"""Cut-first local speech caption workflow for approved edit plans.

This module extracts only video clips whose natural audio is retained, runs
local MLX Whisper on those short audio cuts, and writes reviewable caption
candidates. Meaning is never classified here: Codex or a human supplies an
explicit keep/drop review before captions are applied to the edit plan.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from video_editor.subtitles_local import (
    apply_subtitle_reviews,
    generate_subtitle_candidates,
    validate_subtitle_cues,
)

import video_factory
from transcription_mlx import DEFAULT_MODEL, DEFAULT_MODEL_REVISION, transcribe_audio


CANDIDATES_RELATIVE = "transcripts/caption_candidates.json"
AUTO_CAPTION_ORIGIN = "local-mlx"


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise video_factory.UserFacingError(f"{label} must be a finite number.")
    return float(value)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not video_factory.is_within(path, path.parent):
        raise video_factory.UserFacingError(f"Refusing to write outside the project work directory: {path.name}")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise video_factory.UserFacingError(f"Could not write {path.name}: {exc}") from exc
    finally:
        if temporary:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass


def _plan_file(project: Path) -> Path:
    return video_factory.project_path(project, video_factory.PLAN_RELATIVE_PATH)


def _audio_cut(project: Path, segment: dict[str, Any]) -> tuple[str, str]:
    source_relative, source_path = video_factory.safe_relative_source(
        project, segment.get("source"), f"caption source {segment.get('id', '')!r}"
    )
    source_in = _finite_number(segment.get("source_in"), f"{segment.get('id')}.source_in")
    source_out = _finite_number(segment.get("source_out"), f"{segment.get('id')}.source_out")
    if source_in < 0 or source_out <= source_in:
        raise video_factory.UserFacingError(f"Caption clip {segment.get('id')!r} has an invalid source range.")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise video_factory.UserFacingError("ffmpeg is required to extract retained audio for local captions.")

    source_hash = video_factory.sha256_file(source_path)
    settings = {
        "source": source_relative,
        "source_sha256": source_hash,
        "source_in": round(source_in, 6),
        "source_out": round(source_out, 6),
        "format": "flac",
        "sample_rate": 16000,
        "channels": 1,
        "stream": "first-audio",
    }
    key = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    relative_output = f"work/transcripts/audio/cuts/{key}.flac"
    video_factory.work_path(project, "transcripts/audio/cuts", create_dir=True)
    output = video_factory.work_path(project, f"transcripts/audio/cuts/{key}.flac")
    if output.is_symlink():
        raise video_factory.UserFacingError("Refusing to follow a symlink in the local caption audio cache.")
    expected_duration = source_out - source_in
    if output.is_file() and output.stat().st_size:
        return relative_output, key

    temp_path = output.with_name(f".{output.name}.tmp.flac")
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source_path), "-ss", f"{source_in:.6f}", "-t", f"{expected_duration:.6f}",
        "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", str(temp_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=240, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        temp_path.unlink(missing_ok=True)
        raise video_factory.UserFacingError(f"Could not extract a retained audio cut for {segment.get('id')!r}: {type(exc).__name__}.") from None
    if result.returncode != 0 or not temp_path.is_file() or temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        detail = result.stderr.strip().splitlines()[-1:] if result.stderr else []
        suffix = f" ({detail[0]})" if detail else ""
        raise video_factory.UserFacingError(
            f"Could not extract audio from retained video clip {segment.get('id')!r}{suffix}."
        )
    os.replace(temp_path, output)
    return relative_output, key


def draft_selected_caption_candidates(
    project_value: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    model_revision: str | None = None,
    language: str | None = "zh",
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Extract and locally transcribe only natural-audio clips in the cut."""
    project = video_factory.resolve_project(project_value)
    plan_path = _plan_file(project)
    plan = video_factory.load_json(plan_path, "edit plan")
    errors, _warnings = video_factory.validate_plan_data(project, plan)
    if errors:
        raise video_factory.UserFacingError("Cannot draft captions from an invalid edit plan: " + "; ".join(errors))

    plan_sha256 = video_factory.sha256_file(plan_path)
    transcript_by_clip: dict[str, dict[str, Any]] = {}
    transcription_records: list[dict[str, Any]] = []
    timeline = plan.get("timeline", [])
    selected = [
        segment for segment in timeline
        if isinstance(segment, dict)
        and segment.get("type") == "video"
        and isinstance(segment.get("audio"), dict)
        and segment["audio"].get("preserve_natural") is True
    ]

    for segment in selected:
        clip_id = segment.get("id")
        if not isinstance(clip_id, str) or not clip_id.strip():
            raise video_factory.UserFacingError("Every retained natural-audio video clip needs a stable edit-plan id.")
        source_relative, source_path = video_factory.safe_relative_source(
            project, segment.get("source"), f"caption source {clip_id!r}"
        )
        probe = video_factory.ffprobe_metadata(source_path)
        if not probe.get("audio_streams"):
            transcription_records.append({
                "clip_id": clip_id,
                "source": source_relative,
                "status": "skipped",
                "skip_reason": "no audio stream",
            })
            continue
        relative_audio, cut_key = _audio_cut(project, segment)
        response = transcribe_audio(
            project,
            relative_audio,
            model=model,
            language=language,
            model_revision=model_revision,
            local_files_only=local_files_only,
        )
        transcript_by_clip[clip_id] = response
        transcription_records.append({
            "clip_id": clip_id,
            "source": segment["source"],
            "source_in": segment["source_in"],
            "source_out": segment["source_out"],
            "timeline_start": segment["timeline_start"],
            "timeline_end": segment["timeline_end"],
            "audio_cut": relative_audio,
            "audio_cut_key": cut_key,
            "transcript_text": response.get("text", ""),
            "transcript_language": response.get("language"),
            "transcript_segments": len(response.get("segments", [])),
            "word_count": sum(len(item.get("words", [])) for item in response.get("segments", []) if isinstance(item, dict)),
        })

    candidates = generate_subtitle_candidates(plan, transcript_by_clip, timestamp_reference="clip")
    cue_errors = validate_subtitle_cues(candidates, duration_seconds=plan.get("duration_seconds"))
    if cue_errors:
        raise video_factory.UserFacingError("Generated caption candidates failed timing validation: " + "; ".join(cue_errors))
    candidate_set_sha = hashlib.sha256(
        json.dumps(candidates, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    document = {
        "schema_version": "video-factory.caption-candidates.v1",
        "created_at": video_factory.utc_now(),
        "status": "pending_ai_review" if candidates else "no_caption_candidates",
        "privacy": {"audio_uploaded": False, "audio_processing": "local MLX Whisper"},
        "model": {
            "id": model,
            "revision": model_revision or (DEFAULT_MODEL_REVISION if model == DEFAULT_MODEL else "hub-resolved"),
        },
        "language": language,
        "edit_plan_sha256": plan_sha256,
        "candidate_set_sha256": candidate_set_sha,
        "selected_clip_count": len(selected),
        "transcriptions": transcription_records,
        "candidates": candidates,
        "review_contract": {
            "decision_values": ["keep", "drop"],
            "decisions_key": "candidate id (or explicit phrase:/word: key)",
            "unreviewed_candidates": "must remain pending; do not apply",
            "drop_only_if": "ASR is hallucinated, unintelligible, pure filler, unrelated background speech, or otherwise meaningless",
            "preserve": "meaningful spoken words without inventing or paraphrasing their content",
        },
    }
    video_factory.work_path(project, "transcripts", create_dir=True)
    output = video_factory.work_path(project, CANDIDATES_RELATIVE)
    _atomic_json(output, document)
    return document


def _review_path(project: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise video_factory.UserFacingError("Caption review must use a project-relative path under work/.")
    if not candidate.parts or candidate.parts[0] != "work":
        raise video_factory.UserFacingError("Caption review must be stored under work/.")
    resolved = video_factory.project_path(project, candidate)
    if not video_factory.is_within(resolved, project / "work") or not resolved.is_file():
        raise video_factory.UserFacingError("Caption review file must be an existing file under work/.")
    return resolved


def apply_caption_review(project_value: str | Path, review_value: str | Path) -> tuple[int, int, int]:
    """Apply complete AI/human cue decisions while preserving editorial entries."""
    project = video_factory.resolve_project(project_value)
    plan_path = _plan_file(project)
    plan = video_factory.load_json(plan_path, "edit plan")
    candidate_path = video_factory.work_path(project, CANDIDATES_RELATIVE)
    candidate_doc = video_factory.load_json(candidate_path, "caption candidate file")
    review_path = _review_path(project, review_value)
    review_doc = video_factory.load_json(review_path, "caption review")

    if not isinstance(candidate_doc, dict) or not isinstance(review_doc, dict):
        raise video_factory.UserFacingError("Caption candidate/review root must be a JSON object.")
    if review_doc.get("schema_version") != "video-factory.caption-review.v1":
        raise video_factory.UserFacingError("Caption review schema_version must be video-factory.caption-review.v1.")
    if candidate_doc.get("edit_plan_sha256") != video_factory.sha256_file(plan_path):
        raise video_factory.UserFacingError("Caption candidates are stale because edit_plan.json changed; draft them again.")
    if review_doc.get("edit_plan_sha256") != candidate_doc.get("edit_plan_sha256"):
        raise video_factory.UserFacingError("Caption review belongs to a different edit plan.")
    if review_doc.get("candidate_set_sha256") != candidate_doc.get("candidate_set_sha256"):
        raise video_factory.UserFacingError("Caption review does not match this candidate set.")

    candidates = candidate_doc.get("candidates")
    decisions = review_doc.get("decisions")
    if not isinstance(candidates, list) or not isinstance(decisions, dict):
        raise video_factory.UserFacingError("Caption candidate/review data is malformed.")
    reviewed = apply_subtitle_reviews(candidates, decisions)
    if reviewed["pending"]:
        pending_ids = ", ".join(str(item.get("id")) for item in reviewed["pending"])
        raise video_factory.UserFacingError(f"Every subtitle candidate needs a keep/drop decision before applying; pending: {pending_ids}.")

    keep = reviewed["keep"]
    cue_errors = validate_subtitle_cues(keep, duration_seconds=plan.get("duration_seconds"))
    if cue_errors:
        raise video_factory.UserFacingError("Reviewed subtitles failed validation: " + "; ".join(cue_errors))
    existing = [
        item for item in plan.get("timeline", [])
        if not (isinstance(item, dict) and item.get("caption_origin") == AUTO_CAPTION_ORIGIN)
    ]
    generated = []
    for cue in keep:
        entry = {
            "id": f"spoken-{cue['id']}",
            "type": "subtitle",
            "text": cue["text"],
            "timeline_start": cue["timeline_start"],
            "timeline_end": cue["timeline_end"],
            "source": cue["source"],
            "source_in": cue["source_start"],
            "source_out": cue["source_end"],
            "caption_origin": AUTO_CAPTION_ORIGIN,
            "source_clip_id": cue["source_clip_id"],
            "caption_review": {
                "status": "kept",
                "reason": cue.get("review_reason", "AI judged this spoken phrase meaningful"),
                "reviewer": review_doc.get("reviewer", "Codex"),
            },
        }
        generated.append(entry)
    updated = {**plan, "timeline": existing + generated}
    errors, _warnings = video_factory.validate_plan_data(project, updated)
    if errors:
        raise video_factory.UserFacingError("Reviewed subtitles made the edit plan invalid: " + "; ".join(errors))

    audit = {
        "schema_version": "video-factory.caption-review-audit.v1",
        "created_at": video_factory.utc_now(),
        "reviewer": review_doc.get("reviewer", "Codex"),
        "kept": len(keep),
        "dropped": len(reviewed["drop"]),
        "edit_plan_sha256_before": candidate_doc["edit_plan_sha256"],
        "candidate_set_sha256": candidate_doc["candidate_set_sha256"],
        "decisions": decisions,
    }
    audit_path = video_factory.work_path(project, "transcripts/caption_review_audit.json")
    _atomic_json(plan_path, updated)
    _atomic_json(audit_path, audit)
    return len(keep), len(reviewed["drop"]), len(candidates)


def command_draft_captions(args: Any) -> int:
    document = draft_selected_caption_candidates(
        args.project,
        model=getattr(args, "model", None) or DEFAULT_MODEL,
        model_revision=getattr(args, "model_revision", None),
        language=getattr(args, "language", None) or None,
        local_files_only=bool(getattr(args, "local_only", True)),
    )
    print(
        f"Prepared {len(document['candidates'])} local caption candidate(s) from "
        f"{document['selected_clip_count']} retained natural-audio clip(s): "
        f"{video_factory.work_path(video_factory.resolve_project(args.project), CANDIDATES_RELATIVE)}"
    )
    if not document["candidates"]:
        print("No timed speech candidates were detected; there is no subtitle to add.")
    else:
        print("AI semantic review is required before applying subtitles; audio stayed on this Mac.")
    return 0


def command_apply_caption_review(args: Any) -> int:
    kept, dropped, total = apply_caption_review(args.project, args.review)
    print(f"Applied caption review: kept {kept}, dropped {dropped}, reviewed {total} candidate(s).")
    print("Next: validate-plan, export-srt, render, then inspect the rendered subtitle placements.")
    return 0


__all__ = [
    "apply_caption_review",
    "command_apply_caption_review",
    "command_draft_captions",
    "draft_selected_caption_candidates",
]
