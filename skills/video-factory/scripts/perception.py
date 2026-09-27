"""Perception index contract and local orchestration for Video Factory.

The Mac creates the manifest, audio extracts, proxies and representative
frames.  This module combines those derived artifacts into one cacheable
index.  A cloud worker may later fill the ``pending`` fields, but this layer
never turns a similarity score into an editorial keep decision and never
uploads files by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from product_policy import GPU_POLICIES, PRIVACY_MODES, load_product_policy
except ModuleNotFoundError:  # direct ``video_factory.py perception`` invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from video_editor.product_policy import GPU_POLICIES, PRIVACY_MODES, load_product_policy
from video_factory import (
    MANIFEST_RELATIVE_PATH,
    UserFacingError,
    load_json,
    project_path,
    resolve_project,
    sha256_file,
    utc_now,
    work_path,
    write_json,
)


INDEX_SCHEMA_VERSION = "perception-index.v1"
WORKER_VERSION = "colab-perception.v2"
SPEECH_MODEL = "large-v3-turbo"
SPEECH_ENGINE_VERSION = "faster-whisper-1.2.1"
SPEECH_MODEL_REPO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
SPEECH_MODEL_REVISION = "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"
DIARIZATION_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DIARIZATION_MODEL_REVISION = "0f99f2d0ebe89ac095bcc5903c4dd8f72b367286"
VISUAL_MODEL = "google/siglip2-base-patch16-224"
VISUAL_MODEL_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
TEMPORAL_MODEL = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
TEMPORAL_MODEL_REVISION = "482adb537c021c86670beed01cd58990d01e72e4"
COLAB_PACKAGE_VERSIONS = {
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.6.0",
    "torch": "2.8.0",
    "torchaudio": "2.8.0",
    "transformers": "4.57.6",
    "speechbrain": "1.0.3",
    "numpy": "1.26.4",
    "Pillow": "11.3.0",
    "soundfile": "0.13.1",
    "scipy": "1.15.3",
    "huggingface-hub": "0.36.0",
    "decord": "0.6.0",
    "num2words": "0.5.14",
    "nvidia-cublas-cu12": "12.8.4.1",
    "nvidia-cudnn-cu12": "9.10.2.21",
}
VISUAL_ENGINE_VERSION = f"transformers-{COLAB_PACKAGE_VERSIONS['transformers']}"
TEMPORAL_ADAPTERS = {
    "smolvlm2": {
        "model": TEMPORAL_MODEL,
        "revision": TEMPORAL_MODEL_REVISION,
        "precision": "float16",
        "runner": "run_smolvlm2_temporal",
    },
}
TEMPORAL_BACKENDS = {"none", *TEMPORAL_ADAPTERS}


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _relative_file(project: Path, value: str | Path) -> tuple[str, Path] | None:
    path = Path(value)
    if path.is_absolute():
        candidate = path.resolve(strict=False)
    else:
        candidate = (project / path).resolve(strict=False)
    try:
        relative = candidate.relative_to(project.resolve(strict=True)).as_posix()
    except (ValueError, FileNotFoundError):
        return None
    if not candidate.is_file():
        return None
    return relative, candidate


def _read_json_if(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return load_json(path, path.name)
    except UserFacingError:
        return None


def _manifest_sources(project: Path) -> list[dict[str, Any]]:
    manifest_path = project_path(project, MANIFEST_RELATIVE_PATH)
    value = _read_json_if(manifest_path)
    if not isinstance(value, dict):
        return []
    return [item for item in value.get("assets", []) if isinstance(item, dict) and item.get("source")]


def _transcript_records(project: Path) -> list[dict[str, Any]]:
    root = project / "work" / "transcripts"
    records: list[dict[str, Any]] = []
    if not root.is_dir():
        return records
    for path in sorted(root.glob("*.json")):
        value = _read_json_if(path)
        if not isinstance(value, dict):
            continue
        response = value.get("response") if isinstance(value.get("response"), dict) else value
        segments = response.get("segments") if isinstance(response, dict) else None
        if not isinstance(segments, list):
            continue
        request = value.get("request") if isinstance(value.get("request"), dict) else {}
        records.append({
            "cache_file": path.relative_to(project).as_posix(),
            "source": request.get("source"),
            "source_sha256": request.get("source_sha256"),
            "model": request.get("model") or request.get("engine"),
            "segments": [
                {"start": item.get("start"), "end": item.get("end"), "text": item.get("text", "")}
                for item in segments if isinstance(item, dict)
            ],
        })
    return records


def _frame_records(project: Path) -> list[dict[str, Any]]:
    path = project / "work" / "frames" / "frames.json"
    value = _read_json_if(path)
    if not isinstance(value, dict):
        return []
    output: list[dict[str, Any]] = []
    assets = value.get("assets", {})
    if not isinstance(assets, dict):
        return output
    for source, item in assets.items():
        if not isinstance(item, dict):
            continue
        frames = item.get("frames", [])
        if not isinstance(frames, list):
            continue
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict) or not frame.get("path"):
                continue
            resolved = _relative_file(project, frame["path"])
            if resolved is None:
                continue
            relative, frame_path = resolved
            try:
                frame_hash = sha256_file(frame_path)
            except UserFacingError:
                continue
            output.append({
                "id": f"frame-{len(output)+1:05d}",
                "source": source,
                "path": relative,
                "sha256": frame_hash,
                "timestamp_seconds": frame.get("timestamp_seconds", frame.get("timestamp")),
                "sample_index": index,
            })
    return output


def _audio_upload_candidates(project: Path) -> list[dict[str, Any]]:
    index = _read_json_if(project / "work" / "transcripts" / "audio" / "index.json")
    candidates: list[dict[str, Any]] = []
    if not isinstance(index, dict):
        return candidates
    assets = index.get("assets", {})
    if not isinstance(assets, dict):
        return candidates
    for source, record in assets.items():
        if not isinstance(record, dict) or not record.get("output"):
            continue
        resolved = _relative_file(project, record["output"])
        if resolved is None:
            continue
        relative, path = resolved
        candidates.append({"source": source, "path": relative, "sha256": sha256_file(path)})
    return candidates


def _proxy_upload_candidates(
    project: Path,
    selected_sources: list[str],
    privacy_mode: str,
) -> list[dict[str, Any]]:
    """Return only derived proxies eligible for the selected privacy mode.

    BALANCED deliberately does not send the existing 720p proxy because the
    normal contract caps routine transfers at 480p. MAX_QUALITY may send a
    720p proxy, but only for sources already shortlisted for temporal review.
    """
    if privacy_mode != "MAX_QUALITY" or not selected_sources:
        return []
    if len(selected_sources) != len(set(selected_sources)):
        raise UserFacingError("Temporal source shortlist contains duplicates.")
    manifest = _read_json_if(project / "work" / "proxies" / "manifest.json")
    records = manifest.get("proxies", {}) if isinstance(manifest, dict) else {}
    if not isinstance(records, dict):
        raise UserFacingError("No generated proxy manifest is available for temporal analysis.")
    target_height = manifest.get("target_height") if isinstance(manifest, dict) else None
    if not isinstance(target_height, int) or target_height <= 0 or target_height > 720:
        raise UserFacingError("Temporal analysis requires generated proxies at or below 720p.")
    source_hashes = {
        item.get("source"): item.get("sha256")
        for item in _manifest_sources(project)
        if item.get("kind") == "video"
    }
    sources = set(selected_sources)
    candidates: list[dict[str, Any]] = []
    for source in sorted(sources):
        record = records.get(source)
        if source not in source_hashes or not isinstance(record, dict) or not record.get("proxy_path"):
            raise UserFacingError(f"No derived proxy is available for shortlisted source {source!r}.")
        if record.get("source_sha256") != source_hashes[source]:
            raise UserFacingError(f"The shortlisted proxy is stale for source {source!r}; regenerate proxies first.")
        resolved = _relative_file(project, record["proxy_path"])
        if resolved is None:
            raise UserFacingError(f"The shortlisted proxy is missing for source {source!r}.")
        relative, path = resolved
        if not relative.startswith("work/"):
            raise UserFacingError(f"Temporal proxy is not stored as a project-derived work artifact: {source!r}.")
        digest = sha256_file(path)
        expected_digest = record.get("proxy_sha256")
        if not expected_digest or digest != expected_digest:
            raise UserFacingError(f"The shortlisted proxy hash does not match its manifest for {source!r}.")
        candidates.append({
            "source": source,
            "path": relative,
            "sha256": digest,
            "max_height": target_height,
        })
    return candidates


def assign_anonymous_speakers(
    segments: list[dict[str, Any]],
    embeddings: list[list[float] | None],
) -> list[dict[str, Any]]:
    """Assign clip-local labels from ECAPA embeddings without claiming calibrated confidence.

    This is conservative utterance-level clustering, not overlap-aware
    diarization. It intentionally has no cross-source state and returns
    ``unknown`` when segment duration, vector validity, or cluster margin is
    insufficient.
    """
    if len(segments) != len(embeddings):
        raise ValueError("Every speech segment must have one embedding slot.")
    centers: list[list[float]] = []
    labels: list[str] = []
    counts: list[int] = []
    output: list[dict[str, Any]] = []
    expected_dimension = 192
    minimum_duration = 1.5
    new_speaker_below = 0.62
    assign_above = 0.78
    minimum_margin = 0.06

    for segment, vector in zip(segments, embeddings):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", 0.0))
        duration = end - start
        result = {
            "start": start,
            "end": end,
            "text": segment.get("text", ""),
            "speaker_id": "unknown",
            "assignment_method": "speechbrain_ecapa_cosine_heuristic",
            "similarity_to_cluster": None,
            "confidence": None,
        }
        if duration < minimum_duration or not isinstance(vector, list) or not vector:
            result["assignment_reason"] = "insufficient_segment_or_embedding"
            output.append(result)
            continue
        if len(vector) != expected_dimension or any(not math.isfinite(float(value)) for value in vector):
            result["assignment_reason"] = "invalid_embedding"
            output.append(result)
            continue
        norm = math.sqrt(sum(float(value) * float(value) for value in vector))
        if norm <= 1e-12:
            result["assignment_reason"] = "invalid_embedding"
            output.append(result)
            continue
        normalized = [float(value) / norm for value in vector]
        if not centers:
            centers.append(normalized)
            labels.append("speaker_01")
            counts.append(1)
            result.update({"speaker_id": "speaker_01", "assignment_reason": "first_valid_cluster"})
            output.append(result)
            continue

        similarities = [sum(a * b for a, b in zip(normalized, center)) for center in centers]
        ordered = sorted(enumerate(similarities), key=lambda item: item[1], reverse=True)
        best_index, best_score = ordered[0]
        runner_up = ordered[1][1] if len(ordered) > 1 else -1.0
        result["similarity_to_cluster"] = round(best_score, 6)
        if best_score >= assign_above and best_score - runner_up >= minimum_margin:
            count = counts[best_index]
            centers[best_index] = [
                ((centers[best_index][index] * count) + normalized[index]) / (count + 1)
                for index in range(len(normalized))
            ]
            center_norm = math.sqrt(sum(value * value for value in centers[best_index]))
            centers[best_index] = [value / max(center_norm, 1e-12) for value in centers[best_index]]
            counts[best_index] += 1
            result.update({"speaker_id": labels[best_index], "assignment_reason": "similarity_and_margin_passed"})
        elif best_score < new_speaker_below and len(centers) < 8:
            label = f"speaker_{len(labels) + 1:02d}"
            centers.append(normalized)
            labels.append(label)
            counts.append(1)
            result.update({"speaker_id": label, "assignment_reason": "new_distinct_cluster"})
        else:
            result["assignment_reason"] = "ambiguous_cosine_similarity"
        output.append(result)
    return output


def _duplicate_clusters(frames: list[dict[str, Any]]) -> dict[str, str]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for frame in frames:
        by_hash[str(frame["sha256"])].append(frame["id"])
    result: dict[str, str] = {}
    for index, ids in enumerate(sorted(by_hash.values(), key=lambda values: values[0]), start=1):
        cluster_id = f"visual-{index:04d}"
        for frame_id in ids:
            result[frame_id] = cluster_id
    return result


def _cache_path(project: Path, cache_key: str) -> Path:
    cache_dir = work_path(project, "perception-cache", create_dir=True)
    return cache_dir / f"{cache_key}.json"


def build_perception_index(
    project: Path,
    *,
    privacy_mode: str | None = None,
    gpu_policy: str | None = None,
    cloud: bool | None = None,
    allow_upload: bool = False,
    temporal_backend: str | None = None,
    force: bool = False,
) -> Path:
    """Build a deterministic perception index and return its canonical path.

    ``cloud=True`` only records an authorized transfer plan.  The actual
    Colab transport is intentionally a separate adapter so a normal pipeline
    run cannot silently send media outside the Mac.
    """
    project = resolve_project(project)
    policy = load_product_policy(
        project,
        privacy_mode=privacy_mode,
        gpu_policy=gpu_policy,
        cloud_perception=cloud,
    )
    selected_backend = (temporal_backend or policy.temporal_backend or "none").strip().lower()
    if selected_backend not in TEMPORAL_BACKENDS:
        raise UserFacingError(f"Unknown temporal backend {selected_backend!r}.")
    if policy.gpu_policy not in GPU_POLICIES:
        raise UserFacingError("Perception GPU policy must be T4 or L4.")
    if policy.privacy_mode not in PRIVACY_MODES:
        raise UserFacingError("Unknown perception privacy mode.")
    if policy.gpu_policy == "L4" and not (
        policy.privacy_mode == "MAX_QUALITY" and selected_backend != "none"
    ):
        raise UserFacingError(
            "L4 is reserved for MAX_QUALITY shortlist-only temporal analysis; routine perception uses T4."
        )
    if policy.privacy_mode == "LOCAL_ONLY" and (cloud or policy.cloud_perception):
        raise UserFacingError("LOCAL_ONLY forbids cloud perception.")
    if (cloud or policy.cloud_perception) and not allow_upload:
        cloud_status = "awaiting_authorization"
    elif cloud or policy.cloud_perception:
        cloud_status = "authorized_plan_only"
    else:
        cloud_status = "not_requested"

    manifest_assets = _manifest_sources(project)
    frames = _frame_records(project)
    transcripts = _transcript_records(project)
    audio_candidates = _audio_upload_candidates(project)
    fingerprint = {
        "manifest": [{"source": item.get("source"), "sha256": item.get("sha256")} for item in manifest_assets],
        "frames": [{"path": item["path"], "sha256": item["sha256"]} for item in frames],
        "audio": audio_candidates,
        "privacy_mode": policy.privacy_mode,
        "gpu_policy": policy.gpu_policy,
        "speech": {
            "model": SPEECH_MODEL,
            "engine": SPEECH_ENGINE_VERSION,
            "model_repo": SPEECH_MODEL_REPO,
            "model_revision": SPEECH_MODEL_REVISION,
        },
        "diarization": {
            "model": DIARIZATION_MODEL,
            "model_revision": DIARIZATION_MODEL_REVISION,
            "algorithm": "clip-local-ecapa-cosine-heuristic.v1",
        },
        "visual": {
            "model": VISUAL_MODEL,
            "model_revision": VISUAL_MODEL_REVISION,
            "engine": VISUAL_ENGINE_VERSION,
        },
        "packages": COLAB_PACKAGE_VERSIONS,
        "worker_version": WORKER_VERSION,
    }
    clusters = _duplicate_clusters(frames)
    visual_items: list[dict[str, Any]] = []
    for frame in frames:
        frame_id = frame["id"]
        visual_items.append({
            **frame,
            "cluster_id": clusters.get(frame_id),
            "embedding": {
                "model": VISUAL_MODEL,
                "status": "pending_colab" if cloud_status != "not_requested" else "not_requested",
                "reference": None,
            },
            "similarity": None,
            "confidence": 0.0,
        })
    cache_key = _json_hash(fingerprint)
    cache_path = _cache_path(project, cache_key)
    canonical_path = work_path(project, "perception_index.json", create_dir=False)
    if not force and cache_path.is_file():
        cached = _read_json_if(cache_path)
        if isinstance(cached, dict) and cached.get("cache_key") == cache_key:
            cached_privacy = cached.setdefault("privacy", {})
            cached_privacy["mode"] = policy.privacy_mode
            completed_cloud_result = cached_privacy.get("cloud_status") == "completed"
            if not completed_cloud_result:
                cached_privacy["cloud_status"] = cloud_status
            cached_privacy["original_media_uploaded"] = False
            cached_privacy["eligible_uploads"] = {
                "audio": audio_candidates if policy.privacy_mode != "LOCAL_ONLY" else [],
                "representative_frames": [item["path"] for item in frames] if policy.privacy_mode != "LOCAL_ONLY" else [],
                "proxies": [],
                "proxy_max_height": 720 if policy.privacy_mode == "MAX_QUALITY" else 480,
            }
            cached.setdefault("compute", {})["gpu_policy"] = policy.gpu_policy
            if not completed_cloud_result:
                has_pending_cloud_inputs = bool(
                    cached_privacy["eligible_uploads"]["audio"]
                    or cached_privacy["eligible_uploads"]["representative_frames"]
                )
                pending_status = "pending_colab" if cloud_status != "not_requested" and has_pending_cloud_inputs else "not_requested"
                cached_speech = cached.setdefault("speech", {})
                if cached_speech.get("status") != "completed":
                    cached_speech["status"] = pending_status if audio_candidates else "not_applicable"
                    cached_speech["diarization_status"] = pending_status if audio_candidates else "not_applicable"
                cached_visual = cached.setdefault("visual", {})
                if cached_visual.get("status") != "completed":
                    cached_visual["status"] = pending_status if frames else "not_applicable"
                    for item in cached_visual.get("items", []):
                        embedding = item.setdefault("embedding", {})
                        if embedding.get("status") != "completed":
                            embedding["status"] = pending_status
            cached.setdefault("models", {})["temporal"] = {
                "backend": selected_backend,
                "name": TEMPORAL_ADAPTERS[selected_backend]["model"] if selected_backend in TEMPORAL_ADAPTERS else None,
                "revision": TEMPORAL_ADAPTERS[selected_backend]["revision"] if selected_backend in TEMPORAL_ADAPTERS else None,
            }
            cached.setdefault("temporal", {}).update({
                "backend": selected_backend,
                "candidates": [],
                "results": [],
                "event_groups": [],
                "status": "awaiting_shortlist" if selected_backend != "none" else "not_requested",
            })
            write_json(canonical_path, cached)
            write_json(project_path(project, "outputs/work/perception_index.json"), cached)
            return canonical_path

    for item in visual_items:
        item["embedding"]["reference"] = (
            f"work/perception-cache/{cache_key}/embeddings/{item['id']}.json"
        )

    proxy_candidates: list[dict[str, Any]] = []

    index = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "created_at": utc_now(),
        "cache_key": cache_key,
        "privacy": {
            "mode": policy.privacy_mode,
            "original_media_uploaded": False,
            "cloud_status": cloud_status,
            "eligible_uploads": {
                "audio": audio_candidates if policy.privacy_mode != "LOCAL_ONLY" else [],
                "representative_frames": [item["path"] for item in frames] if policy.privacy_mode != "LOCAL_ONLY" else [],
                # Temporal proxy uploads are populated only by the explicit
                # second-stage Colab command after a director shortlist.
                "proxies": proxy_candidates,
                "proxy_max_height": 720 if policy.privacy_mode == "MAX_QUALITY" else 480,
            },
        },
        "compute": {"gpu_policy": policy.gpu_policy, "premium_gpu_allowed": False},
        "models": {
            "speech": {
                "name": SPEECH_MODEL,
                "engine": SPEECH_ENGINE_VERSION,
                "repo": SPEECH_MODEL_REPO,
                "revision": SPEECH_MODEL_REVISION,
                "word_timestamps": True,
            },
            "diarization": {
                "name": DIARIZATION_MODEL,
                "revision": DIARIZATION_MODEL_REVISION,
                "method": "clip-local-ecapa-cosine-heuristic.v1",
                "confidence_calibrated": False,
            },
            "visual": {"name": VISUAL_MODEL, "revision": VISUAL_MODEL_REVISION, "engine": VISUAL_ENGINE_VERSION},
            "temporal": {
                "backend": selected_backend,
                "name": TEMPORAL_ADAPTERS[selected_backend]["model"] if selected_backend in TEMPORAL_ADAPTERS else None,
                "revision": TEMPORAL_ADAPTERS[selected_backend]["revision"] if selected_backend in TEMPORAL_ADAPTERS else None,
            },
        },
        "speech": {
            "transcripts": transcripts,
            "speech_turns": [],
            "anonymous_speakers": [],
            "audio_events": [],
            "importance": [],
            "status": (
                "pending_colab" if audio_candidates and cloud_status != "not_requested"
                else "local_metadata_only" if audio_candidates else "not_applicable"
            ),
            "diarization_status": (
                "pending_colab" if audio_candidates and cloud_status != "not_requested"
                else "not_requested" if audio_candidates else "not_applicable"
            ),
            "audio_events_status": "not_configured",
            "importance_status": "not_configured",
            "importance_method": None,
            "importance_confidence_calibrated": False,
        },
        "visual": {
            "items": visual_items,
            "retrieval_index": None,
            "scene_similarity": [],
            "status": (
                "pending_colab" if frames and cloud_status != "not_requested"
                else "local_metadata_only" if frames else "not_applicable"
            ),
            "exact_hash_duplicates": clusters,
            "similarity_clusters": {},
            "event_groups": [],
            "event_groups_status": "not_configured",
        },
        "temporal": {
            "backend": selected_backend,
            "candidates": [],
            "results": [],
            "status": "awaiting_shortlist" if selected_backend != "none" else "not_requested",
            "event_groups": [],
        },
        "editorial": {
            "candidate_signals": [
                {"frame_id": item["id"], "signals": ["representative_frame", "visual_cluster"], "keep_decision": None, "confidence": 0.0}
                for item in visual_items
            ],
            "director": "Antigravity/Gemini",
            "decision_boundary": "Perception supplies evidence and candidates; it does not select the final story.",
        },
    }
    write_json(cache_path, index)
    write_json(canonical_path, index)
    # Keep the article's discoverable output path as a compatibility mirror;
    # ``work/perception_index.json`` remains the single source of truth.
    mirror = project_path(project, "outputs/work/perception_index.json")
    write_json(mirror, index)
    return canonical_path


def command_perception(args: Any) -> int:
    try:
        path = build_perception_index(
            resolve_project(args.project),
            privacy_mode=getattr(args, "privacy_mode", None),
            gpu_policy=getattr(args, "gpu", None),
            cloud=getattr(args, "cloud", None),
            allow_upload=getattr(args, "allow_upload", False),
            temporal_backend=getattr(args, "temporal_backend", None),
            force=getattr(args, "force", False),
        )
    except (UserFacingError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Perception index: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a privacy-aware perception index from local derivatives")
    parser.add_argument("project")
    parser.add_argument("--privacy-mode", choices=sorted(PRIVACY_MODES))
    parser.add_argument("--gpu", choices=sorted(GPU_POLICIES), default=None)
    parser.add_argument("--cloud", action="store_true", default=None, help="record an authorized Colab plan; does not upload by itself")
    parser.add_argument("--allow-upload", action="store_true", help="mark the plan authorized for a separate cloud adapter")
    parser.add_argument("--temporal-backend", choices=sorted(TEMPORAL_BACKENDS), default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    return command_perception(args)


__all__ = [
    "INDEX_SCHEMA_VERSION",
    "TEMPORAL_ADAPTERS",
    "TEMPORAL_BACKENDS",
    "VISUAL_MODEL",
    "build_perception_index",
    "command_perception",
]


if __name__ == "__main__":
    raise SystemExit(main())
