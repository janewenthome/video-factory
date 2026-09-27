"""Local subtitle export and rendered-video QA helpers."""

from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from video_factory import (
    PLAN_RELATIVE_PATH,
    MANIFEST_RELATIVE_PATH,
    UserFacingError,
    is_within,
    load_json,
    parse_float,
    parse_ratio,
    project_path,
    read_job_aspect_ratios,
    resolve_project,
    safe_relative_source,
    validate_plan_data,
    work_path,
)


def write_text_atomic(path: Path, text: str, allowed_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not is_within(path, allowed_root):
        raise UserFacingError(f"Refusing to write outside {allowed_root}: {path}")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(text)
        os.replace(temporary, path)
    except OSError as exc:
        if temporary:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass
        raise UserFacingError(f"Could not write {path}: {exc}") from exc


def srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    total_seconds, millis = divmod(total_ms, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def command_export_srt(args: Any) -> int:
    project = resolve_project(args.project)
    plan_path = project_path(project, PLAN_RELATIVE_PATH)
    plan = load_json(plan_path, "edit plan")
    errors, _ = validate_plan_data(project, plan)
    if errors:
        print("Cannot export subtitles because the edit plan has errors:", file=sys.stderr)
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2

    subtitle_rows: list[tuple[float, float, int, str]] = []
    for index, segment in enumerate(plan["timeline"]):
        if not isinstance(segment, dict) or segment.get("type") != "subtitle":
            continue
        start = parse_float(segment.get("timeline_start"))
        end = parse_float(segment.get("timeline_end"))
        text = segment.get("text")
        if start is None or end is None or not isinstance(text, str) or not text.strip():
            print(f"ERROR: timeline[{index}] does not contain valid subtitle times and text.", file=sys.stderr)
            return 2
        subtitle_rows.append((start, end, index, text.replace("\r\n", "\n").replace("\r", "\n").strip()))
    subtitle_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    blocks = [
        f"{number}\n{srt_timestamp(start)} --> {srt_timestamp(end)}\n{text}"
        for number, (start, end, _, text) in enumerate(subtitle_rows, start=1)
    ]
    content = "\n\n".join(blocks) + ("\n" if blocks else "")
    work_path(project, "transcripts", create_dir=True)
    output = work_path(project, "transcripts/captions.srt")
    write_text_atomic(output, content, project_path(project, "work"))
    print(f"Exported {len(subtitle_rows)} subtitle cue(s): {output}")
    if not subtitle_rows:
        print("The edit plan contains no subtitle entries; wrote an empty SRT file.")
    return 0


def read_job_number(project: Path, key: str) -> float | None:
    path = project_path(project, "job.yaml")
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(rf"(?m)^{re.escape(key)}\s*:\s*['\"]?([-+0-9.eE]+)", text)
    return parse_float(match.group(1)) if match else None


def read_job_bool(project: Path, parent: str, key: str) -> bool | None:
    path = project_path(project, "job.yaml")
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    block = re.search(rf"(?ms)^{re.escape(parent)}\s*:\s*\n((?:[ \t]+[^\n]*\n?)*)", text)
    if not block:
        return None
    match = re.search(rf"(?m)^[ \t]+{re.escape(key)}\s*:\s*(true|false)\b", block.group(1), re.IGNORECASE)
    return match.group(1).lower() == "true" if match else None


def video_input_path(project: Path, value: str) -> tuple[Path | None, str]:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        candidate = raw.resolve(strict=False)
        if not is_within(candidate, project):
            return None, "video path must be inside the project folder"
        if not candidate.is_file():
            return None, f"rendered video does not exist: {candidate}"
        return candidate, candidate.relative_to(project).as_posix()
    try:
        relative, candidate = safe_relative_source(project, value, "--video")
    except UserFacingError as exc:
        return None, str(exc)
    return candidate, relative


def ffprobe_render(video: Path) -> tuple[dict[str, Any] | None, str | None]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None, "ffprobe is not installed or not on PATH"
    try:
        result = subprocess.run(
            [ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(video)],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "ffprobe timed out after 180 seconds"
    except OSError as exc:
        return None, f"could not run ffprobe: {exc}"
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        return None, detail[-1] if detail else f"ffprobe exited with status {result.returncode}"
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        return None, f"ffprobe returned invalid JSON: {exc}"
    if not isinstance(data, dict):
        return None, "ffprobe output was not a JSON object"
    return data, None


def run_filter(ffmpeg: str, video: Path, filter_name: str, *, audio_only: bool = False) -> tuple[str | None, str | None]:
    command = [ffmpeg, "-hide_banner", "-loglevel", "info", "-i", str(video)]
    if audio_only:
        command += ["-vn", "-af", filter_name, "-f", "null", "-"]
    else:
        command += ["-vf", filter_name, "-an", "-f", "null", "-"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
    except subprocess.TimeoutExpired:
        return None, f"ffmpeg {filter_name} timed out after 900 seconds"
    except OSError as exc:
        return None, f"could not run ffmpeg: {exc}"
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        return None, detail[-1] if detail else f"ffmpeg exited with status {result.returncode}"
    return result.stderr, None


def check_timeline(plan: dict[str, Any], errors: list[str], warnings: list[str]) -> None:
    timeline = plan.get("timeline")
    if not isinstance(timeline, list):
        return
    media = [
        (parse_float(item.get("timeline_start")), parse_float(item.get("timeline_end")), item.get("type"), item)
        for item in timeline
        if isinstance(item, dict) and item.get("type") in {"photo", "video"}
    ]
    media = [entry for entry in media if entry[0] is not None and entry[1] is not None]
    cards = [
        (parse_float(item.get("timeline_start")), parse_float(item.get("timeline_end")), item.get("type"), item)
        for item in timeline
        if isinstance(item, dict) and item.get("type") in {"title", "text_card"}
    ]
    cards = [entry for entry in cards if entry[0] is not None and entry[1] is not None]
    # Title/text cards that overlap source footage are overlays; standalone cards can bridge footage gaps.
    standalone_cards = [card for card in cards if not any(card[0] < clip[1] and card[1] > clip[0] for clip in media)]
    visual = sorted(media + standalone_cards, key=lambda item: (item[0], item[1]))
    if not media:
        warnings.append("No photo/video source clips were found; confirm the composition is intentionally text-only.")
    for left, right in zip(visual, visual[1:]):
        gap = right[0] - left[1]
        if gap > 0.05:
            warnings.append(f"Possible timeline gap of {gap:.2f}s between {left[2]} and {right[2]}.")
        elif gap < -0.05:
            warnings.append(f"Timeline overlap of {abs(gap):.2f}s between {left[2]} and {right[2]}; confirm a transition or layered edit is intended.")


def command_qa(args: Any) -> int:
    project = resolve_project(args.project)
    errors: list[str] = []
    warnings: list[str] = []
    suggestions: list[str] = []
    checks: list[str] = []

    plan: dict[str, Any] | None = None
    plan_path = project_path(project, PLAN_RELATIVE_PATH)
    if plan_path.is_file():
        try:
            loaded = load_json(plan_path, "edit plan")
            if isinstance(loaded, dict):
                plan = loaded
                plan_errors, plan_warnings = validate_plan_data(project, plan)
                errors.extend(f"Edit plan: {message}" for message in plan_errors)
                warnings.extend(f"Edit plan: {message}" for message in plan_warnings)
                check_timeline(plan, errors, warnings)
            else:
                errors.append("Edit plan root is not a JSON object.")
        except UserFacingError as exc:
            errors.append(str(exc))
    else:
        errors.append(f"Edit plan is missing: {plan_path}")

    video, video_label = video_input_path(project, args.video)
    probe: dict[str, Any] | None = None
    duration: float | None = None
    width: int | None = None
    height: int | None = None
    video_streams: list[dict[str, Any]] = []
    audio_streams: list[dict[str, Any]] = []
    if video is None:
        errors.append(f"Rendered video path is invalid: {video_label}.")
    else:
        probe, probe_error = ffprobe_render(video)
        if probe_error:
            errors.append(f"ffprobe could not read {video_label}: {probe_error}.")
        else:
            streams = probe.get("streams", []) if isinstance(probe, dict) else []
            fmt = probe.get("format", {}) if isinstance(probe, dict) else {}
            video_streams = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video" and not item.get("disposition", {}).get("attached_pic")] if isinstance(streams, list) else []
            audio_streams = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "audio"] if isinstance(streams, list) else []
            duration = parse_float(fmt.get("duration")) if isinstance(fmt, dict) else None
            if not video_streams:
                errors.append("Rendered file has no readable video stream.")
            else:
                width = video_streams[0].get("width") if isinstance(video_streams[0].get("width"), int) else None
                height = video_streams[0].get("height") if isinstance(video_streams[0].get("height"), int) else None
                if not width or not height:
                    errors.append("Rendered video stream has missing or invalid dimensions.")
                else:
                    checks.append(f"Video dimensions: {width}x{height}.")
            if duration is None or duration <= 0:
                errors.append("Rendered file has missing or invalid duration.")
            else:
                checks.append(f"Duration: {duration:.3f}s.")
            checks.append(f"Audio tracks: {len(audio_streams)}.")

    expected_ratios = read_job_aspect_ratios(project)
    if width and height and expected_ratios:
        actual_ratio = width / height
        allowed: list[str] = []
        for ratio in expected_ratios:
            try:
                rw, rh = parse_ratio(ratio)
                allowed.append(ratio)
                if abs(actual_ratio - rw / rh) <= 0.02:
                    break
            except UserFacingError:
                continue
        else:
            errors.append(f"Rendered aspect ratio {width}:{height} does not match any job.yaml ratio: {', '.join(allowed)}.")
        if allowed:
            checks.append(f"Job aspect ratio targets: {', '.join(allowed)}.")
    elif width and height:
        warnings.append("job.yaml has no readable aspect_ratios list; output ratio was not compared to a target.")

    plan_duration: float | None = None
    timeline = plan.get("timeline", []) if isinstance(plan, dict) else []
    if isinstance(plan, dict):
        plan_duration = parse_float(plan.get("duration_seconds"))
    if plan_duration is None and isinstance(timeline, list):
        plan_duration = max((parse_float(item.get("timeline_end")) or 0.0 for item in timeline if isinstance(item, dict)), default=None)
    target_duration = read_job_number(project, "target_duration_seconds")
    if duration is not None and plan_duration and plan_duration > 0:
        tolerance = max(1.0, plan_duration * 0.02)
        if abs(duration - plan_duration) > tolerance:
            errors.append(f"Rendered duration differs from edit plan ({duration:.2f}s vs {plan_duration:.2f}s; tolerance {tolerance:.2f}s).")
        else:
            checks.append(f"Duration is within {tolerance:.2f}s of edit plan ({plan_duration:.2f}s).")
    if duration is not None and target_duration and target_duration > 0:
        tolerance = max(5.0, target_duration * 0.1)
        if abs(duration - target_duration) > tolerance:
            warnings.append(f"Rendered duration differs from job target ({duration:.2f}s vs {target_duration:.2f}s; review the intended pacing).")
        else:
            checks.append(f"Duration is within {tolerance:.2f}s of job target ({target_duration:.2f}s).")

    expected_audio = False
    if isinstance(timeline, list):
        expected_audio = any(isinstance(item, dict) and item.get("type") in {"music", "narration", "natural_audio"} for item in timeline)
        if not expected_audio and read_job_bool(project, "natural_audio", "preserve") is True:
            manifest_path = project_path(project, MANIFEST_RELATIVE_PATH)
            if manifest_path.is_file():
                try:
                    manifest = load_json(manifest_path, "media manifest")
                    audio_sources = {
                        item.get("source") for item in manifest.get("assets", [])
                        if isinstance(item, dict) and item.get("has_audio") is True
                    }
                    expected_audio = any(isinstance(item, dict) and item.get("type") == "video" and item.get("source") in audio_sources for item in timeline)
                except UserFacingError:
                    pass
    if probe is not None and expected_audio and not audio_streams:
        errors.append("The edit plan/job expects audio, but the rendered file has no audio track.")
    elif probe is not None and not audio_streams:
        warnings.append("Rendered file has no audio track; confirm that silence is intentional.")

    if video is not None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            warnings.append("ffmpeg is not installed or not on PATH; black-frame and audio-level analysis were skipped.")
        else:
            black_output, black_error = run_filter(ffmpeg, video, "blackdetect=d=0.5:pix_th=0.10")
            if black_error:
                warnings.append(f"Black-frame scan could not complete: {black_error}.")
            elif black_output is not None:
                black_spans = re.findall(r"black_start:([0-9.]+)\s+black_end:([0-9.]+)\s+black_duration:([0-9.]+)", black_output)
                if black_spans:
                    warnings.append("Black-frame scan found " + "; ".join(f"{start}-{end}s ({length}s)" for start, end, length in black_spans) + "; confirm these are intentional.")
                else:
                    checks.append("Black-frame scan found no intervals of 0.5s or longer.")
            if audio_streams:
                audio_output, audio_error = run_filter(ffmpeg, video, "astats=metadata=1:reset=0", audio_only=True)
                if audio_error:
                    warnings.append(f"Audio-level scan could not complete: {audio_error}.")
                elif audio_output is not None:
                    overall = audio_output.split("Overall", 1)[-1]
                    peak_matches = re.findall(r"Peak level dB:\s*(-?(?:\d+(?:\.\d*)?|inf))", overall, re.IGNORECASE)
                    rms_matches = re.findall(r"RMS level dB:\s*(-?(?:\d+(?:\.\d*)?|inf))", overall, re.IGNORECASE)
                    clip_matches = re.findall(r"Number of clips:\s*(\d+)", overall, re.IGNORECASE)
                    if peak_matches:
                        try:
                            peak = float(peak_matches[-1])
                        except ValueError:
                            peak = None
                        if peak is not None:
                            checks.append(f"Overall audio peak: {peak:.2f} dBFS.")
                            if math.isinf(peak) and peak < 0:
                                warnings.append("Audio peak is -inf dBFS; the rendered audio may be silent.")
                            elif peak >= -0.05:
                                warnings.append(f"Audio peak is near digital full scale ({peak:.2f} dBFS); listen for clipping.")
                    else:
                        warnings.append("Audio-level scan did not report an overall peak value.")
                    if rms_matches:
                        try:
                            rms = float(rms_matches[-1])
                        except ValueError:
                            rms = None
                        if rms is not None:
                            checks.append(f"Overall audio RMS: {rms:.2f} dBFS.")
                            if rms <= -60:
                                warnings.append("Audio RMS is very low; confirm that the soundtrack is audible.")
                    if clip_matches and int(clip_matches[-1]) > 0:
                        warnings.append(f"Audio statistics reported {clip_matches[-1]} clipped sample(s).")
                    if not peak_matches and not rms_matches:
                        warnings.append("Audio-level scan completed without readable level metrics.")

    warnings.append("Visual framing, subject-safe crop, subtitle placement, and text readability require a human viewing pass.")
    if isinstance(timeline, list) and any(isinstance(item, dict) and item.get("type") == "subtitle" for item in timeline):
        suggestions.append("Open the rendered video at the start, middle, and end of subtitle sections and check safe margins and line breaks.")
    if not checks:
        checks.append("No automated media checks completed successfully.")

    report_path = project_path(project, "outputs", create_dir=True) / "qa_report.md"
    report_json_path = project_path(project, "outputs", create_dir=True) / "qa_report.json"
    if report_path.parent.is_symlink():
        raise UserFacingError("outputs/ must be a real directory inside the project.")
    if not is_within(report_path, project_path(project, "outputs")):
        raise UserFacingError("QA report output must stay in outputs/.")
    status = "ERROR" if errors else "WARNING" if warnings else "PASS"
    checked_time = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    lines = [
        "# Video QA Report",
        "",
        f"- Status: **{status}**",
        f"- Render: `{video_label}`",
        f"- Checked at: {checked_time}",
        "",
        "## ERROR",
    ]
    lines.extend(f"- {item}" for item in errors) if errors else lines.append("- None.")
    lines.extend(["", "## WARNING"])
    lines.extend(f"- {item}" for item in warnings) if warnings else lines.append("- None.")
    lines.extend(["", "## SUGGESTION"])
    lines.extend(f"- {item}" for item in suggestions) if suggestions else lines.append("- None.")
    lines.extend(["", "## Checks"])
    lines.extend(f"- {item}" for item in checks)
    lines.append("")
    write_text_atomic(report_path, "\n".join(lines), project_path(project, "outputs"))

    qa_json_data = {
        "status": status,
        "render": video_label,
        "checked_at": checked_time,
        "duration_seconds": duration,
        "width": width,
        "height": height,
        "audio_tracks": len(audio_streams),
        "errors": errors,
        "warnings": warnings,
        "suggestions": suggestions,
        "checks": checks,
    }
    write_text_atomic(report_json_path, json.dumps(qa_json_data, ensure_ascii=False, indent=2) + "\n", project_path(project, "outputs"))

    print(f"QA status: {status}")
    print(f"Report: {report_path}")
    print(f"Report JSON: {report_json_path}")
    for error in errors:
        print(f"ERROR: {error}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 2 if errors else 0


def command_quality(args: Any) -> int:
    if args.command == "export-srt":
        return command_export_srt(args)
    if args.command == "qa":
        return command_qa(args)
    raise UserFacingError(f"Unknown quality command: {args.command}")
