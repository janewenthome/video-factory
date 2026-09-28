"""FFmpeg scene detection, representative-frame, and contact-sheet helpers."""

from __future__ import annotations

import hashlib
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
    IMAGE_SUFFIXES,
    MANIFEST_RELATIVE_PATH,
    UserFacingError,
    VIDEO_SUFFIXES,
    ffprobe_metadata,
    is_within,
    load_json,
    project_path,
    resolve_project,
    sha256_file,
    utc_now,
    work_path,
    write_json,
)


ANALYSIS_RELATIVE_PATH = Path("work/analysis/scene_candidates.json")
FRAMES_INDEX_RELATIVE_PATH = Path("work/frames/frames.json")
SHEETS_INDEX_RELATIVE_PATH = Path("work/contact-sheets/index.json")


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise UserFacingError("ffmpeg is not installed or not on PATH. Install FFmpeg, which provides ffmpeg and ffprobe.")
    return ffmpeg


def get_manifest_assets(project: Path, kinds: set[str] | None = None) -> list[dict[str, Any]]:
    manifest_path = project_path(project, MANIFEST_RELATIVE_PATH)
    if not manifest_path.is_file():
        raise UserFacingError("Media manifest is missing. Run `inspect PROJECT` first.")
    manifest = load_json(manifest_path, "media manifest")
    assets = manifest.get("assets") if isinstance(manifest, dict) else None
    if not isinstance(assets, list):
        raise UserFacingError("Media manifest must contain an assets array. Re-run `inspect PROJECT`.")
    valid: list[dict[str, Any]] = []
    for asset in assets:
        if not isinstance(asset, dict) or not isinstance(asset.get("source"), str):
            continue
        if kinds is not None and asset.get("kind") not in kinds:
            continue
        source = asset["source"]
        if source.startswith("/") or re.match(r"^[A-Za-z]:", source) or "\\" in source or any(part in {"", ".", ".."} for part in source.split("/")):
            raise UserFacingError(f"Media manifest contains an invalid relative source path: {source!r}")
        relative_source = Path(*source.split("/"))
        path = project_path(project, relative_source)
        in_assets = is_within(path, project_path(project, "assets"))
        in_project_root = (
            len(relative_source.parts) == 1
            and path.suffix.lower() in IMAGE_SUFFIXES | VIDEO_SUFFIXES
            and is_within(path, project)
        )
        if not path.is_file() or not (in_assets or in_project_root):
            raise UserFacingError(f"Manifest source is missing or outside supported project media locations: {source}. Re-run `inspect PROJECT`.")
        current_hash = sha256_file(path)
        if asset.get("sha256") != current_hash:
            raise UserFacingError(f"Asset changed since inspection: {source}. Run `inspect PROJECT` again first.")
        valid.append(asset)
    return valid


def run_ffmpeg(args: list[str], *, timeout: int = 900) -> tuple[bool, str]:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return False, f"ffmpeg timed out after {timeout} seconds"
    except OSError as exc:
        return False, str(exc)
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        return False, detail[-1] if detail else f"ffmpeg exited with status {result.returncode}"
    return True, result.stderr


def run_ffmpeg_output(args: list[str], output: Path, *, timeout: int = 900) -> tuple[bool, str]:
    """Render to a work-local temporary file, then atomically replace the derivative."""
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{output.stem}.", suffix=output.suffix, dir=output.parent)
        os.close(descriptor)
    except OSError as exc:
        return False, f"could not create a temporary output in {output.parent}: {exc}"
    command = list(args)
    command[-1] = temporary
    ok, detail = run_ffmpeg(command, timeout=timeout)
    if ok and Path(temporary).stat().st_size == 0:
        ok, detail = False, "ffmpeg produced an empty output"
    if not ok:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError:
            pass
        return False, detail
    try:
        os.replace(temporary, output)
    except OSError as exc:
        try:
            Path(temporary).unlink(missing_ok=True)
        except OSError:
            pass
        return False, f"could not publish derivative {output}: {exc}"
    return True, detail


def decode_heic_output(source: Path, output: Path, *, max_dimension: int = 960) -> tuple[bool, str]:
    """Decode HEIC to a verified, scaled JPEG without touching the source.

    Prefer libheif's converter when available. Otherwise decode the HEIC
    directly with FFmpeg; macOS ``sips`` is intentionally not used because it
    can report success while writing an incomplete or black JPEG for some files.
    Some phone exports retain a ``.HEIC`` filename after converting the
    payload to JPEG, so detect that signature and bypass libheif for those files.
    """
    try:
        with source.open("rb") as handle:
            is_jpeg_payload = handle.read(3) == b"\xff\xd8\xff"
    except OSError as exc:
        return False, f"could not read image source: {exc}"
    heif_convert = None if is_jpeg_payload else shutil.which("heif-convert")
    ffmpeg = require_ffmpeg()
    temporary: str | None = None
    try:
        if heif_convert:
            try:
                descriptor, temporary = tempfile.mkstemp(
                    prefix=f".{output.stem}.heic-source.", suffix=".jpg", dir=output.parent
                )
                os.close(descriptor)
            except OSError as exc:
                return False, f"could not create a temporary output in {output.parent}: {exc}"
            command = [heif_convert, "--quiet", str(source), temporary]
            converter_result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
            converted = Path(temporary)
            if converter_result.returncode != 0 or not converted.is_file() or converted.stat().st_size == 0:
                detail = converter_result.stderr.strip() or converter_result.stdout.strip() or "heif-convert produced no JPEG output"
                return False, detail.splitlines()[-1]
            decode_source = temporary
        else:
            decode_source = str(source)
        scaled = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        if is_jpeg_payload and not heif_convert:
            scaled += ["-f", "mjpeg"]
        scaled += [
            "-i", decode_source,
            "-frames:v", "1", "-vf",
            f"scale={max_dimension}:-2:force_original_aspect_ratio=decrease",
            "-q:v", "2", str(output),
        ]
        ok, detail = run_ffmpeg_output(scaled, output)
        if not ok:
            prefix = "heif-convert output could not be decoded" if heif_convert else "FFmpeg could not decode the HEIC source"
            return False, f"{prefix}: {detail}"
        return True, detail
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    finally:
        if temporary:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass


def load_cached_scenes(project: Path) -> dict[str, Any]:
    path = project_path(project, ANALYSIS_RELATIVE_PATH)
    if not path.is_file():
        return {}
    try:
        data = load_json(path, "scene candidates")
        return data if isinstance(data, dict) else {}
    except UserFacingError:
        return {}


def load_cached_proxies(project: Path) -> dict[str, Any]:
    path = project_path(project, Path("work/proxies/manifest.json"))
    if not path.is_file():
        return {}
    try:
        data = load_json(path, "proxy manifest")
        return data.get("proxies", {}) if isinstance(data, dict) and isinstance(data.get("proxies"), dict) else {}
    except UserFacingError:
        return {}


def analysis_source_path(project: Path, source: str, kind: str, proxies: dict[str, Any]) -> tuple[Path, str | None]:
    original = project_path(project, Path(*source.split("/")))
    if kind != "video":
        return original, None
    record = proxies.get(source) if isinstance(proxies, dict) else None
    proxy_relative = record.get("proxy_path") if isinstance(record, dict) else None
    if isinstance(proxy_relative, str):
        proxy = project_path(project, Path(*proxy_relative.split("/")))
        if proxy.is_file() and proxy.stat().st_size > 0:
            return proxy, proxy_relative
    return original, None
def command_extract_scenes(args: Any) -> int:
    project = resolve_project(args.project)
    threshold = float(args.threshold)
    min_duration = float(args.min_scene_duration)
    if not 0.0 < threshold < 1.0:
        raise UserFacingError("--threshold must be greater than 0 and less than 1.")
    if not math.isfinite(min_duration) or min_duration < 0:
        raise UserFacingError("--min-scene-duration cannot be negative.")
    assets = get_manifest_assets(project, {"video"})
    cached = load_cached_scenes(project)
    cached_records = cached.get("videos", {}) if isinstance(cached.get("videos"), dict) else {}
    output_records: dict[str, Any] = {}
    failures: list[str] = []
    ffmpeg = shutil.which("ffmpeg")
    proxies = load_cached_proxies(project)

    for asset in assets:
        source = asset["source"]
        digest = asset.get("sha256")
        analysis_path, proxy_relative = analysis_source_path(project, source, "video", proxies)
        prior = cached_records.get(source) if isinstance(cached_records, dict) else None
        if (
            isinstance(prior, dict)
            and prior.get("sha256") == digest
            and prior.get("threshold") == threshold
            and prior.get("min_scene_duration") == min_duration
            and prior.get("analysis_source", source) == (proxy_relative or source)
            and isinstance(prior.get("scenes"), list)
        ):
            output_records[source] = prior
            continue
        if not ffmpeg:
            failures.append(f"{source}: ffmpeg is not installed or not on PATH")
            continue
        path = analysis_path
        duration = asset.get("duration_seconds")
        try:
            duration_value = float(duration)
        except (TypeError, ValueError):
            duration_value = 0.0
        if not math.isfinite(duration_value) or duration_value <= 0:
            failures.append(f"{source}: manifest has no positive duration; run inspect again or check the media file")
            continue

        video_filter = f"select='gt(scene,{threshold:.6g})',showinfo"
        command = [
            ffmpeg, "-hide_banner", "-nostats", "-loglevel", "info", "-i", str(path),
            "-vf", video_filter, "-an", "-f", "null", "-",
        ]
        ok, output = run_ffmpeg(command)
        if not ok:
            failures.append(f"{source}: {output}")
            continue
        points = sorted({float(value) for value in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", output) if 0.0 < float(value) < duration_value})
        filtered: list[float] = []
        for point in points:
            if point < min_duration or duration_value - point < min_duration:
                continue
            if filtered and point - filtered[-1] < min_duration:
                continue
            filtered.append(point)
        boundaries = [0.0, *filtered, duration_value]
        scenes: list[dict[str, Any]] = []
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
            if end <= start:
                continue
            key = f"{digest}:{start:.3f}:{end:.3f}"
            scenes.append({
                "scene_id": hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
                "representative_time": round((start + end) / 2, 3),
            })
        output_records[source] = {
            "sha256": digest,
            "analysis_source": proxy_relative or source,
            "threshold": threshold,
            "min_scene_duration": min_duration,
            "scenes": scenes,
        }

    work_path(project, "analysis", create_dir=True)
    output_path = work_path(project, Path("analysis/scene_candidates.json"))
    write_json(output_path, {"generated_by": "video_factory.py extract-scenes", "generated_at": utc_now(), "videos": output_records})
    print(f"Processed {len(output_records)} video(s); scene results: {output_path}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 2
    if not assets:
        print("No video assets were found in media_manifest.json.")
    return 0


def load_cached_frames(project: Path) -> dict[str, Any]:
    path = project_path(project, FRAMES_INDEX_RELATIVE_PATH)
    if not path.is_file():
        return {}
    try:
        data = load_json(path, "frame index")
        return data if isinstance(data, dict) else {}
    except UserFacingError:
        return {}


def select_video_times(asset: dict[str, Any], scenes: dict[str, Any], per_video: int, cap: int) -> list[tuple[float, str | None]]:
    source = asset["source"]
    source_scenes: list[dict[str, Any]] = []
    source_data = scenes.get("videos", {}).get(source) if isinstance(scenes.get("videos"), dict) else None
    if isinstance(source_data, dict) and source_data.get("sha256") == asset.get("sha256") and isinstance(source_data.get("scenes"), list):
        source_scenes = [scene for scene in source_data["scenes"] if isinstance(scene, dict) and isinstance(scene.get("representative_time"), (int, float))]
    if source_scenes:
        if len(source_scenes) > cap:
            indices = sorted({round(index * (len(source_scenes) - 1) / max(1, cap - 1)) for index in range(cap)})
            source_scenes = [source_scenes[index] for index in indices]
        return [(float(scene["representative_time"]), str(scene.get("scene_id") or "")) for scene in source_scenes]
    try:
        duration = float(asset.get("duration_seconds"))
    except (TypeError, ValueError):
        duration = 0.0
    if not math.isfinite(duration) or duration <= 0:
        return []
    count = min(per_video, cap)
    return [((index + 0.5) * duration / count, None) for index in range(count)]


def command_extract_frames(args: Any) -> int:
    project = resolve_project(args.project)
    per_video = int(args.per_video)
    cap = int(args.max_frames_per_video)
    if per_video < 1 or cap < 1:
        raise UserFacingError("--per-video and --max-frames-per-video must be at least 1.")
    ffmpeg = require_ffmpeg()
    assets = get_manifest_assets(project, {"photo", "video"})
    prior_index = load_cached_frames(project)
    prior_records = prior_index.get("assets", {}) if isinstance(prior_index.get("assets"), dict) else {}
    settings = {"per_video": per_video, "max_frames_per_video": cap}
    scene_data = load_cached_scenes(project)
    proxies = load_cached_proxies(project)
    output_records: dict[str, Any] = {}
    failures: list[str] = []

    for asset in assets:
        source = asset["source"]
        digest = asset.get("sha256")
        analysis_path, proxy_relative = analysis_source_path(project, source, asset.get("kind", ""), proxies)
        prior = prior_records.get(source) if isinstance(prior_records, dict) else None
        if asset.get("kind") == "photo":
            sample_times: list[tuple[float | None, str | None]] = [(None, None)]
        else:
            sample_times = [(timestamp, scene_id) for timestamp, scene_id in select_video_times(asset, scene_data, per_video, cap)]
        asset_settings = {
            **settings,
            "analysis_source": proxy_relative or source,
            "samples": [{"time": round(timestamp, 3) if timestamp is not None else None, "scene_id": scene_id} for timestamp, scene_id in sample_times],
        }
        if asset.get("kind") == "photo" and Path(source).suffix.lower() in {".heic", ".heif"}:
            # Include the signature-aware decoder in cache identity so mislabeled
            # JPEG payloads are retried instead of reusing earlier failures.
            asset_settings["still_decoder"] = "signature-aware-heif-v3"
        if not sample_times:
            failures.append(f"{source}: no valid frame sample times; inspect the source duration")
        cached_frames = prior.get("frames") if isinstance(prior, dict) else None
        can_reuse = (
            isinstance(prior, dict)
            and prior.get("sha256") == digest
            and prior.get("settings") == asset_settings
            and isinstance(cached_frames, list)
            and len(cached_frames) == len(sample_times)
            and bool(sample_times)
            and all(
                isinstance(frame, dict)
                and isinstance(frame.get("path"), str)
                and frame["path"].startswith("work/frames/")
                and is_within(project_path(project, Path(*frame["path"].split("/"))), project_path(project, "work/frames"))
                and project_path(project, Path(*frame["path"].split("/"))).is_file()
                for frame in cached_frames
            )
        )
        if can_reuse:
            output_records[source] = prior
            continue

        source_path = analysis_path
        directory = work_path(project, Path("frames") / str(digest)[:16], create_dir=True)
        if not is_within(directory, project_path(project, "work/frames")):
            raise UserFacingError(f"Frame output would escape work/frames/: {directory}")
        frames: list[dict[str, Any]] = []
        for index, (timestamp, scene_id) in enumerate(sample_times, start=1):
            suffix = "still" if timestamp is None else f"{round(timestamp * 1000):09d}ms"
            output = directory / f"frame-{index:03d}-{suffix}.jpg"
            if not is_within(output, project_path(project, "work/frames")):
                raise UserFacingError(f"Frame output would escape work/frames/: {output}")
            if timestamp is None and source_path.suffix.lower() == ".heic":
                ok, detail = decode_heic_output(source_path, output)
                if not ok:
                    failures.append(f"{source}: HEIC decode failed: {detail}")
                    continue
            else:
                command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
                if timestamp is not None:
                    command += ["-ss", f"{timestamp:.3f}"]
                command += ["-i", str(source_path), "-frames:v", "1", "-vf", "scale=960:-2:force_original_aspect_ratio=decrease", "-q:v", "2", str(output)]
                ok, detail = run_ffmpeg_output(command, output)
            if not ok:
                failures.append(f"{source} at {timestamp}: {detail}")
                continue
            relative_output = output.relative_to(project).as_posix()
            frames.append({
                "path": relative_output,
                "source": source,
                "source_sha256": digest,
                "analysis_source": proxy_relative or source,
                "time": round(timestamp, 3) if timestamp is not None else None,
                "scene_id": scene_id,
            })
        output_records[source] = {"sha256": digest, "settings": asset_settings, "frames": frames}

    work_path(project, "frames", create_dir=True)
    index_path = work_path(project, Path("frames/frames.json"))
    write_json(index_path, {"generated_by": "video_factory.py extract-frames", "generated_at": utc_now(), "settings": settings, "assets": output_records})
    frame_count = sum(len(record.get("frames", [])) for record in output_records.values())
    print(f"Extracted or reused {frame_count} frame(s) from {len(output_records)} asset(s). Index: {index_path}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 2
    return 0


def command_extract_audio(args: Any) -> int:
    project = resolve_project(args.project)
    channels = int(args.channels)
    sample_rate = int(args.sample_rate)
    if channels not in {1, 2}:
        raise UserFacingError("--channels must be 1 (mono) or 2 (stereo).")
    if sample_rate < 8000 or sample_rate > 192000:
        raise UserFacingError("--sample-rate must be between 8000 and 192000 Hz.")
    settings = {"format": args.format, "channels": channels, "sample_rate": sample_rate, "stream": "first-audio"}
    settings_key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    candidates = [
        asset for asset in get_manifest_assets(project, {"video"})
        if asset.get("kind") == "video" and asset.get("has_audio") is True
    ]
    index_path = work_path(project, Path("transcripts/audio/index.json"))
    prior_index: dict[str, Any] = {}
    if index_path.is_file():
        try:
            loaded = load_json(index_path, "extracted-audio index")
            prior_index = loaded if isinstance(loaded, dict) else {}
        except UserFacingError:
            prior_index = {}
    prior_records = prior_index.get("assets", {}) if isinstance(prior_index.get("assets"), dict) else {}
    output_dir = work_path(project, "transcripts/audio", create_dir=True)
    if not is_within(output_dir, project_path(project, "work/transcripts")):
        raise UserFacingError("Extracted audio must stay under work/transcripts/audio/.")

    if not candidates:
        write_json(index_path, {"generated_by": "video_factory.py extract-audio", "generated_at": utc_now(), "settings": settings, "assets": {}})
        print(f"No audio-bearing video sources found. Wrote an empty index: {index_path}")
        return 0

    ffmpeg = require_ffmpeg()
    records: dict[str, Any] = {}
    failures: list[str] = []
    for asset in candidates:
        source = asset["source"]
        digest = asset.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            failures.append(f"{source}: manifest has no valid source SHA-256; run inspect again")
            continue
        extension = "flac" if args.format == "flac" else "wav"
        output = output_dir / f"{digest}-{settings_key}.{extension}"
        if not is_within(output, project_path(project, "work/transcripts/audio")):
            raise UserFacingError(f"Extracted audio path escapes work/: {output}")
        prior = prior_records.get(source) if isinstance(prior_records, dict) else None
        cache_hit = False
        if (
            isinstance(prior, dict)
            and prior.get("source_sha256") == digest
            and prior.get("settings") == settings
            and prior.get("output") == output.relative_to(project).as_posix()
            and prior.get("output_sha256")
            and output.is_file()
            and not output.is_symlink()
        ):
            try:
                cache_hit = sha256_file(output) == prior.get("output_sha256")
            except UserFacingError:
                cache_hit = False
        if not cache_hit:
            source_path = project_path(project, Path(*source.split("/")))
            codec = "flac" if args.format == "flac" else "pcm_s16le"
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_path),
                "-map", "0:a:0", "-vn", "-ac", str(channels), "-ar", str(sample_rate),
                "-map_metadata", "-1", "-c:a", codec,
            ]
            if args.format == "wav":
                command += ["-f", "wav"]
            command.append(str(output))
            ok, detail = run_ffmpeg_output(command, output)
            if not ok:
                failures.append(f"{source}: {detail}")
                continue
        try:
            output_metadata = ffprobe_metadata(output)
            output_hash = sha256_file(output)
        except UserFacingError as exc:
            failures.append(f"{source}: extracted audio could not be verified: {exc}")
            continue
        records[source] = {
            "source_sha256": digest,
            "settings": settings,
            "output": output.relative_to(project).as_posix(),
            "output_sha256": output_hash,
            "duration_seconds": output_metadata.get("duration_seconds"),
            "audio_streams": output_metadata.get("audio_streams", []),
            "cache_reused": cache_hit,
        }

    write_json(index_path, {"generated_by": "video_factory.py extract-audio", "generated_at": utc_now(), "settings": settings, "assets": records})
    reused = sum(1 for record in records.values() if record.get("cache_reused"))
    print(f"Extracted audio from {len(records)} video(s); reused {reused} cached output(s). Index: {index_path}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 2
    return 0


def command_build_contact_sheets(args: Any) -> int:
    project = resolve_project(args.project)
    columns = int(args.columns)
    tile_width = int(args.tile_width)
    if columns < 1 or columns > 12:
        raise UserFacingError("--columns must be between 1 and 12.")
    if tile_width < 80 or tile_width > 1000:
        raise UserFacingError("--tile-width must be between 80 and 1000 pixels.")
    ffmpeg = require_ffmpeg()
    frame_index_path = project_path(project, FRAMES_INDEX_RELATIVE_PATH)
    if not frame_index_path.is_file():
        raise UserFacingError("Frame index is missing. Run `extract-frames PROJECT` first.")
    frame_index = load_json(frame_index_path, "frame index")
    frame_records = frame_index.get("assets", {}) if isinstance(frame_index, dict) else {}
    frames: list[dict[str, Any]] = []
    for source, record in frame_records.items() if isinstance(frame_records, dict) else []:
        if not isinstance(record, dict) or not isinstance(record.get("frames"), list):
            continue
        for frame in record["frames"]:
            if not isinstance(frame, dict) or not isinstance(frame.get("path"), str):
                continue
            if frame["path"].startswith("/") or "\\" in frame["path"] or any(part in {"", ".", ".."} for part in frame["path"].split("/")):
                raise UserFacingError(f"Contact sheet index contains an invalid path: {frame['path']!r}")
            frame_path = project_path(project, Path(*frame["path"].split("/")))
            if not is_within(frame_path, project_path(project, "work/frames")):
                raise UserFacingError(f"Refusing to include a frame outside work/frames/: {frame['path']}")
            if not frame_path.is_file():
                raise UserFacingError(f"Frame file is missing: {frame['path']}. Re-run `extract-frames PROJECT`.")
            frames.append(frame)
    if not frames:
        raise UserFacingError("No extracted frames are available. Run `extract-frames PROJECT` first.")

    output_dir = work_path(project, "contact-sheets", create_dir=True)
    if not is_within(output_dir, project_path(project, "work")):
        raise UserFacingError("Contact sheet output must stay inside work/.")
    tile_height = max(1, round(tile_width * 9 / 16))
    per_sheet = columns * 5
    sheet_records: list[dict[str, Any]] = []
    for sheet_number, offset in enumerate(range(0, len(frames), per_sheet), start=1):
        chunk = frames[offset:offset + per_sheet]
        filters: list[str] = []
        layouts: list[str] = []
        for index in range(len(chunk)):
            filters.append(
                f"[{index}:v]scale={tile_width}:{tile_height}:force_original_aspect_ratio=decrease,"
                f"pad={tile_width}:{tile_height}:(ow-iw)/2:(oh-ih)/2:black[v{index}]"
            )
            x = (index % columns) * tile_width
            y = (index // columns) * tile_height
            layouts.append(f"{x}_{y}")
        inputs = "".join(f"[v{index}]" for index in range(len(chunk)))
        if len(chunk) == 1:
            filters.append("[v0]null[grid]")
        else:
            filters.append(f"{inputs}xstack=inputs={len(chunk)}:layout={'|'.join(layouts)}:fill=black[grid]")
        output_path = output_dir / f"sheet-{sheet_number:03d}.jpg"
        if not is_within(output_path, project_path(project, "work")):
            raise UserFacingError(f"Contact sheet path escapes work/: {output_path}")
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        for frame in chunk:
            input_path = project_path(project, Path(*frame["path"].split("/")))
            command += ["-i", str(input_path)]
        command += [
            "-filter_complex", ";".join(filters), "-map", "[grid]", "-frames:v", "1", "-q:v", "3", str(output_path),
        ]
        ok, detail = run_ffmpeg_output(command, output_path)
        if not ok:
            raise UserFacingError(f"Could not build contact sheet {output_path.name}: {detail}")
        sheet_records.append({
            "sheet": output_path.relative_to(project).as_posix(),
            "cells": [
                {"cell": index + 1, "source": frame.get("source"), "time": frame.get("time"), "scene_id": frame.get("scene_id"), "frame": frame.get("path")}
                for index, frame in enumerate(chunk)
            ],
        })

    index_path = work_path(project, Path("contact-sheets/index.json"))
    write_json(index_path, {"generated_at": utc_now(), "columns": columns, "tile_width": tile_width, "tile_height": tile_height, "sheets": sheet_records})
    print(f"Built {len(sheet_records)} contact sheet(s) from {len(frames)} frame(s) under {output_dir}.")
    print(f"Cell index: {index_path}")
    return 0


def command_media(args: Any) -> int:
    if args.command == "extract-scenes":
        return command_extract_scenes(args)
    if args.command == "extract-frames":
        return command_extract_frames(args)
    if args.command == "extract-audio":
        return command_extract_audio(args)
    if args.command == "build-contact-sheets":
        return command_build_contact_sheets(args)
    raise UserFacingError(f"Unknown media helper command: {args.command}")
