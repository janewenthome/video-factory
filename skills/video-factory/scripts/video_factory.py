#!/usr/bin/env python3
"""Deterministic project/media helpers and approved Colab inference for video-factory.

This CLI deliberately uses only the Python standard library. ffprobe/ffmpeg and
optional local image metadata tools are discovered by executable path, never by
shell aliases.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
TEMPLATE_PATH = SKILL_DIR / "templates" / "job.template.yaml"
PROFILES_DIR = SKILL_DIR / "profiles"
PLAN_RELATIVE_PATH = Path("work/edit-plan/edit_plan.json")
MANIFEST_RELATIVE_PATH = Path("work/manifests/media_manifest.json")
CACHE_RELATIVE_PATH = Path("work/manifests/.media_metadata_cache.json")

VIDEO_SUFFIXES = {".3gp", ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".mts", ".m2ts", ".webm"}
IMAGE_SUFFIXES = {".avif", ".bmp", ".gif", ".heic", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
AUDIO_SUFFIXES = {".aac", ".aif", ".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}
KNOWN_SEGMENT_TYPES = {
    "photo", "video", "title", "text_card", "subtitle", "narration", "music",
    "natural_audio", "transition", "overlay", "annotation",
}
SOURCE_REQUIRED_TYPES = {"photo", "video", "narration", "music", "natural_audio"}
SOURCE_RANGE_REQUIRED_TYPES = {"video", "narration", "music", "natural_audio"}
TEXT_REQUIRED_TYPES = {"title", "text_card", "subtitle"}


class UserFacingError(Exception):
    """An actionable error suitable for a CLI user."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def resolve_project(value: str | Path, *, create: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve(strict=False)
    if not path.exists():
        if not create:
            raise UserFacingError(f"Project folder does not exist: {path}\nCreate it first, then run `init PROJECT`.")
        if not path.parent.is_dir():
            raise UserFacingError(f"Project parent folder does not exist: {path.parent}")
        try:
            path.mkdir()
        except OSError as exc:
            raise UserFacingError(f"Could not create project folder {path}: {exc}") from exc
    if not path.is_dir():
        raise UserFacingError(f"Project path is not a folder: {path}")
    return path


def project_path(project: Path, relative: str | Path, *, create_dir: bool = False) -> Path:
    """Resolve a project-relative path while rejecting symlink escapes."""
    rel = Path(relative)
    if rel.is_absolute():
        raise UserFacingError(f"Expected a project-relative path, got: {relative}")
    target = project / rel
    if not is_within(target, project):
        raise UserFacingError(f"Project path escapes the project folder: {relative}")
    if create_dir:
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise UserFacingError(f"Could not create project directory {target}: {exc}") from exc
    if not is_within(target, project):
        raise UserFacingError(f"Project path escapes the project folder: {relative}")
    return target


def work_path(project: Path, relative: str | Path, *, create_dir: bool = False) -> Path:
    work_root = project / "work"
    if work_root.is_symlink():
        raise UserFacingError("work/ must be a real directory inside the project, not a symlink.")
    target = project_path(project, Path("work") / relative, create_dir=create_dir)
    if not is_within(target, work_root):
        raise UserFacingError(f"Generated output must stay under work/: {relative}")
    return target


def get_cloud_processing_setting(project: Path, service: str) -> str | None:
    """Read one value from the small, top-level cloud_processing job block.

    This intentionally parses only the project's constrained two-key block,
    avoiding a new YAML dependency in the standard-library CLI. Unsupported or
    ambiguous syntax fails closed instead of authorizing a transfer.
    """
    if service not in {"colab", "openai"}:
        raise UserFacingError("Unknown cloud-processing service.")
    job_path = project_path(project, "job.yaml")
    try:
        lines = job_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as exc:
        raise UserFacingError("Could not read job.yaml cloud-processing policy.") from exc

    section_indexes = []
    for index, raw_line in enumerate(lines):
        line = raw_line.split("#", 1)[0].rstrip()
        if line == "cloud_processing:":
            section_indexes.append(index)
    if not section_indexes:
        return None
    if len(section_indexes) != 1:
        raise UserFacingError("job.yaml has duplicate cloud_processing sections; resolve them before cloud transfer.")

    values: dict[str, str] = {}
    for raw_line in lines[section_indexes[0] + 1 :]:
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if not raw_line[0].isspace():
            break
        match = re.fullmatch(r" {2}(colab|openai):\s*(.*)", raw_line)
        if not match:
            raise UserFacingError("job.yaml cloud_processing block must contain only indented colab/openai settings.")
        key, raw_value = match.groups()
        value = raw_value.split("#", 1)[0].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key in values:
            raise UserFacingError(f"job.yaml cloud_processing has duplicate {key} settings.")
        if value not in {"ask_each_run", "disabled"}:
            raise UserFacingError(f"job.yaml cloud_processing.{key} must be ask_each_run or disabled.")
        values[key] = value
    return values.get(service)


def load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise UserFacingError(f"{description} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise UserFacingError(f"{description} is not valid JSON ({path}:{exc.lineno}:{exc.colno}): {exc.msg}") from exc
    except OSError as exc:
        raise UserFacingError(f"Could not read {description} {path}: {exc}") from exc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not is_within(path, path.parent):
        raise UserFacingError(f"Refusing to write through a path outside its parent: {path}")
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temp_name = handle.name
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except OSError as exc:
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise UserFacingError(f"Could not write {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise UserFacingError(f"Could not read source file {path}: {exc}") from exc
    return digest.hexdigest()


def parse_ratio(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*:\s*(\d+)\s*", value)
    if not match:
        raise UserFacingError(f"Aspect ratio must use WIDTH:HEIGHT notation, for example 16:9; got {value!r}.")
    width, height = (int(part) for part in match.groups())
    if width <= 0 or height <= 0:
        raise UserFacingError(f"Aspect ratio values must be positive: {value!r}.")
    return width, height


def safe_relative_source(project: Path, source: Any, location: str) -> tuple[str, Path]:
    if not isinstance(source, str) or not source.strip():
        raise UserFacingError(f"{location}: source must be a non-empty project-relative path.")
    if "\\" in source:
        raise UserFacingError(f"{location}: use POSIX path separators in source: {source!r}.")
    posix = PurePosixPath(source)
    raw_parts = source.split("/")
    if posix.is_absolute() or re.match(r"^[A-Za-z]:", source) or any(part in {"", ".", ".."} for part in raw_parts):
        raise UserFacingError(f"{location}: source must stay inside the project and cannot be absolute or contain '..': {source!r}.")
    canonical = posix.as_posix()
    path = project / Path(*posix.parts)
    if not is_within(path, project):
        raise UserFacingError(f"{location}: source resolves outside the project (possibly through a symlink): {source!r}.")
    if not path.is_file():
        raise UserFacingError(f"{location}: source file does not exist: {source!r}.")
    return canonical, path.resolve(strict=True)


def available_profiles() -> list[str]:
    if not PROFILES_DIR.is_dir():
        return []
    return sorted(path.stem for path in PROFILES_DIR.glob("*.yaml") if path.is_file())


def profile_in_template(template: str) -> str | None:
    match = re.search(r"(?m)^profile\s*:\s*['\"]?([A-Za-z0-9_-]+)", template)
    return match.group(1) if match else None


def set_template_profile(template: str, profile: str) -> str:
    line = re.compile(r"(?m)^profile\s*:\s*.*$")
    replacement = f"profile: {profile}"
    if line.search(template):
        return line.sub(replacement, template, count=1)
    return f"profile: {profile}\n\n{template.lstrip()}"


def command_init(args: argparse.Namespace) -> int:
    project = resolve_project(args.project, create=True)
    if not TEMPLATE_PATH.is_file():
        raise UserFacingError(f"Job template is missing: {TEMPLATE_PATH}")
    try:
        template = TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise UserFacingError(f"Could not read job template {TEMPLATE_PATH}: {exc}") from exc

    requested = args.profile or profile_in_template(template)
    if requested is None:
        raise UserFacingError("No default profile is defined in the job template; pass --profile to choose one.")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", requested):
        raise UserFacingError(f"Invalid profile name: {requested!r}")
    profiles = available_profiles()
    if profiles and requested not in profiles:
        raise UserFacingError(f"Unknown profile {requested!r}. Available profiles: {', '.join(profiles)}")
    template = set_template_profile(template, requested)

    directories = (
        "assets/photos", "assets/videos", "assets/music", "assets/narration", "assets/references",
        "work/manifests", "work/frames", "work/contact-sheets", "work/transcripts", "work/analysis",
        "work/edit-plan", "outputs",
    )
    for directory in directories:
        target = project_path(project, directory, create_dir=True)
        if not is_within(target, project):
            raise UserFacingError(f"Directory resolves outside the project: {directory}")

    job_path = project_path(project, "job.yaml")
    draft_path = project_path(project, "job.draft.yaml")
    created = False
    destination = job_path if job_path.exists() else draft_path
    if not job_path.exists() and not draft_path.exists():
        try:
            with destination.open("x", encoding="utf-8") as handle:
                handle.write(template.rstrip() + "\n")
            created = True
        except FileExistsError:
            pass
        except OSError as exc:
            raise UserFacingError(f"Could not create {destination}: {exc}") from exc

    if created:
        print(f"Initialized {project}")
        print(f"Created job.draft.yaml using profile {requested!r}; review it before renaming it to job.yaml.")
    else:
        kept = "job.yaml" if job_path.exists() else "job.draft.yaml"
        print(f"Prepared project folders in {project}; existing {kept} was preserved.")
    print("Created or confirmed assets/, work/, and outputs/ subfolders; no source files were changed.")
    return 0


def parse_float(value: Any) -> float | None:
    try:
        if value is None or value == "N/A":
            return None
        result = float(value)
        return result if result == result and abs(result) != float("inf") else None
    except (TypeError, ValueError):
        return None


def parse_plan_number(value: Any) -> float | None:
    """JSON edit-plan numbers must be numeric values, not strings or booleans."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return parse_float(value)


def parse_rate(value: Any) -> float | None:
    if not isinstance(value, str) or not value or value == "0/0":
        return None
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        top, bottom = parse_float(numerator), parse_float(denominator)
        return top / bottom if top is not None and bottom else None
    return parse_float(value)


def parse_exif_datetime(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = dt.datetime.strptime(value, fmt)
            if parsed.tzinfo is None:
                return parsed.isoformat(timespec="seconds")
            return parsed.isoformat(timespec="seconds")
        except ValueError:
            continue
    return value


def run_json_command(command: list[str], *, timeout: int = 60) -> dict[str, Any] | None:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def ffprobe_metadata(path: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise UserFacingError("ffprobe is not installed or not on PATH. Install FFmpeg, which provides both ffmpeg and ffprobe.")
    data = run_json_command([
        ffprobe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path.resolve()),
    ], timeout=180)
    if data is None:
        raise UserFacingError("ffprobe could not read this media file.")
    streams = data.get("streams") if isinstance(data.get("streams"), list) else []
    video_streams = [s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    audio_streams = [s for s in streams if isinstance(s, dict) and s.get("codec_type") == "audio"]
    format_data = data.get("format") if isinstance(data.get("format"), dict) else {}
    metadata: dict[str, Any] = {
        "duration_seconds": parse_float(format_data.get("duration")),
        "format_name": format_data.get("format_name"),
        "video_streams": [],
        "audio_streams": [],
        "has_audio": bool(audio_streams),
        "captured_at": None,
    }
    for stream in video_streams:
        tags = stream.get("tags") if isinstance(stream.get("tags"), dict) else {}
        side_data = stream.get("side_data_list") if isinstance(stream.get("side_data_list"), list) else []
        rotation = parse_float(tags.get("rotate"))
        if rotation is None:
            for item in side_data:
                if isinstance(item, dict) and item.get("rotation") is not None:
                    rotation = parse_float(item.get("rotation"))
                    break
        item = {
            "codec": stream.get("codec_name"),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "frame_rate": parse_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
            "pixel_format": stream.get("pix_fmt"),
            "rotation_degrees": rotation,
        }
        metadata["video_streams"].append(item)
    for stream in audio_streams:
        tags = stream.get("tags") if isinstance(stream.get("tags"), dict) else {}
        metadata["audio_streams"].append({
            "codec": stream.get("codec_name"),
            "channels": stream.get("channels"),
            "channel_layout": stream.get("channel_layout"),
            "sample_rate": parse_float(stream.get("sample_rate")),
            "language": tags.get("language"),
        })
    tag_sources = [format_data.get("tags", {})] + [s.get("tags", {}) for s in streams if isinstance(s, dict)]
    for tags in tag_sources:
        if not isinstance(tags, dict):
            continue
        for key in ("DateTimeOriginal", "creation_time", "date", "com.apple.quicktime.creationdate", "CreateDate"):
            if tags.get(key):
                metadata["captured_at"] = parse_exif_datetime(tags[key])
                break
        if metadata["captured_at"]:
            break
    if video_streams:
        first = metadata["video_streams"][0]
        coded_width, coded_height = first["width"], first["height"]
        rotation = first["rotation_degrees"]
        rotates_quarter_turn = rotation is not None and abs(round(rotation)) % 180 == 90
        metadata["width"] = coded_height if rotates_quarter_turn else coded_width
        metadata["height"] = coded_width if rotates_quarter_turn else coded_height
        metadata["frame_rate"] = first["frame_rate"]
        metadata["rotation_degrees"] = rotation
        metadata["orientation"] = "landscape" if (metadata["width"] or 0) >= (metadata["height"] or 0) else "portrait"
    return metadata


def image_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"width": None, "height": None, "orientation": None, "captured_at": None, "metadata_sources": []}

    exiftool = shutil.which("exiftool")
    if exiftool:
        try:
            completed = subprocess.run([exiftool, "-j", "-n", "-DateTimeOriginal", "-CreateDate", "-ImageWidth", "-ImageHeight", "-Orientation", str(path.resolve())], capture_output=True, text=True, timeout=30, check=False)
            records = json.loads(completed.stdout) if completed.returncode == 0 else []
            if records and isinstance(records[0], dict):
                record = records[0]
                result["width"] = record.get("ImageWidth")
                result["height"] = record.get("ImageHeight")
                result["orientation"] = record.get("Orientation")
                result["captured_at"] = parse_exif_datetime(record.get("DateTimeOriginal") or record.get("CreateDate"))
                result["metadata_sources"].append("exiftool")
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            pass

    try:
        from PIL import Image, ExifTags  # type: ignore[import-not-found]

        with Image.open(path) as image:
            result["width"], result["height"] = image.size
            exif = image.getexif()
            named = {ExifTags.TAGS.get(key, str(key)): value for key, value in exif.items()}
            orientation_tag = named.get("Orientation")
            if orientation_tag is not None:
                result["orientation"] = orientation_tag
            capture = named.get("DateTimeOriginal") or named.get("DateTimeDigitized") or named.get("DateTime")
            result["captured_at"] = result["captured_at"] or parse_exif_datetime(capture)
            result["metadata_sources"].append("Pillow")
    except (ImportError, OSError, ValueError):
        pass

    if result["width"] is None or result["height"] is None:
        sips = shutil.which("sips")
        if sips:
            try:
                completed = subprocess.run([sips, "-g", "pixelWidth", "-g", "pixelHeight", "-g", "orientation", str(path.resolve())], capture_output=True, text=True, timeout=30, check=False)
                if completed.returncode == 0:
                    for line in completed.stdout.splitlines():
                        key, sep, value = line.partition(":")
                        if not sep:
                            continue
                        try:
                            if key.strip() == "pixelWidth":
                                result["width"] = int(value.strip())
                            elif key.strip() == "pixelHeight":
                                result["height"] = int(value.strip())
                            elif key.strip() == "orientation":
                                result["orientation"] = value.strip()
                        except ValueError:
                            pass
                    result["metadata_sources"].append("sips")
            except (OSError, subprocess.TimeoutExpired):
                pass

    if result["width"] and result["height"] and isinstance(result["width"], (int, float)) and isinstance(result["height"], (int, float)):
        if isinstance(result["orientation"], int):
            result["orientation"] = "portrait" if result["orientation"] in {5, 6, 7, 8} else "landscape" if result["orientation"] in {1, 2, 3, 4} else str(result["orientation"])
        elif not result["orientation"]:
            result["orientation"] = "landscape" if result["width"] >= result["height"] else "portrait"
    return result


def collect_media_files(project: Path) -> list[tuple[str, Path]]:
    assets = project_path(project, "assets")
    if not assets.exists():
        raise UserFacingError(f"assets/ is missing in {project}. Run `init PROJECT` first.")
    if not assets.is_dir() or not is_within(assets, project):
        raise UserFacingError("assets/ must be a folder inside the project.")
    files: list[tuple[str, Path]] = []
    for path in sorted(assets.rglob("*")):
        if not path.is_file():
            continue
        if not is_within(path, project) or not is_within(path, assets):
            raise UserFacingError(f"Asset path escapes assets/ through a symlink: {path.relative_to(project)}")
        source = path.relative_to(project).as_posix()
        files.append((source, path.resolve(strict=True)))
    return files


def command_inspect(args: argparse.Namespace) -> int:
    project = resolve_project(args.project)
    work_path(project, "manifests", create_dir=True)
    manifest_path = work_path(project, Path("manifests/media_manifest.json"))
    cache_path = work_path(project, Path("manifests/.media_metadata_cache.json"))
    cache = load_json(cache_path, "metadata cache") if cache_path.is_file() else {}
    cache_entries = (
        cache.get("entries", {})
        if isinstance(cache, dict) and cache.get("cache_version") == 2
        else {}
    )
    if not isinstance(cache_entries, dict):
        cache_entries = {}

    assets: list[dict[str, Any]] = []
    new_cache: dict[str, Any] = {}
    errors: list[str] = []
    for source, path in collect_media_files(project):
        try:
            digest = sha256_file(path)
        except UserFacingError as exc:
            errors.append(f"{source}: {exc}")
            continue
        cache_record = cache_entries.get(digest)
        reused = (
            isinstance(cache_record, dict)
            and isinstance(cache_record.get("metadata"), dict)
            and not cache_record["metadata"].get("probe_error")
        )
        if reused:
            metadata = dict(cache_record["metadata"])
        else:
            suffix = path.suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                metadata = image_metadata(path)
                probe = shutil.which("ffprobe")
                if probe:
                    probe_result = run_json_command([probe, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path)], timeout=90)
                    streams = probe_result.get("streams", []) if probe_result else []
                    if probe_result and isinstance(probe_result.get("format"), dict):
                        tags = probe_result["format"].get("tags", {})
                        if isinstance(tags, dict):
                            metadata["captured_at"] = next((parse_exif_datetime(tags[key]) for key in ("DateTimeOriginal", "creation_time", "date", "CreateDate") if tags.get(key)), metadata.get("captured_at"))
                    if isinstance(streams, list):
                        metadata["has_audio"] = any(isinstance(stream, dict) and stream.get("codec_type") == "audio" for stream in streams)
                        if not metadata.get("width"):
                            video = next((stream for stream in streams if isinstance(stream, dict) and stream.get("codec_type") == "video"), None)
                            if video:
                                metadata["width"], metadata["height"] = video.get("width"), video.get("height")
                                metadata["orientation"] = "landscape" if (video.get("width") or 0) >= (video.get("height") or 0) else "portrait"
                    metadata["metadata_sources"] = sorted(set(metadata.get("metadata_sources", []) + ["ffprobe"]))
                metadata.update({"duration_seconds": None, "frame_rate": None, "video_streams": [], "audio_streams": metadata.get("audio_streams", [])})
                metadata.setdefault("has_audio", False)
            elif suffix in VIDEO_SUFFIXES or suffix in AUDIO_SUFFIXES:
                try:
                    metadata = ffprobe_metadata(path)
                except UserFacingError as exc:
                    errors.append(f"{source}: {exc}")
                    metadata = {"probe_error": str(exc), "video_streams": [], "audio_streams": [], "has_audio": None}
            else:
                metadata = {"video_streams": [], "audio_streams": [], "has_audio": None}
            if not metadata.get("probe_error"):
                new_cache[digest] = {"metadata": metadata, "cached_at": utc_now()}

        suffix = path.suffix.lower()
        kind = "photo" if suffix in IMAGE_SUFFIXES else "video" if suffix in VIDEO_SUFFIXES else "audio" if suffix in AUDIO_SUFFIXES else "other"
        try:
            modified_at = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            size_bytes = path.stat().st_size
        except OSError:
            modified_at = None
            size_bytes = None
        record = {
            "id": f"sha256:{digest}",
            "sha256": digest,
            "source": source,
            "kind": kind,
            "mime_type": mimetypes.guess_type(path.name)[0],
            "size_bytes": size_bytes,
            "modified_at": modified_at,
            "cache_reused": reused,
            **metadata,
        }
        if record.get("width") is not None and record.get("height") is not None:
            record["resolution"] = {"width": record["width"], "height": record["height"]}
        record["metadata_status"] = "error" if record.get("probe_error") else "ok" if any(record.get(key) is not None for key in ("duration_seconds", "width", "height", "captured_at")) else "partial"
        assets.append(record)
        if reused:
            new_cache[digest] = cache_record

    manifest = {
        "manifest_version": 1,
        "generated_at": utc_now(),
        "assets": assets,
        "errors": errors,
    }
    write_json(manifest_path, manifest)
    write_json(cache_path, {"cache_version": 2, "entries": new_cache})
    print(f"Inspected {len(assets)} asset(s); reused cached metadata for {sum(1 for item in assets if item['cache_reused'])}.")
    print(f"Manifest: {manifest_path}")
    if errors:
        print("Some files could not be fully inspected:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    return 0


def media_lookup(project: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    path = project_path(project, MANIFEST_RELATIVE_PATH)
    if not path.is_file():
        return {}, {}
    manifest = load_json(path, "media manifest")
    by_source: dict[str, dict[str, Any]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for item in manifest.get("assets", []) if isinstance(manifest, dict) else []:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("source"), str):
            by_source[item["source"]] = item
        if isinstance(item.get("id"), str):
            by_id[item["id"]] = item
        if isinstance(item.get("sha256"), str):
            by_id[item["sha256"]] = item
            by_id[f"sha256:{item['sha256']}"] = item
    return by_source, by_id


def plan_source_path(project: Path, source: Any, location: str, by_id: dict[str, dict[str, Any]]) -> tuple[str, Path]:
    return safe_relative_source(project, source, location)


def read_job_aspect_ratios(project: Path) -> list[str]:
    job_path = project_path(project, "job.yaml")
    if not job_path.is_file():
        return []
    try:
        job_text = job_path.read_text(encoding="utf-8")
    except OSError:
        return []
    match = re.search(r"(?ms)^aspect_ratios\s*:\s*\n((?:[ \t]+-[^\n]*\n?)+)", job_text)
    if not match:
        inline = re.search(r"(?m)^aspect_ratios\s*:\s*\[([^\]]*)\]", job_text)
        if not inline:
            return []
        return re.findall(r"['\"]?(\d+\s*:\s*\d+)['\"]?", inline.group(1))
    return re.findall(r"['\"]?(\d+\s*:\s*\d+)['\"]?", match.group(1))


def validate_plan_data(project: Path, plan: Any) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(plan, dict):
        return ["Edit plan root must be a JSON object."], warnings
    timeline = plan.get("timeline")
    if not isinstance(timeline, list):
        return ["Edit plan must contain a timeline array."], warnings
    if not timeline:
        errors.append("Edit plan timeline cannot be empty.")
    by_source, _ = media_lookup(project)
    manifest_path = project_path(project, MANIFEST_RELATIVE_PATH)
    if not manifest_path.is_file():
        warnings.append("No media manifest is available; source time ranges cannot be checked against probed media durations. Run inspect first.")
    else:
        manifest = load_json(manifest_path, "media manifest")
        manifest_errors = manifest.get("errors", []) if isinstance(manifest, dict) else []
        if isinstance(manifest_errors, list) and manifest_errors:
            warnings.append(f"Media manifest records {len(manifest_errors)} inspection issue(s); check it before rendering.")
    for index, segment in enumerate(timeline):
        label = f"timeline[{index}]"
        if not isinstance(segment, dict):
            errors.append(f"{label} must be an object.")
            continue
        segment_type = segment.get("type")
        if not isinstance(segment_type, str) or not segment_type.strip():
            errors.append(f"{label}.type must be a non-empty string.")
            continue
        if segment_type not in KNOWN_SEGMENT_TYPES:
            errors.append(f"{label}.type {segment_type!r} is not supported by the edit-plan schema.")

        if segment_type == "transition":
            errors.append(f"{label}: standalone transition segments are not implemented; use clip fade_in_seconds/fade_out_seconds.")
        for field in ("fade_in_seconds", "fade_out_seconds", "volume"):
            if field in segment:
                value = parse_plan_number(segment[field])
                if value is None or value < 0:
                    errors.append(f"{label}.{field} must be a finite non-negative number.")
        for unsupported in ("caption", "transition"):
            if unsupported in segment:
                errors.append(f"{label}.{unsupported} is not implemented by the renderer.")
        annotations = segment.get("annotations")
        if annotations is not None:
            if not isinstance(annotations, list) or len(annotations) > 1:
                errors.append(f"{label}.annotations supports at most one text annotation.")
            else:
                for annotation in annotations:
                    if not isinstance(annotation, dict) or not isinstance(annotation.get("text"), str) or not annotation["text"].strip():
                        errors.append(f"{label}.annotations requires non-empty text.")
                    elif annotation.get("position", "center") not in ("center", "bottom"):
                        errors.append(f"{label}.annotations.position must be center or bottom.")
        audio = segment.get("audio")
        if audio is not None:
            if not isinstance(audio, dict):
                errors.append(f"{label}.audio must be an object.")
            else:
                if set(audio) - {"volume", "preserve_natural"}:
                    errors.append(f"{label}.audio contains unsupported settings.")
                if "volume" in audio and (parse_plan_number(audio["volume"]) is None or audio["volume"] < 0):
                    errors.append(f"{label}.audio.volume must be a finite non-negative number.")
                if "preserve_natural" in audio and not isinstance(audio["preserve_natural"], bool):
                    errors.append(f"{label}.audio.preserve_natural must be boolean.")
        crop = segment.get("crop")
        if crop is not None:
            if not isinstance(crop, dict):
                errors.append(f"{label}.crop must be an object.")
            else:
                crop_entries = [(f"{label}.crop", crop)]
                for ratio_key in ("16:9", "9:16"):
                    if ratio_key in crop:
                        if isinstance(crop[ratio_key], dict):
                            crop_entries.append((f"{label}.crop.{ratio_key}", crop[ratio_key]))
                        else:
                            errors.append(f"{label}.crop.{ratio_key} must be an object.")
                for crop_label, entry in crop_entries:
                    allowed_crop = {"object_position", "fit", "zoom_start", "zoom_end"}
                    if entry is crop:
                        allowed_crop |= {"16:9", "9:16"}
                    if set(entry) - allowed_crop:
                        errors.append(f"{crop_label} contains unsupported settings.")
                    if "fit" in entry and entry["fit"] not in ("cover", "contain"):
                        errors.append(f"{crop_label}.fit must be cover or contain.")
                    if "object_position" in entry and (not isinstance(entry["object_position"], str) or not entry["object_position"].strip()):
                        errors.append(f"{crop_label}.object_position must be a string.")
                    for field in ("zoom_start", "zoom_end"):
                        if field in entry and (parse_plan_number(entry[field]) is None or entry[field] <= 0):
                            errors.append(f"{crop_label}.{field} must be a positive finite number.")

        start = parse_plan_number(segment.get("timeline_start"))
        end = parse_plan_number(segment.get("timeline_end"))
        if start is None or end is None:
            errors.append(f"{label} requires numeric timeline_start and timeline_end seconds.")
        elif start < 0 or end <= start:
            errors.append(f"{label} has invalid timeline range: {start} to {end}; end must be greater than start and start cannot be negative.")

        fps = parse_plan_number(plan.get("fps")) or 30.0
        if start is not None and end is not None and round(end * fps) <= round(start * fps):
            errors.append(f"{label} must occupy at least one output frame.")

        if segment_type in TEXT_REQUIRED_TYPES and not (isinstance(segment.get("text"), str) and segment["text"].strip()):
            errors.append(f"{label} ({segment_type}) requires non-empty text.")

        source = segment.get("source")
        if segment_type in SOURCE_REQUIRED_TYPES:
            try:
                relative_source, _ = plan_source_path(project, source, f"{label}.source", {})
                source_in = parse_plan_number(segment.get("source_in"))
                source_out = parse_plan_number(segment.get("source_out"))
                if segment_type in SOURCE_RANGE_REQUIRED_TYPES and (source_in is None or source_out is None):
                    errors.append(f"{label} ({segment_type}) requires numeric source_in and source_out seconds.")
                elif (source_in is None) != (source_out is None):
                    errors.append(f"{label} must either omit both source_in/source_out or provide both.")
                elif source_in is not None and (source_in < 0 or source_out is None or source_out <= source_in):
                    errors.append(f"{label} has invalid source range: {source_in} to {source_out}.")
                elif source_out is not None:
                    if segment_type in SOURCE_RANGE_REQUIRED_TYPES and start is not None and end is not None:
                        fps = parse_plan_number(plan.get("fps")) or 30.0
                        if abs((source_out - source_in) - (end - start)) > 1.0 / max(fps, 1.0) + 1e-6:
                            errors.append(f"{label}: source and timeline durations must match; playback-rate changes and looping are not implemented.")
                    media = by_source.get(relative_source)
                    duration = parse_float(media.get("duration_seconds")) if media else None
                    if duration is not None and source_out > duration + 0.05:
                        errors.append(f"{label}.source_out ({source_out}) exceeds source duration ({duration:.3f}s).")
            except UserFacingError as exc:
                errors.append(str(exc))
        elif source not in (None, ""):
            try:
                plan_source_path(project, source, f"{label}.source", {})
            except UserFacingError as exc:
                errors.append(str(exc))

    references = plan.get("references", [])
    if not isinstance(references, list):
        errors.append("Edit plan references must be an array.")
        references = []
    reference_ids: set[str] = set()
    for index, reference in enumerate(references):
        label = f"references[{index}]"
        if not isinstance(reference, dict):
            errors.append(f"{label} must be an object.")
            continue
        for field in ("id", "title", "source_type"):
            if not isinstance(reference.get(field), str) or not reference[field].strip():
                errors.append(f"{label}.{field} must be a non-empty string.")
        if isinstance(reference.get("id"), str):
            if reference["id"] in reference_ids:
                errors.append(f"Duplicate reference id: {reference['id']!r}.")
            reference_ids.add(reference["id"])
        if reference.get("source_type") not in {"user_provided_material", "project_reference_file"}:
            errors.append(f"{label}.source_type must be user_provided_material or project_reference_file.")
        try:
            safe_relative_source(project, reference.get("path"), f"{label}.path")
        except UserFacingError as exc:
            errors.append(str(exc))
    for index, segment in enumerate(timeline):
        if not isinstance(segment, dict) or segment.get("medical_claims") is None:
            continue
        claims = segment.get("medical_claims")
        if not isinstance(claims, list):
            errors.append(f"timeline[{index}].medical_claims must be an array.")
            continue
        for claim_index, claim in enumerate(claims):
            label = f"timeline[{index}].medical_claims[{claim_index}]"
            if not isinstance(claim, dict):
                errors.append(f"{label} must be an object.")
                continue
            if not isinstance(claim.get("statement"), str) or not claim["statement"].strip():
                errors.append(f"{label}.statement must be a non-empty string.")
            claim_refs = claim.get("reference_ids")
            if not isinstance(claim_refs, list) or not claim_refs or any(not isinstance(value, str) or not value for value in claim_refs):
                errors.append(f"{label}.reference_ids must be a non-empty list of reference IDs.")
            else:
                missing_refs = [value for value in claim_refs if value not in reference_ids]
                if missing_refs:
                    errors.append(f"{label} refers to unknown reference ID(s): {', '.join(missing_refs)}.")

    ratio_values: list[str] = []
    if "aspect_ratio" in plan:
        ratio_values.append(plan["aspect_ratio"])
    if isinstance(plan.get("aspect_ratios"), list):
        ratio_values.extend(plan["aspect_ratios"])
    output = plan.get("output")
    if isinstance(output, dict):
        if "aspect_ratio" in output:
            ratio_values.append(output["aspect_ratio"])
        if isinstance(output.get("aspect_ratios"), list):
            ratio_values.extend(output["aspect_ratios"])
    for index, ratio in enumerate(ratio_values):
        if not isinstance(ratio, str):
            errors.append(f"plan aspect ratio #{index + 1} must be a WIDTH:HEIGHT string.")
            continue
        try:
            parse_ratio(ratio)
        except UserFacingError as exc:
            errors.append(str(exc))
    allowed_ratios = read_job_aspect_ratios(project)
    for index, ratio in enumerate(allowed_ratios):
        try:
            parse_ratio(ratio)
        except UserFacingError as exc:
            errors.append(f"job.yaml aspect_ratios entry #{index + 1} is invalid: {exc}")
    if allowed_ratios and ratio_values:
        allowed_normalized = {re.sub(r"\s+", "", value) for value in allowed_ratios}
        for ratio in ratio_values:
            if isinstance(ratio, str) and re.sub(r"\s+", "", ratio) not in allowed_normalized:
                errors.append(f"Plan aspect ratio {ratio!r} is not enabled by job.yaml: {', '.join(allowed_ratios)}.")
    declared_duration = parse_plan_number(plan.get("duration_seconds"))
    if plan.get("duration_seconds") is not None and (declared_duration is None or declared_duration <= 0):
        errors.append("duration_seconds must be a positive number when provided.")
    timeline_end = max((parse_plan_number(item.get("timeline_end")) or 0.0 for item in timeline if isinstance(item, dict)), default=0.0)
    if declared_duration is not None and declared_duration + 1e-6 < timeline_end:
        errors.append(f"duration_seconds ({declared_duration}) is shorter than the final timeline_end ({timeline_end}).")
    declared_fps = parse_plan_number(plan.get("fps"))
    if plan.get("fps") is not None and (declared_fps is None or declared_fps <= 0):
        errors.append("fps must be a positive number when provided.")
    width = parse_plan_number(plan.get("width"))
    height = parse_plan_number(plan.get("height"))
    if (plan.get("width") is None) != (plan.get("height") is None):
        errors.append("width and height must either both be provided or both be omitted.")
    elif plan.get("width") is not None and (width is None or height is None or width <= 0 or height <= 0):
        errors.append("width and height must be positive numbers when provided.")
    elif width is not None and height is not None:
        if not width.is_integer() or not height.is_integer() or int(width) % 2 or int(height) % 2:
            errors.append("width and height must be positive even integers for H.264 output.")
        if "aspect_ratio" in plan and isinstance(plan["aspect_ratio"], str):
            try:
                ratio_width, ratio_height = parse_ratio(plan["aspect_ratio"])
                if abs(width / height - ratio_width / ratio_height) > 0.02:
                    errors.append(f"Plan width/height ({width:g}x{height:g}) do not match aspect_ratio {plan['aspect_ratio']}.")
            except UserFacingError:
                pass
    return errors, warnings


def command_validate_plan(args: argparse.Namespace) -> int:
    project = resolve_project(args.project)
    path = project_path(project, PLAN_RELATIVE_PATH)
    plan = load_json(path, "edit plan")
    errors, warnings = validate_plan_data(project, plan)
    if errors:
        print(f"Edit plan has {len(errors)} error(s):")
        for error in errors:
            print(f"ERROR: {error}")
        for warning in warnings:
            print(f"WARNING: {warning}")
        return 2
    print(f"Edit plan is valid: {len(plan['timeline'])} timeline segment(s).")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


def plan_ratio(plan: dict[str, Any], project: Path, requested: str | None) -> str:
    if requested:
        parse_ratio(requested)
        return requested
    candidates: list[Any] = [plan.get("aspect_ratio")]
    ratios = plan.get("aspect_ratios")
    if isinstance(ratios, list):
        candidates.extend(ratios)
    output = plan.get("output")
    if isinstance(output, dict):
        candidates.extend([output.get("aspect_ratio")])
        if isinstance(output.get("aspect_ratios"), list):
            candidates.extend(output["aspect_ratios"])
    candidates.extend(read_job_aspect_ratios(project))
    for candidate in candidates:
        if isinstance(candidate, str):
            try:
                parse_ratio(candidate)
                return candidate
            except UserFacingError:
                continue
    return "16:9"


def stable_asset_filename(project_relative: str, source_path: Path, content_hash: str) -> str:
    key = hashlib.sha256(project_relative.encode("utf-8")).hexdigest()[:16]
    suffix = source_path.suffix.lower()
    if suffix and not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
        suffix = ""
    return f"asset-{key}-{content_hash[:16]}{suffix}"


def copy_source_to_work(source: Path, output: Path, expected_hash: str) -> bool:
    """Copy a source into work and verify it while streaming; return True for a cache hit."""
    if os.path.lexists(output):
        if output.is_symlink() or not output.is_file():
            raise UserFacingError(f"Render asset destination is not a regular file: {output}")
        existing_hash = sha256_file(output)
        if existing_hash != expected_hash:
            raise UserFacingError(f"Existing render asset has unexpected content and was preserved: {output}")
        source_hash = sha256_file(source)
        if source_hash != expected_hash:
            raise UserFacingError(f"Source changed since its content-addressed render copy was prepared: {source}. Run inspect again.")
        return True

    temporary_name: str | None = None
    digest = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile("wb", dir=output.parent, prefix=f".{output.stem}.", suffix=output.suffix, delete=False) as target:
            temporary_name = target.name
            with source.open("rb") as original:
                for chunk in iter(lambda: original.read(1024 * 1024), b""):
                    digest.update(chunk)
                    target.write(chunk)
        if digest.hexdigest() != expected_hash:
            raise UserFacingError(f"Source changed while preparing the render asset: {source}. Run inspect again.")
        os.replace(temporary_name, output)
        temporary_name = None
    except UserFacingError:
        raise
    except OSError as exc:
        raise UserFacingError(f"Could not copy {source} into the Remotion public assets folder: {exc}") from exc
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
    return False


def prepare_heic_render_asset(source: Path, output: Path, expected_source_hash: str) -> bool:
    """Create or reuse a verified JPEG derivative for a HEIC render source."""
    from media_helpers import decode_heic_output

    conversion = "heic-decoder-verified-jpeg-v2"
    max_dimension = 4096
    metadata_path = output.with_suffix(output.suffix + ".json")
    if sha256_file(source) != expected_source_hash:
        raise UserFacingError(f"Source changed while preparing the HEIC render asset: {source}. Run inspect again.")

    if os.path.lexists(output) or os.path.lexists(metadata_path):
        if output.is_symlink() or metadata_path.is_symlink() or not output.is_file() or not metadata_path.is_file():
            raise UserFacingError(f"Existing HEIC render cache is incomplete or unsafe; preserved: {output}")
        metadata = load_json(metadata_path, "HEIC render cache metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("source_sha256") != expected_source_hash
            or metadata.get("conversion") != conversion
            or metadata.get("max_dimension") != max_dimension
            or metadata.get("output_sha256") != sha256_file(output)
        ):
            raise UserFacingError(f"Existing HEIC render cache failed provenance verification; preserved: {output}")
        return True

    temporary_name: str | None = None
    output_created = False
    metadata_created = False
    succeeded = False
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.stem}.", suffix=".jpg", dir=output.parent
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        ok, detail = decode_heic_output(source, temporary, max_dimension=max_dimension)
        if not ok:
            raise UserFacingError(f"Could not decode HEIC for Remotion: {source}: {detail}")
        output_hash = sha256_file(temporary)
        os.replace(temporary, output)
        temporary_name = None
        output_created = True
        write_json(
            metadata_path,
            {
                "source_sha256": expected_source_hash,
                "output_sha256": output_hash,
                "conversion": conversion,
                "max_dimension": max_dimension,
            },
        )
        metadata_created = True
        succeeded = True
    except UserFacingError:
        raise
    except OSError as exc:
        raise UserFacingError(f"Could not prepare HEIC render asset {output}: {exc}") from exc
    finally:
        if temporary_name:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        if not succeeded:
            for generated in (metadata_path if metadata_created else None, output if output_created else None):
                if generated is not None:
                    try:
                        generated.unlink(missing_ok=True)
                    except OSError:
                        pass
    return False


def command_prepare_render(args: argparse.Namespace) -> int:
    project = resolve_project(args.project)
    plan_path = project_path(project, PLAN_RELATIVE_PATH)
    plan = load_json(plan_path, "edit plan")
    errors, warnings = validate_plan_data(project, plan)
    if errors:
        print("Cannot prepare render input because the edit plan has errors:", file=sys.stderr)
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    if not isinstance(plan, dict):
        raise UserFacingError("Edit plan root must be a JSON object.")
    ratio = plan_ratio(plan, project, args.ratio)
    ratio_width, ratio_height = parse_ratio(ratio)
    allowed_ratios = read_job_aspect_ratios(project)
    if allowed_ratios and re.sub(r"\s+", "", ratio) not in {re.sub(r"\s+", "", value) for value in allowed_ratios}:
        raise UserFacingError(f"Requested ratio {ratio!r} is not enabled by job.yaml: {', '.join(allowed_ratios)}")
    if ratio_width >= ratio_height:
        height = 1080
        width = max(1, round(height * ratio_width / ratio_height / 2) * 2)
    else:
        width = 1080
        height = max(1, round(width * ratio_height / ratio_width / 2) * 2)

    if plan.get("width") is not None and plan.get("height") is not None:
        width, height = int(plan["width"]), int(plan["height"])
        if abs(width / height - ratio_width / ratio_height) > 0.02:
            raise UserFacingError(f"Plan width/height ({width}x{height}) do not match selected ratio {ratio}.")

    public_assets = work_path(project, Path("render-public/assets"), create_dir=True)
    if not is_within(public_assets, project_path(project, "work")):
        raise UserFacingError("render-public/assets must be inside work/.")
    by_source, _ = media_lookup(project)
    rewritten_segments: list[dict[str, Any]] = []
    copies = 0
    cache_hits = 0
    for index, raw_segment in enumerate(plan["timeline"]):
        segment = dict(raw_segment)
        source = segment.get("source")
        if source not in (None, ""):
            relative_source, source_path = plan_source_path(project, source, f"timeline[{index}].source", {})
            manifest_entry = by_source.get(relative_source)
            manifest_hash = manifest_entry.get("sha256") if manifest_entry else None
            if isinstance(manifest_hash, str) and re.fullmatch(r"[0-9a-f]{64}", manifest_hash):
                current_hash = manifest_hash
            else:
                current_hash = sha256_file(source_path)
            name = stable_asset_filename(relative_source, source_path, current_hash)
            if segment.get("type") == "photo" and source_path.suffix.lower() == ".heic":
                name = f"{Path(name).stem}-heic-v2.jpg"
            asset_path = public_assets / name
            if not is_within(asset_path, project_path(project, "work")):
                raise UserFacingError(f"Render asset copy would escape work/: {asset_path}")
            if source_path.suffix.lower() == ".heic" and segment.get("type") == "photo":
                cache_hit = prepare_heic_render_asset(source_path, asset_path, current_hash)
            else:
                cache_hit = copy_source_to_work(source_path, asset_path, current_hash)
            if cache_hit:
                cache_hits += 1
            else:
                copies += 1
            segment["source_url"] = f"assets/{name}"
            segment["source_project_path"] = relative_source
        rewritten_segments.append(segment)

    timeline_duration = max((parse_float(item.get("timeline_end")) or 0.0 for item in rewritten_segments if isinstance(item, dict)), default=0.0)
    duration = parse_float(plan.get("duration_seconds")) or timeline_duration
    render_input = {
        "title": plan.get("title") or "",
        "profile": plan.get("profile"),
        "ratio": ratio,
        "width": width,
        "height": height,
        "fps": parse_float(plan.get("fps")) or 30,
        "duration_seconds": duration,
        "timeline": rewritten_segments,
    }
    output = work_path(project, "render-input.json")
    write_json(output, render_input)
    print(f"Prepared {copies} asset copy/copies and reused {cache_hits} hash-matched copy/copies; render input includes {len(rewritten_segments)} timeline segment(s).")
    print(f"Ratio: {ratio} ({width}x{height})")
    print(f"Render props: {output}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


def add_project_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("project", help="path to the video project folder")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video_factory.py",
        description="Local deterministic media helpers and approved Colab transcription for the video-factory skill.",
        epilog="Examples: video_factory.py init ./projects/trip --profile memory; video_factory.py inspect ./projects/trip; video_factory.py qa ./projects/trip --video outputs/master.mp4",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="create the standard project folders and a reviewable job.draft.yaml")
    add_project_argument(init_parser)
    init_parser.add_argument("--profile", help="profile to place in a newly created job.draft.yaml")
    init_parser.set_defaults(func=command_init)

    inspect_parser = subparsers.add_parser("inspect", help="hash assets and build/cache media metadata")
    add_project_argument(inspect_parser)
    inspect_parser.set_defaults(func=command_inspect)

    validate_parser = subparsers.add_parser("validate-plan", help="validate the project edit_plan.json")
    add_project_argument(validate_parser)
    validate_parser.set_defaults(func=command_validate_plan)

    srt_parser = subparsers.add_parser("export-srt", help="export subtitle timeline entries to a local SRT file")
    add_project_argument(srt_parser)
    srt_parser.set_defaults(func=None)

    transcribe_parser = subparsers.add_parser(
        "transcribe", help="optionally upload one project audio source for timestamped transcription"
    )
    add_project_argument(transcribe_parser)
    transcribe_parser.add_argument("--source", required=True, help="project-relative audio source under assets/ or work/transcripts/audio/")
    transcribe_parser.add_argument("--model", default="whisper-1", help="timestamped transcription model (default: whisper-1)")
    transcribe_parser.add_argument("--language", help="optional language code, such as zh or en")
    transcribe_parser.add_argument("--prompt", help="optional transcription prompt; do not put secrets here")
    transcribe_parser.add_argument(
        "--allow-upload", action="store_true", help="explicitly authorize sending the selected audio to OpenAI on a cache miss"
    )
    transcribe_parser.set_defaults(func=None)

    colab_parser = subparsers.add_parser(
        "colab-transcribe", help="transcribe one approved project audio source on a temporary Colab GPU runtime"
    )
    add_project_argument(colab_parser)
    colab_parser.add_argument("--source", required=True, help="project-relative audio source under assets/ or work/transcripts/audio/")
    colab_parser.add_argument("--model", default="large-v3-turbo", help="faster-whisper model (default: large-v3-turbo)")
    colab_parser.add_argument("--language", help="optional language code, such as zh or en")
    colab_parser.add_argument("--gpu", choices=("T4", "L4"), default="T4", help="Colab GPU (default: T4; L4 only when justified)")
    colab_parser.add_argument(
        "--allow-upload", action="store_true", help="explicitly authorize sending the selected audio to Google Colab on a cache miss"
    )
    colab_parser.add_argument("--dry-run", action="store_true", help="show the transfer and runtime plan without invoking Colab")
    colab_parser.set_defaults(func=None)

    colab_perception_parser = subparsers.add_parser(
        "colab-perception", help="run the derived-data Colab Perception Worker and download one perception index"
    )
    add_project_argument(colab_perception_parser)
    colab_perception_parser.add_argument("--privacy-mode", choices=("BALANCED", "MAX_QUALITY"), default=None)
    colab_perception_parser.add_argument("--gpu", choices=("T4", "L4"), default="T4")
    colab_perception_parser.add_argument(
        "--temporal-backend", choices=("none", "smolvlm2"), default=None,
        help="optional registered temporal model; requires MAX_QUALITY and an explicit source shortlist",
    )
    colab_perception_parser.add_argument(
        "--temporal-source", dest="temporal_sources", action="append", default=[],
        metavar="PROJECT_RELATIVE_SOURCE",
        help="explicit source ID whose derived 720p proxy may be sent for temporal analysis; repeat for each source",
    )
    colab_perception_parser.add_argument("--allow-upload", action="store_true")
    colab_perception_parser.set_defaults(func=None)

    perception_parser = subparsers.add_parser(
        "perception", help="build the privacy-aware speech/visual/temporal perception index from local derivatives"
    )
    add_project_argument(perception_parser)
    perception_parser.add_argument("--privacy-mode", choices=("LOCAL_ONLY", "BALANCED", "MAX_QUALITY"), default=None)
    perception_parser.add_argument("--gpu", choices=("T4", "L4"), default=None)
    perception_parser.add_argument("--cloud", action="store_true", default=None, help="record a Colab plan; a separate adapter performs upload")
    perception_parser.add_argument("--allow-upload", action="store_true", help="authorize a separate cloud adapter for derived files")
    perception_parser.add_argument(
        "--temporal-backend", choices=("none", "smolvlm2"), default=None,
        help="registered temporal analysis backend metadata (deep analysis runs through colab-perception)",
    )
    perception_parser.add_argument("--force", action="store_true")
    perception_parser.set_defaults(func=None)

    qa_parser = subparsers.add_parser("qa", help="check a rendered video and write a QA report")
    add_project_argument(qa_parser)
    qa_parser.add_argument("--video", required=True, help="rendered video path, relative to PROJECT or absolute inside PROJECT")
    qa_parser.set_defaults(func=None)

    prepare_parser = subparsers.add_parser("prepare-render", help="validate a plan and create hash-verified Remotion public asset copies and props under work/")
    add_project_argument(prepare_parser)
    prepare_parser.add_argument("--ratio", help="render ratio WIDTH:HEIGHT; defaults to the plan/job ratio")
    prepare_parser.set_defaults(func=command_prepare_render)

    for name, help_text in (
        ("extract-scenes", "detect FFmpeg scene candidates from inspected videos"),
        ("extract-frames", "extract representative stills into work/frames/"),
        ("extract-audio", "extract local audio from source videos to a hash-cached work/transcripts/audio/"),
        ("build-contact-sheets", "tile extracted frames into contact sheets under work/"),
    ):
        media_parser = subparsers.add_parser(name, help=help_text)
        add_project_argument(media_parser)
        if name == "extract-scenes":
            media_parser.add_argument("--threshold", type=float, default=0.35, help="scene threshold between 0 and 1 (default: 0.35)")
            media_parser.add_argument("--min-scene-duration", type=float, default=1.0, help="ignore scene boundaries closer than this many seconds")
        elif name == "extract-frames":
            media_parser.add_argument("--per-video", type=int, default=6, help="fallback number of evenly spaced frames per video (default: 6)")
            media_parser.add_argument("--max-frames-per-video", type=int, default=24, help="cap frames from scene candidates per video (default: 24)")
        elif name == "extract-audio":
            media_parser.add_argument("--format", choices=("flac", "wav"), default="flac", help="lossless output format (default: flac)")
            media_parser.add_argument("--channels", type=int, choices=(1, 2), default=1, help="output channel count (default: mono)")
            media_parser.add_argument("--sample-rate", type=int, default=48000, help="output sample rate in Hz (default: 48000)")
        else:
            media_parser.add_argument("--columns", type=int, default=5, help="number of thumbnails per row (default: 5)")
            media_parser.add_argument("--tile-width", type=int, default=320, help="thumbnail cell width in pixels (default: 320)")
        media_parser.set_defaults(func=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Direct execution names this module ``__main__``. Helper modules import
    # ``video_factory`` and must see the same exception classes and path
    # helpers so UserFacingError is caught consistently.
    sys.modules.setdefault("video_factory", sys.modules[__name__])
    if args.command in {"extract-scenes", "extract-frames", "extract-audio", "build-contact-sheets", "export-srt", "qa"}:
        try:
            # Keep helper imports bound to this module instance when run as a script.
            sys.modules.setdefault("video_factory", sys.modules[__name__])
            if args.command in {"extract-scenes", "extract-frames", "extract-audio", "build-contact-sheets"}:
                from media_helpers import command_media

                return command_media(args)
            from qa_helpers import command_quality

            return command_quality(args)
        except ImportError as exc:
            raise UserFacingError(f"Media helper could not be loaded: {exc}") from exc
    try:
        if args.command in {"transcribe", "colab-transcribe"}:
            try:
                sys.modules.setdefault("video_factory", sys.modules[__name__])
                if args.command == "transcribe":
                    from transcription_openai import command_transcribe
                else:
                    from colab_transcription import command_colab_transcribe
                    command_transcribe = command_colab_transcribe
            except ImportError as exc:
                raise UserFacingError(f"Transcription helper could not be loaded: {exc}") from exc
            return command_transcribe(args)
        if args.command == "perception":
            try:
                from perception import command_perception
            except ImportError as exc:
                raise UserFacingError(f"Perception helper could not be loaded: {exc}") from exc
            return command_perception(args)
        if args.command == "colab-perception":
            try:
                from colab_perception import command_colab_perception
            except ImportError as exc:
                raise UserFacingError(f"Colab perception helper could not be loaded: {exc}") from exc
            return command_colab_perception(args)
        return args.func(args)
    except UserFacingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserFacingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
