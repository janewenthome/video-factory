"""Optional OpenAI transcription adapter with explicit upload consent.

No request is made on a cache miss unless the caller supplies --allow-upload.
This module uses only the Python standard library and never logs credentials.
"""

from __future__ import annotations

import hashlib
import json
import math
import mimetypes
import os
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

from video_factory import UserFacingError, get_cloud_processing_setting, is_within, resolve_project, sha256_file, work_path


API_URL = "https://api.openai.com/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-1"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
SUPPORTED_AUDIO_SUFFIXES = {".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".wav", ".webm"}
ALLOWED_SOURCE_ROOTS = (PurePosixPath("assets"), PurePosixPath("work/transcripts/audio"))
CACHE_SCHEMA_VERSION = 1


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def resolve_transcript_source(project: Path, source: str) -> tuple[str, Path]:
    if not isinstance(source, str) or not source.strip():
        raise UserFacingError("--source must be a non-empty project-relative audio path.")
    if "\\" in source:
        raise UserFacingError("--source must use POSIX path separators.")
    posix = PurePosixPath(source)
    parts = source.split("/")
    if posix.is_absolute() or re.match(r"^[A-Za-z]:", source) or any(part in {"", ".", ".."} for part in parts):
        raise UserFacingError("--source must be a project-relative path without '.' or '..' components.")
    if not any(posix.parts[: len(root.parts)] == root.parts for root in ALLOWED_SOURCE_ROOTS):
        raise UserFacingError("--source must be under assets/ or work/transcripts/audio/.")

    candidate = project.joinpath(*posix.parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise UserFacingError("--source could not be resolved inside the project.") from exc
    if not is_within(resolved, project):
        raise UserFacingError("--source resolves outside the project; symlink escapes are not allowed.")
    if not any(is_within(resolved, project.joinpath(*root.parts)) for root in ALLOWED_SOURCE_ROOTS):
        raise UserFacingError("--source resolves outside assets/ and work/transcripts/audio/.")
    if not resolved.is_file():
        raise UserFacingError("--source must refer to an existing regular audio file.")
    if resolved.suffix.lower() not in SUPPORTED_AUDIO_SUFFIXES:
        raise UserFacingError("--source file extension is not supported by the transcription endpoint.")
    return posix.as_posix(), resolved


def srt_timestamp(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    total_seconds, millis = divmod(total_ms, 1000)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def normalize_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        raise UserFacingError("Transcription API returned an invalid response: expected a text field.")
    segments = value.get("segments")
    if not isinstance(segments, list):
        raise UserFacingError("Transcription API returned an invalid response: expected timestamped segments.")
    normalized: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise UserFacingError(f"Transcription API returned an invalid segment at index {index}.")
        start = segment.get("start")
        end = segment.get("end")
        text = segment.get("text")
        try:
            start_value = float(start) if not isinstance(start, bool) and isinstance(start, (int, float)) else math.nan
            end_value = float(end) if not isinstance(end, bool) and isinstance(end, (int, float)) else math.nan
        except (OverflowError, TypeError, ValueError):
            start_value = end_value = math.nan
        if (
            not math.isfinite(start_value)
            or not math.isfinite(end_value)
            or start_value < 0
            or end_value < start_value
            or not isinstance(text, str)
        ):
            raise UserFacingError(f"Transcription API returned invalid timestamps or text at segment {index}.")
        normalized.append({"start": start_value, "end": end_value, "text": text})
    return {"text": value["text"], "segments": normalized}


def response_to_srt(response: dict[str, Any]) -> str:
    segments = sorted(response["segments"], key=lambda segment: (segment["start"], segment["end"]))
    blocks: list[str] = []
    for segment in segments:
        text = segment["text"].replace("\r\n", "\n").replace("\r", "\n").strip()
        if not text:
            continue
        number = len(blocks) + 1
        blocks.append(
            f"{number}\n{srt_timestamp(segment['start'])} --> {srt_timestamp(segment['end'])}\n{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def output_path(project: Path, name: str) -> Path:
    path = work_path(project, Path("transcripts") / name, create_dir=False)
    transcripts_dir = project / "work" / "transcripts"
    if transcripts_dir.is_symlink() or not is_within(transcripts_dir, project / "work"):
        raise UserFacingError("work/transcripts/ must be a real directory inside work/.")
    if not transcripts_dir.is_dir():
        raise UserFacingError("Could not access the work/transcripts/ output folder.")
    if os.path.lexists(path) and (path.is_symlink() or not path.is_file()):
        raise UserFacingError(f"Transcript cache destination is not a regular file: {path.name}")
    return path


def write_text_atomic(path: Path, content: str) -> None:
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(content)
        os.replace(temporary, path)
        temporary = None
    except OSError as exc:
        raise UserFacingError(f"Could not write transcript output {path.name}.") from exc
    finally:
        if temporary:
            try:
                Path(temporary).unlink(missing_ok=True)
            except OSError:
                pass


def request_multipart(
    source_path: Path,
    source_bytes: bytes,
    *,
    model: str,
    language: str | None,
    prompt: str | None,
    api_key: str,
) -> dict[str, Any]:
    boundary = "video-factory-" + os.urandom(18).hex()
    fields: list[tuple[str, str]] = [
        ("model", model),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "segment"),
    ]
    if language:
        fields.append(("language", language))
    if prompt:
        fields.append(("prompt", prompt))
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("ascii"),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )
    filename = source_path.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("ascii"),
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"),
            f"Content-Type: {content_type}\r\n\r\n".encode("ascii"),
            source_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        ]
    )
    request = urllib.request.Request(
        API_URL,
        data=b"".join(chunks),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise UserFacingError(f"Transcription API request failed with HTTP {exc.code}; response details were not logged.") from None
    except Exception:
        raise UserFacingError("Transcription API request failed; network details and credentials were not logged.") from None
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise UserFacingError("Transcription API returned invalid JSON.") from None
    return normalize_response(decoded)


def load_valid_cache(json_path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
    if not json_path.is_file():
        return None
    try:
        cached = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(cached, dict)
        or cached.get("schema_version") != CACHE_SCHEMA_VERSION
        or cached.get("cache_key") != expected["cache_key"]
        or cached.get("request") != expected["request"]
    ):
        return None
    try:
        return normalize_response(cached.get("response"))
    except UserFacingError:
        return None


def command_transcribe(args: Any) -> int:
    project = resolve_project(args.project)
    source_relative, source_path = resolve_transcript_source(project, args.source)
    try:
        source_hash = sha256_file(source_path)
    except UserFacingError:
        raise
    model = args.model or DEFAULT_MODEL
    if not isinstance(model, str) or not model.strip() or any(character.isspace() for character in model):
        raise UserFacingError("--model must be a non-empty model identifier without whitespace.")
    language = args.language or None
    if language is not None and (not language.strip() or any(character.isspace() for character in language)):
        raise UserFacingError("--language must be a non-empty language code without whitespace.")
    prompt = args.prompt or None
    prompt_hash = sha256_text(prompt or "")
    request_fingerprint = {
        "source": source_relative,
        "source_sha256": source_hash,
        "model": model,
        "language": language,
        "prompt_sha256": prompt_hash,
    }
    cache_key = hashlib.sha256(
        json.dumps(request_fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    work_path(project, "transcripts", create_dir=True)
    json_path = output_path(project, f"{cache_key}.json")
    srt_path = output_path(project, f"{cache_key}.srt")
    expected = {"cache_key": cache_key, "request": request_fingerprint}
    response = load_valid_cache(json_path, expected)
    if response is not None:
        write_text_atomic(srt_path, response_to_srt(response))
        print(f"Used cached timestamp transcript: {json_path}")
        print(f"SRT: {srt_path}")
        return 0

    if model != DEFAULT_MODEL:
        raise UserFacingError(
            "Timestamped subtitles require --model whisper-1; other OpenAI models do not support this timestamp request."
        )

    if get_cloud_processing_setting(project, "openai") == "disabled":
        raise UserFacingError("job.yaml disables OpenAI cloud processing; no audio was uploaded.")

    if not args.allow_upload:
        raise UserFacingError("No valid cached transcript exists. Add --allow-upload to send this audio to OpenAI.")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise UserFacingError("OPENAI_API_KEY is required when --allow-upload is used.")
    try:
        upload_size = source_path.stat().st_size
    except OSError:
        raise UserFacingError("Could not read the selected audio source.") from None
    if upload_size > MAX_UPLOAD_BYTES:
        raise UserFacingError(
            "Selected audio exceeds the 25 MB transcription upload limit. Compress it or split it into smaller clips first."
        )
    try:
        with source_path.open("rb") as source:
            source_bytes = source.read(MAX_UPLOAD_BYTES + 1)
        if len(source_bytes) > MAX_UPLOAD_BYTES:
            raise UserFacingError("Selected audio exceeds the 25 MB transcription upload limit.")
    except OSError:
        raise UserFacingError("Could not read the selected audio source.") from None
    if hashlib.sha256(source_bytes).hexdigest() != source_hash:
        raise UserFacingError("The audio source changed while preparing the request; run the command again.")

    response = request_multipart(
        source_path,
        source_bytes,
        model=model,
        language=language,
        prompt=prompt,
        api_key=api_key,
    )
    cache_document = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "request": request_fingerprint,
        "response": response,
    }
    write_text_atomic(json_path, json.dumps(cache_document, ensure_ascii=False, indent=2) + "\n")
    write_text_atomic(srt_path, response_to_srt(response))
    print(f"Saved timestamp transcript: {json_path}")
    print(f"SRT: {srt_path}")
    return 0


__all__ = [
    "API_URL",
    "DEFAULT_MODEL",
    "MAX_UPLOAD_BYTES",
    "command_transcribe",
    "normalize_response",
    "request_multipart",
    "resolve_transcript_source",
    "response_to_srt",
    "srt_timestamp",
]
