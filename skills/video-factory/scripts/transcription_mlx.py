"""Local Apple Silicon transcription using MLX Whisper.

Audio is read from the project filesystem and passed to ``mlx_whisper`` as a
local path. The only optional network access is downloading public model
weights from Hugging Face; source audio is never uploaded.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from transcription_openai import (
    CACHE_SCHEMA_VERSION,
    output_path,
    resolve_transcript_source,
    response_to_srt,
    write_text_atomic,
)
from video_factory import UserFacingError, resolve_project, sha256_file, work_path


DEFAULT_MODEL = "mlx-community/whisper-large-v3-mlx"
# Pin a concrete Hub snapshot so a cache key identifies the actual model
# weights, not a moving branch name.
DEFAULT_MODEL_REVISION = "49e6aa286ad60c14352c404340ded53710378a11"
ENGINE = "mlx-whisper"
ENGINE_CONFIG = {
    "word_timestamps": True,
    "temperature": (0.0, 0.2, 0.4),
    "no_speech_threshold": 0.6,
    "condition_on_previous_text": False,
    "verbose": None,
}

_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)+$")
_REVISION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class _Runtime:
    transcribe: Callable[..., Any]
    snapshot_download: Callable[..., str]
    mlx_whisper_version: str
    mlx_version: str


def _validate_model_options(model: str, language: str | None, revision: str | None) -> None:
    if not isinstance(model, str) or not _MODEL_ID_RE.fullmatch(model) or ".." in model.split("/"):
        raise UserFacingError("MLX Whisper model must be a Hugging Face repository id such as mlx-community/whisper-large-v3-mlx.")
    if language is not None and (
        not isinstance(language, str)
        or not language.strip()
        or any(character.isspace() for character in language)
        or not re.fullmatch(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})?", language)
    ):
        raise UserFacingError("--language must be a Whisper language code such as zh or en.")
    if revision is not None and (not isinstance(revision, str) or not _REVISION_RE.fullmatch(revision)):
        raise UserFacingError("MLX Whisper model revision contains unsupported characters.")


def _load_runtime() -> _Runtime:
    if platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}:
        raise UserFacingError("Local MLX transcription requires Apple Silicon macOS.")
    try:
        mlx_whisper = importlib.import_module("mlx_whisper")
        huggingface_hub = importlib.import_module("huggingface_hub")
    except ImportError:
        raise UserFacingError(
            "MLX Whisper is not installed in this Python environment. Install it with `uv pip install mlx-whisper`."
        ) from None
    try:
        package_version = importlib.metadata.version("mlx-whisper")
        mlx_version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        raise UserFacingError("Could not determine the installed MLX Whisper/MLX package versions.") from None
    snapshot_download = getattr(huggingface_hub, "snapshot_download", None)
    transcribe = getattr(mlx_whisper, "transcribe", None)
    if not callable(snapshot_download) or not callable(transcribe):
        raise UserFacingError("The installed MLX Whisper runtime is missing its supported transcription API.")
    return _Runtime(transcribe, snapshot_download, package_version, mlx_version)


def _snapshot_revision(snapshot_path: str | Path) -> str:
    revision = Path(snapshot_path).name
    if not _SHA_RE.fullmatch(revision):
        raise UserFacingError("Hugging Face did not resolve the MLX model to a pinned snapshot revision.")
    return revision


def _resolve_model_snapshot(
    runtime: _Runtime,
    model: str,
    revision: str,
    *,
    local_files_only: bool = True,
) -> tuple[Path, str]:
    # Try a local Hub cache first. This avoids contacting the Hub on subsequent
    # runs while still recording the exact downloaded model commit in cache.
    try:
        snapshot_path = runtime.snapshot_download(
            repo_id=model,
            revision=revision,
            local_files_only=True,
        )
    except Exception:
        if local_files_only:
            raise UserFacingError(
                "The requested MLX Whisper snapshot is not in the local Hugging Face cache; no model was downloaded."
            ) from None
        try:
            snapshot_path = runtime.snapshot_download(repo_id=model, revision=revision)
        except Exception as exc:
            raise UserFacingError(
                "Could not load the pinned MLX Whisper model. Check network access for model weights or install that snapshot in the Hugging Face cache."
            ) from None
    return Path(snapshot_path), _snapshot_revision(snapshot_path)


def normalize_mlx_response(value: Any) -> dict[str, Any]:
    """Validate MLX Whisper output while retaining word timings and confidence."""
    if not isinstance(value, dict) or not isinstance(value.get("text"), str):
        raise UserFacingError("MLX Whisper returned an invalid response: expected transcript text.")
    segments = value.get("segments")
    if not isinstance(segments, list):
        raise UserFacingError("MLX Whisper returned an invalid response: expected timestamped segments.")

    normalized_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise UserFacingError(f"MLX Whisper returned invalid transcript text at segment {index}.")
        start = _finite_number(segment.get("start"))
        end = _finite_number(segment.get("end"))
        if start is None or end is None or start < 0 or end < start:
            raise UserFacingError(f"MLX Whisper returned invalid timestamps at segment {index}.")

        item: dict[str, Any] = {"start": start, "end": end, "text": segment["text"]}
        for confidence_key in ("avg_logprob", "no_speech_prob", "temperature", "compression_ratio"):
            if confidence_key in segment:
                confidence = _finite_number(segment[confidence_key])
                if confidence is not None:
                    item[confidence_key] = confidence

        words = segment.get("words", [])
        if not isinstance(words, list):
            raise UserFacingError(f"MLX Whisper returned invalid word timestamps at segment {index}.")
        normalized_words: list[dict[str, Any]] = []
        for word_index, word in enumerate(words):
            if not isinstance(word, dict) or not isinstance(word.get("word"), str):
                raise UserFacingError(
                    f"MLX Whisper returned invalid word text at segment {index}, word {word_index}."
                )
            word_start = _finite_number(word.get("start"))
            word_end = _finite_number(word.get("end"))
            if word_start is None or word_end is None or word_start < 0 or word_end < word_start:
                raise UserFacingError(
                    f"MLX Whisper returned invalid word timestamps at segment {index}, word {word_index}."
                )
            normalized_word: dict[str, Any] = {
                "start": word_start,
                "end": word_end,
                "word": word["word"],
            }
            probability = _finite_number(word.get("probability"))
            if probability is not None:
                normalized_word["probability"] = probability
            normalized_words.append(normalized_word)
        item["words"] = normalized_words
        normalized_segments.append(item)

    normalized: dict[str, Any] = {"text": value["text"], "segments": normalized_segments}
    language = value.get("language")
    if isinstance(language, str) and language:
        normalized["language"] = language
    return normalized


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_cache(json_path: Path, expected: dict[str, Any]) -> dict[str, Any] | None:
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
        return normalize_mlx_response(cached.get("response"))
    except UserFacingError:
        return None


def _source_relative_path(project: Path, source_path: str | Path) -> str:
    candidate = Path(source_path)
    if candidate.is_absolute():
        try:
            candidate = candidate.resolve(strict=True).relative_to(project.resolve(strict=True))
        except (OSError, RuntimeError, ValueError):
            raise UserFacingError("Audio source must be inside the project assets/ or work/transcripts/audio/ folder.") from None
    return candidate.as_posix()


def transcribe_audio(
    project: str | Path,
    source_path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    model_revision: str | None = None,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Transcribe a project audio file locally, using a hash/versioned cache.

    ``source_path`` may be project-relative or an absolute path under
    ``assets/`` or ``work/transcripts/audio/``. The returned response retains
    MLX's word-level timings for downstream cut-to-timeline mapping.
    """
    project_path = resolve_project(project)
    source_relative, audio_path = resolve_transcript_source(
        project_path, _source_relative_path(project_path, source_path)
    )
    revision = model_revision
    if revision is None and model == DEFAULT_MODEL:
        revision = DEFAULT_MODEL_REVISION
    _validate_model_options(model, language, revision)
    try:
        source_hash = sha256_file(audio_path)
    except UserFacingError:
        raise

    # Resolve the exact local Hub snapshot before looking up the transcript
    # cache. The snapshot lookup is local-only first; a cache hit never needs
    # to download the source or load the MLX model.
    runtime = _load_runtime()
    resolved_revision = revision or "main"
    model_path, resolved_revision = _resolve_model_snapshot(
        runtime, model, resolved_revision, local_files_only=local_files_only
    )
    request = {
        "source": source_relative,
        "source_sha256": source_hash,
        "engine": ENGINE,
        "mlx_whisper_version": runtime.mlx_whisper_version,
        "mlx_version": runtime.mlx_version,
        "model": model,
        "model_revision": resolved_revision,
        "language": language,
        # Canonicalize tuples and other JSON-compatible values so the in-memory
        # fingerprint compares exactly with the representation on disk.
        "parameters": json.loads(json.dumps(ENGINE_CONFIG)),
    }
    cache_key = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    work_path(project_path, "transcripts", create_dir=True)
    json_path = output_path(project_path, f"mlx-{cache_key}.json")
    srt_path = output_path(project_path, f"mlx-{cache_key}.srt")
    expected = {"cache_key": cache_key, "request": request}
    cached_response = _load_cache(json_path, expected)
    if cached_response is not None:
        write_text_atomic(srt_path, response_to_srt(cached_response))
        return cached_response

    options = dict(ENGINE_CONFIG)
    if language is not None:
        options["language"] = language
    try:
        raw_response = runtime.transcribe(
            str(audio_path),
            path_or_hf_repo=str(model_path),
            **options,
        )
    except Exception as exc:
        raise UserFacingError(
            f"Local MLX Whisper inference failed ({type(exc).__name__}). Check the audio format and available memory; source audio was not uploaded."
        ) from None
    response = normalize_mlx_response(raw_response)
    try:
        final_hash = sha256_file(audio_path)
    except UserFacingError:
        raise
    if final_hash != source_hash:
        raise UserFacingError("The audio source changed during local transcription; run the command again.")

    cache_document = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "request": request,
        "response": response,
    }
    write_text_atomic(json_path, json.dumps(cache_document, ensure_ascii=False, indent=2) + "\n")
    write_text_atomic(srt_path, response_to_srt(response))
    return response


def command_transcribe_mlx(args: Any) -> int:
    """CLI-compatible wrapper; the pipeline can call ``transcribe_audio`` directly."""
    response = transcribe_audio(
        args.project,
        args.source,
        model=getattr(args, "model", None) or DEFAULT_MODEL,
        language=getattr(args, "language", None) or None,
        model_revision=getattr(args, "model_revision", None),
        local_files_only=bool(getattr(args, "local_only", False)),
    )
    print(f"Transcribed locally with MLX Whisper: {len(response['segments'])} segments; source audio stayed on this Mac.")
    return 0


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_MODEL_REVISION",
    "ENGINE_CONFIG",
    "command_transcribe_mlx",
    "normalize_mlx_response",
    "transcribe_audio",
]
