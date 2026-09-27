"""Optional Google Colab transcription adapter using the official Colab CLI.

The adapter only uploads the selected audio after an explicit allow_upload
consent. It uses a uniquely named Colab session and attempts to stop that
session in a finally path. Credentials and subprocess output are never logged.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from transcription_openai import (
    CACHE_SCHEMA_VERSION,
    load_valid_cache,
    normalize_response,
    output_path,
    resolve_transcript_source,
    response_to_srt,
    write_text_atomic,
)
from video_factory import (
    UserFacingError,
    get_cloud_processing_setting,
    resolve_project,
    sha256_file,
    work_path,
)


DEFAULT_MODEL = "large-v3-turbo"
DEFAULT_GPU = "T4"
ALLOWED_GPUS = {"T4", "L4"}
FASTER_WHISPER_VERSION = "1.2.1"
COMPUTE_TYPE = "float16"
BEAM_SIZE = 5
WORKER_VERSION = 2
REMOTE_EXEC_TIMEOUT = 3 * 60 * 60
GPU_PACKAGES = [
    f"faster-whisper=={FASTER_WHISPER_VERSION}",
    "ctranslate2==4.6.0",
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12==9.*",
]

_TIMEOUTS = {
    "new": 900,
    "upload": 600,
    "exec": REMOTE_EXEC_TIMEOUT + 120,
    "download": 600,
    "stop": 300,
}


def _inference_source(configuration: dict[str, Any]) -> str:
    encoded = repr(json.dumps(configuration, ensure_ascii=False, separators=(",", ":")))
    return f'''import json
import os

from faster_whisper import WhisperModel

CONFIG = json.loads({encoded})
model = WhisperModel(
    CONFIG["model"],
    device="cuda",
    compute_type=CONFIG["compute_type"],
)
segments, _ = model.transcribe(CONFIG["input_path"], **CONFIG["transcribe"])
normalized = []
for segment in segments:
    normalized.append({{
        "start": float(segment.start),
        "end": float(segment.end),
        "text": segment.text,
    }})
response = {{
    "text": "".join(item["text"] for item in normalized),
    "segments": normalized,
}}
temporary_path = CONFIG["output_path"] + ".tmp"
with open(temporary_path, "w", encoding="utf-8") as handle:
    json.dump(response, handle, ensure_ascii=False)
    handle.write("\\n")
os.replace(temporary_path, CONFIG["output_path"])
'''


def _worker_source(configuration: dict[str, Any]) -> str:
    # Start a fresh interpreter: CUDA's loader reads LD_LIBRARY_PATH at process
    # startup, so changing it in an already running notebook is insufficient.
    inference = _inference_source(configuration)
    return f'''import importlib.util
import os
import subprocess
import sys

hardware = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True, text=True, check=True, timeout=30,
).stdout.strip().splitlines()
if len(hardware) != 1 or {configuration["gpu"]!r} not in hardware[0].split():
    raise RuntimeError("Allocated GPU differs from the requested GPU; inference was not started.")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", *{GPU_PACKAGES!r}],
    check=True, timeout=1800, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
importlib.invalidate_caches()
library_dirs = []
for module in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Required CUDA libraries are missing.")
    library_dirs.extend(spec.submodule_search_locations)
environment = os.environ.copy()
environment["LD_LIBRARY_PATH"] = os.pathsep.join(
    library_dirs + [environment.get("LD_LIBRARY_PATH", "")]
)
subprocess.run(
    [sys.executable, "-c", {inference!r}], env=environment,
    check=True, timeout={REMOTE_EXEC_TIMEOUT - 1900},
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
'''


def _run_colab(cli_path: str, stage: str, arguments: list[str]) -> None:
    try:
        result = subprocess.run(
            [cli_path, "--logtostderr", *arguments],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if stage == "stop" else subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_TIMEOUTS[stage],
        )
    except subprocess.TimeoutExpired:
        raise UserFacingError(f"Colab CLI {stage} command timed out; command output was suppressed.") from None
    except OSError:
        raise UserFacingError(f"Could not start the Colab CLI {stage} command.") from None
    if result.returncode != 0:
        raise UserFacingError(f"Colab CLI {stage} command failed; command output was suppressed.")
    if stage == "stop" and b"[colab] Session terminated." not in (result.stdout or b""):
        raise UserFacingError("Colab CLI did not confirm that the owned session was terminated.")


def _copy_source_for_upload(source_path: Path, destination: Path, expected_hash: str) -> None:
    digest = hashlib.sha256()
    try:
        with source_path.open("rb") as source, destination.open("xb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                digest.update(chunk)
    except OSError:
        raise UserFacingError("Could not prepare the selected audio for upload.") from None
    if digest.hexdigest() != expected_hash:
        raise UserFacingError("The audio source changed while preparing the Colab request; run the command again.")


def _read_remote_response(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise UserFacingError("Colab did not provide the downloaded transcript JSON.") from None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        raise UserFacingError("Colab returned invalid transcript JSON.") from None
    return normalize_response(value)


def _colab_usage_snapshot(cli_path: str) -> dict[str, str] | None:
    """Read and sanitize the documented numeric fields from ``colab usage``."""
    try:
        result = subprocess.run(
            [cli_path, "--logtostderr", "usage"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    output = f"{result.stdout}\n{result.stderr}"
    balance = re.search(r"Current balance:\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s+compute units", output, re.IGNORECASE)
    rate = re.search(r"Usage rate:\s*([0-9][0-9,]*(?:\.[0-9]+)?)/hr", output, re.IGNORECASE)
    assignments = re.search(r"Active assignments:\s*(\d+)\b", output, re.IGNORECASE)
    if not (balance and rate and assignments):
        return None
    return {
        "balance": balance.group(1),
        "rate": rate.group(1),
        "assignments": assignments.group(1),
    }


def _print_usage_snapshot(label: str, snapshot: dict[str, str] | None) -> None:
    if snapshot is None:
        print(f"Colab usage {label}: unavailable.")
        return
    print(
        f"Colab usage {label}: balance {snapshot['balance']} compute units; "
        f"aggregate rate {snapshot['rate']} CU/hr; active assignments {snapshot['assignments']}."
    )


def _print_project_policy_state(project: Path) -> None:
    try:
        policy = get_cloud_processing_setting(project, "colab")
    except UserFacingError as exc:
        print(f"Project Colab policy: invalid ({exc}).")
        return
    if policy is None:
        print("Project Colab policy: missing.")
    else:
        print(f"Project Colab policy: {policy}.")


def command_colab_transcribe(args: Any) -> int:
    """Transcribe one approved project audio source on a Colab GPU."""
    project = resolve_project(args.project)
    source_relative, source_path = resolve_transcript_source(project, args.source)
    source_hash = sha256_file(source_path)

    model = getattr(args, "model", None) or DEFAULT_MODEL
    if not isinstance(model, str) or not model.strip() or any(character.isspace() for character in model):
        raise UserFacingError("--model must be a non-empty faster-whisper model identifier without whitespace.")
    if len(model) > 256 or any(not (character.isalnum() or character in "._/-") for character in model):
        raise UserFacingError("--model contains unsupported characters.")

    gpu = getattr(args, "gpu", None) or DEFAULT_GPU
    if gpu not in ALLOWED_GPUS:
        raise UserFacingError("--gpu must be T4 or L4.")
    if gpu == "L4":
        raise UserFacingError("Routine speech intelligence is pinned to T4; reserve L4 for shortlist-only temporal analysis.")

    language = getattr(args, "language", None) or None
    if language is not None and (
        not isinstance(language, str)
        or not language.strip()
        or any(character.isspace() for character in language)
    ):
        raise UserFacingError("--language must be a non-empty language code without whitespace.")

    transcribe_config: dict[str, Any] = {
        "beam_size": BEAM_SIZE,
        "condition_on_previous_text": True,
        "vad_filter": False,
    }
    if language is not None:
        transcribe_config["language"] = language
    request_fingerprint = {
        "source": source_relative,
        "source_sha256": source_hash,
        "engine": "faster-whisper",
        "engine_version": FASTER_WHISPER_VERSION,
        "worker_version": WORKER_VERSION,
        "packages": GPU_PACKAGES,
        "model": model,
        "gpu": gpu,
        "device": "cuda",
        "compute_type": COMPUTE_TYPE,
        "transcribe": transcribe_config,
    }
    cache_key = hashlib.sha256(
        json.dumps(request_fingerprint, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    work_path(project, "transcripts", create_dir=True)
    json_path = output_path(project, f"{cache_key}.json")
    srt_path = output_path(project, f"{cache_key}.srt")
    expected = {"cache_key": cache_key, "request": request_fingerprint}
    response = load_valid_cache(json_path, expected)

    if getattr(args, "dry_run", False):
        try:
            source_size = source_path.stat().st_size
        except OSError:
            raise UserFacingError("Could not read the selected audio file size for the dry run.") from None
        _print_project_policy_state(project)
        print(f"Selected audio: {source_relative} ({source_size} bytes).")
        if response is not None:
            print("Dry run: a matching local transcript cache exists; no Colab CLI command was called.")
        else:
            print("Dry run: no Colab CLI command was called and no audio was uploaded.")
            print(f"Would request a {gpu} session with faster-whisper {FASTER_WHISPER_VERSION} ({model}).")
            print("No runtime duration or compute-unit estimate is available without a live benchmark.")
        return 0

    if response is not None:
        write_text_atomic(srt_path, response_to_srt(response))
        print(f"Used cached timestamp transcript: {json_path}")
        print(f"SRT: {srt_path}")
        return 0

    policy = get_cloud_processing_setting(project, "colab")
    if policy != "ask_each_run":
        raise UserFacingError(
            "Colab transcription requires job.yaml cloud_processing.colab: ask_each_run; "
            "missing or disabled settings block cloud transfer."
        )

    if not getattr(args, "allow_upload", False):
        raise UserFacingError(
            "No valid cached transcript exists. Add --allow-upload to send this audio to Google Colab."
        )

    try:
        source_size = source_path.stat().st_size
    except OSError:
        raise UserFacingError("Could not read the selected audio file size before upload.") from None
    print(f"Transferring selected audio to Google Colab: {source_relative} ({source_size} bytes).")
    print(f"Requested accelerator: {gpu}; faster-whisper {FASTER_WHISPER_VERSION} ({model}).")

    cli_path = shutil.which("colab")
    if cli_path is None:
        raise UserFacingError("The Colab CLI is not installed. Install google-colab-cli before using this adapter.")

    usage_before = _colab_usage_snapshot(cli_path)
    _print_usage_snapshot("before allocation", usage_before)
    if usage_before is None:
        raise UserFacingError(
            "Could not verify Colab compute-unit usage; refusing to allocate a GPU session."
        )

    session_name = f"vf-transcribe-{secrets.token_hex(16)}"
    remote_nonce = secrets.token_hex(12)
    remote_input = f"/content/vf-audio-{remote_nonce}{source_path.suffix.lower()}"
    remote_output = f"/content/vf-transcript-{remote_nonce}.json"
    session_owned = False
    transcription_saved = False
    try:
        staging_parent = work_path(project, "colab-staging", create_dir=True)
        with tempfile.TemporaryDirectory(
            prefix="video-factory-colab-", dir=str(staging_parent)
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            upload_path = temporary_root / f"selected-audio{source_path.suffix.lower()}"
            worker_path = temporary_root / "colab_transcription_worker.py"
            download_path = temporary_root / "transcript.json"
            _copy_source_for_upload(source_path, upload_path, source_hash)

            worker_configuration = {
                "model": model,
                "gpu": gpu,
                "compute_type": COMPUTE_TYPE,
                "input_path": remote_input,
                "output_path": remote_output,
                "transcribe": transcribe_config,
            }
            try:
                worker_path.write_text(_worker_source(worker_configuration), encoding="utf-8")
            except OSError:
                raise UserFacingError("Could not prepare the generated Colab worker.") from None

            # Reserve this high-entropy name for this run so cleanup can target
            # only the session this adapter asked the CLI to create.
            session_owned = True
            _run_colab(cli_path, "new", ["new", "-s", session_name, "--gpu", gpu])
            _run_colab(cli_path, "upload", ["upload", "-s", session_name, str(upload_path), remote_input])
            _run_colab(cli_path, "exec", [
                "exec", "-s", session_name, "-f", str(worker_path),
                "--timeout", str(REMOTE_EXEC_TIMEOUT),
            ])
            _run_colab(
                cli_path,
                "download",
                ["download", "-s", session_name, remote_output, str(download_path)],
            )
            response = _read_remote_response(download_path)
            cache_document = {
                "schema_version": CACHE_SCHEMA_VERSION,
                "cache_key": cache_key,
                "request": request_fingerprint,
                "response": response,
            }
            write_text_atomic(json_path, json.dumps(cache_document, ensure_ascii=False, indent=2) + "\n")
            write_text_atomic(srt_path, response_to_srt(response))
            transcription_saved = True
    finally:
        if session_owned:
            cleanup_failed = False
            try:
                _run_colab(cli_path, "stop", ["stop", "-s", session_name])
            except UserFacingError:
                cleanup_failed = True
                if not transcription_saved:
                    print(
                        f"WARNING: Could not confirm cleanup. Inspect Colab session {session_name} and stop only that session.",
                        file=sys.stderr,
                    )
            _print_usage_snapshot("after cleanup", _colab_usage_snapshot(cli_path))
            if cleanup_failed and transcription_saved:
                raise UserFacingError(
                    f"Transcript was saved to {json_path}, but cleanup of Colab session {session_name} could not be confirmed."
                ) from None

    print(f"Saved timestamp transcript: {json_path}")
    print(f"SRT: {srt_path}")
    return 0


__all__ = [
    "ALLOWED_GPUS",
    "BEAM_SIZE",
    "COMPUTE_TYPE",
    "DEFAULT_GPU",
    "DEFAULT_MODEL",
    "FASTER_WHISPER_VERSION",
    "command_colab_transcribe",
]
