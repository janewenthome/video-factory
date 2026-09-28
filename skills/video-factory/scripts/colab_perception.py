"""Ephemeral Colab Perception Worker adapter.

This is the only module allowed to transfer derived perception inputs.  It
uploads a manifest plus selected audio/frames/proxies, runs a generated worker
on a T4 by default, downloads one JSON index, verifies the privacy contract,
and releases the owned runtime in ``finally``.  Final editing and rendering
remain local.
"""

from __future__ import annotations

import json
import math
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Any

from colab_transcription import (
    REMOTE_EXEC_TIMEOUT,
    _colab_usage_snapshot,
    _print_usage_snapshot,
    _run_colab,
)
from perception import (
    AUDIO_EVENT_METHOD,
    AUDIO_EVENT_MODEL,
    AUDIO_EVENT_MODEL_REVISION,
    AUDIO_EVENT_TOP_K,
    AUDIO_EVENT_WINDOW_SECONDS,
    COLAB_PACKAGE_VERSIONS,
    DIARIZATION_MODEL,
    DIARIZATION_MODEL_REVISION,
    IMPORTANCE_METHOD,
    IMPORTANCE_PARAMETERS,
    SPEECH_ENGINE_VERSION,
    SPEECH_MODEL,
    SPEECH_MODEL_REPO,
    SPEECH_MODEL_REVISION,
    TEMPORAL_ADAPTERS,
    TEMPORAL_BACKENDS,
    VISUAL_MODEL,
    VISUAL_MODEL_REVISION,
    WORKER_VERSION,
    _proxy_upload_candidates,
    assign_anonymous_speakers,
    build_perception_index,
)
try:
    from product_policy import load_product_policy
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from video_editor.product_policy import load_product_policy
from video_factory import UserFacingError, get_cloud_processing_setting, resolve_project, sha256_file, work_path, write_json


DEFAULT_GPU = "T4"
ALLOWED_GPUS = {"T4", "L4"}
EXPECTED_VISUAL_DIMENSION = 768
TEMPORAL_MAX_FRAMES = 24
TEMPORAL_PROMPT_VERSION = "timeline-json.v1"
PERCEPTION_RUNTIME_PACKAGE_NAMES = {
    "faster-whisper", "ctranslate2", "torch", "torchaudio", "transformers",
    "speechbrain", "numpy", "Pillow", "soundfile", "scipy", "huggingface-hub",
    "num2words", "nvidia-cublas-cu12", "nvidia-cudnn-cu12",
}
TEMPORAL_RUNTIME_PACKAGE_NAMES = {
    "torch", "transformers", "numpy", "Pillow", "huggingface-hub", "decord",
    "nvidia-cublas-cu12", "nvidia-cudnn-cu12",
}


def _find_colab_cli() -> str | None:
    """Resolve the Colab executable through a testable, provider-specific seam."""
    return shutil.which("colab")


def _runtime_packages(stage: str) -> dict[str, str]:
    names = PERCEPTION_RUNTIME_PACKAGE_NAMES if stage == "perception" else TEMPORAL_RUNTIME_PACKAGE_NAMES
    return {name: version for name, version in COLAB_PACKAGE_VERSIONS.items() if name in names}


def _safe_diagnostic(value: Any) -> str:
    """Keep remote error detail useful while excluding credentials and paths."""
    if not isinstance(value, str):
        return ""
    message = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    message = re.sub(r"(?i)\b(hf_[A-Za-z0-9]{8,}|AIza[A-Za-z0-9_-]{20,})\b", "[redacted-token]", message)
    message = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [redacted]", message)
    message = re.sub(
        r"(?i)\b(authorization|api[_-]?key|access[_-]?token|refresh[_-]?token|password|secret)\s*[:=]\s*[^\s,;]+",
        r"\1=[redacted]", message,
    )
    message = re.sub(r"https?://[^\s)]+", "[redacted-url]", message)
    message = re.sub(r"(?<!\w)/(?:content|tmp|root|usr|opt|home|workspace)/[^\s,;]+", "[remote-path]", message)
    lines = [line.strip() for line in message.splitlines() if line.strip()]
    return " | ".join(lines[-6:])[:1000]


def _speech_importance_record(source: str, segment: dict[str, Any], rms_dbfs: float) -> dict[str, Any]:
    """Create transparent, uncalibrated acoustic/transcript candidate signals."""
    start = float(segment["start"])
    end = float(segment["end"])
    duration = max(0.0, end - start)
    words = segment.get("words")
    word_count = (
        sum(1 for word in words if isinstance(word, dict) and isinstance(word.get("word"), str) and word["word"].strip())
        if isinstance(words, list) and words
        else len(re.findall(r"\w+", str(segment.get("text", "")), flags=re.UNICODE))
    )
    word_signal = min(word_count / float(IMPORTANCE_PARAMETERS["word_count_scale"]), 1.0)
    duration_signal = min(duration / float(IMPORTANCE_PARAMETERS["duration_seconds_scale"]), 1.0)
    energy_floor = float(IMPORTANCE_PARAMETERS["rms_dbfs_floor"])
    energy_ceiling = float(IMPORTANCE_PARAMETERS["rms_dbfs_ceiling"])
    energy_signal = max(0.0, min(1.0, (float(rms_dbfs) - energy_floor) / (energy_ceiling - energy_floor)))
    weights = IMPORTANCE_PARAMETERS["weights"]
    return {
        "source": source, "start": start, "end": end,
        "score": round(
            float(weights["words"]) * word_signal
            + float(weights["duration"]) * duration_signal
            + float(weights["energy"]) * energy_signal,
            4,
        ),
        "method": IMPORTANCE_METHOD, "confidence": None,
        "signals": {
            "word_count": word_count, "duration_seconds": round(duration, 3),
            "words_per_second": round(word_count / duration, 3) if duration > 0 else 0.0,
            "rms_dbfs": round(float(rms_dbfs), 2),
            "word_signal": round(word_signal, 4),
            "duration_signal": round(duration_signal, 4),
            "energy_signal": round(energy_signal, 4),
        },
    }


def _worker_source(configuration: dict[str, Any]) -> str:
    """Create a self-contained Colab script with pinned packages and models."""
    import inspect

    config_literal = repr(json.dumps(configuration, ensure_ascii=False, separators=(",", ":")))
    package_literal = repr(configuration.get("packages", COLAB_PACKAGE_VERSIONS))
    assignment_source = inspect.getsource(assign_anonymous_speakers)
    importance_source = inspect.getsource(_speech_importance_record)
    inference = r'''import hashlib, importlib.metadata, json, math, os, re, sys, time, zipfile
from pathlib import Path
from typing import Any

__ASSIGNMENT_HELPER__
__IMPORTANCE_HELPER__

import numpy as np
import torch

CONFIG = json.loads(__CONFIG__)
IMPORTANCE_METHOD = CONFIG["importance_method"]
IMPORTANCE_PARAMETERS = CONFIG["importance_parameters"]
EXPECTED_PACKAGES = __PACKAGES__
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the allocated Colab runtime.")
GPU_NAME = torch.cuda.get_device_name(0)
if CONFIG["gpu"] not in GPU_NAME.split():
    raise RuntimeError("Allocated GPU differs from the requested GPU; perception was not started.")
package_versions = {name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES}
if package_versions != EXPECTED_PACKAGES:
    raise RuntimeError("The Colab package environment differs from the pinned perception runtime.")

root = Path(CONFIG["work_root"]).resolve()
root.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(CONFIG["bundle_path"]) as archive:
    for member in archive.infolist():
        target = (root / member.filename).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError("Input bundle contains an unsafe path.")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                destination.write(source.read())
manifest = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
if manifest.get("stage") != CONFIG["stage"]:
    raise RuntimeError("Input bundle stage does not match the requested worker stage.")
if manifest.get("privacy_mode") != CONFIG["privacy_mode"]:
    raise RuntimeError("Input bundle privacy mode differs from the requested worker policy.")
if CONFIG["stage"] == "perception":
    if CONFIG["privacy_mode"] == "LOCAL_ONLY" or manifest.get("proxies"):
        raise RuntimeError("Perception stage requires an allowed privacy mode and cannot include temporal proxies.")
    if manifest.get("stage") != "perception":
        raise RuntimeError("Perception stage bundle was malformed.")
if CONFIG["stage"] == "temporal":
    if CONFIG["privacy_mode"] != "MAX_QUALITY" or CONFIG["temporal_backend"] != "smolvlm2":
        raise RuntimeError("Temporal analysis requires MAX_QUALITY and the registered SmolVLM2 backend.")
    expected_sources = sorted(CONFIG["temporal_sources"])
    actual_sources = sorted(item.get("source") for item in manifest.get("proxies", []))
    if not expected_sources or actual_sources != expected_sources:
        raise RuntimeError("Temporal proxy manifest differs from the explicit shortlist.")
    if manifest.get("audio") or manifest.get("frames"):
        raise RuntimeError("Temporal stage may upload only shortlisted derived proxies.")
    for item in manifest["proxies"]:
        if int(item.get("max_height", 0)) > 720:
            raise RuntimeError("Temporal proxy exceeds the 720p transfer limit.")

def safe_path(relative):
    path = (root / relative).resolve()
    if root not in path.parents:
        raise RuntimeError("Bundle manifest references an unsafe file path.")
    return path

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def extract_word(word):
    return {"start": float(word.start), "end": float(word.end), "word": word.word}

def classify_audio_events(audio_entries):
    from huggingface_hub import snapshot_download
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    import soundfile as sf
    import torchaudio

    model_dir = snapshot_download(
        repo_id=CONFIG["audio_event_model"], revision=CONFIG["audio_event_model_revision"],
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
    model = AutoModelForAudioClassification.from_pretrained(
        model_dir, torch_dtype=torch.float16,
    ).to("cuda").eval()
    sample_rate = int(feature_extractor.sampling_rate)
    window_samples = CONFIG["audio_event_window_seconds"] * sample_rate
    output = []
    for audio in audio_entries:
        audio_path = safe_path(audio["path"])
        samples, original_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(samples.mean(axis=1).copy())
        if original_rate != sample_rate:
            waveform = torchaudio.functional.resample(waveform, original_rate, sample_rate)
        for start_sample in range(0, waveform.numel(), window_samples):
            end_sample = min(waveform.numel(), start_sample + window_samples)
            chunk = waveform[start_sample:end_sample].numpy()
            if chunk.size == 0:
                continue
            inputs = feature_extractor(
                chunk, sampling_rate=sample_rate, return_tensors="pt",
            )
            inputs = {key: value.to("cuda") for key, value in inputs.items()}
            if "input_values" in inputs:
                inputs["input_values"] = inputs["input_values"].to(dtype=next(model.parameters()).dtype)
            with torch.inference_mode():
                scores = torch.sigmoid(model(**inputs).logits.float()).flatten()
            values, label_indices = torch.topk(
                scores, k=min(CONFIG["audio_event_top_k"], scores.numel()),
            )
            for score, label_index in zip(values.tolist(), label_indices.tolist()):
                output.append({
                    "source": audio["source"],
                    "start_seconds": round(start_sample / sample_rate, 3),
                    "end_seconds": round(end_sample / sample_rate, 3),
                    "label": str(model.config.id2label[int(label_index)]),
                    "score": float(score), "method": CONFIG["audio_event_method"],
                    "confidence": None,
                })
    metadata = {
        "name": CONFIG["audio_event_model"],
        "revision": CONFIG["audio_event_model_revision"],
        "engine": f"transformers-{importlib.metadata.version('transformers')}",
        "status": "completed", "method": CONFIG["audio_event_method"],
        "sampling_rate": sample_rate,
        "window_seconds": CONFIG["audio_event_window_seconds"],
        "top_k": CONFIG["audio_event_top_k"], "score_calibrated": False,
    }
    del model, feature_extractor
    torch.cuda.empty_cache()
    return output, metadata

runtime = {
    "gpu_name": GPU_NAME,
    "python": sys.version.split()[0],
    "packages": package_versions,
    "cuda_version": torch.version.cuda,
}
models = {}
speech = {
    "transcripts": [], "speech_turns": [], "anonymous_speakers": [],
    "audio_events": [], "audio_events_status": "not_applicable",
    "audio_events_method": CONFIG["audio_event_method"],
    "audio_events_confidence_calibrated": False,
    "importance": [], "importance_status": "not_applicable",
    "importance_method": IMPORTANCE_METHOD, "importance_confidence_calibrated": False,
    "vad": "faster-whisper-vad_filter", "status": "not_applicable",
    "diarization_status": "not_applicable",
    "diarization_method": "speechbrain-ecapa-utterance-clustering.v1",
    "diarization_overlap_aware": False,
    "diarization_confidence_calibrated": False,
}
visual = {
    "items": [], "retrieval_index": None, "scene_similarity": [],
    "similarity_clusters": {}, "similarity_method": "siglip2-cosine-threshold-0.92.v1",
    "similarity_confidence_calibrated": False,
    "exact_hash_duplicates": {}, "event_groups": [],
    "event_groups_status": "not_configured", "status": "not_applicable",
}
temporal = {"backend": CONFIG["temporal_backend"], "candidates": [], "results": [], "event_groups": [], "status": "not_requested"}
started = time.monotonic()

# Each temporal model implements this worker contract and is added to the registry.
def run_smolvlm2_temporal(manifest):
    from decord import VideoReader, cpu
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = CONFIG["temporal_model"]
    revision = CONFIG["temporal_model_revision"]
    processor = AutoProcessor.from_pretrained(model_id, revision=revision)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float16,
    ).to("cuda").eval()
    results = []
    for proxy in manifest["proxies"]:
        proxy_path = safe_path(proxy["path"])
        digest = sha256_file(proxy_path)
        if digest != proxy["sha256"]:
            raise RuntimeError("A shortlisted proxy hash does not match its transfer manifest.")
        reader = VideoReader(str(proxy_path), ctx=cpu(0))
        fps = float(reader.get_avg_fps())
        frame_count = len(reader)
        if not math.isfinite(fps) or fps <= 0 or frame_count < 2:
            raise RuntimeError("A temporal proxy has insufficient valid frames for analysis.")
        duration = frame_count / fps
        sample_count = min(CONFIG["temporal_max_frames"], max(2, int(math.ceil(duration))))
        indices = sorted(set(int(round(value)) for value in np.linspace(0, frame_count - 1, sample_count)))
        frame_batch = reader.get_batch(indices).asnumpy()
        content = [{"type": "text", "text": (
            "Analyze this ordered sequence of sampled frames from one short video proxy. "
            "Return ONLY valid JSON with keys summary (string) and events (array). "
            "Each event must have start_seconds, end_seconds, action, and evidence. "
            "Use only observable actions and event progression; do not identify people or infer private traits. "
            "Timestamps must refer to the supplied clip time. If no distinct event is visible, return events as an empty array."
        )}]
        sampled_timestamps = []
        for index, frame_array in zip(indices, frame_batch):
            timestamp = index / fps
            sampled_timestamps.append(round(timestamp, 3))
            content.append({"type": "text", "text": f"Frame timestamp: {timestamp:.3f} seconds."})
            content.append({"type": "image", "image": Image.fromarray(frame_array).convert("RGB")})
        messages = [{"role": "user", "content": content}]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        for key, tensor in list(inputs.items()):
            if torch.is_floating_point(tensor):
                inputs[key] = tensor.to(dtype=torch.float16)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, do_sample=False, max_new_tokens=384)
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        raw = processor.decode(generated, skip_special_tokens=True).strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("SmolVLM2 did not return valid timeline JSON.") from exc
        summary = parsed.get("summary")
        events = parsed.get("events")
        if not isinstance(summary, str) or not summary.strip() or not isinstance(events, list):
            raise RuntimeError("SmolVLM2 timeline JSON is missing required fields.")
        normalized_events = []
        for event in events:
            if not isinstance(event, dict):
                raise RuntimeError("SmolVLM2 timeline contains an invalid event record.")
            start = float(event.get("start_seconds", -1))
            end = float(event.get("end_seconds", -1))
            action = event.get("action")
            evidence = event.get("evidence")
            if (
                not math.isfinite(start) or not math.isfinite(end) or start < 0
                or end <= start or end > duration + 0.05
                or not isinstance(action, str) or not action.strip()
                or not isinstance(evidence, str) or not evidence.strip()
            ):
                raise RuntimeError("SmolVLM2 timeline contains an invalid time range or event description.")
            normalized_events.append({
                "start_seconds": round(start, 3), "end_seconds": round(end, 3),
                "action": action.strip(), "evidence": evidence.strip(),
                "confidence": None,
            })
        normalized_events.sort(key=lambda event: event["start_seconds"])
        results.append({
            "source": proxy["source"], "proxy_sha256": digest,
            "duration_seconds": round(duration, 3),
            "sampled_timestamps_seconds": sampled_timestamps,
            "summary": summary.strip(), "events": normalized_events,
            "status": "completed", "confidence_calibrated": False,
        })
    event_groups = [
        {
            "group_id": f"temporal-{item['proxy_sha256'][:16]}",
            "source": item["source"], "summary": item["summary"],
            "events": item["events"],
        }
        for item in results
    ]
    temporal = {
        "backend": "smolvlm2", "model": model_id, "revision": revision,
        "candidates": [{"source": item["source"], "sha256": item["sha256"], "status": "completed"} for item in manifest["proxies"]],
        "results": results, "event_groups": event_groups, "status": "completed",
        "prompt_version": CONFIG["temporal_prompt_version"],
    }
    model_metadata = {
        "name": model_id, "revision": revision, "status": "completed",
        "precision": "float16", "backend": "smolvlm2",
    }
    return temporal, model_metadata

TEMPORAL_ADAPTER_REGISTRY = {"smolvlm2": run_smolvlm2_temporal}

if CONFIG["stage"] == "perception":
    audio_entries = manifest.get("audio", [])
    if audio_entries:
        from faster_whisper import WhisperModel
        from huggingface_hub import snapshot_download
        from speechbrain.inference.classifiers import EncoderClassifier
        import soundfile as sf
        import torchaudio

        whisper_dir = snapshot_download(
            repo_id=CONFIG["speech_model_repo"], revision=CONFIG["speech_model_revision"],
        )
        whisper = WhisperModel(whisper_dir, device="cuda", compute_type="float16")
        for audio in audio_entries:
            audio_path = safe_path(audio["path"])
            if sha256_file(audio_path) != audio["sha256"]:
                raise RuntimeError("An uploaded audio hash does not match its transfer manifest.")
            segments, info = whisper.transcribe(
                str(audio_path), word_timestamps=True, vad_filter=True,
                language=audio.get("language"),
            )
            segment_values = []
            for segment in segments:
                segment_values.append({
                    "start": float(segment.start), "end": float(segment.end),
                    "text": segment.text,
                    "words": [extract_word(word) for word in (segment.words or [])],
                })
            speech["transcripts"].append({
                "source": audio["source"], "source_sha256": audio["sha256"],
                "duration_seconds": float(getattr(info, "duration", 0.0)),
                "language": getattr(info, "language", None), "segments": segment_values,
                "model": CONFIG["speech_model"],
                "model_repo": CONFIG["speech_model_repo"],
                "model_revision": CONFIG["speech_model_revision"],
            })
        del whisper
        torch.cuda.empty_cache()

        diar_dir = snapshot_download(
            repo_id=CONFIG["diarization_model"], revision=CONFIG["diarization_model_revision"],
        )
        diarizer = EncoderClassifier.from_hparams(
            source=diar_dir, savedir=str(root / "speechbrain-cache"),
            run_opts={"device": "cuda"},
        )
        for audio, transcript in zip(audio_entries, speech["transcripts"]):
            audio_path = safe_path(audio["path"])
            samples, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
            mono_samples = samples.mean(axis=1)
            for segment in transcript["segments"]:
                left = max(0, int(float(segment["start"]) * sample_rate))
                right = min(len(mono_samples), int(float(segment["end"]) * sample_rate))
                clip = mono_samples[left:right]
                rms = float(np.sqrt(np.mean(np.square(clip, dtype=np.float64)))) if clip.size else 0.0
                rms_dbfs = 20.0 * math.log10(max(rms, 1e-6))
                speech["importance"].append(_speech_importance_record(audio["source"], segment, rms_dbfs))
            mono = torch.from_numpy(samples.mean(axis=1).copy())
            if sample_rate != 16000:
                mono = torchaudio.functional.resample(mono, sample_rate, 16000)
            vectors = []
            for segment in transcript["segments"]:
                start = max(0, int(segment["start"] * 16000))
                end = min(mono.numel(), int(segment["end"] * 16000))
                if (end - start) / 16000.0 < 1.5:
                    vectors.append(None)
                    continue
                signal = mono[start:end].unsqueeze(0).to("cuda")
                with torch.inference_mode():
                    encoded = diarizer.encode_batch(signal).detach().float().cpu().numpy().reshape(-1)
                vectors.append([float(value) for value in encoded.tolist()])
            assignments = assign_anonymous_speakers(transcript["segments"], vectors)
            for assignment in assignments:
                speech["speech_turns"].append({"source": audio["source"], **assignment})
            speakers = sorted({
                item["speaker_id"] for item in assignments if item["speaker_id"] != "unknown"
            })
            speech["anonymous_speakers"].extend({
                "source": audio["source"], "speaker_id": speaker,
                "utterance_count": sum(1 for item in assignments if item["speaker_id"] == speaker),
                "scope": "source_only",
            } for speaker in speakers)
        del diarizer
        torch.cuda.empty_cache()
        speech["importance_status"] = "completed"
        speech["importance_method"] = IMPORTANCE_METHOD
        models["speech_importance"] = {
            "method": IMPORTANCE_METHOD,
            "parameters": CONFIG["importance_parameters"],
            "status": "completed", "confidence_calibrated": False,
            "interpretation": "uncalibrated_candidate_signal_not_semantic_importance",
        }
        speech["audio_events"], models["audio_events"] = classify_audio_events(audio_entries)
        speech["audio_events_status"] = "completed"
        speech["audio_events_method"] = CONFIG["audio_event_method"]
        speech["audio_events_confidence_calibrated"] = False
        speech["status"] = "completed"
        speech["diarization_status"] = "completed"
        models["speech"] = {
            "name": CONFIG["speech_model"], "engine": CONFIG["speech_engine_version"],
            "repo": CONFIG["speech_model_repo"], "revision": CONFIG["speech_model_revision"],
            "status": "completed",
        }
        models["diarization"] = {
            "name": CONFIG["diarization_model"], "revision": CONFIG["diarization_model_revision"],
            "status": "completed", "method": speech["diarization_method"],
            "overlap_aware": False, "confidence_calibrated": False,
        }

    frame_entries = manifest.get("frames", [])
    if frame_entries:
        from PIL import Image
        from transformers import AutoModel, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            CONFIG["visual_model"], revision=CONFIG["visual_model_revision"],
        )
        encoder = AutoModel.from_pretrained(
            CONFIG["visual_model"], revision=CONFIG["visual_model_revision"],
            torch_dtype=torch.float16,
        ).to("cuda").eval()
        values_by_id = {}
        for frame in frame_entries:
            frame_path = safe_path(frame["path"])
            digest = sha256_file(frame_path)
            if digest != frame["sha256"]:
                raise RuntimeError("An uploaded representative-frame hash does not match its manifest.")
            with Image.open(frame_path) as opened:
                image = opened.convert("RGB")
            inputs = processor(images=[image], return_tensors="pt")
            inputs = inputs.to("cuda")
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
            with torch.inference_mode():
                features = encoder.get_image_features(**inputs).detach().float().cpu().numpy()
            if features.ndim != 2 or features.shape != (1, CONFIG["visual_dimension"]):
                raise RuntimeError("SigLIP2 returned an unexpected embedding dimension.")
            if not np.isfinite(features).all():
                raise RuntimeError("SigLIP2 returned a non-finite image embedding.")
            vector = features[0].astype(np.float32)
            norm = float(np.linalg.norm(vector))
            if not math.isfinite(norm) or norm <= 1e-12:
                raise RuntimeError("SigLIP2 returned an invalid zero-norm embedding.")
            item = {
                "id": frame["id"], "path": frame["path"], "source": frame.get("source"),
                "timestamp_seconds": frame.get("timestamp_seconds"), "sha256": digest,
                "embedding": {
                    "status": "completed", "model": CONFIG["visual_model"],
                    "revision": CONFIG["visual_model_revision"],
                    "dimension": CONFIG["visual_dimension"],
                    "values": [float(value) for value in vector.tolist()], "reference": None,
                },
            }
            visual["items"].append(item)
            values_by_id[item["id"]] = vector / norm

        ids = list(values_by_id)
        parent = {frame_id: frame_id for frame_id in ids}
        def find(frame_id):
            while parent[frame_id] != frame_id:
                parent[frame_id] = parent[parent[frame_id]]
                frame_id = parent[frame_id]
            return frame_id
        pairs = []
        for index, frame_id in enumerate(ids):
            for other_id in ids[index + 1:]:
                score = float(np.dot(values_by_id[frame_id], values_by_id[other_id]))
                pairs.append({"a": frame_id, "b": other_id, "cosine": round(score, 6)})
                if score >= 0.92:
                    left, right = find(frame_id), find(other_id)
                    if left != right:
                        parent[right] = left
        roots = {}
        for frame_id in ids:
            root_id = find(frame_id)
            roots.setdefault(root_id, f"similarity-{len(roots) + 1:04d}")
        visual["similarity_clusters"] = {frame_id: roots[find(frame_id)] for frame_id in ids}
        visual["scene_similarity"] = sorted(pairs, key=lambda value: value["cosine"], reverse=True)[:100]
        visual["retrieval_index"] = {
            "metric": "cosine", "ids": ids, "dimension": CONFIG["visual_dimension"],
            "model": CONFIG["visual_model"], "revision": CONFIG["visual_model_revision"],
        }
        visual["status"] = "completed"
        models["visual"] = {
            "name": CONFIG["visual_model"], "revision": CONFIG["visual_model_revision"],
            "status": "completed", "dimension": CONFIG["visual_dimension"],
            "engine": CONFIG["visual_engine_version"],
        }
    elif not manifest.get("frames"):
        visual["status"] = "not_applicable"
        models["visual"] = {
            "name": CONFIG["visual_model"], "revision": CONFIG["visual_model_revision"],
            "status": "not_applicable", "dimension": CONFIG["visual_dimension"],
            "engine": CONFIG["visual_engine_version"],
        }

elif CONFIG["stage"] == "temporal":
    adapter = TEMPORAL_ADAPTER_REGISTRY.get(CONFIG["temporal_backend"])
    if adapter is None:
        raise RuntimeError("No registered temporal analysis backend is available.")
    temporal, models["temporal"] = adapter(manifest)
else:
    raise RuntimeError("Unknown perception worker stage.")

expected_audio_sources = sorted(item["source"] for item in manifest.get("audio", []))
expected_frame_ids = sorted(item["id"] for item in manifest.get("frames", []))
if CONFIG["stage"] == "perception":
    if sorted(item["source"] for item in speech["transcripts"]) != expected_audio_sources:
        raise RuntimeError("Speech results do not cover the uploaded audio manifest.")
    if sorted(item["id"] for item in visual["items"]) != expected_frame_ids:
        raise RuntimeError("Visual results do not cover the uploaded frame manifest.")
    speech["status"] = speech["status"] if expected_audio_sources else "not_applicable"
    speech["diarization_status"] = speech["diarization_status"] if expected_audio_sources else "not_applicable"
    top_status = "completed"
else:
    top_status = "completed" if temporal["status"] == "completed" else "failed"

result = {
    "schema_version": "perception-index.v1", "worker_version": CONFIG["worker_version"],
    "stage": CONFIG["stage"], "status": top_status,
    "privacy": {"mode": CONFIG["privacy_mode"], "original_media_uploaded": False},
    "compute": {"gpu_policy": CONFIG["gpu"], "gpu_name": GPU_NAME, "premium_gpu_allowed": False},
    "runtime": runtime, "models": models, "speech": speech,
    "visual": visual, "temporal": temporal,
    "transfer_manifest": manifest,
    "elapsed_seconds": round(time.monotonic() - started, 3),
}
output_path = Path(CONFIG["output_path"])
output_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
temporary_path.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary_path, output_path)
'''
    inference = inference.replace("__ASSIGNMENT_HELPER__", assignment_source)
    inference = inference.replace("__IMPORTANCE_HELPER__", importance_source)
    inference = inference.replace("__CONFIG__", config_literal).replace("__PACKAGES__", package_literal)
    source = r'''import importlib.util, json, os, re, subprocess, sys
from typing import Any

CONFIG = json.loads(__CONFIG__)
PACKAGES = __PACKAGES__
__SAFE_DIAGNOSTIC_HELPER__

def _run_checked(command, *, phase, timeout, env=None):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False,
            timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        detail = _safe_diagnostic(exc.stderr or "")
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"{phase} timed out{suffix}") from None
    if result.returncode:
        detail = _safe_diagnostic(result.stderr or "")
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"{phase} exited with code {result.returncode}{suffix}")
    return result

def _write_failure(error, phase):
    try:
        payload = {
            "schema_version": "perception-index.v1",
            "worker_version": CONFIG.get("worker_version"),
            "stage": CONFIG.get("stage"), "status": "failed",
            "diagnostic": {
                "phase": str(phase)[:80], "error_type": type(error).__name__[:80],
                "message": _safe_diagnostic(str(error)),
            },
        }
        output_path = CONFIG.get("output_path")
        temporary_path = output_path + ".tmp"
        with open(temporary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
        os.replace(temporary_path, output_path)
    except Exception:
        # Do not print worker context or fall back to a success-shaped result.
        pass

phase = "gpu_probe"
hardware = _run_checked(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    phase=phase, timeout=30,
).stdout.strip().splitlines()
if len(hardware) != 1 or CONFIG["gpu"] not in hardware[0].split():
    raise RuntimeError("Allocated GPU differs from the requested GPU; perception was not started.")
phase = "package_install"
_run_checked(
    [sys.executable, "-m", "pip", "install", "--quiet", *[f"{name}=={version}" for name, version in PACKAGES.items()]],
    phase=phase, timeout=1800,
)
importlib.invalidate_caches()
phase = "cuda_library_setup"
library_dirs = []
for module in ("nvidia.cublas.lib", "nvidia.cudnn.lib"):
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Required pinned CUDA libraries are missing.")
    library_dirs.extend(spec.submodule_search_locations)
environment = os.environ.copy()
environment["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs + [environment.get("LD_LIBRARY_PATH", "")])
inference_source = __INFERENCE__
phase = f"{CONFIG['stage']}_inference"
_run_checked(
    [sys.executable, "-c", inference_source], env=environment,
    phase=phase, timeout=__TIMEOUT__,
)
'''
    safe_diagnostic_source = inspect.getsource(_safe_diagnostic)
    source = source.replace("__SAFE_DIAGNOSTIC_HELPER__", safe_diagnostic_source)
    source = source.replace("__CONFIG__", config_literal)
    source = source.replace("__PACKAGES__", package_literal)
    source = source.replace("__INFERENCE__", repr(inference))
    source = source.replace("__TIMEOUT__", str(configuration.get("timeout_seconds", 10_000)))
    import textwrap
    runtime_start = source.index("phase = \"gpu_probe\"")
    prefix, runtime = source[:runtime_start], source[runtime_start:]
    source = (
        prefix + "try:\n" + textwrap.indent(textwrap.dedent(runtime), "    ")
        + "\nexcept Exception as exc:\n    _write_failure(exc, phase)\n"
    )
    return source


def _bundle(
    project: Path,
    index: dict[str, Any],
    destination: Path,
    *,
    stage: str,
    privacy_mode: str,
    selected_sources: list[str] | None = None,
) -> dict[str, Any]:
    """Build a hash-verified stage-specific bundle with no implicit shortlist."""
    if stage not in {"perception", "temporal"}:
        raise UserFacingError(f"Unknown Colab perception stage {stage!r}.")
    entries: dict[str, Any] = {
        "stage": stage, "privacy_mode": privacy_mode,
        "audio": [], "frames": [], "proxies": [],
    }
    if stage == "perception":
        if privacy_mode == "LOCAL_ONLY":
            raise UserFacingError("LOCAL_ONLY forbids Colab perception uploads.")
        eligible = index.get("privacy", {}).get("eligible_uploads", {})
        audio = eligible.get("audio", []) if isinstance(eligible, dict) else []
        frames = eligible.get("representative_frames", []) if isinstance(eligible, dict) else []
        visual_items = index.get("visual", {}).get("items", [])
        frame_by_path = {
            item.get("path"): item for item in visual_items
            if isinstance(item, dict) and isinstance(item.get("path"), str)
        }
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for number, item in enumerate(audio, start=1):
                if not isinstance(item, dict) or not item.get("path") or not item.get("source"):
                    raise UserFacingError("Audio transfer plan contains an invalid entry.")
                path = project / item["path"]
                digest = str(item.get("sha256") or "")
                relative_path = path.resolve(strict=False).relative_to(project.resolve(strict=True)).as_posix()
                if not relative_path.startswith(("work/", "outputs/work/")):
                    raise UserFacingError(f"Audio transfer is not a derived project artifact: {item.get('source')}.")
                if not path.is_file() or len(digest) != 64 or sha256_file(path) != digest:
                    raise UserFacingError(f"Derived audio is missing or stale: {item.get('source')}.")
                arcname = f"audio/{number:05d}-{digest[:16]}{path.suffix.lower()}"
                archive.write(path, arcname=arcname)
                entries["audio"].append({"source": item["source"], "sha256": digest, "path": arcname})
            for number, relative in enumerate(frames, start=1):
                source_item = frame_by_path.get(relative)
                if not isinstance(source_item, dict) or not source_item.get("id") or not source_item.get("source"):
                    raise UserFacingError(f"Representative frame has no index record: {relative}.")
                path = project / relative
                digest = str(source_item.get("sha256") or "")
                relative_path = path.resolve(strict=False).relative_to(project.resolve(strict=True)).as_posix()
                if not relative_path.startswith(("work/", "outputs/work/")):
                    raise UserFacingError(f"Representative frame is not a derived project artifact: {relative}.")
                if not path.is_file() or len(digest) != 64 or sha256_file(path) != digest:
                    raise UserFacingError(f"Representative frame is missing or stale: {relative}.")
                arcname = f"frames/{number:05d}-{source_item['id']}{path.suffix.lower()}"
                archive.write(path, arcname=arcname)
                entries["frames"].append({
                    "id": source_item["id"], "source": source_item["source"],
                    "timestamp_seconds": source_item.get("timestamp_seconds"),
                    "sha256": digest, "path": arcname,
                })
            archive.writestr("bundle.json", json.dumps(entries, ensure_ascii=False, allow_nan=False))
        return entries

    if privacy_mode != "MAX_QUALITY":
        raise UserFacingError("Temporal deep analysis requires MAX_QUALITY privacy mode.")
    selected = list(selected_sources or [])
    if not selected:
        raise UserFacingError("Temporal analysis requires an explicit source shortlist.")
    candidates = _proxy_upload_candidates(project, selected, privacy_mode)
    if sorted(item["source"] for item in candidates) != sorted(selected):
        raise UserFacingError("Derived proxies do not exactly match the explicit temporal shortlist.")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for number, item in enumerate(candidates, start=1):
            path = project / item["path"]
            relative_path = path.resolve(strict=False).relative_to(project.resolve(strict=True)).as_posix()
            if not relative_path.startswith("work/"):
                raise UserFacingError(f"Shortlisted proxy is not a derived project work artifact: {item['source']}.")
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise UserFacingError(f"Shortlisted proxy is missing or stale: {item['source']}.")
            arcname = f"proxies/{number:05d}-{item['sha256'][:16]}{path.suffix.lower()}"
            archive.write(path, arcname=arcname)
            entries["proxies"].append({
                "source": item["source"], "sha256": item["sha256"],
                "path": arcname, "max_height": item["max_height"],
            })
        archive.writestr("bundle.json", json.dumps(entries, ensure_ascii=False, allow_nan=False))
    return entries


def _merge_remote_index(
    project: Path,
    base: dict[str, Any],
    remote: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    """Merge validated remote evidence without changing local editorial decisions."""
    merged = json.loads(json.dumps(base))
    cache_key = str(base.get("cache_key") or "")
    merged["worker_version"] = remote["worker_version"]
    merged.setdefault("privacy", {}).update(remote["privacy"])
    merged["privacy"]["original_media_uploaded"] = False
    merged["privacy"]["cloud_status"] = "completed"
    merged_compute = merged.setdefault("compute", {})
    stage_compute = json.loads(json.dumps(remote["compute"]))
    merged_compute.setdefault("stages", {})[stage] = stage_compute
    if stage == "perception":
        merged_compute.update(stage_compute)
        merged_compute["stages"] = {stage: stage_compute}
    merged.setdefault("runtime", {}).setdefault("stages", {})[stage] = remote["runtime"]
    for name, value in remote.get("models", {}).items():
        if isinstance(value, dict):
            merged.setdefault("models", {}).setdefault(name, {}).update(value)

    if stage == "perception":
        local_transcripts = merged.get("speech", {}).get("transcripts", [])
        merged_speech = json.loads(json.dumps(remote["speech"]))
        uploaded_sources = {item.get("source") for item in remote["speech"].get("transcripts", []) if isinstance(item, dict)}
        retained_local = [
            item for item in local_transcripts
            if isinstance(item, dict) and item.get("source") not in uploaded_sources
        ]
        merged_speech["transcripts"] = retained_local + merged_speech.get("transcripts", [])
        merged["speech"] = merged_speech
        local_visual = merged.setdefault("visual", {})
        remote_visual = remote["visual"]
        remote_by_id = {item["id"]: item for item in remote_visual["items"]}
        embedding_dir = project / "work" / "perception-cache" / cache_key / "embeddings"
        for item in local_visual.get("items", []):
            remote_item = remote_by_id[item["id"]]
            embedding = remote_item["embedding"]
            embedding_dir.mkdir(parents=True, exist_ok=True)
            reference = embedding_dir / f"{item['id']}.json"
            write_json(reference, {
                "model": embedding["model"], "revision": embedding["revision"],
                "dimension": embedding["dimension"], "values": embedding["values"],
            })
            item["embedding"] = {
                "status": "completed", "model": embedding["model"],
                "revision": embedding["revision"], "dimension": embedding["dimension"],
                "reference": reference.relative_to(project).as_posix(),
            }
            item["similarity_cluster_id"] = remote_visual["similarity_clusters"][item["id"]]
        for key in ("retrieval_index", "scene_similarity", "similarity_clusters"):
            local_visual[key] = remote_visual[key]
        # Exact-hash duplicate IDs are Mac-derived; semantic event groups are
        # not produced by SigLIP similarity clustering.
        local_visual["event_groups"] = []
        local_visual["event_groups_status"] = "not_configured"
        local_visual["status"] = remote_visual["status"]
        merged.setdefault("temporal", {}).update({
            "backend": base.get("temporal", {}).get("backend", "none"),
            "candidates": [], "results": [], "event_groups": [],
            "status": "awaiting_shortlist" if base.get("temporal", {}).get("backend") != "none" else "not_requested",
        })
    elif stage == "temporal":
        merged.setdefault("models", {})["temporal"] = remote["models"]["temporal"]
        merged["temporal"] = json.loads(json.dumps(remote["temporal"]))
    else:
        raise UserFacingError(f"Unknown Colab perception stage {stage!r}.")
    merged["cache_key"] = cache_key
    return merged


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _validate_temporal_payload(
    temporal: Any,
    selected_sources: list[str],
    proxies: list[dict[str, Any]],
) -> None:
    adapter = TEMPORAL_ADAPTERS.get("smolvlm2")
    if not adapter:
        raise UserFacingError("No temporal analysis adapter is registered.")
    if not isinstance(temporal, dict):
        raise UserFacingError("Temporal result is missing its result object.")
    expected = {item["source"]: item["sha256"] for item in proxies}
    if sorted(expected) != sorted(selected_sources):
        raise UserFacingError("Temporal proxy manifest differs from the requested shortlist.")
    if (
        temporal.get("backend") != "smolvlm2"
        or temporal.get("model") != adapter["model"]
        or temporal.get("revision") != adapter["revision"]
        or temporal.get("status") != "completed"
        or temporal.get("prompt_version") != TEMPORAL_PROMPT_VERSION
    ):
        raise UserFacingError("Temporal backend/model did not complete with the pinned SmolVLM2 contract.")
    candidates = temporal.get("candidates")
    results = temporal.get("results")
    if not isinstance(candidates, list) or not isinstance(results, list):
        raise UserFacingError("Temporal output is missing candidate or result arrays.")
    if sorted((item.get("source"), item.get("sha256"), item.get("status")) for item in candidates if isinstance(item, dict)) != sorted(
        (source, digest, "completed") for source, digest in expected.items()
    ):
        raise UserFacingError("Temporal candidate results do not exactly match the selected proxy manifest.")
    if sorted(item.get("source") for item in results if isinstance(item, dict)) != sorted(selected_sources):
        raise UserFacingError("Temporal analysis did not return exactly one result per shortlisted source.")
    if len(results) != len(selected_sources):
        raise UserFacingError("Temporal analysis returned duplicate or missing source results.")
    for result in results:
        source = result["source"]
        if result.get("proxy_sha256") != expected[source] or result.get("status") != "completed":
            raise UserFacingError(f"Temporal result failed source/hash verification for {source!r}.")
        duration = result.get("duration_seconds")
        if not _finite_number(duration) or float(duration) <= 0:
            raise UserFacingError(f"Temporal result has an invalid duration for {source!r}.")
        if not isinstance(result.get("summary"), str) or not result["summary"].strip():
            raise UserFacingError(f"Temporal result has no event progression summary for {source!r}.")
        timestamps = result.get("sampled_timestamps_seconds")
        if not isinstance(timestamps, list) or len(timestamps) < 2 or any(not _finite_number(item) for item in timestamps):
            raise UserFacingError(f"Temporal result has invalid sampled frame timestamps for {source!r}.")
        if any(float(value) < 0 or float(value) > float(duration) + 0.05 for value in timestamps):
            raise UserFacingError(f"Temporal sampled timestamps exceed the clip duration for {source!r}.")
        if any(float(right) <= float(left) for left, right in zip(timestamps, timestamps[1:])):
            raise UserFacingError(f"Temporal sampled timestamps are not strictly increasing for {source!r}.")
        if result.get("confidence_calibrated") is not False:
            raise UserFacingError("Temporal confidence must be explicitly marked uncalibrated.")
        events = result.get("events")
        if not isinstance(events, list):
            raise UserFacingError(f"Temporal result has no event array for {source!r}.")
        previous_start = -1.0
        for event in events:
            if not isinstance(event, dict):
                raise UserFacingError(f"Temporal result contains an invalid event for {source!r}.")
            start = event.get("start_seconds")
            end = event.get("end_seconds")
            if (
                not _finite_number(start) or not _finite_number(end)
                or float(start) < 0 or float(end) <= float(start)
                or float(end) > float(duration) + 0.05
                or float(start) < previous_start
                or not isinstance(event.get("action"), str) or not event["action"].strip()
                or not isinstance(event.get("evidence"), str) or not event["evidence"].strip()
                or event.get("confidence") is not None
            ):
                raise UserFacingError(f"Temporal event has invalid timing, evidence, or confidence for {source!r}.")
            previous_start = float(start)
    groups = temporal.get("event_groups")
    expected_groups = [
        {
            "group_id": f"temporal-{expected[result['source']][:16]}",
            "source": result["source"], "summary": result["summary"],
            "events": result["events"],
        }
        for result in results
    ]
    if groups != expected_groups:
        raise UserFacingError("Temporal event groups do not match the per-source timeline results.")


def _validate_remote_result(
    remote: Any,
    base: dict[str, Any],
    *,
    stage: str,
    manifest: dict[str, Any],
    gpu: str,
    selected_sources: list[str] | None = None,
) -> None:
    """Fail closed unless every required remote stage and transfer is verified."""
    if not isinstance(remote, dict):
        raise UserFacingError("Colab perception result has an invalid root object.")
    if (
        remote.get("schema_version") != "perception-index.v1"
        or remote.get("worker_version") != WORKER_VERSION
        or remote.get("stage") != stage
    ):
        raise UserFacingError("Colab perception worker did not report the expected completed stage/version.")
    if remote.get("status") != "completed":
        if remote.get("status") == "failed":
            diagnostic = remote.get("diagnostic")
            if isinstance(diagnostic, dict):
                phase = diagnostic.get("phase")
                error_type = diagnostic.get("error_type")
                message = _safe_diagnostic(diagnostic.get("message"))
                if not isinstance(phase, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", phase):
                    phase = "remote_worker"
                if not isinstance(error_type, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.]{0,79}", error_type):
                    error_type = "RemoteError"
                suffix = f": {message}" if message else ""
                raise UserFacingError(f"Colab {stage} worker failed during {phase} ({error_type}){suffix}.")
        raise UserFacingError("Colab perception worker did not report the expected completed stage/version.")
    privacy = remote.get("privacy")
    if not isinstance(privacy, dict) or privacy.get("mode") != base.get("privacy", {}).get("mode") or privacy.get("original_media_uploaded") is not False:
        raise UserFacingError("Colab perception result failed privacy-mode verification.")
    compute = remote.get("compute")
    if (
        not isinstance(compute, dict) or compute.get("gpu_policy") != gpu
        or compute.get("premium_gpu_allowed") is not False
        or not isinstance(compute.get("gpu_name"), str) or gpu not in compute["gpu_name"].split()
    ):
        raise UserFacingError("Colab perception result does not match the requested allowed GPU.")
    runtime = remote.get("runtime")
    if (
        not isinstance(runtime, dict) or runtime.get("gpu_name") != compute["gpu_name"]
        or not isinstance(runtime.get("python"), str) or not runtime.get("cuda_version")
        or runtime.get("packages") != _runtime_packages(stage)
    ):
        raise UserFacingError("Colab perception runtime does not match the pinned package contract.")
    if remote.get("transfer_manifest") != manifest:
        raise UserFacingError("Colab worker transfer manifest differs from the exact local bundle manifest.")
    models = remote.get("models")
    if not isinstance(models, dict):
        raise UserFacingError("Colab result omitted model execution metadata.")

    if stage == "perception":
        if manifest.get("stage") != "perception" or manifest.get("proxies"):
            raise UserFacingError("Perception stage bundle contains an invalid temporal proxy transfer.")
        expected_audio = {item["source"]: item["sha256"] for item in manifest.get("audio", [])}
        expected_frames = {item["id"]: item for item in manifest.get("frames", [])}
        speech = remote.get("speech")
        visual = remote.get("visual")
        if not isinstance(speech, dict) or not isinstance(visual, dict):
            raise UserFacingError("Colab perception output omitted speech or visual results.")
        if speech.get("vad") != "faster-whisper-vad_filter":
            raise UserFacingError("Speech output did not confirm the configured VAD stage.")
        audio_events = speech.get("audio_events")
        importance = speech.get("importance")
        if not isinstance(audio_events, list) or not isinstance(importance, list):
            raise UserFacingError("Speech event and importance outputs must be arrays.")
        expected_auxiliary_status = "completed" if expected_audio else "not_applicable"
        if speech.get("audio_events_status") != expected_auxiliary_status:
            raise UserFacingError("Audio event candidates did not complete for the uploaded audio workload.")
        if speech.get("importance_status") != expected_auxiliary_status:
            raise UserFacingError("Speech importance metadata did not complete for the uploaded audio workload.")
        if (
            speech.get("importance_method") != IMPORTANCE_METHOD
            or speech.get("importance_confidence_calibrated") is not False
            or speech.get("audio_events_method") != AUDIO_EVENT_METHOD
            or speech.get("audio_events_confidence_calibrated") is not False
        ):
            raise UserFacingError("Speech candidate methods must be explicitly identified as uncalibrated heuristics.")
        if not expected_audio and (audio_events or importance):
            raise UserFacingError("Speech candidates were returned without an uploaded audio source.")
        if speech.get("status") != ("completed" if expected_audio else "not_applicable"):
            raise UserFacingError("Speech stage status does not match the uploaded audio workload.")
        if speech.get("diarization_status") != ("completed" if expected_audio else "not_applicable"):
            raise UserFacingError("Anonymous diarization did not complete for every uploaded audio source.")
        if speech.get("diarization_method") != "speechbrain-ecapa-utterance-clustering.v1" or speech.get("diarization_overlap_aware") is not False or speech.get("diarization_confidence_calibrated") is not False:
            raise UserFacingError("Diarization method/limitations are missing from the result contract.")
        transcripts = speech.get("transcripts")
        if not isinstance(transcripts, list) or sorted(item.get("source") for item in transcripts if isinstance(item, dict)) != sorted(expected_audio):
            raise UserFacingError("Transcription results do not cover the exact uploaded audio source set.")
        if len(transcripts) != len(expected_audio):
            raise UserFacingError("Transcription results contain duplicate source records.")
        transcript_segments: dict[tuple[str, str, str], dict[str, Any]] = {}
        source_durations: dict[str, float] = {}
        for transcript in transcripts:
            source = transcript["source"]
            if transcript.get("source_sha256") != expected_audio[source] or transcript.get("model") != SPEECH_MODEL or transcript.get("model_repo") != SPEECH_MODEL_REPO or transcript.get("model_revision") != SPEECH_MODEL_REVISION:
                raise UserFacingError(f"Transcript source/model fingerprint mismatch for {source!r}.")
            duration = transcript.get("duration_seconds")
            if not _finite_number(duration) or float(duration) <= 0:
                raise UserFacingError(f"Transcript omitted a valid audio duration for {source!r}.")
            source_durations[source] = float(duration)
            segments = transcript.get("segments")
            if not isinstance(segments, list):
                raise UserFacingError(f"Transcript segments are invalid for {source!r}.")
            for segment in segments:
                if (
                    not isinstance(segment, dict) or not _finite_number(segment.get("start"))
                    or not _finite_number(segment.get("end")) or float(segment["start"]) < 0
                    or float(segment["end"]) <= float(segment["start"])
                    or not isinstance(segment.get("text"), str)
                    or not isinstance(segment.get("words"), list)
                ):
                    raise UserFacingError(f"Transcript contains an invalid timestamped segment for {source!r}.")
                for word in segment["words"]:
                    if (
                        not isinstance(word, dict) or not _finite_number(word.get("start"))
                        or not _finite_number(word.get("end")) or float(word["start"]) < float(segment["start"]) - 0.05
                        or float(word["end"]) > float(segment["end"]) + 0.05
                        or float(word["end"]) <= float(word["start"])
                        or not isinstance(word.get("word"), str)
                    ):
                        raise UserFacingError(f"Word timestamps are invalid for {source!r}.")
                transcript_segments[(source, str(segment["start"]), str(segment["end"]))] = segment
        turns = speech.get("speech_turns")
        if not isinstance(turns, list) or len(turns) != sum(len(item["segments"]) for item in transcripts):
            raise UserFacingError("Diarization did not return one anonymous turn per transcript segment.")
        if speech.get("anonymous_speakers") is None or not isinstance(speech.get("anonymous_speakers"), list):
            raise UserFacingError("Diarization omitted anonymous source-local speaker summaries.")
        seen_turns = set()
        for turn in turns:
            if not isinstance(turn, dict) or turn.get("source") not in expected_audio:
                raise UserFacingError("Diarization returned a turn for an unuploaded source.")
            identity = (turn["source"], str(turn.get("start")), str(turn.get("end")))
            if identity not in transcript_segments or identity in seen_turns:
                raise UserFacingError("Diarization turns do not map one-to-one to transcript segments.")
            seen_turns.add(identity)
            if turn.get("text") != transcript_segments[identity]["text"]:
                raise UserFacingError("Diarization turn text differs from its timestamped transcript segment.")
            speaker = turn.get("speaker_id")
            if speaker != "unknown" and not re.fullmatch(r"speaker_[0-9]{2}", str(speaker)):
                raise UserFacingError("Diarization emitted a non-anonymous speaker label.")
            if turn.get("confidence") is not None:
                raise UserFacingError("Diarization similarity is heuristic and cannot be labeled as calibrated confidence.")
            similarity = turn.get("similarity_to_cluster")
            if similarity is not None and (not _finite_number(similarity) or float(similarity) < -1.0 or float(similarity) > 1.0):
                raise UserFacingError("Diarization cosine similarity is outside its valid range.")
        for speaker in speech["anonymous_speakers"]:
            if not isinstance(speaker, dict) or speaker.get("source") not in expected_audio or speaker.get("scope") != "source_only" or not re.fullmatch(r"speaker_[0-9]{2}", str(speaker.get("speaker_id", ""))):
                raise UserFacingError("Anonymous speaker summary violates its source-only label contract.")
        if expected_audio:
            expected_speech_model = models.get("speech", {})
            expected_diarization_model = models.get("diarization", {})
            if expected_speech_model.get("name") != SPEECH_MODEL or expected_speech_model.get("repo") != SPEECH_MODEL_REPO or expected_speech_model.get("revision") != SPEECH_MODEL_REVISION or expected_speech_model.get("engine") != SPEECH_ENGINE_VERSION or expected_speech_model.get("status") != "completed":
                raise UserFacingError("Pinned faster-whisper model did not complete successfully.")
            if (
                expected_diarization_model.get("name") != DIARIZATION_MODEL
                or expected_diarization_model.get("revision") != DIARIZATION_MODEL_REVISION
                or expected_diarization_model.get("status") != "completed"
                or expected_diarization_model.get("method") != speech["diarization_method"]
                or expected_diarization_model.get("overlap_aware") is not False
                or expected_diarization_model.get("confidence_calibrated") is not False
            ):
                raise UserFacingError("Pinned SpeechBrain ECAPA diarization did not complete successfully.")
            importance_model = models.get("speech_importance", {})
            if (
                not isinstance(importance_model, dict)
                or importance_model.get("method") != IMPORTANCE_METHOD
                or importance_model.get("parameters") != IMPORTANCE_PARAMETERS
                or importance_model.get("status") != "completed"
                or importance_model.get("confidence_calibrated") is not False
                or importance_model.get("interpretation") != "uncalibrated_candidate_signal_not_semantic_importance"
            ):
                raise UserFacingError("Speech importance method/parameters must be explicitly pinned and uncalibrated.")
            expected_importance_ids = set(transcript_segments)
            seen_importance_ids = set()
            if len(importance) != len(expected_importance_ids):
                raise UserFacingError("Speech importance metadata does not cover each transcript segment exactly once.")
            for item in importance:
                if not isinstance(item, dict):
                    raise UserFacingError("Speech importance metadata contains an invalid record.")
                identity = (item.get("source"), str(item.get("start")), str(item.get("end")))
                if identity not in expected_importance_ids or identity in seen_importance_ids:
                    raise UserFacingError("Speech importance metadata does not map one-to-one to transcript segments.")
                seen_importance_ids.add(identity)
                signals = item.get("signals")
                if (
                    item.get("method") != IMPORTANCE_METHOD or item.get("confidence") is not None
                    or not _finite_number(item.get("score")) or not 0.0 <= float(item["score"]) <= 1.0
                    or not isinstance(signals, dict) or not _finite_number(signals.get("rms_dbfs"))
                ):
                    raise UserFacingError("Speech importance must remain a non-calibrated, explainable heuristic.")
                expected_item = _speech_importance_record(
                    identity[0], transcript_segments[identity], float(signals["rms_dbfs"]),
                )
                if item != expected_item:
                    raise UserFacingError("Speech importance score and exposed signals do not match the pinned heuristic.")

            event_model = models.get("audio_events", {})
            if (
                not isinstance(event_model, dict)
                or event_model.get("name") != AUDIO_EVENT_MODEL
                or event_model.get("revision") != AUDIO_EVENT_MODEL_REVISION
                or event_model.get("status") != "completed"
                or event_model.get("method") != AUDIO_EVENT_METHOD
                or event_model.get("sampling_rate") != 16000
                or event_model.get("window_seconds") != AUDIO_EVENT_WINDOW_SECONDS
                or event_model.get("top_k") != AUDIO_EVENT_TOP_K
                or event_model.get("score_calibrated") is not False
                or event_model.get("engine") != f"transformers-{COLAB_PACKAGE_VERSIONS['transformers']}"
            ):
                raise UserFacingError("Pinned AudioSet event classifier metadata is missing or inconsistent.")
            event_counts: dict[tuple[str, float], int] = {}
            for event in audio_events:
                if not isinstance(event, dict):
                    raise UserFacingError("Audio event candidates contain an invalid record.")
                source = event.get("source")
                start, end = event.get("start_seconds"), event.get("end_seconds")
                if (
                    source not in expected_audio or not _finite_number(start) or not _finite_number(end)
                    or float(start) < 0 or float(end) <= float(start)
                    or float(end) > source_durations[source] + 0.05
                    or float(end) - float(start) > AUDIO_EVENT_WINDOW_SECONDS + 0.01
                    or not math.isclose(float(start) % AUDIO_EVENT_WINDOW_SECONDS, 0.0, abs_tol=0.002)
                    or not isinstance(event.get("label"), str) or not event["label"].strip()
                    or event.get("method") != AUDIO_EVENT_METHOD or event.get("confidence") is not None
                    or not _finite_number(event.get("score")) or not 0 <= float(event["score"]) <= 1
                ):
                    raise UserFacingError("Audio event candidate has invalid source, timing, model score, or calibration metadata.")
                key = (source, float(start))
                event_counts[key] = event_counts.get(key, 0) + 1
            if any(count > AUDIO_EVENT_TOP_K for count in event_counts.values()):
                raise UserFacingError("AudioSet returned more than the configured candidates per time window.")
        frame_items = visual.get("items")
        if not isinstance(frame_items, list) or sorted(item.get("id") for item in frame_items if isinstance(item, dict)) != sorted(expected_frames):
            raise UserFacingError("SigLIP2 results do not cover the exact representative-frame manifest.")
        if len(frame_items) != len(expected_frames) or visual.get("status") != ("completed" if expected_frames else "not_applicable"):
            raise UserFacingError("Visual embedding stage did not complete the required frame set.")
        similarity_clusters = visual.get("similarity_clusters")
        if not isinstance(similarity_clusters, dict) or set(similarity_clusters) != set(expected_frames):
            raise UserFacingError("Visual similarity clusters do not cover the expected frame IDs.")
        for item in frame_items:
            expected = expected_frames[item["id"]]
            embedding = item.get("embedding")
            if item.get("sha256") != expected["sha256"] or item.get("source") != expected["source"] or not isinstance(embedding, dict):
                raise UserFacingError("Visual result source/hash differs from the representative-frame manifest.")
            values = embedding.get("values")
            if (
                embedding.get("status") != "completed" or embedding.get("model") != VISUAL_MODEL
                or embedding.get("revision") != VISUAL_MODEL_REVISION
                or embedding.get("dimension") != EXPECTED_VISUAL_DIMENSION
                or not isinstance(values, list) or len(values) != EXPECTED_VISUAL_DIMENSION
                or any(not _finite_number(value) for value in values)
            ):
                raise UserFacingError("SigLIP2 embedding failed status, revision, dimension, or finite-value verification.")
            if math.sqrt(sum(float(value) * float(value) for value in values)) <= 1e-12:
                raise UserFacingError("SigLIP2 embedding has zero norm.")
        if visual.get("event_groups") != [] or visual.get("event_groups_status") != "not_configured":
            raise UserFacingError("SigLIP2 similarity clusters must not be reported as semantic event groups.")
        if expected_frames:
            visual_model = models.get("visual", {})
            if visual_model.get("name") != VISUAL_MODEL or visual_model.get("revision") != VISUAL_MODEL_REVISION or visual_model.get("dimension") != EXPECTED_VISUAL_DIMENSION or visual_model.get("status") != "completed":
                raise UserFacingError("Pinned SigLIP2 model did not complete successfully.")
        return

    if stage != "temporal":
        raise UserFacingError(f"Unknown Colab perception result stage {stage!r}.")
    if manifest.get("stage") != "temporal" or manifest.get("audio") or manifest.get("frames"):
        raise UserFacingError("Temporal stage must contain only explicitly shortlisted proxies.")
    if privacy.get("mode") != "MAX_QUALITY":
        raise UserFacingError("Temporal result must use MAX_QUALITY privacy mode.")
    sources = list(selected_sources or [])
    proxies = manifest.get("proxies", [])
    if sorted(item.get("source") for item in proxies if isinstance(item, dict)) != sorted(sources) or len(proxies) != len(sources):
        raise UserFacingError("Temporal stage transfer contains sources outside the explicit shortlist.")
    adapter = TEMPORAL_ADAPTERS.get("smolvlm2")
    model = models.get("temporal", {})
    if not adapter or model.get("name") != adapter["model"] or model.get("revision") != adapter["revision"] or model.get("backend") != "smolvlm2" or model.get("precision") != adapter["precision"] or model.get("status") != "completed":
        raise UserFacingError("Pinned SmolVLM2 FP16 temporal backend did not complete successfully.")
    _validate_temporal_payload(remote.get("temporal"), sources, proxies)


def _perception_cache_complete(project: Path, index: dict[str, Any]) -> bool:
    """Recognize only fully validated v2 cache entries before allocating Colab."""
    if not isinstance(index, dict):
        return False
    privacy_value = index.get("privacy", {})
    if not isinstance(privacy_value, dict):
        return False
    eligible = privacy_value.get("eligible_uploads", {})
    expected_audio = eligible.get("audio", []) if isinstance(eligible, dict) else []
    visual_value = index.get("visual", {})
    expected_frames = visual_value.get("items", []) if isinstance(visual_value, dict) else []
    if not isinstance(expected_audio, list) or not isinstance(expected_frames, list):
        return False
    if not expected_audio and not expected_frames:
        return True
    if index.get("worker_version") != WORKER_VERSION:
        return False
    if privacy_value.get("original_media_uploaded") is not False or privacy_value.get("cloud_status") != "completed":
        return False
    models = index.get("models", {})
    if not isinstance(models, dict):
        return False
    speech = index.get("speech", {})
    visual = index.get("visual", {})
    if not isinstance(speech, dict) or not isinstance(visual, dict):
        return False
    if expected_audio:
        if speech.get("status") != "completed" or speech.get("diarization_status") != "completed":
            return False
        if (
            speech.get("audio_events_status") != "completed"
            or speech.get("audio_events_method") != AUDIO_EVENT_METHOD
            or speech.get("audio_events_confidence_calibrated") is not False
            or speech.get("importance_status") != "completed"
            or speech.get("importance_method") != IMPORTANCE_METHOD
            or speech.get("importance_confidence_calibrated") is not False
        ):
            return False
        speech_model = models.get("speech", {})
        diarization_model = models.get("diarization", {})
        audio_event_model = models.get("audio_events", {})
        importance_model = models.get("speech_importance", {})
        if (
            not isinstance(speech_model, dict) or not isinstance(diarization_model, dict)
            or not isinstance(audio_event_model, dict) or not isinstance(importance_model, dict)
        ):
            return False
        if (
            speech_model.get("name") != SPEECH_MODEL
            or speech_model.get("repo") != SPEECH_MODEL_REPO
            or speech_model.get("revision") != SPEECH_MODEL_REVISION
            or speech_model.get("engine") != SPEECH_ENGINE_VERSION
            or speech_model.get("status") != "completed"
            or diarization_model.get("name") != DIARIZATION_MODEL
            or diarization_model.get("revision") != DIARIZATION_MODEL_REVISION
            or diarization_model.get("status") != "completed"
            or audio_event_model.get("name") != AUDIO_EVENT_MODEL
            or audio_event_model.get("revision") != AUDIO_EVENT_MODEL_REVISION
            or audio_event_model.get("status") != "completed"
            or audio_event_model.get("method") != AUDIO_EVENT_METHOD
            or importance_model.get("method") != IMPORTANCE_METHOD
            or importance_model.get("parameters") != IMPORTANCE_PARAMETERS
            or importance_model.get("status") != "completed"
            or importance_model.get("confidence_calibrated") is not False
        ):
            return False
        if sorted(item.get("source") for item in speech.get("transcripts", []) if isinstance(item, dict)) != sorted(item.get("source") for item in expected_audio):
            return False
    elif speech.get("status") not in {"not_applicable", "local_metadata_only"}:
        return False
    if expected_frames:
        if visual.get("status") != "completed":
            return False
        visual_model = models.get("visual", {})
        if not isinstance(visual_model, dict):
            return False
        if (
            visual_model.get("name") != VISUAL_MODEL
            or visual_model.get("revision") != VISUAL_MODEL_REVISION
            or visual_model.get("dimension") != EXPECTED_VISUAL_DIMENSION
            or visual_model.get("status") != "completed"
        ):
            return False
        if set(visual.get("similarity_clusters", {})) != {item.get("id") for item in expected_frames}:
            return False
        for item in expected_frames:
            embedding = item.get("embedding", {})
            reference = embedding.get("reference")
            if embedding.get("status") != "completed" or not reference:
                return False
            path = (project / reference).resolve(strict=False)
            try:
                path.relative_to(project.resolve(strict=True))
            except (ValueError, FileNotFoundError):
                return False
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                return False
            values = record.get("values") if isinstance(record, dict) else None
            if (
                not isinstance(record, dict)
                or record.get("model") != VISUAL_MODEL or record.get("revision") != VISUAL_MODEL_REVISION
                or record.get("dimension") != EXPECTED_VISUAL_DIMENSION
                or not isinstance(values, list) or len(values) != EXPECTED_VISUAL_DIMENSION
                or any(not _finite_number(value) for value in values)
            ):
                return False
    elif visual.get("status") not in {"not_applicable", None}:
        return False
        if visual.get("event_groups") != [] or visual.get("event_groups_status") != "not_configured":
            return False
    return True


def _temporal_cache_key(base: dict[str, Any], selected_sources: list[str], proxies: list[dict[str, Any]]) -> str:
    payload = {
        "base_cache_key": base.get("cache_key"),
        "backend": "smolvlm2", "model": TEMPORAL_ADAPTERS["smolvlm2"]["model"],
        "revision": TEMPORAL_ADAPTERS["smolvlm2"]["revision"], "prompt_version": TEMPORAL_PROMPT_VERSION,
        "packages": COLAB_PACKAGE_VERSIONS,
        "selected_sources": sorted(selected_sources),
        "proxies": sorted((item["source"], item["sha256"]) for item in proxies),
    }
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_index_outputs(project: Path, index: dict[str, Any], *, cache_path: Path | None = None) -> None:
    if cache_path is not None:
        write_json(cache_path, index)
    write_json(project / "work" / "perception_index.json", index)
    write_json(project / "outputs" / "work" / "perception_index.json", index)


def _owned_session_released(cli_path: str, session_name: str) -> bool | None:
    """Check the server's session list; None means the state was not verifiable."""
    try:
        result = subprocess.run(
            [cli_path, "--logtostderr", "sessions"], stdin=subprocess.DEVNULL,
            capture_output=True, text=True, check=False, timeout=45,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if session_name in output:
        return False
    if output.strip():
        return True
    return None


def _release_owned_session(cli_path: str, session_name: str) -> None:
    stop_error: Exception | None = None
    try:
        _run_colab(cli_path, "stop", ["stop", "-s", session_name])
    except UserFacingError as exc:
        stop_error = exc
    import time

    for attempt in range(5):
        state = _owned_session_released(cli_path, session_name)
        if state is True:
            return
        if state is None:
            break
        if attempt < 4:
            time.sleep(1.5)
    detail = "stop reported an error" if stop_error else "the server still lists the session or its state is unknown"
    raise UserFacingError(f"Colab session release could not be verified ({detail}): {session_name}") from stop_error


def _run_remote_stage(
    project: Path,
    index: dict[str, Any],
    *,
    cli_path: str,
    stage: str,
    gpu: str,
    privacy_mode: str,
    temporal_backend: str,
    selected_sources: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    usage = _colab_usage_snapshot(cli_path)
    _print_usage_snapshot(f"before {stage} perception allocation", usage)
    if usage is None:
        raise UserFacingError("Could not verify Colab compute-unit usage; refusing to allocate a GPU session.")
    session_name = f"vf-perception-{secrets.token_hex(16)}"
    nonce = secrets.token_hex(12)
    remote_bundle = f"/content/vf-perception-{nonce}.zip"
    remote_worker = f"/content/vf-perception-worker-{nonce}.py"
    remote_output = f"/content/vf-perception-result-{nonce}.json"
    owned = False
    merged: dict[str, Any] | None = None
    remote_index: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    try:
        staging = work_path(project, "colab-staging", create_dir=True)
        with tempfile.TemporaryDirectory(prefix="video-factory-perception-", dir=str(staging)) as temp:
            root = Path(temp)
            bundle = root / "perception.zip"
            worker = root / "worker.py"
            download = root / "result.json"
            manifest = _bundle(
                project, index, bundle, stage=stage,
                privacy_mode=privacy_mode, selected_sources=selected_sources,
            )
            configuration = {
                "worker_version": WORKER_VERSION, "stage": stage,
                "packages": _runtime_packages(stage),
                "gpu": gpu, "privacy_mode": privacy_mode,
                "temporal_backend": temporal_backend,
                "temporal_sources": sorted(selected_sources or []),
                "speech_model": SPEECH_MODEL, "speech_engine_version": SPEECH_ENGINE_VERSION,
                "speech_model_repo": SPEECH_MODEL_REPO, "speech_model_revision": SPEECH_MODEL_REVISION,
                "diarization_model": DIARIZATION_MODEL,
                "diarization_model_revision": DIARIZATION_MODEL_REVISION,
                "audio_event_model": AUDIO_EVENT_MODEL,
                "audio_event_model_revision": AUDIO_EVENT_MODEL_REVISION,
                "audio_event_method": AUDIO_EVENT_METHOD,
                "audio_event_window_seconds": AUDIO_EVENT_WINDOW_SECONDS,
                "audio_event_top_k": AUDIO_EVENT_TOP_K,
                "importance_method": IMPORTANCE_METHOD,
                "importance_parameters": IMPORTANCE_PARAMETERS,
                "visual_model": VISUAL_MODEL, "visual_model_revision": VISUAL_MODEL_REVISION,
                "visual_engine_version": f"transformers-{COLAB_PACKAGE_VERSIONS['transformers']}",
                "visual_dimension": EXPECTED_VISUAL_DIMENSION,
                "temporal_model": TEMPORAL_ADAPTERS["smolvlm2"]["model"],
                "temporal_model_revision": TEMPORAL_ADAPTERS["smolvlm2"]["revision"],
                "temporal_prompt_version": TEMPORAL_PROMPT_VERSION,
                "temporal_max_frames": TEMPORAL_MAX_FRAMES,
                "bundle_path": remote_bundle, "work_root": f"/content/vf-perception-{nonce}",
                "output_path": remote_output,
                "timeout_seconds": REMOTE_EXEC_TIMEOUT - 1800,
            }
            worker.write_text(_worker_source(configuration), encoding="utf-8")
            # Mark ownership before `new`: a partial allocation must still be stopped.
            owned = True
            _run_colab(cli_path, "new", ["new", "-s", session_name, "--gpu", gpu])
            _run_colab(cli_path, "upload", ["upload", "-s", session_name, str(bundle), remote_bundle])
            _run_colab(cli_path, "upload", ["upload", "-s", session_name, str(worker), remote_worker])
            exec_error: UserFacingError | None = None
            try:
                _run_colab(cli_path, "exec", ["exec", "-s", session_name, "-f", remote_worker, "--timeout", str(REMOTE_EXEC_TIMEOUT)])
            except UserFacingError as exc:
                exec_error = exc
            try:
                _run_colab(cli_path, "download", ["download", "-s", session_name, remote_output, str(download)])
            except UserFacingError:
                if exec_error is not None:
                    raise exec_error from None
                raise
            try:
                remote_index = json.loads(download.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                if exec_error is not None:
                    raise exec_error from None
                raise UserFacingError("Colab perception did not provide valid JSON output.") from exc
            if exec_error is not None and (
                not isinstance(remote_index, dict) or remote_index.get("status") != "failed"
            ):
                raise UserFacingError(
                    "Colab CLI reported an execution failure and did not provide a verified failure receipt; refusing the result."
                ) from exec_error
            _validate_remote_result(
                remote_index, index, stage=stage, manifest=manifest,
                gpu=gpu, selected_sources=selected_sources,
            )
            merged = _merge_remote_index(project, index, remote_index, stage=stage)
    finally:
        if owned:
            _release_owned_session(cli_path, session_name)
            _print_usage_snapshot(f"after verified {stage} cleanup", _colab_usage_snapshot(cli_path))
    if merged is None or remote_index is None or manifest is None:
        raise UserFacingError(f"Colab {stage} perception did not produce a verified result.")
    return merged, remote_index, manifest


def _apply_temporal_cache(index: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any]:
    merged = json.loads(json.dumps(index))
    merged.setdefault("models", {})["temporal"] = cache["model"]
    merged["temporal"] = cache["temporal"]
    return merged


def _save_temporal_cache(path: Path, key: str, remote: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {
        "cache_key": key,
        "model": remote["models"]["temporal"],
        "temporal": remote["temporal"],
    })


def command_colab_perception(args: Any) -> int:
    project = resolve_project(args.project)
    policy = load_product_policy(
        project,
        privacy_mode=getattr(args, "privacy_mode", None),
        gpu_policy=getattr(args, "gpu", None),
        cloud_perception=True,
    )
    requested_gpu = getattr(args, "gpu", None) or DEFAULT_GPU
    backend = (getattr(args, "temporal_backend", None) or policy.temporal_backend or "none").strip().lower()
    selected_sources = list(getattr(args, "temporal_sources", None) or [])
    if backend not in TEMPORAL_BACKENDS:
        raise UserFacingError(f"Unsupported temporal backend {backend!r}; choose none or smolvlm2.")
    if requested_gpu not in ALLOWED_GPUS:
        raise UserFacingError("Colab perception supports T4 by default and L4 only for explicit temporal analysis.")
    if policy.privacy_mode == "LOCAL_ONLY":
        raise UserFacingError("LOCAL_ONLY forbids Colab perception uploads.")
    if selected_sources and (backend != "smolvlm2" or policy.privacy_mode != "MAX_QUALITY"):
        raise UserFacingError("An explicit temporal shortlist requires SmolVLM2 and MAX_QUALITY privacy mode.")
    if requested_gpu == "L4" and not selected_sources:
        raise UserFacingError("L4 is reserved for an explicit MAX_QUALITY SmolVLM2 shortlist; routine perception uses T4.")
    if selected_sources and requested_gpu == "L4" and policy.privacy_mode != "MAX_QUALITY":
        raise UserFacingError("L4 temporal analysis requires MAX_QUALITY privacy mode.")

    # Stage one is always the default T4 perception batch. L4, if requested,
    # can only be used by the separate, explicit temporal shortlist stage.
    index_path = build_perception_index(
        project, privacy_mode=policy.privacy_mode, gpu_policy="T4",
        cloud=True, allow_upload=False, temporal_backend=backend,
    )
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict):
        raise UserFacingError("The local perception index is invalid.")

    perception_complete = _perception_cache_complete(project, index)
    if not perception_complete:
        eligible = index.get("privacy", {}).get("eligible_uploads", {})
        has_inputs = bool(eligible.get("audio") or eligible.get("representative_frames"))
        if has_inputs:
            if get_cloud_processing_setting(project, "colab") != "ask_each_run":
                raise UserFacingError("Set job.yaml cloud_processing.colab: ask_each_run before a Colab perception upload.")
            if not getattr(args, "allow_upload", False):
                raise UserFacingError("Add --allow-upload after reviewing the derived-audio and representative-frame transfer plan.")
            cli_path = _find_colab_cli()
            if cli_path is None:
                raise UserFacingError("The Colab CLI is not installed.")
            index, remote, _ = _run_remote_stage(
                project, index, cli_path=cli_path, stage="perception", gpu="T4",
                privacy_mode=policy.privacy_mode, temporal_backend=backend,
            )
            base_cache = project / "work" / "perception-cache" / f"{index['cache_key']}.json"
            _write_index_outputs(project, index, cache_path=base_cache)
            perception_complete = _perception_cache_complete(project, index)
            if not perception_complete:
                raise UserFacingError("Downloaded perception result failed local cache completeness verification.")
        else:
            print("No eligible derived audio or representative frames; skipped Colab allocation.")
            _write_index_outputs(project, index)

    if selected_sources:
        candidates = _proxy_upload_candidates(project, selected_sources, policy.privacy_mode)
        temporal_key = _temporal_cache_key(index, selected_sources, candidates)
        temporal_cache_path = project / "work" / "perception-cache" / f"temporal-{temporal_key}.json"
        cached_temporal: dict[str, Any] | None = None
        try:
            value = json.loads(temporal_cache_path.read_text(encoding="utf-8"))
            if value.get("cache_key") == temporal_key and isinstance(value.get("model"), dict):
                _validate_temporal_payload(value.get("temporal"), selected_sources, candidates)
                model = value["model"]
                adapter = TEMPORAL_ADAPTERS["smolvlm2"]
                if model.get("name") == adapter["model"] and model.get("revision") == adapter["revision"] and model.get("backend") == "smolvlm2" and model.get("precision") == adapter["precision"] and model.get("status") == "completed":
                    cached_temporal = value
        except (OSError, UnicodeError, json.JSONDecodeError, UserFacingError, AttributeError, TypeError):
            cached_temporal = None
        if cached_temporal is not None:
            index = _apply_temporal_cache(index, cached_temporal)
            print("Using verified temporal perception cache; no Colab session allocated and no files uploaded.")
        else:
            if get_cloud_processing_setting(project, "colab") != "ask_each_run":
                raise UserFacingError("Set job.yaml cloud_processing.colab: ask_each_run before temporal proxy upload.")
            if not getattr(args, "allow_upload", False):
                raise UserFacingError("Add --allow-upload after reviewing the exact shortlisted 720p proxies.")
            cli_path = _find_colab_cli()
            if cli_path is None:
                raise UserFacingError("The Colab CLI is not installed.")
            temporal_gpu = requested_gpu
            index, remote, manifest = _run_remote_stage(
                project, index, cli_path=cli_path, stage="temporal", gpu=temporal_gpu,
                privacy_mode=policy.privacy_mode, temporal_backend=backend,
                selected_sources=selected_sources,
            )
            _save_temporal_cache(temporal_cache_path, temporal_key, remote)
    elif perception_complete:
        print("Using verified speech/visual perception cache; no Colab session allocated and no files uploaded.")
    elif not index.get("privacy", {}).get("eligible_uploads", {}).get("audio") and not index.get("privacy", {}).get("eligible_uploads", {}).get("representative_frames"):
        print("Perception index is local-only metadata; no Colab session allocated.")

    _write_index_outputs(project, index)
    print(f"Perception index saved: {project / 'work' / 'perception_index.json'}")
    return 0


__all__ = ["command_colab_perception"]
