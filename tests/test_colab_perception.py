from __future__ import annotations

import ast
import contextlib
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "skills" / "video-factory" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import colab_perception as colab
import colab_transcription
from perception import (
    COLAB_PACKAGE_VERSIONS,
    DIARIZATION_MODEL,
    DIARIZATION_MODEL_REVISION,
    IMPORTANCE_PARAMETERS,
    SPEECH_ENGINE_VERSION,
    SPEECH_MODEL,
    SPEECH_MODEL_REPO,
    SPEECH_MODEL_REVISION,
    TEMPORAL_MODEL,
    TEMPORAL_MODEL_REVISION,
    VISUAL_MODEL,
    VISUAL_MODEL_REVISION,
    WORKER_VERSION,
    _proxy_upload_candidates,
    assign_anonymous_speakers,
    build_perception_index,
)
from video_factory import UserFacingError, build_parser, sha256_file, write_json


class ColabPerceptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="vf-colab-perception-")
        self.project = Path(self.temp.name)
        for relative in (
            "assets/videos", "work/frames", "work/proxies", "outputs/work",
        ):
            (self.project / relative).mkdir(parents=True, exist_ok=True)
        self.source = "assets/videos/clip.mp4"
        self.source_path = self.project / self.source
        self.source_path.write_bytes(b"synthetic video source")
        self.frame_path = self.project / "work/frames/frame.jpg"
        self.frame_path.write_bytes(b"synthetic representative frame")
        self.proxy_path = self.project / "work/proxies/clip-720p.mp4"
        self.proxy_path.write_bytes(b"synthetic derived proxy")
        self.source_hash = sha256_file(self.source_path)
        self.proxy_hash = sha256_file(self.proxy_path)
        (self.project / "work/manifests").mkdir(parents=True, exist_ok=True)
        write_json(self.project / "work/manifests/media_manifest.json", {
            "assets": [{
                "source": self.source, "kind": "video", "sha256": self.source_hash,
            }],
        })
        write_json(self.project / "work/frames/frames.json", {
            "assets": {self.source: {"frames": [{
                "path": "work/frames/frame.jpg", "timestamp_seconds": 1.25,
            }]}}
        })
        write_json(self.project / "work/proxies/manifest.json", {
            "target_height": 720,
            "proxies": {self.source: {
                "source": self.source, "source_sha256": self.source_hash,
                "proxy_path": "work/proxies/clip-720p.mp4", "proxy_sha256": self.proxy_hash,
            }},
        })
        (self.project / "job.yaml").write_text(
            "profile: family\n"
            "privacy_mode: MAX_QUALITY\n"
            "temporal_backend: smolvlm2\n"
            "cloud_perception: true\n"
            "cloud_processing:\n  colab: ask_each_run\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def _vector(*, first: float = 0.0, second: float = 0.0) -> list[float]:
        value = [0.0] * 192
        value[0] = first
        value[1] = second
        return value

    def test_speaker_clustering_is_clip_local_conservative_and_uncalibrated(self) -> None:
        segments = [
            {"start": 0.0, "end": 2.0, "text": "one"},
            {"start": 3.0, "end": 5.0, "text": "two"},
            {"start": 6.0, "end": 8.0, "text": "ambiguous"},
            {"start": 9.0, "end": 9.8, "text": "short"},
        ]
        diagonal = [2**-0.5, 2**-0.5] + [0.0] * 190
        assigned = assign_anonymous_speakers(
            segments,
            [self._vector(first=1.0), self._vector(second=1.0), diagonal, self._vector(first=1.0)],
        )
        self.assertEqual([item["speaker_id"] for item in assigned], ["speaker_01", "speaker_02", "unknown", "unknown"])
        self.assertTrue(all(item["confidence"] is None for item in assigned))
        self.assertTrue(all(item["assignment_method"] == "speechbrain_ecapa_cosine_heuristic" for item in assigned))
        self.assertEqual(assign_anonymous_speakers(segments[:1], [self._vector(second=1.0)])[0]["speaker_id"], "speaker_01")

    def test_perception_index_has_no_automatic_temporal_shortlist(self) -> None:
        path = build_perception_index(
            self.project, privacy_mode="MAX_QUALITY", gpu_policy="T4",
            cloud=True, allow_upload=True, temporal_backend="smolvlm2",
        )
        index = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(index["temporal"]["status"], "awaiting_shortlist")
        self.assertEqual(index["temporal"]["candidates"], [])
        self.assertEqual(index["privacy"]["eligible_uploads"]["proxies"], [])
        self.assertEqual(index["visual"]["event_groups"], [])
        self.assertEqual(index["speech"]["audio_events_status"], "not_applicable")
        self.assertEqual(index["speech"]["importance_status"], "not_applicable")
        self.assertEqual(index["models"]["audio_events"]["status"], "not_applicable")
        self.assertEqual(index["models"]["speech_importance"]["method"], colab.IMPORTANCE_METHOD)

    def test_only_explicit_shortlist_proxies_enter_temporal_bundle(self) -> None:
        index_path = build_perception_index(
            self.project, privacy_mode="MAX_QUALITY", gpu_policy="T4",
            cloud=True, allow_upload=True, temporal_backend="smolvlm2",
        )
        index = json.loads(index_path.read_text(encoding="utf-8"))
        candidates = _proxy_upload_candidates(self.project, [self.source], "MAX_QUALITY")
        self.assertEqual([item["source"] for item in candidates], [self.source])
        bundle_path = self.project / "work/temporal-input.zip"
        manifest = colab._bundle(
            self.project, index, bundle_path, stage="temporal",
            privacy_mode="MAX_QUALITY", selected_sources=[self.source],
        )
        self.assertEqual(manifest["audio"], [])
        self.assertEqual(manifest["frames"], [])
        self.assertEqual([item["source"] for item in manifest["proxies"]], [self.source])
        with zipfile.ZipFile(bundle_path) as archive:
            self.assertEqual(sorted(archive.namelist()), ["bundle.json", manifest["proxies"][0]["path"]])
        with self.assertRaises(UserFacingError):
            _proxy_upload_candidates(self.project, ["assets/videos/not-shortlisted.mp4"], "MAX_QUALITY")

    def test_temporal_timeline_contract_and_event_groups_are_independent_of_visual_clusters(self) -> None:
        proxies = _proxy_upload_candidates(self.project, [self.source], "MAX_QUALITY")
        event = {
            "start_seconds": 0.5, "end_seconds": 2.0,
            "action": "A person walks toward the table.",
            "evidence": "The person moves from frame-left to the table across samples.",
            "confidence": None,
        }
        result = {
            "source": self.source, "proxy_sha256": self.proxy_hash,
            "duration_seconds": 5.0, "sampled_timestamps_seconds": [0.0, 2.5, 4.9],
            "summary": "A person approaches a table and sits.",
            "events": [event], "status": "completed", "confidence_calibrated": False,
        }
        temporal = {
            "backend": "smolvlm2", "model": TEMPORAL_MODEL,
            "revision": TEMPORAL_MODEL_REVISION, "status": "completed",
            "prompt_version": "timeline-json.v1",
            "candidates": [{"source": self.source, "sha256": self.proxy_hash, "status": "completed"}],
            "results": [result],
            "event_groups": [{
                "group_id": f"temporal-{self.proxy_hash[:16]}",
                "source": self.source, "summary": result["summary"], "events": [event],
            }],
        }
        colab._validate_temporal_payload(temporal, [self.source], proxies)
        temporal["event_groups"] = [{"group_id": "visual-0001", "items": ["frame-00001"]}]
        with self.assertRaises(UserFacingError):
            colab._validate_temporal_payload(temporal, [self.source], proxies)

    def test_owned_colab_session_must_disappear_from_server_list(self) -> None:
        name = "vf-perception-owned"
        with patch.object(colab, "_run_colab") as stop, patch.object(
            colab, "_owned_session_released", return_value=True,
        ):
            colab._release_owned_session("colab", name)
        stop.assert_called_once_with("colab", "stop", ["stop", "-s", name])
        with patch.object(colab, "_run_colab"), patch.object(
            colab, "_owned_session_released", return_value=False,
        ), patch("time.sleep"):
            with self.assertRaises(UserFacingError):
                colab._release_owned_session("colab", name)

    def test_colab_cli_subprocesses_disable_persistent_debug_logging(self) -> None:
        success = SimpleNamespace(returncode=0, stdout=b"[colab] Session terminated.\n", stderr=b"")
        with patch.object(colab_transcription.subprocess, "run", return_value=success) as run:
            colab_transcription._run_colab("/usr/bin/colab", "stop", ["stop", "-s", "owned"])
        self.assertEqual(run.call_args.args[0][:2], ["/usr/bin/colab", "--logtostderr"])

        usage = SimpleNamespace(
            returncode=0,
            stdout="Current balance: 200.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0\n",
            stderr="",
        )
        with patch.object(colab_transcription.subprocess, "run", return_value=usage) as run:
            snapshot = colab_transcription._colab_usage_snapshot("/usr/bin/colab")
        self.assertEqual(snapshot, {"balance": "200.00", "rate": "0.00", "assignments": "0"})
        self.assertEqual(run.call_args.args[0][:3], ["/usr/bin/colab", "--logtostderr", "usage"])

        sessions = SimpleNamespace(returncode=0, stdout="No active sessions found on server.", stderr="")
        with patch.object(colab.subprocess, "run", return_value=sessions) as run:
            self.assertTrue(colab._owned_session_released("/usr/bin/colab", "owned"))
        self.assertEqual(run.call_args.args[0][:3], ["/usr/bin/colab", "--logtostderr", "sessions"])

    def test_siglip_result_validation_rejects_bad_dimension_and_accepts_complete_stage(self) -> None:
        index_path = build_perception_index(
            self.project, privacy_mode="BALANCED", gpu_policy="T4", cloud=True,
            allow_upload=True, temporal_backend="none",
        )
        base = json.loads(index_path.read_text(encoding="utf-8"))
        manifest = {
            "stage": "perception", "privacy_mode": "BALANCED", "audio": [],
            "frames": [{
                "id": "frame-00001", "source": self.source,
                "timestamp_seconds": 1.25, "sha256": sha256_file(self.frame_path),
                "path": "frames/frame-00001.jpg",
            }],
            "proxies": [],
        }
        vector = [0.0] * 768
        vector[0] = 1.0
        remote = {
            "schema_version": "perception-index.v1", "worker_version": WORKER_VERSION,
            "stage": "perception", "status": "completed",
            "privacy": {"mode": "BALANCED", "original_media_uploaded": False},
            "compute": {"gpu_policy": "T4", "gpu_name": "Tesla T4", "premium_gpu_allowed": False},
            "runtime": {"gpu_name": "Tesla T4", "python": "3.12.0", "cuda_version": "12.8", "packages": colab._runtime_packages("perception")},
            "models": {"visual": {"name": VISUAL_MODEL, "revision": VISUAL_MODEL_REVISION, "dimension": 768, "status": "completed"}},
            "transfer_manifest": manifest,
            "speech": {
                "status": "not_applicable", "diarization_status": "not_applicable",
                "diarization_method": "speechbrain-ecapa-utterance-clustering.v1",
                "diarization_overlap_aware": False, "diarization_confidence_calibrated": False,
                "vad": "faster-whisper-vad_filter",
                "transcripts": [], "speech_turns": [], "anonymous_speakers": [],
                "audio_events": [], "audio_events_status": "not_applicable",
                "audio_events_method": colab.AUDIO_EVENT_METHOD,
                "audio_events_confidence_calibrated": False,
                "importance": [], "importance_status": "not_applicable",
                "importance_method": colab.IMPORTANCE_METHOD,
                "importance_confidence_calibrated": False,
            },
            "visual": {
                "status": "completed", "items": [{
                    "id": "frame-00001", "source": self.source,
                    "sha256": manifest["frames"][0]["sha256"],
                    "embedding": {"status": "completed", "model": VISUAL_MODEL,
                                  "revision": VISUAL_MODEL_REVISION, "dimension": 768,
                                  "values": vector},
                }],
                "similarity_clusters": {"frame-00001": "similarity-0001"},
                "event_groups": [], "event_groups_status": "not_configured",
            },
        }
        colab._validate_remote_result(remote, base, stage="perception", manifest=manifest, gpu="T4")
        remote["visual"]["items"][0]["embedding"]["dimension"] = 767
        with self.assertRaises(UserFacingError):
            colab._validate_remote_result(remote, base, stage="perception", manifest=manifest, gpu="T4")

    def test_generated_worker_has_pins_and_registered_temporal_adapter(self) -> None:
        configuration = {
            "worker_version": WORKER_VERSION, "stage": "temporal", "gpu": "T4",
            "privacy_mode": "MAX_QUALITY", "temporal_backend": "smolvlm2",
            "temporal_sources": [self.source], "speech_model": SPEECH_MODEL,
            "speech_engine_version": SPEECH_ENGINE_VERSION,
            "speech_model_repo": SPEECH_MODEL_REPO, "speech_model_revision": SPEECH_MODEL_REVISION,
            "diarization_model": DIARIZATION_MODEL,
            "diarization_model_revision": DIARIZATION_MODEL_REVISION,
            "visual_model": VISUAL_MODEL, "visual_model_revision": VISUAL_MODEL_REVISION,
            "visual_engine_version": "transformers-4.57.6", "visual_dimension": 768,
            "temporal_model": TEMPORAL_MODEL, "temporal_model_revision": TEMPORAL_MODEL_REVISION,
            "temporal_prompt_version": "timeline-json.v1", "temporal_max_frames": 24,
            "audio_event_model": colab.AUDIO_EVENT_MODEL,
            "audio_event_model_revision": colab.AUDIO_EVENT_MODEL_REVISION,
            "audio_event_method": colab.AUDIO_EVENT_METHOD,
            "audio_event_window_seconds": colab.AUDIO_EVENT_WINDOW_SECONDS,
            "audio_event_top_k": colab.AUDIO_EVENT_TOP_K,
            "importance_method": colab.IMPORTANCE_METHOD,
            "importance_parameters": IMPORTANCE_PARAMETERS,
            "bundle_path": "/content/in.zip", "work_root": "/content/work",
            "output_path": "/content/out.json", "timeout_seconds": 8700,
        }
        source = colab._worker_source(configuration)
        tree = ast.parse(source)
        inference_source = next(
            ast.literal_eval(node.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "inference_source" for target in node.targets)
        )
        ast.parse(inference_source)
        self.assertIn("TEMPORAL_ADAPTER_REGISTRY", inference_source)
        self.assertNotIn("plugin_required", inference_source)
        self.assertIn(VISUAL_MODEL_REVISION, source)
        self.assertIn(colab.AUDIO_EVENT_MODEL_REVISION, inference_source)
        self.assertIn("_write_failure", source)
        self.assertIn('"status": "failed"', source)
        self.assertIn("capture_output=True", source)

    def test_structured_remote_worker_diagnostics_are_safe_and_fail_closed(self) -> None:
        remote = {
            "schema_version": "perception-index.v1", "worker_version": WORKER_VERSION,
            "stage": "perception", "status": "failed",
            "diagnostic": {
                "phase": "package_install", "error_type": "RuntimeError",
                "message": "ImportError hf_abcdefgh12345678 https://example.com/private /content/user.wav",
            },
        }
        with self.assertRaisesRegex(UserFacingError, "package_install") as captured:
            colab._validate_remote_result(
                remote, {}, stage="perception", manifest={}, gpu="T4",
            )
        self.assertIn("[redacted-token]", str(captured.exception))
        self.assertNotIn("hf_abcdefgh12345678", str(captured.exception))
        self.assertNotIn("example.com", str(captured.exception))
        self.assertNotIn("user.wav", str(captured.exception))

    def test_generated_worker_failure_writer_emits_only_a_redacted_receipt(self) -> None:
        configuration = {
            "worker_version": WORKER_VERSION, "stage": "perception",
            "output_path": str(self.project / "remote-result.json"),
        }
        source = colab._worker_source(configuration)
        tree = ast.parse(source)
        helper_nodes = [
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            or isinstance(node, ast.FunctionDef) and node.name in {"_safe_diagnostic", "_write_failure"}
        ]
        namespace = {"CONFIG": configuration}
        exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), "generated-colab-helper", "exec"), namespace)
        namespace["_write_failure"](
            RuntimeError("hf_abcdefgh12345678 https://example.com/token /content/private.wav"),
            "perception_inference",
        )
        receipt = json.loads((self.project / "remote-result.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["diagnostic"]["phase"], "perception_inference")
        self.assertNotIn("hf_abcdefgh12345678", receipt["diagnostic"]["message"])
        self.assertNotIn("example.com", receipt["diagnostic"]["message"])
        self.assertNotIn("private.wav", receipt["diagnostic"]["message"])

    def test_generated_worker_captures_child_failure_without_leaking_logs(self) -> None:
        output = self.project / "simulated-remote-result.json"
        configuration = {
            "worker_version": WORKER_VERSION, "stage": "perception", "gpu": "T4",
            "packages": {"transformers": "4.57.6"}, "output_path": str(output),
        }
        source = colab._worker_source(configuration)
        def fake_run(command, **kwargs):
            if command[0] == "nvidia-smi":
                return SimpleNamespace(returncode=0, stdout="Tesla T4\n", stderr="")
            if "pip" in command:
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(
                returncode=1, stdout="", stderr="RuntimeError hf_abcdefgh12345678 https://example.com/key /content/private.wav",
            )
        with patch("subprocess.run", side_effect=fake_run), patch(
            "importlib.util.find_spec",
            return_value=SimpleNamespace(submodule_search_locations=["/fake/site-packages"]),
        ):
            exec(compile(source, "generated-colab-worker", "exec"), {})
        receipt = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["diagnostic"]["phase"], "perception_inference")
        self.assertNotIn("hf_abcdefgh12345678", receipt["diagnostic"]["message"])
        self.assertNotIn("example.com", receipt["diagnostic"]["message"])
        self.assertNotIn("private.wav", receipt["diagnostic"]["message"])

    def test_remote_exec_failure_downloads_only_nonce_result_and_never_accepts_success(self) -> None:
        index_path = build_perception_index(
            self.project, privacy_mode="BALANCED", gpu_policy="T4",
            cloud=True, allow_upload=True, temporal_backend="none",
        )
        base = json.loads(index_path.read_text(encoding="utf-8"))
        failure_receipt = {
            "schema_version": "perception-index.v1", "worker_version": WORKER_VERSION,
            "stage": "perception", "status": "failed",
            "diagnostic": {
                "phase": "perception_inference", "error_type": "RuntimeError",
                "message": "model import failed hf_abcdefgh12345678 /content/private.wav",
            },
        }

        def run_with_receipt(receipt: dict[str, object]) -> tuple[UserFacingError, list[str]]:
            commands: list[str] = []

            def fake_colab(_cli_path: str, stage: str, arguments: list[str]) -> None:
                commands.append(stage)
                if stage == "exec":
                    raise UserFacingError("Colab CLI exec command failed; command output was suppressed.")
                if stage == "download":
                    Path(arguments[-1]).write_text(json.dumps(receipt), encoding="utf-8")

            with patch.object(colab, "_colab_usage_snapshot", return_value={"balance": "200.00", "rate": "0.00", "assignments": "0"}), patch.object(
                colab, "_print_usage_snapshot",
            ), patch.object(colab, "_run_colab", side_effect=fake_colab), patch.object(
                colab, "_owned_session_released", return_value=True,
            ):
                try:
                    colab._run_remote_stage(
                        self.project, base, cli_path="colab", stage="perception", gpu="T4",
                        privacy_mode="BALANCED", temporal_backend="none",
                    )
                except UserFacingError as exc:
                    return exc, commands
            self.fail("remote result was accepted after the Colab CLI reported an exec failure")

        error, commands = run_with_receipt(failure_receipt)
        self.assertIn("perception_inference", str(error))
        self.assertIn("[redacted-token]", str(error))
        self.assertNotIn("private.wav", str(error))
        self.assertEqual(commands.count("download"), 1)
        self.assertEqual(commands.count("stop"), 1)

        completed_receipt = dict(failure_receipt, status="completed")
        error, commands = run_with_receipt(completed_receipt)
        self.assertIn("did not provide a verified failure receipt", str(error))
        self.assertEqual(commands.count("download"), 1)
        self.assertEqual(commands.count("stop"), 1)

    def test_audio_event_and_importance_payloads_require_pinned_uncalibrated_metadata(self) -> None:
        base = {"privacy": {"mode": "BALANCED"}}
        digest = "a" * 64
        source = self.source
        manifest = {
            "stage": "perception", "privacy_mode": "BALANCED",
            "audio": [{"source": source, "sha256": digest, "path": "audio/test.wav"}],
            "frames": [], "proxies": [],
        }
        segment = {
            "start": 0.0, "end": 1.0, "text": "hello there",
            "words": [{"start": 0.0, "end": 0.4, "word": "hello"},
                      {"start": 0.5, "end": 0.9, "word": "there"}],
        }
        importance = colab._speech_importance_record(source, segment, -24.0)
        remote = {
            "schema_version": "perception-index.v1", "worker_version": WORKER_VERSION,
            "stage": "perception", "status": "completed",
            "privacy": {"mode": "BALANCED", "original_media_uploaded": False},
            "compute": {"gpu_policy": "T4", "gpu_name": "Tesla T4", "premium_gpu_allowed": False},
            "runtime": {
                "gpu_name": "Tesla T4", "python": "3.12.0", "cuda_version": "12.8",
                "packages": colab._runtime_packages("perception"),
            },
            "models": {
                "speech": {"name": SPEECH_MODEL, "repo": SPEECH_MODEL_REPO,
                           "revision": SPEECH_MODEL_REVISION, "engine": SPEECH_ENGINE_VERSION,
                           "status": "completed"},
                "diarization": {"name": DIARIZATION_MODEL, "revision": DIARIZATION_MODEL_REVISION,
                                "status": "completed", "method": "speechbrain-ecapa-utterance-clustering.v1",
                                "overlap_aware": False, "confidence_calibrated": False},
                "audio_events": {"name": colab.AUDIO_EVENT_MODEL,
                                 "revision": colab.AUDIO_EVENT_MODEL_REVISION,
                                 "engine": f"transformers-{COLAB_PACKAGE_VERSIONS['transformers']}",
                                 "status": "completed", "method": colab.AUDIO_EVENT_METHOD,
                                 "sampling_rate": 16000,
                                 "window_seconds": colab.AUDIO_EVENT_WINDOW_SECONDS,
                                 "top_k": colab.AUDIO_EVENT_TOP_K, "score_calibrated": False},
                "speech_importance": {
                    "method": colab.IMPORTANCE_METHOD, "parameters": IMPORTANCE_PARAMETERS,
                    "status": "completed", "confidence_calibrated": False,
                    "interpretation": "uncalibrated_candidate_signal_not_semantic_importance",
                },
            },
            "transfer_manifest": manifest,
            "speech": {
                "vad": "faster-whisper-vad_filter", "status": "completed",
                "diarization_status": "completed",
                "diarization_method": "speechbrain-ecapa-utterance-clustering.v1",
                "diarization_overlap_aware": False, "diarization_confidence_calibrated": False,
                "transcripts": [{"source": source, "source_sha256": digest,
                                 "duration_seconds": 2.0, "model": SPEECH_MODEL,
                                 "model_repo": SPEECH_MODEL_REPO,
                                 "model_revision": SPEECH_MODEL_REVISION,
                                 "segments": [segment]}],
                "speech_turns": [{"source": source, "start": 0.0, "end": 1.0,
                                  "text": "hello there", "speaker_id": "unknown",
                                  "confidence": None, "similarity_to_cluster": None}],
                "anonymous_speakers": [],
                "audio_events": [{"source": source, "start_seconds": 0.0,
                                  "end_seconds": 2.0, "label": "Speech", "score": 0.5,
                                  "method": colab.AUDIO_EVENT_METHOD, "confidence": None}],
                "audio_events_status": "completed",
                "audio_events_method": colab.AUDIO_EVENT_METHOD,
                "audio_events_confidence_calibrated": False,
                "importance": [importance], "importance_status": "completed",
                "importance_method": colab.IMPORTANCE_METHOD,
                "importance_confidence_calibrated": False,
            },
            "visual": {"items": [], "status": "not_applicable", "similarity_clusters": {},
                       "event_groups": [], "event_groups_status": "not_configured"},
        }
        colab._validate_remote_result(remote, base, stage="perception", manifest=manifest, gpu="T4")
        remote["speech"]["audio_events"][0]["confidence"] = 0.99
        with self.assertRaises(UserFacingError):
            colab._validate_remote_result(remote, base, stage="perception", manifest=manifest, gpu="T4")

    def test_speech_importance_is_a_reproducible_uncalibrated_signal(self) -> None:
        segment = {
            "start": 2.0, "end": 7.0, "text": "A useful phrase here.",
            "words": [{"word": word} for word in ("A", "useful", "phrase", "here.")],
        }
        result = colab._speech_importance_record("assets/videos/clip.mp4", segment, -25.0)
        self.assertEqual(result["method"], colab.IMPORTANCE_METHOD)
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["signals"]["word_count"], 4)
        self.assertEqual(result["signals"]["duration_seconds"], 5.0)
        self.assertGreater(result["score"], 0.0)
        self.assertLessEqual(result["score"], 1.0)

    def test_cli_exposes_repeatable_explicit_temporal_sources(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "colab-perception", str(self.project), "--temporal-backend", "smolvlm2",
            "--temporal-source", self.source, "--temporal-source", "assets/videos/clip-2.mp4",
        ])
        self.assertEqual(args.temporal_backend, "smolvlm2")
        self.assertEqual(args.temporal_sources, [self.source, "assets/videos/clip-2.mp4"])
        help_text = parser._subparsers._group_actions[0].choices["colab-perception"].format_help()
        self.assertIn("explicit source shortlist", help_text)
        self.assertIn("repeat for each source", help_text)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["colab-perception", str(self.project), "--temporal-backend", "custom"])

    def test_complete_cache_hit_skips_colab_allocation_and_upload_consent(self) -> None:
        (self.project / "job.yaml").write_text(
            "profile: family\nprivacy_mode: BALANCED\ncloud_perception: true\n"
            "cloud_processing:\n  colab: ask_each_run\n",
            encoding="utf-8",
        )
        index_path = build_perception_index(
            self.project, privacy_mode="BALANCED", gpu_policy="T4",
            cloud=True, allow_upload=False, temporal_backend="none",
        )
        index = json.loads(index_path.read_text(encoding="utf-8"))
        frame_id = index["visual"]["items"][0]["id"]
        reference = f"work/perception-cache/{index['cache_key']}/embeddings/{frame_id}.json"
        embedding_path = self.project / reference
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        values = [0.0] * 768
        values[0] = 1.0
        write_json(embedding_path, {
            "model": VISUAL_MODEL, "revision": VISUAL_MODEL_REVISION,
            "dimension": 768, "values": values,
        })
        index["worker_version"] = WORKER_VERSION
        index["privacy"]["cloud_status"] = "completed"
        index["models"]["visual"] = {
            "name": VISUAL_MODEL, "revision": VISUAL_MODEL_REVISION,
            "dimension": 768, "status": "completed",
        }
        index["visual"]["status"] = "completed"
        index["visual"]["similarity_clusters"] = {frame_id: "similarity-0001"}
        index["visual"]["items"][0]["embedding"] = {
            "status": "completed", "model": VISUAL_MODEL,
            "revision": VISUAL_MODEL_REVISION, "dimension": 768,
            "reference": reference,
        }
        cache_path = self.project / "work/perception-cache" / f"{index['cache_key']}.json"
        write_json(cache_path, index)
        args = SimpleNamespace(
            project=self.project, privacy_mode=None, gpu="T4", temporal_backend=None,
            temporal_sources=[], allow_upload=False,
        )
        with patch.object(colab.shutil, "which", side_effect=AssertionError("must not resolve Colab CLI")), patch.object(
            colab, "_run_colab", side_effect=AssertionError("must not start a Colab session"),
        ):
            self.assertEqual(colab.command_colab_perception(args), 0)


if __name__ == "__main__":
    unittest.main()
