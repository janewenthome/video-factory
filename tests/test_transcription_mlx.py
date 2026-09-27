from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "skills" / "video-factory" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import transcription_mlx as mlx  # noqa: E402
import video_factory  # noqa: E402


class MLXTranscriptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="video-factory-mlx-transcription-")
        self.root = Path(self.temp_dir.name)
        self.project = self.root / "project"
        self.audio = self.project / "work" / "transcripts" / "audio" / "cut.wav"
        self.audio.parent.mkdir(parents=True)
        self.audio.write_bytes(b"local synthetic audio")
        model_cache_name = f"models--{mlx.DEFAULT_MODEL.replace('/', '--')}"
        self.snapshot = (
            self.root / "hf-cache" / model_cache_name
            / "snapshots" / mlx.DEFAULT_MODEL_REVISION
        )
        self.snapshot.mkdir(parents=True)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def response(text: str = "你好，大家好。") -> dict[str, object]:
        return {
            "text": text,
            "language": "zh",
            "segments": [
                {
                    "start": 0.1,
                    "end": 1.2,
                    "text": text,
                    "avg_logprob": -0.2,
                    "no_speech_prob": 0.03,
                    "words": [
                        {"word": "你好，", "start": 0.1, "end": 0.6, "probability": 0.98},
                        {"word": "大家好。", "start": 0.6, "end": 1.2, "probability": 0.91},
                    ],
                }
            ],
        }

    def runtime(self, *, mlx_version: str = "0.29.0") -> mlx._Runtime:
        def transcribe(path: str, **options: object) -> dict[str, object]:
            self.calls.append((path, options))
            return self.response()

        def snapshot_download(**options: object) -> str:
            if options.get("local_files_only") is True:
                return str(self.snapshot)
            self.fail("the local model snapshot should be preferred")

        return mlx._Runtime(
            transcribe=transcribe,
            snapshot_download=snapshot_download,
            mlx_whisper_version="0.4.2",
            mlx_version=mlx_version,
        )

    def test_transcribe_audio_uses_local_word_timestamps_and_reuses_cache(self) -> None:
        with patch.object(mlx, "_load_runtime", return_value=self.runtime()) as runtime_loader:
            first = mlx.transcribe_audio(self.project, self.audio)
            second = mlx.transcribe_audio(self.project, self.audio)

        self.assertEqual(first, second)
        self.assertEqual(len(self.calls), 1)
        path, options = self.calls[0]
        self.assertEqual(Path(path).resolve(), self.audio.resolve())
        self.assertEqual(options["path_or_hf_repo"], str(self.snapshot))
        self.assertTrue(options["word_timestamps"])
        self.assertNotIn("language", options)
        self.assertEqual(len(first["segments"][0]["words"]), 2)
        self.assertEqual(len(list((self.project / "work" / "transcripts").glob("mlx-*.json"))), 1)
        srt = next((self.project / "work" / "transcripts").glob("mlx-*.srt"))
        self.assertIn("你好，大家好。", srt.read_text(encoding="utf-8"))
        runtime_loader.assert_called()

    def test_cache_key_includes_source_hash_language_model_revision_and_runtime_versions(self) -> None:
        with patch.object(mlx, "_load_runtime", side_effect=[
            self.runtime(), self.runtime(), self.runtime(mlx_version="0.30.0"), self.runtime(),
        ]):
            mlx.transcribe_audio(self.project, self.audio, local_files_only=False)
            mlx.transcribe_audio(self.project, self.audio, language="zh")
            mlx.transcribe_audio(self.project, self.audio, language="zh")
            self.audio.write_bytes(b"different local audio")
            mlx.transcribe_audio(self.project, self.audio, language="zh")

        caches = list((self.project / "work" / "transcripts").glob("mlx-*.json"))
        self.assertEqual(len(caches), 4)
        requests = [json.loads(path.read_text(encoding="utf-8"))["request"] for path in caches]
        self.assertEqual({item["mlx_version"] for item in requests}, {"0.29.0", "0.30.0"})
        self.assertEqual({item["model_revision"] for item in requests}, {mlx.DEFAULT_MODEL_REVISION})
        self.assertEqual({item["language"] for item in requests}, {None, "zh"})
        self.assertEqual(len({item["source_sha256"] for item in requests}), 2)

    def test_changed_model_revision_selects_a_distinct_cache(self) -> None:
        second_revision = "b" * 40
        second_snapshot = self.root / "hf-cache" / "other" / "snapshots" / second_revision
        second_snapshot.mkdir(parents=True)

        def snapshot_download(**options: object) -> str:
            if options["revision"] == second_revision:
                return str(second_snapshot)
            return str(self.snapshot)

        runtime = self.runtime()
        runtime = mlx._Runtime(
            transcribe=runtime.transcribe,
            snapshot_download=snapshot_download,
            mlx_whisper_version=runtime.mlx_whisper_version,
            mlx_version=runtime.mlx_version,
        )
        with patch.object(mlx, "_load_runtime", return_value=runtime):
            mlx.transcribe_audio(self.project, self.audio, model_revision=mlx.DEFAULT_MODEL_REVISION)
            mlx.transcribe_audio(self.project, self.audio, model_revision=second_revision)
        self.assertEqual(len(list((self.project / "work" / "transcripts").glob("mlx-*.json"))), 2)

    def test_snapshot_download_falls_back_to_hub_and_caches_exact_resolved_commit(self) -> None:
        downloaded_snapshot = self.root / "hf-cache" / "snapshots" / mlx.DEFAULT_MODEL_REVISION
        downloaded_snapshot.mkdir(parents=True)
        options_seen: list[dict[str, object]] = []

        def snapshot_download(**options: object) -> str:
            options_seen.append(options)
            if options.get("local_files_only"):
                raise OSError("cache miss")
            return str(downloaded_snapshot)

        runtime = self.runtime()
        runtime = mlx._Runtime(
            transcribe=runtime.transcribe,
            snapshot_download=snapshot_download,
            mlx_whisper_version=runtime.mlx_whisper_version,
            mlx_version=runtime.mlx_version,
        )
        with patch.object(mlx, "_load_runtime", return_value=runtime):
            mlx.transcribe_audio(self.project, self.audio, local_files_only=False)

        self.assertEqual(len(options_seen), 2)
        self.assertTrue(options_seen[0]["local_files_only"])
        self.assertNotIn("local_files_only", options_seen[1])

    def test_local_only_is_default_and_fails_without_attempting_a_download(self) -> None:
        options_seen: list[dict[str, object]] = []

        def snapshot_download(**options: object) -> str:
            options_seen.append(options)
            raise OSError("snapshot is not cached")

        runtime = self.runtime()
        runtime = mlx._Runtime(
            transcribe=runtime.transcribe,
            snapshot_download=snapshot_download,
            mlx_whisper_version=runtime.mlx_whisper_version,
            mlx_version=runtime.mlx_version,
        )
        with patch.object(mlx, "_load_runtime", return_value=runtime):
            with self.assertRaisesRegex(video_factory.UserFacingError, "no model was downloaded"):
                mlx.transcribe_audio(self.project, self.audio)

        self.assertEqual(len(options_seen), 1)
        self.assertTrue(options_seen[0]["local_files_only"])
        self.assertEqual(self.calls, [])

    def test_default_model_is_the_reused_large_v3_snapshot(self) -> None:
        self.assertEqual(mlx.DEFAULT_MODEL, "mlx-community/whisper-large-v3-mlx")
        self.assertEqual(mlx.DEFAULT_MODEL_REVISION, "49e6aa286ad60c14352c404340ded53710378a11")

    def test_source_outside_project_and_source_mutation_are_rejected(self) -> None:
        outside = self.root / "outside.wav"
        outside.write_bytes(b"outside")
        with self.assertRaises(video_factory.UserFacingError):
            mlx.transcribe_audio(self.project, outside)

        runtime = self.runtime()

        def mutate_then_transcribe(path: str, **options: object) -> dict[str, object]:
            Path(path).write_bytes(b"changed during inference")
            return self.response()

        runtime = mlx._Runtime(
            transcribe=mutate_then_transcribe,
            snapshot_download=runtime.snapshot_download,
            mlx_whisper_version=runtime.mlx_whisper_version,
            mlx_version=runtime.mlx_version,
        )
        with patch.object(mlx, "_load_runtime", return_value=runtime):
            with self.assertRaisesRegex(video_factory.UserFacingError, "changed during local transcription"):
                mlx.transcribe_audio(self.project, self.audio)
        self.assertEqual(list((self.project / "work" / "transcripts").glob("mlx-*.json")), [])

    def test_normalizer_preserves_words_and_rejects_invalid_timestamps(self) -> None:
        normalized = mlx.normalize_mlx_response(self.response())
        self.assertEqual(normalized["segments"][0]["words"][0]["probability"], 0.98)
        malformed = self.response()
        malformed["segments"][0]["words"][0]["start"] = float("nan")
        with self.assertRaisesRegex(video_factory.UserFacingError, "invalid word timestamps"):
            mlx.normalize_mlx_response(malformed)

    def test_model_language_and_revision_are_validated_before_runtime_load(self) -> None:
        with patch.object(mlx, "_load_runtime", side_effect=AssertionError("runtime must not load")):
            for options in (
                {"model": "../outside/model"},
                {"language": "invalid language"},
                {"model_revision": "bad/revision"},
            ):
                with self.assertRaises(video_factory.UserFacingError):
                    mlx.transcribe_audio(self.project, self.audio, **options)


if __name__ == "__main__":
    unittest.main()
