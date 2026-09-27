from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "skills" / "video-factory" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import transcription_openai  # noqa: E402
import video_factory  # noqa: E402


class FakeResponse:
    def __init__(self, value: object) -> None:
        self.payload = json.dumps(value, ensure_ascii=False).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


class OpenAITranscriptionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="video-factory-transcription-")
        self.root = Path(self.temp_dir.name)
        self.project = self.root / "project"
        self.project.mkdir()
        self.audio = self.project / "assets" / "audio" / "sample.wav"
        self.audio.parent.mkdir(parents=True)
        self.audio.write_bytes(b"synthetic local audio bytes")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def response(self, text: str = "合成逐字稿") -> dict:
        return {
            "text": text,
            "segments": [
                {"start": 65.234, "end": 67.5, "text": "第二句"},
                {"start": 2.0, "end": 3.2, "text": "第一行\r\n第二行"},
            ],
        }

    def run_cli(self, *args: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = video_factory.main(["transcribe", str(self.project), *args])
        return result, stdout.getvalue(), stderr.getvalue()

    def options(self, *extra: str) -> tuple[str, ...]:
        return ("--source", "assets/audio/sample.wav", *extra)

    def test_cli_parser_exposes_explicit_upload_switch_and_model_options(self) -> None:
        args = video_factory.build_parser().parse_args(
            ["transcribe", str(self.project), "--source", "assets/audio/sample.wav", "--language", "zh", "--allow-upload"]
        )

        self.assertEqual(args.command, "transcribe")
        self.assertEqual(args.model, "whisper-1")
        self.assertEqual(args.language, "zh")
        self.assertTrue(args.allow_upload)

    def test_cache_miss_without_explicit_consent_never_calls_network(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch.object(
            transcription_openai.urllib.request, "urlopen", side_effect=AssertionError("unexpected network call")
        ) as request:
            status, _, stderr = self.run_cli(*self.options())

        self.assertEqual(status, 2)
        self.assertIn("--allow-upload", stderr)
        request.assert_not_called()

    def test_upload_consent_without_api_key_never_calls_network(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch.object(
            transcription_openai.urllib.request, "urlopen", side_effect=AssertionError("unexpected network call")
        ) as request:
            status, _, stderr = self.run_cli(*self.options("--allow-upload"))

        self.assertEqual(status, 2)
        self.assertIn("OPENAI_API_KEY is required", stderr)
        request.assert_not_called()

    def test_mocked_upload_requests_segment_timestamps_and_writes_srt(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-secret"}), patch.object(
            transcription_openai.urllib.request, "urlopen", return_value=FakeResponse(self.response())
        ) as request:
            status, stdout, stderr = self.run_cli(*self.options("--allow-upload", "--language", "zh", "--prompt", "中文專有名詞"))

        self.assertEqual(status, 0, stderr)
        self.assertIn("Saved timestamp transcript", stdout)
        self.assertEqual(request.call_count, 1)
        http_request = request.call_args.args[0]
        self.assertEqual(http_request.full_url, transcription_openai.API_URL)
        self.assertEqual(http_request.get_method(), "POST")
        self.assertEqual(http_request.get_header("Authorization"), "Bearer test-only-secret")
        body = http_request.data
        self.assertIn(b'name="response_format"\r\n\r\nverbose_json', body)
        self.assertIn(b'name="timestamp_granularities[]"\r\n\r\nsegment', body)
        self.assertIn(b'name="language"\r\n\r\nzh', body)
        self.assertIn("中文專有名詞".encode(), body)
        self.assertIn(self.audio.read_bytes(), body)

        outputs = list((self.project / "work" / "transcripts").glob("*.srt"))
        self.assertEqual(len(outputs), 1)
        self.assertEqual(
            outputs[0].read_text(encoding="utf-8"),
            "1\n00:00:02,000 --> 00:00:03,200\n第一行\n第二行\n\n"
            "2\n00:01:05,234 --> 00:01:07,500\n第二句\n",
        )
        cache = json.loads(outputs[0].with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(cache["request"]["model"], "whisper-1")
        self.assertEqual(cache["request"]["language"], "zh")
        self.assertNotIn("中文專有名詞", outputs[0].with_suffix(".json").read_text(encoding="utf-8"))

    def test_valid_cache_hit_needs_neither_consent_nor_key(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-secret"}), patch.object(
            transcription_openai.urllib.request, "urlopen", return_value=FakeResponse(self.response())
        ) as request:
            first_status, _, _ = self.run_cli(*self.options("--allow-upload"))
        self.assertEqual(first_status, 0)
        self.assertEqual(request.call_count, 1)

        with patch.dict(os.environ, {}, clear=True), patch.object(
            transcription_openai.urllib.request, "urlopen", side_effect=AssertionError("cache should avoid network")
        ) as request:
            second_status, stdout, stderr = self.run_cli(*self.options())

        self.assertEqual(second_status, 0, stderr)
        self.assertIn("Used cached timestamp transcript", stdout)
        request.assert_not_called()

    def test_changed_source_hash_creates_a_distinct_cache_entry(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-secret"}), patch.object(
            transcription_openai.urllib.request,
            "urlopen",
            side_effect=[FakeResponse(self.response("第一版")), FakeResponse(self.response("第二版"))],
        ) as request:
            first_status, _, first_stderr = self.run_cli(*self.options("--allow-upload"))
            first_json = list((self.project / "work" / "transcripts").glob("*.json"))[0]
            self.audio.write_bytes(b"updated synthetic local audio bytes")
            second_status, _, second_stderr = self.run_cli(*self.options("--allow-upload"))

        self.assertEqual(first_status, 0, first_stderr)
        self.assertEqual(second_status, 0, second_stderr)
        self.assertEqual(request.call_count, 2)
        json_outputs = list((self.project / "work" / "transcripts").glob("*.json"))
        self.assertEqual(len(json_outputs), 2)
        self.assertNotEqual(json_outputs[0].name, json_outputs[1].name)
        self.assertTrue(first_json.is_file())

    def test_model_language_and_prompt_hashes_select_distinct_cache_entries(self) -> None:
        variants = [(), ("--language", "zh"), ("--prompt", "名字")]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-secret"}), patch.object(
            transcription_openai.urllib.request,
            "urlopen",
            side_effect=[FakeResponse(self.response()) for _ in variants],
        ) as request:
            results = [self.run_cli(*self.options("--allow-upload", *variant))[0] for variant in variants]

        self.assertEqual(results, [0, 0, 0])
        self.assertEqual(request.call_count, len(variants))
        self.assertEqual(len(list((self.project / "work" / "transcripts").glob("*.json"))), len(variants))

    def test_symlink_escape_is_rejected_before_network_access(self) -> None:
        outside = self.root / "outside.wav"
        outside.write_bytes(b"outside audio")
        linked = self.project / "assets" / "audio" / "linked.wav"
        linked.symlink_to(outside)

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-secret"}), patch.object(
            transcription_openai.urllib.request, "urlopen", side_effect=AssertionError("unexpected network call")
        ) as request:
            status, _, stderr = self.run_cli("--source", "assets/audio/linked.wav", "--allow-upload")

        self.assertEqual(status, 2)
        self.assertIn("symlink escapes", stderr)
        request.assert_not_called()

    def test_transcription_upload_size_limit_is_checked_before_network_access(self) -> None:
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-only-secret"}), patch.object(
            transcription_openai, "MAX_UPLOAD_BYTES", 4
        ), patch.object(
            transcription_openai.urllib.request, "urlopen", side_effect=AssertionError("unexpected network call")
        ) as request:
            status, _, stderr = self.run_cli(*self.options("--allow-upload"))

        self.assertEqual(status, 2)
        self.assertIn("25 MB", stderr)
        request.assert_not_called()

    def test_invalid_response_is_rejected_without_writing_cache_or_leaking_key(self) -> None:
        invalid = {"text": "bad", "segments": [{"start": 4, "end": 2, "text": "invalid"}]}
        secret = "sensitive-test-key"
        with patch.dict(os.environ, {"OPENAI_API_KEY": secret}), patch.object(
            transcription_openai.urllib.request, "urlopen", return_value=FakeResponse(invalid)
        ):
            status, _, stderr = self.run_cli(*self.options("--allow-upload"))

        self.assertEqual(status, 2)
        self.assertIn("invalid timestamps", stderr)
        self.assertNotIn(secret, stderr)
        self.assertEqual(list((self.project / "work" / "transcripts").glob("*.json")), [])

    def test_non_timestamp_model_rejected_before_upload(self) -> None:
        with patch.object(transcription_openai.urllib.request, "urlopen") as request:
            status, _, stderr = self.run_cli(*self.options("--allow-upload", "--model", "gpt-transcribe"))
        self.assertEqual(status, 2)
        self.assertIn("whisper-1", stderr)
        request.assert_not_called()

    def test_disabled_policy_blocks_upload_even_with_consent(self) -> None:
        (self.project / "job.yaml").write_text("cloud_processing:\n  openai: disabled\n")
        with patch.object(transcription_openai.urllib.request, "urlopen") as request:
            status, _, stderr = self.run_cli(*self.options("--allow-upload"))
        self.assertEqual(status, 2)
        self.assertIn("disables", stderr)
        request.assert_not_called()

    def test_non_utf8_cache_is_treated_as_miss(self) -> None:
        path = self.project / "invalid.json"
        path.write_bytes(b"\xff")
        self.assertIsNone(transcription_openai.load_valid_cache(path, {"cache_key": "x", "request": {}}))

    def test_http_error_details_do_not_leak_credentials(self) -> None:
        secret = "sensitive-test-key"
        failures = [
            urllib.error.HTTPError(
                transcription_openai.API_URL,
                401,
                "Unauthorized",
                hdrs=None,
                fp=io.BytesIO(secret.encode("utf-8")),
            ),
            RuntimeError(secret),
        ]
        for failure in failures:
            with self.subTest(error=type(failure).__name__), patch.dict(
                os.environ, {"OPENAI_API_KEY": secret}
            ), patch.object(transcription_openai.urllib.request, "urlopen", side_effect=failure):
                status, _, stderr = self.run_cli(*self.options("--allow-upload"))

                self.assertEqual(status, 2)
                self.assertNotIn(secret, stderr)
                if isinstance(failure, urllib.error.HTTPError):
                    self.assertIn("HTTP 401", stderr)


if __name__ == "__main__":
    unittest.main()
