from __future__ import annotations

import ast
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "skills" / "video-factory" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))
import colab_transcription as colab
import video_factory


class ColabTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="vf-colab-test-")
        self.project = Path(self.directory.name)
        (self.project / "assets").mkdir()
        (self.project / "assets/sample.wav").write_bytes(b"synthetic audio")
        (self.project / "job.yaml").write_text("cloud_processing:\n  colab: ask_each_run\n")
        self.calls = []

    def tearDown(self):
        self.directory.cleanup()

    def run_cli(self, *extra):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = video_factory.main([
                "colab-transcribe", str(self.project), "--source", "assets/sample.wav", *extra,
            ])
        return status, out.getvalue(), err.getvalue()

    def fake_command(self, executable, stage, arguments):
        self.calls.append((stage, arguments))
        if stage == "exec":
            worker = Path(arguments[arguments.index("-f") + 1]).read_text()
            ast.parse(worker)
            self.assertIn("LD_LIBRARY_PATH", worker)
            self.assertIn("--timeout", arguments)
            self.assertGreater(int(arguments[-1]), 1800)
        if stage == "download":
            Path(arguments[-1]).write_text(json.dumps({
                "text": "hello", "segments": [{"start": 0, "end": 1, "text": "hello"}],
            }))

    def approved_run(self, *extra):
        with patch.object(colab.shutil, "which", return_value="/mock/colab"), patch.object(
            colab, "_colab_usage_snapshot", return_value={"balance": "100", "rate": "0", "assignments": "0"}
        ), patch.object(colab, "_run_colab", side_effect=self.fake_command):
            return self.run_cli("--allow-upload", *extra)

    def test_success_uses_one_owned_session_and_caches(self):
        status, _, err = self.approved_run()
        self.assertEqual(status, 0, err)
        self.assertEqual([stage for stage, _ in self.calls], ["new", "upload", "exec", "download", "stop"])
        names = [args[args.index("-s") + 1] for _, args in self.calls]
        self.assertEqual(len(set(names)), 1)
        self.assertTrue(names[0].startswith("vf-transcribe-"))
        self.assertEqual(self.calls[0][1][-1], "T4")
        self.assertEqual(len(list((self.project / "work/transcripts").glob("*.srt"))), 1)
        self.assertEqual(list((self.project / "work/colab-staging").iterdir()), [])
        with patch.object(colab.subprocess, "run", side_effect=AssertionError("network on cache hit")):
            (self.project / "job.yaml").write_text("cloud_processing:\n  colab: disabled\n")
            status, out, err = self.run_cli()
        self.assertEqual(status, 0, err)
        self.assertIn("cached", out)

    def test_dry_run_and_missing_consent_never_call_cli(self):
        with patch.object(colab.subprocess, "run", side_effect=AssertionError("unexpected CLI")):
            self.assertEqual(self.run_cli("--dry-run")[0], 0)
            self.assertIn("--allow-upload", self.run_cli()[2])

    def test_missing_or_disabled_policy_blocks_upload(self):
        for policy in ("", "cloud_processing:\n  colab: disabled\n"):
            (self.project / "job.yaml").write_text(policy)
            with patch.object(colab.subprocess, "run", side_effect=AssertionError("unexpected CLI")):
                self.assertEqual(self.run_cli("--allow-upload")[0], 2)

    def test_unknown_usage_blocks_allocation(self):
        with patch.object(colab.shutil, "which", return_value="colab"), patch.object(
            colab, "_colab_usage_snapshot", return_value=None
        ), patch.object(colab, "_run_colab") as run:
            self.assertEqual(self.run_cli("--allow-upload")[0], 2)
            run.assert_not_called()

    def test_execution_failure_still_stops_only_owned_session(self):
        normal = self.fake_command
        def failure(executable, stage, arguments):
            if stage == "exec":
                self.calls.append((stage, arguments))
                raise video_factory.UserFacingError("worker failure")
            normal(executable, stage, arguments)
        self.fake_command = failure
        status, _, err = self.approved_run()
        self.assertEqual(status, 2)
        self.assertIn("worker failure", err)
        self.assertEqual(self.calls[-1][0], "stop")
        self.assertEqual(self.calls[0][1][2], self.calls[-1][1][2])
        self.assertEqual(list((self.project / "work/transcripts").glob("*.json")), [])

    def test_cleanup_failure_preserves_transcript_but_is_not_success(self):
        normal = self.fake_command
        def failure(executable, stage, arguments):
            if stage == "stop":
                raise video_factory.UserFacingError("stop failed")
            normal(executable, stage, arguments)
        self.fake_command = failure
        status, _, err = self.approved_run()
        self.assertEqual(status, 2)
        self.assertIn("vf-transcribe-", err)
        self.assertEqual(len(list((self.project / "work/transcripts").glob("*.json"))), 1)

    def test_usage_parser_accepts_cli_074_format_without_leaking_other_output(self):
        result = subprocess.CompletedProcess([], 0, "Current balance: 123.45 compute units\nUsage rate: 1.25/hr\nActive assignments: 2", "secret")
        with patch.object(colab.subprocess, "run", return_value=result):
            self.assertEqual(colab._colab_usage_snapshot("colab"), {"balance": "123.45", "rate": "1.25", "assignments": "2"})

    def test_stop_missing_session_is_not_confirmed_cleanup(self):
        with patch.object(colab.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, b"Session not found.")):
            with self.assertRaisesRegex(video_factory.UserFacingError, "did not confirm"):
                colab._run_colab("colab", "stop", ["stop", "-s", "owned"])

    def test_subprocess_timeout_is_sanitized(self):
        with patch.object(colab.subprocess, "run", side_effect=subprocess.TimeoutExpired("secret", 10, output=b"secret")):
            with self.assertRaises(video_factory.UserFacingError) as raised:
                colab._run_colab("colab", "exec", [])
        self.assertNotIn("secret", str(raised.exception))

    def test_worker_rejects_wrong_gpu_before_install_or_inference(self):
        configuration = {"gpu": "T4", "model": "large-v3-turbo", "compute_type": "float16", "input_path": "audio.wav", "output_path": "out.json", "transcribe": {}}
        with patch.object(subprocess, "run", return_value=types.SimpleNamespace(stdout="NVIDIA A100\n")) as run:
            with self.assertRaisesRegex(RuntimeError, "differs"):
                exec(colab._worker_source(configuration), {})
        self.assertEqual(run.call_count, 1)

    def test_generated_inference_serializes_timestamps(self):
        output = self.project / "result.json"
        configuration = {"model": "large-v3-turbo", "compute_type": "float16", "input_path": "audio.wav", "output_path": str(output), "transcribe": {"language": "zh"}}
        model = types.SimpleNamespace(transcribe=lambda *a, **kw: ([types.SimpleNamespace(start=0.25, end=1.5, text="你好")], None))
        module = types.SimpleNamespace(WhisperModel=lambda *a, **kw: model)
        with patch.dict(sys.modules, {"faster_whisper": module}):
            exec(colab._inference_source(configuration), {})
        response = json.loads(output.read_text())
        self.assertEqual(response["segments"], [{"start": 0.25, "end": 1.5, "text": "你好"}])


if __name__ == "__main__":
    unittest.main()
