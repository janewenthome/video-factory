from __future__ import annotations

import json
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import test_video_factory_cli as fixtures
import media_helpers
import video_factory


class CoreRegressions(unittest.TestCase):
    setUp = fixtures.VideoFactoryCLITests.setUp
    tearDown = fixtures.VideoFactoryCLITests.tearDown
    project = fixtures.VideoFactoryCLITests.project
    run_cli = fixtures.VideoFactoryCLITests.run_cli
    write_json = fixtures.VideoFactoryCLITests.write_json
    project_with_plan_inputs = fixtures.VideoFactoryCLITests.project_with_plan_inputs

    def test_frame_fallback_honors_cap(self):
        times = media_helpers.select_video_times({"source": "clip", "duration_seconds": 20}, {}, 10, 3)
        self.assertEqual(len(times), 3)
        self.assertTrue(all(0 < time < 20 for time, _ in times))

    def test_failed_frame_is_retried(self):
        project = self.project("retry-frames")
        asset = {"source": "assets/photos/photo.jpg", "sha256": "a" * 64, "kind": "photo"}
        with patch.object(media_helpers, "require_ffmpeg", return_value="ffmpeg"), patch.object(
            media_helpers, "get_manifest_assets", return_value=[asset]
        ), patch.object(media_helpers, "run_ffmpeg_output", return_value=(False, "decode failed")) as runner:
            for _ in range(2):
                result, _, _ = self.run_cli("extract-frames", str(project))
                self.assertEqual(result, 2)
            self.assertEqual(runner.call_count, 2)

    def test_zero_byte_ffmpeg_result_preserves_existing_derivative(self):
        output = self.root / "frame.jpg"
        output.write_bytes(b"valid prior output")
        with patch.object(media_helpers, "run_ffmpeg", return_value=(True, "")):
            ok, _ = media_helpers.run_ffmpeg_output(["ffmpeg", str(output)], output)
        self.assertFalse(ok)
        self.assertEqual(output.read_bytes(), b"valid prior output")

    def test_heic_conversion_prefers_libheif_and_scales_via_decodable_jpeg(self):
        source = self.root / "photo.heic"
        output = self.root / "work" / "frame.jpg"
        output.parent.mkdir(parents=True)
        source.write_bytes(b"synthetic HEIC source")

        def fake_which(name):
            return {"heif-convert": "/fake/heif-convert", "ffmpeg": "/fake/ffmpeg"}.get(name)

        def fake_converter(command, **kwargs):
            Path(command[-1]).write_bytes(b"decoded full image")
            return subprocess.CompletedProcess(command, 0, "", "")

        def fake_scale(command, destination):
            self.assertIn("-i", command)
            self.assertIn("scale=960:-2:force_original_aspect_ratio=decrease", command)
            destination.write_bytes(b"verified scaled JPEG")
            return True, ""

        with (
            patch.object(media_helpers.shutil, "which", side_effect=fake_which),
            patch.object(media_helpers.subprocess, "run", side_effect=fake_converter) as converter,
            patch.object(media_helpers, "run_ffmpeg_output", side_effect=fake_scale),
        ):
            ok, _ = media_helpers.decode_heic_output(source, output)

        self.assertTrue(ok)
        self.assertTrue(output.is_file())
        self.assertEqual(converter.call_args.args[0][0], "/fake/heif-convert")

    def test_heic_decode_failure_does_not_publish_frame(self):
        source = self.root / "photo.heic"
        output = self.root / "work" / "frame.jpg"
        output.parent.mkdir(parents=True)
        source.write_bytes(b"synthetic HEIC source")

        def fake_which(name):
            return {"heif-convert": "/fake/heif-convert", "ffmpeg": "/fake/ffmpeg"}.get(name)

        def fake_converter(command, **kwargs):
            Path(command[-1]).write_bytes(b"malformed JPEG")
            return subprocess.CompletedProcess(command, 0, "", "")

        with (
            patch.object(media_helpers.shutil, "which", side_effect=fake_which),
            patch.object(media_helpers.subprocess, "run", side_effect=fake_converter),
            patch.object(media_helpers, "run_ffmpeg_output", return_value=(False, "JPEG decode failed")),
        ):
            ok, detail = media_helpers.decode_heic_output(source, output)

        self.assertFalse(ok)
        self.assertIn("could not be decoded", detail)
        self.assertFalse(output.exists())

    def test_heic_decode_uses_ffmpeg_directly_when_libheif_is_unavailable(self):
        source = self.root / "photo.heic"
        output = self.root / "work" / "frame.jpg"
        output.parent.mkdir(parents=True)
        source.write_bytes(b"synthetic HEIC source")
        looked_up: list[str] = []

        def fake_which(name):
            looked_up.append(name)
            return "/fake/ffmpeg" if name == "ffmpeg" else None

        def fake_scale(command, destination):
            self.assertEqual(command[command.index("-i") + 1], str(source))
            self.assertIn("scale=960:-2:force_original_aspect_ratio=decrease", command)
            destination.write_bytes(b"verified scaled JPEG")
            return True, "ok"

        with (
            patch.object(media_helpers.shutil, "which", side_effect=fake_which),
            patch.object(media_helpers, "run_ffmpeg_output", side_effect=fake_scale),
        ):
            ok, _ = media_helpers.decode_heic_output(source, output)

        self.assertTrue(ok)
        self.assertEqual(looked_up, ["heif-convert", "ffmpeg"])
        self.assertEqual(output.read_bytes(), b"verified scaled JPEG")

    def test_one_frame_contact_sheet_real_ffmpeg(self):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("FFmpeg unavailable")
        project = self.project("one-frame")
        frame = project / "work/frames/source.jpg"
        frame.parent.mkdir(parents=True)
        subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=blue:s=96x54", "-frames:v", "1", str(frame)], check=True, capture_output=True)
        self.write_json(project / "work/frames/frames.json", {"assets": {"source": {"frames": [{"path": "work/frames/source.jpg"}]}}})
        code, _, stderr = self.run_cli("build-contact-sheets", str(project))
        self.assertEqual(code, 0, stderr)
        self.assertGreater((project / "work/contact-sheets/sheet-001.jpg").stat().st_size, 0)

    def test_prepare_preserves_dimensions_and_inline_job_ratio(self):
        project, plan = self.project_with_plan_inputs("dimensions")
        (project / "job.yaml").write_text('aspect_ratios: ["9:16"]\n')
        plan.update(width=360, height=640)
        self.write_json(project / "work/edit-plan/edit_plan.json", plan)
        code, _, stderr = self.run_cli("prepare-render", str(project))
        self.assertEqual(code, 0, stderr)
        rendered = json.loads((project / "work/render-input.json").read_text())
        self.assertEqual((rendered["width"], rendered["height"], rendered["ratio"]), (360, 640, "9:16"))

    def test_prepare_render_converts_heic_photos_to_verified_jpeg_and_reuses_them(self):
        project, plan = self.project_with_plan_inputs("heic-render")
        source = project / "assets/photos/family.HEIC"
        source.parent.mkdir(parents=True)
        original_bytes = b"synthetic HEIC bytes"
        source.write_bytes(original_bytes)
        plan["timeline"].insert(
            0,
            {
                "type": "photo",
                "source": "assets/photos/family.HEIC",
                "timeline_start": 0.0,
                "timeline_end": 1.5,
            },
        )
        self.write_json(project / "work/edit-plan/edit_plan.json", plan)

        def fake_decode(input_path, output_path, *, max_dimension=960):
            self.assertEqual(input_path, source.resolve())
            self.assertEqual(max_dimension, 4096)
            output_path.write_bytes(b"verified full-resolution JPEG")
            return True, "ok"

        with patch.object(media_helpers, "decode_heic_output", side_effect=fake_decode) as decoder:
            result, _, stderr = self.run_cli("prepare-render", str(project))
            self.assertEqual(result, 0, stderr)
            first = json.loads((project / "work/render-input.json").read_text())
            photo = first["timeline"][0]
            self.assertEqual(photo["source_project_path"], "assets/photos/family.HEIC")
            self.assertTrue(photo["source_url"].endswith("-heif-v3.jpg"))
            render_jpeg = project / "work/render-public" / photo["source_url"]
            self.assertTrue(render_jpeg.is_file())
            self.assertEqual(render_jpeg.read_bytes(), b"verified full-resolution JPEG")
            self.assertTrue(render_jpeg.with_suffix(".jpg.json").is_file())
            self.assertEqual(source.read_bytes(), original_bytes)

            result, _, stderr = self.run_cli("prepare-render", str(project))
            self.assertEqual(result, 0, stderr)
            self.assertEqual(decoder.call_count, 1)

    def test_rejects_silently_ignored_or_malformed_render_decisions(self):
        for changes in [
            {"type": " video "}, {"type": "transition"}, {"source_out": 3},
            {"audio": {"volume": "loud"}}, {"crop": {"zoom_start": -1}},
            {"fade_in_seconds": float("nan")},
        ]:
            with self.subTest(changes=changes):
                project, plan = self.project_with_plan_inputs("bad-" + str(len(list(self.root.iterdir()))))
                plan["timeline"][0].update(changes)
                errors, _ = video_factory.validate_plan_data(project, plan)
                self.assertTrue(errors)
