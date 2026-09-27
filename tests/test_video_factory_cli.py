from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "skills" / "video-factory" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import qa_helpers  # noqa: E402
import video_factory  # noqa: E402


class VideoFactoryCLITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="video-factory-tests-")
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def project(self, name: str) -> Path:
        path = self.root / name
        path.mkdir(parents=True)
        return path

    def run_cli(self, *args: str) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = video_factory.main(list(args))
        return result, stdout.getvalue(), stderr.getvalue()

    def write_json(self, path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def project_with_plan_inputs(self, name: str = "plan-project") -> tuple[Path, dict]:
        project = self.project(name)
        video = project / "assets" / "videos" / "clip.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"synthetic video bytes")
        reference = project / "assets" / "references" / "guide.pdf"
        reference.parent.mkdir(parents=True)
        reference.write_bytes(b"synthetic reference bytes")
        (project / "job.yaml").write_text("profile: public-health\n", encoding="utf-8")
        self.write_json(
            project / "work" / "manifests" / "media_manifest.json",
            {
                "assets": [
                    {
                        "source": "assets/videos/clip.mp4",
                        "duration_seconds": 5.0,
                    }
                ],
                "errors": [],
            },
        )
        plan = {
            "references": [
                {
                    "id": "guide-1",
                    "title": "Synthetic source for validator coverage",
                    "path": "assets/references/guide.pdf",
                    "source_type": "project_reference_file",
                }
            ],
            "timeline": [
                {
                    "type": "video",
                    "source": "assets/videos/clip.mp4",
                    "source_in": 0.5,
                    "source_out": 4.5,
                    "timeline_start": 0.0,
                    "timeline_end": 4.0,
                },
                {
                    "type": "text_card",
                    "text": "合成驗證用內容",
                    "timeline_start": 1.0,
                    "timeline_end": 3.0,
                    "medical_claims": [
                        {
                            "statement": "合成驗證用主張",
                            "reference_ids": ["guide-1"],
                        }
                    ],
                },
            ],
        }
        return project, plan

    def test_init_creates_reviewable_draft_and_preserves_existing_jobs(self) -> None:
        project = self.project("family-trip")

        result, stdout, _ = self.run_cli("init", str(project), "--profile", "memory")
        self.assertEqual(result, 0)
        self.assertIn("Created job.draft.yaml", stdout)
        draft = project / "job.draft.yaml"
        self.assertTrue(draft.is_file())
        self.assertIn("profile: memory", draft.read_text(encoding="utf-8"))
        self.assertTrue((project / "assets" / "photos").is_dir())
        self.assertTrue((project / "work" / "manifests").is_dir())
        self.assertTrue((project / "outputs").is_dir())

        draft.write_text("# edited by the project owner\nprofile: memory\n", encoding="utf-8")
        existing_job = project / "job.yaml"
        existing_job.write_text("# approved project settings\nprofile: public-health\n", encoding="utf-8")

        result, stdout, _ = self.run_cli("init", str(project), "--profile", "public-health")
        self.assertEqual(result, 0)
        self.assertIn("existing job.yaml was preserved", stdout)
        self.assertEqual(existing_job.read_text(encoding="utf-8"), "# approved project settings\nprofile: public-health\n")
        self.assertEqual(draft.read_text(encoding="utf-8"), "# edited by the project owner\nprofile: memory\n")

    def test_inspect_reuses_metadata_for_same_hash_and_reprobes_changed_content(self) -> None:
        project = self.project("inspect-project")
        asset = project / "assets" / "videos" / "clip.mp4"
        asset.parent.mkdir(parents=True)
        asset.write_bytes(b"synthetic video version one")
        metadata = {
            "duration_seconds": 8.0,
            "frame_rate": 30.0,
            "width": 640,
            "height": 360,
            "video_streams": [{"codec_type": "video"}],
            "audio_streams": [],
            "has_audio": False,
        }

        with patch.object(video_factory, "ffprobe_metadata", return_value=metadata) as probe:
            result, _, _ = self.run_cli("inspect", str(project))
            self.assertEqual(result, 0)
            first = json.loads((project / "work/manifests/media_manifest.json").read_text(encoding="utf-8"))["assets"][0]
            self.assertFalse(first["cache_reused"])
            first_hash = first["sha256"]

            result, _, _ = self.run_cli("inspect", str(project))
            self.assertEqual(result, 0)
            second = json.loads((project / "work/manifests/media_manifest.json").read_text(encoding="utf-8"))["assets"][0]
            self.assertTrue(second["cache_reused"])
            self.assertEqual(second["sha256"], first_hash)
            self.assertEqual(probe.call_count, 1)

            asset.write_bytes(b"synthetic video version two")
            result, _, _ = self.run_cli("inspect", str(project))
            self.assertEqual(result, 0)
            changed = json.loads((project / "work/manifests/media_manifest.json").read_text(encoding="utf-8"))["assets"][0]
            self.assertFalse(changed["cache_reused"])
            self.assertNotEqual(changed["sha256"], first_hash)
            self.assertEqual(probe.call_count, 2)

    def test_ffprobe_metadata_uses_display_matrix_for_orientation(self) -> None:
        probe_data = {
            "format": {"format_name": "mov,mp4,m4a,3gp,3g2,mj2"},
            "streams": [{
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
                "r_frame_rate": "30/1",
                "tags": {},
                "side_data_list": [{"rotation": -90}],
                "disposition": {"attached_pic": 0},
            }],
        }
        with patch.object(video_factory.shutil, "which", return_value="/fake/ffprobe"), patch.object(
            video_factory, "run_json_command", return_value=probe_data
        ):
            metadata = video_factory.ffprobe_metadata(self.root / "portrait.mov")

        self.assertEqual((metadata["width"], metadata["height"]), (1080, 1920))
        self.assertEqual(metadata["orientation"], "portrait")
        self.assertEqual(metadata["rotation_degrees"], -90)
        self.assertEqual(metadata["video_streams"][0]["width"], 1920)
        self.assertEqual(metadata["video_streams"][0]["height"], 1080)

    def test_validate_plan_accepts_project_local_claim_provenance(self) -> None:
        project, plan = self.project_with_plan_inputs()

        errors, warnings = video_factory.validate_plan_data(project, plan)

        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_validate_plan_rejects_source_range_beyond_inspected_duration(self) -> None:
        project, plan = self.project_with_plan_inputs()
        plan["timeline"][0]["source_out"] = 5.5

        errors, _ = video_factory.validate_plan_data(project, plan)

        self.assertTrue(any("exceeds source duration" in error for error in errors), errors)

    def test_validate_plan_rejects_unregistered_public_health_reference(self) -> None:
        project, plan = self.project_with_plan_inputs()
        plan["timeline"][1]["medical_claims"][0]["reference_ids"] = ["missing-guide"]

        errors, _ = video_factory.validate_plan_data(project, plan)

        self.assertTrue(any("unknown reference ID" in error for error in errors), errors)

    def test_validate_plan_rejects_source_path_traversal(self) -> None:
        project, plan = self.project_with_plan_inputs()
        plan["timeline"][0]["source"] = "../outside.mp4"

        errors, _ = video_factory.validate_plan_data(project, plan)

        self.assertTrue(any("cannot be absolute or contain '..'" in error for error in errors), errors)

    def test_validate_plan_rejects_reference_path_traversal(self) -> None:
        project, plan = self.project_with_plan_inputs()
        plan["references"][0]["path"] = "../outside.pdf"

        errors, _ = video_factory.validate_plan_data(project, plan)

        self.assertTrue(any("references[0].path" in error and "contain '..'" in error for error in errors), errors)

    def test_export_srt_sorts_cues_and_normalizes_line_endings(self) -> None:
        project = self.project("subtitle-project")
        self.write_json(
            project / "work" / "edit-plan" / "edit_plan.json",
            {
                "timeline": [
                    {
                        "type": "subtitle",
                        "text": "第二句",
                        "timeline_start": 65.234,
                        "timeline_end": 67.5,
                    },
                    {
                        "type": "subtitle",
                        "text": "第一行\r\n第二行",
                        "timeline_start": 2.0,
                        "timeline_end": 3.2,
                    },
                ]
            },
        )

        result, stdout, _ = self.run_cli("export-srt", str(project))

        self.assertEqual(result, 0)
        self.assertIn("Exported 2 subtitle cue(s)", stdout)
        subtitle_file = project / "work" / "transcripts" / "captions.srt"
        self.assertEqual(
            subtitle_file.read_text(encoding="utf-8"),
            "1\n00:00:02,000 --> 00:00:03,200\n第一行\n第二行\n\n"
            "2\n00:01:05,234 --> 00:01:07,500\n第二句\n",
        )

    def test_qa_reports_invalid_edit_plan_as_error(self) -> None:
        project = self.project("qa-project")
        rendered = project / "outputs" / "synthetic.mp4"
        rendered.parent.mkdir(parents=True)
        rendered.write_bytes(b"fake rendered media; ffprobe is mocked")
        self.write_json(
            project / "work" / "edit-plan" / "edit_plan.json",
            {
                "duration_seconds": 1.0,
                "timeline": [
                    {
                        "type": "text_card",
                        "text": "synthetic QA",
                        "timeline_start": 1.0,
                        "timeline_end": 0.5,
                    }
                ],
            },
        )
        probe = {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 640,
                    "height": 360,
                    "disposition": {"attached_pic": 0},
                }
            ],
            "format": {"duration": "1.0"},
        }

        with patch.object(qa_helpers, "ffprobe_render", return_value=(probe, None)), patch.object(
            qa_helpers.shutil, "which", return_value=None
        ):
            result, stdout, _ = self.run_cli("qa", str(project), "--video", "outputs/synthetic.mp4")

        self.assertEqual(result, 2)
        self.assertIn("QA status: ERROR", stdout)
        report = (project / "outputs" / "qa_report.md").read_text(encoding="utf-8")
        self.assertIn("Status: **ERROR**", report)
        self.assertIn("invalid timeline range", report)


if __name__ == "__main__":
    unittest.main()
