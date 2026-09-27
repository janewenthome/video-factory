"""Tests for new hybrid video editor pipeline features."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from edit_summary import generate_edit_summary
from select_compute import (
    ComputeSelectionError,
    explain_compute_plan,
    select_compute_for_task,
)
from video_editor.pipeline import PipelineOrchestrator, load_pipeline_state


class TestHybridFeatures(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="vf-test-hybrid-")
        self.project = Path(self.temp_dir.name)
        (self.project / "assets").mkdir(parents=True)
        (self.project / "work").mkdir(parents=True)
        (self.project / "outputs").mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_cloud_job(
        self,
        *,
        cloud_perception: bool,
        colab_policy: str,
        privacy_mode: str = "BALANCED",
        gpu_policy: str = "T4",
        temporal_backend: str = "none",
    ) -> None:
        (self.project / "job.yaml").write_text(
            f"profile: family\n"
            f"privacy_mode: {privacy_mode}\n"
            f"cloud_perception: {'true' if cloud_perception else 'false'}\n"
            f"gpu_policy: {gpu_policy}\n"
            f"temporal_backend: {temporal_backend}\n"
            "cloud_processing:\n"
            f"  colab: {colab_policy}\n",
            encoding="utf-8",
        )

    def _patch_stages_before_perception(self, stack: ExitStack, orch: PipelineOrchestrator) -> None:
        for method_name in (
            "run_stage_1_ingest",
            "run_stage_2_proxy",
            "run_stage_3_scenes",
            "run_stage_4_audio",
            "run_stage_5_transcription",
            "run_stage_6_contact_sheets",
        ):
            stack.enter_context(patch.object(orch, method_name))

    def test_compute_selector_rules(self):
        # Local CPU tasks
        self.assertEqual(select_compute_for_task("probe"), "cpu")
        self.assertEqual(select_compute_for_task("proxy"), "cpu")
        self.assertEqual(select_compute_for_task("render"), "cpu")

        # Default AI task chooses T4
        self.assertEqual(select_compute_for_task("transcribe"), "T4")

        # L4 is reserved for shortlist-only temporal analysis.
        with self.assertRaises(ComputeSelectionError):
            select_compute_for_task("transcribe", requested_gpu="L4")
        self.assertEqual(select_compute_for_task("temporal", requested_gpu="L4"), "L4")

        # Prohibited GPUs raise error unless explicitly allowed
        with self.assertRaises(ComputeSelectionError):
            select_compute_for_task("transcribe", requested_gpu="A100")
        with self.assertRaises(ComputeSelectionError):
            select_compute_for_task("transcribe", requested_gpu="H100")

        # Explicit allow_premium
        self.assertEqual(
            select_compute_for_task("transcribe", requested_gpu="A100", allow_premium=True),
            "A100",
        )

    def test_explain_compute_plan(self):
        cpu_plan = explain_compute_plan("render", "CPU")
        self.assertIn("0 Colab", cpu_plan["cost"])
        t4_plan = explain_compute_plan("transcribe", "T4")
        self.assertEqual(t4_plan["accelerator"], "T4")

    def test_edit_summary_generator(self):
        # Create minimal edit plan
        plan_dir = self.project / "work" / "edit-plan"
        plan_dir.mkdir(parents=True)
        plan_file = plan_dir / "edit_plan.json"
        plan_data = {
            "title": "測試影片",
            "fps": 30.0,
            "duration_seconds": 15.0,
            "timeline": [
                {
                    "type": "video",
                    "source": "assets/videos/clip1.mp4",
                    "timeline_start": 0.0,
                    "timeline_end": 5.0,
                    "source_in": 0.0,
                    "source_out": 5.0,
                    "notes": "開場鏡頭",
                },
                {
                    "type": "subtitle",
                    "timeline_start": 1.0,
                    "timeline_end": 4.0,
                    "text": "這是測試字幕",
                },
                {
                    "type": "music",
                    "source": "assets/music/bgm.mp3",
                    "timeline_start": 0.0,
                    "timeline_end": 15.0,
                },
            ],
        }
        plan_file.write_text(json.dumps(plan_data, ensure_ascii=False), encoding="utf-8")

        summary_path = generate_edit_summary(self.project)
        self.assertTrue(summary_path.is_file())
        text = summary_path.read_text(encoding="utf-8")
        self.assertIn("剪輯計畫審核摘要：測試影片", text)
        self.assertIn("開場鉤子", text)
        self.assertIn("clip1.mp4", text)
        self.assertIn("總字幕段落數：1 段", text)

    def test_pipeline_dry_run(self):
        orch = PipelineOrchestrator(self.project, dry_run=True, no_gpu=False)
        self.assertEqual(orch.run(), 0)

    def test_perception_uses_local_path_when_colab_is_not_requested_or_disabled(self):
        cases = (
            (False, "ask_each_run", "BALANCED", "T4"),
            (True, "disabled", "BALANCED", "T4"),
            (True, "ask_each_run", "LOCAL_ONLY", "L4"),
        )
        for cloud_requested, colab_policy, privacy_mode, gpu_policy in cases:
            with self.subTest(
                cloud_requested=cloud_requested,
                colab_policy=colab_policy,
                privacy_mode=privacy_mode,
                gpu_policy=gpu_policy,
            ):
                self._write_cloud_job(
                    cloud_perception=cloud_requested,
                    colab_policy=colab_policy,
                    privacy_mode=privacy_mode,
                    gpu_policy=gpu_policy,
                )
                events = []
                orch = PipelineOrchestrator(
                    self.project,
                    gpu=gpu_policy,
                    allow_upload=True,
                    resume=False,
                    progress_callback=events.append,
                )
                expected_path = self.project / "work" / "perception_index.json"
                with (
                    patch("video_editor.pipeline.build_perception_index", return_value=expected_path) as local_index,
                    patch("colab_perception.command_colab_perception") as colab_worker,
                ):
                    orch.run_stage_7_perception()

                local_index.assert_called_once()
                self.assertFalse(local_index.call_args.kwargs["cloud"])
                self.assertFalse(local_index.call_args.kwargs["allow_upload"])
                self.assertEqual(local_index.call_args.kwargs["gpu_policy"], "T4")
                colab_worker.assert_not_called()
                self.assertEqual(events[-1]["status"], "completed")
                self.assertEqual(orch.state["perception_execution_mode"], "local")

    def test_perception_without_per_run_consent_stops_before_editorial_or_render(self):
        self._write_cloud_job(cloud_perception=True, colab_policy="ask_each_run")
        events = []
        orch = PipelineOrchestrator(
            self.project,
            allow_upload=False,
            resume=False,
            gate="AUTO",
            progress_callback=events.append,
        )
        with ExitStack() as stack:
            self._patch_stages_before_perception(stack, orch)
            colab_worker = stack.enter_context(patch("colab_perception.command_colab_perception"))

            def fake_local_index(project, **kwargs):
                output = self.project / "work" / "perception_index.json"
                output.write_text(json.dumps({
                    "privacy": {"cloud_status": "awaiting_authorization", "original_media_uploaded": False},
                }), encoding="utf-8")
                return output

            local_index = stack.enter_context(
                patch("video_editor.pipeline.build_perception_index", side_effect=fake_local_index)
            )
            editorial = stack.enter_context(patch.object(orch, "run_stage_8_editorial_analysis"))
            render = stack.enter_context(patch.object(orch, "run_stage_11_render"))
            result = orch.run()

        self.assertEqual(result, 3)
        local_index.assert_called_once()
        self.assertTrue(local_index.call_args.kwargs["cloud"])
        self.assertFalse(local_index.call_args.kwargs["allow_upload"])
        self.assertTrue(local_index.call_args.kwargs["force"])
        colab_worker.assert_not_called()
        editorial.assert_not_called()
        render.assert_not_called()
        self.assertEqual(orch.state["stages"]["perception"]["status"], "pending")
        self.assertNotIn("perception", orch.state["completed_stages"])
        self.assertNotIn("editorial_analysis", orch.state["completed_stages"])
        self.assertFalse((self.project / "work" / "analysis" / "story_plan.json").exists())
        index = json.loads((self.project / "work" / "perception_index.json").read_text(encoding="utf-8"))
        self.assertEqual(index["privacy"]["cloud_status"], "awaiting_authorization")
        self.assertEqual(events[-1]["status"], "pending")

    def test_authorized_perception_calls_colab_worker_with_policy_and_requires_result(self):
        self._write_cloud_job(cloud_perception=True, colab_policy="ask_each_run")
        events = []
        orch = PipelineOrchestrator(
            self.project,
            allow_upload=True,
            resume=False,
            progress_callback=events.append,
        )

        def fake_colab_worker(args):
            output = self.project / "work" / "perception_index.json"
            output.write_text(json.dumps({
                "privacy": {"cloud_status": "completed", "original_media_uploaded": False},
            }), encoding="utf-8")
            return 0

        with (
            patch("video_editor.pipeline.build_perception_index") as local_index,
            patch("colab_perception.command_colab_perception", side_effect=fake_colab_worker) as colab_worker,
        ):
            orch.run_stage_7_perception()

        local_index.assert_not_called()
        args = colab_worker.call_args.args[0]
        self.assertEqual(args.project, str(self.project.resolve()))
        self.assertEqual(args.privacy_mode, "BALANCED")
        self.assertEqual(args.gpu, "T4")
        self.assertEqual(args.temporal_backend, "none")
        self.assertTrue(args.allow_upload)
        self.assertEqual(events[-1]["status"], "completed")
        self.assertEqual(orch.state["perception_execution_mode"], "colab")

    def test_authorized_l4_route_keeps_max_quality_temporal_constraints(self):
        self._write_cloud_job(
            cloud_perception=True,
            colab_policy="ask_each_run",
            privacy_mode="MAX_QUALITY",
            gpu_policy="L4",
            temporal_backend="smolvlm2",
        )
        orch = PipelineOrchestrator(self.project, gpu="L4", allow_upload=True, resume=False)

        def fake_colab_worker(args):
            output = self.project / "work" / "perception_index.json"
            output.write_text(json.dumps({
                "privacy": {"cloud_status": "completed", "original_media_uploaded": False},
            }), encoding="utf-8")
            return 0

        with patch("colab_perception.command_colab_perception", side_effect=fake_colab_worker) as colab_worker:
            orch.run_stage_7_perception()

        args = colab_worker.call_args.args[0]
        self.assertEqual(args.gpu, "L4")
        self.assertEqual(args.privacy_mode, "MAX_QUALITY")
        self.assertEqual(args.temporal_backend, "smolvlm2")

    def test_colab_perception_failure_fails_pipeline_before_story_or_render(self):
        self._write_cloud_job(cloud_perception=True, colab_policy="ask_each_run")
        events = []
        orch = PipelineOrchestrator(
            self.project,
            allow_upload=True,
            resume=False,
            gate="AUTO",
            progress_callback=events.append,
        )
        with ExitStack() as stack:
            self._patch_stages_before_perception(stack, orch)
            colab_worker = stack.enter_context(
                patch(
                    "colab_perception.command_colab_perception",
                    side_effect=RuntimeError("synthetic worker failure"),
                )
            )
            editorial = stack.enter_context(patch.object(orch, "run_stage_8_editorial_analysis"))
            render = stack.enter_context(patch.object(orch, "run_stage_11_render"))
            result = orch.run()

        self.assertEqual(result, 2)
        colab_worker.assert_called_once()
        editorial.assert_not_called()
        render.assert_not_called()
        self.assertEqual(orch.state["stages"]["perception"]["status"], "failed")
        self.assertNotIn("editorial_analysis", orch.state["completed_stages"])
        self.assertFalse(any(event["stage"] == "perception" and event["status"] == "completed" for event in events))

    def test_pipeline_state_persistence_and_resume(self):
        orch = PipelineOrchestrator(self.project, resume=True)
        orch.mark_stage_start("ingest")
        orch.mark_stage_complete("ingest", outputs=["manifest.json"])

        state = load_pipeline_state(self.project)
        self.assertIn("ingest", state["completed_stages"])
        self.assertEqual(state["stages"]["ingest"]["status"], "completed")

        # Resume check
        self.assertTrue(orch.is_stage_completed("ingest"))
        self.assertFalse(orch.is_stage_completed("proxy"))


if __name__ == "__main__":
    unittest.main()
