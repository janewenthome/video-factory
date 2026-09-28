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

import video_factory
from video_editor.queue_runner import QueueApplicationError, VideoFactoryQueue
from video_editor.system_resources import MemorySnapshot
from video_editor.application import ProjectOptions
from video_editor.cli import build_parser


class QueueApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="vf-queue-application-")
        self.root = Path(self.temp_dir.name)
        self.queue = VideoFactoryQueue(
            self.root / "state" / "queue.json",
            probe=lambda: MemorySnapshot(72.0, 0, 0, "fixed"),
            sleep=lambda _seconds: None,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def create_ready_project(self, name: str) -> Path:
        project = self.root / name
        (project / "work/analysis").mkdir(parents=True)
        (project / "work/edit-plan").mkdir(parents=True)
        (project / "job.yaml").write_text("profile: memory\n", encoding="utf-8")
        (project / "work/analysis/story_plan.json").write_text(
            json.dumps({"profile": "memory", "structure": ["opening", "ending"]}),
            encoding="utf-8",
        )
        (project / "work/edit-plan/edit_plan.json").write_text(
            json.dumps({"timeline": []}),
            encoding="utf-8",
        )
        return project

    def test_enqueue_requires_director_plans_and_validates_edit_plan(self) -> None:
        project = self.create_ready_project("prepared")
        with patch.object(video_factory, "command_validate_plan", return_value=0) as validate:
            job = self.queue.enqueue_project(project, max_attempts=1, force=True)

        validate.assert_called_once()
        self.assertEqual(job["status"], "pending")
        self.assertEqual(job["payload"], {"mode": "family", "force": True, "execution": "local"})

    def test_enqueue_rejects_projects_without_a_codex_edit_plan(self) -> None:
        project = self.root / "unprepared"
        (project / "work/analysis").mkdir(parents=True)
        (project / "job.yaml").write_text("profile: family\n", encoding="utf-8")
        (project / "work/analysis/story_plan.json").write_text("{}", encoding="utf-8")

        with self.assertRaisesRegex(QueueApplicationError, "edit-plan/edit_plan.json"):
            self.queue.enqueue_project(project)

    def test_runner_serializes_ready_projects_and_continues_after_failure(self) -> None:
        projects = [self.create_ready_project(f"project-{index}") for index in range(1, 4)]
        with patch.object(video_factory, "command_validate_plan", return_value=0):
            jobs = [self.queue.enqueue_project(project, max_attempts=1) for project in projects]

        active = 0
        maximum_active = 0
        visited: list[str] = []

        def fake_run(job: dict) -> None:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            try:
                visited.append(Path(job["project_path"]).name)
                if job["id"] == jobs[0]["id"]:
                    raise RuntimeError("mock failure")
            finally:
                active -= 1

        with patch.object(self.queue, "_run_project", side_effect=fake_run):
            summary = self.queue.run(cooldown_seconds=0, max_rechecks=0)

        self.assertEqual(maximum_active, 1)
        self.assertEqual(visited, ["project-1", "project-2", "project-3"])
        self.assertEqual(summary.jobs_failed, 1)
        self.assertEqual(summary.jobs_completed, 2)
        self.assertEqual([self.queue.queue.get_job(job["id"])["status"] for job in jobs], [
            "failed", "completed", "completed",
        ])

    def test_resource_pause_is_reported_and_first_cut_defaults_to_auto(self) -> None:
        project = self.create_ready_project("ready")
        with patch.object(video_factory, "command_validate_plan", return_value=0):
            self.queue.enqueue_project(project)

        events: list[dict] = []
        self.queue.on_progress = events.append
        self.queue._probe = lambda: MemorySnapshot(None, None, None, "fixed", "probe unavailable")
        summary = self.queue.run(cooldown_seconds=0, poll_seconds=0.001, max_rechecks=0)

        self.assertTrue(summary.paused)
        self.assertIn("queue_paused", [event["type"] for event in events])
        self.assertEqual(ProjectOptions().review_gate, "AUTO")
        self.assertEqual(build_parser().parse_args(["run", str(project)]).gate, "AUTO")


if __name__ == "__main__":
    unittest.main()
