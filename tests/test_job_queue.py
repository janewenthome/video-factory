from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from video_editor.job_queue import PersistentJobQueue, QueueAlreadyRunning


class PersistentJobQueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = TemporaryDirectory(prefix="video-factory-queue-")
        self.root = Path(self.temp_dir.name)
        self.queue = PersistentJobQueue(self.root / "state" / "queue.json")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def enqueue_three(self, *, max_attempts: int = 2) -> list[dict]:
        return [
            self.queue.enqueue(self.root / f"project-{index}", max_attempts=max_attempts)
            for index in range(1, 4)
        ]

    def test_three_mock_jobs_run_sequentially_and_complete_in_order(self) -> None:
        jobs = self.enqueue_three()
        active = 0
        maximum_active = 0
        visited: list[str] = []
        guard = threading.Lock()

        def run_one(job: dict) -> None:
            nonlocal active, maximum_active
            with guard:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                visited.append(Path(job["project_path"]).name)
                time.sleep(0.01)
            finally:
                with guard:
                    active -= 1

        summary = self.queue.run(run_one)

        self.assertEqual(maximum_active, 1)
        self.assertEqual(visited, ["project-1", "project-2", "project-3"])
        self.assertEqual(summary.attempts_started, 3)
        self.assertEqual(summary.jobs_completed, 3)
        self.assertEqual([job["status"] for job in self.queue.list_jobs()], ["completed"] * 3)
        self.assertEqual([job["id"] for job in self.queue.list_jobs()], [job["id"] for job in jobs])

    def test_failed_job_retries_with_limit_and_does_not_block_later_jobs(self) -> None:
        jobs = self.enqueue_three(max_attempts=2)
        visited: list[str] = []

        def run_one(job: dict) -> None:
            name = Path(job["project_path"]).name
            visited.append(name)
            if job["id"] == jobs[0]["id"]:
                raise RuntimeError("synthetic failure")

        summary = self.queue.run(run_one)

        self.assertEqual(visited, ["project-1", "project-2", "project-3", "project-1"])
        self.assertEqual(summary.attempts_started, 4)
        self.assertEqual(summary.jobs_completed, 2)
        self.assertEqual(summary.jobs_failed, 1)
        by_id = {job["id"]: job for job in self.queue.list_jobs()}
        self.assertEqual(by_id[jobs[0]["id"]]["status"], "failed")
        self.assertEqual(by_id[jobs[0]["id"]]["attempts"], 2)
        self.assertEqual(by_id[jobs[0]["id"]]["last_error"], "RuntimeError: synthetic failure")
        self.assertEqual([by_id[job["id"]]["status"] for job in jobs[1:]], ["completed", "completed"])

    def test_interrupted_job_recovers_after_waiting_jobs_and_keeps_checkpoint(self) -> None:
        interrupted, waiting = self.enqueue_three()[:2]
        state = json.loads(self.queue.queue_file.read_text(encoding="utf-8"))
        state["jobs"][0]["status"] = "running"
        state["jobs"][0]["attempts"] = 1
        state["jobs"][0]["checkpoint"] = {"stage": "transcription", "cache_key": "abc123"}
        self.queue.queue_file.write_text(json.dumps(state), encoding="utf-8")
        visited: list[str] = []

        def run_one(job: dict) -> None:
            visited.append(Path(job["project_path"]).name)

        summary = self.queue.run(run_one)

        self.assertEqual(visited, ["project-2", "project-3", "project-1"])
        self.assertEqual(summary.jobs_completed, 3)
        recovered = self.queue.get_job(interrupted["id"])
        self.assertEqual(recovered["status"], "completed")
        self.assertEqual(recovered["attempts"], 2)
        self.assertEqual(recovered["checkpoint"], {"stage": "transcription", "cache_key": "abc123"})
        self.assertIn("Runner stopped", recovered["attempt_errors"][0]["error"])
        self.assertEqual(self.queue.get_job(waiting["id"])["status"], "completed")

    def test_runner_lock_controls_and_runtime_metadata(self) -> None:
        job = self.queue.enqueue(self.root / "project-lock")
        entered = threading.Event()
        release = threading.Event()
        runner_error: list[BaseException] = []

        def callback(current: dict) -> None:
            self.queue.update_runtime(current["id"], {"pid": 1234, "process_group_id": 1234, "log_path": "/tmp/job.log"})
            entered.set()
            release.wait(timeout=2)
            self.assertTrue(self.queue.get_job(current["id"])["cancel_requested"])

        def run_queue() -> None:
            try:
                self.queue.run(callback)
            except BaseException as exc:  # surfaced in the test thread below
                runner_error.append(exc)

        runner_thread = threading.Thread(target=run_queue)
        runner_thread.start()
        self.assertTrue(entered.wait(timeout=2))
        with self.assertRaises(QueueAlreadyRunning):
            self.queue.run(lambda _: None)

        self.assertEqual(self.queue.cancel(job["id"]), "cancellation_requested")
        self.assertEqual(self.queue.get_job(job["id"])["runtime"]["pid"], 1234)
        release.set()
        runner_thread.join(timeout=2)

        self.assertFalse(runner_thread.is_alive())
        self.assertEqual(runner_error, [])
        self.assertEqual(self.queue.get_job(job["id"])["status"], "cancelled")

    def test_pause_resume_and_pending_reorder(self) -> None:
        jobs = self.enqueue_three()
        self.queue.reorder([jobs[2]["id"], jobs[0]["id"], jobs[1]["id"]])
        self.queue.pause("memory pressure")
        paused_summary = self.queue.run(lambda _: self.fail("paused queue must not start work"))
        self.assertTrue(paused_summary.paused)
        self.assertEqual(paused_summary.pause_reason, "memory pressure")

        self.queue.resume()
        visited: list[str] = []
        self.queue.run(lambda job: visited.append(Path(job["project_path"]).name))
        self.assertEqual(visited, ["project-3", "project-1", "project-2"])

    def test_resource_gate_pauses_and_cooldown_runs_between_jobs(self) -> None:
        self.enqueue_three()
        visited: list[str] = []
        cooldowns: list[str] = []
        gate_checks: list[str] = []

        def gate(job: dict) -> bool:
            name = Path(job["project_path"]).name
            gate_checks.append(name)
            return name != "project-2"

        first = self.queue.run(
            lambda job: visited.append(Path(job["project_path"]).name),
            cooldown=lambda: cooldowns.append("cooled"),
            resource_gate=gate,
        )

        self.assertEqual(visited, ["project-1"])
        self.assertEqual(cooldowns, ["cooled"])
        self.assertEqual(gate_checks, ["project-1", "project-2"])
        self.assertTrue(first.paused)
        self.assertEqual([job["status"] for job in self.queue.list_jobs()], ["completed", "pending", "pending"])

        self.queue.resume()
        second = self.queue.run(lambda job: visited.append(Path(job["project_path"]).name))
        self.assertFalse(second.paused)
        self.assertEqual(visited, ["project-1", "project-2", "project-3"])


if __name__ == "__main__":
    unittest.main()
