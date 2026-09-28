"""Application service for the persistent, serial Video Factory queue."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

from video_editor.job_queue import PersistentJobQueue, RunSummary, default_queue_path
from video_editor.process_lock import (
    HEAVY_LOCK_FD_ENV,
    release_heavy_job,
    wait_for_heavy_job_lock,
)
from video_editor.product_policy import load_product_policy
from video_editor.system_resources import (
    DEFAULT_LOW_FREE_PERCENT,
    MemoryAssessment,
    MemorySnapshot,
    assess_memory_pressure,
    collect_memory_snapshot,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPOSITORY_ROOT / "skills" / "video-factory" / "scripts"
DEFAULT_COOLDOWN_SECONDS = 45.0
DEFAULT_RESOURCE_POLL_SECONDS = 30.0
DEFAULT_RESOURCE_RECHECKS = 3


class QueueApplicationError(RuntimeError):
    """User-facing queue application error."""


class MemoryPressureGate:
    """Pause the queue unless two read-only Mac samples show stable capacity."""

    def __init__(
        self,
        *,
        cooldown_seconds: float,
        poll_seconds: float,
        max_rechecks: int,
        probe: Callable[[], MemorySnapshot] = collect_memory_snapshot,
        sleep: Callable[[float], None] = time.sleep,
        low_free_percent: float = DEFAULT_LOW_FREE_PERCENT,
        on_message: Callable[[str], None] | None = None,
    ):
        if cooldown_seconds < 0 or poll_seconds <= 0 or max_rechecks < 0:
            raise ValueError("Cooldown/recheck settings must be non-negative and poll_seconds must be positive")
        self.cooldown_seconds = cooldown_seconds
        self.poll_seconds = poll_seconds
        self.max_rechecks = max_rechecks
        self.probe = probe
        self.sleep = sleep
        self.low_free_percent = low_free_percent
        self.on_message = on_message or (lambda _message: None)
        self.samples: list[MemorySnapshot] = []
        self.pause_reason: str | None = None

    def cooldown(self) -> None:
        """Wait between jobs and retain before/after pressure samples."""
        self.on_message(f"冷卻 {self.cooldown_seconds:g} 秒並檢查記憶體與 Swapouts")
        first = self.probe()
        if self.cooldown_seconds:
            self.sleep(self.cooldown_seconds)
        second = self.probe()
        self.samples = [first, second]

    def allow_next_job(self, job: dict[str, Any]) -> bool:
        if len(self.samples) < 2:
            self.on_message("先取兩次記憶體狀態樣本，確認可以開始")
            first = self.probe()
            self.sleep(self.poll_seconds)
            self.samples = [first, self.probe()]

        assessment = assess_memory_pressure(
            self.samples,
            low_free_percent=self.low_free_percent,
        )
        rechecks = 0
        while not assessment.allow_start and rechecks < self.max_rechecks:
            reason = "; ".join(assessment.reasons)
            self.on_message(f"資源狀態尚未穩定，等待 {self.poll_seconds:g} 秒後重查：{reason}")
            self.sleep(self.poll_seconds)
            self.samples = [self.samples[-1], self.probe()]
            assessment = assess_memory_pressure(
                self.samples,
                low_free_percent=self.low_free_percent,
            )
            rechecks += 1

        if not assessment.allow_start:
            self.pause_reason = self._reason(assessment)
            self.on_message(f"佇列暫停：{self.pause_reason}")
            return False

        self.on_message(
            f"資源可用：free {self.samples[-1].free_percent:g}%，"
            f"Swapouts 未增加；準備開始 {job['project_path']}"
        )
        return True

    @staticmethod
    def _reason(assessment: MemoryAssessment) -> str:
        return f"Memory pressure gate: {assessment.status}; " + "; ".join(assessment.reasons)


class VideoFactoryQueue:
    """User-facing queue API shared by CLI and the future desktop shell."""

    def __init__(
        self,
        queue_file: str | Path | None = None,
        *,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        probe: Callable[[], MemorySnapshot] = collect_memory_snapshot,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.queue = PersistentJobQueue(queue_file)
        self.on_progress = on_progress
        self._probe = probe
        self._sleep = sleep

    def _emit(self, event: dict[str, Any]) -> None:
        if self.on_progress:
            self.on_progress(event)

    def enqueue_project(
        self,
        project_path: str | Path,
        *,
        max_attempts: int = 2,
        force: bool = False,
    ) -> dict[str, Any]:
        project = Path(project_path).expanduser().resolve()
        if not project.is_dir():
            raise QueueApplicationError(f"Project folder does not exist: {project}")
        if not (project / "job.yaml").is_file():
            raise QueueApplicationError(f"Project needs a finalized job.yaml before it can be queued: {project}")

        story_plan = project / "work" / "analysis" / "story_plan.json"
        edit_plan = project / "work" / "edit-plan" / "edit_plan.json"
        for required in (story_plan, edit_plan):
            if not required.is_file():
                raise QueueApplicationError(
                    f"Prepare the Codex story and edit plans before queueing; missing {required.relative_to(project)}"
                )
            try:
                document = json.loads(required.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise QueueApplicationError(f"Could not read plan {required}: {exc}") from exc
            if not isinstance(document, dict):
                raise QueueApplicationError(f"Plan must contain a JSON object: {required}")
        if not isinstance(json.loads(edit_plan.read_text(encoding="utf-8")).get("timeline"), list):
            raise QueueApplicationError(f"Edit plan has no timeline array: {edit_plan}")

        # Validate the director-owned plan before the unattended local runner
        # accepts it. No media is moved or uploaded by enqueue.
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))
        import video_factory

        validation = video_factory.command_validate_plan(argparse.Namespace(project=str(project)))
        if validation != 0:
            raise QueueApplicationError(f"Edit-plan validation failed for {project}")

        policy = load_product_policy(project)
        queued_path = str(project)
        active = {
            job["status"]
            for job in self.queue.list_jobs()
            if Path(job["project_path"]).resolve() == project
            and job["status"] in {"pending", "running"}
        }
        if active:
            raise QueueApplicationError(f"This project already has a pending or running queue job: {project}")

        job = self.queue.enqueue(
            queued_path,
            max_attempts=max_attempts,
            payload={"mode": policy.mode, "force": bool(force), "execution": "local"},
        )
        self._emit({"type": "queue_job_added", "job": job})
        return job

    def list_jobs(self) -> dict[str, Any]:
        return self.queue.snapshot()

    def pause(self, reason: str = "Paused by user.") -> None:
        self.queue.pause(reason)
        self._emit({"type": "queue_paused", "reason": reason})

    def resume(self) -> None:
        self.queue.resume()
        self._emit({"type": "queue_resumed"})

    def cancel(self, job_id: str) -> str:
        result = self.queue.cancel(job_id)
        self._emit({"type": "queue_job_cancel_requested", "job_id": job_id, "result": result})
        return result

    def reorder(self, pending_job_ids: list[str]) -> None:
        self.queue.reorder(pending_job_ids)
        self._emit({"type": "queue_reordered", "job_ids": pending_job_ids})

    def run(
        self,
        *,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        poll_seconds: float = DEFAULT_RESOURCE_POLL_SECONDS,
        max_rechecks: int = DEFAULT_RESOURCE_RECHECKS,
        max_jobs: int | None = None,
    ) -> RunSummary:
        gate = MemoryPressureGate(
            cooldown_seconds=cooldown_seconds,
            poll_seconds=poll_seconds,
            max_rechecks=max_rechecks,
            probe=self._probe,
            sleep=self._sleep,
            on_message=lambda message: self._emit({"type": "queue_progress", "message": message}),
        )

        def run_one(job: dict[str, Any]) -> None:
            self._run_project(job)

        summary = self.queue.run(
            run_one,
            cooldown=gate.cooldown,
            resource_gate=gate.allow_next_job,
            max_jobs=max_jobs,
        )
        if summary.paused:
            self._emit({
                "type": "queue_paused",
                "reason": summary.pause_reason or gate.pause_reason or "Queue paused.",
            })
        self._emit({
            "type": "queue_finished",
            "summary": {
                "attempts_started": summary.attempts_started,
                "jobs_completed": summary.jobs_completed,
                "jobs_failed": summary.jobs_failed,
                "jobs_cancelled": summary.jobs_cancelled,
                "paused": summary.paused,
                "pause_reason": summary.pause_reason or gate.pause_reason,
            },
        })
        return summary

    def _run_project(self, job: dict[str, Any]) -> None:
        project = Path(job["project_path"]).resolve()
        job_id = str(job["id"])
        log_dir = self.queue.queue_file.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        attempt = int(job.get("attempts") or 1)
        log_path = log_dir / f"{job_id}.attempt-{attempt}.log"
        self._emit({"type": "queue_job_started", "job_id": job_id, "project_path": str(project), "log_path": str(log_path)})

        def cancelled() -> bool:
            return bool(self.queue.get_job(job_id).get("cancel_requested"))

        lock_fd = wait_for_heavy_job_lock(cancelled=cancelled, sleep=self._sleep)
        if lock_fd is None:
            return

        command = [
            sys.executable,
            "-m",
            "video_editor",
            "run",
            str(project),
            "--mode",
            str(job.get("payload", {}).get("mode") or "family"),
            "--gate",
            "AUTO",
            "--approve-review",
            "--no-gpu",
            "--privacy-mode",
            "LOCAL_ONLY",
        ]
        if job.get("payload", {}).get("force"):
            command.append("--force")

        environment = os.environ.copy()
        environment["VIDEO_FACTORY_QUEUE_FILE"] = str(self.queue.queue_file)
        environment[HEAVY_LOCK_FD_ENV] = str(lock_fd)
        environment["PYTHONUNBUFFERED"] = "1"
        process: subprocess.Popen[bytes] | None = None
        process_group_id: int | None = None
        with log_path.open("ab", buffering=0) as log_stream:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=REPOSITORY_ROOT,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    pass_fds=(lock_fd,),
                )
                process_group_id = process.pid
                self.queue.update_runtime(job_id, {
                    "pid": process.pid,
                    "process_group_id": process_group_id,
                    "log_path": str(log_path),
                    "started_at_epoch": time.time(),
                })
                self._emit({"type": "queue_job_process_started", "job_id": job_id, "pid": process.pid})

                while True:
                    if cancelled():
                        self._terminate_process_group(process, process_group_id)
                        self._emit({"type": "queue_job_process_cancelled", "job_id": job_id})
                        return
                    try:
                        return_code = process.wait(timeout=1.0)
                        break
                    except subprocess.TimeoutExpired:
                        continue

                self._ensure_process_group_exited(process_group_id)
                self.queue.update_runtime(job_id, {
                    "pid": None,
                    "process_group_id": None,
                    "finished_at_epoch": time.time(),
                    "exit_code": return_code,
                    "log_path": str(log_path),
                })
                if return_code != 0:
                    raise QueueApplicationError(
                        f"Project pipeline exited with status {return_code}; see {log_path}"
                    )
                self._emit({"type": "queue_job_process_completed", "job_id": job_id, "log_path": str(log_path)})
            except BaseException:
                if process is not None and process_group_id is not None and process.poll() is None:
                    try:
                        self._terminate_process_group(process, process_group_id)
                    except QueueApplicationError:
                        # The termination helper persisted a pause. Keep the
                        # original failure, and let the inherited lock remain
                        # held by any process that has not actually exited.
                        pass
                raise
            finally:
                release_heavy_job(lock_fd)

    def _ensure_process_group_exited(self, process_group_id: int) -> None:
        if self._wait_process_group(process_group_id, timeout=2.0):
            return
        # A successful top-level command that leaves render/transcription
        # descendants behind is still unsafe; clean the group before retrying.
        self._signal_process_group(process_group_id, signal.SIGTERM)
        if not self._wait_process_group(process_group_id, timeout=5.0):
            self._signal_process_group(process_group_id, signal.SIGKILL)
        if not self._wait_process_group(process_group_id, timeout=5.0):
            reason = f"Subprocess group {process_group_id} remained active after TERM/KILL; queue paused."
            self.queue.pause(reason)
            raise QueueApplicationError(reason)
        raise QueueApplicationError(
            f"The pipeline exited but left subprocesses running; they were terminated before the next job."
        )

    def _terminate_process_group(self, process: subprocess.Popen[bytes], process_group_id: int) -> None:
        self._signal_process_group(process_group_id, signal.SIGTERM)
        try:
            process.wait(timeout=15.0)
        except subprocess.TimeoutExpired:
            self._signal_process_group(process_group_id, signal.SIGKILL)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
        if not self._wait_process_group(process_group_id, timeout=5.0):
            self._signal_process_group(process_group_id, signal.SIGKILL)
        group_exited = self._wait_process_group(process_group_id, timeout=5.0)
        process_exited = process.poll() is not None
        if not group_exited or not process_exited:
            reason = f"Subprocess group {process_group_id} could not be confirmed stopped; queue paused."
            self.queue.pause(reason)
            raise QueueApplicationError(reason)

    @staticmethod
    def _signal_process_group(process_group_id: int, signal_number: int) -> None:
        try:
            os.killpg(process_group_id, signal_number)
        except ProcessLookupError:
            pass

    @classmethod
    def _wait_process_group(cls, process_group_id: int, *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                os.killpg(process_group_id, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                return False
            time.sleep(0.2)
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return True
        return False


__all__ = [
    "DEFAULT_COOLDOWN_SECONDS",
    "DEFAULT_RESOURCE_POLL_SECONDS",
    "DEFAULT_RESOURCE_RECHECKS",
    "MemoryPressureGate",
    "QueueApplicationError",
    "VideoFactoryQueue",
]
