"""Durable serial queue for local Video Factory project jobs.

The queue intentionally coordinates work; it does not implement the pipeline.
Callers provide a synchronous callback that runs one project at a time.  The
queue uses separate advisory locks for the single runner and short state
transactions so pause/cancel/reorder requests remain available while a job is
running.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping


QUEUE_FORMAT_VERSION = 1
JOB_STATUSES = {"pending", "running", "completed", "failed", "cancelled"}


class JobQueueError(RuntimeError):
    """Base class for queue errors."""


class QueueAlreadyRunning(JobQueueError):
    """Raised when another process already owns the queue runner lock."""


class QueueStateError(JobQueueError):
    """Raised when the persisted queue cannot be safely read or validated."""


@dataclass(frozen=True)
class RunSummary:
    attempts_started: int
    jobs_completed: int
    jobs_failed: int
    jobs_cancelled: int
    paused: bool
    pause_reason: str | None = None


def default_queue_path() -> Path:
    """Return one persistent queue for this checkout, or an explicit shared path."""
    configured = os.environ.get("VIDEO_FACTORY_QUEUE_FILE")
    if configured:
        return Path(configured).expanduser()
    repository_root = Path(__file__).resolve().parents[1]
    return repository_root / ".video-factory" / "queue.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json_copy(value: Any) -> Any:
    """Validate that caller-owned metadata can be persisted as JSON."""
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Queue metadata must be JSON serializable: {exc}") from exc


class PersistentJobQueue:
    """Atomic JSON job queue with one cross-process runner and resumable jobs."""

    def __init__(self, queue_file: str | Path | None = None):
        self.queue_file = Path(queue_file or default_queue_path()).expanduser().resolve()
        self.queue_file.parent.mkdir(parents=True, exist_ok=True)
        self._state_lock_file = self.queue_file.with_suffix(self.queue_file.suffix + ".state.lock")
        self._runner_lock_file = self.queue_file.with_suffix(self.queue_file.suffix + ".runner.lock")
        with self._state_lock():
            if not self.queue_file.exists():
                self._write_state(self._empty_state())
            else:
                self._read_state()

    @staticmethod
    def _empty_state() -> dict[str, Any]:
        return {
            "format_version": QUEUE_FORMAT_VERSION,
            "paused": False,
            "pause_reason": None,
            "jobs": [],
            "updated_at": _now(),
        }

    @contextmanager
    def _state_lock(self) -> Iterator[None]:
        """Serialize short JSON transactions without blocking UI controls on a job."""
        descriptor = os.open(self._state_lock_file, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @contextmanager
    def _runner_lock(self) -> Iterator[None]:
        descriptor = os.open(self._runner_lock_file, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise QueueAlreadyRunning("A Video Factory queue runner is already active.") from exc
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    def _read_state(self) -> dict[str, Any]:
        try:
            state = json.loads(self.queue_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise QueueStateError(f"Could not read queue state at {self.queue_file}: {exc}") from exc
        if not isinstance(state, dict) or state.get("format_version") != QUEUE_FORMAT_VERSION:
            raise QueueStateError(f"Unsupported or invalid queue state: {self.queue_file}")
        if not isinstance(state.get("jobs"), list):
            raise QueueStateError(f"Queue state has no jobs list: {self.queue_file}")
        for job in state["jobs"]:
            if not isinstance(job, dict) or job.get("status") not in JOB_STATUSES:
                raise QueueStateError(f"Queue contains an invalid job record: {self.queue_file}")
        return state

    def _write_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = _now()
        data = json.dumps(state, ensure_ascii=False, indent=2) + "\n"
        descriptor, temp_name = tempfile.mkstemp(prefix=f".{self.queue_file.name}.", suffix=".tmp", dir=self.queue_file.parent)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                os.fchmod(stream.fileno(), 0o600)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_path, self.queue_file)
            directory_fd = os.open(self.queue_file.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass

    def _transaction(self, update: Callable[[dict[str, Any]], Any]) -> Any:
        with self._state_lock():
            state = self._read_state()
            result = update(state)
            self._write_state(state)
            return result

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock():
            return copy.deepcopy(self._read_state())

    def list_jobs(self) -> list[dict[str, Any]]:
        return self.snapshot()["jobs"]

    def enqueue(
        self,
        project_path: str | Path,
        *,
        max_attempts: int = 2,
        payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be an integer greater than zero")
        job = {
            "id": uuid.uuid4().hex,
            "project_path": str(Path(project_path).expanduser().resolve()),
            "status": "pending",
            "attempts": 0,
            "max_attempts": max_attempts,
            "created_at": _now(),
            "updated_at": _now(),
            "started_at": None,
            "completed_at": None,
            "last_error": None,
            "attempt_errors": [],
            "cancel_requested": False,
            "payload": _json_copy(dict(payload or {})),
            "checkpoint": {},
            "runtime": {},
        }

        def add(state: dict[str, Any]) -> dict[str, Any]:
            state["jobs"].append(job)
            return copy.deepcopy(job)

        return self._transaction(add)

    def pause(self, reason: str = "Paused by user.") -> None:
        def update(state: dict[str, Any]) -> None:
            state["paused"] = True
            state["pause_reason"] = str(reason)

        self._transaction(update)

    def resume(self) -> None:
        def update(state: dict[str, Any]) -> None:
            state["paused"] = False
            state["pause_reason"] = None

        self._transaction(update)

    def cancel(self, job_id: str) -> str:
        """Cancel queued work or request cancellation after an active callback.

        A running media pipeline is not killed forcibly. It is allowed to reach
        its own safe stop, then the queue records the requested cancellation.
        """
        def update(state: dict[str, Any]) -> str:
            job = self._find_job(state, job_id)
            if job["status"] == "running":
                job["cancel_requested"] = True
                job["updated_at"] = _now()
                return "cancellation_requested"
            if job["status"] == "pending":
                job["status"] = "cancelled"
                job["completed_at"] = _now()
                job["updated_at"] = _now()
                return "cancelled"
            return str(job["status"])

        return self._transaction(update)

    def reorder(self, pending_job_ids: list[str]) -> None:
        """Set the exact order of all pending jobs; other states stay in place."""
        def update(state: dict[str, Any]) -> None:
            pending = [job for job in state["jobs"] if job["status"] == "pending"]
            current_ids = {job["id"] for job in pending}
            if len(pending_job_ids) != len(current_ids) or set(pending_job_ids) != current_ids:
                raise ValueError("reorder requires every pending job ID exactly once")
            by_id = {job["id"]: job for job in pending}
            iterator = iter(pending_job_ids)
            state["jobs"] = [by_id[next(iterator)] if job["status"] == "pending" else job for job in state["jobs"]]

        self._transaction(update)

    def update_checkpoint(self, job_id: str, checkpoint: Mapping[str, Any]) -> None:
        """Persist a pipeline-provided resume marker while a job is running."""
        copied = _json_copy(dict(checkpoint))

        def update(state: dict[str, Any]) -> None:
            job = self._find_job(state, job_id)
            job["checkpoint"] = copied
            job["updated_at"] = _now()

        self._transaction(update)

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Return the latest persisted job record, including cancel requests."""
        state = self.snapshot()
        return copy.deepcopy(self._find_job(state, job_id))

    def update_runtime(self, job_id: str, fields: Mapping[str, Any]) -> None:
        """Merge child-process runtime metadata without changing queue status.

        Queue integrations can persist a child PID, process-group ID, log paths,
        or other JSON-safe supervisor metadata here. The process supervisor
        should poll :meth:`get_job` for a cooperative cancellation request,
        terminate and reap its process group, then return from the callback.
        """
        copied = _json_copy(dict(fields))

        def update(state: dict[str, Any]) -> None:
            job = self._find_job(state, job_id)
            runtime = job.setdefault("runtime", {})
            runtime.update(copied)
            job["updated_at"] = _now()

        self._transaction(update)

    @staticmethod
    def _find_job(state: dict[str, Any], job_id: str) -> dict[str, Any]:
        for job in state["jobs"]:
            if job["id"] == job_id:
                return job
        raise KeyError(f"Unknown queue job: {job_id}")

    def _recover_interrupted(self) -> None:
        def update(state: dict[str, Any]) -> None:
            recovered: list[dict[str, Any]] = []
            for job in list(state["jobs"]):
                if job["status"] != "running":
                    continue
                error = "Runner stopped while this job was running; retrying from the project's own checkpoint."
                job["last_error"] = error
                job["attempt_errors"].append({"attempt": job["attempts"], "error": error, "at": _now()})
                if job["attempts"] >= job["max_attempts"]:
                    job["status"] = "failed"
                    job["completed_at"] = _now()
                else:
                    job["status"] = "pending"
                    job["cancel_requested"] = False
                    recovered.append(job)
                job["updated_at"] = _now()
            # Put interrupted jobs after projects which were already waiting.
            if recovered:
                recovered_ids = {job["id"] for job in recovered}
                state["jobs"] = [job for job in state["jobs"] if job["id"] not in recovered_ids] + recovered

        self._transaction(update)

    def run(
        self,
        callback: Callable[[dict[str, Any]], Any],
        *,
        cooldown: Callable[[], None] | None = None,
        resource_gate: Callable[[dict[str, Any]], bool] | None = None,
        max_jobs: int | None = None,
    ) -> RunSummary:
        """Run pending jobs serially until idle, paused, gated, or max_jobs.

        `resource_gate` runs before each job and returns false to pause the queue.
        `cooldown` runs only between jobs. Exceptions from a project callback
        are recorded and retried up to the job limit, then the next job runs.
        """
        if max_jobs is not None and (isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < 1):
            raise ValueError("max_jobs must be an integer greater than zero when provided")
        attempts_started = completed = failed = cancelled = 0
        pause_reason: str | None = None
        with self._runner_lock():
            self._recover_interrupted()
            while max_jobs is None or attempts_started < max_jobs:
                state = self.snapshot()
                if state["paused"]:
                    pause_reason = state.get("pause_reason")
                    break
                job = next((copy.deepcopy(item) for item in state["jobs"] if item["status"] == "pending"), None)
                if job is None:
                    break
                if job.get("cancel_requested"):
                    self.cancel(job["id"])
                    cancelled += 1
                    continue
                if resource_gate is not None:
                    try:
                        allowed = bool(resource_gate(copy.deepcopy(job)))
                    except Exception as exc:
                        allowed = False
                        pause_reason = f"Resource gate failed: {type(exc).__name__}: {exc}"
                    if not allowed:
                        self.pause(pause_reason or "System resource gate requested a pause.")
                        pause_reason = self.snapshot().get("pause_reason")
                        break

                def mark_running(current: dict[str, Any]) -> bool:
                    if current["paused"]:
                        return False
                    selected = self._find_job(current, job["id"])
                    if selected["status"] != "pending":
                        return False
                    selected["status"] = "running"
                    selected["attempts"] += 1
                    selected["started_at"] = _now()
                    selected["updated_at"] = _now()
                    return True

                if not self._transaction(mark_running):
                    continue
                attempts_started += 1
                error_text: str | None = None
                try:
                    callback(self._find_job(self.snapshot(), job["id"]))
                except Exception as exc:
                    error_text = f"{type(exc).__name__}: {exc}"

                def finish(current: dict[str, Any]) -> str:
                    selected = self._find_job(current, job["id"])
                    selected["updated_at"] = _now()
                    if selected.get("cancel_requested"):
                        selected["status"] = "cancelled"
                        selected["completed_at"] = _now()
                        return "cancelled"
                    if error_text is None:
                        selected["status"] = "completed"
                        selected["completed_at"] = _now()
                        selected["last_error"] = None
                        return "completed"
                    selected["last_error"] = error_text
                    selected["attempt_errors"].append({"attempt": selected["attempts"], "error": error_text, "at": _now()})
                    if selected["attempts"] >= selected["max_attempts"]:
                        selected["status"] = "failed"
                        selected["completed_at"] = _now()
                        return "failed"
                    selected["status"] = "pending"
                    # Let already-queued projects run before this retry.
                    current["jobs"].remove(selected)
                    current["jobs"].append(selected)
                    return "retry_pending"

                result = self._transaction(finish)
                if result == "completed":
                    completed += 1
                elif result == "failed":
                    failed += 1
                elif result == "cancelled":
                    cancelled += 1

                state = self.snapshot()
                has_more = any(item["status"] == "pending" for item in state["jobs"])
                if has_more and not state["paused"] and cooldown is not None:
                    try:
                        cooldown()
                    except Exception as exc:
                        pause_reason = f"Cooldown failed: {type(exc).__name__}: {exc}"
                        self.pause(pause_reason)
                        break
            final_state = self.snapshot()
            if pause_reason is None:
                pause_reason = final_state.get("pause_reason") if final_state.get("paused") else None
            return RunSummary(
                attempts_started=attempts_started,
                jobs_completed=completed,
                jobs_failed=failed,
                jobs_cancelled=cancelled,
                paused=bool(final_state["paused"]),
                pause_reason=pause_reason,
            )


__all__ = [
    "JOB_STATUSES",
    "JobQueueError",
    "PersistentJobQueue",
    "QueueAlreadyRunning",
    "QueueStateError",
    "RunSummary",
    "default_queue_path",
]
