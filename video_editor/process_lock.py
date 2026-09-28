"""Cross-process lock that serializes memory-intensive Video Factory work."""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

from video_editor.job_queue import default_queue_path


HEAVY_LOCK_FD_ENV = "VIDEO_FACTORY_HEAVY_LOCK_FD"


class HeavyJobBusy(RuntimeError):
    """Another local Video Factory media job currently owns the heavy lock."""


def heavy_job_lock_path() -> Path:
    return default_queue_path().with_name("heavy-job.lock")


def try_acquire_heavy_job() -> int | None:
    """Try to acquire the shared lock, returning its descriptor when successful."""
    path = heavy_job_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    return descriptor


def release_heavy_job(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def inherited_heavy_lock_fd() -> int | None:
    """Return a verified lock descriptor passed from the queue supervisor."""
    raw = os.environ.get(HEAVY_LOCK_FD_ENV)
    if raw is None:
        return None
    try:
        descriptor = int(raw)
        held = os.fstat(descriptor)
        expected = heavy_job_lock_path().stat()
    except (OSError, ValueError) as exc:
        raise HeavyJobBusy("Inherited Video Factory lock is missing or invalid.") from exc
    if (held.st_dev, held.st_ino) != (expected.st_dev, expected.st_ino):
        raise HeavyJobBusy("Inherited Video Factory lock does not match this workspace queue.")
    return descriptor


@contextmanager
def exclusive_heavy_job() -> Iterator[int]:
    """Acquire the workspace lock, or use a verified lock inherited from a queue.

    Queue subprocesses receive the already-held descriptor with ``pass_fds``.
    They must not unlock it; the supervising process releases its descriptor
    only after the child process tree has exited.
    """
    inherited = inherited_heavy_lock_fd()
    if inherited is not None:
        yield inherited
        return

    descriptor = try_acquire_heavy_job()
    if descriptor is None:
        raise HeavyJobBusy(
            "Another Video Factory media job is running. Add this project with "
            "`python -m video_editor queue add PROJECT` and run the serial queue."
        )
    try:
        yield descriptor
    finally:
        release_heavy_job(descriptor)


def wait_for_heavy_job_lock(
    *,
    cancelled: Callable[[], bool],
    poll_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> int | None:
    """Wait for a running manual media command to finish; return None if cancelled."""
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")
    while True:
        descriptor = try_acquire_heavy_job()
        if descriptor is not None:
            return descriptor
        if cancelled():
            return None
        sleep(poll_seconds)


__all__ = [
    "HEAVY_LOCK_FD_ENV",
    "HeavyJobBusy",
    "exclusive_heavy_job",
    "heavy_job_lock_path",
    "inherited_heavy_lock_fd",
    "release_heavy_job",
    "try_acquire_heavy_job",
    "wait_for_heavy_job_lock",
]
