"""Read-only macOS memory probes for conservative local-workload gating.

The probe runs ``/usr/bin/memory_pressure`` without arguments that alter system
pressure. A queue should take two snapshots separated by its normal cooldown
before starting another heavy job. Starting is allowed only when both readings
are complete, free memory is at or above the configured threshold in both, and
the swapout counter did not increase. A counter increase or a currently low
free-memory reading blocks the next job; a low first reading followed by a
healthy second reading is treated as inconclusive and also blocks until another
pair of readings is taken. Missing free-percent or swapout data is ``unknown``
and fails closed. Swapins are retained as diagnostic telemetry but do not gate
work because they do not by themselves indicate current memory pressure.

The default low-free threshold is 10 percent. This is intentionally a simple,
conservative heuristic, not a claim about Apple's internal memory-pressure
state. The module does not sleep, allocate memory, or add dependencies.
"""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Sequence


MEMORY_PRESSURE_COMMAND = "/usr/bin/memory_pressure"
DEFAULT_LOW_FREE_PERCENT = 10.0

_FREE_PERCENT_RE = re.compile(
    r"System-wide\s+memory\s+free\s+percentage:\s*(\d+(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)
_SWAPINS_RE = re.compile(r"^\s*Swapins:\s*([\d,]+)\s*$", re.IGNORECASE | re.MULTILINE)
_SWAPOUTS_RE = re.compile(r"^\s*Swapouts:\s*([\d,]+)\s*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class MemorySnapshot:
    """One read-only memory-pressure sample; absent readings remain explicit."""

    free_percent: float | None
    swapins: int | None
    swapouts: int | None
    observed_at: str
    error: str | None = None

    @property
    def complete(self) -> bool:
        """Whether the readings required for safe queue gating are present."""
        return self.free_percent is not None and self.swapouts is not None and self.error is None


@dataclass(frozen=True)
class MemoryAssessment:
    """Decision derived from the latest two samples."""

    status: str  # "ready", "pressure", or "unknown"
    allow_start: bool
    reasons: tuple[str, ...]
    swapout_delta: int | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_counter(pattern: re.Pattern[str], text: str) -> int | None:
    match = pattern.search(text)
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


def parse_memory_pressure(text: str, *, observed_at: str | None = None) -> MemorySnapshot:
    """Parse ``memory_pressure`` output without invoking a subprocess.

    This pure parser keeps partial readings for diagnostics while marking any
    absent or malformed critical value as incomplete, so callers can fail
    closed rather than treating an unrecognized output as healthy.
    """
    timestamp = observed_at or _now_iso()
    free_match = _FREE_PERCENT_RE.search(text)
    free_percent: float | None = None
    if free_match:
        try:
            candidate = float(free_match.group(1))
            if 0.0 <= candidate <= 100.0:
                free_percent = candidate
        except ValueError:
            pass

    swapins = _parse_counter(_SWAPINS_RE, text)
    swapouts = _parse_counter(_SWAPOUTS_RE, text)
    missing = []
    if free_percent is None:
        missing.append("system-wide free-memory percentage")
    if swapouts is None:
        missing.append("swapouts counter")

    return MemorySnapshot(
        free_percent=free_percent,
        swapins=swapins,
        swapouts=swapouts,
        observed_at=timestamp,
        error=("Missing or invalid " + " and ".join(missing)) if missing else None,
    )


def collect_memory_snapshot(
    *,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    system_name: str | None = None,
    observed_at: str | None = None,
    timeout_seconds: float = 5.0,
) -> MemorySnapshot:
    """Read one memory snapshot, returning an explicit unavailable result on error."""
    timestamp = observed_at or _now_iso()
    current_system = system_name or platform.system()
    if current_system != "Darwin":
        return MemorySnapshot(None, None, None, timestamp, "memory_pressure is available only on macOS")

    try:
        result = run(
            [MEMORY_PRESSURE_COMMAND],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return MemorySnapshot(None, None, None, timestamp, f"Could not run memory_pressure: {exc}")

    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        return MemorySnapshot(
            None,
            None,
            None,
            timestamp,
            f"memory_pressure exited with status {result.returncode}" + (f": {detail}" if detail else ""),
        )

    parsed = parse_memory_pressure(result.stdout or "", observed_at=timestamp)
    return parsed


def assess_memory_pressure(
    samples: Sequence[MemorySnapshot],
    *,
    low_free_percent: float = DEFAULT_LOW_FREE_PERCENT,
) -> MemoryAssessment:
    """Assess whether another heavy job may start from the latest two samples.

    The threshold is configurable for testing and machine-specific policy. A
    rising swapout counter blocks immediately. Current low free memory blocks;
    a single low first sample followed by recovery is inconclusive and requires
    a fresh pair rather than being called safe. Any missing critical reading or
    fewer than two samples returns ``unknown`` and ``allow_start=False``.
    """
    if isinstance(low_free_percent, bool) or not isinstance(low_free_percent, (int, float)):
        raise ValueError("low_free_percent must be a number from 0 through 100")
    if not 0.0 <= float(low_free_percent) <= 100.0:
        raise ValueError("low_free_percent must be a number from 0 through 100")
    if len(samples) < 2:
        return MemoryAssessment("unknown", False, ("Two memory samples are required",))

    previous, current = samples[-2:]
    if not previous.complete or not current.complete:
        details = tuple(
            f"Sample {index} is incomplete: {sample.error or 'critical reading unavailable'}"
            for index, sample in enumerate((previous, current), start=1)
            if not sample.complete
        )
        return MemoryAssessment("unknown", False, details)

    assert previous.free_percent is not None and current.free_percent is not None
    assert previous.swapouts is not None and current.swapouts is not None
    delta = current.swapouts - previous.swapouts
    if delta < 0:
        return MemoryAssessment(
            "unknown",
            False,
            ("Swapouts counter decreased; its baseline may have reset",),
            swapout_delta=delta,
        )

    reasons: list[str] = []
    if delta > 0:
        reasons.append(f"Swapouts increased by {delta} between samples")
    if current.free_percent < float(low_free_percent):
        reasons.append(
            f"Current free memory is {current.free_percent:g}%, below the "
            f"{float(low_free_percent):g}% threshold"
        )
    elif previous.free_percent < float(low_free_percent):
        return MemoryAssessment(
            "unknown",
            False,
            ("Free-memory readings changed from below threshold to above it; take another pair of samples",),
            swapout_delta=delta,
        )

    if reasons:
        return MemoryAssessment("pressure", False, tuple(reasons), swapout_delta=delta)
    return MemoryAssessment("ready", True, ("Both readings are above the free-memory threshold and swapouts are stable",), swapout_delta=delta)
