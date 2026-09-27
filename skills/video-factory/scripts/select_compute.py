"""Compute selector and GPU cost guardrails for Video Factory."""

from __future__ import annotations

import argparse
import sys
from typing import Literal

ComputeDevice = Literal["cpu", "t4", "l4"]
PROHIBITED_GPUS = {"a100", "h100", "g4", "v100", "l40"}
ALLOWED_GPUS = {"T4", "L4"}

TASK_COMPUTE_MAPPING = {
    "probe": "cpu",
    "ingest": "cpu",
    "proxy": "cpu",
    "scenes": "cpu",
    "contact_sheet": "cpu",
    "validate": "cpu",
    "render": "cpu",
    "qa": "cpu",
    "transcribe": "t4",
    "embeddings": "t4",
    "perception": "t4",
    "temporal": "t4",
    "temporal_analysis": "t4",
    "deep_analysis": "t4",
}


class ComputeSelectionError(Exception):
    """Raised when an invalid or disallowed compute accelerator is requested."""


def select_compute_for_task(
    task: str,
    requested_gpu: str | None = None,
    allow_premium: bool = False,
    estimated_audio_duration_minutes: float = 0.0,
) -> str:
    """Select the appropriate compute resource for a task following the cost hierarchy:

    1. Local CPU for deterministic, media decode/encode, and metadata tasks.
    2. Colab T4 by default for AI inference (faster-whisper, embeddings).
    3. Colab L4 only when explicitly requested, or for huge workloads (> 120 mins audio) with proven benefit.
    4. A100 / H100 / G4 are strictly forbidden unless allow_premium is explicitly True.
    """
    task_normalized = task.lower().strip()
    default_target = TASK_COMPUTE_MAPPING.get(task_normalized, "cpu")

    if default_target == "cpu" and not requested_gpu:
        return "cpu"

    if requested_gpu:
        normalized = requested_gpu.strip().upper()
        lower = normalized.lower()

        if lower in PROHIBITED_GPUS:
            if not allow_premium:
                raise ComputeSelectionError(
                    f"Automatic allocation of premium GPU {normalized} is prohibited by cost-control policy. "
                    "Use T4 (default) or L4, or explicitly set allow_premium=True with user consent."
                )
            return normalized

        if normalized not in ALLOWED_GPUS:
            raise ComputeSelectionError(
                f"Unsupported accelerator: {requested_gpu}. Allowed accelerators: T4, L4 (or CPU for local tasks)."
            )

        if normalized == "L4" and task_normalized not in {"temporal", "temporal_analysis", "deep_analysis"}:
            raise ComputeSelectionError(
                "L4 is reserved for shortlist-only temporal analysis or a measured VRAM requirement. "
                "Routine speech, embeddings and clustering use the default T4."
            )
        return normalized

    # Auto selection for AI tasks
    if task_normalized in {"transcribe", "embeddings", "perception"}:
        return "T4"

    return "cpu"


def explain_compute_plan(task: str, accelerator: str) -> dict[str, str]:
    """Return a human-readable explanation of why this compute was chosen."""
    accel = accelerator.upper()
    if accel == "CPU":
        return {
            "accelerator": "CPU",
            "tier": "Local Mac mini M4",
            "reason": f"Task '{task}' runs most efficiently and cost-effectively on local CPU.",
            "cost": "0 Colab Compute Units",
        }
    if accel == "T4":
        return {
            "accelerator": "T4",
            "tier": "Colab Standard GPU (Default)",
            "reason": f"Task '{task}' requires CUDA GPU inference; T4 provides the best cost/performance balance.",
            "cost": "~1.5-2.0 Compute Units/hr",
        }
    if accel == "L4":
        return {
            "accelerator": "L4",
            "tier": "Colab High-Performance GPU",
            "reason": f"Task '{task}' requires higher VRAM or accelerated inference.",
            "cost": "~4.0-5.0 Compute Units/hr",
        }
    return {
        "accelerator": accel,
        "tier": "Premium GPU",
        "reason": "Explicitly requested override.",
        "cost": "High (> 10 Compute Units/hr)",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Video Factory compute selector")
    parser.add_argument("task", help="task name (e.g. transcribe, proxy, scenes, render)")
    parser.add_argument("--gpu", default=None, help="requested GPU (T4, L4, auto)")
    parser.add_argument("--allow-premium", action="store_true", help="explicitly allow A100/H100/G4")
    parser.add_argument("--duration-min", type=float, default=0.0, help="estimated audio minutes")
    args = parser.parse_args()

    try:
        gpu_arg = None if args.gpu == "auto" else args.gpu
        selected = select_compute_for_task(
            args.task,
            requested_gpu=gpu_arg,
            allow_premium=args.allow_premium,
            estimated_audio_duration_minutes=args.duration_min,
        )
        plan = explain_compute_plan(args.task, selected)
        print(f"Selected Compute: {plan['accelerator']} ({plan['tier']})")
        print(f"Reason: {plan['reason']}")
        print(f"Estimated Cost: {plan['cost']}")
        return 0
    except ComputeSelectionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
