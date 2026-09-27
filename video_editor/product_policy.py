"""Product-level policy shared by the CLI, GUI API and pipeline.

The media workers remain deterministic and local.  This module only resolves
the user-facing choices that a project may make: profile, duration preset,
privacy mode, perception policy and music policy.  It deliberately uses a
small YAML reader for the scalar fields we own so the standard-library CLI
does not acquire a hidden dependency on a YAML implementation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PRIVACY_MODES = {"LOCAL_ONLY", "BALANCED", "MAX_QUALITY"}
GPU_POLICIES = {"T4", "L4"}
MUSIC_MODES = {"auto_open_licensed", "project_music", "local_licensed", "none"}

MODE_ALIASES = {
    "memory": "family",
    "family": "family",
    "public-health": "health_education",
    "health_education": "health_education",
}

DURATION_PRESETS: dict[str, dict[str, dict[str, float]]] = {
    "family": {
        "short": {"target": 90.0, "min": 75.0, "max": 105.0},
        "standard": {"target": 180.0, "min": 150.0, "max": 210.0},
        "full": {"target": 300.0, "min": 250.0, "max": 330.0},
    },
    "health_education": {
        "short": {"target": 30.0, "min": 25.0, "max": 40.0},
        "standard": {"target": 90.0, "min": 75.0, "max": 105.0},
        "full": {"target": 180.0, "min": 150.0, "max": 210.0},
    },
}


def canonical_mode(value: str | None) -> str:
    candidate = (value or "family").strip().lower()
    try:
        return MODE_ALIASES[candidate]
    except KeyError as exc:
        raise ValueError(f"Unsupported editing mode: {value!r}") from exc


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _strip_comment(value: str) -> str:
    # Job values in this project do not use escaped '#' inside scalars.
    return value.split("#", 1)[0].strip()


def read_job_text(project: Path) -> str:
    path = project / "job.yaml"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    except OSError as exc:
        raise ValueError(f"Could not read {path}: {exc}") from exc


def top_level_scalar(text: str, key: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(key)}\s*:\s*(.*?)\s*$")
    for raw in text.splitlines():
        if raw[:1].isspace() or raw.lstrip().startswith("#"):
            continue
        match = pattern.match(_strip_comment(raw))
        if match:
            value = _unquote(match.group(1))
            return value or None
    return None


def nested_scalar(text: str, section: str, key: str) -> str | None:
    in_section = False
    section_indent = 0
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        clean = _strip_comment(raw.strip())
        if indent == 0:
            in_section = clean == f"{section}:"
            section_indent = indent
            continue
        if in_section and indent <= section_indent:
            in_section = False
        if in_section:
            match = re.fullmatch(rf"{re.escape(key)}\s*:\s*(.*?)", clean)
            if match:
                value = _unquote(match.group(1))
                return value or None
    return None


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DurationPolicy:
    mode: str
    preset: str
    target_seconds: float
    minimum_seconds: float
    maximum_seconds: float
    custom: bool = False


@dataclass(frozen=True)
class ProductPolicy:
    mode: str
    profile: str
    duration: DurationPolicy
    privacy_mode: str
    gpu_policy: str
    temporal_backend: str
    cloud_perception: bool
    music_mode: str
    music_license_policy: str
    preserve_natural_audio: bool


def resolve_duration(
    mode: str,
    preset: str | None = None,
    custom_seconds: float | None = None,
) -> DurationPolicy:
    canonical = canonical_mode(mode)
    if custom_seconds is not None:
        value = float(custom_seconds)
        if value <= 0:
            raise ValueError("Custom duration must be greater than zero.")
        return DurationPolicy(canonical, "custom", value, value, value, custom=True)
    selected = (preset or "standard").strip().lower()
    if selected not in DURATION_PRESETS[canonical]:
        raise ValueError(f"Unsupported duration preset {preset!r} for {canonical}.")
    values = DURATION_PRESETS[canonical][selected]
    return DurationPolicy(canonical, selected, values["target"], values["min"], values["max"])


def load_product_policy(
    project: Path,
    mode: str | None = None,
    duration_preset: str | None = None,
    custom_duration_seconds: float | None = None,
    privacy_mode: str | None = None,
    gpu_policy: str | None = None,
    cloud_perception: bool | None = None,
    music_mode: str | None = None,
) -> ProductPolicy:
    text = read_job_text(project)
    requested_mode = mode or top_level_scalar(text, "profile") or "family"
    canonical = canonical_mode(requested_mode)
    # Existing projects may still use target_duration_seconds.  Treat it as a
    # deliberate custom duration and preserve the smoke-test project exactly.
    configured_duration = top_level_scalar(text, "target_duration_seconds")
    configured_preset = duration_preset or top_level_scalar(text, "duration_preset")
    custom = custom_duration_seconds
    if custom is None and configured_duration and (configured_preset is None or configured_preset.strip().lower() == "custom"):
        try:
            custom = float(configured_duration)
        except ValueError:
            raise ValueError("job.yaml target_duration_seconds must be numeric.") from None
    duration = resolve_duration(canonical, configured_preset, custom)

    selected_privacy = (privacy_mode or top_level_scalar(text, "privacy_mode") or "BALANCED").upper()
    if selected_privacy not in PRIVACY_MODES:
        raise ValueError(f"privacy_mode must be one of {sorted(PRIVACY_MODES)}.")
    selected_gpu = (gpu_policy or top_level_scalar(text, "gpu_policy") or "T4").upper()
    if selected_gpu not in GPU_POLICIES:
        raise ValueError("gpu_policy must be T4 or L4; premium accelerators are not allowed by policy.")
    backend = top_level_scalar(text, "temporal_backend") or "none"
    selected_music_mode = (music_mode or nested_scalar(text, "music", "mode") or top_level_scalar(text, "music_mode") or "auto_open_licensed").strip().lower()
    if selected_music_mode not in MUSIC_MODES:
        raise ValueError(f"music mode must be one of {sorted(MUSIC_MODES)}.")
    music_policy = top_level_scalar(text, "music_license_policy") or "SAFE_AUTO"
    music_enabled = nested_scalar(text, "music", "enabled")
    preserve_natural = nested_scalar(text, "natural_audio", "preserve")
    selected_cloud = cloud_perception if cloud_perception is not None else _as_bool(top_level_scalar(text, "cloud_perception"), False)
    return ProductPolicy(
        mode=canonical,
        profile=requested_mode,
        duration=duration,
        privacy_mode=selected_privacy,
        gpu_policy=selected_gpu,
        temporal_backend=backend,
        cloud_perception=selected_cloud,
        music_mode="none" if music_enabled == "false" else selected_music_mode,
        music_license_policy=music_policy,
        preserve_natural_audio=_as_bool(preserve_natural, canonical == "family"),
    )


def duration_summary(policy: DurationPolicy, actual_seconds: float | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "preset": policy.preset,
        "target_seconds": policy.target_seconds,
        "minimum_seconds": policy.minimum_seconds,
        "maximum_seconds": policy.maximum_seconds,
        "custom": policy.custom,
    }
    if actual_seconds is not None:
        result["actual_seconds"] = round(float(actual_seconds), 3)
        result["within_range"] = policy.minimum_seconds <= float(actual_seconds) <= policy.maximum_seconds
    return result


__all__ = [
    "DURATION_PRESETS",
    "DurationPolicy",
    "PRIVACY_MODES",
    "ProductPolicy",
    "canonical_mode",
    "duration_summary",
    "load_product_policy",
    "resolve_duration",
]
