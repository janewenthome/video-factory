"""Proxy-first workflow helper for Video Factory.

Generates 720p low-bitrate proxies for video assets using Apple VideoToolbox
hardware encoding on Mac (with CPU libx264 fallback). Downstream AI analysis
(scene detection, frame sampling, contact sheets) reads from proxies instead
of 4K originals, saving storage, upload bandwidth, and GPU compute units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from video_factory import (
    MANIFEST_RELATIVE_PATH,
    UserFacingError,
    is_within,
    load_json,
    project_path,
    resolve_project,
    sha256_file,
    utc_now,
    work_path,
    write_json,
)

PROXIES_MANIFEST_RELATIVE = Path("proxies/manifest.json")
DEFAULT_TARGET_HEIGHT = 720
DEFAULT_BITRATE = "1500k"


def require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise UserFacingError("ffmpeg is not installed or not on PATH.")
    return ffmpeg


def check_videotoolbox_available(ffmpeg: str) -> bool:
    try:
        res = subprocess.run([ffmpeg, "-encoders"], capture_output=True, text=True, check=False)
        return "h264_videotoolbox" in res.stdout
    except OSError:
        return False


def create_proxy_for_video(
    source_path: Path,
    output_path: Path,
    target_height: int = DEFAULT_TARGET_HEIGHT,
    bitrate: str = DEFAULT_BITRATE,
) -> tuple[bool, str]:
    ffmpeg = require_ffmpeg()
    use_videotoolbox = check_videotoolbox_available(ffmpeg)

    # Scale keeping aspect ratio, width divisible by 2, max height = target_height
    scale_filter = f"scale=-2:'min({target_height},ih)':force_original_aspect_ratio=decrease:force_divisible_by=2"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = output_path.parent
    try:
        descriptor, temp_file = tempfile.mkstemp(
            prefix=f".{output_path.stem}.", suffix=".mp4", dir=str(temp_dir)
        )
        os.close(descriptor)
    except OSError as exc:
        return False, f"Could not create temporary proxy file: {exc}"

    def build_command(encoder: str) -> list[str]:
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source_path),
            "-vf", scale_filter,
            "-c:v", encoder,
        ]
        if encoder == "h264_videotoolbox":
            command += ["-b:v", bitrate, "-allow_sw", "1"]
        else:
            command += ["-preset", "fast", "-crf", "28"]
        # Simple AAC audio track at 96k for proxies.
        return command + ["-c:a", "aac", "-b:a", "96k", "-ac", "2", str(temp_file)]

    command = build_command("h264_videotoolbox" if use_videotoolbox else "libx264")

    try:
        res = subprocess.run(command, capture_output=True, text=True, timeout=1200, check=False)
        if res.returncode != 0 and use_videotoolbox:
            # FFmpeg may list VideoToolbox on a non-Mac host or when the
            # hardware encoder is temporarily unavailable. Retry locally with
            # libx264 before reporting a real proxy failure.
            Path(temp_file).unlink(missing_ok=True)
            res = subprocess.run(build_command("libx264"), capture_output=True, text=True, timeout=1200, check=False)
        if res.returncode != 0:
            Path(temp_file).unlink(missing_ok=True)
            return False, res.stderr.strip() or f"ffmpeg exited with {res.returncode}"

        if Path(temp_file).stat().st_size == 0:
            Path(temp_file).unlink(missing_ok=True)
            return False, "Generated proxy was 0 bytes"

        os.replace(temp_file, output_path)
        return True, ""
    except subprocess.TimeoutExpired:
        Path(temp_file).unlink(missing_ok=True)
        return False, "Proxy generation timed out"
    except OSError as exc:
        Path(temp_file).unlink(missing_ok=True)
        return False, str(exc)


def build_proxies_for_project(
    project_dir: Path,
    target_height: int = DEFAULT_TARGET_HEIGHT,
    force: bool = False,
) -> dict[str, Any]:
    project = resolve_project(project_dir)
    manifest_file = project_path(project, MANIFEST_RELATIVE_PATH)
    if not manifest_file.is_file():
        raise UserFacingError("Media manifest is missing. Run `inspect` first.")

    manifest_data = load_json(manifest_file, "media manifest")
    assets = manifest_data.get("assets", []) if isinstance(manifest_data, dict) else []
    video_assets = [a for a in assets if isinstance(a, dict) and a.get("kind") == "video"]

    proxies_index_file = work_path(project, PROXIES_MANIFEST_RELATIVE, create_dir=False)
    existing_proxies: dict[str, Any] = {}
    if proxies_index_file.is_file() and not force:
        try:
            loaded = load_json(proxies_index_file, "proxies manifest")
            existing_proxies = loaded.get("proxies", {}) if isinstance(loaded, dict) else {}
        except UserFacingError:
            existing_proxies = {}

    proxies_dir = work_path(project, "proxies", create_dir=True)
    records: dict[str, Any] = {}
    successes = 0
    reused = 0
    failures: list[str] = []

    for asset in video_assets:
        source_rel = asset.get("source")
        sha = asset.get("sha256")
        if not source_rel or not sha:
            continue

        source_abs = project_path(project, Path(*source_rel.split("/")))
        proxy_filename = f"proxy-{sha[:16]}-720p.mp4"
        proxy_abs = proxies_dir / proxy_filename
        proxy_rel = proxy_abs.relative_to(project).as_posix()

        # Check cache
        prior = existing_proxies.get(source_rel)
        if (
            not force
            and isinstance(prior, dict)
            and prior.get("source_sha256") == sha
            and prior.get("proxy_path") == proxy_rel
            and proxy_abs.is_file()
            and proxy_abs.stat().st_size > 0
        ):
            records[source_rel] = prior
            reused += 1
            continue

        # If original video resolution is already small (height <= target_height),
        # we can either make a light proxy or note it.
        src_height = asset.get("height")
        src_size = asset.get("size_bytes", 0)

        ok, err = create_proxy_for_video(source_abs, proxy_abs, target_height=target_height)
        if not ok:
            failures.append(f"{source_rel}: {err}")
            continue

        proxy_hash = sha256_file(proxy_abs)
        proxy_size = proxy_abs.stat().st_size
        record = {
            "source": source_rel,
            "source_sha256": sha,
            "source_size_bytes": src_size,
            "proxy_path": proxy_rel,
            "proxy_sha256": proxy_hash,
            "proxy_size_bytes": proxy_size,
            "compression_ratio": round(src_size / max(1, proxy_size), 2),
            "generated_at": utc_now(),
        }
        records[source_rel] = record
        successes += 1

    summary = {
        "manifest_version": 1,
        "generated_at": utc_now(),
        "target_height": target_height,
        "proxies": records,
        "total_videos": len(video_assets),
        "proxies_created": successes,
        "proxies_reused": reused,
        "failures": failures,
    }
    write_json(proxies_index_file, summary)
    return summary


def command_make_proxy(args: Any) -> int:
    project = resolve_project(args.project)
    height = getattr(args, "height", DEFAULT_TARGET_HEIGHT) or DEFAULT_TARGET_HEIGHT
    force = getattr(args, "force", False)

    summary = build_proxies_for_project(project, target_height=height, force=force)
    print(
        f"Proxies prepared: {summary['proxies_created']} created, {summary['proxies_reused']} reused "
        f"for {summary['total_videos']} video(s)."
    )
    if summary["failures"]:
        print("Some proxies failed to generate:", file=sys.stderr)
        for fail in summary["failures"]:
            print(f"- {fail}", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate 720p proxies for video assets.")
    parser.add_argument("project", help="path to project directory")
    parser.add_argument("--height", type=int, default=DEFAULT_TARGET_HEIGHT, help="target proxy height (default: 720)")
    parser.add_argument("--force", action="store_true", help="force recreate proxies")
    args = parser.parse_args()
    return command_make_proxy(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserFacingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
