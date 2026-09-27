"""Backward-compatible entry point for the perception visual index.

New code should call ``perception``.  This command remains for existing
automation and writes a status document without making a final editorial
decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from video_factory import (
    UserFacingError,
    load_json,
    project_path,
    resolve_project,
    utc_now,
    work_path,
    write_json,
)

DEFAULT_ENABLED = False


def command_embeddings(args: Any) -> int:
    project = resolve_project(args.project)
    enabled = getattr(args, "enable", DEFAULT_ENABLED)

    analysis_dir = work_path(project, "analysis", create_dir=True)
    out_file = analysis_dir / "embeddings_status.json"

    if not enabled:
        status_doc = {
            "status": "disabled",
            "reason": "Visual perception is opt-in for cloud work; the local perception index can still record frames, clusters and pending embedding references without uploading media.",
            "updated_at": utc_now(),
        }
        write_json(out_file, status_doc)
        print("Stage 2 Visual AI embeddings: DISABLED (default).")
        print("Codex will perform direct editorial analysis from contact sheets and metadata.")
        return 0

    from perception import build_perception_index
    index_path = build_perception_index(project, privacy_mode="BALANCED", cloud=False)
    print("Stage 2 Visual AI embeddings: delegated to the perception index.")
    status_doc = {
        "status": "delegated",
        "model": "google/siglip2-base-patch16-224",
        "index": str(index_path.relative_to(project)),
        "updated_at": utc_now(),
    }
    write_json(out_file, status_doc)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 Visual AI scene embeddings")
    parser.add_argument("project", help="project directory")
    parser.add_argument("--enable", action="store_true", help="enable stage 2 embeddings")
    args = parser.parse_args()
    return command_embeddings(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except UserFacingError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
