"""Command Line Interface for Video Editor."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Add skills/video-factory/scripts to sys.path
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "skills" / "video-factory" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import video_factory
from video_editor.pipeline import PipelineOrchestrator
from colab_perception import command_colab_perception as run_colab_perception
from perception import command_perception as run_perception
from video_editor.music_catalog import (
    OPENVERSE_NON_ENDORSEMENT,
    OPENVERSE_WARNING,
    MusicCatalogError,
    OpenverseAudioCatalog,
)


def resolve_project_or_stage_raw(target: str, mode: str = "family") -> Path:
    """If target is a raw media folder (like materials/raw), stage it into a project.
    If it's already a project folder, return it directly.
    """
    path = Path(target).resolve()
    if not path.exists():
        raise video_factory.UserFacingError(f"Target folder does not exist: {path}. Create it and add media before running the pipeline.")

    # If it contains job.yaml or has assets/ and work/, it's already a project
    if (path / "job.yaml").is_file() or (path / "job.draft.yaml").is_file() or (path / "assets").is_dir():
        return path

    # Otherwise, it's a raw materials folder. Create a project under projects/
    project_root = Path.cwd() / "projects" / f"auto-{path.name}-{mode}"
    print(f"Staging raw materials from {path} into project: {project_root}...")

    # Init project
    args_init = argparse.Namespace(project=str(project_root), profile=mode)
    video_factory.command_init(args_init)

    # Rename job.draft.yaml to job.yaml if needed
    draft = project_root / "job.draft.yaml"
    job = project_root / "job.yaml"
    if draft.is_file() and not job.is_file():
        os.replace(draft, job)

    # Copy raw media to assets/
    for item in path.iterdir():
        if not item.is_file():
            continue
        suf = item.suffix.lower()
        if suf in video_factory.VIDEO_SUFFIXES:
            dest = project_root / "assets" / "videos" / item.name
            if not dest.exists():
                shutil.copy2(item, dest)
        elif suf in video_factory.IMAGE_SUFFIXES:
            dest = project_root / "assets" / "photos" / item.name
            if not dest.exists():
                shutil.copy2(item, dest)
        elif suf in video_factory.AUDIO_SUFFIXES:
            dest = project_root / "assets" / "narration" / item.name
            if not dest.exists():
                shutil.copy2(item, dest)

    return project_root


def command_prepare(args: argparse.Namespace) -> int:
    project_dir = resolve_project_or_stage_raw(args.target, mode=args.mode)
    orch = PipelineOrchestrator(project_dir, mode=args.mode, dry_run=args.dry_run)
    if args.dry_run:
        orch.print_dry_run_summary()
        return 0
    orch.run_stage_1_ingest()
    orch.run_stage_2_proxy()
    orch.run_stage_3_scenes()
    orch.run_stage_4_audio()
    orch.run_stage_6_contact_sheets()
    print(f"\nPreparation complete for {project_dir}. Media, proxies, scenes, and contact sheets are ready.")
    return 0


def command_analyze(args: argparse.Namespace) -> int:
    project_dir = resolve_project_or_stage_raw(args.target, mode=args.mode)
    orch = PipelineOrchestrator(project_dir, mode=args.mode, dry_run=args.dry_run)
    orch.run_stage_8_editorial_analysis()
    return 0


def command_plan(args: argparse.Namespace) -> int:
    project_dir = resolve_project_or_stage_raw(args.target, mode=args.mode)
    orch = PipelineOrchestrator(project_dir, mode=args.mode, dry_run=args.dry_run)
    orch.run_stage_9_edit_plan()
    orch.run_stage_10_validate()
    return 0


def command_render(args: argparse.Namespace) -> int:
    project_dir = resolve_project_or_stage_raw(args.target, mode="family")
    orch = PipelineOrchestrator(project_dir, dry_run=args.dry_run)
    orch.run_stage_11_render()
    orch.run_stage_12_qa()
    return 0


def command_run(args: argparse.Namespace) -> int:
    project_dir = resolve_project_or_stage_raw(args.target, mode=args.mode)
    orch = PipelineOrchestrator(
        project_dir=project_dir,
        mode=args.mode,
        resume=args.resume,
        force=args.force,
        no_gpu=args.no_gpu,
        gpu=args.gpu,
        allow_upload=args.allow_upload,
        gate=args.gate,
        dry_run=args.dry_run,
        approve_review=args.approve_review,
        duration_preset=args.duration_preset,
        custom_duration_seconds=args.duration_seconds,
        privacy_mode=args.privacy_mode,
        cloud_perception=args.cloud_perception,
        temporal_backend=args.temporal_backend,
        music_mode=args.music_mode,
    )
    return orch.run()


def command_perception_index(args: argparse.Namespace) -> int:
    args.project = args.target
    return run_perception(args)


def command_colab_perception_index(args: argparse.Namespace) -> int:
    args.project = args.target
    return run_colab_perception(args)


def command_music_search(args: argparse.Namespace) -> int:
    project = Path(args.project)
    query = " ".join(args.query)
    catalog = OpenverseAudioCatalog()
    try:
        tracks = catalog.search(query, limit=args.limit)
        snapshot = catalog.save_search(project, query, tracks)
    except MusicCatalogError as exc:
        print(f"[music] {exc}", file=sys.stderr)
        return 2

    print("Openverse search results")
    print(f"Warning: {OPENVERSE_WARNING}")
    print(OPENVERSE_NON_ENDORSEMENT)
    if not tracks:
        print("No results passed the CC0 / PDM / CC BY metadata checks.")
        print(f"Search snapshot: {snapshot}")
        return 0

    for index, track in enumerate(tracks, start=1):
        duration = f"{track.duration_seconds:.0f}s" if track.duration_seconds is not None else "duration unknown"
        print(f"\n{index}. {track.title} — {track.creator or 'creator not supplied'}")
        print(f"   ID: {track.id} | provider: {track.provider} | CC {track.license_slug.upper()} {track.license_version} | {duration}")
        print(f"   Source: {track.source_url}")
        print(f"   License: {track.license_url}")
        print(f"   Preview/download: {track.preview_url}")
    print(f"\nChoose a track explicitly with: video_editor music add <ID> --project {project}")
    print(f"Search snapshot: {snapshot}")
    return 0


def command_music_add(args: argparse.Namespace) -> int:
    catalog = OpenverseAudioCatalog()
    try:
        result = catalog.add(Path(args.project), args.id)
    except MusicCatalogError as exc:
        print(f"[music] {exc}", file=sys.stderr)
        return 2
    track = result["track"]
    print(f"Added selected track: {track.title} — {track.creator or 'creator not supplied'}")
    print(f"License: CC {track.license_slug.upper()} {track.license_version} ({track.license_url})")
    print(f"Source: {track.source_url}")
    print(f"Local file: {result['local_path']}")
    print(f"License/attribution snapshot: {result['evidence_path']}")
    print(f"SHA-256: {result['sha256']} | {result['size_bytes']} bytes | {result['content_type']}")
    print(f"Warning: {OPENVERSE_WARNING}")
    print(OPENVERSE_NON_ENDORSEMENT)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video_editor",
        description="Codex-coordinated local-first video editing on Mac mini M4, with optional deferred cloud adapters",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run the Mac-first editing pipeline; cloud processing stays opt-in")
    run_parser.add_argument("target", help="Project path or raw materials directory (e.g. materials/raw)")
    run_parser.add_argument("--mode", choices=("family", "health_education", "memory", "public-health"), default="family", help="editing mode / profile")
    run_parser.add_argument("--resume", action="store_true", default=True, help="resume from last completed stage")
    run_parser.add_argument("--force", action="store_true", help="force rerun all stages from scratch")
    run_parser.add_argument("--no-gpu", action="store_true", help="run CPU only; skip cloud GPU inference")
    run_parser.add_argument("--gpu", default="T4", help="Colab GPU accelerator (default: T4)")
    run_parser.add_argument("--allow-upload", action="store_true", help="authorize sending audio to Colab on cache miss")
    run_parser.add_argument("--gate", choices=("AUTO", "REVIEW", "MANUAL"), default="REVIEW", help="human review gate policy")
    run_parser.add_argument("--approve-review", action="store_true", help="confirm the current edit summary and allow render")
    run_parser.add_argument("--duration-preset", choices=("short", "standard", "full"), default=None, help="family/health duration preset")
    run_parser.add_argument("--duration-seconds", type=float, default=None, help="custom target duration; overrides preset")
    run_parser.add_argument("--privacy-mode", choices=("LOCAL_ONLY", "BALANCED", "MAX_QUALITY"), default=None, help="perception data-transfer policy")
    run_parser.add_argument("--cloud-perception", action="store_true", default=None, help="prepare a Colab perception plan; upload still requires --allow-upload")
    run_parser.add_argument("--temporal-backend", choices=("none", "smolvlm2"), default=None, help="shortlist-only temporal backend")
    run_parser.add_argument("--music-mode", choices=("auto_open_licensed", "project_music", "local_licensed", "none"), default=None, help="music discovery policy")
    run_parser.add_argument("--dry-run", action="store_true", help="show execution plan and compute estimates without allocating GPU")
    run_parser.set_defaults(func=command_run)

    # prepare command
    prep_parser = subparsers.add_parser("prepare", help="Stage media, extract proxies, scenes, audio, and contact sheets")
    prep_parser.add_argument("target", help="Project path or materials/raw")
    prep_parser.add_argument("--mode", default="family", help="editing mode / profile")
    prep_parser.add_argument("--dry-run", action="store_true", help="dry run only")
    prep_parser.set_defaults(func=command_prepare)

    # analyze command
    ana_parser = subparsers.add_parser("analyze", help="Run editorial asset scoring and story planning")
    ana_parser.add_argument("target", help="Project path")
    ana_parser.add_argument("--mode", default="family", help="editing mode")
    ana_parser.add_argument("--dry-run", action="store_true", help="dry run only")
    ana_parser.set_defaults(func=command_analyze)

    # plan command
    plan_parser = subparsers.add_parser("plan", help="Generate edit_plan.json, edit_summary.md and validate")
    plan_parser.add_argument("target", help="Project path")
    plan_parser.add_argument("--mode", default="family", help="editing mode")
    plan_parser.add_argument("--dry-run", action="store_true", help="dry run only")
    plan_parser.set_defaults(func=command_plan)

    # render command
    ren_parser = subparsers.add_parser("render", help="Render final video on Mac M4 and perform QA")
    ren_parser.add_argument("target", help="Project path")
    ren_parser.add_argument("--dry-run", action="store_true", help="dry run only")
    ren_parser.set_defaults(func=command_render)

    perception_parser = subparsers.add_parser("perception", help="build the local privacy-aware perception index")
    perception_parser.add_argument("target", help="Project path")
    perception_parser.add_argument("--privacy-mode", choices=("LOCAL_ONLY", "BALANCED", "MAX_QUALITY"), default=None)
    perception_parser.add_argument("--gpu", choices=("T4", "L4"), default=None)
    perception_parser.add_argument("--cloud", action="store_true", default=None)
    perception_parser.add_argument("--allow-upload", action="store_true")
    perception_parser.add_argument("--temporal-backend", choices=("none", "smolvlm2"), default=None)
    perception_parser.add_argument("--force", action="store_true")
    perception_parser.set_defaults(func=command_perception_index)

    cloud_perception_parser = subparsers.add_parser("colab-perception", help="run the derived-data Colab Perception Worker")
    cloud_perception_parser.add_argument("target", help="Project path")
    cloud_perception_parser.add_argument("--privacy-mode", choices=("BALANCED", "MAX_QUALITY"), default=None)
    cloud_perception_parser.add_argument("--gpu", choices=("T4", "L4"), default="T4")
    cloud_perception_parser.add_argument("--temporal-backend", choices=("none", "smolvlm2"), default=None)
    cloud_perception_parser.add_argument(
        "--temporal-source", dest="temporal_sources", action="append", default=[],
        metavar="PROJECT_RELATIVE_SOURCE",
        help="shortlisted source whose derived 720p proxy may be sent for temporal analysis; repeat for each source",
    )
    cloud_perception_parser.add_argument("--allow-upload", action="store_true")
    cloud_perception_parser.set_defaults(func=command_colab_perception_index)

    music_parser = subparsers.add_parser("music", help="search and explicitly add openly licensed background music")
    music_subparsers = music_parser.add_subparsers(dest="music_action", required=True)
    music_search_parser = music_subparsers.add_parser("search", help="search Openverse audio without downloading tracks")
    music_search_parser.add_argument("query", nargs="+", help="search words for music")
    music_search_parser.add_argument("--project", required=True, help="project path for the search snapshot")
    music_search_parser.add_argument("--limit", type=int, default=10, help="maximum candidates to list (1-50)")
    music_search_parser.set_defaults(func=command_music_search)
    music_add_parser = music_subparsers.add_parser("add", help="download one explicitly selected search result into work/")
    music_add_parser.add_argument("id", help="track ID printed by `music search`")
    music_add_parser.add_argument("--project", required=True, help="project containing the matching search snapshot")
    music_add_parser.set_defaults(func=command_music_add)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
