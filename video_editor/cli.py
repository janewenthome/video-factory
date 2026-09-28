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
from video_editor.job_queue import JobQueueError, QueueAlreadyRunning
from video_editor.process_lock import HeavyJobBusy, exclusive_heavy_job
from video_editor.queue_runner import QueueApplicationError, VideoFactoryQueue


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


def _queue_service(args: argparse.Namespace) -> VideoFactoryQueue:
    return VideoFactoryQueue(queue_file=getattr(args, "queue_file", None))


def command_queue_add(args: argparse.Namespace) -> int:
    job = _queue_service(args).enqueue_project(
        args.project,
        max_attempts=args.max_attempts,
        force=args.force,
    )
    print(f"Queued {job['project_path']} as {job['id']} (pending).")
    print("The runner executes one local project at a time; the queue is stored outside the public repository.")
    return 0


def command_queue_list(args: argparse.Namespace) -> int:
    state = _queue_service(args).list_jobs()
    if state["paused"]:
        print(f"Queue paused: {state.get('pause_reason') or 'paused by user'}")
    else:
        print("Queue active")
    jobs = state["jobs"]
    if not jobs:
        print("No queued projects.")
        return 0
    for job in jobs:
        attempt_text = f"{job['attempts']}/{job['max_attempts']}"
        print(f"{job['id']}  {job['status']:<10}  attempts {attempt_text:<5}  {job['project_path']}")
        if job.get("last_error"):
            print(f"  Last error: {job['last_error']}")
        if job.get("runtime", {}).get("log_path"):
            print(f"  Log: {job['runtime']['log_path']}")
    return 0


def command_queue_run(args: argparse.Namespace) -> int:
    service = _queue_service(args)

    def emit(event: dict) -> None:
        message = event.get("message")
        if message:
            print(f"[queue] {message}")
        elif event.get("type") == "queue_job_started":
            print(f"[queue] Starting {event['project_path']}")
        elif event.get("type") == "queue_job_process_started":
            print(f"[queue] Pipeline PID {event['pid']}")
        elif event.get("type") == "queue_job_process_completed":
            print(f"[queue] Project complete. Log: {event['log_path']}")

    service.on_progress = emit
    try:
        summary = service.run(
            cooldown_seconds=args.cooldown_seconds,
            poll_seconds=args.poll_seconds,
            max_rechecks=args.max_rechecks,
            max_jobs=args.max_jobs,
        )
    except QueueAlreadyRunning as exc:
        print(f"[queue] {exc}", file=sys.stderr)
        return 75
    except (JobQueueError, QueueApplicationError, HeavyJobBusy, ValueError) as exc:
        print(f"[queue] {exc}", file=sys.stderr)
        return 2

    print(
        "Queue run finished: "
        f"{summary.jobs_completed} completed, {summary.jobs_failed} failed, "
        f"{summary.jobs_cancelled} cancelled, {summary.attempts_started} attempt(s)."
    )
    if summary.paused:
        print(f"Queue paused: {summary.pause_reason or 'system resource gate'}")
        return 75
    return 2 if summary.jobs_failed else 0


def command_queue_pause(args: argparse.Namespace) -> int:
    reason = args.reason or "Paused by user. The current project will finish before the queue stops."
    _queue_service(args).pause(reason)
    print(f"Queue paused. {reason}")
    return 0


def command_queue_resume(args: argparse.Namespace) -> int:
    _queue_service(args).resume()
    print("Queue resumed. Run `python -m video_editor queue run` to continue pending projects.")
    return 0


def command_queue_cancel(args: argparse.Namespace) -> int:
    result = _queue_service(args).cancel(args.job_id)
    if result == "cancellation_requested":
        print("Cancellation requested. The runner will stop and reap this job's process group before starting another.")
    else:
        print(f"Queue job status: {result}")
    return 0


def command_queue_reorder(args: argparse.Namespace) -> int:
    _queue_service(args).reorder(args.job_ids)
    print("Pending projects reordered.")
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
    run_parser.add_argument("--gate", choices=("AUTO", "REVIEW", "MANUAL"), default="AUTO", help="human review gate policy (AUTO creates a first cut by default)")
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

    queue_parser = subparsers.add_parser("queue", help="queue several ready projects and process them serially")
    queue_subparsers = queue_parser.add_subparsers(dest="queue_action", required=True)

    queue_add_parser = queue_subparsers.add_parser("add", help="queue a project with completed story and edit plans")
    queue_add_parser.add_argument("project", help="project path with job.yaml, story_plan.json and edit_plan.json")
    queue_add_parser.add_argument("--max-attempts", type=int, default=2, help="attempt limit including one retry by default")
    queue_add_parser.add_argument("--force", action="store_true", help="rerun cached stages as well as render")
    queue_add_parser.set_defaults(func=command_queue_add)

    queue_list_parser = queue_subparsers.add_parser("list", help="show queue order, status, attempts and logs")
    queue_list_parser.set_defaults(func=command_queue_list)

    queue_run_parser = queue_subparsers.add_parser("run", help="run pending projects one at a time")
    queue_run_parser.add_argument("--cooldown-seconds", type=float, default=45.0, help="wait between projects before checking memory (default: 45)")
    queue_run_parser.add_argument("--poll-seconds", type=float, default=30.0, help="interval for memory-pressure rechecks (default: 30)")
    queue_run_parser.add_argument("--max-rechecks", type=int, default=3, help="pause after this many unsafe/incomplete readings")
    queue_run_parser.add_argument("--max-jobs", type=int, default=None, help="stop after this many attempts; useful for a bounded run")
    queue_run_parser.set_defaults(func=command_queue_run)

    queue_pause_parser = queue_subparsers.add_parser("pause", help="finish the current project, then pause the queue")
    queue_pause_parser.add_argument("--reason", default=None)
    queue_pause_parser.set_defaults(func=command_queue_pause)

    queue_resume_parser = queue_subparsers.add_parser("resume", help="clear a user or memory-pressure pause")
    queue_resume_parser.set_defaults(func=command_queue_resume)

    queue_cancel_parser = queue_subparsers.add_parser("cancel", help="cancel one pending project or stop a running job safely")
    queue_cancel_parser.add_argument("job_id")
    queue_cancel_parser.set_defaults(func=command_queue_cancel)

    queue_reorder_parser = queue_subparsers.add_parser("reorder", help="reorder all pending jobs; include every pending ID once")
    queue_reorder_parser.add_argument("job_ids", nargs="+", help="pending job IDs in the desired order")
    queue_reorder_parser.set_defaults(func=command_queue_reorder)

    for command_parser in queue_subparsers.choices.values():
        command_parser.add_argument(
            "--queue-file",
            default=None,
            help="optional queue JSON path; otherwise use VIDEO_FACTORY_QUEUE_FILE or the checkout-local private queue",
        )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {"run", "prepare", "analyze", "plan", "render", "perception", "colab-perception"}:
        try:
            with exclusive_heavy_job():
                return args.func(args)
        except HeavyJobBusy as exc:
            print(f"[video_editor] {exc}", file=sys.stderr)
            return 75
    try:
        return args.func(args)
    except (JobQueueError, QueueApplicationError, ValueError) as exc:
        print(f"[video_editor] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
