"""Programmatic application boundary for GUI and desktop integrations.

The GUI calls this API instead of parsing terminal output.  It emits compact
human-facing progress events while the processing layer remains the existing
deterministic pipeline.  A Tauri shell can wrap this class later without
moving media files into the GUI process.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from video_editor.pipeline import PipelineOrchestrator, video_factory


@dataclass(frozen=True)
class ProjectOptions:
    # None means inherit the selected project's job.yaml value.
    mode: str | None = None
    duration_preset: str | None = None
    custom_duration_seconds: float | None = None
    privacy_mode: str | None = None
    cloud_perception: bool | None = None
    gpu_policy: str | None = None
    temporal_backend: str | None = None
    music_mode: str | None = None
    ai_quality: str = "standard"
    review_gate: str = "REVIEW"
    allow_upload: bool = False


@dataclass(frozen=True)
class SourceSummary:
    source_path: str
    folder_name: str
    photo_count: int
    video_count: int
    audio_count: int
    total_video_duration_seconds: float
    estimated_source_size_bytes: int


class VideoFactoryApplication:
    """Application service used by a GUI, automation or the CLI."""

    def __init__(self, project_dir: str | Path, options: ProjectOptions | None = None, *, on_progress: Callable[[dict[str, Any]], None] | None = None):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.options = options or ProjectOptions()
        if self.options.ai_quality not in {"standard", "deep"}:
            raise ValueError("ai_quality must be standard or deep")
        self.on_progress = on_progress

    def _orchestrator(self, **overrides: Any) -> PipelineOrchestrator:
        temporal_backend = self.options.temporal_backend
        if self.options.ai_quality == "deep" and temporal_backend in {None, "none"}:
            # Deep quality enables the pluggable temporal stage, but the
            # privacy and GPU policies still decide whether it can run.
            temporal_backend = "smolvlm2"
        values = {
            "project_dir": self.project_dir,
            "mode": self.options.mode,
            "gpu": self.options.gpu_policy,
            "allow_upload": self.options.allow_upload,
            "gate": self.options.review_gate,
            "duration_preset": self.options.duration_preset,
            "custom_duration_seconds": self.options.custom_duration_seconds,
            "privacy_mode": self.options.privacy_mode,
            "cloud_perception": self.options.cloud_perception,
            "temporal_backend": temporal_backend,
            "music_mode": self.options.music_mode,
            "progress_callback": self.on_progress,
        }
        values.update(overrides)
        return PipelineOrchestrator(**values)

    def inspect_source(self, source_dir: str | Path) -> SourceSummary:
        """Return the GUI home-page summary without copying source media."""
        root = Path(source_dir).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Source folder does not exist: {root}")
        photos = videos = audios = 0
        total_duration = 0.0
        total_size = 0
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                total_size += path.stat().st_size
            except OSError:
                continue
            suffix = path.suffix.lower()
            if suffix in video_factory.IMAGE_SUFFIXES:
                photos += 1
            elif suffix in video_factory.VIDEO_SUFFIXES:
                videos += 1
                try:
                    total_duration += float(video_factory.ffprobe_metadata(path).get("duration_seconds") or 0.0)
                except Exception:
                    # The summary remains useful when one file is not probeable;
                    # ingest will report that file as an actual pipeline error.
                    pass
            elif suffix in video_factory.AUDIO_SUFFIXES:
                audios += 1
        return SourceSummary(
            source_path=str(root),
            folder_name=root.name,
            photo_count=photos,
            video_count=videos,
            audio_count=audios,
            total_video_duration_seconds=round(total_duration, 3),
            estimated_source_size_bytes=total_size,
        )

    def run(self, *, approve_review: bool = False, resume: bool = True, force: bool = False, dry_run: bool = False) -> int:
        return self._orchestrator(
            approve_review=approve_review,
            resume=resume,
            force=force,
            dry_run=dry_run,
        ).run()

    def prepare(self) -> None:
        runner = self._orchestrator()
        runner.run_stage_1_ingest()
        runner.run_stage_2_proxy()
        runner.run_stage_3_scenes()
        runner.run_stage_4_audio()
        runner.run_stage_6_contact_sheets()

    def analyze(self) -> None:
        runner = self._orchestrator()
        runner.run_stage_5_transcription()
        runner.run_stage_7_perception()
        runner.run_stage_8_editorial_analysis()

    def plan(self) -> None:
        runner = self._orchestrator()
        runner.run_stage_9_edit_plan()
        runner.run_stage_10_validate()

    def prepare_music(self) -> None:
        """Create the music requirement and licensing artifacts for the plan."""
        runner = self._orchestrator()
        runner.run_stage_9_edit_plan()

    def render(self, *, approve_review: bool = True) -> None:
        runner = self._orchestrator(approve_review=approve_review, gate="AUTO" if approve_review else self.options.review_gate)
        runner.run_stage_11_render()
        runner.run_stage_12_qa()


__all__ = ["ProjectOptions", "SourceSummary", "VideoFactoryApplication"]
