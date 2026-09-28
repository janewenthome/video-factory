"""Mac-first pipeline orchestrator for Video Factory.

Handles:
1. Ingest (manifest, sha256)
2. Proxy (720p VideoToolbox)
3. Scenes (FFmpeg / PySceneDetect)
4. Audio (extraction)
5. Cut-first local transcription (MLX Whisper on Apple Silicon)
6. Contact Sheets (5x5 tiled stills)
7. Perception index (speech, visual and temporal evidence contract)
8. Editorial Analysis (asset scoring & story arc)
9. Edit Plan (JSON + human-readable edit_summary.md)
10. Validate (plan structure, timecodes, references)
11. Render (Remotion + VideoToolbox encoding on Mac M4)
12. QA (technical checks, blackdetect, audio levels, qa_report.json)

State is persisted in work/pipeline_state.json for resume support.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

# Add skills/video-factory/scripts to sys.path so we can reuse core modules
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "skills" / "video-factory" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import video_factory
from edit_summary import generate_edit_summary
from make_proxy import build_proxies_for_project
from perception import build_perception_index
from video_editor.music import prepare_music_artifacts
from video_editor.product_policy import duration_summary, load_product_policy

STATE_FILE_NAME = "pipeline_state.json"
PIPELINE_STAGES = [
    "ingest",
    "proxy",
    "scenes",
    "audio",
    "transcription",
    "contact_sheets",
    "perception",
    "editorial_analysis",
    "edit_plan",
    "validate",
    "render",
    "qa",
]
HUMAN_PROGRESS_MESSAGES = {
    "ingest": "正在準備影片內容",
    "proxy": "正在建立預覽影片",
    "scenes": "正在整理場景",
    "audio": "正在處理音訊",
    "transcription": "正在分析語音",
    "contact_sheets": "正在整理代表畫面",
    "perception": "正在整理相似畫面",
    "editorial_analysis": "正在尋找精彩片段",
    "edit_plan": "正在整理剪輯計畫",
    "validate": "正在檢查剪輯計畫",
    "render": "正在輸出影片",
    "qa": "正在檢查成品",
    "review": "請先檢查剪輯摘要",
    "music_requirements": "正在分析配樂需求",
    "music_discovery": "正在尋找適合的公開授權配樂",
    "music_license_verification": "正在確認配樂授權",
    "music_prepare": "正在準備配樂",
}


class PipelineError(Exception):
    """Raised when a pipeline stage fails."""


class PipelineConsentPending(PipelineError):
    """Raised when a requested cloud stage is waiting for per-run consent."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_pipeline_state(project_dir: Path) -> dict[str, Any]:
    state_file = project_dir / "work" / STATE_FILE_NAME
    if state_file.is_file():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "version": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "current_stage": None,
        "completed_stages": [],
        "stages": {},
    }


def save_pipeline_state(project_dir: Path, state: dict[str, Any]) -> None:
    work_dir = project_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = utc_now()
    state_file = work_dir / STATE_FILE_NAME
    temp_file = work_dir / f".{STATE_FILE_NAME}.tmp"
    temp_file.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_file, state_file)


class PipelineOrchestrator:
    def __init__(
        self,
        project_dir: Path,
        mode: str = "family",
        resume: bool = True,
        force: bool = False,
        no_gpu: bool = False,
        gpu: str = "T4",
        allow_upload: bool = False,
        gate: str = "AUTO",
        dry_run: bool = False,
        approve_review: bool = False,
        duration_preset: str | None = None,
        custom_duration_seconds: float | None = None,
        privacy_mode: str | None = None,
        cloud_perception: bool | None = None,
        temporal_backend: str | None = None,
        music_mode: str | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.project_dir = project_dir.resolve()
        try:
            self.policy = load_product_policy(
                self.project_dir,
                mode=mode,
                duration_preset=duration_preset,
                custom_duration_seconds=custom_duration_seconds,
                privacy_mode=privacy_mode,
                gpu_policy=gpu,
                cloud_perception=cloud_perception,
                music_mode=music_mode,
            )
        except ValueError as exc:
            raise PipelineError(str(exc)) from exc
        self.mode = self.policy.mode
        self.resume = resume
        self.force = force
        self.no_gpu = no_gpu
        self.gpu = "none" if no_gpu else (gpu or "T4")
        self.allow_upload = allow_upload
        self.gate = gate.upper()  # AUTO, REVIEW, MANUAL
        self.dry_run = dry_run
        self.approve_review = approve_review
        self.temporal_backend = temporal_backend
        self.progress_callback = progress_callback
        self.state = load_pipeline_state(self.project_dir)

    def emit_progress(self, stage: str, status: str, message: str) -> None:
        display_message = HUMAN_PROGRESS_MESSAGES.get(stage, message)
        if message.startswith("正在") or message.startswith("請先"):
            display_message = message
        event = {"stage": stage, "status": status, "message": display_message, "project": self.project_dir.name}
        if self.progress_callback:
            self.progress_callback(event)

    def print_dry_run_summary(self) -> None:
        """Display the dry-run resource, compute and transfer expectations."""
        print("==================================================")
        print(" [DRY RUN] Video Factory Execution Plan")
        print("==================================================")
        print(f"Project Target : {self.project_dir}")
        print(f"Workflow Mode  : {self.mode}")
        print(f"Review Gate    : {self.gate}")
        print("--------------------------------------------------")

        # Check existing assets
        assets_dir = self.project_dir / "assets"
        total_size_bytes = 0
        video_count = 0
        photo_count = 0
        audio_count = 0
        if assets_dir.is_dir():
            for p in assets_dir.rglob("*"):
                if p.is_file():
                    total_size_bytes += p.stat().st_size
                    suf = p.suffix.lower()
                    if suf in video_factory.VIDEO_SUFFIXES:
                        video_count += 1
                    elif suf in video_factory.IMAGE_SUFFIXES:
                        photo_count += 1
                    elif suf in video_factory.AUDIO_SUFFIXES:
                        audio_count += 1

        print(f"Local Assets   : {video_count} videos, {photo_count} photos, {audio_count} audio files ({total_size_bytes / (1024*1024):.1f} MB)")
        print("\nLocal CPU Tasks (Mac mini M4):")
        print("  1. Ingest (ffprobe metadata & sha256 hashing)")
        print("  2. Proxy generation (720p hardware VideoToolbox encoding)")
        print("  3. Scene detection (FFmpeg / PySceneDetect)")
        print("  4. Audio extraction (FLAC/WAV for transcription)")
        print("  5. Contact sheets and representative frames")
        print("  6. Editorial validation (edit_plan integrity check)")
        print("  7. Final video assembly & render (Remotion + Apple VideoToolbox)")
        print("  8. Quality Assurance (ffprobe, blackdetect, audio levels)")

        print("\nLocal AI Tasks (Mac mini M4):")
        print("  - After the first edit cut, run MLX Whisper on retained natural-audio clips only.")
        print("  - The selected editorial director reviews timed candidates before any subtitle is applied.")
        print("  - Colab is deferred; this workflow does not upload audio, frames, or video.")

        print("\nResource & Transfer Estimates:")
        print("  - User media transfer: none")
        print("  - Compute Units      : 0 (Colab is not used)")
        print("==================================================")

    def is_stage_completed(self, stage: str) -> bool:
        if self.force:
            return False
        if not self.resume:
            return False
        stage_info = self.state.get("stages", {}).get(stage)
        if not (stage_info and stage_info.get("status") == "completed"):
            return False
        if stage == "perception":
            expected_mode = self._requested_perception_mode()
            return bool(expected_mode and self.state.get("perception_execution_mode") == expected_mode)
        return True

    def _requested_perception_mode(self) -> str | None:
        """Return the perception route selected by current privacy/job policy."""
        if (
            not self.policy.cloud_perception
            or self.no_gpu
            or self.policy.privacy_mode == "LOCAL_ONLY"
        ):
            return "local"
        try:
            cloud_setting = video_factory.get_cloud_processing_setting(self.project_dir, "colab")
        except video_factory.UserFacingError:
            return None
        if cloud_setting == "disabled":
            return "local"
        if cloud_setting == "ask_each_run":
            return "colab"
        # An unset or invalid policy cannot reuse a previous result as if it
        # satisfied the current cloud request. The stage will surface the gate.
        return None

    def mark_stage_start(self, stage: str) -> None:
        self.state["current_stage"] = stage
        if "stages" not in self.state:
            self.state["stages"] = {}
        self.state["stages"][stage] = {
            "status": "in_progress",
            "started_at": utc_now(),
        }
        save_pipeline_state(self.project_dir, self.state)

    def mark_stage_complete(self, stage: str, outputs: list[str] | None = None) -> None:
        if stage not in self.state.get("completed_stages", []):
            self.state.setdefault("completed_stages", []).append(stage)
        self.state["stages"][stage] = {
            "status": "completed",
            "finished_at": utc_now(),
            "outputs": outputs or [],
        }
        self.state["current_stage"] = None
        save_pipeline_state(self.project_dir, self.state)

    def mark_stage_failed(self, stage: str, error: str) -> None:
        self.state["stages"][stage] = {
            "status": "failed",
            "failed_at": utc_now(),
            "error": error,
        }
        save_pipeline_state(self.project_dir, self.state)

    def mark_stage_pending(self, stage: str, reason: str) -> None:
        completed = self.state.setdefault("completed_stages", [])
        if stage in completed:
            completed.remove(stage)
        self.state.setdefault("stages", {})[stage] = {
            "status": "pending",
            "pending_at": utc_now(),
            "reason": reason,
        }
        self.state["current_stage"] = stage
        save_pipeline_state(self.project_dir, self.state)

    # ================= Stage Implementations =================

    def run_stage_1_ingest(self) -> None:
        print("[Stage 1/12] Ingest: Inspecting assets & calculating hash metadata...")
        args = argparse.Namespace(project=str(self.project_dir))
        res = video_factory.command_inspect(args)
        if res != 0:
            raise PipelineError("Stage 1 (ingest) failed.")

    def run_stage_2_proxy(self) -> None:
        print("[Stage 2/12] Proxy: Building 720p VideoToolbox proxies for video assets...")
        summary = build_proxies_for_project(self.project_dir, force=self.force)
        if summary.get("failures"):
            raise PipelineError(
                f"{len(summary['failures'])} proxy file(s) failed; inspect work/proxies/manifest.json before continuing."
            )

    def run_stage_3_scenes(self) -> None:
        print("[Stage 3/12] Scenes: Detecting scene boundaries on local CPU...")
        from media_helpers import command_extract_scenes
        args = argparse.Namespace(
            project=str(self.project_dir),
            threshold=0.35,
            min_scene_duration=1.0,
            command="extract-scenes",
        )
        res = command_extract_scenes(args)
        if res != 0:
            raise PipelineError("Stage 3 (scenes) failed.")

    def run_stage_4_audio(self) -> None:
        print("[Stage 4/12] Audio: Extracting clean audio for transcription...")
        from media_helpers import command_extract_audio
        args = argparse.Namespace(
            project=str(self.project_dir),
            format="flac",
            channels=1,
            sample_rate=48000,
            command="extract-audio",
        )
        res = command_extract_audio(args)
        if res != 0:
            raise PipelineError("Stage 4 (audio) failed.")

    def run_stage_5_transcription(self) -> None:
        print("[Stage 5/12] Speech Intelligence: waiting for the first edit cut...")
        self.emit_progress("speech", "started", "正在分析語音")
        print("Transcription is intentionally delayed until edit_plan.json identifies retained audio clips.")
        self.emit_progress("speech", "deferred", "先完成剪輯選段，再本機分析保留現場聲")

    def run_stage_6_contact_sheets(self) -> None:
        print("[Stage 6/12] Contact Sheets: Extracting key stills & building 5x5 grids...")
        from media_helpers import command_extract_frames, command_build_contact_sheets
        args_frames = argparse.Namespace(
            project=str(self.project_dir),
            per_video=6,
            max_frames_per_video=24,
            command="extract-frames",
        )
        command_extract_frames(args_frames)

        args_sheets = argparse.Namespace(
            project=str(self.project_dir),
            columns=5,
            tile_width=320,
            command="build-contact-sheets",
        )
        command_build_contact_sheets(args_sheets)

    def run_stage_7_perception(self) -> None:
        print("[Stage 7/12] Perception: building speech, visual and temporal evidence index...")
        self.emit_progress("perception", "started", "正在整理相似畫面與影片內容")
        temporal_backend = self.temporal_backend or self.policy.temporal_backend
        cloud_requested = (
            self.policy.cloud_perception
            and not self.no_gpu
            and self.policy.privacy_mode != "LOCAL_ONLY"
        )
        cloud_setting = (
            video_factory.get_cloud_processing_setting(self.project_dir, "colab")
            if cloud_requested
            else None
        )

        if not cloud_requested or cloud_setting == "disabled":
            path = build_perception_index(
                self.project_dir,
                privacy_mode=self.policy.privacy_mode,
                # This path never allocates a Colab accelerator. Use the
                # routine T4 policy metadata so an unused L4 request cannot
                # block otherwise-local processing.
                gpu_policy="T4",
                cloud=False,
                allow_upload=False,
                temporal_backend=temporal_backend,
                force=self.force,
            )
            self.state["perception_execution_mode"] = "local"
            print(f"Local perception index written: {path}")
            if self.policy.cloud_perception and cloud_setting == "disabled":
                message = "正在本機整理相似畫面（job.yaml 已停用 Colab，未傳送檔案）"
            elif self.no_gpu and self.policy.cloud_perception:
                message = "正在本機整理相似畫面（本次執行停用 GPU，未傳送檔案）"
            elif self.policy.privacy_mode == "LOCAL_ONLY" and self.policy.cloud_perception:
                message = "正在本機整理相似畫面（LOCAL_ONLY 未使用 Colab）"
            else:
                message = "正在本機整理相似畫面（未使用 Colab）"
            self.emit_progress("perception", "completed", message)
            return

        if cloud_setting != "ask_each_run":
            reason = (
                "請先在 job.yaml 設定 cloud_processing.colab: ask_each_run；目前只建立本機待授權索引，未傳送檔案"
            )
        elif not self.allow_upload:
            reason = "請先確認本次 Colab 衍生資料上傳，再以 --allow-upload 重跑；目前未傳送檔案"
        else:
            from colab_perception import command_colab_perception

            args = argparse.Namespace(
                project=str(self.project_dir),
                privacy_mode=self.policy.privacy_mode,
                gpu=self.policy.gpu_policy,
                temporal_backend=temporal_backend,
                allow_upload=True,
            )
            try:
                result = command_colab_perception(args)
            except Exception as exc:
                raise PipelineError(f"Colab perception failed; pipeline stopped without editorial analysis: {exc}") from exc
            if result != 0:
                raise PipelineError(
                    f"Colab perception returned status {result}; pipeline stopped without editorial analysis."
                )
            index_path = self.project_dir / "work" / "perception_index.json"
            if not index_path.is_file():
                raise PipelineError("Colab perception reported success but did not write work/perception_index.json.")
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PipelineError("Colab perception wrote an unreadable perception index.") from exc
            privacy = index.get("privacy") if isinstance(index, dict) else None
            if (
                not isinstance(privacy, dict)
                or privacy.get("cloud_status") != "completed"
                or privacy.get("original_media_uploaded") is not False
            ):
                raise PipelineError("Colab perception result did not confirm completion and the original-media privacy invariant.")
            self.state["perception_execution_mode"] = "colab"
            print(f"Colab perception index written: {index_path}")
            self.emit_progress("perception", "completed", "正在整理相似畫面與雲端感知結果")
            return

        # Record a local-only transfer plan so the user can review eligible
        # derived files. Consent is still required before invoking the worker.
        path = build_perception_index(
            self.project_dir,
            privacy_mode=self.policy.privacy_mode,
            gpu_policy=self.policy.gpu_policy,
            cloud=True,
            allow_upload=False,
            temporal_backend=temporal_backend,
            # Force the canonical index to reflect this run's consent gate;
            # older local cache metadata may otherwise say "not_requested".
            force=True,
        )
        self.state.pop("perception_execution_mode", None)
        self.mark_stage_pending("perception", reason)
        print(f"Colab perception pending consent; local index written: {path}")
        self.emit_progress("perception", "pending", reason)
        raise PipelineConsentPending(reason)

    def run_stage_8_editorial_analysis(self) -> None:
        print(f"[Stage 8/12] Editorial Analysis: Evaluating story arc using {self.mode} profile...")
        # Check if asset analysis and story plan exist, or initialize default template
        analysis_dir = self.project_dir / "work" / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        story_plan_file = analysis_dir / "story_plan.json"

        if not story_plan_file.is_file():
            # Generate a structured story plan based on scenes and assets
            manifest_file = self.project_dir / "work" / "manifests" / "media_manifest.json"
            manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.is_file() else {}
            assets = manifest.get("assets", [])

            story_plan = {
                "version": 1,
                "profile": self.mode,
                "created_at": utc_now(),
                "hook": "建立開場視覺吸引力",
                "structure": ["hook", "development", "climax", "ending"],
                "selected_assets": [a.get("source") for a in assets if a.get("kind") in {"video", "photo"}],
            }
            story_plan_file.write_text(json.dumps(story_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Created story plan: {story_plan_file}")

    def run_stage_9_edit_plan(self) -> None:
        print("[Stage 9/12] Edit Plan: Generating edit_plan.json & human-readable edit_summary.md...")
        plan_file = self.project_dir / "work" / "edit-plan" / "edit_plan.json"
        if not plan_file.is_file():
            # If no edit plan exists yet, build a baseline timeline from inspected assets
            manifest_file = self.project_dir / "work" / "manifests" / "media_manifest.json"
            manifest = json.loads(manifest_file.read_text(encoding="utf-8")) if manifest_file.is_file() else {}
            assets = [a for a in manifest.get("assets", []) if a.get("kind") in {"video", "photo"}]

            timeline = []
            cur_time = 0.0
            for i, a in enumerate(assets):
                if cur_time >= self.policy.duration.target_seconds:
                    break
                dur = float(a.get("duration_seconds") or 3.0)
                clip_dur = min(dur, 4.0) if a.get("kind") == "video" else 3.0
                clip_dur = min(clip_dur, self.policy.duration.target_seconds - cur_time)
                if clip_dur <= 0:
                    break
                seg = {
                    "id": f"clip-{i+1:03d}",
                    "type": a.get("kind"),
                    "source": a.get("source"),
                    "timeline_start": round(cur_time, 2),
                    "timeline_end": round(cur_time + clip_dur, 2),
                }
                if a.get("kind") == "video":
                    seg["source_in"] = 0.0
                    seg["source_out"] = round(clip_dur, 2)
                timeline.append(seg)
                cur_time += clip_dur

            edit_plan_data = {
                "version": 1,
                "title": f"{self.project_dir.name} ({self.mode.title()})",
                "profile": self.mode,
                "aspect_ratio": "16:9",
                "fps": 30.0,
                "duration_seconds": round(cur_time, 2),
                "target_duration_seconds": self.policy.duration.target_seconds,
                "duration_policy": duration_summary(self.policy.duration, cur_time),
                "privacy_mode": self.policy.privacy_mode,
                "perception_index": "work/perception_index.json",
                "music_mode": self.policy.music_mode,
                "timeline": timeline,
                "references": [],
            }
            plan_file.parent.mkdir(parents=True, exist_ok=True)
            plan_file.write_text(json.dumps(edit_plan_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Generated initial edit plan with {len(timeline)} segments: {plan_file}")

        else:
            # Upgrade older plans in place with product and perception
            # metadata while preserving the director's existing timeline.
            try:
                existing_plan = json.loads(plan_file.read_text(encoding="utf-8"))
                if isinstance(existing_plan, dict):
                    actual = float(existing_plan.get("duration_seconds") or 0.0)
                    existing_plan.setdefault("target_duration_seconds", self.policy.duration.target_seconds)
                    existing_plan.setdefault("duration_policy", duration_summary(self.policy.duration, actual))
                    existing_plan.setdefault("privacy_mode", self.policy.privacy_mode)
                    existing_plan.setdefault("perception_index", "work/perception_index.json")
                    existing_plan.setdefault("music_mode", self.policy.music_mode)
                    plan_file.write_text(json.dumps(existing_plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise PipelineError(f"Could not upgrade existing edit plan metadata: {exc}") from exc

        # Generate human-readable edit_summary.md
        try:
            plan_data = json.loads(plan_file.read_text(encoding="utf-8"))
            from qa_helpers import read_job_bool
            if read_job_bool(self.project_dir, "captions", "enabled") is not False:
                from video_editor.caption_pipeline import draft_selected_caption_candidates

                self.emit_progress("speech", "started", "正在分析保留片段的語音")
                try:
                    candidate_doc = draft_selected_caption_candidates(
                        self.project_dir, language="zh", local_files_only=True,
                    )
                except video_factory.UserFacingError as exc:
                    print(f"Local speech analysis is pending: {exc}")
                    self.emit_progress("speech", "pending", "本機語音辨識未就緒，字幕保留待處理")
                else:
                    candidate_count = len(candidate_doc.get("candidates", []))
                    print(
                        f"Local speech analysis: {candidate_count} timed caption candidate(s) from "
                        f"{candidate_doc.get('selected_clip_count', 0)} retained natural-audio clip(s)."
                    )
                    if candidate_count:
                        self.emit_progress("speech", "pending", "語音候選已產生，等待判斷字幕是否有意義")
                    else:
                        self.emit_progress("speech", "completed", "保留現場聲未偵測到可用語音字幕")
            else:
                self.emit_progress("speech", "skipped", "專案已停用字幕分析")
            self.emit_progress("music_requirements", "started", "正在分析配樂需求")
            prepare_music_artifacts(self.project_dir, self.policy, plan_data)
            self.emit_progress("music_requirements", "completed", "配樂需求分析完成")
            self.emit_progress("music_discovery", "completed", "公開授權配樂搜尋待 provider 授權")
            self.emit_progress("music_license_verification", "completed", "配樂授權閘門已建立")
            self.emit_progress("music_prepare", "completed", "配樂準備完成；無安全曲目時保留無配樂版本")
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelineError(f"Could not prepare music requirements: {exc}") from exc
        summary_path = generate_edit_summary(self.project_dir)
        print(f"Review summary updated: {summary_path}")

    def run_stage_10_validate(self) -> None:
        print("[Stage 10/12] Validate: Validating edit plan constraints and reference integrity...")
        args = argparse.Namespace(project=str(self.project_dir))
        res = video_factory.command_validate_plan(args)
        if res != 0:
            raise PipelineError("Stage 10 (validate) failed. Check edit plan errors above.")

    def run_stage_11_render(self) -> None:
        print("[Stage 11/12] Render: Preparing Remotion staging & encoding via VideoToolbox on Mac M4...")
        args = argparse.Namespace(project=str(self.project_dir), ratio=None)
        res = video_factory.command_prepare_render(args)
        if res != 0:
            raise PipelineError("Stage 11 (prepare-render) failed.")

        remotion_dir = SCRIPTS_DIR.parent / "runtime" / "remotion"
        remotion_bin = remotion_dir / "node_modules" / ".bin" / "remotion"
        out_mp4 = self.project_dir / "outputs" / "master.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        props_file = self.project_dir / "work" / "render-input.json"
        public_dir = self.project_dir / "work" / "render-public"
        version_dir = self._begin_render_version(out_mp4)

        if remotion_bin.is_file():
            print(f"Rendering Remotion composition to {out_mp4}...")
            env = os.environ.copy()
            env["VIDEO_FACTORY_PUBLIC_DIR"] = str(public_dir)
            cmd = [
                str(remotion_bin), "render", "src/index.ts", "VideoFactory",
                str(out_mp4),
                "--props", str(props_file),
            ]
            run_res = subprocess.run(cmd, cwd=str(remotion_dir), env=env, check=False)
            if run_res.returncode != 0:
                raise PipelineError(f"Remotion render failed with exit code {run_res.returncode}.")
        else:
            print("Remotion runtime node_modules not yet installed in runtime dir.")
            print("Falling back to local FFmpeg VideoToolbox assembly...")
            # FFmpeg fallback concatenation if needed
            self._render_with_ffmpeg_videotoolbox(out_mp4)
        try:
            shutil.copy2(out_mp4, version_dir / "master.mp4")
        except OSError as exc:
            raise PipelineError(f"Could not preserve render version {version_dir.name}: {exc}") from exc
        self.state.setdefault("render_history", [])[-1].update({"status": "rendered", "output": str(version_dir / "master.mp4")})
        save_pipeline_state(self.project_dir, self.state)

    def _begin_render_version(self, output_path: Path) -> Path:
        """Preserve each edit/render decision without copying source assets."""
        version_id = "v-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        version_dir = self.project_dir / "outputs" / "versions" / version_id
        suffix = 1
        while version_dir.exists():
            version_dir = self.project_dir / "outputs" / "versions" / f"{version_id}-{suffix}"
            suffix += 1
        version_dir.mkdir(parents=True, exist_ok=False)
        if output_path.is_file():
            try:
                shutil.copy2(output_path, version_dir / "previous-master.mp4")
            except OSError as exc:
                raise PipelineError(f"Could not preserve the previous render before overwrite: {exc}") from exc
        for relative in (
            Path("work/edit-plan/edit_plan.json"),
            Path("work/edit_summary.md"),
            Path("work/music_requirements.json"),
            Path("work/music_attribution.json"),
        ):
            source = self.project_dir / relative
            if source.is_file():
                destination = version_dir / relative.name
                shutil.copy2(source, destination)
        self.state.setdefault("render_history", []).append({
            "version": version_dir.name,
            "started_at": utc_now(),
            "status": "rendering",
            "edit_plan": str(version_dir / "edit_plan.json"),
        })
        save_pipeline_state(self.project_dir, self.state)
        return version_dir

    def _render_with_ffmpeg_videotoolbox(self, output_path: Path) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise PipelineError("ffmpeg is required for rendering.")
        plan_file = self.project_dir / "work" / "edit-plan" / "edit_plan.json"
        plan = json.loads(plan_file.read_text(encoding="utf-8"))
        timeline = plan.get("timeline", [])

        # The fallback is deliberately limited to real video sources.  A
        # synthetic test card would make a failed/empty edit plan look like a
        # successful deliverable.
        concat_txt = self.project_dir / "work" / "concat.txt"
        lines = []
        for seg in timeline:
            if seg.get("type") == "video":
                src = self.project_dir / seg["source"]
                if src.is_file():
                    lines.append(f"file '{src.resolve()}'")
                    if seg.get("source_in") is not None:
                        lines.append(f"inpoint {float(seg['source_in']):.6f}")
                    if seg.get("source_out") is not None:
                        lines.append(f"outpoint {float(seg['source_out']):.6f}")
        if not lines:
            raise PipelineError("No real video source is available for the local render fallback.")

        concat_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        temporary_output = output_path.with_name(f".{output_path.name}.tmp")
        try:
            subprocess.run([
                ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt),
                "-c:v", "h264_videotoolbox", "-b:v", "4M", "-c:a", "aac", str(temporary_output)
            ], check=True)
            if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
                raise PipelineError("FFmpeg fallback produced an empty render.")
            os.replace(temporary_output, output_path)
        finally:
            temporary_output.unlink(missing_ok=True)

    def run_stage_12_qa(self) -> None:
        print("[Stage 12/12] QA: Running media verification and generating qa_report.json...")
        from qa_helpers import command_qa
        out_mp4 = self.project_dir / "outputs" / "master.mp4"
        if not out_mp4.is_file():
            # Check if any mp4 in outputs
            mp4s = list((self.project_dir / "outputs").glob("*.mp4"))
            if mp4s:
                out_mp4 = mp4s[0]
            else:
                raise PipelineError("No rendered video found in outputs/ to run QA on.")

        args = argparse.Namespace(project=str(self.project_dir), video=str(out_mp4), command="qa")
        command_qa(args)

    # ================= Execution Flow =================

    def run(self) -> int:
        if self.dry_run:
            self.print_dry_run_summary()
            return 0

        stage_runners: dict[str, Callable[[], None]] = {
            "ingest": self.run_stage_1_ingest,
            "proxy": self.run_stage_2_proxy,
            "scenes": self.run_stage_3_scenes,
            "audio": self.run_stage_4_audio,
            "transcription": self.run_stage_5_transcription,
            "contact_sheets": self.run_stage_6_contact_sheets,
            "perception": self.run_stage_7_perception,
            "editorial_analysis": self.run_stage_8_editorial_analysis,
            "edit_plan": self.run_stage_9_edit_plan,
            "validate": self.run_stage_10_validate,
            "render": self.run_stage_11_render,
            "qa": self.run_stage_12_qa,
        }

        print("==================================================")
        print(f" Starting Video Factory Pipeline: {self.project_dir.name}")
        print(f" Mode: {self.mode} | Resume: {self.resume} | Gate: {self.gate}")
        print("==================================================")

        total_stages = len(PIPELINE_STAGES)
        for index, stage_name in enumerate(PIPELINE_STAGES, start=1):
            # The default review gate must stop before the expensive render;
            # it must never auto-confirm a human decision.
            if stage_name == "render" and self.gate in {"REVIEW", "MANUAL"} and not self.approve_review:
                summary_file = self.project_dir / "work" / "edit_summary.md"
                print("\n==================================================")
                print(" [HUMAN REVIEW GATE] Please check edit plan summary:")
                print(f" Summary Path: {summary_file}")
                print(" Pipeline paused. Re-run with --approve-review (or --gate AUTO) after review.")
                print("==================================================\n")
                self.state["review_pending"] = True
                self.state["review_summary"] = str(summary_file)
                save_pipeline_state(self.project_dir, self.state)
                self.emit_progress("review", "pending", "請先檢查剪輯摘要")
                return 3

            if self.is_stage_completed(stage_name):
                print(f"[{index}/{total_stages}] {stage_name.capitalize()}: Reused (already completed in prior run).")
                continue

            self.mark_stage_start(stage_name)
            try:
                runner = stage_runners[stage_name]
                runner()
                self.mark_stage_complete(stage_name)
                self.emit_progress(stage_name, "completed", f"{stage_name} 完成")
            except PipelineConsentPending as exc:
                if self.state.get("stages", {}).get(stage_name, {}).get("status") != "pending":
                    self.mark_stage_pending(stage_name, str(exc))
                print(f"\n[CONSENT GATE] {exc}", file=sys.stderr)
                return 3
            except Exception as exc:
                self.mark_stage_failed(stage_name, str(exc))
                print(f"\nERROR in stage '{stage_name}': {exc}", file=sys.stderr)
                print(f"State saved in {self.project_dir / 'work' / STATE_FILE_NAME}. Resolve and resume with --resume.", file=sys.stderr)
                return 2

        self.state.pop("review_pending", None)
        save_pipeline_state(self.project_dir, self.state)

        print("\n==================================================")
        print(" Pipeline Completed Successfully!")
        print(f" Outputs: {self.project_dir / 'outputs'}")
        print(f" QA Report: {self.project_dir / 'outputs' / 'qa_report.json'}")
        print("==================================================")
        return 0
