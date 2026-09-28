---
name: video-factory
description: "Guide video projects from start/開始 through automatic, local-first first-cut planning, licensed music selection, rendering, QA, and feedback-driven revisions."
metadata:
  short-description: "Build and revise profile-guided videos"
---

# Video Factory

Use this skill when the user invokes $video-factory, asks how to start a video in this repository, or asks to create/revise a project. In a new project conversation, short openers such as `start`, `開始`, `開工`, or a general “how do I begin?” request should show the guided start below. When the user has given enough information to produce a video, Codex creates the first-cut story/edit plans, renders locally, and reports review notes without waiting for plan approval. The user can request revisions after viewing it. Antigravity/Gemini and Claude Opus are optional reviewers, not required workflow gates. Mac mini M4 handles local media preparation, supported local speech inference and final rendering. Colab is deferred and is not part of the current execution path. Do not use CapCut as the core engine.

When the current workspace is `/Volumes/2TB/program/video_factory`, use the repository-managed `skills/video-factory/` directory as `<SKILL_DIR>` for scripts, schemas and references. Treat the installed global copy as the discovery entry; its bundled scripts may lag behind this repository.

## Start and guided onboarding

When the user starts a new video conversation without enough details to act, show this guide and ask only for missing project specifics:

1. Choose a profile: family, travel, and life memories use `memory`; medical and health education use `public-health` (and need citable reference material).
2. Show the matching initializer. For a family example:

   ```bash
   cd /Volumes/2TB/program/video_factory
   python3 skills/video-factory/scripts/video_factory.py init projects/first-edit --profile memory
   ```

   For health education, change `memory` to `public-health`. Initialization creates the project folders and a `job.draft.yaml`; it does not replace an existing project.
3. Source photos and videos may be placed in `assets/photos/` and `assets/videos/`, or directly in the project root. Inventory all supported media already inside the project; do not ask the user to choose a batch or copy media that is already there. Keep originals unchanged. For sources outside the project, ask for the project path or copy only the intended inputs into it.
4. Ask only for missing information: video type/profile, target duration, aspect ratio, topic or moments to keep, and project path. Example: `家庭、60 秒、9:16；中秋團聚，保留生態瓶、烤肉、撈蛤蠣烤蚵與團照；素材在 /Volumes/2TB/program/video_factory/projects/first-edit`.

Do not ask whether the user wants narration; it defaults to none. Do not bring up disabling cloud processing in the ordinary start guide; the supported production workflow is local. If the user already supplied an answer, use it rather than asking again.

For a project already in progress, identify it from the user's path/current workspace and read its job and plans. Resolve the installed Skill directory as the directory containing this SKILL.md. If `job.yaml` is absent and the request plus existing `job.draft.yaml` provide enough details, promote a copy of the draft to `job.yaml` without overwriting either file; otherwise create the draft and ask only for missing details. Read the matching `profiles/<profile>.yaml`; never guess between `memory` and `public-health`, and preserve user-authored fields. Initialize only missing project files. Once the brief is sufficient, do not pause for approval of the job, story plan, edit plan, or subtitle/music choices: create and render a local first cut, then let the user revise it.

## Production workflow

1. Run local inspection and cache reusable metadata: `python3 <SKILL_DIR>/scripts/video_factory.py inspect <PROJECT>`. Inspection scans `assets/` recursively and supported image/video files directly in the project root. Exact-content duplicates are collapsed by SHA-256 with alternate paths recorded as `duplicate_sources`. Check the manifest, chronology and any errors; do not move or copy root-level inputs just to make them visible to the scanner.
2. Build proxy media, scene candidates, representative frames and contact sheets on the Mac. Keep every derivative under project work/; deterministic media preparation, proxy generation, FFmpeg work and final encoding stay local.
3. For JPEG/HEIC/HEIF, `inspect` reads embedded EXIF capture time, timezone, GPS and IPTC place labels with local Apple ImageIO; keep GPS in private project metadata and never reverse-geocode it. The manifest orders assets by parseable embedded capture time; it preserves path order for ties/untimed items and flags timezone-free timestamps as approximate. `perception` runs local Apple Vision OCR over still photos and representative video frames, with text, box, language, confidence, source hash and engine provenance. If Apple Vision is unavailable, mark OCR pending; do not install another OCR package implicitly.
4. Build the local perception index from metadata, OCR, transcript, representative frames and audio evidence. Preserve explicit `not_configured` or `pending` states for unavailable components; do not claim a field is complete because it exists in the schema. Cache each component by source hash + model/version + parameters. Perception returns evidence and candidates only; it never decides that an embedding score means “keep”.
5. Use capture-time order as a starting hypothesis for event progression, then compare it with representative images, within-video frame timestamps, scene changes, speech and the user's description. Treat missing/offset-free timestamps as uncertain; never let filename or file-modified time masquerade as capture time. The editorial director may depart from chronology when the story has a supported reason.
6. Keep text evidence distinct: OCR is literal text visible in an image; ASR is a best-effort transcript of spoken audio; an annotation is a separate editorial suggestion. Record annotation candidates in `work/analysis/asset-analysis.json` using `templates/asset-analysis.schema.json`, with evidence references and confidence. Add a concise, supported annotation directly to the first-cut plan when evidence is strong; omit weak candidates or retain them as pending notes. Do not invent identities, event details, translations or venue names. Never display precise GPS coordinates as on-screen text.
7. Write story_plan.json before editing. Then write work/edit-plan/edit_plan.json as the only render decision source. Keep source paths project-relative, times in seconds, and all captions, annotations, crops, transitions, audio choices, and narration explicit.
8. After the edit plan identifies retained clips, inspect their retained audio for speech. When speech is present, create local transcription candidates with the configured MLX Whisper backend on supported Apple Silicon. Transcribe the retained/cut audio after story selection, then map timestamps to the final timeline; cache by source hash, model/version and parameters. Keep word probabilities and segment signals such as `avg_logprob`, `no_speech_prob`, `temperature` and `compression_ratio` with candidates when available. These are uncalibrated clues, not automatic keep/drop thresholds. Codex judges whether each cue is meaningful in context: include clear, meaningful speech; omit obvious noise, hallucinations and meaningless fragments; leave uncertain wording out of the first cut and explain it in review notes. Do not rewrite speech. Do not send audio to a cloud service or silently fall back to one. If local inference is unavailable, render without those captions and report the limitation.
   - Use `draft-captions --model <MODEL> --model-revision <SNAPSHOT>` to reuse a specific cached model. The command and full pipeline are local-cache-only by default; add `--allow-model-download` only when the user explicitly approves downloading missing weights.
9. Codex records caption keep/drop decisions and reasons in the existing review artifact, then exports and validates SRT timing, overlaps and bounds. This is an internal first-cut quality step; do not wait for the user or another model to approve it. If meaning is uncertain, omit that cue and note the uncertainty for later revision. Narration defaults to none; do not ask whether narration is wanted or synthesize it unless explicitly requested and supported.
10. External transcription or other paid APIs are outside the default workflow. Use one only when the user explicitly selects that provider for this run and authorizes the exact data transfer. Local preprocessing does not authorize upload. Never log secrets.
11. Unless the user or job explicitly turns music off or supplies a chosen track, automatic background music is on. Derive search terms from the video topic, story plan and mood; use Openverse's catalog to find candidates, choose a fitting track, and add it without asking the user to choose. Before download/use, open the original source page and verify the track and license. Use only eligible Public Domain, CC0 or CC BY music with the required attribution recorded; if the upstream license or source cannot be verified, skip that candidate. If no eligible track remains, finish the video without music and explain why; do not block the first cut. Never treat Openverse metadata alone as proof of rights. Write `work/music_requirements.json`, its `outputs/work/` mirror, `work/music_attribution.json`, `outputs/final/music_attribution.json`, `outputs/MUSIC_CREDITS.txt` and `outputs/PUBLISHING_CREDITS.txt`. A direct `video_editor music search` CLI call is a manual utility; this automatic selection behavior belongs to the Codex Skill workflow.
12. A selected background track spans the full video timeline. Preserve retained natural audio at its original gain (normally 1.0); never duck, normalize, or lower it to make room for music. While natural sound is present, smoothly lower only the music to 24% of its base gain over about 0.25 seconds, then restore it over about 0.25 seconds after the natural-sound segment. Keep music under intelligible speech. The current Remotion renderer implements these ramps.
13. Validate the complete plan after captions and music are applied: `python3 <SKILL_DIR>/scripts/video_factory.py validate-plan <PROJECT>`. Resolve every ERROR before rendering.
14. Use the bundled Remotion runtime and official Remotion Agent Skills when they are installed and relevant. If render dependencies are missing, install only in the Skill's runtime/remotion directory using the project's package manager; do not change global tool configuration. Prepare referenced files under work/render-public via `prepare-render`, then render with Remotion to outputs/ on the Mac. Set `VIDEO_FACTORY_PUBLIC_DIR` to `<PROJECT>/work/render-public` and pass `<PROJECT>/work/render-input.json` as `--props`. Keep final render and encoding local.
   - For a selected HEIC photo, `prepare-render` creates a verified, source-hash-keyed JPEG under `work/render-public/assets/`; the HEIC original stays unchanged. The derivative is capped at 4096 pixels on its longest edge and validated by decoding it before Remotion uses it.
   - When a portrait edit uses `contain` for a wide group photo, keep the full photo visible and extend the canvas with a softly blurred, darkened copy of that same photo.
15. Run automated QA and create qa_report.md and review_notes.md. Separate ERROR, WARNING and SUGGESTION. Inspect the rendered video/contact sheet and fix clear technical failures before delivery; include unresolved visual, factual or caption uncertainties in review notes. Deliver the first cut without waiting for approval, and clearly distinguish automated checks from any portion you actually watched.

## Multiple projects and Mac resource safety

Several projects may be prepared and queued together, but local media work must run one project at a time. The persistent queue and the shared heavy-job lock enforce `max_parallel_jobs = 1` for Video Factory CLI and application-service operations in this checkout. Do not start a second proxy, Whisper, perception or render process when the lock is busy; add its project to the queue instead.

For a memory-safe batch, have the user provide all project paths in one Codex task, prepare them sequentially, then run one queue. Do not launch one heavy Codex task per project. The queue controls Video Factory media processes; it does not serialize memory used by separate Codex/ChatGPT tasks or Ollama.

Before adding a project, Codex must finish its `job.yaml`, evidence-based `work/analysis/story_plan.json`, validated `work/edit-plan/edit_plan.json`, subtitle decisions and music/license decision. The queue runner executes that prepared project through the local pipeline; it does not invent the editorial plan or pick music without Codex. `--force` is for an intentional full re-run after revising a completed project.

From the repository root, the CLI workflow is:

```bash
python3 -m video_editor queue add projects/project-a
python3 -m video_editor queue add projects/project-b
python3 -m video_editor queue list
python3 -m video_editor queue run
```

Use `queue pause` to finish the current project and stop before the next one; `queue resume` then `queue run` continues pending work. `queue cancel <JOB_ID>` requests cancellation and waits for that job's process group to stop before another starts. `queue reorder <ID...>` takes every pending job ID once, in the desired order. `queue run --max-jobs N` provides a bounded run; `queue add --max-attempts N` sets the retry limit. A job's full log is retained locally and shown by `queue list`.

The queue persists locally in the ignored `.video-factory/` directory. It records pending/running/completed/failed/cancelled states and retries interrupted work from `work/pipeline_state.json` and content caches. One queue-runner lock prevents duplicate runners; one heavy-job lock also stops direct CLI and application calls from overlapping the queue. Separate clones or worktrees must set the same `VIDEO_FACTORY_QUEUE_FILE` to share this queue and lock.

On Mac, `queue run` samples `/usr/bin/memory_pressure` before starting; subsequent projects get a 45-second default cooldown and before/after samples. If a reading remains unsafe, the runner rechecks every 30 seconds up to three times and then persists a paused queue. It checks free-memory percentage and Swapouts. Free memory below 10%, continuing Swapouts growth, a missing reading, or an unstable sample blocks the next job. Use `queue resume` only after the readings stabilize. The 10% threshold is a conservative heuristic, not Apple's formal pressure level. This gate controls Video Factory jobs; it cannot limit memory used by multiple Codex/ChatGPT sessions, Ollama, or unrelated apps, and one exceptionally large project can still exceed available RAM.

The queue command is the current interface; the Tauri desktop GUI is not implemented yet. `apps/ai-video-studio/gui_contract.json` specifies the future queue controls and structured progress events.

## Colab and interactive tools

- The current supported workflow is Mac-first. Do not allocate Colab sessions, upload project media or request Colab authorization as part of a normal edit.
- Colab CLI/MCP worker code may remain as an optional future adapter. Do not use the MCP until Codex can reliably refresh dynamically added notebook tools; a bootstrap-only connection is not a usable notebook session. Reassess this only after Codex's tool-refresh support changes.
- No live Colab inference, GPU benchmark or quality improvement should be claimed without a separately verified run. Existing schemas/model adapters do not establish that those models ran.
- Use Computer Use only for GUI-dependent actions and visual QA. Do not use it to edit a video timeline or replace deterministic CLI work.
- Read [references/colab.md](references/colab.md) before any explicitly requested future Colab setup. Do not install tools or change account configuration as an implicit part of an ordinary edit.
- GUI code should call `video_editor.application.VideoFactoryApplication` and consume structured progress events such as `正在分析語音`; it must not parse CLI stdout. The CLI remains for development, debug, automation and recovery.

## Revisions

Read the current job, story plan, edit plan, cached analyses and QA report. Map natural-language requests to the smallest affected timeline entries; edit only those entries and update the plan. Reuse unchanged metadata, frames, transcript and analysis. Re-run only relevant preparation, composition, mix and render stages. Do not re-transcribe or re-analyze unchanged inputs.

## Profile and safety rules

- memory: prioritize emotional/story value, people, interactions, unique moments and natural sound. Keep text and transitions restrained; narration defaults to none. A slightly imperfect meaningful moment can beat a technically perfect empty shot.
- health-education: prioritize clarity and factual accuracy. Every medical claim in narration/text must trace to user-provided material or project references. Omit unsupported claims instead of filling gaps from model memory. Always include captions and a references/privacy review; flag identifiable people for review before publication. Factual completeness may justify a duration overrun, which must be explained in `edit_summary.md`. A local first-cut render may be created without approval, but it must not present unsupported claims as facts.
- For 16:9 and 9:16 outputs, make ratio-aware framing/layout decisions independently. Warn when a crop could remove a person or key information.
- Never modify, rename, move or overwrite source assets. Derived files belong only in project work/ or outputs/.
- Read references/pipeline.md for file contracts and cache behavior. Read references/privacy.md and references/colab.md only if the user explicitly requests an external/cloud workflow.

## Completion

Report created/changed files, render/QA evidence, required human review, external APIs/data transfers/costs, and remaining limitations. Report the Colab accelerator/session cleanup status when used. Do not claim an unrun stage or uninspected output is complete.
