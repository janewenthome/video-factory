---
name: video-factory
description: "Create and revise reusable photo/video edits from a project job.yaml using profile-guided story planning, cached local media preparation, an edit plan, Remotion, FFmpeg, subtitles, audio, and QA."
metadata:
  short-description: "Build and revise profile-guided videos"
---

# Video Factory

Use this skill when the user invokes $video-factory or asks to create/revise a video project with this system. Antigravity/Gemini is the editorial director and owns final story choices; Codex coordinates and validates the workflow; Mac mini M4 handles local media preparation, supported local speech inference and final rendering. Claude Opus is an optional deep reviewer. Colab is an optional architecture extension, not part of the current execution path. Defer Colab MCP until Codex can reliably refresh dynamically added tools. Do not use CapCut as the core engine.

## Start

1. Identify the video project from the user's path/current workspace. Resolve the installed Skill directory as the directory containing this SKILL.md.
2. If job.yaml is absent, inspect assets first and create job.draft.yaml from templates/job.template.yaml with only well-supported defaults. Summarize the asset inventory and ask only the few decisions that affect profile, goal/audience, duration, aspect ratio, privacy/cloud analysis, or narration. Do not render until the user confirms the draft.
3. Read job.yaml and the matching profiles/<profile>.yaml. Resolve the family or health-education duration preset (short/standard/full) and privacy mode. Never guess whether the job is memory or public-health. Reject unknown profiles and preserve any user-authored job fields.
4. Initialize only missing project folders/files using scripts/video_factory.py. Never overwrite existing project files or touch source assets.

## Production workflow

1. Run local inspection and cache reusable metadata: python3 <SKILL_DIR>/scripts/video_factory.py inspect <PROJECT>. Check the manifest and any errors.
2. Build proxy media, scene candidates, representative frames and contact sheets on the Mac. Keep every derivative under project work/; do not use Colab for routine scene detection or ordinary FFmpeg work.
3. Build the local perception index from available metadata and visual/audio evidence. Preserve explicit `not_configured` or `pending` states for unavailable components; do not claim a field is complete because it exists in the schema. Cache each component by source hash + model/version + parameters. Perception returns evidence and candidates only; it never decides that an embedding score means “keep”.
4. Write story_plan.json before editing. Then write work/edit-plan/edit_plan.json as the only render decision source. Keep source paths project-relative, times in seconds, and all captions, annotations, crops, transitions, audio choices, and narration explicit.
5. After the edit plan identifies retained clips, inspect their retained audio for speech. When speech is present, create local transcription candidates with the configured MLX Whisper backend on supported Apple Silicon. Transcribe the retained/cut audio after story selection, then map timestamps to the final timeline; cache by source hash, model/version and parameters. Do not send audio to a cloud service or silently fall back to one. If local inference is unavailable, report that and leave subtitle cues pending.
   - Use `draft-captions --model <MODEL> --model-revision <SNAPSHOT>` to reuse a specific cached model. The command and full pipeline are local-cache-only by default; add `--allow-model-download` only when the user explicitly approves downloading missing weights.
6. Have the selected editorial director or the user review each subtitle candidate for meaning in context. Include meaningful speech; omit cues judged to be meaningless, noise, hallucination or unintelligible fragments. Do not invent or rewrite content to make it clearer. If meaning is uncertain, flag it for human review rather than guessing. Codex records the explicit keep/drop decisions and validates cue timing, overlaps and SRT bounds before rendering.
7. External transcription or other paid APIs are outside the default workflow. Use one only when the user explicitly selects that provider for this run and authorizes the exact data transfer. Local preprocessing does not authorize upload. Never log secrets.
8. Validate the plan: python3 <SKILL_DIR>/scripts/video_factory.py validate-plan <PROJECT>. Resolve every ERROR before rendering.
9. Use the bundled Remotion runtime and official Remotion Agent Skills when they are installed and relevant. If render dependencies are missing, install only in the Skill's runtime/remotion directory using the project's package manager; do not change global tool configuration. Prepare referenced files under work/render-public via `prepare-render`, then render with Remotion to outputs/ on the Mac. Set `VIDEO_FACTORY_PUBLIC_DIR` to `<PROJECT>/work/render-public` and pass `<PROJECT>/work/render-input.json` as `--props`. Keep final render and encoding local.
   - For a selected HEIC photo, `prepare-render` creates a verified, source-hash-keyed JPEG under `work/render-public/assets/`; the HEIC original stays unchanged. The derivative is capped at 4096 pixels on its longest edge and validated by decoding it before Remotion uses it.
   - When a portrait edit uses `contain` for a wide group photo, keep the full photo visible and extend the canvas with a softly blurred, darkened copy of that same photo.
10. Export/validate SRT when requested or when meaningful speech was detected in retained audio. TTS is not implemented yet; do not synthesize narration. Run the music requirement/licensing pipeline and write `work/music_requirements.json`, its `outputs/work/` mirror, `work/music_attribution.json`, `outputs/final/music_attribution.json`, `outputs/MUSIC_CREDITS.txt` and `outputs/PUBLISHING_CREDITS.txt`. SAFE_AUTO accepts only Public Domain, CC0 and CC BY with upstream evidence; a provider failure finishes without music. Mix music under speech and important natural audio; do not invent a license.
11. Run automated QA and create qa_report.md and review_notes.md. Separate ERROR, WARNING and SUGGESTION. A successful render alone does not establish that the video looks or sounds correct; inspect the rendered video/contact sheet and note issues requiring human review.

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
- health-education: prioritize clarity and factual accuracy. Every medical claim in narration/text must trace to user-provided material or project references. Flag unsupported claims instead of filling gaps from model memory. Always include captions and a references/privacy review. Factual completeness may justify a duration overrun, which must be explained in `edit_summary.md`.
- For 16:9 and 9:16 outputs, make ratio-aware framing/layout decisions independently. Warn when a crop could remove a person or key information.
- Never modify, rename, move or overwrite source assets. Derived files belong only in project work/ or outputs/.
- Read references/pipeline.md for file contracts and cache behavior. Read references/privacy.md and references/colab.md before any cloud/media transfer.

## Completion

Report created/changed files, render/QA evidence, required human review, external APIs/data transfers/costs, and remaining limitations. Report the Colab accelerator/session cleanup status when used. Do not claim an unrun stage or uninspected output is complete.
