---
name: video-factory
description: "Create and revise reusable photo/video edits from a project job.yaml using profile-guided story planning, cached local media preparation, an edit plan, Remotion, FFmpeg, subtitles, audio, and QA."
metadata:
  short-description: "Build and revise profile-guided videos"
---

# Video Factory

Use this skill when the user invokes $video-factory or asks to create/revise a video project with this system. Antigravity/Gemini is the editorial director; this skill exposes the application service and deterministic processing layer. Mac M4 handles media preparation and final rendering. Colab is an ephemeral perception worker, not a render host. Use the official Google Colab CLI for approved GPU inference. Do not use CapCut as the core engine.

## Start

1. Identify the video project from the user's path/current workspace. Resolve the installed Skill directory as the directory containing this SKILL.md.
2. If job.yaml is absent, inspect assets first and create job.draft.yaml from templates/job.template.yaml with only well-supported defaults. Summarize the asset inventory and ask only the few decisions that affect profile, goal/audience, duration, aspect ratio, privacy/cloud analysis, or narration. Do not render until the user confirms the draft.
3. Read job.yaml and the matching profiles/<profile>.yaml. Resolve the family or health-education duration preset (short/standard/full) and privacy mode. Never guess whether the job is memory or public-health. Reject unknown profiles and preserve any user-authored job fields.
4. Initialize only missing project folders/files using scripts/video_factory.py. Never overwrite existing project files or touch source assets.

## Production workflow

1. Run local inspection and cache reusable metadata: python3 <SKILL_DIR>/scripts/video_factory.py inspect <PROJECT>. Check the manifest and any errors.
2. Build proxy media, scene candidates, representative frames and contact sheets on the Mac. Keep every derivative under project work/; do not use Colab for routine scene detection or ordinary FFmpeg work.
3. If spoken words are needed, extract only the selected audio locally. Check the job's cloud-processing setting and read references/colab.md. On a cache miss, explain which exact derived files will be sent to Google Colab and ask for authorization for this run. Use `colab-perception` for BALANCED/MAX_QUALITY speech, visual embeddings, clustering and shortlist-only temporal analysis; use `colab-transcribe` for a speech-only operation. Default to T4 and `large-v3-turbo`. Run with a dry-run/index build first. The worker stops its own Colab session after normal success, failure or Ctrl-C; after a force-kill, inspect sessions with `--logtostderr` and stop only the session owned by that job. Never upload the whole assets directory or mount Drive for routine perception.
4. Use the existing OpenAI `transcribe` adapter only when the user explicitly chooses OpenAI for this run; it is not an automatic fallback for a Colab failure. Before any other external model or paid API receives images/audio/text, check job privacy settings and obtain clear authorization. Local preprocessing does not authorize upload. Never log secrets.
5. Build `work/perception_index.json` from transcript, speech turns, anonymous speaker labels, configured audio events/importance metadata, scene semantics, duplicate clusters, visual embedding references, temporal results and confidence. Preserve explicit `not_configured` or `pending` states for any unavailable component; do not claim a field is complete because it exists in the schema. Cache every perception component by source hash + model/version + parameters. Perception returns evidence and candidates only; it never decides that an embedding score means “keep”.
6. Write story_plan.json before editing. Then write work/edit-plan/edit_plan.json as the only render decision source. Keep source paths project-relative, times in seconds, and all captions, annotations, crops, transitions, audio choices, and narration explicit.
7. Validate the plan: python3 <SKILL_DIR>/scripts/video_factory.py validate-plan <PROJECT>. Resolve every ERROR before rendering.
8. Use the bundled Remotion runtime and official Remotion Agent Skills when they are installed and relevant. If render dependencies are missing, install only in the Skill's runtime/remotion directory using the project's package manager; do not change global tool configuration. Prepare referenced files under work/render-public via `prepare-render`, then render with Remotion to outputs/ on the Mac. Set `VIDEO_FACTORY_PUBLIC_DIR` to `<PROJECT>/work/render-public` and pass `<PROJECT>/work/render-input.json` as `--props`. Keep final render and encoding local; do not spend Colab GPU time on final video encoding.
   - For a selected HEIC photo, `prepare-render` creates a verified, source-hash-keyed JPEG under `work/render-public/assets/`; the HEIC original stays unchanged. The derivative is capped at 4096 pixels on its longest edge and validated by decoding it before Remotion uses it.
   - When a portrait edit uses `contain` for a wide group photo, keep the full photo visible and extend the canvas with a softly blurred, darkened copy of that same photo.
9. Export SRT when requested. TTS is not implemented yet; do not synthesize narration. Run the music requirement/licensing pipeline and write `work/music_requirements.json`, its `outputs/work/` mirror, `work/music_attribution.json`, `outputs/final/music_attribution.json`, `outputs/MUSIC_CREDITS.txt` and `outputs/PUBLISHING_CREDITS.txt`. SAFE_AUTO accepts only Public Domain, CC0 and CC BY with upstream evidence; a provider failure finishes without music. Mix music under speech and important natural audio; do not invent a license.
10. Run automated QA and create qa_report.md and review_notes.md. Separate ERROR, WARNING and SUGGESTION. A successful render alone does not establish that the video looks or sounds correct; inspect the rendered video/contact sheet and note issues requiring human review.

## Colab and interactive tools

- Use the official `google-colab-cli` command for repeatable batch inference. Request T4 by default for speech, diarization, SigLIP2, embeddings and clustering. Use L4 only for MAX_QUALITY shortlist-only temporal analysis with a measured VRAM/benchmark reason. Never automatically request A100, H100 or G4.
- **Never allocate a premium GPU when the task can reasonably be completed on CPU or T4.**
- **Always release Colab compute after the GPU stage, including failure paths.**
- Use the official Colab MCP only for interactive notebook inspection or debugging when its notebook tools are available in the current Codex session. If the connection exposes only its bootstrap tool, switch to the CLI instead of repeatedly reconnecting.
- Use Computer Use for browser/OAuth account selection, GUI-only Colab inspection, and visual QA. Do not use it to edit a video timeline or replace deterministic CLI work.
- Read [references/colab.md](references/colab.md) before installing, authenticating, or allocating Colab compute. Do not install tools or change account configuration as an implicit part of an ordinary edit.
- Privacy modes are explicit: `LOCAL_ONLY` forbids uploads; `BALANCED` (default) permits only selected audio, representative frames and 360p/480p proxies, while the current routine worker sends selected audio and representative frames only; `MAX_QUALITY` permits 720p proxies only for shortlisted temporal analysis. Original 4K never leaves the Mac.
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
