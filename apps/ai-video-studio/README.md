# AI Video Studio GUI contract

This directory describes the future Tauri 2 thin shell. It is intentionally
not a second media engine: the shell chooses a project folder and calls
`video_editor.application.VideoFactoryApplication` through a local adapter.
It never copies a multi-gigabyte source library into the GUI process and never
parses CLI stdout.

## Home controls

- source folder picker
- after picking, show folder name, photo/video counts, total video duration and estimated source size via `inspect_source`
- Family or Health Education
- Short / Standard / Full / custom duration
- music: auto openly licensed, project music, local licensed music, or none
- review mode: AUTO (default creates the first cut), REVIEW, or MANUAL
- AI quality: standard or deep
- each render keeps a version under `outputs/versions/`
- advanced: privacy mode, cloud perception, GPU policy

Normal status text stays human-facing (`正在理解影片內容`, `正在分析語音`,
`正在整理相似畫面`, `正在尋找精彩片段`, `正在分析配樂需求`,
`正在確認配樂授權`). Technical model names and GPU details are advanced
settings only.

## Current integration point

```python
from video_editor.application import ProjectOptions, VideoFactoryApplication

app = VideoFactoryApplication(project_dir, ProjectOptions(mode="family"), on_progress=emit)
status = app.run()
```

The current GUI contract is complete enough for a desktop shell to be added
without moving deterministic media work or changing the edit-plan contract.

## Multi-project queue

`video_editor.application.VideoFactoryQueue` exposes the same persistent queue
used by the CLI. A future GUI should show pending/running/completed/failed jobs,
the current project and log location, and support add/start-all/pause/resume/
cancel/reorder. The queue runner and shared heavy-job lock enforce one local
pipeline at a time. Pausing waits for the current project to exit; cancelling a
running project stops and reaps its process group before another job starts.

Only a prepared project can be queued: `job.yaml`, a Codex-authored
`work/analysis/story_plan.json`, and a validated `work/edit-plan/edit_plan.json`
must exist first. The runner stays local-only. It samples macOS `memory_pressure`
and Swapouts, cools down between projects, and pauses if readings remain unsafe
or unavailable. Queue UI controls are specified here but the Tauri GUI itself
has not been implemented.

For batches, submit the project paths in one Codex task and prepare/enqueue them
sequentially. Do not run a separate heavy Codex task for every project: the
queue lock cannot limit memory used by other Codex/ChatGPT tasks or Ollama.
