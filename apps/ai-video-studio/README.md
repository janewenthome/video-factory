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
- review mode: REVIEW (default), AUTO, or MANUAL
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
