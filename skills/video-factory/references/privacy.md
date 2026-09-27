# Media privacy and external services

Family footage can reveal children, private locations and identifying details. Public-health footage can include identifiable patients or staff. Treat source media, frame previews, audio, transcripts and references as private.

- Local ffprobe/FFmpeg operations, hashing, frame extraction and rendering stay on this machine.
- Do not upload source media or derived previews automatically. A user request to make a video does not by itself authorize third-party transfer.
- Before cloud vision/transcription/TTS, check `job.yaml` `cloud_processing` and state which data is sent and to which service. `disabled` forbids transfer; `ask_each_run` requires explicit authorization for the selected data on that run. A project setting or general request to make a video does not replace per-run authorization.
- Perception privacy modes are explicit: `LOCAL_ONLY` sends no audio, frames or video; `BALANCED` sends only selected extracted audio, representative frames and 360p/480p proxies; `MAX_QUALITY` sends 720p proxies only for a transcript/embedding shortlist. Original 4K never leaves the Mac.
- Do not mount Drive. The exact GPU and compute-unit use depend on current account availability; report CLI-reported usage, not guessed money costs. T4 is the routine default; L4 is reserved for shortlist temporal analysis.
- Use the minimum selected frames/audio needed. Keep keys in environment variables; never copy them into job.yaml, logs, reports or rendered files.
- For public-health content, retain source materials locally, trace every factual claim, and flag people/privacy review. Do not infer consent to publish from a file being present.
