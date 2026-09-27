# Colab inference

Use Google Colab as a short-lived **Perception Worker** for selected AI inference. Keep proxy creation, metadata, hashing, scene detection, contact sheets, story/edit decisions, and final rendering on the Mac. The worker returns evidence and candidates; Antigravity/Gemini remains the editorial director.

## Account and CLI setup

Install the official Google CLI only when the user asks to set up Colab:

```bash
uv tool install google-colab-cli
colab --logtostderr version
```

On first use, the CLI may request Google authorization. In the browser, select the Google account intended for this project and verify it before granting access. The authorization code is a secret: enter it only in the local terminal, never in a prompt or project file. CLI credentials are stored outside this project; do not inspect, print, copy, or commit them. `colab auth` authenticates code running inside a VM for Google Cloud services; it does not sign the local CLI into Colab.

To complete the first CLI sign-in without allocating a runtime, run `colab --logtostderr --auth=oauth2 sessions`; this lists active sessions after authorization. Computer Use may open the URL printed by the CLI and help select the intended account; paste the resulting code directly into the local terminal yourself. If the intended account is not available in the browser, stop and resolve the Google account selection before proceeding. Do not send the code to Codex.

Before allocating a runtime, inspect usage and sessions. The project adapter passes `--logtostderr` to prevent debug records from being written to the CLI's persistent debug log, and it suppresses raw subprocess output. Do not inspect or print the CLI debug log: it may contain sensitive authorization or runtime data. A paid subscription does not guarantee a particular accelerator or availability. Treat usage/rate output as the source for current account limits; do not infer a dollar amount or monthly allowance. After an authorized GPU job, inspect usage again and confirm the worker-owned session is stopped.

The official CLI supports macOS and provides `new`, `upload`, `install`, `exec`, `download`, `stop`, and `usage` commands. Use the project adapter rather than hand-running a session sequence during ordinary edits:

```bash
python3 <SKILL_DIR>/scripts/video_factory.py colab-transcribe <PROJECT> \
  --source work/transcripts/audio/<hash>.flac \
  --language zh --gpu T4 --dry-run
```

After reviewing the exact source path, size, destination and plan, ask for permission for this run if there is no valid cached transcript. Only then rerun with `--allow-upload`:

```bash
python3 <SKILL_DIR>/scripts/video_factory.py colab-transcribe <PROJECT> \
  --source work/transcripts/audio/<hash>.flac \
  --language zh --gpu T4 --allow-upload
```

For the combined worker, build and inspect the local index first:

```bash
python3 <SKILL_DIR>/scripts/video_factory.py perception <PROJECT> --privacy-mode BALANCED
python3 <SKILL_DIR>/scripts/video_factory.py colab-perception <PROJECT> \
  --privacy-mode BALANCED --gpu T4 --allow-upload
```

`LOCAL_ONLY` never calls the cloud adapter. `BALANCED` may transfer only
selected extracted audio, representative frames and 360p/480p proxies.
`MAX_QUALITY` stage one transfers selected derived audio and representative frames. It may transfer 720p proxies only in a separate second stage, after a director shortlist and a new exact-file review.
Original 4K never leaves the Mac. The worker stores its result in
`work/perception_index.json`; all component caches include source hash,
model/version and parameters.

Use `large-v3-turbo` by default. Request T4 first. Use L4 only when a representative benchmark shows a meaningful total-time or cost benefit. Do not automatically request A100, H100 or G4. Word timestamps are useful when captions or fine-grained speech edits need them. Reuse a cache entry keyed by input hash and inference configuration.

The registered visual model is pinned SigLIP2; its live GPU performance still needs a verified run. Anonymous diarization uses SpeechBrain ECAPA utterance embeddings and conservative clip-local clustering. It does not identify a person, handle overlapping speakers or provide calibrated confidence; short/ambiguous segments remain `unknown`. SmolVLM2 is the currently registered temporal backend behind a backend registry; it is not yet live-benchmarked and its event descriptions remain review evidence, not edit decisions. Audio event detection and speech importance are not configured yet.

## Data and cleanup

- Upload only derived files listed in the local perception index. Do not upload the full project, source video, photo library, or references.
- Do not mount Google Drive for routine jobs. Download the timestamped JSON result into `work/transcripts/`; the runtime is not a cache store.
- The adapter creates a uniquely named session and stops only that session in its cleanup path. If the process is force-killed or the host shuts down, inspect `colab --logtostderr sessions` and stop only a session clearly owned by this job. Never stop unrelated user sessions.
- If a CLI command fails, report the failure and current session state. Do not silently retry on a more expensive GPU or switch to a different cloud provider.
- Keep OpenAI transcription as a separate, user-selected provider. A Colab failure does not authorize uploading the same audio to OpenAI.
- Use T4 for VAD, Whisper, diarization, SigLIP2, embeddings and clustering. L4 is reserved for shortlist-only temporal workloads with a measured reason; A100/H100/G4 are never automatically requested.

## MCP and Computer Use

The official [Google Colab MCP server](https://github.com/googlecolab/colab-mcp) is for interactive notebook inspection and debugging. Its documented launcher is `uvx git+https://github.com/googlecolab/colab-mcp`; follow the upstream client configuration when the user asks to register it. It requires the client to support dynamic tool-list updates (`notifications/tools/list_changed`). If the current Codex session does not expose notebook tools after connecting, use the CLI; do not repeatedly reconnect or install a community fork without an explicit request.

Use Computer Use only for browser-dependent steps such as selecting the intended Google account during OAuth, inspecting Colab's GUI/runtime controls, and visual QA. Do not use it to drag clips on a timeline or as the batch execution path. OAuth codes and tokens must stay out of chat, logs, notebook outputs and repository files.

## Official references

- [Google Colab CLI](https://github.com/googlecolab/google-colab-cli)
- [Google Colab MCP](https://github.com/googlecolab/colab-mcp)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
