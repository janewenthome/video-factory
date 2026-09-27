# Video Factory decisions and lessons

Updated: 2026-09-28

This file records reusable project decisions and production pitfalls. Per-project media, transcripts, credentials, and private OAuth details do not belong in this public record.

## Current defaults

- Mac mini M4 is the local media worker: source inspection, proxy generation, scene sampling, available local inference, subtitles, FFmpeg, Remotion, final encoding, and QA stay on the Mac. Antigravity/Gemini owns final story and clip decisions; Codex coordinates and validates. Colab remains deferred.
- A new job uses `LOCAL_ONLY`, no narration, and automatic open-licensed background music. The opening Codex conversation guides the user to select `memory` or `public-health`, duration, aspect ratio, topic, and project path. It does not ask about narration or cloud processing.
- Preserve every original asset. Put source copies in the project; keep generated media and HEIC-to-JPEG conversions under `work/` or `outputs/`.
- Transcribe only retained clips that contain speech. Review candidate meaning in context; omit meaningless or uncertain captions rather than guessing. A model transcript is evidence, not approved copy.

## Automatic music and natural sound

- Codex derives search terms from the project topic and story, searches Openverse, selects a fitting eligible track, downloads it, and records attribution as part of the ordinary edit workflow. Users should not need to pick a track first. A job or direct user request can disable music or provide a chosen track.
- Openverse aggregates third-party metadata and does not verify each work's rights. Check the original source page before use. Skip any song whose work, license, or attribution requirements cannot be verified; never treat an Openverse license label alone as proof.
- The music bed covers the full timeline. Keep natural sound at its original gain (normally 1.0). Duck only music to 24% of its base gain, ramping down over approximately 0.25 seconds at natural-sound entry and restoring over approximately 0.25 seconds after it ends. The current Remotion renderer implements these ramps.
- A licensed track search/download is different from permission to upload user media: search only with topic/story terms, and keep photos, video, audio, and transcripts on the Mac.

## Pitfalls discovered

| Pitfall | Decision |
|---|---|
| Colab MCP can connect before its dynamic notebook tools become visible to Codex. Creating a notebook does not fix a client that misses `notifications/tools/list_changed`. | Keep Colab deferred. Do not claim a live GPU run, Compute Unit measurement, model benchmark, or quality gain without evidence from an actual run. |
| A locally cached Whisper model may differ from the latest or fastest model. A cache miss can unexpectedly download gigabytes. | Reuse the existing cached MLX Whisper snapshot; default to cache-only and leave captions pending on a miss. Do not download or switch to a remote provider automatically. |
| A converter can report success while writing an unusable/black HEIC derivative. | Prefer a working HEIC decoder and decode-verify each derivative before Remotion uses it. Keep the HEIC original unchanged. |
| Remotion/Chromium may reject certain MOV decoders or fail under a restricted process sandbox. | Use the existing `<OffthreadVideo>` fallback for unsupported MOV decoding, diagnose sandbox permission failures separately from media damage, then inspect the encoded output with ffprobe/FFmpeg. |
| A render, schema check, or sampled contact sheet is not a full review of the finished video. | Report technical QA separately from full visual/audio QA. Only claim full review after actually watching and listening to the output. |
| ASR can produce plausible but contradictory phrases on short noisy natural audio. | Use transcript candidates only after story selection; keep only semantically meaningful speech after review, and leave low-confidence wording out until a person confirms it. |

## Reusable references

- [Video Factory Skill](../skills/video-factory/SKILL.md) defines the guided start and production procedure.
- [Architecture](ARCHITECTURE.md) documents role boundaries, local processing, captions, and the current audio mix.
