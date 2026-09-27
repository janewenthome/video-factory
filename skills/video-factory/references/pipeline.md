# Pipeline and file contracts

## Project contract

Each project contains job.yaml and assets/; work/ stores reconstructable derivatives and outputs/ stores deliverables:

- work/manifests/media_manifest.json: media metadata, relative source path, stable content hash and probe status. JPEG/HEIC/HEIF include locally read EXIF capture time/GPS and IPTC labels when available; assets are ordered by captured time, with timezone uncertainty recorded.
- work/frames and work/contact-sheets: representative local previews.
- work/transcripts/<cache-key>.json and .srt: timestamped transcription response and subtitle export, keyed by source SHA-256 plus provider/model/language/configuration. Colab results also include the `faster-whisper` package version in their cache fingerprint. Plan-based captions export to `work/transcripts/captions.srt`.
- work/analysis: asset/scene summaries and profile scores tied to source hashes.
- work/analysis/story_plan.json: opening, progression, key moments, peak, ending, estimated timing and music arc.
- work/perception_index.json: cache-keyed speech, visual, duplicate/event, photo metadata, OCR and shortlist-temporal evidence. It is input to the director, never the final keep/drop decision.
- work/perception-cache/: content-addressed perception results keyed by source hash + model/version + parameters; local image OCR is cached independently so Family duration variants reuse it.
- work/music_requirements.json and work/music_attribution.json: music discovery and license evidence; mirrors are kept under outputs/work/ and outputs/final/ for delivery.
- work/edit-plan/edit_plan.json: authoritative ordered timeline and rendering decisions.
- work/render-input.json and work/render-public/assets/: validated Remotion props and hash-verified copies of referenced source files.
- outputs/<chosen-name>.mp4, qa_report.md, review_notes.md, MUSIC_CREDITS.txt, PUBLISHING_CREDITS.txt; final music attribution is also under outputs/final/.

All media source values are POSIX paths relative to the project root. Reject .., absolute paths and symlink targets outside the project. Timeline and source in/out values are seconds. A render helper maps source paths to stable names under work/render-public and passes those URLs to the Remotion composition.

## Incremental work

Hash each source once and cache probe, scene, frame, transcription, OCR, perception and render-copy results by content hash plus the preprocessing configuration. Reuse matching cache entries. A job/story/edit-plan change should redo planning/render only. A caption change should redo caption composition/render. If a future TTS adapter is added, narration text changes should redo TTS, mix and render. A source content change invalidates only that source's dependent derivatives and any story/edit decision that used it. Colab receives only derived files permitted by the privacy mode; its runtime is temporary and is not a cache store. GPS and OCR remain local under the current workflow.

Never erase cache/output directories as part of a normal revision. Write reports atomically; use a new render filename or an explicit replacement decision for an existing deliverable.

## Edit plan

The top-level plan identifies version, profile, title, canvas/fps/duration and a timeline array. Entries use type; visual/audio entries have a project-relative source where required; source and timeline times use seconds. Keep the edit plan understandable and auditable: include IDs, purpose/reason, caption/annotation text, crop/focus, audio levels, transition and factual provenance where applicable.

Text, subtitles, narration and music can be separate timeline entries. If one claim or caption is derived from spoken words or a source document, retain its source reference and timestamp/location when available. Renderer code should implement visual/audio behavior only; it must not select footage or invent narrative content.
