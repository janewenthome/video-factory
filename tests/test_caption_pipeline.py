from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "skills" / "video-factory" / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from video_editor import caption_pipeline
import video_factory


class CaptionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory(prefix="video-factory-caption-pipeline-")
        self.project = Path(self.temp_dir.name) / "project"
        source = self.project / "assets" / "clip.mov"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"synthetic source placeholder")
        self.plan_path = self.project / "work" / "edit-plan" / "edit_plan.json"
        self.plan_path.parent.mkdir(parents=True)
        self.plan = {
            "duration_seconds": 2.0,
            "timeline": [
                {
                    "id": "natural-clip",
                    "type": "video",
                    "source": "assets/clip.mov",
                    "source_in": 4.0,
                    "source_out": 6.0,
                    "timeline_start": 0.0,
                    "timeline_end": 2.0,
                    "audio": {"preserve_natural": True},
                },
                {
                    "id": "muted-clip",
                    "type": "video",
                    "source": "assets/clip.mov",
                    "source_in": 0.0,
                    "source_out": 2.0,
                    "timeline_start": 2.0,
                    "timeline_end": 4.0,
                    "audio": {"preserve_natural": False},
                },
            ],
        }
        self.plan["duration_seconds"] = 4.0
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _draft(self) -> dict[str, object]:
        with (
            patch.object(video_factory, "ffprobe_metadata", return_value={"audio_streams": [{"codec": "aac"}]}),
            patch.object(caption_pipeline, "_audio_cut", return_value=("work/transcripts/audio/cuts/a.flac", "cut-key")),
            patch.object(
                caption_pipeline,
                "transcribe_audio",
                return_value={
                    "text": "大家好。嗯",
                    "language": "zh",
                    "segments": [
                        {
                            "start": 0.1,
                            "end": 0.9,
                            "text": "大家好。",
                            "words": [
                                {"word": "大家", "start": 0.1, "end": 0.4},
                                {"word": "好。", "start": 0.45, "end": 0.9},
                            ],
                        },
                        {
                            "start": 1.1,
                            "end": 1.4,
                            "text": "嗯",
                            "words": [{"word": "嗯", "start": 1.1, "end": 1.4}],
                        },
                    ],
                },
            ),
        ):
            return caption_pipeline.draft_selected_caption_candidates(self.project, language="zh")

    def test_drafts_only_retained_audio_cues_with_final_timeline_times(self) -> None:
        document = self._draft()

        self.assertEqual(document["selected_clip_count"], 1)
        self.assertEqual(document["privacy"]["audio_uploaded"], False)
        self.assertEqual([cue["text"] for cue in document["candidates"]], ["大家好。", "嗯"])
        self.assertEqual(document["candidates"][0]["timeline_start"], 0.1)
        output = self.project / "work" / "transcripts" / "caption_candidates.json"
        self.assertTrue(output.is_file())

    def test_apply_complete_review_keeps_meaningful_phrase_and_drops_filler(self) -> None:
        candidates = self._draft()
        candidate_path = self.project / "work" / "transcripts" / "caption_candidates.json"
        review_path = self.project / "work" / "transcripts" / "caption_review.json"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(
            json.dumps({
                "schema_version": "video-factory.caption-review.v1",
                "reviewer": "Codex",
                "edit_plan_sha256": candidates["edit_plan_sha256"],
                "candidate_set_sha256": candidates["candidate_set_sha256"],
                "decisions": {
                    "natural-clip:subtitle-001": {"decision": "keep", "reason": "clear meaningful speech"},
                    "natural-clip:subtitle-002": {"decision": "drop", "reason": "filler only"},
                },
            }),
            encoding="utf-8",
        )

        kept, dropped, total = caption_pipeline.apply_caption_review(self.project, "work/transcripts/caption_review.json")

        updated = json.loads(self.plan_path.read_text(encoding="utf-8"))
        subtitles = [item for item in updated["timeline"] if item.get("type") == "subtitle"]
        self.assertEqual((kept, dropped, total), (1, 1, 2))
        self.assertEqual(len(subtitles), 1)
        self.assertEqual(subtitles[0]["text"], "大家好。")
        self.assertEqual(subtitles[0]["source_in"], 4.1)
        self.assertEqual(subtitles[0]["caption_review"]["status"], "kept")
        self.assertTrue((self.project / "work" / "transcripts" / "caption_review_audit.json").is_file())

        with self.assertRaisesRegex(video_factory.UserFacingError, "stale"):
            caption_pipeline.apply_caption_review(self.project, "work/transcripts/caption_review.json")

    def test_incomplete_review_never_modifies_edit_plan(self) -> None:
        candidates = self._draft()
        original = self.plan_path.read_text(encoding="utf-8")
        review_path = self.project / "work" / "transcripts" / "caption_review.json"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(
            json.dumps({
                "schema_version": "video-factory.caption-review.v1",
                "edit_plan_sha256": candidates["edit_plan_sha256"],
                "candidate_set_sha256": candidates["candidate_set_sha256"],
                "decisions": {"natural-clip:subtitle-001": "keep"},
            }),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(video_factory.UserFacingError, "Every subtitle candidate"):
            caption_pipeline.apply_caption_review(self.project, "work/transcripts/caption_review.json")
        self.assertEqual(self.plan_path.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
