from __future__ import annotations

import unittest

from video_editor.subtitles_local import (
    apply_subtitle_reviews,
    generate_subtitle_candidates,
    validate_subtitle_cues,
)


class TestLocalSubtitleHelpers(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = {
            "duration_seconds": 20,
            "timeline": [
                {
                    "id": "natural-1",
                    "type": "video",
                    "source": "assets/clip.mov",
                    "source_in": 20.0,
                    "source_out": 25.0,
                    "timeline_start": 10.0,
                    "timeline_end": 15.0,
                    "audio": {"preserve_natural": True},
                },
                {
                    "id": "muted-1",
                    "type": "video",
                    "source": "assets/muted.mov",
                    "source_in": 0.0,
                    "source_out": 5.0,
                    "timeline_start": 15.0,
                    "timeline_end": 20.0,
                    "audio": {"preserve_natural": False},
                },
            ],
        }

    def test_maps_cut_relative_words_only_from_natural_audio(self) -> None:
        transcripts = {
            "natural-1": {
                "words": [
                    {"id": "w1", "start": 0.2, "end": 0.6, "word": "大家"},
                    {"id": "w2", "start": 0.65, "end": 1.0, "word": "好，"},
                    {"id": "w3", "start": 1.1, "end": 1.35, "word": "我們"},
                    {"id": "w4", "start": 1.4, "end": 1.75, "word": "開始。"},
                ]
            },
            "muted-1": {"words": [{"start": 0.1, "end": 0.7, "word": "不要字幕"}]},
        }

        cues = generate_subtitle_candidates(self.plan, transcripts)

        self.assertEqual([cue["text"] for cue in cues], ["大家好，", "我們開始。"])
        self.assertEqual(cues[0]["timeline_start"], 10.2)
        self.assertEqual(cues[0]["timeline_end"], 11.0)
        self.assertEqual(cues[0]["source_start"], 20.2)
        self.assertEqual(cues[0]["source_end"], 21.0)
        self.assertEqual(cues[0]["word_ids"], ["w1", "w2"])
        self.assertTrue(all(cue["review_status"] == "pending" for cue in cues))
        self.assertEqual(validate_subtitle_cues(cues, duration_seconds=20), [])

    def test_does_not_caption_words_cut_by_source_or_timeline_bounds(self) -> None:
        plan = {
            "timeline": [
                {
                    **self.plan["timeline"][0],
                    "source_out": 24.0,
                    "timeline_end": 14.0,
                }
            ]
        }
        cues = generate_subtitle_candidates(
            plan,
            {
                "natural-1": {
                    "words": [
                        {"start": -0.1, "end": 0.2, "word": "越界"},
                        {"start": 3.8, "end": 4.2, "word": "部分被剪"},
                        {"start": 3.2, "end": 3.7, "word": "保留"},
                    ]
                }
            },
        )
        self.assertEqual([cue["text"] for cue in cues], ["保留"])
        self.assertGreaterEqual(cues[0]["timeline_start"], 10.0)
        self.assertLessEqual(cues[0]["timeline_end"], 14.0)

    def test_splits_when_speaker_changes_and_keeps_anonymous_speaker_tag(self) -> None:
        cue = generate_subtitle_candidates(
            self.plan,
            {
                "natural-1": {
                    "segments": [
                        {
                            "speaker_id": "speaker_1",
                            "words": [{"start": 0.1, "end": 0.5, "word": "第一位"}],
                        },
                        {
                            "speaker_id": "speaker_2",
                            "words": [{"start": 0.6, "end": 1.0, "word": "第二位"}],
                        },
                    ]
                }
            },
        )
        self.assertEqual(len(cue), 2)
        self.assertEqual([item["speaker_id"] for item in cue], ["speaker_1", "speaker_2"])

    def test_overlapping_word_ranges_are_split_into_non_overlapping_cues(self) -> None:
        cues = generate_subtitle_candidates(
            {"timeline": [self.plan["timeline"][0]]},
            {
                "natural-1": {
                    "words": [
                        {"start": 0.1, "end": 1.0, "word": "第一句，"},
                        {"start": 0.8, "end": 1.5, "word": "第二句。"},
                    ]
                }
            },
        )
        self.assertEqual(len(cues), 2)
        self.assertLessEqual(cues[0]["timeline_end"], cues[1]["timeline_start"])
        self.assertEqual(validate_subtitle_cues(cues, duration_seconds=20), [])

    def test_splits_when_combined_text_exceeds_reading_rate(self) -> None:
        cues = generate_subtitle_candidates(
            {"timeline": [self.plan["timeline"][0]]},
            {
                "natural-1": {
                    "words": [
                        {"start": 0.1, "end": 0.3, "word": "快速文字"},
                        {"start": 0.31, "end": 0.5, "word": "快速文字"},
                    ]
                }
            },
        )
        self.assertEqual([cue["text"] for cue in cues], ["快速文字", "快速文字"])

    def test_review_filter_accepts_explicit_candidate_phrase_and_word_decisions(self) -> None:
        candidates = [
            {"id": "c1", "text": "清楚內容", "word_ids": ["w1", "w2"]},
            {"id": "c2", "text": "背景雜音", "word_ids": ["w3"]},
            {"id": "c3", "text": "尚未判斷", "word_ids": ["w4"]},
        ]
        reviewed = apply_subtitle_reviews(
            candidates,
            {
                "c1": {"decision": "keep", "reason": "語意完整"},
                "phrase:背景雜音": "drop",
                "word:w4": "keep",
            },
        )
        self.assertEqual([cue["id"] for cue in reviewed["keep"]], ["c1", "c3"])
        self.assertEqual(reviewed["keep"][0]["review_reason"], "語意完整")
        self.assertEqual([cue["id"] for cue in reviewed["drop"]], ["c2"])
        self.assertEqual(reviewed["pending"], [])

    def test_a_drop_on_any_word_drops_phrase_as_a_whole(self) -> None:
        result = apply_subtitle_reviews(
            [{"id": "c1", "text": "片語", "word_ids": ["w1", "w2"]}],
            {"word:w1": "keep", "w2": {"decision": "drop", "reason": "聽不清"}},
        )
        self.assertEqual(result["keep"], [])
        self.assertEqual(result["pending"], [])
        self.assertEqual(result["drop"][0]["review_reason"], "AI review dropped at least one constituent word")

    def test_validate_reports_empty_short_out_of_range_and_overlapping_cues(self) -> None:
        errors = validate_subtitle_cues(
            [
                {"id": "a", "text": "第一句", "timeline_start": 1.0, "timeline_end": 2.0},
                {"id": "b", "text": "第二句", "timeline_start": 1.8, "timeline_end": 2.1},
                {"id": "c", "text": "", "timeline_start": 4.0, "timeline_end": 4.05},
                {"id": "d", "text": "太晚", "timeline_start": 9.0, "timeline_end": 10.0},
            ],
            duration_seconds=9.5,
        )
        self.assertTrue(any("a overlaps b" in error for error in errors), errors)
        self.assertTrue(any("c: text is empty" in error for error in errors), errors)
        self.assertTrue(any("c: cue is shorter" in error for error in errors), errors)
        self.assertTrue(any("d: end exceeds" in error for error in errors), errors)

    def test_source_absolute_timestamps_can_be_requested(self) -> None:
        cues = generate_subtitle_candidates(
            {"timeline": [self.plan["timeline"][0]]},
            {"natural-1": {"words": [{"start": 20.25, "end": 20.75, "text": "絕對時間"}]}},
            timestamp_reference="source",
        )
        self.assertEqual(cues[0]["timeline_start"], 10.25)
        self.assertEqual(cues[0]["timeline_end"], 10.75)


if __name__ == "__main__":
    unittest.main()
