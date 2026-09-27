import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from video_editor.music import prepare_music_artifacts
from video_editor.product_policy import load_product_policy


class MusicArtifactPreservationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project = Path(self.temp_dir.name) / "project"
        self.project.mkdir()
        (self.project / "job.yaml").write_text(
            "profile: memory\n"
            "duration_preset: custom\n"
            "target_duration_seconds: 60\n"
            "music:\n"
            "  enabled: true\n"
            "  mode: auto_open_licensed\n"
            "natural_audio:\n"
            "  preserve: true\n",
            encoding="utf-8",
        )
        self.track_path = self.project / "work" / "music" / "track.wav"
        self.track_path.parent.mkdir(parents=True)
        self.track_bytes = b"fake wav data for attribution hash validation"
        self.track_path.write_bytes(self.track_bytes)
        self.track_digest = hashlib.sha256(self.track_bytes).hexdigest()
        self.plan = {
            "duration_seconds": 60,
            "timeline": [
                {"id": "family-track", "type": "music", "timeline_start": 0, "timeline_end": 60,
                 "source": "work/music/track.wav", "source_in": 0, "source_out": 60, "volume": 0.2},
                {"id": "family-photo", "type": "photo", "timeline_start": 0, "timeline_end": 60,
                 "source": "assets/photos/family.jpg"},
            ],
        }
        self.receipt = {
            "schema_version": "music-attribution.v1",
            "tracks": [{
                "id": "family-track", "title": "Family Theme", "creator": "Artist",
                "provider": "Public catalog", "license": "CC0",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "source_url": "https://example.org/track", "download_url": "https://example.org/track.wav",
                "local_asset": "work/music/track.wav", "sha256": self.track_digest,
                "attribution_text": "Family Theme by Artist; CC0; https://example.org/track",
                "source_page_evidence": {"status": "source_page_claim_confirmed"},
            }],
        }
        receipt_path = self.project / "work" / "music_attribution.json"
        receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")
        self.policy = load_product_policy(self.project)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_plan_music_preserves_and_mirrors_manual_attribution(self):
        result = prepare_music_artifacts(self.project, self.policy, self.plan)

        self.assertEqual(result["status"], "selected_in_edit_plan")
        self.assertEqual(result["requirements"], [])
        for relative in (
            "work/music_attribution.json",
            "outputs/work/music_attribution.json",
            "outputs/final/music_attribution.json",
        ):
            self.assertEqual(json.loads((self.project / relative).read_text()), self.receipt)
        credits = (self.project / "outputs/MUSIC_CREDITS.txt").read_text()
        self.assertIn("Family Theme by Artist", credits)
        self.assertNotIn("No automatically selected track", credits)
        self.assertTrue((self.project / "outputs/work/music/licenses/family-track.json").is_file())

    def test_music_segment_without_receipt_fails_closed(self):
        (self.project / "work" / "music_attribution.json").unlink()

        with self.assertRaisesRegex(ValueError, "no work/music_attribution.json receipt"):
            prepare_music_artifacts(self.project, self.policy, self.plan)

        self.assertFalse((self.project / "work" / "music_attribution.json").exists())

    def test_source_hash_mismatch_fails_closed(self):
        self.receipt["tracks"][0]["sha256"] = "0" * 64
        (self.project / "work" / "music_attribution.json").write_text(json.dumps(self.receipt), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "source hash does not match"):
            prepare_music_artifacts(self.project, self.policy, self.plan)


if __name__ == "__main__":
    unittest.main()
