from __future__ import annotations

import json
import ast
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "skills" / "video-factory" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from perception import build_perception_index
from video_factory import build_parser as build_skill_cli_parser
from colab_perception import _worker_source
from video_editor.application import ProjectOptions, VideoFactoryApplication
from video_editor.cli import build_parser as build_product_cli_parser
from video_editor.product_policy import load_product_policy, resolve_duration
from video_editor.music import MusicCandidate, MusicLicenseVerifier


class ProductFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="vf-product-")
        self.project = Path(self.directory.name)
        (self.project / "assets/videos").mkdir(parents=True)
        (self.project / "work/frames").mkdir(parents=True)
        (self.project / "outputs").mkdir()
        (self.project / "assets/videos/clip.mp4").write_bytes(b"video")
        (self.project / "work/frames/frame.jpg").write_bytes(b"frame")
        (self.project / "job.yaml").write_text(
            "profile: family\n"
            "target_duration_seconds: 180\n"
            "privacy_mode: LOCAL_ONLY\n"
            "cloud_perception: false\n"
            "gpu_policy: T4\n"
            "music:\n  enabled: true\n  mode: auto_open_licensed\n",
            encoding="utf-8",
        )
        (self.project / "work/manifests").mkdir(parents=True)
        (self.project / "work/manifests/media_manifest.json").write_text(json.dumps({
            "assets": [{"source": "assets/videos/clip.mp4", "kind": "video", "sha256": "a" * 64}],
        }), encoding="utf-8")
        (self.project / "work/frames/frames.json").write_text(json.dumps({
            "assets": {"assets/videos/clip.mp4": {"frames": [{"path": "work/frames/frame.jpg", "timestamp_seconds": 1.0}]}},
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_duration_presets_are_profile_specific(self) -> None:
        self.assertEqual(resolve_duration("family", "standard").target_seconds, 180)
        self.assertEqual(resolve_duration("health_education", "short").maximum_seconds, 40)
        self.assertEqual(load_product_policy(self.project).privacy_mode, "LOCAL_ONLY")

    def test_application_inherits_job_policy_and_maps_deep_quality_to_supported_backend(self) -> None:
        (self.project / "job.yaml").write_text(
            "profile: memory\n"
            "duration_preset: custom\n"
            "target_duration_seconds: 60\n"
            "privacy_mode: MAX_QUALITY\n"
            "cloud_perception: true\n"
            "gpu_policy: T4\n"
            "temporal_backend: smolvlm2\n",
            encoding="utf-8",
        )
        app = VideoFactoryApplication(self.project)
        policy = app._orchestrator().policy
        self.assertEqual(policy.mode, "family")
        self.assertEqual(policy.duration.target_seconds, 60)
        self.assertEqual(policy.privacy_mode, "MAX_QUALITY")
        self.assertTrue(policy.cloud_perception)
        self.assertEqual(policy.temporal_backend, "smolvlm2")

        deep_app = VideoFactoryApplication(self.project, ProjectOptions(ai_quality="deep"))
        self.assertEqual(deep_app._orchestrator().temporal_backend, "smolvlm2")

    def test_product_cli_only_exposes_temporal_backends_that_are_registered(self) -> None:
        parser = build_product_cli_parser()
        for command in ("run", "perception", "colab-perception"):
            with self.subTest(command=command):
                args = parser.parse_args([command, str(self.project), "--temporal-backend", "smolvlm2"])
                self.assertEqual(args.temporal_backend, "smolvlm2")

    def test_colab_cli_accepts_an_explicit_temporal_source_shortlist(self) -> None:
        args = build_product_cli_parser().parse_args([
            "colab-perception", str(self.project), "--temporal-backend", "smolvlm2",
            "--temporal-source", "assets/videos/one.MOV",
            "--temporal-source", "assets/videos/two.MOV",
        ])
        self.assertEqual(args.temporal_sources, [
            "assets/videos/one.MOV", "assets/videos/two.MOV",
        ])

    def test_perception_subcommand_inherits_job_cloud_setting_when_flag_is_omitted(self) -> None:
        args = build_skill_cli_parser().parse_args(["perception", str(self.project)])
        self.assertIsNone(args.cloud)

    def test_caption_cli_uses_cached_model_by_default_and_requires_opt_in_to_download(self) -> None:
        parser = build_skill_cli_parser()
        local_args = parser.parse_args(["draft-captions", str(self.project)])
        download_args = parser.parse_args([
            "draft-captions", str(self.project), "--allow-model-download",
        ])

        self.assertTrue(local_args.local_only)
        self.assertFalse(download_args.local_only)

    def test_perception_index_never_marks_original_upload(self) -> None:
        path = build_perception_index(self.project)
        index = json.loads(path.read_text(encoding="utf-8"))
        self.assertFalse(index["privacy"]["original_media_uploaded"])
        self.assertEqual(index["privacy"]["mode"], "LOCAL_ONLY")
        self.assertEqual(index["compute"]["gpu_policy"], "T4")
        self.assertIsNone(index["editorial"]["candidate_signals"][0]["keep_decision"])
        self.assertTrue((self.project / "outputs/work/perception_index.json").is_file())

    def test_perception_cache_refreshes_job_cloud_plan_without_transferring(self) -> None:
        (self.project / "job.yaml").write_text(
            "profile: family\nprivacy_mode: BALANCED\ncloud_perception: true\n",
            encoding="utf-8",
        )
        initial = json.loads(build_perception_index(self.project).read_text(encoding="utf-8"))
        self.assertEqual(initial["privacy"]["cloud_status"], "awaiting_authorization")
        self.assertEqual(initial["visual"]["status"], "pending_colab")

        local_only = json.loads(build_perception_index(self.project, cloud=False).read_text(encoding="utf-8"))
        self.assertEqual(local_only["cache_key"], initial["cache_key"])
        self.assertEqual(local_only["privacy"]["cloud_status"], "not_requested")
        self.assertEqual(local_only["visual"]["status"], "not_requested")
        self.assertFalse(local_only["privacy"]["original_media_uploaded"])

    def test_safe_auto_license_verifier_rejects_restrictions(self) -> None:
        candidate = MusicCandidate(
            id="track-1", title="Test", provider="openverse", license="CC BY-NC",
            license_url="https://example.test/license", source_url="https://example.test/source",
            download_url="https://example.test/download",
        )
        decision = MusicLicenseVerifier().verify(candidate)
        self.assertFalse(decision.accepted)

    def test_colab_worker_contract_is_valid_python_and_never_allows_original_media(self) -> None:
        source = _worker_source({
            "gpu": "T4", "privacy_mode": "BALANCED", "temporal_backend": "none",
            "speech_model": "large-v3-turbo", "visual_model": "google/siglip2-base-patch16-224",
            "bundle_path": "/content/input.zip", "work_root": "/content/work",
            "output_path": "/content/result.json",
        })
        ast.parse(source)
        self.assertIn('"original_media_uploaded": False', source)


if __name__ == "__main__":
    unittest.main()
