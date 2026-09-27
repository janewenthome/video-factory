from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from video_editor.music_catalog import (
    CATALOG_MANIFEST,
    LICENSE_FILTER,
    SEARCH_SNAPSHOT,
    TRACK_DIRECTORY,
    HttpResponse,
    MusicCatalogError,
    OpenverseAudioCatalog,
)


def _api_result(
    *,
    track_id: str = "track-001",
    license_slug: str = "by",
    license_version: str = "4.0",
    license_url: str = "https://creativecommons.org/licenses/by/4.0/",
    download_url: str = "https://prod-1.storage.jamendo.com/download/track/1/mp32",
    provider: str = "jamendo",
    source_url: str = "https://www.jamendo.com/track/1",
) -> dict[str, object]:
    return {
        "id": track_id,
        "title": "Warm Evening",
        "creator": "Example Artist",
        "creator_url": "https://www.jamendo.com/artist/1",
        "foreign_landing_url": source_url,
        "license": license_slug,
        "license_version": license_version,
        "license_url": license_url,
        "url": download_url,
        "provider": provider,
        "attribution": '"Warm Evening" by Example Artist, CC BY 4.0',
        "duration": 92_000,
        "filetype": "mp3",
    }


class FakeRequester:
    def __init__(self, responses: list[HttpResponse]):
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def __call__(self, url: str, **kwargs: object) -> HttpResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("Unexpected network request in test")
        return self.responses.pop(0)


class OpenverseAudioCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="vf-music-catalog-")
        self.project = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_search_is_anonymous_and_passes_conservative_license_filter(self) -> None:
        body = json.dumps({"results": [_api_result()]}).encode()
        requester = FakeRequester([HttpResponse(200, {"content-type": "application/json"}, body)])
        catalog = OpenverseAudioCatalog(requester=requester)

        results = catalog.search("warm acoustic", limit=6)

        self.assertEqual(len(results), 1)
        call = requester.calls[0]
        query = parse_qs(urlsplit(str(call["url"])).query)
        self.assertEqual(query["q"], ["warm acoustic"])
        self.assertEqual(query["license"], [LICENSE_FILTER])
        self.assertEqual(query["page_size"], ["6"])
        headers = call["headers"]
        self.assertIn("User-Agent", headers)
        self.assertNotIn("Authorization", headers)
        self.assertEqual(results[0].preview_url, results[0].download_url)
        self.assertEqual(results[0].provider, "jamendo")
        self.assertEqual(results[0].duration_seconds, 92.0)

    def test_search_drops_non_allowlisted_or_inconsistent_license_metadata(self) -> None:
        results = [
            _api_result(track_id="cc-by"),
            _api_result(track_id="cc0", license_slug="cc0", license_version="1.0", license_url="https://creativecommons.org/publicdomain/zero/1.0/"),
            _api_result(track_id="pdm", license_slug="pdm", license_version="1.0", license_url="https://creativecommons.org/publicdomain/mark/1.0/"),
            _api_result(track_id="nc", license_slug="by-nc", license_version="4.0", license_url="https://creativecommons.org/licenses/by-nc/4.0/"),
            _api_result(track_id="wrong-url", license_slug="by", license_version="4.0", license_url="https://example.org/licenses/by/4.0/"),
            _api_result(track_id="wrong-version", license_slug="by", license_version="4.0", license_url="https://creativecommons.org/licenses/by/3.0/"),
            _api_result(track_id="unknown-version", license_slug="by", license_version="7.0", license_url="https://creativecommons.org/licenses/by/7.0/"),
        ]
        requester = FakeRequester([HttpResponse(200, {}, json.dumps({"results": results}).encode())])

        found = OpenverseAudioCatalog(requester=requester).search("ambient")

        self.assertEqual({track.id for track in found}, {"cc-by", "cc0", "pdm"})

    def test_429_respects_retry_after_then_searches(self) -> None:
        requester = FakeRequester([
            HttpResponse(429, {"retry-after": "2"}, b""),
            HttpResponse(200, {}, json.dumps({"results": [_api_result()]}).encode()),
        ])
        sleeps: list[float] = []

        results = OpenverseAudioCatalog(requester=requester, sleep=sleeps.append).search("ambient")

        self.assertEqual(len(results), 1)
        self.assertEqual(sleeps, [2.0])
        self.assertEqual(len(requester.calls), 2)

    def test_long_retry_after_is_not_retried_early(self) -> None:
        requester = FakeRequester([HttpResponse(429, {"retry-after": "120"}, b"")])
        sleeps: list[float] = []

        with self.assertRaisesRegex(MusicCatalogError, "HTTP 429"):
            OpenverseAudioCatalog(requester=requester, sleep=sleeps.append).search("ambient")

        self.assertEqual(sleeps, [])
        self.assertEqual(len(requester.calls), 1)

    def test_transport_error_does_not_expose_request_details(self) -> None:
        class LeakyTransport:
            def __call__(self, url: str, **kwargs: object) -> HttpResponse:
                raise RuntimeError(f"failed request {url}?token=must-not-appear")

        with self.assertRaises(MusicCatalogError) as error:
            OpenverseAudioCatalog(requester=LeakyTransport()).search("ambient")

        self.assertNotIn("must-not-appear", str(error.exception))
        self.assertNotIn("api.openverse.org", str(error.exception))

    def test_explicit_add_downloads_valid_audio_and_saves_provenance(self) -> None:
        audio = b"ID3" + b"test-track-data"
        requester = FakeRequester([
            HttpResponse(200, {}, json.dumps({"results": [_api_result()]}).encode()),
            HttpResponse(200, {"content-type": "audio/mpeg", "content-length": str(len(audio))}, audio),
        ])
        catalog = OpenverseAudioCatalog(requester=requester)
        candidates = catalog.search("warm acoustic")
        catalog.save_search(self.project, "warm acoustic", candidates)

        added = catalog.add(self.project, "track-001")

        path = self.project.resolve() / TRACK_DIRECTORY / "track-001.mp3"
        self.assertEqual(added["local_path"], path)
        self.assertEqual(path.read_bytes(), audio)
        self.assertEqual(added["sha256"], hashlib.sha256(audio).hexdigest())
        manifest = json.loads((self.project / CATALOG_MANIFEST).read_text(encoding="utf-8"))
        self.assertEqual(manifest["tracks"][0]["sha256"], hashlib.sha256(audio).hexdigest())
        evidence_path = self.project / "work/music/openverse/licenses/track-001.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        self.assertFalse(evidence["verification"]["source_license_verified_by_openverse"])
        self.assertIn("not endorsed", evidence["openverse_notice"])
        self.assertEqual(len(requester.calls), 2)
        self.assertEqual(requester.calls[1]["max_bytes"], 50 * 1024 * 1024)

    def test_add_fails_closed_on_content_type_mismatch_without_saving_audio(self) -> None:
        audio = b"ID3" + b"test-track-data"
        requester = FakeRequester([
            HttpResponse(200, {}, json.dumps({"results": [_api_result()]}).encode()),
            HttpResponse(200, {"content-type": "text/html", "content-length": str(len(audio))}, audio),
        ])
        catalog = OpenverseAudioCatalog(requester=requester)
        catalog.save_search(self.project, "warm acoustic", catalog.search("warm acoustic"))

        with self.assertRaisesRegex(MusicCatalogError, "content-type"):
            catalog.add(self.project, "track-001")

        track_dir = self.project / TRACK_DIRECTORY
        self.assertEqual(list(track_dir.glob("*.mp3")), [])

    def test_add_refuses_oversized_audio_before_writing(self) -> None:
        audio = b"ID3" + b"x" * 12
        requester = FakeRequester([
            HttpResponse(200, {}, json.dumps({"results": [_api_result()]}).encode()),
            HttpResponse(200, {"content-type": "audio/mpeg", "content-length": "99"}, audio),
        ])
        catalog = OpenverseAudioCatalog(requester=requester, max_download_bytes=10)
        catalog.save_search(self.project, "warm acoustic", catalog.search("warm acoustic"))

        with self.assertRaisesRegex(MusicCatalogError, "size limit"):
            catalog.add(self.project, "track-001")

        self.assertFalse((self.project / TRACK_DIRECTORY / "track-001.mp3").exists())

    def test_add_revalidates_saved_metadata_and_requires_most_recent_search_id(self) -> None:
        requester = FakeRequester([HttpResponse(200, {}, json.dumps({"results": [_api_result()]}).encode())])
        catalog = OpenverseAudioCatalog(requester=requester)
        catalog.save_search(self.project, "warm acoustic", catalog.search("warm acoustic"))
        snapshot_path = self.project / SEARCH_SNAPSHOT
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot["results"][0]["license_url"] = "https://example.org/not-cc-by"
        snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")

        with self.assertRaisesRegex(MusicCatalogError, "safety check"):
            catalog.add(self.project, "track-001")
        with self.assertRaisesRegex(MusicCatalogError, "not found"):
            catalog.load_search_result(self.project, "other-track")
        self.assertEqual(len(requester.calls), 1)

    def test_cli_exposes_explicit_search_and_add_commands(self) -> None:
        from video_editor.cli import build_parser

        search = build_parser().parse_args(["music", "search", "soft", "piano", "--project", str(self.project)])
        add = build_parser().parse_args(["music", "add", "track-001", "--project", str(self.project)])
        self.assertEqual(search.command, "music")
        self.assertEqual(search.music_action, "search")
        self.assertEqual(search.query, ["soft", "piano"])
        self.assertEqual(add.music_action, "add")
        self.assertEqual(add.id, "track-001")


if __name__ == "__main__":
    unittest.main()
