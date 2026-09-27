"""Explicit Openverse audio discovery and download for project-local use.

Searching only saves a small result snapshot. A user must run ``music add`` to
download a track; this module is deliberately not called by the render
pipeline. Openverse indexes third-party works and does not verify their rights,
so this code enforces a conservative metadata gate and retains provenance for
manual review.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import tempfile
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


OPENVERSE_AUDIO_ENDPOINT = "https://api.openverse.org/v1/audio/"
OPENVERSE_WARNING = (
    "Openverse aggregates third-party audio metadata and does not verify or "
    "guarantee each work's license. Check the original source page before use."
)
OPENVERSE_NON_ENDORSEMENT = (
    "Made using the Openverse API; not endorsed or certified by Openverse."
)
LICENSE_FILTER = "cc0,pdm,by"
SEARCH_SNAPSHOT = Path("work/music/openverse_search.json")
CATALOG_MANIFEST = Path("work/music/openverse_manifest.json")
TRACK_DIRECTORY = Path("work/music/openverse")
LICENSE_DIRECTORY = Path("work/music/openverse/licenses")
MAX_SEARCH_BYTES = 4 * 1024 * 1024
MAX_TRACK_BYTES = 50 * 1024 * 1024
MAX_RETRIES = 3
MAX_AUTOMATIC_RETRY_WAIT = 30.0
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
TRACK_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# The metadata slug, declared version, and canonical URI must agree exactly.
# Support the standard international CC BY releases commonly represented in
# Openverse. Unknown/localized variants fail closed until reviewed explicitly.
_ALLOWED_LICENSE_URLS = {
    ("cc0", "1.0"): "https://creativecommons.org/publicdomain/zero/1.0/",
    ("pdm", "1.0"): "https://creativecommons.org/publicdomain/mark/1.0/",
    **{
        ("by", version): f"https://creativecommons.org/licenses/by/{version}/"
        for version in ("1.0", "2.0", "2.5", "3.0", "4.0")
    },
}
_PROVIDER_DOMAINS = {
    "freesound": ("freesound.org",),
    "jamendo": ("jamendo.com",),
    "wikimedia": ("wikimedia.org",),
    "wikimedia_audio": ("wikimedia.org",),
}
_AUDIO_MIME_ALIASES = {
    "audio/mp3": "audio/mpeg",
    "audio/x-mp3": "audio/mpeg",
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/x-flac": "audio/flac",
    "application/ogg": "audio/ogg",
    "application/x-ogg": "audio/ogg",
}
_AUDIO_EXTENSIONS = {
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/webm": ".webm",
}


class MusicCatalogError(RuntimeError):
    """Safe, user-facing error for catalog searches and downloads."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True)
class OpenverseTrack:
    id: str
    title: str
    creator: str
    creator_url: str
    source_url: str
    license_slug: str
    license_version: str
    license_url: str
    preview_url: str
    download_url: str
    provider: str
    attribution: str
    duration_seconds: float | None
    filetype: str
    catalog_provider: str = "openverse"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _clean_text(value: Any, *, required: bool = False) -> str:
    if not isinstance(value, str):
        return ""
    # Prevent API-provided newlines and terminal control sequences from
    # forging CLI output while preserving normal Unicode text.
    cleaned = "".join(" " if unicodedata.category(char).startswith("C") else char for char in value)
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned[:1000] if required else cleaned[:2000]


def _is_https_url(value: Any) -> bool:
    if not isinstance(value, str) or len(value) > 4096:
        return False
    try:
        parts = urlsplit(value)
        if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
            return False
        if parts.port not in (None, 443):
            return False
        host = parts.hostname.lower().rstrip(".")
        if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost") or host.endswith(".local"):
            return False
        try:
            if not ipaddress.ip_address(host).is_global:
                return False
        except ValueError:
            pass
        return True
    except ValueError:
        return False


def _host_matches(host: str, domains: tuple[str, ...]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _provider_domains(provider: str) -> tuple[str, ...]:
    return _PROVIDER_DOMAINS.get(provider.strip().lower(), ())


def _license_is_allowed(slug: str, version: str, license_url: str) -> bool:
    canonical = _ALLOWED_LICENSE_URLS.get((slug.strip().lower(), version.strip()))
    if canonical is None or not _is_https_url(license_url):
        return False
    try:
        actual = urlsplit(license_url)
    except ValueError:
        return False
    # Ignore only an optional terminal slash; reject alternate hosts, paths,
    # query strings, fragments, and misleading license pages.
    expected = urlsplit(canonical)
    return (
        actual.hostname == expected.hostname
        and actual.port is None
        and actual.path.rstrip("/") == expected.path.rstrip("/")
        and not actual.query
        and not actual.fragment
    )


def _track_from_api(item: Any) -> OpenverseTrack | None:
    if not isinstance(item, dict):
        return None
    track_id = _clean_text(item.get("id"))
    title = _clean_text(item.get("title"), required=True)
    creator = _clean_text(item.get("creator"), required=True)
    creator_url = _clean_text(item.get("creator_url"))
    source_url = _clean_text(item.get("foreign_landing_url"))
    license_slug = _clean_text(item.get("license")).lower()
    license_version = _clean_text(item.get("license_version"))
    license_url = _clean_text(item.get("license_url"))
    download_url = _clean_text(item.get("url"))
    provider = _clean_text(item.get("provider")).lower()
    attribution = _clean_text(item.get("attribution"))
    filetype = _clean_text(item.get("filetype")).lower().lstrip(".")

    if not TRACK_ID_PATTERN.fullmatch(track_id) or not title:
        return None
    if license_slug == "by" and not creator:
        # CC BY requires meaningful attribution data.
        return None
    if not _license_is_allowed(license_slug, license_version, license_url):
        return None
    domains = _provider_domains(provider)
    if not domains or not _is_https_url(source_url) or not _is_https_url(download_url):
        return None
    source_host = urlsplit(source_url).hostname or ""
    download_host = urlsplit(download_url).hostname or ""
    if not _host_matches(source_host, domains) or not _host_matches(download_host, domains):
        return None

    raw_duration = item.get("duration")
    try:
        # Openverse stores audio duration in milliseconds. Convert at the API
        # boundary so the CLI and saved snapshot consistently expose seconds.
        duration_seconds = float(raw_duration) / 1000.0 if raw_duration is not None else None
        if duration_seconds is not None and (duration_seconds < 0 or duration_seconds > 24 * 60 * 60):
            duration_seconds = None
    except (TypeError, ValueError):
        duration_seconds = None

    return OpenverseTrack(
        id=track_id,
        title=title,
        creator=creator,
        creator_url=creator_url if _is_https_url(creator_url) else "",
        source_url=source_url,
        license_slug=license_slug,
        license_version=license_version,
        license_url=license_url,
        preview_url=download_url,
        download_url=download_url,
        provider=provider,
        attribution=attribution,
        duration_seconds=duration_seconds,
        filetype=filetype,
    )


def _safe_json_read(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise MusicCatalogError("Search results are missing. Run `music search` for this project first.") from None
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise MusicCatalogError("The saved music catalog data cannot be read.") from None
    if not isinstance(value, dict):
        raise MusicCatalogError("The saved music catalog data has an invalid format.")
    return value


def _atomic_json_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


class _SafeRedirectHandler(HTTPRedirectHandler):
    def __init__(self, trusted_domains: tuple[str, ...]):
        super().__init__()
        self.trusted_domains = trusted_domains

    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> Request | None:
        if not _is_https_url(newurl):
            raise MusicCatalogError("The catalog returned an unsafe redirect; download was stopped.")
        host = urlsplit(newurl).hostname or ""
        if not _host_matches(host, self.trusted_domains):
            raise MusicCatalogError("The catalog redirected outside the trusted provider; download was stopped.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _http_request(
    url: str,
    headers: Mapping[str, str],
    timeout: float,
    max_bytes: int,
    trusted_domains: tuple[str, ...],
) -> HttpResponse:
    request = Request(url, headers=dict(headers), method="GET")
    opener = build_opener(_SafeRedirectHandler(trusted_domains))
    try:
        with opener.open(request, timeout=timeout) as response:
            response_headers = {key.lower(): value for key, value in response.headers.items()}
            length = response_headers.get("content-length")
            if length and int(length) > max_bytes:
                return HttpResponse(int(response.status), response_headers, b"x" * (max_bytes + 1))
            return HttpResponse(int(response.status), response_headers, response.read(max_bytes + 1))
    except HTTPError as error:
        response_headers = {key.lower(): value for key, value in error.headers.items()} if error.headers else {}
        return HttpResponse(int(error.code), response_headers, b"")
    except MusicCatalogError:
        raise
    except (URLError, TimeoutError, OSError, ValueError):
        raise MusicCatalogError("The Openverse request could not be completed.") from None


class OpenverseAudioCatalog:
    """Search and explicitly add Openverse audio results.

    ``requester`` is injectable for tests and must return :class:`HttpResponse`.
    The default public API path is anonymous and does not use API credentials.
    """

    def __init__(
        self,
        requester: Callable[..., HttpResponse] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        timeout: float = 30.0,
        max_download_bytes: int = MAX_TRACK_BYTES,
        max_retries: int = MAX_RETRIES,
    ):
        self.requester = requester or _http_request
        self.sleep = sleep
        self.timeout = timeout
        self.max_download_bytes = max_download_bytes
        self.max_retries = max_retries

    def _request_with_backoff(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        max_bytes: int,
        trusted_domains: tuple[str, ...],
    ) -> HttpResponse:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.requester(
                    url,
                    headers=headers,
                    timeout=self.timeout,
                    max_bytes=max_bytes,
                    trusted_domains=trusted_domains,
                )
            except MusicCatalogError:
                raise
            except Exception:
                # Avoid exception strings: they can contain request URLs or
                # other transport details that should not reach CLI output.
                raise MusicCatalogError("The Openverse request could not be completed.") from None
            if response.status not in RETRYABLE_STATUS or attempt >= self.max_retries:
                return response
            delay = _retry_delay(response.headers, attempt)
            if delay is None:
                return response
            self.sleep(delay)
        raise MusicCatalogError("The Openverse request could not be completed.")

    def search(self, query: str, *, limit: int = 10) -> list[OpenverseTrack]:
        query = _clean_text(query, required=True)
        if not query:
            raise MusicCatalogError("Enter a search phrase.")
        if len(query) > 200:
            raise MusicCatalogError("Search phrases must be 200 characters or fewer.")
        if not 1 <= limit <= 50:
            raise MusicCatalogError("Search result limit must be between 1 and 50.")
        params = urlencode({"q": query, "license": LICENSE_FILTER, "page_size": str(limit)})
        url = f"{OPENVERSE_AUDIO_ENDPOINT}?{params}"
        response = self._request_with_backoff(
            url,
            headers={"Accept": "application/json", "User-Agent": "VideoFactory/1.0 (Openverse audio search)"},
            max_bytes=MAX_SEARCH_BYTES,
            trusted_domains=("openverse.org",),
        )
        if response.status != 200:
            raise MusicCatalogError(f"Openverse audio search failed with HTTP {response.status}.")
        if len(response.body) > MAX_SEARCH_BYTES:
            raise MusicCatalogError("Openverse returned an oversized search response.")
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            raise MusicCatalogError("Openverse returned an invalid search response.") from None
        raw_results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(raw_results, list):
            raise MusicCatalogError("Openverse returned an invalid search response.")
        tracks: list[OpenverseTrack] = []
        seen: set[str] = set()
        for item in raw_results:
            track = _track_from_api(item)
            if track and track.id not in seen:
                tracks.append(track)
                seen.add(track.id)
        return tracks[:limit]

    def save_search(self, project: Path, query: str, tracks: list[OpenverseTrack]) -> Path:
        root = _project_root(project)
        path = root / SEARCH_SNAPSHOT
        path.parent.mkdir(parents=True, exist_ok=True)
        _ensure_parent_inside_project(root, path)
        _atomic_json_write(path, {
            "schema_version": "openverse-audio-search.v1",
            "catalog_provider": "openverse",
            "query": _clean_text(query),
            "license_filter": LICENSE_FILTER,
            "searched_at": _now(),
            "warning": OPENVERSE_WARNING,
            "non_endorsement": OPENVERSE_NON_ENDORSEMENT,
            "results": [asdict(track) for track in tracks],
        })
        return path

    def load_search_result(self, project: Path, track_id: str) -> OpenverseTrack:
        if not TRACK_ID_PATTERN.fullmatch(track_id):
            raise MusicCatalogError("Track ID has an invalid format.")
        root = _project_root(project)
        snapshot = _safe_json_read(root / SEARCH_SNAPSHOT)
        results = snapshot.get("results")
        if not isinstance(results, list):
            raise MusicCatalogError("The saved music catalog data has an invalid format.")
        item = next((row for row in results if isinstance(row, dict) and row.get("id") == track_id), None)
        if item is None:
            raise MusicCatalogError("Track ID was not found in this project's most recent music search.")
        # Re-apply all gates on persisted metadata so a hand-edited search file
        # cannot turn an ineligible result into a downloadable track.
        track = _track_from_api({
            "id": item.get("id"),
            "title": item.get("title"),
            "creator": item.get("creator"),
            "creator_url": item.get("creator_url"),
            "foreign_landing_url": item.get("source_url"),
            "license": item.get("license_slug"),
            "license_version": item.get("license_version"),
            "license_url": item.get("license_url"),
            "url": item.get("download_url"),
            "provider": item.get("provider"),
            "attribution": item.get("attribution"),
            "duration": item.get("duration_seconds"),
            "filetype": item.get("filetype"),
        })
        if track is None:
            raise MusicCatalogError("Track metadata failed the open-license safety check.")
        return track

    def add(self, project: Path, track_id: str) -> dict[str, Any]:
        root = _project_root(project)
        track = self.load_search_result(root, track_id)
        track_dir = root / TRACK_DIRECTORY
        license_dir = root / LICENSE_DIRECTORY
        track_dir.mkdir(parents=True, exist_ok=True)
        license_dir.mkdir(parents=True, exist_ok=True)
        _ensure_parent_inside_project(root, track_dir / "placeholder")
        _ensure_parent_inside_project(root, license_dir / "placeholder")

        manifest_path = root / CATALOG_MANIFEST
        manifest = _read_manifest(manifest_path)
        existing = next((entry for entry in manifest["tracks"] if entry.get("id") == track.id), None)
        if existing:
            existing_path = root / existing.get("local_path", "")
            if existing_path.is_file():
                raise MusicCatalogError("This track is already in the project's Openverse music library.")
            raise MusicCatalogError("The track is already recorded in the library but its file is missing; review the manifest first.")
        if (license_dir / f"{track.id}.json").exists() or any(track_dir.glob(f"{track.id}.*")):
            raise MusicCatalogError("A file with this track ID already exists; nothing was overwritten.")

        domains = _provider_domains(track.provider)
        response = self._request_with_backoff(
            track.download_url,
            headers={"Accept": "audio/*", "User-Agent": "VideoFactory/1.0 (user-selected Openverse audio)"},
            max_bytes=self.max_download_bytes,
            trusted_domains=domains,
        )
        if response.status != 200:
            raise MusicCatalogError(f"The selected audio could not be downloaded (HTTP {response.status}).")
        if len(response.body) > self.max_download_bytes:
            raise MusicCatalogError(f"The selected audio exceeds the {self.max_download_bytes // (1024 * 1024)} MiB size limit.")
        response_headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
        declared_type = _canonical_audio_mime(response_headers.get("content-type", ""))
        detected_type = _detect_audio_mime(response.body)
        if not declared_type or not detected_type or declared_type != detected_type:
            raise MusicCatalogError("The selected file failed audio content-type verification; nothing was added.")
        if not response.body:
            raise MusicCatalogError("The selected audio file is empty; nothing was added.")
        content_length = response_headers.get("content-length")
        if content_length:
            try:
                if int(content_length) != len(response.body):
                    raise MusicCatalogError("The selected audio size did not match the server metadata; nothing was added.")
            except ValueError:
                raise MusicCatalogError("The server returned an invalid audio size; nothing was added.") from None

        extension = _AUDIO_EXTENSIONS[detected_type]
        local_relative = TRACK_DIRECTORY / f"{track.id}{extension}"
        destination = root / local_relative
        if destination.exists():
            raise MusicCatalogError("A file with this track ID already exists; nothing was overwritten.")
        digest = hashlib.sha256(response.body).hexdigest()
        selected_at = _now()
        evidence = {
            "schema_version": "openverse-audio-license-evidence.v1",
            "catalog_provider": "openverse",
            "catalog_api_url": OPENVERSE_AUDIO_ENDPOINT,
            "openverse_notice": OPENVERSE_NON_ENDORSEMENT,
            "license_warning": OPENVERSE_WARNING,
            "selected_at": selected_at,
            "track": asdict(track),
            "download": {
                "local_path": local_relative.as_posix(),
                "source_download_url": track.download_url,
                "content_type": declared_type,
                "size_bytes": len(response.body),
                "sha256": digest,
            },
            "verification": {
                "status": "metadata_gate_passed_manual_source_review_required",
                "license_slug_version_url_matched": True,
                "source_license_verified_by_openverse": False,
            },
        }
        evidence_path = license_dir / f"{track.id}.json"
        manifest["tracks"].append({
            "id": track.id,
            "title": track.title,
            "creator": track.creator,
            "provider": track.provider,
            "catalog_provider": track.catalog_provider,
            "license_slug": track.license_slug,
            "license_version": track.license_version,
            "license_url": track.license_url,
            "source_url": track.source_url,
            "local_path": local_relative.as_posix(),
            "evidence_path": evidence_path.relative_to(root).as_posix(),
            "sha256": digest,
            "size_bytes": len(response.body),
            "content_type": declared_type,
            "selected_at": selected_at,
        })
        manifest["updated_at"] = selected_at
        try:
            _atomic_bytes_write(destination, response.body)
            _atomic_json_write(evidence_path, evidence)
            _atomic_json_write(manifest_path, manifest)
        except Exception:
            # Only remove outputs created by this explicit add operation. The
            # preflight above prevents replacing a pre-existing track file.
            for created in (evidence_path, destination):
                try:
                    created.unlink(missing_ok=True)
                except OSError:
                    pass
            raise MusicCatalogError("Could not save the selected track and its license record; no track was added.") from None
        return {
            "track": track,
            "local_path": destination,
            "evidence_path": root / evidence_path,
            "sha256": digest,
            "size_bytes": len(response.body),
            "content_type": declared_type,
        }


def _project_root(project: Path) -> Path:
    try:
        root = Path(project).expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        raise MusicCatalogError("Project directory does not exist or cannot be accessed.") from None
    if not root.is_dir():
        raise MusicCatalogError("Project path must be a directory.")
    return root


def _ensure_parent_inside_project(root: Path, path: Path) -> None:
    try:
        parent = path.parent.resolve(strict=True)
        parent.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        raise MusicCatalogError("Music catalog files must remain inside the project's work directory.") from None


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"schema_version": "openverse-music-manifest.v1", "updated_at": None, "tracks": []}
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise MusicCatalogError("The project's Openverse music manifest cannot be read.") from None
    if not isinstance(value, dict) or not isinstance(value.get("tracks"), list):
        raise MusicCatalogError("The project's Openverse music manifest has an invalid format.")
    return value


def _atomic_bytes_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def _retry_delay(headers: Mapping[str, str], attempt: int) -> float | None:
    normalized_headers = {str(key).lower(): str(value) for key, value in headers.items()}
    retry_after = normalized_headers.get("retry-after", "").strip()
    if retry_after:
        try:
            delay = max(0.0, float(retry_after))
            return delay if delay <= MAX_AUTOMATIC_RETRY_WAIT else None
        except ValueError:
            try:
                retry_time = parsedate_to_datetime(retry_after)
                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(tzinfo=timezone.utc)
                delay = max(0.0, (retry_time - datetime.now(timezone.utc)).total_seconds())
                return delay if delay <= MAX_AUTOMATIC_RETRY_WAIT else None
            except (TypeError, ValueError, OverflowError):
                pass
    return min(MAX_AUTOMATIC_RETRY_WAIT, float(2 ** attempt))


def _canonical_audio_mime(value: str) -> str:
    mime = value.split(";", 1)[0].strip().lower()
    mime = _AUDIO_MIME_ALIASES.get(mime, mime)
    return mime if mime in _AUDIO_EXTENSIONS else ""


def _detect_audio_mime(body: bytes) -> str:
    if len(body) >= 2 and body[0] == 0xFF and body[1] & 0xF6 == 0xF0:
        return "audio/aac"
    if body.startswith(b"ID3") or (len(body) >= 2 and body[0] == 0xFF and body[1] & 0xE0 == 0xE0):
        return "audio/mpeg"
    if body.startswith(b"OggS"):
        return "audio/ogg"
    if len(body) >= 12 and body.startswith(b"RIFF") and body[8:12] == b"WAVE":
        return "audio/wav"
    if body.startswith(b"fLaC"):
        return "audio/flac"
    if body.startswith(b"\x1a\x45\xdf\xa3"):
        return "audio/webm"
    if len(body) >= 8 and body[4:8] == b"ftyp":
        return "audio/mp4"
    return ""


__all__ = [
    "HttpResponse",
    "LICENSE_FILTER",
    "MusicCatalogError",
    "OPENVERSE_NON_ENDORSEMENT",
    "OPENVERSE_WARNING",
    "OpenverseAudioCatalog",
    "OpenverseTrack",
]
