"""Music discovery, licensing and placement contracts.

The default policy is ``SAFE_AUTO``: only public-domain, CC0 and CC BY
tracks with verifiable upstream metadata may be selected automatically.
Network providers are adapters, not hidden side effects.  A normal pipeline
run writes requirements and attribution artifacts and can finish without music
when a provider is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol

from video_editor.product_policy import ProductPolicy


SAFE_LICENSES = {"public_domain", "public domain", "public_domain_mark", "public domain mark", "cc0", "cc-by", "cc by", "cc_by"}
REJECTED_LICENSE_MARKERS = {"nc", "nd", "sa", "unknown", "all rights reserved"}
PROVIDER_ENDPOINTS = {
    "openverse": "https://api.openverse.org/v1/audio/",
    "jamendo": "https://api.jamendo.com/v3.0/tracks/",
    "wikimedia": "https://commons.wikimedia.org/w/api.php",
}


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _safe_relative(project: Path, value: str) -> bool:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        return False
    return (project / path).resolve(strict=False).is_relative_to(project.resolve(strict=True))


@dataclass(frozen=True)
class MusicRequirement:
    id: str
    start_seconds: float
    end_seconds: float
    tags: tuple[str, ...]
    avoid_tags: tuple[str, ...]
    preserve_natural_audio: bool
    dialogue_protection_db: float = -12.0
    content_type: str = "video"
    energy: str = "medium"
    instrumental_preferred: bool = True
    dialogue_present: bool = False
    original_audio_importance: str = "none"
    loop_allowed: bool = True
    fade_preference: str = "fade"
    music_required: bool = False
    music_preferred: bool = False


@dataclass(frozen=True)
class MusicCandidate:
    id: str
    title: str
    provider: str
    license: str
    license_url: str
    source_url: str
    download_url: str
    duration_seconds: float | None = None
    attribution: str = ""
    tags: tuple[str, ...] = ()
    checksum: str | None = None
    creator: str = ""
    creator_url: str = ""
    download_allowed: bool = True


@dataclass(frozen=True)
class LicenseDecision:
    accepted: bool
    reason: str
    normalized_license: str
    evidence: dict[str, str] = field(default_factory=dict)


class MusicProvider(Protocol):
    name: str

    def search(self, requirement: MusicRequirement) -> Iterable[MusicCandidate]:
        ...


class _CatalogProvider:
    """Network boundary for a public catalog; disabled until injected."""

    name = "catalog"
    endpoint = ""

    def __init__(self, requester: Any | None = None):
        self.requester = requester

    def search(self, requirement: MusicRequirement) -> Iterable[MusicCandidate]:
        if self.requester is None:
            return ()
        # Requesters are injected by an application that has explicit network
        # authorization; this package never imports requests or calls a URL.
        value = self.requester(self.endpoint, requirement)
        return tuple(value) if value else ()


class OpenverseProvider(_CatalogProvider):
    name = "openverse"
    endpoint = PROVIDER_ENDPOINTS["openverse"]


class JamendoProvider(_CatalogProvider):
    name = "jamendo"
    endpoint = PROVIDER_ENDPOINTS["jamendo"]


class WikimediaProvider(_CatalogProvider):
    name = "wikimedia"
    endpoint = PROVIDER_ENDPOINTS["wikimedia"]


class WikimediaCommonsProvider(WikimediaProvider):
    """Explicit name used by the product contract; keep the short alias too."""

    name = "wikimedia_commons"


class MusicRequirementAnalyzer:
    def analyze(self, plan: dict[str, Any], policy: ProductPolicy) -> list[MusicRequirement]:
        requirements: list[MusicRequirement] = []
        for index, segment in enumerate(plan.get("timeline", [])):
            if not isinstance(segment, dict) or segment.get("type") not in {"video", "photo"}:
                continue
            start = _number(segment.get("timeline_start"), 0.0)
            end = _number(segment.get("timeline_end"), start)
            if end <= start:
                continue
            content_type = str(segment.get("content_type") or segment.get("type") or "video").lower()
            classification = str(segment.get("audio_classification") or "").upper()
            dialogue_present = bool(segment.get("dialogue_present")) or classification == "DIALOGUE"
            audio_importance = str(segment.get("original_audio_importance") or "").lower()
            if not audio_importance:
                if content_type == "photo" or segment.get("has_audio") is False:
                    audio_importance = "none"
                elif classification == "IMPORTANT_NATURAL_SOUND":
                    audio_importance = "important"
                elif dialogue_present:
                    audio_importance = "dialogue"
                else:
                    audio_importance = "ambient"
            music_required = bool(segment.get("music_required")) or audio_importance in {"none", "silent"}
            music_preferred = bool(segment.get("music_preferred")) or (
                not music_required and not dialogue_present and audio_importance in {"low", "ambient"}
            )
            # SAFE_AUTO starts with silent/photo segments. Important speech or
            # natural sound is never turned into a music slot implicitly.
            if not music_required and not music_preferred:
                continue
            requirements.append(MusicRequirement(
                id=f"music-slot-{index + 1:03d}",
                start_seconds=start,
                end_seconds=end,
                tags=("warm", "reflective") if policy.mode == "family" else ("clear", "reassuring"),
                avoid_tags=("dramatic", "lyrics", "high_energy") if policy.mode == "family" else ("lyrics", "dramatic", "anxiety"),
                preserve_natural_audio=policy.preserve_natural_audio,
                content_type=content_type,
                energy="medium" if policy.mode == "family" else "low_to_medium",
                instrumental_preferred=True,
                dialogue_present=dialogue_present,
                original_audio_importance=audio_importance,
                loop_allowed=True,
                fade_preference="crossfade" if end - start > 8 else "fade",
                music_required=music_required,
                music_preferred=music_preferred,
            ))
        return requirements


class MusicLicenseVerifier:
    def verify(self, candidate: MusicCandidate) -> LicenseDecision:
        raw = (candidate.license or "").strip().lower().replace("/", "-")
        if any(marker in raw for marker in REJECTED_LICENSE_MARKERS):
            return LicenseDecision(False, "license contains a non-commercial or share-alike restriction", raw)
        normalized = raw.replace(" ", "_")
        is_cc_by_version = bool(re.match(r"^cc[- ]by(?:[- ]|$)", raw)) and not any(marker in raw for marker in ("nc", "nd", "sa"))
        if raw not in SAFE_LICENSES and normalized not in SAFE_LICENSES and not is_cc_by_version:
            return LicenseDecision(False, "license is not in SAFE_AUTO allowlist", normalized)
        if is_cc_by_version:
            normalized = "cc_by"
        urls = (candidate.license_url, candidate.source_url, candidate.download_url)
        if not candidate.download_allowed:
            return LicenseDecision(False, "provider did not authorize a downloadable file", normalized)
        if not all(re.fullmatch(r"https?://[^\s]+", value or "") for value in urls):
            return LicenseDecision(False, "upstream license, source and download URLs are required", normalized)
        return LicenseDecision(True, "verified allowlisted license with upstream evidence", normalized, {
            "license_url": candidate.license_url,
            "source_url": candidate.source_url,
            "download_url": candidate.download_url,
        })


class MusicCandidateRanker:
    def rank(
        self,
        candidates: Iterable[MusicCandidate],
        requirement: MusicRequirement,
        verifier: MusicLicenseVerifier | None = None,
    ) -> list[MusicCandidate]:
        verifier = verifier or MusicLicenseVerifier()

        def score(candidate: MusicCandidate) -> tuple[int, int, str]:
            tag_hits = len(set(candidate.tags).intersection(requirement.tags))
            avoid_hits = len(set(candidate.tags).intersection(requirement.avoid_tags))
            duration_fit = 0 if candidate.duration_seconds is None else abs(candidate.duration_seconds - (requirement.end_seconds - requirement.start_seconds))
            return (avoid_hits * 100 - tag_hits, int(duration_fit), candidate.id)

        # License safety is a hard gate, never a ranking bonus.
        return sorted((candidate for candidate in candidates if verifier.verify(candidate).accepted), key=score)


class MusicSearchService:
    def __init__(self, providers: Iterable[MusicProvider] = ()):
        self.providers = tuple(providers)
        self.health: dict[str, str] = {}

    def search(self, requirement: MusicRequirement) -> list[MusicCandidate]:
        candidates: list[MusicCandidate] = []
        for provider in self.providers:
            try:
                candidates.extend(provider.search(requirement))
                self.health[getattr(provider, "name", provider.__class__.__name__)] = "AVAILABLE"
            except Exception:
                # A provider outage must not fail the whole video job.
                self.health[getattr(provider, "name", provider.__class__.__name__)] = "UNAVAILABLE"
        return candidates


class MusicDownloader:
    """Interface boundary; no network download is performed implicitly."""

    def __init__(self, downloader: Any | None = None):
        self.downloader = downloader

    def download(self, candidate: MusicCandidate, destination: Path) -> Path:
        if self.downloader is None:
            raise RuntimeError("Music downloads require an explicit provider operation and are not enabled in the default pipeline.")
        result = self.downloader(candidate, destination)
        return Path(result or destination)


class MusicAnalyzer:
    def analyze(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {"path": path.as_posix(), "status": "missing"}
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return {"path": path.as_posix(), "status": "ffprobe_unavailable"}
        try:
            result = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration:stream=channels,sample_rate", "-of", "json", str(path)],
                capture_output=True, text=True, check=False, timeout=60,
            )
            value = json.loads(result.stdout) if result.returncode == 0 else {}
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            value = {}
        streams = value.get("streams", []) if isinstance(value, dict) else []
        audio = streams[0] if streams and isinstance(streams[0], dict) else {}
        duration = (value.get("format", {}) or {}).get("duration") if isinstance(value, dict) else None
        return {
            "path": path.as_posix(),
            "status": "completed_local" if value else "analysis_failed",
            "duration_seconds": _number(duration, 0.0),
            "channels": audio.get("channels"),
            "sample_rate": audio.get("sample_rate"),
        }


class MusicPlacementPlanner:
    def plan(self, requirements: Iterable[MusicRequirement], candidates: Iterable[MusicCandidate]) -> list[dict[str, Any]]:
        selected = list(candidates)
        placements: list[dict[str, Any]] = []
        for requirement, candidate in zip(requirements, selected):
            placements.append({
                "requirement_id": requirement.id,
                "candidate_id": candidate.id,
                "timeline_start": requirement.start_seconds,
                "timeline_end": requirement.end_seconds,
                "volume": 0.2,
                "ducking_db": requirement.dialogue_protection_db,
                "fade_in_seconds": 0.5,
                "fade_out_seconds": 0.8,
            })
        return placements


class MusicMixer:
    def mix(self, placements: Iterable[dict[str, Any]]) -> dict[str, Any]:
        return {"status": "planned", "placements": list(placements), "renderer": "Mac-local"}


class MusicCache:
    def __init__(self, project: Path):
        self.project = project
        self.directory = project / "work" / "music-cache"
        self.directory.mkdir(parents=True, exist_ok=True)

    def key(self, requirement: MusicRequirement, policy: ProductPolicy) -> str:
        return _hash({"requirement": requirement.__dict__, "policy": policy.music_license_policy})

    def path(self, key: str) -> Path:
        return self.directory / f"{key}.json"


class MusicAttributionGenerator:
    def write(self, project: Path, candidates: Iterable[MusicCandidate], decisions: Iterable[LicenseDecision]) -> dict[str, Any]:
        entries = []
        retrieved_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        for candidate, decision in zip(candidates, decisions):
            if not decision.accepted:
                continue
            entries.append({
                "id": candidate.id,
                "title": candidate.title,
                "provider": candidate.provider,
                "license": decision.normalized_license,
                "license_url": candidate.license_url,
                "source_url": candidate.source_url,
                "download_url": candidate.download_url,
                "attribution": candidate.attribution,
                "creator": candidate.creator,
                "creator_url": candidate.creator_url,
                "checksum": candidate.checksum,
                "retrieved_at": retrieved_at,
                "verification_status": "verified",
                "verification_reason": decision.reason,
                "license_version": decision.normalized_license,
                "trimmed": False,
                "looped": False,
                "fades_added": False,
                "gain_changed": False,
            })
        payload = {"schema_version": "music-attribution.v1", "tracks": entries}
        work = project / "work"
        work.mkdir(parents=True, exist_ok=True)
        outputs = project / "outputs"
        outputs.mkdir(parents=True, exist_ok=True)
        outputs_work = outputs / "work"
        outputs_work.mkdir(parents=True, exist_ok=True)
        outputs_final = outputs / "final"
        outputs_final.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        (work / "music_attribution.json").write_text(serialized, encoding="utf-8")
        (outputs_work / "music_attribution.json").write_text(serialized, encoding="utf-8")
        (outputs_final / "music_attribution.json").write_text(serialized, encoding="utf-8")
        licenses = outputs_work / "music" / "licenses"
        licenses.mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": "music-manifest.v1", "tracks": []}
        for entry in entries:
            evidence = {key: entry.get(key) for key in (
                "id", "title", "provider", "license", "license_url", "source_url", "creator", "creator_url",
                "download_url", "retrieved_at", "verification_status", "verification_reason", "checksum",
            )}
            (licenses / f"{entry['id']}.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            manifest["tracks"].append({"id": entry["id"], "title": entry["title"], "provider": entry["provider"], "license": entry["license"]})
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        (outputs_work / "music" / "music_manifest.json").write_text(manifest_text, encoding="utf-8")
        credits = ["# MUSIC_CREDITS", "", "No automatically selected track." if not entries else ""]
        publishing = ["# PUBLISHING_CREDITS", "", "No external music was downloaded by the default pipeline." if not entries else ""]
        for entry in entries:
            line = f"- {entry['title']} — {entry['license']} — {entry['source_url']}"
            credits.append(line)
            publishing.append(line)
        credits_text = "\n".join(credits) + "\n"
        publishing_text = "\n".join(publishing) + "\n"
        for directory in (outputs, outputs_final):
            (directory / "MUSIC_CREDITS.txt").write_text(credits_text, encoding="utf-8")
            (directory / "PUBLISHING_CREDITS.txt").write_text(publishing_text, encoding="utf-8")
        return payload


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def prepare_music_artifacts(project: Path, policy: ProductPolicy, plan: dict[str, Any]) -> dict[str, Any]:
    analyzer = MusicRequirementAnalyzer()
    requirements = analyzer.analyze(plan, policy)
    status = "disabled" if policy.music_mode == "none" else ("no_music_required" if not requirements else "pending_discovery")
    payload = {
        "schema_version": "music-requirements.v1",
        "mode": policy.music_mode,
        "license_policy": policy.music_license_policy,
        "status": status,
        "silent_segments_only": True,
        "provider_order": ["openverse", "jamendo", "wikimedia_commons"],
        "provider_endpoints": PROVIDER_ENDPOINTS,
        "requirements": [requirement.__dict__ for requirement in requirements],
        "fallback": "finish_without_music_if_provider_or_license_verification_fails",
    }
    work = project / "work"
    work.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    (work / "music_requirements.json").write_text(serialized, encoding="utf-8")
    outputs_work = project / "outputs" / "work"
    outputs_work.mkdir(parents=True, exist_ok=True)
    (outputs_work / "music_requirements.json").write_text(serialized, encoding="utf-8")
    MusicAttributionGenerator().write(project, (), ())
    return payload


__all__ = [
    "MusicAttributionGenerator",
    "MusicCandidate",
    "MusicCandidateRanker",
    "MusicDownloader",
    "MusicLicenseVerifier",
    "MusicMixer",
    "MusicPlacementPlanner",
    "MusicProvider",
    "OpenverseProvider",
    "JamendoProvider",
    "WikimediaProvider",
    "WikimediaCommonsProvider",
    "MusicRequirement",
    "MusicRequirementAnalyzer",
    "MusicSearchService",
    "MusicAnalyzer",
    "prepare_music_artifacts",
]
