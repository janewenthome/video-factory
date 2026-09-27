"""Deterministic subtitle cue preparation for locally transcribed speech.

This module does not decide whether dialogue is meaningful. It creates timed
caption candidates from transcript words and exposes a separate explicit
keep/drop review boundary for an AI or human reviewer.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any


_CLOSING_PUNCTUATION = tuple("。！？!?；;，,：:")
_ASCII_WORD = re.compile(r"[A-Za-z0-9]$")
_ASCII_START = re.compile(r"^[A-Za-z0-9]")


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _clip_source(segment: Mapping[str, Any], index: int) -> tuple[str, float, float, float, float, str]:
    source = segment.get("source") or segment.get("source_url")
    if not isinstance(source, str) or not source.strip():
        raise ValueError(f"video timeline[{index}] is missing source/source_url")

    timeline_start = _number(segment.get("timeline_start"), f"timeline[{index}].timeline_start")
    timeline_end = _number(segment.get("timeline_end"), f"timeline[{index}].timeline_end")
    source_in = _number(segment.get("source_in", 0), f"timeline[{index}].source_in")
    if timeline_start < 0 or timeline_end <= timeline_start or source_in < 0:
        raise ValueError(f"video timeline[{index}] has invalid bounds")

    source_out_value = segment.get("source_out")
    source_out = (
        _number(source_out_value, f"timeline[{index}].source_out")
        if source_out_value is not None
        else source_in + (timeline_end - timeline_start)
    )
    if source_out <= source_in:
        raise ValueError(f"video timeline[{index}] has invalid source bounds")

    clip_id = segment.get("id")
    if not isinstance(clip_id, str) or not clip_id.strip():
        clip_id = f"clip-{index:04d}"
    return source, source_in, source_out, timeline_start, timeline_end, clip_id


def _word_items(transcript: Any) -> list[Mapping[str, Any]]:
    """Accept flat word records or common Whisper segment/word payloads."""
    if isinstance(transcript, Mapping):
        flat = transcript.get("words")
        if isinstance(flat, Sequence) and not isinstance(flat, (str, bytes)):
            return [item for item in flat if isinstance(item, Mapping)]
        segments = transcript.get("segments")
        if isinstance(segments, Sequence) and not isinstance(segments, (str, bytes)):
            result: list[Mapping[str, Any]] = []
            for segment in segments:
                if not isinstance(segment, Mapping):
                    continue
                words = segment.get("words")
                if isinstance(words, Sequence) and not isinstance(words, (str, bytes)):
                    for item in words:
                        if not isinstance(item, Mapping):
                            continue
                        enriched = dict(item)
                        if "speaker_id" not in enriched and segment.get("speaker_id") is not None:
                            enriched["speaker_id"] = segment["speaker_id"]
                        result.append(enriched)
                elif segment.get("text"):
                    # Whisper can return useful segment timestamps even when
                    # word timestamps are unavailable. Do not invent finer timing.
                    result.append(segment)
            return result
        return []
    if isinstance(transcript, Sequence) and not isinstance(transcript, (str, bytes)):
        return [item for item in transcript if isinstance(item, Mapping)]
    return []


def _token_text(word: Mapping[str, Any]) -> str:
    value = word.get("text", word.get("word", ""))
    return str(value).strip()


def _join_tokens(tokens: Sequence[str]) -> str:
    output = ""
    for token in tokens:
        if not token:
            continue
        if output and _ASCII_WORD.search(output) and _ASCII_START.search(token):
            output += " "
        output += token
    return output.strip()


def _ends_clause(text: str) -> bool:
    return text.endswith(_CLOSING_PUNCTUATION)


def _group_words(
    words: list[tuple[float, float, str, str | None, str]],
    *,
    max_chars: int,
    max_chars_per_second: float,
    max_duration: float,
    max_gap: float,
) -> list[list[tuple[float, float, str]]]:
    groups: list[list[tuple[float, float, str]]] = []
    current: list[tuple[float, float, str, str | None, str]] = []
    for word in words:
        if not current:
            current = [word]
            continue
        candidate_text = _join_tokens([item[2] for item in current] + [word[2]])
        gap = max(0.0, word[0] - current[-1][1])
        exceeds = (
            len(candidate_text) > max_chars
            or word[1] - current[0][0] > max_duration
            or len(candidate_text) / max(0.001, word[1] - current[0][0]) > max_chars_per_second
            or gap > max_gap
            or _ends_clause(current[-1][2])
            or word[3] != current[-1][3]
        )
        if exceeds:
            groups.append(current)
            current = [word]
        else:
            current.append(word)
    if current:
        groups.append(current)
    return groups


def _resolve_candidate_overlaps(candidates: list[dict[str, Any]]) -> None:
    """Split overlapping display intervals at the midpoint of their overlap."""
    for previous, current in zip(candidates, candidates[1:]):
        previous_start = float(previous["timeline_start"])
        previous_end = float(previous["timeline_end"])
        current_start = float(current["timeline_start"])
        current_end = float(current["timeline_end"])
        if current_start >= previous_end:
            continue
        overlap_start = max(previous_start, current_start)
        overlap_end = min(previous_end, current_end)
        boundary = round((overlap_start + overlap_end) / 2, 3)
        # Keep both cues positive when millisecond rounding hits an edge.
        if boundary <= previous_start:
            boundary = round(previous_start + 0.001, 3)
        if boundary >= current_end:
            boundary = round(current_end - 0.001, 3)
        previous["timeline_end"] = boundary
        current["timeline_start"] = boundary


def generate_subtitle_candidates(
    edit_plan: Mapping[str, Any],
    transcripts_by_source: Mapping[str, Any],
    *,
    timestamp_reference: str = "clip",
    max_chars: int = 18,
    max_chars_per_second: float = 8.0,
    max_duration_seconds: float = 4.5,
    max_gap_seconds: float = 0.9,
) -> list[dict[str, Any]]:
    """Map transcript timing into candidate cues for retained natural audio.

    Only ``video`` segments with ``audio.preserve_natural is True`` are used.
    Transcript records are looked up by clip ID first, then source path. By
    default their timestamps are relative to that clip's extracted/cut audio;
    pass ``timestamp_reference='source'`` for source-file absolute timestamps.
    A word must fit wholly inside the retained trim. Cues remain pending until
    an explicit semantic review result is supplied.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if max_chars_per_second <= 0:
        raise ValueError("max_chars_per_second must be positive")
    if timestamp_reference not in {"clip", "source"}:
        raise ValueError("timestamp_reference must be 'clip' or 'source'")
    if max_duration_seconds <= 0 or max_gap_seconds < 0:
        raise ValueError("duration and gap limits must be positive/non-negative")

    timeline = edit_plan.get("timeline", [])
    if not isinstance(timeline, Sequence) or isinstance(timeline, (str, bytes)):
        raise ValueError("edit_plan.timeline must be a sequence")

    candidates: list[dict[str, Any]] = []
    for index, segment in enumerate(timeline):
        if not isinstance(segment, Mapping) or segment.get("type") != "video":
            continue
        audio = segment.get("audio")
        if not isinstance(audio, Mapping) or audio.get("preserve_natural") is not True:
            continue

        source, source_in, source_out, timeline_start, timeline_end, clip_id = _clip_source(segment, index)
        transcript = transcripts_by_source.get(clip_id)
        if transcript is None:
            transcript = transcripts_by_source.get(source)
        if transcript is None:
            continue

        retained_duration = min(source_out - source_in, timeline_end - timeline_start)
        timed_words: list[tuple[float, float, str, str | None, str]] = []
        for word_index, item in enumerate(_word_items(transcript)):
            text = _token_text(item)
            if not text:
                continue
            start = _number(item.get("start"), f"{source} word[{word_index}].start")
            end = _number(item.get("end"), f"{source} word[{word_index}].end")
            if end <= start:
                raise ValueError(f"{source} word[{word_index}] has end <= start")
            # Do not emit partial words at edit boundaries.
            if timestamp_reference == "clip":
                if start < 0 or end > retained_duration:
                    continue
                relative_start, relative_end = start, end
            else:
                if start < source_in or end > source_out:
                    continue
                relative_start, relative_end = start - source_in, end - source_in
                if relative_end > retained_duration:
                    continue
            mapped_start = timeline_start + relative_start
            mapped_end = timeline_start + relative_end
            if mapped_end > mapped_start:
                speaker_id = item.get("speaker_id", item.get("speaker"))
                speaker_id = str(speaker_id) if speaker_id is not None else None
                word_id = item.get("id") or f"{clip_id}:word-{word_index + 1:04d}"
                timed_words.append((mapped_start, mapped_end, text, speaker_id, str(word_id)))

        timed_words.sort(key=lambda item: (item[0], item[1], item[2], item[4]))
        groups = _group_words(
            timed_words,
            max_chars=max_chars,
            max_chars_per_second=max_chars_per_second,
            max_duration=max_duration_seconds,
            max_gap=max_gap_seconds,
        )
        for cue_index, group in enumerate(groups, start=1):
            text = _join_tokens([word[2] for word in group])
            if not text:
                continue
            cue_start = max(timeline_start, group[0][0])
            cue_end = min(timeline_end, group[-1][1])
            if cue_end <= cue_start:
                continue
            cue_id = f"{clip_id}:subtitle-{cue_index:03d}"
            candidates.append(
                {
                    "id": cue_id,
                    "type": "subtitle",
                    "text": text,
                    "timeline_start": round(cue_start, 3),
                    "timeline_end": round(cue_end, 3),
                    "source": source,
                    "source_clip_id": clip_id,
                    "source_start": round(group[0][0] - timeline_start + source_in, 3),
                    "source_end": round(group[-1][1] - timeline_start + source_in, 3),
                    "word_ids": [word[4] for word in group],
                    "review_status": "pending",
                }
            )
            speaker_ids = {word[3] for word in group}
            speaker_id = next(iter(speaker_ids)) if len(speaker_ids) == 1 else None
            if speaker_id is not None:
                candidates[-1]["speaker_id"] = speaker_id

    candidates.sort(key=lambda cue: (cue["timeline_start"], cue["timeline_end"], cue["id"]))
    _resolve_candidate_overlaps(candidates)
    return candidates


def apply_subtitle_reviews(
    candidates: Sequence[Mapping[str, Any]],
    review_results: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Apply explicit semantic keep/drop results without inferring meaning.

    Review values may be ``'keep'``/``'drop'`` or mappings with a
    ``decision`` and optional ``reason``. Decisions can use the cue ID, a
    ``phrase:<exact caption text>`` key (or the exact text), or word IDs
    (optionally prefixed with ``word:``). A drop on any word drops the whole
    cue so the helper never creates a semantically damaged phrase. Missing or
    partial decisions remain pending. Returns separate lists for auditability.
    """
    result: dict[str, list[dict[str, Any]]] = {"keep": [], "drop": [], "pending": []}
    for candidate in candidates:
        cue = dict(candidate)
        cue_id = cue.get("id")
        raw = review_results.get(str(cue_id)) if cue_id is not None else None
        if raw is None:
            text = str(cue.get("text", ""))
            raw = review_results.get(f"phrase:{text}", review_results.get(text))
        if raw is None:
            word_ids = cue.get("word_ids", [])
            word_decisions: list[tuple[str, Any]] = []
            if isinstance(word_ids, Sequence) and not isinstance(word_ids, (str, bytes)):
                for word_id in word_ids:
                    word_result = review_results.get(f"word:{word_id}", review_results.get(str(word_id)))
                    if word_result is not None:
                        if isinstance(word_result, Mapping):
                            decision_value = str(word_result.get("decision", "")).strip().lower()
                        else:
                            decision_value = str(word_result).strip().lower()
                        word_decisions.append((str(word_id), decision_value))
            if any(value == "drop" for _, value in word_decisions):
                raw = {"decision": "drop", "reason": "AI review dropped at least one constituent word"}
            elif word_ids and len(word_decisions) == len(word_ids) and all(
                value == "keep" for _, value in word_decisions
            ):
                raw = {"decision": "keep", "reason": "all constituent words were kept"}
            elif any(value not in {"keep", "drop"} for _, value in word_decisions):
                raise ValueError(f"word review for {cue_id!r} must be 'keep' or 'drop'")
        if raw is None:
            cue["review_status"] = "pending"
            result["pending"].append(cue)
            continue
        if isinstance(raw, Mapping):
            decision = str(raw.get("decision", "")).strip().lower()
            reason = raw.get("reason")
        else:
            decision = str(raw).strip().lower()
            reason = None
        if decision not in {"keep", "drop"}:
            raise ValueError(f"review for {cue_id!r} must be 'keep' or 'drop'")
        cue["review_status"] = decision
        if reason is not None:
            cue["review_reason"] = str(reason)
        result[decision].append(cue)
    return result


def validate_subtitle_cues(
    cues: Sequence[Mapping[str, Any]],
    *,
    duration_seconds: float | None = None,
    minimum_duration_seconds: float = 0.15,
) -> list[str]:
    """Return deterministic QA errors for malformed, overlapping, or out-of-range cues."""
    errors: list[str] = []
    duration = _number(duration_seconds, "duration_seconds") if duration_seconds is not None else None
    minimum = _number(minimum_duration_seconds, "minimum_duration_seconds")
    if minimum < 0:
        raise ValueError("minimum_duration_seconds must be non-negative")

    parsed: list[tuple[int, float, float, str]] = []
    for index, cue in enumerate(cues):
        cue_id = str(cue.get("id", f"cue-{index + 1}"))
        try:
            start = _number(cue.get("timeline_start"), f"{cue_id}.timeline_start")
            end = _number(cue.get("timeline_end"), f"{cue_id}.timeline_end")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        text = str(cue.get("text", "")).strip()
        if not text:
            errors.append(f"{cue_id}: text is empty")
        if start < 0:
            errors.append(f"{cue_id}: start is before timeline zero")
        if end <= start:
            errors.append(f"{cue_id}: end must be after start")
        elif end - start < minimum:
            errors.append(f"{cue_id}: cue is shorter than {minimum:.3f}s")
        if duration is not None and end > duration + 0.001:
            errors.append(f"{cue_id}: end exceeds edit duration")
        parsed.append((index, start, end, cue_id))

    parsed.sort(key=lambda row: (row[1], row[2], row[0]))
    for previous, current in zip(parsed, parsed[1:]):
        if current[1] < previous[2] - 0.001:
            errors.append(f"{previous[3]} overlaps {current[3]}")
    return errors
