"""Split diarized ASR segments without guessing word times or dropping text."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict
from typing import Any

from workers.semantic_boundaries import (
    BoundaryCandidate,
    BoundaryPlan,
    BoundarySelector,
    TokenizerAdapter,
    generate_boundary_plan,
    rank_candidate_ids,
)

MAX_SECONDS = 14.0
# When no natural pause exists before MAX_SECONDS, allow a controlled overflow so
# that CJK compounds (which WhisperX tokenizes per character) are not split mid-word.
HARD_MAX_SECONDS = 20.0
PREFERRED_SECONDS = 8.0
MIN_FRAGMENT_SECONDS = 2.5
MERGE_GAP_SECONDS = 0.65
PAUSE_SECONDS = 0.65
TERMINAL_PUNCTUATION = re.compile(r"""[.!?。！？…]["'”’）)\]]*$""")
DEFAULT_SPEAKER = "SPEAKER_00"
UNKNOWN_SPEAKER = "UNKNOWN"


def _time(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        return round(value, 3)
    return None


def _word_offsets(text: str, words: list[dict]) -> list[int] | None:
    # WhisperX can insert spaces when grouping CJK words. Match the original text
    # ignoring whitespace, then cut the original string at the matched offsets.
    letters = [(offset, char) for offset, char in enumerate(text) if not char.isspace()]
    compact = "".join(char for _, char in letters)
    cursor = 0
    offsets = []
    for word in words:
        token = "".join(str(word.get("word") or "").split())
        if not token or not compact.startswith(token, cursor):
            return None
        offsets.append(letters[cursor][0])
        cursor += len(token)
    return offsets if cursor == len(compact) else None


def aligned_cut_indices(raw: dict) -> list[int]:
    """Return word indices that can start an evidence-backed second line."""
    words = raw.get("words") or []
    start, end = _time(raw.get("start")), _time(raw.get("end"))
    if start is None or end is None or not _word_offsets(str(raw.get("text") or "").strip(), words):
        return []
    times = [(_time(w.get("start")), _time(w.get("end"))) for w in words]
    if any(a is None or b is None or not start <= a < b <= end for a, b in times):
        return []
    if any(left[1] > right[0] for left, right in zip(times, times[1:])):
        return []
    # Reject single-frame fragments and implausibly stretched boundary words.
    return [
        i for i, (left, right) in enumerate(zip(times, times[1:]), 1)
        if left[1] - start >= 0.15 and end - right[0] >= 0.15
        and left[1] - left[0] <= 1.0 and right[1] - right[0] <= 1.0
    ]


def _split_segment(
    raw: dict, diarized: bool, *, language: str | None = None,
    tokenizer: TokenizerAdapter | None = None,
    boundary_selector: BoundarySelector | None = None,
    trim_to_aligned_words: bool = False,
    split_sentences: bool = False,
    semantic_cuts: frozenset[int] | None = None,
) -> list[dict]:
    text = str(raw.get("text") or "").strip()
    if not text:
        return []
    raw_start, raw_end = _time(raw.get("start")), _time(raw.get("end"))
    start = raw_start if raw_start is not None else 0.0
    end = raw_end if raw_end is not None and raw_end > start else round(start + 0.25, 3)
    invalid_source_bounds = raw_start is None or raw_end != end
    rank_speaker = (str(raw.get("speaker") or "").strip() or None) if diarized else None
    words = []
    for item in raw.get("words") or []:
        word_start, word_end = _time(item.get("start")), _time(item.get("end"))
        if word_start is None or word_end is None or not start <= word_start < word_end <= end:
            word_start = word_end = None
        speaker = (str(item.get("speaker") or "").strip() or None) if diarized else None
        words.append(dict(word=str(item.get("word") or ""), start=word_start, end=word_end, speaker=speaker))

    offsets = _word_offsets(text, words)
    # ASR/VAD windows can include silence or another speaker outside the text.
    # Tighten only a complete, ordered alignment. The source evidence retains
    # the ASR window; never crop missing text or change individual word times.
    if (trim_to_aligned_words and offsets and raw_start is not None and raw_end == end
            and all(word["start"] is not None and word["end"] is not None for word in words)
            and all(left["end"] <= right["start"] for left, right in zip(words, words[1:]))):
        start, end = words[0]["start"], words[-1]["end"]
    # A valid cut needs aligned words on both sides and must preserve their audio.
    cuts = [
        i for i in range(1, len(words))
        if words[i - 1]["end"] is not None and words[i]["start"] is not None
        and words[i - 1]["end"] <= words[i]["start"]
    ] if offsets else []

    plan = BoundaryPlan(()) if semantic_cuts is not None else generate_boundary_plan(
        text, words, language=language, tokenizer=tokenizer, pause_seconds=PAUSE_SECONDS,
    )
    review_reasons = list(plan.review_reasons)
    semantic = {candidate.word_index: candidate for candidate in plan.candidates}
    chosen: dict[int, dict] = {}
    source_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    def record(candidate: BoundaryCandidate) -> dict:
        value = asdict(candidate)
        value["candidate_id"] = value.pop("id")
        value["reasons"] = list(candidate.reasons)
        value["source_text_sha256"] = source_text_sha256
        return value

    def speaker_at(index: int) -> str | None:
        return words[index]["speaker"] or rank_speaker

    def part(left: int, right: int) -> dict:
        part_start = words[left]["start"] if left else start
        part_end = words[right - 1]["end"] if right < len(words) else end
        left_offset = offsets[left] if left and offsets else 0
        right_offset = offsets[right] if right < len(words) and offsets else len(text)
        part_words = words[left:right]
        alignment, speaker_review = _review_flags(dict(
            start=part_start,
            end=part_end,
            words=part_words,
        ), diarized, alignment_extra=(not offsets or invalid_source_bounds))
        return dict(
            start=part_start,
            end=part_end,
            speaker=DEFAULT_SPEAKER if not diarized else (
                next((speaker_at(i) for i in range(left, right) if speaker_at(i)), None)
                or rank_speaker or DEFAULT_SPEAKER
            ),
            orig_text=text[left_offset:right_offset].strip(),
            words=part_words,
            alignment_needs_review=alignment or bool(review_reasons),
            speaker_needs_review=speaker_review,
            boundary_provenance=[chosen[index] for index in (left, right) if index in chosen],
            boundary_review_reasons=list(review_reasons),
        )

    # A missing word label is a review signal rather than a speaker transition.
    # Otherwise gaps in diarization can isolate individual aligned characters.
    speaker_cuts = set()
    for i in cuts:
        previous, current = words[i - 1]["speaker"], words[i]["speaker"]
        if previous not in {None, UNKNOWN_SPEAKER} and current not in {None, UNKNOWN_SPEAKER} and previous != current:
            speaker_cuts.add(i)
    sentence_cuts = {i for i in cuts if i in semantic} if split_sentences else set()
    if semantic_cuts is None:
        forced = sorted(speaker_cuts | sentence_cuts)
    else:
        speaker_cuts.clear()
        forced = sorted(semantic_cuts.intersection(aligned_cut_indices(raw)))
    for index in forced:
        candidate = semantic.get(index) or BoundaryCandidate(
            id=f"boundary:{source_text_sha256[:16]}:{offsets[index]}:{words[index - 1]['end']:.6f}:{words[index]['start']:.6f}",
            word_index=index, char_offset=offsets[index],
            left_end=words[index - 1]["end"], right_start=words[index]["start"],
            reasons=("semantic_sentence_boundary",) if semantic_cuts is not None else ("speaker_change",),
        )
        chosen[index] = record(candidate)
        if index in speaker_cuts and "speaker_change" not in chosen[index]["reasons"]:
            chosen[index]["reasons"].append("speaker_change")
    segments = []
    left = 0
    for right in [*forced, len(words)]:
        part_start = words[left]["start"] if left else start
        part_end = words[right - 1]["end"] if right < len(words) else end
        while semantic_cuts is None and part_end - part_start > MAX_SECONDS:
            available = [
                candidate for candidate in plan.candidates
                if left < candidate.word_index < right
                and candidate.left_end - part_start <= HARD_MAX_SECONDS
            ]
            preferred = [
                candidate for candidate in available
                if PREFERRED_SECONDS <= candidate.left_end - part_start <= MAX_SECONDS
            ]
            extended = [candidate for candidate in available if candidate.left_end - part_start > MAX_SECONDS]
            early = [candidate for candidate in available if candidate.left_end - part_start < PREFERRED_SECONDS]
            pool = preferred or extended or early
            if not pool:
                if "no_reliable_boundary" not in review_reasons:
                    review_reasons.append("no_reliable_boundary")
                break
            # Prefer the latest natural boundary in the normal window; when
            # extending, stop at the first valid boundary rather than hard-cutting.
            ordered = tuple(sorted(pool, key=lambda item: item.left_end, reverse=bool(preferred or early)))
            ranked, selector_issues = rank_candidate_ids(ordered, boundary_selector)
            review_reasons.extend(reason for reason in selector_issues if reason not in review_reasons)
            candidate = ranked[0]
            cut = candidate.word_index
            chosen[cut] = record(candidate)
            segments.append(part(left, cut))
            left = cut
            part_start = words[left]["start"]
        segments.append(part(left, right))
        left = right
    return segments


def _join_text(left: str, right: str) -> str:
    left = left.rstrip()
    right = right.lstrip()
    if not left:
        return right
    if not right:
        return left
    if bool(re.search(r"[A-Za-z0-9]$", left)) and bool(re.match(r"[A-Za-z0-9]", right)):
        return f"{left} {right}"
    if left[-1] in ",;:" and bool(re.match(r"[A-Za-z0-9]", right)):
        return f"{left} {right}"
    return f"{left}{right}"


def _review_flags(seg: dict, diarized: bool, *, alignment_extra: bool = False) -> tuple[bool, bool]:
    words = seg.get("words") or []
    starts = [word["start"] for word in words if word["start"] is not None]
    alignment = (
        alignment_extra
        or not words or len(starts) != len(words) or starts != sorted(starts)
        or seg["end"] - seg["start"] > MAX_SECONDS
    )
    speakers = {word.get("speaker") for word in words if word.get("start") is not None}
    speaker_review = (
        not diarized
        or len(speakers) != 1
        or None in speakers
        or speakers == {UNKNOWN_SPEAKER}
        or any(word.get("start") is not None and not word.get("speaker") for word in words)
    )
    return alignment, speaker_review


def _merge_short_segments(segments: list[dict], *, diarized: bool) -> list[dict]:
    """合并过短的同角色相邻片段，避免 ASR/VAD 把一句话切成碎片。"""
    merged: list[dict] = []
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        prev = merged[-1]
        gap = round(seg["start"] - prev["end"], 3)
        short_piece = (
            prev["end"] - prev["start"] < MIN_FRAGMENT_SECONDS
            or seg["end"] - seg["start"] < MIN_FRAGMENT_SECONDS
        )
        review_ok = (
            (not diarized and not prev["alignment_needs_review"] and not seg["alignment_needs_review"])
            or (
                diarized
                and not prev["alignment_needs_review"] and not prev["speaker_needs_review"]
                and not seg["alignment_needs_review"] and not seg["speaker_needs_review"]
            )
        )
        can_merge = (
            prev["speaker"] == seg["speaker"]
            and 0 <= gap <= MERGE_GAP_SECONDS
            and (seg["end"] - prev["start"]) <= MAX_SECONDS
            and short_piece
            and not TERMINAL_PUNCTUATION.search(prev["orig_text"].rstrip())
            and review_ok
        )
        if not can_merge:
            merged.append(seg)
            continue
        combined = dict(prev)
        combined["end"] = seg["end"]
        combined["orig_text"] = _join_text(prev["orig_text"], seg["orig_text"])
        combined["words"] = [*prev.get("words", []), *seg.get("words", [])]
        alignment, speaker = _review_flags(combined, diarized)
        combined["alignment_needs_review"] = alignment or prev["alignment_needs_review"] or seg["alignment_needs_review"]
        combined["speaker_needs_review"] = speaker or prev["speaker_needs_review"] or seg["speaker_needs_review"]
        combined["boundary_provenance"] = [*prev.get("boundary_provenance", []), *seg.get("boundary_provenance", [])]
        combined["boundary_review_reasons"] = list(dict.fromkeys(
            [*prev.get("boundary_review_reasons", []), *seg.get("boundary_review_reasons", [])]
        ))
        merged[-1] = combined
    return merged


def normalize_segments(
    raw_segments: list[dict], *, diarized: bool = False,
    language: str | None = None, tokenizer: TokenizerAdapter | None = None,
    boundary_selector: BoundarySelector | None = None,
    trim_to_aligned_words: bool = False,
    split_sentences: bool = False,
    semantic_cuts: frozenset[int] | None = None,
) -> list[dict]:
    segments = [part for raw in raw_segments for part in _split_segment(
        raw, diarized, language=language, tokenizer=tokenizer, boundary_selector=boundary_selector,
        trim_to_aligned_words=trim_to_aligned_words,
        split_sentences=split_sentences,
        semantic_cuts=semantic_cuts,
    )]
    segments.sort(key=lambda item: (item["start"], item["end"]))
    if not split_sentences and semantic_cuts is None:
        segments = _merge_short_segments(segments, diarized=diarized)
    segments.sort(key=lambda item: (item["start"], item["end"]))
    return [dict(segment, id=index) for index, segment in enumerate(segments)]
