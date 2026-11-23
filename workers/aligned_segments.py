"""Split diarized ASR segments without guessing word times or dropping text."""

from __future__ import annotations

import math
import re
from typing import Any

MAX_SECONDS = 14.0
PREFERRED_SECONDS = 8.0
MIN_FRAGMENT_SECONDS = 2.5
MERGE_GAP_SECONDS = 0.65
PAUSE_SECONDS = 0.65
PUNCTUATION = re.compile(r"""[.!?;,:。！？；，、：…]["'”’）)\]]*$""")
TERMINAL_PUNCTUATION = re.compile(r"""[.!?。！？…]["'”’）)\]]*$""")
LATIN_CHAR = re.compile(r"[A-Za-z0-9]")
DEFAULT_SPEAKER = "SPEAKER_00"


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
        token = "".join(word["word"].split())
        if not token or not compact.startswith(token, cursor):
            return None
        offsets.append(letters[cursor][0])
        cursor += len(token)
    return offsets if cursor == len(compact) else None


def _split_segment(raw: dict, diarized: bool) -> list[dict]:
    text = str(raw.get("text") or "").strip()
    if not text:
        return []
    raw_start, raw_end = _time(raw.get("start")), _time(raw.get("end"))
    start = raw_start if raw_start is not None else 0.0
    end = raw_end if raw_end is not None and raw_end > start else round(start + 0.25, 3)
    rank_speaker = (str(raw.get("speaker") or "").strip() or None) if diarized else None
    words = []
    for item in raw.get("words") or []:
        word_start, word_end = _time(item.get("start")), _time(item.get("end"))
        if word_start is None or word_end is None or not start <= word_start < word_end <= end:
            word_start = word_end = None
        speaker = (str(item.get("speaker") or "").strip() or None) if diarized else None
        words.append(dict(word=str(item.get("word") or ""), start=word_start, end=word_end, speaker=speaker))

    offsets = _word_offsets(text, words)
    # A valid cut needs aligned words on both sides and must preserve their audio.
    cuts = [
        i for i in range(1, len(words))
        if words[i - 1]["end"] is not None and words[i]["start"] is not None
        and words[i - 1]["end"] <= words[i]["start"]
    ] if offsets else []

    def speaker_at(index: int) -> str | None:
        return words[index]["speaker"] or rank_speaker

    def part(left: int, right: int) -> dict:
        part_start = words[left]["start"] if left else start
        part_end = words[right - 1]["end"] if right < len(words) else end
        left_offset = offsets[left] if left and offsets else 0
        right_offset = offsets[right] if right < len(words) and offsets else len(text)
        speakers = {speaker_at(i) for i in range(left, right)}
        if not speakers:
            speakers = {rank_speaker}
        speaker = next((speaker_at(i) for i in range(left, right) if speaker_at(i)), rank_speaker)
        part_words = words[left:right]
        ordered_starts = [word["start"] for word in part_words if word["start"] is not None]
        return dict(
            start=part_start,
            end=part_end,
            speaker=speaker or DEFAULT_SPEAKER,
            orig_text=text[left_offset:right_offset].strip(),
            words=part_words,
            alignment_needs_review=(
                not offsets or raw_start is None or raw_end != end
                or len(ordered_starts) != len(part_words) or ordered_starts != sorted(ordered_starts)
                or part_end - part_start > MAX_SECONDS
            ),
            speaker_needs_review=(
                diarized and (
                    len(speakers) != 1 or None in speakers
                    or any(word["start"] is not None and not word["speaker"] for word in part_words)
                )
            ),
        )

    forced = [i for i in cuts if speaker_at(i - 1) and speaker_at(i) and speaker_at(i - 1) != speaker_at(i)]
    segments = []
    left = 0
    for right in [*forced, len(words)]:
        part_start = words[left]["start"] if left else start
        part_end = words[right - 1]["end"] if right < len(words) else end
        while part_end - part_start > MAX_SECONDS:
            available = [i for i in cuts if left < i < right]
            if not available:
                break
            fitting = [i for i in available if words[i - 1]["end"] - part_start <= MAX_SECONDS]
            natural = [
                i for i in fitting
                if words[i - 1]["end"] - part_start >= PREFERRED_SECONDS
                and (PUNCTUATION.search(text[offsets[i - 1]:offsets[i]].rstrip())
                     or words[i]["start"] - words[i - 1]["end"] >= PAUSE_SECONDS)
            ]
            cut = natural[-1] if natural else (fitting[-1] if fitting else available[0])
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
    if bool(LATIN_CHAR.search(left[-1])) and bool(LATIN_CHAR.search(right[0])):
        return f"{left} {right}"
    return f"{left}{right}"


def _review_flags(seg: dict, diarized: bool) -> tuple[bool, bool]:
    words = seg.get("words") or []
    starts = [word["start"] for word in words if word["start"] is not None]
    alignment = (
        not words or len(starts) != len(words) or starts != sorted(starts)
        or seg["end"] - seg["start"] > MAX_SECONDS
    )
    speakers = {word.get("speaker") for word in words if word.get("start") is not None}
    speaker_review = diarized and (len(speakers) != 1 or None in speakers)
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
        review_ok = not diarized or (
            not prev["alignment_needs_review"] and not prev["speaker_needs_review"]
            and not seg["alignment_needs_review"] and not seg["speaker_needs_review"]
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
        combined["alignment_needs_review"], combined["speaker_needs_review"] = _review_flags(combined, diarized)
        merged[-1] = combined
    return merged


def normalize_segments(raw_segments: list[dict], *, diarized: bool = False) -> list[dict]:
    segments = [part for raw in raw_segments for part in _split_segment(raw, diarized)]
    segments.sort(key=lambda item: (item["start"], item["end"]))
    segments = _merge_short_segments(segments, diarized=diarized)
    segments.sort(key=lambda item: (item["start"], item["end"]))
    return [dict(segment, id=index) for index, segment in enumerate(segments)]
