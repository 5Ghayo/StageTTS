"""Audio-anchored dialogue metadata and optional text-subtitle references."""

from __future__ import annotations

import html
import json
import re
import subprocess
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from backend.schemas import Segment, SpeechCue
from backend.services.speech_loudness import active_rms_dbfs, training_loudness_reason

SUBTITLE_MAX_BYTES = 2 * 1024 * 1024
DUB_TAIL_ALLOWANCE_SECONDS = 1.0
_TIME = r"(?:\d{1,3}:)?\d{2}:\d{2}[,.]\d{3}"
_INTERVAL = re.compile(rf"^\s*({_TIME})\s*-->\s*({_TIME})(?:\s.*)?$")
_TAG = re.compile(r"<[^>]+>|\{\\[^}]*\}")


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str


def _seconds(stamp: str) -> float:
    value = 0.0
    for field in stamp.replace(",", ".").split(":"):
        value = value * 60 + float(field)
    return value


def parse_subtitle_cues(raw: bytes) -> list[SubtitleCue]:
    if len(raw) > SUBTITLE_MAX_BYTES:
        raise ValueError("字幕文件不能超过 2 MB")
    source = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
    subtitles = []
    for block in re.split(r"\n[ \t]*\n", source.replace("\r\n", "\n").replace("\r", "\n")):
        lines = block.strip().splitlines()
        for index, line in enumerate(lines[:2]):
            match = _INTERVAL.fullmatch(line)
            if not match:
                continue
            start, end = map(_seconds, match.groups())
            text = " ".join(html.unescape(_TAG.sub("", "\n".join(lines[index + 1:]))).split())
            if start < end and text:
                subtitles.append(SubtitleCue(start, end, text))
            break
    return sorted(subtitles, key=lambda item: (item.start, item.end))


def subtitle_path(project_dir: Path, language: str) -> Path:
    return project_dir / f"subtitle-reference-{language}.txt"


def _normalized(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", text).casefold() if ch.isalnum())


def apply_subtitle_references(segments: list[Segment], language: str, subtitles: list[SubtitleCue]) -> None:
    """Match onto existing audio rows by overlap, with Japanese text as extra evidence."""
    ordered = sorted((row for row in segments if row.speech_cue), key=lambda row: row.dub_start)
    first = 0
    for row in ordered:
        anchor = row.speech_cue
        start, end = anchor.speech_start, anchor.speech_end
        while first < len(subtitles) and subtitles[first].end < start - 0.35:
            first += 1
        matches = []
        heard = _normalized(row.orig_text)
        for index in range(first, len(subtitles)):
            item = subtitles[index]
            if item.start > end + 0.35:
                break
            overlap = min(end, item.end) - max(start, item.start)
            coverage = overlap / min(end - start, item.end - item.start)
            if coverage >= 0.5 and overlap >= min(0.15, 0.5 * (end - start)):
                matches.append(item.text)
            elif language == "ja" and overlap > -0.35 and heard:
                reference = _normalized(item.text)
                if reference and SequenceMatcher(None, heard, reference).ratio() >= 0.72:
                    matches.append(item.text)
        setattr(anchor, f"reference_{language}", " ".join(dict.fromkeys(matches)) or None)


def load_subtitle_references(segments: list[Segment], project_dir: Path, media: Path | None = None) -> None:
    explicit = {lang: subtitle_path(project_dir, lang) for lang in ("ja", "zh")}
    for lang, path in explicit.items():
        if path.is_file():
            apply_subtitle_references(segments, lang, parse_subtitle_cues(path.read_bytes()))
    missing = {lang for lang, path in explicit.items() if not path.is_file()}
    if not missing or media is None or not segments:
        return
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "s", "-show_streams", "-of", "json", str(media)],
            check=True, capture_output=True, timeout=30,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        for stream in streams:
            tag = stream.get("tags", {}).get("language", "").lower().replace("_", "-").split("-")[0]
            lang = {"jpn": "ja", "chi": "zh", "zho": "zh", "cmn": "zh"}.get(tag, tag)
            if lang not in missing or stream.get("codec_name") not in {"ass", "ssa", "mov_text", "subrip", "webvtt", "text"}:
                continue
            result = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", str(media), "-map", f"0:{int(stream['index'])}",
                 "-f", "srt", "pipe:1"], capture_output=True, timeout=45,
            )
            if result.returncode == 0:
                cues = parse_subtitle_cues(result.stdout)
                if cues:
                    apply_subtitle_references(segments, lang, cues)
                    missing.remove(lang)
    except (OSError, subprocess.SubprocessError, ValueError):
        # Optional embedded references must never block audio parsing.
        return


def _active_rms_dbfs(audio, start: float, end: float, rate: int) -> float | None:
    import numpy as np

    samples = audio[round(start * rate):round(end * rate)]
    if not samples.size:
        return None
    hop = round(0.02 * rate)
    complete = samples.size // hop * hop
    frames = samples[:complete].reshape(-1, hop)
    energies = np.mean(np.square(frames, dtype=np.float64), axis=1).tolist()
    if complete < samples.size:
        energies.append(float(np.mean(np.square(samples[complete:], dtype=np.float64))))
    return active_rms_dbfs(energies)


def attach_speech_cues(segments: list[Segment], raw_segments: list[dict], audio, duration: float) -> None:
    """Anchor all usable lines to their acoustic onset, leaving bounded pause slack."""
    raw_by_id = {row["id"]: row for row in raw_segments}
    ordered = sorted(segments, key=lambda row: (row.source_audio_start if row.source_audio_start is not None else row.start,
                                                row.id))
    starts = [row.source_audio_start if row.source_audio_start is not None else row.start for row in ordered]
    for i, (row, start) in enumerate(zip(ordered, starts, strict=True)):
        end = min(duration, raw_by_id[row.id].get("speech_end", row.end))
        if end <= start:
            start, end = row.start, min(duration, row.end)
        if end <= start:
            continue
        next_start = starts[i + 1] if i + 1 < len(starts) else duration
        latest = min(duration, end + DUB_TAIL_ALLOWANCE_SECONDS, max(end, next_start))
        rms = _active_rms_dbfs(audio, start, end, 16000) if getattr(audio, "ndim", None) == 1 else None
        row.speech_cue = SpeechCue(speech_start=start, speech_end=end, latest_end=latest, rms_dbfs=rms)


def exclude_quiet_training_clips(segments: list[Segment]) -> None:
    for row in segments:
        reason = training_loudness_reason(row.speech_cue.rms_dbfs if row.speech_cue else None)
        if reason:
            row.training_excluded = True
            if reason not in row.training_exclusion_reasons:
                row.training_exclusion_reasons.append(reason)
