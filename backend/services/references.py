"""Store manual references and cut automatic clips from reviewed source ranges."""

from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
import unicodedata
import uuid
import wave
from pathlib import Path
from typing import BinaryIO

from backend import config
from backend.schemas import AlignedWord, ProjectScript, ReferencePreview, Segment
from backend.services.text_normalization import normalize_tts_lang, prepare_tts_text

MIN_REFERENCE_SECONDS = 3
MAX_REFERENCE_SECONDS = 10
MAX_REFERENCE_BYTES = 32 * 1024 * 1024
SUPPORTED_SOURCE_LANGS = {"zh", "en", "ja", "ko", "yue"}


def _copy_bounded(source: BinaryIO, target: BinaryIO, max_bytes: int) -> None:
    total = 0
    while chunk := source.read(65536):
        total += len(chunk)
        if total > max_bytes:
            raise ValueError("参考音文件不能超过 32 MB")
        target.write(chunk)


def store_reference_audio(project_id: str, source: BinaryIO) -> str:
    dest = config.WORKSPACE_DIR / project_id / "references"
    dest.mkdir(parents=True, exist_ok=True)
    name = f"{uuid.uuid4().hex}.wav"
    with tempfile.TemporaryDirectory(dir=dest) as temp:
        original, converted = Path(temp) / "input", Path(temp) / name
        with original.open("wb") as target:
            _copy_bounded(source, target, MAX_REFERENCE_BYTES)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
                    "-format_whitelist", "wav,mp3,flac,ogg,mov,aac,matroska,webm",
                    "-i", str(original), "-map", "0:a:0", "-t", "10.1", "-ac", "1",
                    "-ar", "32000", "-c:a", "pcm_s16le", str(converted),
                ],
                check=True, capture_output=True, timeout=30,
            )
        except FileNotFoundError as exc:
            raise ValueError("上传参考音需要安装 FFmpeg") from exc
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise ValueError("无法读取参考音，请上传可播放的音频文件") from exc
        with wave.open(str(converted), "rb") as audio:
            duration = audio.getnframes() / audio.getframerate()
        if not MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
            raise ValueError("GPT-SoVITS 参考音长度须在 3–10 秒之间")
        converted.replace(dest / name)
    return f"/workspace/{project_id}/references/{name}"


def _candidate_clips(script: ProjectScript, speaker: str) -> list[list[Segment]]:
    ordered = sorted(script.segments, key=lambda seg: (seg.start, seg.id))
    candidates = []
    for index, seg in enumerate(ordered):
        if seg.speaker != speaker or seg.alignment_needs_review or seg.speaker_needs_review or not seg.orig_text.strip():
            continue
        clip = []
        for following in range(index, len(ordered)):
            other = ordered[following]
            if (other.speaker != speaker or other.alignment_needs_review or other.speaker_needs_review
                    or not other.orig_text.strip() or other.end - seg.start > MAX_REFERENCE_SECONDS
                    or (clip and not 0 <= round(other.start - clip[-1].end, 3) <= 0.35)):
                break
            clip.append(other)
            duration = other.end - seg.start
            if MIN_REFERENCE_SECONDS <= duration <= MAX_REFERENCE_SECONDS:
                if not any(
                    item.start < other.end and item.end > seg.start
                    and item.speaker != speaker
                    for item in ordered
                ):
                    transcript = " ".join(part.orig_text.strip() for part in clip)
                    if len(re.sub(r"\s+", "", transcript)) <= duration * 150:
                        candidates.append(clip)
                    break
    return sorted(candidates, key=lambda clip: (
        len(clip),
        abs(clip[-1].end - clip[0].start - 5),
        len("".join(part.orig_text for part in clip)),
    ))


def _word_char_spans(text: str, words: list[AlignedWord]) -> list[tuple[int, int]] | None:
    """将 Segment.words 逐词映射回 orig_text 的字符切片范围，失败说明词级对齐缺失或已过期。"""
    if not words or not text:
        return None
    letters = [(offset, char) for offset, char in enumerate(text) if not char.isspace()]
    compact = "".join(char for _, char in letters)
    cursor = 0
    spans: list[tuple[int, int]] = []
    for word in words:
        token = "".join(word.word.split())
        if not token or not compact.startswith(token, cursor):
            return None
        start_char = letters[cursor][0]
        cursor += len(token)
        end_char = letters[cursor - 1][0] + 1
        spans.append((start_char, end_char))
    return spans if cursor == len(compact) else None


def _trimmed_transcript(seg: Segment, start: float, end: float) -> str:
    """Prefer the exact words in the trim window, but fall back to reviewed full text."""
    spans = _word_char_spans(seg.orig_text, seg.words)
    if spans is None:
        return seg.orig_text.strip()
    indices = [
        idx
        for idx, word in enumerate(seg.words)
        if word.word.strip()
        and word.start is not None
        and word.end is not None
        and word.start >= start - 0.001
        and word.end <= end + 0.001
    ]
    if not indices:
        return seg.orig_text.strip()
    return seg.orig_text[spans[indices[0]][0] : spans[indices[-1]][1]].strip() or seg.orig_text.strip()


def _aligned_fallback_window(
    seg: Segment, ordered: list[Segment], speaker: str,
) -> tuple[float, float, str] | None:
    """当长句包含有效词级时间戳时，按词边界截取 3–10 秒（优先 ≤5s 且接近 5s）的无重叠片段。"""
    if _word_char_spans(seg.orig_text, seg.words) is None:
        return None
    words = seg.words
    candidates: list[tuple[tuple[bool, bool, float], float, float]] = []
    for left in range(len(words)):
        w_start = words[left].start
        if w_start is None or words[left].end is None:
            continue
        for right in range(left, len(words)):
            w_end = words[right].end
            if words[right].start is None or w_end is None or w_end <= w_start:
                break
            dur = round(w_end - w_start, 3)
            if dur > MAX_REFERENCE_SECONDS:
                break
            if dur < MIN_REFERENCE_SECONDS:
                continue
            overlaps_other = any(
                item.start < w_end and item.end > w_start and item.speaker != speaker
                for item in ordered
            )
            if not overlaps_other:
                candidates.append(((left > 0, dur > 5.0, abs(dur - 5.0)), w_start, w_end))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    _, best_start, best_end = candidates[0]
    transcript = _trimmed_transcript(seg, best_start, best_end)
    if not transcript:
        return None
    return best_start, best_end, transcript


def automatic_reference(script: ProjectScript, speaker: str) -> ReferencePreview | None:
    lang = normalize_tts_lang(script.source_lang, default="")
    if lang not in SUPPORTED_SOURCE_LANGS or not script.media.vocals_wav:
        return None
    candidates = _candidate_clips(script, speaker)
    windows: list[tuple[float, float, str, list[int]]] = []
    for clip in candidates:
        windows.append((
            clip[0].start,
            clip[-1].end,
            " ".join(part.orig_text.strip() for part in clip),
            [part.id for part in clip],
        ))
    if not windows:
        ordered = sorted(script.segments, key=lambda item: (item.start, item.id))
        for seg in ordered:
            if seg.speaker != speaker or seg.speaker_needs_review or not seg.orig_text.strip():
                continue
            aligned_window = _aligned_fallback_window(seg, ordered, speaker)
            if aligned_window is not None:
                w_start, w_end, transcript = aligned_window
                windows.append((w_start, w_end, transcript, [seg.id]))
                break
            duration = min(5.0, max(0.0, seg.end - seg.start))
            if duration >= MIN_REFERENCE_SECONDS:
                end = round(seg.start + duration, 3)
                windows.append((seg.start, end, _trimmed_transcript(seg, seg.start, end), [seg.id]))
                break
    if not windows:
        return None

    try:
        source = config.resolve_media_path(script.media.vocals_wav)
        project_dir = (config.WORKSPACE_DIR / script.project_id).resolve()
        is_banquet_dialogue = (
            script.media.cinematic_stems is not None
            and script.media.vocals_wav == script.media.cinematic_stems.dialogue
            and source.name == "dialogue.wav"
            and source.parent.name.startswith("cinematic-")
            and source.parent.parent == project_dir
        )
        if source != project_dir / "vocals.wav" and not is_banquet_dialogue:
            return None
        with wave.open(str(source), "rb") as vocals:
            rate, channels, width = vocals.getframerate(), vocals.getnchannels(), vocals.getsampwidth()
            for window_start, window_end, transcript, segment_ids in windows:
                start = max(0, round(window_start * rate))
                # Segment timestamps can extend a few ms past the decoded audio length.
                end = min(round(window_end * rate), vocals.getnframes())
                if end - start < MIN_REFERENCE_SECONDS * rate:
                    continue
                key = (str(source), source.stat().st_mtime_ns, start, end, channels, width)
                name = f"auto_{hashlib.sha256(repr(key).encode()).hexdigest()[:24]}.wav"
                path = project_dir / "references" / name
                if not path.is_file():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
                        draft = Path(temp) / name
                        vocals.setpos(start)
                        frames = vocals.readframes(end - start)
                        if channels == 1 and width == 2 and rate == 32000:
                            with wave.open(str(draft), "wb") as output:
                                output.setnchannels(1)
                                output.setsampwidth(2)
                                output.setframerate(32000)
                                output.writeframes(frames)
                        else:
                            raw = Path(temp) / "raw.wav"
                            with wave.open(str(raw), "wb") as source_wav:
                                source_wav.setnchannels(channels)
                                source_wav.setsampwidth(width)
                                source_wav.setframerate(rate)
                                source_wav.writeframes(frames)
                            subprocess.run(
                                ["ffmpeg", "-nostdin", "-v", "error", "-i", str(raw), "-ac", "1", "-ar", "32000",
                                 "-c:a", "pcm_s16le", str(draft)],
                                check=True, capture_output=True, timeout=30,
                            )
                        draft.replace(path)
                text = prepare_tts_text(unicodedata.normalize("NFKC", transcript), lang)
                if not text:
                    continue
                return ReferencePreview(
                    wav=f"/workspace/{script.project_id}/references/{name}",
                    text=text,
                    lang=lang,
                    segment_ids=segment_ids,
                )
    except (ValueError, OSError, EOFError, wave.Error, subprocess.SubprocessError):
        return None
    return None
