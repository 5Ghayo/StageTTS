"""Store manual references and cut automatic clips from reviewed source ranges."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import uuid
import wave
from pathlib import Path
from typing import BinaryIO

from backend import config
from backend.schemas import ProjectScript, ReferencePreview, Segment

MIN_REFERENCE_SECONDS = 3
MAX_REFERENCE_SECONDS = 10
MAX_REFERENCE_BYTES = 32 * 1024 * 1024


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
        if seg.speaker != speaker:
            continue
        if seg.alignment_needs_review or seg.speaker_needs_review or not seg.orig_text.strip():
            continue
        clip = []
        for following in range(index, len(ordered)):
            other = ordered[following]
            if (other.speaker != speaker or other.alignment_needs_review or other.speaker_needs_review
                    or not other.orig_text.strip() or other.end - seg.start > MAX_REFERENCE_SECONDS
                    or (clip and not 0 <= round(other.start - clip[-1].end, 3) <= 0.35)):
                break
            clip.append(other)
            if other.end - seg.start >= MIN_REFERENCE_SECONDS:
                # Reject clips that overlap another speaker or an unconfirmed turn.
                if not any(
                    item.start < other.end and item.end > seg.start
                    and (item.speaker != speaker or item.speaker_needs_review)
                    for item in ordered
                ):
                    candidates.append(clip)
                break
    # Prefer an uninterrupted sentence, then a clip close to five seconds.
    return sorted(candidates, key=lambda clip: (len(clip), abs(clip[-1].end - clip[0].start - 5)))


def automatic_reference(script: ProjectScript, speaker: str) -> ReferencePreview | None:
    lang = (script.source_lang or "").strip().lower().replace("_", "-").split("-")[0]
    if lang not in {"zh", "en", "ja", "ko", "yue"} or not script.media.vocals_wav:
        return None
    candidates = _candidate_clips(script, speaker)
    if not candidates:
        return None
    try:
        source = config.resolve_media_path(script.media.vocals_wav)
        project_dir = (config.WORKSPACE_DIR / script.project_id).resolve()
        if source != project_dir / "vocals.wav":
            return None
        with wave.open(str(source), "rb") as vocals:
            rate = vocals.getframerate()
            for clip in candidates:
                start = round(clip[0].start * rate)
                end = round(clip[-1].end * rate)
                if end > vocals.getnframes() or not MIN_REFERENCE_SECONDS * rate <= end - start <= MAX_REFERENCE_SECONDS * rate:
                    continue
                key = (str(source), source.stat().st_mtime_ns, start, end)
                name = f"auto_{hashlib.sha256(repr(key).encode()).hexdigest()[:24]}.wav"
                path = project_dir / "references" / name
                if not path.is_file():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
                        draft = Path(temp) / name
                        vocals.setpos(start)
                        with wave.open(str(draft), "wb") as output:
                            output.setparams(vocals.getparams())
                            output.writeframes(vocals.readframes(end - start))
                        draft.replace(path)
                separator = "" if lang in {"zh", "ja", "yue"} else " "
                return ReferencePreview(
                    wav=f"/workspace/{script.project_id}/references/{name}",
                    text=separator.join(seg.orig_text.strip() for seg in clip),
                    lang=lang,
                    segment_ids=[seg.id for seg in clip],
                )
    except (ValueError, OSError, EOFError, wave.Error):
        return None
    return None
