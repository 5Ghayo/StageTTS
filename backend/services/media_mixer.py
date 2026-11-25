"""音视频后期合成服务：原声保留、pydub 毫秒拼轨与 FFmpeg 混音封装。"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from pydub import AudioSegment

from backend import config
from backend.schemas import ProjectScript
from backend.services.task_runner import probe_media_duration


def check_ffmpeg_available() -> bool:
    """检查系统 PATH 中是否可用 ffmpeg。"""
    return shutil.which("ffmpeg") is not None


def apply_gentle_tempo(wav_path: Path, speed_factor: float, tmp_dir: Path) -> Path:
    """调用 FFmpeg atempo 滤镜对单句做不变调温和变速（限 0.5 ~ 2.0）。"""
    factor = max(0.5, min(2.0, float(speed_factor)))
    if round(factor, 3) == 1.0:
        return wav_path

    out_path = tmp_dir / f"{wav_path.stem}_spd.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(wav_path),
        "-filter:a",
        f"atempo={factor:.3f}",
        "-vn",
        str(out_path),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg atempo 处理失败: {proc.stderr[-400:]}")
    return out_path


def _find_watermark_font() -> Optional[str]:
    """定位系统中文字体文件并转义为 FFmpeg drawtext fontfile 参数格式。"""
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for font_path in candidates:
        if font_path.is_file():
            # FFmpeg filtergraph 中 Windows 盘符冒号需转义为 \:
            posix = font_path.as_posix()
            return posix.replace(":", "\\:")
    return None


def _build_vocal_track(
    script: ProjectScript, project_dir: Path, *, timeline_duration: float | None = None,
) -> Path:
    """Mix complete clips, blocking any overflow that cannot use empty timeline space."""
    timeline_duration = script.media.duration if timeline_duration is None else timeline_duration
    if timeline_duration <= 0:
        raise RuntimeError("导出待审核：项目时长无效，无法安全安排配音。")
    total_ms = round(timeline_duration * 1000)
    for seg in script.segments:
        if seg.end > timeline_duration + 0.000001:
            raise RuntimeError(f"片段 #{seg.id} 导出待审核：原始时间范围超过媒体结尾。")

    vocals_audio: Optional[AudioSegment] = None
    if script.media.vocals_wav:
        try:
            vocals_path = config.resolve_media_path(script.media.vocals_wav)
            if vocals_path.is_file():
                with vocals_path.open("rb") as source:
                    vocals_audio = AudioSegment.from_file(source)
        except ValueError:
            vocals_audio = None

    frame_rate = vocals_audio.frame_rate if vocals_audio is not None else 44100
    vocal_track = AudioSegment.silent(duration=total_ms, frame_rate=frame_rate).set_channels(2)

    with tempfile.TemporaryDirectory(prefix="stagetts_mix_") as tmp_name:
        tmp_dir = Path(tmp_name)
        for seg in sorted(script.segments, key=lambda item: item.dub_start):
            pos_ms = max(0, round(seg.dub_start * 1000))
            if seg.keep_original:
                if vocals_audio is None:
                    continue
                speech_end = seg.speech_cue.speech_end if seg.speech_cue else seg.end
                end_ms = min(len(vocals_audio), max(pos_ms, int(speech_end * 1000)))
                if end_ms <= pos_ms:
                    continue
                clip = vocals_audio[pos_ms:end_ms]
                vocal_track = vocal_track.overlay(clip, position=pos_ms)
                continue

            if not seg.tts_wav or seg.status != "ready":
                continue
            try:
                tts_path = config.resolve_media_path(seg.tts_wav)
            except ValueError:
                continue
            if not tts_path.is_file():
                continue

            with tts_path.open("rb") as source:
                raw_clip = AudioSegment.from_wav(source)
            clip = raw_clip
            if seg.applied_speed_factor is None:
                # 旧项目的 WAV 仍沿用导出时变速。
                speed = seg.speed_factor
                raw_sec = len(raw_clip) / 1000.0
                slot_sec = max(0.05, seg.dub_end - seg.dub_start)
                if abs(speed - 1.0) < 0.01 and slot_sec * 1.02 < raw_sec <= slot_sec * config.MAX_GENTLE_SPEED:
                    speed = min(config.MAX_GENTLE_SPEED, round(raw_sec / slot_sec, 3))
                if abs(speed - 1.0) >= 0.02:
                    adjusted_path = apply_gentle_tempo(tts_path, speed, tmp_dir)
                    with adjusted_path.open("rb") as source:
                        clip = AudioSegment.from_wav(source)

            clip_end = seg.dub_start + clip.frame_count() / clip.frame_rate
            if clip_end > timeline_duration + 0.000001:
                raise RuntimeError(
                    f"片段 #{seg.id} 导出待审核：完整配音超过媒体结尾 "
                    f"{clip_end - timeline_duration:.3f} 秒，不能截尾。"
                )
            if seg.speech_cue and clip_end > seg.dub_end + 0.001:
                raise RuntimeError(f"片段 #{seg.id} 超出配音允许时间窗，请调整后导出。")
            if not seg.speech_cue and clip_end > seg.end:
                collision = next((
                    other for other in script.segments
                    if other.id != seg.id and other.dub_start < clip_end and other.dub_end > seg.end
                ), None)
                if collision is not None:
                    raise RuntimeError(
                        f"片段 #{seg.id} 导出待审核：完整配音侵入片段 #{collision.id}，"
                        "请调整配音或审核排程后导出。"
                    )
            placed_frames = round(pos_ms * vocal_track.frame_rate / 1000)
            clip_frames = clip.frame_count() * vocal_track.frame_rate / clip.frame_rate
            if placed_frames + clip_frames > vocal_track.frame_count():
                raise RuntimeError(f"片段 #{seg.id} 导出待审核：配音超出混音画布，不能截尾。")
            vocal_track = vocal_track.overlay(clip, position=pos_ms)

    dubbed_path = project_dir / "dubbed_vocals.wav"
    with dubbed_path.open("wb") as output:
        vocal_track.export(output, format="wav")
    return dubbed_path


def _amix_normalize_option() -> str:
    """Return the amix no-renormalization option supported by the local FFmpeg."""
    try:
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", "filter=amix"],
            capture_output=True, text=True, errors="replace", timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return "normalize=0"
    text = probe.stdout + probe.stderr
    if "normalize" in text:
        return "normalize=0"
    # Older amix builds always normalize by active inputs; compensate after mixing.
    return "weights=1 1,volume=2"


def _run_ffmpeg_cmd(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


def _build_drawtext_filter() -> str:
    font_spec = _find_watermark_font()
    if font_spec:
        text_expr = "AI 合成语音 · StageTTS"
        font_arg = f"fontfile='{font_spec}':"
    else:
        text_expr = "AI Synthesized Voice - StageTTS"
        font_arg = ""

    return (
        f"drawtext={font_arg}text='{text_expr}':"
        "x=w-tw-24:y=h-th-24:fontsize=22:fontcolor=white@0.88:"
        "box=1:boxcolor=black@0.48:boxborderw=8"
    )


def mix_and_render(
    script: ProjectScript,
    burn_watermark: bool = False,
) -> Tuple[str, float]:
    """执行 pydub 毫秒拼轨与 FFmpeg 混音封装导出。

    严格遵守 ARCHITECTURE.md 铁律 #7：
    - 混音使用 `amix=inputs=2:duration=first:normalize=0` 避免音量衰减。
    - `burn_watermark=False` 时使用 `-c:v copy`（严禁混用 `-vf`）。
    - `burn_watermark=True` 时使用 `-vf drawtext=...` 烧录合规水印，优先 `h264_nvenc` 并支持 `libx264` 兜底。
    """
    if not check_ffmpeg_available():
        raise RuntimeError("未在系统 PATH 中检测到 ffmpeg，请确认已安装 FFmpeg 6.x+。")

    project_dir: Path = config.WORKSPACE_DIR / script.project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    source_video_path: Optional[Path] = None
    timeline_duration = script.media.duration
    if script.media.source_video:
        candidate_video = config.resolve_media_path(script.media.source_video)
        if candidate_video.is_file():
            source_video_path = candidate_video
            source_duration = probe_media_duration(candidate_video)
            if source_duration <= 0:
                raise RuntimeError("导出待审核：无法确认源视频时长，不能安全安排配音。")
            timeline_duration = min(timeline_duration, source_duration)
    dubbed_vocals_path = _build_vocal_track(
        script, project_dir, timeline_duration=timeline_duration,
    )

    bgm_path: Optional[Path] = None
    if script.media.bgm_wav:
        try:
            candidate_bgm = config.resolve_media_path(script.media.bgm_wav)
            if candidate_bgm.is_file():
                bgm_path = candidate_bgm
        except ValueError:
            bgm_path = None
    if script.media.cinematic_stems is not None and bgm_path is None:
        raise FileNotFoundError("影视四轨的混合背景轨缺失，请重新分离后导出。")

    bgm_vol = max(0.0, min(2.0, float(script.media.bgm_volume)))
    amix_gain = _amix_normalize_option()

    # 模式 B 虚拟舞台若无源视频文件，导出完整混音音频 output.wav
    if script.mode == "virtual_stage" and source_video_path is None:
        output_wav = project_dir / "output.wav"
        if bgm_path is not None:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(dubbed_vocals_path),
                "-i",
                str(bgm_path),
                "-filter_complex",
                f"[1:a]volume={bgm_vol:.3f}[bgm];[0:a][bgm]amix=inputs=2:duration=first:{amix_gain}[aout]",
                "-map",
                "[aout]",
                str(output_wav),
            ]
            proc = _run_ffmpeg_cmd(cmd)
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg 音频混音失败: {proc.stderr[-600:]}")
        else:
            shutil.copyfile(dubbed_vocals_path, output_wav)

        out_duration = probe_media_duration(output_wav) or script.media.duration
        return f"/workspace/{script.project_id}/output.wav", out_duration

    if source_video_path is None:
        raise FileNotFoundError(
            f"未找到源视频文件 ({script.media.source_video})，无法封装 output.mp4"
        )

    output_mp4 = project_dir / "output.mp4"
    base_inputs: List[str] = ["ffmpeg", "-y", "-i", str(source_video_path), "-i", str(dubbed_vocals_path)]
    if bgm_path is not None:
        base_inputs.extend(["-i", str(bgm_path)])
        audio_filter = (
            f"[2:a]volume={bgm_vol:.3f}[bgm];"
            f"[1:a][bgm]amix=inputs=2:duration=first:{amix_gain}[aout]"
        )
        audio_map = ["-map", "0:v:0", "-map", "[aout]"]
    else:
        audio_filter = None
        audio_map = ["-map", "0:v:0", "-map", "1:a:0"]

    if not burn_watermark:
        cmd = list(base_inputs)
        if audio_filter:
            cmd.extend(["-filter_complex", audio_filter])
        cmd.extend(audio_map)
        cmd.extend(["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(output_mp4)])
        proc = _run_ffmpeg_cmd(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg 无损封装失败: {proc.stderr[-600:]}")
    else:
        drawtext = _build_drawtext_filter()
        for video_codec_args in (
            ["-c:v", "h264_nvenc", "-preset", "p4"],
            ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"],
        ):
            cmd = list(base_inputs)
            if audio_filter:
                cmd.extend(["-filter_complex", audio_filter])
            cmd.extend(audio_map)
            cmd.extend(["-vf", drawtext])
            cmd.extend(video_codec_args)
            cmd.extend(["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", str(output_mp4)])
            proc = _run_ffmpeg_cmd(cmd)
            if proc.returncode == 0:
                break
        else:
            raise RuntimeError(f"FFmpeg 硬水印导出失败: {proc.stderr[-600:]}")

    out_duration = probe_media_duration(output_mp4) or script.media.duration
    return f"/workspace/{script.project_id}/output.mp4", out_duration
