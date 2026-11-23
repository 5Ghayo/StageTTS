"""音视频后期合成服务：原声保留、响度对齐、四轨独立混音、自动闪避与 FFmpeg 封装。"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydub import AudioSegment

from backend import config
from backend.schemas import ProjectScript, Segment, StemVolumes, SubtitleBurnMode, SubtitleLayout
from backend.services.subtitles import write_ass_file
from backend.services.task_runner import probe_media_duration

_GAP_GUARD_MS = 50
_MAX_GAP_SPILLOVER_MS = 800
_FADE_OUT_MS = 30
_KEEP_ORIG_FADE_MS = 10


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
    if os.name == "nt":
        win_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        candidates = [
            win_fonts / "msyh.ttc",
            win_fonts / "simhei.ttf",
            win_fonts / "msyhbd.ttc",
            win_fonts / "arial.ttf",
        ]
    elif sys.platform == "darwin":
        candidates = [
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/System/Library/Fonts/STHeiti Light.ttc"),
            Path("/System/Library/Fonts/Hiragino Sans GB.ttc"),
            Path("/Library/Fonts/Arial Unicode.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        ]
    else:
        candidates = [
            Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        ]
    for font_path in candidates:
        if font_path.is_file():
            return _escape_ffmpeg_filter_path(font_path)
    return None


def _video_encoder_candidates() -> Tuple[List[str], ...]:
    """按平台返回优先硬件加速与 libx264 兜底视频编码参数。"""
    fallback = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"]
    if sys.platform == "darwin":
        return (["-c:v", "h264_videotoolbox", "-q:v", "65"], fallback)
    return (["-c:v", "h264_nvenc", "-preset", "p4"], fallback)


def _normalize_clip_loudness(
    clip: AudioSegment,
    target_dbfs: float = -18.0,
    max_gain_db: float = 12.0,
    min_gain_db: float = -12.0,
    peak_ceiling_dbfs: float = -1.0,
) -> AudioSegment:
    """将人声片段响度对齐至目标 dBFS，并限制最大增益与峰值防削波。"""
    if len(clip) == 0 or clip.rms == 0:
        return clip
    current_dbfs = clip.dBFS
    if current_dbfs < -55.0:
        return clip
    desired_gain = max(min_gain_db, min(max_gain_db, target_dbfs - current_dbfs))
    headroom = peak_ceiling_dbfs - clip.max_dBFS
    applied_gain = min(desired_gain, headroom)
    if abs(applied_gain) < 0.25:
        return clip
    return clip.apply_gain(applied_gain)


def _probe_wav_sample_rate(wav_path: Path, default: int = 44100) -> int:
    try:
        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate()
            if rate > 0:
                return rate
    except (OSError, wave.Error, EOFError):
        pass
    return default


def _build_vocal_track(script: ProjectScript, project_dir: Path) -> Path:
    """使用 pydub 按毫秒 overlay 拼接完整人声轨（含间隙延展、淡出与响度对齐）。"""
    max_seg_end = max((seg.end for seg in script.segments), default=0.0)
    total_ms = max(1000, int(max(script.media.duration, max_seg_end) * 1000))

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

    audible_items: List[Tuple[Segment, int, Optional[Path]]] = []
    for seg in sorted(script.segments, key=lambda item: (item.start, item.id)):
        pos_ms = max(0, int(seg.start * 1000))
        if pos_ms >= total_ms:
            continue
        if seg.keep_original:
            if vocals_audio is None:
                continue
            end_ms = min(len(vocals_audio), max(pos_ms, int(seg.end * 1000)))
            if end_ms > pos_ms:
                audible_items.append((seg, pos_ms, None))
            continue

        if not seg.tts_wav or seg.status != "ready":
            continue
        try:
            tts_path = config.resolve_media_path(seg.tts_wav)
        except ValueError:
            continue
        if tts_path.is_file():
            audible_items.append((seg, pos_ms, tts_path))

    with tempfile.TemporaryDirectory(prefix="stagetts_mix_") as tmp_name:
        tmp_dir = Path(tmp_name)
        for idx, (seg, pos_ms, tts_path) in enumerate(audible_items):
            if seg.keep_original:
                assert vocals_audio is not None
                end_ms = min(len(vocals_audio), max(pos_ms, int(seg.end * 1000)))
                clip = vocals_audio[pos_ms:end_ms]
                fade_ms = min(_KEEP_ORIG_FADE_MS, len(clip) // 4)
                if fade_ms >= 2:
                    clip = clip.fade_in(fade_ms).fade_out(fade_ms)
                clip = _normalize_clip_loudness(clip)
                vocal_track = vocal_track.overlay(clip, position=pos_ms)
                continue

            assert tts_path is not None
            with tts_path.open("rb") as source:
                raw_clip = AudioSegment.from_wav(source)
            clip = raw_clip

            if seg.applied_speed_factor is None:
                # 旧项目的 WAV 仍沿用导出时变速。
                speed = seg.speed_factor
                raw_sec = len(raw_clip) / 1000.0
                slot_sec = max(0.05, seg.end - seg.start)
                if abs(speed - 1.0) < 0.01 and slot_sec * 1.02 < raw_sec <= slot_sec * config.MAX_GENTLE_SPEED:
                    speed = min(config.MAX_GENTLE_SPEED, round(raw_sec / slot_sec, 3))
                if abs(speed - 1.0) >= 0.02:
                    adjusted_path = apply_gentle_tempo(tts_path, speed, tmp_dir)
                    with adjusted_path.open("rb") as source:
                        clip = AudioSegment.from_wav(source)

            slot_ms = max(1, int((seg.end - seg.start) * 1000))
            next_start_ms = audible_items[idx + 1][1] if idx + 1 < len(audible_items) else total_ms
            gap_limit_ms = (
                max(slot_ms, next_start_ms - pos_ms - _GAP_GUARD_MS)
                if idx + 1 < len(audible_items)
                else total_ms - pos_ms
            )
            max_allowed_ms = min(total_ms - pos_ms, gap_limit_ms, slot_ms + _MAX_GAP_SPILLOVER_MS)
            max_allowed_ms = max(min(slot_ms, total_ms - pos_ms), max_allowed_ms)

            if max_allowed_ms > 0 and len(clip) > max_allowed_ms:
                clip = clip[:max_allowed_ms]
                fade_ms = min(_FADE_OUT_MS, len(clip) // 4)
                if fade_ms >= 2:
                    clip = clip.fade_out(fade_ms)

            clip = _normalize_clip_loudness(clip)
            vocal_track = vocal_track.overlay(clip, position=pos_ms)

    dubbed_path = project_dir / "dubbed_vocals.wav"
    with dubbed_path.open("wb") as output:
        vocal_track.export(output, format="wav")
    return dubbed_path


@functools.lru_cache(maxsize=8)
def _ffmpeg_filter_help(filter_name: str) -> str:
    """查询本机 FFmpeg 滤镜帮助文本。"""
    try:
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-h", f"filter={filter_name}"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return probe.stdout + probe.stderr if probe.returncode == 0 else ""


def _ffmpeg_has_filter(filter_name: str) -> bool:
    return bool(_ffmpeg_filter_help(filter_name))


def _ffmpeg_filter_has_option(help_text: str, option: str) -> bool:
    for line in help_text.splitlines():
        name = line.strip().split("<", 1)[0].strip()
        if name == option:
            return True
    return False


def _amix_normalize_option(inputs: int = 2) -> str:
    """Return the amix no-renormalization option supported by the local FFmpeg."""
    text = _ffmpeg_filter_help("amix")
    if not text or "normalize" in text:
        return f"amix=inputs={inputs}:duration=first:normalize=0"
    if _ffmpeg_filter_has_option(text, "weights"):
        weights = " ".join(["1"] * inputs)
        return f"amix=inputs={inputs}:duration=first:weights={weights},volume={inputs}"
    return f"amix=inputs={inputs}:duration=first,volume={inputs}"


def _alimiter_filter() -> str:
    help_text = _ffmpeg_filter_help("alimiter")
    if not help_text:
        return "volume=0.98"
    options = ["limit=0.98"]
    if _ffmpeg_filter_has_option(help_text, "level"):
        options.append("level=0")
    if _ffmpeg_filter_has_option(help_text, "latency"):
        options.append("latency=1")
    return "alimiter=" + ":".join(options)


def _resolve_background_tracks(
    script: ProjectScript,
) -> Tuple[Optional[Tuple[Path, Path, Path]], Optional[Path]]:
    """优先解析独立三背景轨 (effects, music, singing)，否则回退单 bgm.wav。"""
    if script.media.cinematic_stems is not None:
        stem_urls = (
            script.media.cinematic_stems.effects,
            script.media.cinematic_stems.music,
            script.media.cinematic_stems.singing,
        )
        try:
            stem_paths = tuple(config.resolve_media_path(url) for url in stem_urls)
        except ValueError:
            stem_paths = ()
        if len(stem_paths) == 3 and all(path.is_file() for path in stem_paths):
            return (stem_paths[0], stem_paths[1], stem_paths[2]), None

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
    return None, bgm_path


def resolve_effective_auto_ducking(
    script: ProjectScript,
    auto_ducking: Optional[bool] = None,
) -> bool:
    """计算本次导出实际生效的自动闪避开关。"""
    requested = script.media.auto_ducking if auto_ducking is None else bool(auto_ducking)
    return requested and _ffmpeg_has_filter("sidechaincompress")


def _build_audio_mix_graph(
    vocal_idx: int,
    bg_inputs: Tuple[int, ...],
    sample_rate: int,
    script: ProjectScript,
    auto_ducking: Optional[bool] = None,
) -> Tuple[List[str], str, bool]:
    """构建支持四轨独立增益与 sidechain 自动闪避的 FFmpeg filter_complex。"""
    master = max(0.0, min(2.0, float(script.media.bgm_volume)))
    sv = script.media.stem_volumes or StemVolumes()
    eff_gain = min(2.0, max(0.0, float(sv.effects)) * master)
    mus_gain = min(2.0, max(0.0, float(sv.music)) * master)
    sng_gain = min(2.0, max(0.0, float(sv.singing)) * master)

    effective_duck = resolve_effective_auto_ducking(script, auto_ducking)
    amix2 = _amix_normalize_option(2)
    amix3 = _amix_normalize_option(3)
    limiter = _alimiter_filter()
    fmt = f"aformat=sample_rates={sample_rate}:channel_layouts=stereo"

    chains: List[str] = []
    if len(bg_inputs) == 3:
        eff_idx, mus_idx, sng_idx = bg_inputs
        if effective_duck:
            chains.extend([
                f"[{mus_idx}:a]volume={mus_gain:.3f},{fmt}[mus_f]",
                f"[{sng_idx}:a]volume={sng_gain:.3f},{fmt}[sng_f]",
                f"[mus_f][sng_f]{amix2}[ms_bed]",
                f"[{vocal_idx}:a]asplit=2[v_mix][v_sc]",
                f"[v_sc]{fmt}[v_sc_f]",
                "[ms_bed][v_sc_f]sidechaincompress=threshold=0.03:ratio=4:attack=25:release=300:makeup=1[ms_duck]",
                f"[{eff_idx}:a]volume={eff_gain:.3f},{fmt}[eff_f]",
                f"[ms_duck][eff_f]{amix2},{limiter}[bgm_ready]",
                f"[v_mix]{fmt}[v_mix_f]",
                f"[v_mix_f][bgm_ready]{amix2},{limiter}[aout]",
            ])
        else:
            chains.extend([
                f"[{eff_idx}:a]volume={eff_gain:.3f},{fmt}[eff_f]",
                f"[{mus_idx}:a]volume={mus_gain:.3f},{fmt}[mus_f]",
                f"[{sng_idx}:a]volume={sng_gain:.3f},{fmt}[sng_f]",
                f"[eff_f][mus_f][sng_f]{amix3},{limiter}[bgm_ready]",
                f"[{vocal_idx}:a]{fmt}[v_mix_f]",
                f"[v_mix_f][bgm_ready]{amix2},{limiter}[aout]",
            ])
        return chains, "[aout]", effective_duck

    (bgm_idx,) = bg_inputs
    if effective_duck:
        chains.extend([
            f"[{bgm_idx}:a]volume={master:.3f},{fmt}[bgm_f]",
            f"[{vocal_idx}:a]asplit=2[v_mix][v_sc]",
            f"[v_sc]{fmt}[v_sc_f]",
            "[bgm_f][v_sc_f]sidechaincompress=threshold=0.03:ratio=4:attack=25:release=300:makeup=1[bgm_duck]",
            f"[v_mix]{fmt}[v_mix_f]",
            f"[v_mix_f][bgm_duck]{amix2},{limiter}[aout]",
        ])
    else:
        chains.extend([
            f"[{bgm_idx}:a]volume={master:.3f},{fmt}[bgm_f]",
            f"[{vocal_idx}:a]{fmt}[v_mix_f]",
            f"[v_mix_f][bgm_f]{amix2},{limiter}[aout]",
        ])
    return chains, "[aout]", effective_duck


def _run_ffmpeg_cmd(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


def _escape_ffmpeg_filter_path(path: Path) -> str:
    """转义 FFmpeg filtergraph 文件路径参数（如 ass=filename='...'）。"""
    text = path.resolve().as_posix()
    for ch in ("\\", "'", ":", ",", "[", "]"):
        text = text.replace(ch, f"\\{ch}")
    return text


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


_WINDOWS_RESERVED_STEM_RE = re.compile(r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$", re.IGNORECASE)


def _sanitize_output_filename(raw_name: Optional[str], expected_ext: str) -> str:
    """校验并规范化导出文件名，防止路径穿越、非法字符或 Windows 保留设备名。"""
    cleaned = (raw_name or "").strip()
    if not cleaned:
        return f"output{expected_ext}"
    if "/" in cleaned or "\\" in cleaned or ".." in cleaned or any(ch in '<>:"|?*' for ch in cleaned) or cleaned in {".", ".."}:
        raise ValueError("输出文件名不能包含路径分隔符或非法字符")
    suffix = Path(cleaned).suffix.lower()
    if not suffix:
        cleaned = f"{cleaned}{expected_ext}"
    elif suffix != expected_ext:
        raise ValueError(f"当前模式导出文件后缀必须为 {expected_ext}")
    stem = Path(cleaned).stem
    if not stem or stem.endswith((" ", ".")) or _WINDOWS_RESERVED_STEM_RE.match(stem):
        raise ValueError("输出文件名不能使用系统保留名称或以空格/点结尾")
    return cleaned


def _resolve_export_dir(raw_dir: Optional[str]) -> Optional[Path]:
    """校验并创建外部导出目录（必须为绝对路径或以 ~ 开头）。"""
    cleaned = (raw_dir or "").strip()
    if not cleaned:
        return None
    candidate = Path(cleaned).expanduser()
    if not candidate.is_absolute():
        raise ValueError("保存目录请填写绝对路径（支持 ~ 开头）")
    if candidate.exists() and not candidate.is_dir():
        raise ValueError(f"保存路径不是目录：{candidate}")
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ValueError(f"无法创建保存目录 {candidate}：{exc}") from exc
    return candidate.resolve()


def _copy_file_atomic(src: Path, dest_dir: Path, filename: str) -> Path:
    """在目标目录内原子拷贝产物文件，兼容跨盘/跨分区移动。"""
    target = (dest_dir / filename).resolve()
    if target == src.resolve():
        return target
    with tempfile.TemporaryDirectory(dir=dest_dir, prefix=".stagetts_export_") as tmp_dir:
        tmp_file = Path(tmp_dir) / filename
        shutil.copyfile(src, tmp_file)
        tmp_file.replace(target)
    return target


def mix_and_render(
    script: ProjectScript,
    burn_watermark: bool = False,
    burn_subtitles: SubtitleBurnMode = "none",
    auto_ducking: Optional[bool] = None,
    output_filename: Optional[str] = None,
) -> Tuple[str, float]:
    """执行 pydub 毫秒拼轨与 FFmpeg 混音封装导出。

    严格遵守 ARCHITECTURE.md 铁律 #7：
    - 混音支持四轨独立增益与可选自动闪避，并避免 amix 音量异常衰减。
    - 无硬字幕且 `burn_watermark=False` 时使用 `-c:v copy`（严禁混用 `-vf`）。
    - 需要烧录 ASS 硬字幕或合规水印时合并为单次 `-vf` 重编码，优先 `h264_nvenc` 并支持 `libx264` 兜底。
    """
    if not check_ffmpeg_available():
        raise RuntimeError("未在系统 PATH 中检测到 ffmpeg，请确认已安装 FFmpeg 6.x+。")

    source_video_path: Optional[Path] = None
    if script.media.source_video:
        candidate_video = config.resolve_media_path(script.media.source_video)
        if candidate_video.is_file():
            source_video_path = candidate_video

    is_audio_export = script.mode == "virtual_stage" and source_video_path is None
    if not is_audio_export and source_video_path is None:
        raise FileNotFoundError(
            f"未找到源视频文件 ({script.media.source_video})，无法封装 output.mp4"
        )
    target_filename = _sanitize_output_filename(output_filename, ".wav" if is_audio_export else ".mp4")

    project_dir: Path = config.WORKSPACE_DIR / script.project_id
    project_dir.mkdir(parents=True, exist_ok=True)
    dubbed_vocals_path = _build_vocal_track(script, project_dir)
    vocal_sr = _probe_wav_sample_rate(dubbed_vocals_path, default=44100)
    stems_paths, bgm_path = _resolve_background_tracks(script)

    # 模式 B 虚拟舞台若无源视频文件，导出完整混音音频
    if is_audio_export:
        output_wav = project_dir / target_filename
        if stems_paths is not None or bgm_path is not None:
            cmd = ["ffmpeg", "-y", "-i", str(dubbed_vocals_path)]
            if stems_paths is not None:
                for p in stems_paths:
                    cmd.extend(["-i", str(p)])
                bg_inputs: Tuple[int, ...] = (1, 2, 3)
            else:
                assert bgm_path is not None
                cmd.extend(["-i", str(bgm_path)])
                bg_inputs = (1,)
            chains, out_label, _ = _build_audio_mix_graph(
                vocal_idx=0,
                bg_inputs=bg_inputs,
                sample_rate=vocal_sr,
                script=script,
                auto_ducking=auto_ducking,
            )
            cmd.extend(["-filter_complex", ";".join(chains), "-map", out_label, str(output_wav)])
            proc = _run_ffmpeg_cmd(cmd)
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg 音频混音失败: {proc.stderr[-600:]}")
        else:
            shutil.copyfile(dubbed_vocals_path, output_wav)

        out_duration = probe_media_duration(output_wav) or script.media.duration
        return f"/workspace/{script.project_id}/{target_filename}", out_duration

    assert source_video_path is not None
    output_mp4 = project_dir / target_filename
    base_inputs: List[str] = ["ffmpeg", "-y", "-i", str(source_video_path), "-i", str(dubbed_vocals_path)]
    if stems_paths is not None:
        for p in stems_paths:
            base_inputs.extend(["-i", str(p)])
        chains, out_label, _ = _build_audio_mix_graph(
            vocal_idx=1,
            bg_inputs=(2, 3, 4),
            sample_rate=vocal_sr,
            script=script,
            auto_ducking=auto_ducking,
        )
        audio_filter: Optional[str] = ";".join(chains)
        audio_map = ["-map", "0:v:0", "-map", out_label]
    elif bgm_path is not None:
        base_inputs.extend(["-i", str(bgm_path)])
        chains, out_label, _ = _build_audio_mix_graph(
            vocal_idx=1,
            bg_inputs=(2,),
            sample_rate=vocal_sr,
            script=script,
            auto_ducking=auto_ducking,
        )
        audio_filter = ";".join(chains)
        audio_map = ["-map", "0:v:0", "-map", out_label]
    else:
        audio_filter = None
        audio_map = ["-map", "0:v:0", "-map", "1:a:0"]

    with tempfile.TemporaryDirectory(prefix="stagetts_sub_") as sub_tmp:
        video_filters: List[str] = []
        if burn_subtitles != "none":
            ass_path = write_ass_file(script, burn_subtitles, Path(sub_tmp) / "subtitles.ass")
            video_filters.append(f"ass=filename='{_escape_ffmpeg_filter_path(ass_path)}'")
        if burn_watermark:
            video_filters.append(_build_drawtext_filter())

        if not video_filters:
            cmd = list(base_inputs)
            if audio_filter:
                cmd.extend(["-filter_complex", audio_filter])
            cmd.extend(audio_map)
            cmd.extend(["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_mp4)])
            proc = _run_ffmpeg_cmd(cmd)
            if proc.returncode != 0:
                raise RuntimeError(f"FFmpeg 无损封装失败: {proc.stderr[-600:]}")
        else:
            vf_expr = ",".join(video_filters)
            for video_codec_args in _video_encoder_candidates():
                cmd = list(base_inputs)
                if audio_filter:
                    cmd.extend(["-filter_complex", audio_filter])
                cmd.extend(audio_map)
                cmd.extend(["-vf", vf_expr])
                cmd.extend(video_codec_args)
                cmd.extend(["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(output_mp4)])
                proc = _run_ffmpeg_cmd(cmd)
                if proc.returncode == 0:
                    break
            else:
                raise RuntimeError(f"FFmpeg 视频重编码导出失败: {proc.stderr[-600:]}")

    out_duration = probe_media_duration(output_mp4) or script.media.duration
    return f"/workspace/{script.project_id}/{target_filename}", out_duration


def export_project_deliverables(
    script: ProjectScript,
    *,
    burn_watermark: bool = False,
    burn_subtitles: SubtitleBurnMode = "none",
    export_ass_sidecar: Optional[SubtitleLayout] = None,
    auto_ducking: Optional[bool] = None,
    output_filename: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """导出成片与可选外挂 ASS 字幕，并按需原子落盘至用户指定本机目录。"""
    target_dir = _resolve_export_dir(output_dir)
    output_url, duration = mix_and_render(
        script,
        burn_watermark=burn_watermark,
        burn_subtitles=burn_subtitles,
        auto_ducking=auto_ducking,
        output_filename=output_filename,
    )
    workspace_output = config.resolve_media_path(output_url)
    saved_path = (
        _copy_file_atomic(workspace_output, target_dir, workspace_output.name)
        if target_dir is not None
        else workspace_output
    )

    subtitle_url: Optional[str] = None
    subtitle_saved_path: Optional[str] = None
    if export_ass_sidecar is not None:
        ass_name = f"{workspace_output.stem}.ass"
        workspace_ass = workspace_output.parent / ass_name
        write_ass_file(script, export_ass_sidecar, workspace_ass)
        subtitle_url = f"/workspace/{script.project_id}/{ass_name}"
        copied_ass = (
            _copy_file_atomic(workspace_ass, target_dir, ass_name)
            if target_dir is not None
            else workspace_ass
        )
        subtitle_saved_path = str(copied_ass)

    return {
        "output_video": output_url,
        "duration": duration,
        "burn_watermark": burn_watermark,
        "burn_subtitles": burn_subtitles,
        "auto_ducking": resolve_effective_auto_ducking(script, auto_ducking),
        "saved_path": str(saved_path),
        "subtitle_url": subtitle_url,
        "subtitle_saved_path": subtitle_saved_path,
    }
