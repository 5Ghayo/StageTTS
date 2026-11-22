"""GPT-SoVITS 推理客户端：按说话人分组批量合成 + 动态情绪参考音频路由。"""

from __future__ import annotations

import wave
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from pydub import AudioSegment

from backend import config
from backend.schemas import (
    CharacterConfig,
    EmotionRef,
    ProjectScript,
    Segment,
    SegmentDurationCheck,
)
from backend.services.task_runner import ensure_tts_running, gpu_task_guard


class TTSServiceUnavailableError(RuntimeError):
    """GPT-SoVITS 推理服务不可达或响应超时。"""


# 模块级权重缓存，避免连续单句精修同一角色时重复重载模型
_active_gpt_weights: Optional[str] = None
_active_sovits_weights: Optional[str] = None


def reset_active_weights_cache() -> None:
    """GPT-SoVITS 按需重启后清空模块级权重缓存。"""
    global _active_gpt_weights, _active_sovits_weights
    _active_gpt_weights = None
    _active_sovits_weights = None


def check_tts_service(timeout: float = 1.5) -> bool:
    """快速探测 GPT-SoVITS api_v2.py 是否在线。"""
    try:
        resp = requests.get(f"{config.TTS_API_URL}/", timeout=timeout)
        return resp.status_code < 500
    except requests.RequestException:
        return False


def resolve_emotion_reference(
    character: Optional[CharacterConfig],
    segment: Segment,
) -> Optional[EmotionRef]:
    """根据片段情绪动态路由参考音频；缺失时依次回退 neutral -> 首个可用参考音。"""
    if character is None or not character.emotion_refs:
        return None
    if segment.emotion in character.emotion_refs:
        return character.emotion_refs[segment.emotion]
    if "neutral" in character.emotion_refs:
        return character.emotion_refs["neutral"]
    return next(iter(character.emotion_refs.values()))


def _guess_prompt_lang(text: str, fallback: str = "zh") -> str:
    if not text:
        return fallback
    # 检测日文平假名/片假名，避免日语原声参考被误判为中文
    if any("\u3040" <= ch <= "\u30ff" for ch in text):
        return "ja"
    ascii_letters = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    return "en" if ascii_letters > len(text) * 0.5 else fallback


def _extract_zeroshot_ref(
    script: ProjectScript,
    speaker: str,
    tts_dir: Path,
) -> Optional[Tuple[EmotionRef, Path]]:
    """当未配置情绪参考音时，从 vocals.wav 截取 3~8s 原声作为零样本参考兜底。"""
    if not script.media.vocals_wav:
        return None
    try:
        vocals_path = config.resolve_media_path(script.media.vocals_wav)
    except ValueError:
        return None
    if not vocals_path.is_file():
        return None

    speaker_segs = [s for s in script.segments if s.speaker == speaker]
    if not speaker_segs:
        return None

    ideal = [s for s in speaker_segs if 3.0 <= (s.end - s.start) <= 8.0]
    chosen = ideal[0] if ideal else max(speaker_segs, key=lambda s: s.end - s.start)

    start_ms = max(0, int(chosen.start * 1000))
    end_ms = max(start_ms + 3000, min(int(chosen.end * 1000), start_ms + 8000))

    audio = AudioSegment.from_file(str(vocals_path))
    if start_ms >= len(audio):
        return None
    clip = audio[start_ms : min(len(audio), end_ms)]
    if len(clip) < 1000:
        return None

    out_path = tts_dir / f"_zeroshot_{speaker}.wav"
    clip.export(str(out_path), format="wav")
    prompt_text = chosen.orig_text.strip() or chosen.target_text.strip() or "这是一段参考语音。"
    prompt_lang = _guess_prompt_lang(prompt_text, chosen.target_lang or "zh")
    return EmotionRef(wav=str(out_path), text=prompt_text, lang=prompt_lang), out_path


def _resolve_ref_for_segment(
    script: ProjectScript,
    character: Optional[CharacterConfig],
    segment: Segment,
    tts_dir: Path,
) -> Optional[Tuple[EmotionRef, Path]]:
    ref = resolve_emotion_reference(character, segment)
    if ref is not None:
        try:
            ref_path = config.resolve_media_path(ref.wav)
            if ref_path.is_file():
                return ref, ref_path
        except ValueError:
            pass
    return _extract_zeroshot_ref(script, segment.speaker, tts_dir)


def _switch_weights(
    session: requests.Session,
    gpt_weights: Optional[str],
    sovits_weights: Optional[str],
) -> None:
    """按需切换 GPT/SoVITS 权重（缓存命中时跳过，避免重复重载引发显存抖动）。"""
    global _active_gpt_weights, _active_sovits_weights

    if gpt_weights:
        resolved_gpt = config.resolve_weights_path(gpt_weights)
        if resolved_gpt != _active_gpt_weights:
            try:
                resp = session.get(
                    f"{config.TTS_API_URL}/set_gpt_weights",
                    params={"weights_path": resolved_gpt},
                    timeout=config.TTS_TIMEOUT_SEC,
                )
            except requests.RequestException as exc:
                raise TTSServiceUnavailableError(
                    f"无法连接 GPT-SoVITS 权重接口 ({config.TTS_API_URL}): {exc}"
                ) from exc
            if resp.status_code != 200:
                raise RuntimeError(
                    f"切换 GPT 权重失败 ({resp.status_code}): {resp.text[:300]}"
                )
            _active_gpt_weights = resolved_gpt

    if sovits_weights:
        resolved_sovits = config.resolve_weights_path(sovits_weights)
        if resolved_sovits != _active_sovits_weights:
            try:
                resp = session.get(
                    f"{config.TTS_API_URL}/set_sovits_weights",
                    params={"weights_path": resolved_sovits},
                    timeout=config.TTS_TIMEOUT_SEC,
                )
            except requests.RequestException as exc:
                raise TTSServiceUnavailableError(
                    f"无法连接 GPT-SoVITS 权重接口 ({config.TTS_API_URL}): {exc}"
                ) from exc
            if resp.status_code != 200:
                raise RuntimeError(
                    f"切换 SoVITS 权重失败 ({resp.status_code}): {resp.text[:300]}"
                )
            _active_sovits_weights = resolved_sovits


def _measure_wav_duration(wav_path: Path) -> float:
    """精确测量合成 WAV 文件时长（秒）。"""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate()
            if rate > 0:
                return round(wf.getnframes() / float(rate), 3)
    except (wave.Error, EOFError):
        pass
    audio = AudioSegment.from_file(str(wav_path))
    return round(len(audio) / 1000.0, 3)


def _synthesize_single_line(
    session: requests.Session,
    seg: Segment,
    ref: EmotionRef,
    ref_path: Path,
    out_wav: Path,
) -> float:
    payload = {
        "text": seg.target_text,
        "text_lang": seg.target_lang or "zh",
        "ref_audio_path": str(ref_path),
        "prompt_text": ref.text,
        "prompt_lang": ref.lang or "zh",
        "speed_factor": float(seg.speed_factor),
        "media_type": "wav",
    }
    try:
        resp = session.post(
            f"{config.TTS_API_URL}/tts",
            json=payload,
            timeout=config.TTS_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        raise TTSServiceUnavailableError(
            f"无法连接 GPT-SoVITS 服务 ({config.TTS_API_URL})，请确认已启动 api_v2.py: {exc}"
        ) from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"片段 #{seg.id} TTS 推理失败 ({resp.status_code}): {resp.text[:300]}"
        )
    if not resp.content:
        raise RuntimeError(f"片段 #{seg.id} TTS 返回空音频流")

    out_wav.write_bytes(resp.content)
    return _measure_wav_duration(out_wav)


def synthesize_segments_grouped(
    script: ProjectScript,
    segment_ids: Optional[List[int]] = None,
) -> Tuple[List[int], List[SegmentDurationCheck], List[str]]:
    """按 speaker 分组批量调用 GPT-SoVITS (api_v2.py) 合成单句 WAV。

    严格遵守 ARCHITECTURE.md 铁律：
    1. 过滤 keep_original == True 的片段；segment_ids 为 None 时合成全部非 ready 片段。
    2. 先按 speaker 分组，每个角色最多切换一次 GPT/SoVITS 权重。
    3. 动态情绪参考音频路由 + 缺失时零样本截取 vocals.wav 兜底。
    4. 检测 actual_duration > (end - start) * 1.15 并返回 overflow_segments 供反思闭环处理。
    """
    target_set = set(segment_ids) if segment_ids is not None else None
    grouped: Dict[str, List[Segment]] = defaultdict(list)

    for seg in script.segments:
        if seg.keep_original:
            continue
        if target_set is not None and seg.id not in target_set:
            continue
        if target_set is None and seg.status == "ready":
            continue
        grouped[seg.speaker].append(seg)

    if not grouped:
        return [], [], []

    tts_dir = config.WORKSPACE_DIR / script.project_id / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    synthesized_ids: List[int] = []
    overflow_segments: List[SegmentDurationCheck] = []
    errors: List[str] = []

    with gpu_task_guard():
        try:
            ensure_tts_running()
        except RuntimeError as exc:
            raise TTSServiceUnavailableError(str(exc)) from exc
        with requests.Session() as session:
            for speaker, segs in grouped.items():
                character = script.characters.get(speaker)
                if character and (character.gpt_weights or character.sovits_weights):
                    _switch_weights(
                        session,
                        character.gpt_weights,
                        character.sovits_weights,
                    )

                for seg in segs:
                    if not seg.target_text.strip():
                        seg.status = "error"
                        errors.append(f"片段 #{seg.id} 目标台词为空，跳过合成")
                        continue

                    resolved = _resolve_ref_for_segment(
                        script, character, seg, tts_dir
                    )
                    if resolved is None:
                        seg.status = "error"
                        errors.append(
                            f"片段 #{seg.id} ({speaker}) 缺少有效情绪参考音频且无法从 vocals.wav 兜底"
                        )
                        continue

                    ref_obj, ref_path = resolved
                    out_name = f"seg_{seg.id:04d}.wav"
                    out_wav = tts_dir / out_name
                    try:
                        seg.status = "synthesizing"
                        actual_duration = _synthesize_single_line(
                            session, seg, ref_obj, ref_path, out_wav
                        )
                    except TTSServiceUnavailableError:
                        seg.status = "error"
                        raise
                    except Exception as exc:
                        seg.status = "error"
                        errors.append(str(exc))
                        continue

                    seg.tts_wav = f"/workspace/{script.project_id}/tts/{out_name}"
                    seg.actual_duration = actual_duration
                    seg.status = "ready"
                    synthesized_ids.append(seg.id)

                    slot_duration = max(0.01, round(seg.end - seg.start, 3))
                    overflow_ratio = round(actual_duration / slot_duration, 3)
                    overflow_seconds = round(actual_duration - slot_duration, 3)
                    if actual_duration > slot_duration * config.DURATION_OVERFLOW_RATIO:
                        overflow_segments.append(
                            SegmentDurationCheck(
                                segment_id=seg.id,
                                slot_duration=slot_duration,
                                actual_duration=actual_duration,
                                overflow_ratio=overflow_ratio,
                                overflow_seconds=overflow_seconds,
                                needs_reflection=True,
                            )
                        )

    return synthesized_ids, overflow_segments, errors
