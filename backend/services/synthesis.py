"""配音时长质检、双向改写重配与最终试听 WAV 调速。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from openai import OpenAI, OpenAIError

from backend import config
from backend.agent.prompts import DURATION_REFLECTION_PROMPT
from backend.schemas import ProjectScript, Segment, SegmentDurationCheck
from backend.services.media_mixer import apply_gentle_tempo
from backend.services.translation import DialogueContext, LLMNotConfiguredError, complete_dialogue, create_openai_client
from backend.services.tts_client import measure_wav_duration, synthesize_segments_grouped


def _check_duration(seg: Segment) -> SegmentDurationCheck | None:
    """判断片段是否显著偏离原时间槽。

    - too_long：actual > slot * 1.15
    - too_short：actual < slot * 0.80（且槽位不小于 1 秒，避免把纯叹词强行扩写）
    """
    slot = max(0.05, seg.dub_end - seg.dub_start)
    speech_duration = seg.speech_cue.speech_end - seg.dub_start if seg.speech_cue else slot
    actual = seg.actual_duration
    if actual is None:
        return None
    fill = round(actual / slot, 3)
    overflow = actual > seg.dub_end - seg.dub_start + 1e-6 if seg.speech_cue else fill > config.DURATION_OVERFLOW_RATIO
    if overflow:
        state = "too_long"
    elif (actual < speech_duration * config.DURATION_UNDERFLOW_RATIO
          and speech_duration >= config.MIN_DURATION_REFLECTION_SECONDS):
        state = "too_short"
        slot = speech_duration
        fill = round(actual / slot, 3)
    else:
        return None
    return SegmentDurationCheck(
        segment_id=seg.id,
        slot_duration=round(slot, 3),
        actual_duration=actual,
        fill_ratio=fill,
        duration_state=state,
        overflow_ratio=fill,
        overflow_seconds=round(actual - slot, 3),
        needs_reflection=True,
    )


def _rewrite_overflow(
    client: OpenAI, script: ProjectScript, segments: dict[int, Segment], checks: list[SegmentDurationCheck]
) -> dict[int, str]:
    """让 LLM 按实际偏差改写：太长则精简，太短则自然补足衔接。"""
    context = DialogueContext(script.model_copy(update={"segments": list(segments.values())}))
    payload = []
    for check in checks:
        seg = segments[check.segment_id]
        payload.append({
            **context.entry(seg, actual_duration=check.actual_duration),
            "target_text": seg.target_text,
            "duration_state": check.duration_state,
            "fill_ratio": check.fill_ratio,
            "actual_duration_sec": check.actual_duration,
            "overflow_seconds": check.overflow_seconds,
            "constraints": {
                "preserve": "实体、数字、否定、指代、术语必须保留；min_units/max_units 是软约束",
                "too_long": "只删冗余口气词/重复衔接，不删核心信息",
                "too_short": "只自然补足语气和衔接，不编造新事实",
                "glossary": context.script.glossary,
            },
        })
    entries = complete_dialogue(client, DURATION_REFLECTION_PROMPT, context.payload(payload))
    requested = {check.segment_id for check in checks}
    rewrites = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        seg_id, text = entry.get("id"), entry.get("target_text")
        if type(seg_id) is int and seg_id in requested and isinstance(text, str):
            text = text.strip()
            if text and text != segments[seg_id].target_text:
                rewrites[seg_id] = text
    return rewrites


def _fit_wav(seg: Segment, check: SegmentDurationCheck) -> None:
    """在不改变用户显式 speed_factor 的前提下，对仍偏离的音频做温和兜底。"""
    if not seg.tts_wav or seg.speed_factor != 1.0:
        return
    speed = seg.applied_speed_factor or 1.0
    if check.duration_state == "too_long":
        extra_speed = round(min(check.fill_ratio, config.MAX_GENTLE_SPEED / speed), 3)
        if extra_speed <= 1.01:
            return
    else:
        extra_speed = round(max(check.fill_ratio, config.MIN_GENTLE_SPEED / speed), 3)
        if extra_speed >= 0.99:
            return

    wav_path = config.resolve_media_path(seg.tts_wav)
    with tempfile.TemporaryDirectory(dir=wav_path.parent) as tmp_dir:
        adjusted = apply_gentle_tempo(wav_path, extra_speed, Path(tmp_dir))
        duration = measure_wav_duration(adjusted)
        improved = (
            duration < check.actual_duration
            if check.duration_state == "too_long"
            else duration > check.actual_duration
        )
        if improved:
            adjusted.replace(wav_path)
            seg.actual_duration = duration
            seg.applied_speed_factor = round(speed * extra_speed, 3)


def _retry_rewrites(
    script: ProjectScript,
    segments: dict[int, Segment],
    rewrites: dict[int, str],
    state: str,
    errors: list[str],
) -> bool:
    if not rewrites:
        return False
    candidates = {
        seg_id: segments[seg_id].model_copy(update={"target_text": text})
        for seg_id, text in rewrites.items()
    }
    trial = script.model_copy(update={
        "segments": [candidates.get(seg.id, seg) for seg in segments.values()],
    })
    kwargs = {"require_shorter": state == "too_long", "require_longer": state == "too_short"}
    try:
        updated, retry_errors = synthesize_segments_grouped(trial, list(rewrites), **kwargs)
    except RuntimeError as exc:
        ids = ", ".join(f"#{seg_id}" for seg_id in rewrites)
        errors.append(f"片段 {ids} 重配失败，保留上一版配音：{exc}")
        return False
    errors.extend(retry_errors)
    if not updated:
        return False
    segments.update((seg_id, candidates[seg_id]) for seg_id in updated)
    return True


def synthesize_with_duration_check(
    script: ProjectScript, segment_ids: list[int] | None, *, auto_reflect: bool = True
) -> tuple[list[int], list[SegmentDurationCheck], list[str]]:
    synthesized_ids, errors = synthesize_segments_grouped(script, segment_ids)
    segments = {seg.id: seg for seg in script.segments}
    client = None
    if auto_reflect:
        for _ in range(config.MAX_REFLECTION_ROUNDS):
            pending = [
                check for seg_id in synthesized_ids
                if (check := _check_duration(segments[seg_id])) is not None
            ]
            if not pending:
                break
            ids = ", ".join(f"#{check.segment_id}" for check in pending)
            try:
                if client is None:
                    client = create_openai_client()
                rewrites = _rewrite_overflow(client, script, segments, pending)
            except LLMNotConfiguredError:
                errors.append(f"片段 {ids} 未自动改写：未配置 LLM。")
                break
            except (OpenAIError, ValueError, IndexError):
                errors.append(f"片段 {ids} 自动改写失败，保留上一版配音。")
                break
            if not rewrites:
                errors.append(f"片段 {ids} 未取得可用改写，保留上一版配音。")
                break

            grouped: dict[str, dict[int, str]] = {"too_long": {}, "too_short": {}}
            for check in pending:
                text = rewrites.get(check.segment_id)
                if text:
                    grouped[check.duration_state][check.segment_id] = text
            progressed = False
            for state, group in grouped.items():
                progressed |= _retry_rewrites(script, segments, group, state, errors)
            if not progressed:
                break

    script.segments = [segments[seg.id] for seg in script.segments]
    remaining: list[SegmentDurationCheck] = []
    for seg_id in synthesized_ids:
        seg = segments[seg_id]
        check = _check_duration(seg)
        if check is None:
            continue
        try:
            _fit_wav(seg, check)
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"片段 #{seg_id} 自动调速失败：{exc}")
        if final_check := _check_duration(seg):
            remaining.append(final_check)
    return synthesized_ids, remaining, errors
