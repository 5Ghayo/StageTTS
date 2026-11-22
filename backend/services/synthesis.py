"""配音时长质检、改写重配与最终试听 WAV 调速。"""

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
    slot = seg.end - seg.start
    actual = seg.actual_duration
    if actual is None or actual - slot <= 0.020:
        return None
    return SegmentDurationCheck(
        segment_id=seg.id,
        slot_duration=round(slot, 3),
        actual_duration=actual,
        overflow_ratio=round(actual / slot, 3),
        overflow_seconds=round(actual - slot, 3),
        needs_reflection=actual > slot * config.DURATION_OVERFLOW_RATIO,
    )


def _rewrite_overflow(
    client: OpenAI, script: ProjectScript, segments: dict[int, Segment], checks: list[SegmentDurationCheck]
) -> dict[int, str]:
    context = DialogueContext(script.model_copy(update={"segments": list(segments.values())}))
    payload = []
    for check in checks:
        seg = segments[check.segment_id]
        payload.append({
            **context.entry(seg, actual_duration=check.actual_duration),
            "target_text": seg.target_text,
            "actual_duration_sec": check.actual_duration,
            "overflow_seconds": check.overflow_seconds,
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
    speed = seg.applied_speed_factor
    if speed is None or not seg.tts_wav or seg.speed_factor != 1.0:
        return
    extra_speed = round(min(check.overflow_ratio, config.MAX_GENTLE_SPEED / speed), 3)
    if extra_speed <= 1.0:
        return
    wav_path = config.resolve_media_path(seg.tts_wav)
    with tempfile.TemporaryDirectory(dir=wav_path.parent) as tmp_dir:
        adjusted = apply_gentle_tempo(wav_path, extra_speed, Path(tmp_dir))
        duration = measure_wav_duration(adjusted)
        if duration < check.actual_duration:
            adjusted.replace(wav_path)
            seg.actual_duration = duration
            seg.applied_speed_factor = round(speed * extra_speed, 3)


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
                if (check := _check_duration(segments[seg_id])) and check.needs_reflection
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
            candidates = {
                seg_id: segments[seg_id].model_copy(update={"target_text": text})
                for seg_id, text in rewrites.items()
            }
            trial = script.model_copy(update={
                "segments": [candidates.get(seg.id, seg) for seg in segments.values()],
            })
            try:
                updated, retry_errors = synthesize_segments_grouped(trial, list(rewrites), require_shorter=True)
            except RuntimeError as exc:
                errors.append(f"片段 {ids} 重配失败，保留上一版配音：{exc}")
                break
            errors.extend(retry_errors)
            if not updated:
                break
            segments.update((seg_id, candidates[seg_id]) for seg_id in updated)

    script.segments = [segments[seg.id] for seg in script.segments]
    overflow = []
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
            overflow.append(final_check)
    return synthesized_ids, overflow, errors
