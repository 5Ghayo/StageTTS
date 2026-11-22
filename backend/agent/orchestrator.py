"""大模型 Function Calling 多轮调度器与时长约束反思闭环。"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from backend import config
from backend.agent.prompts import (
    DURATION_REFLECTION_PROMPT,
    EDITING_AGENT_SYSTEM_PROMPT,
)
from backend.agent.tools import (
    TOOL_REGISTRY,
    TOOLS_SCHEMA,
    batch_update_segments,
    load_project_script,
    synthesize_segments,
)
from backend.schemas import (
    AgentChatResponse,
    SegmentDurationCheck,
    ToolCallLog,
)


class LLMNotConfiguredError(RuntimeError):
    """未配置有效的 LLM API Key 时抛出，由接口层映射为 503。"""


def _create_openai_client() -> OpenAI:
    if not config.is_llm_configured():
        raise LLMNotConfiguredError(
            "未配置有效的 LLM_API_KEY，请在 backend/.env 中设置 LLM_API_KEY、LLM_BASE_URL 与 LLM_MODEL。"
        )
    return OpenAI(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
        timeout=config.LLM_TIMEOUT_SEC,
    )


def _parse_json_array(raw_text: str) -> List[Dict[str, Any]]:
    """从大模型返回文本中容错提取 JSON 数组。"""
    cleaned = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    start_idx = cleaned.find("[")
    end_idx = cleaned.rfind("]")
    if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
        return []
    parsed = json.loads(cleaned[start_idx : end_idx + 1])
    return [item for item in parsed if isinstance(item, dict)] if isinstance(parsed, list) else []


def _summarize_tool_result(tool_name: str, result: Dict[str, Any]) -> str:
    if tool_name == "query_script":
        return f"查询到 {result.get('count', 0)} 条片段"
    if tool_name == "batch_update_segments":
        updated = result.get("updated_ids", [])
        pending = result.get("marked_pending_ids", [])
        return f"更新片段 {updated}（待重合成: {pending}）"
    if tool_name == "synthesize_segments":
        synth = result.get("synthesized_ids", [])
        overflow = result.get("overflow_segments", [])
        return f"合成片段 {synth}（超时需反思: {len(overflow)} 条）"
    if tool_name == "update_mix_settings":
        return f"已更新 BGM 音量为 {result.get('bgm_volume')}"
    if tool_name == "render_final_video":
        return f"已导出成片 {result.get('output_video')}"
    return "执行完成"


def run_duration_reflection_loop(
    project_id: str,
    overflow_segments: List[SegmentDurationCheck],
    max_rounds: int = config.MAX_REFLECTION_ROUNDS,
    tool_logs: Optional[List[ToolCallLog]] = None,
) -> List[int]:
    """时长约束自动反思改写闭环（最多迭代 max_rounds=2 轮）。"""
    pending = [item for item in overflow_segments if item.needs_reflection]
    if not pending:
        return []

    client = _create_openai_client()
    revised_ids: List[int] = []

    for round_idx in range(max_rounds):
        if not pending:
            break

        script = load_project_script(project_id)
        seg_map = {seg.id: seg for seg in script.segments}
        feedback_items: List[Dict[str, Any]] = []
        pending_ids = set()

        for item in pending:
            seg = seg_map.get(item.segment_id)
            if seg is None:
                continue
            pending_ids.add(seg.id)
            recommended_chars = max(2, int(item.slot_duration * 4.8))
            feedback_items.append(
                {
                    "id": seg.id,
                    "emotion": seg.emotion,
                    "orig_text": seg.orig_text,
                    "current_target_text": seg.target_text,
                    "slot_duration_sec": item.slot_duration,
                    "actual_duration_sec": item.actual_duration,
                    "overflow_seconds": item.overflow_seconds,
                    "recommended_max_chars": recommended_chars,
                }
            )

        if not feedback_items:
            break

        completion = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": DURATION_REFLECTION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"第 {round_idx + 1} 轮时长反思校对，请精简以下超长台词：\n"
                        + json.dumps(feedback_items, ensure_ascii=False, indent=2)
                    ),
                },
            ],
            temperature=0.3,
        )
        raw_content = completion.choices[0].message.content or ""
        rewrites = _parse_json_array(raw_content)

        updates: List[Dict[str, Any]] = []
        for entry in rewrites:
            seg_id = entry.get("id")
            new_text = str(entry.get("target_text", "")).strip()
            if isinstance(seg_id, int) and seg_id in pending_ids and new_text:
                updates.append({"id": seg_id, "target_text": new_text})

        if not updates:
            break

        upd_args = {"project_id": project_id, "updates": updates}
        upd_res = batch_update_segments(**upd_args)
        if tool_logs is not None:
            tool_logs.append(
                ToolCallLog(
                    tool_name="batch_update_segments",
                    arguments=upd_args,
                    result_summary=f"时长反思第 {round_idx + 1} 轮改写: {_summarize_tool_result('batch_update_segments', upd_res)}",
                )
            )

        target_ids = [u["id"] for u in updates]
        revised_ids.extend(target_ids)
        synth_args = {"project_id": project_id, "segment_ids": target_ids}
        synth_res = synthesize_segments(**synth_args)
        if tool_logs is not None:
            tool_logs.append(
                ToolCallLog(
                    tool_name="synthesize_segments",
                    arguments=synth_args,
                    result_summary=f"时长反思第 {round_idx + 1} 轮重合成: {_summarize_tool_result('synthesize_segments', synth_res)}",
                )
            )

        pending = [
            SegmentDurationCheck.model_validate(item)
            for item in synth_res.get("overflow_segments", [])
            if item.get("needs_reflection")
        ]

    # 若达到反思轮数上限仍略超标，施加 <= MAX_GENTLE_SPEED (1.15x) 温和变速兜底
    if pending:
        gentle_patches: List[Dict[str, Any]] = []
        for item in pending:
            gentle_speed = min(config.MAX_GENTLE_SPEED, max(1.0, round(item.overflow_ratio, 3)))
            gentle_patches.append({"id": item.segment_id, "speed_factor": gentle_speed})
        if gentle_patches:
            spd_args = {"project_id": project_id, "updates": gentle_patches}
            spd_res = batch_update_segments(**spd_args)
            if tool_logs is not None:
                tool_logs.append(
                    ToolCallLog(
                        tool_name="batch_update_segments",
                        arguments=spd_args,
                        result_summary=f"达到反思上限，应用温和变速 <= {config.MAX_GENTLE_SPEED}x: {spd_res.get('updated_ids')}",
                    )
                )

    return sorted(set(revised_ids))


def run_agent_chat(
    project_id: str,
    user_message: str,
    auto_synthesize: bool = True,
) -> AgentChatResponse:
    """自然语言批量剪辑 Agent 主调度入口（严格通过 5 大确定性工具白名单执行）。"""
    client = _create_openai_client()
    script = load_project_script(project_id)

    project_context = {
        "project_id": script.project_id,
        "mode": script.mode,
        "duration": script.media.duration,
        "bgm_volume": script.media.bgm_volume,
        "characters": {
            key: char.display_name for key, char in script.characters.items()
        },
        "segment_count": len(script.segments),
        "auto_synthesize": auto_synthesize,
    }

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": EDITING_AGENT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"当前项目上下文：\n{json.dumps(project_context, ensure_ascii=False)}\n\n"
                f"用户指令：{user_message}"
            ),
        },
    ]

    tool_logs: List[ToolCallLog] = []
    final_reply = "已完成指令处理。"
    pending_modified = False
    last_overflow: List[SegmentDurationCheck] = []

    for _ in range(6):
        completion = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,  # type: ignore[arg-type]
            tools=TOOLS_SCHEMA,  # type: ignore[arg-type]
            tool_choice="auto",
            temperature=0.2,
        )
        assistant_msg = completion.choices[0].message
        tool_calls = getattr(assistant_msg, "tool_calls", None) or []

        if not tool_calls:
            final_reply = assistant_msg.content or final_reply
            break

        assistant_record: Dict[str, Any] = {
            "role": "assistant",
            "content": assistant_msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        }
        messages.append(assistant_record)

        for tc in tool_calls:
            tool_name = tc.function.name
            func = TOOL_REGISTRY.get(tool_name)
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("工具参数必须为 JSON 对象")
                # 强制锁定 project_id，禁止越权读写其他项目
                args["project_id"] = project_id

                if func is None:
                    raise ValueError(f"禁止调用未授权工具: {tool_name}")

                result = func(**args)
                summary = _summarize_tool_result(tool_name, result)
                tool_logs.append(
                    ToolCallLog(
                        tool_name=tool_name,
                        arguments=args,
                        result_summary=summary,
                    )
                )

                if tool_name == "batch_update_segments" and result.get("marked_pending_ids"):
                    pending_modified = True
                if tool_name == "synthesize_segments":
                    pending_modified = False
                    last_overflow = [
                        SegmentDurationCheck.model_validate(item)
                        for item in result.get("overflow_segments", [])
                    ]
            except Exception as exc:
                result = {"error": str(exc)}
                tool_logs.append(
                    ToolCallLog(
                        tool_name=tool_name,
                        arguments={"project_id": project_id},
                        result_summary=f"调用失败: {exc}",
                    )
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    # 兜底闭环：若 LLM 修改了台词/情绪但未收尾合成，或存在超标片段未反思，自动补全闭环
    if auto_synthesize:
        if pending_modified:
            synth_args = {"project_id": project_id, "segment_ids": None}
            synth_res = synthesize_segments(**synth_args)
            tool_logs.append(
                ToolCallLog(
                    tool_name="synthesize_segments",
                    arguments=synth_args,
                    result_summary=f"自动收尾合成: {_summarize_tool_result('synthesize_segments', synth_res)}",
                )
            )
            last_overflow = [
                SegmentDurationCheck.model_validate(item)
                for item in synth_res.get("overflow_segments", [])
            ]

        if any(item.needs_reflection for item in last_overflow):
            revised = run_duration_reflection_loop(
                project_id=project_id,
                overflow_segments=last_overflow,
                tool_logs=tool_logs,
            )
            if revised:
                final_reply = f"{final_reply}\n（已自动完成超时片段 {revised} 的反思改写与重合成）"

    latest_script = load_project_script(project_id)
    return AgentChatResponse(
        project_id=project_id,
        reply=final_reply,
        tool_logs=tool_logs,
        script=latest_script,
    )
