"""大模型 Function Calling 多轮调度器与时长约束反思闭环。"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from backend import config
from backend.agent.prompts import EDITING_AGENT_SYSTEM_PROMPT
from backend.agent.tools import (
    TOOL_REGISTRY,
    TOOLS_SCHEMA,
    load_project_script,
    synthesize_segments,
)
from backend.schemas import (
    AgentChatResponse,
    SegmentDurationCheck,
    ToolCallLog,
)
from backend.services.synthesis import create_openai_client


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
        return f"合成片段 {synth}（仍超长: {len(overflow)} 条）"
    if tool_name == "update_mix_settings":
        return f"已更新 BGM 音量为 {result.get('bgm_volume')}"
    if tool_name == "render_final_video":
        return f"已导出成片 {result.get('output_video')}"
    return "执行完成"


def run_agent_chat(
    project_id: str,
    user_message: str,
    auto_synthesize: bool = True,
) -> AgentChatResponse:
    """自然语言批量剪辑 Agent 主调度入口（严格通过 5 大确定性工具白名单执行）。"""
    client = create_openai_client()
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
    pending_modified: set[int] = set()
    synthesized_ids: set[int] = set()
    overflow: dict[int, SegmentDurationCheck] = {}
    synthesis_errors: list[str] = []

    def record_synthesis(result: Dict[str, Any]) -> None:
        completed = result["synthesized_ids"]
        synthesized_ids.update(completed)
        pending_modified.difference_update(completed)
        for seg_id in completed:
            overflow.pop(seg_id, None)
        for item in result["overflow_segments"]:
            check = SegmentDurationCheck.model_validate(item)
            overflow[check.segment_id] = check
        synthesis_errors.extend(result["errors"])

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

                if tool_name == "synthesize_segments":
                    result = synthesize_segments(**(args | {"auto_reflect": auto_synthesize}))
                    record_synthesis(result)
                else:
                    result = func(**args)
                summary = _summarize_tool_result(tool_name, result)
                tool_logs.append(
                    ToolCallLog(
                        tool_name=tool_name,
                        arguments=args,
                        result_summary=summary,
                    )
                )

                if tool_name == "batch_update_segments":
                    pending_modified.update(result["marked_pending_ids"])
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

    if auto_synthesize and pending_modified:
        synth_args = {"project_id": project_id, "segment_ids": sorted(pending_modified)}
        synth_res = synthesize_segments(**synth_args)
        record_synthesis(synth_res)
        tool_logs.append(
            ToolCallLog(
                tool_name="synthesize_segments",
                arguments=synth_args,
                result_summary=f"自动收尾合成: {_summarize_tool_result('synthesize_segments', synth_res)}",
            )
        )

    if overflow:
        details = "、".join(f"#{item.segment_id} 超 {item.overflow_seconds:.3f}s" for item in overflow.values())
        final_reply += f"\n配音仍超出时间槽：{details}。"
    if synthesis_errors:
        final_reply += "\n" + "\n".join(dict.fromkeys(synthesis_errors))

    latest_script = load_project_script(project_id)
    return AgentChatResponse(
        project_id=project_id,
        reply=final_reply,
        tool_logs=tool_logs,
        synthesized_ids=sorted(synthesized_ids),
        script=latest_script,
    )
