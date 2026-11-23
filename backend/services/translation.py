"""Dialogue context shared by parse-time translation and duration rewrites."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable

from openai import OpenAI, OpenAIError

from backend import config
from backend.agent.prompts import DIRECTOR_TRANSLATE_PROMPT
from backend.schemas import ProjectScript, Segment

EMOTIONS = {"neutral", "angry", "happy", "sad", "whisper"}
CONTEXT_INSTRUCTIONS = (
    "先通读完整对白理解剧情、角色关系和前后语气，再按原句 id 还原译文。"
    "参照相邻台词、角色语气和术语表；术语出现时使用固定译法。"
    "每条的时长预算以 max_units 和 unit 表示。只处理 lines 中的片段 ID。"
)


class LLMNotConfiguredError(RuntimeError):
    """未配置有效的 LLM API Key。"""


class LLMServiceError(RuntimeError):
    """调用上游大模型网关失败或返回非法结果。"""


def create_openai_client() -> OpenAI:
    if not config.is_llm_configured():
        raise LLMNotConfiguredError("请在 backend/.env 中配置 LLM_API_KEY、LLM_BASE_URL 与 LLM_MODEL。")
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, timeout=config.LLM_TIMEOUT_SEC)


def complete_dialogue(client: OpenAI, system: str, payload: dict) -> list[dict]:
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system + "\n" + CONTEXT_INSTRUCTIONS},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.3,
    )
    content = (response.choices[0].message.content or "").strip()
    fences = re.findall(r"```(json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
    if fences:
        content = (
            next((body.strip() for lang, body in fences if lang and "[" in body and "]" in body), None)
            or next((body.strip() for _, body in fences if "[" in body and "]" in body), None)
            or fences[0][1].strip()
        )
    left, right = content.find("["), content.rfind("]")
    if left == -1 or right <= left:
        raise ValueError("LLM 未返回 JSON 数组")
    entries = json.loads(content[left:right + 1])
    if not isinstance(entries, list):
        raise ValueError("LLM 未返回 JSON 数组")
    return [entry for entry in entries if isinstance(entry, dict)]


class DialogueContext:
    def __init__(self, script: ProjectScript):
        self.script = script
        self.ordered = sorted(script.segments, key=lambda seg: (seg.start, seg.id))
        self.positions = {seg.id: index for index, seg in enumerate(self.ordered)}

    def _line(self, seg: Segment) -> dict:
        character = self.script.characters.get(seg.speaker)
        return {
            "id": seg.id,
            "speaker": seg.speaker,
            "speaker_name": character.display_name if character else seg.speaker,
            "speaking_style": character.speaking_style if character else "",
            "orig_text": seg.orig_text,
            "target_text": seg.target_text if seg.target_text != seg.orig_text else "",
        }

    def entry(self, seg: Segment, *, actual_duration: float | None = None) -> dict:
        index = self.positions[seg.id]
        slot = seg.end - seg.start
        cjk = seg.target_lang.removeprefix("all_").split("-")[0] in {"zh", "ja", "yue", "ko"}
        budget = max(1, int(slot * (5.0 if cjk else 2.8)))
        if actual_duration is not None:
            units = len(re.sub(r"\s+", "", seg.target_text)) if cjk else len(seg.target_text.split())
            budget = min(budget, max(1, int(units * slot / actual_duration * 0.95)))
        return {
            **self._line(seg),
            "target_lang": seg.target_lang,
            "emotion": seg.emotion,
            "slot_sec": round(slot, 3),
            "unit": "characters" if cjk else "words",
            "min_units": max(1, int(budget * 0.55)),
            "max_units": budget,
            "before": [self._line(other) for other in self.ordered[max(0, index - 2):index]],
            "after": [self._line(other) for other in self.ordered[index + 1:index + 3]],
        }

    def payload(self, entries: Iterable[dict]) -> dict:
        return {"source_lang": self.script.source_lang, "glossary": self.script.glossary, "lines": list(entries)}

    def whole_dialogue_payload(self, segments: Iterable[Segment]) -> dict:
        """构造一次性通读全文的翻译请求，并保留原句 id 供结果还原。"""
        ordered = sorted(segments, key=lambda seg: (seg.start, seg.id))
        lines = [self.entry(seg) for seg in ordered]
        dialogue_lines = []
        for seg in ordered:
            character = self.script.characters.get(seg.speaker)
            speaker_name = character.display_name if character else seg.speaker
            dialogue_lines.append(f"[{seg.id}] {speaker_name}: {seg.orig_text.strip()}")
        target_lang = next((seg.target_lang for seg in ordered if seg.target_lang), "zh")
        return {
            "source_lang": self.script.source_lang,
            "target_lang": target_lang,
            "glossary": self.script.glossary,
            "translation_mode": "whole_dialogue_once",
            "dialogue": "\n".join(dialogue_lines),
            "lines": lines,
            "output_requirements": (
                "先理解整段对白，再逐条还原到 lines 中的 id；"
                "只输出 JSON 数组，不输出完整译文正文；"
                "每项必须包含 id、target_text、emotion、action、keep_original。"
            ),
        }


def translate_script(
    script: ProjectScript,
    client: OpenAI,
    log: Callable[[str], None] | None = None,
    errors: list[str] | None = None,
) -> list[int]:
    context = DialogueContext(script)
    segments = [seg for seg in context.ordered if seg.orig_text.strip()]
    if not segments:
        return []

    payload = context.whole_dialogue_payload(segments)
    if log:
        log(f"调用导演 LLM 一次性通读并翻译全部 {len(segments)} 条台词")

    try:
        entries = complete_dialogue(client, DIRECTOR_TRANSLATE_PROMPT, payload)
    except (OpenAIError, ValueError, IndexError, json.JSONDecodeError) as exc:
        if errors is not None:
            errors.append(f"全文翻译失败：{exc}")
        if log:
            log("全文翻译失败，保留当前台词")
        return []

    translated: list[int] = []
    pending = {seg.id: seg for seg in segments}
    for entry in entries:
        seg_id = entry.get("id")
        text = entry.get("target_text")
        if type(seg_id) is not int or seg_id not in pending or not isinstance(text, str) or not text.strip():
            continue
        seg = pending.pop(seg_id, None)
        if seg is None:
            continue
        emotion = entry.get("emotion")
        emotion = emotion if isinstance(emotion, str) and emotion in EMOTIONS else "neutral"
        action = entry.get("action")
        keep_original = entry.get("keep_original") is True
        needs_resynth = (text.strip(), emotion, keep_original) != (seg.target_text, seg.emotion, seg.keep_original)
        seg.target_text = text.strip()
        seg.emotion = emotion
        seg.action = action.strip() if isinstance(action, str) and action.strip() else "idle"
        seg.keep_original = keep_original
        if needs_resynth:
            seg.invalidate_audio()
        translated.append(seg.id)
    return translated
