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
    "参照相邻台词理解语境，遵循角色语气和术语表。术语出现时使用表中的固定译法。"
    "每条的时长预算以 max_units 和 unit 表示。只处理 lines 中的片段 ID。"
)


class LLMNotConfiguredError(RuntimeError):
    """未配置有效的 LLM API Key。"""


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
    content = response.choices[0].message.content or ""
    entries = json.loads(content[content.find("["):content.rfind("]") + 1])
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
            "max_units": budget,
            "before": [self._line(other) for other in self.ordered[max(0, index - 2):index]],
            "after": [self._line(other) for other in self.ordered[index + 1:index + 3]],
        }

    def payload(self, entries: Iterable[dict]) -> dict:
        return {"source_lang": self.script.source_lang, "glossary": self.script.glossary, "lines": list(entries)}


def translate_script(
    script: ProjectScript, client: OpenAI, log: Callable[[str], None] | None = None,
) -> list[int]:
    context = DialogueContext(script)
    segments = [seg for seg in context.ordered if seg.orig_text.strip()]
    translated = []
    batch_size = 25
    for offset in range(0, len(segments), batch_size):
        batch = segments[offset:offset + batch_size]
        pending = {seg.id: seg for seg in batch}
        if log:
            log(f"调用导演 LLM 标注片段 #{batch[0].id}..#{batch[-1].id}")
        try:
            entries = complete_dialogue(
                client, DIRECTOR_TRANSLATE_PROMPT, context.payload(context.entry(seg) for seg in batch),
            )
        except (OpenAIError, ValueError, IndexError):
            if log:
                log("LLM 标注批次失败，保留当前台词")
            continue

        for entry in entries:
            seg_id = entry.get("id")
            text = entry.get("target_text")
            if type(seg_id) is not int or seg_id not in pending or not isinstance(text, str) or not text.strip():
                continue
            seg = pending.pop(seg_id)
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
            translated.append(seg_id)
    return translated
