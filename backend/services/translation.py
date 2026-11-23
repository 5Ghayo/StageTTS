"""Dialogue context shared by parse-time translation and duration rewrites."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable

from openai import OpenAI, OpenAIError

from backend import config
from backend.agent.prompts import DIRECTOR_TRANSLATE_PROMPT, TRANSLATION_REPAIR_PROMPT
from backend.schemas import ProjectScript, Segment

EMOTIONS = {"neutral", "angry", "happy", "sad", "whisper"}
WHOLE_DIALOGUE_MAX_SEGMENTS = 80
WINDOW_CONTEXT_MARGIN = 20
_CJK_LANGS = {"zh", "ja", "yue", "ko"}
_CJK_CHAR_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_NON_CJK_WORD_RE = re.compile(r"[^\W_]+(?:['-][^\W_]+)*")
_ASCII_GLOSSARY_TERM_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\s\-']*[A-Za-z0-9]|[A-Za-z0-9]")
CONTEXT_INSTRUCTIONS = (
    "先通读完整对白理解剧情、角色关系和前后语气，再按原句 id 还原译文。"
    "参照相邻台词、角色语气和术语表；术语出现时使用固定译法。"
    "忠实优先：不得摘要、概括、省略、删词或改成名词短语；min_units/max_units 只是软预算。"
    "只处理 lines 中的片段 ID。"
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
        return {
            "source_lang": self.script.source_lang,
            "glossary": self.script.glossary,
            "constraints": {
                "fidelity": "逐句完整翻译，不得摘要、概括、省略或改成名词短语",
                "glossary": "术语按固定译法，不得同义改写",
                "spoken_form": "中文输出中的英文缩写/单字母必须是可朗读形式",
                "budget": "min_units/max_units 是软预算，语义完整优先",
            },
            "lines": list(entries),
        }

    def whole_dialogue_payload(
        self,
        segments: Iterable[Segment],
        *,
        full_segments: Iterable[Segment] | None = None,
        dialogue_window: tuple[int, int, int] | None = None,
    ) -> dict:
        """构造通读全文或滑动窗口上下文的翻译请求，并保留原句 id 供结果还原。"""
        ordered = sorted(segments, key=lambda seg: (seg.start, seg.id))
        dialogue_source = sorted(full_segments or segments, key=lambda seg: (seg.start, seg.id))
        lines = [self.entry(seg) for seg in ordered]
        dialogue_lines = []
        for seg in dialogue_source:
            character = self.script.characters.get(seg.speaker)
            speaker_name = character.display_name if character else seg.speaker
            dialogue_lines.append(f"[{seg.id}] {speaker_name}: {seg.orig_text.strip()}")
        target_lang = next((seg.target_lang for seg in ordered if seg.target_lang), "zh")
        payload = self.payload(lines)
        payload.update({
            "target_lang": target_lang,
            "translation_mode": "windowed_dialogue" if dialogue_window is not None else "whole_dialogue_once",
            "dialogue": "\n".join(dialogue_lines),
            "output_requirements": (
                "先理解整段对白，再逐条还原到 lines 中的 id；"
                "只输出 JSON 数组，不输出完整译文正文；"
                "每项必须包含 id、target_text、emotion、action、keep_original、self_check。"
            ),
        })
        if dialogue_window is not None:
            start_idx, end_idx, total_count = dialogue_window
            payload["dialogue_window"] = {
                "start_index": start_idx,
                "end_index": end_idx,
                "total_segments": total_count,
            }
        return payload


def _self_check_flags(entry: dict) -> tuple[bool, str]:
    check = entry.get("self_check")
    if not isinstance(check, dict):
        return True, ""
    complete = check.get("semantic_complete")
    omitted = str(check.get("omitted_info") or "").strip()
    return complete is not False, omitted


def _count_non_cjk_words(text: str) -> int:
    cleaned = _CJK_CHAR_RE.sub(" ", text)
    return len(_NON_CJK_WORD_RE.findall(cleaned))


def _is_suspicious(seg: Segment, text: str, glossary: dict[str, str] | None = None) -> bool:
    """启发式判断是否可能摘要化/漏译或遗漏术语表规定译法；只触发一次 repair。"""
    source = seg.orig_text.strip()
    candidate = text.strip()
    target_cjk = seg.target_lang.removeprefix("all_").split("-")[0] in _CJK_LANGS
    src_cjk = len(_CJK_CHAR_RE.findall(source))
    src_words = _count_non_cjk_words(source)
    compact_src = re.sub(r"\s+", "", source)
    compact_cand = re.sub(r"\s+", "", candidate)

    if target_cjk:
        if src_cjk >= 12 and len(compact_cand) <= max(2, int(len(compact_src) * 0.30)):
            return True
        if src_cjk < 12 and src_words >= 6 and len(compact_cand) <= max(2, int(src_words * 0.60)):
            return True
    else:
        cand_words = _count_non_cjk_words(candidate)
        if src_cjk >= 10 and cand_words <= max(2, int(len(compact_src) * 0.25)):
            return True
        if src_cjk < 10 and src_words >= 6 and cand_words <= max(2, int(src_words * 0.35)):
            return True

    for raw_source_term, raw_target_term in (glossary or {}).items():
        source_term = raw_source_term.strip()
        target_term = raw_target_term.strip()
        if not source_term or not target_term:
            continue
        if _ASCII_GLOSSARY_TERM_RE.fullmatch(source_term):
            matched = bool(
                re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(source_term)}(?![A-Za-z0-9])",
                    source,
                    flags=re.IGNORECASE,
                )
            )
        else:
            matched = source_term.casefold() in source.casefold()
        if matched and target_term.casefold() not in candidate.casefold():
            return True
    return False


def _repair_entries(
    client: OpenAI,
    context: DialogueContext,
    suspicious: list[tuple[Segment, str]],
) -> dict[int, str]:
    if not suspicious:
        return {}
    payload = context.payload([
        {
            **context.entry(seg),
            "previous_target_text": text,
            "repair_reason": "疑似摘要化或漏译",
        }
        for seg, text in suspicious
    ])
    try:
        entries = complete_dialogue(client, TRANSLATION_REPAIR_PROMPT, payload)
    except (OpenAIError, ValueError, IndexError, json.JSONDecodeError):
        return {}
    repaired: dict[int, str] = {}
    for entry in entries:
        seg_id, text = entry.get("id"), entry.get("target_text")
        if type(seg_id) is int and isinstance(text, str) and text.strip():
            repaired[seg_id] = text.strip()
    return repaired


def _is_non_speech(text: str) -> bool:
    """Only genuine interjections/laughter deserve keep_original=true.

    A full sentence that merely looks short or colloquial must still be dubbed;
    otherwise a normal line silently keeps the source audio.
    """
    stripped = re.sub(r"[\s\W_]+", "", text or "")
    if not stripped:
        return True
    if len(stripped) > 8:
        return False
    return not re.search(r"[A-Za-z\u3040-\u30ff\u4e00-\u9fff]{4,}", stripped)


def _apply_translation(seg: Segment, text: str, entry: dict) -> bool:
    text = text.strip()
    complete, omitted = _self_check_flags(entry)
    if complete is False or omitted or not text:
        return False
    emotion = entry.get("emotion")
    emotion = emotion if isinstance(emotion, str) and emotion in EMOTIONS else "neutral"
    action = entry.get("action")
    requested_keep = entry.get("keep_original") is True
    # Guard against the model over-marking normal dialogue as "keep original".
    keep_original = requested_keep and _is_non_speech(seg.orig_text)
    needs_resynth = (text, emotion, keep_original) != (seg.target_text, seg.emotion, seg.keep_original)
    seg.target_text = text
    seg.emotion = emotion
    seg.action = action.strip() if isinstance(action, str) and action.strip() else "idle"
    seg.keep_original = keep_original
    if needs_resynth:
        seg.invalidate_audio()
    return True


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

    use_window = len(segments) > WHOLE_DIALOGUE_MAX_SEGMENTS
    if log:
        if use_window:
            log(f"剧本共 {len(segments)} 条台词，启用滑动窗口上下文分批翻译")
        else:
            log(f"调用导演 LLM 通读全文并翻译全部 {len(segments)} 条台词")

    translated: list[int] = []
    pending = {seg.id: seg for seg in segments}
    completed_ids: set[int] = set()
    chunk_size = config.TRANSLATION_CHUNK_SIZE
    for offset in range(0, len(segments), chunk_size):
        batch = segments[offset:offset + chunk_size]
        if use_window:
            win_start = max(0, offset - WINDOW_CONTEXT_MARGIN)
            win_end = min(len(segments), offset + len(batch) + WINDOW_CONTEXT_MARGIN)
            batch_payload = context.whole_dialogue_payload(
                batch,
                full_segments=segments[win_start:win_end],
                dialogue_window=(win_start, win_end, len(segments)),
            )
        else:
            batch_payload = context.whole_dialogue_payload(batch, full_segments=segments)
        if log:
            mode_label = "按滑动窗口上下文翻译" if use_window else "通读全文并翻译"
            log(f"调用导演 LLM {mode_label}片段 #{batch[0].id}..#{batch[-1].id}（{len(batch)} 条）")
        try:
            entries = complete_dialogue(client, DIRECTOR_TRANSLATE_PROMPT, batch_payload)
        except (OpenAIError, ValueError, IndexError, json.JSONDecodeError) as exc:
            if errors is not None:
                errors.append(f"片段 #{batch[0].id}..#{batch[-1].id} 翻译失败：{exc}")
            if log:
                log("本批翻译失败，保留当前台词")
            continue

        suspicious: list[tuple[Segment, str, dict]] = []
        for entry in entries:
            seg_id = entry.get("id")
            text = entry.get("target_text")
            if type(seg_id) is not int or seg_id not in pending or seg_id in completed_ids or not isinstance(text, str) or not text.strip():
                continue
            seg = pending[seg_id]
            complete, omitted = _self_check_flags(entry)
            if not complete or omitted or _is_suspicious(seg, text, script.glossary):
                suspicious.append((seg, text.strip(), entry))
                continue
            if _apply_translation(seg, text, entry):
                completed_ids.add(seg.id)
                translated.append(seg.id)

        if not suspicious:
            continue
        repaired = _repair_entries(
            client,
            context,
            [(seg, old_text) for seg, old_text, _ in suspicious],
        )
        for seg, _old_text, orig_entry in suspicious:
            text = repaired.get(seg.id)
            repaired_entry = {
                "id": seg.id,
                "target_text": text,
                "emotion": orig_entry.get("emotion"),
                "action": orig_entry.get("action"),
                "keep_original": orig_entry.get("keep_original"),
            }
            if text and seg.id not in completed_ids and _apply_translation(seg, text, repaired_entry):
                completed_ids.add(seg.id)
                translated.append(seg.id)
            elif errors is not None:
                errors.append(f"片段 #{seg.id} 译文疑似摘要化，已保留上一版")

    return sorted(set(translated))
