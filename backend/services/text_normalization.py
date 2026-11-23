"""Shared TTS-facing text and language normalization helpers."""

from __future__ import annotations

import hashlib
import re
import unicodedata


def normalize_tts_lang(value: str | None, *, default: str = "zh") -> str:
    """Canonicalize language codes accepted by GPT-SoVITS api_v2.py."""
    lang = (value or "").strip().lower().replace("_", "-")
    aliases = {
        "zh-cn": "zh", "zh-hans": "zh", "zh-hant": "zh", "chinese": "zh", "中文": "zh",
        "en-us": "en", "en-gb": "en", "english": "en", "英语": "en",
        "ja-jp": "ja", "japanese": "ja", "日语": "ja",
        "ko-kr": "ko", "korean": "ko", "韩语": "ko",
        "yue-hk": "yue", "cantonese": "yue", "粤语": "yue",
    }
    lang = aliases.get(lang, lang)
    supported = {"zh", "en", "ja", "ko", "yue", "auto", "", "all_zh", "all_en", "all_ja", "all_ko", "all_yue"}
    if lang not in supported:
        return default
    return lang


def prepare_tts_text(text: str, lang: str, glossary: dict[str, str] | None = None) -> str:
    """Create a TTS-facing spoken-form copy without mutating reviewed subtitles."""
    value = unicodedata.normalize("NFKC", text or "")
    value = "".join(ch for ch in value if ch.isprintable() or ch.isspace())
    value = re.sub(r"\s+", " ", value).strip()
    for source, target in (glossary or {}).items():
        if source and source.casefold() in value.casefold():
            value = re.sub(re.escape(source), target or source, value, flags=re.IGNORECASE)
    if normalize_tts_lang(lang) == "zh":
        value = re.sub(r"(?<=[\u4e00-\u9fff])([A-Za-z0-9]+)", r" \1", value)
        value = re.sub(r"([A-Za-z0-9]+)(?=[\u4e00-\u9fff])", r"\1 ", value)
    return value


def deterministic_seed(*parts: object) -> int:
    digest = hashlib.sha1("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)
