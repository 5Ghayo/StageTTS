"""Offline semantic boundary candidates; all times come from existing alignment."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.metadata import version
import math
import re
from typing import Callable, Protocol, Sequence

TERMINAL = re.compile(r'''[.!?。！？…]["'”’）)\]]*$''')
PUNCTUATION = re.compile(r'''[.!?;,:。！？；，、：…]["'”’）)\]]*$''')


@dataclass(frozen=True)
class LexicalToken:
    """Offsets address the exact original Python string, never normalized text."""
    text: str
    start: int
    end: int


class TokenizerAdapter(Protocol):
    name: str
    version: str

    def tokenize(self, text: str) -> Sequence[LexicalToken]: ...


@dataclass(frozen=True)
class BoundaryCandidate:
    id: str
    word_index: int
    char_offset: int
    left_end: float
    right_start: float
    reasons: tuple[str, ...]
    tokenizer: str | None = None
    tokenizer_version: str | None = None

    @property
    def pause_seconds(self) -> float:
        return self.right_start - self.left_end


@dataclass(frozen=True)
class BoundaryPlan:
    candidates: tuple[BoundaryCandidate, ...]
    review_reasons: tuple[str, ...] = ()


BoundarySelector = Callable[[tuple[BoundaryCandidate, ...]], Sequence[str]]


class JiebaTokenizer:
    name = "jieba"

    def __init__(self):
        import jieba
        self.version = version("jieba")
        self._tokenizer = jieba.Tokenizer()

    def tokenize(self, text: str) -> Sequence[LexicalToken]:
        return [LexicalToken(token, start, end)
                for token, start, end in self._tokenizer.tokenize(text)]


class SudachiTokenizer:
    name = "sudachipy/core/C"

    def __init__(self):
        from sudachipy import dictionary, tokenizer
        self.version = f"{version('sudachipy')}/{version('sudachidict-core')}"
        self._tokenizer = dictionary.Dictionary(dict="core").create()
        self._mode = tokenizer.Tokenizer.SplitMode.C

    def tokenize(self, text: str) -> Sequence[LexicalToken]:
        return [LexicalToken(item.surface(), item.begin(), item.end())
                for item in self._tokenizer.tokenize(text, self._mode)]


@lru_cache(maxsize=2)
def load_tokenizer(language: str) -> TokenizerAdapter:
    """Only reads installed packages/dictionaries. Never installs or downloads."""
    if language == "zh":
        return JiebaTokenizer()
    if language == "ja":
        return SudachiTokenizer()
    raise ValueError(f"unsupported_tokenizer_language:{language}")


def _compact(text: str) -> str:
    return "".join(text.split())


def _lexical_ends(text: str, tokens: Sequence[LexicalToken]) -> set[int]:
    cursor = 0
    ends = set()
    for token in tokens:
        if (type(token.start) is not int or type(token.end) is not int
                or not 0 <= token.start < token.end <= len(text)
                or token.start < cursor or text[token.start:token.end] != token.text
                or _compact(text[cursor:token.start])):
            raise ValueError("tokenizer_offset_mismatch")
        cursor = token.end
        ends.add(len(_compact(text[:cursor])))
    if _compact(text[cursor:]) or not tokens:
        raise ValueError("tokenizer_incomplete_coverage")
    return ends


def _valid_time(value: object) -> bool:
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= 0)


def generate_boundary_plan(
    text: str, words: Sequence[dict], *, language: str | None = None,
    tokenizer: TokenizerAdapter | None = None, pause_seconds: float = 0.65,
) -> BoundaryPlan:
    """Map lexical ends onto exact aligned-word boundaries without guessing time.

    Missing/invalid CJK tokenization permits sentence punctuation only. With a
    valid tokenizer, a pause inside a lexical token is never a cut candidate.
    With no language hint, retain the historical punctuation/pause behavior.
    """
    compact = _compact(text)
    word_ends = []
    cursor = 0
    for word in words:
        token = _compact(str(word.get("word") or ""))
        if not token or not compact.startswith(token, cursor):
            return BoundaryPlan((), ("alignment_text_mismatch",))
        cursor += len(token)
        word_ends.append(cursor)
    if cursor != len(compact):
        return BoundaryPlan((), ("alignment_text_mismatch",))
    lang = (language or "").lower().replace("_", "-").split("-")[0]
    reasons = []
    lexical = None
    if tokenizer is None and lang in {"ja", "zh"}:
        try:
            tokenizer = load_tokenizer(lang)
        except (ImportError, OSError, RuntimeError, ValueError, LookupError) as exc:
            reasons.append(f"tokenizer_unavailable:{lang}:{type(exc).__name__}")
    if tokenizer is not None:
        try:
            lexical = _lexical_ends(text, tokenizer.tokenize(text))
        except Exception as exc:
            reasons.append(f"tokenizer_invalid_offsets:{type(exc).__name__}")
    chars = [offset for offset, char in enumerate(text) if not char.isspace()]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    candidates = []
    for i in range(1, len(words)):
        end, start = words[i - 1].get("end"), words[i].get("start")
        if not (_valid_time(end) and _valid_time(start) and end <= start):
            continue
        char_offset = chars[word_ends[i - 1]]
        prefix = text[:char_offset].rstrip()
        terminal = bool(TERMINAL.search(prefix))
        punctuation = bool(PUNCTUATION.search(prefix))
        pause = start - end >= pause_seconds
        token_end = lexical is not None and word_ends[i - 1] in lexical
        if lexical is not None and not token_end:
            continue
        # For CJK without trustworthy lexical offsets, pause-only splitting can
        # still divide a compound. Preserve it for review instead.
        conservative = lexical is None and (lang in {"ja", "zh"} or bool(reasons))
        if not (terminal or (punctuation and pause) or (pause and not conservative)):
            continue
        boundary_reasons = []
        if terminal:
            boundary_reasons.append("terminal_punctuation")
        elif punctuation:
            boundary_reasons.append("punctuation")
        if pause:
            boundary_reasons.append("aligned_pause")
        if token_end:
            boundary_reasons.append("lexical_token_end")
        candidates.append(BoundaryCandidate(
            id=f"boundary:{digest}:{char_offset}:{float(end):.6f}:{float(start):.6f}", word_index=i,
            char_offset=char_offset, left_end=float(end), right_start=float(start),
            reasons=tuple(boundary_reasons),
            tokenizer=tokenizer.name if lexical is not None else None,
            tokenizer_version=tokenizer.version if lexical is not None else None,
        ))
    return BoundaryPlan(tuple(candidates), tuple(reasons))


def rank_candidate_ids(
    candidates: tuple[BoundaryCandidate, ...], selector: BoundarySelector | None,
) -> tuple[tuple[BoundaryCandidate, ...], tuple[str, ...]]:
    """An optional LLM/callback can reorder only the supplied immutable candidates."""
    if selector is None:
        return candidates, ()
    try:
        ids = list(selector(candidates))
        by_id = {candidate.id: candidate for candidate in candidates}
        if (any(not isinstance(item, str) or item not in by_id for item in ids)
                or len(ids) != len(set(ids))):
            return candidates, ("boundary_selector_invalid_ids",)
        ranked = tuple(by_id[item] for item in ids)
        selected = set(ids)
        return ranked + tuple(c for c in candidates if c.id not in selected), ()
    except Exception:
        return candidates, ("boundary_selector_failed",)
