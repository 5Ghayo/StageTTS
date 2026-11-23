"""Build immutable source evidence separately from editable dubbing segments.

Overlap seconds are geometric measurements, never speaker probabilities. This
module does not load models or infer missing speaker identities.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping

from backend.schemas import (
    DiarizationSpan, EvidenceProvenance, EvidenceReview, PerformanceUnit, Segment,
    SourceEvidence, SourceToken, SourceTurn, SourceUtterance, SpeakerCandidate, TranslationUnit,
)


def _id(kind: str, source_id: str, *parts: object) -> str:
    value = json.dumps([source_id, *parts], ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return f'{kind}_{hashlib.sha256(value.encode()).hexdigest()[:24]}'


def _time(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def _review(reasons: list[str], provenance: EvidenceProvenance | None = None,
            score: float | None = None, score_kind: str | None = None) -> EvidenceReview:
    return EvidenceReview(needs_review=bool(reasons), reasons=list(dict.fromkeys(reasons)),
                          score=score, score_kind=score_kind, provenance=provenance)


def _spans(rows: Iterable[Mapping[str, object]], source_id: str, *, derived: bool,
           provenance: EvidenceProvenance | None, reasons: list[str]) -> list[DiarizationSpan]:
    normalized = []
    for row in rows:
        start, end = _time(row.get('start')), _time(row.get('end'))
        speaker = str(row.get('speaker') or '').strip()
        if start is None or end is None or end <= start or not speaker or speaker == 'UNKNOWN':
            reasons.append('invalid_exclusive_span' if derived else 'invalid_diarization_span')
            continue
        row_reasons = [reason for reason in row.get('review_reasons', [])
                       if isinstance(reason, str) and reason.strip()]
        if provenance and provenance.parameters.get('activity_kind') == 'joint_generated_derived_timestamps':
            row_reasons.append('moss_joint_generated_activity_unvalidated')
        normalized.append((start, end, speaker, tuple(sorted(set(row_reasons)))))
    # Sorting makes ids independent of dataframe row order. Duplicate activity is
    # retained with an occurrence index rather than silently removed.
    result = []
    occurrences: dict[tuple[float, float, str], int] = {}
    for start, end, speaker, row_reasons in sorted(normalized):
        key = start, end, speaker
        occurrence = occurrences.get(key, 0)
        occurrences[key] = occurrence + 1
        result.append(DiarizationSpan(
            id=_id('exclusive' if derived else 'activity', source_id, *key, occurrence),
            start=start, end=end, speaker=speaker,
            review=_review([*row_reasons, *(['derived_exclusive_track'] if derived else [])], provenance),
        ))
    return result


def build_source_evidence(
    raw_segments: list[dict],
    diarization_spans: Iterable[Mapping[str, object]] | None,
    segments: list[Segment],
    *,
    source_id: str,
    provenance: list[EvidenceProvenance] | None = None,
    fallback_reason: str | None = None,
    exclusive_spans: Iterable[Mapping[str, object]] | None = None,
    raw_generation: str | None = None,
    speaker_scope: str | None = None,
) -> SourceEvidence:
    """Snapshot real source ASR/diarizer outputs and link legacy segments in place.

    Call with a deepcopy of aligned ASR *before* WhisperX speaker assignment.
    ``source_id`` must identify immutable source media (prefer a content hash).
    Empty/missing diarization is unknown even if ASR dictionaries have speaker
    labels. Existing target_text/emotion/audio never enters source token/turn ids.
    Raw unaligned text is retained with unknown token timing rather than guessed.
    """
    if not source_id.strip():
        raise ValueError('source_id must identify the source artifact')
    provenance = provenance or []
    diar_provenance = next((item for item in provenance if item.stage == 'diarization'), None)
    asr_provenance = next((item for item in provenance if item.stage in {'alignment', 'asr'}), None)
    reasons: list[str] = [fallback_reason] if fallback_reason else []
    spans = _spans(diarization_spans or [], source_id, derived=False,
                   provenance=diar_provenance, reasons=reasons)
    derived = _spans(exclusive_spans or [], source_id, derived=True,
                     provenance=diar_provenance, reasons=reasons)
    if not spans:
        reasons.append(fallback_reason or 'diarization_evidence_missing')
    tokens: list[SourceToken] = []
    utterances: list[SourceUtterance] = []
    raw_ranges: list[tuple[float, float, list[str]]] = []
    for raw_index, raw in enumerate(raw_segments):
        raw_start, raw_end = _time(raw.get('start')), _time(raw.get('end'))
        words = raw.get('words') or []
        if not words and str(raw.get('text') or '').strip():
            words = [{'word': str(raw['text'])}]
        row_tokens = []
        previous_end = raw_start
        for word_index, word in enumerate(words):
            text = str(word.get('word') or '')
            start, end = _time(word.get('start')), _time(word.get('end'))
            token_reasons = [fallback_reason] if fallback_reason else []
            if (start is None or end is None or end <= start
                    or raw_start is None or raw_end is None
                    or not raw_start <= start < end <= raw_end):
                start = end = None
                token_reasons.append('alignment_missing_or_invalid')
            elif previous_end is not None and start < previous_end:
                token_reasons.append('overlapping_or_out_of_order_alignment')
            if end is not None:
                previous_end = end
            overlaps: dict[str, float] = {}
            if start is not None and end is not None:
                # Union per speaker avoids double-counting duplicate same-speaker spans.
                intervals: dict[str, list[tuple[float, float]]] = {}
                for span in spans:
                    left, right = max(start, span.start), min(end, span.end)
                    if right > left:
                        intervals.setdefault(span.speaker, []).append((left, right))
                for speaker, activity in intervals.items():
                    merged = []
                    for left, right in sorted(activity):
                        if merged and left <= merged[-1][1]:
                            merged[-1] = (merged[-1][0], max(right, merged[-1][1]))
                        else:
                            merged.append((left, right))
                    overlaps[speaker] = sum(right - left for left, right in merged)
            if not overlaps:
                token_reasons.append(fallback_reason or 'speaker_evidence_missing')
            elif len(overlaps) > 1:
                token_reasons.append('multiple_speaker_activity')
            elif next(iter(overlaps.values())) < end - start - 0.000001:
                token_reasons.append('partial_speaker_coverage')
            score = word.get('score')
            score = float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score) else None
            token = SourceToken(
                id=_id('token', source_id, raw_index, word_index, text, start, end),
                text=text, start=start, end=end,
                speaker_candidates=[SpeakerCandidate(speaker=speaker, overlap_seconds=seconds)
                                    for speaker, seconds in sorted(overlaps.items())],
                review=_review(token_reasons, asr_provenance, score,
                               'alignment_model_score' if score is not None else None),
            )
            tokens.append(token)
            row_tokens.append(token.id)
        original_text = str(raw.get('text') or '')
        token_text = ''.join(token.text for token in tokens if token.id in row_tokens)
        utterance_reasons = [str(reason) for reason in raw.get('review_reasons', [])
                             if isinstance(reason, str) and reason.strip()]
        def generated_time(value: object) -> float | None:
            return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None
        generated_start = generated_time(raw.get('generated_start'))
        generated_end = generated_time(raw.get('generated_end'))
        if ''.join(original_text.split()) != ''.join(token_text.split()):
            utterance_reasons.append('asr_text_token_mismatch')
        utterance_start, utterance_end = raw_start, raw_end
        if raw_start is None or raw_end is None or raw_end <= raw_start:
            utterance_reasons.append('source_utterance_timing_missing')
            utterance_start = utterance_end = None
        utterances.append(SourceUtterance(
            id=_id('utterance', source_id, raw_index, original_text, raw_start, raw_end),
            text=original_text, start=utterance_start, end=utterance_end,
            generated_start=generated_start, generated_end=generated_end,
            source_token_ids=row_tokens, review=_review(utterance_reasons, asr_provenance),
        ))
        if raw_start is not None and raw_end is not None and raw_end > raw_start:
            raw_ranges.append((raw_start, raw_end, row_tokens))
        elif row_tokens:
            reasons.append('source_turn_timing_missing')

    turns = [SourceTurn(
        id=_id('turn', source_id, span.id), start=span.start, end=span.end,
        speaker=span.speaker, diarization_span_ids=[span.id],
        source_token_ids=[token.id for token in tokens if token.start is not None
                          and token.start < span.end and token.end > span.start],
        review=span.review.model_copy(deep=True),
    ) for span in spans]
    for start, end, token_ids in raw_ranges:
        missing = [token.id for token in tokens if token.id in token_ids and not token.speaker_candidates]
        if missing:
            turns.append(SourceTurn(
                id=_id('turn_unknown', source_id, start, end, missing), start=start, end=end,
                speaker=None, source_token_ids=missing,
                review=_review([fallback_reason or 'speaker_evidence_missing'], diar_provenance),
            ))
    units = []
    performances = []
    for seg in segments:
        linked_tokens = [token for token in tokens if token.start is not None
                         and token.start < seg.end and token.end > seg.start]
        token_ids = [token.id for token in linked_tokens]
        # Segment-level ASR may have exact utterance boundaries but no word
        # timestamps. Link the immutable text evidence without fabricating times.
        exact_utterances = [utterance for utterance in utterances
                            if utterance.start == seg.start and utterance.end == seg.end
                            and ''.join(utterance.text.split()) == ''.join(seg.orig_text.split())]
        for utterance in exact_utterances:
            token_ids.extend(token_id for token_id in utterance.source_token_ids if token_id not in token_ids)
        turn_ids = [turn.id for turn in turns if turn.start < seg.end and turn.end > seg.start]
        link_reasons = ['unit_boundaries_require_review']
        if not token_ids:
            link_reasons.append('source_token_links_missing')
        # These are initial editable-unit proposals, explicitly distinct from turns.
        unit_id = _id('translation', source_id, seg.id, token_ids, turn_ids)
        performance_id = _id('performance', source_id, unit_id)
        units.append(TranslationUnit(id=unit_id, source_turn_ids=turn_ids, source_token_ids=token_ids,
                                     segment_ids=[seg.id], source_text=seg.orig_text,
                                     review=_review(link_reasons)))
        performances.append(PerformanceUnit(id=performance_id, translation_unit_ids=[unit_id],
                                            source_turn_ids=turn_ids, segment_ids=[seg.id],
                                            review=_review(link_reasons)))
        seg.source_token_ids = token_ids
        seg.source_turn_ids = turn_ids
        seg.translation_unit_ids = [unit_id]
        seg.performance_unit_ids = [performance_id]
    return SourceEvidence(source_id=source_id, raw_generation=raw_generation, speaker_scope=speaker_scope,
                          provenance=provenance, source_utterances=utterances, source_tokens=tokens,
                          diarization_spans=spans, exclusive_spans=derived, source_turns=turns,
                          translation_units=units, performance_units=performances,
                          review_reasons=list(dict.fromkeys(reasons)))
