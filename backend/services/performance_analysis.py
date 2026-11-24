"""Explicit, offline source performance analysis; legacy TTS routing stays unchanged."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from backend import config
from backend.schemas import EvidenceLabel, EvidenceProvenance, EvidenceReview, ProjectScript, ProsodyMeasurement, SourceAnnotation
from backend.services.engine_capabilities import all_capabilities
from backend.services.references import _verified_word_offsets

Status = Literal['ok', 'partial', 'not_run', 'not_configured', 'blocked', 'error']


class StageReport(BaseModel):
    status: Status = 'not_run'
    model: dict = Field(default_factory=dict)
    annotations: list[dict] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class PerformanceAnalysisReport(BaseModel):
    version: Literal['1.0'] = '1.0'
    project_id: str
    status: Status = 'not_configured'
    selected_segment_ids: list[int] = Field(default_factory=list)
    audio_path: str | None = None
    emotion: StageReport = Field(default_factory=StageReport)
    prosody: StageReport = Field(default_factory=StageReport)
    engine_capabilities: list[dict] = Field(default_factory=all_capabilities)
    review_reasons: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


def _subprocess_runner(python: str, worker: Path, payload: dict) -> dict:
    process = subprocess.run([python, '-I', '-B', str(worker)], input=json.dumps(payload), text=True,
                             capture_output=True, timeout=300, check=False, encoding='utf-8')
    if process.returncode:
        raise ValueError(f'worker exited {process.returncode}: {process.stderr[-1000:]}')
    return json.loads(process.stdout)


def _validate_stage(value: dict, stage: str, spans: list[dict], *,
                    expected_source: dict | None = None, expected_model: dict | None = None) -> StageReport:
    if not isinstance(value, dict) or value.get('version') != '1.0' or value.get('stage') != stage:
        raise ValueError(f'invalid {stage} worker version/stage')
    if expected_source is not None:
        returned_source = value.get('source') or {}
        if any(returned_source.get(key) != expected_source.get(key) for key in ('id', 'kind', 'audio_path')):
            raise ValueError('worker changed source identity')
    if expected_model is not None:
        returned_model = value.get('model') or {}
        if any(returned_model.get(key) != expected_model.get(key) for key in ('id', 'revision')):
            raise ValueError('worker changed model identity')
    status = value.get('status')
    if status not in {'ok', 'partial', 'not_run', 'not_configured', 'blocked', 'error'}:
        raise ValueError('invalid worker status')
    allowed = {item['id']: item for item in spans}
    seen: set[str] = set()
    annotations = value.get('annotations', [])
    if not isinstance(annotations, list):
        raise ValueError('worker annotations must be a list')
    for annotation in annotations:
        if not isinstance(annotation, dict):
            raise ValueError('worker annotation must be an object')
        annotation_id = annotation.get('id')
        if annotation_id not in allowed or annotation_id in seen:
            raise ValueError('unexpected/duplicate worker annotation id')
        seen.add(annotation_id)
        source = allowed[annotation_id]
        for key in ('start', 'end'):
            val = annotation.get(key)
            if not isinstance(val, (int, float)) or not math.isfinite(val) or abs(val - source[key]) > 1e-6:
                raise ValueError('worker changed source range')
        for key in ('source_token_ids', 'source_turn_ids'):
            if annotation.get(key, []) != source[key]:
                raise ValueError('worker changed source links')
        if annotation.get('status') not in {'ok', 'partial', 'not_run', 'not_configured', 'blocked', 'error'}:
            raise ValueError('invalid annotation status')
        if source['blocked_reasons'] and annotation.get('status') in {'ok', 'partial'}:
            raise ValueError('worker analyzed blocked source')
        if stage == 'emotion':
            labels, scores = annotation.get('labels', []), annotation.get('scores', [])
            if not isinstance(labels, list) or not isinstance(scores, list):
                raise ValueError('emotion labels/scores must be arrays')
            if len(labels) != len(scores) or any(not isinstance(label, str) for label in labels):
                raise ValueError('invalid emotion labels/scores')
            if any(isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) for score in scores):
                raise ValueError('nonfinite emotion scores')
            if annotation['status'] in {'ok', 'partial'} and not labels:
                raise ValueError('successful emotion requires predictions')
            if labels and annotation.get('score_kind') != 'model_softmax_uncalibrated':
                raise ValueError('unsupported/claimed calibrated emotion score kind')
            if annotation['status'] not in {'ok', 'partial'} and (labels or scores):
                raise ValueError('unrun emotion annotation contains predictions')
        elif stage == 'prosody':
            for metric in annotation.get('measurements', []):
                val = metric.get('value')
                if val is not None and (isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val)):
                    raise ValueError('nonfinite prosody measurement')
                if annotation['status'] in {'not_run', 'not_configured', 'error'} and val is not None:
                    raise ValueError('unrun prosody annotation contains measurements')
    missing = set(allowed) - seen
    reasons = list(value.get('review_reasons') or [])
    if missing:
        reasons.append('worker_missing_annotations:' + ','.join(sorted(missing)))
        if status == 'ok':
            status = 'partial'
    return StageReport(status=status, model=value.get('model') or {}, annotations=annotations,
                       review_reasons=reasons, errors=value.get('errors') or [])


def _spans(script: ProjectScript, selected: list) -> list[dict]:
    spans = []
    evidence = script.evidence
    tokens = {item.id: item for item in evidence.source_tokens} if evidence else {}
    turns = {item.id: item for item in evidence.source_turns} if evidence else {}
    compact = lambda text: ''.join(text.split())
    for seg in selected:
        reasons = []
        segment_confirmed = not getattr(seg, 'segment_timing_needs_review', True)
        word_confirmed = not seg.alignment_needs_review and _verified_word_offsets(seg) is not None
        if seg.speaker == 'UNKNOWN' or not seg.speaker or seg.speaker_needs_review:
            reasons.append('source_speaker_unconfirmed')
        if not (segment_confirmed or word_confirmed):
            reasons.append('source_alignment_unconfirmed')
        timed_count = 0
        if not evidence:
            reasons.append('source_evidence_missing')
        else:
            if not seg.source_turn_ids or not seg.source_token_ids:
                reasons.append('source_links_missing')
            if segment_confirmed:
                # Manual whole-segment review never invents word alignment. It
                # must match an immutable complete source utterance/range.
                matched = [utterance for utterance in evidence.source_utterances
                           if utterance.start is not None and utterance.end is not None
                           and abs(utterance.start - seg.start) <= 1e-6
                           and abs(utterance.end - seg.end) <= 1e-6
                           and compact(utterance.text) == compact(seg.orig_text)]
                if not matched:
                    reasons.append('confirmed_segment_source_utterance_mismatch')
                elif not any(set(seg.source_token_ids) == set(utterance.source_token_ids) for utterance in matched):
                    reasons.append('confirmed_segment_source_token_mismatch')
                if any(token_id not in tokens for token_id in seg.source_token_ids):
                    reasons.append('source_token_links_missing')
            else:
                for token_id in seg.source_token_ids:
                    token = tokens.get(token_id)
                    if token is None or token.review.needs_review:
                        reasons.append('source_token_requires_review')
            for token_id in seg.source_token_ids:
                token = tokens.get(token_id)
                if (token is not None and token.start is not None and token.end is not None
                        and token.start >= seg.start and token.end <= seg.end
                        and not token.review.needs_review):
                    timed_count += 1
            if segment_confirmed:
                coverage = sorted((max(seg.start, span.start), min(seg.end, span.end))
                                  for span in evidence.diarization_spans if span.speaker == seg.speaker
                                  and span.start < seg.end and span.end > seg.start)
                cursor = seg.start
                for left, right in coverage:
                    if left > cursor + 1e-6:
                        break
                    cursor = max(cursor, right)
                if cursor < seg.end - 1e-6:
                    reasons.append('confirmed_segment_speaker_activity_incomplete')
            for turn_id in seg.source_turn_ids:
                turn = turns.get(turn_id)
                if (turn is None or (turn.speaker != seg.speaker and not (segment_confirmed and turn.speaker is None))
                        or (turn.review.needs_review and not segment_confirmed)
                        or (segment_confirmed and not (turn.start < seg.end and turn.end > seg.start))):
                    reasons.append('source_turn_requires_review')
            if any(span.speaker != seg.speaker and span.start < seg.end and span.end > seg.start
                   for span in evidence.diarization_spans):
                reasons.append('source_overlap')
        spans.append({
            'id': f'segment_{seg.id}', 'start': seg.start, 'end': seg.end, 'speaker': seg.speaker,
            'source_token_ids': list(seg.source_token_ids), 'source_turn_ids': list(seg.source_turn_ids),
            'token_count': timed_count, 'speaker_needs_review': seg.speaker_needs_review,
            # This describes eligibility for whole-span analysis, never edits the
            # original Segment.alignment_needs_review or missing token timestamps.
            'alignment_needs_review': not (segment_confirmed or word_confirmed),
            'source_word_alignment_needs_review': seg.alignment_needs_review,
            'alignment_granularity': getattr(seg, 'alignment_granularity', 'unknown'),
            'segment_timing_needs_review': not segment_confirmed,
            'blocked_reasons': list(dict.fromkeys(reasons)),
            'overlap': 'source_overlap' in reasons,
        })
    return spans


def analyze_project_performance(
    script: ProjectScript, *, segment_ids: list[int] | None = None,
    audio_path: str | Path | None = None, enable_emotion: bool = False, enable_prosody: bool = True,
    enable_f0: bool = True, emotion_device: str = 'cuda:0',
    emotion_python: str | None = None, emotion_worker_path: str | Path | None = None,
    emotion_model_path: str | None = None, emotion_model_id: str | None = None,
    emotion_revision: str | None = None, emotion_funasr_version: str | None = None,
    emotion_user_authorized: bool = False, prosody_python: str | None = None,
    prosody_worker_path: str | Path | None = None,
    emotion_runner: Callable | None = None, prosody_runner: Callable | None = None,
) -> PerformanceAnalysisReport:
    """Run configured stages sequentially. Caller owns project/GPU locks and persistence."""
    available = {seg.id for seg in script.segments}
    if segment_ids is not None and not set(segment_ids) <= available:
        raise ValueError('unknown segment id')
    selected = [seg for seg in script.segments if segment_ids is None or seg.id in segment_ids]
    report = PerformanceAnalysisReport(project_id=script.project_id,
                                       selected_segment_ids=[seg.id for seg in selected])
    if audio_path is None and script.media.vocals_wav:
        audio_path = config.resolve_media_path(script.media.vocals_wav)
    if audio_path is None or not Path(audio_path).is_file():
        report.review_reasons.append('source_audio_missing')
        return report
    audio = Path(audio_path).resolve()
    report.audio_path = str(audio)
    source_id = script.evidence.source_id if script.evidence else 'unlinked_source'
    spans = _spans(script, selected)
    source = {'id': source_id, 'kind': 'dialogue', 'audio_path': str(audio)}
    root = Path(__file__).resolve().parents[2]
    if enable_prosody:
        try:
            payload = {'version': '1.0', 'source': source, 'spans': spans, 'enable_f0': enable_f0}
            result = (prosody_runner or _subprocess_runner)(
                prosody_python or sys.executable, Path(prosody_worker_path or root / 'workers/prosody_worker.py'), payload)
            report.prosody = _validate_stage(result, 'prosody', spans, expected_source=source,
                                             expected_model={'id': 'pcm_rms_energy_gate_v1', 'revision': '1.0'})
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            report.prosody = StageReport(status='error', errors=[str(exc)])
    if enable_emotion:
        eligible = [span for span in spans if not span['blocked_reasons']]
        blocked = [{**{key: span[key] for key in ('id', 'start', 'end', 'speaker', 'source_token_ids', 'source_turn_ids')},
                    'status': 'blocked', 'labels': [], 'scores': [],
                    'review_reasons': span['blocked_reasons'], 'score_kind': None} for span in spans if span['blocked_reasons']]
        if not eligible:
            report.emotion = StageReport(status='blocked', annotations=blocked,
                                         review_reasons=['no_eligible_source_spans'])
        elif not all((emotion_python, emotion_model_path, emotion_model_id, emotion_revision,
                    emotion_funasr_version, emotion_user_authorized)):
            report.emotion = StageReport(status='not_configured', review_reasons=['emotion_local_model_not_configured_or_authorized'])
        else:
            try:
                # Conversion is explicit, temporary, time-preserving, and never edits the source.
                with tempfile.TemporaryDirectory(prefix='stagetts_analysis_') as temp:
                    converted = Path(temp) / 'source_16k_mono.wav'
                    command = ['ffmpeg', '-v', 'error', '-nostdin', '-y', '-i', str(audio), '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(converted)]
                    subprocess.run(command, check=True, capture_output=True, timeout=120)
                    with wave.open(str(audio), 'rb') as original:
                        original_info = {'sample_rate': original.getframerate(), 'channels': original.getnchannels(), 'sample_width': original.getsampwidth()}
                    payload = {
                        'version': '1.0', 'source': {**source, 'audio_path': str(converted)}, 'spans': eligible,
                        'model': {'id': emotion_model_id, 'revision': emotion_revision,
                                  'local_path': emotion_model_path, 'funasr_version': emotion_funasr_version,
                                  'license_approved': emotion_user_authorized, 'device': emotion_device},
                    }
                    result = (emotion_runner or _subprocess_runner)(
                        emotion_python, Path(emotion_worker_path or root / 'workers/emotion_worker.py'), payload)
                    report.emotion = _validate_stage(result, 'emotion', eligible,
                                                     expected_source=payload['source'], expected_model=payload['model'])
                    report.emotion.annotations.extend(blocked)
                    if blocked and report.emotion.status == 'ok':
                        report.emotion.status = 'partial'
                    report.emotion.model['preprocessing'] = {'method': 'ffmpeg_pcm16_mono_16k', 'original': original_info,
                                                             'source_audio_path': str(audio), 'absolute_time_preserved': True}
            except (ValueError, OSError, wave.Error, subprocess.SubprocessError) as exc:
                report.emotion = StageReport(status='error', errors=[str(exc)])
    active = [stage for enabled, stage in ((enable_emotion, report.emotion), (enable_prosody, report.prosody)) if enabled]
    if not active:
        report.status = 'not_run'
    elif all(stage.status == 'ok' for stage in active):
        report.status = 'ok'
    elif all(stage.status in {'not_configured', 'not_run', 'blocked'} for stage in active):
        report.status = 'not_configured' if any(stage.status == 'not_configured' for stage in active) else 'blocked'
    elif all(stage.status == 'error' for stage in active):
        report.status = 'error'
    else:
        report.status = 'partial'
    return report


def apply_analysis_to_evidence(script: ProjectScript, report: PerformanceAnalysisReport) -> list[SourceAnnotation]:
    """Return validated rich annotations only; no mutation, legacy emotion5 stays untouched."""
    if script.evidence is None:
        return []
    annotations = []
    for stage_name in ('emotion', 'prosody'):
        stage = getattr(report, stage_name)
        for raw in stage.annotations:
            identity = [script.evidence.source_id, stage_name, raw['start'], raw['end'],
                        raw.get('source_token_ids') or [], raw.get('source_turn_ids') or []]
            annotation_id = 'analysis:' + stage_name + ':' + hashlib.sha256(
                json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
            emotions = [EvidenceLabel(label=label, score=score, score_kind=raw.get('score_kind'))
                        for label, score in zip(raw.get('labels', []), raw.get('scores', []))]
            prosody = [ProsodyMeasurement(name=metric['name'], value=metric.get('value'), unit=metric.get('unit'),
                                           status=metric.get('status'), reason=metric.get('reason'))
                       for metric in raw.get('measurements', [])]
            model = {key: str(value) for key, value in stage.model.items() if value is not None and not isinstance(value, dict)}
            parameters: dict[str, str | int | float | bool | None] = {}
            processing = stage.model.get('preprocessing') or {}
            if isinstance(processing, dict):
                parameters['preprocessing_method'] = processing.get('method')
                parameters['source_audio_path'] = processing.get('source_audio_path')
                parameters['absolute_time_preserved'] = processing.get('absolute_time_preserved')
                for key, value in (processing.get('original') or {}).items():
                    parameters['original_' + key] = value
            raw_provenance = raw.get('provenance') or {}
            if isinstance(raw_provenance, dict):
                for key in ('method', 'source_audio_path', 'frame_sec', 'energy_gate',
                            'activity_is_speech_detection', 'normalization', 'normalization_sample_count',
                            'f0_method', 'f0_voiced_probability_threshold', 'f0_center', 'f0_frame_length',
                            'alignment_granularity', 'source_word_alignment_needs_review', 'segment_timing_needs_review'):
                    if key in raw_provenance:
                        parameters[key] = raw_provenance[key]
                for key, value in (raw_provenance.get('wav') or {}).items():
                    parameters['wav_' + key] = value
            review_reasons = list(raw.get('review_reasons') or [])
            if raw.get('reliability') is not None:
                parameters['reliability'] = str(raw['reliability'])
                if raw['reliability'] != 'validated':
                    review_reasons.append('model_reliability:' + str(raw['reliability']))
            provenance = EvidenceProvenance(stage=stage_name, model=model.get('id'),
                                            model_version=model.get('revision'), source_id=script.evidence.source_id,
                                            parameters=parameters)
            for metric in prosody:
                metric.provenance = provenance.model_copy(deep=True)
            annotations.append(SourceAnnotation(
                id=annotation_id, start=raw['start'], end=raw['end'],
                source_token_ids=raw.get('source_token_ids') or [], source_turn_ids=raw.get('source_turn_ids') or [],
                status=raw['status'], model=model, emotions=emotions, prosody=prosody,
                review=EvidenceReview(needs_review=raw.get('needs_review') is True or raw['status'] != 'ok' or bool(review_reasons),
                                      reasons=list(dict.fromkeys(review_reasons)),
                                      provenance=EvidenceProvenance(stage=stage_name, model=model.get('id'),
                                                                    model_version=model.get('revision'), source_id=script.evidence.source_id,
                                                                    parameters=parameters)),
            ))
    candidate = script.evidence.model_copy(deep=True)
    candidate.annotations = annotations
    type(candidate).model_validate(candidate.model_dump())
    return annotations
