#!/usr/bin/env python3
"""Offline WAV energy/activity measurements; optional F0 is never guessed from text.

stdin/stdout protocol: {version:1.0, source:{id,kind,audio_path}, spans:[...]}
A 20ms energy gate measures acoustic activity, not linguistic speech detection.
Token-rate units are source tokens/active second (may be characters, not words).
"""
from __future__ import annotations

import argparse
import array
import contextlib
import importlib.util
import json
import math
import statistics
import sys
import wave
from pathlib import Path
from typing import Callable

VERSION = '1.0'
METHOD = 'pcm_rms_energy_gate_v1'


def read_wav(path: str | Path, *, start: float | None = None,
             end: float | None = None) -> tuple[list[float], int, dict]:
    with wave.open(str(path), 'rb') as audio:
        channels, width, rate = audio.getnchannels(), audio.getsampwidth(), audio.getframerate()
        if audio.getcomptype() != 'NONE' or width not in (1, 2, 3, 4) or channels < 1 or rate < 1:
            raise ValueError('only integer PCM WAV with 8/16/24/32 bits is supported')
        total_count = audio.getnframes()
        first = 0 if start is None else round(start * rate)
        last = total_count if end is None else round(end * rate)
        if first < 0 or last > total_count or last <= first:
            raise ValueError('invalid/out-of-bounds source range')
        count = last - first
        audio.setpos(first)
        raw = audio.readframes(count)
    if len(raw) != count * width * channels:
        raise ValueError('truncated WAV sample data')
    if width in (2, 4):
        values = array.array('h' if width == 2 else 'i')
        values.frombytes(raw)
        if sys.byteorder != 'little':
            values.byteswap()
        decoded = [value / float(1 << (width * 8 - 1)) for value in values]
    elif width == 1:
        decoded = [(value - 128) / 128.0 for value in raw]
    else:
        decoded = [int.from_bytes(raw[index:index + 3], 'little', signed=True) / 8388608.0
                   for index in range(0, len(raw), 3)]
    mono = [sum(decoded[index:index + channels]) / channels for index in range(0, len(decoded), channels)]
    return mono, rate, {'sample_rate': rate, 'channels': channels, 'sample_width': width,
                        'frames': count, 'downmix': 'arithmetic_mean' if channels > 1 else 'none'}


def _measurement(name: str, value: float | None, unit: str, *, status: str = 'ok', reason: str | None = None) -> dict:
    if value is not None and not math.isfinite(value):
        raise ValueError('measurement must be finite')
    result = {'name': name, 'value': value, 'unit': unit, 'status': status}
    if reason:
        result['reason'] = reason
    return result


def _rms(samples: list[float]) -> float:
    return math.sqrt(sum(value * value for value in samples) / len(samples)) if samples else 0.0


def _optional_f0(samples: list[float], rate: int, active_mask: list[bool], hop: int) -> list[dict]:
    if importlib.util.find_spec('librosa') is None or importlib.util.find_spec('numpy') is None:
        return [_measurement('f0_median_hz', None, 'Hz', status='not_available', reason='optional_librosa_or_numpy_missing')]
    # pYIN supplies an explicit voiced decision. Acoustic energy alone is not
    # voicing: fricatives/noise/music must not be assigned an F0 by YIN.
    frame_length = 2048
    if len(samples) < frame_length:
        return [_measurement('f0_median_hz', None, 'Hz', status='not_available', reason='clip_too_short_for_pyin')]
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np
        import librosa
        signal = np.asarray(samples)
        f0, voiced_flag, voiced_probability = librosa.pyin(
            signal, fmin=60.0, fmax=min(600.0, rate / 2 - 1), sr=rate,
            frame_length=frame_length, hop_length=hop, center=False, fill_na=np.nan,
        )
    energy_floor = max(10 ** (-45 / 20), _rms(samples) * 0.1)
    # pYIN frame i covers [i*hop, i*hop+frame_length), matching center=False.
    voiced = [float(value) for index, value in enumerate(f0)
              if bool(voiced_flag[index]) and float(voiced_probability[index]) >= 0.8
              and math.isfinite(float(value))
              and _rms(samples[index * hop:index * hop + frame_length]) >= energy_floor]
    if not voiced:
        return [_measurement('f0_median_hz', None, 'Hz', status='not_available', reason='no_reliable_pyin_voiced_frames')]
    return [_measurement('f0_median_hz', statistics.median(voiced), 'Hz'),
            _measurement('f0_range_hz', max(voiced) - min(voiced), 'Hz')]


def measure_clip(samples: list[float], rate: int, *, token_count: int = 0,
                 enable_f0: bool = False, f0_provider: Callable | None = None) -> list[dict]:
    if rate <= 0 or not samples:
        raise ValueError('empty clip or invalid sample rate')
    hop = max(1, round(rate * 0.02))
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    peak = max(abs(sample) for sample in samples)
    threshold = max(10 ** (-45 / 20), rms * 0.1)
    frames = [samples[index:index + hop] for index in range(0, len(samples), hop)]
    active = [math.sqrt(sum(value * value for value in frame) / len(frame)) >= threshold for frame in frames]
    active_seconds = sum(len(frame) / rate for frame, flag in zip(frames, active) if flag)
    duration = len(samples) / rate
    longest_silence = current = 0.0
    for frame, flag in zip(frames, active):
        current = 0.0 if flag else current + len(frame) / rate
        longest_silence = max(longest_silence, current)
    values = [
        _measurement('duration_sec', duration, 's'),
        _measurement('rms_amplitude', rms, 'full_scale'),
        _measurement('rms_dbfs', 20 * math.log10(rms) if rms > 0 else None, 'dBFS',
                     status='ok' if rms else 'not_available', reason=None if rms else 'silent_clip'),
        _measurement('peak_amplitude', peak, 'full_scale'),
        _measurement('energy_active_sec', active_seconds, 's'),
        _measurement('energy_active_ratio', active_seconds / duration, 'ratio'),
        _measurement('energy_silent_sec', duration - active_seconds, 's'),
        _measurement('longest_silence_sec', longest_silence, 's'),
        _measurement('source_token_rate_active', token_count / active_seconds if active_seconds and token_count else None,
                     'source_tokens/s', status='ok' if active_seconds and token_count else 'not_available',
                     reason=None if active_seconds and token_count else 'source_token_count_or_activity_missing'),
    ]
    if not enable_f0:
        values.append(_measurement('f0_median_hz', None, 'Hz', status='not_available', reason='f0_not_enabled'))
    else:
        try:
            values.extend((f0_provider or _optional_f0)(samples, rate, active, hop))
        except (ImportError, ValueError, RuntimeError, OSError) as exc:
            values.append(_measurement('f0_median_hz', None, 'Hz', status='not_available', reason=f'f0_estimator_unavailable:{exc}'))
    return values


def _normalize(annotations: list[dict]) -> None:
    """Z scores within one speaker/source report, never invented reference population."""
    for speaker in {item.get('speaker') for item in annotations if item['status'] == 'ok'}:
        group = [item for item in annotations if item['status'] == 'ok' and item.get('speaker') == speaker]
        for metric_name in ('rms_dbfs', 'source_token_rate_active', 'f0_median_hz'):
            measured = [(item, next((metric['value'] for metric in item['measurements'] if metric['name'] == metric_name), None))
                        for item in group]
            valid = [value for _, value in measured if value is not None]
            deviation = statistics.pstdev(valid) if len(valid) >= 2 else 0.0
            for item, value in measured:
                if value is None or deviation <= 1e-12:
                    item['measurements'].append(_measurement(metric_name + '_speaker_z', None, 'z_score',
                        status='not_available', reason='insufficient_speaker_variance_or_samples'))
                else:
                    item['measurements'].append(_measurement(metric_name + '_speaker_z',
                        (value - statistics.mean(valid)) / deviation, 'z_score'))
                item['provenance']['normalization'] = 'within_speaker_current_source_report_population_z'
                item['provenance']['normalization_sample_count'] = len(valid)


def analyze_request(request: dict) -> dict:
    if request.get('version') != VERSION:
        raise ValueError('unsupported prosody request version')
    source = request.get('source') or {}
    with wave.open(str(source['audio_path']), 'rb') as audio:
        duration = audio.getnframes() / audio.getframerate()
    annotations = []
    for span in request.get('spans', []):
        start, end = span.get('start'), span.get('end')
        if (isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float))
                or not isinstance(end, (int, float)) or not math.isfinite(start) or not math.isfinite(end)
                or start < 0 or end <= start or end > duration + 1e-6):
            raise ValueError('invalid/out-of-bounds source range')
        reasons = list(span.get('blocked_reasons') or [])
        if span.get('speaker') in (None, '', 'UNKNOWN') or span.get('speaker_needs_review'):
            reasons.append('source_speaker_unconfirmed')
        if span.get('alignment_needs_review'):
            reasons.append('source_alignment_unconfirmed')
        if span.get('overlap'):
            reasons.append('source_overlap')
        clip, rate, wav_meta = read_wav(source['audio_path'], start=start, end=end)
        # Unlinked old projects may expose raw CPU energy, never trustworthy
        # speaker normalization or automatic emotion inference.
        can_measure = not any(reason != 'source_evidence_missing' for reason in reasons)
        values = measure_clip(clip, rate, token_count=span.get('token_count') or 0,
                              enable_f0=request.get('enable_f0') is True) if can_measure else []
        annotations.append({
            'id': span['id'], 'start': start, 'end': end, 'speaker': span.get('speaker'),
            'source_token_ids': list(span.get('source_token_ids') or []),
            'source_turn_ids': list(span.get('source_turn_ids') or []),
            'status': 'blocked' if reasons else 'ok', 'review_reasons': list(dict.fromkeys(reasons)),
            'measurements': values,
            'provenance': {'stage': 'prosody', 'model': METHOD, 'source_id': source.get('id'),
                           'source_audio_path': source['audio_path'], 'wav': wav_meta,
                           'frame_sec': 0.02, 'energy_gate': 'max(-45dBFS, clip_rms*0.1)',
                           'activity_is_speech_detection': False,
                           'f0_method': 'librosa.pyin_voiced_flag',
                           'f0_voiced_probability_threshold': 0.8, 'f0_center': False,
                           'f0_frame_length': 2048,
                           'alignment_granularity': span.get('alignment_granularity', 'unknown'),
                           'source_word_alignment_needs_review': span.get('source_word_alignment_needs_review'),
                           'segment_timing_needs_review': span.get('segment_timing_needs_review')},
        })
    _normalize(annotations)
    status = 'blocked' if annotations and all(item['status'] == 'blocked' for item in annotations) else (
        'partial' if any(item['status'] != 'ok' for item in annotations) else 'ok')
    return {'version': VERSION, 'stage': 'prosody', 'status': status,
            'model': {'id': METHOD, 'revision': VERSION}, 'source': source, 'annotations': annotations, 'errors': []}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', help='JSON request file; stdin when omitted')
    parser.add_argument('--output', help='JSON result file; stdout when omitted')
    args = parser.parse_args(argv)
    try:
        request = json.loads(Path(args.input).read_text(encoding='utf-8') if args.input else sys.stdin.read())
        report = analyze_request(request)
    except (ValueError, KeyError, TypeError, OSError, wave.Error) as exc:
        report = {'version': VERSION, 'stage': 'prosody', 'status': 'error', 'annotations': [], 'errors': [str(exc)]}
    output = json.dumps(report, ensure_ascii=False, allow_nan=False)
    if args.output:
        Path(args.output).write_text(output, encoding='utf-8')
    else:
        print(output)
    return 0 if report['status'] != 'error' else 1


if __name__ == '__main__':
    raise SystemExit(main())
