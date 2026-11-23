"""Build clip-specific reference transcripts for MOSS segment-only projects."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import wave
from pathlib import Path

from backend import config
from backend.schemas import ProjectScript, ReferencePreview
from backend.services.text_normalization import normalize_tts_lang, prepare_tts_text


def transcribe_reference(audio: Path, directory: Path) -> dict:
    if not config.MOSS_MODEL_PATH or not config.MOSS_CUSTOM_CODE_APPROVED:
        raise ValueError("自动参考音需要已配置并批准的本地 MOSS 模型")
    mono = directory / 'reference_16k.wav'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(audio),
                    '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le', str(mono)],
                   check=True, capture_output=True, timeout=30)
    request = dict(version='1.0', model={
        'id': 'OpenMOSS-Team/MOSS-Transcribe-Diarize', 'local_path': str(config.MOSS_MODEL_PATH),
        'revision': config.MOSS_REVISION, 'custom_code_approved': config.MOSS_CUSTOM_CODE_APPROVED,
        'torch_version': config.MOSS_TORCH_VERSION, 'transformers_version': config.MOSS_TRANSFORMERS_VERSION},
        source={'id': 'sha256:' + hashlib.sha256(mono.read_bytes()).hexdigest(),
                'kind': 'dialogue', 'audio_path': str(mono.resolve()), 'window_id': 'reference_clip'},
        device='cpu', max_audio_seconds=10, max_new_tokens=1024)
    incoming, outgoing = directory / 'request.json', directory / 'result.json'
    incoming.write_text(json.dumps(request), encoding='utf-8')
    env = {**os.environ, 'PYTHONUTF8': '1'}
    result = subprocess.run([config.ENV_PARSE_PYTHON, str(config.ROOT_DIR / 'workers' / 'moss_worker.py'),
                             '--input', str(incoming), '--output', str(outgoing)],
                            env=env, cwd=config.ROOT_DIR, capture_output=True, timeout=config.PARSE_TIMEOUT_SEC)
    if result.returncode or not outgoing.is_file():
        raise ValueError('自动参考音 MOSS 识别进程失败')
    report = json.loads(outgoing.read_text(encoding='utf-8'))
    rows = report.get('segments', [])
    if (report.get('status') != 'ok' or report.get('review_reasons') or not rows
            or report.get('generation_limit_reached') or not report.get('ended_with_eos')
            or len({row.get('speaker') for row in rows}) != 1
            or any(row.get('speaker') in {None, 'UNKNOWN'} for row in rows)):
        raise ValueError('自动参考音识别不完整或含多个说话人，请换一段参考音')
    return report


def automatic_moss_reference(script: ProjectScript, speaker: str) -> ReferencePreview | None:
    lang = normalize_tts_lang(script.source_lang, default='')
    if not lang or speaker == 'UNKNOWN' or not script.media.vocals_wav:
        return None
    source = config.resolve_media_path(script.media.vocals_wav)
    project = (config.WORKSPACE_DIR / script.project_id).resolve()
    if not source.is_file() or project not in source.parents:
        return None
    candidates = [seg for seg in script.segments if seg.speaker == speaker
                  and seg.timing_source == 'moss_joint_generated' and seg.alignment_granularity == 'segment'
                  and seg.end - seg.start >= 3.2 and seg.orig_text.strip()
                  and not any(other.id != seg.id and other.start < seg.end and other.end > seg.start
                              for other in script.segments)]
    candidates.sort(key=lambda seg: (seg.end - seg.start > 10, abs(seg.end - seg.start - 6)))
    failures = []
    for seg in candidates:
        # Preserve the turn onset instead of cutting into its first word.
        start = seg.start
        duration = min(8.0, seg.end - start)
        key = (str(source), source.stat().st_mtime_ns, source.stat().st_size,
               speaker, seg.id, seg.orig_text, start, duration, lang, config.MOSS_REVISION, 'moss_ref_v1')
        stem = 'moss_ref_' + hashlib.sha256(repr(key).encode()).hexdigest()[:24]
        folder = project / 'references'
        audio, metadata = folder / (stem + '.wav'), folder / (stem + '.json')
        if audio.is_file() and metadata.is_file():
            cached = json.loads(metadata.read_text(encoding='utf-8'))
            return ReferencePreview.model_validate(cached['reference'])
        folder.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=folder) as temporary:
                directory = Path(temporary)
                draft = directory / 'reference.wav'
                subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', str(start),
                                '-i', str(source), '-t', str(duration), '-ac', '1', '-ar', '32000',
                                '-c:a', 'pcm_s16le', str(draft)], check=True, capture_output=True, timeout=30)
                with wave.open(str(draft), 'rb') as stream:
                    actual_duration = stream.getnframes() / stream.getframerate()
                if not 3 <= actual_duration <= 10:
                    raise ValueError('裁剪后的参考音长度不在 3–10 秒内')
                report = transcribe_reference(draft, directory)
                text = prepare_tts_text(' '.join(row['text'] for row in report['segments']), lang)
                if not text.strip():
                    raise ValueError('参考音识别结果为空')
                reference = ReferencePreview(wav=f'/workspace/{script.project_id}/references/{audio.name}',
                                             text=text, lang=lang, segment_ids=[seg.id])
                record = {'reference': reference.model_dump(), 'source_start': start, 'source_end': start + duration,
                          'speaker_assignment': 'inherited_from_moss_source_turn_not_human_verified',
                          'transcript_source': 'moss_clip_retranscription', 'model_revision': config.MOSS_REVISION,
                          'report': report}
                draft.replace(audio)
                metadata.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
                return reference
        except (ValueError, OSError, subprocess.SubprocessError) as exc:
            failures.append(str(exc))
    if failures:
        raise ValueError('自动参考音生成失败：' + failures[-1])
    return None
