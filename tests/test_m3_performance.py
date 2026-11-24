import copy
import math
import shutil
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from backend.schemas import AlignedWord, CharacterConfig, EmotionRef, MediaInfo, ProjectScript, Segment, SourceTurn
from backend.services.evidence import build_source_evidence
from backend.services.performance_analysis import (
    _validate_stage, analyze_project_performance, apply_analysis_to_evidence,
)
from backend.services.engine_capabilities import build_gpt_sovits_request, get_capabilities, resolve_reference_route
from workers.prosody_worker import analyze_request, measure_clip, read_wav


def write_wave(path, amplitudes=(0.1, 0.4), rate=16000, channels=1):
    data = b''.join(struct.pack('<h', round(amplitude * math.sin(2 * math.pi * 200 * index / rate) * 32767)) * channels
                    for amplitude in amplitudes for index in range(rate))
    with wave.open(str(path), 'wb') as audio:
        audio.setparams((channels, 2, rate, 0, 'NONE', 'not compressed'))
        audio.writeframes(data)


def project(path, *, evidence=True):
    segments = [Segment(id=index, start=index, end=index + 1, speaker='A', orig_text='hello',
                        target_text='你好', emotion='happy', speaker_needs_review=False,
                        words=[AlignedWord(word='hello', start=index, end=index + 1, speaker='A')],
                        alignment_needs_review=False) for index in range(2)]
    script = ProjectScript(project_id='m3', media=MediaInfo(duration=2), source_lang='en', segments=segments)
    if evidence:
        script.evidence = build_source_evidence(
            [dict(start=0, end=2, text='hello hello', words=[dict(word='hello', start=0, end=1),
                                                          dict(word='hello', start=1, end=2)])],
            [dict(start=0, end=2, speaker='A')], segments, source_id='sha256:test')
    return script


class PerformanceTest(unittest.TestCase):
    def test_pcm_stereo_downmix_and_energy_activity(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'audio.wav'
            write_wave(path, amplitudes=(0.0, 0.2), rate=48000, channels=2)
            samples, rate, metadata = read_wav(path)
            metrics = {item['name']: item for item in measure_clip(samples, rate, token_count=4)}
            self.assertEqual(metadata['channels'], 2)
            self.assertAlmostEqual(metrics['rms_amplitude']['value'], 0.1, places=3)
            self.assertAlmostEqual(metrics['energy_active_sec']['value'], 1, places=2)
            self.assertAlmostEqual(metrics['source_token_rate_active']['value'], 4, places=2)
            self.assertIsNone(metrics['f0_median_hz']['value'])
            self.assertEqual(metrics['f0_median_hz']['status'], 'not_available')

    def test_silence_never_invents_f0_or_rate(self):
        result = {item['name']: item for item in measure_clip([0.0] * 16000, 16000)}
        self.assertEqual(result['energy_active_ratio']['value'], 0)
        self.assertIsNone(result['rms_dbfs']['value'])
        self.assertIsNone(result['source_token_rate_active']['value'])

    def test_missing_optional_f0_is_not_available(self):
        with patch('workers.prosody_worker.importlib.util.find_spec', return_value=None):
            metrics = measure_clip([0.2] * 16000, 16000, enable_f0=True)
        self.assertEqual(metrics[-1]['status'], 'not_available')
        self.assertIsNone(metrics[-1]['value'])

    def test_real_subprocess_prosody_and_speaker_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'audio.wav'
            write_wave(path)
            script = project(path)
            before = script.model_dump()
            report = analyze_project_performance(script, audio_path=path)
            self.assertEqual(report.status, 'ok', report.model_dump())
            values = [{item['name']: item['value'] for item in annotation['measurements']}
                      for annotation in report.prosody.annotations]
            self.assertAlmostEqual(values[0]['rms_dbfs_speaker_z'], -1)
            self.assertAlmostEqual(values[1]['rms_dbfs_speaker_z'], 1)
            annotations = apply_analysis_to_evidence(script, report)
            self.assertEqual(len(annotations), 2)
            self.assertEqual(annotations[0].review.provenance.parameters['wav_sample_rate'], 16000)
            self.assertEqual(script.model_dump(), before)
            self.assertEqual(script.segments[0].emotion, 'happy')

    def test_unknown_or_overlapped_source_never_calls_emotion(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'audio.wav'
            write_wave(path)
            script = project(path)
            script.segments[0].speaker = 'UNKNOWN'
            script.segments[0].speaker_needs_review = True
            script.segments[1].alignment_needs_review = True
            def forbidden(*args):
                self.fail('blocked source reached SER')
            report = analyze_project_performance(script, audio_path=path, enable_prosody=False,
                                                enable_emotion=True, emotion_runner=forbidden)
            self.assertEqual(report.emotion.status, 'blocked')
            self.assertTrue(all(not item['labels'] for item in report.emotion.annotations))

    def test_missing_model_config_is_not_neutral(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'audio.wav'
            write_wave(path)
            report = analyze_project_performance(project(path), audio_path=path,
                                                enable_prosody=False, enable_emotion=True)
            self.assertEqual(report.emotion.status, 'not_configured')
            self.assertEqual(report.emotion.annotations, [])

    def test_old_project_does_not_fabricate_source_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'audio.wav'
            write_wave(path)
            script = project(path, evidence=False)
            report = analyze_project_performance(script, audio_path=path)
            self.assertEqual(report.prosody.status, 'blocked')
            self.assertTrue(report.prosody.annotations[0]['measurements'])
            self.assertEqual(apply_analysis_to_evidence(script, report), [])
            self.assertIsNone(script.evidence)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'requires ffmpeg')
    def test_emotion_conversion_is_temporary_and_provenance_survives(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'audio.wav'
            write_wave(path, rate=48000, channels=2)
            script = project(path)
            converted_paths = []
            def runner(python, worker, payload):
                converted = Path(payload['source']['audio_path'])
                converted_paths.append(converted)
                with wave.open(str(converted)) as audio:
                    self.assertEqual((audio.getframerate(), audio.getnchannels(), audio.getsampwidth()), (16000, 1, 2))
                return {'version': '1.0', 'stage': 'emotion', 'status': 'ok', 'model': payload['model'],
                        'source': payload['source'],
                        'annotations': [{**{key: span[key] for key in ('id', 'start', 'end', 'source_token_ids', 'source_turn_ids')},
                                         'status': 'ok', 'labels': ['sad', 'happy'], 'scores': [0.1, 0.9],
                                         'needs_review': True, 'reliability': 'unvalidated',
                                         'score_kind': 'model_softmax_uncalibrated'} for span in payload['spans']]}
            report = analyze_project_performance(script, audio_path=path, enable_prosody=False, enable_emotion=True,
                emotion_python=sys.executable, emotion_model_path=temp, emotion_model_id='test-local',
                emotion_revision='commit-hash', emotion_funasr_version='1.4.28', emotion_user_authorized=True,
                emotion_runner=runner)
            self.assertEqual(report.emotion.status, 'ok', report.model_dump())
            self.assertFalse(converted_paths[0].exists())
            annotation = apply_analysis_to_evidence(script, report)[0]
            self.assertEqual(annotation.emotions[0].score_kind, 'model_softmax_uncalibrated')
            self.assertTrue(annotation.review.needs_review)
            self.assertIn('model_reliability:unvalidated', annotation.review.reasons)
            self.assertEqual(annotation.review.provenance.parameters['reliability'], 'unvalidated')
            self.assertEqual(annotation.review.provenance.parameters['original_sample_rate'], 48000)
            self.assertEqual(annotation.review.provenance.parameters['preprocessing_method'], 'ffmpeg_pcm16_mono_16k')

    def test_worker_outputs_cannot_change_source_or_claim_probability(self):
        span = dict(id='one', start=0, end=1, source_token_ids=[], source_turn_ids=[], blocked_reasons=[])
        base = {'version': '1.0', 'stage': 'emotion', 'status': 'ok',
                'annotations': [dict(id='one', start=0, end=1, status='ok', labels=['happy'], scores=[0.8],
                                     score_kind='probability')]}
        with self.assertRaisesRegex(ValueError, 'calibrated'):
            _validate_stage(base, 'emotion', [span])
        base['annotations'][0]['score_kind'] = 'model_softmax_uncalibrated'
        base['annotations'][0]['start'] = 0.5
        with self.assertRaisesRegex(ValueError, 'range'):
            _validate_stage(base, 'emotion', [span])
        base['stage'] = 'prosody'
        base['annotations'][0].update(start=0, status='not_run', measurements=[dict(name='rms', value=0.5)])
        with self.assertRaisesRegex(ValueError, 'unrun prosody'):
            _validate_stage(base, 'prosody', [span])

    def test_request_identity_empty_predictions_and_annotation_ids(self):
        span = dict(id='one', start=0, end=1, source_token_ids=[], source_turn_ids=[], blocked_reasons=[])
        result = {'version': '1.0', 'stage': 'emotion', 'status': 'ok',
                  'source': {'id': 'wrong', 'kind': 'dialogue', 'audio_path': 'x'},
                  'model': {'id': 'model', 'revision': 'commit'},
                  'annotations': [{**span, 'status': 'ok', 'labels': [], 'scores': []}]}
        with self.assertRaisesRegex(ValueError, 'source identity'):
            _validate_stage(result, 'emotion', [span], expected_source={'id': 'source', 'kind': 'dialogue', 'audio_path': 'x'})
        with self.assertRaisesRegex(ValueError, 'model identity'):
            _validate_stage(result, 'emotion', [span], expected_model={'id': 'another', 'revision': 'commit'})
        with self.assertRaisesRegex(ValueError, 'requires predictions'):
            _validate_stage(result, 'emotion', [span])

    def test_pyin_uses_voiced_flags_and_drops_unvoiced_noise(self):
        import types
        import numpy as np
        def fake_pyin(signal, **kwargs):
            self.assertFalse(kwargs['center'])
            return np.array([220., 300., 200.]), np.array([True, False, False]), np.array([0.95, 0.99, 0.2])
        fake_librosa = types.SimpleNamespace(pyin=fake_pyin)
        with patch('workers.prosody_worker.importlib.util.find_spec', return_value=object()), patch.dict(sys.modules, {'librosa': fake_librosa}):
            result = {item['name']: item for item in measure_clip([0.2] * 8000, 16000, enable_f0=True)}
            self.assertEqual(result['f0_median_hz']['value'], 220)
            fake_librosa.pyin = lambda *args, **kwargs: (np.array([220.]), np.array([False]), np.array([0.99]))
            result = {item['name']: item for item in measure_clip([0.2] * 8000, 16000, enable_f0=True)}
            self.assertIsNone(result['f0_median_hz']['value'])
            self.assertIn('no_reliable', result['f0_median_hz']['reason'])

    def test_segment_only_confirmation_preserves_unknown_words_and_null_rate(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source.wav'
            write_wave(path)
            seg = Segment(id=0, start=0, end=2, speaker='A', orig_text='A complete sentence.',
                          target_text='完整句子。', words=[], speaker_needs_review=False,
                          alignment_needs_review=True, alignment_granularity='segment',
                          timing_source='moss_joint_generated', segment_timing_needs_review=False)
            script = ProjectScript(project_id='moss_segment', media=MediaInfo(duration=2), segments=[seg])
            script.evidence = build_source_evidence(
                [dict(start=0, end=2, text=seg.orig_text, words=[])],
                [dict(start=0, end=2, speaker='A')], [seg], source_id='sha256:segment')
            # Builder links exact utterance tokens without inventing their times.
            self.assertEqual(seg.source_token_ids, script.evidence.source_utterances[0].source_token_ids)
            report = analyze_project_performance(script, audio_path=path, enable_f0=False)
            self.assertEqual(report.prosody.status, 'ok', report.model_dump())
            metrics = {item['name']: item for item in report.prosody.annotations[0]['measurements']}
            self.assertIsNone(metrics['source_token_rate_active']['value'])
            self.assertTrue(seg.alignment_needs_review)
            self.assertEqual(seg.words, [])
            self.assertIsNone(script.evidence.source_tokens[0].start)
            self.assertTrue(script.evidence.source_tokens[0].review.needs_review)
            stored = apply_analysis_to_evidence(script, report)[0]
            self.assertTrue(stored.review.provenance.parameters['source_word_alignment_needs_review'])
            seg.segment_timing_needs_review = True
            seg.alignment_needs_review = False  # Old/no-word false is insufficient evidence.
            report = analyze_project_performance(script, audio_path=path, enable_f0=False)
            self.assertEqual(report.prosody.status, 'blocked')
            seg.segment_timing_needs_review = False
            seg.alignment_needs_review = True
            seg.orig_text = 'Edited text does not match source.'
            report = analyze_project_performance(script, audio_path=path, enable_f0=False)
            self.assertEqual(report.prosody.status, 'blocked')
            self.assertIn('confirmed_segment_source_utterance_mismatch', report.prosody.annotations[0]['review_reasons'])

    def test_confirmed_segment_rejects_other_utterance_token_and_disjoint_turn(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'source.wav'
            write_wave(path)
            script = project(path)
            seg = script.segments[0]
            seg.segment_timing_needs_review = False
            utterance = script.evidence.source_utterances[0].model_copy(deep=True)
            utterance.id, utterance.start, utterance.end, utterance.text = 'exact_0', 0, 1, 'hello'
            utterance.source_token_ids = [script.evidence.source_tokens[0].id]
            script.evidence.source_utterances.append(utterance)
            seg.source_token_ids = list(script.segments[1].source_token_ids)
            report = analyze_project_performance(script, audio_path=path, segment_ids=[0], enable_f0=False)
            self.assertEqual(report.prosody.status, 'blocked')
            self.assertIn('confirmed_segment_source_token_mismatch', report.prosody.annotations[0]['review_reasons'])
            # Use one exact utterance per segment so the correct token link can
            # pass; then a same-speaker non-overlapping turn must still fail.
            seg.source_token_ids = [script.evidence.source_tokens[0].id]
            wrong_turn = SourceTurn(id='other_time', start=1, end=2, speaker='A')
            script.evidence.source_turns.append(wrong_turn)
            seg.source_turn_ids = [wrong_turn.id]
            report = analyze_project_performance(script, audio_path=path, segment_ids=[0], enable_f0=False)
            self.assertEqual(report.prosody.status, 'blocked')
            self.assertIn('source_turn_requires_review', report.prosody.annotations[0]['review_reasons'])

    def test_gpt_engine_only_routes_reference_no_fake_emotion_control(self):
        engine = get_capabilities('gpt_sovits_v2')
        self.assertFalse(engine.direct_emotion_control)
        self.assertFalse(engine.direct_prosody_control)
        self.assertEqual(engine.status, 'wired')
        self.assertEqual(get_capabilities('cosyvoice3_0_5b').status, 'planned_not_wired')
        reference = EmotionRef(wav='reference.wav', text='Actual English reference.', lang='en')
        character = CharacterConfig(display_name='A', emotion_refs={'neutral': reference})
        routed, reason = resolve_reference_route(character, 'angry')
        self.assertEqual(reason, 'neutral_reference_fallback')
        payload = build_gpt_sovits_request(text='中文台词', text_lang='zh', reference=routed,
                                          ref_audio_path='reference.wav')
        self.assertEqual(payload['prompt_lang'], 'en')
        self.assertEqual(set(payload), {'text', 'text_lang', 'ref_audio_path', 'prompt_text',
                                        'prompt_lang', 'speed_factor', 'media_type'})


if __name__ == '__main__':
    unittest.main()
