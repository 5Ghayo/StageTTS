import json
import math
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import config
from backend.main import app
from backend.schemas import AlignedWord, CharacterConfig, MediaInfo, ProjectScript, Segment
from backend.services.evidence import build_source_evidence
from backend.services.task_runner import gpu_task_guard


class M123IntegrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(config, 'WORKSPACE_DIR', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.project = self.root / 'evidence-api'
        self.project.mkdir()
        with wave.open(str(self.project / 'vocals.wav'), 'wb') as wav:
            wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            wav.writeframes(array('h', [int(4000 * math.sin(2 * math.pi * 160 * i / 16000))
                                      for i in range(3 * 16000)]).tobytes())
        self.script = ProjectScript(
            project_id='evidence-api', source_lang='en',
            media=MediaInfo(duration=3, vocals_wav='/workspace/evidence-api/vocals.wav'),
            characters={'A': CharacterConfig(display_name='A')},
            segments=[Segment(id=0, start=0, end=2, speaker='A', orig_text='Hello there.',
                              target_text='你好。', alignment_needs_review=False, speaker_needs_review=False,
                              words=[AlignedWord(word='Hello', start=0, end=0.8, speaker='A'),
                                     AlignedWord(word='there.', start=1, end=2, speaker='A')])],
        )
        self.script.evidence = build_source_evidence(
            [{'start': 0, 'end': 2, 'text': 'Hello there.', 'words': [
                {'word': 'Hello', 'start': 0, 'end': 0.8},
                {'word': 'there.', 'start': 1, 'end': 2},
            ]}],
            [{'start': 0, 'end': 2, 'speaker': 'A'}], self.script.segments, source_id='sha256:synthetic',
        )
        self.save(self.script)
        # Source input for the parse API, separate from the cached dialogue stem.
        (self.project / 'input.wav').write_bytes((self.project / 'vocals.wav').read_bytes())

    def save(self, script):
        (self.project / 'script.json').write_text(script.model_dump_json(), encoding='utf8')

    def test_legacy_json_is_readable_and_keeps_no_fabricated_source_evidence(self):
        data = self.script.model_dump()
        data.pop('evidence')
        for segment in data['segments']:
            for key in ('source_token_ids', 'source_turn_ids', 'translation_unit_ids', 'performance_unit_ids', 'boundary_provenance'):
                segment.pop(key, None)
        (self.project / 'script.json').write_text(json.dumps(data), encoding='utf8')
        with TestClient(app) as client:
            response = client.get('/api/projects/evidence-api/script')
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()['evidence'])

    def test_human_role_review_persists_and_invalidates_the_previous_voice(self):
        self.script.characters['B'] = CharacterConfig(display_name='B')
        segment = self.script.segments[0]
        segment.tts_wav = '/workspace/evidence-api/tts/seg_0000.wav'
        segment.status = 'ready'
        self.save(self.script)
        payload = self.script.model_dump()
        payload['segments'][0].update(speaker='B', speaker_reviewed_by_user=True)
        with TestClient(app) as client:
            saved = client.put('/api/projects/evidence-api/script', json=payload)
            self.assertEqual(saved.status_code, 200, saved.text)
        with TestClient(app) as new_client:
            result = new_client.get('/api/projects/evidence-api/script')
        row = result.json()['segments'][0]
        self.assertEqual(row['speaker'], 'B')
        self.assertTrue(row['speaker_reviewed_by_user'])
        self.assertFalse(row['speaker_needs_review'])
        self.assertEqual(row['status'], 'pending')
        self.assertIsNone(row['tts_wav'])

    def test_real_prosody_subprocess_persists_rich_measurements_without_emotion(self):
        with TestClient(app) as client, patch.object(config, 'ENV_ANALYSIS_PYTHON', ''):
            response = client.post('/api/projects/evidence-api/analyze-performance', json={'segment_ids': [0]})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data['stage_status']['emotion'], 'not_run')
        self.assertEqual(data['stage_status']['prosody'], 'ok')
        self.assertEqual(data['script']['segments'][0]['emotion'], 'neutral')
        annotations = data['script']['evidence']['annotations']
        self.assertTrue(annotations)
        self.assertFalse(any(item['emotions'] for item in annotations))
        self.assertTrue(any(metric['value'] is not None for item in annotations for metric in item['prosody']))
        saved = ProjectScript.model_validate_json((self.project / 'script.json').read_text(encoding='utf8'))
        self.assertEqual(len(saved.evidence.annotations), len(annotations))

    def test_missing_ser_config_returns_not_configured_not_neutral_prediction(self):
        with TestClient(app) as client, patch.object(config, 'ENV_ANALYSIS_PYTHON', ''), patch.object(config, 'EMOTION_USER_AUTHORIZED', False):
            response = client.post('/api/projects/evidence-api/analyze-performance', json={
                'segment_ids': [0], 'enable_emotion': True, 'enable_prosody': False,
            })
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['status'], 'not_configured')
        self.assertEqual(response.json()['stage_status']['emotion'], 'not_configured')
        self.assertEqual(response.json()['script']['evidence']['annotations'], [])

    def test_emotion_analysis_respects_existing_gpu_task_lock(self):
        with TestClient(app) as client, gpu_task_guard():
            response = client.post('/api/projects/evidence-api/analyze-performance', json={'enable_emotion': True})
        self.assertEqual(response.status_code, 409, response.text)

    def test_configured_ser_releases_only_managed_tts_and_blocks_external_service(self):
        with (
            TestClient(app) as client,
            patch.object(config, 'ENV_ANALYSIS_PYTHON', 'configured-python'),
            patch.object(config, 'EMOTION_MODEL_PATH', Path('configured-model')),
            patch.object(config, 'EMOTION_MODEL_ID', 'emotion2vec_plus_base'),
            patch.object(config, 'EMOTION_REVISION', 'pinned-revision'),
            patch.object(config, 'EMOTION_FUNASR_VERSION', '1.4.28'),
            patch.object(config, 'EMOTION_USER_AUTHORIZED', True),
            patch.object(config, 'TTS_AUTO_MANAGE', True),
            patch.object(config, 'TTS_API_URL', 'http://127.0.0.1:9880'),
            patch('backend.main.ensure_tts_stopped') as stop,
            patch('backend.main.check_tts_service', return_value=True),
            patch('backend.services.performance_analysis.analyze_project_performance') as analyze,
        ):
            response = client.post('/api/projects/evidence-api/analyze-performance', json={'enable_emotion': True})
        self.assertEqual(response.status_code, 409, response.text)
        stop.assert_called_once()
        analyze.assert_not_called()

    def test_segment_confirmation_keeps_word_review_and_resets_on_source_edit(self):
        self.script.segments[0].words = []
        self.script.segments[0].alignment_granularity = 'segment'
        self.script.segments[0].timing_source = 'moss_joint_generated'
        self.script.segments[0].alignment_needs_review = True
        self.save(self.script)
        url = '/api/projects/evidence-api/script'
        with TestClient(app) as client:
            data = client.get(url).json()
            data['segments'][0]['segment_timing_needs_review'] = False
            response = client.put(url, json=data)
            self.assertEqual(response.status_code, 200, response.text)
            confirmed = response.json()['segments'][0]
            self.assertFalse(confirmed['segment_timing_needs_review'])
            self.assertTrue(confirmed['alignment_needs_review'])
            self.assertEqual(confirmed['words'], [])
            data = response.json()
            data['segments'][0]['alignment_granularity'] = 'word'
            data['segments'][0]['timing_source'] = 'forced_alignment'
            provenance_update = client.put(url, json=data)
            self.assertEqual(provenance_update.status_code, 200)
            self.assertEqual(provenance_update.json()['segments'][0]['alignment_granularity'], 'segment')
            self.assertEqual(provenance_update.json()['segments'][0]['timing_source'], 'moss_joint_generated')
            data = provenance_update.json()
            data['segments'][0]['orig_text'] = 'Edited source.'
            response = client.put(url, json=data)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertTrue(response.json()['segments'][0]['segment_timing_needs_review'])
            self.assertTrue(response.json()['segments'][0]['alignment_needs_review'])

    def test_parse_api_forwards_explicit_and_existing_source_language(self):
        with TestClient(app) as client, patch('backend.main.run_parse_worker') as run:
            response = client.post('/api/projects/evidence-api/parse', json={'source_lang': 'ja'})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(run.call_args.kwargs['source_lang'], 'ja')
            response = client.post('/api/projects/evidence-api/parse', json={})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(run.call_args.kwargs['source_lang'], 'en')

    def test_unknown_segment_id_cannot_modify_project(self):
        before = (self.project / 'script.json').read_bytes()
        with TestClient(app) as client:
            response = client.post('/api/projects/evidence-api/analyze-performance', json={'segment_ids': [999]})
        self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual((self.project / 'script.json').read_bytes(), before)

    def test_unknown_source_is_blocked_without_fabricated_predictions(self):
        self.script.segments[0].speaker = 'UNKNOWN'
        self.script.segments[0].speaker_needs_review = True
        self.save(self.script)
        with TestClient(app) as client:
            response = client.post('/api/projects/evidence-api/analyze-performance', json={'segment_ids': [0]})
        self.assertEqual(response.status_code, 200, response.text)
        annotations = response.json()['script']['evidence']['annotations']
        self.assertTrue(annotations)
        self.assertTrue(all(item['status'] == 'blocked' for item in annotations))
        self.assertTrue(all(not item['emotions'] for item in annotations))


if __name__ == '__main__':
    unittest.main()
