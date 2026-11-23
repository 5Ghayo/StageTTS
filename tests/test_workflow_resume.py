"""Saved-source translation resume: real persistence, mocked upstream LLM only."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend import config
from backend.main import app
from backend.schemas import AlignedWord, MediaInfo, ProjectScript, Segment
from backend.services.evidence import build_source_evidence


def completion(rows):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(rows)))])


class WorkflowResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.workspace = patch.object(config, 'WORKSPACE_DIR', self.directory)
        self.workspace.start()
        self.addCleanup(self.workspace.stop)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)
        self.project = ProjectScript(
            project_id='resume', source_lang='en', media=MediaInfo(duration=6), glossary={'Beacon': '信标'},
            segments=[Segment(id=index, start=index * 3, end=index * 3 + 2, speaker='A',
                              orig_text='Beacon is ready.', target_text='Beacon is ready.', target_lang='en',
                              words=[AlignedWord(word='Beacon is ready.', start=index * 3, end=index * 3 + 2, speaker='A')],
                              alignment_needs_review=True, speaker_needs_review=True,
                              alignment_granularity='segment', timing_source='moss_joint_generated',
                              segment_timing_needs_review=True, emotion='happy', action='nod',
                              tts_wav=f'old_{index}.wav', actual_duration=1.7, applied_speed_factor=1,
                              status='ready') for index in range(2)])
        self.project.evidence = build_source_evidence(
            [dict(start=seg.start, end=seg.end, text=seg.orig_text, words=[])
             for seg in self.project.segments],
            [dict(start=0, end=5, speaker='A')], self.project.segments, source_id='sha256:source')
        (self.directory / self.project.project_id).mkdir()
        self.path = self.directory / self.project.project_id / 'script.json'
        self.save()
        self.llm = patch('backend.main.create_openai_client')
        self.factory = self.llm.start()
        self.addCleanup(self.llm.stop)
        self.parse = patch('backend.main.run_parse_worker', side_effect=AssertionError('translation must not parse'))
        self.parse_mock = self.parse.start()
        self.addCleanup(self.parse.stop)

    def save(self):
        self.path.write_text(self.project.model_dump_json(indent=2), encoding='utf-8')

    def source_fields(self, script):
        data = script.model_dump()
        for segment in data['segments']:
            for field in ('target_text', 'target_lang', 'emotion', 'action', 'keep_original',
                          'tts_wav', 'actual_duration', 'applied_speed_factor', 'status'):
                segment.pop(field)
        return data

    def rows(self):
        return [{'id': seg.id, 'target_text': '信标已经准备好了。', 'emotion': 'sad', 'action': 'point',
                 'keep_original': False, 'self_check': {'semantic_complete': True, 'omitted_info': ''}}
                for seg in self.project.segments]

    def test_override_translates_existing_sources_and_preserves_evidence(self):
        before = self.source_fields(self.project)
        self.factory.return_value.chat.completions.create.return_value = completion(self.rows())
        response = self.client.post('/api/projects/resume/translate', json={'target_lang': 'zh'})
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result['translated_ids'], [0, 1])
        saved = ProjectScript.model_validate_json(self.path.read_text(encoding='utf-8'))
        self.assertEqual(self.source_fields(saved), before)
        for segment in saved.segments:
            self.assertEqual(segment.target_lang, 'zh')
            self.assertEqual(segment.target_text, '信标已经准备好了。')
            self.assertEqual(segment.emotion, 'sad')
            self.assertEqual(segment.action, 'point')
            self.assertIsNone(segment.tts_wav)
            self.assertEqual(segment.status, 'pending')
        payload = json.loads(self.factory.return_value.chat.completions.create.call_args.kwargs['messages'][-1]['content'])
        self.assertEqual(payload['target_lang'], 'zh')
        self.assertEqual(payload['glossary'], self.project.glossary)
        self.parse_mock.assert_not_called()

    def test_old_bodyless_request_keeps_each_existing_target_language(self):
        self.factory.return_value.chat.completions.create.return_value = completion(self.rows())
        response = self.client.post('/api/projects/resume/translate')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(all(segment['target_lang'] == 'en' for segment in response.json()['script']['segments']))
        schema = self.client.get('/openapi.json').json()
        body = schema['paths']['/api/projects/{project_id}/translate']['post']['requestBody']
        self.assertFalse(body.get('required', False))
        self.assertIn('TranslationRequest', schema['components']['schemas'])

    def test_partial_failure_keeps_unsuccessful_text_language_and_audio(self):
        self.factory.return_value.chat.completions.create.return_value = completion(self.rows()[:1])
        before_failed = self.project.segments[1].model_dump()
        response = self.client.post('/api/projects/resume/translate', json={'target_lang': 'zh'})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()['translated_ids'], [0])
        self.assertEqual(response.json()['skipped_count'], 1)
        self.assertTrue(response.json()['errors'])
        saved = ProjectScript.model_validate_json(self.path.read_text(encoding='utf-8'))
        self.assertEqual(saved.segments[1].model_dump(), before_failed)
        self.assertEqual(saved.segments[0].target_lang, 'zh')
        self.assertIsNone(saved.segments[0].tts_wav)

    def test_all_failures_never_write_or_relabel_saved_script(self):
        for upstream in (completion([]), ValueError('gateway failed')):
            with self.subTest(upstream=type(upstream).__name__):
                before = self.path.read_bytes()
                if isinstance(upstream, Exception):
                    self.factory.return_value.chat.completions.create.side_effect = upstream
                else:
                    self.factory.return_value.chat.completions.create.side_effect = None
                    self.factory.return_value.chat.completions.create.return_value = upstream
                response = self.client.post('/api/projects/resume/translate', json={'target_lang': 'ja'})
                self.assertEqual(response.status_code, 502, response.text)
                self.assertEqual(self.path.read_bytes(), before)

    def test_empty_source_preflight_does_not_construct_llm_or_parse(self):
        for segments in ([], [Segment(id=0, start=0, end=1, speaker='A', orig_text='   ')]):
            with self.subTest(count=len(segments)):
                self.project.evidence = None
                self.project.segments = segments
                self.save()
                before = self.path.read_bytes()
                response = self.client.post('/api/projects/resume/translate', json={'target_lang': 'zh'})
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn('原文', response.json()['detail'])
                self.assertEqual(self.path.read_bytes(), before)
        self.factory.assert_not_called()
        self.parse_mock.assert_not_called()

    def test_language_only_change_invalidates_successful_audio(self):
        for segment in self.project.segments:
            segment.target_text = '信标已经准备好了。'
            segment.emotion = 'sad'
            segment.action = 'point'
        self.save()
        self.factory.return_value.chat.completions.create.return_value = completion(self.rows()[:1])
        response = self.client.post('/api/projects/resume/translate', json={'target_lang': 'zh'})
        self.assertEqual(response.status_code, 200, response.text)
        segments = response.json()['script']['segments']
        self.assertIsNone(segments[0]['tts_wav'])
        self.assertEqual(segments[0]['target_lang'], 'zh')
        self.assertEqual(segments[1]['tts_wav'], 'old_1.wav')
        self.assertEqual(segments[1]['target_lang'], 'en')

    def test_invalid_target_request_is_rejected_without_llm(self):
        for lang in ('', ' ', 'zh;command', 42):
            with self.subTest(lang=lang):
                response = self.client.post('/api/projects/resume/translate', json={'target_lang': lang})
                self.assertEqual(response.status_code, 422)
        self.factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
