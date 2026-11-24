"""UI LLM preferences, secret handling and immutable request configuration."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from backend import config
from backend.main import app
from backend.services import llm_settings as settings
from backend.services import translation


class LLMSettingsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'llm_settings.local.json'
        for target, value in [('SETTINGS_PATH', self.path)]:
            context = patch.object(settings, target, value)
            context.start()
            self.addCleanup(context.stop)
        for key, value in [('LLM_API_KEY', 'old-secret-key'), ('LLM_BASE_URL', 'https://old.example/v1'), ('LLM_MODEL', 'old-model')]:
            context = patch.object(config, key, value)
            context.start()
            self.addCleanup(context.stop)
        context = patch.dict('os.environ', {'STAGETTS_LLM_SNAPSHOT': '0'})
        context.start()
        self.addCleanup(context.stop)
        self.client = TestClient(app)

    def put(self, **changes):
        return self.client.put('/api/settings/llm', json={'base_url': 'https://new.example/v1', 'model': 'new-model', **changes})

    def test_get_uses_environment_defaults_but_never_returns_key(self):
        response = self.client.get('/api/settings/llm')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['model'], 'old-model')
        self.assertTrue(response.json()['api_key_configured'])
        self.assertNotIn('old-secret-key', response.text)
        self.assertFalse(self.path.exists())

    def test_save_persists_across_reads_and_blank_key_preserves_secret(self):
        response = self.put(api_key='new-secret')
        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn('new-secret', response.text)
        self.assertEqual(json.loads(self.path.read_text())['api_key'], 'new-secret')
        response = self.put(model='third-model', api_key='')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(settings.get_llm_snapshot().api_key, 'new-secret')
        self.assertEqual(self.client.get('/api/settings/llm').json()['model'], 'third-model')
        self.assertNotIn('new-secret', repr(settings.get_llm_snapshot()))

    def test_clear_key_does_not_fall_back_to_environment_secret(self):
        response = self.put(clear_api_key=True)
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()['configured'])
        self.assertFalse(config.is_llm_configured())
        self.assertEqual(settings.get_llm_snapshot().api_key, '')

    def test_invalid_settings_never_replace_existing_configuration(self):
        self.put()
        before = self.path.read_bytes()
        for changes in [{'base_url': 'file:///private'}, {'base_url': 'https://user:pass@example.com/v1'},
                        {'base_url': 'https://example.com/v1?key=secret'}, {'model': ' '},
                        {'clear_api_key': True, 'api_key': 'new-key'}]:
            with self.subTest(changes=changes):
                self.assertIn(self.put(**changes).status_code, [400, 422])
                self.assertEqual(self.path.read_bytes(), before)

    def test_atomic_write_failure_preserves_previous_settings(self):
        self.put()
        before = self.path.read_bytes()
        with patch.object(settings.os, 'replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                settings.save_llm_settings(settings.LLMSettingsUpdate(base_url='https://other.example/v1', model='other'))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob('.llm-settings-*.tmp')), [])

    def test_client_keeps_model_url_and_key_snapshot_after_settings_change(self):
        self.put(api_key='first-key')
        with patch.object(translation, 'OpenAI') as factory:
            first = MagicMock()
            second = MagicMock()
            factory.side_effect = [first, second]
            self.assertIs(translation.create_openai_client(), first)
            self.assertEqual(factory.call_args.kwargs['api_key'], 'first-key')
            self.assertEqual(factory.call_args.kwargs['base_url'], 'https://new.example/v1')
            self.put(model='another-model', api_key='second-key')
            self.assertIs(translation.create_openai_client(), second)
            first.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='[]'))])
            translation.complete_dialogue(first, 'test', {})
            self.assertEqual(first.chat.completions.create.call_args.kwargs['model'], 'new-model')
            self.assertEqual(translation.client_model(second), 'another-model')

    def test_real_sdk_client_accepts_bound_model_without_any_network_request(self):
        self.put(api_key='sdk-test-key')
        client = translation.create_openai_client()
        try:
            self.assertEqual(translation.client_model(client), 'new-model')
            self.assertEqual(str(client.base_url), 'https://new.example/v1/')
        finally:
            client.close()

    def test_parse_snapshot_does_not_read_later_ui_changes(self):
        self.put(model='ui-model', api_key='ui-key')
        with patch.dict('os.environ', {'STAGETTS_LLM_SNAPSHOT': '1'}):
            snapshot = settings.get_llm_snapshot()
        self.assertEqual(snapshot.model, 'old-model')
        self.assertEqual(snapshot.api_key, 'old-secret-key')

    def test_corrupt_settings_reports_safe_error_without_secret_contents(self):
        self.path.write_text('{broken: super-secret-key', encoding='utf-8')
        response = self.client.get('/api/settings/llm')
        self.assertEqual(response.status_code, 400)
        self.assertNotIn('super-secret-key', response.text)


if __name__ == '__main__':
    unittest.main()
