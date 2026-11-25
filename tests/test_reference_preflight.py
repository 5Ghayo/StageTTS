"""Missing references fail before GPU startup and preserve review metadata."""
import unittest
import tempfile
from pathlib import Path
from backend import config
from unittest.mock import MagicMock, patch
from backend.schemas import CharacterConfig, EmotionRef, ModelParameters, ProjectScript, MediaInfo, Segment
from backend.services import tts_client, references


class ReferencePreflightTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = patch.object(config, 'WORKSPACE_DIR', Path(temporary.name))
        workspace.start()
        self.addCleanup(workspace.stop)

    def script(self, duration=6):
        return ProjectScript(project_id='reference_preflight', source_lang='ja',
            media=MediaInfo(duration=30), segments=[Segment(id=0, start=0, end=duration,
                speaker='MOSS_A', orig_text='原文', target_text='译文',
                segment_timing_needs_review=True, speaker_needs_review=True)])

    def test_missing_reference_does_not_start_tts_and_reports_candidate(self):
        script = self.script()
        with patch.object(tts_client, '_resolve_ref_for_segment', return_value=None), \
             patch.object(tts_client, 'ensure_tts_running') as start:
            ids, errors = tts_client.synthesize_segments_grouped(script)
        start.assert_not_called()
        self.assertEqual(ids, [])
        self.assertIn('#0', errors[0])
        self.assertIn('句段时间与原文', errors[0])
        self.assertIn('当前说话人', errors[0])
        self.assertEqual(script.segments[0].status, 'error')
        self.assertTrue(script.segments[0].speaker_needs_review)
        self.assertTrue(script.segments[0].segment_timing_needs_review)

    def test_long_unaligned_reference_requires_manual_clip_not_guessed_crop(self):
        reason = references.reference_unavailable_reason(self.script(15), 'MOSS_A')
        self.assertIn('不能直接截取', reason)
        self.assertIn('上传', reason)
        self.assertIn('逐字稿', reason)

    def test_empty_target_does_not_resolve_reference_or_start_tts(self):
        script = self.script()
        script.segments[0].target_text = ' '
        with patch.object(tts_client, '_resolve_ref_for_segment') as resolve, \
             patch.object(tts_client, 'ensure_tts_running') as start:
            ids, errors = tts_client.synthesize_segments_grouped(script)
        resolve.assert_not_called()
        start.assert_not_called()
        self.assertEqual(ids, [])
        self.assertIn('目标台词为空', errors[0])

    def test_video_role_uses_neutral_reference_across_emotions(self):
        script = self.script()
        neutral = config.WORKSPACE_DIR / 'neutral.wav'
        angry = config.WORKSPACE_DIR / 'angry.wav'
        neutral.touch()
        angry.touch()
        voice = CharacterConfig(display_name='角色', emotion_refs={
            'neutral': EmotionRef(wav=str(neutral), text='默认原文'),
            'angry': EmotionRef(wav=str(angry), text='生气原文'),
        })
        row = script.segments[0]
        row.emotion = 'angry'
        self.assertEqual(tts_client._resolve_ref_for_segment(script, voice, row)[1], neutral.resolve())
        script.mode = 'virtual_stage'
        self.assertEqual(tts_client._resolve_ref_for_segment(script, voice, row)[1], angry.resolve())

    def test_project_sampling_reaches_the_tts_request(self):
        script = self.script()
        script.model_parameters = ModelParameters(tts_top_k=2, tts_temperature=0.65)
        ref_path = config.WORKSPACE_DIR / 'ref.wav'
        ref_path.touch()
        reference = EmotionRef(wav=str(ref_path), text='原文', lang='ja')
        session = MagicMock()
        session.post.return_value.status_code = 200
        session.post.return_value.content = b'wav'
        with patch.object(tts_client, '_resolve_ref_for_segment', return_value=(reference, ref_path)), \
             patch.object(tts_client, 'ensure_tts_running'), \
             patch.object(tts_client.httpx, 'Client') as client, \
             patch.object(tts_client, 'measure_wav_duration', return_value=2.0):
            client.return_value.__enter__.return_value = session
            self.assertEqual(tts_client.synthesize_segments_grouped(script), ([0], []))
        request = session.post.call_args.kwargs['json']
        self.assertEqual((request['top_k'], request['temperature'], request['repetition_penalty']), (2, 0.65, 1.35))
        self.assertNotIn('top_p', request)


if __name__ == '__main__':
    unittest.main()
