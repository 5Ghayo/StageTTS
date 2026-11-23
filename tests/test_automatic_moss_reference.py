import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from backend import config
from backend.schemas import ProjectScript, Segment, MediaInfo
from backend.services.references import automatic_reference
from backend.services import automatic_moss_reference as module
import wave


def write_audio(path, seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as stream:
        stream.setparams((1, 2, 32000, 0, 'NONE', 'not compressed'))
        stream.writeframes(b'\x00\x00' * round(seconds * 32000))


class AutomaticMossReferenceTest(unittest.TestCase):
    def test_unreviewed_long_moss_turn_uses_clip_transcript_and_caches_without_clearing_reviews(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(module, 'transcribe_reference') as infer:
            root = Path(directory)
            write_audio(root / 'demo' / 'vocals.wav', 16)
            script = ProjectScript(project_id='demo', source_lang='ja', media=MediaInfo(duration=16, vocals_wav='/workspace/demo/vocals.wav'),
                segments=[Segment(id=1,start=0,end=15,speaker='MOSS_A',orig_text='whole long transcript',
                    alignment_granularity='segment', timing_source='moss_joint_generated',
                    speaker_needs_review=True, segment_timing_needs_review=True)])
            infer.return_value={'status':'ok','segments':[{'text':'clip transcript','speaker':'S01'}]}
            with patch.object(config, 'WORKSPACE_DIR', root):
                first = automatic_reference(script,'MOSS_A')
                second = automatic_reference(script,'MOSS_A')
            self.assertEqual(first.text, 'clip transcript')
            self.assertEqual(first, second)
            infer.assert_called_once()
            self.assertTrue(script.segments[0].speaker_needs_review)
            self.assertTrue(script.segments[0].segment_timing_needs_review)
            record=json.loads(next((root/'demo'/'references').glob('*.json')).read_text(encoding='utf-8'))
            self.assertLessEqual(record['source_end']-record['source_start'],10)
            self.assertEqual(record['transcript_source'],'moss_clip_retranscription')

    def test_unknown_or_overlapping_moss_turn_never_retranscribed(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(module, 'transcribe_reference') as infer:
            root=Path(directory);write_audio(root/'demo'/'vocals.wav',10)
            script=ProjectScript(project_id='demo',source_lang='ja',media=MediaInfo(duration=16, vocals_wav='/workspace/demo/vocals.wav'),
                segments=[Segment(id=0,start=0,end=6,speaker='A',orig_text='a',alignment_granularity='segment',timing_source='moss_joint_generated'),
                          Segment(id=1,start=1,end=4,speaker='B',orig_text='b')])
            with patch.object(config,'WORKSPACE_DIR',root):
                self.assertIsNone(automatic_reference(script,'A'))
                self.assertIsNone(automatic_reference(script,'UNKNOWN'))
            infer.assert_not_called()
