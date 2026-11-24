"""Regression coverage for complete audio scheduling and verified reference crops."""
import shutil
import tempfile
import unittest
import wave
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend import config
from backend.schemas import AlignedWord, CinematicStems, MediaInfo, ProjectScript, Segment
from backend.services import media_mixer, references


def write_audio(path, seconds, *, rate=32000, sentinel=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = array('h', [0] * round(seconds * rate))
    if sentinel:
        samples[-round(0.05 * rate):] = array('h', [12000] * round(0.05 * rate))
    with wave.open(str(path), 'wb') as output:
        output.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        output.writeframes(samples.tobytes())


class AudioRegressionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = self.root / 'demo'
        self.project.mkdir()
        patcher = patch.object(config, 'WORKSPACE_DIR', self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def script(self, segments, duration=4):
        return ProjectScript(project_id='demo', media=MediaInfo(duration=duration), segments=segments)

    def ready(self, *, end=1, duration=1.1, **kwargs):
        write_audio(self.project / 'tts.wav', duration, sentinel=True)
        return Segment(id=0, start=0, end=end, speaker='A', status='ready',
                       tts_wav='/workspace/demo/tts.wav', applied_speed_factor=1, **kwargs)

    def test_accepted_small_overflow_retains_tail_in_empty_gap(self):
        seg = self.ready()
        script = self.script([seg, Segment(id=1, start=2, end=3, speaker='B')])
        output = media_mixer._build_vocal_track(script, self.project)
        with wave.open(str(output), 'rb') as audio:
            samples = array('h', audio.readframes(audio.getnframes()))
            tail = samples[round(1.05 * audio.getframerate()) * 2:round(1.1 * audio.getframerate()) * 2]
            self.assertGreater(max(tail), 10000)
            self.assertEqual(audio.getnframes(), 4 * audio.getframerate())

    def test_overflow_collision_blocks_even_when_next_line_is_pending(self):
        seg = self.ready(duration=1.2)
        script = self.script([seg, Segment(id=1, start=1.1, end=2, speaker='B')])
        with self.assertRaisesRegex(RuntimeError, '待审核.*#1'):
            media_mixer._build_vocal_track(script, self.project)
        self.assertFalse((self.project / 'dubbed_vocals.wav').exists())

    def test_project_end_blocks_instead_of_overlay_truncation(self):
        with self.assertRaisesRegex(RuntimeError, '媒体结尾'):
            media_mixer._build_vocal_track(self.script([self.ready(duration=1.2)], duration=1), self.project)

    def test_original_overlap_is_allowed_without_extending_it(self):
        script = self.script([self.ready(end=1.1, duration=1),
                              Segment(id=1, start=.9, end=2, speaker='B')])
        self.assertTrue(media_mixer._build_vocal_track(script, self.project).is_file())

    def test_legacy_tempo_is_validated_after_adjustment(self):
        seg = self.ready(duration=1.4)
        seg.applied_speed_factor = None
        seg.speed_factor = 1.15
        adjusted = self.project / 'adjusted.wav'
        write_audio(adjusted, 1.22, sentinel=True)
        with patch.object(media_mixer, 'apply_gentle_tempo', return_value=adjusted) as tempo:
            with self.assertRaisesRegex(RuntimeError, '待审核'):
                media_mixer._build_vocal_track(self.script([seg], duration=1.2), self.project)
            tempo.assert_called_once()

    def test_video_duration_blocks_before_render_and_never_uses_shortest(self):
        script = self.script([self.ready(duration=1.2)], duration=4)
        video = self.project / 'input.mp4'
        video.touch()
        script.media.source_video = '/workspace/demo/input.mp4'
        with patch.object(media_mixer, 'check_ffmpeg_available', return_value=True), \
                patch.object(media_mixer, 'probe_media_duration', return_value=1), \
                patch.object(media_mixer, '_run_ffmpeg_cmd') as render:
            with self.assertRaisesRegex(RuntimeError, '媒体结尾'):
                media_mixer.mix_and_render(script)
            render.assert_not_called()
        script.segments[0].end = 2
        with patch.object(media_mixer, 'check_ffmpeg_available', return_value=True), \
                patch.object(media_mixer, 'probe_media_duration', return_value=4), \
                patch.object(media_mixer, '_run_ffmpeg_cmd', return_value=SimpleNamespace(returncode=0)) as render:
            media_mixer.mix_and_render(script)
            self.assertNotIn('-shortest', render.call_args.args[0])

    def reference_script(self, seg, *, source_duration=15, banquet=False):
        name = 'cinematic-test/dialogue.wav' if banquet else 'vocals.wav'
        write_audio(self.project / name, source_duration)
        media = MediaInfo(duration=15, vocals_wav=f'/workspace/demo/{name}')
        if banquet:
            media.cinematic_stems = CinematicStems(**{
                stem: f'/workspace/demo/cinematic-test/{stem}.wav'
                for stem in ('dialogue', 'singing', 'music', 'effects')})
        return ProjectScript(project_id='demo', source_lang='en', media=media, segments=[seg])

    def long_segment(self):
        words = [AlignedWord(word=f'word{i}', start=i, end=i + .8, speaker='A') for i in range(12)]
        return Segment(id=0, start=0, end=12, speaker='A',
                       orig_text=' '.join(word.word for word in words), words=words,
                       alignment_needs_review=False, speaker_needs_review=False)

    def test_long_reference_crop_has_exact_boundary_transcript(self):
        script = self.reference_script(self.long_segment())
        preview = references.automatic_reference(script, 'A')
        self.assertIsNotNone(preview)
        self.assertEqual(preview.text, 'word0 word1 word2 word3 word4')
        self.assertEqual(preview.lang, 'en')
        with wave.open(str(config.resolve_media_path(preview.wav)), 'rb') as audio:
            self.assertEqual(audio.getnframes(), round(4.8 * 32000))
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getframerate(), 32000)

    def test_crop_refuses_edited_text_missing_timing_or_unknown_word_speaker(self):
        for mutation in ('text', 'timing', 'speaker', 'review', 'overlap'):
            with self.subTest(mutation=mutation):
                seg = self.long_segment()
                script = self.reference_script(seg)
                if mutation == 'text':
                    seg.orig_text = 'Corrected text does not match old words.'
                elif mutation == 'timing':
                    seg.words[2].end = None
                elif mutation == 'speaker':
                    seg.words[2].speaker = None
                elif mutation == 'review':
                    seg.alignment_needs_review = True
                else:
                    script.segments.append(Segment(id=1, start=0, end=12, speaker='A'))
                self.assertIsNone(references.automatic_reference(script, 'A'))

    def test_complete_reviewed_reference_needs_no_crop_or_word_guess(self):
        seg = Segment(id=0, start=0, end=4, speaker='A', orig_text='A complete reviewed line.',
                      segment_timing_needs_review=False,
                      alignment_needs_review=False, speaker_needs_review=False)
        for banquet in (False, True):
            with self.subTest(banquet=banquet):
                script = self.reference_script(seg, source_duration=4, banquet=banquet)
                preview = references.automatic_reference(script, 'A')
                self.assertEqual(preview.text, seg.orig_text)
                with wave.open(str(config.resolve_media_path(preview.wav)), 'rb') as audio:
                    self.assertEqual(audio.getnframes(), 4 * 32000)

    def test_full_reference_preferred_and_overlap_review_are_respected(self):
        script = self.reference_script(self.long_segment())
        script.segments.append(Segment(id=1, start=12, end=15, speaker='A', orig_text='Full line.',
                                       segment_timing_needs_review=False,
                                       alignment_needs_review=False, speaker_needs_review=False))
        self.assertEqual(references.automatic_reference(script, 'A').segment_ids, [1])
        script.segments[1].speaker_needs_review = True
        script.segments[0].speaker_needs_review = True
        self.assertIsNone(references.automatic_reference(script, 'A'))

    def test_source_tail_margin_requires_evidence_no_word_was_cut(self):
        for word_end, allowed in ((3.9, True), (4.02, False), (None, False)):
            with self.subTest(word_end=word_end):
                words = [AlignedWord(word='Complete.', start=0, end=word_end, speaker='A')] if word_end else []
                seg = Segment(id=0, start=0, end=4.03, speaker='A', orig_text='Complete.', words=words,
                              alignment_needs_review=False, speaker_needs_review=False)
                script = self.reference_script(seg, source_duration=4)
                self.assertEqual(references.automatic_reference(script, 'A') is not None, allowed)

    @unittest.skipUnless(shutil.which('ffmpeg'), 'requires FFmpeg')
    def test_banquet_stereo_48khz_converts_to_32khz_mono(self):
        seg = Segment(id=0, start=0, end=4, speaker='A', orig_text='Reviewed line.',
                      segment_timing_needs_review=False,
                      alignment_needs_review=False, speaker_needs_review=False)
        script = self.reference_script(seg, source_duration=4, banquet=True)
        source = config.resolve_media_path(script.media.vocals_wav)
        with wave.open(str(source), 'wb') as audio:
            audio.setparams((2, 2, 48000, 0, 'NONE', 'not compressed'))
            audio.writeframes(bytes(4 * 48000 * 2 * 2))
        preview = references.automatic_reference(script, 'A')
        self.assertIsNotNone(preview)
        with wave.open(str(config.resolve_media_path(preview.wav)), 'rb') as audio:
            self.assertEqual((audio.getframerate(), audio.getnchannels(), audio.getnframes()),
                             (32000, 1, 4 * 32000))

    def test_moss_segment_confirmation_allows_whole_reference_not_long_crop(self):
        seg = Segment(id=0, start=0, end=4, speaker='A', orig_text='A full reviewed source line.',
                      alignment_granularity='segment', timing_source='moss_joint_generated',
                      alignment_needs_review=True, speaker_needs_review=False)
        script = self.reference_script(seg, source_duration=12)
        self.assertIsNone(references._reviewed_automatic_reference(script, 'A'))
        seg.segment_timing_needs_review = False
        preview = references._reviewed_automatic_reference(script, 'A')
        self.assertEqual(preview.text, seg.orig_text)
        self.assertEqual(seg.words, [])
        self.assertTrue(seg.alignment_needs_review)
        seg.end = 12
        self.assertIsNone(references._reviewed_automatic_reference(script, 'A'))
        seg.end = 4
        seg.speaker_needs_review = True
        self.assertIsNone(references._reviewed_automatic_reference(script, 'A'))

    def test_old_false_word_review_without_words_is_not_a_reference_approval(self):
        seg = Segment(id=0,start=0,end=4,speaker='A',orig_text='Old source text.',
                      alignment_needs_review=False,speaker_needs_review=False)
        script = self.reference_script(seg,source_duration=4)
        self.assertIsNone(references.automatic_reference(script,'A'))
        seg.segment_timing_needs_review = False
        self.assertIsNotNone(references.automatic_reference(script,'A'))
        seg.segment_timing_needs_review = True
        seg.words = [AlignedWord(word='Old',start=0,end=1,speaker='A'),
                     AlignedWord(word='source',start=1,end=2,speaker='A'),
                     AlignedWord(word='text.',start=2,end=3.9,speaker='A')]
        self.assertIsNotNone(references.automatic_reference(script,'A'))

    def test_missing_source_language_never_uses_target_language_for_reference(self):
        seg = Segment(id=0,start=0,end=4,speaker='A',orig_text='Actual English audio.',target_lang='zh',
                      segment_timing_needs_review=False,speaker_needs_review=False)
        script = self.reference_script(seg,source_duration=4)
        script.source_lang = None
        self.assertIsNone(references.automatic_reference(script,'A'))
        script.source_lang = 'en'
        preview = references.automatic_reference(script,'A')
        self.assertEqual(preview.lang,'en')
        self.assertEqual(preview.text,seg.orig_text)

    def test_unknown_bucket_is_never_used_as_a_reference(self):
        seg = Segment(id=0, start=0, end=4, speaker='UNKNOWN', orig_text='Unknown voice.',
                      alignment_needs_review=False, speaker_needs_review=False)
        script = self.reference_script(seg, source_duration=4)
        self.assertIsNone(references.automatic_reference(script, 'UNKNOWN'))

    def test_same_speaker_overlapping_reference_is_rejected(self):
        seg = Segment(id=0, start=0, end=4, speaker='A', orig_text='Full line.',
                      alignment_needs_review=False, speaker_needs_review=False)
        script = self.reference_script(seg, source_duration=5)
        script.segments.append(Segment(id=1, start=1, end=2, speaker='A', orig_text='Overlap.'))
        self.assertIsNone(references.automatic_reference(script, 'A'))


if __name__ == '__main__':
    unittest.main()
