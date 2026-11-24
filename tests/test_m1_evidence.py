import copy
import unittest

from pydantic import ValidationError

from backend.schemas import (
    EvidenceProvenance, MediaInfo, ProjectScript, Segment, SourceAnnotation,
    SourceEvidence, SourceToken, TranslationUnit,
)
from backend.services.evidence import build_source_evidence


class SourceEvidenceTest(unittest.TestCase):
    def raw(self):
        return [dict(start=0, end=3, text='Hello there.', words=[
            dict(word='Hello', start=0, end=1, score=.8),
            dict(word='there.', start=1.2, end=2.5),
        ])]

    def segments(self):
        return [Segment(id=0, start=0, end=3, speaker='A', orig_text='Hello there.', target_text='你好。')]

    def build(self, **kwargs):
        return build_source_evidence(self.raw(), [dict(start=0, end=3, speaker='A')],
                                     self.segments(), source_id='sha256:fixture', **kwargs)

    def test_legacy_json_defaults_and_openapi_remain_available(self):
        project = ProjectScript.model_validate(dict(project_id='old', media={'duration':3},
                                                   segments=[dict(id=0,start=0,end=3,speaker='A')]))
        self.assertIsNone(project.evidence)
        self.assertEqual(project.segments[0].source_turn_ids, [])
        self.assertIn('SourceEvidence', ProjectScript.model_json_schema()['$defs'])

    def test_overlap_is_retained_with_all_speaker_candidates(self):
        spans = [dict(start=0, end=2, speaker='A'), dict(start=.5, end=3, speaker='B')]
        evidence = build_source_evidence(self.raw(), spans, self.segments(), source_id='sha256:fixture',
                                         exclusive_spans=[dict(start=0,end=3,speaker='A')])
        self.assertEqual(len(evidence.diarization_spans), 2)
        self.assertEqual(len(evidence.source_turns), 2)
        self.assertEqual(len(evidence.exclusive_spans), 1)
        self.assertEqual({item.speaker for item in evidence.source_tokens[0].speaker_candidates}, {'A','B'})
        self.assertIn('multiple_speaker_activity', evidence.source_tokens[0].review.reasons)
        self.assertTrue(all(item.score is None for item in evidence.source_tokens[0].speaker_candidates))
        self.assertEqual(evidence.source_tokens[0].review.score_kind, 'alignment_model_score')

    def test_source_ids_survive_editable_segmentation_translation_and_order(self):
        raw = self.raw()
        before = copy.deepcopy(raw)
        spans = [dict(start=0,end=2,speaker='A'), dict(start=.5,end=3,speaker='B')]
        first = build_source_evidence(raw, spans, self.segments(), source_id='sha256:fixture')
        changed = [Segment(id=8,start=0,end=1,speaker='C',orig_text='Hello',target_text='新译文'),
                   Segment(id=9,start=1.2,end=3,speaker='C',orig_text='there.',emotion='angry')]
        second = build_source_evidence(raw, list(reversed(spans)), changed, source_id='sha256:fixture')
        self.assertEqual([item.id for item in first.source_tokens], [item.id for item in second.source_tokens])
        self.assertEqual([item.id for item in first.source_turns], [item.id for item in second.source_turns])
        self.assertEqual(raw, before)
        self.assertNotEqual(first.source_tokens[0].id,
                            build_source_evidence(raw, spans, [], source_id='sha256:other').source_tokens[0].id)

    def test_assignment_failure_reason_survives_available_activity_spans(self):
        evidence = self.build(fallback_reason='speaker_assignment_missing')
        self.assertTrue(evidence.diarization_spans)
        self.assertIn('speaker_assignment_missing', evidence.review_reasons)
        self.assertTrue(all('speaker_assignment_missing' in token.review.reasons
                            for token in evidence.source_tokens))

    def test_missing_diarization_does_not_trust_asr_speaker_labels(self):
        raw = self.raw()
        raw[0]['speaker'] = 'A'
        raw[0]['words'][0]['speaker'] = 'A'
        evidence = build_source_evidence(raw, None, self.segments(), source_id='source',
                                         fallback_reason='token_not_configured')
        self.assertEqual(evidence.diarization_spans, [])
        self.assertTrue(all(not token.speaker_candidates for token in evidence.source_tokens))
        self.assertTrue(all(turn.speaker is None for turn in evidence.source_turns))
        self.assertIn('token_not_configured', evidence.review_reasons)
        self.assertTrue(all(turn.review.needs_review for turn in evidence.source_turns))

    def test_unaligned_text_is_preserved_without_guessed_timestamps(self):
        evidence = build_source_evidence([dict(start=0,end=3,text='No alignment.')], [], [], source_id='source')
        token, = evidence.source_tokens
        self.assertEqual(token.text, 'No alignment.')
        self.assertIsNone(token.start)
        self.assertIsNone(token.end)
        self.assertIn('alignment_missing_or_invalid', token.review.reasons)
        self.assertEqual(evidence.source_turns[0].source_token_ids, [token.id])

    def test_duplicate_spans_retained_without_double_counting_overlap(self):
        spans = [dict(start=0,end=3,speaker='A')] * 2
        evidence = build_source_evidence(self.raw(), spans, [], source_id='source')
        self.assertEqual(len(evidence.diarization_spans), 2)
        candidate, = evidence.source_tokens[0].speaker_candidates
        self.assertEqual(candidate.overlap_seconds, 1)
        self.assertEqual(len({span.id for span in evidence.diarization_spans}), 2)

    def test_invalid_time_evidence_is_not_converted_into_fake_source_ranges(self):
        raw = self.raw()
        raw[0]['words'][0]['start'] = float('nan')
        evidence = build_source_evidence(raw, [dict(start=0,end=-1,speaker='A')], [], source_id='source')
        self.assertIsNone(evidence.source_tokens[0].start)
        self.assertIn('invalid_diarization_span', evidence.review_reasons)

    def test_segment_links_and_reference_validation(self):
        segments = self.segments()
        provenance = [EvidenceProvenance(stage='asr',model='test',model_version='pinned')]
        evidence = build_source_evidence(self.raw(), [dict(start=0,end=3,speaker='A')], segments,
                                         source_id='source',provenance=provenance)
        project = ProjectScript(project_id='demo',media=MediaInfo(duration=3),segments=segments,evidence=evidence)
        self.assertEqual(project.segments[0].source_token_ids, [token.id for token in evidence.source_tokens])
        self.assertNotEqual(project.segments[0].source_turn_ids, project.segments[0].translation_unit_ids)
        payload = project.model_dump()
        payload['segments'][0]['source_turn_ids'] = ['missing']
        with self.assertRaisesRegex(ValidationError, 'dangling'):
            ProjectScript.model_validate(payload)
        with self.assertRaisesRegex(ValidationError, 'Dangling'):
            SourceEvidence(source_id='source',translation_units=[TranslationUnit(id='unit',source_token_ids=['missing'])])

    def test_impossible_or_partial_utterance_ranges_are_unknown(self):
        for raw in (dict(start=3,end=1,text='Hello'), dict(start=1,text='Hello'),
                    dict(start=float('nan'),end=3,text='Hello')):
            with self.subTest(raw=raw):
                evidence = build_source_evidence([raw], [], [], source_id='source')
                utterance, = evidence.source_utterances
                self.assertIsNone(utterance.start)
                self.assertIsNone(utterance.end)
                self.assertTrue(utterance.review.needs_review)
        from backend.schemas import SourceUtterance
        for values in ({'start':3,'end':1}, {'start':1}, {'end':2}):
            with self.assertRaises(ValidationError):
                SourceUtterance(id='utterance',text='Hello',**values)

    def test_overlapping_word_times_are_retained_as_evidence_and_reviewed(self):
        raw = self.raw()
        raw[0]['words'][1]['start'] = .5
        evidence = build_source_evidence(raw, [], [], source_id='source')
        self.assertEqual(evidence.source_tokens[1].start, .5)
        self.assertIn('overlapping_or_out_of_order_alignment', evidence.source_tokens[1].review.reasons)

    def test_original_asr_text_survives_partial_word_list(self):
        raw = self.raw()
        raw[0]['text'] = 'Hello there. Unaligned material.'
        evidence = build_source_evidence(raw, [], [], source_id='source')
        self.assertEqual(evidence.source_utterances[0].text, raw[0]['text'])
        self.assertIn('asr_text_token_mismatch', evidence.source_utterances[0].review.reasons)

    def test_segment_only_utterance_links_unknown_tokens_without_invented_word_times(self):
        segments = [Segment(id=0,start=0,end=3,speaker='scope_S01',orig_text='MOSS source text.',
                            alignment_granularity='segment',timing_source='moss_joint_generated')]
        evidence = build_source_evidence([dict(start=0,end=3,text='MOSS source text.')],
                                         [dict(start=0,end=3,speaker='scope_S01')], segments,source_id='source')
        self.assertEqual(segments[0].source_token_ids,[evidence.source_tokens[0].id])
        self.assertIsNone(evidence.source_tokens[0].start)
        self.assertTrue(evidence.source_tokens[0].review.needs_review)
        self.assertTrue(segments[0].source_turn_ids)
        segments[0].orig_text = 'Edited source.'
        build_source_evidence([dict(start=0,end=3,text='MOSS source text.')],
                              [dict(start=0,end=3,speaker='scope_S01')], segments,source_id='source')
        self.assertEqual(segments[0].source_token_ids,[])

    def test_joint_generated_activity_and_turns_preserve_unvalidated_review(self):
        provenance = [EvidenceProvenance(stage='diarization',model='MOSS',
                                         parameters={'activity_kind':'joint_generated_derived_timestamps'})]
        evidence = build_source_evidence(self.raw(),[dict(start=0,end=3,speaker='S01',
            review_reasons=['timestamp_clamped_to_source_end'])],[],source_id='source',provenance=provenance)
        for review in (evidence.diarization_spans[0].review,evidence.source_turns[0].review):
            self.assertTrue(review.needs_review)
            self.assertIn('moss_joint_generated_activity_unvalidated',review.reasons)
            self.assertIn('timestamp_clamped_to_source_end',review.reasons)
        legacy = self.build()
        self.assertFalse(legacy.diarization_spans[0].review.needs_review)

    def test_generated_timestamps_and_correction_reason_survive_snapshot(self):
        raw = [dict(start=0,end=30,text='Complete sentence.',generated_start=0,generated_end=30.02,
                    review_reasons=['timestamp_clamped_to_source_end'])]
        evidence = build_source_evidence(raw,[],[],source_id='source')
        utterance, = evidence.source_utterances
        self.assertEqual(utterance.end,30)
        self.assertEqual(utterance.generated_end,30.02)
        self.assertIn('timestamp_clamped_to_source_end',utterance.review.reasons)
        self.assertTrue(utterance.review.needs_review)
        restored = SourceEvidence.model_validate_json(evidence.model_dump_json())
        self.assertEqual(restored.source_utterances[0].generated_end,30.02)

    def test_moss_generation_scope_and_segment_capability_preserved(self):
        evidence = self.build(raw_generation='[0][S01]Hello.[3]', speaker_scope='media-window-1')
        self.assertEqual(evidence.raw_generation, '[0][S01]Hello.[3]')
        self.assertEqual(evidence.speaker_scope, 'media-window-1')
        segment = Segment(id=0,start=0,end=3,speaker='S01',orig_text='Hello.',
                          alignment_granularity='segment',timing_source='moss_joint_generated',
                          segment_timing_needs_review=False,alignment_needs_review=True)
        restored = Segment.model_validate_json(segment.model_dump_json())
        self.assertEqual(restored.words, [])
        self.assertTrue(restored.alignment_needs_review)
        self.assertFalse(restored.segment_timing_needs_review)
        old = Segment(id=1,start=0,end=3,speaker='A')
        self.assertEqual(old.alignment_granularity,'unknown')
        self.assertEqual(old.timing_source,'unknown')
        self.assertTrue(old.segment_timing_needs_review)

    def test_annotations_roundtrip_and_scores_are_nullable_not_fake_neutral(self):
        evidence = self.build()
        self.assertEqual(evidence.annotations, [])
        annotation = SourceAnnotation(id='ann',start=0,end=3,source_turn_ids=[evidence.source_turns[0].id])
        self.assertEqual(annotation.emotions, [])
        self.assertIsNone(annotation.review.score)
        evidence.annotations = [annotation]
        restored = SourceEvidence.model_validate_json(evidence.model_dump_json())
        self.assertEqual(restored.annotations[0].emotions, [])
        with self.assertRaises(ValidationError):
            SourceEvidence(source_id='s',source_tokens=[SourceToken(id='duplicate',text='a'),
                                                       SourceToken(id='duplicate',text='b')])


if __name__ == '__main__':
    unittest.main()
