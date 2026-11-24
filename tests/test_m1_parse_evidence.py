"""Mock-only M1 parse snapshot regression tests; no model or user project loads."""
import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from backend.schemas import ProjectScript
from workers import parse_worker


class ParseEvidenceTest(unittest.TestCase):
    def setUp(self):
        patch.object(parse_worker, "_release_cuda_memory").start()
        self.addCleanup(patch.stopall)

    def test_snapshot_precedes_mutating_word_assignment_and_preserves_overlap(self):
        raw = {"segments": [{"start": 0, "end": 2, "text": "Hello there.", "words": [
            {"word": "Hello", "start": 0, "end": 1, "score": 0.73},
            {"word": "there.", "start": 1, "end": 2}]}]}
        original = copy.deepcopy(raw)
        rows = [{"start": 0, "end": 2, "speaker": "A"}, {"start": 0.5, "end": 1.5, "speaker": "B"}]
        context = {}
        result = {"diarization_spans": rows, "model_revision": "fixture", "source_audio_sha256": "fixture"}
        with tempfile.TemporaryDirectory() as temporary, \
             patch("workers.speaker_pipeline.run_speaker_job", return_value=result):
            audio = Path(temporary) / "dialogue.wav"
            segments = parse_worker._stage_diarize(audio, raw, "cpu", False, context)
        self.assertEqual(raw, original)
        self.assertEqual(context["raw_segments"], original["segments"])
        self.assertEqual(context["diarization_spans"], rows)
        self.assertEqual(context["exclusive_spans"], [])
        self.assertTrue(segments[0]["speaker_needs_review"])
        self.assertEqual(segments[0]["speaker_fallback_reason"], "overlapping_speaker_activity")
        self.assertNotIn("speaker", context["raw_segments"][0]["words"][0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocals = root / "source.wav"
            vocals.write_bytes(b"immutable-source-fixture")
            args = Namespace(project_id="evidence", input=vocals, output_dir=root,
                             mode="video_dubbing", target_lang="zh", skip_llm=True, whisper_model="fixture")
            saved = parse_worker._build_and_save_script(args, 2, vocals, None, segments, "en",
                                                        evidence_context=context)
            script = ProjectScript.model_validate_json(saved.read_text(encoding="utf-8"))
            evidence = script.evidence
            self.assertTrue(evidence.source_id.startswith("sha256:"))
            self.assertEqual(len(evidence.diarization_spans), 2)
            self.assertEqual({turn.speaker for turn in evidence.source_turns}, {"A", "B"})
            self.assertEqual({c.speaker for c in evidence.source_tokens[0].speaker_candidates}, {"A", "B"})
            self.assertTrue(evidence.source_tokens[0].review.needs_review)
            self.assertEqual(evidence.source_tokens[0].review.score, 0.73)
            self.assertTrue(script.segments[0].source_token_ids)
            self.assertTrue(script.segments[0].source_turn_ids)
            self.assertTrue(script.segments[0].translation_unit_ids)
            self.assertTrue(script.segments[0].performance_unit_ids)
            before = evidence.model_dump()
            script.segments[0].target_text = "你好。"
            script.segments[0].tts_wav = "new.wav"
            self.assertEqual(script.evidence.model_dump(), before)
            self.assertEqual(json.loads(saved.read_text(encoding="utf-8"))["evidence"]["exclusive_spans"], [])

    def test_skipped_diarization_snapshot_is_unknown_not_stale_asr_speaker(self):
        raw = {"segments": [{"start": 0, "end": 1, "text": "Wait.", "speaker": "stale",
                             "words": [{"word": "Wait.", "start": 0, "end": 1, "speaker": "stale"}]}]}
        context = {}
        segments = parse_worker._stage_diarize(None, raw, "cpu", True, context)
        self.assertEqual(context["fallback_reason"], "diarization_skipped")
        self.assertEqual(context["raw_segments"], raw["segments"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocals = root / "source.wav"
            vocals.write_bytes(b"fixture")
            args = Namespace(project_id="unknown-evidence", input=vocals, output_dir=root,
                             mode="video_dubbing", target_lang="zh", skip_llm=True)
            saved = parse_worker._build_and_save_script(args, 1, vocals, None, segments, "en",
                                                        evidence_context=context)
            script = ProjectScript.model_validate_json(saved.read_text(encoding="utf-8"))
            self.assertEqual(script.evidence.diarization_spans, [])
            self.assertIn("diarization_skipped", script.evidence.review_reasons)
            self.assertEqual(script.evidence.source_tokens[0].speaker_candidates, [])
            self.assertIsNone(script.evidence.source_turns[0].speaker)
            self.assertEqual(script.segments[0].speaker, "UNKNOWN")
            self.assertEqual(script.segments[0].speaker_fallback_reason, "diarization_skipped")

    def test_sequential_activity_is_not_overlap_and_duplicate_same_speaker_is_not_overlap(self):
        segment = {"start": 0, "end": 2, "speaker_needs_review": False}
        rows = [{"start": 0, "end": 1, "speaker": "A"}, {"start": 1, "end": 2, "speaker": "B"}]
        parse_worker._mark_overlapping_activity([segment], rows)
        self.assertFalse(segment["speaker_needs_review"])
        parse_worker._mark_overlapping_activity([segment], [{"start": 0, "end": 2, "speaker": "A"}] * 2)
        self.assertFalse(segment["speaker_needs_review"])

    def test_detected_language_reaches_success_and_fallback_normalization(self):
        raw = {"segments": [{"start": 0, "end": 1, "text": "待って。",
                             "words": [{"word": "待って。", "start": 0, "end": 1}]}]}
        real_normalize = parse_worker.normalize_segments
        with patch.object(parse_worker, "normalize_segments", wraps=real_normalize) as normalize:
            parse_worker._stage_diarize(None, raw, "cpu", True, {}, language="ja")
            self.assertEqual(normalize.call_args.kwargs["language"], "ja")
        result = {"diarization_spans": [{"start": 0, "end": 1, "speaker": "A"}]}
        with tempfile.TemporaryDirectory() as temporary, \
             patch("workers.speaker_pipeline.run_speaker_job", return_value=result), \
             patch.object(parse_worker, "normalize_segments", wraps=real_normalize) as normalize:
            parse_worker._stage_diarize(Path(temporary) / "audio.wav", raw, "cpu", False, {}, language="ja")
            self.assertEqual(normalize.call_args.kwargs["language"], "ja")

    def test_production_pipeline_passes_detected_language_and_snapshot_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            source.touch()
            output = root / "output"
            args = Namespace(project_id="mock-pipeline", input=source, output_dir=output,
                             device="cpu", skip_diarization=False)
            with patch("workers.speaker_pipeline.run_speaker_job", return_value={}), \
                 patch.object(parse_worker, "_resolve_device", return_value="cpu"), \
                 patch.object(parse_worker, "_extract_wav_and_probe", return_value=1), \
                 patch.object(parse_worker, "_stage_banquet"), \
                 patch.object(parse_worker, "_finish_four_stems", return_value=(source, None, None)), \
                 patch.object(parse_worker, "_stage_whisperx_asr_align", return_value=(None, {"segments": []}, "ja")), \
                 patch.object(parse_worker, "_stage_diarize", return_value=[]) as diarize, \
                 patch.object(parse_worker, "_build_and_save_script", return_value=output / "script.json") as save, \
                 patch.object(parse_worker, "_cleanup_stale_cinematic_dirs"):
                parse_worker.run_pipeline(args)
            self.assertEqual(diarize.call_args.kwargs["language"], "ja")
            self.assertIs(diarize.call_args.kwargs["evidence_context"], save.call_args.kwargs["evidence_context"])


if __name__ == "__main__":
    unittest.main()
