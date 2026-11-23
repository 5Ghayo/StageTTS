"""M0 regressions: diarization failures must survive script saving as unknown evidence."""

import copy
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

from backend import config
from backend.agent import tools
from backend.schemas import CharacterConfig, MediaInfo, ProjectScript, Segment
from workers import parse_worker


def source_segments():
    # Adjacent short lines have no evidence that they belong to the same person.
    return [
        {"start": 0.0, "end": 1.0, "text": "Wait", "speaker": "STALE_LABEL",
         "words": [{"word": "Wait", "start": 0.0, "end": 1.0, "speaker": "STALE_LABEL"}]},
        {"start": 1.1, "end": 2.0, "text": "here.", "speaker": "STALE_LABEL",
         "words": [{"word": "here.", "start": 1.1, "end": 2.0, "speaker": "STALE_LABEL"}]},
    ]


def fake_diarization(pipeline, assigned=None):
    module = ModuleType("whisperx.diarize")
    module.DiarizationPipeline = pipeline
    module.assign_word_speakers = MagicMock(return_value=assigned)
    return module


class DiarizationFallbackTest(unittest.TestCase):
    def setUp(self):
        self.compat = patch.object(parse_worker, "_enable_pyannote_torch_load_compat").start()
        self.release = patch.object(parse_worker, "_release_cuda_memory").start()
        self.addCleanup(patch.stopall)

    def assert_saved_unknown(self, segments, reason):
        self.assertEqual(len(segments), 2, "unknown ownership must not merge source lines")
        self.assertEqual([seg["id"] for seg in segments], [0, 1])
        self.assertEqual([seg["orig_text"] for seg in segments], ["Wait", "here."])
        for seg in segments:
            self.assertEqual(seg["speaker"], "UNKNOWN")
            self.assertTrue(seg["speaker_needs_review"])
            self.assertEqual(seg["speaker_fallback_reason"], reason)
            self.assertTrue(all(word["speaker"] is None for word in seg["words"]))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocals = root / "dialogue.wav"
            vocals.touch()
            args = Namespace(project_id="m0-test", input=vocals, output_dir=root,
                             mode="video_dubbing", target_lang="zh", skip_llm=True)
            saved = parse_worker._build_and_save_script(args, 2.0, vocals, None, segments, "en")
            script = ProjectScript.model_validate_json(saved.read_text(encoding="utf-8"))
            self.assertEqual(script.characters["UNKNOWN"].display_name, "待确认说话人")
            for seg in script.segments:
                self.assertEqual(seg.speaker, "UNKNOWN")
                self.assertTrue(seg.speaker_needs_review)
                self.assertEqual(seg.speaker_fallback_reason, reason)
            # Persisted JSON is the actual contract; no reliance on ignored extra fields.
            data = json.loads(saved.read_text(encoding="utf-8"))
            self.assertEqual(data["segments"][0]["speaker_fallback_reason"], reason)

    def test_skip_preserves_unknown_reason_and_source_boundaries(self):
        original = {"segments": source_segments()}
        untouched = copy.deepcopy(original)
        with patch.object(parse_worker.config, "is_hf_token_configured", return_value=True):
            result = parse_worker._stage_diarize(None, original, "cuda", True)
        self.assert_saved_unknown(result, "diarization_skipped")
        self.assertEqual(original, untouched)
        self.compat.assert_not_called()

    def test_no_token_persists_unknown(self):
        with patch.object(parse_worker.config, "is_hf_token_configured", return_value=False):
            result = parse_worker._stage_diarize(None, {"segments": source_segments()}, "cuda", False)
        self.assert_saved_unknown(result, "hf_token_missing")
        self.compat.assert_not_called()

    def test_import_failure_persists_unknown(self):
        with patch.object(parse_worker.config, "is_hf_token_configured", return_value=True), \
             patch.dict(sys.modules, {"whisperx.diarize": None}):
            result = parse_worker._stage_diarize(None, {"segments": source_segments()}, "cuda", False)
        self.assert_saved_unknown(result, "diarization_import_failed")

    def test_both_device_inference_failures_persist_unknown(self):
        model = MagicMock(side_effect=RuntimeError("mock inference failed"))
        pipeline = MagicMock(return_value=model)
        module = fake_diarization(pipeline)
        with patch.object(parse_worker.config, "is_hf_token_configured", return_value=True), \
             patch.dict(sys.modules, {"whisperx.diarize": module}):
            result = parse_worker._stage_diarize(None, {"segments": source_segments()}, "cuda", False)
        self.assertEqual([call.kwargs["device"] for call in pipeline.call_args_list], ["cuda", "cpu"])
        self.assertEqual(self.release.call_count, 2)
        module.assign_word_speakers.assert_not_called()
        self.assert_saved_unknown(result, "diarization_failed")

    def test_empty_assignment_is_not_successful_single_speaker(self):
        module = fake_diarization(MagicMock(return_value=MagicMock()), {"segments": []})
        with patch.object(parse_worker.config, "is_hf_token_configured", return_value=True), \
             patch.dict(sys.modules, {"whisperx.diarize": module}):
            result = parse_worker._stage_diarize(None, {"segments": source_segments()}, "cpu", False)
        self.assert_saved_unknown(result, "speaker_assignment_missing")

    def test_success_preserves_confirmed_and_missing_word_evidence(self):
        assigned = {"segments": [
            {"start": 0.0, "end": 2.0, "text": "Hello there.", "speaker": "A",
             "words": [{"word": "Hello", "start": 0.0, "end": 0.8, "speaker": "A"},
                       {"word": "there.", "start": 0.9, "end": 2.0, "speaker": "A"}]},
            {"start": 3.0, "end": 5.0, "text": "Not certain.", "speaker": "B",
             "words": [{"word": "Not", "start": 3.0, "end": 3.8, "speaker": "B"},
                       {"word": "certain.", "start": 3.9, "end": 5.0}]},
            {"start": 6.0, "end": 7.0, "text": "Who?",
             "words": [{"word": "Who?", "start": 6.0, "end": 7.0}]},
        ]}
        module = fake_diarization(MagicMock(return_value=MagicMock()), assigned)
        with patch.object(parse_worker.config, "is_hf_token_configured", return_value=True), \
             patch.dict(sys.modules, {"whisperx.diarize": module}):
            result = parse_worker._stage_diarize("mock-audio", {"segments": source_segments()}, "cuda", False)
        self.assertEqual([seg["speaker"] for seg in result], ["A", "B", "UNKNOWN"])
        self.assertFalse(result[0]["speaker_needs_review"])
        self.assertTrue(result[1]["speaker_needs_review"])
        self.assertIsNone(result[1]["words"][1]["speaker"])
        self.assertEqual(result[1]["speaker_fallback_reason"], "word_speaker_missing")
        self.assertEqual(result[2]["speaker_fallback_reason"], "speaker_assignment_missing")
        self.assertTrue(result[2]["speaker_needs_review"])
        self.assertFalse(module.assign_word_speakers.call_args.kwargs["fill_nearest"])
        # Successful-path review metadata also survives actual Pydantic JSON saving.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vocals = root / "dialogue.wav"
            vocals.touch()
            args = Namespace(project_id="m0-success", input=vocals, output_dir=root,
                             mode="video_dubbing", target_lang="zh", skip_llm=True)
            saved = parse_worker._build_and_save_script(args, 7, vocals, None, result, "en")
            script = ProjectScript.model_validate_json(saved.read_text(encoding="utf-8"))
            self.assertIsNone(script.segments[0].speaker_fallback_reason)
            self.assertEqual(script.segments[1].speaker_fallback_reason, "word_speaker_missing")
            self.assertEqual(script.characters["UNKNOWN"].display_name, "待确认说话人")

    def test_success_still_merges_confirmed_same_speaker_fragments(self):
        assigned = {"segments": source_segments()}
        for raw in assigned["segments"]:
            raw["speaker"] = "A"
            raw["words"][0]["speaker"] = "A"
        module = fake_diarization(MagicMock(return_value=MagicMock()), assigned)
        with patch.object(parse_worker.config, "is_hf_token_configured", return_value=True), \
             patch.dict(sys.modules, {"whisperx.diarize": module}):
            result = parse_worker._stage_diarize(None, {"segments": source_segments()}, "cpu", False)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["orig_text"], "Wait here.")
        self.assertEqual(result[0]["speaker"], "A")
        self.assertFalse(result[0]["speaker_needs_review"])

    def test_cpu_retry_success_is_not_marked_as_total_failure(self):
        assigned = {"segments": [{**source_segments()[0], "speaker": "A",
                                 "words": [{"word": "Wait", "start": 0, "end": 1, "speaker": "A"}]}]}
        pipeline = MagicMock(side_effect=[RuntimeError("mock CUDA failure"), MagicMock()])
        module = fake_diarization(pipeline, assigned)
        with patch.object(parse_worker.config, "is_hf_token_configured", return_value=True), \
             patch.dict(sys.modules, {"whisperx.diarize": module}):
            result = parse_worker._stage_diarize(None, {"segments": source_segments()}, "cuda", False)
        self.assertEqual(result[0]["speaker"], "A")
        self.assertFalse(result[0]["speaker_needs_review"])
        self.assertNotIn("speaker_fallback_reason", result[0])

    def test_unknown_cannot_be_confirmed_by_schema_or_saved_mutation(self):
        seg = Segment(id=0, start=0, end=1, speaker="UNKNOWN", speaker_needs_review=False)
        self.assertTrue(seg.speaker_needs_review)
        self.assertEqual(seg.speaker_fallback_reason, "speaker_assignment_missing")
        script = ProjectScript(project_id="invariant", media=MediaInfo(duration=1),
                               characters={"UNKNOWN": CharacterConfig(display_name="待确认说话人")},
                               segments=[seg])
        seg.speaker_needs_review = False  # Simulate an existing Python caller mutating without validation.
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "WORKSPACE_DIR", Path(directory)):
            tools.save_project_script(script)
            restored = tools.load_project_script("invariant")
            self.assertTrue(restored.segments[0].speaker_needs_review)
            data = json.loads((Path(directory) / "invariant" / "script.json").read_text(encoding="utf-8"))
            self.assertTrue(data["segments"][0]["speaker_needs_review"])

    def test_agent_patch_preserves_reason_until_known_speaker_assignment(self):
        script = ProjectScript(
            project_id="patches", media=MediaInfo(duration=1),
            characters={"UNKNOWN": CharacterConfig(display_name="待确认说话人"),
                        "A": CharacterConfig(display_name="角色 A")},
            segments=[Segment(id=0, start=0, end=1, speaker="UNKNOWN", orig_text="Hi.",
                              speaker_fallback_reason="hf_token_missing")],
        )
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "WORKSPACE_DIR", Path(directory)):
            tools.save_project_script(script)
            tools.batch_update_segments("patches", [{"id": 0, "orig_text": "Hello.",
                                                     "speaker_needs_review": False}])
            seg = tools.load_project_script("patches").segments[0]
            self.assertTrue(seg.speaker_needs_review)
            self.assertEqual(seg.speaker_fallback_reason, "hf_token_missing")
            tools.batch_update_segments("patches", [{"id": 0, "speaker": "UNKNOWN",
                                                     "speaker_needs_review": False}])
            seg = tools.load_project_script("patches").segments[0]
            self.assertTrue(seg.speaker_needs_review)
            self.assertEqual(seg.speaker_fallback_reason, "hf_token_missing")
            tools.batch_update_segments("patches", [{"id": 0, "speaker": "A"}])
            seg = tools.load_project_script("patches").segments[0]
            self.assertEqual(seg.speaker, "A")
            self.assertFalse(seg.speaker_needs_review)
            self.assertIsNone(seg.speaker_fallback_reason)

    def test_older_script_segment_remains_compatible(self):
        seg = Segment(id=0, start=0, end=1, speaker="A", orig_text="Hi.")
        self.assertIsNone(seg.speaker_fallback_reason)


if __name__ == "__main__":
    unittest.main()
