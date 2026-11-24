"""DiariZen assignment failures, unknown ownership and saved schema invariants."""
import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from backend import config
from backend.agent import tools
from backend.schemas import CharacterConfig, MediaInfo, ProjectScript, Segment
from workers import parse_worker


def source_segments():
    return [
        {"start": 0.0, "end": 1.0, "text": "Wait", "speaker": "STALE_LABEL",
         "words": [{"word": "Wait", "start": 0.0, "end": 1.0, "speaker": "STALE_LABEL"}]},
        {"start": 1.1, "end": 2.0, "text": "here.", "speaker": "STALE_LABEL",
         "words": [{"word": "here.", "start": 1.1, "end": 2.0, "speaker": "STALE_LABEL"}]},
    ]


def worker_result(spans):
    return {"status": "completed", "diarization_spans": spans, "model_revision": "fixed-model",
            "source_audio_sha256": "fixture"}


class DiarizationFallbackTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.audio = Path(self.temporary.name) / "dialogue.wav"
        self.audio.touch()
        patch.object(parse_worker, "_release_cuda_memory").start()
        self.addCleanup(patch.stopall)

    def assert_saved_unknown(self, segments, reason):
        self.assertEqual(len(segments), 2)
        for seg in segments:
            self.assertEqual(seg["speaker"], "UNKNOWN")
            self.assertTrue(seg["speaker_needs_review"])
            self.assertEqual(seg["speaker_fallback_reason"], reason)
            self.assertTrue(all(word["speaker"] is None for word in seg["words"]))
        args = Namespace(project_id="m0-test", input=self.audio, output_dir=self.audio.parent,
                         mode="video_dubbing", target_lang="zh", skip_llm=True)
        saved = parse_worker._build_and_save_script(args, 2, self.audio, None, segments, "en")
        script = ProjectScript.model_validate_json(saved.read_text())
        self.assertEqual(script.characters["UNKNOWN"].display_name, "待确认说话人")
        self.assertTrue(all(seg.speaker_fallback_reason == reason for seg in script.segments))

    def test_skip_preserves_unknown_reason_and_source_boundaries(self):
        original = {"segments": source_segments()}
        untouched = copy.deepcopy(original)
        with patch("workers.speaker_pipeline.run_speaker_job") as run:
            result = parse_worker._stage_diarize(self.audio, original, "cuda", True)
        self.assert_saved_unknown(result, "diarization_skipped")
        self.assertEqual(original, untouched)
        run.assert_not_called()

    def test_missing_configuration_fails_explicitly_without_loading_old_model(self):
        with patch.object(config, "ENV_DIARIZATION_PYTHON", ""):
            with self.assertRaisesRegex(ValueError, "ENV_DIARIZATION_PYTHON"):
                parse_worker._stage_diarize(self.audio, {"segments": source_segments()}, "cpu", False)

    def test_failed_inference_does_not_publish_a_report_or_retry_other_models(self):
        with patch("workers.speaker_pipeline.run_speaker_job", side_effect=RuntimeError("model failed")) as run:
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                parse_worker._stage_diarize(self.audio, {"segments": source_segments()}, "cuda", False)
        self.assertEqual(run.call_count, 1)
        self.assertFalse((self.audio.parent / "speaker_analysis.json").exists())

    def test_empty_activity_remains_unknown_and_is_saved(self):
        with patch("workers.speaker_pipeline.run_speaker_job", return_value=worker_result([])):
            result = parse_worker._stage_diarize(self.audio, {"segments": source_segments()}, "cpu", False)
        self.assert_saved_unknown(result, "speaker_assignment_missing")

    def test_success_needs_no_hf_token_and_clears_stale_labels(self):
        raw = {"segments": source_segments()}
        untouched = copy.deepcopy(raw)
        spans = [{"start": 0, "end": 1, "speaker": "A"}, {"start": 1.1, "end": 2, "speaker": "B"}]
        with patch("workers.speaker_pipeline.run_speaker_job", return_value=worker_result(spans)), \
             patch.object(config, "is_hf_token_configured", return_value=False):
            result = parse_worker._stage_diarize(self.audio, raw, "cpu", False)
        self.assertEqual([s["speaker"] for s in result], ["A", "B"])
        self.assertTrue(all(not s["speaker_needs_review"] for s in result))
        self.assertEqual(raw, untouched)
        self.assertTrue((self.audio.parent / "speaker_analysis.json").is_file())

    def test_aligned_words_split_a_real_speaker_change_without_losing_text(self):
        raw = {"segments": [{"start": 0, "end": 2, "text": "Wait here.", "words": [
            {"word": "Wait", "start": 0, "end": 1}, {"word": "here.", "start": 1, "end": 2}]}]}
        spans = [{"start": 0, "end": 1, "speaker": "A"}, {"start": 1, "end": 2, "speaker": "B"}]
        with patch("workers.speaker_pipeline.run_speaker_job", return_value=worker_result(spans)):
            result = parse_worker._stage_diarize(self.audio, raw, "cpu", False)
        self.assertEqual([s["speaker"] for s in result], ["A", "B"])
        self.assertEqual([s["orig_text"] for s in result], ["Wait", "here."])

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
