from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock
import wave

from workers.emotion_worker import _load_audio, run_request
from experiments.eval_emotion import compare_request, summarize, validate_result


class EmotionWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / "model"
        self.model.mkdir()
        for name in ("config.yaml", "configuration.json", "model.pt", "tokens.txt"):
            (self.model / name).write_text("test fixture, never a real checkpoint", encoding="utf-8")
        self.audio = self.root / "speech.wav"
        with wave.open(str(self.audio), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(struct.pack("<h", 2000) * 32000)
        self.request = {
            "model": {"id": "emotion2vec/emotion2vec_plus_base", "revision": "abc123",
                      "local_path": str(self.model), "funasr_version": "1.4.28", "license_approved": True},
            "source": {"id": "source-hash", "kind": "dialogue", "audio_path": str(self.audio)},
            "spans": [{"id": "turn1", "start": 0.0, "end": 2.0, "speaker": "A", "source_token_ids": ["t1"]}],
        }

    def test_unauthorized_and_remote_paths_never_load_model(self):
        factory = Mock(side_effect=AssertionError("must not load"))
        for mutation, reason in (({"license_approved": False}, "model_license_not_approved"),
                                 ({"local_path": "emotion2vec/emotion2vec_plus_base"}, "local_model_directory_required_no_download"),
                                 ({"revision": "main"}, "explicit_model_revision_required")):
            request = copy.deepcopy(self.request)
            request["model"].update(mutation)
            result = run_request(request, model_factory=factory)
            self.assertEqual(result["status"], "not_run")
            self.assertIn(reason, result["review_reasons"])
            self.assertEqual(result["annotations"], [])
        factory.assert_not_called()

    def test_missing_dependency_is_not_run_with_no_neutral(self):
        result = run_request(self.request, model_factory=Mock(side_effect=ImportError("funasr")))
        self.assertEqual(result["status"], "not_run")
        self.assertEqual(result["annotations"][0]["labels"], [])
        self.assertIsNone(result["annotations"][0]["selected_label"])

    def test_exact_raw_labels_scores_and_source_links_preserved(self):
        model = Mock()
        model.generate.return_value = [{"labels": ["中立/neutral", "开心/happy"], "scores": [0.17, 0.83]}]
        original = copy.deepcopy(self.request)
        result = run_request(self.request, model_factory=lambda *_: model)
        self.assertEqual(self.request, original)
        self.assertEqual(result["status"], "ok")
        item = result["annotations"][0]
        self.assertEqual(item["labels"], ["中立/neutral", "开心/happy"])
        self.assertEqual(item["scores"], [0.17, 0.83])
        self.assertEqual(item["source_token_ids"], ["t1"])
        self.assertEqual(item["score_kind"], "model_softmax_uncalibrated")
        self.assertTrue(item["needs_review"])
        model.generate.assert_called_once()
        self.assertEqual(model.generate.call_args.kwargs["batch_size"], 1)
        self.assertEqual(len(model.generate.call_args.kwargs["input"]), 32000)

    def test_unknown_overlap_and_review_are_blocked_before_model_load(self):
        factory = Mock(side_effect=AssertionError("must not load"))
        for changes in ({"speaker": "UNKNOWN"}, {"overlap": True}, {"speaker_needs_review": True}, {"alignment_needs_review": True}, {"blocked_reasons": ["source_token_requires_review"]}):
            request = copy.deepcopy(self.request)
            request["spans"][0].update(changes)
            result = run_request(request, model_factory=factory)
            self.assertEqual(result["annotations"][0]["status"], "blocked")
            self.assertEqual(result["annotations"][0]["labels"], [])
        factory.assert_not_called()

    def test_span_bounds_and_too_long_never_silently_crop(self):
        request = copy.deepcopy(self.request)
        request["spans"][0]["end"] = 3.0
        model = Mock()
        result = run_request(request, model_factory=lambda *_: model)
        self.assertEqual(result["annotations"][0]["status"], "error")
        self.assertIn("source_span_outside_audio", result["annotations"][0]["review_reasons"][-1])
        model.generate.assert_not_called()
        request["max_span_seconds"] = 1.0
        factory = Mock()
        result = run_request(request, model_factory=factory)
        self.assertEqual(result["status"], "not_run")
        factory.assert_not_called()

    def test_hash_mismatch_and_invalid_scores_do_not_make_labels(self):
        request = copy.deepcopy(self.request)
        request["model"]["weight_sha256"] = "0" * 64
        factory = Mock()
        result = run_request(request, model_factory=factory)
        self.assertEqual(result["status"], "not_run")
        factory.assert_not_called()
        model = Mock()
        model.generate.return_value = [{"labels": ["happy"], "scores": [float("nan")]}]
        result = run_request(self.request, model_factory=lambda *_: model)
        self.assertEqual(result["annotations"][0]["status"], "error")
        self.assertEqual(result["annotations"][0]["labels"], [])

    def test_download_manifest_revision_cannot_be_relabelled(self):
        (self.model / "download_manifest.json").write_text(json.dumps({
            "model_id": self.request["model"]["id"], "revision": "different-revision", "files": []
        }), encoding="utf-8")
        factory = Mock()
        result = run_request(self.request, model_factory=factory)
        self.assertEqual(result["status"], "not_run")
        self.assertIn("local_manifest_model_revision_mismatch", result["review_reasons"][0])
        factory.assert_not_called()

    def test_stdin_cli_missing_model_is_valid_json_without_importing_ml(self):
        request = copy.deepcopy(self.request)
        request["model"]["local_path"] = str(self.root / "missing")
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "workers" / "emotion_worker.py")],
                                   input=json.dumps(request), capture_output=True, text=True, encoding="utf-8", check=False)
        self.assertEqual(completed.returncode, 0)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "not_run")
        self.assertIn("local_model_directory_required_no_download", result["review_reasons"])

    def test_no_gold_metrics_are_null_and_subset_metrics_are_explicit(self):
        result = {"status": "ok", "source": {"id": "x", "kind": "dialogue"}, "annotations": [
            {"id": "a", "status": "ok", "selected_label": "happy"},
            {"id": "b", "status": "ok", "selected_label": "sad"}]}
        self.assertIsNone(summarize(result)["accuracy"])
        measured = summarize(result, {"labels": {"a": "happy", "b": "happy", "c": "sad"}})
        self.assertEqual(measured["accuracy"], 0.5)
        self.assertEqual(measured["evaluated_spans"], 2)
        self.assertEqual(measured["unevaluated_gold_spans"], ["c"])

    def test_comparison_rejects_wrong_source_duplicate_and_changed_ranges(self):
        valid = {"version": "1.0", "stage": "emotion", "status": "ok",
                 "source": self.request["source"], "model": self.request["model"],
                 "annotations": [{**self.request["spans"][0], "labels": ["happy"], "scores": [1.0], "selected_label": "happy", "status": "ok"}]}
        wrong = copy.deepcopy(valid)
        wrong["source"]["id"] = "another-source"
        with self.assertRaisesRegex(ValueError, "source_mismatch"):
            validate_result(wrong, self.request)
        duplicate = copy.deepcopy(valid)
        duplicate["annotations"].append(copy.deepcopy(duplicate["annotations"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate_annotation"):
            validate_result(duplicate, self.request)
        changed = copy.deepcopy(valid)
        changed["annotations"][0]["end"] = 1.0
        with self.assertRaisesRegex(ValueError, "range_mismatch"):
            validate_result(changed, self.request)
        empty = {**valid, "annotations": []}
        summary = summarize(validate_result(empty, self.request))
        self.assertEqual(summary["requested_spans"], 1)
        self.assertEqual(summary["missing_span_ids"], ["turn1"])

    def test_comparison_is_serial_distinct_sources_no_project_mutation(self):
        request = {"model": self.request["model"], "spans": self.request["spans"], "sources": [
            self.request["source"], {**self.request["source"], "id": "mix", "kind": "original"}]}
        original = copy.deepcopy(request)
        calls = []
        def launch(payload, _python, _timeout):
            calls.append(payload["source"]["id"])
            return {"version": "1.0", "stage": "emotion", "status": "not_run", "source": payload["source"], "model": payload["model"], "annotations": []}
        result = compare_request(request, worker_python="unused", launcher=launch)
        self.assertEqual(calls, ["source-hash", "mix"])
        self.assertEqual(request, original)
        self.assertTrue(all(x["accuracy"] is None for x in result["summaries"]))


if __name__ == "__main__":
    unittest.main()
