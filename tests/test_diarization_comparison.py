"""Offline comparison protocol checks with synthetic annotations and mock subprocesses."""
import importlib.metadata
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from experiments import eval_diarization
from workers import diarization_worker


class Annotation:
    def __init__(self, rows):
        self.rows = rows
    def itertracks(self, yield_label=False):
        for index, (start, end, speaker) in enumerate(self.rows):
            yield SimpleNamespace(start=start, end=end), index, speaker


class DiarizationComparisonTest(unittest.TestCase):
    def request(self, root, **updates):
        model = root / "pipeline"
        model.mkdir(exist_ok=True)
        (model / "config.yaml").write_text("pipeline: {}\n", encoding="utf-8")
        audio = root / "audio.wav"
        audio.touch()
        return {"protocol_version": "1.0", "backend": "community-1", "source_id": "fixture",
                "audio_source": "dialogue", "audio_path": str(audio.resolve()), "model_path": str(model.resolve()),
                "expected_package_version": "4.0.1", "device": "cpu", **updates}

    def test_community_full_overlap_retained_exclusive_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.request(Path(directory))
            output = SimpleNamespace(speaker_diarization=Annotation([(0, 2, "A"), (0.5, 1.5, "B")]),
                                     exclusive_speaker_diarization=Annotation([(0, 2, "A")]))
            with patch.object(importlib.metadata, "version", return_value="4.0.1"), \
                 patch.object(diarization_worker, "_load_pipeline", return_value=MagicMock(return_value=output)):
                result = diarization_worker.run_request(request)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(result["diarization_spans"]), 2)
        self.assertEqual(len(result["exclusive_spans"]), 1)
        self.assertTrue(result["exclusive_is_derived"])
        self.assertIsNone(result["diarization_spans"][0]["score"])

    def test_legacy_annotation_and_version_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.request(Path(directory), backend="legacy-3.1", expected_package_version="3.4.0")
            with patch.object(importlib.metadata, "version", return_value="3.4.0"), \
                 patch.object(diarization_worker, "_load_pipeline", return_value=MagicMock(return_value=Annotation([(0, 1, "A")]))):
                result = diarization_worker.run_request(request)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["exclusive_spans"], [])

    def test_local_directory_resolves_config_for_actual_legacy_loader_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = self.request(root, backend="legacy-3.1", expected_package_version="3.4.0")
            config_path = root / "pipeline" / "config.yaml"
            self.assertEqual(diarization_worker.local_pipeline_config(root / "pipeline"), config_path)
            self.assertEqual(diarization_worker.local_pipeline_config(config_path), config_path)
            with patch.object(importlib.metadata, "version", return_value="3.4.0"), \
                 patch.object(diarization_worker, "_load_pipeline", return_value=MagicMock(return_value=Annotation([]))) as loader:
                result = diarization_worker.run_request(request)
            loader.assert_called_once_with(config_path)
            self.assertEqual(result["hash_scope"], "pipeline_config_only_not_model_weights")
            self.assertEqual(len(result["pipeline_config_sha256"]), 64)
            config_path.unlink()
            with self.assertRaises(ValueError):
                diarization_worker.local_pipeline_config(root / "pipeline")

    def test_missing_dependency_or_wrong_version_is_not_run(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.request(Path(directory))
            with patch.object(importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError):
                self.assertEqual(diarization_worker.run_request(request)["status"], "not_run")
            with patch.object(importlib.metadata, "version", return_value="3.4.0"), \
                 patch.object(diarization_worker, "_load_pipeline") as loader:
                result = diarization_worker.run_request(request)
                self.assertEqual(result["status"], "not_run")
                loader.assert_not_called()

    def test_hub_id_refused_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            request = self.request(Path(directory), model_path="pyannote/speaker-diarization-community-1")
            with patch.object(diarization_worker, "_load_pipeline") as loader:
                result = diarization_worker.run_request(request)
            self.assertEqual(result["status"], "not_run")
            loader.assert_not_called()

    def test_no_gold_never_reports_zero_der_or_imports_metric(self):
        with patch.object(eval_diarization.importlib, "import_module") as importer:
            report = eval_diarization.evaluate_der({"status": "completed"}, None, uri="fixture")
        self.assertEqual(report["status"], "not_evaluated")
        self.assertIsNone(report["der"])
        importer.assert_not_called()

    def test_optional_metric_missing_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            gold = Path(directory) / "gold.rttm"
            gold.touch()
            with patch.object(eval_diarization.importlib, "import_module", side_effect=ImportError):
                report = eval_diarization.evaluate_der({"status": "completed"}, gold, uri="fixture")
        self.assertEqual(report["status"], "not_evaluated")
        self.assertIn("optional", report["reason"])

    def test_der_uses_original_activity_and_declares_scoring_policy(self):
        tracks = []
        class FakeAnnotation:
            def __init__(self, uri):
                pass
            def __setitem__(self, key, value):
                tracks.append((key, value))
        metric = MagicMock(return_value={"total": 2.0, "diarization error rate": 0.25,
                                         "confusion": 0.5, "missed detection": 0.0, "false alarm": 0.0})
        constructor = MagicMock(return_value=metric)
        modules = {"pyannote.metrics.diarization": SimpleNamespace(DiarizationErrorRate=constructor),
                   "pyannote.core": SimpleNamespace(Annotation=FakeAnnotation, Segment=lambda start, end: (start, end)),
                   "pyannote.database.util": SimpleNamespace(load_rttm=lambda path: {"fixture": Annotation([(0, 2, "A")])})}
        with tempfile.TemporaryDirectory() as directory:
            gold = Path(directory) / "gold.rttm"
            gold.touch()
            result = {"status": "completed", "diarization_spans": [{"start": 0, "end": 2, "speaker": "A"},
                       {"start": 0.5, "end": 1.5, "speaker": "B"}], "exclusive_spans": []}
            with patch.object(eval_diarization.importlib, "import_module", side_effect=lambda name: modules[name]):
                report = eval_diarization.evaluate_der(result, gold, uri="fixture")
        self.assertEqual(report["der"], 0.25)
        self.assertEqual(report["der_unit"], "fraction")
        self.assertEqual(len(tracks), 2)
        constructor.assert_called_once_with(collar=0.0, skip_overlap=False)

    def test_failed_rerun_does_not_consume_stale_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "raw.wav"
            audio.touch()
            output = root / "results"
            output.mkdir()
            stale = output / "sample-0-raw-worker-0.result.json"
            stale.write_text(json.dumps({"protocol_version": "1.0", "backend": "legacy-3.1", "source_id": "fixture",
                                         "audio_source": "raw", "status": "completed", "diarization_spans": []}), encoding="utf-8")
            manifest = {"protocol_version": "1.0", "samples": [{"id": "fixture", "raw_audio": str(audio.resolve())}],
                        "workers": [{"backend": "legacy-3.1", "python": sys.executable,
                                     "model_path": str(root.resolve()), "package_version": "3.4.0"}]}
            with patch.object(eval_diarization.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stderr="")):
                report = eval_diarization.run_comparison(manifest, output)
            self.assertEqual(report["comparisons"][0]["result"]["status"], "error")
            self.assertEqual(report["comparisons"][0]["evaluation"]["status"], "not_evaluated")

    def test_runner_sequential_raw_dialogue_and_worker_identity_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw, dialogue = root / "raw.wav", root / "dialogue.wav"
            raw.touch()
            dialogue.touch()
            manifest = {"protocol_version": "1.0", "samples": [{"id": "fixture", "raw_audio": str(raw.resolve()),
                         "dialogue_audio": str(dialogue.resolve())}], "workers": [{"backend": "legacy-3.1",
                         "python": sys.executable, "model_path": str(root.resolve()), "package_version": "3.4.0"}]}
            requests = []
            def run(command, **kwargs):
                request = json.loads(Path(command[3]).read_text(encoding="utf-8"))
                requests.append(request)
                Path(command[5]).write_text(json.dumps({**request, "status": "completed", "diarization_spans": []}), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, stderr="")
            with patch.object(eval_diarization.subprocess, "run", side_effect=run):
                result = eval_diarization.run_comparison(manifest, root / "results")
            self.assertEqual([item["audio_source"] for item in requests], ["raw", "dialogue"])
            self.assertTrue(all(item["evaluation"]["status"] == "not_evaluated" for item in result["comparisons"]))
            self.assertTrue((root / "results" / "comparison.json").is_file())


if __name__ == "__main__":
    unittest.main()
