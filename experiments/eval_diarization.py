"""Offline legacy-3.1/community-1 × raw/dialogue comparison.

Manifest JSON: protocol_version=1.0, samples=[{id, raw_audio?, dialogue_audio?,
gold_rttm?, uem?}], workers=[{backend, python, model_path, package_version,
device?}]. Every path is an explicit local file. Models/environments must already
be prepared with their licenses accepted by the user. Runs serially. No ASR/WER
is claimed by this diarization-only experiment. No gold RTTM means not_evaluated.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKER = ROOT / "workers" / "diarization_worker.py"


def evaluate_der(result: dict, gold_rttm: Path | None, *, uri: str,
                 collar: float = 0.0, skip_overlap: bool = False, uem: Path | None = None) -> dict:
    evaluation = {"status": "not_evaluated", "der": None, "reason": None,
                  "collar_seconds": collar, "skip_overlap": skip_overlap,
                  "gold_rttm": str(gold_rttm) if gold_rttm else None}
    if not math.isfinite(collar) or collar < 0:
        evaluation["reason"] = "collar must be finite and nonnegative"
        return evaluation
    if result.get("status") != "completed":
        evaluation["reason"] = "diarization_not_completed"
        return evaluation
    if gold_rttm is None:
        evaluation["reason"] = "gold_rttm_missing; no DER claimed"
        return evaluation
    if not gold_rttm.is_file():
        evaluation["reason"] = "gold_rttm_file_missing"
        return evaluation
    try:
        metrics = importlib.import_module("pyannote.metrics.diarization")
        core = importlib.import_module("pyannote.core")
        utils = importlib.import_module("pyannote.database.util")
    except ImportError:
        evaluation["reason"] = "optional pyannote.metrics/core/database unavailable in comparison interpreter"
        return evaluation
    try:
        references = utils.load_rttm(str(gold_rttm))
        if uri not in references:
            raise ValueError(f"gold RTTM has no matching URI: {uri}")
        reference = references[uri]
        if not list(reference.itertracks(yield_label=True)):
            raise ValueError("gold RTTM has no speaker activity")
        hypothesis = core.Annotation(uri=uri)
        for index, span in enumerate(result.get("diarization_spans", [])):
            start, end = float(span["start"]), float(span["end"])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                raise ValueError("invalid hypothesis activity times")
            hypothesis[core.Segment(start, end), str(index)] = str(span["speaker"])
        timeline = None
        if uem is not None:
            if not uem.is_file():
                raise ValueError("UEM file missing")
            timeline = utils.load_uem(str(uem)).get(uri)
            if timeline is None:
                raise ValueError(f"UEM has no matching URI: {uri}")
        metric = metrics.DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
        details = metric(reference, hypothesis, uem=timeline, detailed=True)
        denominator = float(details.get("total", 0))
        if denominator <= 0:
            raise ValueError("gold reference has zero scored speaker time")
        der = float(details["diarization error rate"])
        if not math.isfinite(der):
            raise ValueError("metric returned nonfinite DER")
        evaluation.update(status="evaluated", der=der, reason=None,
                          details={str(key): float(value) for key, value in details.items()},
                          der_unit="fraction")
    except Exception as exc:
        evaluation["reason"] = f"DER evaluation failed: {type(exc).__name__}: {exc}"
    return evaluation


def _local_file(value: str, name: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{name} must be an existing absolute local file")
    return path


def run_comparison(manifest: dict, output_dir: Path, *, collar: float = 0.0,
                   skip_overlap: bool = False, timeout_seconds: float = 3600) -> dict:
    if manifest.get("protocol_version") != "1.0":
        raise ValueError("manifest protocol_version must be 1.0")
    workers, samples = manifest.get("workers"), manifest.get("samples")
    if not isinstance(workers, list) or not workers or not isinstance(samples, list) or not samples:
        raise ValueError("manifest requires nonempty workers and samples lists")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    for sample_index, sample in enumerate(samples):
        uri = str(sample.get("id") or "")
        if not uri:
            raise ValueError("each sample requires an id matching the gold RTTM URI")
        for audio_source in ("raw", "dialogue"):
            audio_value = sample.get(f"{audio_source}_audio")
            if not audio_value:
                continue
            audio_path = _local_file(audio_value, f"{audio_source}_audio")
            for worker_index, worker in enumerate(workers):
                interpreter = _local_file(worker["python"], "worker python")
                run_id = f"sample-{sample_index}-{audio_source}-worker-{worker_index}"
                request_path, result_path = output_dir / f"{run_id}.request.json", output_dir / f"{run_id}.result.json"
                request = {"protocol_version": "1.0", "backend": worker["backend"], "source_id": uri,
                           "audio_source": audio_source, "audio_path": str(audio_path),
                           "model_path": worker["model_path"],
                           "expected_package_version": worker["package_version"],
                           "device": worker.get("device", "cpu")}
                request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2), encoding="utf-8")
                # A failed rerun must never consume a previous experiment's result.
                result_path.unlink(missing_ok=True)
                try:
                    process = subprocess.run([str(interpreter), str(WORKER), "--request", str(request_path),
                                              "--output", str(result_path)], capture_output=True, text=True,
                                             encoding="utf-8", errors="replace", timeout=timeout_seconds)
                    if process.returncode != 0 or not result_path.is_file():
                        raise RuntimeError(f"worker exit {process.returncode}: {process.stderr[-1500:]}")
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if (result.get("protocol_version") != "1.0" or result.get("backend") != worker["backend"]
                            or result.get("source_id") != uri or result.get("audio_source") != audio_source):
                        raise ValueError("worker result identity/protocol mismatch")
                except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    result = {"status": "error", "reason": str(exc), "diarization_spans": []}
                gold = Path(sample["gold_rttm"]) if sample.get("gold_rttm") else None
                uem = Path(sample["uem"]) if sample.get("uem") else None
                comparisons.append({"sample_id": uri, "audio_source": audio_source,
                                    "backend": worker["backend"], "request": request, "result": result,
                                    "evaluation": evaluate_der(result, gold, uri=uri, collar=collar,
                                                               skip_overlap=skip_overlap, uem=uem)})
    report = {"protocol_version": "1.0", "comparisons": comparisons,
              "scoring": {"collar_seconds": collar, "skip_overlap": skip_overlap},
              "note": "DER requires gold RTTM; raw/exclusive tracks are not interchangeable. No model ranking without scored gold."}
    (output_dir / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--collar", type=float, default=0.0)
    parser.add_argument("--skip-overlap", action="store_true", help="Nondefault metric variant; reported explicitly")
    args = parser.parse_args()
    report = run_comparison(json.loads(args.manifest.read_text(encoding="utf-8")), args.output_dir.resolve(),
                            collar=args.collar, skip_overlap=args.skip_overlap)
    print(json.dumps({"runs": len(report["comparisons"]), "output": str(args.output_dir / "comparison.json")}))


if __name__ == "__main__":
    main()
