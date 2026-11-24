"""Isolated, offline diarization experiment worker (JSON protocol version 1.0).

This worker never changes the production 3.1 pipeline or downloads checkpoints.
Run with the Python interpreter of the chosen, separately prepared environment.
Both legacy-3.1 (pyannote.audio 3.x) and community-1 (4.x) require an explicit
local pipeline path and the exact installed pyannote.audio version in the request.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "1.0"
BACKENDS = {"legacy-3.1": 3, "community-1": 4}


def _validate_request(request: dict) -> dict:
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("protocol_version must be 1.0")
    backend = request.get("backend")
    if backend not in BACKENDS:
        raise ValueError("backend must be legacy-3.1 or community-1")
    if request.get("device", "cpu") not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if request.get("audio_source") not in {"raw", "dialogue"}:
        raise ValueError("audio_source must explicitly be raw or dialogue")
    if not isinstance(request.get("source_id"), str) or not request["source_id"].strip():
        raise ValueError("source_id is required")
    if not isinstance(request.get("expected_package_version"), str) or not request["expected_package_version"].strip():
        raise ValueError("expected_package_version is required; prepare the isolated environment first")
    model_path = Path(request.get("model_path", ""))
    audio_path = Path(request.get("audio_path", ""))
    if not model_path.is_absolute() or not model_path.exists():
        raise ValueError("model_path must be an existing absolute local pipeline path; no hub IDs/downloads")
    if not audio_path.is_absolute() or not audio_path.is_file():
        raise ValueError("audio_path must be an existing absolute local audio file")
    return request


def annotation_spans(annotation: Any) -> list[dict]:
    """Serialize every activity track; sorting must never collapse overlaps."""
    spans = []
    for turn, track, speaker in annotation.itertracks(yield_label=True):
        start, end = float(turn.start), float(turn.end)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("diarizer returned an invalid activity time range")
        spans.append({"start": start, "end": end, "speaker": str(speaker),
                      "track": str(track), "score": None, "score_kind": None})
    return sorted(spans, key=lambda span: (span["start"], span["end"], span["speaker"], span["track"]))


def local_pipeline_config(model_path: Path) -> Path:
    """Both supported APIs accept a config file; legacy 3.x does not accept dirs."""
    config_path = model_path / "config.yaml" if model_path.is_dir() else model_path
    if not config_path.is_file():
        raise ValueError("local pipeline must contain an existing config.yaml")
    return config_path


def _load_pipeline(model_path: Path):
    # Documented offline modes: nested HF model references may use already cached
    # artifacts, but missing files must fail instead of silently contacting the hub.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from pyannote.audio import Pipeline
    return Pipeline.from_pretrained(str(model_path))


def run_request(request: dict) -> dict:
    """Return explicit not_run/error states; absence of a model never means one speaker."""
    result = {"protocol_version": PROTOCOL_VERSION, "status": "not_run",
              "backend": request.get("backend"), "source_id": request.get("source_id"),
              "audio_source": request.get("audio_source"), "audio_path": request.get("audio_path"),
              "model_path": request.get("model_path"), "package_version": None,
              "diarization_spans": [], "exclusive_spans": [],
              "exclusive_is_derived": True, "review_reasons": [],
              "elapsed_seconds": None, "peak_allocated_bytes": None, "peak_reserved_bytes": None}
    try:
        _validate_request(request)
        config_path = local_pipeline_config(Path(request["model_path"]))
        result["pipeline_config_path"] = str(config_path)
        result["pipeline_config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        result["hash_scope"] = "pipeline_config_only_not_model_weights"
    except (OSError, ValueError, TypeError) as exc:
        result["reason"] = str(exc)
        return result
    try:
        installed = importlib.metadata.version("pyannote.audio")
    except importlib.metadata.PackageNotFoundError:
        result["reason"] = "pyannote.audio is not installed in this worker interpreter"
        return result
    result["package_version"] = installed
    if installed != request["expected_package_version"]:
        result["reason"] = "installed pyannote.audio version differs from expected_package_version"
        return result
    if installed.split(".")[0] != str(BACKENDS[request["backend"]]):
        result["reason"] = "backend/package major version mismatch; use separate legacy/community environments"
        return result
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(sys.stderr):
            pipeline = _load_pipeline(config_path)
            if pipeline is None:
                raise RuntimeError("local pipeline could not be loaded")
            device = request.get("device", "cpu")
            cuda = None
            if device == "cuda":
                import torch
                if not torch.cuda.is_available():
                    result["reason"] = "CUDA requested but unavailable"
                    return result
                pipeline.to(torch.device("cuda"))
                cuda = torch.cuda
                cuda.reset_peak_memory_stats()
            output = pipeline(str(Path(request["audio_path"])))
            if cuda is not None:
                cuda.synchronize()
                result["peak_allocated_bytes"] = cuda.max_memory_allocated()
                result["peak_reserved_bytes"] = cuda.max_memory_reserved()
        full = getattr(output, "speaker_diarization", output)
        result["diarization_spans"] = annotation_spans(full)
        exclusive = getattr(output, "exclusive_speaker_diarization", None)
        if exclusive is not None:
            result["exclusive_spans"] = annotation_spans(exclusive)
        if not result["diarization_spans"]:
            result["review_reasons"].append("no_speaker_activity_returned")
        result["status"] = "completed"
        result["reason"] = None
    except (ImportError, ModuleNotFoundError) as exc:
        result["reason"] = f"worker dependency unavailable: {type(exc).__name__}: {exc}"
    except Exception as exc:
        result["status"] = "error"
        result["reason"] = f"offline diarization failed: {type(exc).__name__}: {exc}"
    finally:
        result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path, help="JSON request (no credentials)")
    parser.add_argument("--output", required=True, type=Path, help="JSON result outside user project data")
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        result = run_request(request)
    except (OSError, ValueError) as exc:
        result = {"protocol_version": PROTOCOL_VERSION, "status": "not_run", "reason": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
