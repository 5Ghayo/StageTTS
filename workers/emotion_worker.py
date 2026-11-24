"""Offline emotion2vec+ JSON worker; optional ML dependencies stay in this process.

Run: python workers/emotion_worker.py --input request.json --output result.json
Omit --input/--output to use stdin/stdout. Never downloads models or infers emotion
from text. Model selection/license approval is supplied explicitly by the caller.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import wave
from typing import Any, Callable

MODEL_IDS = {f"emotion2vec/emotion2vec_plus_{size}" for size in ("seed", "base", "large")}
REQUIRED_FILES = ("config.yaml", "configuration.json", "model.pt", "tokens.txt")


def _base_result(request: dict) -> dict:
    return {
        "version": "1.0", "stage": "emotion", "status": "not_run",
        "model": dict(request.get("model") or {}),
        "source": dict(request.get("source") or {}),
        "annotations": [], "review_reasons": [],
        "measurement_note": "Model scores are uncalibrated; no emotion accuracy measured without gold labels.",
    }


def _validate(request: dict) -> tuple[Path, Path]:
    model = request.get("model") or {}
    source = request.get("source") or {}
    if model.get("license_approved") is not True:
        raise ValueError("model_license_not_approved")
    if model.get("id") not in MODEL_IDS:
        raise ValueError("unsupported_model_id")
    if not model.get("revision") or model["revision"] in {"main", "master", "latest"}:
        raise ValueError("explicit_model_revision_required")
    if not model.get("funasr_version"):
        raise ValueError("explicit_funasr_version_required")
    local = Path(str(model.get("local_path", "")))
    if not local.is_absolute() or not local.is_dir():
        raise ValueError("local_model_directory_required_no_download")
    if any(not (local / name).is_file() for name in REQUIRED_FILES):
        raise ValueError("local_model_files_missing")
    if not source.get("id") or source.get("kind") not in {"original", "dialogue"}:
        raise ValueError("source_id_and_kind_required")
    audio = Path(str(source.get("audio_path", "")))
    if not audio.is_absolute() or not audio.is_file():
        raise ValueError("local_audio_file_required")
    spans = request.get("spans")
    if not isinstance(spans, list) or not spans:
        raise ValueError("source_spans_required")
    ids = set()
    for span in spans:
        if not isinstance(span, dict) or not isinstance(span.get("id"), str) or not span["id"]:
            raise ValueError("source_span_id_required")
        if span["id"] in ids:
            raise ValueError("duplicate_source_span_id")
        ids.add(span["id"])
        for key in ("start", "end"):
            value = span.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("invalid_source_span_time")
        if span["start"] < 0 or span["end"] <= span["start"]:
            raise ValueError("invalid_source_span_range")
    return local, audio


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_audio(path: Path, start: float, end: float):
    """Read only the requested source span. No resampling/truncation is hidden."""
    import numpy as np
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (16000, 1, 2, "NONE"):
            raise ValueError("audio_requires_pcm16_mono_16000hz")
        first, last = round(start * 16000), round(end * 16000)
        if last > wav.getnframes() or first >= last:
            raise ValueError("source_span_outside_audio")
        wav.setpos(first)
        samples = np.frombuffer(wav.readframes(last - first), dtype="<i2")
    if len(samples) != last - first:
        raise ValueError("incomplete_source_audio")
    if not samples.any():
        raise ValueError("silent_audio_no_emotion_evidence")
    return samples.astype("float32") / 32768.0


def _create_model(local: Path, config: dict):
    actual = importlib.metadata.version("funasr")
    if actual != config["funasr_version"]:
        raise RuntimeError(f"funasr_version_mismatch: expected={config['funasr_version']} actual={actual}")
    # Documented offline settings; local file preflight happens before any ML import.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from funasr import AutoModel
    model = AutoModel(
        model=str(local), hub="ms", device=config.get("device", "cpu"),
        disable_update=True, check_latest=False, trust_remote_code=False,
        batch_size=1, ncpu=1,
    )
    return model


def _annotation(span: dict, source: dict, model: dict) -> dict:
    reasons = list(dict.fromkeys(str(reason) for reason in (span.get("blocked_reasons") or [])))
    if not span.get("speaker") or span.get("speaker") == "UNKNOWN":
        reasons.append("speaker_unknown")
    if span.get("overlap"):
        reasons.append("overlapping_speakers")
    if span.get("speaker_needs_review"):
        reasons.append("speaker_needs_review")
    if span.get("alignment_needs_review"):
        reasons.append("alignment_needs_review")
    if span["end"] - span["start"] < 1.0:
        reasons.append("short_audio_context")
    if source["kind"] == "original":
        reasons.append("original_mix_may_include_music_or_effects")
    return {
        "id": span["id"], "start": span["start"], "end": span["end"],
        "speaker": span.get("speaker") or "UNKNOWN",
        "source_token_ids": list(span.get("source_token_ids") or []),
        "source_turn_ids": list(span.get("source_turn_ids") or []),
        "status": "not_run", "labels": [], "scores": [], "selected_label": None,
        "score_kind": None, "needs_review": True, "review_reasons": reasons,
        "reliability": "unvalidated",
        "provenance": {"stage": "emotion", "model": model["id"],
                       "model_version": model["revision"], "source_id": source["id"],
                       "parameters": {"source_kind": source["kind"], "granularity": "utterance", "batch_size": 1}},
    }


def _parse_scores(raw: Any) -> tuple[list[str], list[float]]:
    if not isinstance(raw, list) or len(raw) != 1 or not isinstance(raw[0], dict):
        raise ValueError("invalid_model_result_count")
    labels, scores = raw[0].get("labels"), raw[0].get("scores")
    if not isinstance(labels, list) or not labels or not all(isinstance(x, str) for x in labels):
        raise ValueError("invalid_model_labels")
    if not isinstance(scores, list) or len(scores) != len(labels):
        raise ValueError("model_label_score_mismatch")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in scores):
        raise ValueError("invalid_model_scores")
    return list(labels), list(scores)


def run_request(request: dict, *, model_factory: Callable | None = None,
                audio_loader: Callable | None = None) -> dict:
    """One model instance, batch one, immutable source spans, no project writes."""
    result = _base_result(request)
    try:
        local, audio = _validate(request)
    except (ValueError, TypeError) as exc:
        result["review_reasons"].append(str(exc))
        return result
    result["annotations"] = [_annotation(s, result["source"], result["model"]) for s in request["spans"]]
    maximum = request.get("max_span_seconds", 30.0)
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)) or not math.isfinite(maximum) or maximum <= 0:
        result["review_reasons"].append("invalid_max_span_seconds")
        return result
    for span, item in zip(request["spans"], result["annotations"]):
        if span.get("blocked_reasons") or any(reason in item["review_reasons"] for reason in (
            "speaker_unknown", "overlapping_speakers", "speaker_needs_review", "alignment_needs_review"
        )):
            item["status"] = "blocked"
        elif span["end"] - span["start"] > maximum:
            item["review_reasons"].append("span_exceeds_maximum_no_hidden_crop")
    eligible = [item for item in result["annotations"] if item["status"] == "not_run" and not any(
        reason == "span_exceeds_maximum_no_hidden_crop" for reason in item["review_reasons"])]
    if not eligible:
        result["review_reasons"].append("no_eligible_source_spans")
        return result
    try:
        expected_hash = result["model"].get("weight_sha256")
        manifest_path = local / "download_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (manifest.get("model_id"), manifest.get("revision")) != (result["model"]["id"], result["model"]["revision"]):
                raise ValueError("local_manifest_model_revision_mismatch")
            manifest_hash = next((item.get("sha256") for item in manifest.get("files", [])
                                  if item.get("name") == "model.pt"), None)
            if expected_hash and manifest_hash and expected_hash.lower() != manifest_hash.lower():
                raise ValueError("requested_and_manifest_hash_mismatch")
            expected_hash = expected_hash or manifest_hash
        actual_hash = _sha256(local / "model.pt")
        if expected_hash and expected_hash.lower() != actual_hash:
            raise ValueError("model_weight_hash_mismatch")
        result["model"]["weight_sha256"] = actual_hash
        # Missing deps are not classified as neutral or scored as model failure.
        model = (model_factory or _create_model)(local, result["model"])
    except (ImportError, importlib.metadata.PackageNotFoundError) as exc:
        result["review_reasons"].append(f"dependency_unavailable: {exc}")
        return result
    except Exception as exc:
        result["review_reasons"].append(f"model_not_loaded: {exc}")
        return result
    for span, item in zip(request["spans"], result["annotations"]):
        if item not in eligible:
            continue
        try:
            samples = (audio_loader or _load_audio)(audio, span["start"], span["end"])
            raw = model.generate(input=samples, fs=16000, granularity="utterance",
                                 extract_embedding=False, batch_size=1, disable_pbar=True)
            labels, scores = _parse_scores(raw)
            item.update(status="ok", labels=labels, scores=scores,
                        selected_label=labels[max(range(len(scores)), key=scores.__getitem__)],
                        score_kind="model_softmax_uncalibrated")
        except Exception as exc:
            item["status"] = "error"
            item["review_reasons"].append(f"classification_failed: {exc}")
    successful = sum(x["status"] == "ok" for x in result["annotations"])
    result["status"] = "ok" if successful == len(result["annotations"]) else "partial" if successful else "error"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.input and args.output and args.input.resolve() == args.output.resolve():
        parser.error("output must not overwrite the input request")
    try:
        text = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
        request = json.loads(text)
        if not isinstance(request, dict):
            raise ValueError("request_must_be_object")
        # Third-party progress/logs must never corrupt stdout JSON framing.
        with redirect_stdout(sys.stderr):
            result = run_request(request)
    except Exception as exc:
        result = {"version": "1.0", "stage": "emotion", "status": "error", "annotations": [], "review_reasons": [str(exc)]}
    text = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if result["status"] in {"ok", "partial", "not_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
