"""Offline DiariZen Large v2 speaker activity with recording-scoped labels.

Run in its own environment containing the DiariZen author's pyannote fork.
No pyannote 3.1/community checkpoint or online model lookup is used here.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from workers.speaker_pipeline import DIARIZEN_MODEL, DIARIZEN_REVISION, validate_spans


def validate_request(request: dict) -> None:
    if request.get("protocol_version") != "1.0":
        raise ValueError("protocol_version must be 1.0")
    for key, directory in (("model_path", True), ("embedding_path", False)):
        path = Path(request.get(key) or "")
        if not path.is_absolute() or not (path.is_dir() if directory else path.is_file()):
            raise ValueError(f"{key} must be an existing absolute local {'directory' if directory else 'file'}")
    model = Path(request["model_path"])
    for name in ("config.toml", "pytorch_model.bin", "plda/plda.npz", "plda/xvec_transform.npz"):
        if not (model / name).is_file():
            raise ValueError(f"DiariZen model is missing {name}")
    if request.get("device", "cpu") not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    batch = request.get("batch_size", 1)
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError("batch_size must be a positive integer")
    if not request.get("preflight"):
        audio = Path(request.get("audio_path") or "")
        if not audio.is_absolute() or not audio.is_file():
            raise ValueError("audio_path must be an existing absolute local file")


def source_activity(rows: list[dict], duration: float) -> list[dict]:
    """Intersect frame-model output with real audio; retain original rows in the report."""
    result = []
    for row in rows:
        start, end = float(row["start"]), float(row["end"])
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            raise ValueError("DiariZen returned invalid speaker activity")
        left, right = max(0.0, start), min(duration, end)
        if left >= right:
            continue
        clipped = {**row, "start": left, "end": right}
        if left != start or right != end:
            clipped["review_reasons"] = ["clipped_diarizer_padding_to_source_bounds"]
        result.append(clipped)
    return validate_spans(result, duration)


def scope_activity(spans: list[dict], scope: str) -> tuple[list[dict], dict]:
    """Namespace model labels by recording without changing speaker groups or times."""
    mapping = {label: f"DZ_{scope}_{label}" for label in sorted({row["speaker"] for row in spans})}
    return [dict(row, speaker=mapping[row["speaker"]]) for row in spans], {
        "speaker_mapping": mapping,
        "identity_scope": "recording_local_candidate_not_confirmed_character",
    }


def _load_diarizen(request: dict):
    import toml
    import torch
    from diarizen.pipelines.inference import DiariZenPipeline

    # The official local constructor avoids the hub IDs used by from_pretrained.
    root = Path(request["model_path"])
    config = toml.load(str(root / "config.toml"))
    config["inference"]["args"]["batch_size"] = request.get("batch_size", 1)
    # Mapping form avoids upstream's filename heuristic interpreting 'wespeaker' as ONNX.
    embedding = {"checkpoint": request["embedding_path"], "map_location": "cpu"}
    pipeline = DiariZenPipeline(diarizen_hub=root, embedding_model=embedding, config_parse=config)
    pipeline.to(torch.device(request.get("device", "cpu")))
    return pipeline


def run_request(request: dict) -> dict:
    result = {"protocol_version": "1.0", "status": "error", "backend": "diarizen",
              "model": DIARIZEN_MODEL, "model_revision": None, "expected_model_revision": DIARIZEN_REVISION,
              "diarization_spans": [], "raw_diarization_spans": [], "exclusive_spans": []}
    started = time.perf_counter()
    try:
        validate_request(request)
        # Only claim a model revision when the downloaded asset manifest matches the files.
        asset_root = Path(request["model_path"]).parent
        manifest_path = asset_root / "asset_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = [Path(request["embedding_path"]),
                     *[p for p in Path(request["model_path"]).rglob("*") if p.is_file()]]
            for path in files:
                expected = manifest.get("sha256", {}).get(str(path.relative_to(asset_root)))
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if not expected or expected != digest.hexdigest():
                    raise ValueError(f"Speaker model asset hash mismatch: {path.name}")
            result["model_revision"] = manifest.get("diarizen_model_revision")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        # Official fork's Lightning checkpoints need the legacy loader in this isolated process.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        with contextlib.redirect_stdout(sys.stderr):
            import inspect

            import torch
            from diarizen.pipelines.inference import DiariZenPipeline  # noqa: F401
            from pyannote.audio.pipelines import SpeakerDiarization

            if "config" not in inspect.signature(SpeakerDiarization.__init__).parameters:
                raise RuntimeError("Install DiariZen's bundled pyannote fork in the isolated environment")

            if request.get("device") == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            if request.get("preflight"):
                result.update(status="completed", reason=None)
                return result
            from io import BytesIO

            import torchaudio

            waveform, rate = torchaudio.load(request["audio_path"])
            if waveform.ndim != 2 or waveform.shape[1] == 0 or not torch.isfinite(waveform).all():
                raise ValueError("Invalid input audio")
            duration = waveform.shape[1] / rate
            # Upstream selects channel 0; explicitly average channels before passing it audio.
            buffer = BytesIO()
            torchaudio.save(buffer, waveform.mean(dim=0, keepdim=True), rate, format="wav")
            buffer.seek(0)
            pipeline = _load_diarizen(request)
            try:
                annotation = pipeline(buffer)
                original = [
                    {"start": float(turn.start), "end": float(turn.end), "speaker": str(speaker)}
                    for turn, _, speaker in annotation.itertracks(yield_label=True)
                ]
                raw = source_activity(original, duration)
            finally:
                del pipeline
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            digest = hashlib.sha256()
            with Path(request["audio_path"]).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            scope = digest.hexdigest()[:12]
            spans, grouping = scope_activity(raw, scope)
            result.update(status="completed", reason=None, diarization_spans=spans,
                          raw_diarization_spans=original, grouping=grouping, source_audio_sha256=digest.hexdigest(),
                          duration_seconds=duration, device=request.get("device", "cpu"),
                          review_reasons=["no_speaker_activity_returned"] if not spans else [])
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["elapsed_seconds"] = round(time.perf_counter() - started, 6)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = json.loads(args.request.read_text(encoding="utf-8"))
        result = run_request(request)
    except (ValueError, OSError, TypeError) as exc:
        result = {"protocol_version": "1.0", "status": "error", "reason": str(exc)}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
