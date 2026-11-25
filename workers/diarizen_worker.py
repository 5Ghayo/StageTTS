"""Offline DiariZen speaker clustering for externally supplied dialogue intervals.

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

AHC_THRESHOLD = 0.35
VBX_FA = 0.15
VBX_FB = 0.37
CLUSTERING_ASSETS = ("config.toml", "plda/plda.npz", "plda/xvec_transform.npz")


def clustering_parameters(request: dict) -> dict[str, float]:
    defaults = {"ahc_threshold": AHC_THRESHOLD, "Fa": VBX_FA, "Fb": VBX_FB}
    overrides = request.get("clustering_parameters", {})
    if not isinstance(overrides, dict) or not overrides.keys() <= defaults.keys():
        raise ValueError("Invalid clustering_parameters keys")
    for key, value in overrides.items():
        if type(value) not in (int, float) or not 0 < value <= 1:
            raise ValueError(f"Invalid DiariZen clustering parameter {key}")
    return {**defaults, **overrides}


def validate_request(request: dict) -> None:
    if request.get("protocol_version") != "1.0":
        raise ValueError("protocol_version must be 1.0")
    for key, directory in (("model_path", True), ("embedding_path", False)):
        path = Path(request.get(key) or "")
        if not path.is_absolute() or not (path.is_dir() if directory else path.is_file()):
            raise ValueError(f"{key} must be an existing absolute local {'directory' if directory else 'file'}")
    model = Path(request["model_path"])
    for name in CLUSTERING_ASSETS:
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
        if not isinstance(request.get("dialogue_units"), list):
            raise ValueError("DiariZen requires dialogue_units; audio segmentation is disabled")


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
    from pyannote.audio.pipelines.clustering import VBxClustering
    from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding

    root = Path(request["model_path"])
    config = toml.load(str(root / "config.toml"))["clustering"]["args"]
    # Mapping form avoids upstream's filename heuristic interpreting 'wespeaker' as ONNX.
    embedding = PretrainedSpeakerEmbedding(
        {"checkpoint": request["embedding_path"], "map_location": "cpu"},
        device=torch.device(request.get("device", "cpu")),
    )
    clustering = VBxClustering(metric=embedding.metric, plda_dir=str(root / "plda"),
                               lda_dim=config["lda_dim"], maxIters=config["max_iters"])
    clustering.instantiate({"ahc_criterion": config["ahc_criterion"], **clustering_parameters(request)})
    return embedding, clustering


def cluster_dialogue_units(embedding, clustering, waveform, rate: int, units: list[dict]) -> list[dict]:
    """Cluster one DiariZen WeSpeaker embedding per final dialogue interval."""
    import numpy as np
    import torchaudio
    from pyannote.core import SlidingWindow, SlidingWindowFeature

    duration = waveform.shape[-1] / rate
    mono = waveform.mean(dim=0, keepdim=True)
    if units and rate != embedding.sample_rate:
        mono = torchaudio.functional.resample(mono, rate, embedding.sample_rate)
    rows, vectors, valid = [], [], []
    seen = set()
    for unit in units:
        index, start, end = unit["id"], float(unit["start"]), float(unit["end"])
        if (type(index) is not int or index in seen or not math.isfinite(start) or not math.isfinite(end)
                or not 0 <= start < end <= duration + 0.001):
            raise ValueError("Invalid dialogue unit ID or timestamp")
        seen.add(index)
        row = {"id": index, "start": start, "end": end, "speaker": "UNKNOWN", "reason": None}
        rows.append(row)
        clip = mono[:, round(start * embedding.sample_rate):round(end * embedding.sample_rate)]
        if clip.shape[-1] < embedding.min_num_samples:
            row["reason"] = "dialogue_too_short_for_embedding"
            continue
        vector = embedding(clip[None])[0]
        if not np.isfinite(vector).all() or not np.linalg.norm(vector) > 0:
            row["reason"] = "invalid_dialogue_embedding"
            continue
        vectors.append(vector)
        valid.append(row)

    if vectors:
        # VBx accepts (chunks, local speakers, dimensions). Here every chunk
        # is a dialogue unit, with one candidate speaker and one active slot.
        embeddings = np.stack(vectors)[:, None, :]
        active = SlidingWindowFeature(np.ones((len(valid), 1, 1), dtype=np.float32), SlidingWindow())
        labels, _, _ = clustering(embeddings=embeddings, segmentations=active)
        for row, label in zip(valid, labels[:, 0], strict=True):
            row["speaker"] = str(int(label))
    return rows


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
                     *(Path(request["model_path"]) / name for name in CLUSTERING_ASSETS)]
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
            from pyannote.audio.pipelines import SpeakerDiarization

            if "config" not in inspect.signature(SpeakerDiarization.__init__).parameters:
                raise RuntimeError("Install DiariZen's bundled pyannote fork in the isolated environment")

            if request.get("device") == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            if request.get("preflight"):
                result.update(status="completed", reason=None)
                return result
            import torchaudio

            waveform, rate = torchaudio.load(request["audio_path"])
            if waveform.ndim != 2 or waveform.shape[1] == 0 or not torch.isfinite(waveform).all():
                raise ValueError("Invalid input audio")
            duration = waveform.shape[1] / rate
            embedding, clustering = _load_diarizen(request)
            try:
                assignments = cluster_dialogue_units(embedding, clustering, waveform, rate, request["dialogue_units"])
                original = [
                    {key: row[key] for key in ("start", "end", "speaker")}
                    for row in assignments if row["speaker"] != "UNKNOWN"
                ]
                raw = source_activity(original, duration)
            finally:
                del embedding, clustering
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            digest = hashlib.sha256()
            with Path(request["audio_path"]).open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            scope = digest.hexdigest()[:12]
            parameters = clustering_parameters(request)
            if parameters != clustering_parameters({}):
                fingerprint = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
                scope = hashlib.sha256(f"{digest.hexdigest()}:{fingerprint}".encode()).hexdigest()[:12]
            spans, grouping = scope_activity(raw, scope)
            mapping = grouping["speaker_mapping"]
            result["dialogue_assignments"] = [
                dict(row, speaker=mapping.get(row["speaker"], "UNKNOWN")) for row in assignments
            ]
            result.update(status="completed", reason=None, diarization_spans=spans,
                          grouping=grouping, source_audio_sha256=digest.hexdigest(),
                          duration_seconds=duration, device=request.get("device", "cpu"),
                          parameters=parameters,
                          clustering_input="dialogue_units",
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
