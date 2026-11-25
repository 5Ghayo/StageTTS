"""Run an independent pyannote pass and preserve concurrent speaker onsets."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from backend import config
from workers.speaker_pipeline import validate_spans


def scan_speaker_activity(audio: Path, device: str = "cpu") -> list[dict]:
    if not config.PYANNOTE_OVERLAP_PYTHON or not config.PYANNOTE_OVERLAP_MODEL:
        raise ValueError("请配置 PYANNOTE_OVERLAP_PYTHON 和 PYANNOTE_OVERLAP_MODEL")
    worker = Path(__file__).resolve().parents[1] / "experiments" / "diarization_worker.py"
    request = {
        "protocol_version": "1.0", "backend": "community-1", "device": device,
        "audio_path": str(audio.resolve()), "audio_source": "dialogue",
        "source_id": audio.name, "model_path": str(config.PYANNOTE_OVERLAP_MODEL),
        "expected_package_version": config.PYANNOTE_OVERLAP_VERSION,
    }
    with tempfile.TemporaryDirectory(prefix="stagetts_overlap_") as directory:
        path = Path(directory)
        source, target = path / "request.json", path / "result.json"
        source.write_text(json.dumps(request), encoding="utf-8")
        subprocess.run([config.PYANNOTE_OVERLAP_PYTHON, str(worker), "--request", str(source),
                        "--output", str(target)], check=True, timeout=config.PARSE_TIMEOUT_SEC)
        result = json.loads(target.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise RuntimeError(result.get("reason") or "pyannote 未完成叠话检查")
    return validate_spans(result["diarization_spans"])


def overlap_pairs(spans: list[dict]) -> list[tuple[float, float, dict, dict]]:
    active = []
    overlaps = []
    for row in sorted(spans, key=lambda item: item["start"]):
        start = row["start"]
        active = [item for item in active if item["end"] > start]
        for previous in active:
            if previous["speaker"] != row["speaker"]:
                overlaps.append((start, min(previous["end"], row["end"]), previous, row))
        active.append(row)
    return overlaps


def annotate_overlaps(segments: list[dict], spans: list[dict] | None) -> None:
    overlaps = overlap_pairs(spans) if spans is not None else []
    for row in segments:
        start = row.get("source_audio_start")
        end = row.get("source_audio_end")
        start = row["start"] if start is None else start
        end = row["end"] if end is None else end
        turns = sorted({(turn["start"], turn["end"], turn["speaker"])
                        for left, right, first, second in overlaps if left < end and right > start
                        for turn in (first, second)})
        covered = spans is not None and any(turn["start"] < end and turn["end"] > start for turn in spans)
        row["overlap_status"] = "detected" if turns else ("clear" if covered else "unchecked")
        row["overlap_turns"] = [{"start": a, "end": b, "speaker": label} for a, b, label in turns]
        row["training_excluded"] = row.get("training_excluded", False) or not covered or bool(turns)
