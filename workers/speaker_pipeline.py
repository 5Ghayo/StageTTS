"""Model-independent speaker assignment and isolated DiariZen orchestration."""

from __future__ import annotations

import copy
import json
import math
import subprocess
import tempfile
from pathlib import Path

DIARIZEN_MODEL = "BUT-FIT/diarizen-wavlm-large-s80-md-v2"
DIARIZEN_REVISION = "027b3e221b9d81dce2be794084f1dbd3ba2da403"
DIARIZEN_SOURCE = "844f5555b0a98acd0931511fc641a8c5b8ba92c7"
DIARIZEN_PYANNOTE_SOURCE = "221d6165a1a9cc858a755cd85dfec68ea60e6a5e"


def validate_spans(spans: list[dict], duration: float | None = None) -> list[dict]:
    result = []
    for row in spans:
        start, end = float(row["start"]), float(row["end"])
        speaker = str(row.get("speaker") or "").strip()
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start
                or not speaker or speaker == "UNKNOWN" or (duration is not None and end > duration + 0.05)):
            raise ValueError("DiariZen returned invalid speaker activity")
        result.append({**row, "start": start, "end": end, "speaker": speaker})
    return sorted(result, key=lambda row: (row["start"], row["end"], row["speaker"]))


def assign_interval(start: float, end: float, spans: list[dict], min_coverage: float = 0.8) -> tuple[str, str | None]:
    """Assign only an unambiguous interval; never fill a gap with the nearest voice."""
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return "UNKNOWN", "speaker_assignment_missing"
    active = [s for s in spans if s["start"] < end and s["end"] > start]
    labels = {s["speaker"] for s in active}
    if len(labels) > 1:
        overlapping = any(
            a["speaker"] != b["speaker"] and min(end, a["end"], b["end"]) > max(start, a["start"], b["start"])
            for i, a in enumerate(active) for b in active[i + 1:]
        )
        return "UNKNOWN", "overlapping_speaker_activity" if overlapping else "multiple_speaker_turns"
    if not labels:
        return "UNKNOWN", "speaker_assignment_missing"
    cursor, covered = start, 0.0
    for span in sorted(active, key=lambda s: s["start"]):
        right = min(end, span["end"])
        covered += max(0.0, right - max(cursor, span["start"]))
        cursor = max(cursor, right)
    if covered < min_coverage * (end - start):
        return "UNKNOWN", "insufficient_speaker_coverage"
    return next(iter(labels)), None


def assign_transcript(segments: list[dict], spans: list[dict]) -> list[dict]:
    """Preserve source text/timestamps; no character-based splitting of ASR sentences."""
    result = copy.deepcopy(segments)
    for segment in result:
        speaker, reason = assign_interval(float(segment["start"]), float(segment["end"]), spans)
        segment["speaker"] = speaker
        segment.pop("speaker_fallback_reason", None)
        segment["speaker_needs_review"] = bool(reason)
        if reason:
            segment["speaker_fallback_reason"] = reason
        for word in segment.get("words") or []:
            # Clear stale ASR speaker labels even when word alignment is missing.
            word["speaker"] = None
            if word.get("start") is not None and word.get("end") is not None:
                label, word_reason = assign_interval(float(word["start"]), float(word["end"]), spans)
                word["speaker"] = label if not word_reason else None
    return result


def run_speaker_job(audio_path: Path | None, settings: dict, *, preflight: bool = False) -> dict:
    """Keep the forked DiariZen dependencies out of WhisperX/backend environments."""
    interpreter = settings.get("python")
    if not interpreter:
        raise ValueError("ENV_DIARIZATION_PYTHON / diarization_python is required")
    request = {**settings, "protocol_version": "1.0", "preflight": preflight,
               "audio_path": str(audio_path.resolve()) if audio_path else None}
    worker = Path(__file__).with_name("diarizen_worker.py").resolve()
    with tempfile.TemporaryDirectory(prefix="stagetts_diarizen_") as temporary:
        path = Path(temporary)
        input_file, output_file = path / "request.json", path / "result.json"
        input_file.write_text(json.dumps(request), encoding="utf-8")
        subprocess.run([str(interpreter), str(worker), "--request", str(input_file), "--output", str(output_file)],
                       check=True, timeout=settings.get("timeout", 1800))
        result = json.loads(output_file.read_text(encoding="utf-8"))
    if result.get("protocol_version") != "1.0" or result.get("status") != "completed":
        raise RuntimeError(f"DiariZen speaker pipeline failed: {result.get('reason', 'invalid worker response')}")
    if not preflight:
        result["diarization_spans"] = validate_spans(result.get("diarization_spans", []))
    return result
