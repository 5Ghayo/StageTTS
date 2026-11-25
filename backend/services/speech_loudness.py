"""RMS gate shared by parsed dialogue and training export."""

from __future__ import annotations

import math
import subprocess
import sys
from array import array
from collections.abc import Iterable
from pathlib import Path

MIN_TRAINING_RMS_DBFS = -45.0


def active_rms_dbfs(frame_energy: Iterable[float]) -> float | None:
    energies = list(frame_energy)
    average = math.fsum(energies) / len(energies) if energies else 0.0
    if average <= 0:
        return None
    active = [value for value in energies if value >= average * 0.01]
    return round(10 * math.log10(math.fsum(active) / len(active)), 2)


def training_loudness_reason(dbfs: float | None) -> str | None:
    if dbfs is None or not math.isfinite(dbfs):
        return "rms_unavailable"
    return "low_rms_dbfs" if dbfs < MIN_TRAINING_RMS_DBFS else None


def measure_training_audio(path: Path) -> float | None:
    """Measure normalized mono speech from legacy or standalone corpus clips."""
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
         "-f", "s16le", "pipe:1"],
        capture_output=True, check=True, timeout=60,
    )
    samples = array("h")
    samples.frombytes(result.stdout)
    if sys.byteorder == "big":
        samples.byteswap()
    return active_rms_dbfs(
        math.fsum(sample * sample for sample in frame) / len(frame) / (32768 * 32768)
        for i in range(0, len(samples), 320) if (frame := samples[i:i + 320])
    )
