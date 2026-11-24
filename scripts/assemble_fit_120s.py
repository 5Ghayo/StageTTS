"""Assemble existing s1/s2 segments with light silence trim and <=1.15x tempo fit.

This mirrors the StageTTS placement rule closely enough for a listening sample:
trim non-speech edges, gently speed up clips that overflow their slot, then
place them at the original start time. The original e8/e6 track is left untouched.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import os
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT = Path(os.environ["STAGETTS_EXPERIMENT_PROJECT"])
SR = 32000
MAX_SPEED = 1.15
PAD = 0.08


def speech_bounds(audio: np.ndarray, threshold: float = 0.015) -> tuple[int, int]:
    idx = np.flatnonzero(np.abs(audio) > threshold)
    if idx.size == 0:
        return 0, 0
    return int(idx[0]), int(idx[-1]) + 1


def main() -> None:
    script = json.loads((PROJECT / "script.json").read_text(encoding="utf-8"))
    duration = float(script["media"].get("duration") or 120.0)
    segments = sorted(script["segments"], key=lambda s: s["id"])

    for mode in ("s1", "s2"):
        mode_dir = PROJECT / "ab-120s" / mode
        total = int(round(duration * SR))
        track = np.zeros(total, dtype=np.float32)
        with tempfile.TemporaryDirectory(prefix="fit_") as tmp:
            tmp_dir = Path(tmp)
            for seg in segments:
                src = mode_dir / f"seg_{seg['id']:04d}.wav"
                if not src.is_file():
                    continue
                audio, sr = sf.read(src, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != SR:
                    raise RuntimeError(f"unexpected sample rate {sr}")

                start, end = speech_bounds(audio)
                if end <= start:
                    continue
                pad = int(PAD * SR)
                start = max(0, start - pad)
                end = min(len(audio), end + pad)
                clip = audio[start:end]

                slot = max(0.05, float(seg["end"]) - float(seg["start"]))
                raw = len(clip) / SR
                speed = 1.0
                if raw > slot * 1.02:
                    speed = min(MAX_SPEED, round(raw / slot, 3))
                if speed > 1.02:
                    trimmed_path = tmp_dir / f"{mode}_{seg['id']:04d}_trim.wav"
                    fitted_path = tmp_dir / f"{mode}_{seg['id']:04d}_fit.wav"
                    sf.write(trimmed_path, clip, SR, subtype="PCM_16")
                    subprocess.run([
                        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(trimmed_path),
                        "-filter:a", f"atempo={speed}", "-ar", str(SR), "-ac", "1",
                        "-c:a", "pcm_s16le", str(fitted_path),
                    ], check=True)
                    clip, _ = sf.read(fitted_path, dtype="float32")

                pos = int(round(float(seg["start"]) * SR))
                stop = min(total, pos + len(clip))
                if stop > pos:
                    track[pos:stop] += clip[: stop - pos]

        peak = float(np.max(np.abs(track))) if track.size else 0.0
        if peak > 0.98:
            track *= 0.98 / peak
        raw_path = mode_dir / f"{mode}_vocals_120s_fit_32k.wav"
        sf.write(raw_path, track, SR, subtype="PCM_16")
        out = PROJECT / "ab-120s" / f"{mode}_vocals_120s_fit.wav"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(raw_path),
            "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(out),
        ], check=True)
        print(f"{mode} {out} {out.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
