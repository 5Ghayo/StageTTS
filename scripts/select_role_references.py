"""Select one representative original dialogue clip per role for GPT-SoVITS."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median

import librosa
import numpy as np
import soundfile as sf


def measure_f0(samples: np.ndarray, rate: int) -> float | None:
    hop = rate // 50
    pitch, voiced, _ = librosa.pyin(
        samples, fmin=85, fmax=730, sr=rate, frame_length=768, hop_length=hop,
        max_transition_rate=30, fill_na=np.nan,
    )
    energy = librosa.feature.rms(y=samples, frame_length=512, hop_length=hop)[0]
    usable = pitch[voiced & (energy >= max(float(energy.max()) * 0.09, 0.0004))]
    if len(usable) >= 8:
        return float(np.median(usable))
    return None


def choose_reference(clips: list[dict]) -> dict:
    typical_pitch = median(math.log2(clip["f0_hz"]) for clip in clips)
    typical_rms = median(clip["rms_dbfs"] for clip in clips)
    candidates = [clip for clip in clips if 3 <= clip["end"] - clip["start"] <= 10]
    if not candidates:
        raise ValueError("角色缺少 3–10 秒的可用参考音")

    def score(clip: dict) -> float:
        pitch_distance = abs(math.log2(clip["f0_hz"]) - typical_pitch) * 12
        loudness_distance = abs(clip["rms_dbfs"] - typical_rms)
        shortfall = max(0.0, 7 - clip["speech_seconds"])
        return pitch_distance / 3 + loudness_distance / 6 + shortfall / 4

    best = min(candidates, key=lambda clip: (score(clip), -clip["speech_seconds"], clip["segment_id"]))
    return {**best, "role_median_f0_hz": 2**typical_pitch, "role_median_rms_dbfs": typical_rms,
            "measured_clips": len(clips), "candidates": len(candidates)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voices", required=True, type=Path)
    parser.add_argument("--script", required=True, type=Path)
    parser.add_argument("--dialogue", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = json.loads(args.script.read_text(encoding="utf-8"))
    segments = {row["id"]: row for row in source["segments"]}
    voices = json.loads(args.voices.read_text(encoding="utf-8"))["voices"]
    chosen = {}
    with sf.SoundFile(args.dialogue) as audio:
        if audio.samplerate != 16000 or audio.channels != 1:
            raise ValueError("分析需要 16 kHz 单声道对白轨")
        rate = audio.samplerate
        for voice in voices:
            clips = []
            for row in voice["clips"]:
                seg = segments[row["segment_id"]]
                cue = seg["speech_cue"]
                if seg["speaker_needs_review"] or seg["overlap_status"] != "clear":
                    continue
                start, end = cue["speech_start"], cue["speech_end"]
                audio.seek(round(start * rate))
                samples = audio.read(round((end - start) * rate), dtype="float32")
                if len(samples) < 768:
                    continue
                pitch = measure_f0(samples, rate)
                if pitch is None:
                    continue
                # For a short utterance keep only its existing tail margin,
                # allowing GPT-SoVITS to receive a natural, valid 3 s reference.
                ref_end = max(end, min(row["end"], start + 3.01))
                clips.append({
                    "segment_id": seg["id"], "start": start, "end": ref_end,
                    "speech_seconds": round(end - start, 3), "f0_hz": pitch,
                    "rms_dbfs": cue["rms_dbfs"], "text": row["text"], "lang": source["source_lang"],
                })
            selected = choose_reference(clips)
            chosen[voice["speaker"]] = selected
            print(voice["voice"], selected["segment_id"], f"{selected['end']-selected['start']:.2f}s",
                  f"{selected['f0_hz']:.0f}Hz", f"{selected['rms_dbfs']:.1f}dBFS", flush=True)
    args.output.write_text(json.dumps(chosen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
