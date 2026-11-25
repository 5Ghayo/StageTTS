"""Prepare the seven current DiariZen voices used by the classic 120 s scene."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import wave
from pathlib import Path

CLUSTERS = (2, 3, 5, 6, 7, 9, 13)
# Automatically selected full-sentence references, plus a verified 4.31 s
# candidate for cluster 02, whose automatic preflight had no reference.
REFERENCE_IDS = {2: 86, 3: 235, 5: 426, 6: 92, 7: 581, 9: 495, 13: 240}


def run(*args: object) -> None:
    subprocess.run([str(arg) for arg in args], check=True)


def cut_audio(source: Path, start: float, end: float, output: Path) -> None:
    run("ffmpeg", "-nostdin", "-v", "error", "-ss", f"{start:.3f}", "-i", source,
        "-t", f"{end - start:.3f}", "-ar", "32000", "-ac", "1", "-c:a", "pcm_s16le", output)


def repeat_to_minimum(path: Path, minimum: float = 0.8) -> int:
    with wave.open(str(path), "rb") as audio:
        params = audio.getparams()
        rate = audio.getframerate()
        pcm = audio.readframes(audio.getnframes())
    repeats = max(1, math.ceil(minimum * rate * 2 / len(pcm)))
    if repeats > 1:
        with wave.open(str(path), "wb") as audio:
            audio.setparams(params)
            audio.writeframes(pcm * repeats)
    return repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    script = json.loads(args.script.read_text(encoding="utf-8"))
    by_id = {seg["id"]: seg for seg in script["segments"]}
    voices = []
    for number in CLUSTERS:
        speaker = f"DZ_6e5330e3aa4b_{number}"
        name = f"cluster_{number:02d}"
        folder = args.output / "training-candidates" / name
        folder.mkdir(parents=True)
        rows = []
        for seg in script["segments"]:
            if seg.get("speaker") != speaker or seg.get("overlap_status") == "detected" or seg.get("overlap_turns"):
                continue
            cue = seg.get("speech_cue") or {}
            text = " ".join((cue.get("corrected_source_text") or seg.get("orig_text") or "").replace("|", " ").split())
            start, end, rms = seg.get("source_audio_start"), seg.get("source_audio_end"), cue.get("rms_dbfs")
            if rms is not None and rms < -45:
                continue
            if not text or start is None or end is None or end <= start:
                raise ValueError(f"Segment {seg['id']} lacks a training transcript or valid audio range")
            index = len(rows)
            audio = folder / f"{index:04d}.wav"
            cut_audio(args.audio, float(start), float(end), audio)
            repeats = repeat_to_minimum(audio)
            rows.append({
                "segment_id": seg["id"], "audio": audio.name, "start": start, "end": end,
                "rms_dbfs": rms, "text": text, "repeat": repeats,
            })

        if not rows:
            raise RuntimeError(f"No eligible training audio for {speaker}")
        labels = [
            f"{folder / row['audio']}|{name}|ja|{' '.join([row['text']] * row['repeat'])}"
            for row in rows
        ]
        (folder / "labels.list").write_text("\n".join(labels) + "\n", encoding="utf-8")

        ref_id = REFERENCE_IDS[number]
        ref = by_id[ref_id]
        if ref["speaker"] != speaker:
            raise ValueError(f"Reference {ref_id} does not belong to {speaker}")
        cue = ref.get("speech_cue") or {}
        ref_text = " ".join((cue.get("corrected_source_text") or ref["orig_text"]).split())
        ref_wav = folder / "reference.wav"
        cut_audio(args.audio, float(ref["source_audio_start"]), float(ref["source_audio_end"]), ref_wav)
        reference = {"wav": str(ref_wav), "text": ref_text, "lang": "ja", "segment_id": ref_id}
        (folder / "reference.json").write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        voices.append({
            "voice": name, "speaker": speaker, "display_name": script["characters"][speaker]["display_name"],
            "clips": rows, "seconds": round(sum(row["end"] - row["start"] for row in rows), 3),
            "reference": reference,
        })
        print(f"{name}: {len(rows)} clips, {voices[-1]['seconds']:.3f}s, reference segment {ref_id}", flush=True)

    (args.output / "voices.json").write_text(json.dumps({
        "source_script": str(args.script), "source_audio": str(args.audio),
        "grouping": "diarizen_current_16", "voices": voices,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    repo = args.output.parent
    gsv_link = args.output / "GPT-SoVITS"
    v5_pipeline = repo / ".v5-pipeline"
    gsv_link.symlink_to(v5_pipeline / "GPT-SoVITS")
    shutil.copyfile(v5_pipeline / "train_v5.py", args.output / "train_v5.py")
    print(f"PREPARED {len(voices)} voices at {args.output}", flush=True)


if __name__ == "__main__":
    main()
