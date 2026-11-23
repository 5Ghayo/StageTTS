"""Run MOSS in its own environment and export source-aligned training clips."""

import argparse
import json
import math
import subprocess
from pathlib import Path


def export_segments(segments: list[dict], source: Path, output: Path, language: str) -> list[dict]:
    duration = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(source),
    ], text=True).strip())
    rows = []
    for segment in segments:
        start, speech_end = float(segment["start"]), float(segment["end"])
        text = segment["text"].strip()
        if not all(math.isfinite(t) for t in (start, speech_end)) or start < 0 or speech_end <= start:
            raise ValueError(f"Invalid MOSS timestamp: {segment}")
        if not text or start >= duration:
            continue
        speech_end = min(speech_end, duration)
        end = min(speech_end + 0.15, duration)
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-ss", str(start), "-i", str(source),
            "-t", str(end - start), "-ar", "32000", "-ac", "1",
            str(output / f"{source.stem}_{len(rows)}.mp3"),
        ], check=True)
        rows.append({**segment, "start": start, "end": end, "speech_end": speech_end,
                     "text": text, "language": language})
    return rows


def main() -> None:
    from moss_transcribe_diarize.app.model_runner import ModelRunner
    from moss_transcribe_diarize.subtitle import subtitle_segments_from_transcript

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--language", required=True)
    args = parser.parse_args()
    runner = ModelRunner("OpenMOSS-Team/MOSS-Transcribe-Diarize", device="auto", dtype="bf16")
    for source in sorted(args.inputs.iterdir()):
        if not source.is_file():
            continue
        output = args.inputs.with_name("emilia_processed") / source.stem
        output.mkdir(parents=True)
        print(f"MOSS transcribing {source.name}", flush=True)
        result = runner.transcribe(source, max_new_tokens=65536, max_length=131072)
        segments = [s.to_dict() for s in subtitle_segments_from_transcript(result.text, postprocess=False)]
        rows = export_segments(segments, source, output, args.language)
        (output / f"{source.stem}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        (output / "moss-result.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8",
        )
        print(f"MOSS exported {len(rows)} clips", flush=True)


if __name__ == "__main__":
    main()
