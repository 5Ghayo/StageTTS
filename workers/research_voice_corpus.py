"""Build per-voice GPT-SoVITS training lists from a DiariZen-labelled Emilia corpus."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from backend.services.speech_loudness import measure_training_audio, training_loudness_reason
from workers.research_training import file_path, output_directory, write_json


def local_asset(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Corpus asset leaves the input directory: {relative}")
    return file_path(path)


def build_voice_corpus(args: argparse.Namespace) -> None:
    corpus = args.corpus.resolve()
    output = args.output.resolve()
    rows = [row for row in json.loads(file_path(corpus / "manifest.json").read_text(encoding="utf-8"))["clips"]
            if not row.get("training_excluded") and row.get("overlap_status") != "detected"
            and row.get("provenance", {}).get("origin") != "dialogue-sidon"]
    keys = [
        (row["provenance"]["source"], row["provenance"]["origin"],
         row["provenance"].get("track"), row["provenance"].get("start"), row.get("speaker"))
        for row in rows
    ]
    groups = {key: index for index, key in enumerate(dict.fromkeys(keys))}
    labels = [groups[key] for key in keys]
    durations = [float(row.get("speech_end", row["end"])) - float(row["start"]) for row in rows]
    transcripts = [" ".join((row.get("text") or "").replace("|", " ").split()) for row in rows]
    report = []
    with output_directory(output) as staging:
        for label in sorted(set(labels)):
            candidates = [
                index for index, assigned in enumerate(labels) if assigned == label
                and rows[index].get("speaker") not in {None, "", "UNKNOWN"}
                and args.min_clip_seconds <= durations[index] <= args.max_clip_seconds
                and len(transcripts[index]) >= 2 and rows[index].get("language")
            ]
            selected = []
            for index in candidates:
                row = rows[index]
                dbfs = row["rms_dbfs"] if "rms_dbfs" in row else measure_training_audio(local_asset(corpus, row["audio"]))
                if not training_loudness_reason(dbfs):
                    selected.append(index)
            seconds = math.fsum(durations[index] for index in selected)
            if len(selected) < args.min_clips or seconds < args.min_seconds:
                continue

            name = f"cluster_{label:02d}"
            folder = Path("training-candidates") / name
            (staging / folder / "audio").mkdir(parents=True)
            lines = []
            clips = []
            for index in selected:
                row = rows[index]
                relative = folder / "audio" / f"{index:04d}.wav"
                source = local_asset(corpus, row["audio"])
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
                     "-ar", "32000", "-ac", "1", "-c:a", "pcm_s16le", str(staging / relative)],
                    check=True,
                )
                lines.append(f"{output / relative}|{name}|{row['language']}|{transcripts[index]}")
                clips.append({"row": index, "audio": row["audio"]})
            (staging / folder / "labels.list").write_text("\n".join(lines) + "\n", encoding="utf-8")
            report.append({
                "voice": name, "cluster": int(label), "clips": clips, "seconds": round(seconds, 2),
            })

        if not report:
            raise ValueError("No voice groups meet the configured thresholds")
        write_json(staging / "voices.json", {
            "source_corpus": str(corpus), "group_by": "speaker",
            "min_clips": args.min_clips, "min_seconds": args.min_seconds,
            "min_clip_seconds": args.min_clip_seconds, "max_clip_seconds": args.max_clip_seconds,
            "voices": report,
        })
    print(f"Prepared {len(report)} groups from {len(rows)} corpus clips")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-clips", type=int, default=3)
    parser.add_argument("--min-seconds", type=float, default=12)
    parser.add_argument("--min-clip-seconds", type=float, default=2)
    parser.add_argument("--max-clip-seconds", type=float, default=18)
    args = parser.parse_args()
    if not (args.min_clips >= 1 and args.min_seconds > 0
            and 0 < args.min_clip_seconds <= args.max_clip_seconds <= 54):
        parser.error("Invalid clustering or training clip thresholds")
    try:
        build_voice_corpus(args)
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Voice corpus preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
