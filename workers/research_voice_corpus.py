"""Build per-voice GPT-SoVITS training lists from an Emilia/KyaraEmbed corpus."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from workers.research_training import file_path, output_directory, write_json


def local_asset(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Corpus asset leaves the input directory: {relative}")
    return file_path(path)


def select_members(
    member_indices: list[int], similarities: list[float], minimum_similarity: float,
) -> list[tuple[int, float]]:
    """Rank corpus row indices using similarities in cluster-local order."""
    pairs = zip(member_indices, similarities, strict=True)
    return sorted(
        ((index, score) for index, score in pairs if score >= minimum_similarity),
        key=lambda item: (-item[1], item[0]),
    )


def build_voice_corpus(args: argparse.Namespace) -> None:
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    corpus = args.corpus.resolve()
    output = args.output.resolve()
    rows = json.loads(file_path(corpus / "manifest.json").read_text(encoding="utf-8"))["clips"]
    vectors = []
    for row in rows:
        with np.load(local_asset(corpus, row["features"])) as features:
            vector = np.asarray(features["kyaraembed"], dtype=np.float32)
        norm = np.linalg.norm(vector)
        if vector.shape != (192,) or not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"Invalid KyaraEmbed vector: {row['features']}")
        vectors.append(vector / norm)
    matrix = np.stack(vectors)
    if args.group_by == "speaker":
        keys = [
            (row["provenance"]["source"], row["provenance"]["origin"],
             row["provenance"].get("track"), row["provenance"].get("start"), row["speaker"])
            for row in rows
        ]
        groups = {key: index for index, key in enumerate(dict.fromkeys(keys))}
        labels = np.asarray([groups[key] for key in keys])
    elif len(rows) == 1:
        labels = np.zeros(1, dtype=int)
    else:
        labels = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=args.distance,
        ).fit_predict(matrix)
    durations = [float(row.get("speech_end", row["end"])) - float(row["start"]) for row in rows]
    transcripts = [" ".join((row.get("text") or "").replace("|", " ").split()) for row in rows]
    report = []
    with output_directory(output) as staging:
        for label in sorted(set(labels)):
            candidates = [
                int(index) for index in np.flatnonzero(labels == label)
                if args.min_clip_seconds <= durations[index] <= args.max_clip_seconds
                and len(transcripts[index]) >= 2 and rows[index].get("language")
            ]
            if len(candidates) < args.min_clips:
                continue
            members = matrix[candidates]
            center = members.mean(axis=0)
            scores = members @ (center / np.linalg.norm(center))
            minimum = -1.0 if args.group_by == "speaker" else args.min_similarity
            selected = select_members(candidates, scores.tolist(), minimum)
            seconds = math.fsum(durations[index] for index, _ in selected)
            if len(selected) < args.min_clips or seconds < args.min_seconds:
                continue

            name = f"cluster_{label:02d}"
            folder = Path("training-candidates") / name
            (staging / folder / "audio").mkdir(parents=True)
            lines = []
            clips = []
            for index, similarity in selected:
                row = rows[index]
                relative = folder / "audio" / f"{index:04d}.wav"
                source = local_asset(corpus, row["audio"])
                subprocess.run(
                    ["ffmpeg", "-nostdin", "-v", "error", "-i", str(source),
                     "-ar", "32000", "-ac", "1", "-c:a", "pcm_s16le", str(staging / relative)],
                    check=True,
                )
                lines.append(f"{output / relative}|{name}|{row['language']}|{transcripts[index]}")
                clips.append({"row": index, "audio": row["audio"], "similarity": round(similarity, 4)})
            (staging / folder / "labels.list").write_text("\n".join(lines) + "\n", encoding="utf-8")
            report.append({
                "voice": name, "cluster": int(label), "clips": clips, "seconds": round(seconds, 2),
            })

        if not report:
            raise ValueError("No voice groups meet the configured thresholds")
        write_json(staging / "voices.json", {
            "source_corpus": str(corpus), "distance": args.distance, "group_by": args.group_by,
            "min_clips": args.min_clips, "min_seconds": args.min_seconds,
            "min_clip_seconds": args.min_clip_seconds, "max_clip_seconds": args.max_clip_seconds,
            "min_similarity": args.min_similarity, "voices": report,
        })
    print(f"Prepared {len(report)} groups from {len(rows)} corpus clips")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--distance", type=float, default=0.25)
    parser.add_argument("--group-by", choices=["embedding", "speaker"], default="embedding")
    parser.add_argument("--min-clips", type=int, default=3)
    parser.add_argument("--min-seconds", type=float, default=12)
    parser.add_argument("--min-clip-seconds", type=float, default=2)
    parser.add_argument("--max-clip-seconds", type=float, default=18)
    parser.add_argument("--min-similarity", type=float, default=0.50)
    args = parser.parse_args()
    if not (0 < args.distance <= 2 and args.min_clips >= 1 and args.min_seconds > 0
            and 0 < args.min_clip_seconds <= args.max_clip_seconds <= 54
            and -1 <= args.min_similarity <= 1):
        parser.error("Invalid clustering or training clip thresholds")
    try:
        build_voice_corpus(args)
    except (KeyError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Voice corpus preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
