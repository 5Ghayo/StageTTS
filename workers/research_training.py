"""Prepare corpus audio and representations with the StageTTS research toolchain."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from workers.research_models import RELEASES

MODEL_RUNNER = Path(__file__).with_name("research_models.py").resolve()
EMILIA_MAIN = Path("preprocessors/Emilia/main.py")
EMILIA_AUDIO = {".mp3", ".wav", ".flac", ".m4a", ".aac"}
SOURCES = {
    "amphion": ["https://github.com/open-mmlab/Amphion", "26f6883110181f1dbfe95c70a7c7dbaf4de5f42a"],
    "sidon": ["https://github.com/sarulab-speech/Sidon", "45210939881b8a38a0752fb33451a79a9d3703ee"],
    "kyarabench": ["https://github.com/sizigi/kyarabench", "6453727862d3bdf54a8e50a4d26a254ca3b47e14"],
    "emotion2vec": ["https://github.com/ddlBoJack/emotion2vec", "a0e606c668227ab0c01527e196a4acc42b1efad9"],
}


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def file_path(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    # Resolving a venv's python symlink would select the system interpreter.
    return {name: os.path.abspath(path.parent / Path(value).expanduser()) for name, value in config.items() if value}


def run_job(config: dict, stage: str, job: dict) -> None:
    interpreter = "sidon_python" if stage == "dialogue-sidon" else "feature_python"
    with tempfile.TemporaryDirectory(prefix="stagetts_research_job_") as tmp:
        path = Path(tmp) / "job.json"
        write_json(path, {**config, **job})
        subprocess.run([str(file_path(Path(config[interpreter]))), str(MODEL_RUNNER), stage, str(path)], check=True)


@contextmanager
def output_directory(path: Path):
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{path.name}-", dir=path.parent) as tmp:
        staging = Path(tmp) / "dataset"
        staging.mkdir()
        yield staging
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
        staging.rename(path)


def project_asset(script: Path, project_id: str, url: str) -> Path:
    parts = PurePosixPath(url).parts
    if parts[:3] != ("/", "workspace", project_id) or len(parts) < 4:
        raise ValueError(f"Expected an asset in project {project_id}: {url}")
    asset = (script.parent / Path(*parts[3:])).resolve()
    if not asset.is_relative_to(script.parent.resolve()):
        raise ValueError("Asset path leaves the project directory")
    return file_path(asset)


def extract_audio(source: Path, dest: Path, start: float | None = None, end: float | None = None) -> None:
    trim = ["-ss", str(start), "-t", str(end - start)] if start is not None and end is not None else []
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", *trim, "-i", str(source),
         "-map", "0:a:0", "-c:a", "pcm_s16le", str(dest)],
        check=True,
    )


def overlap_windows(segments: list[dict], duration: float) -> list[dict]:
    """Find overlapping approved two-person windows on a StageTTS timeline."""
    edges = sorted({t for seg in segments for t in (seg["start"], seg["end"])})
    windows = []
    for start, end in zip(edges, edges[1:]):
        active = [seg for seg in segments if seg["start"] < end and seg["end"] > start]
        speakers = sorted({seg["speaker"] for seg in active})
        if len(speakers) != 2 or any(seg.get("speaker_needs_review", True) for seg in active):
            continue
        if windows and windows[-1]["end"] == start and windows[-1]["candidate_characters"] == speakers:
            windows[-1]["end"] = end
        else:
            windows.append({"start": start, "end": end, "candidate_characters": speakers})
    for window in windows:
        start, end = window["start"], window["end"]
        actors = set(window["candidate_characters"])
        other = [seg for seg in segments if seg["speaker"] not in actors]
        floor = max((seg["end"] for seg in other if seg["end"] <= start), default=0.0)
        ceiling = min((seg["start"] for seg in other if seg["start"] >= end), default=duration)
        # A little two-speaker context gives Emilia's minimum-length filter a chance
        # to keep an isolated turn even if the intersection itself is very brief.
        padding = max(0.0, (4.0 - (end - start)) / 2)
        window.update(overlap_start=start, overlap_end=end, start=max(floor, start - padding),
                      end=min(ceiling, end + padding))
    return windows


def character_for_clip(segments: list[dict], start: float, end: float) -> str | None:
    overlapping = [seg for seg in segments if seg["start"] < end and seg["end"] > start]
    speakers = {seg["speaker"] for seg in overlapping}
    # Keep ambiguous portions out of any named character's training set.
    if len(speakers) != 1 or any(seg.get("speaker_needs_review", True) for seg in overlapping):
        return None
    covered = 0.0
    cursor = start
    for seg in sorted(overlapping, key=lambda item: item["start"]):
        boundary = min(end, seg["end"])
        covered += max(0.0, boundary - max(cursor, seg["start"]))
        cursor = max(cursor, boundary)
    if covered < 0.8 * (end - start):
        return None
    return next(iter(speakers))


def run_emilia(config: dict, inputs: Path) -> None:
    main = file_path(Path(config["amphion_root"]) / EMILIA_MAIN)
    subprocess.run(
        [str(file_path(Path(config["emilia_python"]))), str(main), "--config_path",
         str(file_path(Path(config["emilia_config"]))), "--input_folder_path", str(inputs),
         "--whisper_arch", "medium"],
        cwd=main.parent, check=True,
    )


def validate_emilia(config: dict) -> None:
    main = file_path(Path(config["amphion_root"]) / EMILIA_MAIN)
    settings = json.loads(Path(config["emilia_config"]).read_text(encoding="utf-8"))
    token = settings.get("huggingface_token", "")
    if not isinstance(token, str) or not token.startswith("hf_"):
        raise ValueError("Set an authorized Hugging Face token in the local Emilia-Pipe config")
    for model, expected in (
        (settings["separate"]["step1"]["model_path"], "UVR-MDX-NET-Inst_HQ_3.onnx"),
        (settings["mos_model"]["primary_model_path"], "sig_bak_ovr.onnx"),
    ):
        weights = main.parent / Path(model)
        if file_path(weights).name != expected:
            raise ValueError(f"Emilia-Pipe expects the paper's {expected}")


def attach_features(config: dict, root: Path, records: list[dict]) -> None:
    if not records:
        raise RuntimeError("Emilia-Pipe accepted no training clips")
    (root / "features").mkdir()
    for index, record in enumerate(records):
        record["features"] = f"features/{index:06}.npz"
    ranking = root / "character_candidates.json"
    run_job(config, "features", {
        "clips": [
            {
                "audio": str(root / row["audio"]), "features": str(root / row["features"]),
                "character_id": row["character_id"],
                "candidate_characters": row.get("provenance", {}).get("candidate_characters", []),
            }
            for row in records
        ],
        "ranking": str(ranking),
    })
    for record in records:
        file_path(root / record["features"])
    if ranking.is_file():
        for index, candidates in json.loads(ranking.read_text(encoding="utf-8")).items():
            records[int(index)]["character_candidates"] = candidates
        ranking.unlink()


def pack_clips(config: dict, output: Path, inputs: list[Path]) -> None:
    """Extract representations from audio that has already passed corpus review."""
    with output_directory(output) as root:
        (root / "audio").mkdir()
        rows = []
        for index, source in enumerate(inputs):
            audio = f"audio/{index:06}{source.suffix.lower()}"
            shutil.copyfile(file_path(source), root / audio)
            rows.append({"audio": audio, "source": str(source), "character_id": None})
        attach_features(config, root, rows)
        write_manifest(root, rows, ["kyaraembed", "emotion2vec", "facodec"])


def write_manifest(root: Path, rows: list[dict], models: list[str]) -> None:
    write_json(root / "manifest.json", {
        "sources": SOURCES,
        "model_releases": {name: RELEASES[name][:2] for name in models},
        "clips": rows,
    })


def prepare_dataset(
    config: dict, output: Path, input_audio: Path | None, script_path: Path | None, overlaps: list[Path],
) -> None:
    validate_emilia(config)
    segments = []
    dialogue_source = None
    duration = 0.0
    if script_path is not None:
        from backend.schemas import ProjectScript

        project = ProjectScript.model_validate_json(script_path.read_text(encoding="utf-8"))
        duration = project.media.duration
        segments = [seg.model_dump() for seg in project.segments]
        if project.media.source_video:
            input_audio = project_asset(script_path, project.project_id, project.media.source_video)
        else:
            originals = sorted(p for p in script_path.parent.glob("input.*") if p.is_file())
            if len(originals) != 1:
                raise ValueError("Expected one original input file in the StageTTS project")
            input_audio = originals[0]
        if project.media.vocals_wav:
            dialogue_source = project_asset(script_path, project.project_id, project.media.vocals_wav)

    if input_audio is None:
        raise ValueError("An input file or StageTTS project is required")
    source = file_path(input_audio)
    models = ["kyaraembed", "emotion2vec", "facodec"]
    with output_directory(output) as root, tempfile.TemporaryDirectory(prefix="stagetts_research_inputs_") as tmp_name:
        temp = Path(tmp_name)
        inputs = temp / "emilia"
        inputs.mkdir()
        raw = inputs / ("original" + (source.suffix.lower() if source.suffix.lower() in EMILIA_AUDIO else ".wav"))
        if source.suffix.lower() in EMILIA_AUDIO:
            shutil.copyfile(source, raw)
        else:
            extract_audio(source, raw)

        metadata = {"original": {"source": str(source), "origin": "original"}}
        windows = overlap_windows(segments, duration)
        mixtures = []
        annotated_mixtures = []
        for index, window in enumerate(windows):
            wav = temp / f"window_{index:04}.wav"
            extract_audio(dialogue_source or raw, wav, window["start"], window["end"])
            annotated_mixtures.append((wav, {**window, "source": str(source)}))
        annotated_mixtures += [(file_path(p), {"source": str(p)}) for p in overlaps]
        for index, (wav, window) in enumerate(annotated_mixtures):
            outputs = []
            for channel in (1, 2):
                stem = f"overlap_{index:04}_track_{channel}"
                metadata[stem] = {**window, "origin": "dialogue-sidon", "track": channel}
                outputs.append(str(inputs / f"{stem}.wav"))
            mixtures.append({"input": str(wav), "outputs": outputs})
        if mixtures:
            run_job(config, "dialogue-sidon", {"mixtures": mixtures, "seed": 0})
            for mixture in mixtures:
                for track in mixture["outputs"]:
                    file_path(Path(track))
            models.append("dialogue-sidon")

        run_emilia(config, inputs)
        native = inputs.with_name("emilia_processed")
        if not native.is_dir():
            raise RuntimeError("Emilia-Pipe did not write its native output directory")
        target = root / "emilia_processed"
        shutil.move(native, target)
        rows = []
        for stem, provenance in metadata.items():
            transcript = file_path(target / stem / f"{stem}.json")
            for index, row in enumerate(json.loads(transcript.read_text(encoding="utf-8"))):
                start, end = row["start"], row["end"]
                if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
                    raise ValueError(f"Invalid time range in Emilia-Pipe output: {transcript}")
                origin = provenance["origin"]
                audio = file_path(transcript.parent / f"{stem}_{index}.mp3")
                rows.append({
                    **row, "provenance": provenance, "audio": audio.relative_to(root).as_posix(),
                    "character_id": character_for_clip(segments, start, end) if origin == "original" else None,
                })
        attach_features(config, root, rows)
        write_manifest(root, rows, models)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolchain", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Run Emilia-Pipe and extract training features")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--project", type=Path, help="Path to a StageTTS script.json")
    prepare.add_argument("--overlap", type=Path, action="append", default=[], help="Confirmed two-person audio window")
    prepare.add_argument("--output", type=Path, required=True)
    features = subparsers.add_parser("features", help="Extract representations from reviewed speech clips")
    features.add_argument("--input", type=Path, action="append", required=True)
    features.add_argument("--output", type=Path, required=True)
    separate = subparsers.add_parser("separate", help="Run DialogueSidon on a two-speaker WAV")
    separate.add_argument("--input", type=Path, required=True)
    separate.add_argument("--output", type=Path, required=True)
    benchmark = subparsers.add_parser("kyarabench", help="Run the official benchmark on restored Anim-400K clips")
    benchmark.add_argument("--clips", type=Path, required=True)
    benchmark.add_argument("--anim400k-source", type=Path, help="Restore the official source clips before scoring")
    args = parser.parse_args()

    try:
        config = load_config(args.toolchain.resolve())
        if args.command == "prepare":
            prepare_dataset(
                config, args.output.resolve(), args.input.resolve() if args.input else None,
                args.project.resolve() if args.project else None, [p.resolve() for p in args.overlap],
            )
        elif args.command == "features":
            pack_clips(config, args.output.resolve(), [p.resolve() for p in args.input])
        elif args.command == "separate":
            with output_directory(args.output.resolve()) as root:
                tracks = [root / f"track_{index}.wav" for index in (1, 2)]
                mixture = {"input": str(file_path(args.input.resolve())), "outputs": list(map(str, tracks))}
                run_job(config, "dialogue-sidon", {"mixtures": [mixture], "seed": 0})
                for track in tracks:
                    file_path(track)
        else:
            if args.anim400k_source:
                kyara = Path(config["kyara_root"])
                subprocess.run(
                    [config["sidon_python"], str(kyara / "prepare/restore.py"), "--input-root",
                     str(args.anim400k_source.resolve()), "--output-root", str(args.clips.resolve()),
                     "--clips-json", str(kyara / "data/kyarabench_v1.0_source_clips.json")],
                    cwd=kyara, check=True,
                )
            run_job(config, "kyarabench", {"clips_root": str(args.clips.resolve())})
    except (KeyError, ValueError, TypeError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Training data preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
