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
from workers.subtitle_boundaries import detect_subtitle_boundaries

MODEL_RUNNER = Path(__file__).with_name("research_models.py").resolve()
EMILIA_RUNNER = Path(__file__).with_name("emilia_runner.py").resolve()
EMILIA_MAIN = Path("preprocessors/Emilia/main.py")
EMILIA_AUDIO = {".mp3", ".wav", ".flac", ".m4a", ".aac"}
SOURCES = {
    "amphion": ["https://github.com/open-mmlab/Amphion", "26f6883110181f1dbfe95c70a7c7dbaf4de5f42a"],
    "sidon": ["https://github.com/sarulab-speech/Sidon", "45210939881b8a38a0752fb33451a79a9d3703ee"],
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
    scalars = {"diarization_device", "diarizen_batch_size"}
    return {name: value if name in scalars else os.path.abspath(path.parent / Path(value).expanduser())
            for name, value in config.items() if value is not None and value != ""}


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


def run_emilia(config: dict, inputs: Path, filter_overrides: dict[str, float | bool | str] | None = None) -> None:
    main = file_path(Path(config["amphion_root"]) / EMILIA_MAIN)
    command = [str(file_path(Path(config["emilia_python"])))]
    overrides = filter_overrides or {"min_clip_seconds": 3.0, "min_dnsmos": 3.0}
    command.extend([
        str(EMILIA_RUNNER), str(main),
        "--min-clip-seconds", str(overrides["min_clip_seconds"]),
        "--min-dnsmos", str(overrides["min_dnsmos"]),
    ])
    if overrides.get("fine_turns"):
        command.append("--fine-turns")
        for name in ("max_turn_seconds", "min_silence_ms", "tail_ms", "turn_boundaries"):
            if name in overrides:
                command.extend(["--" + name.replace("_", "-"), str(overrides[name])])
    with tempfile.TemporaryDirectory(prefix="stagetts_emilia_settings_") as temporary:
        settings = Path(temporary) / "diarization.json"
        write_json(settings, research_speaker_settings(config))
        subprocess.run(
            [*command, "--diarization-settings", str(settings),
             "--config_path", str(file_path(Path(config["emilia_config"]))),
             "--input_folder_path", str(inputs), "--whisper_arch", "medium"],
            cwd=main.parent, check=True,
        )


def research_speaker_settings(config: dict) -> dict:
    """Resolve the independent speaker runtime from the research toolchain."""
    return {"python": config["diarization_python"], "model_path": config["diarizen_model_path"],
            "embedding_path": config["diarizen_embedding_path"],
            "device": config.get("diarization_device", "cpu"),
            "batch_size": config.get("diarizen_batch_size", 1)}


def validate_emilia(config: dict) -> None:
    main = file_path(Path(config["amphion_root"]) / EMILIA_MAIN)
    settings = json.loads(Path(config["emilia_config"]).read_text(encoding="utf-8"))
    from workers.speaker_pipeline import run_speaker_job
    run_speaker_job(None, research_speaker_settings(config), preflight=True)
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
    run_job(config, "features", {
        "clips": [
            {
                "audio": str(root / row["audio"]), "features": str(root / row["features"]),
            }
            for row in records
        ],
    })
    for record in records:
        file_path(root / record["features"])

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
        write_manifest(root, rows, ["emotion2vec", "facodec"])


def write_manifest(
    root: Path, rows: list[dict], models: list[str], *, emilia_filter: dict[str, float | bool | str] | None = None,
    transcription_backend: str | None = None,
) -> None:
    manifest = {
        "sources": SOURCES,
        "model_releases": {name: RELEASES[name][:2] for name in models},
        "clips": rows,
    }
    if emilia_filter is not None:
        manifest["emilia_filter"] = emilia_filter
    if transcription_backend is not None:
        manifest["transcription_backend"] = transcription_backend
        if transcription_backend == "emilia":
            from workers.speaker_pipeline import DIARIZEN_MODEL, DIARIZEN_REVISION
            manifest["speaker_pipeline"] = {
                "diarization_model": DIARIZEN_MODEL, "diarization_revision": DIARIZEN_REVISION,
                "identity_scope": "recording_local_candidate",
            }
    write_json(root / "manifest.json", manifest)


def prepare_dataset(
    config: dict, output: Path, input_audio: Path | None, script_path: Path | None, overlaps: list[Path],
    *, emilia_filter: dict[str, float | bool | str] | None = None, subtitle_source: Path | None = None,
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
    models = ["emotion2vec", "facodec"]
    with output_directory(output) as root, tempfile.TemporaryDirectory(prefix="stagetts_research_inputs_") as tmp_name:
        temp = Path(tmp_name)
        inputs = temp / "emilia"
        inputs.mkdir()
        runtime_filter = emilia_filter
        if emilia_filter and emilia_filter.get("fine_turns") and "turn_boundaries" not in emilia_filter:
            subtitle_media = subtitle_source or source
            if subtitle_source is not None or source.suffix.lower() not in EMILIA_AUDIO:
                detected = detect_subtitle_boundaries(subtitle_media, videosubfinder_cli=config.get("videosubfinder_cli"))
                if detected:
                    boundary_file = root / "subtitle-boundaries.json"
                    write_json(boundary_file, detected)
                    emilia_filter = {**emilia_filter, "turn_boundaries": boundary_file.name}
                    runtime_filter = {**emilia_filter, "turn_boundaries": str(boundary_file)}
                    label = detected.get("detector") or f"subtitle stream {detected['stream']} ({detected['codec']})"
                    print(f"Detected {len(detected['cut_seconds'])} cut points via {label}", flush=True)
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

        run_emilia(config, inputs, runtime_filter)
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
                    "character_id": (
                        character_for_clip(segments, start, row.get("speech_end", end))
                        if origin == "original" else None
                    ),
                })
        attach_features(config, root, rows)
        write_manifest(root, rows, models, emilia_filter=emilia_filter, transcription_backend="emilia")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolchain", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Transcribe, slice, and extract training features")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--project", type=Path, help="Path to a StageTTS script.json")
    prepare.add_argument("--overlap", type=Path, action="append", default=[], help="Confirmed two-person audio window")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--emilia-min-clip-seconds", type=float, help="Emilia minimum clip length (default: 3)")
    prepare.add_argument("--emilia-min-dnsmos", type=float, help="Emilia minimum DNSMOS score (default: 3)")
    prepare.add_argument("--emilia-fine-turns", action="store_true", help="Run VAD on every exclusive speaker turn")
    prepare.add_argument("--emilia-max-turn-seconds", type=float, help="Maximum continuous VAD turn in fine mode")
    prepare.add_argument("--emilia-min-silence-ms", type=int, help="Shortest silence kept as a cut in fine mode")
    prepare.add_argument("--emilia-tail-ms", type=int, help="Maximum trailing padding in fine mode (default: 150)")
    prepare.add_argument("--emilia-turn-boundaries", type=Path, help="Source-aligned JSON list of cut_seconds")
    prepare.add_argument("--emilia-subtitle-source", type=Path, help="Aligned video to probe when the input is audio")
    features = subparsers.add_parser("features", help="Extract representations from reviewed speech clips")
    features.add_argument("--input", type=Path, action="append", required=True)
    features.add_argument("--output", type=Path, required=True)
    separate = subparsers.add_parser("separate", help="Run DialogueSidon on a two-speaker WAV")
    separate.add_argument("--input", type=Path, required=True)
    separate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        config = load_config(args.toolchain.resolve())
        if args.command == "prepare":
            if not args.emilia_fine_turns and any(value is not None for value in (
                args.emilia_max_turn_seconds, args.emilia_min_silence_ms,
                args.emilia_tail_ms, args.emilia_turn_boundaries, args.emilia_subtitle_source,
            )):
                raise ValueError("Turn tuning options require --emilia-fine-turns")
            if args.emilia_turn_boundaries is not None and args.emilia_subtitle_source is not None:
                raise ValueError("Use either a cut-point JSON or a source video for subtitle detection")
            overrides = None
            if args.emilia_min_clip_seconds is not None or args.emilia_min_dnsmos is not None or args.emilia_fine_turns:
                overrides = {
                    "min_clip_seconds": (
                        3.0 if args.emilia_min_clip_seconds is None else args.emilia_min_clip_seconds
                    ),
                    "min_dnsmos": 3.0 if args.emilia_min_dnsmos is None else args.emilia_min_dnsmos,
                }
                if not (0 < overrides["min_clip_seconds"] <= 30 and 1 <= overrides["min_dnsmos"] <= 5):
                    raise ValueError("Emilia minimum clip length must be in (0, 30] and DNSMOS in [1, 5]")
                if args.emilia_fine_turns:
                    overrides["fine_turns"] = True
                    tail_ms = 150 if args.emilia_tail_ms is None else args.emilia_tail_ms
                    if not 0 <= tail_ms <= 1000:
                        raise ValueError("Trailing padding must be within 0–1000 ms")
                    overrides["tail_ms"] = tail_ms
                    if args.emilia_max_turn_seconds is not None:
                        if not (overrides["min_clip_seconds"] <= args.emilia_max_turn_seconds <= 30):
                            raise ValueError("Maximum VAD turn must be between the minimum clip length and 30 s")
                        overrides["max_turn_seconds"] = args.emilia_max_turn_seconds
                    if args.emilia_min_silence_ms is not None:
                        if not (20 <= args.emilia_min_silence_ms <= 1000):
                            raise ValueError("VAD silence must be within 20–1000 ms")
                        overrides["min_silence_ms"] = args.emilia_min_silence_ms
                    if args.emilia_turn_boundaries is not None:
                        overrides["turn_boundaries"] = str(file_path(args.emilia_turn_boundaries.resolve()))
            prepare_dataset(
                config, args.output.resolve(), args.input.resolve() if args.input else None,
                args.project.resolve() if args.project else None, [p.resolve() for p in args.overlap],
                emilia_filter=overrides,
                subtitle_source=file_path(args.emilia_subtitle_source.resolve()) if args.emilia_subtitle_source else None,
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
    except (KeyError, ValueError, TypeError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"Training data preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
