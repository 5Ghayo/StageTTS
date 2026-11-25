"""Run the pinned Emilia CLI with explicit upstream clip-filter overrides."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def apply_diarizen_override(source: str) -> str:
    """Replace only the pinned Emilia diarizer, retaining its ASR/VAD/filter/export."""
    import_line = "from pyannote.audio import Pipeline"
    start_marker = '    if not cfg["huggingface_token"].startswith("hf"):'
    end_marker = "    dia_pipeline.to(device)"
    if source.count(import_line) != 1 or source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise ValueError("Unexpected Emilia diarization source; refusing to run the old model")
    source = source.replace(import_line, "", 1)
    start = source.index(start_marker)
    end = source.index(end_marker, start) + len(end_marker)
    source = source[:start] + "    dia_pipeline = stage_tts_diarizen_pipeline" + source[end:]
    call = "    speakerdia = speaker_diarization(audio)"
    if source.count(call) != 1:
        raise ValueError("Unexpected Emilia processing function")
    return source.replace(
        call, "    stage_tts_diarizen_pipeline.vad = vad\n" + call
        + "\n    stage_tts_diarizen_pipeline.save_report(save_path)", 1,
    )


class EmiliaDiarizenPipeline:
    """Adapt Emilia's waveform call to the isolated, local-only speaker worker."""

    def __init__(self, settings: dict):
        self.settings = settings
        self.last_result = None
        self.vad = None

    def save_report(self, directory):
        (Path(directory) / "speaker_analysis.json").write_text(
            json.dumps(self.last_result, ensure_ascii=False, indent=2), encoding="utf-8")

    def __call__(self, audio):
        import librosa
        import soundfile as sf
        from pyannote.core import Annotation, Segment

        from workers.emilia_segments import exclusive_turns
        from workers.speaker_pipeline import run_speaker_job

        waveform = audio["waveform"].detach().cpu().numpy().mean(axis=0)
        rate = audio["sample_rate"]
        vad_rate = 16000
        samples = librosa.resample(waveform, orig_sr=rate, target_sr=vad_rate)
        regions = self.vad.get_speech_timestamps(
            samples, self.vad.vad_model, sampling_rate=vad_rate, max_speech_duration_s=15,
        )
        units = [{"id": i, "start": row["start"] / vad_rate,
                  "end": min(row["end"] / vad_rate, len(waveform) / rate)}
                 for i, row in enumerate(regions)]
        with tempfile.TemporaryDirectory(prefix="stagetts_emilia_diarizen_") as temporary:
            path = Path(temporary) / "dialogue.wav"
            sf.write(str(path), waveform, rate)
            result = run_speaker_job(path, {**self.settings, "dialogue_units": units})
        self.last_result = result
        annotation = Annotation()
        for index, (start, end, label) in enumerate(exclusive_turns(result["diarization_spans"])):
            annotation[Segment(start, end), index] = label
        return annotation


def apply_filter_overrides(
    source: str, min_clip_seconds: float, min_dnsmos: float, *, fine_turns: bool = False,
) -> str:
    short_text = ", min_char_count=1" if fine_turns else ""
    replacements = {
        "MIN_SEGMENT_LENGTH = 3  # min segment length in seconds":
            f"MIN_SEGMENT_LENGTH = {min_clip_seconds!r}  # min segment length in seconds",
        "calculate_audio_stats(mos_list)":
            f"calculate_audio_stats(mos_list, min_duration={min_clip_seconds!r}, "
            f"min_dnsmos={min_dnsmos!r}{short_text})",
        "    filtered_segment = len(filtered_audio_stats)":
            "    if not all_audio_stats:\n        return []\n    filtered_segment = len(filtered_audio_stats)",
    }
    if fine_turns:
        replacements.update({
            "vad_list = vad.vad(speakerdia, audio)":
                "vad_list = stage_tts_refine_turns(vad, speakerdia, audio, audio_path)",
            "MERGE_GAP = 2  # merge gap in seconds, if smaller than this, merge": "MERGE_GAP = 0",
            "    export_to_mp3(audio, filtered_list, save_path, audio_name)": (
                "    filtered_list = stage_tts_pad_tails(filtered_list, audio)\n"
                "    export_to_mp3(audio, filtered_list, save_path, audio_name)"
            ),
            "            language, prob = asr_model.detect_language(segment_audio)": (
                "            if len(supported_languages) == 1 and len(segment_audio) < 32000:\n"
                "                language, prob = supported_languages[0], 1.0\n"
                "            else:\n"
                "                language, prob = asr_model.detect_language(segment_audio)"
            ),
        })
    for original, revised in replacements.items():
        if source.count(original) != 1:
            raise ValueError(f"Unexpected Emilia CLI source. Could not identify {original!r}.")
        source = source.replace(original, revised, 1)
    return source


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entrypoint", type=Path)
    parser.add_argument("--min-clip-seconds", type=float, required=True)
    parser.add_argument("--min-dnsmos", type=float, required=True)
    parser.add_argument("--fine-turns", action="store_true")
    parser.add_argument("--max-turn-seconds", type=float, default=15)
    parser.add_argument("--min-silence-ms", type=int, default=150)
    parser.add_argument("--tail-ms", type=int, default=150)
    parser.add_argument("--diarization-settings", type=Path, required=True)
    args, official_args = parser.parse_known_args()
    entrypoint = args.entrypoint.resolve()
    source = apply_filter_overrides(
        entrypoint.read_text(encoding="utf-8"), args.min_clip_seconds, args.min_dnsmos,
        fine_turns=args.fine_turns,
    )
    source = apply_diarizen_override(source)
    sys.path.insert(0, str(entrypoint.parent))
    sys.argv = [str(entrypoint), *official_args]
    launch = {"__name__": "__main__", "__file__": str(entrypoint), "__package__": None}
    from workers.speaker_pipeline import run_speaker_job

    settings = json.loads(args.diarization_settings.read_text(encoding="utf-8"))
    run_speaker_job(None, settings, preflight=True)
    launch["stage_tts_diarizen_pipeline"] = EmiliaDiarizenPipeline(settings)
    if args.fine_turns:
        from emilia_segments import pad_exported_tails, refine_turns

        def refine_file(vad, diarization, audio, _path):
            return refine_turns(
                vad, diarization, audio,
                max_turn_seconds=args.max_turn_seconds, min_silence_ms=args.min_silence_ms,
            )

        def pad_tails(segments, audio):
            duration = len(audio["waveform"]) / audio["sample_rate"]
            return pad_exported_tails(segments, duration, args.tail_ms)

        launch["stage_tts_refine_turns"] = refine_file
        launch["stage_tts_pad_tails"] = pad_tails
    exec(compile(source, str(entrypoint), "exec"), launch)


if __name__ == "__main__":
    main()
