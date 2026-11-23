"""Run the pinned Emilia CLI with explicit upstream clip-filter overrides."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


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
    parser.add_argument("--turn-boundaries", type=Path)
    args, official_args = parser.parse_known_args()
    entrypoint = args.entrypoint.resolve()
    source = apply_filter_overrides(
        entrypoint.read_text(encoding="utf-8"), args.min_clip_seconds, args.min_dnsmos,
        fine_turns=args.fine_turns,
    )
    sys.path.insert(0, str(entrypoint.parent))
    sys.argv = [str(entrypoint), *official_args]
    launch = {"__name__": "__main__", "__file__": str(entrypoint), "__package__": None}
    if args.fine_turns:
        from emilia_segments import pad_exported_tails, refine_turns

        boundaries = []
        if args.turn_boundaries is not None:
            data = json.loads(args.turn_boundaries.read_text(encoding="utf-8"))
            boundaries = sorted({float(t) for t in data["cut_seconds"]})
            if any(not math.isfinite(t) or t < 0 for t in boundaries):
                parser.error("Turn boundaries must be finite nonnegative seconds")

        def refine_file(vad, diarization, audio, path):
            is_original = Path(path).stem == "original"
            return refine_turns(
                vad, diarization, audio,
                boundaries=boundaries if is_original else (),
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
