"""Refine Emilia's diarized intervals without joining separate speech turns."""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict


def exclusive_turns(turns):
    events = defaultdict(Counter)
    for turn in turns:
        start, end, speaker = float(turn["start"]), float(turn["end"]), turn["speaker"]
        if start < end:
            events[start][speaker] += 1
            events[end][speaker] -= 1
    active = Counter()
    previous = None
    spans = []
    for boundary in sorted(events):
        if previous is not None and len(active) == 1:
            speaker = next(iter(active))
            if spans and spans[-1][2] == speaker and spans[-1][1] == previous:
                spans[-1] = (spans[-1][0], boundary, speaker)
            else:
                spans.append((previous, boundary, speaker))
        for speaker, delta in events[boundary].items():
            active[speaker] += delta
            if not active[speaker]:
                del active[speaker]
        previous = boundary
    return spans


def split_turns_at_boundaries(turns, boundaries):
    for start, end, speaker in turns:
        cuts = [start, *boundaries[bisect_right(boundaries, start):bisect_left(boundaries, end)], end]
        yield from ((left, right, speaker) for left, right in zip(cuts, cuts[1:]))


def pad_exported_tails(segments, duration, tail_ms):
    """Extend the WAV ranges after transcription and filtering."""
    return [
        {**clip, "speech_end": clip["end"], "end": min(duration, clip["end"] + tail_ms / 1000)}
        for clip in segments
    ] if tail_ms else segments


def refine_turns(
    vad, diarization, audio, *, boundaries=(), max_turn_seconds=15, min_silence_ms=150,
):
    import librosa

    rate = 16000
    waveform = librosa.resample(audio["waveform"], orig_sr=audio["sample_rate"], target_sr=rate)
    segments = []
    turns = exclusive_turns(diarization.to_dict(orient="records"))
    for start, end, speaker in split_turns_at_boundaries(turns, boundaries):
        first, last = max(0, round(start * rate)), min(len(waveform), round(end * rate))
        if last - first < round(0.16 * rate):
            continue
        timestamps = vad.get_speech_timestamps(
            waveform[first:last], vad.vad_model, sampling_rate=rate,
            min_speech_duration_ms=160, min_silence_duration_ms=min_silence_ms, speech_pad_ms=30,
            max_speech_duration_s=max_turn_seconds,
        )
        for stamp in timestamps:
            clip_start = max(start, (first + stamp["start"]) / rate)
            clip_end = min(end, (first + stamp["end"]) / rate)
            if clip_start < clip_end:
                segments.append({
                    "index": f"{len(segments):05d}", "start": clip_start,
                    "end": clip_end, "speaker": speaker,
                })
    return segments
