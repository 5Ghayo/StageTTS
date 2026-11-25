"""Sentence-sized ASR input from WhisperX's own audio VAD scores."""

from __future__ import annotations

import math
from bisect import bisect_right

ASR_MAX_SPEECH_SECONDS = 8.0
ASR_MIN_SILENCE_SECONDS = 0.25
ASR_SPEECH_PAD_SECONDS = 0.03
ASR_VAD_THRESHOLD = 0.65
RECHECK_VAD_THRESHOLD = 0.80
PLAYBACK_TAIL_PAD_SECONDS = 0.25


def speech_playback_ranges(
    speech: list[tuple[float, float]], duration: float,
) -> list[tuple[float, float]]:
    """Start at detected speech and keep the tail of each midpoint cut."""
    bounds = [(start, min(duration, end + PLAYBACK_TAIL_PAD_SECONDS)) for start, end in speech]
    for i, ((_, left_end), (right_start, _)) in enumerate(zip(speech, speech[1:])):
        end = (left_end + right_start) / 2 if left_end <= right_start else left_end
        bounds[i] = (bounds[i][0], end)
    return bounds


def attach_playback_ranges(
    segments: list[dict], chunks: list[dict], duration: float,
) -> None:
    """Apply speech-onset trimming and midpoint tails to each VAD window."""
    speech = [(chunk["segments"][0][0], chunk["segments"][-1][1]) for chunk in chunks]
    ranges = speech_playback_ranges(speech, duration)
    starts = [chunk["start"] for chunk in chunks]
    by_window: dict[int, list[int]] = {}
    for i, segment in enumerate(segments):
        window = bisect_right(starts, segment["start"] + 0.002) - 1
        if window >= 0 and segment["end"] <= chunks[window]["end"] + 0.002:
            by_window.setdefault(window, []).append(i)
    for window, indices in by_window.items():
        start, end = ranges[window]
        first, last = indices[0], indices[-1]
        if last + 1 < len(segments):
            end = min(end, segments[last + 1]["start"])
        for i in indices:
            segment = segments[i]
            playback_end = round(max(segment["end"], end), 3) if i == last else segment["end"]
            playback_start = round(start, 3)
            segment["source_audio_start"] = playback_start if i == first and playback_start < playback_end else segment["start"]
            segment["source_audio_end"] = playback_end


def speech_chunks(
    regions: list[tuple[float, float]], *, max_seconds: float = ASR_MAX_SPEECH_SECONDS,
    min_silence: float = ASR_MIN_SILENCE_SECONDS, duration: float | None = None,
) -> list[dict]:
    """Keep pauses between utterances; join only small gaps within a short window.

    Long continuous regions must already be split by WhisperX's minimum-score
    VAD cut. No evenly spaced timestamps or text-derived audio cuts are created.
    """
    if not math.isfinite(max_seconds) or max_seconds <= 0 or not math.isfinite(min_silence) or min_silence < 0:
        raise ValueError("Invalid speech segmentation settings")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("Invalid audio duration")
    chunks = []
    for start, end in sorted(regions):
        if not all(math.isfinite(t) for t in (start, end)) or not 0 <= start < end:
            raise ValueError("Invalid VAD speech region")
        if end - start > max_seconds + 0.05:
            raise ValueError("WhisperX VAD did not split a long speech region")
        if duration is not None:
            end = min(end, duration)
        if end <= start:
            continue
        if chunks and start < chunks[-1]["end"]:
            raise ValueError("Overlapping VAD regions cannot be assigned separate ASR windows")
        if (chunks and start - chunks[-1]["end"] < min_silence
                and end - chunks[-1]["start"] <= max_seconds):
            chunks[-1]["end"] = end
            chunks[-1]["segments"].append((start, end))
        else:
            chunks.append({"start": start, "end": end, "segments": [(start, end)]})
    # Small context protects consonants at VAD boundaries. Padding never overlaps
    # a neighbor; original VAD regions remain available in ``segments``.
    for i, chunk in enumerate(chunks):
        previous_end = chunks[i - 1]["segments"][-1][1] if i else 0.0
        next_start = chunks[i + 1]["segments"][0][0] if i + 1 < len(chunks) else duration
        chunk["start"] = max(0.0, chunk["start"] - ASR_SPEECH_PAD_SECONDS,
                             (previous_end + chunk["start"]) / 2 if i else 0.0)
        right = chunk["end"] + ASR_SPEECH_PAD_SECONDS
        if next_start is not None:
            right = min(right, (chunk["end"] + next_start) / 2 if i + 1 < len(chunks) else next_start)
        chunk["end"] = right
    return chunks


def enable_sentence_vad(pipeline):
    """Reuse the loaded official Pyannote VAD model; replace only chunk merging."""
    from whisperx.vads.pyannote import Binarize, Pyannote
    from whisperx.vads.vad import Vad

    original = pipeline.vad_model
    if not isinstance(original, Pyannote):
        raise RuntimeError("Sentence segmentation requires WhisperX's Pyannote VAD adapter")

    class SentenceVad(Vad):
        def __init__(self):
            self.duration = None
            self.last_chunks = []
            self.recheck_regions = []

        def preprocess_audio(self, audio):
            return original.preprocess_audio(audio)

        def __call__(self, audio):
            self.duration = audio["waveform"].shape[-1] / audio["sample_rate"]
            return original(audio)

        def merge_chunks(self, scores, chunk_size, onset=0.5, offset=None):
            # Keep the author's acoustic binarization and minimum-score cut.
            annotation = Binarize(max_duration=chunk_size, onset=onset, offset=offset)(scores)
            regions = [(float(turn.start), float(turn.end)) for turn in annotation.get_timeline()]
            self.last_chunks = speech_chunks(regions, max_seconds=chunk_size, duration=self.duration)
            # Reuse exactly the same WhisperX frame scores. Only nominated
            # dialogue cuts will consume the stricter VAD regions.
            review = Binarize(onset=RECHECK_VAD_THRESHOLD, offset=RECHECK_VAD_THRESHOLD)(scores)
            self.recheck_regions = [
                (float(turn.start), float(turn.end)) for turn in review.get_timeline()
            ]
            return self.last_chunks

    adapter = SentenceVad()
    pipeline.vad_model = adapter
    return adapter
