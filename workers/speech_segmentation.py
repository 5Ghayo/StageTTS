"""Sentence-sized ASR input from WhisperX's own audio VAD scores."""

from __future__ import annotations

import math

ASR_MAX_SPEECH_SECONDS = 8.0
ASR_MIN_SILENCE_SECONDS = 0.25
ASR_SPEECH_PAD_SECONDS = 0.03


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
            return self.last_chunks

    adapter = SentenceVad()
    pipeline.vad_model = adapter
    return adapter
