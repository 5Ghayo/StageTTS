"""Audio VAD chunking and sentence boundaries without subtitle input."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from workers.aligned_segments import normalize_segments
from workers.speech_segmentation import speech_chunks


def test_real_pauses_separate_short_utterances_instead_of_merging_to_30_seconds():
    regions = [(1, 2), (2.1, 3), (3.3, 4), (8, 9)]
    chunks = speech_chunks(regions, duration=10)
    assert [c["segments"] for c in chunks] == [[(1, 2), (2.1, 3)], [(3.3, 4)], [(8, 9)]]
    assert [(round(c["start"], 2), round(c["end"], 2)) for c in chunks] == [(0.97, 3.03), (3.27, 4.03), (7.97, 9.03)]


def test_short_responses_survive_and_padding_never_overlaps_or_leaves_audio():
    chunks = speech_chunks([(0, 0.12), (0.13, 0.3), (0.31, 0.5)], max_seconds=0.2, duration=0.5)
    assert len(chunks) == 3
    assert chunks[0]["start"] == 0
    assert chunks[-1]["end"] == 0.5
    assert all(a["end"] <= b["start"] for a, b in zip(chunks, chunks[1:]))


def test_no_evenly_spaced_cuts_are_invented_for_long_continuous_speech():
    with pytest.raises(ValueError, match="did not split"):
        speech_chunks([(0, 20)])
    assert speech_chunks([]) == []


@pytest.mark.parametrize("regions", [[(0, float('nan'))], [(2, 1)], [(-1, 1)], [(0, 2), (1, 3)]])
def test_invalid_or_overlapping_vad_cannot_publish_chunks(regions):
    with pytest.raises(ValueError):
        speech_chunks(regions)


def test_sentence_punctuation_splits_even_short_asr_windows_and_is_not_remerged():
    raw = [{"start": 0, "end": 2, "text": "Hi. Yes?", "speaker": "A", "words": [
        {"word": "Hi.", "start": 0, "end": 0.8, "speaker": "A"},
        {"word": "Yes?", "start": 1, "end": 2, "speaker": "A"}]}]
    result = normalize_segments(raw, diarized=True, split_sentences=True)
    assert [r["orig_text"] for r in result] == ["Hi.", "Yes?"]
    assert all(r["speaker"] == "A" for r in result)
    assert all(r["boundary_provenance"][0]["reasons"] == ["terminal_punctuation"] for r in result)


def test_long_pause_splits_sentence_mode_but_compound_guard_still_applies():
    from workers.semantic_boundaries import LexicalToken

    class Tokenizer:
        name, version = "fixture", "1"

        def tokenize(self, text):
            return [LexicalToken("一室", 0, 2), LexicalToken("好", 2, 3)]

    raw = [{"start": 0, "end": 3, "text": "一室好", "words": [
        {"word": "一", "start": 0, "end": 0.5}, {"word": "室", "start": 1.2, "end": 1.5},
        {"word": "好", "start": 2.2, "end": 3}]}]
    result = normalize_segments(raw, language="zh", tokenizer=Tokenizer(), split_sentences=True)
    assert [r["orig_text"] for r in result] == ["一室", "好"]


def test_parse_stage_uses_fine_audio_vad_before_asr_and_keeps_normal_alignment(tmp_path):
    from workers import parse_worker

    events = []
    model = SimpleNamespace(transcribe=lambda *a, **kw: (
        events.append(("asr", kw)) or {"language": "ja", "segments": [{"start": 0, "end": 1, "text": "はい"}]}))
    result = {"segments": [{"start": 0, "end": 1, "text": "はい", "words": []}]}
    whisper = SimpleNamespace(load_audio=lambda _: "audio", load_model=lambda *a, **kw: model,
                              load_align_model=lambda **kw: ("aligner", {}), align=lambda *a, **kw: result)
    with patch.dict(sys.modules, {"whisperx": whisper}), \
         patch.object(parse_worker, "_enable_pyannote_torch_load_compat"), \
         patch.object(parse_worker, "_release_cuda_memory"), \
         patch("workers.speech_segmentation.enable_sentence_vad", side_effect=lambda _: events.append(("vad", {}))):
        audio, aligned, lang = parse_worker._stage_whisperx_asr_align(tmp_path / "dialogue.wav", "float16", "large-v3", "cuda")
    assert [e[0] for e in events] == ["vad", "asr"]
    assert events[1][1]["chunk_size"] == 8
    assert aligned is result and lang == "ja" and audio == "audio"


def test_retired_subtitle_toolchain_configuration_is_rejected(tmp_path):
    from workers.research_training import load_config

    config = tmp_path / "toolchain.json"
    config.write_text('{"videosubfinder_cli":"/old/tool"}')
    with pytest.raises(ValueError, match="Subtitle/OCR input"):
        load_config(config)
