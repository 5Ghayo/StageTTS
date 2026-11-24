"""Turn-boundary regressions using fixed alignment and speaker activity fixtures."""

import copy
from unittest.mock import patch

import pytest

from workers.parse_worker import _stage_diarize


def word(text, start=None, end=None):
    return {"word": text, "start": start, "end": end}


def activity(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def diarize(tmp_path, raw, spans):
    context = {}
    with patch("workers.speaker_pipeline.run_speaker_job", return_value={"diarization_spans": spans}):
        result = _stage_diarize(tmp_path / "dialogue.wav", raw, "cpu", False, context)
    return result, context


def test_speaker_change_uses_word_bounds_and_preserves_raw_evidence(tmp_path):
    raw = {"segments": [{"start": 0, "end": 4.5, "text": "Wait here.",
                         "words": [word("Wait", 1, 2), word("here.", 3, 4)]}]}
    untouched = copy.deepcopy(raw)
    segments, context = diarize(tmp_path, raw, [activity(1, 2, "A"), activity(3, 4, "B")])
    assert [(s["start"], s["end"], s["speaker"]) for s in segments] == [(1, 2, "A"), (3, 4, "B")]
    assert [s["orig_text"] for s in segments] == ["Wait", "here."]
    assert all(not s["speaker_needs_review"] for s in segments)
    assert all(not s["alignment_needs_review"] for s in segments)
    assert raw == untouched
    assert context["raw_segments"] == untouched["segments"]
    assert all("speaker_change" in s["boundary_provenance"][0]["reasons"] for s in segments)


def test_next_speaker_in_asr_tail_does_not_override_the_aligned_speaker(tmp_path):
    raw = {"segments": [{"start": 0, "end": 3, "text": "Wait.", "words": [word("Wait.", 1, 2)]}]}
    segments, _ = diarize(tmp_path, raw, [activity(1, 2, "A"), activity(2.4, 3, "B")])
    assert len(segments) == 1
    assert (segments[0]["start"], segments[0]["end"], segments[0]["speaker"]) == (1, 2, "A")
    assert not segments[0]["speaker_needs_review"]


@pytest.mark.parametrize("words", [[], [word("Wait.")], [word("Wait.", 1, 4)],
                                    [word("Wrong.", 1, 2)]])
def test_incomplete_or_mismatched_alignment_does_not_trim_or_confirm(tmp_path, words):
    raw = {"segments": [{"start": 0, "end": 3, "text": "Wait.", "words": words}]}
    segments, _ = diarize(tmp_path, raw, [activity(1, 2, "A"), activity(2.4, 3, "B")])
    assert (segments[0]["start"], segments[0]["end"]) == (0, 3)
    assert segments[0]["speaker"] == "UNKNOWN"
    assert segments[0]["speaker_needs_review"]
    assert segments[0]["orig_text"] == "Wait."


def test_word_crossing_turn_boundary_stays_unknown_between_known_turns(tmp_path):
    raw = {"segments": [{"start": 0, "end": 2, "text": "Go now please.", "words": [
        word("Go", 0.1, 0.8), word("now", 0.85, 1.15), word("please.", 1.2, 1.9)]}]}
    segments, _ = diarize(tmp_path, raw, [activity(0, 1, "A"), activity(1, 2, "B")])
    assert [s["speaker"] for s in segments] == ["A", "UNKNOWN", "B"]
    assert [s["orig_text"] for s in segments] == ["Go", "now", "please."]
    assert segments[1]["speaker_needs_review"]
    assert segments[1]["speaker_fallback_reason"] == "multiple_speaker_turns"


def test_genuine_overlapping_activity_is_not_fixed_by_trimming(tmp_path):
    raw = {"segments": [{"start": 0, "end": 3, "text": "Wait.", "words": [word("Wait.", 1, 2)]}]}
    segments, _ = diarize(tmp_path, raw, [activity(1, 2, "A"), activity(1.5, 2.5, "B")])
    assert segments[0]["speaker"] == "UNKNOWN"
    assert segments[0]["speaker_fallback_reason"] == "overlapping_speaker_activity"


def test_japanese_fast_exchange_preserves_character_text_and_real_times(tmp_path):
    raw = {"segments": [{"start": 0, "end": 2, "text": "待ってはい", "words": [
        word("待", 0.2, 0.4), word("っ", 0.4, 0.5), word("て", 0.5, 0.7),
        word("は", 0.9, 1.1), word("い", 1.1, 1.2)]}]}
    context = {}
    with patch("workers.speaker_pipeline.run_speaker_job", return_value={
        "diarization_spans": [activity(0.2, 0.7, "A"), activity(0.9, 1.2, "B")]
    }):
        segments = _stage_diarize(tmp_path / "dialogue.wav", raw, "cpu", False, context, language="ja")
    assert [s["orig_text"] for s in segments] == ["待って", "はい"]
    assert [(s["start"], s["end"], s["speaker"]) for s in segments] == [(0.2, 0.7, "A"), (0.9, 1.2, "B")]


def test_overlapping_word_alignment_keeps_original_window_for_review(tmp_path):
    raw = {"segments": [{"start": 0, "end": 3, "text": "Wait here.", "words": [
        word("Wait", 1, 1.8), word("here.", 1.7, 2)]}]}
    segments, _ = diarize(tmp_path, raw, [activity(1, 2, "A"), activity(2.4, 3, "B")])
    assert (segments[0]["start"], segments[0]["end"]) == (0, 3)
    assert segments[0]["speaker"] == "UNKNOWN"
