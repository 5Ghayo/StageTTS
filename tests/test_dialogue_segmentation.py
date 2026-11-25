from unittest.mock import Mock, patch

import pytest

from workers.dialogue_segmentation import resolve_dialogue_boundaries


@pytest.mark.parametrize("has_pause,probabilities,expected_cut", [
    (True, [0, .95, 0, 0], True),
    (False, [0, .95, 0, 0], False),
    (True, [0, .20, 0, 0], False),
    (True, [.99], False),
])
def test_parse_uses_sat_nomination_and_whisperx_pause_midpoint(has_pause, probabilities, expected_cut):
    raw = {"start": 0.3, "end": 1.04, "text": "はいそれ", "words": [
        {"word": "はい", "start": 0.3, "end": 1.0},
        {"word": "それ", "start": 1.0, "end": 1.04},
    ]}
    alignment = {
        "segments": [raw],
        "vad_chunks": [{"start": 0.27, "end": 2.03, "segments": [(0.3, 2.0)]}],
        "vad_duration": 2.5,
        "vad_recheck_regions": [(0.3, 1.1), (1.5, 2.0)] if has_pause else [(0.3, 2.0)],
    }
    with patch("workers.dialogue_segmentation.predict_boundaries", return_value=[probabilities]):
        parts = resolve_dialogue_boundaries(alignment, Mock())
    if expected_cut:
        assert [(p["orig_text"], p["start"], p["end"]) for p in parts] == [
            ("はい", 0.3, 1.3), ("それ", 1.3, 2.0),
        ]
        assert parts[0]["source_audio_start"] == 0.3
        assert parts[0]["source_audio_end"] == 1.3
        assert parts[1]["source_audio_start"] == 1.5
        assert parts[1]["source_audio_end"] == 2.25
        assert parts[1]["words"][0]["start"] is None
    else:
        assert [(p["orig_text"], p["start"], p["end"]) for p in parts] == [("はいそれ", 0.3, 1.04)]


def test_sat_midpoint_tail_is_capped_without_changing_the_timeline():
    alignment = {
        "segments": [{"start": 0.3, "end": 6.0, "text": "はいそれ", "words": [
            {"word": "はい", "start": 0.3, "end": 3.2},
            {"word": "それ", "start": 3.2, "end": 6.0},
        ]}],
        "vad_chunks": [{"start": 0.27, "end": 6.03, "segments": [(0.3, 6.0)]}],
        "vad_duration": 6.5,
        "vad_recheck_regions": [(0.3, 1.0), (5.4, 6.0)],
    }
    with patch("workers.dialogue_segmentation.predict_boundaries", return_value=[[0, .95, 0, 0]]):
        left, right = resolve_dialogue_boundaries(alignment, Mock())
    assert left["end"] == right["start"] == 3.2
    assert left["source_audio_end"] == 3.0
    assert right["source_audio_start"] == 5.4
