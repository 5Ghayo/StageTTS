"""The speaker stage receives fixed audio clips and cannot change their boundaries."""

import copy
from unittest.mock import patch

import pytest

from workers.aligned_segments import normalize_segments
from workers.parse_worker import _stage_diarize


def test_cluster_uses_trimmed_source_clip_and_keeps_dubbing_and_word_times(tmp_path):
    raw = {"segments": [{"start": 1, "end": 2, "text": "Wait.", "words": [
        {"word": "Wait.", "start": 1, "end": 2},
    ]}]}
    prepared = normalize_segments(raw["segments"], semantic_cuts=frozenset())
    prepared[0].update(source_audio_start=1.2, source_audio_end=2.4)
    untouched = copy.deepcopy(prepared)
    worker = {
        "diarization_spans": [{"start": 1.2, "end": 2.4, "speaker": "A"}],
        "dialogue_assignments": [{"id": 0, "speaker": "A"}],
    }

    with patch("workers.speaker_pipeline.run_speaker_job", return_value=worker) as run:
        line, = _stage_diarize(tmp_path / "dialogue.wav", raw, "cpu", False, prepared_segments=prepared)

    assert run.call_args.args[1]["dialogue_units"] == [{"id": 0, "start": 1.2, "end": 2.4}]
    assert (line["start"], line["end"], line["speaker"]) == (1, 2, "A")
    assert (line["source_audio_start"], line["source_audio_end"]) == (1.2, 2.4)
    assert line["words"] == [{"word": "Wait.", "start": 1, "end": 2, "speaker": "A"}]
    assert not line["speaker_needs_review"]
    assert prepared == untouched


@pytest.mark.parametrize("assignments", [[], [{"id": 1, "speaker": "A"}], [{"id": 0, "speaker": "A"}] * 2])
def test_cluster_rejects_missing_wrong_or_duplicate_ids_before_publishing(tmp_path, assignments):
    raw = {"segments": [{"start": 1, "end": 2, "text": "Wait."}]}
    with patch("workers.speaker_pipeline.run_speaker_job", return_value={
        "diarization_spans": [], "dialogue_assignments": assignments,
    }), pytest.raises(RuntimeError, match="one assignment per dialogue unit"):
        _stage_diarize(tmp_path / "dialogue.wav", raw, "cpu", False)
    assert not (tmp_path / "speaker_analysis.json").exists()
