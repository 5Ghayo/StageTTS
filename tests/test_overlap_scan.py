from unittest.mock import patch

from backend.schemas import Segment
from workers.aligned_segments import normalize_segments
from workers.overlap_scan import annotate_overlaps
from workers.parse_worker import _stage_diarize


def test_overlapping_turns_stay_in_clustering_but_are_excluded_from_training(tmp_path):
    raw = {"segments": [
        {"start": 1, "end": 3.8, "text": "こんにちは。"},
        {"start": 5, "end": 6, "text": "はい。"},
    ]}
    prepared = normalize_segments(raw["segments"], semantic_cuts=frozenset())
    annotate_overlaps(prepared, [
        {"start": 1.2, "end": 3.2, "speaker": "A"},
        {"start": 2.1, "end": 3.8, "speaker": "B"},
        {"start": 5.0, "end": 6.0, "speaker": "A"},
    ])
    worker = {
        "diarization_spans": [{"start": 1, "end": 3.8, "speaker": "DZ_A"},
                              {"start": 5, "end": 6, "speaker": "DZ_A"}],
        "dialogue_assignments": [{"id": 0, "speaker": "DZ_A"}, {"id": 1, "speaker": "DZ_A"}],
    }
    with patch("workers.speaker_pipeline.run_speaker_job", return_value=worker) as run:
        mixed, solo = _stage_diarize(tmp_path / "dialogue.wav", raw, "cpu", False, prepared_segments=prepared)
    assert run.call_args.args[1]["dialogue_units"] == [{"id": 0, "start": 1, "end": 3.8}, {"id": 1, "start": 5, "end": 6}]
    model = Segment(**mixed)
    assert model.training_excluded and model.overlap_status == "detected" and model.speaker == "DZ_A"
    assert model.speaker_needs_review and model.speaker_fallback_reason == "source_overlap"
    assert [turn.start for turn in model.overlap_turns] == [1.2, 2.1]
    assert solo["overlap_status"] == "clear" and not solo["training_excluded"]


def test_unavailable_detector_never_marks_audio_as_cleared_for_training():
    row = {"start": 1, "end": 2}
    annotate_overlaps([row], None)
    assert row["overlap_status"] == "unchecked" and row["training_excluded"]
