"""Speaker assignment, overlap constraints and isolation without loading ML models."""

import copy
import json
import math
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from workers.diarizen_worker import run_request, scope_activity, source_activity, validate_request
from workers.speaker_pipeline import assign_interval, assign_transcript, run_speaker_job, validate_spans


def span(start, end, speaker):
    return {"start": start, "end": end, "speaker": speaker}


def test_assignment_uses_union_coverage_and_never_nearest_or_majority():
    activity = [span(0, 0.5, "A"), span(0.2, 1, "A"), span(1, 2, "B"), span(4, 4.1, "C")]
    assert assign_interval(0, 1, activity) == ("A", None)
    assert assign_interval(0, 2, activity) == ("UNKNOWN", "multiple_speaker_turns")
    assert assign_interval(2, 3, activity) == ("UNKNOWN", "speaker_assignment_missing")
    assert assign_interval(4, 5, activity) == ("UNKNOWN", "insufficient_speaker_coverage")
    assert assign_interval(0, 1, [span(0, 1, "A"), span(0.5, 1, "B")]) == (
        "UNKNOWN", "overlapping_speaker_activity")


def test_assignment_is_immutable_and_clears_stale_word_labels():
    raw = [{"start": 0, "end": 2, "text": "One two", "speaker": "STALE", "words": [
        {"word": "One", "start": 0, "end": 1, "speaker": "STALE"},
        {"word": "two", "speaker": "STALE"}]}]
    untouched = copy.deepcopy(raw)
    result = assign_transcript(raw, [span(0, 1, "A"), span(1, 2, "B")])
    assert raw == untouched
    assert result[0]["text"] == "One two"
    assert result[0]["speaker"] == "UNKNOWN"
    assert result[0]["words"][0]["speaker"] == "A"
    assert result[0]["words"][1]["speaker"] is None


@pytest.mark.parametrize("row", [span(0, math.nan, "A"), span(-1, 2, "A"), span(1, 1, "A"),
                                 span(0, 2, "UNKNOWN"), span(0, 100, "A")])
def test_invalid_worker_activity_is_rejected(row):
    with pytest.raises(ValueError):
        validate_spans([row], duration=10)


def test_frame_padding_is_clipped_to_real_audio_and_recorded_for_review():
    original = [span(-0.01, 2, "A"), span(4, 7, "B"), span(7, 8, "C")]
    activity = source_activity(original, 5)
    assert [(s["start"], s["end"]) for s in activity] == [(0, 2), (4, 5)]
    assert all("clipped_diarizer_padding_to_source_bounds" in s["review_reasons"] for s in activity)
    assert original[0]["start"] == -0.01


def test_worker_rejects_missing_models_without_importing_or_downloading():
    result = run_request({"protocol_version": "1.0", "model_path": "BUT-FIT/diarizen"})
    assert result["status"] == "error"
    assert "absolute local" in result["reason"]
    with pytest.raises(ValueError):
        validate_request({"protocol_version": "2.0"})


def test_orchestrator_runs_the_selected_interpreter_and_validates_response(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.touch()

    def subprocess_run(command, **kwargs):
        assert command[0] == sys.executable
        assert Path(command[1]).name == "diarizen_worker.py"
        request = json.loads(Path(command[command.index("--request") + 1]).read_text())
        assert request["audio_path"] == str(audio)
        assert kwargs["timeout"] == 123
        Path(command[command.index("--output") + 1]).write_text(json.dumps({
            "protocol_version": "1.0", "status": "completed", "diarization_spans": [span(0, 1, "A")]}))

    with patch("workers.speaker_pipeline.subprocess.run", side_effect=subprocess_run):
        result = run_speaker_job(audio, {"python": sys.executable, "timeout": 123})
    assert result["diarization_spans"] == [span(0, 1, "A")]


def test_scoping_preserves_diarizen_groups_overlaps_and_input():
    original = [span(0, 2, "A"), span(1, 3, "B"), span(4, 6, "A")]
    untouched = copy.deepcopy(original)
    first, report = scope_activity(original, "recording1")
    second, _ = scope_activity(original, "recording2")
    assert original == untouched
    assert [(row["start"], row["end"]) for row in first] == [(0, 2), (1, 3), (4, 6)]
    assert first[0]["speaker"] == first[2]["speaker"]
    assert first[0]["speaker"] != first[1]["speaker"]
    assert first[0]["speaker"] != second[0]["speaker"]
    assert len(report["speaker_mapping"]) == 2


def test_preflight_requires_only_diarizen_and_its_internal_embedding(tmp_path):
    model = tmp_path / "model"
    for name in ("config.toml", "pytorch_model.bin", "plda/plda.npz", "plda/xvec_transform.npz"):
        file = model / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    embedding = tmp_path / "wespeaker.bin"
    embedding.touch()
    validate_request({"protocol_version": "1.0", "model_path": str(model),
                      "embedding_path": str(embedding), "preflight": True, "device": "cpu"})
