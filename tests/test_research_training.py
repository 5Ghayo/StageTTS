import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from workers.research_training import character_for_clip, load_config, overlap_windows, prepare_dataset


def turn(speaker, start, end, needs_review=False):
    return {"speaker": speaker, "start": start, "end": end, "speaker_needs_review": needs_review}


def test_overlap_context_does_not_cross_a_third_role():
    timeline = [turn("C", 0, 2), turn("A", 2, 4), turn("B", 3, 6), turn("D", 5, 7)]
    assert overlap_windows(timeline[:3], 7) == [
        {"start": 2, "end": 5.5, "overlap_start": 3, "overlap_end": 4, "candidate_characters": ["A", "B"]},
    ]
    assert overlap_windows(timeline, 7)[0]["end"] == 5
    assert not overlap_windows([turn("A", 2, 4, True), turn("B", 3, 6)], 7)
    assert character_for_clip([turn("A", 2, 4)], 1, 4) is None
    assert character_for_clip([turn("A", 2, 4)], 2, 4) == "A"


def test_toolchain_keeps_venv_python_link(tmp_path):
    env = tmp_path / "venv"
    env.mkdir()
    python = env / "python"
    try:
        python.symlink_to(sys.executable)
    except OSError as exc:
        pytest.skip(f"Symbolic links are unavailable: {exc}")
    config = tmp_path / "toolchain.json"
    config.write_text(json.dumps({"feature_python": "venv/python"}), encoding="utf-8")
    assert load_config(config)["feature_python"] == str(python)


def test_prepare_preserves_native_clips_and_links_model_features(tmp_path):
    original = tmp_path / "input.wav"
    original.write_bytes(b"source audio fixture")
    overlap = tmp_path / "overlap.wav"
    overlap.touch()
    output = tmp_path / "corpus"
    stages = []

    def model_job(_config, stage, job):
        stages.append(stage)
        if stage == "dialogue-sidon":
            assert job["mixtures"][0]["input"] == str(overlap)
            for path in job["mixtures"][0]["outputs"]:
                Path(path).touch()
        else:
            for clip in job["clips"]:
                assert Path(clip["audio"]).is_file()
                Path(clip["features"]).touch()

    def emilia(_config, inputs):
        stages.append("emilia")
        assert (inputs / "original.wav").read_bytes() == original.read_bytes()
        for source in inputs.glob("*.wav"):
            target = inputs.with_name("emilia_processed") / source.stem
            target.mkdir(parents=True)
            (target / f"{source.stem}.json").write_text(json.dumps([
                {"start": 0, "end": 4, "text": "大丈夫。", "language": "ja", "dnsmos": 3.7},
            ]), encoding="utf-8")
            (target / f"{source.stem}_0.mp3").touch()

    with (
        patch("workers.research_training.validate_emilia"),
        patch("workers.research_training.run_job", side_effect=model_job),
        patch("workers.research_training.run_emilia", side_effect=emilia),
    ):
        prepare_dataset({}, output, original, None, [overlap])

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert stages == ["dialogue-sidon", "emilia", "features"]
    assert [row["provenance"].get("track") for row in manifest["clips"]] == [None, 1, 2]
    for clip in manifest["clips"]:
        assert clip["character_id"] is None
        assert (output / clip["audio"]).is_file()
        assert (output / clip["features"]).is_file()
