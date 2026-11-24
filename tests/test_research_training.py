import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from workers.emilia_runner import apply_diarizen_override, apply_filter_overrides
from workers.emilia_segments import exclusive_turns, pad_exported_tails, split_turns_at_boundaries
from workers.research_training import character_for_clip, load_config, overlap_windows, prepare_dataset
from workers.subtitle_boundaries import detect_subtitle_boundaries, srt_cut_seconds


def turn(speaker, start, end, needs_review=False):
    return {"speaker": speaker, "start": start, "end": end, "speaker_needs_review": needs_review}


def test_emilia_diarizer_override_removes_token_gate_and_keeps_processing():
    source = '''from pyannote.audio import Pipeline
def main_process():
    speakerdia = speaker_diarization(audio)
    vad_list = vad.vad(speakerdia, audio)
if __name__ == "__main__":
    if not cfg["huggingface_token"].startswith("hf"):
        raise ValueError("token")
    dia_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    dia_pipeline.to(device)
    asr_model = load_asr()
'''
    result = apply_diarizen_override(source)
    compile(result, "fixture", "exec")
    assert "huggingface_token" not in result
    assert "Pipeline.from_pretrained" not in result
    assert "dia_pipeline = stage_tts_diarizen_pipeline" in result
    assert "asr_model = load_asr()" in result
    assert "save_report(save_path)" in result
    with pytest.raises(ValueError, match="Unexpected Emilia"):
        apply_diarizen_override("changed upstream")


def test_emilia_overrides_apply_before_asr_and_during_final_filter():
    source = (
        "MIN_SEGMENT_LENGTH = 3  # min segment length in seconds\n"
        "calculate_audio_stats(mos_list)\n"
        "    filtered_segment = len(filtered_audio_stats)\n"
    )
    adjusted = apply_filter_overrides(source, 2.0, 2.6)

    assert "MIN_SEGMENT_LENGTH = 2.0" in adjusted
    assert "calculate_audio_stats(mos_list, min_duration=2.0, min_dnsmos=2.6)" in adjusted
    assert "if not all_audio_stats:\n        return []" in adjusted
    with pytest.raises(ValueError, match="Unexpected Emilia CLI source"):
        apply_filter_overrides("calculate_audio_stats(mos_list)", 2.0, 2.6)


def test_exclusive_turns_retains_short_turn_and_removes_shared_audio():
    timeline = [turn("A", 0, 1), turn("B", 0.8, 1.15), turn("A", 1.15, 1.5)]

    assert exclusive_turns(timeline) == [(0, 0.8, "A"), (1, 1.15, "B"), (1.15, 1.5, "A")]


def test_subtitle_cut_points_split_even_one_continuous_diarization_turn():
    spans = [(0.5, 6.5, "A")]
    cuts = [0.5, 1.5, 4.0, 6.5]

    assert list(split_turns_at_boundaries(spans, cuts)) == [
        (0.5, 1.5, "A"), (1.5, 4.0, "A"), (4.0, 6.5, "A"),
    ]


def test_exported_tails_can_overlap_a_neighbor_without_changing_speech_times():
    speech = [turn("A", 0, 1), turn("B", 1, 1.95)]
    exported = pad_exported_tails(speech, duration=2, tail_ms=150)

    assert [clip["end"] for clip in exported] == [1.15, 2]
    assert [clip["speech_end"] for clip in exported] == [1, 1.95]
    assert speech[0]["end"] == 1


def test_embedded_subtitle_times_are_deduplicated_and_sorted():
    subtitles = (
        "1\r\n00:00:04,250 --> 00:00:06,000\r\nこんにちは\r\n\r\n"
        "2\r\n00:00:04,250 --> 00:00:06,000\r\n你好\r\n\r\n"
        "3\r\n00:00:06,080 --> 00:00:07,100\r\n次\r\n"
    )

    assert srt_cut_seconds(subtitles) == [4.25, 6.0, 6.08, 7.1]


@pytest.mark.parametrize("has_text", [False, True])
def test_subtitle_detection_prefers_text_track_and_falls_back_to_video(tmp_path, has_text):
    video = tmp_path / "video with spaces.mkv"
    srt = "1\n00:00:01,250 --> 00:00:03,500\n[2.25]\n"
    streams = [{"codec_type": "video", "index": 0}]
    if has_text:
        streams.append({"codec_type": "subtitle", "codec_name": "ass", "index": 2})

    def run(args, **kwargs):
        if args[0] == "ffprobe":
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"streams": streams}))
        if args[0] == "ffmpeg":
            return subprocess.CompletedProcess(args, 0, stdout=srt)
        assert args[0] == "/tools/VideoSubFinderCli.run"
        assert args[args.index("-i") + 1] == str(video)
        assert "-dsi" in args
        Path(args[args.index("-ces") + 1]).write_text(srt, encoding="utf-8-sig")
        return subprocess.CompletedProcess(args, 0)

    with patch("workers.subtitle_boundaries.subprocess.run", side_effect=run) as command:
        result = detect_subtitle_boundaries(video, videosubfinder_cli="/tools/VideoSubFinderCli.run")

    assert result["cut_seconds"] == [1.25, 3.5]
    assert command.call_count == 2
    if has_text:
        assert result["stream"] == 2
    else:
        assert result["detector"] == "videosubfinder"


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

    def emilia(_config, inputs, filters):
        assert filters["min_clip_seconds"] == 2.0
        assert filters["min_dnsmos"] == 2.6
        assert json.loads(Path(filters["turn_boundaries"]).read_text())["cut_seconds"] == [1, 2]
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
        patch("workers.research_training.detect_subtitle_boundaries",
              return_value={"stream": 7, "codec": "ass", "cut_seconds": [1, 2]}) as detect,
        patch("workers.research_training.run_job", side_effect=model_job),
        patch("workers.research_training.run_emilia", side_effect=emilia),
    ):
        prepare_dataset({}, output, original, None, [overlap],
                        emilia_filter={"fine_turns": True, "min_clip_seconds": 2.0, "min_dnsmos": 2.6},
                        subtitle_source=original)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    detect.assert_called_once_with(original, videosubfinder_cli=None)
    assert stages == ["dialogue-sidon", "emilia", "features"]
    assert manifest["emilia_filter"] == {
        "fine_turns": True, "min_clip_seconds": 2.0, "min_dnsmos": 2.6, "turn_boundaries": "subtitle-boundaries.json",
    }
    assert (output / "subtitle-boundaries.json").is_file()
    assert [row["provenance"].get("track") for row in manifest["clips"]] == [None, 1, 2]
    for clip in manifest["clips"]:
        assert clip["character_id"] is None
        assert (output / clip["audio"]).is_file()
        assert (output / clip["features"]).is_file()


def test_voice_corpus_uses_scoped_speaker_ids_without_feature_models(tmp_path):
    from argparse import Namespace

    from workers.research_voice_corpus import build_voice_corpus

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    rows = []
    for i, (source, speaker) in enumerate([("one", "UNKNOWN"), ("one", "A"), ("one", "A"), ("two", "A")]):
        audio = corpus / f"{i}.wav"
        audio.touch()
        rows.append({"audio": audio.name, "start": 0, "end": 4, "speaker": speaker,
                     "text": "はい。", "language": "ja",
                     "provenance": {"source": source, "origin": "original"}})
    (corpus / "manifest.json").write_text(json.dumps({"clips": rows}))
    output = tmp_path / "voices"
    args = Namespace(corpus=corpus, output=output, min_clips=1, min_seconds=1,
                     min_clip_seconds=1, max_clip_seconds=10)

    def ffmpeg(command, **kwargs):
        Path(command[-1]).touch()

    with patch("workers.research_voice_corpus.subprocess.run", side_effect=ffmpeg):
        build_voice_corpus(args)
    report = json.loads((output / "voices.json").read_text())
    assert report["group_by"] == "speaker"
    assert [[clip["row"] for clip in voice["clips"]] for voice in report["voices"]] == [[1, 2], [3]]
    for voice in report["voices"]:
        labels = (output / "training-candidates" / voice["voice"] / "labels.list").read_text()
        assert "|ja|はい。" in labels
        assert all("similarity" not in clip for clip in voice["clips"])
