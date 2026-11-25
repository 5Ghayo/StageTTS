import math
from unittest.mock import Mock, patch

import pytest

from backend.schemas import MediaInfo, ProjectScript, Segment
from backend.services.dialogue_cues import (
    apply_subtitle_references,
    attach_speech_cues,
    exclude_quiet_training_clips,
    parse_subtitle_cues,
)
from backend.services.subtitles import generate_ass_subtitles
from backend.services.synthesis import _check_duration
from backend.services.translation import DialogueContext, translate_script


def test_audio_cue_is_the_anchor_for_subtitles_translation_and_dubbing():
    rows = [
        {"id": 0, "start": 0.8, "end": 2.0, "speech_end": 1.8,
         "source_audio_start": 1.0, "source_audio_end": 2.1, "speaker": "A", "orig_text": "よろいかさん？"},
        {"id": 1, "start": 2.1, "end": 3.0, "speech_end": 3.0,
         "source_audio_start": 2.4, "source_audio_end": 3.3, "speaker": "B", "orig_text": "はい。"},
    ]
    segments = [Segment(**row) for row in rows]
    script = ProjectScript(project_id="cue-test", media=MediaInfo(duration=5), source_lang="ja", segments=segments)
    attach_speech_cues(script.segments, rows, None, 5)
    raw = "1\n00:00:01,010 --> 00:00:01,750\n鎧塚さん？\n\n2\n00:00:04,200 --> 00:00:04,800\n字幕仅有。\n"
    apply_subtitle_references(script.segments, "ja", parse_subtitle_cues(raw.encode()))
    raw = "WEBVTT\n\n00:01.000 --> 00:01.750\n<v A>铠塚同学？\n"
    apply_subtitle_references(script.segments, "zh", parse_subtitle_cues(raw.encode()))

    first, second = script.segments
    cue = first.speech_cue
    assert (first.start, first.end, first.dub_start, first.dub_end) == (0.8, 2.0, 1.0, 2.4)
    assert (cue.reference_ja, cue.reference_zh) == ("鎧塚さん？", "铠塚同学？")
    assert second.speech_cue.reference_ja is None and second.speech_cue.rms_dbfs is None
    assert len(script.segments) == 2

    entries = [{"id": 0, "target_text": "是铠塚同学吗？", "corrected_source_text": "鎧塚さん？",
                "self_check": {"semantic_complete": True, "omitted_info": ""}},
               {"id": 1, "target_text": "是。", "corrected_source_text": "はい。",
                "self_check": {"semantic_complete": True, "omitted_info": ""}}]
    with patch("backend.services.translation.complete_dialogue", return_value=entries) as llm:
        assert translate_script(script, Mock()) == [0, 1]
    payload = llm.call_args.args[2]["lines"][0]
    assert (payload["orig_text"], payload["reference_ja"], payload["reference_zh"]) == (
        "よろいかさん？", "鎧塚さん？", "铠塚同学？",
    )
    assert DialogueContext(script).entry(first)["slot_sec"] == 1.4
    assert cue.corrected_source_text == "鎧塚さん？" and first.orig_text == "よろいかさん？"
    assert first.target_text == "是铠塚同学吗？"

    first.status = "ready"
    first.actual_duration = 1.1
    assert _check_duration(first) is None
    assert "0:00:01.00,0:00:02.10" in generate_ass_subtitles(script)
    first.actual_duration = 1.5
    assert _check_duration(first).duration_state == "too_long"



@pytest.mark.parametrize("amplitude", [0.25, 0.001])
def test_cue_loudness_uses_active_speech_and_ignores_tail_silence(amplitude):
    np = pytest.importorskip("numpy")
    row = {"id": 0, "start": 0.8, "end": 2.0, "speech_end": 1.8,
           "source_audio_start": 1.0, "source_audio_end": 2.1, "speaker": "A"}
    segment = Segment(**row)
    samples = np.zeros(5 * 16000, dtype=np.float32)
    samples[16000:round(1.6 * 16000)] = amplitude
    attach_speech_cues([segment], [row], samples, 5)
    assert segment.speech_cue.rms_dbfs == pytest.approx(20 * math.log10(amplitude), abs=0.1)
    exclude_quiet_training_clips([segment])
    assert segment.training_excluded is (amplitude == 0.001)
    assert segment.training_exclusion_reasons == (["low_rms_dbfs"] if amplitude == 0.001 else [])
