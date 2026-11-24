import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend import config
from backend.schemas import CharacterConfig, MediaInfo, ProjectScript, Segment
from backend.services import translation


def completion(entries):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(entries)))])


def make_script(count=1, **segment_kwargs):
    return ProjectScript(
        project_id="m0_translation",
        media=MediaInfo(duration=max(10, count * 3)),
        source_lang="en",
        glossary={},
        characters={"A": CharacterConfig(display_name="主播")},
        segments=[
            Segment(
                id=i,
                start=i * 3,
                end=i * 3 + 2,
                speaker="A",
                orig_text="Please wait here.",
                **segment_kwargs,
            )
            for i in range(count)
        ],
    )


class TranslationUnicodeNonSpeechTest(unittest.TestCase):
    def test_cjk_non_speech_markers_keep_source_audio(self):
        for text in ("[笑声]", "（叹气）", "【掌声】", "[sigh]"):
            with self.subTest(text=text):
                self.assertTrue(translation._is_non_speech(text))

    def test_spoken_short_answers_are_not_non_speech(self):
        for text in ("Yes", "Go", "你好", "是的", "はい", "한국어", "ha ha ha", "哈哈哈", "呵呵", "Here's the basic logic."):
            with self.subTest(text=text):
                self.assertFalse(translation._is_non_speech(text))


class M0TranslationTest(unittest.TestCase):
    def test_normal_english_to_chinese_does_not_trigger_repair(self):
        script = make_script()
        client = MagicMock()
        client.chat.completions.create.return_value = completion([{
            "id": 0,
            "target_text": "请在这里等。",
            "emotion": "neutral",
            "action": "idle",
            "keep_original": False,
            "self_check": {"semantic_complete": True, "omitted_info": ""},
        }])
        self.assertEqual(translation.translate_script(script, client), [0])
        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertEqual(script.segments[0].target_text, "请在这里等。")

    def test_suspicious_translation_triggers_repair_without_losing_metadata(self):
        script = make_script()
        seg = script.segments[0]
        seg.emotion = "happy"
        seg.action = "nod"
        seg.keep_original = True
        seg.status = "ready"
        seg.tts_wav = "human.wav"
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            completion([{
                "id": 0,
                "target_text": "思路",
                "emotion": "sad",
                "action": "shake_head",
                "keep_original": False,
                "self_check": {"semantic_complete": False, "omitted_info": "省略here's basic logic"},
            }]),
            completion([{"id": 0, "target_text": "这是基本思路。", "self_check": {"semantic_complete": True, "omitted_info": ""}}]),
        ]
        self.assertEqual(translation.translate_script(script, client), [0])
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertEqual(seg.target_text, "这是基本思路。")
        self.assertEqual(seg.emotion, "happy")
        self.assertEqual(seg.action, "nod")
        self.assertTrue(seg.keep_original)
        self.assertEqual(seg.status, "ready")
        self.assertIsNone(seg.tts_wav)  # text changed; keep-original audio cache is invalidated, not cancelled

    def test_model_cannot_keep_normal_dialogue_without_human_setting(self):
        script = make_script()
        script.segments[0].orig_text = "Yes."
        client = MagicMock()
        client.chat.completions.create.return_value = completion([{
            "id": 0,
            "target_text": "是。",
            "emotion": "neutral",
            "action": "idle",
            "keep_original": True,
            "self_check": {"semantic_complete": True, "omitted_info": ""},
        }])
        self.assertEqual(translation.translate_script(script, client), [0])
        self.assertFalse(script.segments[0].keep_original)

    def test_explicit_event_marker_can_keep_source_audio(self):
        script = make_script()
        script.segments[0].orig_text = "[laughs]"
        client = MagicMock()
        client.chat.completions.create.return_value = completion([{
            "id": 0,
            "target_text": "[笑声]",
            "emotion": "happy",
            "action": "idle",
            "keep_original": True,
            "self_check": {"semantic_complete": True, "omitted_info": ""},
        }])
        self.assertEqual(translation.translate_script(script, client), [0])
        self.assertTrue(script.segments[0].keep_original)

    def test_current_batch_ids_strict_whitelist_and_partial_success(self):
        script = make_script(count=3)
        for seg in script.segments:
            seg.orig_text = f"Line {seg.id}. The sky is blue and the weather is fine."
        client = MagicMock()
        with patch.object(config, "TRANSLATION_CHUNK_SIZE", 2):
            client.chat.completions.create.side_effect = [
                completion([
                    {"id": 0, "target_text": "第0句。天空是蓝色的，天气很好。", "self_check": {"semantic_complete": True}},
                    {"id": 0, "target_text": "重复第0句。", "self_check": {"semantic_complete": True}},
                    {"id": 2, "target_text": "未来批次。", "self_check": {"semantic_complete": True}},
                ]),
                completion([
                    {"id": 1, "target_text": "第1句。天空是蓝色的，天气很好。", "self_check": {"semantic_complete": True}},
                ]),
            ]
            errors: list[str] = []
            translated = translation.translate_script(script, client, errors=errors)
        self.assertEqual(translated, [0])
        self.assertEqual(
            [script.segments[i].target_text for i in range(3)],
            ["第0句。天空是蓝色的，天气很好。", "", ""],
        )
        joined = " | ".join(errors)
        self.assertIn("后续批次", joined)
        self.assertIn("重复", joined)

    def test_missing_id_reported_and_valid_partial_preserved(self):
        script = make_script(count=2)
        client = MagicMock()
        client.chat.completions.create.return_value = completion([
            {"id": 0, "target_text": "只译第0句。", "self_check": {"semantic_complete": True}},
        ])
        errors: list[str] = []
        self.assertEqual(translation.translate_script(script, client, errors=errors), [0])
        self.assertEqual(script.segments[0].target_text, "只译第0句。")
        self.assertEqual(script.segments[1].target_text, "")
        self.assertTrue(any("未在本批次返回" in message for message in errors))

    def test_repair_self_check_false_is_rejected(self):
        script = make_script(count=1)
        script.segments[0].orig_text = "Here's the basic logic."
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            completion([{
                "id": 0,
                "target_text": "思路",
                "emotion": "neutral",
                "action": "idle",
                "keep_original": False,
                "self_check": {"semantic_complete": True, "omitted_info": ""},
            }]),
            completion([{
                "id": 0,
                "target_text": "这是基本思路。",
                "self_check": {"semantic_complete": False, "omitted_info": "遗漏 here's"},
            }]),
        ]
        errors: list[str] = []
        self.assertEqual(translation.translate_script(script, client, errors=errors), [])
        self.assertEqual(script.segments[0].target_text, "")
        self.assertTrue(any("未通过自检" in message for message in errors))

    def test_non_speech_marker_with_spoken_text_is_not_event(self):
        self.assertFalse(translation._is_non_speech("[laughs] Yes"))

    def test_repair_whitelist_and_failed_repair_rejected(self):
        script = make_script(count=1)
        script.segments[0].orig_text = "Here's the basic logic."
        seg = script.segments[0]
        seg.emotion = "happy"
        seg.action = "point"
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            completion([{
                "id": 0,
                "target_text": "思路",
                "emotion": "neutral",
                "action": "idle",
                "keep_original": False,
                "self_check": {"semantic_complete": True, "omitted_info": ""},
            }]),
            completion([
                {"id": 0, "target_text": "点子", "emotion": "sad"},
                {"id": 99, "target_text": "越界修复", "emotion": "angry"},
            ]),
        ]
        errors: list[str] = []
        self.assertEqual(translation.translate_script(script, client, errors=errors), [])
        self.assertEqual(seg.target_text, "")
        self.assertEqual(seg.emotion, "happy")
        self.assertEqual(seg.action, "point")
        self.assertTrue(any("未通过自检" in message for message in errors))
        self.assertTrue(any("#99" in message for message in errors))


if __name__ == "__main__":
    unittest.main()
