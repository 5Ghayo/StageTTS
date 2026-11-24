import io
import json
import shutil
import tempfile
import unittest
import wave
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend import config
from backend.main import app
from backend.schemas import AlignedWord, CharacterConfig, EmotionRef, MediaInfo, ProjectScript, Segment
from backend.services import references, synthesis, translation, tts_client
from workers.aligned_segments import normalize_segments
from workers.parse_worker import _build_and_save_script


class DialogueWorkflowTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_long_turns_and_reference_roundtrip(self):
        words = [
            dict(word=f"word{i}{'.' if i % 6 == 5 else ''}", start=float(i), end=i + 0.7,
                 speaker="A" if i < 12 else "B")
            for i in range(24)
        ]
        transcript = " ".join(word["word"] for word in words)
        raw = dict(start=0.0, end=24.0, text=transcript, speaker="A", words=words)
        parts = normalize_segments([raw], diarized=True)
        self.assertEqual(" ".join(part["orig_text"] for part in parts), transcript)
        self.assertEqual([part["speaker"] for part in parts], ["A", "B"])
        self.assertEqual([(part["start"], part["end"]) for part in parts],
                         [(0, 11.7), (12, 24)])
        self.assertEqual([word for part in parts for word in part["words"]], words)
        self.assertFalse(any(part["alignment_needs_review"] for part in parts))
        unaligned, = normalize_segments([{**raw, "words": []}])
        self.assertTrue(unaligned["alignment_needs_review"])
        # No diarization evidence is not confirmation of a single speaker.
        self.assertTrue(unaligned["speaker_needs_review"])
        self.assertEqual(unaligned["orig_text"], transcript)

        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            project_dir = Path(temp) / "sample"
            project_dir.mkdir()
            vocals = project_dir / "vocals.wav"
            rate = 8000
            with wave.open(str(vocals), "wb") as audio:
                audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                audio.writeframes(b"".join((i + 1).to_bytes(2, "little") * rate for i in range(24)))
            args = Namespace(project_id="sample", input=vocals, output_dir=project_dir,
                             mode="video_dubbing", target_lang="zh", skip_llm=True)
            _build_and_save_script(args, 24, vocals, None, parts, "en")
            with TestClient(app) as client:
                project_url = "/api/projects/sample"
                script_url = f"{project_url}/script"
                data = client.get(script_url).json()
                self.assertEqual(data["source_lang"], "en")
                for item in data["segments"][:2]:
                    item["orig_text"] = f"Corrected line {item['id']}."
                edited = client.put(script_url, json=data)
                self.assertEqual(edited.status_code, 200)
                data = edited.json()
                self.assertTrue(all(item["alignment_needs_review"] for item in data["segments"][:2]))
                self.assertTrue(all(item["segment_timing_needs_review"] for item in data["segments"][:2]))
                self.assertEqual(client.get(script_url).json(), data)

                preview_url = f"{project_url}/characters/A/reference-preview"
                # Correcting text invalidates the old word transcript as crop evidence.
                # A short crop must not advertise the newly edited full sentence.
                self.assertEqual(client.post(preview_url).status_code, 400)
                for item, part in zip(data["segments"][:2], parts[:2]):
                    item["orig_text"] = part["orig_text"]
                restored = client.put(script_url, json=data)
                self.assertEqual(restored.status_code, 200)
                data = restored.json()
                # Restored text still needs a fresh explicit review of its words.
                for item in data["segments"][:2]:
                    item["alignment_needs_review"] = False
                confirmed = client.put(script_url, json=data)
                self.assertEqual(confirmed.status_code, 200)
                data = confirmed.json()
                preview = client.post(preview_url)
                self.assertEqual(preview.status_code, 200, preview.text)
                result = preview.json()
                source, = (item for item in data["segments"] if item["id"] in result["segment_ids"])
                self.assertEqual(result["lang"], "en")
                self.assertEqual(source["speaker"], "A")
                with wave.open(str(config.resolve_media_path(result["wav"])), "rb") as clip:
                    self.assertEqual(clip.getframerate(), 32000)
                    self.assertEqual(clip.getnchannels(), 1)
                    clip_seconds = clip.getnframes() / clip.getframerate()
                    self.assertGreaterEqual(clip_seconds, references.MIN_REFERENCE_SECONDS)
                    self.assertLessEqual(clip_seconds, references.MAX_REFERENCE_SECONDS)
                    window_start = source["start"]
                    window_end = window_start + clip_seconds
                    expected_words = [word["word"] for word in source["words"]
                                      if word["start"] >= window_start - 0.001
                                      and word["end"] <= window_end + 0.001]
                    self.assertTrue(expected_words)
                    self.assertEqual(result["text"], " ".join(expected_words))
                    self.assertNotEqual(result["text"], source["orig_text"])

                with config.resolve_media_path(result["wav"]).open("rb") as clip_file:
                    upload = client.post(f"{project_url}/references", files={"file": ("clip.wav", clip_file)})
                self.assertEqual(upload.status_code, 200, upload.text)
                data["characters"]["A"]["emotion_refs"]["neutral"] = {
                    "wav": upload.json()["wav"], "text": "A reviewed transcript.", "lang": "en",
                }
                for item in data["segments"][:2]:
                    item["speaker_needs_review"] = True
                self.assertEqual(client.put(script_url, json=data).status_code, 200)
                self.assertEqual(client.post(preview_url).status_code, 400)
                self.assertEqual(client.get(script_url).json(), data)

    def test_unknown_speaker_cannot_be_confirmed_by_put_or_agent_patch(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            project_dir = Path(temp) / "unknown-review"
            project_dir.mkdir()
            script = ProjectScript(
                project_id="unknown-review", media=MediaInfo(duration=5),
                characters={"UNKNOWN": CharacterConfig(display_name="待确认说话人"),
                            "A": CharacterConfig(display_name="角色 A")},
                segments=[Segment(id=0, start=0, end=4, speaker="UNKNOWN", orig_text="Hello.",
                                  speaker_fallback_reason="hf_token_missing")],
            )
            (project_dir / "script.json").write_text(script.model_dump_json(), encoding="utf-8")
            with TestClient(app) as client:
                url = "/api/projects/unknown-review/script"
                data = client.get(url).json()
                data["segments"][0]["speaker_needs_review"] = False
                response = client.put(url, json=data)
                self.assertEqual(response.status_code, 200, response.text)
                saved = client.get(url).json()
                self.assertTrue(saved["segments"][0]["speaker_needs_review"])
                self.assertEqual(saved["segments"][0]["speaker_fallback_reason"], "hf_token_missing")
                self.assertEqual(client.post("/api/projects/unknown-review/characters/UNKNOWN/reference-preview").status_code, 400)
                from backend.agent.tools import batch_update_segments
                batch_update_segments("unknown-review", [{"id": 0, "orig_text": "Hello there."}])
                saved = client.get(url).json()
                self.assertTrue(saved["segments"][0]["speaker_needs_review"])
                self.assertEqual(saved["segments"][0]["speaker_fallback_reason"], "hf_token_missing")
                batch_update_segments("unknown-review", [{"id": 0, "speaker": "A"}])
                saved = client.get(url).json()
                self.assertFalse(saved["segments"][0]["speaker_needs_review"])
                self.assertIsNone(saved["segments"][0]["speaker_fallback_reason"])

    def test_translation_and_rewrite_share_cross_batch_context(self):
        script = ProjectScript(
            project_id="terms", media=MediaInfo(duration=78), source_lang="en",
            glossary={"Beacon": "信标"},
            characters={"A": CharacterConfig(display_name="领队", speaking_style="简洁、沉着")},
            segments=[
                Segment(id=i, start=i * 3, end=i * 3 + 2, speaker="A", orig_text="Beacon is ready.",
                        target_text="旧稿", status="ready", tts_wav="old.wav", actual_duration=5)
                for i in range(26)
            ],
        )
        client = MagicMock()
        calls = []

        def reply(**kwargs):
            payload = json.loads(kwargs["messages"][-1]["content"])
            calls.append(payload)
            entries = [{"id": line["id"], "target_text": "信标已就绪"} for line in payload["lines"]]
            if len(calls) == 2:
                entries.append({"id": 24, "target_text": "越界返回，不能覆盖"})
            if len(calls) == 3:
                entries = [{"id": 25, "target_text": "信标就绪"}]
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(entries)))])

        client.chat.completions.create.side_effect = reply
        self.assertEqual(translation.translate_script(script, client), list(range(26)))
        self.assertEqual(script.segments[24].target_text, "信标已就绪")
        self.assertTrue(all(seg.status == "pending" and seg.tts_wav is None for seg in script.segments))
        self.assertEqual(calls[0]["lines"][-1]["after"][0]["id"], 25)
        self.assertEqual(calls[1]["lines"][0]["before"][-1]["target_text"], "信标已就绪")
        seg = script.segments[-1]
        seg.actual_duration = 3.5
        check = synthesis._check_duration(seg)
        self.assertEqual(synthesis._rewrite_overflow(client, script, {seg.id: seg for seg in script.segments}, [check]),
                         {25: "信标就绪"})
        self.assertEqual([payload["glossary"] for payload in calls], [script.glossary] * 3)
        self.assertEqual(calls[2]["lines"][0]["speaking_style"], "简洁、沉着")
        self.assertLess(calls[2]["lines"][0]["max_units"], calls[1]["lines"][0]["max_units"])

    def test_role_reference_and_optional_weights_do_not_bleed(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            root = Path(temp)
            ref_paths = {speaker: root / f"{speaker}.wav" for speaker in "ABC"}
            for path in ref_paths.values():
                with wave.open(str(path), "wb") as audio:
                    audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
                    audio.writeframes(bytes(4 * 8000 * 2))
            script = ProjectScript(
                project_id="voices", media=MediaInfo(duration=15),
                characters={
                    speaker: CharacterConfig(
                        display_name=speaker, emotion_refs={"neutral": EmotionRef(
                            wav=str(ref_paths[speaker]), text=f"Script {speaker}", lang="en"
                        )},
                    ) for speaker in "ABC"
                },
                segments=[
                    Segment(id=i, start=i * 5, end=i * 5 + 4, speaker=speaker, target_text=f"Line {speaker}")
                    for i, speaker in enumerate("ABC")
                ],
            )
            script.characters["A"].gpt_weights = "voice-gpt"
            script.characters["A"].sovits_weights = "voice-sovits"
            script.characters["B"].gpt_weights = "second-gpt"
            defaults = {"gpt": "base-gpt", "sovits": "base-sovits"}
            active = defaults.copy()
            heard = []
            session = MagicMock()

            def switch(url, *, params, **_):
                kind = url.rsplit("/set_", 1)[-1].removesuffix("_weights")
                active[kind] = params["weights_path"]
                return SimpleNamespace(status_code=200)

            def speak(_url, *, json, **_):
                heard.append((json["prompt_text"], Path(json["ref_audio_path"]).name, active.copy()))
                return SimpleNamespace(status_code=200, content=ref_paths["A"].read_bytes())

            session.get.side_effect = switch
            session.post.side_effect = speak
            self.addCleanup(tts_client.reset_active_weights_cache)
            with (
                patch.object(tts_client, "ensure_tts_running"),
                patch.object(tts_client, "_default_weights", return_value=defaults),
                patch.object(tts_client.httpx, "Client") as factory,
            ):
                factory.return_value.__enter__.return_value = session
                self.assertEqual(tts_client.synthesize_segments_grouped(script), ([0, 1, 2], []))
                self.assertEqual(tts_client.synthesize_segments_grouped(script, [0]), ([0], []))
            self.assertEqual(heard[:3], [
                ("Script A", "A.wav", {"gpt": "voice-gpt", "sovits": "voice-sovits"}),
                ("Script B", "B.wav", {"gpt": "second-gpt", "sovits": "base-sovits"}),
                ("Script C", "C.wav", defaults),
            ])
            self.assertEqual(active, defaults)

    def test_retranslate_surfaces_llm_errors_and_handles_fenced_json(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            script = ProjectScript(
                project_id="retrans",
                media=MediaInfo(duration=80),
                segments=[
                    Segment(id=i, start=i * 2, end=i * 2 + 1.5, speaker="A", orig_text=f"Line {i}.")
                    for i in range(26)
                ],
            )
            (Path(temp) / "retrans").mkdir()
            (Path(temp) / "retrans" / "script.json").write_text(script.model_dump_json(), encoding="utf-8")

            with TestClient(app) as client, patch("backend.main.create_openai_client") as factory:
                factory.return_value.chat.completions.create.side_effect = ValueError("gateway down")
                failed = client.post("/api/projects/retrans/translate")
                self.assertEqual(failed.status_code, 502)
                self.assertIn("gateway down", failed.json()["detail"])

                factory.return_value.chat.completions.create.side_effect = [
                    SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                        content="说明 [hint]\n```json\n" + json.dumps([
                            {"id": i, "target_text": f"台词 {i}"} for i in range(25)
                        ]) + "\n```"
                    ))]),
                    ValueError("second batch failed"),
                ]
                partial = client.post("/api/projects/retrans/translate")
                self.assertEqual(partial.status_code, 200, partial.text)
                payload = partial.json()
                self.assertEqual(payload["translated_ids"], list(range(25)))
                self.assertEqual(payload["skipped_count"], 1)
                self.assertEqual(len(payload["errors"]), 1)

    def test_reference_store_enforces_size_limit_when_stream_size_unknown(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            with patch.object(references, "MAX_REFERENCE_BYTES", 16):
                with self.assertRaisesRegex(ValueError, "32 MB"):
                    references.store_reference_audio("oversized", io.BytesIO(b"0" * 32))

    def test_en_to_zh_translation_avoids_false_repair_and_repair_preserves_tags(self):
        normal_script = ProjectScript(
            project_id="en_zh_ok",
            source_lang="en",
            glossary={"AI": "人工智能"},
            media=MediaInfo(duration=12),
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=4.0,
                    speaker="SPEAKER_00",
                    orig_text="Wait, are you sure this local video dubbing tool works without cloud servers?",
                ),
            ],
        )
        ok_client = MagicMock()
        ok_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps([
                            {
                                "id": 0,
                                "target_text": "等等，你确定这个本地视频配音工具不需要云服务器就能运行吗？",
                                "emotion": "angry",
                                "action": "shake_head",
                                "keep_original": False,
                                "self_check": {"semantic_complete": True, "omitted_info": ""},
                            }
                        ])
                    )
                )
            ]
        )
        self.assertEqual(translation.translate_script(normal_script, ok_client), [0])
        self.assertEqual(ok_client.chat.completions.create.call_count, 1)
        seg0 = normal_script.segments[0]
        self.assertEqual(seg0.emotion, "angry")
        self.assertEqual(seg0.action, "shake_head")
        self.assertFalse(seg0.keep_original)

        repair_script = ProjectScript(
            project_id="en_zh_repair",
            source_lang="en",
            glossary={"Beacon": "信标"},
            media=MediaInfo(duration=20),
            segments=[
                Segment(
                    id=0,
                    start=0.0,
                    end=4.0,
                    speaker="SPEAKER_00",
                    orig_text="Check the Beacon status before launching the mission.",
                ),
                Segment(
                    id=1,
                    start=4.2,
                    end=7.0,
                    speaker="SPEAKER_00",
                    orig_text="Please check every single sensor on deck right now.",
                ),
                Segment(
                    id=2,
                    start=7.2,
                    end=12.0,
                    speaker="SPEAKER_00",
                    orig_text="We need to verify every single audio track carefully before exporting the final film.",
                ),
                Segment(
                    id=3,
                    start=12.2,
                    end=13.5,
                    speaker="SPEAKER_00",
                    orig_text="Oh...",
                ),
            ],
        )
        repair_client = MagicMock()
        repair_client.chat.completions.create.side_effect = [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps([
                                {
                                    "id": 0,
                                    "target_text": "在发射任务前先检查一下设备状态。",
                                    "emotion": "angry",
                                    "action": "point",
                                    "keep_original": False,
                                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                                },
                                {
                                    "id": 1,
                                    "target_text": "（叹气）检查甲板。",
                                    "emotion": "sad",
                                    "action": "sigh",
                                    "keep_original": True,
                                    "self_check": {"semantic_complete": False, "omitted_info": "sensor"},
                                },
                                {
                                    "id": 2,
                                    "target_text": "导出前",
                                    "emotion": "whisper",
                                    "action": "nod",
                                    "keep_original": False,
                                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                                },
                                {
                                    "id": 3,
                                    "target_text": "（停顿）",
                                    "emotion": "sad",
                                    "action": "sigh",
                                    "keep_original": True,
                                    "self_check": {"semantic_complete": False, "omitted_info": "tone"},
                                },
                            ])
                        )
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps([
                                {"id": 0, "target_text": "在启动任务前先检查信标状态。"},
                                {"id": 1, "target_text": "请现在就检查甲板上的每一个传感器。"},
                                {"id": 2, "target_text": "导出成片前我们需要仔细核对每一条音轨。"},
                                {"id": 3, "target_text": "哦……"},
                            ])
                        )
                    )
                ]
            ),
        ]
        self.assertEqual(translation.translate_script(repair_script, repair_client), [0, 1, 2, 3])
        self.assertEqual(repair_client.chat.completions.create.call_count, 2)
        s0, s1, s2, s3 = repair_script.segments
        self.assertEqual((s0.emotion, s0.action, s0.keep_original), ("angry", "point", False))
        # 上游 _is_non_speech 守卫会过滤长整句误标的 keep_original=True，仅短叹词保留原声
        self.assertEqual((s1.emotion, s1.action, s1.keep_original), ("sad", "sigh", False))
        self.assertEqual((s2.emotion, s2.action, s2.keep_original), ("whisper", "nod", False))
        self.assertEqual((s3.emotion, s3.action, s3.keep_original), ("sad", "sigh", True))

    def test_long_script_translation_uses_sliding_dialogue_window(self):
        script = ProjectScript(
            project_id="long_win",
            source_lang="en",
            media=MediaInfo(duration=200),
            segments=[
                Segment(
                    id=i,
                    start=i * 2.0,
                    end=i * 2.0 + 1.5,
                    speaker="SPEAKER_00",
                    orig_text=f"Sentence number {i} is ready now.",
                )
                for i in range(65)
            ],
        )
        client = MagicMock()
        seen_payloads = []

        def reply(**kwargs):
            payload = json.loads(kwargs["messages"][-1]["content"])
            seen_payloads.append(payload)
            entries = [
                {
                    "id": line["id"],
                    "target_text": f"第{line['id']}句已经准备好了。",
                    "emotion": "neutral",
                    "action": "idle",
                    "keep_original": False,
                    "self_check": {"semantic_complete": True, "omitted_info": ""},
                }
                for line in payload["lines"]
            ]
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(entries)))]
            )

        client.chat.completions.create.side_effect = reply
        with patch.object(translation, "WHOLE_DIALOGUE_MAX_SEGMENTS", 30):
            translated_ids = translation.translate_script(script, client)

        self.assertEqual(translated_ids, list(range(65)))
        self.assertEqual(len(seen_payloads), 3)
        for idx, payload in enumerate(seen_payloads):
            self.assertEqual(payload["translation_mode"], "windowed_dialogue")
            self.assertIn("dialogue_window", payload)
            dialogue_lines = payload["dialogue"].splitlines()
            self.assertLess(len(dialogue_lines), 65)
            if idx == 1:
                self.assertIn("[24] SPEAKER_00:", payload["dialogue"])
                self.assertIn("[50] SPEAKER_00:", payload["dialogue"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_automatic_reference_fallback_trims_transcript_to_window(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            project_dir = Path(temp) / "trim_ref"
            project_dir.mkdir()
            vocals = project_dir / "vocals.wav"
            rate = 8000
            with wave.open(str(vocals), "wb") as audio:
                audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                audio.writeframes(bytes(13 * rate * 2))

            en_words = [
                AlignedWord(word=token, start=idx * 1.2, end=idx * 1.2 + 1.1, speaker="A")
                for idx, token in enumerate(
                    ["Alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
                )
            ]
            en_text = " ".join(w.word for w in en_words)
            en_script = ProjectScript(
                project_id="trim_ref",
                source_lang="en",
                media=MediaInfo(duration=13.0, vocals_wav="/workspace/trim_ref/vocals.wav"),
                segments=[
                    Segment(
                        id=0,
                        start=0.0,
                        end=12.0,
                        speaker="A",
                        orig_text=en_text,
                        words=en_words,
                        alignment_needs_review=True,
                        speaker_needs_review=False,
                    )
                ],
            )
            en_preview = references.automatic_reference(en_script, "A")
            self.assertIsNotNone(en_preview)
            assert en_preview is not None
            self.assertEqual(en_preview.text, "Alpha bravo charlie delta")
            self.assertNotIn("foxtrot", en_preview.text)
            with wave.open(str(config.resolve_media_path(en_preview.wav)), "rb") as clip:
                self.assertEqual(clip.getframerate(), 32000)
                self.assertEqual(clip.getnframes(), round(4.7 * 32000))

            cjk_tokens = ["今天", "天气", "真的", "非常", "不错", "我们", "一起", "出发", "去海边", "玩吧"]
            cjk_words = [
                AlignedWord(word=tok, start=idx * 1.2, end=idx * 1.2 + 1.1, speaker="A")
                for idx, tok in enumerate(cjk_tokens)
            ]
            cjk_script = ProjectScript(
                project_id="trim_ref",
                source_lang="zh",
                media=MediaInfo(duration=13.0, vocals_wav="/workspace/trim_ref/vocals.wav"),
                segments=[
                    Segment(
                        id=0,
                        start=0.0,
                        end=12.0,
                        speaker="A",
                        orig_text="".join(cjk_tokens),
                        words=cjk_words,
                        alignment_needs_review=True,
                        speaker_needs_review=False,
                    )
                ],
            )
            cjk_preview = references.automatic_reference(cjk_script, "A")
            self.assertIsNotNone(cjk_preview)
            assert cjk_preview is not None
            self.assertEqual(cjk_preview.text, "今天天气真的非常")
            self.assertNotIn(" ", cjk_preview.text)

    def test_suspicious_check_handles_non_ascii_latin_and_cyrillic(self):
        fr_seg = Segment(
            id=0,
            start=0.0,
            end=5.0,
            speaker="A",
            orig_text="今天天气真好，我们一起去河边散步吧，顺便买杯咖啡。",
            target_lang="fr",
        )
        self.assertFalse(
            translation._is_suspicious(
                fr_seg,
                "Allons nous promener près de la rivière et prenons un café.",
            )
        )
        ru_seg = Segment(
            id=1,
            start=0.0,
            end=5.0,
            speaker="A",
            orig_text="今天天气真好，我们一起去河边散步吧，顺便买杯咖啡。",
            target_lang="ru",
        )
        self.assertFalse(
            translation._is_suspicious(
                ru_seg,
                "Погода сегодня отличная, пойдём гулять к реке и купим кофе.",
            )
        )

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_automatic_reference_fallback_respects_overlap_and_scans_subsequent_segments(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            project_dir = Path(temp) / "overlap_ref"
            project_dir.mkdir()
            vocals = project_dir / "vocals.wav"
            rate = 8000
            with wave.open(str(vocals), "wb") as audio:
                audio.setparams((1, 2, rate, 0, "NONE", "not compressed"))
                audio.writeframes(bytes(20 * rate * 2))

            words0 = [
                AlignedWord(word="Hello", start=0.0, end=2.0, speaker="A"),
                AlignedWord(word="world", start=2.0, end=4.5, speaker="A"),
            ]
            script = ProjectScript(
                project_id="overlap_ref",
                source_lang="en",
                media=MediaInfo(duration=20.0, vocals_wav="/workspace/overlap_ref/vocals.wav"),
                segments=[
                    Segment(
                        id=0,
                        start=0.0,
                        end=4.5,
                        speaker="A",
                        orig_text="Hello world",
                        words=words0,
                        alignment_needs_review=True,
                        speaker_needs_review=False,
                    ),
                    Segment(
                        id=1,
                        start=1.0,
                        end=3.5,
                        speaker="B",
                        orig_text="Interrupt",
                        alignment_needs_review=True,
                        speaker_needs_review=False,
                    ),
                    Segment(
                        id=2,
                        start=10.0,
                        end=14.2,
                        speaker="A",
                        orig_text="This later line is completely clear for reference.",
                        alignment_needs_review=True,
                        speaker_needs_review=False,
                    ),
                ],
            )
            preview = references.automatic_reference(script, "A")
            self.assertIsNotNone(preview)
            assert preview is not None
            self.assertEqual(preview.segment_ids, [2])
            self.assertIn("completely clear", preview.text)
