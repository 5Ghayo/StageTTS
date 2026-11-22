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
from backend.schemas import CharacterConfig, EmotionRef, MediaInfo, ProjectScript, Segment
from backend.services import synthesis, translation, tts_client
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
        self.assertEqual([part["speaker"] for part in parts], ["A", "A", "B", "B"])
        self.assertEqual([(part["start"], part["end"]) for part in parts],
                         [(0, 5.7), (6, 11.7), (12, 17.7), (18, 24)])
        self.assertEqual([word for part in parts for word in part["words"]], words)
        self.assertFalse(any(part["alignment_needs_review"] for part in parts))
        unaligned, = normalize_segments([{**raw, "words": []}])
        self.assertTrue(unaligned["alignment_needs_review"])
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
                self.assertEqual(client.put(script_url, json=data).status_code, 200)
                self.assertEqual(client.get(script_url).json(), data)

                preview_url = f"{project_url}/characters/A/reference-preview"
                preview = client.post(preview_url)
                self.assertEqual(preview.status_code, 200, preview.text)
                result = preview.json()
                source, = (item for item in data["segments"] if item["id"] in result["segment_ids"])
                self.assertEqual(result["text"], source["orig_text"])
                self.assertEqual(result["lang"], "en")
                self.assertEqual(source["speaker"], "A")
                with wave.open(str(config.resolve_media_path(result["wav"])), "rb") as clip:
                    self.assertEqual(clip.getnframes(), round((source["end"] - source["start"]) * rate))
                    self.assertEqual(int.from_bytes(clip.readframes(1), "little"), source["start"] + 1)

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
