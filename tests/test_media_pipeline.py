import json
import shutil
import tempfile
import unittest
import wave
from array import array
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend import config
from backend.agent import orchestrator
from backend.agent.tools import save_project_script
from backend.main import app
from backend.schemas import MediaInfo, ProjectScript, Segment
from backend.services import media_mixer, synthesis, task_runner, tts_client
from workers import demucs_stage, parse_worker


def write_wav(path: Path, seconds: float, rate: int = 8000, amplitude: int = 1000) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(array("h", [amplitude] * int(seconds * rate)).tobytes())


def completion(content=None, tool_calls=()):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))])


@unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
class SynthesisTest(unittest.TestCase):
    def setUp(self):
        context = ExitStack()
        self.addCleanup(context.close)
        workspace = context.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(workspace)
        context.enter_context(patch.object(config, "WORKSPACE_DIR", self.root))
        context.enter_context(patch.object(tts_client, "ensure_tts_running"))
        context.enter_context(patch.object(tts_client, "_resolve_ref_for_segment", return_value=(None, None)))

        durations = {"long": 1.5, "short": 0.85, "stubborn": 1.8, "retry": 1.6, "shorter": 1.45}

        def fake_tts(_session, segment, _ref, _ref_path, output):
            write_wav(output, durations[segment.target_text])

        context.enter_context(patch.object(tts_client, "_synthesize_single_line", side_effect=fake_tts))
        self.client = context.enter_context(TestClient(app))

    def _create_project(self, project_id):
        save_project_script(ProjectScript(
            project_id=project_id, mode="virtual_stage", media=MediaInfo(duration=5),
            segments=[
                Segment(id=1, start=0, end=1, speaker="A", target_text="long"),
                Segment(id=2, start=2, end=3, speaker="A", target_text="stubborn"),
            ],
        ))

    def test_api_and_agent_share_reflection_and_final_wav(self):
        with patch.object(synthesis, "create_openai_client") as llm:
            rewrite = llm.return_value.chat.completions.create
            rewrite.side_effect = [
                completion('[{"id":1,"target_text":"short"},{"id":2,"target_text":"retry"}]'),
                completion('[{"id":2,"target_text":"shorter"}]'),
            ] * 2
            self._create_project("batch")
            response = self.client.post("/api/projects/batch/synthesize", json={"segment_ids": [1, 2]})
            self.assertEqual(response.status_code, 200, response.text)
            batch = response.json()
            self.assertEqual([item["segment_id"] for item in batch["overflow_segments"]], [2])
            self.assertEqual(batch["errors"], [])

            self._create_project("agent")
            call = SimpleNamespace(id="tool1", function=SimpleNamespace(
                name="synthesize_segments", arguments=json.dumps({"segment_ids": [1, 2]}),
            ))
            with patch.object(orchestrator, "create_openai_client") as agent_llm:
                agent_llm.return_value.chat.completions.create.side_effect = [
                    completion(tool_calls=[call]), completion("配音完成"),
                ]
                response = self.client.post("/api/projects/agent/agent/chat", json={"message": "为两句配音"})
            self.assertEqual(response.status_code, 200, response.text)
            agent = response.json()
            self.assertIn("#2", agent["reply"])
            self.assertEqual(rewrite.call_count, 4)

        for result in (batch, agent):
            self.assertEqual(result["synthesized_ids"], [1, 2])
            short, stubborn = result["script"]["segments"]
            self.assertEqual(short["target_text"], "short")
            self.assertEqual(stubborn["target_text"], "shorter")
            self.assertEqual(stubborn["applied_speed_factor"], 1.15)
            self.assertLess(stubborn["actual_duration"], 1.45)
            self.assertGreater(stubborn["actual_duration"], 1)
            wav = config.resolve_media_path(stubborn["tts_wav"])
            self.assertEqual(tts_client.measure_wav_duration(wav), stubborn["actual_duration"])
            project_id = result["script"]["project_id"]
            with patch.object(media_mixer, "apply_gentle_tempo", side_effect=AssertionError("second tempo pass")):
                for _ in range(2):
                    response = self.client.post(f"/api/projects/{project_id}/render", json={})
                    self.assertEqual(response.status_code, 200, response.text)

    def test_missing_llm_and_failed_rewrite_keep_usable_audio(self):
        self._create_project("test")
        with patch.object(synthesis, "create_openai_client", side_effect=synthesis.LLMNotConfiguredError):
            first = self.client.post("/api/projects/test/synthesize", json={"segment_ids": [1]}).json()
        self.assertIn("未配置 LLM", first["errors"][0])
        self.assertEqual(first["overflow_segments"][0]["segment_id"], 1)
        with patch.object(synthesis, "create_openai_client") as llm:
            llm.return_value.chat.completions.create.return_value = completion('[{"id":1,"target_text":"invalid"}]')
            second = self.client.post("/api/projects/test/synthesize", json={"segment_ids": [1]}).json()
        self.assertEqual(second["script"]["segments"][0], first["script"]["segments"][0])
        self.assertTrue(any("#1" in error for error in second["errors"]))

    def test_legacy_wav_still_uses_export_tempo(self):
        self._create_project("legacy")
        script = self.client.get("/api/projects/legacy/script").json()
        project_dir = self.root / "legacy"
        write_wav(project_dir / "legacy.wav", 1.4)
        seg = script["segments"][0]
        seg.pop("applied_speed_factor")
        seg.update(tts_wav="/workspace/legacy/legacy.wav", speed_factor=1.15, status="ready")
        response = self.client.put("/api/projects/legacy/script", json=script)
        self.assertEqual(response.status_code, 200, response.text)
        with patch.object(media_mixer, "apply_gentle_tempo", wraps=media_mixer.apply_gentle_tempo) as tempo:
            response = self.client.post("/api/projects/legacy/render", json={})
        self.assertEqual(response.status_code, 200, response.text)
        tempo.assert_called_once()
        self.assertEqual(tempo.call_args.args[1], 1.15)


class ParseTest(unittest.TestCase):
    def test_managed_process_exits_and_external_service_blocks_parse(self):
        managed = MagicMock()
        managed.poll.return_value = None
        with (
            patch.object(config, "TTS_AUTO_MANAGE", True),
            patch.object(config, "TTS_API_URL", "http://127.0.0.1:9880"),
            patch.object(task_runner, "_tts_proc", managed),
            patch.object(task_runner.time, "sleep"),
            patch.object(task_runner.socket, "create_connection", side_effect=OSError),
        ):
            task_runner._prepare_gpu_for_parse()
            managed.terminate.assert_called_once()
            managed.wait.assert_called_once()
            self.assertIsNone(task_runner._tts_proc)
        with (
            patch.object(config, "TTS_AUTO_MANAGE", False),
            patch.object(config, "TTS_API_URL", "http://127.0.0.1:9880"),
            patch.object(task_runner.socket, "create_connection"),
            patch.object(task_runner, "ensure_tts_stopped") as stop,
        ):
            with self.assertRaisesRegex(task_runner.GpuBusyError, "9880"):
                task_runner._prepare_gpu_for_parse()
            stop.assert_not_called()

    def test_long_cuda_retry_preserves_samples_and_cpu_requires_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "input.wav"
            write_wav(source, 575, rate=800)
            messages, devices = [], []

            def fake_demucs(chunk, output, device, _timeout):
                devices.append(device)
                with wave.open(str(chunk), "rb") as wav:
                    seconds = wav.getnframes() / wav.getframerate()
                if seconds > 31:
                    raise demucs_stage.DemucsError("CUDA out of memory")
                stems = output / "htdemucs" / chunk.stem
                stems.mkdir(parents=True)
                amplitude = 1000 if len(devices) % 2 else 3000
                for name in ("vocals.wav", "no_vocals.wav"):
                    write_wav(stems / name, seconds, rate=800, amplitude=amplitude)

            with (
                patch.object(demucs_stage, "_sample_gpu_memory"),
                patch.object(demucs_stage, "_run_demucs", side_effect=fake_demucs),
            ):
                stems = demucs_stage.separate_stems(source, root, "cuda", False, messages.append)
            self.assertTrue(any("30s 分块重试" in message for message in messages))
            self.assertEqual(set(devices), {"cuda"})
            self.assertIsNotNone(stems)
            for path in stems:
                with wave.open(str(path), "rb") as wav:
                    self.assertEqual(wav.getnframes(), 575 * 800)
                    wav.setpos(30 * 800 - 1)
                    samples = array("h", wav.readframes(802))
                    self.assertLess(max(abs(after - before) for before, after in zip(samples, samples[1:])), 10)

            with (
                patch.object(demucs_stage, "_sample_gpu_memory"),
                patch.object(demucs_stage, "_run_demucs", side_effect=demucs_stage.DemucsError("CUDA out of memory")) as run,
            ):
                vocals, bgm = parse_worker._stage_demucs(source, root, "cuda", False, False)
                self.assertIsNone(bgm)
                self.assertEqual(vocals.read_bytes(), source.read_bytes())
                self.assertEqual([call.args[2] for call in run.call_args_list], ["cuda", "cuda"])
                run.reset_mock()
                parse_worker._stage_demucs(source, root, "cuda", False, True)
                self.assertEqual([call.args[2] for call in run.call_args_list], ["cuda", "cuda", "cpu"])
