import io
import json
import os
import shutil
import subprocess
import sys
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
from backend.schemas import CinematicStems, MediaInfo, ProjectScript, Segment
from backend.services import media_mixer, references, synthesis, task_runner, tts_client
from workers import parse_worker


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

        def fake_tts(_session, segment, _ref, _ref_path, output, _glossary=None):
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
            # The overflow is allowed into the empty scene gap, not silently cut
            # at the original 3.0-second slot end during the API export route.
            with wave.open(str(self.root / project_id / "dubbed_vocals.wav"), "rb") as rendered:
                rendered.setpos(round(3.05 * rendered.getframerate()))
                tail = array("h", rendered.readframes(round(0.05 * rendered.getframerate())))
                self.assertTrue(tail)
                self.assertGreater(max(abs(sample) for sample in tail), 0)

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

    def test_legacy_error_segment_is_not_mixed_into_render(self):
        self._create_project("legacy_err")
        script = self.client.get("/api/projects/legacy_err/script").json()
        project_dir = self.root / "legacy_err"
        write_wav(project_dir / "stale.wav", 1.0, amplitude=2000)
        seg = script["segments"][0]
        seg.pop("applied_speed_factor")
        seg.update(tts_wav="/workspace/legacy_err/stale.wav", status="error")
        self.assertEqual(self.client.put("/api/projects/legacy_err/script", json=script).status_code, 200)
        response = self.client.post("/api/projects/legacy_err/render", json={})
        self.assertEqual(response.status_code, 200, response.text)
        with wave.open(str(project_dir / "dubbed_vocals.wav"), "rb") as out:
            frames = array("h", out.readframes(out.getnframes()))
            self.assertEqual(max(abs(sample) for sample in frames), 0)

    def test_tts_unavailable_returns_503_and_agent_does_not_repeat_failed_synthesis(self):
        self._create_project("down")
        with patch.object(
            tts_client, "_synthesize_single_line", side_effect=tts_client.TTSServiceUnavailableError("tts down"),
        ) as synth_call:
            resp = self.client.post("/api/projects/down/synthesize", json={"segment_ids": [1, 2]})
            self.assertEqual(resp.status_code, 503)
            self.assertEqual(synth_call.call_count, 1)

        with patch.object(
            tts_client, "_switch_weights", side_effect=tts_client.TTSServiceUnavailableError("weights down"),
        ):
            resp = self.client.post("/api/projects/down/synthesize", json={"segment_ids": [1, 2]})
            self.assertEqual(resp.status_code, 503)

        self._create_project("agent_fail")
        update_call = SimpleNamespace(id="t1", function=SimpleNamespace(
            name="batch_update_segments",
            arguments=json.dumps({"updates": [{"id": 1, "target_text": ""}]}),
        ))
        synth_call = SimpleNamespace(id="t2", function=SimpleNamespace(
            name="synthesize_segments",
            arguments=json.dumps({"segment_ids": [1]}),
        ))
        with (
            patch.object(orchestrator, "create_openai_client") as agent_llm,
            patch.object(orchestrator, "synthesize_segments", wraps=orchestrator.synthesize_segments) as wrapped_synth,
        ):
            agent_llm.return_value.chat.completions.create.side_effect = [
                completion(tool_calls=[update_call]),
                completion(tool_calls=[synth_call]),
                completion("合成失败，台词为空"),
            ]
            resp = self.client.post("/api/projects/agent_fail/agent/chat", json={"message": "清空第1句并合成"})
            self.assertEqual(resp.status_code, 200, resp.text)
            self.assertEqual(wrapped_synth.call_count, 1)


class ParseTest(unittest.TestCase):
    def test_probe_duration_parses_ffprobe_json(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "sample.wav"
            write_wav(media, 1.0, rate=800)
            proc = SimpleNamespace(stdout='{"format": {"duration": "12.3456"}}')
            with (
                patch.object(parse_worker.shutil, "which", return_value="/usr/bin/ffprobe"),
                patch.object(parse_worker.subprocess, "run", return_value=proc),
            ):
                self.assertEqual(parse_worker._probe_duration(media), 12.346)

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

    def test_banquet_retries_cuda_oom_on_cpu_and_streams_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bandit_dir = root / "src" / "models" / "bandit"
            bandit_dir.mkdir(parents=True)
            (bandit_dir / "bandit.py").touch()
            (bandit_dir / "film.py").touch()
            checkpoint = root / "banquet-split.ckpt"
            checkpoint.touch()

            def fake_proc(output: str, returncode: int):
                proc = MagicMock()
                proc.stdout = io.StringIO(output)
                proc.wait.return_value = returncode
                return proc

            cuda = fake_proc("[banquet] 加载 Banquet 模型 (cuda)...\nCUDA out of memory\n", 1)
            cpu = fake_proc("[banquet] 加载 Banquet 模型 (cpu)...\n[banquet] 分离进度 1/2 窗口 (50%)\n", 0)
            logged: list[tuple[str, str]] = []
            with (
                patch.dict(os.environ, BANQUET_ROOT=str(root), BANQUET_CHECKPOINT=str(checkpoint)),
                patch.object(parse_worker.subprocess, "Popen", side_effect=[cuda, cpu]) as popen,
                patch.object(parse_worker, "_log", side_effect=lambda stage, msg: logged.append((stage, msg))),
            ):
                parse_worker._stage_banquet(root / "input.wav", root, "cuda")
            self.assertEqual([call.args[0][-1] for call in popen.call_args_list], ["cuda", "cpu"])
            self.assertTrue(any(stage == "Stage1" and "分离进度 1/2 窗口 (50%)" in msg for stage, msg in logged))
            self.assertTrue(cuda.stdout.closed and cpu.stdout.closed)

            non_oom = fake_proc("", -9)
            with (
                patch.dict(os.environ, BANQUET_ROOT=str(root), BANQUET_CHECKPOINT=str(checkpoint)),
                patch.object(parse_worker.subprocess, "Popen", return_value=non_oom) as popen_non_oom,
            ):
                with self.assertRaisesRegex(RuntimeError, r"code=-9"):
                    parse_worker._stage_banquet(root / "input.wav", root, "cuda")
            self.assertEqual(popen_non_oom.call_count, 1)
            self.assertTrue(non_oom.stdout.closed)

            timed_out = MagicMock()
            timed_out.stdout = io.StringIO("")
            timed_out.wait.side_effect = [subprocess.TimeoutExpired("banquet", 900), -9]
            with (
                patch.dict(os.environ, BANQUET_ROOT=str(root), BANQUET_CHECKPOINT=str(checkpoint)),
                patch.object(parse_worker.subprocess, "Popen", return_value=timed_out),
            ):
                with self.assertRaisesRegex(RuntimeError, "分离超时"):
                    parse_worker._stage_banquet(root / "input.wav", root, "cpu")
            timed_out.kill.assert_called_once()
            self.assertTrue(timed_out.stdout.closed)

            interrupted = MagicMock()
            interrupted.stdout = io.StringIO("")
            interrupted.poll.return_value = None
            interrupted.wait.side_effect = [KeyboardInterrupt, -9]
            with (
                patch.dict(os.environ, BANQUET_ROOT=str(root), BANQUET_CHECKPOINT=str(checkpoint)),
                patch.object(parse_worker.subprocess, "Popen", return_value=interrupted),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    parse_worker._stage_banquet(root / "input.wav", root, "cpu")
            interrupted.kill.assert_called_once()
            self.assertTrue(interrupted.stdout.closed)

    def test_banquet_readiness_and_fail_fast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bandit_dir = root / "src" / "models" / "bandit"
            bandit_dir.mkdir(parents=True)
            (bandit_dir / "bandit.py").touch()
            checkpoint = root / "banquet-split.ckpt"
            checkpoint.touch()
            input_media = root / "input.mp4"
            input_media.write_bytes(b"fake")

            with patch.dict(os.environ, {"BANQUET_ROOT": "", "BANQUET_CHECKPOINT": ""}):
                with (
                    patch.object(config, "BANQUET_ROOT", None),
                    patch.object(config, "BANQUET_CHECKPOINT", None),
                ):
                    self.assertFalse(config.is_banquet_configured())
                    with TestClient(app) as client:
                        health = client.get("/api/health")
                        self.assertEqual(health.status_code, 200)
                        self.assertFalse(health.json()["readiness"]["banquet_configured"])
                    with (
                        patch.object(task_runner.shutil, "which", return_value="/usr/bin/python"),
                        patch.object(task_runner, "_prepare_gpu_for_parse") as prepare_gpu,
                    ):
                        with self.assertRaisesRegex(task_runner.ParseWorkerError, "BANQUET_ROOT"):
                            task_runner.run_parse_worker("demo", input_media)
                        prepare_gpu.assert_not_called()

            # Missing film.py should still fail validation.
            with patch.dict(os.environ, {"BANQUET_ROOT": str(root), "BANQUET_CHECKPOINT": str(checkpoint)}):
                self.assertFalse(config.is_banquet_configured())
                (bandit_dir / "film.py").touch()
                self.assertTrue(config.is_banquet_configured())
                with TestClient(app) as client:
                    health = client.get("/api/health")
                    self.assertEqual(health.status_code, 200)
                    self.assertTrue(health.json()["readiness"]["banquet_configured"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_finish_four_stems_mixes_bgm_and_validates_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            project_dir = Path(directory) / "demo"
            cinematic_dir = project_dir / "cinematic-abc"
            cinematic_dir.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "未生成完整"):
                parse_worker._finish_four_stems(cinematic_dir, "demo")
            for stem in ("dialogue", "singing", "music", "effects"):
                write_wav(cinematic_dir / f"{stem}.wav", 0.5, rate=48000, amplitude=25000)
            dialogue, bgm, stems = parse_worker._finish_four_stems(cinematic_dir, "demo")
            self.assertEqual(dialogue, cinematic_dir / "dialogue.wav")
            self.assertTrue(bgm.is_file() and bgm.stat().st_size > 44)
            with wave.open(str(bgm), "rb") as mixed:
                self.assertEqual(mixed.getsampwidth(), 2)
                self.assertEqual(mixed.getframerate(), 48000)
                self.assertEqual(mixed.getnframes(), 24000)
            self.assertEqual(stems.dialogue, "/workspace/demo/cinematic-abc/dialogue.wav")
            self.assertEqual(stems.singing, "/workspace/demo/cinematic-abc/singing.wav")
            self.assertEqual(stems.music, "/workspace/demo/cinematic-abc/music.wav")
            self.assertEqual(stems.effects, "/workspace/demo/cinematic-abc/effects.wav")

    def test_reparse_cleans_stale_cinematic_dirs_only_on_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "demo"
            output_dir.mkdir()
            old_dir = output_dir / "cinematic-old"
            old_dir.mkdir()
            (old_dir / "dialogue.wav").write_bytes(b"old")
            input_file = output_dir / "input.wav"
            write_wav(input_file, 1.0)
            args = SimpleNamespace(
                project_id="demo",
                input=input_file,
                output_dir=output_dir,
                target_lang="zh",
                mode="video_dubbing",
                device="cpu",
                skip_diarization=True,
                skip_llm=True,
            )

            with (
                patch.object(parse_worker, "_extract_wav_and_probe", return_value=1.0),
                patch.object(parse_worker, "_stage_banquet", side_effect=RuntimeError("boom")),
            ):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    parse_worker.run_pipeline(args)
            self.assertEqual([p.name for p in output_dir.glob("cinematic-*")], ["cinematic-old"])

            def fake_finish(cinematic_dir: Path, project_id: str):
                stems = CinematicStems(
                    dialogue=f"/workspace/{project_id}/{cinematic_dir.name}/dialogue.wav",
                    singing=f"/workspace/{project_id}/{cinematic_dir.name}/singing.wav",
                    music=f"/workspace/{project_id}/{cinematic_dir.name}/music.wav",
                    effects=f"/workspace/{project_id}/{cinematic_dir.name}/effects.wav",
                )
                return cinematic_dir / "dialogue.wav", cinematic_dir / "bgm.wav", stems

            with (
                patch.object(parse_worker, "_extract_wav_and_probe", return_value=1.0),
                patch.object(parse_worker, "_stage_banquet"),
                patch.object(parse_worker, "_finish_four_stems", side_effect=fake_finish),
                patch.object(parse_worker, "_stage_whisperx_asr_align", return_value=(None, {"segments": []}, "zh")),
                patch.object(parse_worker, "_stage_diarize", return_value=[]),
            ):
                script_out = parse_worker.run_pipeline(args)
            self.assertTrue(script_out.is_file())
            remaining = [p.name for p in output_dir.glob("cinematic-*")]
            self.assertEqual(len(remaining), 1)
            self.assertNotEqual(remaining[0], "cinematic-old")

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_render_fails_when_cinematic_bgm_missing(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "WORKSPACE_DIR", Path(directory)):
            script = ProjectScript(
                project_id="demo",
                mode="virtual_stage",
                media=MediaInfo(
                    duration=2.0,
                    bgm_wav="/workspace/demo/cinematic-x/bgm.wav",
                    cinematic_stems=CinematicStems(
                        dialogue="/workspace/demo/cinematic-x/dialogue.wav",
                        singing="/workspace/demo/cinematic-x/singing.wav",
                        music="/workspace/demo/cinematic-x/music.wav",
                        effects="/workspace/demo/cinematic-x/effects.wav",
                    ),
                ),
                segments=[],
            )
            with self.assertRaisesRegex(FileNotFoundError, "混合背景轨缺失"):
                media_mixer.mix_and_render(script)

    def test_vocal_track_preserves_48khz_and_defaults_to_44100(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "WORKSPACE_DIR", Path(directory)):
            project_dir = Path(directory) / "demo"
            project_dir.mkdir()
            vocals_path = project_dir / "cinematic-a" / "dialogue.wav"
            vocals_path.parent.mkdir()
            write_wav(vocals_path, 2.0, rate=48000, amplitude=500)
            script_48k = ProjectScript(
                project_id="demo",
                mode="virtual_stage",
                media=MediaInfo(duration=2.0, vocals_wav="/workspace/demo/cinematic-a/dialogue.wav"),
                segments=[],
            )
            out_48k = media_mixer._build_vocal_track(script_48k, project_dir)
            with wave.open(str(out_48k), "rb") as wav:
                self.assertEqual(wav.getframerate(), 48000)

            script_no_vocals = ProjectScript(
                project_id="demo",
                mode="virtual_stage",
                media=MediaInfo(duration=2.0),
                segments=[],
            )
            out_default = media_mixer._build_vocal_track(script_no_vocals, project_dir)
            with wave.open(str(out_default), "rb") as wav:
                self.assertEqual(wav.getframerate(), 44100)

    @unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
    def test_automatic_reference_accepts_only_project_banquet_dialogue(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(config, "WORKSPACE_DIR", Path(directory)):
            dialogue = Path(directory) / "demo" / "cinematic-example" / "dialogue.wav"
            dialogue.parent.mkdir(parents=True)
            write_wav(dialogue, 5)
            root = "/workspace/demo/cinematic-example"
            script = ProjectScript(
                project_id="demo",
                source_lang="ja",
                media=MediaInfo(
                    duration=5,
                    vocals_wav=f"{root}/dialogue.wav",
                    cinematic_stems=CinematicStems(
                        dialogue=f"{root}/dialogue.wav",
                        singing=f"{root}/singing.wav",
                        music=f"{root}/music.wav",
                        effects=f"{root}/effects.wav",
                    ),
                ),
                segments=[
                    Segment(id=0, start=0, end=4, speaker="A", orig_text="大丈夫。",
                            alignment_needs_review=True, speaker_needs_review=False,
                            alignment_granularity="segment", segment_timing_needs_review=False),
                ],
            )
            preview = references.automatic_reference(script, "A")
            self.assertIsNotNone(preview)
            self.assertTrue(config.resolve_media_path(preview.wav).is_file())
            foreign = Path(directory) / "other" / "cinematic-example" / "dialogue.wav"
            foreign.parent.mkdir(parents=True)
            write_wav(foreign, 5)
            script.media.vocals_wav = "/workspace/other/cinematic-example/dialogue.wav"
            script.media.cinematic_stems.dialogue = script.media.vocals_wav
            self.assertIsNone(references.automatic_reference(script, "A"))

    def test_config_imports_without_pydantic_settings(self):
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules['pydantic_settings'] = None; "
                "from backend import config; assert config.DURATION_OVERFLOW_RATIO == 1.15",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0)
