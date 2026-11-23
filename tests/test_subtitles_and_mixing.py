import math
import shutil
import subprocess
import tempfile
import unittest
import wave
from array import array
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend import config
from backend.agent import tools
from backend.main import app
from backend.schemas import CharacterConfig, CinematicStems, MediaInfo, ProjectScript, Segment, StemVolumes
from backend.services import media_mixer, subtitles


def _write_tone_wav(
    path: Path,
    seconds: float,
    *,
    rate: int = 48000,
    channels: int = 2,
    amplitude: int = 8000,
    freq: float = 440.0,
) -> None:
    total_frames = int(seconds * rate)
    samples = array("h")
    for idx in range(total_frames):
        val = int(amplitude * math.sin(2 * math.pi * freq * idx / rate)) if amplitude else 0
        for _ in range(channels):
            samples.append(val)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())


def _rms_between(wav_path: Path, start_sec: float, end_sec: float) -> float:
    with wave.open(str(wav_path), "rb") as wav:
        rate = wav.getframerate()
        channels = wav.getnchannels()
        start_frame = max(0, int(start_sec * rate))
        end_frame = min(wav.getnframes(), int(end_sec * rate))
        wav.setpos(start_frame)
        raw = array("h", wav.readframes(max(0, end_frame - start_frame)))
    if not raw:
        return 0.0
    step = max(1, channels)
    mono = [raw[i] for i in range(0, len(raw), step)]
    return math.sqrt(sum(sample * sample for sample in mono) / len(mono))


@unittest.skipUnless(shutil.which("ffmpeg"), "requires FFmpeg")
class SubtitlesAndMixingTest(unittest.TestCase):
    def test_ass_generation_and_download_endpoint(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            script = ProjectScript(
                project_id="ass_demo",
                mode="video_dubbing",
                media=MediaInfo(duration=8.0),
                characters={
                    "SPEAKER_00": CharacterConfig(display_name="主讲, 甲"),
                },
                segments=[
                    Segment(
                        id=0,
                        start=0.5,
                        end=1.999,
                        speaker="SPEAKER_00",
                        orig_text="Hello {world}\nNext line",
                        target_text="你好｛世界｝\n下一行",
                    ),
                    Segment(
                        id=1,
                        start=2.5,
                        end=4.0,
                        speaker="SPEAKER_00",
                        orig_text="Same",
                        target_text="Same",
                    ),
                ],
            )
            tools.save_project_script(script)

            target_ass = subtitles.generate_ass_subtitles(script, layout="target")
            self.assertIn("[Script Info]", target_ass)
            self.assertIn("[V4+ Styles]", target_ass)
            self.assertIn("Dialogue: 0,0:00:00.50,0:00:02.00,Primary,主讲  甲,0,0,0,,你好｛世界｝\\N下一行", target_ass)
            self.assertNotIn(r"{\rSecondary}", target_ass)

            bilingual_ass = subtitles.generate_ass_subtitles(script, layout="bilingual")
            self.assertIn(r"你好｛世界｝\N下一行\N{\rSecondary}Hello ｛world｝\NNext line", bilingual_ass)
            self.assertNotIn(r"Same\N{\rSecondary}Same", bilingual_ass)

            with TestClient(app) as client:
                resp = client.get("/api/projects/ass_demo/subtitles?layout=bilingual")
                self.assertEqual(resp.status_code, 200, resp.text)
                self.assertTrue(resp.content.startswith(b"\xef\xbb\xbf"))
                self.assertIn("ass_demo_bilingual.ass", resp.headers["content-disposition"])
                self.assertEqual(resp.content.decode("utf-8-sig"), bilingual_ass)

    def test_vocal_track_gap_spillover_fade_and_loudness_normalization(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            project_dir = Path(temp) / "vocal_mix"
            project_dir.mkdir()
            tts_dir = project_dir / "tts"
            tts_dir.mkdir()

            _write_tone_wav(tts_dir / "seg_0000.wav", 1.5, rate=44100, channels=1, amplitude=14000, freq=330.0)
            _write_tone_wav(tts_dir / "seg_0001.wav", 1.6, rate=44100, channels=1, amplitude=6000, freq=440.0)
            _write_tone_wav(tts_dir / "seg_0002.wav", 0.8, rate=44100, channels=1, amplitude=1500, freq=550.0)

            script = ProjectScript(
                project_id="vocal_mix",
                mode="virtual_stage",
                media=MediaInfo(duration=5.0),
                segments=[
                    Segment(
                        id=0, start=0.0, end=1.0, speaker="A", target_text="one",
                        tts_wav="/workspace/vocal_mix/tts/seg_0000.wav", status="ready",
                    ),
                    Segment(
                        id=1, start=2.0, end=3.0, speaker="A", target_text="two",
                        tts_wav="/workspace/vocal_mix/tts/seg_0001.wav", status="ready",
                    ),
                    Segment(
                        id=2, start=3.1, end=4.0, speaker="A", target_text="three",
                        tts_wav="/workspace/vocal_mix/tts/seg_0002.wav", status="ready",
                    ),
                ],
            )
            out_wav = media_mixer._build_vocal_track(script, project_dir)

            # 句 0 后方有 1.0s 空隙，1.5s 音频应完整延展保留而非在 1.0s 硬截断
            self.assertGreater(_rms_between(out_wav, 1.15, 1.40), 1000.0)
            # 句 1 后方紧邻 3.1s 句 2，应保留 >=50ms 保护间隙并在 3.05s 前淡出归零
            self.assertLess(_rms_between(out_wav, 3.055, 3.095), 5.0)
            # 极弱句 2 经响度对齐后，RMS 与句 0 收敛
            rms0 = _rms_between(out_wav, 0.2, 0.8)
            rms2 = _rms_between(out_wav, 3.2, 3.8)
            self.assertGreater(rms2, 1200.0)
            self.assertLess(max(rms0, rms2) / max(1.0, min(rms0, rms2)), 2.2)

    def test_four_stem_independent_mix_and_auto_ducking_real_ffmpeg(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            project_dir = Path(temp) / "stems_mix"
            stems_dir = project_dir / "cinematic-1"
            tts_dir = project_dir / "tts"
            stems_dir.mkdir(parents=True)
            tts_dir.mkdir(parents=True)

            _write_tone_wav(stems_dir / "dialogue.wav", 4.5, amplitude=0)
            _write_tone_wav(stems_dir / "singing.wav", 4.5, amplitude=0)
            _write_tone_wav(stems_dir / "music.wav", 4.5, amplitude=10000, freq=220.0)
            _write_tone_wav(stems_dir / "effects.wav", 4.5, amplitude=4000, freq=880.0)
            _write_tone_wav(tts_dir / "seg_0000.wav", 1.8, rate=48000, channels=1, amplitude=12000, freq=660.0)

            base_media = MediaInfo(
                duration=4.5,
                bgm_volume=1.0,
                vocals_wav="/workspace/stems_mix/cinematic-1/dialogue.wav",
                cinematic_stems=CinematicStems(
                    dialogue="/workspace/stems_mix/cinematic-1/dialogue.wav",
                    singing="/workspace/stems_mix/cinematic-1/singing.wav",
                    music="/workspace/stems_mix/cinematic-1/music.wav",
                    effects="/workspace/stems_mix/cinematic-1/effects.wav",
                ),
            )
            script = ProjectScript(
                project_id="stems_mix",
                mode="virtual_stage",
                media=base_media.model_copy(
                    update={"stem_volumes": StemVolumes(singing=0.0, music=0.0, effects=1.0), "auto_ducking": False}
                ),
                segments=[
                    Segment(
                        id=0, start=1.0, end=2.8, speaker="A", target_text="speech",
                        tts_wav="/workspace/stems_mix/tts/seg_0000.wav", status="ready",
                    )
                ],
            )
            media_mixer.mix_and_render(script)
            effects_only_rms = _rms_between(project_dir / "output.wav", 0.1, 0.7)
            self.assertGreater(effects_only_rms, 1500.0)

            script.media.stem_volumes = StemVolumes(singing=0.0, music=1.0, effects=0.0)
            script.media.auto_ducking = False
            media_mixer.mix_and_render(script)
            unducked_idle = _rms_between(project_dir / "output.wav", 0.1, 0.7)
            unducked_speech = _rms_between(project_dir / "output.wav", 1.2, 2.5)
            self.assertGreater(unducked_idle, effects_only_rms * 1.5)

            script.media.auto_ducking = True
            media_mixer.mix_and_render(script)
            ducked_idle = _rms_between(project_dir / "output.wav", 0.1, 0.7)
            ducked_speech = _rms_between(project_dir / "output.wav", 1.2, 2.5)
            self.assertAlmostEqual(ducked_idle, unducked_idle, delta=unducked_idle * 0.12)
            self.assertLess(ducked_speech, unducked_speech * 0.85)

    def test_single_pass_ass_subtitle_and_watermark_burn_real_ffmpeg(self):
        with ExitStack() as stack:
            temp = stack.enter_context(tempfile.TemporaryDirectory())
            stack.enter_context(patch.object(config, "WORKSPACE_DIR", Path(temp)))
            project_dir = Path(temp) / "burn_demo"
            tts_dir = project_dir / "tts"
            tts_dir.mkdir(parents=True)

            input_mp4 = project_dir / "input.mp4"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x180:d=2.0",
                    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo:d=2.0",
                    "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(input_mp4),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            _write_tone_wav(tts_dir / "seg_0000.wav", 1.2, rate=44100, channels=1, amplitude=6000)

            script = ProjectScript(
                project_id="burn_demo",
                mode="video_dubbing",
                media=MediaInfo(source_video="/workspace/burn_demo/input.mp4", duration=2.0),
                segments=[
                    Segment(
                        id=0, start=0.2, end=1.6, speaker="SPEAKER_00",
                        orig_text="Hello StageTTS!",
                        target_text="你好 StageTTS！",
                        tts_wav="/workspace/burn_demo/tts/seg_0000.wav",
                        status="ready",
                    )
                ],
            )
            tools.save_project_script(script)

            with TestClient(app) as client:
                lossless = client.post("/api/projects/burn_demo/render", json={})
                self.assertEqual(lossless.status_code, 200, lossless.text)
                self.assertEqual(lossless.json()["burn_subtitles"], "none")
                self.assertTrue((project_dir / "output.mp4").is_file())

                burned = client.post(
                    "/api/projects/burn_demo/render",
                    json={"burn_watermark": True, "burn_subtitles": "bilingual", "auto_ducking": False},
                )
                self.assertEqual(burned.status_code, 200, burned.text)
                payload = burned.json()
                self.assertTrue(payload["burn_watermark"])
                self.assertEqual(payload["burn_subtitles"], "bilingual")
                self.assertFalse(payload["auto_ducking"])
                self.assertGreater(payload["duration"], 1.5)
                self.assertGreater((project_dir / "output.mp4").stat().st_size, 1000)

    def test_agent_update_mix_settings_supports_stems_and_ducking(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(config, "WORKSPACE_DIR", Path(temp)):
            tools.save_project_script(
                ProjectScript(project_id="mix_agent", mode="video_dubbing", media=MediaInfo(duration=4.0))
            )
            res1 = tools.update_mix_settings("mix_agent", bgm_volume=0.8)
            self.assertEqual(res1["bgm_volume"], 0.8)
            self.assertIsNone(res1["stem_volumes"])
            self.assertTrue(res1["auto_ducking"])

            res2 = tools.update_mix_settings(
                "mix_agent",
                music_volume=0.35,
                effects_volume=1.25,
                auto_ducking=False,
            )
            self.assertEqual(res2["bgm_volume"], 0.8)
            self.assertEqual(res2["stem_volumes"], {"singing": 1.0, "music": 0.35, "effects": 1.25})
            self.assertFalse(res2["auto_ducking"])
