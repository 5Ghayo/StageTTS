from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend import config
from backend.agent.tools import batch_update_segments, load_project_script, save_project_script
from backend.main import app
from backend.schemas import CharacterConfig, MediaInfo, ProjectScript, Segment


def _sample_script(project_id: str = "proj_m2_test") -> ProjectScript:
    return ProjectScript(
        project_id=project_id,
        media=MediaInfo(duration=15.0, vocals_wav=f"/workspace/{project_id}/vocals.wav"),
        characters={
            "SPEAKER_00": CharacterConfig(display_name="角色A", color="#5B8DEF"),
        },
        segments=[
            Segment(
                id=0,
                start=0.5,
                end=2.5,
                speaker="SPEAKER_00",
                orig_text="Hello world.",
                target_text="你好世界。",
                target_lang="zh",
                tts_wav=f"/workspace/{project_id}/tts/seg_0000.wav",
                actual_duration=1.8,
                status="ready",
            ),
            Segment(
                id=1,
                start=4.0,
                end=6.5,
                speaker="SPEAKER_00",
                orig_text="Second line.",
                target_text="第二句台词。",
                target_lang="zh",
                tts_wav=f"/workspace/{project_id}/tts/seg_0001.wav",
                actual_duration=2.1,
                status="ready",
            ),
        ],
    )


class M2EditingHistoryPresetsTests(unittest.TestCase):
    def test_batch_update_segments_timing_and_invalid_range(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            with patch.object(config, "WORKSPACE_DIR", workspace_dir):
                script = _sample_script("proj_timing")
                save_project_script(script)

                # 将片段 #1 移动到片段 #0 之前，触发音频失效并按时间重排
                result = batch_update_segments(
                    "proj_timing",
                    [{"id": 1, "start": 0.1, "end": 0.4}],
                )
                self.assertEqual(result["updated_ids"], [1])
                self.assertEqual(result["marked_pending_ids"], [1])

                reloaded = load_project_script("proj_timing")
                self.assertEqual([seg.id for seg in reloaded.segments], [1, 0])
                self.assertAlmostEqual(reloaded.segments[0].start, 0.1)
                self.assertAlmostEqual(reloaded.segments[0].end, 0.4)
                self.assertEqual(reloaded.segments[0].status, "pending")
                self.assertIsNone(reloaded.segments[0].tts_wav)
                # 片段 #0 未受影响，仍为 ready
                self.assertEqual(reloaded.segments[1].status, "ready")

                # 非法时间范围应抛 ValueError
                with self.assertRaises(ValueError):
                    batch_update_segments(
                        "proj_timing",
                        [{"id": 0, "start": 3.0, "end": 2.0}],
                    )

    def test_project_script_rejects_duplicate_segment_ids(self) -> None:
        script = _sample_script("proj_dup")
        payload = script.model_dump(mode="json")
        payload["segments"].append({**payload["segments"][0], "start": 7.0, "end": 8.0})

        with self.assertRaises(ValidationError):
            ProjectScript.model_validate(payload)

        with TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            with patch.object(config, "WORKSPACE_DIR", workspace_dir):
                client = TestClient(app)
                res = client.put("/api/projects/proj_dup/script", json=payload)
                self.assertEqual(res.status_code, 422)

    def test_snapshot_dedup_pruning_and_restore_api(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            with patch.object(config, "WORKSPACE_DIR", workspace_dir):
                client = TestClient(app)
                script = _sample_script("proj_hist")
                payload = script.model_dump(mode="json")

                # 连续两次保存完全相同的内容，只应生成 1 条去重快照
                res1 = client.put("/api/projects/proj_hist/script", json=payload)
                self.assertEqual(res1.status_code, 200)
                res2 = client.put("/api/projects/proj_hist/script", json=payload)
                self.assertEqual(res2.status_code, 200)

                hist_initial = client.get("/api/projects/proj_hist/history")
                self.assertEqual(hist_initial.status_code, 200)
                self.assertEqual(len(hist_initial.json()["snapshots"]), 1)

                # 再写入 24 次不同改动，验证最多保留 20 条（倒序）
                for step in range(1, 25):
                    modified = json.loads(json.dumps(payload))
                    modified["segments"][0]["target_text"] = f"第 {step} 版台词"
                    put_res = client.put("/api/projects/proj_hist/script", json=modified)
                    self.assertEqual(put_res.status_code, 200)

                hist_after = client.get("/api/projects/proj_hist/history")
                self.assertEqual(hist_after.status_code, 200)
                snapshots = hist_after.json()["snapshots"]
                self.assertEqual(len(snapshots), 20)

                # 恢复到倒数第 6 条（第 19 版 -> index 5）
                target_snap_id = snapshots[5]["snapshot_id"]
                restore_res = client.post(
                    f"/api/projects/proj_hist/history/{target_snap_id}/restore"
                )
                self.assertEqual(restore_res.status_code, 200)
                self.assertEqual(restore_res.json()["segments"][0]["target_text"], "第 19 版台词")

                # 非法快照 ID 应被拦截
                bad_res = client.post("/api/projects/proj_hist/history/bad.snap!id/restore")
                self.assertEqual(bad_res.status_code, 400)

    def test_voice_presets_crud_and_isolated_audio_copy(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace_dir = Path(tmp) / "workspace"
            assets_dir = Path(tmp) / "assets"
            src_ref_dir = workspace_dir / "proj_preset_src" / "refs"
            src_ref_dir.mkdir(parents=True, exist_ok=True)
            src_wav = src_ref_dir / "ref_happy.wav"
            src_wav.write_bytes(b"RIFFfakewavpayload")

            with (
                patch.object(config, "WORKSPACE_DIR", workspace_dir),
                patch.object(config, "ASSETS_DIR", assets_dir),
            ):
                client = TestClient(app)

                # 缺失音频文件应返回 400
                bad_create = client.post(
                    "/api/voice-presets",
                    json={
                        "name": "坏预设",
                        "emotion_refs": {
                            "happy": {
                                "wav": "/workspace/proj_preset_src/refs/missing.wav",
                                "text": "不存在",
                                "lang": "zh",
                            }
                        },
                    },
                )
                self.assertEqual(bad_create.status_code, 400)

                # 创建全局预设：将参考音频独立复制到 assets/refs/presets/
                create_res = client.post(
                    "/api/voice-presets",
                    json={
                        "name": "阳光少年",
                        "speaking_style": "元气明快",
                        "gpt_weights": "assets/weights/boy.ckpt",
                        "emotion_refs": {
                            "happy": {
                                "wav": "/workspace/proj_preset_src/refs/ref_happy.wav",
                                "text": "太棒了！",
                                "lang": "zh",
                            }
                        },
                    },
                )
                self.assertEqual(create_res.status_code, 200)
                preset = create_res.json()
                preset_id = preset["preset_id"]
                copied_url = preset["emotion_refs"]["happy"]["wav"]
                self.assertTrue(copied_url.startswith(f"/assets/refs/presets/{preset_id}/"))

                copied_file = assets_dir / "refs" / "presets" / preset_id / "happy.wav"
                self.assertTrue(copied_file.is_file())
                self.assertEqual(copied_file.read_bytes(), b"RIFFfakewavpayload")

                # 删除原项目音频后，全局预设音频依然完好
                src_wav.unlink()
                self.assertTrue(copied_file.is_file())

                list_res = client.get("/api/voice-presets")
                self.assertEqual(list_res.status_code, 200)
                self.assertEqual([item["preset_id"] for item in list_res.json()["presets"]], [preset_id])

                # 删除预设并清理音频目录
                del_res = client.delete(f"/api/voice-presets/{preset_id}")
                self.assertEqual(del_res.status_code, 200)
                self.assertEqual(del_res.json()["deleted_id"], preset_id)
                self.assertFalse((assets_dir / "refs" / "presets" / preset_id).exists())

                list_after = client.get("/api/voice-presets")
                self.assertEqual(list_after.status_code, 200)
                self.assertEqual(list_after.json()["presets"], [])
