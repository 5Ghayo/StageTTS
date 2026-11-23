import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend import config, main
from backend.services import media_mixer, subtitles, task_runner


class CrossPlatformTest(unittest.TestCase):
    def test_resolve_gpt_sovits_python_windows_and_posix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "runtime" / "bin").mkdir(parents=True)
            posix_bin = root / "runtime" / "bin" / "python"
            posix_bin.touch()

            with patch.object(config.os, "name", "posix"):
                self.assertEqual(config._resolve_gpt_sovits_python("", root), str(posix_bin))

            (root / "runtime" / "python.exe").touch()
            with patch.object(config.os, "name", "nt"):
                self.assertEqual(
                    config._resolve_gpt_sovits_python("", root),
                    str(root / "runtime" / "python.exe"),
                )

            win_venv = Path(temp) / "win_only"
            (win_venv / ".venv" / "Scripts").mkdir(parents=True)
            win_py = win_venv / ".venv" / "Scripts" / "python.exe"
            win_py.touch()
            with patch.object(config.os, "name", "nt"):
                self.assertEqual(config._resolve_gpt_sovits_python("", win_venv), str(win_py))

            self.assertEqual(
                config._resolve_gpt_sovits_python("~/custom/python3", root),
                str(Path("~/custom/python3").expanduser()),
            )

            base_bin = root / "base_python"
            base_bin.touch()
            venv_dir = root / "venv" / "bin"
            venv_dir.mkdir(parents=True)
            venv_python = venv_dir / "python"
            try:
                os.symlink(base_bin, venv_python)
            except OSError:
                pass
            else:
                self.assertEqual(
                    config._resolve_gpt_sovits_python(str(venv_python), root),
                    str(venv_python),
                )

    def test_pick_ass_font_name_across_platforms(self):
        subtitles._pick_ass_font_name.cache_clear()
        self.addCleanup(subtitles._pick_ass_font_name.cache_clear)

        with patch.object(subtitles.os, "name", "nt"), patch.object(subtitles.sys, "platform", "win32"):
            self.assertEqual(subtitles._pick_ass_font_name(), "Microsoft YaHei")

        subtitles._pick_ass_font_name.cache_clear()
        with patch.object(subtitles.os, "name", "posix"), patch.object(subtitles.sys, "platform", "darwin"):
            self.assertEqual(subtitles._pick_ass_font_name(), "PingFang SC")

        subtitles._pick_ass_font_name.cache_clear()
        proc = SimpleNamespace(returncode=0, stdout="Noto Sans CJK SC,Noto Sans CJK SC Regular\nDejaVu Sans\n")
        with (
            patch.object(subtitles.os, "name", "posix"),
            patch.object(subtitles.sys, "platform", "linux"),
            patch.object(subtitles.shutil, "which", return_value="/usr/bin/fc-list"),
            patch.object(subtitles.subprocess, "run", return_value=proc),
        ):
            self.assertEqual(subtitles._pick_ass_font_name(), "Noto Sans CJK SC")

    def test_video_encoder_candidates_macos_and_other(self):
        with patch.object(media_mixer.sys, "platform", "darwin"):
            cands = media_mixer._video_encoder_candidates()
            self.assertEqual(cands[0][:2], ["-c:v", "h264_videotoolbox"])
            self.assertEqual(cands[-1][:2], ["-c:v", "libx264"])

        with patch.object(media_mixer.sys, "platform", "linux"):
            cands = media_mixer._video_encoder_candidates()
            self.assertEqual(cands[0][:2], ["-c:v", "h264_nvenc"])
            self.assertEqual(cands[-1][:2], ["-c:v", "libx264"])

    def test_sanitize_output_filename_blocks_windows_reserved_names(self):
        for bad in ("CON.mp4", "nul.mp4", "COM1.mp4", "lpt9", "clip .mp4", "clip..mp4"):
            with self.assertRaises(ValueError):
                media_mixer._sanitize_output_filename(bad, ".mp4")
        self.assertEqual(media_mixer._sanitize_output_filename("demo_clip", ".mp4"), "demo_clip.mp4")

    def test_resolve_export_dir_wraps_oserror_as_value_error(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "locked" / "sub"
            with patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
                with self.assertRaisesRegex(ValueError, "无法创建保存目录"):
                    media_mixer._resolve_export_dir(str(target))

    def test_discover_user_export_dirs_macos_and_localized_linux(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            (home / "Movies").mkdir()
            (home / "Downloads").mkdir()
            (home / "视频").mkdir()
            (home / "下载").mkdir()

            with (
                patch.object(main.Path, "home", return_value=home),
                patch.object(main.sys, "platform", "darwin"),
                patch.object(main.os, "name", "posix"),
            ):
                dirs = main._discover_user_export_dirs()
                self.assertEqual(
                    [(d.label, d.path) for d in dirs],
                    [
                        ("影片目录", str((home / "Movies").resolve())),
                        ("下载目录", str((home / "Downloads").resolve())),
                    ],
                )

            def fake_run(cmd, **_kwargs):
                kind = cmd[1]
                mapping = {"VIDEOS": str(home / "视频"), "DOWNLOAD": str(home / "下载"), "DESKTOP": str(home)}
                return SimpleNamespace(returncode=0, stdout=mapping.get(kind, "") + "\n")

            with (
                patch.object(main.Path, "home", return_value=home),
                patch.object(main.sys, "platform", "linux"),
                patch.object(main.os, "name", "posix"),
                patch.object(main.shutil, "which", return_value="/usr/bin/xdg-user-dir"),
                patch.object(main.subprocess, "run", side_effect=fake_run),
            ):
                dirs = main._discover_user_export_dirs()
                paths = [d.path for d in dirs]
                self.assertIn(str((home / "视频").resolve()), paths)
                self.assertIn(str((home / "下载").resolve()), paths)
                self.assertNotIn(str(home.resolve()), paths)
                self.assertEqual(len(paths), len(set(paths)))

    def test_terminate_process_tree_posix_and_windows_branches(self):
        posix_proc = MagicMock()
        posix_proc.pid = 4321
        posix_proc.poll.side_effect = [None, None]
        posix_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5.0), 0]
        with (
            patch.object(task_runner.os, "name", "posix"),
            patch.object(task_runner.os, "killpg", create=True) as killpg,
        ):
            task_runner._terminate_process_tree(posix_proc, grace_sec=1.0)
            self.assertEqual(
                [call.args for call in killpg.call_args_list],
                [(4321, signal.SIGTERM), (4321, signal.SIGKILL)],
            )

        win_proc = MagicMock()
        win_proc.pid = 9876
        win_proc.poll.side_effect = [None, None]
        win_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5.0), 0]
        with (
            patch.object(task_runner.os, "name", "nt"),
            patch.object(task_runner.subprocess, "run") as sub_run,
        ):
            task_runner._terminate_process_tree(win_proc, grace_sec=1.0)
            win_proc.terminate.assert_called_once()
            sub_run.assert_called_once_with(
                ["taskkill", "/T", "/F", "/PID", "9876"],
                capture_output=True,
                timeout=5,
            )
            win_proc.kill.assert_called_once()
