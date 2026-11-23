"""分块运行 Demucs，并在采样级拼接人声和伴奏。"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from array import array
from pathlib import Path
from typing import Callable

_CHUNK_SECONDS = 90
_RETRY_SECONDS = 30
_OVERLAP_SECONDS = 1


class DemucsError(RuntimeError):
    pass


def _sample_gpu_memory(stop: threading.Event, peaks: dict[str, int]) -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return
    while not stop.is_set():
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=2,
            )
            for line in result.stdout.splitlines():
                index, used = (value.strip() for value in line.split(","))
                peaks[index] = max(peaks.get(index, 0), int(used))
        except (OSError, subprocess.SubprocessError, ValueError):
            return
        stop.wait(1)


def _run_demucs(chunk: Path, output: Path, device: str, timeout: float) -> None:
    if timeout <= 0:
        raise DemucsError("分离超时")
    command = [
        sys.executable, "-m", "demucs", "-n", "htdemucs", "--two-stems=vocals",
        "--segment", "7", "-d", device, "-o", str(output), str(chunk),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise DemucsError("分离超时") from exc
    if result.returncode != 0:
        details = (result.stderr or result.stdout).splitlines()
        # 保留识别 OOM 所需的信息，日志仅输出错误末尾。
        raise DemucsError("\n".join(line for line in details if "seconds/s]" not in line and "%|" not in line))


def _read_stem(path: Path, frames: int, channels: int, rate: int) -> bytes:
    try:
        with wave.open(str(path), "rb") as stem:
            if (stem.getnchannels(), stem.getsampwidth(), stem.getframerate()) == (channels, 2, rate):
                data = stem.readframes(frames + 1)
                if len(data) == frames * channels * 2:
                    return data
    except (wave.Error, EOFError):
        pass

    # 某些音频后端会输出浮点 WAV，统一到输入 PCM 格式再拼接。
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(path), "-f", "s16le", "-ac", str(channels), "-ar", str(rate), "pipe:1"],
            check=True, capture_output=True, timeout=60,
        )
    except subprocess.SubprocessError as exc:
        raise DemucsError(f"无法读取分离音轨 {path.name}") from exc
    if len(result.stdout) != frames * channels * 2:
        raise DemucsError(f"分离音轨 {path.name} 与输入块的采样数不一致")
    return result.stdout


def _crossfade(before: bytes, after: bytes, channels: int) -> bytes:
    left, right = array("h", before), array("h", after)
    if sys.byteorder != "little":
        left.byteswap()
        right.byteswap()
    frames = len(left) // channels
    for idx in range(len(left)):
        weight = (idx // channels + 1) / (frames + 1)
        left[idx] = round(left[idx] * (1 - weight) + right[idx] * weight)
    if sys.byteorder != "little":
        left.byteswap()
    return left.tobytes()


def _separate_chunks(
    extracted_wav: Path, tmp_dir: Path, device: str, chunk_seconds: int,
    deadline: float, log: Callable[[str], None],
) -> None:
    outputs = tmp_dir / "stems"
    with (
        wave.open(str(extracted_wav), "rb") as source,
        wave.open(str(tmp_dir / "vocals.wav"), "wb") as vocals,
        wave.open(str(tmp_dir / "bgm.wav"), "wb") as bgm,
    ):
        stems = {"vocals": vocals, "no_vocals": bgm}
        channels, width, rate, frames, *_ = source.getparams()
        if width != 2 or frames == 0:
            raise DemucsError("输入音频应为非空 16-bit PCM WAV")
        for writer in stems.values():
            writer.setparams(source.getparams())
        chunk_frames = rate * chunk_seconds
        total = (frames + chunk_frames - 1) // chunk_frames
        frame_bytes = channels * width
        tails = {name: b"" for name in stems}
        for idx, start in enumerate(range(0, frames, chunk_frames), start=1):
            end = min(frames, start + chunk_frames)
            padded_end = min(frames, end + rate * _OVERLAP_SECONDS)
            source.setpos(start)
            chunk = tmp_dir / "chunk.wav"
            with wave.open(str(chunk), "wb") as writer:
                writer.setparams(source.getparams())
                writer.writeframes(source.readframes(padded_end - start))

            log(f"Demucs {device} 分离块 {idx}/{total}（{start / rate:.1f}–{padded_end / rate:.1f}s）")
            _run_demucs(chunk, outputs, device, deadline - time.monotonic())
            stem_dir = outputs / "htdemucs" / chunk.stem
            core_bytes = (end - start) * frame_bytes
            for name, writer in stems.items():
                data = _read_stem(stem_dir / f"{name}.wav", padded_end - start, channels, rate)
                overlap = len(tails[name])
                if overlap:
                    writer.writeframesraw(_crossfade(tails[name], data[:overlap], channels))
                writer.writeframesraw(data[overlap:core_bytes])
                tails[name] = data[core_bytes:]
            shutil.rmtree(stem_dir)


def separate_stems(
    extracted_wav: Path, output_dir: Path, device: str, allow_cpu: bool, log: Callable[[str], None],
) -> tuple[Path, Path] | None:
    attempts = []
    if device == "cuda":
        attempts.append(("cuda", _CHUNK_SECONDS))
    if allow_cpu:
        attempts.append(("cpu", _CHUNK_SECONDS))
    with wave.open(str(extracted_wav), "rb") as source:
        duration = source.getnframes() / source.getframerate()

    while attempts:
        target, size = attempts.pop(0)
        started = time.monotonic()
        if target == "cpu":
            log("已启用 Demucs CPU 分离，长素材可能耗时较长")
        peaks: dict[str, int] = {}
        stop = threading.Event()
        monitor = threading.Thread(target=_sample_gpu_memory, args=(stop, peaks), daemon=True)
        if target == "cuda":
            monitor.start()
        try:
            with tempfile.TemporaryDirectory(prefix="stagetts_demucs_", dir=output_dir) as temp:
                tmp_dir = Path(temp)
                _separate_chunks(extracted_wav, tmp_dir, target, size, started + 900, log)
                for name in ("vocals.wav", "bgm.wav"):
                    (tmp_dir / name).replace(output_dir / name)
            return output_dir / "vocals.wav", output_dir / "bgm.wav"
        except (DemucsError, OSError, wave.Error, EOFError) as exc:
            details = str(exc)
            log(f"Demucs {target} 分离失败：{details.strip()[-260:]}")
            oom = any(marker in details.lower() for marker in (
                "out of memory", "cudaerrormemoryallocation", "cublas_status_alloc_failed",
            ))
            if target == "cuda" and oom and size > _RETRY_SECONDS and duration > _RETRY_SECONDS:
                log(f"CUDA 显存不足，改用 {_RETRY_SECONDS}s 分块重试")
                attempts.insert(0, ("cuda", _RETRY_SECONDS))
        finally:
            stop.set()
            if target == "cuda":
                monitor.join()
            peak = "、".join(f"GPU {index} {used} MiB" for index, used in sorted(peaks.items()))
            usage = f"，采样总显存峰值 {peak or '不可用'}" if target == "cuda" else ""
            log(f"Demucs {target} 尝试耗时 {time.monotonic() - started:.1f}s{usage}")
    return None
