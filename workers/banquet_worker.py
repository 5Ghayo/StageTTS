"""Run the four-stem Banquet checkpoint from the paper Facing the Music."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

SAMPLE_RATE = 48_000
# Embedding row order in banquet-split.ckpt, checked against isolated sources.
STEMS = ("dialogue", "effects", "singing", "music")
CHUNK_SECONDS = 300
SEARCH_SECONDS = 30
CONTEXT_SECONDS = 8


def load_model(source_root: Path, checkpoint: Path, device: str) -> nn.Module:
    sys.path.insert(0, str(source_root))
    from src.models.bandit.bandit import BaseBandit, OverlappingMaskEstimationModule
    from src.models.bandit.film import BTFBroadcastedFiLM

    class Banquet(BaseBandit):
        def __init__(self) -> None:
            super().__init__(
                in_channels=1, fs=SAMPLE_RATE, n_sqm_modules=8,
                rnn_type="GRU", pad_mode="reflect",
            )
            self.mask_estim = OverlappingMaskEstimationModule(
                in_channels=1, band_specs=self.band_specs.get_band_specs(),
                freq_weights=self.band_specs.get_freq_weights(), n_freq=1025,
                emb_dim=128, mlp_dim=512, hidden_activation_kwargs={},
                complex_mask=True, use_freq_weights=True,
            )
            self.substems_gammas = nn.Embedding(len(STEMS), 128)
            self.substems_betas = nn.Embedding(len(STEMS), 128)
            self.broadcasted_film = BTFBroadcastedFiLM()

        def separate(self, batch):
            spectrum, embedding, length = self.encode(batch)
            estimates = {}
            for index, stem in enumerate(STEMS):
                conditioned = self.broadcasted_film(
                    embedding, self.substems_gammas.weight[index], self.substems_betas.weight[index],
                )
                mask = self.mask_estim(conditioned)
                estimates[stem] = self.istft(spectrum * mask, length)
            return estimates

    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)["state_dict"]
    model = Banquet()
    model.load_state_dict({key[6:]: value for key, value in state.items() if key.startswith("model.")})
    return model.eval().to(device)


@torch.inference_mode()
def separate(
    model: nn.Module,
    waveform: torch.Tensor,
    device: str,
    progress: Callable[[str], None] | None = None,
    hop_seconds: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Use the paper's 8 s window and configurable hop (default 1 s), with one inference for a short clip."""
    if not (0 < hop_seconds <= 4.0):
        raise ValueError("Banquet hop_seconds 必须在 (0, 4.0] 秒范围内。")

    def infer(chunk: torch.Tensor) -> dict[str, torch.Tensor]:
        # The checkpoint is mono. Run one channel at a time to keep peak VRAM low on 8GB GPUs.
        per_stem: dict[str, list[torch.Tensor]] = {}
        for channel in range(chunk.shape[0]):
            mono = chunk[channel : channel + 1, None].to(device)
            signals = model({"mixture": {"audio": mono}})
            for stem, signal in signals.items():
                per_stem.setdefault(stem, []).append(signal[0, 0].cpu())
        return {stem: torch.stack(channels, dim=0) for stem, channels in per_stem.items()}

    length = waveform.shape[-1]
    chunk_size = 8 * SAMPLE_RATE
    hop_size = max(1, round(hop_seconds * SAMPLE_RATE))
    if length <= chunk_size:
        if progress is not None:
            progress("正在分离音频（1/1 窗口）")
        # Reflect padding in the STFT requires more than n_fft // 2 samples.
        waveform = F.pad(waveform, (0, max(0, 1025 - length)))
        return {stem: signal[..., :length] for stem, signal in infer(waveform).items()}

    overlap = chunk_size - hop_size
    waveform = F.pad(waveform, (overlap, overlap), mode="reflect")
    window = torch.hann_window(chunk_size)
    weights = torch.zeros(length)
    estimates = {stem: torch.zeros((waveform.shape[0], length)) for stem in STEMS}

    # Accumulate only the part of each window that overlaps the original audio.
    starts = range(0, overlap + length, hop_size)
    total = len(starts)
    report_step = max(1, total // 20)
    for idx, start in enumerate(starts, start=1):
        chunk = waveform[..., start:start + chunk_size]
        chunk = F.pad(chunk, (0, chunk_size - chunk.shape[-1]))
        signals = infer(chunk)
        left = max(start, overlap)
        right = min(start + chunk_size, overlap + length)
        output_slice = slice(left - overlap, right - overlap)
        chunk_slice = slice(left - start, right - start)
        fade = window[chunk_slice]
        weights[output_slice] += fade
        for stem, signal in signals.items():
            estimates[stem][..., output_slice] += signal[..., chunk_slice] * fade
        if progress is not None and (idx == 1 or idx == total or idx % report_step == 0):
            progress(f"分离进度 {idx}/{total} 窗口 ({idx * 100 // total}%)")

    for signal in estimates.values():
        signal.div_(weights)
    return estimates


def chunk_boundaries(source: sf.SoundFile, target_seconds: float = CHUNK_SECONDS,
                     search_seconds: float = SEARCH_SECONDS) -> list[int]:
    """Find a low-energy seam close to each target on the source sample grid."""
    rate, total = source.samplerate, source.frames
    target, search = round(target_seconds * rate), round(search_seconds * rate)
    hop = round(rate * 0.02)
    points = [0]
    while total - points[-1] > target + search:
        nominal = points[-1] + target
        first = max(points[-1] + hop, nominal - search)
        # Avoid starting an almost empty final chunk.
        last = min(nominal + search, total - min(target // 5, 60 * rate))
        source.seek(first)
        samples = source.read(last - first, dtype="float32", always_2d=True)
        count = len(samples) // hop
        if not count:
            points.append(nominal)
            continue
        frames = samples[:count * hop].reshape(count, hop, source.channels)
        # Take the louder channel so out-of-phase stereo does not appear silent.
        energy = np.mean(np.square(frames), axis=1).max(axis=1)
        quiet = energy <= 10 ** (-45 / 10)
        edges = np.flatnonzero(np.diff(np.r_[False, quiet, False]))
        centres = [first + (int(a) + int(b)) * hop // 2
                   for a, b in edges.reshape(-1, 2) if (b - a) * hop >= 0.25 * rate]
        if centres:
            cut = min(centres, key=lambda point: abs(point - nominal))
        else:
            candidates = np.flatnonzero(energy == energy.min())
            nearest = min(candidates, key=lambda index: abs(first + index * hop + hop // 2 - nominal))
            cut = first + int(nearest) * hop + hop // 2
        points.append(cut)
    return [*points, total]


def separate_file(model: nn.Module, input_path: Path, output_dir: Path, device: str,
                  progress: Callable[[str], None], hop_seconds: float,
                  target_seconds: float = CHUNK_SECONDS, search_seconds: float = SEARCH_SECONDS) -> None:
    """Separate bounded chunks and write their non-overlapping cores back in order."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".banquet-", dir=output_dir) as directory:
        temp = Path(directory)
        if sf.info(input_path).samplerate != SAMPLE_RATE:
            normalized = temp / "input.wav"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(input_path), "-vn",
                            "-ar", str(SAMPLE_RATE), "-c:a", "pcm_f32le", str(normalized)], check=True)
            input_path = normalized
        with sf.SoundFile(input_path) as source, ExitStack() as stack:
            if not source.frames:
                raise ValueError("Banquet 输入音频为空。")
            boundaries = chunk_boundaries(source, target_seconds, search_seconds)
            total_chunks = len(boundaries) - 1
            context = round(CONTEXT_SECONDS * SAMPLE_RATE)
            hop_samples = max(1, round(hop_seconds * SAMPLE_RATE))
            buffers = {stem: stack.enter_context(sf.SoundFile(
                temp / f"{stem}-float.wav", mode="w", samplerate=SAMPLE_RATE,
                channels=source.channels, subtype="FLOAT",
            )) for stem in STEMS}
            peaks = dict.fromkeys(STEMS, 0.0)
            for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]), start=1):
                progress(f"分段 {index}/{total_chunks}：{start / SAMPLE_RATE:.2f}–{end / SAMPLE_RATE:.2f} 秒")
                left = max(0, (start - context) // hop_samples * hop_samples)
                right = min(source.frames, end + context)
                source.seek(left)
                samples = source.read(right - left, dtype="float32", always_2d=True)
                estimates = separate(model, torch.from_numpy(samples.T), device, progress, hop_seconds)
                if estimates.keys() != buffers.keys():
                    raise RuntimeError("Banquet 输出轨道与对白、音效、歌声、音乐不一致。")
                for stem in STEMS:
                    signal = estimates[stem]
                    core = signal[:, start - left:end - left].T.numpy()
                    if len(core) != end - start or not np.isfinite(core).all():
                        raise RuntimeError(f"Banquet 生成的 {stem} 分段无效。")
                    peaks[stem] = max(peaks[stem], float(np.abs(core).max()))
                    buffers[stem].write(core)
                del estimates, signal, core, samples
            channels, expected = source.channels, source.frames

        progress("正在拼接并保存四轨 WAV 文件...")
        for stem in STEMS:
            gain = 0.99 / max(0.99, peaks[stem])
            with sf.SoundFile(temp / f"{stem}-float.wav") as separated:
                if separated.frames != expected:
                    raise RuntimeError(f"Banquet 生成的 {stem} 轨长度与原音频不一致。")
                with sf.SoundFile(temp / f"{stem}.wav", "w", SAMPLE_RATE, channels, "PCM_16") as output:
                    for block in separated.blocks(blocksize=5 * SAMPLE_RATE, dtype="float32"):
                        output.write(block * gain)
        for stem in STEMS:
            (temp / f"{stem}.wav").replace(output_dir / f"{stem}.wav")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--hop-seconds", type=float, default=1.0)
    args = parser.parse_args()

    def emit(message: str) -> None:
        print(f"[banquet] {message}", flush=True)

    emit(f"加载 Banquet 模型 ({args.device})...")
    model = load_model(args.source_root, args.checkpoint, args.device)
    separate_file(model, args.input, args.output_dir, args.device, emit, args.hop_seconds)


if __name__ == "__main__":
    main()
