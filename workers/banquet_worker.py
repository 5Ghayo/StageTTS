"""Run the four-stem Banquet checkpoint from the paper Facing the Music."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

SAMPLE_RATE = 48_000
# Embedding row order in banquet-split.ckpt, checked against isolated sources.
STEMS = ("dialogue", "effects", "singing", "music")


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
def _normalize_peak(signal: torch.Tensor, limit: float = 0.99) -> torch.Tensor:
    if signal.numel() == 0:
        return signal
    peak = float(signal.abs().max().item())
    if peak > limit:
        return signal * (limit / peak)
    return signal


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

    audio, sample_rate = sf.read(args.input, dtype="float32", always_2d=True)
    if len(audio) == 0:
        raise ValueError("Banquet 输入音频为空。")
    waveform = torch.from_numpy(audio.T)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

    emit(f"加载 Banquet 模型 ({args.device})...")
    model = load_model(args.source_root, args.checkpoint, args.device)
    estimates = separate(model, waveform, args.device, progress=emit, hop_seconds=args.hop_seconds)
    emit("正在写入四轨 WAV 文件...")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stem, signal in estimates.items():
        if not torch.isfinite(signal).all():
            raise RuntimeError(f"Banquet 生成的 {stem} 轨包含非有限值。")
        # Write 16-bit PCM WAV so dialogue works with stdlib wave and all stems play reliably in browsers.
        normalized = _normalize_peak(signal)
        sf.write(args.output_dir / f"{stem}.wav", normalized.T.numpy(), SAMPLE_RATE, subtype="PCM_16")


if __name__ == "__main__":
    main()
