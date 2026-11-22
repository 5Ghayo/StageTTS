"""Run the four-stem Banquet checkpoint from the paper Facing the Music."""

from __future__ import annotations

import argparse
import sys
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
def separate(model: nn.Module, waveform: torch.Tensor, device: str) -> dict[str, torch.Tensor]:
    """Use the paper's 8 s window and 1 s hop, with one inference for a short clip."""
    def infer(chunk: torch.Tensor) -> dict[str, torch.Tensor]:
        # The checkpoint is mono. Treat each source channel as a batch item.
        signals = model({"mixture": {"audio": chunk[:, None].to(device)}})
        return {stem: signal[:, 0].cpu() for stem, signal in signals.items()}

    length = waveform.shape[-1]
    chunk_size = 8 * SAMPLE_RATE
    hop_size = SAMPLE_RATE
    if length <= chunk_size:
        # Reflect padding in the STFT requires more than n_fft // 2 samples.
        waveform = F.pad(waveform, (0, max(0, 1025 - length)))
        return {stem: signal[..., :length] for stem, signal in infer(waveform).items()}

    overlap = chunk_size - hop_size
    waveform = F.pad(waveform, (overlap, overlap), mode="reflect")
    window = torch.hann_window(chunk_size)
    weights = torch.zeros(length)
    estimates = {stem: torch.zeros((waveform.shape[0], length)) for stem in STEMS}

    # Accumulate only the part of each window that overlaps the original audio.
    for start in range(0, overlap + length, hop_size):
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
    args = parser.parse_args()

    audio, sample_rate = sf.read(args.input, dtype="float32", always_2d=True)
    if len(audio) == 0:
        raise ValueError("Banquet 输入音频为空。")
    waveform = torch.from_numpy(audio.T)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

    model = load_model(args.source_root, args.checkpoint, args.device)
    estimates = separate(model, waveform, args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stem, signal in estimates.items():
        if not torch.isfinite(signal).all():
            raise RuntimeError(f"Banquet 生成的 {stem} 轨包含非有限值。")
        sf.write(args.output_dir / f"{stem}.wav", signal.T.numpy(), SAMPLE_RATE, subtype="FLOAT")


if __name__ == "__main__":
    main()
