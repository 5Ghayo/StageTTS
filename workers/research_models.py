"""Adapters around the paper authors' model code. Run in an inference environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

RELEASES = {
    "dialogue-sidon": (
        "sarulab-speech/DialogueSidon", "d43d7478402a5527136c6733c3f4359c37b312ab",
        ("ssl_encoder.pt2", "diffusion_head.pt2", "vae_decoder.pt2", "metadata.json"),
    ),
    "emotion2vec": (
        "emotion2vec/emotion2vec_base", "0c9a3152734f9d7a7a05b4ee6bfb3c109d288664",
        ("emotion2vec_base.pt", "config.yaml", "configuration.json"),
    ),
    "facodec": (
        "amphion/naturalspeech3_facodec", "314afc3ea1455ba881a0e484ef9408b6cb996736",
        ("ns3_facodec_encoder.bin", "ns3_facodec_decoder.bin"),
    ),
}


def release_path(model: str, weights_dir: Path | None) -> Path:
    from huggingface_hub import snapshot_download

    repo, revision, files = RELEASES[model]
    local = weights_dir / model if weights_dir else None
    if local and all((local / name).is_file() for name in files):
        return local
    return Path(snapshot_download(repo, revision=revision, allow_patterns=list(files), local_dir=local))


def dialogue_sidon(job: dict) -> None:
    import soundfile as sf
    import torch

    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    spec = importlib.util.spec_from_file_location("dialogue_sidon_official", Path(job["sidon_root"]) / "spaces/app.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load official DialogueSidon inference entry point")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)

    directory = release_path("dialogue-sidon", Path(job["weights_dir"]) if job.get("weights_dir") else None)
    # Keep the author's scheduler, speech features, inference and stitching. Resolve its four
    # downloads to the pinned official export, also allowing preloaded offline weights.
    def local_weight(repo_id: str, filename: str, **_kwargs) -> str:
        if repo_id != RELEASES["dialogue-sidon"][0] or filename not in RELEASES["dialogue-sidon"][2]:
            raise ValueError(f"Unexpected DialogueSidon file {repo_id}/{filename}")
        return str(directory / filename)

    official.hf_hub_download = local_weight
    torch.manual_seed(job["seed"])
    device = official.get_device()
    if device.type == "cpu":
        from torch.export.passes import move_to_device_pass

        # The official exports were saved on CUDA. Map their serialized storages
        # and graph device arguments to CPU before calling the same inference.
        torch.serialization.register_package(
            0, lambda _storage: None,
            lambda storage, location: storage if location.startswith("cuda") else None,
        )
        load_export = torch.export.load
        torch.export.load = lambda path: move_to_device_pass(load_export(path), device)
    for item in job["mixtures"]:
        samples, rate = sf.read(item["input"], dtype="float32", always_2d=True)
        with torch.no_grad():
            separated, out_rate = official.separate(
                torch.from_numpy(samples.mean(axis=1)).unsqueeze(0), rate, 30, device,
            )
        if separated.ndim != 2 or separated.shape[0] != 2 or not torch.isfinite(separated).all():
            raise RuntimeError(f"DialogueSidon produced invalid tracks for {item['input']}")
        for track, path in zip(separated, item["outputs"]):
            sf.write(path, track.cpu().numpy(), out_rate, subtype="PCM_16")


def facodec(root: Path, weights: Path):
    import torch

    sys.path.insert(0, str(root / "models/codec"))
    from ns3_codec import FACodecDecoder, FACodecEncoder

    encoder = FACodecEncoder(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256)
    decoder = FACodecDecoder(
        in_channels=256, upsample_initial_channel=1024, ngf=32,
        up_ratios=[5, 5, 4, 2], vq_num_q_c=2, vq_num_q_p=1, vq_num_q_r=3,
        vq_dim=256, codebook_dim=8, codebook_size_prosody=10,
        codebook_size_content=10, codebook_size_residual=10,
        use_gr_x_timbre=True, use_gr_residual_f0=True, use_gr_residual_phone=True,
    )
    for component, name in ((encoder, "encoder"), (decoder, "decoder")):
        component.load_state_dict(torch.load(weights / f"ns3_facodec_{name}.bin", map_location="cpu", weights_only=True))
        component.eval()
    return encoder, decoder


def extract_features(job: dict) -> None:
    import librosa
    import numpy as np
    import torch
    from funasr import AutoModel

    weights_dir = Path(job["weights_dir"]) if job.get("weights_dir") else None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The authors document FunASR for extracting the ACL paper's original base embeddings.
    emotion_model = AutoModel(
        model=str(release_path("emotion2vec", weights_dir)), device=device, disable_update=True,
    )
    encoder, decoder = facodec(Path(job["amphion_root"]), release_path("facodec", weights_dir))
    encoder, decoder = encoder.to(device), decoder.to(device)
    for item in job["clips"]:
        path = item["audio"]
        speech = librosa.load(path, sr=16_000)[0]
        emotion = np.asarray(emotion_model.generate(
            input=speech, granularity="utterance", extract_embedding=True,
        )[0]["feats"], dtype=np.float32)
        with torch.no_grad():
            _, codes, _, _, timbre = decoder(
                encoder(torch.from_numpy(speech)[None, None].to(device)), eval_vq=False, vq=True,
            )
        if codes.shape[0] != 6 or emotion.shape != (768,) or not np.isfinite(emotion).all():
            raise RuntimeError(f"Invalid FACodec codes or emotion2vec vector for {path}")
        np.savez_compressed(
            item["features"], emotion2vec=emotion,
            prosody=codes[:1].cpu().numpy(), content=codes[1:3].cpu().numpy(),
            acoustic_detail=codes[3:].cpu().numpy(), timbre=timbre.cpu().numpy(),
        )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dialogue-sidon", "features"))
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    if args.stage == "dialogue-sidon":
        dialogue_sidon(job)
    elif args.stage == "features":
        extract_features(job)


if __name__ == "__main__":
    main()
