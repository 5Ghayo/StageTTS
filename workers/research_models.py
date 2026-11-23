"""Adapters around the paper authors' model code. Run in an inference environment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import runpy
import sys
from functools import partial
from pathlib import Path

RELEASES = {
    "dialogue-sidon": (
        "sarulab-speech/DialogueSidon", "d43d7478402a5527136c6733c3f4359c37b312ab",
        ("ssl_encoder.pt2", "diffusion_head.pt2", "vae_decoder.pt2", "metadata.json"),
    ),
    "kyaraembed": (
        "spellbrush/kyaraembed", "25bb09b1aa13e1fe51e1894b17eae6b0bb6c7797",
        ("kyaraembed_v1.0.pt", "kyaraembed_v1.0.pt.sha256"),
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


def kyara_checkpoint(weights_dir: Path | None) -> Path:
    from hashlib import sha256

    directory = release_path("kyaraembed", weights_dir)
    path = directory / "kyaraembed_v1.0.pt"
    digest = sha256()
    with path.open("rb") as input_file:
        for block in iter(partial(input_file.read, 1024 * 1024), b""):
            digest.update(block)
    expected = (directory / "kyaraembed_v1.0.pt.sha256").read_text().split()[0]
    if digest.hexdigest() != expected:
        raise ValueError("KyaraEmbed checkpoint did not match the author's SHA-256 file")
    return path


def kyara_adapter(root: Path, weights_dir: Path | None):
    sys.path.insert(0, str(root))
    from embedders import kyaraembed

    checkpoint = str(kyara_checkpoint(weights_dir))
    os.environ["KYARAEMBED_CKPT"] = checkpoint
    # The official adapter strictly loads KyaraEmbed after constructing this
    # architecture. Use the same checkpoint for initialization to avoid a
    # redundant download of the general-purpose base checkpoint.
    kyaraembed.AnimeSpeakerEmbedding = partial(kyaraembed.AnimeSpeakerEmbedding, ckpt_path=checkpoint)
    return kyaraembed


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
    kyaraembed = kyara_adapter(Path(job["kyara_root"]), weights_dir)
    from kyarabench.audio import load_clip
    from kyarabench.protocol import pool

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The authors document FunASR for extracting the ACL paper's original base embeddings.
    emotion_model = AutoModel(
        model=str(release_path("emotion2vec", weights_dir)), device=device, disable_update=True,
    )
    encoder, decoder = facodec(Path(job["amphion_root"]), release_path("facodec", weights_dir))
    encoder, decoder = encoder.to(device), decoder.to(device)
    embeddings = []
    references = {}
    for item in job["clips"]:
        path = item["audio"]
        fixed, length = load_clip(path)
        kyara = np.asarray(kyaraembed.embed([fixed], [length]), dtype=np.float32)[0]
        norm = np.linalg.norm(kyara)
        if kyara.shape != (192,) or not np.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"Invalid KyaraEmbed vector for {path}")
        kyara = kyara / norm
        if item["character_id"]:
            references.setdefault(item["character_id"], []).append(len(embeddings))
        embeddings.append(kyara)
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
            item["features"], kyaraembed=kyara, emotion2vec=emotion,
            prosody=codes[:1].cpu().numpy(), content=codes[1:3].cpu().numpy(),
            acoustic_detail=codes[3:].cpu().numpy(), timbre=timbre.cpu().numpy(),
        )
    if references:
        table = np.stack(embeddings)
        # KyaraBench uses cosine similarity against a normalized mean-pooled bundle.
        prototypes = {name: pool(table, indices[:4]) for name, indices in references.items()}
        ranking = {}
        for index, item in enumerate(job["clips"]):
            matches = [
                {"character_id": name, "similarity": float(table[index] @ prototypes[name])}
                for name in item["candidate_characters"] if name in prototypes
            ]
            if matches:
                ranking[index] = sorted(matches, key=lambda match: match["similarity"], reverse=True)
        Path(job["ranking"]).write_text(json.dumps(ranking), encoding="utf-8")


def run_kyarabench(job: dict) -> None:
    directory = Path(job["kyara_root"])
    weights_dir = Path(job["weights_dir"]) if job.get("weights_dir") else None
    kyara_adapter(directory, weights_dir)
    from kyarabench.audio import clip_path

    clips = json.loads((directory / "data/kyarabench_v1.0_source_clips.json").read_text(encoding="utf-8"))["clips"]
    missing = sum(not Path(clip_path(job["clips_root"], clip["uuid"])).is_file() for clip in clips)
    if missing:
        raise FileNotFoundError(f"KyaraBench is missing {missing} of {len(clips)} official restored clips")
    scorer = str(directory / "score.py")
    sys.argv = [scorer, "--clips", job["clips_root"], "--embedder", "embedders.kyaraembed"]
    runpy.run_path(scorer, run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dialogue-sidon", "features", "kyarabench"))
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text(encoding="utf-8"))
    if args.stage == "dialogue-sidon":
        dialogue_sidon(job)
    elif args.stage == "features":
        extract_features(job)
    else:
        run_kyarabench(job)


if __name__ == "__main__":
    main()
