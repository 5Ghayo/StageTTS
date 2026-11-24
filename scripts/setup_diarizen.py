"""Acquire pinned public DiariZen assets and install an isolated runtime.

Example (Python 3.10):
  python scripts/setup_diarizen.py --root .cache/diarizen --environment .venv-diarizen
For CUDA machines pass --torch-version 2.8.0 --torch-index https://download.pytorch.org/whl/cu128.
Never installs packages into the caller's parse, backend or TTS interpreter.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workers.speaker_pipeline import DIARIZEN_MODEL, DIARIZEN_REVISION, DIARIZEN_SOURCE


def download(url: str, destination: Path) -> None:
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    try:
        print(f"Downloading {destination.name}", flush=True)
        with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--torch-version", default="2.5.1")
    parser.add_argument("--torch-index")
    args = parser.parse_args()
    root, environment = args.root.resolve(), args.environment.absolute()
    if environment == Path(sys.prefix).absolute():
        parser.error("Choose a separate environment, not the running interpreter")
    root.mkdir(parents=True, exist_ok=True)
    archive = root / f"DiariZen-{DIARIZEN_SOURCE}.zip"
    download(f"https://codeload.github.com/BUTSpeechFIT/DiariZen/zip/{DIARIZEN_SOURCE}", archive)
    source = root / f"DiariZen-{DIARIZEN_SOURCE}"
    if not source.is_dir():
        with zipfile.ZipFile(archive) as bundle, tempfile.TemporaryDirectory(dir=root) as staging:
            for entry in bundle.infolist():
                target = Path(staging) / entry.filename
                if not target.resolve().is_relative_to(Path(staging).resolve()):
                    raise ValueError("Unexpected path in upstream source archive")
                # Upstream's training recipe data link is unnecessary for inference.
                if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    continue
                bundle.extract(entry, staging)
            (Path(staging) / source.name).rename(source)
    model = root / "diarizen-large-v2"
    for name in ("config.toml", "config.json", "pytorch_model.bin", "plda/plda.npz", "plda/xvec_transform.npz"):
        download(f"https://huggingface.co/{DIARIZEN_MODEL}/resolve/{DIARIZEN_REVISION}/{name}", model / name)
    embedding = root / "wespeaker-resnet34.bin"
    download("https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/resolve/"
             "837717ddb9ff5507820346191109dc79c958d614/pytorch_model.bin", embedding)
    uv = shutil.which("uv")
    if not environment.exists():
        if uv:
            subprocess.run([uv, "venv", "--python", sys.executable, str(environment)], check=True)
        else:
            venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt").create(environment)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    install = [uv, "pip", "install", "--python", str(python)] if uv else [str(python), "-m", "pip", "install"]
    torch_command = [*install, f"torch=={args.torch_version}", f"torchaudio=={args.torch_version}"]
    if args.torch_index:
        torch_command.extend(["--index-url", args.torch_index])
    subprocess.run(torch_command, check=True)
    requirements = Path(__file__).resolve().parent.parent / "workers/requirements-diarizen.txt"
    with tempfile.TemporaryDirectory(prefix="diarizen_constraints_") as temporary:
        constraints = Path(temporary) / "constraints.txt"
        constraints.write_text(f"torch=={args.torch_version}\ntorchaudio=={args.torch_version}\n" +
                               requirements.read_text(), encoding="utf-8")
        subprocess.run([*install, "-r", str(requirements), "-c", str(constraints)], check=True)
        subprocess.run([*install, "-e", str(source), "-e", str(source / "pyannote-audio"),
                        "-c", str(constraints)], check=True)
    check = [uv, "pip", "check", "--python", str(python)] if uv else [str(python), "-m", "pip", "check"]
    subprocess.run(check, check=True)
    settings = {"python": str(python), "model_path": str(model), "embedding_path": str(embedding),
                "device": "cpu", "batch_size": 1}
    (root / "speaker_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    manifest = {"diarizen_source": DIARIZEN_SOURCE, "diarizen_model_revision": DIARIZEN_REVISION,
                "sha256": {str(p.relative_to(root)): digest(p) for p in [archive, embedding, *model.rglob("*")] if p.is_file()}}
    (root / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(settings, indent=2))


if __name__ == "__main__":
    main()
