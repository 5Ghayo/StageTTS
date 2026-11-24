"""120s Japanese dub with the highest-epoch speaker fine-tunes, plain path.

Same neutral references and weights as the Chinese plain run, but the text is
the original Japanese `orig_text` and `text_lang=ja`. No emotion routing, no
trimming, no atempo, no pronunciation rewriting.

Run on the WSL training host inside the 120s project workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
import os
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

API = "http://127.0.0.1:9880"
PROJECT = Path(os.environ["STAGETTS_EXPERIMENT_PROJECT"])
WEIGHTS = Path(os.environ["STAGETTS_EXPERIMENT_WEIGHTS"])
OUT = PROJECT / "ab-120s" / "high-epoch-plain-ja"


def prepare(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    value = "".join(ch for ch in value if ch.isprintable() or ch.isspace())
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"(?<=[\u4e00-\u9fff])([A-Za-z0-9]+)", r" \1", value)
    value = re.sub(r"([A-Za-z0-9]+)(?=[\u4e00-\u9fff])", r"\1 ", value)
    return value


def seed(*parts: object) -> int:
    digest = hashlib.sha1("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def set_weights(kind: str, path: Path) -> None:
    r = requests.get(f"{API}/set_{kind}_weights", params={"weights_path": str(path)}, timeout=180)
    r.raise_for_status()
    print(f"  set {kind}: {path.name}", flush=True)


def highest_gpt(speaker: str) -> Path:
    for epoch in (18, 15, 12, 10, 8, 6, 4, 3, 2, 1):
        candidate = WEIGHTS / f"{speaker}-e{epoch}.ckpt"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no GPT weight for {speaker}")


def highest_sovits(speaker: str) -> Path:
    candidates = sorted(WEIGHTS.glob(f"{speaker}_e*_s*.pth"))
    if not candidates:
        raise FileNotFoundError(f"no SoVITS weight for {speaker}")

    def epoch_of(path: Path) -> int:
        m = re.search(r"_e(\d+)_s", path.name)
        return int(m.group(1)) if m else -1

    return max(candidates, key=epoch_of)


def synthesize(text: str, ref_wav: Path, ref_text: str, target_text: str, seg_id: int, out: Path) -> None:
    payload = {
        "text": text,
        "text_lang": "ja",
        "ref_audio_path": str(ref_wav),
        "prompt_text": ref_text,
        "prompt_lang": "ja",
        "speed_factor": 1.0,
        "media_type": "wav",
        "text_split_method": "cut0",
        "fragment_interval": 0.0,
        "batch_size": 1,
        "seed": seed(seg_id, target_text, ref_text, ref_wav.name),
        "parallel_infer": True,
        "repetition_penalty": 1.35,
    }
    r = requests.post(f"{API}/tts", json=payload, timeout=300)
    if r.status_code != 200:
        raise RuntimeError(f"seg {seg_id} HTTP {r.status_code}: {r.text[:400]}")
    out.write_bytes(r.content)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="high_epoch_plain_ja")
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    script = json.loads((PROJECT / "script.json").read_text(encoding="utf-8"))
    duration = float(args.duration or script["media"].get("duration") or 120.0)
    OUT.mkdir(parents=True, exist_ok=True)

    chars = script["characters"]
    segments = sorted(script["segments"], key=lambda s: s["id"])
    errors: list[str] = []

    by_speaker: dict[str, list[dict]] = {}
    for seg in segments:
        by_speaker.setdefault(seg["speaker"], []).append(seg)

    for speaker, rows in by_speaker.items():
        char = chars[speaker]
        ref = char["emotion_refs"]["neutral"]
        ref_wav = Path(ref["wav"])
        if not ref_wav.is_file():
            raise FileNotFoundError(ref_wav)
        ref_text = prepare(ref["text"])

        gpt = highest_gpt(speaker)
        sovits = highest_sovits(speaker)
        print(f"[{speaker}] gpt={gpt.name} sovits={sovits.name} ref={ref_wav.name}", flush=True)
        set_weights("gpt", gpt)
        set_weights("sovits", sovits)

        for seg in rows:
            source_text = prepare(seg["orig_text"])
            out = OUT / f"seg_{seg['id']:04d}.wav"
            try:
                synthesize(source_text, ref_wav, ref_text, seg["orig_text"], seg["id"], out)
                print(f"seg {seg['id']:04d} {speaker} {len(source_text)} chars ok", flush=True)
            except Exception as exc:  # keep going, report at the end
                errors.append(f"seg {seg['id']}: {exc}")
                print(f"seg {seg['id']:04d} FAILED {exc}", flush=True)

    # Assemble at 32k mono using original start times; no trimming or tempo fit.
    sample_rate = 32000
    total = int(round(duration * sample_rate))
    track = np.zeros(total, dtype=np.float32)
    used = 0
    for seg in segments:
        wav_path = OUT / f"seg_{seg['id']:04d}.wav"
        if not wav_path.is_file():
            continue
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != sample_rate:
            raise RuntimeError(f"seg {seg['id']} sample rate {sr}, expected {sample_rate}")
        start = int(round(float(seg["start"]) * sample_rate))
        end = min(total, start + len(audio))
        if end > start:
            track[start:end] += audio[: end - start]
            used += 1

    peak = float(np.max(np.abs(track))) if track.size else 0.0
    if peak > 0.98:
        track *= 0.98 / peak

    raw = OUT / f"{args.tag}_raw_32k.wav"
    sf.write(raw, track, sample_rate, subtype="PCM_16")

    final = OUT / f"{args.tag}_120s.wav"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(raw),
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", str(final),
    ], check=True)
    mp3 = OUT / f"{args.tag}_120s.mp3"
    subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(final),
        "-c:a", "libmp3lame", "-b:a", "192k", str(mp3),
    ], check=True)

    if errors:
        (OUT / f"{args.tag}_errors.txt").write_text("\n".join(errors), encoding="utf-8")
    print(f"DONE: {mp3} ({mp3.stat().st_size} bytes), segments={used}, errors={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
