"""120s Japanese dub with the 15-group recluster weights, plain path.

Uses the `liz-ja-d027-15voices-120s` project and the 9 clusters that actually
have lines in it. Text is the original Japanese `orig_text`, text_lang=ja.
Only neutral references; no emotion routing, trimming, atempo, or rewriting.

Run on the WSL training host inside the 15-voice project workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

API = "http://127.0.0.1:9880"
PROJECT = Path("/home/gage/projects/StageTTS-41860f0.TR7af4/workspace/liz-ja-d027-15voices-120s")
MODELS_JSON = Path("/home/gage/projects/StageTTS-f47a517/.recluster027-train/latest-models.json")
OUT = PROJECT / "ab-120s" / "high-epoch-plain-zh-15"


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


def synthesize(text: str, ref_wav: Path, ref_text: str, target_text: str, seg_id: int, out: Path) -> None:
    payload = {
        "text": text,
        "text_lang": "zh",
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
    parser.add_argument("--tag", default="high_epoch_plain_zh_15")
    parser.add_argument("--duration", type=float, default=None)
    args = parser.parse_args()

    script = json.loads((PROJECT / "script.json").read_text(encoding="utf-8"))
    models = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
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

        model = models.get(speaker)
        if not model:
            raise KeyError(f"no 15-group model entry for {speaker}")
        gpt = Path(model["gpt_weights"])
        sovits = Path(model["sovits_weights"])
        if not gpt.is_file():
            raise FileNotFoundError(gpt)
        if not sovits.is_file():
            raise FileNotFoundError(sovits)

        print(f"[{speaker}] gpt={gpt.name} sovits={sovits.name} ref={ref_wav.name}", flush=True)
        set_weights("gpt", gpt)
        set_weights("sovits", sovits)

        for seg in rows:
            source_text = prepare(seg["target_text"])
            out = OUT / f"seg_{seg['id']:04d}.wav"
            try:
                synthesize(source_text, ref_wav, ref_text, seg["target_text"], seg["id"], out)
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
