"""Use SaT to select sentence breaks within existing WhisperX dialogue windows."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from pathlib import Path

from workers.aligned_segments import normalize_segments
from workers.speech_segmentation import MAX_PLAYBACK_TAIL_SILENCE_SECONDS, attach_playback_ranges

ROOT = Path(__file__).resolve().parent.parent
SAT_THRESHOLD = 0.25
RECHECK_RADIUS_SECONDS = 1.0
RECHECK_MIN_PAUSE_SECONDS = 0.20


def resolve_dialogue_boundaries(result: dict, log: Callable[[str], None]) -> list[dict]:
    """Accept SaT cuts only at midpoints of pauses found by WhisperX VAD."""
    raw = result.get("segments") or []
    base = normalize_segments(raw, semantic_cuts=frozenset())
    chunks = result.get("vad_chunks") or []
    if not chunks:
        return base
    attach_playback_ranges(base, chunks, result["vad_duration"])
    regions = result.get("vad_recheck_regions") or []
    pauses = [(left, right) for (_, left), (right, _) in zip(regions, regions[1:])
              if right - left >= RECHECK_MIN_PAUSE_SECONDS]
    if not pauses:
        return base
    middles = [(left + right) / 2 for left, right in pauses]
    starts = [chunk["start"] for chunk in chunks]

    try:
        predictions = predict_boundaries([row["orig_text"] for row in base])
        return [
            dict(row, id=i) for i, row in enumerate(
                part for j, (row, probabilities) in enumerate(zip(base, predictions, strict=True))
                for part in _split_at_vad_pause(
                    row, probabilities, chunks, starts, pauses, middles,
                    base[j + 1]["start"] if j + 1 < len(base) else result["vad_duration"],
                )
            )
        ]
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        log(f"本地 SaT 复查不可用，保留 WhisperX 切分：{exc}")
        return base


def _split_at_vad_pause(row, probabilities, chunks, starts, pauses, middles, next_start):
    text = row["orig_text"]
    if len(probabilities) != len(text) or not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities):
        raise ValueError("SaT 返回的概率与原文位置不匹配")
    words = row["words"]
    tokens = ["".join(word["word"].split()) for word in words]
    if not words or any(not token for token in tokens) or "".join(tokens) != "".join(text.split()):
        return [row]
    window = bisect_right(starts, row["start"] + 0.002) - 1
    if window < 0 or row["end"] > chunks[window]["end"] + 0.002:
        return [row]
    speech = chunks[window]["segments"]
    start = row["start"]
    last_speech_end = min(speech[-1][1], next_start)
    characters = [i for i, char in enumerate(text) if not char.isspace()]
    digest = hashlib.sha256(text.encode()).hexdigest()
    positions = []
    cursor = 0
    for i, token in enumerate(tokens[:-1], 1):
        cursor += len(token)
        char_offset = characters[cursor]
        score = max(probabilities[characters[cursor - 1]:char_offset])
        left, right = words[i - 1]["end"], words[i]["start"]
        if score <= SAT_THRESHOLD or left is None or right is None:
            continue
        target = (left + right) / 2
        lo = bisect_left(middles, target - RECHECK_RADIUS_SECONDS)
        hi = bisect_right(middles, target + RECHECK_RADIUS_SECONDS)
        candidates = [k for k in range(lo, hi)
                      if start < pauses[k][0] and pauses[k][1] < last_speech_end]
        if not candidates:
            continue
        k = min(candidates, key=lambda index: abs(middles[index] - target))
        start = round(middles[k], 3)
        positions.append((i, char_offset, start, {
            "candidate_id": f"sat:{digest[:16]}:{char_offset}:{start:.3f}",
            "source_text_sha256": digest, "char_offset": char_offset, "word_index": i,
            "left_end": round(pauses[k][0], 3), "right_start": round(pauses[k][1], 3),
            "reasons": ["sat_sentence_boundary", "whisperx_vad_pause_midpoint"],
        }))
    if not positions:
        return [row]
    end = max(row["end"], round(last_speech_end, 3))
    points = [(0, 0, row["start"], None), *positions, (len(words), len(text), end, None)]
    output = []
    for (i, a, start, left), (j, b, stop, right) in zip(points, points[1:]):
        part, = normalize_segments(
            [{"start": start, "end": stop, "text": text[a:b].strip(), "words": words[i:j]}],
            semantic_cuts=frozenset(),
        )
        part["source_audio_start"] = left["right_start"] if left else row.get("source_audio_start", start)
        part["speech_end"] = right["left_end"] if right else row.get("speech_end", stop)
        part["source_audio_end"] = (
            min(stop, right["left_end"] + MAX_PLAYBACK_TAIL_SILENCE_SECONDS)
            if right else max(stop, row.get("source_audio_end", stop))
        )
        part["boundary_provenance"] = [boundary for boundary in (left, right) if boundary]
        output.append(part)
    return output


def predict_boundaries(texts: list[str]) -> list[list[float]]:
    """Run the official model in an isolated, local-only CPU process."""
    default_python = ROOT / ".venv-sat" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python = os.getenv("SAT_PYTHON") or str(default_python)
    assets = Path(os.getenv("SAT_ASSET_DIR") or ROOT / ".cache/sat").resolve()
    if not (assets / "model/model_optimized.onnx").is_file():
        raise FileNotFoundError("缺少 SaT 模型，请按 docs/diarizen_setup.md 安装")
    result = subprocess.run(
        [python, "-m", "workers.dialogue_segmentation", str(assets)],
        cwd=ROOT, input=json.dumps(texts, ensure_ascii=False), capture_output=True,
        encoding="utf-8", timeout=600,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
    )
    if result.returncode:
        last_line = result.stderr.strip().splitlines()[-1:] or ["SaT worker failed"]
        raise RuntimeError(last_line[0][:300])
    return json.loads(result.stdout)


def main() -> None:
    assets = Path(sys.argv[1])
    with contextlib.redirect_stdout(sys.stderr):
        import onnxruntime as ort
        from wtpsplit import SaT

        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        model = SaT(
            assets / "model", tokenizer_name_or_path=assets / "tokenizer",
            ort_providers=["CPUExecutionProvider"], ort_kwargs={"sess_options": options},
        )
        texts = json.load(sys.stdin)
        probabilities = [row.tolist() for row in model.predict_proba(texts, batch_size=16)]
    print(json.dumps(probabilities, allow_nan=False))


if __name__ == "__main__":
    main()
