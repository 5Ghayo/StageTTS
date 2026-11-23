"""Compare original/dialogue SER evidence in isolated workers; gold labels optional.

python experiments/eval_emotion.py --request request.json --output report.json
    --worker-python F:/env_emotion/Scripts/python.exe

Request is the emotion_worker request with `sources` instead of `source`, e.g.
[{"id":"movie-original","kind":"original","audio_path":"..."},
 {"id":"movie-dialogue","kind":"dialogue","audio_path":"..."}].
Optional gold: {"labels": {"span-id": "exact model label"}}. No gold => no accuracy.
The runner never modifies script.json, installs packages, or downloads weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "workers" / "emotion_worker.py"


def launch_worker(request: dict, python: str, timeout: float) -> dict:
    completed = subprocess.run(
        [python, "-I", "-B", str(WORKER)], input=json.dumps(request, ensure_ascii=False),
        capture_output=True, text=True, encoding="utf-8", timeout=timeout,
        cwd=str(ROOT), check=False,
    )
    try:
        result = json.loads(completed.stdout)
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"worker_invalid_json (exit {completed.returncode}): {completed.stderr[-2000:]}") from exc
    if not isinstance(result, dict) or result.get("version") != "1.0" or result.get("stage") != "emotion":
        raise ValueError("worker_result_contract_mismatch")
    result["process"] = {"returncode": completed.returncode, "stderr_tail": completed.stderr[-2000:]}
    return result


def validate_result(result: dict, request: dict) -> dict:
    """Bind every returned observation to the exact requested source and span."""
    if not isinstance(result, dict) or result.get("version") != "1.0" or result.get("stage") != "emotion":
        raise ValueError("worker_result_contract_mismatch")
    if result.get("status") not in {"ok", "partial", "not_run", "not_configured", "blocked", "error"}:
        raise ValueError("worker_status_invalid")
    source = result.get("source") or {}
    for key in ("id", "kind", "audio_path"):
        if source.get(key) != request["source"].get(key):
            raise ValueError("worker_source_mismatch")
    model = result.get("model") or {}
    for key in ("id", "revision"):
        if model.get(key) != request["model"].get(key):
            raise ValueError("worker_model_mismatch")
    allowed = {span["id"]: span for span in request["spans"]}
    items = result.get("annotations")
    if not isinstance(items, list):
        raise ValueError("worker_annotations_must_be_list")
    seen = set()
    for item in items:
        if not isinstance(item, dict) or item.get("id") not in allowed or item["id"] in seen:
            raise ValueError("unexpected_or_duplicate_annotation_id")
        seen.add(item["id"])
        span = allowed[item["id"]]
        if item.get("status") not in {"ok", "partial", "not_run", "not_configured", "blocked", "error"}:
            raise ValueError("annotation_status_invalid")
        labels, scores = item.get("labels", []), item.get("scores", [])
        if not isinstance(labels, list) or not isinstance(scores, list) or len(labels) != len(scores):
            raise ValueError("annotation_label_score_mismatch")
        if not all(isinstance(label, str) for label in labels) or any(
            isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) for score in scores
        ):
            raise ValueError("annotation_label_score_invalid")
        if item["status"] in {"ok", "partial"}:
            if not labels or item.get("selected_label") != labels[max(range(len(scores)), key=scores.__getitem__)]:
                raise ValueError("annotation_selected_label_invalid")
        elif labels or scores or item.get("selected_label") is not None:
            raise ValueError("unrun_annotation_contains_prediction")
        for key in ("start", "end"):
            value = item.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value - span[key]) > 1e-6:
                raise ValueError("worker_annotation_range_mismatch")
        for key in ("source_token_ids", "source_turn_ids"):
            if item.get(key, []) != span.get(key, []):
                raise ValueError("worker_annotation_source_links_mismatch")
    result["requested_span_ids"] = list(allowed)
    result["missing_span_ids"] = sorted(set(allowed) - seen)
    return result


def summarize(result: dict, gold: dict | None = None) -> dict:
    items = result.get("annotations") or []
    successful = [x for x in items if x.get("status") == "ok" and x.get("selected_label") is not None]
    summary = {
        "source_id": result.get("source", {}).get("id"),
        "source_kind": result.get("source", {}).get("kind"),
        "status": result.get("status"),
        "requested_spans": len(result.get("requested_span_ids", [x["id"] for x in items])),
        "missing_span_ids": list(result.get("missing_span_ids") or []),
        "classified_spans": len(successful),
        "label_counts": dict(Counter(x["selected_label"] for x in successful)),
        "accuracy": None, "macro_f1": None, "evaluated_spans": 0,
        "evaluation_status": "not_evaluated_no_gold",
    }
    if gold is None:
        return summary
    labels = gold.get("labels")
    if not isinstance(labels, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in labels.items()):
        raise ValueError("gold_requires_exact_label_map")
    pairs = [(labels[x["id"]], x["selected_label"]) for x in successful if x["id"] in labels]
    if not pairs:
        summary["evaluation_status"] = "not_evaluated_no_matching_classified_spans"
        return summary
    classes = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    f1s = []
    for label in classes:
        tp = sum(a == label and b == label for a, b in pairs)
        fp = sum(a != label and b == label for a, b in pairs)
        fn = sum(a == label and b != label for a, b in pairs)
        f1s.append(2 * tp / (2 * tp + fp + fn))
    summary.update(
        evaluation_status="evaluated_on_provided_gold_subset",
        evaluated_spans=len(pairs), accuracy=sum(a == b for a, b in pairs) / len(pairs),
        macro_f1=sum(f1s) / len(f1s), gold_spans=len(labels),
        unevaluated_gold_spans=sorted(set(labels) - {x["id"] for x in successful}),
        metric_note="Exact provided model-label strings; no label remapping or in-domain accuracy guarantee.",
    )
    return summary


def compare_request(request: dict, *, worker_python: str, timeout: float = 600,
                    gold: dict | None = None, launcher: Callable | None = None) -> dict:
    sources = request.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources_required")
    if len({x.get("id") for x in sources}) != len(sources):
        raise ValueError("duplicate_source_ids")
    results = []
    for source in sources:
        payload = {k: v for k, v in request.items() if k not in {"sources", "source"}}
        payload["source"] = source
        # Serial subprocesses release one model before starting the next source.
        try:
            result = (launcher or launch_worker)(payload, worker_python, timeout)
            result = validate_result(result, payload)
        except Exception as exc:
            result = {"version": "1.0", "stage": "emotion", "status": "error", "source": source,
                      "model": request.get("model", {}),
                      "requested_span_ids": [span["id"] for span in request["spans"]],
                      "missing_span_ids": [span["id"] for span in request["spans"]],
                      "annotations": [], "review_reasons": [f"worker_failed: {exc}"]}
        results.append(result)
    return {
        "version": "1.0", "stage": "emotion_comparison",
        "evaluation_note": "Mix/dialogue agreement is a diagnostic, not ground truth or calibrated reliability.",
        "summaries": [summarize(result, gold) for result in results],
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--worker-python", default=sys.executable)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args(argv)
    request = json.loads(args.request.read_text(encoding="utf-8"))
    gold = json.loads(args.gold.read_text(encoding="utf-8")) if args.gold else None
    if args.output.resolve() == args.request.resolve() or (args.gold and args.output.resolve() == args.gold.resolve()):
        parser.error("output must not overwrite input request or gold")
    result = compare_request(request, worker_python=args.worker_python, timeout=args.timeout, gold=gold)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return 1 if any(x["status"] == "error" for x in result["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
