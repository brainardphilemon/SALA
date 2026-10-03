#!/usr/bin/env python3
"""Run CPU-only LAYA hallucination detection over saved TriviaQA generations.

The LAYA model receives only the question and the generated answer.  Saved
BLEURT scores/references are joined afterward for evaluation, so the detector
does not see its ground-truth signal at inference time.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable


QUESTION = {
    "hallucination": {
        "type": "noul",
        "instructions": (
            "Is the assistant answer factually incorrect, unsupported, irrelevant, "
            "or otherwise hallucinated for the given trivia question?"
        ),
        "criteria": {
            "false": "The answer is factually correct and directly answers the question.",
            "true": "The answer is incorrect, unsupported, irrelevant, or hallucinated.",
        },
        # Neutral labels reduce the documented true/false label sensitivity.
        "labels": {"false": "B", "true": "A"},
    }
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--domain", default="triviaqa")
    parser.add_argument("--model", default="convaiinnovations/laya")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty result file")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def build_state(row: dict[str, Any]) -> dict[str, str]:
    return {
        "trivia_question": str(row["input_text"]),
        "assistant_answer": str(row["output_text"]),
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(r["ground_truth_hallucination_label"] == 1 and r["predicted_label"] == 1 for r in rows)
    tn = sum(r["ground_truth_hallucination_label"] == 0 and r["predicted_label"] == 0 for r in rows)
    fp = sum(r["ground_truth_hallucination_label"] == 0 and r["predicted_label"] == 1 for r in rows)
    fn = sum(r["ground_truth_hallucination_label"] == 1 and r["predicted_label"] == 0 for r in rows)
    total = len(rows)
    return {
        "samples": total,
        "accuracy": (tp + tn) / total if total else None,
        "precision_hallucinated": tp / (tp + fp) if tp + fp else None,
        "recall_hallucinated": tp / (tp + fn) if tp + fn else None,
        "f1_hallucinated": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    artifact_dir = Path(args.artifact_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_path = artifact_dir / "generation" / args.domain / "rows.jsonl"
    bleurt_path = artifact_dir / "bleurt" / args.domain / "scores.jsonl"
    if not generation_path.is_file() or not bleurt_path.is_file():
        raise FileNotFoundError(f"Missing generation or BLEURT file under {artifact_dir}")

    generated = read_jsonl(generation_path)
    scored = {str(row["id"]): row for row in read_jsonl(bleurt_path)}
    if args.max_samples:
        generated = generated[: args.max_samples]
    missing = [str(row["id"]) for row in generated if str(row["id"]) not in scored]
    if missing:
        raise ValueError(f"Missing BLEURT scores for {len(missing)} rows; first={missing[0]}")

    checkpoint_path = output_dir / "laya_triviaqa_rows.jsonl"
    completed_rows = read_jsonl(checkpoint_path) if checkpoint_path.exists() else []
    completed = {str(row["id"]): row for row in completed_rows}
    pending = [row for row in generated if str(row["id"]) not in completed]
    print(
        f"TriviaQA rows={len(generated)} completed={len(completed_rows)} pending={len(pending)}; "
        f"CPU-only LAYA threshold={args.threshold}",
        flush=True,
    )

    if pending:
        import laya

        load_kwargs: dict[str, Any] = {"device": "cpu"}
        if args.revision:
            load_kwargs["revision"] = args.revision
        agent = laya.load(args.model, **load_kwargs)

        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            states = [build_state(row) for row in batch]
            predictions = agent.predict_batch(
                states,
                QUESTION,
                batch_size=len(states),
                sort_by_length=False,
            )
            saved = []
            for source, prediction in zip(batch, predictions):
                row_id = str(source["id"])
                judged = scored[row_id]
                answer = prediction["answers"]["hallucination"]
                probability = float(answer["noul"])
                # BLEURT >= 0.5 means correct in the original run, hence not hallucinated.
                ground_truth_hallucination = int(float(judged["bleurt_score"]) < 0.5)
                saved.append({
                    "id": row_id,
                    "input_text": str(source["input_text"]),
                    "output_text": str(source["output_text"]),
                    "ground_truth_text": str(judged["ground_truth_text"]),
                    "bleurt_score": float(judged["bleurt_score"]),
                    "ground_truth_hallucination_label": ground_truth_hallucination,
                    "laya_confidence_score": probability,
                    "predicted_label": int(probability > args.threshold),
                    "predicted_label_text": "hallucinated" if probability > args.threshold else "not_hallucinated",
                    "threshold": args.threshold,
                    "laya_answer_confidence": float(answer.get("confidence", max(probability, 1.0 - probability))),
                    "laya_model": args.model,
                    "device": "cpu",
                })
            append_jsonl(checkpoint_path, saved)
            completed.update({str(row["id"]): row for row in saved})
            done = min(start + len(batch), len(pending))
            print(f"LAYA scored {done}/{len(pending)} pending rows ({len(completed)}/{len(generated)} total)", flush=True)

    ordered = [completed[str(row["id"])] for row in generated]
    write_csv(output_dir / "results.csv", ordered)
    summary = {
        "domain": args.domain,
        "model": args.model,
        "device": "cpu",
        "threshold_rule": f"hallucinated iff laya_confidence_score > {args.threshold}",
        "ground_truth_rule": "hallucinated iff BLEURT-20 score < 0.5",
        **metrics(ordered),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {output_dir / 'results.csv'}", flush=True)


if __name__ == "__main__":
    main()
