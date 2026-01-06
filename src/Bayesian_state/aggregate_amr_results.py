"""Aggregate per-subject AMR results into a single JSON for ModelEval.

Usage:
    python -m src.Bayesian_state.aggregate_amr_results \
        --input-dir results/state-based-AMR-result \
        --output results/state-based-AMR-result/all_subjects.json

The input dir should contain per-subject JSON files produced by
``run_amr_optimization.py``.
The aggregated file is a dict keyed by subject_id with fields that
`model_evaluation.ModelEval` expects (condition, sliding_* arrays, etc.).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def aggregate(input_dir: Path) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for file in sorted(input_dir.glob("subject_*.json")):
        payload = load_json(file)
        sid = int(payload["subject_id"])
        metrics = payload.get("metrics", {})
        results[sid] = {
            "condition": payload.get("condition"),
            # Arrays for ModelEval.plot_accuracy_comparison
            "sliding_true_acc": metrics.get("sliding_true_acc"),
            "sliding_pred_acc": metrics.get("sliding_pred_acc"),
            "sliding_pred_acc_std": metrics.get("sliding_pred_acc_std"),
            # Store mean error and params for reference
            "mean_error": payload.get("mean_error"),
            "std_error": payload.get("std_error"),
            "best_params": payload.get("best_params"),
        }
    if not results:
        raise RuntimeError(f"No subject_*.json found in {input_dir}")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate per-subject AMR results")
    p.add_argument("--input-dir", type=Path, required=True, help="Directory containing subject_*.json")
    p.add_argument("--output", type=Path, required=True, help="Path to save aggregated JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    agg = aggregate(args.input_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    print(f"Aggregated {len(agg)} subjects -> {args.output}")


if __name__ == "__main__":
    main()
