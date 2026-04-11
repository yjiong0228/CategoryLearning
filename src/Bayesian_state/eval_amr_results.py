"""Aggregate per-subject AMR results and generate evaluation plots.

Usage example:

    conda activate cate_learn
    python -m src.Bayesian_state.eval_amr_results \
        --input-dir results/state-based-AMR-result/pmh/cond1 \
        --aggregate-output results/state-based-AMR-result/pmh/cond1_agg.json \
        --plot-accuracy results/state-based-AMR-result/pmh/cond1_accuracy.png

- Aggregates subject_*.json in the input directory (same format as run_amr_optimization outputs).
- Saves aggregated JSON.
- Runs ModelEval.plot_accuracy_comparison to produce accuracy figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

# Use non-interactive backend for batch plotting
matplotlib.use("Agg")

from src.Bayesian_state.utils.model_evaluation import ModelEval
from src.Bayesian_state.utils.oral_process import Oral_center_analysis
from src.Bayesian_state.utils.paths import TASK2_PROCESSED_PATH
import pandas as pd


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _strategy_to_best_step_amount(strategy_step: Dict[str, Any]) -> Dict[str, List[float]]:
    converted: Dict[str, List[float]] = {}
    for key, value in (strategy_step or {}).items():
        if key == "active_total":
            continue
        if key == "random":
            converted["random"] = [_to_float(value)]
        else:
            converted[f"{key}_posterior"] = [_to_float(value)]
    converted.setdefault("random", [0.0])
    return converted


def _build_step_results(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    existing = payload.get("best_step_results") or payload.get("step_results")
    if isinstance(existing, list) and existing:
        return existing

    posterior_log = payload.get("posterior_log") or []
    strategy_counts = payload.get("strategy_counts_log") or []
    if not isinstance(posterior_log, list) or not posterior_log:
        return []

    step_results: List[Dict[str, Any]] = []
    for idx, posterior in enumerate(posterior_log):
        if not isinstance(posterior, list):
            continue
        hypo_details = {
            int(hypo_idx): {"post_max": _to_float(prob)}
            for hypo_idx, prob in enumerate(posterior)
            if _to_float(prob) > 0
        }
        step_item: Dict[str, Any] = {"hypo_details": hypo_details}

        if idx < len(strategy_counts) and isinstance(strategy_counts[idx], dict):
            step_item["best_step_amount"] = _strategy_to_best_step_amount(strategy_counts[idx])

        if hypo_details:
            best_k = max(hypo_details.items(), key=lambda item: item[1]["post_max"])[0]
            step_item["best_k"] = int(best_k)

        step_results.append(step_item)
    return step_results


def aggregate(input_dir: Path) -> dict:
    """Aggregate subject_*.json files into ModelEval-friendly structure."""
    results = {}
    for file in sorted(input_dir.glob("subject_*.json")):
        payload = load_json(file)
        sid = int(payload["subject_id"])
        metrics = payload.get("best_metrics") or payload.get("metrics", {}) or {}
        step_results = _build_step_results(payload)
        results[sid] = {
            "condition": payload.get("condition"),
            "sliding_true_acc": metrics.get("sliding_true_acc"),
            "sliding_pred_acc": metrics.get("sliding_pred_acc"),
            "sliding_pred_acc_std": metrics.get("sliding_pred_acc_std"),
            "mean_error": payload.get("best_error", payload.get("mean_error")),
            "std_error": payload.get("refit_std_error", payload.get("std_error")),
            "best_params": payload.get("best_params"),
            "best_step_results": step_results,
            "step_results": step_results,
            "strategy_counts_log": payload.get("strategy_counts_log"),
            "posterior_log": payload.get("posterior_log"),
            "prior_log": payload.get("prior_log"),
            "sample_errors": payload.get("sample_errors"),
            "selection_meta": payload.get("selection_meta"),
        }
    if not results:
        raise RuntimeError(f"No subject_*.json found in {input_dir}")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate AMR results and plot evaluation charts")
    p.add_argument("--input-dir", type=Path, required=True, help="Directory containing subject_*.json")
    p.add_argument(
        "--aggregate-output",
        type=Path,
        default=None,
        help="Where to save aggregated JSON (default: <input-dir>/all_subjects.json)",
    )
    p.add_argument(
        "--plot-accuracy",
        type=Path,
        default=None,
        help="Where to save accuracy comparison plot (default: <input-dir>/accuracy.png)",
    )
    p.add_argument(
        "--plot-oral",
        type=Path,
        default=None,
        help="Optional: save oral vs model k comparison (requires best_step_results)",
    )
    p.add_argument(
        "--plot-cluster",
        type=Path,
        default=None,
        help="Optional: save cluster amount comparison (requires best_step_results)",
    )
    p.add_argument(
        "--oral-data",
        type=Path,
        default=TASK2_PROCESSED_PATH,
        help="Path to Task2 processed CSV with oral fields (for oral vs model plot)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Defaults based on input_dir
    agg_out = args.aggregate_output or (input_dir / "all_subjects.json")
    plot_out = args.plot_accuracy or (input_dir / "accuracy.png")
    plot_oral = args.plot_oral or (input_dir / "oral_vs_model.png")
    plot_cluster = args.plot_cluster or (input_dir / "cluster_amount.png")
    oral_data_path = args.oral_data

    # Aggregate per-subject results
    aggregated = aggregate(input_dir)
    agg_out.parent.mkdir(parents=True, exist_ok=True)
    agg_out.write_text(json.dumps(aggregated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Aggregated {len(aggregated)} subjects -> {agg_out}")

    # Plot accuracy comparison
    me = ModelEval()
    me.plot_accuracy_comparison(aggregated, save_path=str(plot_out))
    print(f"Saved accuracy plot -> {plot_out}")

    # Optional plots when step logs exist
    has_steps = any(v.get("best_step_results") for v in aggregated.values())
    if has_steps:
        if args.plot_cluster is not None or plot_cluster:
            me.plot_cluster_amount(aggregated, save_path=str(plot_cluster))
            print(f"Saved cluster amount plot -> {plot_cluster}")
        else:
            print("Cluster plot targ0et not provided; skipping.")
        # Oral comparison: build oral hits from processed CSV if available
        if oral_data_path and oral_data_path.exists():
            oral_df = pd.read_csv(oral_data_path)
            oral_hits = Oral_center_analysis().get_oral_hypo_hits(oral_df)
            me.plot_k_oral_comparison(aggregated, oral_hits, save_path=str(plot_oral))
            print(f"Saved oral vs model plot -> {plot_oral}")
        else:
            print(f"Oral data not found at {oral_data_path}; skipping oral vs model plot.")
    else:
        print("No best_step_results found; skipping oral/cluster plots.")


if __name__ == "__main__":
    main()
