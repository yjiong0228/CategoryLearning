"""Aggregate per-subject GRID results and generate evaluation plots.

Usage example:

    conda activate cate_learn
    python -m src.Bayesian_state.eval_grid_results \
        --input-dir results/state-based-grid-result/pmh/cond3

This script reads subject_*.json produced by run_grid_optimization.py,
adapts them to ModelEval's expected schema, saves an aggregated JSON,
and outputs plots (accuracy, grid, posterior, cluster dynamics, oral alignment)
when required fields are available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import pandas as pd

# Use non-interactive backend for batch plotting
matplotlib.use("Agg")

from src.Bayesian_state.utils.model_evaluation import ModelEval
from src.Bayesian_state.utils.oral_process import Oral_center_analysis
from src.Bayesian_state.utils.paths import TASK2_PROCESSED_PATH


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_grid_errors(payload: Dict[str, Any]) -> Dict[Tuple[float, float], List[float]]:
    """Adapt run_grid_optimization's grid_summary -> ModelEval grid_errors format."""
    grid_errors: Dict[Tuple[float, float], List[float]] = {}
    for item in payload.get("grid_summary", []) or []:
        params = item.get("params", {}) or {}
        if "gamma" not in params or "w0" not in params:
            continue
        key = (_to_float(params.get("gamma"), float("nan")), _to_float(params.get("w0"), float("nan")))
        grid_errors.setdefault(key, []).append(_to_float(item.get("mean_error"), float("nan")))
    return grid_errors


def _strategy_to_best_step_amount(strategy_step: Dict[str, Any]) -> Dict[str, List[float]]:
    """Convert strategy_counts_log format into best_step_amount-like format.

    plot_cluster_amount expects values as lists and relies on key names containing
    'posterior' plus a 'random' channel.
    """
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
    """Build/normalize step_results for ModelEval.

    Priority:
    1) best_step_results directly from payload
    2) fallback from posterior_log (+ optional strategy_counts_log)
    """
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


def aggregate_grid_results(input_dir: Path) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}

    for file in sorted(input_dir.glob("subject_*.json")):
        payload = load_json(file)
        sid = int(payload["subject_id"])
        metrics = payload.get("metrics", {}) or {}
        step_results = _build_step_results(payload)

        results[sid] = {
            "condition": payload.get("condition"),
            "sliding_true_acc": metrics.get("sliding_true_acc"),
            "sliding_pred_acc": metrics.get("sliding_pred_acc"),
            "sliding_pred_acc_std": metrics.get("sliding_pred_acc_std"),
            "mean_error": payload.get("mean_error"),
            "std_error": payload.get("std_error"),
            "best_params": payload.get("best_params"),
            "best_step_results": step_results,
            "step_results": step_results,
            "strategy_counts_log": payload.get("strategy_counts_log"),
            "posterior_log": payload.get("posterior_log"),
            "prior_log": payload.get("prior_log"),
            "grid_errors": _build_grid_errors(payload),
            "grid_summary": payload.get("grid_summary", []),
        }

    if not results:
        raise RuntimeError(f"No subject_*.json found in {input_dir}")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate GRID results and plot evaluation charts")
    p.add_argument("--input-dir", type=Path, required=True, help="Directory containing subject_*.json")
    p.add_argument(
        "--aggregate-output",
        type=Path,
        default=None,
        help="Where to save aggregated JSON (default: <input-dir>/all_subjects.json)",
    )
    p.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory to save generated plots (default: <input-dir>/plots)",
    )
    p.add_argument("--plot-accuracy", type=Path, default=None, help="Accuracy plot path")
    p.add_argument("--plot-grid", type=Path, default=None, help="Error grid plot path")
    p.add_argument("--plot-posterior", type=Path, default=None, help="Posterior plot path")
    p.add_argument("--plot-cluster", type=Path, default=None, help="Cluster dynamics plot path")
    p.add_argument("--plot-oral", type=Path, default=None, help="Oral-vs-model plot path")
    p.add_argument(
        "--oral-data",
        type=Path,
        default=TASK2_PROCESSED_PATH,
        help="Path to Task2 processed CSV with oral fields (for oral vs model plot)",
    )
    return p.parse_args()


def _has_grid(aggregated: Dict[int, Dict[str, Any]]) -> bool:
    return any(v.get("grid_errors") for v in aggregated.values())


def _has_steps(aggregated: Dict[int, Dict[str, Any]]) -> bool:
    return any(v.get("best_step_results") for v in aggregated.values())


def _serialize_grid_errors(grid_errors: Dict[Tuple[float, float], List[float]]) -> Dict[str, List[float]]:
    """Make grid_errors JSON friendly by stringifying tuple keys."""
    return {f"gamma={g},w0={w0}": errs for (g, w0), errs in grid_errors.items()}


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    agg_out = args.aggregate_output or (input_dir / "all_subjects.json")
    plots_dir = args.plots_dir or (input_dir / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy = args.plot_accuracy or (plots_dir / "accuracy.png")
    plot_grid = args.plot_grid or (plots_dir / "error_grid.png")
    plot_posterior = args.plot_posterior or (plots_dir / "posterior.png")
    plot_cluster = args.plot_cluster or (plots_dir / "cluster_amount.png")
    plot_oral = args.plot_oral or (plots_dir / "oral_vs_model.png")
    oral_data_path = args.oral_data

    aggregated = aggregate_grid_results(input_dir)
    aggregated_serializable = {
        sid: {**info, "grid_errors": _serialize_grid_errors(info.get("grid_errors", {}))}
        for sid, info in aggregated.items()
    }

    agg_out.parent.mkdir(parents=True, exist_ok=True)
    agg_out.write_text(json.dumps(aggregated_serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Aggregated {len(aggregated)} subjects -> {agg_out}")

    me = ModelEval()

    me.plot_accuracy_comparison(aggregated, save_path=str(plot_accuracy))
    print(f"Saved accuracy plot -> {plot_accuracy}")

    if _has_grid(aggregated):
        me.plot_error_grids(aggregated, fname=["gamma", "w0"], save_path=str(plot_grid))
        print(f"Saved error grid plot -> {plot_grid}")
    else:
        print("No grid data found; skipping error grid plot.")

    if _has_steps(aggregated):
        me.plot_posterior_probabilities(aggregated, save_path=str(plot_posterior))
        print(f"Saved posterior plot -> {plot_posterior}")

        me.plot_cluster_amount(aggregated, save_path=str(plot_cluster))
        print(f"Saved cluster dynamics plot -> {plot_cluster}")

        if oral_data_path and oral_data_path.exists():
            oral_df = pd.read_csv(oral_data_path)
            oral_hits = Oral_center_analysis().get_oral_hypo_hits(oral_df)
            me.plot_k_oral_comparison(aggregated, oral_hits, save_path=str(plot_oral))
            print(f"Saved oral vs model plot -> {plot_oral}")
        else:
            print(f"Oral data not found at {oral_data_path}; skipping oral plot.")
    else:
        print("No step-level logs found; skipping posterior/cluster/oral plots.")


if __name__ == "__main__":
    main()
