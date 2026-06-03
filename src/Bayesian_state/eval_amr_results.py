"""Aggregate per-subject AMR results and generate evaluation plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

from src.Bayesian_state.utils.model_evaluation import ModelEval


DEFAULT_EVAL_PREDICTION_MODE = "posterior_t_minus_1"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _subject_json_files(input_dir: Path) -> List[Path]:
    return sorted((input_dir / "subjects").glob("subject_*.json"))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_eval_metrics(payload: Dict[str, Any], eval_prediction_mode: str) -> Dict[str, Any]:
    metrics_by_mode = payload.get("metrics_by_mode")
    if not isinstance(metrics_by_mode, dict) or not metrics_by_mode:
        raise ValueError(
            f"subject_{payload.get('subject_id')} missing metrics_by_mode. "
            "Please regenerate results with schema v4."
        )
    if eval_prediction_mode not in metrics_by_mode:
        available = sorted(metrics_by_mode.keys())
        raise ValueError(
            f"subject_{payload.get('subject_id')} does not include eval mode '{eval_prediction_mode}'. "
            f"Available: {available}"
        )
    metrics = metrics_by_mode[eval_prediction_mode]
    if not isinstance(metrics, dict):
        raise ValueError(
            f"Invalid metrics_by_mode['{eval_prediction_mode}'] for subject_{payload.get('subject_id')}"
        )
    return metrics


def _strategy_to_best_step_amount(strategy_step: Dict[str, Any]) -> Dict[str, List[float]]:
    converted: Dict[str, List[float]] = {}
    for key, value in (strategy_step or {}).items():
        if key in {"active_total", "strategies"}:
            continue
        if not isinstance(value, (int, float)):
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


def aggregate(input_dir: Path, eval_prediction_mode: str) -> dict:
    results = {}
    for file in _subject_json_files(input_dir):
        payload = load_json(file)
        sid = int(payload["subject_id"])
        metrics = _resolve_eval_metrics(payload, eval_prediction_mode)
        step_results = _build_step_results(payload)
        results[sid] = {
            "condition": payload.get("condition"),
            "sliding_true_acc": metrics.get("sliding_true_acc"),
            "sliding_pred_acc": metrics.get("sliding_pred_acc"),
            "sliding_pred_acc_std": metrics.get("sliding_pred_acc_std"),
            "sliding_true_family_acc": metrics.get("sliding_true_family_acc"),
            "sliding_pred_family_acc": metrics.get("sliding_pred_family_acc"),
            "sliding_pred_family_acc_std": metrics.get("sliding_pred_family_acc_std"),
            "mean_error": metrics.get("mean_error", payload.get("best_error", payload.get("mean_error"))),
            "std_error": payload.get("refit_std_error", payload.get("std_error")),
            "best_params": payload.get("best_params"),
            "best_step_results": step_results,
            "step_results": step_results,
            "strategy_counts_log": payload.get("strategy_counts_log"),
            "posterior_log": payload.get("posterior_log"),
            "prior_log": payload.get("prior_log"),
            "beta_log": payload.get("beta_log"),
            "sample_errors": payload.get("sample_errors"),
            "selection_meta": payload.get("selection_meta"),
            "eval_prediction_mode": eval_prediction_mode,
            "available_prediction_modes": payload.get("available_prediction_modes", []),
        }
    if not results:
        raise RuntimeError(f"No subject_*.json found in {input_dir / 'subjects'}")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate AMR results and plot evaluation charts")
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing subjects/subject_*.json",
    )
    p.add_argument("--eval-prediction-mode", type=str, default=DEFAULT_EVAL_PREDICTION_MODE)
    p.add_argument("--aggregate-output", type=Path, default=None)
    p.add_argument("--plot-accuracy", type=Path, default=None)
    p.add_argument("--plot-cluster", type=Path, default=None)
    p.add_argument("--plot-strategy-amount-details", type=Path, default=None)
    p.add_argument(
        "--strategy-amount-window-size",
        type=int,
        default=1,
        help="Rolling window for detailed per-strategy amount plots. Defaults to raw per-trial counts.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    agg_out = args.aggregate_output or (input_dir / "all_subjects.json")
    plot_out = args.plot_accuracy or (input_dir / "accuracy.png")
    plot_cluster = args.plot_cluster or (input_dir / "cluster_amount.png")
    plot_strategy_amount_details = args.plot_strategy_amount_details or (
        input_dir / "strategy_amount_details.png"
    )

    aggregated = aggregate(input_dir, eval_prediction_mode=args.eval_prediction_mode)
    agg_out.parent.mkdir(parents=True, exist_ok=True)
    agg_out.write_text(json.dumps(aggregated, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Aggregated {len(aggregated)} subjects -> {agg_out}")

    me = ModelEval()
    me.plot_accuracy_comparison(aggregated, save_path=str(plot_out))
    print(f"Saved accuracy plot -> {plot_out}")

    has_steps = any(v.get("best_step_results") for v in aggregated.values())
    if not has_steps:
        raise RuntimeError("No best_step_results found; cannot generate cluster/oral plots.")

    me.plot_cluster_amount(aggregated, save_path=str(plot_cluster))
    print(f"Saved cluster amount plot -> {plot_cluster}")

    me.plot_strategy_amount_details(
        aggregated,
        window_size=int(args.strategy_amount_window_size),
        save_path=str(plot_strategy_amount_details),
        min_periods=1,
    )
    print(f"Saved detailed strategy amount plot -> {plot_strategy_amount_details}")


if __name__ == "__main__":
    main()
