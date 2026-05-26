"""Aggregate per-subject GRID results and generate evaluation plots."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import pandas as pd
import yaml

matplotlib.use("Agg")

from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.utils.model_evaluation import ModelEval
from src.Bayesian_state.utils.oral_model_alignment import Oral_center_analysis, Oral_region_analysis


ORAL_MODE_CHOICES = ("center", "region")
COMMON_REQUIRED_COLS = ("iSub", "condition", "choice")
CENTER_REQUIRED_COLS = ("oral_center",)
REGION_REQUIRED_COLS = ("oral_A", "oral_b")
DEFAULT_REGION_N_SAMPLES = 1000
DEFAULT_EVAL_PREDICTION_MODE = "posterior_t_minus_1"
TRIAL_DATA_COLS = ("feature1", "feature2", "feature3", "feature4", "category", "choice", "feedback")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _subject_json_files(input_dir: Path) -> List[Path]:
    return sorted((input_dir / "subjects").glob("subject_*.json"))


def _stream_ref_relative_to(
    ref: Dict[str, Any] | None,
    source_json_path: Path,
    target_base_dir: Path,
) -> Dict[str, Any] | None:
    if not isinstance(ref, dict) or "path" not in ref:
        return ref
    adjusted = dict(ref)
    abs_path = (source_json_path.parent / str(ref["path"])).resolve()
    adjusted["path"] = os.path.relpath(abs_path, target_base_dir.resolve())
    return adjusted


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


def _build_grid_errors(payload: Dict[str, Any]) -> Dict[Tuple[float, float], List[float]]:
    grid_errors: Dict[Tuple[float, float], List[float]] = {}

    raw_grid_errors = payload.get("grid_errors")
    if isinstance(raw_grid_errors, list) and raw_grid_errors:
        for item in raw_grid_errors:
            if not isinstance(item, dict):
                continue
            params = item.get("params", {}) or {}
            if "gamma" not in params or "w0" not in params:
                continue
            key = (_to_float(params.get("gamma"), float("nan")), _to_float(params.get("w0"), float("nan")))

            samples = item.get("errors")
            if isinstance(samples, list) and samples:
                grid_errors[key] = [_to_float(v, float("nan")) for v in samples]
            else:
                grid_errors.setdefault(key, []).append(_to_float(item.get("mean_error"), float("nan")))
        if grid_errors:
            return grid_errors

    for item in payload.get("grid_summary", []) or []:
        params = item.get("params", {}) or {}
        if "gamma" not in params or "w0" not in params:
            continue
        key = (_to_float(params.get("gamma"), float("nan")), _to_float(params.get("w0"), float("nan")))
        grid_errors.setdefault(key, []).append(_to_float(item.get("mean_error"), float("nan")))
    return grid_errors


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


def _build_subject_trials(trial_df: pd.DataFrame | None, subject_id: int, n_trials: int) -> Dict[str, List[Any]] | None:
    if trial_df is None:
        return None
    missing = [col for col in TRIAL_DATA_COLS if col not in trial_df.columns]
    if missing:
        return None
    subj_df = trial_df[trial_df["iSub"] == subject_id].reset_index(drop=True)
    if subj_df.empty:
        return None
    if n_trials:
        subj_df = subj_df.iloc[:n_trials]
    return {col: subj_df[col].tolist() for col in TRIAL_DATA_COLS}


def aggregate_grid_results(
    input_dir: Path,
    eval_prediction_mode: str,
    trial_df: pd.DataFrame | None = None,
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}

    for file in _subject_json_files(input_dir):
        payload = load_json(file)
        sid = int(payload["subject_id"])
        metrics = _resolve_eval_metrics(payload, eval_prediction_mode)
        step_results = _build_step_results(payload)
        mean_error = metrics.get("mean_error", payload.get("best_error", payload.get("mean_error")))
        std_error = payload.get("refit_std_error", payload.get("std_error"))
        true_acc = metrics.get("true_acc") or []
        sliding_pred_acc = metrics.get("sliding_pred_acc") or []
        n_trials = len(true_acc) if isinstance(true_acc, list) else 0
        n_sliding = len(sliding_pred_acc) if isinstance(sliding_pred_acc, list) else 0
        window_size = n_trials - n_sliding if n_trials and n_sliding else None
        subject_trials = _build_subject_trials(trial_df, sid, n_trials)

        results[sid] = {
            "condition": payload.get("condition"),
            "sliding_true_acc": metrics.get("sliding_true_acc"),
            "sliding_pred_acc": metrics.get("sliding_pred_acc"),
            "sliding_pred_acc_std": metrics.get("sliding_pred_acc_std"),
            "sliding_true_family_acc": metrics.get("sliding_true_family_acc"),
            "sliding_pred_family_acc": metrics.get("sliding_pred_family_acc"),
            "sliding_pred_family_acc_std": metrics.get("sliding_pred_family_acc_std"),
            "n_trials": n_trials,
            "window_size": window_size,
            "mean_error": mean_error,
            "std_error": std_error,
            "best_params": payload.get("best_params"),
            "best_step_results": step_results,
            "step_results": step_results,
            "strategy_counts_log": payload.get("strategy_counts_log"),
            "posterior_log": payload.get("posterior_log"),
            "prior_log": payload.get("prior_log"),
            "beta_log": payload.get("beta_log"),
            "raw_runs_ref": _stream_ref_relative_to(payload.get("raw_runs_ref"), file, input_dir),
            "sample_errors": payload.get("sample_errors"),
            "selection_meta": payload.get("selection_meta"),
            "eval_prediction_mode": eval_prediction_mode,
            "available_prediction_modes": payload.get("available_prediction_modes", []),
            "grid_errors": _build_grid_errors(payload),
            "grid_summary": payload.get("grid_summary", []),
            "subject_trials": subject_trials,
            "subject_json_path": str(file),
        }

    if not results:
        raise RuntimeError(f"No subject_*.json found in {input_dir / 'subjects'}")
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate GRID results and plot evaluation charts")
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing subjects/subject_*.json",
    )
    p.add_argument("--eval-prediction-mode", type=str, default=DEFAULT_EVAL_PREDICTION_MODE)
    p.add_argument("--config", type=Path, default=None, help="Optional optimization YAML to resolve oral config defaults")
    p.add_argument("--aggregate-output", type=Path, default=None)
    p.add_argument("--plots-dir", type=Path, default=None)
    p.add_argument("--plot-accuracy", type=Path, default=None)
    p.add_argument("--plot-grid", type=Path, default=None)
    p.add_argument("--plot-posterior", type=Path, default=None)
    p.add_argument("--plot-cluster", type=Path, default=None)
    p.add_argument("--plot-beta", type=Path, default=None)
    p.add_argument("--plot-accuracy-family", type=Path, default=None)
    p.add_argument("--trajectory-dir", type=Path, default=None)
    p.add_argument("--trajectory-posterior-dir", type=Path, default=None)
    p.add_argument("--plot-oral", type=Path, default=None)
    p.add_argument("--plot-oral-alignment", type=Path, default=None)
    p.add_argument("--plot-choice-conditioned-oral", type=Path, default=None)
    p.add_argument("--plot-active-topn-capture", type=Path, default=None)
    p.add_argument("--plot-active-topn-subjectwise", type=Path, default=None)
    p.add_argument("--oral-mode", type=str, choices=ORAL_MODE_CHOICES, default=None)
    p.add_argument("--oral-data", type=Path, default=None)
    p.add_argument("--oral-region-n-samples", type=int, default=None)
    return p.parse_args()


def _has_grid(aggregated: Dict[int, Dict[str, Any]]) -> bool:
    return any(v.get("grid_errors") for v in aggregated.values())


def _has_steps(aggregated: Dict[int, Dict[str, Any]]) -> bool:
    return any(v.get("best_step_results") for v in aggregated.values())


def _serialize_grid_errors(grid_errors: Dict[Tuple[float, float], List[float]]) -> Dict[str, List[float]]:
    return {f"gamma={g},w0={w0}": errs for (g, w0), errs in grid_errors.items()}


def _resolve_plot_window_size(results: Dict[int, Dict[str, Any]], default: int = 16) -> int:
    values = []
    for info in results.values():
        try:
            value = int(info.get("window_size"))
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.append(value)
    if not values:
        return int(default)
    values = sorted(values)
    return int(values[len(values) // 2])


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must parse to a dict: {path}")
    return loaded


def _resolve_oral_settings(args: argparse.Namespace) -> Tuple[str, Path, int]:
    config_mode = None
    config_data_path = None
    config_region_n_samples = None

    if args.config is not None:
        config_path = args.config.resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = _load_yaml(config_path)
        dataset_paths = resolve_dataset_paths(config, config_path.parent)
        config_data_path = dataset_paths["learning_data"]
        oral_cfg = config.get("oral")
        if isinstance(oral_cfg, dict):
            raw_mode = oral_cfg.get("mode")
            if isinstance(raw_mode, str):
                raw_mode = raw_mode.strip().lower()
                if raw_mode in ORAL_MODE_CHOICES:
                    config_mode = raw_mode
                else:
                    raise ValueError(f"Invalid oral.mode '{raw_mode}' in {config_path}.")
            raw_region_n_samples = oral_cfg.get("region_n_samples")
            if raw_region_n_samples is not None:
                config_region_n_samples = int(raw_region_n_samples)

    final_mode = args.oral_mode or config_mode
    if final_mode is None:
        raise ValueError("Oral mode is required. Provide --oral-mode or set oral.mode in --config YAML.")

    final_data_path = args.oral_data or config_data_path
    if final_data_path is None:
        raise ValueError(
            f"Oral data path is required for mode='{final_mode}'. "
            "Provide --oral-data or set dataset.learning_data in --config YAML."
        )

    region_n_samples = args.oral_region_n_samples
    if region_n_samples is None:
        region_n_samples = config_region_n_samples
    if region_n_samples is None:
        region_n_samples = DEFAULT_REGION_N_SAMPLES
    if int(region_n_samples) <= 0:
        raise ValueError(f"oral region n_samples must be > 0, got {region_n_samples}")

    return final_mode, final_data_path.resolve(), int(region_n_samples)


def _build_oral_hits(
    mode: str,
    oral_df: pd.DataFrame,
    oral_data_path: Path,
    region_n_samples: int,
) -> Dict[int, Dict[str, Any]]:
    if mode == "center":
        required_cols = COMMON_REQUIRED_COLS + CENTER_REQUIRED_COLS
        missing = [col for col in required_cols if col not in oral_df.columns]
        if missing:
            raise ValueError(f"Oral center evaluation failed for {oral_data_path}: missing columns {missing}.")
        oral_hits = Oral_center_analysis().get_oral_hypo_hits(oral_df)
    else:
        required_cols = COMMON_REQUIRED_COLS + REGION_REQUIRED_COLS
        missing = [col for col in required_cols if col not in oral_df.columns]
        if missing:
            raise ValueError(f"Oral region evaluation failed for {oral_data_path}: missing columns {missing}.")
        oral_hits = Oral_region_analysis().get_oral_hypo_hits(oral_df, n_samples=region_n_samples)

    if not oral_hits:
        raise RuntimeError(f"Oral {mode} evaluation produced no subject-level hits for {oral_data_path}.")
    return oral_hits


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
    plot_beta = args.plot_beta or (plots_dir / "beta.png")
    plot_accuracy_family = args.plot_accuracy_family or (plots_dir / "accuracy_family.png")
    trajectory_dir = args.trajectory_dir or (plots_dir / "trajectory_accuracy")
    trajectory_posterior_dir = args.trajectory_posterior_dir or (plots_dir / "trajectory_posterior")
    plot_oral = args.plot_oral or (plots_dir / "oral_vs_model.png")
    plot_oral_alignment = args.plot_oral_alignment or (plots_dir / "oral_model_alignment.png")
    plot_choice_conditioned_oral = args.plot_choice_conditioned_oral or (plots_dir / "oral_choice_conditioned_alignment.png")
    plot_active_topn = args.plot_active_topn_capture or (plots_dir / "oral_active_topn_capture.png")
    plot_active_topn_subjectwise = (
        args.plot_active_topn_subjectwise or (plots_dir / "oral_active_topn_capture_subjectwise.png")
    )
    oral_mode, oral_data_path, oral_region_n_samples = _resolve_oral_settings(args)

    if not oral_data_path.is_file():
        raise FileNotFoundError(f"Oral data file not found: {oral_data_path}")

    oral_df = pd.read_csv(oral_data_path)
    aggregated = aggregate_grid_results(input_dir, eval_prediction_mode=args.eval_prediction_mode, trial_df=oral_df)

    me = ModelEval()

    me.plot_accuracy_comparison(aggregated, save_path=str(plot_accuracy))
    print(f"Saved accuracy plot -> {plot_accuracy}")

    me.plot_accuracy_family_comparison(aggregated, save_path=str(plot_accuracy_family))
    print(f"Saved family accuracy plot -> {plot_accuracy_family}")

    me.plot_beta_dynamics(aggregated, save_path=str(plot_beta))
    print(f"Saved beta dynamics plot -> {plot_beta}")

    trajectory_summary = me.plot_trajectory_analysis(
        input_dir,
        trajectory_dir,
        ranks=ModelEval.DEFAULT_TRAJECTORY_RANKS,
        n_cols=4,
        eval_prediction_mode=args.eval_prediction_mode,
    )
    if trajectory_summary.empty:
        print("No run-level trajectories found; skipping trajectory accuracy plots.")
    else:
        print(f"Saved trajectory accuracy plots -> {trajectory_dir}")

    posterior_trajectory_summary = me.plot_trajectory_posteriors(
        input_dir,
        trajectory_posterior_dir,
        ranks=ModelEval.DEFAULT_TOP16_RANKS,
        n_cols=4,
    )
    if posterior_trajectory_summary.empty:
        print("No run-level posterior trajectories found; skipping trajectory posterior plots.")
    else:
        print(f"Saved trajectory posterior plots -> {trajectory_posterior_dir}")

    if _has_grid(aggregated):
        me.plot_error_grids(aggregated, fname=["gamma", "w0"], save_path=str(plot_grid))
        print(f"Saved error grid plot -> {plot_grid}")
    else:
        print("No grid data found; skipping error grid plot.")

    if not _has_steps(aggregated):
        raise RuntimeError("No step-level logs found in aggregated results; cannot generate posterior/cluster/oral plots.")

    me.plot_posterior_probabilities(aggregated, save_path=str(plot_posterior))
    print(f"Saved posterior plot -> {plot_posterior}")

    me.plot_cluster_amount(aggregated, save_path=str(plot_cluster))
    print(f"Saved cluster dynamics plot -> {plot_cluster}")

    oral_hits = _build_oral_hits(oral_mode, oral_df, oral_data_path, oral_region_n_samples)
    me.plot_k_oral_comparison(aggregated, oral_hits, save_path=str(plot_oral))
    print(f"Saved oral vs model plot -> {plot_oral}")

    oral_alignment = me.compute_oral_model_alignment(
        aggregated,
        oral_df,
        oral_mode=oral_mode,
        region_n_samples=oral_region_n_samples,
    )
    me.plot_oral_model_alignment(oral_alignment, save_path=str(plot_oral_alignment))
    print(f"Saved oral-model alignment plot -> {plot_oral_alignment}")

    choice_conditioned_alignment = me.compute_choice_conditioned_oral_alignment(
        aggregated,
        oral_df,
        oral_mode=oral_mode,
        region_n_samples=oral_region_n_samples,
    )
    me.plot_choice_conditioned_oral_alignment(
        choice_conditioned_alignment,
        save_path=str(plot_choice_conditioned_oral),
    )
    print(f"Saved choice-conditioned oral alignment plot -> {plot_choice_conditioned_oral}")

    active_topn_capture = me.compute_oral_active_topn_capture(
        aggregated,
        oral_df,
        oral_mode=oral_mode,
        region_n_samples=oral_region_n_samples,
    )
    active_topn_outputs = me.save_oral_active_topn_capture_outputs(
        active_topn_capture,
        plots_dir,
        group_plot_path=str(plot_active_topn),
        subjectwise_plot_path=str(plot_active_topn_subjectwise),
        window_size=_resolve_plot_window_size(aggregated),
    )
    print(f"Saved oral active top-N capture plot -> {active_topn_outputs['group_plot']}")
    print(f"Saved oral active top-N subject-wise plot -> {active_topn_outputs['subjectwise_plot']}")

    aggregated_serializable = {
        sid: {
            **info,
            "grid_errors": _serialize_grid_errors(info.get("grid_errors", {})),
            "oral_model_alignment": oral_alignment.get(sid),
            "choice_conditioned_oral_alignment": choice_conditioned_alignment.get(sid),
        }
        for sid, info in aggregated.items()
    }

    agg_out.parent.mkdir(parents=True, exist_ok=True)
    agg_out.write_text(json.dumps(aggregated_serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Aggregated {len(aggregated)} subjects -> {agg_out}")


if __name__ == "__main__":
    main()
