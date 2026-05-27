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


ORAL_MODE_CHOICES = ("center", "region")
MODEL_DISTRIBUTION_CHOICES = ("posterior", "prior")
ORAL_BASED_MODEL_STATE_CHOICES = ("choice_conditioned_prior", "prior", "posterior")
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
    p.add_argument("--plot-oral-mass", type=Path, default=None)
    p.add_argument("--plot-distribution-alignment-group", type=Path, default=None)
    p.add_argument("--plot-distribution-alignment-subject", type=Path, default=None)
    p.add_argument("--plot-oral-based-alignment-group", type=Path, default=None)
    p.add_argument("--plot-oral-based-alignment-subject", type=Path, default=None)
    p.add_argument("--plot-target-based-alignment-group", type=Path, default=None)
    p.add_argument("--plot-target-based-alignment-subject", type=Path, default=None)
    p.add_argument("--plot-hit-based-alignment-group", type=Path, default=None)
    p.add_argument("--plot-hit-based-alignment-subject", type=Path, default=None)
    p.add_argument("--plot-coverage-based-alignment-group", type=Path, default=None)
    p.add_argument("--plot-coverage-based-alignment-subject", type=Path, default=None)
    p.add_argument("--oral-mode", type=str, choices=ORAL_MODE_CHOICES, default=None)
    p.add_argument(
        "--distribution-alignment-model-state",
        type=str,
        choices=MODEL_DISTRIBUTION_CHOICES,
        default="prior",
        help=(
            "Model distribution state used in distribution_alignment_group/subject plots. "
            "Defaults to prior_t to match the pre-feedback timing of oral reports."
        ),
    )
    p.add_argument(
        "--oral-based-alignment-model-state",
        type=str,
        choices=ORAL_BASED_MODEL_STATE_CHOICES,
        default="choice_conditioned_prior",
        help=(
            "Model belief state projected into oral center/region space. "
            "Defaults to choice-conditioned prior_t, matching oral-report timing."
        ),
    )
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


def _resolve_hit_rank_topk(results: Dict[int, Dict[str, Any]]) -> Tuple[Dict[int, int], str]:
    """Use cond1 top2 and cond2/cond3 top4 for rank-threshold hit alignment."""
    mapping = {1: 2, 2: 4, 3: 4}
    conditions = sorted({int(info.get("condition")) for info in results.values() if info.get("condition") is not None})
    if not conditions:
        return mapping, "top_by_condition"

    topk_by_condition = {condition: mapping.get(condition, 4) for condition in conditions}
    unique_topk = sorted(set(topk_by_condition.values()))
    if len(unique_topk) == 1:
        return topk_by_condition, f"top{unique_topk[0]}"
    return topk_by_condition, "top_by_condition"


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


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    agg_out = args.aggregate_output or (input_dir / "all_subjects.json")
    plots_dir = args.plots_dir or (input_dir / "plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    oral_mode, oral_data_path, oral_region_n_samples = _resolve_oral_settings(args)
    oral_plots_dir = plots_dir / f"oral_{oral_mode}_mode"
    oral_plots_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy = args.plot_accuracy or (plots_dir / "accuracy.png")
    plot_grid = args.plot_grid or (plots_dir / "error_grid.png")
    plot_posterior = args.plot_posterior or (plots_dir / "posterior.png")
    plot_cluster = args.plot_cluster or (plots_dir / "cluster_amount.png")
    plot_beta = args.plot_beta or (plots_dir / "beta.png")
    plot_accuracy_family = args.plot_accuracy_family or (plots_dir / "accuracy_family.png")
    trajectory_dir = args.trajectory_dir or (plots_dir / "trajectory_accuracy")
    trajectory_posterior_dir = args.trajectory_posterior_dir or (plots_dir / "trajectory_posterior")
    plot_oral_mass = args.plot_oral_mass or (oral_plots_dir / "oral_mass.png")
    plot_distribution_alignment_group = (
        args.plot_distribution_alignment_group or (oral_plots_dir / "distribution_based_alignment_group.png")
    )
    plot_distribution_alignment_subject = (
        args.plot_distribution_alignment_subject or (oral_plots_dir / "distribution_based_alignment_subject.png")
    )
    plot_oral_based_alignment_group = (
        args.plot_oral_based_alignment_group or (oral_plots_dir / "oral_based_alignment_group.png")
    )
    plot_oral_based_alignment_subject = (
        args.plot_oral_based_alignment_subject or (oral_plots_dir / "oral_based_alignment_subject.png")
    )
    plot_target_based_alignment_group = (
        args.plot_target_based_alignment_group or (oral_plots_dir / "target_based_alignment_group.png")
    )
    plot_target_based_alignment_subject = (
        args.plot_target_based_alignment_subject or (oral_plots_dir / "target_based_alignment_subject.png")
    )
    plot_hit_based_alignment_group = (
        args.plot_hit_based_alignment_group or (oral_plots_dir / "hit_based_alignment_group.png")
    )
    plot_hit_based_alignment_subject = (
        args.plot_hit_based_alignment_subject or (oral_plots_dir / "hit_based_alignment_subject.png")
    )
    plot_coverage_based_alignment_group = (
        args.plot_coverage_based_alignment_group
        or (oral_plots_dir / "coverage_based_alignment_group.png")
    )
    plot_coverage_based_alignment_subject = (
        args.plot_coverage_based_alignment_subject
        or (oral_plots_dir / "coverage_based_alignment_subject.png")
    )

    if not oral_data_path.is_file():
        raise FileNotFoundError(f"Oral data file not found: {oral_data_path}")

    oral_df = pd.read_csv(oral_data_path)
    aggregated = aggregate_grid_results(input_dir, eval_prediction_mode=args.eval_prediction_mode, trial_df=oral_df)
    oral_eval_df = oral_df[oral_df["iSub"].isin(aggregated.keys())].copy()

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

    oral_mass_cache = oral_plots_dir / "oral_mass_probabilities.npz"
    if oral_mass_cache.is_file():
        oral_mass = me.load_oral_mass_probabilities(oral_mass_cache)
        print(f"Loaded oral mass probabilities -> {oral_mass_cache}")
    else:
        oral_mass = me.compute_oral_mass_probabilities(
            oral_eval_df,
            oral_mode=oral_mode,
            region_n_samples=oral_region_n_samples,
        )
        me.save_oral_mass_probabilities(oral_mass, oral_mass_cache)
        print(f"Saved oral mass probabilities -> {oral_mass_cache}")
    me.plot_oral_mass_probabilities(oral_mass, save_path=str(plot_oral_mass))
    print(f"Saved oral mass plot -> {plot_oral_mass}")

    distribution_alignment = me.compute_distribution_based_alignment(
        aggregated,
        oral_eval_df,
        oral_mode=oral_mode,
        region_n_samples=oral_region_n_samples,
        model_distribution=args.distribution_alignment_model_state,
        oral_mass_results=oral_mass,
    )
    distribution_alignment_outputs = me.save_distribution_based_alignment_outputs(
        distribution_alignment,
        oral_plots_dir,
        prefix="distribution_based_alignment",
        group_plot_path=str(plot_distribution_alignment_group),
        subjectwise_plot_path=str(plot_distribution_alignment_subject),
        window_size=_resolve_plot_window_size(aggregated),
    )
    print(f"Saved distribution alignment group plot -> {distribution_alignment_outputs['group_plot']}")
    print(f"Saved distribution alignment subject-wise plot -> {distribution_alignment_outputs['subjectwise_plot']}")

    oral_based_alignment = me.compute_oral_based_alignment(
        aggregated,
        oral_eval_df,
        oral_mode=oral_mode,
        region_n_samples=oral_region_n_samples,
        model_distribution=args.oral_based_alignment_model_state,
    )
    oral_based_alignment_outputs = me.save_oral_based_alignment_outputs(
        oral_based_alignment,
        oral_plots_dir,
        group_plot_path=str(plot_oral_based_alignment_group),
        subjectwise_plot_path=str(plot_oral_based_alignment_subject),
        window_size=_resolve_plot_window_size(aggregated),
    )
    print(f"Saved oral-based alignment group plot -> {oral_based_alignment_outputs['group_plot']}")
    print(f"Saved oral-based alignment subject-wise plot -> {oral_based_alignment_outputs['subjectwise_plot']}")

    target_based_alignment = me.compute_target_based_alignment(
        aggregated,
        oral_eval_df,
        oral_mode=oral_mode,
        region_n_samples=oral_region_n_samples,
        oral_mass_results=oral_mass,
    )
    target_based_alignment_outputs = me.save_target_based_alignment_outputs(
        target_based_alignment,
        oral_plots_dir,
        group_plot_path=str(plot_target_based_alignment_group),
        subjectwise_plot_path=str(plot_target_based_alignment_subject),
        window_size=_resolve_plot_window_size(aggregated),
    )
    print(f"Saved target-based alignment group plot -> {target_based_alignment_outputs['group_plot']}")
    for space, path in target_based_alignment_outputs.get("subjectwise_plots", {}).items():
        print(f"Saved target-based alignment {space} subject-wise plot -> {path}")

    hit_based_alignment = me.compute_hit_based_alignment(
        aggregated,
        oral_eval_df,
        oral_mode=oral_mode,
        region_n_samples=oral_region_n_samples,
        oral_mass_results=oral_mass,
    )
    hit_based_alignment_outputs = me.save_hit_based_alignment_outputs(
        hit_based_alignment,
        oral_plots_dir,
        group_plot_path=str(plot_hit_based_alignment_group),
        subjectwise_plot_path=str(plot_hit_based_alignment_subject),
        window_size=_resolve_plot_window_size(aggregated),
    )
    print(f"Saved hit-based alignment group plot -> {hit_based_alignment_outputs['group_plot']}")
    print(f"Saved hit-based alignment subject-wise plot -> {hit_based_alignment_outputs['subjectwise_plot']}")

    hit_rank_topk, hit_rank_label = _resolve_hit_rank_topk(aggregated)
    hit_rank_based_alignment = me.compute_hit_based_alignment(
        aggregated,
        oral_eval_df,
        oral_mode=oral_mode,
        region_n_samples=oral_region_n_samples,
        oral_mass_results=oral_mass,
        rank_top_k=hit_rank_topk,
    )
    hit_rank_based_alignment_outputs = me.save_hit_based_alignment_outputs(
        hit_rank_based_alignment,
        oral_plots_dir,
        prefix=f"hit_based_alignment_{hit_rank_label}",
        window_size=_resolve_plot_window_size(aggregated),
        title_prefix=f"Condition-specific {hit_rank_label}",
    )
    print(f"Saved {hit_rank_label} hit-based alignment group plot -> {hit_rank_based_alignment_outputs['group_plot']}")
    print(
        f"Saved {hit_rank_label} hit-based alignment subject-wise plot -> "
        f"{hit_rank_based_alignment_outputs['subjectwise_plot']}"
    )

    coverage_based_alignment = me.compute_coverage_based_alignment(
        aggregated,
        oral_eval_df,
        oral_mode=oral_mode,
        region_n_samples=oral_region_n_samples,
        oral_mass_results=oral_mass,
    )
    coverage_based_alignment_outputs = me.save_coverage_based_alignment_outputs(
        coverage_based_alignment,
        oral_plots_dir,
        group_plot_path=str(plot_coverage_based_alignment_group),
        subjectwise_plot_path=str(plot_coverage_based_alignment_subject),
        window_size=_resolve_plot_window_size(aggregated),
    )
    print(f"Saved coverage-based alignment group plot -> {coverage_based_alignment_outputs['group_plot']}")
    print(f"Saved coverage-based alignment subject-wise plot -> {coverage_based_alignment_outputs['subjectwise_plot']}")

    aggregated_serializable = {
        sid: {
            **info,
            "grid_errors": _serialize_grid_errors(info.get("grid_errors", {})),
        }
        for sid, info in aggregated.items()
    }

    agg_out.parent.mkdir(parents=True, exist_ok=True)
    agg_out.write_text(json.dumps(aggregated_serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Aggregated {len(aggregated)} subjects -> {agg_out}")


if __name__ == "__main__":
    main()
