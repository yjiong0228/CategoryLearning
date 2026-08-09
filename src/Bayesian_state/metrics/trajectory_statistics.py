"""Statistics aggregated across repeated stochastic trajectories."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

from .behavior_metrics import (
    accuracy_curve_metrics,
    accuracy_scalar_metrics,
    history_kernel_metrics,
    switch_behavior_metrics,
)
from ._numeric import (
    finite_array,
    nanmean_or_nan,
    nanmedian_or_nan,
    nanquantile_or_nan,
    safe_float,
)
from .prediction_metrics import RunPrediction, empirical_crps
from .trajectory_selection import (
    accuracy_shape_score,
    distribution_selection_metrics,
    history_kernel_score,
    switch_behavior_score,
)


def behavior_ppc_group_metrics(values: Mapping[str, Any]) -> dict[str, float]:
    """Aggregate the established behavior-PPC columns for one subject."""
    choice_error = finite_array(values.get("choice_error", []))
    accuracy_mae = finite_array(values.get("acc_mae", []))
    volatility_ratio = finite_array(values.get("vol_ratio", []))
    history_correlation = finite_array(values.get("history_corr", []))
    switch_score = finite_array(values.get("switch_score", []))

    def mean(array: np.ndarray) -> float:
        return float(np.mean(array)) if array.size else float("nan")

    def median(array: np.ndarray) -> float:
        return float(np.median(array)) if array.size else float("nan")

    def quantile(array: np.ndarray, probability: float) -> float:
        return float(np.quantile(array, probability)) if array.size else float("nan")

    return {
        "choice_error_mean": mean(choice_error),
        "acc_mae_mean": mean(accuracy_mae),
        "acc_mae_median": median(accuracy_mae),
        "vol_ratio_median": median(volatility_ratio),
        "vol_ratio_q10": quantile(volatility_ratio, 0.10),
        "vol_ratio_q90": quantile(volatility_ratio, 0.90),
        "history_corr_mean": mean(history_correlation),
        "history_corr_median": median(history_correlation),
        "switch_score_mean": mean(switch_score),
        "switch_score_median": median(switch_score),
    }


def _metrics_for_run(run: Any, prediction_mode: str) -> Mapping[str, Any] | None:
    if isinstance(run, RunPrediction):
        if run.prediction_mode != prediction_mode:
            return None
        return run.to_metrics_mapping()
    if isinstance(run, Mapping):
        metrics_by_mode = run.get("metrics_by_mode") or {}
    else:
        metrics_by_mode = getattr(run, "metrics_by_mode", {}) or {}
    metrics = metrics_by_mode.get(prediction_mode) if isinstance(metrics_by_mode, Mapping) else None
    return metrics if isinstance(metrics, Mapping) else None


def marginal_prediction_metrics_from_runs(
    runs: Sequence[Any],
    *,
    selection_prediction_mode: str,
) -> dict[str, Any]:
    """Score marginal probabilities and the empirical trajectory distribution.

    This preserves the established simulation-statistics schema while accepting
    either ``RunPrediction`` objects, mappings, or legacy ``SingleRunResult``
    instances.
    """
    probability_rows: list[np.ndarray] = []
    sliding_rows: list[np.ndarray] = []
    observed_choice: np.ndarray | None = None
    valid_trial_mask: np.ndarray | None = None
    sliding_true: np.ndarray | None = None

    for run in runs:
        metrics = _metrics_for_run(run, selection_prediction_mode)
        if metrics is None:
            continue
        probabilities = np.asarray(metrics.get("pred_category_probs"), dtype=float)
        choices = np.asarray(metrics.get("observed_choice_index"), dtype=float).reshape(-1)
        valid = np.asarray(
            metrics.get("valid_trial_mask", np.ones(choices.size, dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        if probabilities.ndim == 2 and probabilities.shape[0] == choices.size and valid.size == choices.size:
            if observed_choice is None:
                observed_choice = choices.copy()
                valid_trial_mask = valid.copy()
            if (
                observed_choice.shape == choices.shape
                and probabilities.shape[0] == observed_choice.size
                and probabilities.shape[1] > 0
            ):
                probability_rows.append(probabilities)

        pred_curve = np.asarray(metrics.get("sliding_pred_acc"), dtype=float).reshape(-1)
        true_curve = np.asarray(metrics.get("sliding_true_acc"), dtype=float).reshape(-1)
        if pred_curve.size and pred_curve.shape == true_curve.shape:
            if sliding_true is None:
                sliding_true = true_curve.copy()
            if sliding_true.shape == pred_curve.shape:
                sliding_rows.append(pred_curve)

    out: dict[str, Any] = {
        "run_count": int(len(probability_rows)),
        "choice_brier": float("nan"),
        "choice_nll": float("nan"),
        "trajectory_run_count": int(len(sliding_rows)),
        "trajectory_crps": float("nan"),
        "trajectory_mean_mae": float("nan"),
        "trajectory_median_mae": float("nan"),
        "trajectory_coverage_90": float("nan"),
        "trajectory_median_vol_ratio": float("nan"),
    }

    if probability_rows and observed_choice is not None and valid_trial_mask is not None:
        stack = np.stack(probability_rows, axis=0)
        finite = np.all(np.isfinite(stack), axis=2)
        masked = np.where(finite[:, :, None], stack, np.nan)
        finite_counts = np.sum(np.isfinite(masked), axis=0)
        marginal = np.divide(
            np.nansum(masked, axis=0),
            finite_counts,
            out=np.full(masked.shape[1:], np.nan, dtype=float),
            where=finite_counts > 0,
        )
        row_sums = np.nansum(marginal, axis=1)
        finite_choice = np.isfinite(observed_choice)
        choice_index = np.full(observed_choice.shape, -1, dtype=int)
        choice_index[finite_choice] = observed_choice[finite_choice].astype(int)
        keep = (
            valid_trial_mask
            & finite_choice
            & (choice_index >= 0)
            & (choice_index < marginal.shape[1])
            & np.all(np.isfinite(marginal), axis=1)
            & (row_sums > 0.0)
        )
        if np.any(keep):
            normalized = marginal[keep] / row_sums[keep, None]
            selected = choice_index[keep]
            one_hot = np.zeros_like(normalized)
            one_hot[np.arange(normalized.shape[0]), selected] = 1.0
            out["choice_brier"] = float(
                np.mean(np.sum(np.square(normalized - one_hot), axis=1))
            )
            selected_probability = normalized[np.arange(normalized.shape[0]), selected]
            out["choice_nll"] = float(
                np.mean(-np.log(np.clip(selected_probability, 1e-12, 1.0)))
            )

    if sliding_rows and sliding_true is not None:
        stack = np.stack(sliding_rows, axis=0)
        finite_true = np.isfinite(sliding_true)
        crps = np.asarray(
            [
                empirical_crps(stack[:, index], sliding_true[index])
                for index in range(sliding_true.size)
            ],
            dtype=float,
        )
        finite_crps = crps[np.isfinite(crps) & finite_true]
        if finite_crps.size:
            out["trajectory_crps"] = float(np.mean(finite_crps))

        mean_curve = np.full(sliding_true.shape, np.nan, dtype=float)
        median_curve = np.full(sliding_true.shape, np.nan, dtype=float)
        q05 = np.full(sliding_true.shape, np.nan, dtype=float)
        q95 = np.full(sliding_true.shape, np.nan, dtype=float)
        for index in range(sliding_true.size):
            values = stack[:, index]
            values = values[np.isfinite(values)]
            if values.size:
                mean_curve[index] = float(np.mean(values))
                median_curve[index] = float(np.median(values))
                q05[index], q95[index] = np.quantile(values, [0.05, 0.95])
        keep = finite_true & np.isfinite(mean_curve) & np.isfinite(median_curve)
        if np.any(keep):
            out["trajectory_mean_mae"] = float(
                np.mean(np.abs(mean_curve[keep] - sliding_true[keep]))
            )
            out["trajectory_median_mae"] = float(
                np.mean(np.abs(median_curve[keep] - sliding_true[keep]))
            )
            out["trajectory_coverage_90"] = float(
                np.mean((sliding_true[keep] >= q05[keep]) & (sliding_true[keep] <= q95[keep]))
            )
            true_values = sliding_true[keep]
            median_values = median_curve[keep]
            true_volatility = (
                float(np.mean(np.abs(np.diff(true_values))))
                if true_values.size > 1
                else float("nan")
            )
            median_volatility = (
                float(np.mean(np.abs(np.diff(median_values))))
                if median_values.size > 1
                else float("nan")
            )
            if (
                np.isfinite(true_volatility)
                and true_volatility > 0.0
                and np.isfinite(median_volatility)
            ):
                out["trajectory_median_vol_ratio"] = float(
                    median_volatility / true_volatility
                )
    return out


def simulation_error_summary(errors: Sequence[Any]) -> Dict[str, Any]:
    """Summarize the repeated objective errors for one parameter setting."""

    values = np.asarray([float(value) for value in errors], dtype=float)
    if values.size == 0:
        raise ValueError("errors must contain at least one value")
    best_index = int(np.argmin(values))
    return {
        "sample_errors": [float(value) for value in values],
        "mean_error": float(np.mean(values)),
        "std_error": float(np.std(values)) if values.size > 1 else 0.0,
        "best_error": float(values[best_index]),
        "best_index": best_index,
    }


def loss_metric_summary_from_runs(
    runs: Sequence[Any],
    *,
    selection_prediction_mode: str,
) -> Dict[str, Dict[str, Any]]:
    values_by_metric: Dict[str, list[float]] = {}
    for run in runs:
        metrics_by_mode = getattr(run, "metrics_by_mode", {}) or {}
        metrics = metrics_by_mode.get(selection_prediction_mode)
        if not isinstance(metrics, Mapping):
            continue
        loss_values = metrics.get("loss_values")
        if not isinstance(loss_values, Mapping):
            loss_metric = metrics.get("loss_metric")
            mean_error = metrics.get("mean_error")
            if loss_metric is not None:
                loss_values = {str(loss_metric): mean_error}
            else:
                continue
        for name, value in loss_values.items():
            numeric = safe_float(value)
            if np.isfinite(numeric):
                values_by_metric.setdefault(str(name), []).append(float(numeric))

    summary: Dict[str, Dict[str, Any]] = {}
    for name, values in sorted(values_by_metric.items()):
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        ordered = np.sort(arr)
        best10_count = max(1, int(np.ceil(ordered.size * 0.10)))
        best25_count = max(1, int(np.ceil(ordered.size * 0.25)))
        best10_mean = float(np.mean(ordered[:best10_count]))
        best25_mean = float(np.mean(ordered[:best25_count]))
        summary[name] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "best": float(np.min(arr)),
            "best10_mean": best10_mean,
            "best10_count": int(best10_count),
            "best25_mean": best25_mean,
            "best25-mean": best25_mean,
            "best25_count": int(best25_count),
            "q10": float(np.quantile(arr, 0.10)),
            "std": float(np.std(arr)) if arr.size > 1 else 0.0,
            "count": int(arr.size),
        }
    return summary


def accuracy_shape_metrics_from_runs(
    runs: Sequence[Any],
    *,
    selection_prediction_mode: str,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    for repeat_index, run in enumerate(runs):
        metrics_by_mode = getattr(run, "metrics_by_mode", {}) or {}
        metrics = metrics_by_mode.get(selection_prediction_mode)
        if not isinstance(metrics, Mapping):
            continue
        curve = accuracy_curve_metrics(metrics)
        score = accuracy_shape_score(curve, config)
        choice_error = safe_float(getattr(run, "mean_error", np.nan))
        rows.append(
            {
                "repeat_index": int(repeat_index),
                "choice_error": choice_error,
                "accuracy_shape_score": score,
                **curve,
            }
        )
    if not rows:
        return {}

    finite_choice = np.asarray([row["choice_error"] for row in rows], dtype=float)
    finite_choice = finite_choice[np.isfinite(finite_choice)]
    if finite_choice.size == 0:
        return {}
    ordered_choice = np.sort(finite_choice)
    gate_count = max(1, int(np.ceil(len(rows) * float(config["run_choice_fraction"]))))
    gate_count = min(gate_count, ordered_choice.size)
    run_choice_cutoff = float(ordered_choice[gate_count - 1])
    eligible = [
        row for row in rows
        if np.isfinite(row["choice_error"]) and row["choice_error"] <= run_choice_cutoff
    ]
    if not eligible:
        eligible = rows
    best = min(
        eligible,
        key=lambda row: (
            safe_float(row.get("accuracy_shape_score"), float("inf")),
            safe_float(row.get("choice_error"), float("inf")),
        ),
    )
    all_scores = np.asarray([row["accuracy_shape_score"] for row in rows], dtype=float)
    eligible_scores = np.asarray([row["accuracy_shape_score"] for row in eligible], dtype=float)
    return {
        "accuracy_shape_score": safe_float(best.get("accuracy_shape_score"), float("inf")),
        "accuracy_shape_choice_error": safe_float(best.get("choice_error")),
        "accuracy_shape_repeat_index": int(best.get("repeat_index", -1)),
        "accuracy_shape_acc_mae": safe_float(best.get("acc_mae")),
        "accuracy_shape_acc_rmse": safe_float(best.get("acc_rmse")),
        "accuracy_shape_acc_corr": safe_float(best.get("acc_corr")),
        "accuracy_shape_vol_ratio": safe_float(best.get("vol_ratio")),
        "accuracy_shape_range_ratio": safe_float(best.get("range_ratio")),
        "accuracy_shape_slope_agree": safe_float(best.get("slope_agree")),
        "accuracy_shape_run_choice_cutoff": run_choice_cutoff,
        "accuracy_shape_eligible_run_count": int(len(eligible)),
        "accuracy_shape_all_run_count": int(len(rows)),
        "accuracy_shape_score_mean": float(np.nanmean(all_scores)),
        "accuracy_shape_score_q10": float(np.nanquantile(all_scores, 0.1)),
        "accuracy_shape_eligible_score_mean": float(np.nanmean(eligible_scores)),
    }


def history_kernel_metrics_from_runs(
    runs: Sequence[Any],
    *,
    selection_prediction_mode: str,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    for repeat_index, run in enumerate(runs):
        metrics_by_mode = getattr(run, "metrics_by_mode", {}) or {}
        metrics = metrics_by_mode.get(selection_prediction_mode)
        if not isinstance(metrics, Mapping):
            continue
        kernel = history_kernel_metrics(
            metrics,
            max_lag=int(config["history_max_lag"]),
            ridge=float(config["history_ridge"]),
            standardize=bool(config["history_standardize"]),
        )
        score = history_kernel_score(kernel, config)
        choice_error = safe_float(getattr(run, "mean_error", np.nan))
        rows.append(
            {
                "repeat_index": int(repeat_index),
                "choice_error": choice_error,
                "history_kernel_score": score,
                **kernel,
            }
        )
    if not rows:
        return {}

    finite_choice = np.asarray([row["choice_error"] for row in rows], dtype=float)
    finite_choice = finite_choice[np.isfinite(finite_choice)]
    if finite_choice.size == 0:
        return {}
    ordered_choice = np.sort(finite_choice)
    gate_count = max(1, int(np.ceil(len(rows) * float(config["run_choice_fraction"]))))
    gate_count = min(gate_count, ordered_choice.size)
    run_choice_cutoff = float(ordered_choice[gate_count - 1])
    eligible = [
        row for row in rows
        if np.isfinite(row["choice_error"]) and row["choice_error"] <= run_choice_cutoff
    ]
    if not eligible:
        eligible = rows
    best = min(
        eligible,
        key=lambda row: (
            safe_float(row.get("history_kernel_score"), float("inf")),
            safe_float(row.get("choice_error"), float("inf")),
        ),
    )
    all_scores = np.asarray([row["history_kernel_score"] for row in rows], dtype=float)
    eligible_scores = np.asarray([row["history_kernel_score"] for row in eligible], dtype=float)
    return {
        "history_kernel_score": safe_float(best.get("history_kernel_score"), float("inf")),
        "history_kernel_choice_error": safe_float(best.get("choice_error")),
        "history_kernel_repeat_index": int(best.get("repeat_index", -1)),
        "history_kernel_mse": safe_float(best.get("kernel_mse")),
        "history_kernel_corr": safe_float(best.get("kernel_corr")),
        "history_kernel_corr_loss": safe_float(best.get("kernel_corr_loss")),
        "history_kernel_norm_ratio": safe_float(best.get("kernel_norm_ratio")),
        "history_kernel_human_norm": safe_float(best.get("human_kernel_norm")),
        "history_kernel_model_norm": safe_float(best.get("model_kernel_norm")),
        "history_kernel_max_lag": int(best.get("max_lag", int(config["history_max_lag"]))),
        "history_kernel_n_rows": int(best.get("n_rows", 0)),
        "history_kernel_human": list(best.get("human_kernel") or []),
        "history_kernel_model": list(best.get("model_kernel") or []),
        "history_kernel_run_choice_cutoff": run_choice_cutoff,
        "history_kernel_eligible_run_count": int(len(eligible)),
        "history_kernel_all_run_count": int(len(rows)),
        "history_kernel_score_mean": float(np.nanmean(all_scores)),
        "history_kernel_score_q10": float(np.nanquantile(all_scores, 0.1)),
        "history_kernel_eligible_score_mean": float(np.nanmean(eligible_scores)),
    }


def switch_behavior_metrics_from_runs(
    runs: Sequence[Any],
    *,
    selection_prediction_mode: str,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    for repeat_index, run in enumerate(runs):
        metrics_by_mode = getattr(run, "metrics_by_mode", {}) or {}
        metrics = metrics_by_mode.get(selection_prediction_mode)
        if not isinstance(metrics, Mapping):
            continue
        switch = switch_behavior_metrics(
            metrics,
            min_trials=int(config["min_switch_trials"]),
        )
        score = switch_behavior_score(switch, config)
        choice_error = safe_float(getattr(run, "mean_error", np.nan))
        rows.append(
            {
                "repeat_index": int(repeat_index),
                "choice_error": choice_error,
                "switch_behavior_score": score,
                **switch,
            }
        )
    if not rows:
        return {}

    finite_choice = np.asarray([row["choice_error"] for row in rows], dtype=float)
    finite_choice = finite_choice[np.isfinite(finite_choice)]
    if finite_choice.size == 0:
        return {}
    ordered_choice = np.sort(finite_choice)
    gate_count = max(1, int(np.ceil(len(rows) * float(config["run_choice_fraction"]))))
    gate_count = min(gate_count, ordered_choice.size)
    run_choice_cutoff = float(ordered_choice[gate_count - 1])
    eligible = [
        row for row in rows
        if np.isfinite(row["choice_error"]) and row["choice_error"] <= run_choice_cutoff
    ]
    if not eligible:
        eligible = rows
    best = min(
        eligible,
        key=lambda row: (
            safe_float(row.get("switch_behavior_score"), float("inf")),
            safe_float(row.get("choice_error"), float("inf")),
        ),
    )
    all_scores = np.asarray([row["switch_behavior_score"] for row in rows], dtype=float)
    eligible_scores = np.asarray([row["switch_behavior_score"] for row in eligible], dtype=float)
    return {
        "switch_behavior_score": safe_float(best.get("switch_behavior_score"), float("inf")),
        "switch_behavior_choice_error": safe_float(best.get("choice_error")),
        "switch_behavior_repeat_index": int(best.get("repeat_index", -1)),
        "switch_behavior_switch_human": safe_float(best.get("switch_human")),
        "switch_behavior_switch_model": safe_float(best.get("switch_model")),
        "switch_behavior_switch_abs_diff": safe_float(best.get("switch_abs_diff")),
        "switch_behavior_perseveration_human": safe_float(best.get("perseveration_human")),
        "switch_behavior_perseveration_model": safe_float(best.get("perseveration_model")),
        "switch_behavior_perseveration_abs_diff": safe_float(best.get("perseveration_abs_diff")),
        "switch_behavior_win_stay_human": safe_float(best.get("win_stay_human")),
        "switch_behavior_win_stay_model": safe_float(best.get("win_stay_model")),
        "switch_behavior_win_stay_abs_diff": safe_float(best.get("win_stay_abs_diff")),
        "switch_behavior_lose_shift_human": safe_float(best.get("lose_shift_human")),
        "switch_behavior_lose_shift_model": safe_float(best.get("lose_shift_model")),
        "switch_behavior_lose_shift_abs_diff": safe_float(best.get("lose_shift_abs_diff")),
        "switch_behavior_n_pairs": int(best.get("n_pairs", 0)),
        "switch_behavior_n_win_pairs": int(best.get("n_win_pairs", 0)),
        "switch_behavior_n_loss_pairs": int(best.get("n_loss_pairs", 0)),
        "switch_behavior_run_choice_cutoff": run_choice_cutoff,
        "switch_behavior_eligible_run_count": int(len(eligible)),
        "switch_behavior_all_run_count": int(len(rows)),
        "switch_behavior_score_mean": float(np.nanmean(all_scores)),
        "switch_behavior_score_q10": float(np.nanquantile(all_scores, 0.1)),
        "switch_behavior_eligible_score_mean": float(np.nanmean(eligible_scores)),
    }


def distribution_behavior_metrics_from_runs(
    runs: Sequence[Any],
    *,
    selection_prediction_mode: str,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    for repeat_index, run in enumerate(runs):
        metrics_by_mode = getattr(run, "metrics_by_mode", {}) or {}
        metrics = metrics_by_mode.get(selection_prediction_mode)
        if not isinstance(metrics, Mapping):
            continue
        acc_scalar = accuracy_scalar_metrics(metrics)
        curve = accuracy_curve_metrics(metrics)
        kernel = history_kernel_metrics(
            metrics,
            max_lag=int(config["history_max_lag"]),
            ridge=float(config["history_ridge"]),
            standardize=bool(config["history_standardize"]),
        )
        switch = switch_behavior_metrics(
            metrics,
            min_trials=int(config["min_switch_trials"]),
        )
        rows.append(
            {
                "repeat_index": int(repeat_index),
                "choice_error": safe_float(getattr(run, "mean_error", np.nan)),
                "shape_score": accuracy_shape_score(curve, config),
                "history_score": history_kernel_score(kernel, config),
                "switch_score": switch_behavior_score(switch, config),
                **{f"acc_{key}": value for key, value in acc_scalar.items()},
                **{f"curve_{key}": value for key, value in curve.items()},
                **{f"kernel_{key}": value for key, value in kernel.items()},
                **{f"switch_{key}": value for key, value in switch.items()},
            }
        )
    if len(rows) < int(config["distribution_min_run_count"]):
        return {}

    choice_errors = [safe_float(row.get("choice_error")) for row in rows]
    acc_mae_mean = nanmean_or_nan(row.get("curve_acc_mae") for row in rows)
    vol_ratio_median = nanmedian_or_nan(row.get("curve_vol_ratio") for row in rows)
    slope_agree_mean = nanmean_or_nan(row.get("curve_slope_agree") for row in rows)
    distribution_curve = {
        "acc_mae": acc_mae_mean,
        "vol_ratio": vol_ratio_median,
        "slope_agree": slope_agree_mean,
    }
    distribution_shape_score = accuracy_shape_score(distribution_curve, config)

    history_summary = _distribution_history_summary(rows)
    distribution_history_score = history_kernel_score(history_summary, config)

    switch_summary = _distribution_switch_summary(rows)
    distribution_switch_score = switch_behavior_score(switch_summary, config)

    component_scores = {
        "choice_error": nanmean_or_nan(choice_errors),
        "accuracy_shape": distribution_shape_score,
        "history_kernel": distribution_history_score,
        "switch_behavior": distribution_switch_score,
    }
    selection_metrics = distribution_selection_metrics(
        rows,
        accuracy_mae=acc_mae_mean,
        volatility_ratio=vol_ratio_median,
        history_summary=history_summary,
        switch_summary=switch_summary,
        component_scores=component_scores,
        config=config,
    )

    return {
        **selection_metrics,
        "distribution_run_count": int(len(rows)),
        "distribution_choice_error_mean": nanmean_or_nan(choice_errors),
        "distribution_choice_error_median": nanmedian_or_nan(choice_errors),
        "distribution_choice_error_q10": nanquantile_or_nan(choice_errors, 0.10),
        "distribution_choice_error_std": (
            float(np.std(finite_array(choice_errors)))
            if finite_array(choice_errors).size > 1
            else 0.0
        ),
        "distribution_accuracy_shape_score": distribution_shape_score,
        "distribution_acc_mae_mean": acc_mae_mean,
        "distribution_acc_mae_median": nanmedian_or_nan(
            row.get("curve_acc_mae") for row in rows
        ),
        "distribution_acc_mae_q90": nanquantile_or_nan(
            (row.get("curve_acc_mae") for row in rows),
            0.90,
        ),
        "distribution_acc_rmse_mean": nanmean_or_nan(
            row.get("curve_acc_rmse") for row in rows
        ),
        "distribution_vol_ratio_mean": nanmean_or_nan(
            row.get("curve_vol_ratio") for row in rows
        ),
        "distribution_vol_ratio_median": vol_ratio_median,
        "distribution_vol_ratio_q10": nanquantile_or_nan(
            (row.get("curve_vol_ratio") for row in rows),
            0.10,
        ),
        "distribution_vol_ratio_q90": nanquantile_or_nan(
            (row.get("curve_vol_ratio") for row in rows),
            0.90,
        ),
        "distribution_slope_agree_mean": slope_agree_mean,
        "distribution_history_kernel_score": distribution_history_score,
        "distribution_history_kernel_mse": safe_float(history_summary.get("kernel_mse")),
        "distribution_history_kernel_corr": safe_float(history_summary.get("kernel_corr")),
        "distribution_history_kernel_corr_loss": safe_float(history_summary.get("kernel_corr_loss")),
        "distribution_history_kernel_norm_ratio": safe_float(history_summary.get("kernel_norm_ratio")),
        "distribution_history_kernel_human_norm": safe_float(history_summary.get("human_kernel_norm")),
        "distribution_history_kernel_model_norm": safe_float(history_summary.get("model_kernel_norm")),
        "distribution_history_kernel_run_count": int(history_summary.get("run_count", 0)),
        "distribution_switch_behavior_score": distribution_switch_score,
        "distribution_switch_human": safe_float(switch_summary.get("switch_human")),
        "distribution_switch_model": safe_float(switch_summary.get("switch_model")),
        "distribution_switch_abs_diff": safe_float(switch_summary.get("switch_abs_diff")),
        "distribution_perseveration_human": safe_float(switch_summary.get("perseveration_human")),
        "distribution_perseveration_model": safe_float(switch_summary.get("perseveration_model")),
        "distribution_perseveration_abs_diff": safe_float(switch_summary.get("perseveration_abs_diff")),
        "distribution_win_stay_human": safe_float(switch_summary.get("win_stay_human")),
        "distribution_win_stay_model": safe_float(switch_summary.get("win_stay_model")),
        "distribution_win_stay_abs_diff": safe_float(switch_summary.get("win_stay_abs_diff")),
        "distribution_lose_shift_human": safe_float(switch_summary.get("lose_shift_human")),
        "distribution_lose_shift_model": safe_float(switch_summary.get("lose_shift_model")),
        "distribution_lose_shift_abs_diff": safe_float(switch_summary.get("lose_shift_abs_diff")),
        "distribution_switch_run_count": int(switch_summary.get("run_count", 0)),
    }


def _distribution_history_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    human_kernels: list[np.ndarray] = []
    model_kernels: list[np.ndarray] = []
    for row in rows:
        human = np.asarray(row.get("kernel_human_kernel") or [], dtype=float)
        model = np.asarray(row.get("kernel_model_kernel") or [], dtype=float)
        if human.shape != model.shape or human.size == 0:
            continue
        finite = np.isfinite(human) & np.isfinite(model)
        if not finite.any():
            continue
        human_kernels.append(np.where(finite, human, np.nan))
        model_kernels.append(np.where(finite, model, np.nan))
    if not human_kernels:
        return {
            "kernel_mse": float("nan"),
            "kernel_corr": float("nan"),
            "kernel_corr_loss": float("nan"),
            "kernel_norm_ratio": float("nan"),
            "human_kernel_norm": float("nan"),
            "model_kernel_norm": float("nan"),
            "run_count": 0,
        }

    human_stack = np.vstack(human_kernels)
    model_stack = np.vstack(model_kernels)
    human_mean = np.nanmean(human_stack, axis=0)
    model_mean = np.nanmean(model_stack, axis=0)
    finite = np.isfinite(human_mean) & np.isfinite(model_mean)
    if not finite.any():
        return {
            "kernel_mse": float("nan"),
            "kernel_corr": float("nan"),
            "kernel_corr_loss": float("nan"),
            "kernel_norm_ratio": float("nan"),
            "human_kernel_norm": float("nan"),
            "model_kernel_norm": float("nan"),
            "run_count": 0,
        }
    human_f = human_mean[finite]
    model_f = model_mean[finite]
    diff = model_f - human_f
    human_norm = float(np.linalg.norm(human_f))
    model_norm = float(np.linalg.norm(model_f))
    if human_f.size > 1 and np.nanstd(human_f) > 1e-12 and np.nanstd(model_f) > 1e-12:
        corr = float(np.corrcoef(human_f, model_f)[0, 1])
    else:
        corr = float("nan")
    corr_loss = 0.5 * (1.0 - corr) if np.isfinite(corr) else 1.0
    return {
        "kernel_mse": float(np.mean(diff * diff)),
        "kernel_corr": corr,
        "kernel_corr_loss": float(corr_loss),
        "kernel_norm_ratio": float(model_norm / human_norm) if human_norm > 0 else float("nan"),
        "human_kernel_norm": human_norm,
        "model_kernel_norm": model_norm,
        "run_count": int(len(model_kernels)),
    }


def _distribution_switch_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def strip_prefix(key: str) -> str:
        prefix = "switch_"
        return key[len(prefix):] if key.startswith(prefix) else key

    def pair_summary(human_key: str, model_key: str, diff_key: str) -> Dict[str, float]:
        human_mean = nanmean_or_nan(row.get(human_key) for row in rows)
        model_mean = nanmean_or_nan(row.get(model_key) for row in rows)
        return {
            strip_prefix(human_key): human_mean,
            strip_prefix(model_key): model_mean,
            strip_prefix(diff_key): (
                float(abs(model_mean - human_mean))
                if np.isfinite(human_mean) and np.isfinite(model_mean)
                else float("nan")
            ),
        }

    out: Dict[str, float] = {}
    for human_key, model_key, diff_key in (
        ("switch_switch_human", "switch_switch_model", "switch_switch_abs_diff"),
        ("switch_perseveration_human", "switch_perseveration_model", "switch_perseveration_abs_diff"),
        ("switch_win_stay_human", "switch_win_stay_model", "switch_win_stay_abs_diff"),
        ("switch_lose_shift_human", "switch_lose_shift_model", "switch_lose_shift_abs_diff"),
    ):
        out.update(pair_summary(human_key, model_key, diff_key))
    out["run_count"] = int(
        np.sum(np.isfinite([safe_float(row.get("switch_switch_abs_diff")) for row in rows]))
    )
    return out



__all__ = [
    "accuracy_shape_metrics_from_runs",
    "behavior_ppc_group_metrics",
    "distribution_behavior_metrics_from_runs",
    "history_kernel_metrics_from_runs",
    "loss_metric_summary_from_runs",
    "marginal_prediction_metrics_from_runs",
    "simulation_error_summary",
    "switch_behavior_metrics_from_runs",
]
