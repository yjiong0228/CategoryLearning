"""Composite scores used to compare and select stochastic trajectories."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .behavior import accuracy_curve_metrics
from .numeric import finite_array, safe_float


def accuracy_shape_score(curve: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    """Combine curve error, volatility, and slope agreement."""

    accuracy_mae = safe_float(curve.get("acc_mae"))
    if not np.isfinite(accuracy_mae):
        return float("inf")
    volatility_ratio = safe_float(
        curve.get("vol_ratio"),
        config["min_volatility_ratio"],
    )
    volatility_ratio = max(
        float(volatility_ratio),
        float(config["min_volatility_ratio"]),
    )
    target_volatility = float(config["target_volatility_ratio"])
    volatility_penalty = abs(np.log(volatility_ratio / target_volatility))
    slope_agreement = safe_float(curve.get("slope_agree"), 0.0)
    slope_agreement = min(1.0, max(0.0, slope_agreement))
    return float(
        float(config["accuracy_weight"]) * accuracy_mae
        + float(config["volatility_weight"]) * volatility_penalty
        + float(config["slope_weight"]) * (1.0 - slope_agreement)
    )


def history_kernel_score(kernel: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    """Combine history-kernel discrepancy components."""

    kernel_mse = safe_float(kernel.get("kernel_mse"))
    if not np.isfinite(kernel_mse):
        return float("inf")
    correlation_loss = safe_float(kernel.get("kernel_corr_loss"), 1.0)
    norm_ratio = safe_float(kernel.get("kernel_norm_ratio"), 1.0)
    norm_ratio = max(float(norm_ratio), float(config["history_min_norm"]))
    norm_penalty = abs(np.log(norm_ratio))
    return float(
        float(config["history_kernel_weight"]) * kernel_mse
        + float(config["history_corr_weight"]) * correlation_loss
        + float(config["history_norm_weight"]) * norm_penalty
    )


def switch_behavior_score(switch: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    """Return the weighted mean of available switching discrepancies."""

    components = (
        ("switch_abs_diff", "switch_weight"),
        ("win_stay_abs_diff", "win_stay_weight"),
        ("lose_shift_abs_diff", "lose_shift_weight"),
        ("perseveration_abs_diff", "perseveration_weight"),
    )
    total = 0.0
    weight_sum = 0.0
    for metric_key, weight_key in components:
        weight = float(config[weight_key])
        if weight <= 0:
            continue
        value = safe_float(switch.get(metric_key))
        if not np.isfinite(value):
            continue
        total += weight * value
        weight_sum += weight
    if weight_sum <= 0:
        return float("inf")
    return float(total / weight_sum)


def _upper_bound_violation(value: Any, upper_bound: Any) -> float:
    value = safe_float(value, float("inf"))
    upper_bound = safe_float(upper_bound, float("nan"))
    if not np.isfinite(value) or not np.isfinite(upper_bound):
        return float("inf")
    scale = max(abs(upper_bound), 1e-12)
    return float(max(0.0, (value - upper_bound) / scale))


def _lower_bound_violation(value: Any, lower_bound: Any) -> float:
    value = safe_float(value, float("nan"))
    lower_bound = safe_float(lower_bound, float("nan"))
    if not np.isfinite(value) or not np.isfinite(lower_bound):
        return float("inf")
    scale = max(abs(lower_bound), 1e-12)
    return float(max(0.0, (lower_bound - value) / scale))


def _interval_violation(value: Any, lower_bound: Any, upper_bound: Any) -> float:
    value = safe_float(value, float("nan"))
    lower_bound = safe_float(lower_bound, float("nan"))
    upper_bound = safe_float(upper_bound, float("nan"))
    if not (np.isfinite(value) and np.isfinite(lower_bound) and np.isfinite(upper_bound)):
        return float("inf")
    width = max(float(upper_bound - lower_bound), 1e-12)
    scale = max(width, abs(value), abs(lower_bound), abs(upper_bound), 1e-12)
    if value < lower_bound:
        return float((lower_bound - value) / scale)
    if value > upper_bound:
        return float((value - upper_bound) / scale)
    return 0.0


def _ppc_interval_selection(
    rows: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
) -> dict[str, Any]:
    stat_specs = (
        ("acc_mean", "acc_human_mean", "acc_model_mean"),
        ("acc_vol", "curve_true_vol", "curve_pred_vol"),
        ("acc_range", "curve_true_range", "curve_pred_range"),
        ("history_kernel_norm", "kernel_human_kernel_norm", "kernel_model_kernel_norm"),
        ("switch_rate", "switch_switch_human", "switch_switch_model"),
        ("win_stay", "switch_win_stay_human", "switch_win_stay_model"),
        ("lose_shift", "switch_lose_shift_human", "switch_lose_shift_model"),
        ("perseveration", "switch_perseveration_human", "switch_perseveration_model"),
    )
    alpha = float(alpha)
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q
    out: dict[str, Any] = {
        "score": float("inf"),
        "accept": False,
        "violation_count": 0,
        "stat_count": int(len(stat_specs)),
        "alpha": alpha,
        "lower_quantile": lower_q,
        "upper_quantile": upper_q,
    }
    tail_scores: list[float] = []
    violations: list[float] = []
    complete = True
    for label, human_key, model_key in stat_specs:
        human_values = finite_array(row.get(human_key) for row in rows)
        model_values = finite_array(row.get(model_key) for row in rows)
        if human_values.size == 0 or model_values.size == 0:
            complete = False
            human_value = float("nan")
            lower = float("nan")
            upper = float("nan")
            median = float("nan")
            violation = float("inf")
            percentile = float("nan")
            tail_score = float("inf")
        else:
            human_value = float(np.median(human_values))
            lower = float(np.quantile(model_values, lower_q))
            upper = float(np.quantile(model_values, upper_q))
            median = float(np.median(model_values))
            violation = _interval_violation(human_value, lower, upper)
            percentile = float(np.mean(model_values <= human_value))
            tail_scale = max(0.5 - lower_q, 1e-12)
            tail_score = float(abs(percentile - 0.5) / tail_scale)
            tail_scores.append(tail_score)
            violations.append(violation)
        out[f"{label}_human"] = human_value
        out[f"{label}_model_q05"] = lower
        out[f"{label}_model_q95"] = upper
        out[f"{label}_model_median"] = median
        out[f"{label}_percentile"] = percentile
        out[f"{label}_tail_score"] = tail_score
        out[f"{label}_violation"] = violation
        out[f"{label}_accept"] = bool(np.isfinite(violation) and violation <= 0.0)

    if complete and len(violations) == len(stat_specs):
        out["score"] = float(max(tail_scores)) if tail_scores else float("inf")
        out["accept"] = bool(all(value <= 0.0 for value in violations))
        out["violation_count"] = int(sum(value > 0.0 for value in violations))
    return out


def distribution_selection_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    accuracy_mae: Any,
    volatility_ratio: Any,
    history_summary: Mapping[str, Any],
    switch_summary: Mapping[str, Any],
    component_scores: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply configured trajectory-distribution selection rules."""
    intersection_violations = {
        "acc_mae": _upper_bound_violation(
            accuracy_mae, config["distribution_accept_acc_mae_max"]
        ),
        "vol_ratio_low": _lower_bound_violation(
            volatility_ratio, config["distribution_accept_vol_ratio_min"]
        ),
        "vol_ratio_high": _upper_bound_violation(
            volatility_ratio, config["distribution_accept_vol_ratio_max"]
        ),
        "history_corr": _lower_bound_violation(
            history_summary.get("kernel_corr"),
            config["distribution_accept_history_corr_min"],
        ),
        "switch_rate": _upper_bound_violation(
            switch_summary.get("switch_abs_diff"),
            config["distribution_accept_switch_score_max"],
        ),
        "win_stay": _upper_bound_violation(
            switch_summary.get("win_stay_abs_diff"),
            config["distribution_accept_switch_score_max"],
        ),
        "lose_shift": _upper_bound_violation(
            switch_summary.get("lose_shift_abs_diff"),
            config["distribution_accept_switch_score_max"],
        ),
        "perseveration": _upper_bound_violation(
            switch_summary.get("perseveration_abs_diff"),
            config["distribution_accept_switch_score_max"],
        ),
    }
    finite_violations = [
        float(value)
        for value in intersection_violations.values()
        if np.isfinite(value)
    ]
    intersection_score = (
        float(max(finite_violations))
        if len(finite_violations) == len(intersection_violations)
        else float("inf")
    )
    ppc_interval = _ppc_interval_selection(
        rows,
        alpha=float(config["distribution_interval_alpha"]),
    )

    weights = config.get("multiobjective_weights") or {}
    weighted_total = 0.0
    weight_sum = 0.0
    finite_components: list[float] = []
    for component, raw_value in component_scores.items():
        value = safe_float(raw_value, float("inf"))
        if not np.isfinite(value):
            continue
        finite_components.append(value)
        weight = float(weights.get(component, 1.0))
        if weight <= 0:
            continue
        weighted_total += weight * value
        weight_sum += weight
    weighted_score = (
        float(weighted_total / weight_sum) if weight_sum > 0 else float("inf")
    )

    return {
        "distribution_score": weighted_score,
        "distribution_component_max_raw": (
            float(max(finite_components)) if finite_components else float("inf")
        ),
        "distribution_intersection_score": intersection_score,
        "distribution_intersection_violation_count": int(
            sum(float(value) > 0.0 for value in finite_violations)
        ),
        "distribution_intersection_accept": bool(intersection_score <= 0.0),
        "distribution_ppc_interval_score": safe_float(
            ppc_interval.get("score"), float("inf")
        ),
        "distribution_ppc_interval_violation_count": int(
            ppc_interval.get("violation_count", 0)
        ),
        "distribution_ppc_interval_accept": bool(ppc_interval.get("accept", False)),
        "distribution_ppc_interval_alpha": safe_float(ppc_interval.get("alpha")),
        "distribution_ppc_interval_stat_count": int(ppc_interval.get("stat_count", 0)),
        **{
            f"distribution_ppc_interval_{key}": value
            for key, value in ppc_interval.items()
            if key not in {"score", "violation_count", "accept", "alpha", "stat_count"}
        },
        **{
            f"distribution_intersection_{key}_violation": value
            for key, value in intersection_violations.items()
        },
    }


def representative_accuracy_shape_score(metrics: Mapping[str, Any]) -> float:
    """Legacy representative-run curve score, expressed as a shared metric."""

    curve = accuracy_curve_metrics(metrics)
    accuracy_mae = safe_float(curve.get("acc_mae"))
    if not np.isfinite(accuracy_mae):
        return float("inf")
    volatility_ratio = safe_float(curve.get("vol_ratio"))
    volatility_penalty = (
        abs(np.log(max(volatility_ratio, 1e-6)))
        if np.isfinite(volatility_ratio)
        else 1.0
    )
    return float(accuracy_mae + 0.06 * volatility_penalty)


def representative_switch_score(metrics: Mapping[str, Any]) -> float:
    """Absolute mismatch in mean human and model switching probability."""

    required = ("pred_category_probs", "observed_choice_index", "valid_trial_mask")
    if any(key not in metrics for key in required):
        return float("inf")
    probabilities = np.asarray(metrics.get("pred_category_probs"), dtype=float)
    choices = np.asarray(metrics.get("observed_choice_index"), dtype=float)
    valid = np.asarray(metrics.get("valid_trial_mask"), dtype=bool)
    if probabilities.ndim != 2 or choices.ndim != 1 or valid.ndim != 1:
        return float("inf")
    if (
        probabilities.shape[0] != choices.shape[0]
        or valid.shape[0] != choices.shape[0]
        or choices.size <= 1
    ):
        return float("inf")
    previous_choice = choices[:-1]
    next_choice = choices[1:]
    pair_mask = (
        valid[1:]
        & np.isfinite(previous_choice)
        & np.isfinite(next_choice)
        & (previous_choice >= 0)
        & (next_choice >= 0)
        & (previous_choice < probabilities.shape[1])
        & (next_choice < probabilities.shape[1])
        & np.all(np.isfinite(probabilities[1:, :]), axis=1)
    )
    if not np.any(pair_mask):
        return float("inf")
    rows = np.arange(1, choices.size)[pair_mask]
    previous_index = previous_choice[pair_mask].astype(int)
    next_index = next_choice[pair_mask].astype(int)
    model_switch = 1.0 - np.clip(
        probabilities[rows, previous_index],
        0.0,
        1.0,
    )
    human_switch = (next_index != previous_index).astype(float)
    return float(abs(np.mean(model_switch) - np.mean(human_switch)))


def representative_behavior_score(metrics: Mapping[str, Any]) -> float:
    """Mean of the finite representative-run curve and switching scores."""

    values = (
        representative_accuracy_shape_score(metrics),
        representative_switch_score(metrics),
    )
    finite_values = [value for value in values if np.isfinite(value)]
    return float(np.mean(finite_values)) if finite_values else float("inf")


__all__ = [
    "accuracy_shape_score",
    "distribution_selection_metrics",
    "history_kernel_score",
    "representative_accuracy_shape_score",
    "representative_behavior_score",
    "representative_switch_score",
    "switch_behavior_score",
]
