"""Shared repeated-simulation statistics for hyper selectors and final runs."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np


MULTIOBJECTIVE_WEIGHT_DEFAULTS = {
    "choice_error": 0.0,
    "accuracy_shape": 1.0,
    "history_kernel": 1.0,
    "switch_behavior": 1.0,
}

SELECTION_METRIC_ALIASES = {
    "mean_simulation_error": "simulation.mean_error",
    "best_simulation_error": "simulation.best_error",
    "best10_mean_simulation_error": "simulation.best10_mean_error",
    "q10_simulation_error": "simulation.q10_error",
    "selection_error": "selection.primary.value",
    "accuracy_shape_score": "statistics.scores.accuracy_shape.value",
    "history_kernel_score": "statistics.scores.history_kernel.value",
    "switch_behavior_score": "statistics.scores.switch_behavior.value",
    "distribution_score": "statistics.scores.distribution.multiobjective.score",
    "distribution_component_max_raw": "statistics.scores.distribution.multiobjective.component_max_raw",
    "distribution_intersection_score": "statistics.scores.distribution.intersection.score",
    "distribution_ppc_interval_score": "statistics.scores.distribution.ppc_interval.score",
}

SIMULATION_STAT_DEFAULTS = {
    "enabled": False,
    "mode": "accuracy_shape",
    "primary_tolerance_abs": 0.02,
    "primary_tolerance_rel": 0.08,
    "run_choice_fraction": 0.10,
    "accuracy_weight": 1.0,
    "volatility_weight": 0.03,
    "slope_weight": 0.02,
    "target_volatility_ratio": 1.0,
    "min_volatility_ratio": 1e-6,
    "history_max_lag": 8,
    "history_ridge": 1e-3,
    "history_standardize": True,
    "history_kernel_weight": 1.0,
    "history_corr_weight": 0.05,
    "history_norm_weight": 0.0,
    "history_min_norm": 1e-6,
    "switch_weight": 1.0,
    "win_stay_weight": 1.0,
    "lose_shift_weight": 1.0,
    "perseveration_weight": 0.5,
    "min_switch_trials": 5,
    "multiobjective_weights": MULTIOBJECTIVE_WEIGHT_DEFAULTS,
    "distribution_min_run_count": 10,
    "distribution_interval_alpha": 0.10,
    "distribution_accept_acc_mae_max": 0.10,
    "distribution_accept_vol_ratio_min": 0.60,
    "distribution_accept_vol_ratio_max": 2.00,
    "distribution_accept_history_corr_min": 0.80,
    "distribution_accept_switch_score_max": 0.10,
}


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def get_stat_value(root: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Read a nested statistic by dot path."""
    current: Any = root
    for part in str(path).split("."):
        if not part:
            return default
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def resolve_selection_metric_path(metric: Any) -> str:
    """Normalize a configured selection metric to a structured statistics path."""
    path = str(metric or "").strip()
    if not path:
        raise ValueError("selection_metric cannot be empty")
    path = SELECTION_METRIC_ALIASES.get(path, path)
    if "." not in path:
        raise ValueError(
            "selection_metric must be a structured path such as "
            "'simulation.mean_error' or 'statistics.loss.choice_brier.mean'"
        )
    return path


def finite_array(values: Sequence[Any]) -> np.ndarray:
    arr = np.asarray([safe_float(value) for value in values], dtype=float)
    return arr[np.isfinite(arr)]


def nanmean_or_nan(values: Sequence[Any]) -> float:
    arr = finite_array(values)
    return float(np.mean(arr)) if arr.size else float("nan")


def nanmedian_or_nan(values: Sequence[Any]) -> float:
    arr = finite_array(values)
    return float(np.median(arr)) if arr.size else float("nan")


def nanquantile_or_nan(values: Sequence[Any], q: float) -> float:
    arr = finite_array(values)
    return float(np.quantile(arr, float(q))) if arr.size else float("nan")


def minimize_rank01(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    ranks = np.ones(arr.shape, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(arr))
    if finite_positions.size == 0:
        return ranks
    if finite_positions.size == 1:
        ranks[finite_positions[0]] = 0.0
        return ranks

    finite_values = arr[finite_positions]
    order = np.argsort(finite_values, kind="mergesort")
    sorted_positions = finite_positions[order]
    sorted_values = finite_values[order]
    denom = float(max(1, sorted_positions.size - 1))
    start = 0
    while start < sorted_positions.size:
        end = start + 1
        while end < sorted_positions.size and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = ((start + end - 1) / 2.0) / denom
        ranks[sorted_positions[start:end]] = float(rank)
        start = end
    return ranks


def resolve_simulation_stat_config(
    raw: Any = None,
    *,
    setting_name: str = "statistics_config",
) -> Dict[str, Any]:
    cfg = dict(SIMULATION_STAT_DEFAULTS)
    if raw is not None:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{setting_name} must be a mapping when provided")
        cfg.update(dict(raw))
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["mode"] = str(cfg.get("mode", "accuracy_shape")).strip().lower()
    mode_aliases = {
        "shape": "accuracy_shape",
        "accuracy": "accuracy_shape",
        "accuracy_curve": "accuracy_shape",
        "history": "history_kernel",
        "kernel": "history_kernel",
        "history_feedback": "history_kernel",
        "switch": "switch_behavior",
        "switching": "switch_behavior",
        "exploration": "switch_behavior",
        "perseveration": "switch_behavior",
        "multi": "multiobjective",
        "multi_objective": "multiobjective",
        "multi-objective": "multiobjective",
        "distribution": "distribution_multiobjective",
        "predictive_distribution": "distribution_multiobjective",
        "distribution_multi": "distribution_multiobjective",
        "distribution_multi_objective": "distribution_multiobjective",
        "distribution-multi-objective": "distribution_multiobjective",
        "intersection": "distribution_intersection",
        "acceptance": "distribution_intersection",
        "distribution_acceptance": "distribution_intersection",
        "distribution_intersection": "distribution_intersection",
        "ppc": "distribution_ppc_interval",
        "ppc_interval": "distribution_ppc_interval",
        "distribution_ppc": "distribution_ppc_interval",
        "distribution_ppc_interval": "distribution_ppc_interval",
    }
    cfg["mode"] = mode_aliases.get(cfg["mode"], cfg["mode"])
    if cfg["mode"] not in {
        "accuracy_shape",
        "history_kernel",
        "switch_behavior",
        "multiobjective",
        "distribution_multiobjective",
        "distribution_intersection",
        "distribution_ppc_interval",
    }:
        raise ValueError(
            f"{setting_name}.mode must be one of: "
            "'accuracy_shape', 'history_kernel', 'switch_behavior', "
            "'multiobjective', 'distribution_multiobjective', "
            "'distribution_intersection', 'distribution_ppc_interval'"
        )
    for key in (
        "primary_tolerance_abs",
        "primary_tolerance_rel",
        "run_choice_fraction",
        "accuracy_weight",
        "volatility_weight",
        "slope_weight",
        "target_volatility_ratio",
        "min_volatility_ratio",
        "history_ridge",
        "history_kernel_weight",
        "history_corr_weight",
        "history_norm_weight",
        "history_min_norm",
        "switch_weight",
        "win_stay_weight",
        "lose_shift_weight",
        "perseveration_weight",
        "distribution_interval_alpha",
        "distribution_accept_acc_mae_max",
        "distribution_accept_vol_ratio_min",
        "distribution_accept_vol_ratio_max",
        "distribution_accept_history_corr_min",
        "distribution_accept_switch_score_max",
    ):
        cfg[key] = float(cfg[key])
    cfg["history_max_lag"] = int(cfg["history_max_lag"])
    cfg["min_switch_trials"] = int(cfg["min_switch_trials"])
    cfg["distribution_min_run_count"] = int(cfg["distribution_min_run_count"])
    cfg["history_standardize"] = bool(cfg.get("history_standardize", True))
    raw_weights = cfg.get("multiobjective_weights")
    weights = dict(MULTIOBJECTIVE_WEIGHT_DEFAULTS)
    if raw_weights is not None:
        if not isinstance(raw_weights, Mapping):
            raise ValueError(f"{setting_name}.multiobjective_weights must be a mapping")
        weights.update({str(key): float(value) for key, value in raw_weights.items()})
    allowed_weight_keys = set(MULTIOBJECTIVE_WEIGHT_DEFAULTS)
    unknown_weight_keys = sorted(set(weights) - allowed_weight_keys)
    if unknown_weight_keys:
        raise ValueError(
            f"{setting_name}.multiobjective_weights contains unsupported keys: "
            + ", ".join(unknown_weight_keys)
        )
    if any(value < 0 for value in weights.values()):
        raise ValueError(f"{setting_name}.multiobjective_weights must be non-negative")
    if sum(weights.values()) <= 0:
        raise ValueError(f"{setting_name}.multiobjective_weights must include at least one positive weight")
    cfg["multiobjective_weights"] = weights
    if cfg["run_choice_fraction"] <= 0 or cfg["run_choice_fraction"] > 1:
        raise ValueError(f"{setting_name}.run_choice_fraction must be in (0, 1]")
    if cfg["target_volatility_ratio"] <= 0:
        raise ValueError(f"{setting_name}.target_volatility_ratio must be > 0")
    if cfg["min_volatility_ratio"] <= 0:
        raise ValueError(f"{setting_name}.min_volatility_ratio must be > 0")
    if cfg["history_max_lag"] <= 0:
        raise ValueError(f"{setting_name}.history_max_lag must be positive")
    if cfg["history_min_norm"] <= 0:
        raise ValueError(f"{setting_name}.history_min_norm must be positive")
    if cfg["min_switch_trials"] <= 0:
        raise ValueError(f"{setting_name}.min_switch_trials must be positive")
    if cfg["distribution_min_run_count"] <= 0:
        raise ValueError(f"{setting_name}.distribution_min_run_count must be positive")
    if cfg["distribution_interval_alpha"] <= 0 or cfg["distribution_interval_alpha"] >= 1:
        raise ValueError(f"{setting_name}.distribution_interval_alpha must be in (0, 1)")
    if cfg["distribution_accept_vol_ratio_min"] <= 0:
        raise ValueError(f"{setting_name}.distribution_accept_vol_ratio_min must be > 0")
    if cfg["distribution_accept_vol_ratio_max"] <= 0:
        raise ValueError(f"{setting_name}.distribution_accept_vol_ratio_max must be > 0")
    if cfg["distribution_accept_vol_ratio_max"] < cfg["distribution_accept_vol_ratio_min"]:
        raise ValueError(
            f"{setting_name}.distribution_accept_vol_ratio_max must be >= "
            "distribution_accept_vol_ratio_min"
        )
    return cfg


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


def accuracy_scalar_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    true_acc = np.asarray(metrics.get("true_acc"), dtype=float)
    pred_acc = np.asarray(metrics.get("pred_acc"), dtype=float)
    valid = np.asarray(metrics.get("valid_trial_mask", np.ones_like(true_acc, dtype=bool)), dtype=bool)
    empty = {
        "human_mean": float("nan"),
        "model_mean": float("nan"),
        "human_final": float("nan"),
        "model_final": float("nan"),
    }
    if true_acc.ndim != 1 or pred_acc.ndim != 1 or true_acc.shape != pred_acc.shape:
        return empty
    if valid.shape != true_acc.shape:
        return empty
    mask = valid & np.isfinite(true_acc) & np.isfinite(pred_acc)
    if not mask.any():
        return empty
    true = true_acc[mask]
    pred = pred_acc[mask]
    return {
        "human_mean": float(np.mean(true)),
        "model_mean": float(np.mean(pred)),
        "human_final": float(true[-1]),
        "model_final": float(pred[-1]),
    }


def _ppc_interval_summary(
    rows: Sequence[Mapping[str, Any]],
    stat_specs: Sequence[tuple[str, str, str]],
    *,
    alpha: float,
) -> Dict[str, Any]:
    alpha = float(alpha)
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q
    out: Dict[str, Any] = {
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


def accuracy_curve_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    true = np.asarray(metrics.get("sliding_true_acc"), dtype=float)
    pred = np.asarray(metrics.get("sliding_pred_acc"), dtype=float)
    empty = {
        "acc_mae": float("nan"),
        "acc_rmse": float("nan"),
        "acc_corr": float("nan"),
        "true_vol": float("nan"),
        "pred_vol": float("nan"),
        "vol_ratio": float("nan"),
        "true_range": float("nan"),
        "pred_range": float("nan"),
        "range_ratio": float("nan"),
        "slope_agree": float("nan"),
    }
    if true.shape != pred.shape or true.size == 0:
        return empty
    mask = np.isfinite(true) & np.isfinite(pred)
    if not mask.any():
        return empty
    true = true[mask]
    pred = pred[mask]
    diff = pred - true
    true_vol = float(np.mean(np.abs(np.diff(true)))) if true.size > 1 else float("nan")
    pred_vol = float(np.mean(np.abs(np.diff(pred)))) if pred.size > 1 else float("nan")
    true_range = float(np.nanmax(true) - np.nanmin(true))
    pred_range = float(np.nanmax(pred) - np.nanmin(pred))
    if np.nanstd(true) > 1e-12 and np.nanstd(pred) > 1e-12:
        acc_corr = float(np.corrcoef(true, pred)[0, 1])
    else:
        acc_corr = float("nan")
    d_true = np.diff(true)
    d_pred = np.diff(pred)
    slope_mask = (np.abs(d_true) > 1e-12) & (np.abs(d_pred) > 1e-12)
    slope_agree = (
        float(np.mean(np.sign(d_true[slope_mask]) == np.sign(d_pred[slope_mask])))
        if slope_mask.any()
        else float("nan")
    )
    return {
        "acc_mae": float(np.mean(np.abs(diff))),
        "acc_rmse": float(np.sqrt(np.mean(diff * diff))),
        "acc_corr": acc_corr,
        "true_vol": true_vol,
        "pred_vol": pred_vol,
        "vol_ratio": float(pred_vol / true_vol) if true_vol > 0 else float("nan"),
        "true_range": true_range,
        "pred_range": pred_range,
        "range_ratio": float(pred_range / true_range) if true_range > 0 else float("nan"),
        "slope_agree": slope_agree,
    }


def _standardized_lag_kernel(
    x: np.ndarray,
    y: np.ndarray,
    *,
    ridge: float,
    standardize: bool,
) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0] or x.shape[0] <= x.shape[1]:
        return np.full(x.shape[1] if x.ndim == 2 else 0, np.nan, dtype=float)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    y = y - float(np.nanmean(y))
    if standardize:
        x_scale = np.nanstd(x, axis=0, keepdims=True)
        x_scale = np.where(x_scale > 1e-12, x_scale, 1.0)
        x = x / x_scale
        y_scale = float(np.nanstd(y))
        if y_scale > 1e-12:
            y = y / y_scale
    xtx = x.T @ x
    penalty = max(0.0, float(ridge)) * np.eye(xtx.shape[0], dtype=float)
    try:
        return np.linalg.solve(xtx + penalty, x.T @ y)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(xtx + penalty) @ (x.T @ y)


def history_kernel_metrics(
    metrics: Mapping[str, Any],
    *,
    max_lag: int,
    ridge: float,
    standardize: bool,
) -> Dict[str, Any]:
    empty = {
        "kernel_mse": float("nan"),
        "kernel_corr": float("nan"),
        "kernel_corr_loss": float("nan"),
        "kernel_norm_ratio": float("nan"),
        "human_kernel_norm": float("nan"),
        "model_kernel_norm": float("nan"),
        "human_kernel": [],
        "model_kernel": [],
        "max_lag": int(max_lag),
        "n_rows": 0,
    }
    max_lag = int(max(1, max_lag))
    true_acc = np.asarray(metrics.get("true_acc"), dtype=float)
    pred_acc = np.asarray(metrics.get("pred_acc"), dtype=float)
    valid = np.asarray(metrics.get("valid_trial_mask", np.ones_like(true_acc, dtype=bool)), dtype=bool)
    if true_acc.ndim != 1 or pred_acc.ndim != 1 or true_acc.shape != pred_acc.shape:
        return empty
    if valid.shape != true_acc.shape or true_acc.size <= max_lag:
        return empty

    rows: list[list[float]] = []
    human_y: list[float] = []
    model_y: list[float] = []
    for trial_idx in range(max_lag, true_acc.size):
        if not bool(valid[trial_idx]):
            continue
        y_h = float(true_acc[trial_idx])
        y_m = float(pred_acc[trial_idx])
        if not (np.isfinite(y_h) and np.isfinite(y_m)):
            continue
        lag_values = [float(true_acc[trial_idx - lag]) for lag in range(1, max_lag + 1)]
        if not all(np.isfinite(value) for value in lag_values):
            continue
        rows.append(lag_values)
        human_y.append(y_h)
        model_y.append(y_m)
    if len(rows) <= max_lag:
        return empty

    x = np.asarray(rows, dtype=float)
    human = _standardized_lag_kernel(
        x,
        np.asarray(human_y, dtype=float),
        ridge=float(ridge),
        standardize=bool(standardize),
    )
    model = _standardized_lag_kernel(
        x,
        np.asarray(model_y, dtype=float),
        ridge=float(ridge),
        standardize=bool(standardize),
    )
    if human.shape != model.shape or human.size == 0:
        return empty
    finite = np.isfinite(human) & np.isfinite(model)
    if not finite.any():
        return empty
    human_f = human[finite]
    model_f = model[finite]
    diff = model_f - human_f
    human_norm = float(np.linalg.norm(human_f))
    model_norm = float(np.linalg.norm(model_f))
    if human_f.size > 1 and np.nanstd(human_f) > 1e-12 and np.nanstd(model_f) > 1e-12:
        kernel_corr = float(np.corrcoef(human_f, model_f)[0, 1])
    else:
        kernel_corr = float("nan")
    corr_loss = 0.5 * (1.0 - kernel_corr) if np.isfinite(kernel_corr) else 1.0
    return {
        "kernel_mse": float(np.mean(diff * diff)),
        "kernel_corr": kernel_corr,
        "kernel_corr_loss": float(corr_loss),
        "kernel_norm_ratio": float(model_norm / human_norm) if human_norm > 0 else float("nan"),
        "human_kernel_norm": human_norm,
        "model_kernel_norm": model_norm,
        "human_kernel": [float(value) if np.isfinite(value) else None for value in human],
        "model_kernel": [float(value) if np.isfinite(value) else None for value in model],
        "max_lag": int(max_lag),
        "n_rows": int(len(rows)),
    }


def switch_behavior_metrics(
    metrics: Mapping[str, Any],
    *,
    min_trials: int,
) -> Dict[str, Any]:
    empty = {
        "switch_human": float("nan"),
        "switch_model": float("nan"),
        "switch_abs_diff": float("nan"),
        "perseveration_human": float("nan"),
        "perseveration_model": float("nan"),
        "perseveration_abs_diff": float("nan"),
        "win_stay_human": float("nan"),
        "win_stay_model": float("nan"),
        "win_stay_abs_diff": float("nan"),
        "lose_shift_human": float("nan"),
        "lose_shift_model": float("nan"),
        "lose_shift_abs_diff": float("nan"),
        "n_pairs": 0,
        "n_win_pairs": 0,
        "n_loss_pairs": 0,
    }
    required = ("pred_category_probs", "observed_choice_index", "true_acc", "valid_trial_mask")
    if any(key not in metrics for key in required):
        return empty

    probs = np.asarray(metrics.get("pred_category_probs"), dtype=float)
    choices = np.asarray(metrics.get("observed_choice_index"), dtype=float)
    true_acc = np.asarray(metrics.get("true_acc"), dtype=float)
    valid = np.asarray(metrics.get("valid_trial_mask"), dtype=bool)
    if probs.ndim != 2 or choices.ndim != 1 or true_acc.ndim != 1 or valid.ndim != 1:
        return empty
    n_trials = probs.shape[0]
    if choices.shape[0] != n_trials or true_acc.shape[0] != n_trials or valid.shape[0] != n_trials:
        return empty
    if n_trials <= 1:
        return empty

    prev_choice = choices[:-1]
    next_choice = choices[1:]
    choice_pair_valid = (
        np.isfinite(prev_choice)
        & np.isfinite(next_choice)
        & (prev_choice >= 0)
        & (next_choice >= 0)
        & (prev_choice < probs.shape[1])
        & (next_choice < probs.shape[1])
        & (np.floor(prev_choice) == prev_choice)
        & (np.floor(next_choice) == next_choice)
    )
    pair_mask = valid[1:] & choice_pair_valid & np.all(np.isfinite(probs[1:, :]), axis=1)
    if not np.any(pair_mask):
        return empty

    rows = np.arange(1, n_trials)[pair_mask]
    prev_idx = prev_choice[pair_mask].astype(int)
    next_idx = next_choice[pair_mask].astype(int)
    prev_acc = true_acc[:-1][pair_mask]
    model_stay = probs[rows, prev_idx]
    finite = np.isfinite(model_stay) & np.isfinite(prev_acc)
    if int(np.sum(finite)) < int(min_trials):
        return empty

    model_stay = np.clip(model_stay[finite], 0.0, 1.0)
    prev_idx = prev_idx[finite]
    next_idx = next_idx[finite]
    prev_acc = prev_acc[finite]
    human_stay = (next_idx == prev_idx).astype(float)
    human_switch = 1.0 - human_stay
    model_switch = 1.0 - model_stay

    def summarize(human_values: np.ndarray, model_values: np.ndarray) -> tuple[float, float, float]:
        if human_values.size == 0 or model_values.size == 0:
            return float("nan"), float("nan"), float("nan")
        human_mean = float(np.mean(human_values))
        model_mean = float(np.mean(model_values))
        return human_mean, model_mean, float(abs(model_mean - human_mean))

    switch_h, switch_m, switch_diff = summarize(human_switch, model_switch)
    stay_h, stay_m, stay_diff = summarize(human_stay, model_stay)

    win_mask = prev_acc >= 0.5
    loss_mask = prev_acc < 0.5
    win_h, win_m, win_diff = summarize(human_stay[win_mask], model_stay[win_mask])
    loss_h, loss_m, loss_diff = summarize(human_switch[loss_mask], model_switch[loss_mask])

    return {
        "switch_human": switch_h,
        "switch_model": switch_m,
        "switch_abs_diff": switch_diff,
        "perseveration_human": stay_h,
        "perseveration_model": stay_m,
        "perseveration_abs_diff": stay_diff,
        "win_stay_human": win_h,
        "win_stay_model": win_m,
        "win_stay_abs_diff": win_diff,
        "lose_shift_human": loss_h,
        "lose_shift_model": loss_m,
        "lose_shift_abs_diff": loss_diff,
        "n_pairs": int(model_stay.size),
        "n_win_pairs": int(np.sum(win_mask)),
        "n_loss_pairs": int(np.sum(loss_mask)),
    }


def shape_score(curve: Mapping[str, Any], cfg: Mapping[str, Any]) -> float:
    acc_mae = safe_float(curve.get("acc_mae"))
    if not np.isfinite(acc_mae):
        return float("inf")
    vol_ratio = safe_float(curve.get("vol_ratio"), cfg["min_volatility_ratio"])
    vol_ratio = max(float(vol_ratio), float(cfg["min_volatility_ratio"]))
    target_vol = float(cfg["target_volatility_ratio"])
    vol_penalty = abs(np.log(vol_ratio / target_vol))
    slope = safe_float(curve.get("slope_agree"), 0.0)
    slope = min(1.0, max(0.0, slope))
    return float(
        float(cfg["accuracy_weight"]) * acc_mae
        + float(cfg["volatility_weight"]) * vol_penalty
        + float(cfg["slope_weight"]) * (1.0 - slope)
    )


def history_kernel_score(kernel: Mapping[str, Any], cfg: Mapping[str, Any]) -> float:
    kernel_mse = safe_float(kernel.get("kernel_mse"))
    if not np.isfinite(kernel_mse):
        return float("inf")
    corr_loss = safe_float(kernel.get("kernel_corr_loss"), 1.0)
    norm_ratio = safe_float(kernel.get("kernel_norm_ratio"), 1.0)
    norm_ratio = max(float(norm_ratio), float(cfg["history_min_norm"]))
    norm_penalty = abs(np.log(norm_ratio))
    return float(
        float(cfg["history_kernel_weight"]) * kernel_mse
        + float(cfg["history_corr_weight"]) * corr_loss
        + float(cfg["history_norm_weight"]) * norm_penalty
    )


def switch_behavior_score(switch: Mapping[str, Any], cfg: Mapping[str, Any]) -> float:
    components = (
        ("switch_abs_diff", "switch_weight"),
        ("win_stay_abs_diff", "win_stay_weight"),
        ("lose_shift_abs_diff", "lose_shift_weight"),
        ("perseveration_abs_diff", "perseveration_weight"),
    )
    total = 0.0
    weight_sum = 0.0
    for metric_key, weight_key in components:
        weight = float(cfg[weight_key])
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


def _loss_metric_summary_from_runs(
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


def _empirical_crps(samples: Sequence[Any], observation: Any) -> float:
    """Compute CRPS for an empirical one-dimensional predictive sample."""
    values = finite_array(samples)
    observed = safe_float(observation)
    if values.size == 0 or not np.isfinite(observed):
        return float("nan")
    ordered = np.sort(values)
    n_values = int(ordered.size)
    coefficients = 2.0 * np.arange(n_values, dtype=float) - n_values + 1.0
    pairwise_mean = 2.0 * float(np.sum(coefficients * ordered)) / float(n_values * n_values)
    return float(np.mean(np.abs(ordered - observed)) - 0.5 * pairwise_mean)


def marginal_prediction_metrics_from_runs(
    runs: Sequence[Any],
    *,
    selection_prediction_mode: str,
) -> Dict[str, Any]:
    """Score the marginal predictive distribution across stochastic runs.

    Per-run lower-tail scores answer whether a model can occasionally match the
    data. These metrics instead average category probabilities over all runs and
    score that marginal prediction. The trajectory CRPS additionally evaluates
    the calibration and sharpness of the run-level accuracy-curve distribution.
    """
    probability_rows: list[np.ndarray] = []
    sliding_rows: list[np.ndarray] = []
    observed_choice: np.ndarray | None = None
    valid_trial_mask: np.ndarray | None = None
    sliding_true: np.ndarray | None = None

    for run in runs:
        metrics_by_mode = getattr(run, "metrics_by_mode", {}) or {}
        metrics = metrics_by_mode.get(selection_prediction_mode)
        if not isinstance(metrics, Mapping):
            continue

        probs = np.asarray(metrics.get("pred_category_probs"), dtype=float)
        choices = np.asarray(metrics.get("observed_choice_index"), dtype=float).reshape(-1)
        valid = np.asarray(
            metrics.get("valid_trial_mask", np.ones(choices.size, dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        if probs.ndim == 2 and probs.shape[0] == choices.size and valid.size == choices.size:
            if observed_choice is None:
                observed_choice = choices.copy()
                valid_trial_mask = valid.copy()
            if (
                observed_choice.shape == choices.shape
                and probs.shape[0] == observed_choice.size
                and probs.shape[1] > 0
            ):
                probability_rows.append(probs)

        pred_curve = np.asarray(metrics.get("sliding_pred_acc"), dtype=float).reshape(-1)
        true_curve = np.asarray(metrics.get("sliding_true_acc"), dtype=float).reshape(-1)
        if pred_curve.size and pred_curve.shape == true_curve.shape:
            if sliding_true is None:
                sliding_true = true_curve.copy()
            if sliding_true.shape == pred_curve.shape:
                sliding_rows.append(pred_curve)

    out: Dict[str, Any] = {
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
                _empirical_crps(stack[:, idx], sliding_true[idx])
                for idx in range(sliding_true.size)
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
        for idx in range(sliding_true.size):
            values = stack[:, idx]
            values = values[np.isfinite(values)]
            if values.size:
                mean_curve[idx] = float(np.mean(values))
                median_curve[idx] = float(np.median(values))
                q05[idx], q95[idx] = np.quantile(values, [0.05, 0.95])
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
            true_vol = float(np.mean(np.abs(np.diff(true_values)))) if true_values.size > 1 else float("nan")
            median_vol = float(np.mean(np.abs(np.diff(median_values)))) if median_values.size > 1 else float("nan")
            if np.isfinite(true_vol) and true_vol > 0.0 and np.isfinite(median_vol):
                out["trajectory_median_vol_ratio"] = float(median_vol / true_vol)

    return out


def _score_stats(rows: Sequence[Mapping[str, Any]], score_key: str, eligible: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    all_scores = finite_array(row.get(score_key) for row in rows)
    eligible_scores = finite_array(row.get(score_key) for row in eligible)
    return {
        "mean": float(np.mean(all_scores)) if all_scores.size else float("nan"),
        "q10": float(np.quantile(all_scores, 0.10)) if all_scores.size else float("nan"),
        "eligible_mean": float(np.mean(eligible_scores)) if eligible_scores.size else float("nan"),
    }


def _accuracy_shape_nested(flat: Mapping[str, Any]) -> Dict[str, Any]:
    if not flat:
        return {}
    rmse = safe_float(flat.get("accuracy_shape_acc_rmse"))
    return {
        "diagnostics": {
            "selected": {
                "repeat_index": int(flat.get("accuracy_shape_repeat_index", -1)),
                "choice_error": safe_float(flat.get("accuracy_shape_choice_error")),
                "metrics": {
                    "mae": safe_float(flat.get("accuracy_shape_acc_mae")),
                    "rmse": rmse,
                    "mse": float(rmse * rmse) if np.isfinite(rmse) else float("nan"),
                    "corr": safe_float(flat.get("accuracy_shape_acc_corr")),
                    "volatility_ratio": safe_float(flat.get("accuracy_shape_vol_ratio")),
                    "range_ratio": safe_float(flat.get("accuracy_shape_range_ratio")),
                    "slope_agree": safe_float(flat.get("accuracy_shape_slope_agree")),
                },
            },
            "run_gate": {
                "choice_error_cutoff": safe_float(flat.get("accuracy_shape_run_choice_cutoff")),
                "eligible_count": int(flat.get("accuracy_shape_eligible_run_count", 0)),
                "run_count": int(flat.get("accuracy_shape_all_run_count", 0)),
            },
            "score_summary": {
                "mean": safe_float(flat.get("accuracy_shape_score_mean")),
                "q10": safe_float(flat.get("accuracy_shape_score_q10")),
                "eligible_mean": safe_float(flat.get("accuracy_shape_eligible_score_mean")),
            },
        },
        "score": {
            "value": safe_float(flat.get("accuracy_shape_score"), float("inf")),
            "repeat_index": int(flat.get("accuracy_shape_repeat_index", -1)),
            "choice_error": safe_float(flat.get("accuracy_shape_choice_error")),
            "components": {
                "accuracy_mae": safe_float(flat.get("accuracy_shape_acc_mae")),
                "volatility_ratio": safe_float(flat.get("accuracy_shape_vol_ratio")),
                "slope_agree": safe_float(flat.get("accuracy_shape_slope_agree")),
            },
        },
    }


def _history_kernel_nested(flat: Mapping[str, Any]) -> Dict[str, Any]:
    if not flat:
        return {}
    return {
        "diagnostics": {
            "selected": {
                "repeat_index": int(flat.get("history_kernel_repeat_index", -1)),
                "choice_error": safe_float(flat.get("history_kernel_choice_error")),
                "metrics": {
                    "mse": safe_float(flat.get("history_kernel_mse")),
                    "corr": safe_float(flat.get("history_kernel_corr")),
                    "corr_loss": safe_float(flat.get("history_kernel_corr_loss")),
                    "norm_ratio": safe_float(flat.get("history_kernel_norm_ratio")),
                    "human_norm": safe_float(flat.get("history_kernel_human_norm")),
                    "model_norm": safe_float(flat.get("history_kernel_model_norm")),
                    "max_lag": int(flat.get("history_kernel_max_lag", 0)),
                    "n_rows": int(flat.get("history_kernel_n_rows", 0)),
                    "human_kernel": list(flat.get("history_kernel_human") or []),
                    "model_kernel": list(flat.get("history_kernel_model") or []),
                },
            },
            "run_gate": {
                "choice_error_cutoff": safe_float(flat.get("history_kernel_run_choice_cutoff")),
                "eligible_count": int(flat.get("history_kernel_eligible_run_count", 0)),
                "run_count": int(flat.get("history_kernel_all_run_count", 0)),
            },
            "score_summary": {
                "mean": safe_float(flat.get("history_kernel_score_mean")),
                "q10": safe_float(flat.get("history_kernel_score_q10")),
                "eligible_mean": safe_float(flat.get("history_kernel_eligible_score_mean")),
            },
        },
        "score": {
            "value": safe_float(flat.get("history_kernel_score"), float("inf")),
            "repeat_index": int(flat.get("history_kernel_repeat_index", -1)),
            "choice_error": safe_float(flat.get("history_kernel_choice_error")),
            "components": {
                "mse": safe_float(flat.get("history_kernel_mse")),
                "corr_loss": safe_float(flat.get("history_kernel_corr_loss")),
                "norm_ratio": safe_float(flat.get("history_kernel_norm_ratio")),
            },
        },
    }


def _switch_behavior_nested(flat: Mapping[str, Any]) -> Dict[str, Any]:
    if not flat:
        return {}
    return {
        "diagnostics": {
            "selected": {
                "repeat_index": int(flat.get("switch_behavior_repeat_index", -1)),
                "choice_error": safe_float(flat.get("switch_behavior_choice_error")),
                "metrics": {
                    "switch": {
                        "human": safe_float(flat.get("switch_behavior_switch_human")),
                        "model": safe_float(flat.get("switch_behavior_switch_model")),
                        "abs_diff": safe_float(flat.get("switch_behavior_switch_abs_diff")),
                    },
                    "perseveration": {
                        "human": safe_float(flat.get("switch_behavior_perseveration_human")),
                        "model": safe_float(flat.get("switch_behavior_perseveration_model")),
                        "abs_diff": safe_float(flat.get("switch_behavior_perseveration_abs_diff")),
                    },
                    "win_stay": {
                        "human": safe_float(flat.get("switch_behavior_win_stay_human")),
                        "model": safe_float(flat.get("switch_behavior_win_stay_model")),
                        "abs_diff": safe_float(flat.get("switch_behavior_win_stay_abs_diff")),
                    },
                    "lose_shift": {
                        "human": safe_float(flat.get("switch_behavior_lose_shift_human")),
                        "model": safe_float(flat.get("switch_behavior_lose_shift_model")),
                        "abs_diff": safe_float(flat.get("switch_behavior_lose_shift_abs_diff")),
                    },
                    "counts": {
                        "pairs": int(flat.get("switch_behavior_n_pairs", 0)),
                        "win_pairs": int(flat.get("switch_behavior_n_win_pairs", 0)),
                        "loss_pairs": int(flat.get("switch_behavior_n_loss_pairs", 0)),
                    },
                },
            },
            "run_gate": {
                "choice_error_cutoff": safe_float(flat.get("switch_behavior_run_choice_cutoff")),
                "eligible_count": int(flat.get("switch_behavior_eligible_run_count", 0)),
                "run_count": int(flat.get("switch_behavior_all_run_count", 0)),
            },
            "score_summary": {
                "mean": safe_float(flat.get("switch_behavior_score_mean")),
                "q10": safe_float(flat.get("switch_behavior_score_q10")),
                "eligible_mean": safe_float(flat.get("switch_behavior_eligible_score_mean")),
            },
        },
        "score": {
            "value": safe_float(flat.get("switch_behavior_score"), float("inf")),
            "repeat_index": int(flat.get("switch_behavior_repeat_index", -1)),
            "choice_error": safe_float(flat.get("switch_behavior_choice_error")),
            "components": {
                "switch_abs_diff": safe_float(flat.get("switch_behavior_switch_abs_diff")),
                "win_stay_abs_diff": safe_float(flat.get("switch_behavior_win_stay_abs_diff")),
                "lose_shift_abs_diff": safe_float(flat.get("switch_behavior_lose_shift_abs_diff")),
                "perseveration_abs_diff": safe_float(flat.get("switch_behavior_perseveration_abs_diff")),
            },
        },
    }


def _ppc_stats_nested(flat: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for label in (
        "acc_mean",
        "acc_vol",
        "acc_range",
        "history_kernel_norm",
        "switch_rate",
        "win_stay",
        "lose_shift",
        "perseveration",
    ):
        prefix = f"distribution_ppc_interval_{label}_"
        if f"{prefix}human" not in flat:
            continue
        stats[label] = {
            "human": safe_float(flat.get(f"{prefix}human")),
            "model_lower": safe_float(flat.get(f"{prefix}model_q05")),
            "model_upper": safe_float(flat.get(f"{prefix}model_q95")),
            "model_median": safe_float(flat.get(f"{prefix}model_median")),
            "percentile": safe_float(flat.get(f"{prefix}percentile")),
            "tail_score": safe_float(flat.get(f"{prefix}tail_score")),
            "violation": safe_float(flat.get(f"{prefix}violation"), float("inf")),
            "accept": bool(flat.get(f"{prefix}accept", False)),
        }
    return stats


def _distribution_nested(flat: Mapping[str, Any]) -> Dict[str, Any]:
    if not flat:
        return {}
    violations = {
        "acc_mae": safe_float(flat.get("distribution_intersection_acc_mae_violation"), float("inf")),
        "vol_ratio_low": safe_float(flat.get("distribution_intersection_vol_ratio_low_violation"), float("inf")),
        "vol_ratio_high": safe_float(flat.get("distribution_intersection_vol_ratio_high_violation"), float("inf")),
        "history_corr": safe_float(flat.get("distribution_intersection_history_corr_violation"), float("inf")),
        "switch_rate": safe_float(flat.get("distribution_intersection_switch_rate_violation"), float("inf")),
        "win_stay": safe_float(flat.get("distribution_intersection_win_stay_violation"), float("inf")),
        "lose_shift": safe_float(flat.get("distribution_intersection_lose_shift_violation"), float("inf")),
        "perseveration": safe_float(flat.get("distribution_intersection_perseveration_violation"), float("inf")),
    }
    components = {
        "choice_error": safe_float(flat.get("distribution_choice_error_mean")),
        "accuracy_shape": safe_float(flat.get("distribution_accuracy_shape_score"), float("inf")),
        "history_kernel": safe_float(flat.get("distribution_history_kernel_score"), float("inf")),
        "switch_behavior": safe_float(flat.get("distribution_switch_behavior_score"), float("inf")),
    }
    return {
        "diagnostics": {
            "run_count": int(flat.get("distribution_run_count", 0)),
            "choice_error": {
                "mean": safe_float(flat.get("distribution_choice_error_mean")),
                "median": safe_float(flat.get("distribution_choice_error_median")),
                "q10": safe_float(flat.get("distribution_choice_error_q10")),
                "std": safe_float(flat.get("distribution_choice_error_std")),
            },
            "accuracy_curve": {
                "mae": {
                    "mean": safe_float(flat.get("distribution_acc_mae_mean")),
                    "median": safe_float(flat.get("distribution_acc_mae_median")),
                    "q90": safe_float(flat.get("distribution_acc_mae_q90")),
                },
                "rmse": {"mean": safe_float(flat.get("distribution_acc_rmse_mean"))},
                "volatility_ratio": {
                    "mean": safe_float(flat.get("distribution_vol_ratio_mean")),
                    "median": safe_float(flat.get("distribution_vol_ratio_median")),
                    "q10": safe_float(flat.get("distribution_vol_ratio_q10")),
                    "q90": safe_float(flat.get("distribution_vol_ratio_q90")),
                },
                "slope_agree": {"mean": safe_float(flat.get("distribution_slope_agree_mean"))},
            },
            "history_kernel": {
                "mse": safe_float(flat.get("distribution_history_kernel_mse")),
                "corr": safe_float(flat.get("distribution_history_kernel_corr")),
                "corr_loss": safe_float(flat.get("distribution_history_kernel_corr_loss")),
                "norm_ratio": safe_float(flat.get("distribution_history_kernel_norm_ratio")),
                "human_norm": safe_float(flat.get("distribution_history_kernel_human_norm")),
                "model_norm": safe_float(flat.get("distribution_history_kernel_model_norm")),
                "run_count": int(flat.get("distribution_history_kernel_run_count", 0)),
            },
            "switch_behavior": {
                "switch": {
                    "human": safe_float(flat.get("distribution_switch_human")),
                    "model": safe_float(flat.get("distribution_switch_model")),
                    "abs_diff": safe_float(flat.get("distribution_switch_abs_diff")),
                },
                "perseveration": {
                    "human": safe_float(flat.get("distribution_perseveration_human")),
                    "model": safe_float(flat.get("distribution_perseveration_model")),
                    "abs_diff": safe_float(flat.get("distribution_perseveration_abs_diff")),
                },
                "win_stay": {
                    "human": safe_float(flat.get("distribution_win_stay_human")),
                    "model": safe_float(flat.get("distribution_win_stay_model")),
                    "abs_diff": safe_float(flat.get("distribution_win_stay_abs_diff")),
                },
                "lose_shift": {
                    "human": safe_float(flat.get("distribution_lose_shift_human")),
                    "model": safe_float(flat.get("distribution_lose_shift_model")),
                    "abs_diff": safe_float(flat.get("distribution_lose_shift_abs_diff")),
                },
                "run_count": int(flat.get("distribution_switch_run_count", 0)),
            },
        },
        "score": {
            "multiobjective": {
                "score": safe_float(flat.get("distribution_score"), float("inf")),
                "component_max_raw": safe_float(flat.get("distribution_component_max_raw"), float("inf")),
                "components": components,
            },
            "intersection": {
                "score": safe_float(flat.get("distribution_intersection_score"), float("inf")),
                "accept": bool(flat.get("distribution_intersection_accept", False)),
                "violation_count": int(flat.get("distribution_intersection_violation_count", 0)),
                "violations": violations,
            },
            "ppc_interval": {
                "score": safe_float(flat.get("distribution_ppc_interval_score"), float("inf")),
                "accept": bool(flat.get("distribution_ppc_interval_accept", False)),
                "violation_count": int(flat.get("distribution_ppc_interval_violation_count", 0)),
                "alpha": safe_float(flat.get("distribution_ppc_interval_alpha")),
                "stat_count": int(flat.get("distribution_ppc_interval_stat_count", 0)),
                "lower_quantile": safe_float(flat.get("distribution_ppc_interval_lower_quantile")),
                "upper_quantile": safe_float(flat.get("distribution_ppc_interval_upper_quantile")),
                "stats": _ppc_stats_nested(flat),
            },
        },
    }


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
        score = shape_score(curve, config)
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
                "shape_score": shape_score(curve, config),
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
    distribution_shape_score = shape_score(distribution_curve, config)

    history_summary = _distribution_history_summary(rows)
    distribution_history_score = history_kernel_score(history_summary, config)

    switch_summary = _distribution_switch_summary(rows)
    distribution_switch_score = switch_behavior_score(switch_summary, config)

    intersection_violations = {
        "acc_mae": _upper_bound_violation(
            acc_mae_mean,
            config["distribution_accept_acc_mae_max"],
        ),
        "vol_ratio_low": _lower_bound_violation(
            vol_ratio_median,
            config["distribution_accept_vol_ratio_min"],
        ),
        "vol_ratio_high": _upper_bound_violation(
            vol_ratio_median,
            config["distribution_accept_vol_ratio_max"],
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
    distribution_intersection_score = (
        float(max(finite_violations))
        if len(finite_violations) == len(intersection_violations)
        else float("inf")
    )
    distribution_intersection_violation_count = int(
        sum(float(value) > 0.0 for value in finite_violations)
    )
    ppc_interval = _ppc_interval_summary(
        rows,
        (
            ("acc_mean", "acc_human_mean", "acc_model_mean"),
            ("acc_vol", "curve_true_vol", "curve_pred_vol"),
            ("acc_range", "curve_true_range", "curve_pred_range"),
            ("history_kernel_norm", "kernel_human_kernel_norm", "kernel_model_kernel_norm"),
            ("switch_rate", "switch_switch_human", "switch_switch_model"),
            ("win_stay", "switch_win_stay_human", "switch_win_stay_model"),
            ("lose_shift", "switch_lose_shift_human", "switch_lose_shift_model"),
            ("perseveration", "switch_perseveration_human", "switch_perseveration_model"),
        ),
        alpha=float(config["distribution_interval_alpha"]),
    )

    component_scores = {
        "choice_error": nanmean_or_nan(choice_errors),
        "accuracy_shape": distribution_shape_score,
        "history_kernel": distribution_history_score,
        "switch_behavior": distribution_switch_score,
    }
    weights = config.get("multiobjective_weights") or {}
    weighted_total = 0.0
    weight_sum = 0.0
    finite_components = []
    for component, value in component_scores.items():
        value = safe_float(value, float("inf"))
        if not np.isfinite(value):
            continue
        finite_components.append(value)
        weight = float(weights.get(component, 1.0))
        if weight <= 0:
            continue
        weighted_total += weight * value
        weight_sum += weight
    distribution_weighted_score = (
        float(weighted_total / weight_sum)
        if weight_sum > 0
        else float("inf")
    )

    return {
        "distribution_score": distribution_weighted_score,
        "distribution_component_max_raw": (
            float(max(finite_components)) if finite_components else float("inf")
        ),
        "distribution_intersection_score": distribution_intersection_score,
        "distribution_intersection_violation_count": distribution_intersection_violation_count,
        "distribution_intersection_accept": bool(distribution_intersection_score <= 0.0),
        "distribution_ppc_interval_score": safe_float(ppc_interval.get("score"), float("inf")),
        "distribution_ppc_interval_violation_count": int(ppc_interval.get("violation_count", 0)),
        "distribution_ppc_interval_accept": bool(ppc_interval.get("accept", False)),
        "distribution_ppc_interval_alpha": safe_float(ppc_interval.get("alpha")),
        "distribution_ppc_interval_stat_count": int(ppc_interval.get("stat_count", 0)),
        **{
            f"distribution_ppc_interval_{key}": value
            for key, value in ppc_interval.items()
            if key not in {"score", "violation_count", "accept", "alpha", "stat_count"}
        },
        "distribution_intersection_acc_mae_violation": intersection_violations["acc_mae"],
        "distribution_intersection_vol_ratio_low_violation": intersection_violations["vol_ratio_low"],
        "distribution_intersection_vol_ratio_high_violation": intersection_violations["vol_ratio_high"],
        "distribution_intersection_history_corr_violation": intersection_violations["history_corr"],
        "distribution_intersection_switch_rate_violation": intersection_violations["switch_rate"],
        "distribution_intersection_win_stay_violation": intersection_violations["win_stay"],
        "distribution_intersection_lose_shift_violation": intersection_violations["lose_shift"],
        "distribution_intersection_perseveration_violation": intersection_violations["perseveration"],
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


def compute_simulation_statistics(
    runs: Sequence[Any],
    *,
    selection_prediction_mode: str,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return structured statistics derived from all repeated simulation runs."""
    cfg = resolve_simulation_stat_config(config)
    summary: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}
    scores: Dict[str, Any] = {}

    loss_summary = _loss_metric_summary_from_runs(
        runs,
        selection_prediction_mode=selection_prediction_mode,
    )
    if loss_summary:
        summary["loss"] = loss_summary

    marginal = marginal_prediction_metrics_from_runs(
        runs,
        selection_prediction_mode=selection_prediction_mode,
    )
    if marginal:
        summary["marginal_prediction"] = marginal

    shape = _accuracy_shape_nested(
        accuracy_shape_metrics_from_runs(
            runs,
            selection_prediction_mode=selection_prediction_mode,
            config=cfg,
        )
    )
    if shape:
        diagnostics["accuracy_curve"] = shape["diagnostics"]
        scores["accuracy_shape"] = shape["score"]

    history = _history_kernel_nested(
        history_kernel_metrics_from_runs(
            runs,
            selection_prediction_mode=selection_prediction_mode,
            config=cfg,
        )
    )
    if history:
        diagnostics["history_kernel"] = history["diagnostics"]
        scores["history_kernel"] = history["score"]

    switch = _switch_behavior_nested(
        switch_behavior_metrics_from_runs(
            runs,
            selection_prediction_mode=selection_prediction_mode,
            config=cfg,
        )
    )
    if switch:
        diagnostics["switch_behavior"] = switch["diagnostics"]
        scores["switch_behavior"] = switch["score"]

    distribution = _distribution_nested(
        distribution_behavior_metrics_from_runs(
            runs,
            selection_prediction_mode=selection_prediction_mode,
            config=cfg,
        )
    )
    if distribution:
        diagnostics["distribution"] = distribution["diagnostics"]
        scores["distribution"] = distribution["score"]

    if diagnostics:
        summary["diagnostics"] = diagnostics
    if scores:
        summary["scores"] = scores
    return summary


__all__ = [
    "MULTIOBJECTIVE_WEIGHT_DEFAULTS",
    "SELECTION_METRIC_ALIASES",
    "SIMULATION_STAT_DEFAULTS",
    "accuracy_curve_metrics",
    "accuracy_scalar_metrics",
    "compute_simulation_statistics",
    "finite_array",
    "get_stat_value",
    "history_kernel_metrics",
    "marginal_prediction_metrics_from_runs",
    "minimize_rank01",
    "nanmean_or_nan",
    "nanmedian_or_nan",
    "nanquantile_or_nan",
    "resolve_selection_metric_path",
    "resolve_simulation_stat_config",
    "safe_float",
    "switch_behavior_metrics",
]
