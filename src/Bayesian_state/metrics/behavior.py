"""Learning-curve, choice-history, and switching behavior metrics."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def exponential_smooth_curve(
    values: Sequence[float] | np.ndarray,
    *,
    alpha: float,
    init_value: float,
) -> np.ndarray:
    """Return an exponentially smoothed curve aligned to the input trials."""
    if not np.isfinite(alpha) or alpha <= 0.0 or alpha > 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha!r}.")
    if not np.isfinite(init_value):
        raise ValueError(f"init_value must be finite, got {init_value!r}.")
    array = np.asarray(values, dtype=float).reshape(-1)
    state = float(init_value)
    out = np.empty(array.shape[0], dtype=float)
    for index, value in enumerate(array):
        if np.isfinite(value):
            state = float(alpha) * float(value) + (1.0 - float(alpha)) * state
        out[index] = state
    return out


def accuracy_scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    true_acc = np.asarray(metrics.get("true_acc"), dtype=float)
    pred_acc = np.asarray(metrics.get("pred_acc"), dtype=float)
    valid = np.asarray(
        metrics.get("valid_trial_mask", np.ones_like(true_acc, dtype=bool)),
        dtype=bool,
    )
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


def curve_discrepancy_metrics(
    observed: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Compare two aligned learning curves without assigning decision thresholds."""
    true = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
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
    difference = pred - true
    true_vol = float(np.mean(np.abs(np.diff(true)))) if true.size > 1 else float("nan")
    pred_vol = float(np.mean(np.abs(np.diff(pred)))) if pred.size > 1 else float("nan")
    true_range = float(np.nanmax(true) - np.nanmin(true))
    pred_range = float(np.nanmax(pred) - np.nanmin(pred))
    if np.nanstd(true) > 1e-12 and np.nanstd(pred) > 1e-12:
        correlation = float(np.corrcoef(true, pred)[0, 1])
    else:
        correlation = float("nan")
    true_difference = np.diff(true)
    pred_difference = np.diff(pred)
    slope_mask = (np.abs(true_difference) > 1e-12) & (np.abs(pred_difference) > 1e-12)
    slope_agree = (
        float(np.mean(np.sign(true_difference[slope_mask]) == np.sign(pred_difference[slope_mask])))
        if slope_mask.any()
        else float("nan")
    )
    return {
        "acc_mae": float(np.mean(np.abs(difference))),
        "acc_rmse": float(np.sqrt(np.mean(difference * difference))),
        "acc_corr": correlation,
        "true_vol": true_vol,
        "pred_vol": pred_vol,
        "vol_ratio": float(pred_vol / true_vol) if true_vol > 0 else float("nan"),
        "true_range": true_range,
        "pred_range": pred_range,
        "range_ratio": float(pred_range / true_range) if true_range > 0 else float("nan"),
        "slope_agree": slope_agree,
    }


def centered_curve_metrics(
    observed: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Separate overall level bias from centered curve-shape discrepancy."""
    true = np.asarray(observed, dtype=float).reshape(-1)
    pred = np.asarray(predicted, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError(f"curve shapes differ: {true.shape} vs {pred.shape}")
    valid = np.isfinite(true) & np.isfinite(pred)
    if not np.any(valid):
        return {
            "level_bias": float("nan"),
            "absolute_level_bias": float("nan"),
            "centered_mae": float("nan"),
            "centered_rmse": float("nan"),
            "n_observations": 0,
        }
    true = true[valid]
    pred = pred[valid]
    level_bias = float(np.mean(pred) - np.mean(true))
    centered_difference = (pred - np.mean(pred)) - (true - np.mean(true))
    return {
        "level_bias": level_bias,
        "absolute_level_bias": abs(level_bias),
        "centered_mae": float(np.mean(np.abs(centered_difference))),
        "centered_rmse": float(np.sqrt(np.mean(np.square(centered_difference)))),
        "n_observations": int(true.size),
    }


def accuracy_curve_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return curve_discrepancy_metrics(
        metrics.get("sliding_true_acc"),
        metrics.get("sliding_pred_acc"),
    )


def _standardized_lag_kernel(
    predictors: np.ndarray,
    outcome: np.ndarray,
    *,
    ridge: float,
    standardize: bool,
) -> np.ndarray:
    predictors = np.asarray(predictors, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    if (
        predictors.ndim != 2
        or outcome.ndim != 1
        or predictors.shape[0] != outcome.shape[0]
        or predictors.shape[0] <= predictors.shape[1]
    ):
        return np.full(predictors.shape[1] if predictors.ndim == 2 else 0, np.nan)
    predictors = predictors - np.nanmean(predictors, axis=0, keepdims=True)
    outcome = outcome - float(np.nanmean(outcome))
    if standardize:
        predictor_scale = np.nanstd(predictors, axis=0, keepdims=True)
        predictor_scale = np.where(predictor_scale > 1e-12, predictor_scale, 1.0)
        predictors = predictors / predictor_scale
        outcome_scale = float(np.nanstd(outcome))
        if outcome_scale > 1e-12:
            outcome = outcome / outcome_scale
    cross_product = predictors.T @ predictors
    penalty = max(0.0, float(ridge)) * np.eye(cross_product.shape[0], dtype=float)
    try:
        return np.linalg.solve(cross_product + penalty, predictors.T @ outcome)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cross_product + penalty) @ (predictors.T @ outcome)


def history_kernel_metrics(
    metrics: Mapping[str, Any],
    *,
    max_lag: int,
    ridge: float,
    standardize: bool,
) -> dict[str, Any]:
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
    valid = np.asarray(
        metrics.get("valid_trial_mask", np.ones_like(true_acc, dtype=bool)),
        dtype=bool,
    )
    if true_acc.ndim != 1 or pred_acc.ndim != 1 or true_acc.shape != pred_acc.shape:
        return empty
    if valid.shape != true_acc.shape or true_acc.size <= max_lag:
        return empty

    rows: list[list[float]] = []
    human_y: list[float] = []
    model_y: list[float] = []
    for trial_index in range(max_lag, true_acc.size):
        if not bool(valid[trial_index]):
            continue
        human_value = float(true_acc[trial_index])
        model_value = float(pred_acc[trial_index])
        if not (np.isfinite(human_value) and np.isfinite(model_value)):
            continue
        lag_values = [float(true_acc[trial_index - lag]) for lag in range(1, max_lag + 1)]
        if not all(np.isfinite(value) for value in lag_values):
            continue
        rows.append(lag_values)
        human_y.append(human_value)
        model_y.append(model_value)
    if len(rows) <= max_lag:
        return empty

    predictors = np.asarray(rows, dtype=float)
    human = _standardized_lag_kernel(
        predictors,
        np.asarray(human_y, dtype=float),
        ridge=float(ridge),
        standardize=bool(standardize),
    )
    model = _standardized_lag_kernel(
        predictors,
        np.asarray(model_y, dtype=float),
        ridge=float(ridge),
        standardize=bool(standardize),
    )
    if human.shape != model.shape or human.size == 0:
        return empty
    finite = np.isfinite(human) & np.isfinite(model)
    if not finite.any():
        return empty
    human_finite = human[finite]
    model_finite = model[finite]
    difference = model_finite - human_finite
    human_norm = float(np.linalg.norm(human_finite))
    model_norm = float(np.linalg.norm(model_finite))
    if (
        human_finite.size > 1
        and np.nanstd(human_finite) > 1e-12
        and np.nanstd(model_finite) > 1e-12
    ):
        correlation = float(np.corrcoef(human_finite, model_finite)[0, 1])
    else:
        correlation = float("nan")
    correlation_loss = 0.5 * (1.0 - correlation) if np.isfinite(correlation) else 1.0
    return {
        "kernel_mse": float(np.mean(difference * difference)),
        "kernel_corr": correlation,
        "kernel_corr_loss": float(correlation_loss),
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
) -> dict[str, Any]:
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
        "switch_score": float("nan"),
        "n_pairs": 0,
        "n_win_pairs": 0,
        "n_loss_pairs": 0,
    }
    required = ("pred_category_probs", "observed_choice_index", "true_acc", "valid_trial_mask")
    if any(key not in metrics for key in required):
        return empty

    probabilities = np.asarray(metrics.get("pred_category_probs"), dtype=float)
    choices = np.asarray(metrics.get("observed_choice_index"), dtype=float)
    true_acc = np.asarray(metrics.get("true_acc"), dtype=float)
    valid = np.asarray(metrics.get("valid_trial_mask"), dtype=bool)
    if probabilities.ndim != 2 or choices.ndim != 1 or true_acc.ndim != 1 or valid.ndim != 1:
        return empty
    n_trials = probabilities.shape[0]
    if choices.shape[0] != n_trials or true_acc.shape[0] != n_trials or valid.shape[0] != n_trials:
        return empty
    if n_trials <= 1:
        return empty

    previous_choice = choices[:-1]
    next_choice = choices[1:]
    choice_pair_valid = (
        np.isfinite(previous_choice)
        & np.isfinite(next_choice)
        & (previous_choice >= 0)
        & (next_choice >= 0)
        & (previous_choice < probabilities.shape[1])
        & (next_choice < probabilities.shape[1])
        & (np.floor(previous_choice) == previous_choice)
        & (np.floor(next_choice) == next_choice)
    )
    pair_mask = valid[1:] & choice_pair_valid & np.all(np.isfinite(probabilities[1:, :]), axis=1)
    if not np.any(pair_mask):
        return empty

    rows = np.arange(1, n_trials)[pair_mask]
    previous_index = previous_choice[pair_mask].astype(int)
    next_index = next_choice[pair_mask].astype(int)
    previous_accuracy = true_acc[:-1][pair_mask]
    model_stay = probabilities[rows, previous_index]
    finite = np.isfinite(model_stay) & np.isfinite(previous_accuracy)
    if int(np.sum(finite)) < int(min_trials):
        return empty

    model_stay = np.clip(model_stay[finite], 0.0, 1.0)
    previous_index = previous_index[finite]
    next_index = next_index[finite]
    previous_accuracy = previous_accuracy[finite]
    human_stay = (next_index == previous_index).astype(float)
    human_switch = 1.0 - human_stay
    model_switch = 1.0 - model_stay

    def summarize(human_values: np.ndarray, model_values: np.ndarray) -> tuple[float, float, float]:
        if human_values.size == 0 or model_values.size == 0:
            return float("nan"), float("nan"), float("nan")
        human_mean = float(np.mean(human_values))
        model_mean = float(np.mean(model_values))
        return human_mean, model_mean, float(abs(model_mean - human_mean))

    switch_h, switch_m, switch_difference = summarize(human_switch, model_switch)
    stay_h, stay_m, stay_difference = summarize(human_stay, model_stay)
    win_mask = previous_accuracy >= 0.5
    loss_mask = previous_accuracy < 0.5
    win_h, win_m, win_difference = summarize(human_stay[win_mask], model_stay[win_mask])
    loss_h, loss_m, loss_difference = summarize(human_switch[loss_mask], model_switch[loss_mask])
    score_components = np.asarray(
        [switch_difference, win_difference, loss_difference], dtype=float
    )
    score_components = score_components[np.isfinite(score_components)]
    return {
        "switch_human": switch_h,
        "switch_model": switch_m,
        "switch_abs_diff": switch_difference,
        "perseveration_human": stay_h,
        "perseveration_model": stay_m,
        "perseveration_abs_diff": stay_difference,
        "win_stay_human": win_h,
        "win_stay_model": win_m,
        "win_stay_abs_diff": win_difference,
        "lose_shift_human": loss_h,
        "lose_shift_model": loss_m,
        "lose_shift_abs_diff": loss_difference,
        "switch_score": (
            float(np.mean(score_components)) if score_components.size else float("nan")
        ),
        "n_pairs": int(model_stay.size),
        "n_win_pairs": int(np.sum(win_mask)),
        "n_loss_pairs": int(np.sum(loss_mask)),
    }


__all__ = [
    "accuracy_curve_metrics",
    "accuracy_scalar_metrics",
    "centered_curve_metrics",
    "curve_discrepancy_metrics",
    "exponential_smooth_curve",
    "history_kernel_metrics",
    "switch_behavior_metrics",
]
