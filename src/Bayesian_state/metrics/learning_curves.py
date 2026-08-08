"""Learning-curve construction and discrepancy metrics."""
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


__all__ = [
    "accuracy_curve_metrics",
    "accuracy_scalar_metrics",
    "centered_curve_metrics",
    "curve_discrepancy_metrics",
    "exponential_smooth_curve",
]
