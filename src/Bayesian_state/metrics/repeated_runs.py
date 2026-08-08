"""Metrics that marginalize predictions across repeated stochastic runs."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import RunPrediction
from .predictive import empirical_crps


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


__all__ = ["marginal_prediction_metrics_from_runs"]
