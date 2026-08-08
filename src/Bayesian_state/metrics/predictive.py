"""Proper scores and calibration metrics for categorical predictions."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .contracts import MetricResult, TrialPrediction
from .numeric import finite_array, safe_float


def choice_brier(prediction: TrialPrediction) -> MetricResult:
    """Return the multiclass Brier score over valid trials."""
    valid = prediction.valid_trial_mask
    probabilities = prediction.category_probabilities[valid]
    choices = prediction.observed_choice_index[valid].astype(int)
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(probabilities.shape[0]), choices] = 1.0
    per_trial = np.sum(np.square(probabilities - one_hot), axis=1)
    value = float(np.mean(per_trial)) if per_trial.size else float("nan")
    return MetricResult(value, int(per_trial.size), {"per_trial": per_trial})


def choice_nll(
    prediction: TrialPrediction,
    *,
    probability_floor: float = 1e-12,
) -> MetricResult:
    """Return mean negative log probability of the observed choice."""
    floor = float(probability_floor)
    if not 0.0 < floor < 1.0:
        raise ValueError("probability_floor must lie in (0, 1)")
    selected = prediction.selected_probabilities()
    per_trial = -np.log(np.clip(selected, floor, 1.0))
    value = float(np.mean(per_trial)) if per_trial.size else float("nan")
    return MetricResult(value, int(per_trial.size), {"per_trial": per_trial})


def choice_probability_metrics(
    prediction: TrialPrediction,
    *,
    probability_floor: float = 1e-12,
    extreme_probability_threshold: float = 0.01,
) -> dict[str, Any]:
    """Return a compact collection of choice-probability scores."""
    threshold = float(extreme_probability_threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("extreme_probability_threshold must lie in [0, 1]")
    selected = prediction.selected_probabilities()
    brier = choice_brier(prediction)
    nll = choice_nll(prediction, probability_floor=probability_floor)
    return {
        "choice_brier": brier.value,
        "choice_nll": nll.value,
        "mean_observed_choice_probability": (
            float(np.mean(selected)) if selected.size else float("nan")
        ),
        "extreme_low_probability_count": int(np.sum(selected < threshold)),
        "n_observations": int(selected.size),
    }


def expected_calibration_error(
    prediction: TrialPrediction,
    *,
    n_bins: int = 10,
) -> MetricResult:
    """Compute top-label ECE using fixed-width confidence bins."""
    n_bins = int(n_bins)
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    probabilities = prediction.category_probabilities[prediction.valid_trial_mask]
    choices = prediction.observed_choice_index[prediction.valid_trial_mask].astype(int)
    if probabilities.shape[0] == 0:
        return MetricResult(float("nan"), 0, {"bins": [], "mce": float("nan")})

    confidence = np.max(probabilities, axis=1)
    predicted_choice = np.argmax(probabilities, axis=1)
    correct = (predicted_choice == choices).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.minimum(np.searchsorted(edges, confidence, side="right") - 1, n_bins - 1)
    bin_index = np.maximum(bin_index, 0)

    rows: list[dict[str, Any]] = []
    weighted_gap = 0.0
    max_gap = 0.0
    for index in range(n_bins):
        selected = bin_index == index
        count = int(np.sum(selected))
        if count:
            mean_confidence = float(np.mean(confidence[selected]))
            accuracy = float(np.mean(correct[selected]))
            gap = abs(accuracy - mean_confidence)
            weighted_gap += (count / confidence.size) * gap
            max_gap = max(max_gap, gap)
        else:
            mean_confidence = float("nan")
            accuracy = float("nan")
            gap = float("nan")
        rows.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )
    return MetricResult(
        float(weighted_gap),
        int(confidence.size),
        {"bins": rows, "mce": float(max_gap), "n_bins": n_bins},
    )


def empirical_crps(samples: Sequence[Any], observation: Any) -> float:
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


def predictive_interval_metrics(
    samples: np.ndarray,
    observations: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.10,
) -> dict[str, Any]:
    """Summarize empirical predictive intervals, width, and CRPS.

    ``samples`` is shaped ``(draw, observation)``. Non-finite draws are
    discarded separately for each observation.
    """
    sample_array = np.asarray(samples, dtype=float)
    observed = np.asarray(observations, dtype=float).reshape(-1)
    alpha = float(alpha)
    if sample_array.ndim != 2:
        raise ValueError(f"samples must be 2-D, got shape {sample_array.shape}")
    if sample_array.shape[1] != observed.size:
        raise ValueError(
            "sample observation width must match observations, "
            f"got {sample_array.shape[1]} and {observed.size}"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    lower = np.full(observed.shape, np.nan, dtype=float)
    upper = np.full(observed.shape, np.nan, dtype=float)
    median = np.full(observed.shape, np.nan, dtype=float)
    crps = np.full(observed.shape, np.nan, dtype=float)
    for index in range(observed.size):
        finite = sample_array[:, index]
        finite = finite[np.isfinite(finite)]
        if finite.size and np.isfinite(observed[index]):
            lower[index], median[index], upper[index] = np.quantile(
                finite,
                [alpha / 2.0, 0.5, 1.0 - alpha / 2.0],
            )
            crps[index] = empirical_crps(finite, observed[index])
    valid = np.isfinite(observed) & np.isfinite(lower) & np.isfinite(upper)
    width = upper - lower
    return {
        "alpha": alpha,
        "n_observations": int(np.sum(valid)),
        "coverage": (
            float(np.mean((observed[valid] >= lower[valid]) & (observed[valid] <= upper[valid])))
            if np.any(valid)
            else float("nan")
        ),
        "mean_width": float(np.mean(width[valid])) if np.any(valid) else float("nan"),
        "mean_crps": float(np.nanmean(crps[valid])) if np.any(valid) else float("nan"),
        "lower": lower,
        "median": median,
        "upper": upper,
        "per_observation_crps": crps,
    }


__all__ = [
    "choice_brier",
    "choice_nll",
    "choice_probability_metrics",
    "empirical_crps",
    "expected_calibration_error",
    "predictive_interval_metrics",
]
