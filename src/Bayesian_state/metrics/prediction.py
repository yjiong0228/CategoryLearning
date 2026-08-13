"""Prediction data contracts, proper scores, and calibration metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .numeric import finite_array, safe_float


def _readonly_array(values: Any, *, dtype: Any, ndim: int, name: str) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {array.shape}")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TrialPrediction:
    """One trial-aligned categorical predictive distribution.

    Choice indices are zero-based. Invalid or deliberately unscored trials may
    use any choice value and may contain non-finite probability rows; all rows
    selected by ``valid_trial_mask`` must be finite, non-negative, normalized,
    and paired with an in-range integer choice index.
    """

    category_probabilities: np.ndarray
    observed_choice_index: np.ndarray
    valid_trial_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        probabilities = _readonly_array(
            self.category_probabilities,
            dtype=float,
            ndim=2,
            name="category_probabilities",
        )
        choices = _readonly_array(
            self.observed_choice_index,
            dtype=float,
            ndim=1,
            name="observed_choice_index",
        )
        if probabilities.shape[0] != choices.shape[0]:
            raise ValueError(
                "category_probabilities and observed_choice_index must have the "
                f"same trial count, got {probabilities.shape[0]} and {choices.shape[0]}"
            )
        if probabilities.shape[1] < 2:
            raise ValueError("category_probabilities must contain at least two categories")

        if self.valid_trial_mask is None:
            valid = np.ones(choices.shape[0], dtype=bool)
            valid.setflags(write=False)
        else:
            valid = _readonly_array(
                self.valid_trial_mask,
                dtype=bool,
                ndim=1,
                name="valid_trial_mask",
            )
            if valid.shape != choices.shape:
                raise ValueError(
                    "valid_trial_mask must match observed_choice_index shape, "
                    f"got {valid.shape} and {choices.shape}"
                )

        valid_probabilities = probabilities[valid]
        valid_choices = choices[valid]
        if not np.all(np.isfinite(valid_probabilities)):
            raise ValueError("valid category-probability rows contain non-finite values")
        if np.any(valid_probabilities < 0.0):
            raise ValueError("valid category-probability rows contain negative values")
        if valid_probabilities.size and not np.allclose(
            valid_probabilities.sum(axis=1),
            1.0,
            rtol=1e-8,
            atol=1e-10,
        ):
            raise ValueError("valid category-probability rows are not normalized")
        if not np.all(np.isfinite(valid_choices)):
            raise ValueError("valid observed choices contain non-finite values")
        if not np.all(np.floor(valid_choices) == valid_choices):
            raise ValueError("valid observed choices must be integer-valued")
        if np.any(valid_choices < 0) or np.any(valid_choices >= probabilities.shape[1]):
            raise ValueError("valid observed choices fall outside the category range")

        object.__setattr__(self, "category_probabilities", probabilities)
        object.__setattr__(self, "observed_choice_index", choices)
        object.__setattr__(self, "valid_trial_mask", valid)

    @property
    def n_trials(self) -> int:
        return int(self.category_probabilities.shape[0])

    @property
    def n_categories(self) -> int:
        return int(self.category_probabilities.shape[1])

    @property
    def n_valid(self) -> int:
        return int(np.sum(self.valid_trial_mask))

    def selected_probabilities(self) -> np.ndarray:
        rows = np.flatnonzero(self.valid_trial_mask)
        choices = self.observed_choice_index[self.valid_trial_mask].astype(int)
        return self.category_probabilities[rows, choices]


@dataclass(frozen=True)
class RunPrediction:
    """Minimal metric input for one stochastic trajectory or marginal run."""

    trial: TrialPrediction
    prediction_mode: str
    sliding_true_accuracy: np.ndarray | None = None
    sliding_pred_accuracy: np.ndarray | None = None
    run_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = str(self.prediction_mode).strip()
        if not mode:
            raise ValueError("prediction_mode cannot be empty")
        object.__setattr__(self, "prediction_mode", mode)
        object.__setattr__(self, "metadata", dict(self.metadata))

        true_curve = self.sliding_true_accuracy
        pred_curve = self.sliding_pred_accuracy
        if (true_curve is None) != (pred_curve is None):
            raise ValueError(
                "sliding_true_accuracy and sliding_pred_accuracy must be supplied together"
            )
        if true_curve is not None and pred_curve is not None:
            true_array = _readonly_array(
                true_curve,
                dtype=float,
                ndim=1,
                name="sliding_true_accuracy",
            )
            pred_array = _readonly_array(
                pred_curve,
                dtype=float,
                ndim=1,
                name="sliding_pred_accuracy",
            )
            if true_array.shape != pred_array.shape:
                raise ValueError(
                    "sliding accuracy curves must have the same shape, "
                    f"got {true_array.shape} and {pred_array.shape}"
                )
            object.__setattr__(self, "sliding_true_accuracy", true_array)
            object.__setattr__(self, "sliding_pred_accuracy", pred_array)

    @classmethod
    def from_metrics(
        cls,
        metrics: Mapping[str, Any],
        *,
        prediction_mode: str,
        run_index: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "RunPrediction":
        probabilities = np.asarray(metrics.get("pred_category_probs"), dtype=float)
        choices = np.asarray(metrics.get("observed_choice_index"), dtype=float).reshape(-1)
        valid = np.asarray(
            metrics.get("valid_trial_mask", np.ones(choices.size, dtype=bool)),
            dtype=bool,
        ).reshape(-1)
        true_curve = metrics.get("sliding_true_acc")
        pred_curve = metrics.get("sliding_pred_acc")
        return cls(
            trial=TrialPrediction(probabilities, choices, valid),
            prediction_mode=prediction_mode,
            sliding_true_accuracy=true_curve,
            sliding_pred_accuracy=pred_curve,
            run_index=run_index,
            metadata=metadata or {},
        )

    def to_metrics_mapping(self) -> dict[str, Any]:
        metrics: dict[str, Any] = {
            "pred_category_probs": self.trial.category_probabilities,
            "observed_choice_index": self.trial.observed_choice_index,
            "valid_trial_mask": self.trial.valid_trial_mask,
        }
        if self.sliding_true_accuracy is not None:
            metrics["sliding_true_acc"] = self.sliding_true_accuracy
            metrics["sliding_pred_acc"] = self.sliding_pred_accuracy
        return metrics


@dataclass(frozen=True)
class MetricResult:
    """A scalar metric with its effective sample count and diagnostics."""

    value: float
    n_observations: int
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.n_observations) < 0:
            raise ValueError("n_observations cannot be negative")
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "n_observations", int(self.n_observations))
        object.__setattr__(self, "details", dict(self.details))

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "n_observations": self.n_observations,
            "details": dict(self.details),
        }


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
    "MetricResult",
    "RunPrediction",
    "TrialPrediction",
    "choice_brier",
    "choice_nll",
    "choice_probability_metrics",
    "empirical_crps",
    "expected_calibration_error",
    "predictive_interval_metrics",
]
