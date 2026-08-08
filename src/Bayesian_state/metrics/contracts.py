"""Backend- and workflow-neutral data contracts for model metrics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


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


__all__ = ["MetricResult", "RunPrediction", "TrialPrediction"]
