"""Scalar optimization losses computed from the shared prediction schema.

This module is the canonical implementation of every value accepted by the
``loss_metric`` configuration key.  Optimization code may choose a loss, but
must not reimplement its numerical definition.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np


LOSS_METRIC_ACCURACY_CURVE_MAE = "accuracy_curve_mae"
LOSS_METRIC_ACCURACY_CURVE_MSE = "accuracy_curve_mse"
LOSS_METRIC_ACCURACY_CURVE_FAMILY_MSE = "accuracy_curve_family_mse"
LOSS_METRIC_ACCURACY_CURVE_BERHU = "accuracy_curve_berhu"
LOSS_METRIC_ACCURACY_BRIER = "accuracy_brier"
LOSS_METRIC_ACCURACY_FAMILY_BRIER = "accuracy_family_brier"
LOSS_METRIC_ACCURACY_NLL = "accuracy_nll"
LOSS_METRIC_CHOICE_BRIER = "choice_brier"
LOSS_METRIC_CHOICE_NLL = "choice_nll"
LOSS_METRIC_WRONG_CHOICE_NLL = "wrong_choice_nll"
LOSS_METRIC_CONDITIONAL_WRONG_CHOICE_NLL = "conditional_wrong_choice_nll"
LOSS_METRIC_TARGET_PROB_BRIER = "target_prob_brier"

# Backward-compatible aliases used by existing configs and callers.
LOSS_METRIC_ACCURACY_MAE = LOSS_METRIC_ACCURACY_CURVE_MAE
LOSS_METRIC_ACCURACY_MSE = LOSS_METRIC_ACCURACY_CURVE_MSE
LOSS_METRIC_ACCURACY_BERHU = LOSS_METRIC_ACCURACY_CURVE_BERHU
LOSS_METRIC_MAE = LOSS_METRIC_ACCURACY_CURVE_MAE
LOSS_METRIC_MSE = LOSS_METRIC_ACCURACY_CURVE_MSE
LOSS_METRIC_BERHU = LOSS_METRIC_ACCURACY_CURVE_BERHU

ACCURACY_LOSS_METRIC_CHOICES = (
    LOSS_METRIC_ACCURACY_CURVE_MAE,
    LOSS_METRIC_ACCURACY_CURVE_MSE,
    LOSS_METRIC_ACCURACY_CURVE_FAMILY_MSE,
    LOSS_METRIC_ACCURACY_CURVE_BERHU,
    LOSS_METRIC_ACCURACY_BRIER,
    LOSS_METRIC_ACCURACY_FAMILY_BRIER,
    LOSS_METRIC_ACCURACY_NLL,
)
CHOICE_LOSS_METRIC_CHOICES = (
    LOSS_METRIC_CHOICE_BRIER,
    LOSS_METRIC_CHOICE_NLL,
    LOSS_METRIC_WRONG_CHOICE_NLL,
    LOSS_METRIC_CONDITIONAL_WRONG_CHOICE_NLL,
)
PROBABILISTIC_LOSS_METRIC_CHOICES = (LOSS_METRIC_TARGET_PROB_BRIER,)
LOSS_METRIC_CHOICES = (
    ACCURACY_LOSS_METRIC_CHOICES
    + CHOICE_LOSS_METRIC_CHOICES
    + PROBABILISTIC_LOSS_METRIC_CHOICES
)


class LossStrategy(ABC):
    name: str

    @abstractmethod
    def compute(self, metrics: Mapping[str, Any]) -> float:
        raise NotImplementedError


def accuracy_curve_mae(metrics: Mapping[str, Any]) -> float:
    true = np.asarray(metrics["sliding_true_acc"], dtype=float)
    predicted = np.asarray(metrics["sliding_pred_acc"], dtype=float)
    error = np.abs(true - predicted)
    return float(np.nanmean(error)) if error.size else float("nan")


def accuracy_curve_mse(metrics: Mapping[str, Any]) -> float:
    true = np.asarray(metrics["sliding_true_acc"], dtype=float)
    predicted = np.asarray(metrics["sliding_pred_acc"], dtype=float)
    error = np.square(true - predicted)
    return float(np.nanmean(error)) if error.size else float("nan")


def accuracy_curve_family_mse(metrics: Mapping[str, Any]) -> float:
    true = np.asarray(metrics["sliding_true_family_acc"], dtype=float)
    predicted = np.asarray(metrics["sliding_pred_family_acc"], dtype=float)
    error = np.square(true - predicted)
    return float(np.nanmean(error)) if error.size else float("nan")


def accuracy_curve_berhu(metrics: Mapping[str, Any], *, delta: float) -> float:
    """Reverse-Huber loss for the aligned accuracy curve."""
    delta = float(delta)
    if delta <= 0:
        raise ValueError(f"loss_delta must be > 0 for accuracy_curve_berhu, got {delta}")
    true = np.asarray(metrics["sliding_true_acc"], dtype=float)
    predicted = np.asarray(metrics["sliding_pred_acc"], dtype=float)
    absolute_error = np.abs(true - predicted)
    piecewise = np.where(
        absolute_error <= delta,
        absolute_error,
        (np.square(absolute_error) + delta**2) / (2.0 * delta),
    )
    return float(np.nanmean(piecewise)) if piecewise.size else float("nan")


def accuracy_berhu(metrics: Mapping[str, Any], *, delta: float) -> float:
    """Compatibility spelling for :func:`accuracy_curve_berhu`."""
    return accuracy_curve_berhu(metrics, delta=delta)


def _valid_trial_accuracy_data(
    metrics: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(metrics["pred_category_probs"], dtype=float)
    true_index = np.asarray(metrics["true_category_index"], dtype=int)
    true_accuracy = np.asarray(metrics["true_acc"], dtype=float)
    valid = np.asarray(metrics["valid_trial_mask"], dtype=bool)
    if probabilities.ndim != 2:
        raise ValueError(
            f"pred_category_probs must be 2-D, got shape {probabilities.shape}"
        )
    n_trials = probabilities.shape[0]
    for name, values in (
        ("valid_trial_mask", valid),
        ("true_category_index", true_index),
        ("true_acc", true_accuracy),
    ):
        if values.shape[0] != n_trials:
            raise ValueError(
                f"{name} length does not match pred_category_probs rows: "
                f"{values.shape[0]} vs {n_trials}"
            )

    probabilities = probabilities[valid]
    true_index = true_index[valid]
    true_accuracy = true_accuracy[valid]
    if probabilities.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    n_categories = probabilities.shape[1]
    keep = (
        (true_index >= 0)
        & (true_index < n_categories)
        & np.all(np.isfinite(probabilities), axis=1)
        & np.isfinite(true_accuracy)
    )
    if not np.any(keep):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    probabilities = probabilities[keep]
    true_index = true_index[keep]
    true_accuracy = np.clip(true_accuracy[keep], 0.0, 1.0)
    return probabilities[np.arange(probabilities.shape[0]), true_index], true_accuracy


def accuracy_brier(metrics: Mapping[str, Any]) -> float:
    predicted, observed = _valid_trial_accuracy_data(metrics)
    if predicted.size == 0:
        return float("nan")
    return float(np.mean(np.square(predicted - observed)))


def _valid_trial_family_accuracy_data(
    metrics: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(metrics["pred_family_acc"], dtype=float)
    observed = np.asarray(metrics["true_family_acc"], dtype=float)
    valid = np.asarray(metrics["valid_trial_mask"], dtype=bool)
    if valid.shape[0] != predicted.shape[0]:
        raise ValueError(
            "valid_trial_mask length does not match pred_family_acc length: "
            f"{valid.shape[0]} vs {predicted.shape[0]}"
        )
    if observed.shape[0] != predicted.shape[0]:
        raise ValueError(
            "true_family_acc length does not match pred_family_acc length: "
            f"{observed.shape[0]} vs {predicted.shape[0]}"
        )
    predicted = predicted[valid]
    observed = observed[valid]
    keep = np.isfinite(predicted) & np.isfinite(observed)
    if not np.any(keep):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return predicted[keep], np.clip(observed[keep], 0.0, 1.0)


def accuracy_family_brier(metrics: Mapping[str, Any]) -> float:
    predicted, observed = _valid_trial_family_accuracy_data(metrics)
    if predicted.size == 0:
        return float("nan")
    return float(np.mean(np.square(predicted - observed)))


def accuracy_nll(metrics: Mapping[str, Any], *, eps: float = 1e-12) -> float:
    predicted, observed = _valid_trial_accuracy_data(metrics)
    if predicted.size == 0:
        return float("nan")
    predicted = np.clip(predicted, float(eps), 1.0 - float(eps))
    return float(
        np.mean(
            -(observed * np.log(predicted) + (1.0 - observed) * np.log(1.0 - predicted))
        )
    )


def _valid_trial_two_target_data(
    metrics: Mapping[str, Any],
    first_target_key: str,
    second_target_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray(metrics["pred_category_probs"], dtype=float)
    first_index = np.asarray(metrics[first_target_key], dtype=int)
    second_index = np.asarray(metrics[second_target_key], dtype=int)
    valid = np.asarray(metrics["valid_trial_mask"], dtype=bool)
    if probabilities.ndim != 2:
        raise ValueError(
            f"pred_category_probs must be 2-D, got shape {probabilities.shape}"
        )
    n_trials = probabilities.shape[0]
    for name, values in (
        ("valid_trial_mask", valid),
        (first_target_key, first_index),
        (second_target_key, second_index),
    ):
        if values.shape[0] != n_trials:
            raise ValueError(
                f"{name} length does not match pred_category_probs rows: "
                f"{values.shape[0]} vs {n_trials}"
            )

    probabilities = probabilities[valid]
    first_index = first_index[valid]
    second_index = second_index[valid]
    if probabilities.size == 0:
        return probabilities, first_index, second_index
    n_categories = probabilities.shape[1]
    keep = (
        (first_index >= 0)
        & (first_index < n_categories)
        & (second_index >= 0)
        & (second_index < n_categories)
        & np.all(np.isfinite(probabilities), axis=1)
    )
    return probabilities[keep], first_index[keep], second_index[keep]


def _valid_trial_classification_data(
    metrics: Mapping[str, Any], target_key: str
) -> tuple[np.ndarray, np.ndarray]:
    probabilities, target_index, _ = _valid_trial_two_target_data(
        metrics, target_key, target_key
    )
    return probabilities, target_index


def choice_brier_loss(metrics: Mapping[str, Any]) -> float:
    probabilities, choice_index = _valid_trial_classification_data(
        metrics, "observed_choice_index"
    )
    if probabilities.size == 0:
        return float("nan")
    n_trials, n_categories = probabilities.shape
    one_hot = np.zeros((n_trials, n_categories), dtype=float)
    one_hot[np.arange(n_trials), choice_index] = 1.0
    return float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1)))


def choice_nll_loss(metrics: Mapping[str, Any], *, eps: float = 1e-12) -> float:
    probabilities, choice_index = _valid_trial_classification_data(
        metrics, "observed_choice_index"
    )
    if probabilities.size == 0:
        return float("nan")
    selected = probabilities[np.arange(probabilities.shape[0]), choice_index]
    return float(np.mean(-np.log(np.clip(selected, float(eps), 1.0))))


def wrong_choice_nll(metrics: Mapping[str, Any], *, eps: float = 1e-12) -> float:
    probabilities, choice_index, true_index = _valid_trial_two_target_data(
        metrics, "observed_choice_index", "true_category_index"
    )
    if probabilities.size == 0:
        return float("nan")
    wrong = choice_index != true_index
    if not np.any(wrong):
        return float("nan")
    probabilities = probabilities[wrong]
    choice_index = choice_index[wrong]
    selected = probabilities[np.arange(probabilities.shape[0]), choice_index]
    return float(np.mean(-np.log(np.clip(selected, float(eps), 1.0))))


def conditional_wrong_choice_nll(
    metrics: Mapping[str, Any], *, eps: float = 1e-12
) -> float:
    probabilities, choice_index, true_index = _valid_trial_two_target_data(
        metrics, "observed_choice_index", "true_category_index"
    )
    if probabilities.size == 0:
        return float("nan")
    wrong = choice_index != true_index
    if not np.any(wrong):
        return float("nan")
    probabilities = probabilities[wrong]
    choice_index = choice_index[wrong]
    true_index = true_index[wrong]
    row = np.arange(probabilities.shape[0])
    selected = probabilities[row, choice_index]
    true_probability = probabilities[row, true_index]
    wrong_mass = np.clip(1.0 - true_probability, float(eps), 1.0)
    conditional = np.clip(selected / wrong_mass, float(eps), 1.0)
    return float(np.mean(-np.log(conditional)))


def target_prob_brier(metrics: Mapping[str, Any]) -> float:
    probabilities = np.asarray(metrics["pred_category_probs"], dtype=float)
    targets = np.asarray(metrics.get("target_probs"), dtype=float)
    valid = np.asarray(metrics["valid_trial_mask"], dtype=bool)
    if probabilities.ndim != 2:
        raise ValueError(
            f"pred_category_probs must be 2-D, got shape {probabilities.shape}"
        )
    if targets.ndim != 2:
        raise ValueError(f"target_probs must be 2-D, got shape {targets.shape}")
    if probabilities.shape != targets.shape:
        raise ValueError(
            "target_probs shape does not match pred_category_probs: "
            f"{targets.shape} vs {probabilities.shape}"
        )
    if valid.shape[0] != probabilities.shape[0]:
        raise ValueError(
            "valid_trial_mask length does not match pred_category_probs rows: "
            f"{valid.shape[0]} vs {probabilities.shape[0]}"
        )
    finite = np.all(np.isfinite(probabilities), axis=1) & np.all(
        np.isfinite(targets), axis=1
    )
    keep = valid & finite
    if not np.any(keep):
        return float("nan")
    return float(
        np.mean(np.sum(np.square(probabilities[keep] - targets[keep]), axis=1))
    )


class AccuracyCurveMAELoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_CURVE_MAE

    def compute(self, metrics: Mapping[str, Any]) -> float:
        return accuracy_curve_mae(metrics)


class AccuracyCurveMSELoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_CURVE_MSE

    def compute(self, metrics: Mapping[str, Any]) -> float:
        return accuracy_curve_mse(metrics)


class AccuracyCurveFamilyMSELoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_CURVE_FAMILY_MSE

    def compute(self, metrics: Mapping[str, Any]) -> float:
        return accuracy_curve_family_mse(metrics)


class AccuracyCurveBerHuLoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_CURVE_BERHU

    def __init__(self, delta: float):
        if delta <= 0:
            raise ValueError(
                f"loss_delta must be > 0 for accuracy_curve_berhu, got {delta}"
            )
        self.delta = float(delta)

    def compute(self, metrics: Mapping[str, Any]) -> float:
        return accuracy_curve_berhu(metrics, delta=self.delta)


class AccuracyBrierLoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_BRIER

    def compute(self, metrics: Mapping[str, Any]) -> float:
        return accuracy_brier(metrics)


class AccuracyFamilyBrierLoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_FAMILY_BRIER

    def compute(self, metrics: Mapping[str, Any]) -> float:
        return accuracy_family_brier(metrics)


class AccuracyNLLLoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_NLL

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def compute(self, metrics: Mapping[str, Any]) -> float:
        return accuracy_nll(metrics, eps=self.eps)


class ChoiceBrierLoss(LossStrategy):
    name = LOSS_METRIC_CHOICE_BRIER

    def compute(self, metrics: Mapping[str, Any]) -> float:
        return choice_brier_loss(metrics)


class ChoiceNLLLoss(LossStrategy):
    name = LOSS_METRIC_CHOICE_NLL

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def compute(self, metrics: Mapping[str, Any]) -> float:
        return choice_nll_loss(metrics, eps=self.eps)


class WrongChoiceNLLLoss(LossStrategy):
    name = LOSS_METRIC_WRONG_CHOICE_NLL

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def compute(self, metrics: Mapping[str, Any]) -> float:
        return wrong_choice_nll(metrics, eps=self.eps)


class ConditionalWrongChoiceNLLLoss(LossStrategy):
    name = LOSS_METRIC_CONDITIONAL_WRONG_CHOICE_NLL

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def compute(self, metrics: Mapping[str, Any]) -> float:
        return conditional_wrong_choice_nll(metrics, eps=self.eps)


class TargetProbBrierLoss(LossStrategy):
    name = LOSS_METRIC_TARGET_PROB_BRIER

    def compute(self, metrics: Mapping[str, Any]) -> float:
        return target_prob_brier(metrics)


AccuracyBerHuLoss = AccuracyCurveBerHuLoss


def build_loss_strategy(loss_metric: str, loss_delta: float | None = None) -> LossStrategy:
    metric = str(loss_metric).strip().lower()
    strategies: dict[str, type[LossStrategy]] = {
        LOSS_METRIC_ACCURACY_CURVE_MAE: AccuracyCurveMAELoss,
        LOSS_METRIC_ACCURACY_CURVE_MSE: AccuracyCurveMSELoss,
        LOSS_METRIC_ACCURACY_CURVE_FAMILY_MSE: AccuracyCurveFamilyMSELoss,
        LOSS_METRIC_ACCURACY_BRIER: AccuracyBrierLoss,
        LOSS_METRIC_ACCURACY_FAMILY_BRIER: AccuracyFamilyBrierLoss,
        LOSS_METRIC_ACCURACY_NLL: AccuracyNLLLoss,
        LOSS_METRIC_CHOICE_BRIER: ChoiceBrierLoss,
        LOSS_METRIC_CHOICE_NLL: ChoiceNLLLoss,
        LOSS_METRIC_WRONG_CHOICE_NLL: WrongChoiceNLLLoss,
        LOSS_METRIC_CONDITIONAL_WRONG_CHOICE_NLL: ConditionalWrongChoiceNLLLoss,
        LOSS_METRIC_TARGET_PROB_BRIER: TargetProbBrierLoss,
    }
    if metric == LOSS_METRIC_ACCURACY_CURVE_BERHU:
        if loss_delta is None:
            raise ValueError(
                "loss_delta is required when loss_metric='accuracy_curve_berhu'"
            )
        return AccuracyCurveBerHuLoss(float(loss_delta))
    strategy = strategies.get(metric)
    if strategy is None:
        raise ValueError(f"Unsupported loss_metric '{loss_metric}'. Valid: {LOSS_METRIC_CHOICES}")
    return strategy()


def compute_loss_values(
    metrics: Mapping[str, Any], *, loss_delta: float | None = None
) -> dict[str, float]:
    """Compute all configured scalar losses available for one prediction mapping."""
    output: dict[str, float] = {}
    for metric in LOSS_METRIC_CHOICES:
        if metric == LOSS_METRIC_ACCURACY_CURVE_BERHU and loss_delta is None:
            continue
        try:
            value = float(build_loss_strategy(metric, loss_delta=loss_delta).compute(metrics))
        except Exception:
            value = float("nan")
        output[str(metric)] = value
    return output


def attach_loss_metrics(
    metrics: Mapping[str, Any],
    *,
    loss_metric: str,
    loss_delta: float | None = None,
) -> dict[str, Any]:
    """Return a copy with the selected objective and all available losses attached."""
    output = dict(metrics)
    strategy = build_loss_strategy(loss_metric, loss_delta=loss_delta)
    objective = float(strategy.compute(output))
    values = compute_loss_values(output, loss_delta=loss_delta)
    values[strategy.name] = objective
    output.update(
        {
            "mean_error": objective,
            "objective_error": objective,
            "loss_metric": strategy.name,
            "loss_values": values,
        }
    )
    output.update({f"loss_{name}": value for name, value in values.items()})
    if loss_delta is not None:
        output["loss_delta"] = float(loss_delta)
    return output


__all__ = [
    "ACCURACY_LOSS_METRIC_CHOICES",
    "CHOICE_LOSS_METRIC_CHOICES",
    "PROBABILISTIC_LOSS_METRIC_CHOICES",
    "LOSS_METRIC_CHOICES",
    "LOSS_METRIC_ACCURACY_CURVE_MAE",
    "LOSS_METRIC_ACCURACY_CURVE_MSE",
    "LOSS_METRIC_ACCURACY_CURVE_FAMILY_MSE",
    "LOSS_METRIC_ACCURACY_CURVE_BERHU",
    "LOSS_METRIC_ACCURACY_BRIER",
    "LOSS_METRIC_ACCURACY_FAMILY_BRIER",
    "LOSS_METRIC_ACCURACY_NLL",
    "LOSS_METRIC_CHOICE_BRIER",
    "LOSS_METRIC_CHOICE_NLL",
    "LOSS_METRIC_WRONG_CHOICE_NLL",
    "LOSS_METRIC_CONDITIONAL_WRONG_CHOICE_NLL",
    "LOSS_METRIC_TARGET_PROB_BRIER",
    "LOSS_METRIC_ACCURACY_MAE",
    "LOSS_METRIC_ACCURACY_MSE",
    "LOSS_METRIC_ACCURACY_BERHU",
    "LOSS_METRIC_MAE",
    "LOSS_METRIC_MSE",
    "LOSS_METRIC_BERHU",
    "LossStrategy",
    "AccuracyCurveMAELoss",
    "AccuracyCurveMSELoss",
    "AccuracyCurveFamilyMSELoss",
    "AccuracyCurveBerHuLoss",
    "AccuracyBerHuLoss",
    "AccuracyBrierLoss",
    "AccuracyFamilyBrierLoss",
    "AccuracyNLLLoss",
    "ChoiceBrierLoss",
    "ChoiceNLLLoss",
    "WrongChoiceNLLLoss",
    "ConditionalWrongChoiceNLLLoss",
    "TargetProbBrierLoss",
    "accuracy_curve_mae",
    "accuracy_curve_mse",
    "accuracy_curve_family_mse",
    "accuracy_curve_berhu",
    "accuracy_berhu",
    "accuracy_brier",
    "accuracy_family_brier",
    "accuracy_nll",
    "choice_brier_loss",
    "choice_nll_loss",
    "wrong_choice_nll",
    "conditional_wrong_choice_nll",
    "target_prob_brier",
    "build_loss_strategy",
    "compute_loss_values",
    "attach_loss_metrics",
]
