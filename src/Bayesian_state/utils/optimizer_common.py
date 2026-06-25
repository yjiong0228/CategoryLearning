"""Shared utilities for StateModel hyperparameter selection and simulations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Mapping

import numpy as np
import pandas as pd
from .paths import PROCESSED_DATA_DIR, TASK2_PROCESSED_PATH

PREDICTION_MODE_POSTERIOR_T_MINUS_1 = "posterior_t_minus_1"
PREDICTION_MODE_PRIOR_T = "prior_t"
PREDICTION_MODE_BOTH = "both"
PREDICTION_MODE_CHOICES = (
    PREDICTION_MODE_POSTERIOR_T_MINUS_1,
    PREDICTION_MODE_PRIOR_T,
    PREDICTION_MODE_BOTH,
)

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

# Short aliases for internal call sites. Configs should use the explicit names above.
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
PROBABILISTIC_LOSS_METRIC_CHOICES = (
    LOSS_METRIC_TARGET_PROB_BRIER,
)
LOSS_METRIC_CHOICES = (
    ACCURACY_LOSS_METRIC_CHOICES
    + CHOICE_LOSS_METRIC_CHOICES
    + PROBABILISTIC_LOSS_METRIC_CHOICES
)
SEED_MODULUS = 2 ** 32

OUTPUT_NOISE_TARGET_UNIFORM = "uniform"
OUTPUT_NOISE_TARGET_PREVIOUS_CHOICE = "previous_choice"
OUTPUT_NOISE_TARGET_LOSE_SHIFT = "lose_shift"
OUTPUT_NOISE_TARGET_CHOICES = (
    OUTPUT_NOISE_TARGET_UNIFORM,
    OUTPUT_NOISE_TARGET_PREVIOUS_CHOICE,
    OUTPUT_NOISE_TARGET_LOSE_SHIFT,
)
OUTPUT_NOISE_KWARG_KEYS = (
    "enabled",
    "base_lapse",
    "post_error_lapse",
    "low_accuracy_lapse",
    "low_accuracy_threshold",
    "recent_accuracy_window",
    "lapse_decay",
    "max_lapse",
    "lapse_target",
    "latent_volatility_lapse",
    "latent_volatility_power",
)


def _seedable(obj: Any) -> Any:
    """Convert common Python/numpy/path values to stable JSON seed payloads."""
    if isinstance(obj, np.ndarray):
        return [_seedable(x) for x in obj.tolist()]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, Mapping):
        return {str(k): _seedable(v) for k, v in sorted(obj.items(), key=lambda item: str(item[0]))}
    if isinstance(obj, (list, tuple)):
        return [_seedable(x) for x in obj]
    return obj


def stable_seed(payload: Any) -> int:
    """Derive a deterministic uint32 seed from a JSON-serializable payload."""
    encoded = json.dumps(_seedable(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % SEED_MODULUS


def derive_hyper_candidate_seed(
    hyper_base_seed: int,
    stage: str,
    combination_index: int,
    hyperparams: Mapping[str, Any],
    extra_context: Mapping[str, Any] | None = None,
) -> int:
    payload: Dict[str, Any] = {
        "seed_role": "hyper_candidate_seed",
        "hyper_base_seed": int(hyper_base_seed),
        "stage": str(stage),
        "combination_index": int(combination_index),
        "hyperparams": dict(hyperparams),
    }
    if extra_context:
        payload["extra_context"] = dict(extra_context)
    return stable_seed(payload)


def derive_simulation_point_seed(
    hyper_candidate_seed: int,
    subject_id: int,
    params: Mapping[str, Any],
) -> int:
    return stable_seed(
        {
            "seed_role": "simulation_point_seed",
            "hyper_candidate_seed": int(hyper_candidate_seed),
            "subject_id": int(subject_id),
            "params": dict(params),
        }
    )


def derive_trajectory_seed(
    simulation_point_seed: int,
    phase: str,
    repeat_index: int,
) -> int:
    return stable_seed(
        {
            "seed_role": "trajectory_seed",
            "simulation_point_seed": int(simulation_point_seed),
            "phase": str(phase),
            "repeat_index": int(repeat_index),
        }
    )


def derive_module_seed(
    trajectory_seed: int,
    module_name: str = "hypo_transitions_mod",
) -> int:
    return stable_seed(
        {
            "seed_role": "module_seed",
            "trajectory_seed": int(trajectory_seed),
            "module_name": str(module_name),
        }
    )


def inject_module_seed_from_trajectory(
    engine_config: Dict[str, Any],
    trajectory_seed: int | None,
    module_name: str = "hypo_transitions_mod",
) -> int | None:
    if trajectory_seed is None:
        return None
    module_seed = derive_module_seed(int(trajectory_seed), module_name=module_name)
    modules = engine_config.get("modules")
    if not isinstance(modules, dict) or module_name not in modules:
        return None
    module_cfg = modules[module_name]
    if not isinstance(module_cfg, dict):
        return None
    kwargs = module_cfg.setdefault("kwargs", {})
    if not isinstance(kwargs, dict):
        return None
    kwargs["module_seed"] = int(module_seed)
    return int(module_seed)


class LossStrategy(ABC):
    name: str

    @abstractmethod
    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        raise NotImplementedError


# Accuracy-based losses compare predicted and observed correctness.
class AccuracyCurveMAELoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_CURVE_MAE

    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        true_acc = np.asarray(metrics["sliding_true_acc"], dtype=float)
        pred_acc = np.asarray(metrics["sliding_pred_acc"], dtype=float)
        err = np.abs(true_acc - pred_acc)
        return float(np.nanmean(err)) if err.size else float("nan")


class AccuracyCurveMSELoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_CURVE_MSE

    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        true_acc = np.asarray(metrics["sliding_true_acc"], dtype=float)
        pred_acc = np.asarray(metrics["sliding_pred_acc"], dtype=float)
        err = np.square(true_acc - pred_acc)
        return float(np.nanmean(err)) if err.size else float("nan")


class AccuracyCurveFamilyMSELoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_CURVE_FAMILY_MSE

    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        true_acc = np.asarray(metrics["sliding_true_family_acc"], dtype=float)
        pred_acc = np.asarray(metrics["sliding_pred_family_acc"], dtype=float)
        err = np.square(true_acc - pred_acc)
        return float(np.nanmean(err)) if err.size else float("nan")


class AccuracyCurveBerHuLoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_CURVE_BERHU

    def __init__(self, delta: float):
        if delta <= 0:
            raise ValueError(f"loss_delta must be > 0 for accuracy_curve_berhu, got {delta}")
        self.delta = float(delta)

    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        true_acc = np.asarray(metrics["sliding_true_acc"], dtype=float)
        pred_acc = np.asarray(metrics["sliding_pred_acc"], dtype=float)
        abs_err = np.abs(true_acc - pred_acc)
        piecewise = np.where(
            abs_err <= self.delta,
            abs_err,
            (np.square(abs_err) + self.delta ** 2) / (2.0 * self.delta),
        )
        return float(np.nanmean(piecewise)) if piecewise.size else float("nan")


def _valid_trial_accuracy_data(
    metrics: Dict[str, np.ndarray | float],
) -> Tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(metrics["pred_category_probs"], dtype=float)
    true_idx = np.asarray(metrics["true_category_index"], dtype=int)
    true_acc = np.asarray(metrics["true_acc"], dtype=float)
    valid_mask = np.asarray(metrics["valid_trial_mask"], dtype=bool)
    if probs.ndim != 2:
        raise ValueError(f"pred_category_probs must be 2-D, got shape {probs.shape}")
    if valid_mask.shape[0] != probs.shape[0]:
        raise ValueError(
            "valid_trial_mask length does not match pred_category_probs rows: "
            f"{valid_mask.shape[0]} vs {probs.shape[0]}"
        )
    if true_idx.shape[0] != probs.shape[0]:
        raise ValueError(
            "true_category_index length does not match pred_category_probs rows: "
            f"{true_idx.shape[0]} vs {probs.shape[0]}"
        )
    if true_acc.shape[0] != probs.shape[0]:
        raise ValueError(
            "true_acc length does not match pred_category_probs rows: "
            f"{true_acc.shape[0]} vs {probs.shape[0]}"
        )

    probs = probs[valid_mask]
    true_idx = true_idx[valid_mask]
    true_acc = true_acc[valid_mask]
    if probs.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    n_cats = probs.shape[1]
    valid_true = (true_idx >= 0) & (true_idx < n_cats)
    finite_probs = np.all(np.isfinite(probs), axis=1)
    finite_acc = np.isfinite(true_acc)
    keep = valid_true & finite_probs & finite_acc
    if not np.any(keep):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    probs = probs[keep]
    true_idx = true_idx[keep]
    true_acc = np.clip(true_acc[keep], 0.0, 1.0)
    p_true = probs[np.arange(probs.shape[0]), true_idx]
    return p_true, true_acc


class AccuracyBrierLoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_BRIER

    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        p_true, true_acc = _valid_trial_accuracy_data(metrics)
        if p_true.size == 0:
            return float("nan")
        return float(np.mean(np.square(p_true - true_acc)))


def _valid_trial_family_accuracy_data(
    metrics: Dict[str, np.ndarray | float],
) -> Tuple[np.ndarray, np.ndarray]:
    pred_family_acc = np.asarray(metrics["pred_family_acc"], dtype=float)
    true_family_acc = np.asarray(metrics["true_family_acc"], dtype=float)
    valid_mask = np.asarray(metrics["valid_trial_mask"], dtype=bool)
    if valid_mask.shape[0] != pred_family_acc.shape[0]:
        raise ValueError(
            "valid_trial_mask length does not match pred_family_acc length: "
            f"{valid_mask.shape[0]} vs {pred_family_acc.shape[0]}"
        )
    if true_family_acc.shape[0] != pred_family_acc.shape[0]:
        raise ValueError(
            "true_family_acc length does not match pred_family_acc length: "
            f"{true_family_acc.shape[0]} vs {pred_family_acc.shape[0]}"
        )

    pred_family_acc = pred_family_acc[valid_mask]
    true_family_acc = true_family_acc[valid_mask]
    keep = np.isfinite(pred_family_acc) & np.isfinite(true_family_acc)
    if not np.any(keep):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return pred_family_acc[keep], np.clip(true_family_acc[keep], 0.0, 1.0)


class AccuracyFamilyBrierLoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_FAMILY_BRIER

    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        pred_family_acc, true_family_acc = _valid_trial_family_accuracy_data(metrics)
        if pred_family_acc.size == 0:
            return float("nan")
        return float(np.mean(np.square(pred_family_acc - true_family_acc)))


class AccuracyNLLLoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_NLL

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        p_true, true_acc = _valid_trial_accuracy_data(metrics)
        if p_true.size == 0:
            return float("nan")
        p_true = np.clip(p_true, self.eps, 1.0 - self.eps)
        return float(
            np.mean(
                -(true_acc * np.log(p_true) + (1.0 - true_acc) * np.log(1.0 - p_true))
            )
        )


# Choice-based losses compare predicted category probabilities with observed choices.
def _valid_trial_classification_data(
    metrics: Dict[str, np.ndarray | float],
    target_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    probs, target_idx, _ = _valid_trial_two_target_data(
        metrics,
        target_key,
        target_key,
    )
    return probs, target_idx


def _valid_trial_two_target_data(
    metrics: Dict[str, np.ndarray | float],
    first_target_key: str,
    second_target_key: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs = np.asarray(metrics["pred_category_probs"], dtype=float)
    first_idx = np.asarray(metrics[first_target_key], dtype=int)
    second_idx = np.asarray(metrics[second_target_key], dtype=int)
    valid_mask = np.asarray(metrics["valid_trial_mask"], dtype=bool)
    if probs.ndim != 2:
        raise ValueError(f"pred_category_probs must be 2-D, got shape {probs.shape}")
    if valid_mask.shape[0] != probs.shape[0]:
        raise ValueError(
            "valid_trial_mask length does not match pred_category_probs rows: "
            f"{valid_mask.shape[0]} vs {probs.shape[0]}"
        )
    if first_idx.shape[0] != probs.shape[0]:
        raise ValueError(
            f"{first_target_key} length does not match pred_category_probs rows: "
            f"{first_idx.shape[0]} vs {probs.shape[0]}"
        )
    if second_idx.shape[0] != probs.shape[0]:
        raise ValueError(
            f"{second_target_key} length does not match pred_category_probs rows: "
            f"{second_idx.shape[0]} vs {probs.shape[0]}"
        )

    probs = probs[valid_mask]
    first_idx = first_idx[valid_mask]
    second_idx = second_idx[valid_mask]
    if probs.size == 0:
        return probs, first_idx, second_idx

    n_cats = probs.shape[1]
    valid_first = (first_idx >= 0) & (first_idx < n_cats)
    valid_second = (second_idx >= 0) & (second_idx < n_cats)
    finite_probs = np.all(np.isfinite(probs), axis=1)
    keep = valid_first & valid_second & finite_probs
    return probs[keep], first_idx[keep], second_idx[keep]


class ChoiceBrierLoss(LossStrategy):
    name = LOSS_METRIC_CHOICE_BRIER

    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        probs, choice_idx = _valid_trial_classification_data(
            metrics, "observed_choice_index"
        )
        if probs.size == 0:
            return float("nan")
        n_trials, n_cats = probs.shape
        one_hot = np.zeros((n_trials, n_cats), dtype=float)
        one_hot[np.arange(n_trials), choice_idx] = 1.0
        return float(np.mean(np.sum(np.square(probs - one_hot), axis=1)))


class ChoiceNLLLoss(LossStrategy):
    name = LOSS_METRIC_CHOICE_NLL

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        probs, choice_idx = _valid_trial_classification_data(
            metrics, "observed_choice_index"
        )
        if probs.size == 0:
            return float("nan")
        p_choice = probs[np.arange(probs.shape[0]), choice_idx]
        p_choice = np.clip(p_choice, self.eps, 1.0)
        return float(np.mean(-np.log(p_choice)))


class WrongChoiceNLLLoss(LossStrategy):
    name = LOSS_METRIC_WRONG_CHOICE_NLL

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        probs, choice_idx, true_idx = _valid_trial_two_target_data(
            metrics,
            "observed_choice_index",
            "true_category_index",
        )
        if probs.size == 0:
            return float("nan")
        wrong_mask = choice_idx != true_idx
        if not np.any(wrong_mask):
            return float("nan")
        wrong_probs = probs[wrong_mask]
        wrong_choice_idx = choice_idx[wrong_mask]
        p_choice = wrong_probs[np.arange(wrong_probs.shape[0]), wrong_choice_idx]
        p_choice = np.clip(p_choice, self.eps, 1.0)
        return float(np.mean(-np.log(p_choice)))


class ConditionalWrongChoiceNLLLoss(LossStrategy):
    name = LOSS_METRIC_CONDITIONAL_WRONG_CHOICE_NLL

    def __init__(self, eps: float = 1e-12):
        self.eps = float(eps)

    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        probs, choice_idx, true_idx = _valid_trial_two_target_data(
            metrics,
            "observed_choice_index",
            "true_category_index",
        )
        if probs.size == 0:
            return float("nan")
        wrong_mask = choice_idx != true_idx
        if not np.any(wrong_mask):
            return float("nan")
        wrong_probs = probs[wrong_mask]
        wrong_choice_idx = choice_idx[wrong_mask]
        wrong_true_idx = true_idx[wrong_mask]
        row = np.arange(wrong_probs.shape[0])
        p_choice = wrong_probs[row, wrong_choice_idx]
        p_true = wrong_probs[row, wrong_true_idx]
        wrong_mass = np.clip(1.0 - p_true, self.eps, 1.0)
        conditional_p_choice = np.clip(p_choice / wrong_mass, self.eps, 1.0)
        return float(np.mean(-np.log(conditional_p_choice)))


class TargetProbBrierLoss(LossStrategy):
    name = LOSS_METRIC_TARGET_PROB_BRIER

    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        probs = np.asarray(metrics["pred_category_probs"], dtype=float)
        target_probs = np.asarray(metrics.get("target_probs"), dtype=float)
        valid_mask = np.asarray(metrics["valid_trial_mask"], dtype=bool)
        if probs.ndim != 2:
            raise ValueError(f"pred_category_probs must be 2-D, got shape {probs.shape}")
        if target_probs.ndim != 2:
            raise ValueError(f"target_probs must be 2-D, got shape {target_probs.shape}")
        if probs.shape != target_probs.shape:
            raise ValueError(
                "target_probs shape does not match pred_category_probs: "
                f"{target_probs.shape} vs {probs.shape}"
            )
        if valid_mask.shape[0] != probs.shape[0]:
            raise ValueError(
                "valid_trial_mask length does not match pred_category_probs rows: "
                f"{valid_mask.shape[0]} vs {probs.shape[0]}"
            )
        finite = np.all(np.isfinite(probs), axis=1) & np.all(np.isfinite(target_probs), axis=1)
        keep = valid_mask & finite
        if not np.any(keep):
            return float("nan")
        return float(np.mean(np.sum(np.square(probs[keep] - target_probs[keep]), axis=1)))


def build_loss_strategy(loss_metric: str, loss_delta: float | None = None) -> LossStrategy:
    metric = str(loss_metric).strip().lower()
    if metric == LOSS_METRIC_ACCURACY_CURVE_MAE:
        return AccuracyCurveMAELoss()
    if metric == LOSS_METRIC_ACCURACY_CURVE_MSE:
        return AccuracyCurveMSELoss()
    if metric == LOSS_METRIC_ACCURACY_CURVE_FAMILY_MSE:
        return AccuracyCurveFamilyMSELoss()
    if metric == LOSS_METRIC_ACCURACY_CURVE_BERHU:
        if loss_delta is None:
            raise ValueError("loss_delta is required when loss_metric='accuracy_curve_berhu'")
        return AccuracyCurveBerHuLoss(float(loss_delta))
    if metric == LOSS_METRIC_ACCURACY_BRIER:
        return AccuracyBrierLoss()
    if metric == LOSS_METRIC_ACCURACY_FAMILY_BRIER:
        return AccuracyFamilyBrierLoss()
    if metric == LOSS_METRIC_ACCURACY_NLL:
        return AccuracyNLLLoss()
    if metric == LOSS_METRIC_CHOICE_BRIER:
        return ChoiceBrierLoss()
    if metric == LOSS_METRIC_CHOICE_NLL:
        return ChoiceNLLLoss()
    if metric == LOSS_METRIC_WRONG_CHOICE_NLL:
        return WrongChoiceNLLLoss()
    if metric == LOSS_METRIC_CONDITIONAL_WRONG_CHOICE_NLL:
        return ConditionalWrongChoiceNLLLoss()
    if metric == LOSS_METRIC_TARGET_PROB_BRIER:
        return TargetProbBrierLoss()
    raise ValueError(f"Unsupported loss_metric '{loss_metric}'. Valid: {LOSS_METRIC_CHOICES}")


def compute_loss_values(
    metrics: Dict[str, np.ndarray | float],
    *,
    loss_delta: float | None = None,
) -> Dict[str, float]:
    """Compute all scalar loss/statistic values available for one metrics dict."""
    out: Dict[str, float] = {}
    for metric in LOSS_METRIC_CHOICES:
        if metric == LOSS_METRIC_ACCURACY_CURVE_BERHU and loss_delta is None:
            continue
        try:
            strategy = build_loss_strategy(metric, loss_delta=loss_delta)
            value = float(strategy.compute(metrics))
        except Exception:
            value = float("nan")
        out[str(metric)] = value
    return out


@dataclass
class SimulationResult:
    """Container for repeated simulations under one fixed parameter setting."""

    params: Dict[str, Any]
    mean_error: float
    metrics_by_mode: Dict[str, Dict[str, np.ndarray | float]]
    selection_prediction_mode: str
    state_log: Optional[Dict[str, Sequence[np.ndarray]]] = None
    trial_events: Optional[Sequence[Dict[str, Any]]] = None
    transition_counts: Optional[Sequence[Dict[str, Any]]] = None
    raw_runs: Optional[Sequence[Dict[str, Any]]] = None
    sample_errors: Optional[Sequence[float]] = None
    best_error: Optional[float] = None
    representative_run_index: Optional[int] = None
    simulation_repeats: int = 0
    simulation_point_seed: Optional[int] = None
    std_error: float = 0.0
    statistics_summary: Optional[Dict[str, Any]] = None

    @property
    def gamma(self) -> float:
        memory_kwargs = self.params.get("engine.modules.memory_mod.kwargs")
        if isinstance(memory_kwargs, Mapping) and "gamma" in memory_kwargs:
            return memory_kwargs["gamma"]
        return self.params.get(
            "gamma",
            self.params.get("engine.modules.memory_mod.kwargs.gamma", float("nan")),
        )

    @property
    def w0(self) -> float:
        memory_kwargs = self.params.get("engine.modules.memory_mod.kwargs")
        if isinstance(memory_kwargs, Mapping) and "w0" in memory_kwargs:
            return memory_kwargs["w0"]
        return self.params.get(
            "w0",
            self.params.get("engine.modules.memory_mod.kwargs.w0", float("nan")),
        )


@dataclass
class SingleRunResult:
    params: Dict[str, Any]
    mean_error: float
    metrics_by_mode: Dict[str, Dict[str, np.ndarray | float]]
    selection_prediction_mode: str
    loss_metric: str
    loss_delta: Optional[float]
    state_log: Optional[Dict[str, Sequence[np.ndarray]]] = None
    trial_events: Optional[Sequence[Dict[str, Any]]] = None
    transition_counts: Optional[Sequence[Dict[str, Any]]] = None
    simulation_point_seed: Optional[int] = None
    trajectory_seed: Optional[int] = None
    module_seed: Optional[int] = None
    seed_context: Optional[Dict[str, Any]] = None
    posterior_log: Optional[Any] = None
    prior_log: Optional[Any] = None
    beta_log: Optional[Any] = None
    step_log: Optional[Any] = None
    strategy_counts_log: Optional[Any] = None


@dataclass
class TrialArrays:
    """Subject trial arrays with optional hard and probabilistic targets."""

    stimulus: np.ndarray
    choices: np.ndarray
    feedback: np.ndarray
    categories: Optional[np.ndarray] = None
    target_probs: Optional[np.ndarray] = None


def _coerce_trial_arrays(arrays: TrialArrays | tuple | list) -> TrialArrays:
    if isinstance(arrays, TrialArrays):
        return arrays
    if not isinstance(arrays, (tuple, list)) or len(arrays) < 3:
        raise ValueError("arrays must be a TrialArrays instance or a tuple/list with at least 3 entries")
    categories = arrays[3] if len(arrays) >= 4 else None
    target_probs = arrays[4] if len(arrays) >= 5 else None
    return TrialArrays(
        stimulus=np.asarray(arrays[0], dtype=float),
        choices=np.asarray(arrays[1], dtype=int),
        feedback=np.asarray(arrays[2], dtype=float),
        categories=None if categories is None else np.asarray(categories, dtype=int),
        target_probs=None if target_probs is None else np.asarray(target_probs, dtype=float),
    )


def _normalize_probability_rows(values: np.ndarray, *, context: str) -> np.ndarray:
    probs = np.asarray(values, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"{context} must be a 2-D matrix, got shape {probs.shape}")
    if not np.all(np.isfinite(probs)):
        raise ValueError(f"{context} contains non-finite values")
    if np.any(probs < 0):
        raise ValueError(f"{context} contains negative values")
    denom = probs.sum(axis=1, keepdims=True)
    if np.any(denom <= 0):
        raise ValueError(f"{context} has rows that sum to zero")
    return probs / denom


def _probability_columns_from_frame(subject_frame: pd.DataFrame) -> list[str]:
    cols: list[tuple[int, str]] = []
    for col in subject_frame.columns:
        name = str(col)
        if not name.startswith("probCat"):
            continue
        suffix = name[len("probCat"):]
        if suffix.isdigit():
            cols.append((int(suffix), name))
    return [name for _, name in sorted(cols)]


class BaseStateOptimizer:
    """Common data preparation and subject slicing logic."""

    def __init__(
        self,
        engine_config: Dict[str, Any],
        processed_data_dir: Optional[Path | str] = None,
        n_jobs: int = 1,
        dataset_paths: Optional[Mapping[str, Path | str]] = None,
    ) -> None:
        self._engine_config_template = deepcopy(engine_config)
        self._processed_data_dir = (
            Path(processed_data_dir).resolve()
            if processed_data_dir is not None
            else PROCESSED_DATA_DIR
        )
        self._dataset_paths = dict(dataset_paths or {})
        self.learning_data: Optional[pd.DataFrame] = None
        self.n_jobs = n_jobs
        data_cfg = self._engine_config_template.get("data", {}) or {}
        self._feature_columns = list(
            data_cfg.get("feature_columns", ["feature1", "feature2", "feature3", "feature4"])
        )
        self._condition_column = str(data_cfg.get("condition_column", "condition"))
        self._subject_column = str(data_cfg.get("subject_column", "iSub"))
        self._category_column = str(data_cfg.get("category_column", "category"))
        self._target_type = str(data_cfg.get("target_type", "auto")).strip().lower()
        self._probability_columns = list(data_cfg.get("probability_columns", []))

    def prepare_data(self, data_path: Path | str = TASK2_PROCESSED_PATH) -> None:
        data_path = Path(data_path).resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        self.learning_data = pd.read_csv(data_path, encoding="utf-8-sig")

    def _get_subject_frame(self, subject_id: int, stop_at: float) -> pd.DataFrame:
        if self.learning_data is None:
            self.prepare_data()
        assert self.learning_data is not None

        if self._subject_column not in self.learning_data.columns:
            raise ValueError(f"Subject column '{self._subject_column}' not found in dataset")

        subject_frame = self.learning_data[self.learning_data[self._subject_column] == subject_id]
        if subject_frame.empty:
            raise ValueError(f"Subject {subject_id} not found in dataset")

        stop_index = max(1, int(len(subject_frame) * stop_at + 0.5))
        return subject_frame.iloc[:stop_index].copy()

    def _extract_arrays(
        self,
        subject_frame: pd.DataFrame,
        max_trials: Optional[int],
    ) -> TrialArrays:
        missing_features = [col for col in self._feature_columns if col not in subject_frame.columns]
        if missing_features:
            raise ValueError(
                "Dataset is missing configured feature columns: "
                + ", ".join(missing_features)
            )
        stimulus = subject_frame[self._feature_columns].to_numpy(dtype=float)
        choices = subject_frame["choice"].to_numpy(dtype=int)
        feedback = subject_frame["feedback"].to_numpy(dtype=float)

        probabilistic_target_types = {"probabilistic", "probability", "soft", "soft_category"}
        categories: Optional[np.ndarray] = None
        target_probs: Optional[np.ndarray] = None

        prob_cols = list(self._probability_columns)
        if not prob_cols:
            prob_cols = _probability_columns_from_frame(subject_frame)

        if self._target_type in probabilistic_target_types:
            if not prob_cols:
                raise ValueError(
                    "data.target_type is probabilistic, but no probability columns were configured "
                    "and no probCat* columns were found."
                )
        elif self._target_type not in {"auto", "hard", "category", "categorical"}:
            raise ValueError(
                "data.target_type must be auto, hard/category/categorical, or probabilistic/probability/soft"
            )

        if prob_cols:
            missing_probs = [col for col in prob_cols if col not in subject_frame.columns]
            if missing_probs:
                raise ValueError(
                    "Dataset is missing configured probability columns: "
                    + ", ".join(missing_probs)
                )
            target_probs = _normalize_probability_rows(
                subject_frame[prob_cols].to_numpy(dtype=float),
                context="target probability columns",
            )

        if self._target_type not in probabilistic_target_types and self._category_column in subject_frame.columns:
            categories = subject_frame[self._category_column].to_numpy(dtype=int)
        elif self._target_type in {"hard", "category", "categorical"}:
            raise ValueError(f"Dataset is missing configured category column: {self._category_column}")

        if max_trials is not None:
            usable = min(max_trials, stimulus.shape[0])
            stimulus = stimulus[:usable]
            choices = choices[:usable]
            feedback = feedback[:usable]
            if categories is not None:
                categories = categories[:usable]
            if target_probs is not None:
                target_probs = target_probs[:usable]

        return TrialArrays(
            stimulus=stimulus,
            choices=choices,
            feedback=feedback,
            categories=categories,
            target_probs=target_probs,
        )

    def _get_condition_value(self, subject_frame: pd.DataFrame) -> int:
        if self._condition_column in subject_frame.columns:
            return int(subject_frame[self._condition_column].iloc[0])
        if "ruleID" in subject_frame.columns:
            return int(subject_frame["ruleID"].iloc[0])
        return 1


def prepare_trial_sequence(
    stimulus: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
) -> List[List[float]]:
    trials: List[List[float]] = []
    for stim, choice, fb in zip(stimulus, choices, feedback):
        trial: List[float] = [stim, int(choice), float(fb)]
        trials.append(trial)
    return trials


def _get_prediction_modes(prediction_mode: str) -> List[str]:
    if prediction_mode not in PREDICTION_MODE_CHOICES:
        raise ValueError(
            f"Unsupported prediction_mode '{prediction_mode}'. "
            f"Valid values: {PREDICTION_MODE_CHOICES}"
        )
    if prediction_mode == PREDICTION_MODE_BOTH:
        return [PREDICTION_MODE_POSTERIOR_T_MINUS_1, PREDICTION_MODE_PRIOR_T]
    return [prediction_mode]


def _family_correct(categories: np.ndarray, choices: np.ndarray, n_cats: int) -> np.ndarray:
    if n_cats >= 4:
        category_family = np.where(np.isin(categories, [1, 2]), 0, 1)
        choice_family = np.where(np.isin(choices, [1, 2]), 0, 1)
        return (category_family == choice_family).astype(float)
    return (categories == choices).astype(float)


def _family_indices(category: int, n_cats: int) -> np.ndarray:
    category_idx = int(category) - 1
    if n_cats >= 4:
        if category_idx in (0, 1):
            return np.array([0, 1], dtype=int)
        return np.array([2, 3], dtype=int)
    return np.array([category_idx], dtype=int)


def _target_majority_indices(target_probs: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return the unique highest-probability category for each trial, or -1 for ties/missing."""
    if target_probs is None:
        return None
    probs = np.asarray(target_probs, dtype=float)
    if probs.ndim != 2 or probs.shape[0] == 0:
        return None
    finite = np.all(np.isfinite(probs), axis=1)
    max_prob = np.full(probs.shape[0], np.nan, dtype=float)
    max_prob[finite] = np.max(probs[finite], axis=1)
    is_max = np.isclose(probs, max_prob[:, None], rtol=0.0, atol=1e-12)
    unique = finite & (np.sum(is_max, axis=1) == 1)
    majority = np.full(probs.shape[0], -1, dtype=int)
    majority[unique] = np.argmax(probs[unique], axis=1)
    return majority


def _safe_nanmean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    keep = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(keep)) < 2:
        return float("nan")
    x = x[keep]
    y = y[keep]
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _mapping_get_path(root: Mapping[str, Any] | None, path: str) -> Any:
    curr: Any = root
    for part in path.split("."):
        if not isinstance(curr, Mapping) or part not in curr:
            return None
        curr = curr[part]
    return curr


def _float_from_mapping(
    values: Mapping[str, Any],
    key: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    raw = values.get(key, default)
    try:
        out = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"output_noise.kwargs.{key} must be numeric, got {raw!r}") from exc
    if not np.isfinite(out):
        raise ValueError(f"output_noise.kwargs.{key} must be finite, got {raw!r}")
    if min_value is not None and out < min_value:
        raise ValueError(
            f"output_noise.kwargs.{key} must be >= {min_value}, got {out!r}"
        )
    if max_value is not None and out > max_value:
        raise ValueError(
            f"output_noise.kwargs.{key} must be <= {max_value}, got {out!r}"
        )
    return out


def _bool_from_mapping(values: Mapping[str, Any], key: str, default: bool) -> bool:
    raw = values.get(key, default)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(raw, (int, float, np.integer, np.floating)):
        return bool(raw)
    raise ValueError(f"output_noise.kwargs.{key} must be boolean-like, got {raw!r}")


def _extract_output_noise_config(
    params: Mapping[str, Any] | None,
    engine_config: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    raw: Dict[str, Any] = {}
    sources = [
        _mapping_get_path(engine_config, "output_noise.kwargs"),
        _mapping_get_path(engine_config, "engine.output_noise.kwargs"),
        _mapping_get_path(params, "output_noise.kwargs"),
        _mapping_get_path(params, "engine.output_noise.kwargs"),
    ]
    for source in sources:
        if isinstance(source, Mapping):
            raw.update(dict(source))

    flat_sources = [params, engine_config]
    for source in flat_sources:
        if not isinstance(source, Mapping):
            continue
        for key in OUTPUT_NOISE_KWARG_KEYS:
            for prefix in ("engine.output_noise.kwargs.", "output_noise.kwargs."):
                full_key = f"{prefix}{key}"
                if full_key in source:
                    raw[key] = source[full_key]

    if not raw:
        return {"enabled": False}

    cfg = {
        "enabled": _bool_from_mapping(raw, "enabled", True),
        "base_lapse": _float_from_mapping(raw, "base_lapse", 0.0, min_value=0.0, max_value=1.0),
        "post_error_lapse": _float_from_mapping(
            raw, "post_error_lapse", 0.0, min_value=0.0, max_value=1.0
        ),
        "low_accuracy_lapse": _float_from_mapping(
            raw, "low_accuracy_lapse", 0.0, min_value=0.0, max_value=1.0
        ),
        "low_accuracy_threshold": _float_from_mapping(
            raw, "low_accuracy_threshold", 0.70, min_value=1e-9, max_value=1.0
        ),
        "recent_accuracy_window": int(raw.get("recent_accuracy_window", 8)),
        "lapse_decay": _float_from_mapping(raw, "lapse_decay", 0.0, min_value=0.0, max_value=1.0),
        "max_lapse": _float_from_mapping(raw, "max_lapse", 0.40, min_value=0.0, max_value=1.0),
        "lapse_target": str(raw.get("lapse_target", OUTPUT_NOISE_TARGET_UNIFORM)),
        "latent_volatility_lapse": _float_from_mapping(
            raw,
            "latent_volatility_lapse",
            0.0,
            min_value=0.0,
            max_value=1.0,
        ),
        "latent_volatility_power": _float_from_mapping(
            raw,
            "latent_volatility_power",
            1.0,
            min_value=1e-9,
        ),
    }
    if int(cfg["recent_accuracy_window"]) <= 0:
        raise ValueError(
            "output_noise.kwargs.recent_accuracy_window must be positive, "
            f"got {cfg['recent_accuracy_window']!r}"
        )
    if cfg["lapse_target"] not in OUTPUT_NOISE_TARGET_CHOICES:
        raise ValueError(
            "output_noise.kwargs.lapse_target must be one of "
            f"{OUTPUT_NOISE_TARGET_CHOICES}, got {cfg['lapse_target']!r}"
        )
    if cfg["max_lapse"] < cfg["base_lapse"]:
        raise ValueError(
            "output_noise.kwargs.max_lapse must be >= base_lapse, "
            f"got max_lapse={cfg['max_lapse']!r}, base_lapse={cfg['base_lapse']!r}"
        )
    has_lapse = (
        cfg["base_lapse"] > 0.0
        or cfg["post_error_lapse"] > 0.0
        or cfg["low_accuracy_lapse"] > 0.0
        or cfg["latent_volatility_lapse"] > 0.0
    )
    cfg["enabled"] = bool(cfg["enabled"] and has_lapse and cfg["max_lapse"] > 0.0)
    return cfg


def _normalize_probability_vector(values: np.ndarray, n_cats: int) -> np.ndarray:
    probs = np.asarray(values, dtype=float).reshape(-1)
    if probs.shape[0] != n_cats:
        raise ValueError(f"Probability vector width mismatch: expected {n_cats}, got {probs.shape[0]}")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0):
        return np.full(n_cats, 1.0 / max(1, n_cats), dtype=float)
    denom = float(np.sum(probs))
    if denom <= 0.0:
        return np.full(n_cats, 1.0 / max(1, n_cats), dtype=float)
    return probs / denom


def _one_hot_or_uniform(index: int, n_cats: int) -> np.ndarray:
    out = np.full(n_cats, 1.0 / max(1, n_cats), dtype=float)
    if 0 <= int(index) < n_cats:
        out[:] = 0.0
        out[int(index)] = 1.0
    return out


def _output_noise_target_vector(
    lapse_target: str,
    trial_idx: int,
    choices: np.ndarray,
    feedback: np.ndarray,
    n_cats: int,
) -> np.ndarray:
    uniform = np.full(n_cats, 1.0 / max(1, n_cats), dtype=float)
    if trial_idx <= 0:
        return uniform
    prev_choice_idx = int(choices[trial_idx - 1]) - 1
    prev_feedback = float(feedback[trial_idx - 1]) if np.isfinite(feedback[trial_idx - 1]) else 1.0
    if lapse_target == OUTPUT_NOISE_TARGET_PREVIOUS_CHOICE:
        return _one_hot_or_uniform(prev_choice_idx, n_cats)
    if lapse_target == OUTPUT_NOISE_TARGET_LOSE_SHIFT:
        if prev_feedback >= 1.0 or not (0 <= prev_choice_idx < n_cats):
            return uniform
        if n_cats == 2:
            return _one_hot_or_uniform(1 - prev_choice_idx, n_cats)
        out = np.ones(n_cats, dtype=float)
        out[prev_choice_idx] = 0.0
        denom = float(np.sum(out))
        return out / denom if denom > 0.0 else uniform
    return uniform


def _recent_feedback_accuracy(feedback: np.ndarray, trial_idx: int, window: int) -> float:
    start = max(0, int(trial_idx) - int(window))
    recent = np.asarray(feedback[start:trial_idx], dtype=float)
    recent = recent[np.isfinite(recent)]
    if recent.size == 0:
        return 1.0
    return float(np.clip(np.mean(recent), 0.0, 1.0))


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
    arr = np.asarray(values, dtype=float).reshape(-1)
    state = float(init_value)
    out = np.empty(arr.shape[0], dtype=float)
    for idx, value in enumerate(arr):
        if np.isfinite(value):
            state = float(alpha) * float(value) + (1.0 - float(alpha)) * state
        out[idx] = state
    return out


def _apply_output_noise_to_category_prob(
    category_prob: np.ndarray,
    *,
    trial_idx: int,
    choices: np.ndarray,
    feedback: np.ndarray,
    n_cats: int,
    output_noise_config: Mapping[str, Any],
    post_error_lapse_state: float,
    latent_volatility_value: float = 0.0,
) -> tuple[np.ndarray, float, float]:
    prob = _normalize_probability_vector(category_prob, n_cats)
    if not bool(output_noise_config.get("enabled", False)):
        return prob, 0.0, 0.0

    prev_feedback = float(feedback[trial_idx - 1]) if trial_idx > 0 and np.isfinite(feedback[trial_idx - 1]) else 1.0
    error_severity = float(np.clip(1.0 - prev_feedback, 0.0, 1.0))
    post_error_state = (
        float(output_noise_config["lapse_decay"]) * float(post_error_lapse_state)
        + float(output_noise_config["post_error_lapse"]) * error_severity
    )

    recent_acc = _recent_feedback_accuracy(
        feedback,
        trial_idx,
        int(output_noise_config["recent_accuracy_window"]),
    )
    threshold = float(output_noise_config["low_accuracy_threshold"])
    low_acc_scale = max(0.0, threshold - recent_acc) / max(threshold, 1e-12)
    low_acc_lapse = float(output_noise_config["low_accuracy_lapse"]) * low_acc_scale
    latent_value = float(np.clip(latent_volatility_value, 0.0, 1.0))
    latent_lapse = float(output_noise_config["latent_volatility_lapse"]) * (
        latent_value ** float(output_noise_config["latent_volatility_power"])
    )
    lapse = float(output_noise_config["base_lapse"]) + post_error_state + low_acc_lapse + latent_lapse
    lapse = float(np.clip(lapse, 0.0, float(output_noise_config["max_lapse"])))
    if lapse <= 0.0:
        return prob, 0.0, post_error_state

    target = _output_noise_target_vector(
        str(output_noise_config["lapse_target"]),
        trial_idx,
        choices,
        feedback,
        n_cats,
    )
    mixed = (1.0 - lapse) * prob + lapse * target
    return _normalize_probability_vector(mixed, n_cats), lapse, post_error_state


def _compute_single_mode_metrics(
    mode: str,
    model,
    post_arr: np.ndarray,
    prior_arr: np.ndarray,
    step_log: Sequence[Dict[str, Any]],
    stimulus: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    categories: Optional[np.ndarray],
    target_probs: Optional[np.ndarray],
    window_size: int,
    engine_beta: np.ndarray,
    hypotheses: Sequence[int],
    output_noise_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, np.ndarray | float]:
    partition = model.partition_model
    distance_mode = getattr(model.engine, "distance_mode", "prototype")
    n_trials = len(feedback)
    n_features = int(stimulus.shape[1])
    partition_n_cats = getattr(partition, "n_cats", None)
    if partition_n_cats is not None:
        n_cats = int(partition_n_cats)
    elif categories is not None and len(categories):
        n_cats = int(np.nanmax(categories))
    elif target_probs is not None:
        n_cats = int(target_probs.shape[1])
    else:
        n_cats = int(np.nanmax(choices)) if len(choices) else 2

    if target_probs is not None:
        target_probs = _normalize_probability_rows(target_probs, context="target_probs")
        if target_probs.shape[0] != n_trials:
            raise ValueError(
                "target_probs length does not match number of trials: "
                f"{target_probs.shape[0]} vs {n_trials}"
            )
        if target_probs.shape[1] != n_cats:
            raise ValueError(
                "target_probs category width does not match partition.n_cats: "
                f"{target_probs.shape[1]} vs {n_cats}"
            )

    true_acc = (feedback == 1.0).astype(float)
    has_categories = categories is not None
    true_family_acc = (
        _family_correct(categories, choices, n_cats)
        if has_categories
        else np.full(n_trials, np.nan, dtype=float)
    )
    pred_acc = np.full(n_trials, np.nan, dtype=float)
    pred_family_acc = np.full(n_trials, np.nan, dtype=float)
    pred_category_probs = np.full((n_trials, n_cats), np.nan, dtype=float)
    output_lapse_values = np.zeros(n_trials, dtype=float)
    output_noise_config = output_noise_config or {"enabled": False}
    latent_volatility_values = np.asarray(
        output_noise_config.get("latent_volatility", np.zeros(n_trials, dtype=float)),
        dtype=float,
    ).reshape(-1)
    post_error_lapse_state = 0.0
    true_category_index = (
        np.asarray(categories, dtype=int) - 1
        if has_categories
        else np.full(n_trials, -1, dtype=int)
    )
    target_prob_matrix = (
        np.asarray(target_probs, dtype=float)
        if target_probs is not None
        else np.full((n_trials, n_cats), np.nan, dtype=float)
    )
    observed_choice_index = np.asarray(choices, dtype=int) - 1
    target_majority_index = _target_majority_indices(target_prob_matrix)
    if target_majority_index is None:
        target_majority_index = np.full(n_trials, -1, dtype=int)
    target_majority_acc = np.full(n_trials, np.nan, dtype=float)
    pred_target_majority_acc = np.full(n_trials, np.nan, dtype=float)
    target_choice_valid = (
        (target_majority_index >= 0)
        & (observed_choice_index >= 0)
        & (observed_choice_index < n_cats)
    )
    target_majority_acc[target_choice_valid] = (
        observed_choice_index[target_choice_valid] == target_majority_index[target_choice_valid]
    ).astype(float)
    valid_trial_mask = np.zeros(n_trials, dtype=bool)

    for trial_idx in range(1, n_trials):
        step_item = step_log[trial_idx]
        if "perceived_stimulus" not in step_item:
            raise ValueError(f"Missing perceived_stimulus in step log at trial index {trial_idx}")
        perceived_stimulus = np.asarray(step_item["perceived_stimulus"], dtype=float)
        if perceived_stimulus.ndim != 1 or perceived_stimulus.shape[0] != n_features:
            raise ValueError(
                "Invalid perceived_stimulus shape at trial index "
                f"{trial_idx}: expected ({n_features},), got {perceived_stimulus.shape}"
            )

        if mode == PREDICTION_MODE_POSTERIOR_T_MINUS_1:
            current_dist = post_arr[trial_idx - 1]
        elif mode == PREDICTION_MODE_PRIOR_T:
            current_dist = prior_arr[trial_idx]
        else:
            raise ValueError(f"Unexpected mode: {mode}")

        weighted_cat_prob = np.zeros(n_cats, dtype=float)
        trial_slice = (
            [perceived_stimulus],
            [choices[trial_idx]],
            [feedback[trial_idx]],
        )
        category_idx = int(categories[trial_idx]) - 1 if has_categories else -1
        family_idx = _family_indices(int(categories[trial_idx]), n_cats) if has_categories else np.asarray([], dtype=int)
        for weight, hypo in zip(current_dist, hypotheses):
            if weight <= 0:
                continue
            beta_for_hypo = float(engine_beta[hypo]) if hypo < len(engine_beta) else 10.0
            prob = partition.get_category_probabilities(
                hypo,
                trial_slice,
                beta_for_hypo,
                distance_mode=distance_mode,
            )
            if prob.ndim == 1:
                prob = prob.reshape(-1, 1)
            prob_vec = np.asarray(prob[:, 0], dtype=float)
            if prob_vec.shape[0] != n_cats:
                raise ValueError(
                    f"Category probability shape mismatch at trial {trial_idx}: expected {n_cats}, got {prob_vec.shape[0]}"
                )
            weighted_cat_prob += weight * prob_vec

        weighted_cat_prob, output_lapse, post_error_lapse_state = _apply_output_noise_to_category_prob(
            weighted_cat_prob,
            trial_idx=trial_idx,
            choices=choices,
            feedback=feedback,
            n_cats=n_cats,
            output_noise_config=output_noise_config,
            post_error_lapse_state=post_error_lapse_state,
            latent_volatility_value=(
                float(latent_volatility_values[trial_idx])
                if trial_idx < latent_volatility_values.size and np.isfinite(latent_volatility_values[trial_idx])
                else 0.0
            ),
        )
        output_lapse_values[trial_idx] = output_lapse
        if has_categories:
            if 0 <= category_idx < weighted_cat_prob.shape[0]:
                pred_acc[trial_idx] = float(weighted_cat_prob[category_idx])
            valid_family_idx = family_idx[family_idx < weighted_cat_prob.shape[0]]
            if valid_family_idx.size:
                pred_family_acc[trial_idx] = float(np.sum(weighted_cat_prob[valid_family_idx]))
        else:
            choice_idx = int(choices[trial_idx]) - 1
            if 0 <= choice_idx < weighted_cat_prob.shape[0]:
                pred_acc[trial_idx] = float(weighted_cat_prob[choice_idx])
        majority_idx = int(target_majority_index[trial_idx])
        if 0 <= majority_idx < weighted_cat_prob.shape[0]:
            pred_target_majority_acc[trial_idx] = float(weighted_cat_prob[majority_idx])
        pred_category_probs[trial_idx, :] = weighted_cat_prob
        valid_trial_mask[trial_idx] = True

    sliding_true_acc: List[float] = []
    sliding_pred_acc: List[float] = []
    sliding_pred_std: List[float] = []
    sliding_true_family_acc: List[float] = []
    sliding_pred_family_acc: List[float] = []
    sliding_pred_family_std: List[float] = []
    sliding_target_majority_acc: List[float] = []
    sliding_pred_target_majority_acc: List[float] = []
    sliding_pred_target_majority_std: List[float] = []

    for start in range(1, n_trials - window_size + 1):
        end = start + window_size
        true_window = true_acc[start:end]
        pred_window = pred_acc[start:end]
        true_family_window = true_family_acc[start:end]
        pred_family_window = pred_family_acc[start:end]
        target_majority_window = target_majority_acc[start:end]
        pred_target_majority_window = pred_target_majority_acc[start:end]
        sliding_true_acc.append(float(np.mean(true_window)))
        sliding_pred_acc.append(float(np.nanmean(pred_window)))
        valid = pred_window[~np.isnan(pred_window)]
        if valid.size == 0:
            sliding_pred_std.append(np.nan)
        else:
            sliding_pred_std.append(float(np.sqrt(np.sum(valid * (1 - valid))) / window_size))
        sliding_true_family_acc.append(
            float(np.nanmean(true_family_window))
            if np.any(np.isfinite(true_family_window))
            else np.nan
        )
        sliding_pred_family_acc.append(
            float(np.nanmean(pred_family_window))
            if np.any(np.isfinite(pred_family_window))
            else np.nan
        )
        valid_family = pred_family_window[~np.isnan(pred_family_window)]
        if valid_family.size == 0:
            sliding_pred_family_std.append(np.nan)
        else:
            sliding_pred_family_std.append(
                float(np.sqrt(np.sum(valid_family * (1 - valid_family))) / window_size)
            )
        sliding_target_majority_acc.append(_safe_nanmean(target_majority_window))
        sliding_pred_target_majority_acc.append(_safe_nanmean(pred_target_majority_window))
        valid_target_majority = pred_target_majority_window[np.isfinite(pred_target_majority_window)]
        if valid_target_majority.size == 0:
            sliding_pred_target_majority_std.append(np.nan)
        else:
            denom = max(1, int(valid_target_majority.size))
            sliding_pred_target_majority_std.append(
                float(np.sqrt(np.sum(valid_target_majority * (1 - valid_target_majority))) / denom)
            )

    exp_alpha = float(2.0 / (float(window_size) + 1.0))
    chance_level = 1.0 / float(max(1, n_cats))
    exp_true_acc = exponential_smooth_curve(true_acc, alpha=exp_alpha, init_value=chance_level)
    exp_pred_acc = exponential_smooth_curve(pred_acc, alpha=exp_alpha, init_value=chance_level)
    exp_true_family_acc = exponential_smooth_curve(true_family_acc, alpha=exp_alpha, init_value=chance_level)
    exp_pred_family_acc = exponential_smooth_curve(pred_family_acc, alpha=exp_alpha, init_value=chance_level)
    exp_target_majority_acc = exponential_smooth_curve(target_majority_acc, alpha=exp_alpha, init_value=chance_level)
    exp_pred_target_majority_acc = exponential_smooth_curve(
        pred_target_majority_acc,
        alpha=exp_alpha,
        init_value=chance_level,
    )

    family_error = np.abs(np.array(sliding_true_family_acc) - np.array(sliding_pred_family_acc))
    finite_family_error = family_error[np.isfinite(family_error)]
    family_mean_error = float(np.mean(finite_family_error)) if finite_family_error.size else float("nan")
    target_prob_finite = (
        valid_trial_mask
        & np.all(np.isfinite(pred_category_probs), axis=1)
        & np.all(np.isfinite(target_prob_matrix), axis=1)
    )
    if np.any(target_prob_finite):
        target_prob_brier = float(
            np.mean(np.sum(np.square(pred_category_probs[target_prob_finite] - target_prob_matrix[target_prob_finite]), axis=1))
        )
        target_prob_corr_by_cat = np.asarray(
            [
                _safe_pearson(
                    pred_category_probs[target_prob_finite, cat_idx],
                    target_prob_matrix[target_prob_finite, cat_idx],
                )
                for cat_idx in range(n_cats)
            ],
            dtype=float,
        )
    else:
        target_prob_brier = float("nan")
        target_prob_corr_by_cat = np.full(n_cats, np.nan, dtype=float)
    latent_volatility_for_mean = latent_volatility_values[:n_trials]
    latent_volatility_mean = _safe_nanmean(
        latent_volatility_for_mean[valid_trial_mask[:latent_volatility_for_mean.size]]
    )

    return {
        "true_acc": true_acc,
        "pred_acc": pred_acc,
        "true_family_acc": true_family_acc,
        "pred_family_acc": pred_family_acc,
        "sliding_true_acc": np.asarray(sliding_true_acc, dtype=float),
        "sliding_pred_acc": np.asarray(sliding_pred_acc, dtype=float),
        "sliding_pred_acc_std": np.asarray(sliding_pred_std, dtype=float),
        "sliding_true_family_acc": np.asarray(sliding_true_family_acc, dtype=float),
        "sliding_pred_family_acc": np.asarray(sliding_pred_family_acc, dtype=float),
        "sliding_pred_family_acc_std": np.asarray(sliding_pred_family_std, dtype=float),
        "exp_true_acc": exp_true_acc,
        "exp_pred_acc": exp_pred_acc,
        "exp_true_family_acc": exp_true_family_acc,
        "exp_pred_family_acc": exp_pred_family_acc,
        "exp_accuracy_alpha": float(exp_alpha),
        "target_majority_acc": target_majority_acc,
        "pred_target_majority_acc": pred_target_majority_acc,
        "sliding_target_majority_acc": np.asarray(sliding_target_majority_acc, dtype=float),
        "sliding_pred_target_majority_acc": np.asarray(sliding_pred_target_majority_acc, dtype=float),
        "sliding_pred_target_majority_acc_std": np.asarray(sliding_pred_target_majority_std, dtype=float),
        "exp_target_majority_acc": exp_target_majority_acc,
        "exp_pred_target_majority_acc": exp_pred_target_majority_acc,
        "family_mean_error": family_mean_error,
        "pred_category_probs": pred_category_probs,
        "output_lapse": output_lapse_values,
        "output_lapse_mean": _safe_nanmean(output_lapse_values[valid_trial_mask]),
        "output_lapse_max": float(np.nanmax(output_lapse_values)) if output_lapse_values.size else float("nan"),
        "output_lapse_target": str(output_noise_config.get("lapse_target", OUTPUT_NOISE_TARGET_UNIFORM)),
        "latent_volatility": latent_volatility_values,
        "latent_volatility_mean": latent_volatility_mean,
        "latent_volatility_max": float(np.nanmax(latent_volatility_values)) if latent_volatility_values.size else float("nan"),
        "target_probs": target_prob_matrix,
        "target_prob_brier": target_prob_brier,
        "target_prob_corr_by_cat": target_prob_corr_by_cat,
        "target_prob_corr_cat1": float(target_prob_corr_by_cat[0]) if target_prob_corr_by_cat.size else float("nan"),
        "true_category_index": true_category_index,
        "observed_choice_index": observed_choice_index,
        "target_majority_index": target_majority_index,
        "valid_trial_mask": valid_trial_mask,
    }


def compute_prediction_metrics(
    model,
    post_log: Sequence[np.ndarray],
    prior_log: Sequence[np.ndarray],
    step_log: Sequence[Dict[str, Any]],
    stimulus: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    categories: Optional[np.ndarray],
    target_probs: Optional[np.ndarray],
    window_size: int,
    prediction_mode: str,
    loss_metric: str,
    loss_delta: float | None = None,
    output_noise_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Dict[str, np.ndarray | float]]:
    hypotheses = list(model.hypotheses_set)
    loss_strategy = build_loss_strategy(loss_metric, loss_delta=loss_delta)

    engine_beta = getattr(model.engine, "beta", None)
    if engine_beta is None:
        beta_param = 10.0
        if hasattr(model.engine, "likelihood_mod"):
            lik_mod = getattr(model.engine, "likelihood_mod")
            beta_param = float(lik_mod.kwargs.get("beta", 10.0))
        engine_beta = np.full(len(hypotheses), beta_param)

    post_arr = np.asarray(post_log, dtype=float)
    if post_arr.ndim == 1:
        post_arr = post_arr.reshape(1, -1)
    prior_arr = np.asarray(prior_log, dtype=float)
    if prior_arr.ndim == 1:
        prior_arr = prior_arr.reshape(1, -1)

    n_trials = len(feedback)
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    min_trials_for_window = window_size + 1
    if n_trials < min_trials_for_window:
        raise ValueError(
            "Not enough trials for sliding-window metrics with t-1 posterior alignment: "
            f"need at least {min_trials_for_window} trials, got {n_trials}"
        )
    if post_arr.shape[0] != n_trials:
        raise ValueError(
            "Post log length does not match number of trials: "
            f"{post_arr.shape[0]} vs {n_trials}"
        )
    if post_arr.shape[1] != len(hypotheses):
        raise ValueError(
            "Posterior width does not match hypothesis set size: "
            f"{post_arr.shape[1]} vs {len(hypotheses)}"
        )
    if prior_arr.shape[0] != n_trials:
        raise ValueError(
            "Prior log length does not match number of trials: "
            f"{prior_arr.shape[0]} vs {n_trials}"
        )
    if prior_arr.shape[1] != len(hypotheses):
        raise ValueError(
            "Prior width does not match hypothesis set size: "
            f"{prior_arr.shape[1]} vs {len(hypotheses)}"
        )
    if len(step_log) != n_trials:
        raise ValueError(
            "Step log length does not match number of trials: "
            f"{len(step_log)} vs {n_trials}"
        )

    metrics_by_mode: Dict[str, Dict[str, np.ndarray | float]] = {}
    for mode in _get_prediction_modes(prediction_mode):
        metrics = _compute_single_mode_metrics(
            mode=mode,
            model=model,
            post_arr=post_arr,
            prior_arr=prior_arr,
            step_log=step_log,
            stimulus=stimulus,
            choices=choices,
            feedback=feedback,
            categories=categories,
            target_probs=target_probs,
            window_size=window_size,
            engine_beta=np.asarray(engine_beta, dtype=float),
            hypotheses=hypotheses,
            output_noise_config=output_noise_config,
        )
        objective_error = float(loss_strategy.compute(metrics))
        loss_values = compute_loss_values(metrics, loss_delta=loss_delta)
        loss_values[loss_strategy.name] = objective_error
        metrics["mean_error"] = objective_error
        metrics["objective_error"] = objective_error
        metrics["loss_metric"] = loss_strategy.name
        metrics["loss_values"] = loss_values
        for key, value in loss_values.items():
            metrics[f"loss_{key}"] = value
        if loss_delta is not None:
            metrics["loss_delta"] = float(loss_delta)
        metrics_by_mode[mode] = metrics
    return metrics_by_mode


def inject_params(config: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Inject runtime params into engine config (supports dot-path and shortcuts)."""
    shortcuts = {
        "gamma": "modules.memory_mod.kwargs.gamma",
        "w0": "modules.memory_mod.kwargs.w0",
    }

    def set_by_path(root: Dict[str, Any], path: str, value: Any) -> None:
        parts = path.split(".")
        curr = root
        for part in parts[:-1]:
            curr = curr.setdefault(part, {})
        curr[parts[-1]] = value

    for key, value in params.items():
        if key == "beta":
            continue
        path = shortcuts.get(key, key)
        set_by_path(config, path, value)


def derive_run_seed(
    base_seed: int | None,
    subject_id: int,
    params: Mapping[str, Any],
    phase: str,
    repeat_index: int,
) -> int | None:
    """Derive a deterministic per-run seed from stable optimizer inputs."""
    if base_seed is None:
        return None
    payload = {
        "base_seed": int(base_seed),
        "subject_id": int(subject_id),
        "params": dict(params),
        "phase": str(phase),
        "repeat_index": int(repeat_index),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def set_hypothesis_transition_seed(engine_config: Dict[str, Any], seed: int | None) -> None:
    """Set the transition RNG seed only when the module is present."""
    if seed is None:
        return
    modules = engine_config.get("modules", {})
    if not isinstance(modules, dict) or "hypo_transitions_mod" not in modules:
        return
    hypo_cfg = modules.get("hypo_transitions_mod")
    if not isinstance(hypo_cfg, dict):
        return
    kwargs = hypo_cfg.setdefault("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("hypo_transitions_mod.kwargs must be a dictionary to set random_seed.")
    kwargs["random_seed"] = int(seed)


def get_hypothesis_transition_seed(engine_config: Mapping[str, Any]) -> int | None:
    modules = engine_config.get("modules", {})
    if not isinstance(modules, Mapping):
        return None
    hypo_cfg = modules.get("hypo_transitions_mod", {})
    if not isinstance(hypo_cfg, Mapping):
        return None
    kwargs = hypo_cfg.get("kwargs", {})
    if not isinstance(kwargs, Mapping) or "random_seed" not in kwargs:
        return None
    seed = kwargs.get("random_seed")
    return None if seed is None else int(seed)


def evaluate_state_model_run(
    subject_id: int,
    condition: int,
    arrays: TrialArrays | Tuple[np.ndarray, ...],
    params: Dict[str, Any],
    engine_config_template: Dict[str, Any],
    processed_data_dir: Path,
    window_size: int,
    dataset_paths: Optional[Mapping[str, Path | str]] = None,
    keep_logs: bool = True,
    include_step_log: bool = False,
    prediction_mode: str = PREDICTION_MODE_POSTERIOR_T_MINUS_1,
    selection_prediction_mode: str = PREDICTION_MODE_POSTERIOR_T_MINUS_1,
    loss_metric: str = LOSS_METRIC_MAE,
    loss_delta: float | None = None,
    run_seed: int | None = None,
    simulation_point_seed: int | None = None,
    trajectory_seed: int | None = None,
    seed_context: Optional[Mapping[str, Any]] = None,
) -> SingleRunResult:
    """Run one parameter evaluation for StateModel and return normalized outputs."""
    trial_arrays = _coerce_trial_arrays(arrays)
    stimulus = trial_arrays.stimulus
    choices = trial_arrays.choices
    feedback = trial_arrays.feedback
    categories = trial_arrays.categories
    target_probs = trial_arrays.target_probs
    trial_sequence = prepare_trial_sequence(stimulus, choices, feedback)

    from ..problems import StateModel

    engine_config = deepcopy(engine_config_template)
    inject_params(engine_config, params)
    effective_trajectory_seed = trajectory_seed if trajectory_seed is not None else run_seed
    module_seed = inject_module_seed_from_trajectory(engine_config, effective_trajectory_seed)
    if effective_trajectory_seed is not None:
        # Keep legacy modules that still call global np.random reproducible per trajectory.
        np.random.seed(int(effective_trajectory_seed))
    model = StateModel(
        engine_config,
        condition=condition,
        subject_id=subject_id,
        processed_data_dir=processed_data_dir,
        dataset_paths=dataset_paths,
    )

    posterior_log, prior_log = model.fit_step_by_step(trial_sequence)
    all_step_log = getattr(model, "step_log", None)
    if all_step_log is None:
        raise ValueError("StateModel.step_log is missing after fit_step_by_step")
    trial_events = all_step_log if include_step_log else None

    strategy_log = None
    latent_volatility_log = None
    hypo_mod = getattr(model.engine, "modules", {}).get("hypo_transitions_mod") if hasattr(model, "engine") else None
    if hypo_mod is not None and hasattr(hypo_mod, "strategy_counts_log"):
        strategy_log = getattr(hypo_mod, "strategy_counts_log")
    if hypo_mod is not None and hasattr(hypo_mod, "latent_volatility_log"):
        latent_volatility_log = getattr(hypo_mod, "latent_volatility_log")

    beta_log = None
    beta_mod = getattr(model.engine, "modules", {}).get("beta_mod") if hasattr(model, "engine") else None
    if beta_mod is not None and hasattr(beta_mod, "beta_log"):
        beta_log = getattr(beta_mod, "beta_log")

    output_noise_config = _extract_output_noise_config(params, engine_config)
    if latent_volatility_log is not None:
        latent_values: List[float] = []
        for item in latent_volatility_log:
            if isinstance(item, Mapping):
                raw_value = item.get("state", 0.0)
            else:
                raw_value = item
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = 0.0
            latent_values.append(value if np.isfinite(value) else 0.0)
        output_noise_config["latent_volatility"] = np.asarray(latent_values, dtype=float)
    metrics_by_mode = compute_prediction_metrics(
        model,
        posterior_log,
        prior_log,
        all_step_log,
        stimulus,
        choices,
        feedback,
        categories,
        target_probs,
        window_size,
        prediction_mode=prediction_mode,
        loss_metric=loss_metric,
        loss_delta=loss_delta,
        output_noise_config=output_noise_config,
    )

    if selection_prediction_mode not in metrics_by_mode:
        raise ValueError(
            f"selection_prediction_mode '{selection_prediction_mode}' is unavailable. "
            f"Available: {tuple(metrics_by_mode.keys())}"
        )

    if not keep_logs:
        state_log = None
        trial_events = None
        transition_counts = None
    else:
        state_log = {
            "posterior": posterior_log,
            "prior": prior_log,
            "beta": beta_log,
            "latent_volatility": latent_volatility_log,
        }
        transition_counts = strategy_log

    selected_mean_error = float(metrics_by_mode[selection_prediction_mode]["mean_error"])

    return SingleRunResult(
        params=dict(params),
        mean_error=selected_mean_error,
        metrics_by_mode=metrics_by_mode,
        selection_prediction_mode=selection_prediction_mode,
        loss_metric=str(loss_metric).lower(),
        loss_delta=float(loss_delta) if loss_delta is not None else None,
        state_log=state_log,
        trial_events=trial_events,
        transition_counts=transition_counts,
        simulation_point_seed=int(simulation_point_seed) if simulation_point_seed is not None else None,
        trajectory_seed=int(effective_trajectory_seed) if effective_trajectory_seed is not None else None,
        module_seed=module_seed,
        seed_context=dict(seed_context) if seed_context is not None else None,
    )
