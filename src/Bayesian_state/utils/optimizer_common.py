"""Shared utilities for StateModel optimizers (grid / AMR)."""
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

LOSS_METRIC_ACCURACY_MAE = "accuracy_mae"
LOSS_METRIC_ACCURACY_MSE = "accuracy_mse"
LOSS_METRIC_ACCURACY_BERHU = "accuracy_berhu"
LOSS_METRIC_CHOICE_BRIER = "choice_brier"
LOSS_METRIC_CHOICE_NLL = "choice_nll"
LOSS_METRIC_WRONG_CHOICE_NLL = "wrong_choice_nll"
LOSS_METRIC_CONDITIONAL_WRONG_CHOICE_NLL = "conditional_wrong_choice_nll"

# Backward-compatible constant names for call sites that import the old symbols.
# The accepted config strings are the explicit names above.
LOSS_METRIC_MAE = LOSS_METRIC_ACCURACY_MAE
LOSS_METRIC_MSE = LOSS_METRIC_ACCURACY_MSE
LOSS_METRIC_BERHU = LOSS_METRIC_ACCURACY_BERHU
ACCURACY_LOSS_METRIC_CHOICES = (
    LOSS_METRIC_ACCURACY_MAE,
    LOSS_METRIC_ACCURACY_MSE,
    LOSS_METRIC_ACCURACY_BERHU,
)
CHOICE_LOSS_METRIC_CHOICES = (
    LOSS_METRIC_CHOICE_BRIER,
    LOSS_METRIC_CHOICE_NLL,
    LOSS_METRIC_WRONG_CHOICE_NLL,
    LOSS_METRIC_CONDITIONAL_WRONG_CHOICE_NLL,
)
LOSS_METRIC_CHOICES = ACCURACY_LOSS_METRIC_CHOICES + CHOICE_LOSS_METRIC_CHOICES
LOSS_METRIC_ALIASES = {
    "mae": LOSS_METRIC_ACCURACY_MAE,
    "mse": LOSS_METRIC_ACCURACY_MSE,
    "berhu": LOSS_METRIC_ACCURACY_BERHU,
}


def normalize_loss_metric(loss_metric: str) -> str:
    metric = str(loss_metric).strip().lower()
    return LOSS_METRIC_ALIASES.get(metric, metric)


class LossStrategy(ABC):
    name: str

    @abstractmethod
    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        raise NotImplementedError


# Accuracy-based losses compare predicted and observed correctness curves.
class AccuracyMAELoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_MAE

    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        true_acc = np.asarray(metrics["sliding_true_acc"], dtype=float)
        pred_acc = np.asarray(metrics["sliding_pred_acc"], dtype=float)
        err = np.abs(true_acc - pred_acc)
        return float(np.nanmean(err)) if err.size else float("nan")


class AccuracyMSELoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_MSE

    def compute(self, metrics: Dict[str, np.ndarray | float]) -> float:
        true_acc = np.asarray(metrics["sliding_true_acc"], dtype=float)
        pred_acc = np.asarray(metrics["sliding_pred_acc"], dtype=float)
        err = np.square(true_acc - pred_acc)
        return float(np.nanmean(err)) if err.size else float("nan")


class AccuracyBerHuLoss(LossStrategy):
    name = LOSS_METRIC_ACCURACY_BERHU

    def __init__(self, delta: float):
        if delta <= 0:
            raise ValueError(f"loss_delta must be > 0 for accuracy_berhu, got {delta}")
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


def build_loss_strategy(loss_metric: str, loss_delta: float | None = None) -> LossStrategy:
    metric = normalize_loss_metric(loss_metric)
    if metric == LOSS_METRIC_ACCURACY_MAE:
        return AccuracyMAELoss()
    if metric == LOSS_METRIC_ACCURACY_MSE:
        return AccuracyMSELoss()
    if metric == LOSS_METRIC_ACCURACY_BERHU:
        if loss_delta is None:
            raise ValueError("loss_delta is required when loss_metric='accuracy_berhu'")
        return AccuracyBerHuLoss(float(loss_delta))
    if metric == LOSS_METRIC_CHOICE_BRIER:
        return ChoiceBrierLoss()
    if metric == LOSS_METRIC_CHOICE_NLL:
        return ChoiceNLLLoss()
    if metric == LOSS_METRIC_WRONG_CHOICE_NLL:
        return WrongChoiceNLLLoss()
    if metric == LOSS_METRIC_CONDITIONAL_WRONG_CHOICE_NLL:
        return ConditionalWrongChoiceNLLLoss()
    raise ValueError(f"Unsupported loss_metric '{loss_metric}'. Valid: {LOSS_METRIC_CHOICES}")


@dataclass
class GridPointResult:
    """Container for a single parameter combination evaluation."""

    params: Dict[str, Any]
    mean_error: float
    metrics_by_mode: Dict[str, Dict[str, np.ndarray | float]]
    selection_prediction_mode: str
    posterior_log: Optional[Sequence[np.ndarray]] = None
    prior_log: Optional[Sequence[np.ndarray]] = None
    beta_log: Optional[Sequence[np.ndarray]] = None
    step_results: Optional[Sequence[Dict[str, Any]]] = None
    strategy_counts_log: Optional[Sequence[Dict[str, Any]]] = None
    raw_runs: Optional[Sequence[Dict[str, Any]]] = None
    raw_step_results: Optional[Sequence[Sequence[Dict[str, Any]]]] = None
    sample_errors: Optional[Sequence[float]] = None
    best_error: Optional[float] = None
    refit_mean_error: Optional[float] = None
    refit_std_error: Optional[float] = None
    representative_run_index: Optional[int] = None
    n_repeats: int = 1
    std_error: float = 0.0

    @property
    def gamma(self) -> float:
        return self.params.get("gamma", float("nan"))

    @property
    def w0(self) -> float:
        return self.params.get("w0", float("nan"))


@dataclass
class SingleRunResult:
    params: Dict[str, Any]
    mean_error: float
    metrics_by_mode: Dict[str, Dict[str, np.ndarray | float]]
    selection_prediction_mode: str
    loss_metric: str
    loss_delta: Optional[float]
    posterior_log: Optional[Sequence[np.ndarray]]
    prior_log: Optional[Sequence[np.ndarray]]
    beta_log: Optional[Sequence[np.ndarray]]
    step_log: Optional[Sequence[Dict[str, Any]]]
    strategy_counts_log: Optional[Sequence[Dict[str, Any]]]


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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        missing_features = [col for col in self._feature_columns if col not in subject_frame.columns]
        if missing_features:
            raise ValueError(
                "Dataset is missing configured feature columns: "
                + ", ".join(missing_features)
            )
        stimulus = subject_frame[self._feature_columns].to_numpy(dtype=float)
        choices = subject_frame["choice"].to_numpy(dtype=int)
        feedback = subject_frame["feedback"].to_numpy(dtype=float)
        categories = subject_frame["category"].to_numpy(dtype=int)

        if max_trials is not None:
            usable = min(max_trials, stimulus.shape[0])
            stimulus = stimulus[:usable]
            choices = choices[:usable]
            feedback = feedback[:usable]
            categories = categories[:usable]

        return stimulus, choices, feedback, categories

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


def _extract_distribution_from_step(
    step_item: Dict[str, Any],
    key: str,
    set_size: int,
    trial_idx: int,
) -> np.ndarray:
    if key not in step_item:
        raise ValueError(f"Missing {key} in step log at trial index {trial_idx}")
    dist = np.asarray(step_item[key], dtype=float)
    if dist.ndim != 1 or dist.shape[0] != set_size:
        raise ValueError(
            f"Invalid {key} shape at trial index {trial_idx}: "
            f"expected ({set_size},), got {dist.shape}"
        )
    return dist


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


def _compute_single_mode_metrics(
    mode: str,
    model,
    post_arr: np.ndarray,
    step_log: Sequence[Dict[str, Any]],
    stimulus: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    categories: np.ndarray,
    window_size: int,
    engine_beta: np.ndarray,
    hypotheses: Sequence[int],
) -> Dict[str, np.ndarray | float]:
    partition = model.partition_model
    distance_mode = getattr(model.engine, "distance_mode", "prototype")
    n_trials = len(feedback)
    n_features = int(stimulus.shape[1])
    n_cats = int(getattr(partition, "n_cats", int(np.nanmax(categories)) if len(categories) else 2))

    true_acc = (feedback == 1.0).astype(float)
    true_family_acc = _family_correct(categories, choices, n_cats)
    pred_acc = np.full(n_trials, np.nan, dtype=float)
    pred_family_acc = np.full(n_trials, np.nan, dtype=float)
    pred_category_probs = np.full((n_trials, n_cats), np.nan, dtype=float)
    true_category_index = np.asarray(categories, dtype=int) - 1
    observed_choice_index = np.asarray(choices, dtype=int) - 1
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
            current_dist = _extract_distribution_from_step(
                step_item=step_item,
                key="prior",
                set_size=len(hypotheses),
                trial_idx=trial_idx,
            )
        else:
            raise ValueError(f"Unexpected mode: {mode}")

        weighted_prob = 0.0
        weighted_family_prob = 0.0
        weighted_cat_prob = np.zeros(n_cats, dtype=float)
        trial_slice = (
            [perceived_stimulus],
            [choices[trial_idx]],
            [feedback[trial_idx]],
            [categories[trial_idx]],
        )
        category_idx = int(categories[trial_idx]) - 1
        family_idx = _family_indices(int(categories[trial_idx]), n_cats)
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
            weighted_prob += weight * float(prob_vec[category_idx])
            family_idx = family_idx[family_idx < prob_vec.shape[0]]
            if family_idx.size:
                weighted_family_prob += weight * float(np.sum(prob_vec[family_idx]))

        pred_acc[trial_idx] = weighted_prob
        pred_family_acc[trial_idx] = weighted_family_prob
        pred_category_probs[trial_idx, :] = weighted_cat_prob
        valid_trial_mask[trial_idx] = True

    sliding_true_acc: List[float] = []
    sliding_pred_acc: List[float] = []
    sliding_pred_std: List[float] = []
    sliding_true_family_acc: List[float] = []
    sliding_pred_family_acc: List[float] = []
    sliding_pred_family_std: List[float] = []

    for start in range(1, n_trials - window_size + 1):
        end = start + window_size
        true_window = true_acc[start:end]
        pred_window = pred_acc[start:end]
        true_family_window = true_family_acc[start:end]
        pred_family_window = pred_family_acc[start:end]
        sliding_true_acc.append(float(np.mean(true_window)))
        sliding_pred_acc.append(float(np.nanmean(pred_window)))
        valid = pred_window[~np.isnan(pred_window)]
        if valid.size == 0:
            sliding_pred_std.append(np.nan)
        else:
            sliding_pred_std.append(float(np.sqrt(np.sum(valid * (1 - valid))) / window_size))
        sliding_true_family_acc.append(float(np.mean(true_family_window)))
        sliding_pred_family_acc.append(float(np.nanmean(pred_family_window)))
        valid_family = pred_family_window[~np.isnan(pred_family_window)]
        if valid_family.size == 0:
            sliding_pred_family_std.append(np.nan)
        else:
            sliding_pred_family_std.append(
                float(np.sqrt(np.sum(valid_family * (1 - valid_family))) / window_size)
            )

    family_error = np.abs(np.array(sliding_true_family_acc) - np.array(sliding_pred_family_acc))
    family_mean_error = float(np.nanmean(family_error)) if family_error.size else float("nan")

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
        "family_mean_error": family_mean_error,
        "pred_category_probs": pred_category_probs,
        "true_category_index": true_category_index,
        "observed_choice_index": observed_choice_index,
        "valid_trial_mask": valid_trial_mask,
    }


def compute_prediction_metrics(
    model,
    post_log: Sequence[np.ndarray],
    step_log: Sequence[Dict[str, Any]],
    stimulus: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    categories: np.ndarray,
    window_size: int,
    prediction_mode: str,
    loss_metric: str,
    loss_delta: float | None = None,
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
            step_log=step_log,
            stimulus=stimulus,
            choices=choices,
            feedback=feedback,
            categories=categories,
            window_size=window_size,
            engine_beta=np.asarray(engine_beta, dtype=float),
            hypotheses=hypotheses,
        )
        objective_error = float(loss_strategy.compute(metrics))
        metrics["mean_error"] = objective_error
        metrics["objective_error"] = objective_error
        metrics["loss_metric"] = loss_strategy.name
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
    arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
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
) -> SingleRunResult:
    """Run one parameter evaluation for StateModel and return normalized outputs."""
    stimulus, choices, feedback, categories = arrays
    trial_sequence = prepare_trial_sequence(stimulus, choices, feedback)

    from ..problems import StateModel

    engine_config = deepcopy(engine_config_template)
    inject_params(engine_config, params)
    set_hypothesis_transition_seed(engine_config, run_seed)
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
    step_log = all_step_log if include_step_log else None

    strategy_log = None
    hypo_mod = getattr(model.engine, "modules", {}).get("hypo_transitions_mod") if hasattr(model, "engine") else None
    if hypo_mod is not None and hasattr(hypo_mod, "strategy_counts_log"):
        strategy_log = getattr(hypo_mod, "strategy_counts_log")

    beta_log = None
    beta_mod = getattr(model.engine, "modules", {}).get("beta_mod") if hasattr(model, "engine") else None
    if beta_mod is not None and hasattr(beta_mod, "beta_log"):
        beta_log = getattr(beta_mod, "beta_log")

    metrics_by_mode = compute_prediction_metrics(
        model,
        posterior_log,
        all_step_log,
        stimulus,
        choices,
        feedback,
        categories,
        window_size,
        prediction_mode=prediction_mode,
        loss_metric=loss_metric,
        loss_delta=loss_delta,
    )

    if selection_prediction_mode not in metrics_by_mode:
        raise ValueError(
            f"selection_prediction_mode '{selection_prediction_mode}' is unavailable. "
            f"Available: {tuple(metrics_by_mode.keys())}"
        )

    if not keep_logs:
        posterior_log = None
        prior_log = None
        beta_log = None
        step_log = None
        strategy_log = None

    selected_mean_error = float(metrics_by_mode[selection_prediction_mode]["mean_error"])

    return SingleRunResult(
        params=dict(params),
        mean_error=selected_mean_error,
        metrics_by_mode=metrics_by_mode,
        selection_prediction_mode=selection_prediction_mode,
        loss_metric=str(loss_metric).lower(),
        loss_delta=float(loss_delta) if loss_delta is not None else None,
        posterior_log=posterior_log,
        prior_log=prior_log,
        beta_log=beta_log,
        step_log=step_log,
        strategy_counts_log=strategy_log,
    )
