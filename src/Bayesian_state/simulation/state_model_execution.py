"""Trial data, result objects, and single-run StateModel execution."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from ..metrics.losses import LOSS_METRIC_MAE, attach_loss_metrics
from ..metrics.trial_metrics import (
    build_prediction_metric_bundle,
)
from ..problems.modules.readout import (
    CHOICE_READOUT_EXPECTATION,
    CHOICE_READOUT_SHARPENED,
    OUTPUT_NOISE_TARGET_UNIFORM,
    apply_output_noise_to_category_prob as _apply_output_noise_to_category_prob,
    choice_readout_weights as _choice_readout_weights,
    resolve_choice_readout_config as _extract_choice_readout_config,
    resolve_output_noise_config as _extract_output_noise_config,
)
from ..utils.paths import PROCESSED_DATA_DIR, TASK2_PROCESSED_PATH
from ..utils.seeding import stable_seed

PREDICTION_MODE_POSTERIOR_T_MINUS_1 = "posterior_t_minus_1"
PREDICTION_MODE_PRIOR_T = "prior_t"
PREDICTION_MODE_BOTH = "both"
PREDICTION_MODE_CHOICES = (
    PREDICTION_MODE_POSTERIOR_T_MINUS_1,
    PREDICTION_MODE_PRIOR_T,
    PREDICTION_MODE_BOTH,
)


# Public run data
@dataclass
class SimulationResult:
    """Repeated runs under one fixed parameter setting."""

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
    """Normalized output of one trajectory or particle-marginal run."""

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


# Trial-data preparation
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


def sequential_importance_marginal(
    probability_stack: np.ndarray,
    observed_choice_index: Sequence[int] | np.ndarray,
    valid_trial_mask: Sequence[bool] | np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Causally marginalize stochastic state paths using observed choices.

    ``probability_stack`` has shape ``[n_particles, n_trials, n_categories]``.
    The prediction for trial ``t`` uses particle weights after trial ``t-1``.
    Only after producing that marginal prediction is the weight of each path
    multiplied by its probability for the observed choice on trial ``t``.

    Returns
    -------
    marginal_probabilities, effective_sample_size
        Trial-aligned arrays with shapes ``[n_trials, n_categories]`` and
        ``[n_trials]``.  ESS is recorded for the pre-choice weights used at
        each trial.
    """
    stack = np.asarray(probability_stack, dtype=float)
    if stack.ndim != 3 or stack.shape[0] <= 0 or stack.shape[2] <= 0:
        raise ValueError(
            "probability_stack must have shape [particles, trials, categories], "
            f"got {stack.shape}."
        )
    n_particles, n_trials, n_categories = stack.shape
    choices = np.asarray(observed_choice_index, dtype=int).reshape(-1)
    if choices.shape[0] != n_trials:
        raise ValueError(
            "observed_choice_index length does not match probability trials: "
            f"{choices.shape[0]} vs {n_trials}."
        )
    if valid_trial_mask is None:
        valid = np.ones(n_trials, dtype=bool)
    else:
        valid = np.asarray(valid_trial_mask, dtype=bool).reshape(-1)
        if valid.shape[0] != n_trials:
            raise ValueError(
                "valid_trial_mask length does not match probability trials: "
                f"{valid.shape[0]} vs {n_trials}."
            )

    log_weights = np.full(n_particles, -np.log(float(n_particles)), dtype=float)
    marginal = np.full((n_trials, n_categories), np.nan, dtype=float)
    effective_sample_size = np.full(n_trials, np.nan, dtype=float)

    for trial_idx in range(n_trials):
        trial_probability = stack[:, trial_idx, :]
        finite_particle = (
            np.all(np.isfinite(trial_probability), axis=1)
            & np.all(trial_probability >= 0.0, axis=1)
            & (np.sum(trial_probability, axis=1) > 0.0)
        )
        finite_weight = np.isfinite(log_weights)
        available = finite_particle & finite_weight
        if not np.any(available):
            log_weights[:] = -np.log(float(n_particles))
            continue

        normalized_rows = trial_probability[available]
        normalized_rows = normalized_rows / np.sum(
            normalized_rows,
            axis=1,
            keepdims=True,
        )
        available_log_weights = log_weights[available]
        available_log_weights -= float(np.max(available_log_weights))
        weights = np.exp(available_log_weights)
        weights /= float(np.sum(weights))
        marginal[trial_idx] = np.sum(weights[:, None] * normalized_rows, axis=0)
        effective_sample_size[trial_idx] = 1.0 / float(np.sum(np.square(weights)))

        choice = int(choices[trial_idx])
        if not bool(valid[trial_idx]) or not 0 <= choice < n_categories:
            continue
        row_sums = np.sum(trial_probability, axis=1)
        choice_probability = np.divide(
            trial_probability[:, choice],
            row_sums,
            out=np.zeros(n_particles, dtype=float),
            where=row_sums > 0.0,
        )
        choice_probability = np.clip(choice_probability, 1e-12, 1.0)
        choice_probability[~finite_particle] = 1e-12
        log_weights += np.log(choice_probability)
        finite_after = np.isfinite(log_weights)
        if not np.any(finite_after):
            log_weights[:] = -np.log(float(n_particles))
        else:
            log_weights -= float(np.max(log_weights[finite_after]))

    return marginal, effective_sample_size


def _build_single_mode_prediction_payload(
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
    choice_readout_config: Optional[Mapping[str, Any]] = None,
    readout_seed: int | None = None,
    score_trial_mask: Optional[Sequence[bool] | np.ndarray] = None,
) -> Dict[str, np.ndarray | float]:
    partition = model.partition_model
    distance_mode = getattr(model.engine, "distance_mode", "prototype")
    n_trials = len(feedback)
    if score_trial_mask is None:
        resolved_score_mask = np.ones(n_trials, dtype=bool)
    else:
        resolved_score_mask = np.asarray(score_trial_mask, dtype=bool).reshape(-1)
        if resolved_score_mask.shape[0] != n_trials:
            raise ValueError(
                "score_trial_mask length does not match number of trials: "
                f"{resolved_score_mask.shape[0]} vs {n_trials}"
            )
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

    pred_category_probs = np.full((n_trials, n_cats), np.nan, dtype=float)
    output_lapse_values = np.zeros(n_trials, dtype=float)
    output_noise_config = output_noise_config or {"enabled": False}
    choice_readout_config = choice_readout_config or {"method": CHOICE_READOUT_EXPECTATION}
    readout_rng = np.random.default_rng(0 if readout_seed is None else int(readout_seed))
    readout_selected_hypothesis = np.full(n_trials, -1, dtype=int)
    readout_switch = np.zeros(n_trials, dtype=bool)
    readout_confidence = np.full(n_trials, np.nan, dtype=float)
    readout_switch_probability = np.full(n_trials, np.nan, dtype=float)
    sticky_state: Dict[str, Any] = {}
    latent_volatility_values = np.asarray(
        output_noise_config.get("latent_volatility", np.zeros(n_trials, dtype=float)),
        dtype=float,
    ).reshape(-1)
    post_error_lapse_state = 0.0
    beta_arr = np.asarray(engine_beta, dtype=float)
    if beta_arr.ndim == 1:
        if beta_arr.shape[0] != len(hypotheses):
            raise ValueError(
                "engine_beta width does not match hypothesis set size: "
                f"{beta_arr.shape[0]} vs {len(hypotheses)}"
            )
    elif beta_arr.ndim == 2:
        if beta_arr.shape[0] != n_trials or beta_arr.shape[1] != len(hypotheses):
            raise ValueError(
                "engine_beta log shape does not match trials/hypotheses: "
                f"{beta_arr.shape} vs ({n_trials}, {len(hypotheses)})"
            )
    else:
        raise ValueError(f"engine_beta must be 1-D or 2-D, got shape {beta_arr.shape}")
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

        hypo_cat_probs = np.zeros((len(hypotheses), n_cats), dtype=float)
        beta_for_trial = beta_arr[trial_idx] if beta_arr.ndim == 2 else beta_arr
        trial_slice = (
            [perceived_stimulus],
            [choices[trial_idx]],
            [feedback[trial_idx]],
        )
        for hypo_arg, hypo in enumerate(hypotheses):
            beta_for_hypo = float(beta_for_trial[hypo]) if hypo < len(beta_for_trial) else 10.0
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
            hypo_cat_probs[hypo_arg, :] = prob_vec

        readout_weights, readout_log = _choice_readout_weights(
            current_dist,
            trial_idx=trial_idx,
            feedback=feedback,
            config=choice_readout_config,
            rng=readout_rng,
            sticky_state=sticky_state,
        )
        executed_hypothesis = step_item.get("executed_hypothesis")
        if executed_hypothesis is not None:
            executed_value = int(executed_hypothesis)
            try:
                selected_arg = [int(value) for value in hypotheses].index(
                    executed_value
                )
            except ValueError as exc:
                raise ValueError(
                    "step-log executed_hypothesis is outside the hypothesis set."
                ) from exc
            readout_weights = np.zeros(len(hypotheses), dtype=float)
            readout_weights[selected_arg] = 1.0
            readout_log.update(
                {
                    "selected_arg": int(selected_arg),
                    "switched": bool(
                        step_item.get("execution_switch_event", False)
                    ),
                    "switch_probability": float(
                        step_item.get("execution_switch_probability", 0.0)
                    ),
                    "persistent_execution_enabled": True,
                }
            )
        weighted_cat_prob = np.sum(readout_weights[:, None] * hypo_cat_probs, axis=0)
        selected_arg = int(readout_log.get("selected_arg", -1))
        if 0 <= selected_arg < len(hypotheses):
            readout_selected_hypothesis[trial_idx] = int(hypotheses[selected_arg])
        readout_switch[trial_idx] = bool(readout_log.get("switched", False))
        readout_confidence[trial_idx] = float(readout_log.get("confidence", np.nan))
        if "switch_probability" in readout_log:
            readout_switch_probability[trial_idx] = float(readout_log["switch_probability"])

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
        pred_category_probs[trial_idx, :] = weighted_cat_prob
        valid_trial_mask[trial_idx] = bool(resolved_score_mask[trial_idx])

    return build_prediction_metric_bundle(
        pred_category_probs,
        choices=choices,
        feedback=feedback,
        categories=categories,
        target_probabilities=target_probs,
        window_size=window_size,
        score_trial_mask=resolved_score_mask,
        valid_trial_mask=valid_trial_mask,
        diagnostics={
            "choice_readout_method": str(
                choice_readout_config.get("method", CHOICE_READOUT_EXPECTATION)
            ),
            "readout_selected_hypothesis": readout_selected_hypothesis,
            "readout_switch": readout_switch,
            "readout_confidence": readout_confidence,
            "readout_switch_probability": readout_switch_probability,
            "output_lapse": output_lapse_values,
            "output_lapse_target": str(
                output_noise_config.get("lapse_target", OUTPUT_NOISE_TARGET_UNIFORM)
            ),
            "latent_volatility": latent_volatility_values,
        },
    )


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
    choice_readout_config: Optional[Mapping[str, Any]] = None,
    readout_seed: int | None = None,
    beta_log: Optional[Sequence[np.ndarray]] = None,
    score_trial_mask: Optional[Sequence[bool] | np.ndarray] = None,
) -> Dict[str, Dict[str, np.ndarray | float]]:
    hypotheses = list(model.hypotheses_set)
    engine_beta = beta_log if beta_log is not None else getattr(model.engine, "beta", None)
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
        mode_readout_seed = stable_seed(
            {
                "seed_role": "choice_readout",
                "base": readout_seed,
                "mode": mode,
            }
        )
        metrics = _build_single_mode_prediction_payload(
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
            choice_readout_config=choice_readout_config,
            readout_seed=mode_readout_seed,
            score_trial_mask=score_trial_mask,
        )
        metrics_by_mode[mode] = attach_loss_metrics(
            metrics,
            loss_metric=loss_metric,
            loss_delta=loss_delta,
        )
    return metrics_by_mode


def compute_metrics_from_category_probabilities(
    probabilities: np.ndarray,
    *,
    choices: np.ndarray,
    feedback: np.ndarray,
    categories: Optional[np.ndarray],
    target_probs: Optional[np.ndarray],
    window_size: int,
    loss_metric: str,
    loss_delta: float | None = None,
    score_trial_mask: Optional[Sequence[bool] | np.ndarray] = None,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, np.ndarray | float]:
    """Build standard optimizer metrics from already-marginalized choices.

    Particle backends produce category probabilities directly and therefore do
    not need to reconstruct them from one representative posterior trajectory.
    This adapter keeps their output compatible with all standard loss and
    repeated-simulation statistics consumers.
    """

    probs = _normalize_probability_rows(
        np.asarray(probabilities, dtype=float),
        context="particle marginal probabilities",
    )
    n_trials, _ = probs.shape
    if target_probs is None:
        target_prob_matrix = None
    else:
        target_prob_matrix = _normalize_probability_rows(
            np.asarray(target_probs, dtype=float), context="target_probs"
        )
        if target_prob_matrix.shape != probs.shape:
            raise ValueError(
                "target_probs shape does not match particle probabilities: "
                f"{target_prob_matrix.shape} vs {probs.shape}."
            )

    metric_diagnostics: Dict[str, Any] = {
        "choice_readout_method": "particle_marginal",
        "readout_selected_hypothesis": np.full(n_trials, -1, dtype=int),
        "readout_switch": np.zeros(n_trials, dtype=bool),
        "readout_confidence": np.full(n_trials, np.nan, dtype=float),
        "readout_switch_probability": np.full(n_trials, np.nan, dtype=float),
        "output_lapse": np.zeros(n_trials, dtype=float),
        "output_lapse_target": OUTPUT_NOISE_TARGET_UNIFORM,
        "latent_volatility": np.zeros(n_trials, dtype=float),
    }
    if diagnostics:
        metric_diagnostics.update(dict(diagnostics))
    metrics = build_prediction_metric_bundle(
        probs,
        choices=choices,
        feedback=feedback,
        categories=categories,
        target_probabilities=target_prob_matrix,
        window_size=window_size,
        score_trial_mask=score_trial_mask,
        diagnostics=metric_diagnostics,
        mask_choice_prediction_by_validity=True,
        target_std_denominator="window",
    )
    return attach_loss_metrics(
        metrics,
        loss_metric=loss_metric,
        loss_delta=loss_delta,
    )


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


def _evaluate_state_model_particle_filter_run(
    *,
    subject_id: int,
    condition: int,
    trial_arrays: TrialArrays,
    params: Mapping[str, Any],
    engine_config: Dict[str, Any],
    processed_data_dir: Path,
    dataset_paths: Optional[Mapping[str, Path | str]],
    window_size: int,
    keep_logs: bool,
    prediction_mode: str,
    selection_prediction_mode: str,
    loss_metric: str,
    loss_delta: float | None,
    filter_seed: int | None,
    simulation_point_seed: int | None,
    seed_context: Optional[Mapping[str, Any]],
    score_trial_mask: Optional[Sequence[bool] | np.ndarray],
) -> SingleRunResult:
    if int(condition) != 1:
        raise ValueError("the current StateModel particle backend supports condition 1 only.")
    if prediction_mode not in {PREDICTION_MODE_PRIOR_T, PREDICTION_MODE_BOTH}:
        raise ValueError(
            "particle-filter prediction_mode must be 'prior_t' or 'both'; "
            "the marginal is a pre-choice prior_t prediction."
        )
    if selection_prediction_mode != PREDICTION_MODE_PRIOR_T:
        raise ValueError(
            "particle-filter selection_prediction_mode must be 'prior_t'."
        )

    readout = _extract_choice_readout_config(params, engine_config)
    method = str(readout.get("method", CHOICE_READOUT_EXPECTATION))
    if method not in {CHOICE_READOUT_EXPECTATION, CHOICE_READOUT_SHARPENED}:
        raise ValueError(
            "particle filtering currently supports expectation or "
            "sharpened_expectation choice readout."
        )
    if float(readout.get("weight_floor", 0.0)) != 0.0:
        raise ValueError("particle-filter choice readout does not support weight_floor.")
    readout_power = (
        float(readout.get("power", 1.0))
        if method == CHOICE_READOUT_SHARPENED
        else 1.0
    )
    strategy_confidence_gain = float(
        readout.get("strategy_confidence_gain", 0.0)
    )

    output_noise = _extract_output_noise_config(params, engine_config)
    unsupported_lapse = (
        float(output_noise.get("post_error_lapse", 0.0))
        + float(output_noise.get("low_accuracy_lapse", 0.0))
        + float(output_noise.get("latent_volatility_lapse", 0.0))
    )
    if unsupported_lapse > 0.0:
        raise ValueError(
            "particle filtering currently accepts only output_noise.base_lapse; "
            "history-dependent lapse must be represented by a stateful module."
        )
    if str(output_noise.get("lapse_target", OUTPUT_NOISE_TARGET_UNIFORM)) != OUTPUT_NOISE_TARGET_UNIFORM:
        raise ValueError("particle-filter output lapse must target the uniform distribution.")
    output_lapse = (
        float(output_noise.get("base_lapse", 0.0))
        if bool(output_noise.get("enabled", False))
        else 0.0
    )

    from ..inference_engine.dispatcher import (
        BACKEND_PARTICLE_FILTER,
        run_inference_backend,
    )

    resolved_seed = int(filter_seed if filter_seed is not None else 20260806)
    result = run_inference_backend(
        engine_config=engine_config,
        subject_id=int(subject_id),
        condition=int(condition),
        stimulus=trial_arrays.stimulus,
        choices=trial_arrays.choices,
        feedback=trial_arrays.feedback,
        inference_seed=resolved_seed,
        choice_readout_power=readout_power,
        strategy_confidence_gain=strategy_confidence_gain,
        output_lapse=output_lapse,
        valid_trial_mask=np.ones(trial_arrays.choices.size, dtype=bool),
        processed_data_dir=processed_data_dir,
        dataset_paths=dataset_paths,
    )
    result.require_backend(BACKEND_PARTICLE_FILTER)
    probabilities = np.asarray(
        result.observation_probabilities[PREDICTION_MODE_PRIOR_T], dtype=float
    )
    state_probabilities = result.state_probabilities
    latent = result.latent_summaries
    diagnostics = result.diagnostics
    audit_state_log: dict[str, Any] = {}
    for key in (
        "audit_hypothesis_map",
        "audit_adaptive_sharpening",
        "audit_exploration_lapse",
        "audit_unsharpened_expectation",
        "audit_sharpened_no_lapse",
        "audit_strategy_confidence_no_lapse",
        "audit_persistent_execution_no_lapse",
    ):
        value = result.observation_probabilities.get(key)
        if value is not None:
            audit_state_log[key] = value
    for key in (
        "audit_particle_correct_q10",
        "audit_particle_correct_q50",
        "audit_particle_correct_q90",
        "audit_correct_predicting_available_probability",
        "audit_correct_predicting_prior_mass",
        "audit_best_active_correct_probability",
    ):
        value = diagnostics.get(key)
        if value is not None:
            audit_state_log[key] = value
    ancestral_paths = result.artifacts.get("audit_ancestral_paths")
    if isinstance(ancestral_paths, Mapping):
        for key, value in ancestral_paths.items():
            audit_state_log[f"audit_ancestral_{key}"] = value
    diagnostic_arrays = {
        "particle_pre_choice_ess": diagnostics["pre_choice_ess"],
        "particle_post_choice_ess": diagnostics["post_choice_ess"],
        "particle_resampled": diagnostics["resampled"],
        "particle_unique_ancestors": diagnostics["resampling_unique_ancestors"],
        "particle_transition_rate": latent["transition_rate"],
        "particle_search_range": latent["search_range"],
        "particle_swap_probability": latent["swap_probability"],
        "particle_swap_event_probability": latent["swap_event_probability"],
        "particle_replacement_count": latent["replacement_count"],
        "particle_replacement_fraction": latent["replacement_fraction"],
        "particle_removed_mass": latent["removed_mass"],
        "particle_newcomer_distance": latent["newcomer_distance"],
        "particle_feedback_surprise": latent["feedback_surprise"],
        "particle_feedback_uncertainty": latent["feedback_uncertainty"],
        "particle_predictive_transition_rate": latent["predictive_transition_rate"],
        "particle_predictive_search_range": latent["predictive_search_range"],
        "particle_predictive_swap_probability": latent["predictive_swap_probability"],
        "particle_predictive_swap_event_probability": latent[
            "predictive_swap_event_probability"
        ],
        "particle_predictive_replacement_fraction": latent[
            "predictive_replacement_fraction"
        ],
        "particle_predictive_newcomer_distance": latent[
            "predictive_newcomer_distance"
        ],
        "particle_predictive_strategy_exploit": latent[
            "predictive_strategy_exploit"
        ],
        "particle_predictive_strategy_local_explore": latent[
            "predictive_strategy_local_explore"
        ],
        "particle_predictive_strategy_global_explore": latent[
            "predictive_strategy_global_explore"
        ],
        "particle_predictive_failure_pressure": latent[
            "predictive_failure_pressure"
        ],
        "particle_predictive_mastery_evidence": latent[
            "predictive_mastery_evidence"
        ],
        "particle_predictive_choice_confidence_signal": latent[
            "predictive_choice_confidence_signal"
        ],
        "particle_predictive_strategy_choice_precision": latent[
            "predictive_strategy_choice_precision"
        ],
        "particle_predictive_exploration_target": latent[
            "predictive_exploration_target"
        ],
        "particle_predictive_global_target": latent[
            "predictive_global_target"
        ],
        "particle_predictive_prior_reset_strength": latent[
            "predictive_prior_reset_strength"
        ],
        "particle_predictive_prior_reset_mass_shift": latent[
            "predictive_prior_reset_mass_shift"
        ],
        "particle_predictive_execution_switch_probability": latent[
            "predictive_execution_switch_probability"
        ],
        "particle_predictive_execution_switch_event_probability": latent[
            "predictive_execution_switch_event_probability"
        ],
        "particle_predictive_execution_dwell_trials": latent[
            "predictive_execution_dwell_trials"
        ],
        "particle_predictive_misconception_capture_eligible_probability": latent[
            "predictive_misconception_capture_eligible_probability"
        ],
        "particle_predictive_misconception_capture_hold_probability": latent[
            "predictive_misconception_capture_hold_probability"
        ],
        "particle_predictive_misconception_capture_switch_event_probability": latent[
            "predictive_misconception_capture_switch_event_probability"
        ],
        "particle_predictive_executed_choice_compatibility": latent[
            "predictive_executed_choice_compatibility"
        ],
        "particle_predictive_best_alternative_choice_compatibility": latent[
            "predictive_best_alternative_choice_compatibility"
        ],
        "particle_predictive_executed_beta": latent[
            "predictive_executed_beta"
        ],
        "particle_filtered_executed_beta": latent[
            "filtered_executed_beta"
        ],
        "particle_execution_switch_event_probability": latent[
            "execution_switch_event_probability"
        ],
        "particle_execution_dwell_trials": latent[
            "execution_dwell_trials"
        ],
        "particle_count": int(result.metadata["particle_count"]),
        "filter_seed": int(result.metadata["filter_seed"]),
        "choice_readout_method": method,
        "choice_readout_strategy_confidence_gain": strategy_confidence_gain,
        "output_lapse": np.full(trial_arrays.choices.size, output_lapse, dtype=float),
        "output_lapse_mean": float(output_lapse),
        "output_lapse_max": float(output_lapse),
    }
    if "executed_probability" in state_probabilities:
        diagnostic_arrays["particle_executed_probability"] = state_probabilities[
            "executed_probability"
        ]
        diagnostic_arrays["particle_filtered_executed_probability"] = (
            state_probabilities["filtered_executed_probability"]
        )
    metrics = compute_metrics_from_category_probabilities(
        probabilities,
        choices=trial_arrays.choices,
        feedback=trial_arrays.feedback,
        categories=trial_arrays.categories,
        target_probs=trial_arrays.target_probs,
        window_size=int(window_size),
        loss_metric=loss_metric,
        loss_delta=loss_delta,
        score_trial_mask=score_trial_mask,
        diagnostics=diagnostic_arrays,
    )
    metrics_by_mode = {PREDICTION_MODE_PRIOR_T: metrics}
    transition_counts = None
    state_log = None
    if keep_logs:
        transition_counts = [
            {
                "trial_index": int(index),
                "predictive_m": float(latent["predictive_transition_rate"][index]),
                "predictive_g": float(latent["predictive_search_range"][index]),
                "predictive_swap_probability": float(
                    latent["predictive_swap_probability"][index]
                ),
                "predictive_swap_event_probability": float(
                    latent["predictive_swap_event_probability"][index]
                ),
                "strategy_exploit": float(
                    latent["predictive_strategy_exploit"][index]
                ),
                "strategy_local_explore": float(
                    latent["predictive_strategy_local_explore"][index]
                ),
                "strategy_global_explore": float(
                    latent["predictive_strategy_global_explore"][index]
                ),
                "failure_pressure": float(
                    latent["predictive_failure_pressure"][index]
                ),
                "mastery_evidence": float(
                    latent["predictive_mastery_evidence"][index]
                ),
                "strategy_confidence_signal": float(
                    latent["predictive_choice_confidence_signal"][index]
                ),
                "strategy_choice_precision": float(
                    latent["predictive_strategy_choice_precision"][index]
                ),
                "exploration_target": float(
                    latent["predictive_exploration_target"][index]
                ),
                "global_target": float(
                    latent["predictive_global_target"][index]
                ),
                "prior_reset_strength": float(
                    latent["predictive_prior_reset_strength"][index]
                ),
                "prior_reset_mass_shift": float(
                    latent["predictive_prior_reset_mass_shift"][index]
                ),
                "execution_switch_probability": float(
                    latent["predictive_execution_switch_probability"][index]
                ),
                "execution_switch_event_probability": float(
                    latent[
                        "predictive_execution_switch_event_probability"
                    ][index]
                ),
                "execution_dwell_trials": float(
                    latent["predictive_execution_dwell_trials"][index]
                ),
                "misconception_capture_eligible_probability": float(
                    latent[
                        "predictive_misconception_capture_eligible_probability"
                    ][index]
                ),
                "misconception_capture_hold_probability": float(
                    latent[
                        "predictive_misconception_capture_hold_probability"
                    ][index]
                ),
                "misconception_capture_switch_event_probability": float(
                    latent[
                        "predictive_misconception_capture_switch_event_probability"
                    ][index]
                ),
                "executed_choice_compatibility": float(
                    latent["predictive_executed_choice_compatibility"][index]
                ),
                "best_alternative_choice_compatibility": float(
                    latent["predictive_best_alternative_choice_compatibility"][index]
                ),
                "executed_beta": float(
                    latent["predictive_executed_beta"][index]
                ),
                "executed_hypothesis_mode": (
                    int(np.argmax(state_probabilities["executed_probability"][index]))
                    if "executed_probability" in state_probabilities
                    else -1
                ),
                "replacement_count": float(latent["replacement_count"][index]),
                "replacement_fraction": float(
                    latent["replacement_fraction"][index]
                ),
                "removed_mass": float(latent["removed_mass"][index]),
                "newcomer_distance": float(
                    latent["newcomer_distance"][index]
                ),
                "feedback_surprise": float(latent["feedback_surprise"][index]),
                "feedback_uncertainty": float(latent["feedback_uncertainty"][index]),
                "pre_choice_ess": float(diagnostics["pre_choice_ess"][index]),
                "post_choice_ess": float(diagnostics["post_choice_ess"][index]),
                "resampled": bool(diagnostics["resampled"][index]),
                "active_total": float(
                    np.sum(state_probabilities["active_probability"][index])
                ),
                "strategies": [],
            }
            for index in range(trial_arrays.choices.size)
        ]
        state_log = {
            "marginal_prior": state_probabilities["hypothesis_prior"],
            "marginal_active_probability": state_probabilities["active_probability"],
            "transition_rate": latent["transition_rate"],
            "search_range": latent["search_range"],
            "swap_probability": latent["swap_probability"],
            "swap_event_probability": latent["swap_event_probability"],
            "replacement_count": latent["replacement_count"],
            "replacement_fraction": latent["replacement_fraction"],
            "removed_mass": latent["removed_mass"],
            "newcomer_distance": latent["newcomer_distance"],
            "feedback_surprise": latent["feedback_surprise"],
            "feedback_uncertainty": latent["feedback_uncertainty"],
            "predictive_transition_rate": latent["predictive_transition_rate"],
            "predictive_search_range": latent["predictive_search_range"],
            "predictive_swap_probability": latent["predictive_swap_probability"],
            "predictive_swap_event_probability": latent[
                "predictive_swap_event_probability"
            ],
            "predictive_replacement_fraction": latent[
                "predictive_replacement_fraction"
            ],
            "predictive_newcomer_distance": latent[
                "predictive_newcomer_distance"
            ],
            "predictive_strategy_exploit": latent["predictive_strategy_exploit"],
            "predictive_strategy_local_explore": latent[
                "predictive_strategy_local_explore"
            ],
            "predictive_strategy_global_explore": latent[
                "predictive_strategy_global_explore"
            ],
            "predictive_failure_pressure": latent[
                "predictive_failure_pressure"
            ],
            "predictive_mastery_evidence": latent[
                "predictive_mastery_evidence"
            ],
            "predictive_choice_confidence_signal": latent[
                "predictive_choice_confidence_signal"
            ],
            "predictive_strategy_choice_precision": latent[
                "predictive_strategy_choice_precision"
            ],
            "predictive_exploration_target": latent[
                "predictive_exploration_target"
            ],
            "predictive_global_target": latent[
                "predictive_global_target"
            ],
            "predictive_prior_reset_strength": latent[
                "predictive_prior_reset_strength"
            ],
            "predictive_prior_reset_mass_shift": latent[
                "predictive_prior_reset_mass_shift"
            ],
            "predictive_execution_switch_probability": latent[
                "predictive_execution_switch_probability"
            ],
            "predictive_execution_switch_event_probability": latent[
                "predictive_execution_switch_event_probability"
            ],
            "predictive_execution_dwell_trials": latent[
                "predictive_execution_dwell_trials"
            ],
            "predictive_misconception_capture_eligible_probability": latent[
                "predictive_misconception_capture_eligible_probability"
            ],
            "predictive_misconception_capture_hold_probability": latent[
                "predictive_misconception_capture_hold_probability"
            ],
            "predictive_misconception_capture_switch_event_probability": latent[
                "predictive_misconception_capture_switch_event_probability"
            ],
            "predictive_executed_choice_compatibility": latent[
                "predictive_executed_choice_compatibility"
            ],
            "predictive_best_alternative_choice_compatibility": latent[
                "predictive_best_alternative_choice_compatibility"
            ],
            "predictive_executed_beta": latent["predictive_executed_beta"],
            "filtered_executed_beta": latent["filtered_executed_beta"],
            "execution_switch_event_probability": latent[
                "execution_switch_event_probability"
            ],
            "execution_dwell_trials": latent["execution_dwell_trials"],
            "pre_choice_ess": diagnostics["pre_choice_ess"],
            "post_choice_ess": diagnostics["post_choice_ess"],
            "resampled": diagnostics["resampled"],
            **audit_state_log,
        }
        if "executed_probability" in state_probabilities:
            state_log.update(
                {
                    "marginal_executed_probability": state_probabilities[
                        "executed_probability"
                    ],
                    "filtered_executed_probability": state_probabilities[
                        "filtered_executed_probability"
                    ],
                }
            )

    return SingleRunResult(
        params=dict(params),
        mean_error=float(metrics["mean_error"]),
        metrics_by_mode=metrics_by_mode,
        selection_prediction_mode=PREDICTION_MODE_PRIOR_T,
        loss_metric=str(loss_metric).lower(),
        loss_delta=float(loss_delta) if loss_delta is not None else None,
        state_log=state_log,
        trial_events=None,
        transition_counts=transition_counts,
        simulation_point_seed=(
            int(simulation_point_seed) if simulation_point_seed is not None else None
        ),
        trajectory_seed=resolved_seed,
        module_seed=resolved_seed,
        seed_context=dict(seed_context) if seed_context is not None else None,
        # The filter exposes a predictive marginal prior, not a post-feedback
        # marginal posterior.  Keep the semantic distinction explicit.
        posterior_log=None,
        prior_log=state_probabilities["hypothesis_prior"] if keep_logs else None,
        strategy_counts_log=transition_counts,
    )


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
    score_trial_mask: Optional[Sequence[bool] | np.ndarray] = None,
) -> SingleRunResult:
    """Run one parameter evaluation for StateModel and return normalized outputs."""
    trial_arrays = _coerce_trial_arrays(arrays)
    stimulus = trial_arrays.stimulus
    choices = trial_arrays.choices
    feedback = trial_arrays.feedback
    categories = trial_arrays.categories
    target_probs = trial_arrays.target_probs
    engine_config = deepcopy(engine_config_template)
    inject_params(engine_config, params)
    effective_trajectory_seed = trajectory_seed if trajectory_seed is not None else run_seed
    if effective_trajectory_seed is not None:
        # Keep legacy modules that still call global np.random reproducible per trajectory.
        np.random.seed(int(effective_trajectory_seed))
    from ..inference_engine.dispatcher import (
        BACKEND_PARTICLE_FILTER,
        resolve_inference_backend,
        run_inference_backend,
    )

    backend_config = resolve_inference_backend(engine_config)
    if backend_config.backend == BACKEND_PARTICLE_FILTER:
        return _evaluate_state_model_particle_filter_run(
            subject_id=subject_id,
            condition=condition,
            trial_arrays=trial_arrays,
            params=params,
            engine_config=engine_config,
            processed_data_dir=processed_data_dir,
            dataset_paths=dataset_paths,
            window_size=window_size,
            keep_logs=keep_logs,
            prediction_mode=prediction_mode,
            selection_prediction_mode=selection_prediction_mode,
            loss_metric=loss_metric,
            loss_delta=loss_delta,
            filter_seed=effective_trajectory_seed,
            simulation_point_seed=simulation_point_seed,
            seed_context=seed_context,
            score_trial_mask=score_trial_mask,
        )
    inference_result = run_inference_backend(
        engine_config=engine_config,
        subject_id=int(subject_id),
        condition=condition,
        stimulus=stimulus,
        choices=choices,
        feedback=feedback,
        inference_seed=effective_trajectory_seed,
        processed_data_dir=processed_data_dir,
        dataset_paths=dataset_paths,
    )
    inference_result.require_backend("trajectory")
    model = inference_result.artifacts["model"]
    posterior_log = inference_result.state_probabilities["hypothesis_posterior"]
    prior_log = inference_result.state_probabilities["hypothesis_prior"]
    all_step_log = inference_result.diagnostics["step_log"]
    module_seed = inference_result.metadata.get("module_seed")
    trial_events = all_step_log if include_step_log else None
    strategy_log = inference_result.latent_summaries.get("transition_events")
    latent_volatility_log = inference_result.latent_summaries.get("latent_volatility")
    beta_log = inference_result.latent_summaries.get("beta")

    output_noise_config = _extract_output_noise_config(params, engine_config)
    choice_readout_config = _extract_choice_readout_config(params, engine_config)
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
        choice_readout_config=choice_readout_config,
        beta_log=beta_log,
        score_trial_mask=score_trial_mask,
        readout_seed=stable_seed(
            {
                "seed_role": "choice_readout_run",
                "trajectory_seed": effective_trajectory_seed,
                "subject_id": int(subject_id),
                "params": params,
            }
        ),
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


__all__ = [
    "PREDICTION_MODE_BOTH",
    "PREDICTION_MODE_CHOICES",
    "PREDICTION_MODE_POSTERIOR_T_MINUS_1",
    "PREDICTION_MODE_PRIOR_T",
    "BaseStateOptimizer",
    "SimulationResult",
    "SingleRunResult",
    "TrialArrays",
    "compute_metrics_from_category_probabilities",
    "compute_prediction_metrics",
    "derive_run_seed",
    "evaluate_state_model_run",
    "get_hypothesis_transition_seed",
    "inject_params",
    "prepare_trial_sequence",
    "sequential_importance_marginal",
    "set_hypothesis_transition_seed",
]
