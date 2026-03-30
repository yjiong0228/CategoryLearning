"""Shared utilities for StateModel optimizers (grid / AMR)."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from .paths import PROCESSED_DATA_DIR, TASK2_PROCESSED_PATH


@dataclass
class GridPointResult:
    """Container for a single parameter combination evaluation."""

    params: Dict[str, Any]
    mean_error: float
    metrics: Dict[str, np.ndarray | float]
    posterior_log: Optional[Sequence[np.ndarray]] = None
    prior_log: Optional[Sequence[np.ndarray]] = None
    step_results: Optional[Sequence[Dict[str, Any]]] = None
    strategy_counts_log: Optional[Sequence[Dict[str, Any]]] = None
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
    metrics: Dict[str, np.ndarray | float]
    posterior_log: Optional[Sequence[np.ndarray]]
    prior_log: Optional[Sequence[np.ndarray]]
    step_log: Optional[Sequence[Dict[str, Any]]]
    strategy_counts_log: Optional[Sequence[Dict[str, Any]]]


class BaseStateOptimizer:
    """Common data preparation and subject slicing logic."""

    def __init__(
        self,
        engine_config: Dict[str, Any],
        processed_data_dir: Optional[Path | str] = None,
        n_jobs: int = 1,
    ) -> None:
        self._engine_config_template = deepcopy(engine_config)
        self._processed_data_dir = (
            Path(processed_data_dir).resolve()
            if processed_data_dir is not None
            else PROCESSED_DATA_DIR
        )
        self.learning_data: Optional[pd.DataFrame] = None
        self.n_jobs = n_jobs

    def prepare_data(self, data_path: Path | str = TASK2_PROCESSED_PATH) -> None:
        data_path = Path(data_path).resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        self.learning_data = pd.read_csv(data_path)

    def _get_subject_frame(self, subject_id: int, stop_at: float) -> pd.DataFrame:
        if self.learning_data is None:
            self.prepare_data()
        assert self.learning_data is not None

        subject_frame = self.learning_data[self.learning_data["iSub"] == subject_id]
        if subject_frame.empty:
            raise ValueError(f"Subject {subject_id} not found in dataset")

        stop_index = max(1, int(len(subject_frame) * stop_at + 0.5))
        return subject_frame.iloc[:stop_index].copy()

    def _extract_arrays(
        self,
        subject_frame: pd.DataFrame,
        max_trials: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        stimulus = subject_frame[["feature1", "feature2", "feature3", "feature4"]].to_numpy(dtype=float)
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


def compute_prediction_metrics(
    model,
    post_log: Sequence[np.ndarray],
    stimulus: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    categories: np.ndarray,
    window_size: int,
) -> Dict[str, np.ndarray | float]:
    partition = model.partition_model
    hypotheses = list(model.hypotheses_set)

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
    if post_arr.shape[0] != n_trials:
        raise ValueError(
            "Post log length does not match number of trials: "
            f"{post_arr.shape[0]} vs {n_trials}"
        )

    true_acc = (feedback == 1.0).astype(float)
    pred_acc = np.full(n_trials, np.nan, dtype=float)

    for trial_idx in range(n_trials):
        current_post = post_arr[trial_idx]
        weighted_prob = 0.0
        trial_slice = (
            [stimulus[trial_idx]],
            [choices[trial_idx]],
            [feedback[trial_idx]],
            [categories[trial_idx]],
        )
        for weight, hypo in zip(current_post, hypotheses):
            if weight <= 0:
                continue
            beta_for_hypo = float(engine_beta[hypo]) if hypo < len(engine_beta) else 10.0
            lik = partition.calc_trueprob_entry(
                hypo,
                trial_slice,
                beta_for_hypo,
                use_cached_dist=False,
            )
            weighted_prob += weight * float(np.ravel(lik)[0])
        pred_acc[trial_idx] = weighted_prob

    sliding_true_acc: List[float] = []
    sliding_pred_acc: List[float] = []
    sliding_pred_std: List[float] = []

    for start in range(1, n_trials - window_size + 2):
        end = start + window_size
        true_window = true_acc[start:end]
        pred_window = pred_acc[start:end]
        sliding_true_acc.append(float(np.mean(true_window)))
        sliding_pred_acc.append(float(np.nanmean(pred_window)))
        valid = pred_window[~np.isnan(pred_window)]
        if valid.size == 0:
            sliding_pred_std.append(np.nan)
        else:
            sliding_pred_std.append(float(np.sqrt(np.sum(valid * (1 - valid))) / window_size))

    error = np.abs(np.array(sliding_true_acc) - np.array(sliding_pred_acc))
    mean_error = float(np.nanmean(error)) if error.size else float("nan")

    return {
        "true_acc": true_acc,
        "pred_acc": pred_acc,
        "sliding_true_acc": np.asarray(sliding_true_acc, dtype=float),
        "sliding_pred_acc": np.asarray(sliding_pred_acc, dtype=float),
        "sliding_pred_acc_std": np.asarray(sliding_pred_std, dtype=float),
        "mean_error": mean_error,
    }


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


def evaluate_state_model_run(
    subject_id: int,
    condition: int,
    arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    params: Dict[str, Any],
    engine_config_template: Dict[str, Any],
    processed_data_dir: Path,
    window_size: int,
    keep_logs: bool = True,
    include_step_log: bool = False,
) -> SingleRunResult:
    """Run one parameter evaluation for StateModel and return normalized outputs."""
    stimulus, choices, feedback, categories = arrays
    trial_sequence = prepare_trial_sequence(stimulus, choices, feedback)

    from ..problems import StateModel

    engine_config = deepcopy(engine_config_template)
    inject_params(engine_config, params)
    model = StateModel(
        engine_config,
        condition=condition,
        subject_id=subject_id,
        processed_data_dir=processed_data_dir,
    )

    model.precompute_distances(stimulus)
    posterior_log, prior_log = model.fit_step_by_step(trial_sequence)
    step_log = getattr(model, "step_log", None) if include_step_log else None

    strategy_log = None
    hypo_mod = getattr(model.engine, "modules", {}).get("hypo_transitions_mod") if hasattr(model, "engine") else None
    if hypo_mod is not None and hasattr(hypo_mod, "strategy_counts_log"):
        strategy_log = getattr(hypo_mod, "strategy_counts_log")

    metrics = compute_prediction_metrics(
        model,
        posterior_log,
        stimulus,
        choices,
        feedback,
        categories,
        window_size,
    )

    if not keep_logs:
        posterior_log = None
        prior_log = None
        step_log = None
        strategy_log = None

    return SingleRunResult(
        params=dict(params),
        mean_error=float(metrics["mean_error"]),
        metrics=metrics,
        posterior_log=posterior_log,
        prior_log=prior_log,
        step_log=step_log,
        strategy_counts_log=strategy_log,
    )
