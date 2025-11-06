"""Grid search utilities for tuning StateModel memory parameters."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..problems import StateModel
from ..utils.base import LOGGER

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = _REPO_ROOT / "data" / "processed" / "Task2_processed.csv"


@dataclass(slots=True)
class GridPointResult:
    """Container for a single (gamma, w0) evaluation."""

    gamma: float
    w0: float
    mean_error: float
    metrics: Dict[str, np.ndarray | float]
    posterior_log: Sequence[np.ndarray]


def _prepare_trial_sequence(
    stimulus: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
) -> List[List[float]]:
    trials: List[List[float]] = []
    for stim, choice, fb in zip(stimulus, choices, feedback):
        trial: List[float] = [stim, int(choice), float(fb)]
        trials.append(trial)
    return trials


def _compute_prediction_metrics(
    model: StateModel,
    posterior_log: Sequence[np.ndarray],
    stimulus: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    categories: np.ndarray,
    gamma: float,
    w0: float,
    window_size: int,
) -> Dict[str, np.ndarray | float]:
    partition = model.partition_model
    hypotheses = list(model.hypotheses_set)
    beta_param = float(
        getattr(model.engine, "likelihood_mod", None).kwargs.get("beta", 10.0)
        if hasattr(model.engine, "likelihood_mod")
        else 10.0
    )

    post_arr = np.asarray(posterior_log, dtype=float)
    if post_arr.ndim == 1:
        post_arr = post_arr.reshape(1, -1)

    n_trials = len(feedback)
    if post_arr.shape[0] != n_trials:
        raise ValueError(
            "Posterior log length does not match number of trials: "
            f"{post_arr.shape[0]} vs {n_trials}"
        )

    true_acc = (feedback == 1.0).astype(float)
    pred_acc = np.full(n_trials, np.nan, dtype=float)

    for trial_idx in range(1, n_trials):
        prev_post = post_arr[trial_idx - 1]
        weighted_prob = 0.0

        trial_slice = (
            [stimulus[trial_idx]],
            [choices[trial_idx]],
            [feedback[trial_idx]],
            [categories[trial_idx]],
        )

        for weight, hypo in zip(prev_post, hypotheses):
            if weight <= 0:
                continue
            lik = partition.calc_trueprob_entry(
                hypo,
                trial_slice,
                beta_param,
                use_cached_dist=True,
                gamma=gamma,
                w0=w0,
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
            sliding_pred_std.append(
                float(np.sqrt(np.sum(valid * (1 - valid))) / window_size)
            )

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


class StateModelGridOptimizer:
    """Grid-search helper for StateModel memory parameters."""

    def __init__(
        self,
        engine_config: Dict,
        processed_data_dir: Optional[Path | str] = None,
    ) -> None:
        self._engine_config_template = deepcopy(engine_config)
        self._processed_data_dir = (
            Path(processed_data_dir).resolve()
            if processed_data_dir is not None
            else _REPO_ROOT / "data" / "processed"
        )
        self.learning_data: Optional[pd.DataFrame] = None

    def prepare_data(self, data_path: Path | str = DEFAULT_DATA_PATH) -> None:
        data_path = Path(data_path).resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset not found: {data_path}")
        self.learning_data = pd.read_csv(data_path)

    def _get_subject_frame(
        self,
        subject_id: int,
        stop_at: float,
    ) -> pd.DataFrame:
        if self.learning_data is None:
            self.prepare_data()
        assert self.learning_data is not None

        subject_frame = self.learning_data[self.learning_data["iSub"] == subject_id]
        if subject_frame.empty:
            raise ValueError(f"Subject {subject_id} not found in dataset")

        stop_index = max(1, int(len(subject_frame) * stop_at + 0.5))
        return subject_frame.iloc[:stop_index].copy()

    def _default_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        mod_cfg = self._engine_config_template.get("modules", {}).get("memory_mod", {})
        kwargs = mod_cfg.get("kwargs", {})

        personal_range = kwargs.get(
            "personal_memory_range",
            {"gamma": (0.05, 1.0), "w0": (0.075, 0.15)},
        )
        param_resolution = max(1, int(kwargs.get("param_resolution", 20)))

        gamma_grid = kwargs.get("gamma_grid")
        if gamma_grid is None:
            gamma_low, gamma_high = personal_range.get("gamma", (0.05, 1.0))
            gamma_grid = np.linspace(float(gamma_low), float(gamma_high), param_resolution, endpoint=True)
        else:
            gamma_grid = np.asarray(gamma_grid, dtype=float)

        w0_grid = kwargs.get("w0_grid")
        if w0_grid is None:
            w0_high = float(personal_range.get("w0", (0.075, 0.15))[1])
            w0_grid = np.array([w0_high / (i + 1) for i in range(param_resolution)], dtype=float)
        else:
            w0_grid = np.asarray(w0_grid, dtype=float)

        return gamma_grid, w0_grid

    def grid_search_subject(
        self,
        subject_id: int,
        gamma_grid: Optional[Sequence[float]] = None,
        w0_grid: Optional[Sequence[float]] = None,
        window_size: int = 16,
        stop_at: float = 1.0,
        max_trials: Optional[int] = None,
    ) -> Dict[str, object]:
        subject_frame = self._get_subject_frame(subject_id, stop_at)
        gamma_candidates, w0_candidates = self._coerce_grids(gamma_grid, w0_grid)

        condition = int(subject_frame["condition"].iloc[0])
        arrays = self._extract_arrays(subject_frame, max_trials)

        results: List[GridPointResult] = []
        for gamma_val, w0_val in product(gamma_candidates, w0_candidates):
            result = self._evaluate_combination(
                subject_id,
                condition,
                arrays,
                float(gamma_val),
                float(w0_val),
                window_size,
            )
            results.append(result)

        if not results:
            raise RuntimeError("No parameter combinations were evaluated")

        best_result = min(results, key=lambda item: item.mean_error)

        return {
            "subject_id": subject_id,
            "condition": condition,
            "best": best_result,
            "grid": results,
            "gamma_grid": np.asarray(gamma_candidates, dtype=float),
            "w0_grid": np.asarray(w0_candidates, dtype=float),
        }

    def _coerce_grids(
        self,
        gamma_grid: Optional[Sequence[float]],
        w0_grid: Optional[Sequence[float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if gamma_grid is None or w0_grid is None:
            default_gamma, default_w0 = self._default_grids()
            gamma_grid = default_gamma if gamma_grid is None else gamma_grid
            w0_grid = default_w0 if w0_grid is None else w0_grid

        gamma_arr = np.asarray(list(gamma_grid), dtype=float)
        w0_arr = np.asarray(list(w0_grid), dtype=float)

        if gamma_arr.size == 0 or w0_arr.size == 0:
            raise ValueError("Parameter grids must be non-empty")

        return gamma_arr, w0_arr

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

    def _evaluate_combination(
        self,
        subject_id: int,
        condition: int,
        arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        gamma: float,
        w0: float,
        window_size: int,
    ) -> GridPointResult:
        stimulus, choices, feedback, categories = arrays
        trial_sequence = _prepare_trial_sequence(stimulus, choices, feedback)

        engine_config = deepcopy(self._engine_config_template)
        memory_cfg = engine_config.setdefault("modules", {}).setdefault("memory_mod", {})
        kwargs = memory_cfg.setdefault("kwargs", {})
        kwargs.update({"gamma": gamma, "w0": w0})

        model = StateModel(
            engine_config,
            condition=condition,
            subject_id=subject_id,
            processed_data_dir=self._processed_data_dir,
        )

        model.precompute_distances(stimulus)
        posterior_log = model.fit_step_by_step(trial_sequence)

        metrics = _compute_prediction_metrics(
            model,
            posterior_log,
            stimulus,
            choices,
            feedback,
            categories,
            gamma,
            w0,
            window_size,
        )

        LOGGER.debug(
            "Evaluated subject %s gamma=%.3f w0=%.3f -> error %.4f",
            subject_id,
            gamma,
            w0,
            metrics["mean_error"],
        )

        return GridPointResult(
            gamma=gamma,
            w0=w0,
            mean_error=float(metrics["mean_error"]),
            metrics=metrics,
            posterior_log=posterior_log,
        )
