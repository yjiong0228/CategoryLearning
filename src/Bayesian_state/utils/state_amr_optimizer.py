"""Grid search utilities for tuning StateModel memory parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any, Union
from collections import defaultdict

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..problems import StateModel
from ..utils.base import LOGGER

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = _REPO_ROOT / "data" / "processed" / "Task2_processed.csv"


@dataclass
class GridPointResult:
    """Container for a single parameter combination evaluation."""
    
    params: Dict[str, Any]
    mean_error: float
    metrics: Dict[str, np.ndarray | float]
    posterior_log: Optional[Sequence[np.ndarray]] = None
    prior_log: Optional[Sequence[np.ndarray]] = None
    n_repeats: int = 1
    std_error: float = 0.0

    @property
    def gamma(self) -> float:
        return self.params.get("gamma", float("nan"))

    @property
    def w0(self) -> float:
        return self.params.get("w0", float("nan"))


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
    prior_log: Sequence[np.ndarray],
    stimulus: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    categories: np.ndarray,
    window_size: int,
) -> Dict[str, np.ndarray | float]:
    # Memory weighting is assumed to be embedded in the prior trajectory
    # produced during fitting.
    partition = model.partition_model
    hypotheses = list(model.hypotheses_set)
    
    # Try to find beta in likelihood_mod, default to 10.0
    beta_param = 10.0
    if hasattr(model.engine, "likelihood_mod"):
        lik_mod = getattr(model.engine, "likelihood_mod")
        beta_param = float(lik_mod.kwargs.get("beta", 10.0))


    prior_arr = np.asarray(prior_log, dtype=float)
    if prior_arr.ndim == 1:
        prior_arr = prior_arr.reshape(1, -1)

    n_trials = len(feedback)
    if prior_arr.shape[0] != n_trials:
        raise ValueError(
            "Prior log length does not match number of trials: "
            f"{prior_arr.shape[0]} vs {n_trials}"
        )

    true_acc = (feedback == 1.0).astype(float)
    pred_acc = np.full(n_trials, np.nan, dtype=float)

    for trial_idx in range(n_trials):
        current_prior = prior_arr[trial_idx]
        weighted_prob = 0.0

        trial_slice = (
            [stimulus[trial_idx]],
            [choices[trial_idx]],
            [feedback[trial_idx]],
            [categories[trial_idx]],
        )
    
        for idx, (weight, hypo) in enumerate(zip(current_prior, hypotheses)):
            if weight <= 0:
                continue
            lik = partition.calc_trueprob_entry(
                hypo,
                trial_slice,
                beta_param,
                use_cached_dist=False, # FIXME：必须是false
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


def _inject_params(config: Dict, params: Dict[str, Any]) -> None:
    """
    Inject parameters into the configuration dictionary.
    Supports dot notation (e.g. 'modules.memory_mod.kwargs.gamma')
    and shortcuts for common parameters.
    """
    shortcuts = {
        "gamma": "modules.memory_mod.kwargs.gamma",
        "w0": "modules.memory_mod.kwargs.w0",
        "beta": "modules.likelihood_mod.kwargs.beta",
    }

    def set_by_path(root: Dict, path: str, value: Any):
        parts = path.split(".")
        curr = root
        for part in parts[:-1]:
            curr = curr.setdefault(part, {})
        curr[parts[-1]] = value

    for key, value in params.items():
        path = shortcuts.get(key, key)
        set_by_path(config, path, value)


def _evaluate_single_run(
    subject_id: int,
    condition: int,
    arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    params: Dict[str, Any],
    engine_config_template: Dict,
    processed_data_dir: Path,
    window_size: int,
) -> Tuple[Dict[str, Any], float, Dict[str, Any], Sequence[np.ndarray], Sequence[np.ndarray]]:
    """
    Worker function to evaluate a single parameter combination.
    """
    stimulus, choices, feedback, categories = arrays
    trial_sequence = _prepare_trial_sequence(stimulus, choices, feedback)

    # Prepare config
    engine_config = deepcopy(engine_config_template)
    _inject_params(engine_config, params)

    # Initialize model
    from ..problems import StateModel
    model = StateModel(
        engine_config,
        condition=condition,
        subject_id=subject_id,
        processed_data_dir=processed_data_dir,
    )

    # Run model
    model.precompute_distances(stimulus)
    posterior_log, prior_log = model.fit_step_by_step(trial_sequence)

    # Compute metrics
    metrics = _compute_prediction_metrics(
        model,
        prior_log,
        stimulus,
        choices,
        feedback,
        categories,
        window_size,
    )
    
    return params, float(metrics["mean_error"]), metrics, posterior_log, prior_log


class StateModelGridOptimizer:
    """
    Grid-search helper for StateModel parameters with parallel execution support.
    """

    def __init__(
        self,
        engine_config: Dict,
        processed_data_dir: Optional[Path | str] = None,
        n_jobs: int = 1,
    ) -> None:
        self._engine_config_template = deepcopy(engine_config)
        self._processed_data_dir = (
            Path(processed_data_dir).resolve()
            if processed_data_dir is not None
            else _REPO_ROOT / "data" / "processed"
        )
        self.learning_data: Optional[pd.DataFrame] = None
        self.n_jobs = n_jobs

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

    def optimize_subject(
        self,
        subject_id: int,
        param_grid: Dict[str, Sequence[Any]],
        n_repeats: int = 1,
        refit_repeats: int = 0,
        window_size: int = 16,
        stop_at: float = 1.0,
        max_trials: Optional[int] = None,
        keep_logs: bool = False,
    ) -> Dict[str, object]:
        """
        Optimize parameters for a single subject using parallel grid search.
        
        Args:
            subject_id: The subject ID to optimize for.
            param_grid: Dictionary mapping parameter names to lists of values.
            n_repeats: Number of times to repeat each parameter combination (grid search).
            refit_repeats: Number of times to repeat the best parameter combination (refinement).
            window_size: Sliding window size for error calculation.
            stop_at: Fraction of data to use (0.0 to 1.0).
            max_trials: Maximum number of trials to use (overrides stop_at if set).
            keep_logs: Whether to keep posterior/prior logs for all grid points. 
                       If False, only the best result's logs are kept.
        """
        subject_frame = self._get_subject_frame(subject_id, stop_at)
        condition = int(subject_frame["condition"].iloc[0])
        arrays = self._extract_arrays(subject_frame, max_trials)

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        # Create tasks: (params, repeat_idx)
        tasks = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            for _ in range(n_repeats):
                tasks.append(params)

        LOGGER.info(
            f"Optimizing subject {subject_id}: {len(combinations)} combos * {n_repeats} repeats = {len(tasks)} tasks"
        )

        # Run parallel execution
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_evaluate_single_run)(
                subject_id,
                condition,
                arrays,
                params,
                self._engine_config_template,
                self._processed_data_dir,
                window_size
            )
            for params in tqdm(tasks, desc=f"Sub {subject_id} Grid Search")
        )

        # Aggregate results
        grouped_results = defaultdict(list)
        for params, error, metrics, posterior_log, prior_log in results:
            # Create a hashable key for the parameters
            param_key = tuple(sorted(params.items()))
            grouped_results[param_key].append((error, metrics, posterior_log, prior_log))

        final_grid_results: List[GridPointResult] = []
        
        for param_key, runs in grouped_results.items():
            params = dict(param_key)
            errors = [r[0] for r in runs]
            mean_error = float(np.mean(errors))
            std_error = float(np.std(errors)) if len(errors) > 1 else 0.0
            
            # Pick the run closest to the mean error as the representative metrics
            best_run_idx = np.argmin([abs(e - mean_error) for e in errors])
            _, best_metrics, best_posterior, best_prior = runs[best_run_idx]
            
            # Memory optimization: discard logs if not requested
            if not keep_logs:
                best_posterior = None
                best_prior = None

            final_grid_results.append(GridPointResult(
                params=params,
                mean_error=mean_error,
                metrics=best_metrics,
                posterior_log=best_posterior,
                prior_log=best_prior,
                n_repeats=n_repeats,
                std_error=std_error
            ))

        if not final_grid_results:
            raise RuntimeError("No results produced.")

        best_result = min(final_grid_results, key=lambda item: item.mean_error)

        # --- Refit Stage ---
        if refit_repeats > 0:
            LOGGER.info(f"Refitting best params for subject {subject_id} with {refit_repeats} repeats.")
            refit_tasks = [best_result.params] * refit_repeats
            
            refit_results = Parallel(n_jobs=self.n_jobs)(
                delayed(_evaluate_single_run)(
                    subject_id,
                    condition,
                    arrays,
                    params,
                    self._engine_config_template,
                    self._processed_data_dir,
                    window_size
                )
                for params in tqdm(refit_tasks, desc=f"Sub {subject_id} Refit")
            )
            
            # Aggregate refit results
            refit_errors = [r[1] for r in refit_results]
            refit_mean_error = float(np.mean(refit_errors))
            refit_std_error = float(np.std(refit_errors))
            
            # Find representative run
            best_refit_idx = np.argmin([abs(e - refit_mean_error) for e in refit_errors])
            _, _, best_refit_metrics, best_refit_posterior, best_refit_prior = refit_results[best_refit_idx]
            
            # Update best_result
            best_result.mean_error = refit_mean_error
            best_result.std_error = refit_std_error
            best_result.metrics = best_refit_metrics
            best_result.posterior_log = best_refit_posterior
            best_result.prior_log = best_refit_prior
            best_result.n_repeats = refit_repeats

        return {
            "subject_id": subject_id,
            "condition": condition,
            "best": best_result,
            "grid": final_grid_results,
            "param_grid": param_grid,
        }

    # -------------------------------------------------------------------------
    # Deprecated / Backward Compatibility Methods
    # -------------------------------------------------------------------------

    def grid_search_subject(
        self,
        subject_id: int,
        gamma_grid: Optional[Sequence[float]] = None,
        w0_grid: Optional[Sequence[float]] = None,
        window_size: int = 16,
        stop_at: float = 1.0,
        max_trials: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Legacy wrapper for backward compatibility.
        Translates gamma_grid/w0_grid to generic param_grid.
        """
        # Resolve default grids if None
        if gamma_grid is None or w0_grid is None:
            d_gamma, d_w0 = self._default_grids()
            gamma_grid = d_gamma if gamma_grid is None else gamma_grid
            w0_grid = d_w0 if w0_grid is None else w0_grid
            
        param_grid: Dict[str, Sequence[Any]] = {
            "gamma": list(gamma_grid) if gamma_grid is not None else [],
            "w0": list(w0_grid) if w0_grid is not None else []
        }
        
        return self.optimize_subject(
            subject_id=subject_id,
            param_grid=param_grid,
            n_repeats=1,
            window_size=window_size,
            stop_at=stop_at,
            max_trials=max_trials
        )

    def _default_grids(self) -> Tuple[np.ndarray, np.ndarray]:
        """Legacy helper to get default grids from config."""
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

    def _evaluate_combination(
        self,
        subject_id: int,
        condition: int,
        arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        gamma: float,
        w0: float,
        window_size: int,
    ) -> GridPointResult:
        """
        Legacy helper for single evaluation.
        """
        params = {"gamma": gamma, "w0": w0}
        _, mean_error, metrics, posterior_log, prior_log = _evaluate_single_run(
            subject_id,
            condition,
            arrays,
            params,
            self._engine_config_template,
            self._processed_data_dir,
            window_size
        )
        return GridPointResult(
            params=params,
            mean_error=mean_error,
            metrics=metrics,
            posterior_log=posterior_log
        )