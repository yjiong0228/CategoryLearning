"""Grid search utilities for tuning StateModel memory parameters."""
from __future__ import annotations

from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple, Any
from collections import defaultdict

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ..utils.base import LOGGER
from .optimizer_common import (
    BaseStateOptimizer,
    GridPointResult,
    evaluate_state_model_run,
)


class StateModelGridOptimizer(BaseStateOptimizer):
    """
    Grid-search helper for StateModel parameters with parallel execution support.
    """


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
            delayed(evaluate_state_model_run)(
                subject_id,
                condition,
                arrays,
                params,
                self._engine_config_template,
                self._processed_data_dir,
                window_size,
                True,
                False,
            )
            for params in tqdm(tasks, desc=f"Sub {subject_id} Grid Search")
        )

        # Aggregate results
        grouped_results = defaultdict(list)
        for run in results:
            # Create a hashable key for the parameters
            param_key = tuple(sorted(run.params.items()))
            grouped_results[param_key].append(run)

        final_grid_results: List[GridPointResult] = []
        
        for param_key, runs in grouped_results.items():
            params = dict(param_key)
            errors = [r.mean_error for r in runs]
            mean_error = float(np.mean(errors))
            std_error = float(np.std(errors)) if len(errors) > 1 else 0.0
            
            # Pick the run closest to the mean error as the representative metrics
            best_run_idx = np.argmin([abs(e - mean_error) for e in errors])
            best_run = runs[best_run_idx]
            best_metrics = best_run.metrics
            best_posterior = best_run.posterior_log
            best_prior = best_run.prior_log
            best_strategy_log = best_run.strategy_counts_log
            
            # Memory optimization: discard logs if not requested
            if not keep_logs:
                best_posterior = None
                best_prior = None
                best_strategy_log = None

            final_grid_results.append(GridPointResult(
                params=params,
                mean_error=mean_error,
                metrics=best_metrics,
                posterior_log=best_posterior,
                prior_log=best_prior,
                strategy_counts_log=best_strategy_log,
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
                delayed(evaluate_state_model_run)(
                    subject_id,
                    condition,
                    arrays,
                    params,
                    self._engine_config_template,
                    self._processed_data_dir,
                    window_size,
                    True,
                    False,
                )
                for params in tqdm(refit_tasks, desc=f"Sub {subject_id} Refit")
            )
            
            # Aggregate refit results
            refit_errors = [r.mean_error for r in refit_results]
            refit_mean_error = float(np.mean(refit_errors))
            refit_std_error = float(np.std(refit_errors))
            
            # Find representative run
            best_refit_idx = np.argmin([abs(e - refit_mean_error) for e in refit_errors])
            best_refit = refit_results[best_refit_idx]
            best_refit_metrics = best_refit.metrics
            best_refit_posterior = best_refit.posterior_log
            best_refit_prior = best_refit.prior_log
            best_refit_strategy = best_refit.strategy_counts_log

            if not keep_logs:
                best_refit_posterior = None
                best_refit_prior = None
                best_refit_strategy = None
            
            # Update best_result
            best_result.mean_error = refit_mean_error
            best_result.std_error = refit_std_error
            best_result.metrics = best_refit_metrics
            best_result.posterior_log = best_refit_posterior
            best_result.prior_log = best_refit_prior
            best_result.strategy_counts_log = best_refit_strategy
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
        run = evaluate_state_model_run(
            subject_id,
            condition,
            arrays,
            params,
            self._engine_config_template,
            self._processed_data_dir,
            window_size,
            True,
            False,
        )
        return GridPointResult(
            params=params,
            mean_error=run.mean_error,
            metrics=run.metrics,
            posterior_log=run.posterior_log
        )
