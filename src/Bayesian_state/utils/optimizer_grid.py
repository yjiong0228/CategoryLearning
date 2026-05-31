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
    SingleRunResult,
    evaluate_state_model_run,
    PREDICTION_MODE_POSTERIOR_T_MINUS_1,
    LOSS_METRIC_MAE,
)


def _build_run_record(
    run: SingleRunResult,
    run_index: int,
    subject_id: int,
    condition: int,
    window_size: int,
    phase: str,
) -> Dict[str, Any]:
    return {
        "run_index": int(run_index),
        "subject_id": int(subject_id),
        "condition": int(condition),
        "phase": str(phase),
        "window_size": int(window_size),
        "params": dict(run.params),
        "mean_error": float(run.mean_error),
        "selection_prediction_mode": str(run.selection_prediction_mode),
        "loss_metric": str(run.loss_metric),
        "metrics_by_mode": run.metrics_by_mode,
        "step_log": run.step_log,
        "posterior_log": run.posterior_log,
        "prior_log": run.prior_log,
        "strategy_counts_log": run.strategy_counts_log,
    }


class StateModelGridOptimizer(BaseStateOptimizer):
    """Grid-search helper for StateModel parameters with parallel execution support."""

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
        prediction_mode: str = PREDICTION_MODE_POSTERIOR_T_MINUS_1,
        selection_prediction_mode: str = PREDICTION_MODE_POSTERIOR_T_MINUS_1,
        loss_metric: str = LOSS_METRIC_MAE,
    ) -> Dict[str, object]:
        subject_frame = self._get_subject_frame(subject_id, stop_at)
        condition = self._get_condition_value(subject_frame)
        arrays = self._extract_arrays(subject_frame, max_trials)

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        tasks = []
        for combo in combinations:
            params = dict(zip(param_names, combo))
            for _ in range(n_repeats):
                tasks.append(params)

        LOGGER.info(
            f"Optimizing subject {subject_id}: {len(combinations)} combos * {n_repeats} repeats = {len(tasks)} tasks"
        )

        raw_results = list(Parallel(n_jobs=self.n_jobs)(
            delayed(evaluate_state_model_run)(
                subject_id,
                condition,
                arrays,
                params,
                self._engine_config_template,
                self._processed_data_dir,
                window_size,
                self._dataset_paths,
                True,
                True,
                prediction_mode,
                selection_prediction_mode,
                loss_metric,
            )
            for params in tqdm(tasks, desc=f"Sub {subject_id} Grid Search")
        ))
        results: List[SingleRunResult] = [r for r in raw_results if r is not None]
        if len(results) != len(tasks):
            raise RuntimeError("Grid search produced empty runs unexpectedly.")

        grouped_results = defaultdict(list)
        for run in results:
            param_key = tuple(sorted(run.params.items()))
            grouped_results[param_key].append(run)

        final_grid_results: List[GridPointResult] = []

        for param_key, runs in grouped_results.items():
            params = dict(param_key)
            errors = [r.mean_error for r in runs]
            mean_error = float(np.mean(errors))
            std_error = float(np.std(errors)) if len(errors) > 1 else 0.0
            run_records = [
                _build_run_record(
                    run=r,
                    run_index=i,
                    subject_id=subject_id,
                    condition=condition,
                    window_size=window_size,
                    phase="grid",
                )
                for i, r in enumerate(runs)
            ]

            best_run_idx = int(np.argmin(errors))
            best_run = runs[best_run_idx]
            best_metrics_by_mode = best_run.metrics_by_mode
            best_posterior = best_run.posterior_log
            best_prior = best_run.prior_log
            best_beta_log = best_run.beta_log
            best_strategy_log = best_run.strategy_counts_log
            best_step_log = best_run.step_log
            sample_errors = [float(e) for e in errors]
            raw_step_results = [r.step_log for r in runs if r.step_log is not None]

            if not keep_logs:
                best_posterior = None
                best_prior = None
                best_beta_log = None
                best_strategy_log = None
                best_step_log = None
                run_records = None
                raw_step_results = None

            final_grid_results.append(GridPointResult(
                params=params,
                mean_error=mean_error,
                metrics_by_mode=best_metrics_by_mode,
                selection_prediction_mode=selection_prediction_mode,
                posterior_log=best_posterior,
                prior_log=best_prior,
                beta_log=best_beta_log,
                step_results=best_step_log,
                strategy_counts_log=best_strategy_log,
                raw_runs=run_records,
                raw_step_results=raw_step_results,
                sample_errors=sample_errors,
                best_error=float(errors[best_run_idx]),
                refit_mean_error=mean_error,
                refit_std_error=std_error,
                representative_run_index=best_run_idx,
                n_repeats=n_repeats,
                std_error=std_error,
            ))

        if not final_grid_results:
            raise RuntimeError("No results produced.")

        best_result = min(final_grid_results, key=lambda item: item.mean_error)

        if refit_repeats > 0:
            LOGGER.info(f"Refitting best params for subject {subject_id} with {refit_repeats} repeats.")
            refit_tasks = [best_result.params] * refit_repeats

            raw_refit_results = list(Parallel(n_jobs=self.n_jobs)(
                delayed(evaluate_state_model_run)(
                    subject_id,
                    condition,
                    arrays,
                    params,
                    self._engine_config_template,
                    self._processed_data_dir,
                    window_size,
                    self._dataset_paths,
                    True,
                    True,
                    prediction_mode,
                    selection_prediction_mode,
                    loss_metric,
                )
                for params in tqdm(refit_tasks, desc=f"Sub {subject_id} Refit")
            ))
            refit_results: List[SingleRunResult] = [r for r in raw_refit_results if r is not None]
            if len(refit_results) != len(refit_tasks):
                raise RuntimeError("Refit stage produced empty runs unexpectedly.")

            refit_errors = [r.mean_error for r in refit_results]
            refit_mean_error = float(np.mean(refit_errors))
            refit_std_error = float(np.std(refit_errors))

            best_refit_idx = int(np.argmin(refit_errors))
            best_refit = refit_results[best_refit_idx]
            best_refit_metrics_by_mode = best_refit.metrics_by_mode
            best_refit_posterior = best_refit.posterior_log
            best_refit_prior = best_refit.prior_log
            best_refit_beta_log = best_refit.beta_log
            best_refit_strategy = best_refit.strategy_counts_log
            best_refit_step = best_refit.step_log
            refit_sample_errors = [float(e) for e in refit_errors]
            refit_run_records = [
                _build_run_record(
                    run=r,
                    run_index=i,
                    subject_id=subject_id,
                    condition=condition,
                    window_size=window_size,
                    phase="refit",
                )
                for i, r in enumerate(refit_results)
            ]
            refit_raw_steps = [r.step_log for r in refit_results if r.step_log is not None]

            if not keep_logs:
                best_refit_posterior = None
                best_refit_prior = None
                best_refit_beta_log = None
                best_refit_strategy = None
                best_refit_step = None
                refit_run_records = None
                refit_raw_steps = None

            best_result.mean_error = refit_mean_error
            best_result.std_error = refit_std_error
            best_result.metrics_by_mode = best_refit_metrics_by_mode
            best_result.posterior_log = best_refit_posterior
            best_result.prior_log = best_refit_prior
            best_result.beta_log = best_refit_beta_log
            best_result.step_results = best_refit_step
            best_result.strategy_counts_log = best_refit_strategy
            best_result.raw_runs = refit_run_records
            best_result.raw_step_results = refit_raw_steps
            best_result.sample_errors = refit_sample_errors
            best_result.best_error = float(refit_errors[best_refit_idx])
            best_result.refit_mean_error = refit_mean_error
            best_result.refit_std_error = refit_std_error
            best_result.representative_run_index = best_refit_idx
            best_result.n_repeats = refit_repeats
        else:
            if best_result.best_error is None:
                best_result.best_error = float(best_result.mean_error)
            if best_result.refit_mean_error is None:
                best_result.refit_mean_error = float(best_result.mean_error)
            if best_result.refit_std_error is None:
                best_result.refit_std_error = float(best_result.std_error)
            if best_result.sample_errors is None:
                best_result.sample_errors = [float(best_result.mean_error)]
            if best_result.raw_runs is None:
                best_result.raw_runs = []
            if best_result.representative_run_index is None:
                best_result.representative_run_index = 0

        return {
            "subject_id": subject_id,
            "condition": condition,
            "best": best_result,
            "grid": final_grid_results,
            "param_grid": param_grid,
            "selection_meta": {
                "param_selection": "min_mean_error",
                "run_selection": "min_error",
                "prediction_mode": prediction_mode,
                "selection_prediction_mode": selection_prediction_mode,
                "loss_metric": loss_metric,
            },
        }

    def grid_search_subject(
        self,
        subject_id: int,
        gamma_grid: Optional[Sequence[float]] = None,
        w0_grid: Optional[Sequence[float]] = None,
        window_size: int = 16,
        stop_at: float = 1.0,
        max_trials: Optional[int] = None,
    ) -> Dict[str, object]:
        if gamma_grid is None or w0_grid is None:
            d_gamma, d_w0 = self._default_grids()
            gamma_grid = list(map(float, d_gamma)) if gamma_grid is None else gamma_grid
            w0_grid = list(map(float, d_w0)) if w0_grid is None else w0_grid

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
            max_trials=max_trials,
        )

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

    def _evaluate_combination(
        self,
        subject_id: int,
        condition: int,
        arrays: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        gamma: float,
        w0: float,
        window_size: int,
    ) -> GridPointResult:
        params = {"gamma": gamma, "w0": w0}
        run = evaluate_state_model_run(
            subject_id,
            condition,
            arrays,
            params,
            self._engine_config_template,
            self._processed_data_dir,
            window_size,
            self._dataset_paths,
            True,
            False,
            PREDICTION_MODE_POSTERIOR_T_MINUS_1,
            PREDICTION_MODE_POSTERIOR_T_MINUS_1,
            LOSS_METRIC_MAE,
        )
        return GridPointResult(
            params=params,
            mean_error=run.mean_error,
            metrics_by_mode=run.metrics_by_mode,
            selection_prediction_mode=run.selection_prediction_mode,
            posterior_log=run.posterior_log,
            prior_log=run.prior_log,
            beta_log=run.beta_log,
        )
