"""Repeated simulation utilities for fixed StateModel hyperparameters."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, List

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ..utils.base import LOGGER
from .optimizer_common import (
    BaseStateOptimizer,
    SimulationResult,
    SingleRunResult,
    derive_simulation_point_seed,
    derive_trajectory_seed,
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
) -> Dict[str, Any]:
    return {
        "run_index": int(run_index),
        "subject_id": int(subject_id),
        "condition": int(condition),
        "phase": "simulation",
        "window_size": int(window_size),
        "params": dict(run.params),
        "mean_error": float(run.mean_error),
        "selection_prediction_mode": str(run.selection_prediction_mode),
        "loss_metric": str(run.loss_metric),
        "loss_delta": run.loss_delta,
        "simulation_point_seed": run.simulation_point_seed,
        "trajectory_seed": run.trajectory_seed,
        "module_seed": run.module_seed,
        "seed_context": run.seed_context,
        "metrics_by_mode": run.metrics_by_mode,
        "step_log": run.step_log,
        "posterior_log": run.posterior_log,
        "prior_log": run.prior_log,
        "strategy_counts_log": run.strategy_counts_log,
    }


class StateModelSimulationRunner(BaseStateOptimizer):
    """Run repeated simulations for one fixed hyperparameter setting."""

    def simulate_subject(
        self,
        subject_id: int,
        simulation_repeats: int | None = None,
        fixed_hyperparams: Mapping[str, Any] | None = None,
        runtime_params: Mapping[str, Any] | None = None,
        window_size: int = 16,
        stop_at: float = 1.0,
        max_trials: Optional[int] = None,
        keep_logs: bool = False,
        prediction_mode: str = PREDICTION_MODE_POSTERIOR_T_MINUS_1,
        selection_prediction_mode: str = PREDICTION_MODE_POSTERIOR_T_MINUS_1,
        loss_metric: str = LOSS_METRIC_MAE,
        loss_delta: float | None = None,
        hyper_candidate_seed: int | None = None,
        seed_hyperparams: Mapping[str, Any] | None = None,
    ) -> Dict[str, object]:
        if simulation_repeats is None:
            raise ValueError("simulation_repeats is required.")
        simulation_repeats = int(simulation_repeats)
        if simulation_repeats <= 0:
            raise ValueError(f"simulation_repeats must be positive, got {simulation_repeats}")

        subject_frame = self._get_subject_frame(subject_id, stop_at)
        condition = self._get_condition_value(subject_frame)
        arrays = self._extract_arrays(subject_frame, max_trials)

        fixed_payload = dict(fixed_hyperparams or {})
        eval_params = dict(runtime_params or {})
        seed_params = dict(seed_hyperparams) if seed_hyperparams is not None else (fixed_payload or eval_params)
        simulation_point_seed = (
            derive_simulation_point_seed(int(hyper_candidate_seed), subject_id, seed_params)
            if hyper_candidate_seed is not None
            else None
        )

        tasks = []
        for repeat_index in range(simulation_repeats):
            trajectory_seed = (
                derive_trajectory_seed(int(simulation_point_seed), "simulation", repeat_index)
                if simulation_point_seed is not None
                else None
            )
            tasks.append(
                {
                    "repeat_index": int(repeat_index),
                    "trajectory_seed": trajectory_seed,
                }
            )

        LOGGER.info(
            "Simulating subject %s: fixed hyperparams * %s repeats = %s tasks",
            subject_id,
            simulation_repeats,
            len(tasks),
        )

        raw_results = list(
            Parallel(n_jobs=self.n_jobs)(
                delayed(evaluate_state_model_run)(
                    subject_id,
                    condition,
                    arrays,
                    eval_params,
                    self._engine_config_template,
                    self._processed_data_dir,
                    window_size,
                    self._dataset_paths,
                    keep_logs,
                    keep_logs,
                    prediction_mode,
                    selection_prediction_mode,
                    loss_metric,
                    loss_delta,
                    simulation_point_seed=simulation_point_seed,
                    trajectory_seed=task["trajectory_seed"],
                    seed_context={
                        "hyper_candidate_seed": hyper_candidate_seed,
                        "simulation_point_seed": simulation_point_seed,
                        "trajectory_seed": task["trajectory_seed"],
                        "phase": "simulation",
                        "repeat_index": task["repeat_index"],
                    },
                )
                for task in tqdm(tasks, desc=f"Sub {subject_id} Simulation")
            )
        )
        runs: List[SingleRunResult] = [r for r in raw_results if r is not None]
        if len(runs) != len(tasks):
            raise RuntimeError("Simulation produced empty runs unexpectedly.")

        errors = [float(r.mean_error) for r in runs]
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors)) if len(errors) > 1 else 0.0
        best_run_idx = int(np.argmin(errors))
        best_run = runs[best_run_idx]

        raw_runs = [
            _build_run_record(
                run=run,
                run_index=i,
                subject_id=subject_id,
                condition=condition,
                window_size=window_size,
            )
            for i, run in enumerate(runs)
        ]
        if not keep_logs:
            raw_runs = []

        best_result = SimulationResult(
            params=fixed_payload or eval_params,
            mean_error=mean_error,
            metrics_by_mode=best_run.metrics_by_mode,
            selection_prediction_mode=selection_prediction_mode,
            posterior_log=best_run.posterior_log,
            prior_log=best_run.prior_log,
            beta_log=best_run.beta_log,
            step_results=best_run.step_log,
            strategy_counts_log=best_run.strategy_counts_log,
            raw_runs=raw_runs,
            raw_step_results=[r.step_log for r in runs if r.step_log is not None] if keep_logs else [],
            sample_errors=[float(e) for e in errors],
            best_error=float(errors[best_run_idx]),
            representative_run_index=best_run_idx,
            simulation_repeats=simulation_repeats,
            simulation_point_seed=simulation_point_seed,
            std_error=std_error,
        )

        return {
            "subject_id": subject_id,
            "condition": condition,
            "best": best_result,
            "fixed_hyperparams": fixed_payload,
            "runtime_params": eval_params,
            "selection_meta": {
                "param_selection": "fixed_hyperparams",
                "run_selection": "min_error",
                "prediction_mode": prediction_mode,
                "selection_prediction_mode": selection_prediction_mode,
                "loss_metric": loss_metric,
                "loss_delta": loss_delta,
                "hyper_candidate_seed": hyper_candidate_seed,
                "simulation_point_seed": simulation_point_seed,
                "seed_hyperparams": seed_params,
                "simulation_repeats": simulation_repeats,
            },
        }


__all__ = ["StateModelSimulationRunner"]
