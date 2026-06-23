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
from .simulation_statistics import compute_simulation_statistics


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _run_accuracy_shape_score(metrics: Mapping[str, Any]) -> float:
    true = np.asarray(metrics.get("sliding_true_acc"), dtype=float)
    pred = np.asarray(metrics.get("sliding_pred_acc"), dtype=float)
    if true.shape != pred.shape or true.size == 0:
        return float("inf")
    mask = np.isfinite(true) & np.isfinite(pred)
    if not mask.any():
        return float("inf")
    true = true[mask]
    pred = pred[mask]
    acc_mae = float(np.mean(np.abs(pred - true)))
    true_vol = float(np.mean(np.abs(np.diff(true)))) if true.size > 1 else float("nan")
    pred_vol = float(np.mean(np.abs(np.diff(pred)))) if pred.size > 1 else float("nan")
    vol_ratio = float(pred_vol / true_vol) if true_vol > 0 else float("nan")
    vol_penalty = abs(np.log(max(vol_ratio, 1e-6))) if np.isfinite(vol_ratio) else 1.0
    return float(acc_mae + 0.06 * vol_penalty)


def _run_switch_score(metrics: Mapping[str, Any]) -> float:
    required = ("pred_category_probs", "observed_choice_index", "valid_trial_mask")
    if any(key not in metrics for key in required):
        return float("inf")
    probs = np.asarray(metrics.get("pred_category_probs"), dtype=float)
    choices = np.asarray(metrics.get("observed_choice_index"), dtype=float)
    valid = np.asarray(metrics.get("valid_trial_mask"), dtype=bool)
    if probs.ndim != 2 or choices.ndim != 1 or valid.ndim != 1:
        return float("inf")
    if probs.shape[0] != choices.shape[0] or valid.shape[0] != choices.shape[0] or choices.size <= 1:
        return float("inf")
    prev_choice = choices[:-1]
    next_choice = choices[1:]
    pair_mask = (
        valid[1:]
        & np.isfinite(prev_choice)
        & np.isfinite(next_choice)
        & (prev_choice >= 0)
        & (next_choice >= 0)
        & (prev_choice < probs.shape[1])
        & (next_choice < probs.shape[1])
        & np.all(np.isfinite(probs[1:, :]), axis=1)
    )
    if not np.any(pair_mask):
        return float("inf")
    rows = np.arange(1, choices.size)[pair_mask]
    prev_idx = prev_choice[pair_mask].astype(int)
    next_idx = next_choice[pair_mask].astype(int)
    model_switch = 1.0 - np.clip(probs[rows, prev_idx], 0.0, 1.0)
    human_switch = (next_idx != prev_idx).astype(float)
    return float(abs(np.mean(model_switch) - np.mean(human_switch)))


def _behavior_representative_score(run: SingleRunResult, selection_prediction_mode: str) -> float:
    metrics = (run.metrics_by_mode or {}).get(selection_prediction_mode)
    if not isinstance(metrics, Mapping):
        return float("inf")
    shape = _run_accuracy_shape_score(metrics)
    switch = _run_switch_score(metrics)
    parts = [value for value in (shape, switch) if np.isfinite(value)]
    if not parts:
        return float("inf")
    return float(np.mean(parts))


def _select_representative_run_index(
    runs: Sequence[SingleRunResult],
    *,
    selection_prediction_mode: str,
    representative_run_selection: str,
    representative_choice_fraction: float,
) -> int:
    errors = np.asarray([float(run.mean_error) for run in runs], dtype=float)
    mode = str(representative_run_selection or "min_error").strip().lower()
    if mode in {"min_error", "choice_brier", "choice", "best"}:
        return int(np.nanargmin(errors))
    if mode not in {"behavior_composite", "composite_behavior", "multiobjective"}:
        raise ValueError(
            "representative_run_selection must be 'min_error' or 'behavior_composite'"
        )
    finite_errors = errors[np.isfinite(errors)]
    if finite_errors.size == 0:
        return 0
    frac = min(1.0, max(0.0, float(representative_choice_fraction)))
    gate_count = max(1, int(np.ceil(finite_errors.size * frac)))
    cutoff = float(np.sort(finite_errors)[gate_count - 1])
    eligible = [
        idx for idx, error in enumerate(errors)
        if np.isfinite(error) and float(error) <= cutoff
    ]
    if not eligible:
        eligible = list(range(len(runs)))
    return min(
        eligible,
        key=lambda idx: (
            _behavior_representative_score(runs[idx], selection_prediction_mode),
            float(errors[idx]),
            int(idx),
        ),
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
        "state_log": run.state_log,
        "trial_events": run.trial_events,
        "transition_counts": run.transition_counts,
    }


def aggregate_simulation_runs(
    runs: Sequence[SingleRunResult],
    *,
    params: Mapping[str, Any],
    subject_id: int,
    condition: int,
    window_size: int,
    selection_prediction_mode: str,
    simulation_repeats: int,
    simulation_point_seed: int | None,
    keep_logs: bool,
    representative_run_selection: str = "min_error",
    representative_choice_fraction: float = 0.10,
    statistics_config: Mapping[str, Any] | None = None,
) -> SimulationResult:
    """Aggregate repeated state-model runs using the standard simulation summary semantics."""
    runs = list(runs)
    if len(runs) != int(simulation_repeats):
        raise RuntimeError(
            f"Simulation produced {len(runs)} runs for {simulation_repeats} requested repeats."
        )
    if not runs:
        raise RuntimeError("Simulation produced no runs.")

    errors = [float(r.mean_error) for r in runs]
    mean_error = float(np.mean(errors))
    std_error = float(np.std(errors)) if len(errors) > 1 else 0.0
    min_error_idx = int(np.argmin(errors))
    representative_run_idx = _select_representative_run_index(
        runs,
        selection_prediction_mode=selection_prediction_mode,
        representative_run_selection=representative_run_selection,
        representative_choice_fraction=representative_choice_fraction,
    )
    representative_run = runs[representative_run_idx]

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
    statistics_summary = compute_simulation_statistics(
        runs,
        selection_prediction_mode=selection_prediction_mode,
        config=statistics_config,
    )

    return SimulationResult(
        params=dict(params),
        mean_error=mean_error,
        metrics_by_mode=representative_run.metrics_by_mode,
        selection_prediction_mode=selection_prediction_mode,
        state_log=representative_run.state_log,
        trial_events=representative_run.trial_events,
        transition_counts=representative_run.transition_counts,
        raw_runs=raw_runs,
        sample_errors=[float(e) for e in errors],
        best_error=float(errors[min_error_idx]),
        representative_run_index=representative_run_idx,
        simulation_repeats=int(simulation_repeats),
        simulation_point_seed=simulation_point_seed,
        std_error=std_error,
        statistics_summary=statistics_summary,
    )


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
        representative_run_selection: str = "min_error",
        representative_choice_fraction: float = 0.10,
        statistics_config: Mapping[str, Any] | None = None,
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

        best_result = aggregate_simulation_runs(
            runs,
            params=fixed_payload or eval_params,
            subject_id=subject_id,
            condition=condition,
            window_size=window_size,
            selection_prediction_mode=selection_prediction_mode,
            simulation_repeats=simulation_repeats,
            simulation_point_seed=simulation_point_seed,
            keep_logs=keep_logs,
            representative_run_selection=representative_run_selection,
            representative_choice_fraction=representative_choice_fraction,
            statistics_config=statistics_config,
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
                "representative_run_selection": representative_run_selection,
                "representative_choice_fraction": float(representative_choice_fraction),
                "hyper_candidate_seed": hyper_candidate_seed,
                "simulation_point_seed": simulation_point_seed,
                "seed_hyperparams": seed_params,
                "simulation_repeats": simulation_repeats,
                "window_size": int(window_size),
                "statistics_config": statistics_config,
            },
        }


__all__ = ["StateModelSimulationRunner", "aggregate_simulation_runs"]
