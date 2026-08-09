"""Repeated StateModel simulation, representative-run selection, and summaries."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from ..metrics.losses import LOSS_METRIC_MAE
from ..metrics._numeric import safe_float
from ..metrics.trajectory_statistics import (
    marginal_prediction_metrics_from_runs,
    accuracy_shape_metrics_from_runs,
    distribution_behavior_metrics_from_runs,
    history_kernel_metrics_from_runs,
    loss_metric_summary_from_runs,
    simulation_error_summary,
    switch_behavior_metrics_from_runs,
)
from ..metrics.trajectory_selection import representative_behavior_score
from ..utils.base import LOGGER
from ..utils.seeding import derive_simulation_point_seed, derive_trajectory_seed
from .state_model_execution import (
    PREDICTION_MODE_POSTERIOR_T_MINUS_1,
    BaseStateOptimizer,
    SimulationResult,
    SingleRunResult,
    evaluate_state_model_run,
)
from .simulation_config import (
    EVALUATION_ROLE_SIMULATION,
    resolve_evaluation_score_mask,
)


# Repeated-run execution and representative-run selection
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
            representative_behavior_score(
                (runs[idx].metrics_by_mode or {}).get(
                    selection_prediction_mode,
                    {},
                )
            ),
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

    error_summary = simulation_error_summary([run.mean_error for run in runs])
    errors = error_summary["sample_errors"]
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
        mean_error=error_summary["mean_error"],
        metrics_by_mode=representative_run.metrics_by_mode,
        selection_prediction_mode=selection_prediction_mode,
        state_log=representative_run.state_log,
        trial_events=representative_run.trial_events,
        transition_counts=representative_run.transition_counts,
        raw_runs=raw_runs,
        sample_errors=[float(e) for e in errors],
        best_error=error_summary["best_error"],
        representative_run_index=representative_run_idx,
        simulation_repeats=int(simulation_repeats),
        simulation_point_seed=simulation_point_seed,
        std_error=error_summary["std_error"],
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
        score_trial_mask: Sequence[bool] | np.ndarray | None = None,
        evaluation_protocol: Mapping[str, Any] | None = None,
        evaluation_role: str = EVALUATION_ROLE_SIMULATION,
    ) -> Dict[str, object]:
        if simulation_repeats is None:
            raise ValueError("simulation_repeats is required.")
        simulation_repeats = int(simulation_repeats)
        if simulation_repeats <= 0:
            raise ValueError(f"simulation_repeats must be positive, got {simulation_repeats}")

        subject_frame = self._get_subject_frame(subject_id, stop_at)
        condition = self._get_condition_value(subject_frame)
        arrays = self._extract_arrays(subject_frame, max_trials)
        if score_trial_mask is not None and evaluation_protocol is not None:
            raise ValueError(
                "Provide score_trial_mask or evaluation_protocol, not both."
            )
        if score_trial_mask is not None:
            resolved_score_mask = np.asarray(score_trial_mask, dtype=bool).reshape(-1)
            if resolved_score_mask.shape[0] != arrays.feedback.shape[0]:
                raise ValueError(
                    "score_trial_mask length does not match the evaluated subject trials: "
                    f"{resolved_score_mask.shape[0]} vs {arrays.feedback.shape[0]}"
                )
            score_context = {
                "enabled": True,
                "mode": "explicit_mask",
                "role": str(evaluation_role),
                "partition": "explicit",
                "n_trials": int(arrays.feedback.shape[0]),
                "split_index": None,
                "score_trial_count": int(np.sum(resolved_score_mask)),
            }
        else:
            resolved_score_mask, score_context = resolve_evaluation_score_mask(
                int(arrays.feedback.shape[0]),
                evaluation_protocol,
                role=evaluation_role,
            )

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
                    score_trial_mask=resolved_score_mask,
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
                "evaluation_protocol": dict(evaluation_protocol or {}),
                "evaluation_role": str(evaluation_role),
                "score_context": score_context,
                "score_trial_count": int(score_context["score_trial_count"]),
            },
        }


# Repeated-run statistic configuration and result schema
MULTIOBJECTIVE_WEIGHT_DEFAULTS = {
    "choice_error": 0.0,
    "accuracy_shape": 1.0,
    "history_kernel": 1.0,
    "switch_behavior": 1.0,
}

SELECTION_METRIC_ALIASES = {
    "mean_simulation_error": "simulation.mean_error",
    "best_simulation_error": "simulation.best_error",
    "best10_mean_simulation_error": "simulation.best10_mean_error",
    "q10_simulation_error": "simulation.q10_error",
    "selection_error": "selection.primary.value",
    "accuracy_shape_score": "statistics.scores.accuracy_shape.value",
    "history_kernel_score": "statistics.scores.history_kernel.value",
    "switch_behavior_score": "statistics.scores.switch_behavior.value",
    "distribution_score": "statistics.scores.distribution.multiobjective.score",
    "distribution_component_max_raw": "statistics.scores.distribution.multiobjective.component_max_raw",
    "distribution_intersection_score": "statistics.scores.distribution.intersection.score",
    "distribution_ppc_interval_score": "statistics.scores.distribution.ppc_interval.score",
}

SIMULATION_STAT_DEFAULTS = {
    "enabled": False,
    "mode": "accuracy_shape",
    "primary_tolerance_abs": 0.02,
    "primary_tolerance_rel": 0.08,
    "run_choice_fraction": 0.10,
    "accuracy_weight": 1.0,
    "volatility_weight": 0.03,
    "slope_weight": 0.02,
    "target_volatility_ratio": 1.0,
    "min_volatility_ratio": 1e-6,
    "history_max_lag": 8,
    "history_ridge": 1e-3,
    "history_standardize": True,
    "history_kernel_weight": 1.0,
    "history_corr_weight": 0.05,
    "history_norm_weight": 0.0,
    "history_min_norm": 1e-6,
    "switch_weight": 1.0,
    "win_stay_weight": 1.0,
    "lose_shift_weight": 1.0,
    "perseveration_weight": 0.5,
    "min_switch_trials": 5,
    "multiobjective_weights": MULTIOBJECTIVE_WEIGHT_DEFAULTS,
    "distribution_min_run_count": 10,
    "distribution_interval_alpha": 0.10,
    "distribution_accept_acc_mae_max": 0.10,
    "distribution_accept_vol_ratio_min": 0.60,
    "distribution_accept_vol_ratio_max": 2.00,
    "distribution_accept_history_corr_min": 0.80,
    "distribution_accept_switch_score_max": 0.10,
}


def get_stat_value(root: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """Read a nested statistic by dot path."""
    current: Any = root
    for part in str(path).split("."):
        if not part:
            return default
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def resolve_selection_metric_path(metric: Any) -> str:
    """Normalize a configured selection metric to a structured statistics path."""
    path = str(metric or "").strip()
    if not path:
        raise ValueError("selection_metric cannot be empty")
    path = SELECTION_METRIC_ALIASES.get(path, path)
    if "." not in path:
        raise ValueError(
            "selection_metric must be a structured path such as "
            "'simulation.mean_error' or 'statistics.loss.choice_brier.mean'"
        )
    return path


def resolve_simulation_stat_config(
    raw: Any = None,
    *,
    setting_name: str = "statistics_config",
) -> Dict[str, Any]:
    cfg = dict(SIMULATION_STAT_DEFAULTS)
    if raw is not None:
        if not isinstance(raw, Mapping):
            raise ValueError(f"{setting_name} must be a mapping when provided")
        cfg.update(dict(raw))
    cfg["enabled"] = bool(cfg.get("enabled", False))
    cfg["mode"] = str(cfg.get("mode", "accuracy_shape")).strip().lower()
    mode_aliases = {
        "shape": "accuracy_shape",
        "accuracy": "accuracy_shape",
        "accuracy_curve": "accuracy_shape",
        "history": "history_kernel",
        "kernel": "history_kernel",
        "history_feedback": "history_kernel",
        "switch": "switch_behavior",
        "switching": "switch_behavior",
        "exploration": "switch_behavior",
        "perseveration": "switch_behavior",
        "multi": "multiobjective",
        "multi_objective": "multiobjective",
        "multi-objective": "multiobjective",
        "distribution": "distribution_multiobjective",
        "predictive_distribution": "distribution_multiobjective",
        "distribution_multi": "distribution_multiobjective",
        "distribution_multi_objective": "distribution_multiobjective",
        "distribution-multi-objective": "distribution_multiobjective",
        "intersection": "distribution_intersection",
        "acceptance": "distribution_intersection",
        "distribution_acceptance": "distribution_intersection",
        "distribution_intersection": "distribution_intersection",
        "ppc": "distribution_ppc_interval",
        "ppc_interval": "distribution_ppc_interval",
        "distribution_ppc": "distribution_ppc_interval",
        "distribution_ppc_interval": "distribution_ppc_interval",
    }
    cfg["mode"] = mode_aliases.get(cfg["mode"], cfg["mode"])
    if cfg["mode"] not in {
        "accuracy_shape",
        "history_kernel",
        "switch_behavior",
        "multiobjective",
        "distribution_multiobjective",
        "distribution_intersection",
        "distribution_ppc_interval",
    }:
        raise ValueError(
            f"{setting_name}.mode must be one of: "
            "'accuracy_shape', 'history_kernel', 'switch_behavior', "
            "'multiobjective', 'distribution_multiobjective', "
            "'distribution_intersection', 'distribution_ppc_interval'"
        )
    for key in (
        "primary_tolerance_abs",
        "primary_tolerance_rel",
        "run_choice_fraction",
        "accuracy_weight",
        "volatility_weight",
        "slope_weight",
        "target_volatility_ratio",
        "min_volatility_ratio",
        "history_ridge",
        "history_kernel_weight",
        "history_corr_weight",
        "history_norm_weight",
        "history_min_norm",
        "switch_weight",
        "win_stay_weight",
        "lose_shift_weight",
        "perseveration_weight",
        "distribution_interval_alpha",
        "distribution_accept_acc_mae_max",
        "distribution_accept_vol_ratio_min",
        "distribution_accept_vol_ratio_max",
        "distribution_accept_history_corr_min",
        "distribution_accept_switch_score_max",
    ):
        cfg[key] = float(cfg[key])
    cfg["history_max_lag"] = int(cfg["history_max_lag"])
    cfg["min_switch_trials"] = int(cfg["min_switch_trials"])
    cfg["distribution_min_run_count"] = int(cfg["distribution_min_run_count"])
    cfg["history_standardize"] = bool(cfg.get("history_standardize", True))
    raw_weights = cfg.get("multiobjective_weights")
    weights = dict(MULTIOBJECTIVE_WEIGHT_DEFAULTS)
    if raw_weights is not None:
        if not isinstance(raw_weights, Mapping):
            raise ValueError(f"{setting_name}.multiobjective_weights must be a mapping")
        weights.update({str(key): float(value) for key, value in raw_weights.items()})
    allowed_weight_keys = set(MULTIOBJECTIVE_WEIGHT_DEFAULTS)
    unknown_weight_keys = sorted(set(weights) - allowed_weight_keys)
    if unknown_weight_keys:
        raise ValueError(
            f"{setting_name}.multiobjective_weights contains unsupported keys: "
            + ", ".join(unknown_weight_keys)
        )
    if any(value < 0 for value in weights.values()):
        raise ValueError(f"{setting_name}.multiobjective_weights must be non-negative")
    if sum(weights.values()) <= 0:
        raise ValueError(f"{setting_name}.multiobjective_weights must include at least one positive weight")
    cfg["multiobjective_weights"] = weights
    if cfg["run_choice_fraction"] <= 0 or cfg["run_choice_fraction"] > 1:
        raise ValueError(f"{setting_name}.run_choice_fraction must be in (0, 1]")
    if cfg["target_volatility_ratio"] <= 0:
        raise ValueError(f"{setting_name}.target_volatility_ratio must be > 0")
    if cfg["min_volatility_ratio"] <= 0:
        raise ValueError(f"{setting_name}.min_volatility_ratio must be > 0")
    if cfg["history_max_lag"] <= 0:
        raise ValueError(f"{setting_name}.history_max_lag must be positive")
    if cfg["history_min_norm"] <= 0:
        raise ValueError(f"{setting_name}.history_min_norm must be positive")
    if cfg["min_switch_trials"] <= 0:
        raise ValueError(f"{setting_name}.min_switch_trials must be positive")
    if cfg["distribution_min_run_count"] <= 0:
        raise ValueError(f"{setting_name}.distribution_min_run_count must be positive")
    if cfg["distribution_interval_alpha"] <= 0 or cfg["distribution_interval_alpha"] >= 1:
        raise ValueError(f"{setting_name}.distribution_interval_alpha must be in (0, 1)")
    if cfg["distribution_accept_vol_ratio_min"] <= 0:
        raise ValueError(f"{setting_name}.distribution_accept_vol_ratio_min must be > 0")
    if cfg["distribution_accept_vol_ratio_max"] <= 0:
        raise ValueError(f"{setting_name}.distribution_accept_vol_ratio_max must be > 0")
    if cfg["distribution_accept_vol_ratio_max"] < cfg["distribution_accept_vol_ratio_min"]:
        raise ValueError(
            f"{setting_name}.distribution_accept_vol_ratio_max must be >= "
            "distribution_accept_vol_ratio_min"
        )
    return cfg


def _accuracy_shape_nested(flat: Mapping[str, Any]) -> Dict[str, Any]:
    if not flat:
        return {}
    rmse = safe_float(flat.get("accuracy_shape_acc_rmse"))
    return {
        "diagnostics": {
            "selected": {
                "repeat_index": int(flat.get("accuracy_shape_repeat_index", -1)),
                "choice_error": safe_float(flat.get("accuracy_shape_choice_error")),
                "metrics": {
                    "mae": safe_float(flat.get("accuracy_shape_acc_mae")),
                    "rmse": rmse,
                    "mse": float(rmse * rmse) if np.isfinite(rmse) else float("nan"),
                    "corr": safe_float(flat.get("accuracy_shape_acc_corr")),
                    "volatility_ratio": safe_float(flat.get("accuracy_shape_vol_ratio")),
                    "range_ratio": safe_float(flat.get("accuracy_shape_range_ratio")),
                    "slope_agree": safe_float(flat.get("accuracy_shape_slope_agree")),
                },
            },
            "run_gate": {
                "choice_error_cutoff": safe_float(flat.get("accuracy_shape_run_choice_cutoff")),
                "eligible_count": int(flat.get("accuracy_shape_eligible_run_count", 0)),
                "run_count": int(flat.get("accuracy_shape_all_run_count", 0)),
            },
            "score_summary": {
                "mean": safe_float(flat.get("accuracy_shape_score_mean")),
                "q10": safe_float(flat.get("accuracy_shape_score_q10")),
                "eligible_mean": safe_float(flat.get("accuracy_shape_eligible_score_mean")),
            },
        },
        "score": {
            "value": safe_float(flat.get("accuracy_shape_score"), float("inf")),
            "repeat_index": int(flat.get("accuracy_shape_repeat_index", -1)),
            "choice_error": safe_float(flat.get("accuracy_shape_choice_error")),
            "components": {
                "accuracy_mae": safe_float(flat.get("accuracy_shape_acc_mae")),
                "volatility_ratio": safe_float(flat.get("accuracy_shape_vol_ratio")),
                "slope_agree": safe_float(flat.get("accuracy_shape_slope_agree")),
            },
        },
    }


def _history_kernel_nested(flat: Mapping[str, Any]) -> Dict[str, Any]:
    if not flat:
        return {}
    return {
        "diagnostics": {
            "selected": {
                "repeat_index": int(flat.get("history_kernel_repeat_index", -1)),
                "choice_error": safe_float(flat.get("history_kernel_choice_error")),
                "metrics": {
                    "mse": safe_float(flat.get("history_kernel_mse")),
                    "corr": safe_float(flat.get("history_kernel_corr")),
                    "corr_loss": safe_float(flat.get("history_kernel_corr_loss")),
                    "norm_ratio": safe_float(flat.get("history_kernel_norm_ratio")),
                    "human_norm": safe_float(flat.get("history_kernel_human_norm")),
                    "model_norm": safe_float(flat.get("history_kernel_model_norm")),
                    "max_lag": int(flat.get("history_kernel_max_lag", 0)),
                    "n_rows": int(flat.get("history_kernel_n_rows", 0)),
                    "human_kernel": list(flat.get("history_kernel_human") or []),
                    "model_kernel": list(flat.get("history_kernel_model") or []),
                },
            },
            "run_gate": {
                "choice_error_cutoff": safe_float(flat.get("history_kernel_run_choice_cutoff")),
                "eligible_count": int(flat.get("history_kernel_eligible_run_count", 0)),
                "run_count": int(flat.get("history_kernel_all_run_count", 0)),
            },
            "score_summary": {
                "mean": safe_float(flat.get("history_kernel_score_mean")),
                "q10": safe_float(flat.get("history_kernel_score_q10")),
                "eligible_mean": safe_float(flat.get("history_kernel_eligible_score_mean")),
            },
        },
        "score": {
            "value": safe_float(flat.get("history_kernel_score"), float("inf")),
            "repeat_index": int(flat.get("history_kernel_repeat_index", -1)),
            "choice_error": safe_float(flat.get("history_kernel_choice_error")),
            "components": {
                "mse": safe_float(flat.get("history_kernel_mse")),
                "corr_loss": safe_float(flat.get("history_kernel_corr_loss")),
                "norm_ratio": safe_float(flat.get("history_kernel_norm_ratio")),
            },
        },
    }


def _switch_behavior_nested(flat: Mapping[str, Any]) -> Dict[str, Any]:
    if not flat:
        return {}
    return {
        "diagnostics": {
            "selected": {
                "repeat_index": int(flat.get("switch_behavior_repeat_index", -1)),
                "choice_error": safe_float(flat.get("switch_behavior_choice_error")),
                "metrics": {
                    "switch": {
                        "human": safe_float(flat.get("switch_behavior_switch_human")),
                        "model": safe_float(flat.get("switch_behavior_switch_model")),
                        "abs_diff": safe_float(flat.get("switch_behavior_switch_abs_diff")),
                    },
                    "perseveration": {
                        "human": safe_float(flat.get("switch_behavior_perseveration_human")),
                        "model": safe_float(flat.get("switch_behavior_perseveration_model")),
                        "abs_diff": safe_float(flat.get("switch_behavior_perseveration_abs_diff")),
                    },
                    "win_stay": {
                        "human": safe_float(flat.get("switch_behavior_win_stay_human")),
                        "model": safe_float(flat.get("switch_behavior_win_stay_model")),
                        "abs_diff": safe_float(flat.get("switch_behavior_win_stay_abs_diff")),
                    },
                    "lose_shift": {
                        "human": safe_float(flat.get("switch_behavior_lose_shift_human")),
                        "model": safe_float(flat.get("switch_behavior_lose_shift_model")),
                        "abs_diff": safe_float(flat.get("switch_behavior_lose_shift_abs_diff")),
                    },
                    "counts": {
                        "pairs": int(flat.get("switch_behavior_n_pairs", 0)),
                        "win_pairs": int(flat.get("switch_behavior_n_win_pairs", 0)),
                        "loss_pairs": int(flat.get("switch_behavior_n_loss_pairs", 0)),
                    },
                },
            },
            "run_gate": {
                "choice_error_cutoff": safe_float(flat.get("switch_behavior_run_choice_cutoff")),
                "eligible_count": int(flat.get("switch_behavior_eligible_run_count", 0)),
                "run_count": int(flat.get("switch_behavior_all_run_count", 0)),
            },
            "score_summary": {
                "mean": safe_float(flat.get("switch_behavior_score_mean")),
                "q10": safe_float(flat.get("switch_behavior_score_q10")),
                "eligible_mean": safe_float(flat.get("switch_behavior_eligible_score_mean")),
            },
        },
        "score": {
            "value": safe_float(flat.get("switch_behavior_score"), float("inf")),
            "repeat_index": int(flat.get("switch_behavior_repeat_index", -1)),
            "choice_error": safe_float(flat.get("switch_behavior_choice_error")),
            "components": {
                "switch_abs_diff": safe_float(flat.get("switch_behavior_switch_abs_diff")),
                "win_stay_abs_diff": safe_float(flat.get("switch_behavior_win_stay_abs_diff")),
                "lose_shift_abs_diff": safe_float(flat.get("switch_behavior_lose_shift_abs_diff")),
                "perseveration_abs_diff": safe_float(flat.get("switch_behavior_perseveration_abs_diff")),
            },
        },
    }


def _ppc_stats_nested(flat: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for label in (
        "acc_mean",
        "acc_vol",
        "acc_range",
        "history_kernel_norm",
        "switch_rate",
        "win_stay",
        "lose_shift",
        "perseveration",
    ):
        prefix = f"distribution_ppc_interval_{label}_"
        if f"{prefix}human" not in flat:
            continue
        stats[label] = {
            "human": safe_float(flat.get(f"{prefix}human")),
            "model_lower": safe_float(flat.get(f"{prefix}model_q05")),
            "model_upper": safe_float(flat.get(f"{prefix}model_q95")),
            "model_median": safe_float(flat.get(f"{prefix}model_median")),
            "percentile": safe_float(flat.get(f"{prefix}percentile")),
            "tail_score": safe_float(flat.get(f"{prefix}tail_score")),
            "violation": safe_float(flat.get(f"{prefix}violation"), float("inf")),
            "accept": bool(flat.get(f"{prefix}accept", False)),
        }
    return stats


def _distribution_nested(flat: Mapping[str, Any]) -> Dict[str, Any]:
    if not flat:
        return {}
    violations = {
        "acc_mae": safe_float(flat.get("distribution_intersection_acc_mae_violation"), float("inf")),
        "vol_ratio_low": safe_float(flat.get("distribution_intersection_vol_ratio_low_violation"), float("inf")),
        "vol_ratio_high": safe_float(flat.get("distribution_intersection_vol_ratio_high_violation"), float("inf")),
        "history_corr": safe_float(flat.get("distribution_intersection_history_corr_violation"), float("inf")),
        "switch_rate": safe_float(flat.get("distribution_intersection_switch_rate_violation"), float("inf")),
        "win_stay": safe_float(flat.get("distribution_intersection_win_stay_violation"), float("inf")),
        "lose_shift": safe_float(flat.get("distribution_intersection_lose_shift_violation"), float("inf")),
        "perseveration": safe_float(flat.get("distribution_intersection_perseveration_violation"), float("inf")),
    }
    components = {
        "choice_error": safe_float(flat.get("distribution_choice_error_mean")),
        "accuracy_shape": safe_float(flat.get("distribution_accuracy_shape_score"), float("inf")),
        "history_kernel": safe_float(flat.get("distribution_history_kernel_score"), float("inf")),
        "switch_behavior": safe_float(flat.get("distribution_switch_behavior_score"), float("inf")),
    }
    return {
        "diagnostics": {
            "run_count": int(flat.get("distribution_run_count", 0)),
            "choice_error": {
                "mean": safe_float(flat.get("distribution_choice_error_mean")),
                "median": safe_float(flat.get("distribution_choice_error_median")),
                "q10": safe_float(flat.get("distribution_choice_error_q10")),
                "std": safe_float(flat.get("distribution_choice_error_std")),
            },
            "accuracy_curve": {
                "mae": {
                    "mean": safe_float(flat.get("distribution_acc_mae_mean")),
                    "median": safe_float(flat.get("distribution_acc_mae_median")),
                    "q90": safe_float(flat.get("distribution_acc_mae_q90")),
                },
                "rmse": {"mean": safe_float(flat.get("distribution_acc_rmse_mean"))},
                "volatility_ratio": {
                    "mean": safe_float(flat.get("distribution_vol_ratio_mean")),
                    "median": safe_float(flat.get("distribution_vol_ratio_median")),
                    "q10": safe_float(flat.get("distribution_vol_ratio_q10")),
                    "q90": safe_float(flat.get("distribution_vol_ratio_q90")),
                },
                "slope_agree": {"mean": safe_float(flat.get("distribution_slope_agree_mean"))},
            },
            "history_kernel": {
                "mse": safe_float(flat.get("distribution_history_kernel_mse")),
                "corr": safe_float(flat.get("distribution_history_kernel_corr")),
                "corr_loss": safe_float(flat.get("distribution_history_kernel_corr_loss")),
                "norm_ratio": safe_float(flat.get("distribution_history_kernel_norm_ratio")),
                "human_norm": safe_float(flat.get("distribution_history_kernel_human_norm")),
                "model_norm": safe_float(flat.get("distribution_history_kernel_model_norm")),
                "run_count": int(flat.get("distribution_history_kernel_run_count", 0)),
            },
            "switch_behavior": {
                "switch": {
                    "human": safe_float(flat.get("distribution_switch_human")),
                    "model": safe_float(flat.get("distribution_switch_model")),
                    "abs_diff": safe_float(flat.get("distribution_switch_abs_diff")),
                },
                "perseveration": {
                    "human": safe_float(flat.get("distribution_perseveration_human")),
                    "model": safe_float(flat.get("distribution_perseveration_model")),
                    "abs_diff": safe_float(flat.get("distribution_perseveration_abs_diff")),
                },
                "win_stay": {
                    "human": safe_float(flat.get("distribution_win_stay_human")),
                    "model": safe_float(flat.get("distribution_win_stay_model")),
                    "abs_diff": safe_float(flat.get("distribution_win_stay_abs_diff")),
                },
                "lose_shift": {
                    "human": safe_float(flat.get("distribution_lose_shift_human")),
                    "model": safe_float(flat.get("distribution_lose_shift_model")),
                    "abs_diff": safe_float(flat.get("distribution_lose_shift_abs_diff")),
                },
                "run_count": int(flat.get("distribution_switch_run_count", 0)),
            },
        },
        "score": {
            "multiobjective": {
                "score": safe_float(flat.get("distribution_score"), float("inf")),
                "component_max_raw": safe_float(flat.get("distribution_component_max_raw"), float("inf")),
                "components": components,
            },
            "intersection": {
                "score": safe_float(flat.get("distribution_intersection_score"), float("inf")),
                "accept": bool(flat.get("distribution_intersection_accept", False)),
                "violation_count": int(flat.get("distribution_intersection_violation_count", 0)),
                "violations": violations,
            },
            "ppc_interval": {
                "score": safe_float(flat.get("distribution_ppc_interval_score"), float("inf")),
                "accept": bool(flat.get("distribution_ppc_interval_accept", False)),
                "violation_count": int(flat.get("distribution_ppc_interval_violation_count", 0)),
                "alpha": safe_float(flat.get("distribution_ppc_interval_alpha")),
                "stat_count": int(flat.get("distribution_ppc_interval_stat_count", 0)),
                "lower_quantile": safe_float(flat.get("distribution_ppc_interval_lower_quantile")),
                "upper_quantile": safe_float(flat.get("distribution_ppc_interval_upper_quantile")),
                "stats": _ppc_stats_nested(flat),
            },
        },
    }


def compute_simulation_statistics(
    runs: Sequence[Any],
    *,
    selection_prediction_mode: str,
    config: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return structured statistics derived from all repeated simulation runs."""
    cfg = resolve_simulation_stat_config(config)
    summary: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}
    scores: Dict[str, Any] = {}

    loss_summary = loss_metric_summary_from_runs(
        runs,
        selection_prediction_mode=selection_prediction_mode,
    )
    if loss_summary:
        summary["loss"] = loss_summary

    marginal = marginal_prediction_metrics_from_runs(
        runs,
        selection_prediction_mode=selection_prediction_mode,
    )
    if marginal:
        summary["marginal_prediction"] = marginal

    shape = _accuracy_shape_nested(
        accuracy_shape_metrics_from_runs(
            runs,
            selection_prediction_mode=selection_prediction_mode,
            config=cfg,
        )
    )
    if shape:
        diagnostics["accuracy_curve"] = shape["diagnostics"]
        scores["accuracy_shape"] = shape["score"]

    history = _history_kernel_nested(
        history_kernel_metrics_from_runs(
            runs,
            selection_prediction_mode=selection_prediction_mode,
            config=cfg,
        )
    )
    if history:
        diagnostics["history_kernel"] = history["diagnostics"]
        scores["history_kernel"] = history["score"]

    switch = _switch_behavior_nested(
        switch_behavior_metrics_from_runs(
            runs,
            selection_prediction_mode=selection_prediction_mode,
            config=cfg,
        )
    )
    if switch:
        diagnostics["switch_behavior"] = switch["diagnostics"]
        scores["switch_behavior"] = switch["score"]

    distribution = _distribution_nested(
        distribution_behavior_metrics_from_runs(
            runs,
            selection_prediction_mode=selection_prediction_mode,
            config=cfg,
        )
    )
    if distribution:
        diagnostics["distribution"] = distribution["diagnostics"]
        scores["distribution"] = distribution["score"]

    if diagnostics:
        summary["diagnostics"] = diagnostics
    if scores:
        summary["scores"] = scores
    return summary


__all__ = [
    "MULTIOBJECTIVE_WEIGHT_DEFAULTS",
    "SELECTION_METRIC_ALIASES",
    "SIMULATION_STAT_DEFAULTS",
    "StateModelSimulationRunner",
    "aggregate_simulation_runs",
    "compute_simulation_statistics",
    "get_stat_value",
    "resolve_selection_metric_path",
    "resolve_simulation_stat_config",
]
