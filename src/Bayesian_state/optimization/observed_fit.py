"""Exploratory observed-data fitting helpers for frozen Model 0818."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.Bayesian_state.optimization.parameter_space import (
    reactive_error_probability,
    spike_and_positive_values,
)
from src.Bayesian_state.simulation.config import (
    expand_profile_candidate_hyperparams,
)


WORKSPACE_PROFILE_KEY = "__profile_candidate__:workspace_execution"
REACTIVE_PROFILE_KEY = "__profile_candidate__:reactive_event"
GLOBAL_PROFILE_KEY = "__profile_candidate__:global_search"

CAPACITY_PATH = "engine.modules.hypo_transitions_mod.kwargs.capacity"
EXECUTION_PATH = (
    "engine.modules.hypo_transitions_mod.kwargs.persistent_execution.enabled"
)
CONTROLLER_PATH = (
    "engine.modules.hypo_transitions_mod.kwargs."
    "nested_feedback_accumulator_controller"
)
EVENT_CORRECT_PATH = f"{CONTROLLER_PATH}.event_after_correct"
EVENT_ERROR_PATH = f"{CONTROLLER_PATH}.event_after_error"
INITIAL_EVENT_PATH = f"{CONTROLLER_PATH}.initial_event_probability"
GLOBAL_SEARCH_PATH = f"{CONTROLLER_PATH}.global_search"
ACCUMULATOR_GAIN_PATH = f"{CONTROLLER_PATH}.accumulator_logit_gain"
GLOBAL_GAIN_PATH = f"{CONTROLLER_PATH}.global_search_failure_gain"
GAMMA_PATH = "engine.modules.memory_mod.kwargs.gamma"
BETA_PATH = "engine.modules.beta_mod.kwargs.beta_init"
ETA_PLUS_PATH = "engine.modules.beta_mod.kwargs.increase_rate"
ETA_MINUS_PATH = "engine.modules.beta_mod.kwargs.decrease_rate"


def _repo_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _parameter_values(
    parameter_space: Mapping[str, Any],
    name: str,
) -> list[Any]:
    spec = dict(parameter_space["subject_parameters"][name])
    if spec["kind"] == "spike_and_positive_grid":
        return list(spike_and_positive_values(parameter_space, name))
    if name == "workspace_execution":
        return [dict(value) for value in spec["candidates"]]
    return list(spec["coarse_values"])


def build_model_0818_hyper_config(
    analysis_config: Mapping[str, Any],
    parameter_space: Mapping[str, Any],
    *,
    root: Path,
    parallel_budget: int,
) -> dict[str, Any]:
    """Translate named Model 0818 parameters into executable Hyper-CD paths."""

    optimization = dict(analysis_config["optimization"])
    search = dict(optimization["search"])
    max_trials_raw = analysis_config["trial_scope"].get("max_trials")
    max_trials = None if max_trials_raw is None else int(max_trials_raw)
    output = _repo_path(root, analysis_config["output_dir"]) / "search"
    base_sim = _repo_path(root, analysis_config["base_simulation_config"])

    workspace_values = [
        {
            CAPACITY_PATH: int(value["M"]),
            EXECUTION_PATH: bool(int(value["chi"])),
        }
        for value in _parameter_values(
            parameter_space, "workspace_execution"
        )
    ]
    reactive_values = [
        {
            EVENT_CORRECT_PATH: float(event_correct),
            EVENT_ERROR_PATH: reactive_error_probability(
                float(event_correct), float(delta_e)
            ),
            INITIAL_EVENT_PATH: float(event_correct),
        }
        for event_correct in _parameter_values(parameter_space, "E_C")
        for delta_e in _parameter_values(parameter_space, "delta_E")
    ]
    global_values = [
        {
            GLOBAL_SEARCH_PATH: float(global_search),
            GLOBAL_GAIN_PATH: float(global_gain),
        }
        for global_search in _parameter_values(parameter_space, "g_0")
        for global_gain in _parameter_values(parameter_space, "c_G")
    ]
    hyperparam_space = {
        WORKSPACE_PROFILE_KEY: {"values": workspace_values},
        GAMMA_PATH: {
            "values": _parameter_values(parameter_space, "gamma")
        },
        REACTIVE_PROFILE_KEY: {"values": reactive_values},
        ACCUMULATOR_GAIN_PATH: {
            "values": _parameter_values(parameter_space, "c_A")
        },
        GLOBAL_PROFILE_KEY: {"values": global_values},
        BETA_PATH: {
            "values": _parameter_values(parameter_space, "beta_0")
        },
        ETA_PLUS_PATH: {
            "values": _parameter_values(parameter_space, "eta_plus")
        },
        ETA_MINUS_PATH: {
            "values": _parameter_values(parameter_space, "eta_minus")
        },
    }

    anchors = parameter_space["subject_parameters"]
    workspace_start = [
        {CAPACITY_PATH: int(value["M"]), EXECUTION_PATH: bool(int(value["chi"]))}
        for value in anchors["workspace_execution"]["start_candidates"]
    ]
    reactive_anchor = {
        EVENT_CORRECT_PATH: float(anchors["E_C"]["anchor"]),
        EVENT_ERROR_PATH: reactive_error_probability(
            float(anchors["E_C"]["anchor"]),
            float(anchors["delta_E"]["positive_anchor"]),
        ),
        INITIAL_EVENT_PATH: float(anchors["E_C"]["anchor"]),
    }
    global_anchor = {
        GLOBAL_SEARCH_PATH: float(anchors["g_0"]["anchor"]),
        GLOBAL_GAIN_PATH: float(anchors["c_G"]["zero_value"]),
    }
    common_anchor = {
        GAMMA_PATH: float(anchors["gamma"]["anchor"]),
        REACTIVE_PROFILE_KEY: reactive_anchor,
        ACCUMULATOR_GAIN_PATH: float(anchors["c_A"]["zero_value"]),
        GLOBAL_PROFILE_KEY: global_anchor,
        BETA_PATH: float(anchors["beta_0"]["anchor"]),
        ETA_PLUS_PATH: float(anchors["eta_plus"]["anchor"]),
        ETA_MINUS_PATH: float(anchors["eta_minus"]["anchor"]),
    }
    initial_points = [
        {WORKSPACE_PROFILE_KEY: workspace, **deepcopy(common_anchor)}
        for workspace in workspace_start
    ]

    return {
        "analysis_id": str(analysis_config["analysis_id"]),
        "base_sim_config_path": str(base_sim),
        "subjects": [int(value) for value in analysis_config["subjects"]],
        "output_dir": str(output),
        "hyperparam_selection_mode": "per_subject",
        "common_random_numbers_within_candidate_comparisons": True,
        "objective_order": [
            {
                "path": "simulation.mean_error",
                "rel_tolerance": 0.0,
                "abs_tolerance": 0.0,
                "scale_floor": 0.0,
                "anchor_guard": True,
            }
        ],
        "save_level": "compact",
        "hyper_base_seed": int(analysis_config["hyper_base_seed"]),
        "loss_metric": "choice_nll",
        "window_size": int(analysis_config["window_size"]),
        "statistics_config": {"enabled": False},
        "cd": {
            "n_restarts": len(initial_points),
            "max_outer_iters": int(search["max_outer_iters"]),
            "init_strategy": "anchor",
            "initial_points": initial_points,
            "coordinate_order": "fixed",
            "patience": 1,
            "min_delta": 0.0,
            "parallel_budget": int(parallel_budget),
        },
        "refine_policy": {
            "top_k": int(search["shortlist_size"]),
        },
        "hyperparam_space": hyperparam_space,
        "stages": {
            "coarse": {
                "cd_parallel": {
                    "max_repeat_jobs": int(search["filter_seed_count"]),
                },
                "simulation_overrides": {
                    "simulation_repeats": int(search["filter_seed_count"]),
                    "repeat_aggregation": "mean_probability",
                    "keep_logs": False,
                    "max_trials": max_trials,
                    "engine_config": {
                        "inference": {
                            "particle_count": int(search["particle_count"]),
                            "resample_threshold_fraction": float(
                                analysis_config["resample_threshold_fraction"]
                            ),
                        }
                    },
                },
            }
        },
    }


def extract_model_0818_parameters(
    hyperparams: Mapping[str, Any],
) -> dict[str, Any]:
    """Return named scientific parameters from executable hyperparameter paths."""

    expanded = expand_profile_candidate_hyperparams(hyperparams)
    event_correct = float(expanded[EVENT_CORRECT_PATH])
    event_error = float(expanded[EVENT_ERROR_PATH])
    logit_correct = np.log(event_correct) - np.log1p(-event_correct)
    logit_error = np.log(event_error) - np.log1p(-event_error)
    delta_e = float(max(0.0, logit_error - logit_correct))
    if abs(delta_e) < 1e-12:
        delta_e = 0.0
    return {
        "M": int(expanded[CAPACITY_PATH]),
        "chi": int(bool(expanded[EXECUTION_PATH])),
        "gamma": float(expanded[GAMMA_PATH]),
        "E_C": event_correct,
        "delta_E": delta_e,
        "E_E": event_error,
        "g_0": float(expanded[GLOBAL_SEARCH_PATH]),
        "c_A": float(expanded[ACCUMULATOR_GAIN_PATH]),
        "c_G": float(expanded[GLOBAL_GAIN_PATH]),
        "beta_0": float(expanded[BETA_PATH]),
        "eta_plus": float(expanded[ETA_PLUS_PATH]),
        "eta_minus": float(expanded[ETA_MINUS_PATH]),
    }


def _mean_nll(probabilities: np.ndarray, choices: np.ndarray) -> float:
    selected = probabilities[np.arange(choices.size), choices - 1]
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())


def summarize_observed_choice_fit(
    probability_runs: np.ndarray,
    choices: Sequence[int] | np.ndarray,
    categories: Sequence[int] | np.ndarray,
    *,
    window_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Compute transparent fit and baseline metrics from B PF probability runs."""

    runs = np.asarray(probability_runs, dtype=float)
    observed = np.asarray(choices, dtype=int).reshape(-1)
    correct = np.asarray(categories, dtype=int).reshape(-1)
    if runs.ndim != 3 or runs.shape[1:] != (observed.size, 2):
        raise ValueError("probability_runs must have shape (B, T, 2)")
    if runs.shape[0] < 2 or observed.size < 2:
        raise ValueError("at least two filter seeds and two trials are required")
    if int(window_size) <= 0:
        raise ValueError("window_size must be positive")
    if correct.shape != observed.shape:
        raise ValueError("categories and choices must have equal length")
    if not np.all(np.isin(observed, [1, 2])) or not np.all(
        np.isin(correct, [1, 2])
    ):
        raise ValueError("choices and categories must be encoded as 1 or 2")
    if not np.all(np.isfinite(runs)) or np.any(runs < 0.0):
        raise ValueError("probability runs must be finite and nonnegative")
    if not np.allclose(runs.sum(axis=2), 1.0, atol=1e-8):
        raise ValueError("probability rows must sum to one")

    mean_probability = runs.mean(axis=0)
    n_trials = observed.size
    one_hot = np.eye(2, dtype=float)[observed - 1]
    model_mean_nll = _mean_nll(mean_probability, observed)
    model_total_nll = model_mean_nll * float(n_trials)
    choice_brier = float(np.mean(np.sum((mean_probability - one_hot) ** 2, axis=1)))
    predicted_choice = np.argmax(mean_probability, axis=1) + 1
    choice_accuracy = float(np.mean(predicted_choice == observed))

    random_probability = np.full((n_trials, 2), 0.5, dtype=float)
    random_mean_nll = _mean_nll(random_probability, observed)
    causal_bias = np.empty((n_trials, 2), dtype=float)
    choice_two_count = 0
    for trial_index in range(n_trials):
        probability_two = (0.5 + float(choice_two_count)) / (1.0 + trial_index)
        causal_bias[trial_index] = [1.0 - probability_two, probability_two]
        choice_two_count += int(observed[trial_index] == 2)
    causal_bias_mean_nll = _mean_nll(causal_bias, observed)

    model_correct_probability = mean_probability[
        np.arange(n_trials), correct - 1
    ]
    observed_correct = (observed == correct).astype(float)
    width = max(1, min(int(window_size), n_trials))
    kernel = np.ones(width, dtype=float) / float(width)
    rolling_model = np.convolve(model_correct_probability, kernel, mode="valid")
    rolling_observed = np.convolve(observed_correct, kernel, mode="valid")
    if rolling_model.size > 1 and np.std(rolling_model) > 0.0 and np.std(
        rolling_observed
    ) > 0.0:
        learning_curve_correlation = float(
            np.corrcoef(rolling_model, rolling_observed)[0, 1]
        )
    else:
        learning_curve_correlation = float("nan")
    learning_curve_mae = float(np.mean(np.abs(rolling_model - rolling_observed)))

    probability_two = mean_probability[:, 1]
    observed_two = (observed == 2).astype(float)
    calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (probability_two >= lower) & (
            probability_two <= upper if upper >= 1.0 else probability_two < upper
        )
        if np.any(mask):
            calibration_error += float(np.mean(mask)) * abs(
                float(np.mean(probability_two[mask]))
                - float(np.mean(observed_two[mask]))
            )

    selected_by_seed = runs[:, np.arange(n_trials), observed - 1]
    seed_nll = -np.log(np.clip(selected_by_seed, 1e-12, 1.0)).mean(axis=1)
    probability_mcse = runs[:, :, 1].std(axis=0, ddof=1) / np.sqrt(
        float(runs.shape[0])
    )
    midpoint = n_trials // 2
    summary = {
        "trial_count": int(n_trials),
        "filter_seed_count": int(runs.shape[0]),
        "mean_trial_nll": model_mean_nll,
        "total_nll": model_total_nll,
        "choice_brier": choice_brier,
        "choice_accuracy": choice_accuracy,
        "random_mean_trial_nll": random_mean_nll,
        "nll_improvement_vs_random": random_mean_nll - model_mean_nll,
        "causal_bias_mean_trial_nll": causal_bias_mean_nll,
        "nll_improvement_vs_causal_bias": causal_bias_mean_nll - model_mean_nll,
        "early_half_mean_trial_nll": _mean_nll(
            mean_probability[:midpoint], observed[:midpoint]
        ),
        "late_half_mean_trial_nll": _mean_nll(
            mean_probability[midpoint:], observed[midpoint:]
        ),
        "learning_curve_correlation": learning_curve_correlation,
        "learning_curve_mae": learning_curve_mae,
        "choice_calibration_ece10": float(calibration_error),
        "seed_nll_sd": float(np.std(seed_nll, ddof=1)),
        "maximum_trial_probability_mcse": float(np.max(probability_mcse)),
        "median_trial_probability_mcse": float(np.median(probability_mcse)),
    }
    return summary, mean_probability


__all__ = [
    "ACCUMULATOR_GAIN_PATH",
    "BETA_PATH",
    "CAPACITY_PATH",
    "ETA_MINUS_PATH",
    "ETA_PLUS_PATH",
    "EVENT_CORRECT_PATH",
    "EVENT_ERROR_PATH",
    "EXECUTION_PATH",
    "GAMMA_PATH",
    "GLOBAL_GAIN_PATH",
    "GLOBAL_SEARCH_PATH",
    "build_model_0818_hyper_config",
    "extract_model_0818_parameters",
    "summarize_observed_choice_fit",
]
