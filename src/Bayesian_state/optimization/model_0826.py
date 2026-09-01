"""Named Model0826 architecture cells and Hyper-CD recovery configuration."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.Bayesian_state.optimization.parameter_space import (
    reactive_error_probability,
)
from src.Bayesian_state.simulation.config import (
    expand_profile_candidate_hyperparams,
)


MODEL_0826_CELLS = ("P", "PM", "PH", "PMH")

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

STANDARD_MEMORY_CLASS = (
    "src.Bayesian_state.model.modules.memory.BayesianMemoryModule"
)
DUAL_MEMORY_CLASS = "src.Bayesian_state.model.modules.memory.DualMemoryModule"


def build_model_0826_cell_engine(
    base_engine: Mapping[str, Any],
    cell: str,
) -> dict[str, Any]:
    """Return one P/PM/PH/PMH engine while changing only M and H."""

    cell_name = str(cell).strip().upper()
    if cell_name not in MODEL_0826_CELLS:
        raise ValueError(f"cell must be one of {MODEL_0826_CELLS}")
    engine = deepcopy(dict(base_engine))
    provenance = engine.get("provenance") or {}
    if provenance.get("model_id") != "model_0826":
        raise ValueError("base engine provenance.model_id must be 'model_0826'")
    modules = engine.get("modules")
    agenda = engine.get("agenda")
    if not isinstance(modules, dict) or not isinstance(agenda, list):
        raise ValueError("base Model0826 engine requires modules and agenda")

    has_memory = cell_name in {"PM", "PMH"}
    has_hypothesis_search = cell_name in {"PH", "PMH"}
    if has_memory:
        memory = modules.get("memory_mod")
        if not isinstance(memory, dict):
            raise ValueError("base Model0826 engine is missing memory_mod")
        memory["class"] = DUAL_MEMORY_CLASS
    else:
        modules["memory_mod"] = {"class": STANDARD_MEMORY_CLASS}

    if has_hypothesis_search:
        if "hypo_transitions_mod" not in modules:
            raise ValueError("base Model0826 engine is missing hypo_transitions_mod")
        if "hypo_transitions_mod" not in agenda:
            memory_index = agenda.index("memory_mod")
            agenda.insert(memory_index, "hypo_transitions_mod")
    else:
        modules.pop("hypo_transitions_mod", None)
        engine["agenda"] = [
            name for name in agenda if name != "hypo_transitions_mod"
        ]

    beta = modules.get("beta_mod") or {}
    beta_kwargs = beta.get("kwargs") or {}
    if (
        beta_kwargs.get("update_scope") != "active_hypotheses"
        or float(beta_kwargs.get("increase_rate", 0.0)) <= 0.0
        or float(beta_kwargs.get("decrease_rate", 0.0)) <= 0.0
    ):
        raise ValueError("Model0826 cells require dynamic active-hypotheses beta")
    readout = (engine.get("choice_readout") or {}).get("kwargs") or {}
    if (
        readout.get("method") != "expectation"
        or float(readout.get("power", np.nan)) != 1.0
        or float(readout.get("strategy_confidence_gain", np.nan)) != 0.0
    ):
        raise ValueError("Model0826 cells require the frozen expectation readout")
    output_noise = (engine.get("output_noise") or {}).get("kwargs") or {}
    if float(output_noise.get("base_lapse", np.nan)) != 0.0:
        raise ValueError("Model0826 cells require zero output lapse")

    engine.setdefault("recovery", {})["architecture_cell"] = cell_name
    return engine


def _parameter_values(
    parameter_space: Mapping[str, Any],
    name: str,
    stage: str,
) -> list[Any]:
    specification = dict(parameter_space["subject_parameters"][name])
    if name == "workspace_execution":
        key = "fine_candidates" if stage == "fine" else "candidates"
        return [deepcopy(value) for value in specification[key]]
    if specification["kind"] == "spike_and_positive_grid":
        key = "fine_positive_values" if stage == "fine" else "positive_values"
        return [float(specification["zero_value"]), *map(float, specification[key])]
    key = "fine_values" if stage == "fine" else "coarse_values"
    return [float(value) for value in specification[key]]


def _profile_spaces(
    parameter_space: Mapping[str, Any],
    free_parameters: Sequence[str],
    stage: str,
) -> dict[str, dict[str, list[Any]]]:
    free = set(free_parameters)
    space: dict[str, dict[str, list[Any]]] = {}
    if "workspace_execution" in free:
        space[WORKSPACE_PROFILE_KEY] = {
            "values": [
                {
                    CAPACITY_PATH: int(value["M"]),
                    EXECUTION_PATH: bool(int(value["chi"])),
                }
                for value in _parameter_values(
                    parameter_space, "workspace_execution", stage
                )
            ]
        }
    if "gamma" in free:
        space[GAMMA_PATH] = {
            "values": _parameter_values(parameter_space, "gamma", stage)
        }
    if {"E_C", "delta_E"}.issubset(free):
        space[REACTIVE_PROFILE_KEY] = {
            "values": [
                {
                    EVENT_CORRECT_PATH: float(event_correct),
                    EVENT_ERROR_PATH: reactive_error_probability(
                        float(event_correct), float(delta_e)
                    ),
                    INITIAL_EVENT_PATH: float(event_correct),
                }
                for event_correct in _parameter_values(
                    parameter_space, "E_C", stage
                )
                for delta_e in _parameter_values(
                    parameter_space, "delta_E", stage
                )
            ]
        }
    if "g_0" in free and "c_G" in free:
        space[GLOBAL_PROFILE_KEY] = {
            "values": [
                {
                    GLOBAL_SEARCH_PATH: float(global_search),
                    GLOBAL_GAIN_PATH: float(global_gain),
                }
                for global_search in _parameter_values(
                    parameter_space, "g_0", stage
                )
                for global_gain in _parameter_values(
                    parameter_space, "c_G", stage
                )
            ]
        }
    direct_paths = {
        "c_A": ACCUMULATOR_GAIN_PATH,
        "beta_0": BETA_PATH,
        "eta_plus": ETA_PLUS_PATH,
        "eta_minus": ETA_MINUS_PATH,
    }
    for parameter, path in direct_paths.items():
        if parameter in free:
            space[path] = {
                "values": _parameter_values(parameter_space, parameter, stage)
            }
    return space


def _anchor_point(
    parameter_space: Mapping[str, Any],
    free_parameters: Sequence[str],
    workspace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    parameters = parameter_space["subject_parameters"]
    free = set(free_parameters)
    point: dict[str, Any] = {}
    if "workspace_execution" in free:
        if workspace is None:
            raise ValueError("workspace start is required for H cells")
        point[WORKSPACE_PROFILE_KEY] = {
            CAPACITY_PATH: int(workspace["M"]),
            EXECUTION_PATH: bool(int(workspace["chi"])),
        }
    if "gamma" in free:
        point[GAMMA_PATH] = float(parameters["gamma"]["anchor"])
    if {"E_C", "delta_E"}.issubset(free):
        event_correct = float(parameters["E_C"]["anchor"])
        delta_e = float(parameters["delta_E"]["positive_anchor"])
        point[REACTIVE_PROFILE_KEY] = {
            EVENT_CORRECT_PATH: event_correct,
            EVENT_ERROR_PATH: reactive_error_probability(event_correct, delta_e),
            INITIAL_EVENT_PATH: event_correct,
        }
    if "g_0" in free and "c_G" in free:
        point[GLOBAL_PROFILE_KEY] = {
            GLOBAL_SEARCH_PATH: float(parameters["g_0"]["anchor"]),
            GLOBAL_GAIN_PATH: float(parameters["c_G"]["zero_value"]),
        }
    if "c_A" in free:
        point[ACCUMULATOR_GAIN_PATH] = float(parameters["c_A"]["zero_value"])
    for parameter, path in (
        ("beta_0", BETA_PATH),
        ("eta_plus", ETA_PLUS_PATH),
        ("eta_minus", ETA_MINUS_PATH),
    ):
        if parameter in free:
            point[path] = float(parameters[parameter]["anchor"])
    return point


def _stage_overrides(
    budget: Mapping[str, Any],
    analysis_config: Mapping[str, Any],
) -> dict[str, Any]:
    repeats = int(budget["filter_seed_count"])
    overrides: dict[str, Any] = {
        "simulation_repeats": repeats,
        "repeat_aggregation": "mean_probability",
        "keep_logs": False,
        "max_trials": analysis_config.get("max_trials"),
        "engine_config": {
            "inference": {
                "particle_count": int(budget["particle_count"]),
                "resample_threshold_fraction": float(
                    analysis_config.get("resample_threshold_fraction", 0.5)
                ),
            }
        },
    }
    if analysis_config.get("evaluation_protocol") is not None:
        overrides["evaluation_protocol"] = deepcopy(
            analysis_config["evaluation_protocol"]
        )
    return overrides


def build_model_0826_hyper_config(
    analysis_config: Mapping[str, Any],
    parameter_space: Mapping[str, Any],
    cell: str,
    base_sim_config_path: str | Path,
    output_dir: str | Path,
    stage_budgets: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a cell-specific, schema-v2 coarse/fine Hyper-CD configuration."""

    cell_name = str(cell).strip().upper()
    if cell_name not in MODEL_0826_CELLS:
        raise ValueError(f"cell must be one of {MODEL_0826_CELLS}")
    cells = parameter_space.get("architecture_cells") or {}
    if cell_name not in cells:
        raise ValueError(f"parameter space is missing architecture cell {cell_name}")
    free_parameters = [
        str(value) for value in cells[cell_name]["free_parameters"]
    ]
    coarse_space = _profile_spaces(parameter_space, free_parameters, "coarse")
    fine_space = _profile_spaces(parameter_space, free_parameters, "fine")
    workspace_starts: list[Mapping[str, Any] | None]
    if "workspace_execution" in free_parameters:
        workspace_starts = list(
            parameter_space["subject_parameters"]["workspace_execution"][
                "start_candidates"
            ]
        )
    else:
        workspace_starts = [None]
    initial_points = [
        _anchor_point(parameter_space, free_parameters, workspace)
        for workspace in workspace_starts
    ]
    if set(initial_points[0]) != set(coarse_space):
        raise ValueError("Model0826 initial point does not match coarse coordinates")

    coarse_budget = dict(stage_budgets["coarse"])
    fine_budget = dict(stage_budgets["fine"])
    final_budget = dict(stage_budgets["final_rescore"])
    cd_config = dict(analysis_config.get("cd") or {})
    shortlist_size = int(analysis_config.get("shortlist_size", 4))
    return {
        "search_schema_version": 2,
        "analysis_id": str(analysis_config["analysis_id"]),
        "base_sim_config_path": str(Path(base_sim_config_path)),
        "subjects": [int(value) for value in analysis_config["subjects"]],
        "output_dir": str(Path(output_dir)),
        "hyperparam_selection_mode": str(
            analysis_config.get("hyperparam_selection_mode", "per_subject")
        ),
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
        "window_size": int(analysis_config.get("window_size", 16)),
        "statistics_config": {"enabled": False},
        "cd": {
            "n_restarts": len(initial_points),
            "max_outer_iters": int(cd_config.get("max_outer_iters", 6)),
            "init_strategy": "anchor",
            "initial_points": initial_points,
            "coordinate_order": str(cd_config.get("coordinate_order", "fixed")),
            "patience": int(cd_config.get("patience", 2)),
            "min_delta": float(cd_config.get("min_delta", 0.0)),
            "parallel_budget": int(cd_config.get("parallel_budget", 1)),
            "resume_mode": "explicit",
            "checkpoint_every_coordinate": True,
        },
        "refine_policy": {
            "top_k": shortlist_size,
            "fine_initialization": "coarse_shortlist",
        },
        "hyperparam_space": coarse_space,
        "stages": {
            "coarse": {
                "hyperparam_space": coarse_space,
                "cd_parallel": {
                    "max_repeat_jobs": int(coarse_budget["filter_seed_count"])
                },
                "simulation_overrides": _stage_overrides(
                    coarse_budget, analysis_config
                ),
            },
            "fine": {
                "hyperparam_space": fine_space,
                "cd_parallel": {
                    "max_repeat_jobs": int(fine_budget["filter_seed_count"])
                },
                "simulation_overrides": _stage_overrides(
                    fine_budget, analysis_config
                ),
            },
        },
        "final_rescore": {
            "enabled": True,
            "shortlist_size": shortlist_size,
            "seed_family": str(final_budget["seed_family"]),
            "simulation_overrides": _stage_overrides(
                final_budget, analysis_config
            ),
        },
        "recovery": {
            "model_id": "model_0826",
            "architecture_cell": cell_name,
            "free_parameters": free_parameters,
        },
    }


def extract_model_0826_parameters(
    hyperparams: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert executable parameter paths into named Model0826 parameters."""

    expanded = expand_profile_candidate_hyperparams(hyperparams)
    named: dict[str, Any] = {}
    if CAPACITY_PATH in expanded:
        named["M"] = int(expanded[CAPACITY_PATH])
        named["chi"] = int(bool(expanded[EXECUTION_PATH]))
    if GAMMA_PATH in expanded:
        named["gamma"] = float(expanded[GAMMA_PATH])
    if EVENT_CORRECT_PATH in expanded:
        event_correct = float(expanded[EVENT_CORRECT_PATH])
        event_error = float(expanded[EVENT_ERROR_PATH])
        logit_correct = np.log(event_correct) - np.log1p(-event_correct)
        logit_error = np.log(event_error) - np.log1p(-event_error)
        delta_e = float(max(0.0, logit_error - logit_correct))
        named.update(
            {
                "E_C": event_correct,
                "delta_E": 0.0 if abs(delta_e) < 1e-12 else delta_e,
                "E_E": event_error,
            }
        )
    for name, path in (
        ("g_0", GLOBAL_SEARCH_PATH),
        ("c_A", ACCUMULATOR_GAIN_PATH),
        ("c_G", GLOBAL_GAIN_PATH),
        ("beta_0", BETA_PATH),
        ("eta_plus", ETA_PLUS_PATH),
        ("eta_minus", ETA_MINUS_PATH),
    ):
        if path in expanded:
            named[name] = float(expanded[path])
    return named


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
    "MODEL_0826_CELLS",
    "build_model_0826_cell_engine",
    "build_model_0826_hyper_config",
    "extract_model_0826_parameters",
]
