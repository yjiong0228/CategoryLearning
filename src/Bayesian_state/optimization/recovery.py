"""Fit a recovery dataset under a frozen search budget."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping
import pandas as pd
from src.Bayesian_state.optimization.model_0826 import (
    build_model_0826_cell_engine,
    build_model_0826_hyper_config,
)
from src.Bayesian_state.optimization.search.coordinate_descent import HyperCDOptimizer
from src.Bayesian_state.optimization.parameter_space import load_model_parameter_space
from src.Bayesian_state.utils.seeding import stable_seed
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.simulation.recovery import (
    CELL_FREE_PARAMETERS,
    RecoveryDatasetSpec,
    RecoveryDesign,
)
from src.Bayesian_state.utils.recovery_artifacts import (
    _load_yaml,
    _write_immutable_yaml,
)


def resolve_recovery_stage_budgets(
    search_config: Mapping[str, Any],
    frozen_budget: Mapping[str, Any],
) -> dict[str, dict[str, int]]:
    """Resolve low-cost search stages and the frozen final-score budget."""

    final_budget = {
        "particle_count": int(frozen_budget["particle_count"]),
        "filter_seed_count": int(frozen_budget["filter_seed_count"]),
    }
    if final_budget["particle_count"] < 2 or final_budget["filter_seed_count"] < 1:
        raise ValueError("frozen PF budget is invalid")
    configured = search_config.get("stage_budgets")
    if configured is None:
        return {
            stage: deepcopy(final_budget)
            for stage in ("coarse", "fine", "final_rescore")
        }
    if not isinstance(configured, Mapping):
        raise ValueError("search.stage_budgets must be a mapping")

    resolved: dict[str, dict[str, int]] = {}
    for stage in ("coarse", "fine"):
        raw = configured.get(stage)
        if not isinstance(raw, Mapping):
            raise ValueError(f"search.stage_budgets.{stage} must be a mapping")
        budget = {
            "particle_count": int(raw["particle_count"]),
            "filter_seed_count": int(raw["filter_seed_count"]),
        }
        if budget["particle_count"] < 2 or budget["filter_seed_count"] < 1:
            raise ValueError(f"search.stage_budgets.{stage} is invalid")
        if (
            budget["particle_count"] > final_budget["particle_count"]
            or budget["filter_seed_count"] > final_budget["filter_seed_count"]
        ):
            raise ValueError(
                f"search.stage_budgets.{stage} cannot exceed final_rescore"
            )
        resolved[stage] = budget
    resolved["final_rescore"] = final_budget
    return resolved


def fit_recovery_dataset(
    design: RecoveryDesign,
    specification: RecoveryDatasetSpec,
    *,
    candidate_cell: str,
    synthetic_csv: str | Path,
    frozen_budget: Mapping[str, Any],
    output_dir: str | Path,
    resume: bool = False,
    optimizer_factory: Callable[..., Any] = HyperCDOptimizer,
) -> dict[str, Any]:
    """Fit one cell to one synthetic dataset under its registered score mask."""

    cell = str(candidate_cell).strip().upper()
    if cell not in CELL_FREE_PARAMETERS:
        raise ValueError("candidate_cell must be P, PM, PH, or PMH")
    particle_count = int(frozen_budget["particle_count"])
    filter_seed_count = int(frozen_budget["filter_seed_count"])
    if particle_count < 2 or filter_seed_count < 1:
        raise ValueError("frozen PF budget is invalid")
    synthetic_path = Path(synthetic_csv).resolve()
    if not synthetic_path.is_file():
        raise ValueError(f"synthetic recovery CSV does not exist: {synthetic_path}")
    synthetic = pd.read_csv(synthetic_path)
    if len(synthetic) != specification.trial_count:
        raise ValueError("synthetic recovery CSV trial count does not match design")

    base_simulation = _load_yaml(design.base_simulation_config)
    dataset_paths = resolve_dataset_paths(
        base_simulation,
        design.base_simulation_config.parent,
    )
    base_engine = _load_yaml(design.model_engine_config)
    cell_engine = build_model_0826_cell_engine(base_engine, cell)
    if specification.family == "module":
        evaluation_protocol: dict[str, Any] = {
            "mode": "sequential_holdout",
            "train_fraction": 0.70,
            "optimization_partition": "train",
            "simulation_partition": "evaluation",
        }
    elif specification.family == "parameter":
        evaluation_protocol = {"mode": "all"}
    else:
        raise ValueError("unknown recovery dataset family")

    output = Path(output_dir)
    config_dir = output / "configs"
    search_dir = output / "search"
    simulation_config_path = config_dir / "simulation.yaml"
    hyper_config_path = config_dir / "hyper_cd.yaml"
    resolved_simulation = deepcopy(base_simulation)
    resolved_simulation.pop("engine_config_path", None)
    resolved_simulation.update(
        {
            "subjects": [int(specification.subject_id)],
            "engine_config": cell_engine,
            "dataset": {
                "processed_dir": str(dataset_paths["processed_dir"]),
                "learning_data": str(synthetic_path),
                "perception_summary": str(dataset_paths["perception_summary"]),
                "perception_summary_72": str(
                    dataset_paths["perception_summary_72"]
                ),
                "feature_order_data": str(dataset_paths["feature_order_data"]),
            },
            "output_dir": str(output / "base_simulation"),
            "simulation_repeats": filter_seed_count,
            "repeat_aggregation": "mean_probability",
            "max_trials": None,
            "evaluation_protocol": evaluation_protocol,
            "keep_logs": False,
        }
    )
    _write_immutable_yaml(
        simulation_config_path,
        resolved_simulation,
        resume=resume,
    )

    search_config = dict(design.config["search"])
    final_seed_family = (
        f"{search_config['final_rescore_seed_family']}:"
        f"{specification.dataset_id}:{cell}"
    )
    analysis_config = {
        "analysis_id": (
            f"{design.analysis_id}:{specification.dataset_id}:{cell}"
        ),
        "subjects": [int(specification.subject_id)],
        "hyper_base_seed": stable_seed(
            {
                "seed_role": "model0826_recovery_hyper_cd",
                "analysis_id": design.analysis_id,
                "dataset_id": specification.dataset_id,
                "candidate_cell": cell,
            }
        ),
        "max_trials": None,
        "evaluation_protocol": evaluation_protocol,
        "shortlist_size": int(search_config["shortlist_size"]),
        "cd": deepcopy(dict(search_config["cd"])),
    }
    stage_budgets = resolve_recovery_stage_budgets(
        search_config,
        {
            "particle_count": particle_count,
            "filter_seed_count": filter_seed_count,
        },
    )
    hyper_config = build_model_0826_hyper_config(
        analysis_config,
        load_model_parameter_space(
            design.parameter_space_path,
            expected_model_id="model_0826",
        ),
        cell,
        simulation_config_path,
        search_dir,
        {
            "coarse": stage_budgets["coarse"],
            "fine": stage_budgets["fine"],
            "final_rescore": {
                **stage_budgets["final_rescore"],
                "seed_family": final_seed_family,
            },
        },
    )
    _write_immutable_yaml(hyper_config_path, hyper_config, resume=resume)
    optimizer = optimizer_factory(hyper_config, hyper_config_path)
    checkpoint_path = (
        search_dir
        / f"subject_{int(specification.subject_id)}"
        / "search_checkpoint.json"
    )
    fit_result = optimizer.run(
        [int(specification.subject_id)],
        stage="all",
        resume=bool(resume and checkpoint_path.is_file()),
    )
    return {
        "dataset_id": specification.dataset_id,
        "candidate_cell": cell,
        "simulation_config_path": str(simulation_config_path),
        "hyper_config_path": str(hyper_config_path),
        "search_output_dir": str(search_dir),
        "fit_result": fit_result,
    }
