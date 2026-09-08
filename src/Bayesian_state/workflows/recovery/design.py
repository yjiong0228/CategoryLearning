"""Load and validate registered recovery experiment designs."""
from __future__ import annotations
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
from src.Bayesian_state.optimization.artifacts import to_builtin
from src.Bayesian_state.optimization.parameter_space import load_model_parameter_space
from src.Bayesian_state.utils.seeding import stable_seed
from src.Bayesian_state.optimization.recovery_parameters import (
    _declared_support,
)
from src.Bayesian_state.simulation.recovery import (
    CELL_FREE_PARAMETERS,
    MODEL_PARAMETER_NAMES,
    RecoveryDatasetSpec,
    RecoveryDesign,
)
from src.Bayesian_state.utils.recovery_artifacts import (
    _load_yaml,
)


def _resolve_relative(source: Path, raw: Any) -> Path:
    path = Path(str(raw))
    return path.resolve() if path.is_absolute() else (source.parent / path).resolve()


def _generation_seed(
    analysis_id: str,
    family: str,
    dataset_id: str,
    base_seed: int,
    truth: Mapping[str, Any],
) -> int:
    return stable_seed(
        {
            "seed_role": "model0826_recovery_autonomous_generation",
            "analysis_id": str(analysis_id),
            "family": str(family),
            "dataset_id": str(dataset_id),
            "base_seed": int(base_seed),
            "truth": to_builtin(dict(truth)),
        }
    )


def _module_specs(
    config: Mapping[str, Any],
    subject_trial_counts: Mapping[int, int],
    analysis_id: str,
    base_seed: int,
) -> tuple[RecoveryDatasetSpec, ...]:
    module = dict(config["module_recovery"])
    cells = [str(value).upper() for value in module["cells"]]
    if cells != ["P", "PM", "PH", "PMH"]:
        raise ValueError("module_recovery.cells must be P, PM, PH, PMH in order")
    templates = [int(value) for value in module["templates"]]
    if templates != [101, 111, 118]:
        raise ValueError("module_recovery.templates must be 101, 111, 118")
    repeats = int(module["replicates_per_cell_template"])
    if repeats != 3:
        raise ValueError("module recovery requires exactly three trajectories per cell/template")
    train_fraction = float(module["train_fraction"])
    if train_fraction != 0.70:
        raise ValueError("module recovery train_fraction must equal 0.70")
    shared_truth = dict(module["truth"])
    rows: list[RecoveryDatasetSpec] = []
    for cell in cells:
        truth = {
            name: deepcopy(shared_truth[name])
            for name in CELL_FREE_PARAMETERS[cell]
        }
        for subject_id in templates:
            trial_count = int(subject_trial_counts[subject_id])
            train_count = int(np.floor(trial_count * train_fraction))
            for replicate in range(1, repeats + 1):
                dataset_id = (
                    f"module_{cell}_subject_{subject_id}_replicate_{replicate:02d}"
                )
                rows.append(
                    RecoveryDatasetSpec(
                        dataset_id=dataset_id,
                        family="module",
                        subject_id=subject_id,
                        trial_count=trial_count,
                        replicate=replicate,
                        truth_cell=cell,
                        truth_profile="module_anchor",
                        truth=deepcopy(truth),
                        generation_seed=_generation_seed(
                            analysis_id,
                            "module",
                            dataset_id,
                            base_seed,
                            truth,
                        ),
                        train_trial_count=train_count,
                        evaluation_trial_count=trial_count - train_count,
                    )
                )
    return tuple(rows)


def _parameter_specs(
    config: Mapping[str, Any],
    subject_trial_counts: Mapping[int, int],
    analysis_id: str,
    base_seed: int,
) -> tuple[RecoveryDatasetSpec, ...]:
    parameter = dict(config["parameter_recovery"])
    profile_rows: list[tuple[str, list[int], dict[str, Any]]] = []
    local = dict(parameter["local_center"])
    profile_rows.append(
        (
            str(local["profile_id"]),
            [int(value) for value in local["template_assignments"]],
            dict(local["truth"]),
        )
    )
    contrasts = dict(parameter["contrast_profiles"])
    if list(contrasts) != ["C1", "C2", "C3", "C4", "C5", "C6"]:
        raise ValueError("parameter contrast profiles must be ordered C1--C6")
    for profile_id, raw in contrasts.items():
        specification = dict(raw)
        profile_rows.append(
            (
                str(profile_id),
                [int(value) for value in specification["template_assignments"]],
                dict(specification["truth"]),
            )
        )

    rows: list[RecoveryDatasetSpec] = []
    for profile_id, assignments, truth in profile_rows:
        if set(truth) != set(MODEL_PARAMETER_NAMES):
            raise ValueError(
                f"parameter profile {profile_id} must define every PMH parameter"
            )
        for replicate, subject_id in enumerate(assignments, start=1):
            if subject_id not in subject_trial_counts:
                raise ValueError(f"unknown template subject {subject_id}")
            dataset_id = (
                f"parameter_{profile_id}_subject_{subject_id}_replicate_{replicate:02d}"
            )
            trial_count = int(subject_trial_counts[subject_id])
            rows.append(
                RecoveryDatasetSpec(
                    dataset_id=dataset_id,
                    family="parameter",
                    subject_id=subject_id,
                    trial_count=trial_count,
                    replicate=replicate,
                    truth_cell="PMH",
                    truth_profile=profile_id,
                    truth=deepcopy(truth),
                    generation_seed=_generation_seed(
                        analysis_id,
                        "parameter",
                        dataset_id,
                        base_seed,
                        truth,
                    ),
                    train_trial_count=trial_count,
                    evaluation_trial_count=0,
                )
            )
    return tuple(rows)


def _validate_parameter_truth_support(
    parameter_space: Mapping[str, Any],
    datasets: Sequence[RecoveryDatasetSpec],
) -> None:
    for dataset in datasets:
        truth = dataset.truth
        if "M" in truth:
            candidate = {"M": int(truth["M"]), "chi": int(truth["chi"])}
            if candidate not in _declared_support(
                parameter_space, "workspace_execution"
            ):
                raise ValueError(f"{dataset.dataset_id} workspace truth is off support")
        for name in (
            "gamma", "E_C", "delta_E", "g_0", "c_A", "c_G",
            "beta_0", "eta_plus", "eta_minus",
        ):
            if name in truth and float(truth[name]) not in _declared_support(
                parameter_space, name
            ):
                raise ValueError(
                    f"{dataset.dataset_id} truth {name}={truth[name]} is off support"
                )


def load_recovery_design(path: str | Path) -> RecoveryDesign:
    """Load and materialize the pre-registered 36+40 recovery datasets."""

    source = Path(path).resolve()
    config = _load_yaml(source)
    analysis_id = str(config.get("analysis_id", ""))
    if analysis_id not in {
        "model_0826_recovery_v1",
        "model_0826_recovery_v2",
    }:
        raise ValueError(
            "analysis_id must be model_0826_recovery_v1 or "
            "model_0826_recovery_v2"
        )
    if config.get("model_id") != "model_0826":
        raise ValueError("model_id must be model_0826")
    subject_rows = list(config["subjects"])
    subject_trial_counts = {
        int(row["subject_id"]): int(row["trial_count"])
        for row in subject_rows
    }
    if subject_trial_counts != {101: 320, 111: 320, 118: 256}:
        raise ValueError("recovery subjects must have trial counts 320/320/256")
    generation = dict(config["generation"])
    if generation.get("observed_choices_used") is not False:
        raise ValueError("recovery generation must not use observed choices")
    base_seed = int(generation["base_seed"])
    module_datasets = _module_specs(
        config, subject_trial_counts, analysis_id, base_seed
    )
    parameter_datasets = _parameter_specs(
        config, subject_trial_counts, analysis_id, base_seed
    )
    if len(module_datasets) != 36 or len(parameter_datasets) != 40:
        raise ValueError("recovery design must materialize exactly 36+40 datasets")
    all_ids = [row.dataset_id for row in module_datasets + parameter_datasets]
    all_seeds = [row.generation_seed for row in module_datasets + parameter_datasets]
    if len(all_ids) != len(set(all_ids)) or len(all_seeds) != len(set(all_seeds)):
        raise ValueError("recovery dataset ids and generation seeds must be unique")
    if sum(int(row.truth["chi"]) for row in parameter_datasets) != 20:
        raise ValueError("parameter recovery must contain 20 chi=0 and 20 chi=1")

    parameter_space_path = _resolve_relative(source, config["parameter_space"])
    model_engine_config = _resolve_relative(source, config["model_engine_config"])
    base_simulation_config = _resolve_relative(
        source, config["base_simulation_config"]
    )
    for required_path in (
        parameter_space_path,
        model_engine_config,
        base_simulation_config,
    ):
        if not required_path.is_file():
            raise ValueError(f"Model0826 recovery input does not exist: {required_path}")
    parameter_space = load_model_parameter_space(
        parameter_space_path,
        expected_model_id="model_0826",
    )
    _validate_parameter_truth_support(
        parameter_space,
        module_datasets + parameter_datasets,
    )
    base_simulation = _load_yaml(base_simulation_config)
    if [int(value) for value in base_simulation.get("subjects", [])] != [
        101, 111, 118
    ]:
        raise ValueError("recovery base simulation subjects must be 101/111/118")
    if base_simulation.get("max_trials", "missing") is not None:
        raise ValueError("recovery base simulation max_trials must be null")
    return RecoveryDesign(
        analysis_id=analysis_id,
        source_path=source,
        model_engine_config=model_engine_config,
        parameter_space_path=parameter_space_path,
        base_simulation_config=base_simulation_config,
        output_root=_resolve_relative(source, config["output_root"]),
        subject_trial_counts=subject_trial_counts,
        module_datasets=module_datasets,
        parameter_datasets=parameter_datasets,
        config=config,
    )
