"""Reproducible Model0826 module and parameter recovery utilities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from src.Bayesian_state.optimization.artifacts import to_builtin
from src.Bayesian_state.optimization.model_0826 import (
    ACCUMULATOR_GAIN_PATH,
    BETA_PATH,
    CAPACITY_PATH,
    ETA_MINUS_PATH,
    ETA_PLUS_PATH,
    EVENT_CORRECT_PATH,
    EVENT_ERROR_PATH,
    EXECUTION_PATH,
    GAMMA_PATH,
    GLOBAL_GAIN_PATH,
    GLOBAL_SEARCH_PATH,
    INITIAL_EVENT_PATH,
    build_model_0826_cell_engine,
)
from src.Bayesian_state.optimization.parameter_space import (
    load_model_parameter_space,
    reactive_error_probability,
)
from src.Bayesian_state.simulation.autonomous import (
    run_autonomous_category_learning,
)
from src.Bayesian_state.simulation.parameters import (
    apply_fixed_hyperparams_to_engine_config,
)
from src.Bayesian_state.utils.seeding import stable_seed


FEATURE_COLUMNS = ("feature1", "feature2", "feature3", "feature4")
ORDER_COLUMNS = ("iSession", "iBlock", "iTrial")
SCHEDULE_COLUMNS = (*ORDER_COLUMNS, *FEATURE_COLUMNS, "category")
MODEL_PARAMETER_NAMES = (
    "M",
    "chi",
    "gamma",
    "E_C",
    "delta_E",
    "g_0",
    "c_A",
    "c_G",
    "beta_0",
    "eta_plus",
    "eta_minus",
)
CELL_FREE_PARAMETERS = {
    "P": ("beta_0", "eta_plus", "eta_minus"),
    "PM": ("gamma", "beta_0", "eta_plus", "eta_minus"),
    "PH": (
        "M", "chi", "E_C", "delta_E", "g_0", "c_A", "c_G",
        "beta_0", "eta_plus", "eta_minus",
    ),
    "PMH": MODEL_PARAMETER_NAMES,
}


@dataclass(frozen=True)
class RecoveryDatasetSpec:
    dataset_id: str
    family: str
    subject_id: int
    trial_count: int
    replicate: int
    truth_cell: str
    truth_profile: str
    truth: dict[str, Any]
    generation_seed: int
    train_trial_count: int
    evaluation_trial_count: int


@dataclass(frozen=True)
class RecoveryDesign:
    analysis_id: str
    source_path: Path
    model_engine_config: Path
    parameter_space_path: Path
    base_simulation_config: Path
    output_root: Path
    subject_trial_counts: dict[int, int]
    module_datasets: tuple[RecoveryDatasetSpec, ...]
    parameter_datasets: tuple[RecoveryDatasetSpec, ...]
    config: dict[str, Any]

    @property
    def all_datasets(self) -> tuple[RecoveryDatasetSpec, ...]:
        return self.module_datasets + self.parameter_datasets


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read Model0826 recovery config: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid Model0826 recovery YAML: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Model0826 recovery YAML root must be a mapping")
    return deepcopy(dict(payload))


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


def _declared_support(parameter_space: Mapping[str, Any], name: str) -> list[Any]:
    specification = dict(parameter_space["subject_parameters"][name])
    if name == "workspace_execution":
        return [dict(value) for value in specification["fine_candidates"]]
    if specification["kind"] == "spike_and_positive_grid":
        return [
            float(specification["zero_value"]),
            *map(float, specification["fine_positive_values"]),
        ]
    return [float(value) for value in specification["fine_values"]]


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
    if config.get("analysis_id") != "model_0826_recovery_v1":
        raise ValueError("analysis_id must be model_0826_recovery_v1")
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
    analysis_id = str(config["analysis_id"])
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


def schedule_fingerprint(schedule_frame: pd.DataFrame) -> str:
    """Hash only task order, stimulus, and category—not observed behavior."""

    missing = set(SCHEDULE_COLUMNS) - set(schedule_frame.columns)
    if missing:
        raise ValueError(f"schedule is missing columns: {sorted(missing)}")
    schedule = schedule_frame.loc[:, list(SCHEDULE_COLUMNS)].copy()
    if schedule.empty:
        raise ValueError("schedule cannot be empty")
    encoded = schedule.to_csv(index=False, float_format="%.17g").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def synthetic_dataset_frame(
    schedule_frame: pd.DataFrame,
    *,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Replace observed behavior with one autonomously generated trajectory."""

    generated_choice = np.asarray(choices, dtype=int).reshape(-1)
    generated_feedback = np.asarray(feedback, dtype=float).reshape(-1)
    if len(schedule_frame) != generated_choice.size or (
        generated_feedback.size != generated_choice.size
    ):
        raise ValueError("generated choices/feedback must align with the schedule")
    if not np.all(np.isin(generated_choice, [1, 2])):
        raise ValueError("generated choices must be encoded as 1 or 2")
    if not np.all(np.isfinite(generated_feedback)):
        raise ValueError("generated feedback must be finite")
    leading = [
        name
        for name in ("iSub", "condition", *SCHEDULE_COLUMNS)
        if name in schedule_frame.columns
    ]
    frame = schedule_frame.loc[:, leading].copy().reset_index(drop=True)
    frame["choice"] = generated_choice
    frame["feedback"] = generated_feedback
    frame.attrs["observed_choices_used"] = False
    frame.attrs["schedule_fingerprint"] = schedule_fingerprint(schedule_frame)
    return frame


def _truth_hyperparams(specification: RecoveryDatasetSpec) -> dict[str, Any]:
    truth = specification.truth
    hyperparams: dict[str, Any] = {
        BETA_PATH: float(truth["beta_0"]),
        ETA_PLUS_PATH: float(truth["eta_plus"]),
        ETA_MINUS_PATH: float(truth["eta_minus"]),
    }
    if "gamma" in truth:
        hyperparams[GAMMA_PATH] = float(truth["gamma"])
    if "M" in truth:
        event_correct = float(truth["E_C"])
        hyperparams.update(
            {
                CAPACITY_PATH: int(truth["M"]),
                EXECUTION_PATH: bool(int(truth["chi"])),
                EVENT_CORRECT_PATH: event_correct,
                EVENT_ERROR_PATH: reactive_error_probability(
                    event_correct, float(truth["delta_E"])
                ),
                INITIAL_EVENT_PATH: event_correct,
                GLOBAL_SEARCH_PATH: float(truth["g_0"]),
                ACCUMULATOR_GAIN_PATH: float(truth["c_A"]),
                GLOBAL_GAIN_PATH: float(truth["c_G"]),
            }
        )
    return hyperparams


def _canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        to_builtin(dict(payload)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(
                to_builtin(dict(payload)),
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def generate_synthetic_dataset(
    specification: RecoveryDatasetSpec,
    *,
    schedule_frame: pd.DataFrame,
    base_engine_config: Mapping[str, Any],
    output_dir: str | Path,
    processed_data_dir: str | Path | None = None,
    dataset_paths: Mapping[str, str | Path] | None = None,
    resume: bool = False,
    generator: Callable[..., Any] = run_autonomous_category_learning,
) -> dict[str, Any]:
    """Generate and atomically store one autonomous recovery observation."""

    if len(schedule_frame) != int(specification.trial_count):
        raise ValueError(
            f"{specification.dataset_id} requires {specification.trial_count} trials"
        )
    fingerprint_payload = {
        "specification": asdict(specification),
        "schedule_fingerprint": schedule_fingerprint(schedule_frame),
        "base_engine_config": to_builtin(dict(base_engine_config)),
    }
    fingerprint = _canonical_fingerprint(fingerprint_payload)
    output = Path(output_dir)
    csv_path = output / f"{specification.dataset_id}.csv"
    npz_path = output / f"{specification.dataset_id}.npz"
    manifest_path = output / f"{specification.dataset_id}.manifest.json"
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(
                f"synthetic recovery dataset already exists: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError("synthetic recovery manifest fingerprint does not match")
        if (
            manifest.get("status") != "complete"
            or not csv_path.is_file()
            or not npz_path.is_file()
        ):
            raise ValueError("synthetic recovery cache is incomplete")
        return dict(manifest)

    engine = build_model_0826_cell_engine(
        base_engine_config,
        specification.truth_cell,
    )
    engine = apply_fixed_hyperparams_to_engine_config(
        engine,
        _truth_hyperparams(specification),
    )
    stimulus = schedule_frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    categories = schedule_frame["category"].to_numpy(dtype=int)
    result = generator(
        engine_config=engine,
        subject_id=int(specification.subject_id),
        condition=1,
        stimulus=stimulus,
        categories=categories,
        trajectory_seed=int(specification.generation_seed),
        processed_data_dir=processed_data_dir,
        dataset_paths=dataset_paths,
    )
    trajectory = result.trajectory
    choices = np.asarray(trajectory.choices, dtype=int).reshape(-1)
    feedback = np.asarray(trajectory.feedback, dtype=float).reshape(-1)
    probabilities = np.asarray(
        trajectory.observed_probabilities,
        dtype=float,
    )
    if choices.size != specification.trial_count or feedback.size != choices.size:
        raise ValueError("autonomous generation returned the wrong trial count")
    if probabilities.shape != (choices.size, 2):
        raise ValueError("generated choice probabilities must have shape (T, 2)")
    if not np.all(np.isfinite(probabilities)) or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-8
    ):
        raise ValueError("generated choice probabilities are invalid")
    generated_frame = synthetic_dataset_frame(
        schedule_frame,
        choices=choices,
        feedback=feedback,
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "fingerprint": fingerprint,
        "dataset_id": specification.dataset_id,
        "family": specification.family,
        "subject_id": int(specification.subject_id),
        "trial_count": int(specification.trial_count),
        "truth_cell": specification.truth_cell,
        "truth_profile": specification.truth_profile,
        "truth": deepcopy(specification.truth),
        "generation_seed": int(specification.generation_seed),
        "schedule_fingerprint": fingerprint_payload["schedule_fingerprint"],
        "generated_accuracy": float(np.mean(feedback)),
        "observed_choices_used": False,
        "csv_path": str(csv_path),
        "npz_path": str(npz_path),
    }
    _atomic_npz(
        npz_path,
        stimulus=stimulus.astype(np.float64),
        categories=categories.astype(np.int8),
        choices=choices.astype(np.int8),
        feedback=feedback.astype(np.float64),
        generated_choice_probabilities=probabilities.astype(np.float64),
        metadata_json=np.asarray(
            json.dumps(to_builtin(manifest), sort_keys=True, allow_nan=False)
        ),
    )
    _atomic_csv(csv_path, generated_frame)
    _atomic_json(manifest_path, manifest)
    return manifest


__all__ = [
    "CELL_FREE_PARAMETERS",
    "FEATURE_COLUMNS",
    "MODEL_PARAMETER_NAMES",
    "ORDER_COLUMNS",
    "RecoveryDatasetSpec",
    "RecoveryDesign",
    "generate_synthetic_dataset",
    "load_recovery_design",
    "schedule_fingerprint",
    "synthetic_dataset_frame",
]
