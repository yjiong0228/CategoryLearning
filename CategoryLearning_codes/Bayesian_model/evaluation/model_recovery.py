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
from joblib import Parallel, delayed
from scipy.stats import spearmanr

from CategoryLearning_codes.Bayesian_model.inference.backends.particle_filter import (
    run_state_model_particle_filter,
)
from CategoryLearning_codes.Bayesian_model.hypothesis_space.geometry import warmup_dykstra_numba
from CategoryLearning_codes.Bayesian_model.model.readout import (
    resolve_choice_readout_config,
    resolve_output_noise_config,
)
from CategoryLearning_codes.Bayesian_model.optimization.artifacts import to_builtin
from CategoryLearning_codes.Bayesian_model.optimization.model_0826 import (
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
    build_model_0826_hyper_config,
)
from CategoryLearning_codes.Bayesian_model.optimization.search.coordinate_descent import (
    HyperCDOptimizer,
)
from CategoryLearning_codes.Bayesian_model.optimization.parameter_space import (
    load_model_parameter_space,
    reactive_error_probability,
)
from CategoryLearning_codes.Bayesian_model.simulation.autonomous import (
    run_autonomous_category_learning,
)
from CategoryLearning_codes.Bayesian_model.simulation.config import (
    EVALUATION_ROLE_SIMULATION,
    resolve_evaluation_score_mask,
)
from CategoryLearning_codes.Bayesian_model.simulation.parameters import (
    apply_fixed_hyperparams_to_engine_config,
)
from CategoryLearning_codes.Bayesian_model.utils.seeding import stable_seed
from CategoryLearning_codes.Bayesian_model.utils.datasets import resolve_dataset_paths


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


def model_0826_truth_hyperparams(
    truth: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert named recovery truth values to executable engine paths."""

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


def _truth_hyperparams(specification: RecoveryDatasetSpec) -> dict[str, Any]:
    return model_0826_truth_hyperparams(specification.truth)


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


def build_calibration_bank(anchor: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen eight-candidate numerical calibration bank."""

    required = set(MODEL_PARAMETER_NAMES) - {"chi"}
    if set(anchor) != required:
        raise ValueError(
            "calibration anchor must define every PMH parameter except chi"
        )
    variants: list[tuple[str, dict[str, Any]]] = []
    variants.append(("anchor", deepcopy(dict(anchor))))
    gamma = deepcopy(dict(anchor))
    gamma["gamma"] = 0.50
    variants.append(("gamma_050", gamma))
    gains = deepcopy(dict(anchor))
    gains.update({"c_A": 0.0, "c_G": 0.0})
    variants.append(("gains_zero", gains))
    beta = deepcopy(dict(anchor))
    beta.update({"beta_0": 1.0, "eta_plus": 0.01, "eta_minus": 0.03})
    variants.append(("beta_slow", beta))

    bank: list[dict[str, Any]] = []
    for variant, values in variants:
        for chi in (0, 1):
            truth = deepcopy(values)
            truth["chi"] = chi
            bank.append(
                {
                    "candidate_id": f"{variant}_chi_{chi}",
                    "variant": variant,
                    "truth": truth,
                }
            )
    return bank


def resolve_calibration_filter_seeds(
    *,
    dataset_id: str,
    base_seed: int,
    ensemble: str,
    count: int,
) -> list[int]:
    """Resolve nested logical seeds within A/B and disjoint seeds between them."""

    ensemble_name = str(ensemble).strip().upper()
    if ensemble_name not in {"A", "B"}:
        raise ValueError("calibration ensemble must be A or B")
    seed_count = int(count)
    if seed_count <= 0:
        raise ValueError("calibration seed count must be positive")
    return [
        stable_seed(
            {
                "seed_role": "model0826_recovery_pf_calibration",
                "dataset_id": str(dataset_id),
                "base_seed": int(base_seed),
                "ensemble": ensemble_name,
                "logical_seed_index": int(index),
            }
        )
        for index in range(seed_count)
    ]


def _frozen_readout_args(engine_config: Mapping[str, Any]) -> dict[str, float]:
    readout = resolve_choice_readout_config(None, engine_config)
    noise = resolve_output_noise_config(None, engine_config)
    if (
        readout["method"] != "expectation"
        or float(readout["power"]) != 1.0
        or float(readout["strategy_confidence_gain"]) != 0.0
        or float(readout["rule_commitment_confidence_gain"]) != 0.0
    ):
        raise ValueError("Model0826 recovery requires the frozen expectation readout")
    noise_terms = (
        "base_lapse",
        "post_error_lapse",
        "low_accuracy_lapse",
        "latent_volatility_lapse",
    )
    if any(float(noise.get(name, 0.0)) != 0.0 for name in noise_terms):
        raise ValueError("Model0826 recovery requires zero output lapse")
    return {
        "choice_readout_power": 1.0,
        "strategy_confidence_gain": 0.0,
        "rule_commitment_confidence_gain": 0.0,
        "output_lapse": 0.0,
    }


def score_pf_bank(
    *,
    dataset_id: str,
    subject_id: int,
    stimulus: Sequence[Sequence[float]] | np.ndarray,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    base_engine_config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    particle_count: int,
    filter_seeds: Sequence[int],
    ensemble: str,
    resample_threshold_fraction: float = 0.5,
    processed_data_dir: str | Path | None = None,
    dataset_paths: Mapping[str, str | Path] | None = None,
    pf_runner: Callable[..., Any] = run_state_model_particle_filter,
) -> list[dict[str, Any]]:
    """Score one fixed candidate bank using paired PF seeds."""

    physical = np.asarray(stimulus, dtype=float)
    observed_choice = np.asarray(choices, dtype=int).reshape(-1)
    observed_feedback = np.asarray(feedback, dtype=float).reshape(-1)
    if physical.ndim != 2 or physical.shape[0] != observed_choice.size:
        raise ValueError("PF calibration stimulus and choices are misaligned")
    if observed_feedback.size != observed_choice.size:
        raise ValueError("PF calibration feedback and choices are misaligned")
    seeds = [int(value) for value in filter_seeds]
    if not seeds:
        raise ValueError("PF calibration requires at least one filter seed")
    if len(seeds) != len(set(seeds)):
        raise ValueError("PF calibration filter seeds must be unique")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        truth = dict(candidate["truth"])
        engine = build_model_0826_cell_engine(base_engine_config, "PMH")
        engine = apply_fixed_hyperparams_to_engine_config(
            engine,
            model_0826_truth_hyperparams(truth),
        )
        readout_args = _frozen_readout_args(engine)
        probability_runs: list[np.ndarray] = []
        for filter_seed in seeds:
            result = pf_runner(
                engine_config=engine,
                subject_id=int(subject_id),
                stimulus=physical,
                choices=observed_choice,
                feedback=observed_feedback,
                particle_count=int(particle_count),
                filter_seed=int(filter_seed),
                resample_threshold_fraction=float(resample_threshold_fraction),
                processed_data_dir=processed_data_dir,
                dataset_paths=dataset_paths,
                **readout_args,
            )
            probabilities = np.asarray(result.marginal_probabilities, dtype=float)
            if probabilities.shape != (observed_choice.size, 2):
                raise ValueError("PF calibration probabilities must have shape (T, 2)")
            if not np.all(np.isfinite(probabilities)) or not np.allclose(
                probabilities.sum(axis=1), 1.0, atol=1e-8
            ):
                raise ValueError("PF calibration returned invalid probabilities")
            probability_runs.append(probabilities)
        stack = np.stack(probability_runs, axis=0)
        mean_probability = np.mean(stack, axis=0)
        selected = mean_probability[
            np.arange(observed_choice.size), observed_choice - 1
        ]
        total_nll = float(-np.log(np.clip(selected, 1e-12, 1.0)).sum())
        if stack.shape[0] > 1:
            probability_mcse = np.std(stack[:, :, 1], axis=0, ddof=1) / np.sqrt(
                float(stack.shape[0])
            )
        else:
            probability_mcse = np.zeros(observed_choice.size, dtype=float)
        rows.append(
            {
                "dataset_id": str(dataset_id),
                "subject_id": int(subject_id),
                "candidate_id": candidate_id,
                "variant": str(candidate["variant"]),
                "candidate_chi": int(truth["chi"]),
                "particle_count": int(particle_count),
                "filter_seed_count": int(len(seeds)),
                "ensemble": str(ensemble).upper(),
                "filter_seeds": seeds,
                "total_nll": total_nll,
                "mean_trial_nll": total_nll / float(observed_choice.size),
                "mean_probability": mean_probability,
                "probability_runs": stack,
                "trial_probability_mcse": probability_mcse,
                "probability_aggregation": "mean_probability_then_nll",
            }
        )
    return rows


def _score_pf_candidate_seed(
    *,
    common_kwargs: Mapping[str, Any],
    candidate: Mapping[str, Any],
    filter_seed: int,
) -> dict[str, Any]:
    return score_pf_bank(
        **dict(common_kwargs),
        candidates=[dict(candidate)],
        filter_seeds=[int(filter_seed)],
    )[0]


def score_pf_bank_parallel(
    *,
    dataset_id: str,
    subject_id: int,
    stimulus: Sequence[Sequence[float]] | np.ndarray,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    base_engine_config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    particle_count: int,
    filter_seeds: Sequence[int],
    ensemble: str,
    n_jobs: int,
    resample_threshold_fraction: float = 0.5,
    processed_data_dir: str | Path | None = None,
    dataset_paths: Mapping[str, str | Path] | None = None,
    pf_runner: Callable[..., Any] = run_state_model_particle_filter,
) -> list[dict[str, Any]]:
    """Parallelize independent candidate×seed PF runs, then aggregate exactly."""

    candidate_rows = [deepcopy(dict(candidate)) for candidate in candidates]
    seeds = [int(value) for value in filter_seeds]
    if not candidate_rows:
        raise ValueError("parallel PF calibration requires at least one candidate")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("parallel PF calibration requires unique filter seeds")
    jobs = min(int(n_jobs), len(candidate_rows) * len(seeds))
    if jobs < 1:
        raise ValueError("parallel PF calibration n_jobs must be positive")
    common_kwargs = {
        "dataset_id": str(dataset_id),
        "subject_id": int(subject_id),
        "stimulus": np.asarray(stimulus, dtype=float),
        "choices": np.asarray(choices, dtype=int),
        "feedback": np.asarray(feedback, dtype=float),
        "base_engine_config": deepcopy(dict(base_engine_config)),
        "particle_count": int(particle_count),
        "ensemble": str(ensemble),
        "resample_threshold_fraction": float(resample_threshold_fraction),
        "processed_data_dir": processed_data_dir,
        "dataset_paths": dataset_paths,
        "pf_runner": pf_runner,
    }
    if jobs == 1:
        return score_pf_bank(
            **common_kwargs,
            candidates=candidate_rows,
            filter_seeds=seeds,
        )
    warmup_dykstra_numba()
    single_rows = Parallel(n_jobs=jobs, backend="loky", verbose=10)(
        delayed(_score_pf_candidate_seed)(
            common_kwargs=common_kwargs,
            candidate=candidate,
            filter_seed=filter_seed,
        )
        for candidate in candidate_rows
        for filter_seed in seeds
    )
    observed = np.asarray(choices, dtype=int).reshape(-1)
    combined: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidate_rows):
        start = candidate_index * len(seeds)
        candidate_seed_rows = single_rows[start : start + len(seeds)]
        stack = np.stack(
            [
                np.asarray(row["mean_probability"], dtype=float)
                for row in candidate_seed_rows
            ],
            axis=0,
        )
        if stack.shape[0] > 1:
            probability_mcse = np.std(stack[:, :, 1], axis=0, ddof=1) / np.sqrt(
                float(stack.shape[0])
            )
        else:
            probability_mcse = np.zeros(observed.size, dtype=float)
        total_nll = mean_probability_nll(stack, observed)
        row = dict(candidate_seed_rows[0])
        row.update(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "filter_seed_count": int(len(seeds)),
                "filter_seeds": seeds,
                "total_nll": total_nll,
                "mean_trial_nll": total_nll / float(observed.size),
                "mean_probability": np.mean(stack, axis=0),
                "probability_runs": stack,
                "trial_probability_mcse": probability_mcse,
                "parallel_n_jobs": int(jobs),
            }
        )
        combined.append(row)
    return combined


def _setting_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int, str], dict[str, Mapping[str, Any]]]:
    settings: dict[
        tuple[str, int, int, str], dict[str, Mapping[str, Any]]
    ] = {}
    for row in rows:
        key = (
            str(row["dataset_id"]),
            int(row["particle_count"]),
            int(row["filter_seed_count"]),
            str(row["ensemble"]).upper(),
        )
        candidate_id = str(row["candidate_id"])
        candidate_rows = settings.setdefault(key, {})
        if candidate_id in candidate_rows:
            raise ValueError("duplicate PF calibration candidate row")
        candidate_rows[candidate_id] = row
    return settings


def _compare_pf_settings(
    settings: Mapping[
        tuple[str, int, int, str], Mapping[str, Mapping[str, Any]]
    ],
    left: tuple[int, int, str],
    right: tuple[int, int, str],
) -> list[dict[str, Any]]:
    dataset_ids = sorted({key[0] for key in settings})
    comparisons: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        left_rows = settings.get((dataset_id, *left))
        right_rows = settings.get((dataset_id, *right))
        if left_rows is None or right_rows is None:
            continue
        candidate_ids = sorted(left_rows)
        if candidate_ids != sorted(right_rows):
            raise ValueError("PF calibration candidate banks differ between settings")
        left_nll = np.asarray(
            [float(left_rows[name]["total_nll"]) for name in candidate_ids]
        )
        right_nll = np.asarray(
            [float(right_rows[name]["total_nll"]) for name in candidate_ids]
        )
        right_winner_index = int(np.argmin(right_nll))
        left_order = np.argsort(left_nll, kind="stable")
        right_winner_rank_in_left = int(
            np.flatnonzero(left_order == right_winner_index)[0] + 1
        )
        rho = float(spearmanr(left_nll, right_nll).statistic)
        if not np.isfinite(rho):
            rho = 1.0 if np.allclose(left_nll, right_nll) else 0.0
        probability_differences = []
        for name in candidate_ids:
            left_probability = np.asarray(
                left_rows[name]["mean_probability"], dtype=float
            )
            right_probability = np.asarray(
                right_rows[name]["mean_probability"], dtype=float
            )
            if left_probability.shape != right_probability.shape:
                raise ValueError("PF calibration probability shapes differ")
            probability_differences.append(
                np.square(left_probability - right_probability).reshape(-1)
            )
        comparisons.append(
            {
                "dataset_id": dataset_id,
                "left": {"particle_count": left[0], "filter_seed_count": left[1], "ensemble": left[2]},
                "right": {"particle_count": right[0], "filter_seed_count": right[1], "ensemble": right[2]},
                "candidate_nll_spearman": rho,
                "winner_agreement": bool(
                    candidate_ids[int(np.argmin(left_nll))]
                    == candidate_ids[right_winner_index]
                ),
                "right_winner_rank_in_left": right_winner_rank_in_left,
                "probability_rmse": float(
                    np.sqrt(np.mean(np.concatenate(probability_differences)))
                ),
            }
        )
    return comparisons


def summarize_search_budget_retention(
    score_rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Check whether low-budget stages retain the high-budget winner."""

    settings = _setting_rows(score_rows)
    reference_config = dict(policy.get("reference") or {})
    reference = (
        int(reference_config["particle_count"]),
        int(reference_config["filter_seed_count"]),
        str(reference_config.get("ensemble", "A")).upper(),
    )
    stage_configs = policy.get("stages") or {}
    if not isinstance(stage_configs, Mapping) or not stage_configs:
        raise ValueError("search budget retention requires at least one stage")

    stage_summaries: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for stage_name, raw_config in stage_configs.items():
        config = dict(raw_config)
        low = (
            int(config["particle_count"]),
            int(config["filter_seed_count"]),
            str(config.get("ensemble", "A")).upper(),
        )
        top_k = int(config["winner_top_k"])
        minimum_count = int(config["minimum_dataset_count"])
        if top_k < 1 or minimum_count < 1:
            raise ValueError(
                "winner_top_k and minimum_dataset_count must be positive"
            )
        rows = _compare_pf_settings(settings, low, reference)
        ranks = [int(row["right_winner_rank_in_left"]) for row in rows]
        retained_count = sum(rank <= top_k for rank in ranks)
        stage_summaries[str(stage_name)] = {
            "particle_count": low[0],
            "filter_seed_count": low[1],
            "ensemble": low[2],
            "reference": {
                "particle_count": reference[0],
                "filter_seed_count": reference[1],
                "ensemble": reference[2],
            },
            "winner_top_k": top_k,
            "minimum_dataset_count": minimum_count,
            "comparison_dataset_count": len(rows),
            "retained_dataset_count": int(retained_count),
            "right_winner_ranks": ranks,
            "median_candidate_nll_spearman": (
                float(np.median([row["candidate_nll_spearman"] for row in rows]))
                if rows
                else None
            ),
            "passes": bool(
                len(rows) >= minimum_count and retained_count >= minimum_count
            ),
        }
        comparison_rows.extend(
            {"stage": str(stage_name), **row} for row in rows
        )

    return {
        "status": (
            "passed"
            if all(row["passes"] for row in stage_summaries.values())
            else "failed"
        ),
        "stages": stage_summaries,
        "comparisons": comparison_rows,
    }


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


def _budget_mcse_q95(
    settings: Mapping[
        tuple[str, int, int, str], Mapping[str, Mapping[str, Any]]
    ],
    particle_count: int,
    filter_seed_count: int,
) -> float:
    values: list[np.ndarray] = []
    for (dataset_id, particles, seeds, ensemble), candidates in settings.items():
        del dataset_id
        if particles == particle_count and seeds == filter_seed_count and ensemble in {"A", "B"}:
            values.extend(
                np.asarray(row["trial_probability_mcse"], dtype=float).reshape(-1)
                for row in candidates.values()
            )
    if not values:
        return float("inf")
    return float(np.quantile(np.concatenate(values), 0.95))


def summarize_pf_calibration(
    score_rows: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the pre-registered rank, winner, RMSE, and MCSE gates."""

    settings = _setting_rows(score_rows)
    dataset_count = int(gates.get("dataset_count", 6))

    def decision(
        *,
        particle_count: int,
        filter_seed_count: int,
        scaling_pairs: Sequence[
            tuple[tuple[int, int, str], tuple[int, int, str]]
        ],
        independent_left: tuple[int, int, str],
        independent_right: tuple[int, int, str],
        extra_probability_comparison: tuple[
            tuple[int, int, str], tuple[int, int, str]
        ] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        scaling_by_pair = [
            _compare_pf_settings(settings, left, right)
            for left, right in scaling_pairs
        ]
        scaling = [row for pair_rows in scaling_by_pair for row in pair_rows]
        independent = _compare_pf_settings(
            settings, independent_left, independent_right
        )
        probability_rows = list(scaling) + list(independent)
        if extra_probability_comparison is not None:
            probability_rows.extend(
                _compare_pf_settings(settings, *extra_probability_comparison)
            )
        rank_values = [row["candidate_nll_spearman"] for row in scaling]
        scaling_winner_counts = [
            sum(bool(row["winner_agreement"]) for row in pair_rows)
            for pair_rows in scaling_by_pair
        ]
        winner_top_k = int(gates.get("adjacent_winner_top_k", 1))
        if winner_top_k < 1:
            raise ValueError("adjacent_winner_top_k must be positive")
        winner_top_k_min_count = int(
            gates.get(
                "adjacent_winner_top_k_min_count",
                gates.get("adjacent_winner_agreement_min_count", dataset_count),
            )
        )
        scaling_winner_top_k_counts = [
            sum(
                int(row["right_winner_rank_in_left"]) <= winner_top_k
                for row in pair_rows
            )
            for pair_rows in scaling_by_pair
        ]
        independent_winners = sum(
            bool(row["winner_agreement"]) for row in independent
        )
        probability_rmse = [row["probability_rmse"] for row in probability_rows]
        mcse_q95 = _budget_mcse_q95(
            settings, particle_count, filter_seed_count
        )
        complete = all(
            len(pair_rows) == dataset_count for pair_rows in scaling_by_pair
        ) and len(independent) == dataset_count
        metrics = {
            "particle_count": int(particle_count),
            "filter_seed_count": int(filter_seed_count),
            "complete": complete,
            "median_adjacent_rank_spearman": (
                float(np.median(rank_values)) if rank_values else None
            ),
            "minimum_adjacent_rank_spearman": (
                float(np.min(rank_values)) if rank_values else None
            ),
            "adjacent_winner_agreement_counts": [
                int(value) for value in scaling_winner_counts
            ],
            "minimum_adjacent_winner_agreement_count": (
                int(min(scaling_winner_counts)) if scaling_winner_counts else 0
            ),
            "adjacent_high_budget_winner_top_k": winner_top_k,
            "adjacent_high_budget_winner_top_k_counts": [
                int(value) for value in scaling_winner_top_k_counts
            ],
            "minimum_adjacent_high_budget_winner_top_k_count": (
                int(min(scaling_winner_top_k_counts))
                if scaling_winner_top_k_counts
                else 0
            ),
            "independent_winner_agreement_count": int(independent_winners),
            "median_probability_rmse": (
                float(np.median(probability_rmse)) if probability_rmse else None
            ),
            "trial_probability_mcse_q95": mcse_q95,
        }
        metrics["passes_all_gates"] = bool(
            complete
            and metrics["median_adjacent_rank_spearman"]
            >= float(gates["median_adjacent_rank_spearman_min"])
            and metrics["minimum_adjacent_rank_spearman"]
            >= float(gates["minimum_adjacent_rank_spearman_min"])
            and metrics["minimum_adjacent_high_budget_winner_top_k_count"]
            >= winner_top_k_min_count
            and independent_winners
            >= int(gates["independent_winner_agreement_min_count"])
            and metrics["median_probability_rmse"]
            <= float(gates["median_probability_rmse_max"])
            and mcse_q95 <= float(gates["trial_probability_mcse_q95_max"])
        )
        return metrics, probability_rows

    decisions: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    primary, primary_comparisons = decision(
        particle_count=64,
        filter_seed_count=8,
        scaling_pairs=(
            ((16, 4, "A"), (32, 4, "A")),
            ((32, 4, "A"), (64, 4, "A")),
        ),
        independent_left=(64, 8, "A"),
        independent_right=(64, 8, "B"),
        extra_probability_comparison=((64, 4, "A"), (64, 8, "A")),
    )
    decisions.append(primary)
    comparisons.extend(primary_comparisons)
    high_setting_keys = {
        (key[1], key[2], key[3]) for key in settings
    }
    if (128, 16, "A") in high_setting_keys and (128, 16, "B") in high_setting_keys:
        escalated, escalated_comparisons = decision(
            particle_count=128,
            filter_seed_count=16,
            scaling_pairs=(((64, 8, "A"), (128, 16, "A")),),
            independent_left=(128, 16, "A"),
            independent_right=(128, 16, "B"),
        )
        decisions.append(escalated)
        comparisons.extend(escalated_comparisons)
    return {
        "status": (
            "passed" if any(row["passes_all_gates"] for row in decisions) else "failed"
        ),
        "budget_decisions": decisions,
        "comparisons": comparisons,
    }


def freeze_smallest_passing_budget(
    summary: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, int] | None:
    """Return and optionally persist the smallest budget passing every gate."""

    decisions = [
        dict(row)
        for row in summary.get("budget_decisions", [])
        if bool(row.get("passes_all_gates", False))
    ]
    if not decisions:
        return None
    selected = min(
        decisions,
        key=lambda row: (
            int(row["particle_count"]) * int(row["filter_seed_count"]),
            int(row["particle_count"]),
            int(row["filter_seed_count"]),
        ),
    )
    budget = {
        "particle_count": int(selected["particle_count"]),
        "filter_seed_count": int(selected["filter_seed_count"]),
    }
    if output_path is not None:
        _atomic_json(
            Path(output_path),
            {
                "status": "frozen",
                **budget,
                "source_decision": selected,
            },
        )
    return budget


def mean_probability_nll(
    probability_runs: Sequence[Any] | np.ndarray,
    choices: Sequence[int] | np.ndarray,
    mask: Sequence[bool] | np.ndarray | None = None,
) -> float:
    """Average PF probabilities first, then compute masked total choice NLL."""

    runs = np.asarray(probability_runs, dtype=float)
    observed = np.asarray(choices, dtype=int).reshape(-1)
    if runs.ndim != 3 or runs.shape[1:] != (observed.size, 2):
        raise ValueError("probability_runs must have shape (B, T, 2)")
    if runs.shape[0] < 1 or not np.all(np.isfinite(runs)) or np.any(runs < 0.0):
        raise ValueError("probability_runs must contain finite nonnegative values")
    if not np.allclose(runs.sum(axis=2), 1.0, atol=1e-8):
        raise ValueError("probability rows must sum to one")
    if not np.all(np.isin(observed, [1, 2])):
        raise ValueError("choices must be encoded as 1 or 2")
    score_mask = np.ones(observed.size, dtype=bool)
    if mask is not None:
        score_mask = np.asarray(mask, dtype=bool).reshape(-1)
        if score_mask.size != observed.size:
            raise ValueError("NLL mask must align with choices")
    if not np.any(score_mask):
        raise ValueError("NLL mask must select at least one trial")
    mean_probability = np.mean(runs, axis=0)
    selected = mean_probability[np.arange(observed.size), observed - 1]
    return float(-np.log(np.clip(selected[score_mask], 1e-12, 1.0)).sum())


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(
                to_builtin(dict(payload)),
                stream,
                sort_keys=False,
                allow_unicode=True,
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_immutable_yaml(
    path: Path,
    payload: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if path.exists():
        existing = _load_yaml(path)
        if _canonical_fingerprint(existing) != _canonical_fingerprint(payload):
            raise ValueError(f"recovery config fingerprint does not match: {path}")
        if not resume:
            raise FileExistsError(f"recovery config already exists: {path}")
        return
    _atomic_yaml(path, payload)


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


def score_frozen_candidate(
    *,
    subject_id: int,
    stimulus: Sequence[Sequence[float]] | np.ndarray,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    base_engine_config: Mapping[str, Any],
    candidate_cell: str,
    fixed_hyperparams: Mapping[str, Any],
    particle_count: int,
    filter_seeds: Sequence[int],
    evaluation_protocol: Mapping[str, Any] | None,
    n_jobs: int = 1,
    resample_threshold_fraction: float = 0.5,
    processed_data_dir: str | Path | None = None,
    dataset_paths: Mapping[str, str | Path] | None = None,
    pf_runner: Callable[..., Any] = run_state_model_particle_filter,
) -> dict[str, Any]:
    """Run frozen parameters on the full sequence and score only the held-out mask."""

    physical = np.asarray(stimulus, dtype=float)
    observed = np.asarray(choices, dtype=int).reshape(-1)
    observed_feedback = np.asarray(feedback, dtype=float).reshape(-1)
    if physical.ndim != 2 or physical.shape[0] != observed.size:
        raise ValueError("frozen scoring arrays are misaligned")
    if observed_feedback.size != observed.size:
        raise ValueError("frozen scoring feedback is misaligned")
    score_mask, score_context = resolve_evaluation_score_mask(
        observed.size,
        evaluation_protocol,
        role=EVALUATION_ROLE_SIMULATION,
    )
    engine = build_model_0826_cell_engine(base_engine_config, candidate_cell)
    engine = apply_fixed_hyperparams_to_engine_config(engine, fixed_hyperparams)
    readout_args = _frozen_readout_args(engine)
    seeds = [int(value) for value in filter_seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("frozen scoring requires unique filter seeds")
    jobs = min(int(n_jobs), len(seeds))
    if jobs < 1:
        raise ValueError("frozen scoring n_jobs must be positive")

    def run_seed(filter_seed: int) -> np.ndarray:
        result = pf_runner(
            engine_config=engine,
            subject_id=int(subject_id),
            stimulus=physical,
            choices=observed,
            feedback=observed_feedback,
            particle_count=int(particle_count),
            filter_seed=int(filter_seed),
            resample_threshold_fraction=float(resample_threshold_fraction),
            processed_data_dir=processed_data_dir,
            dataset_paths=dataset_paths,
            **readout_args,
        )
        probabilities = np.asarray(result.marginal_probabilities, dtype=float)
        if probabilities.shape != (observed.size, 2):
            raise ValueError("frozen scoring probabilities must have shape (T, 2)")
        return probabilities

    if jobs == 1:
        probability_runs = [run_seed(filter_seed) for filter_seed in seeds]
    else:
        warmup_dykstra_numba()
        probability_runs = list(
            Parallel(n_jobs=jobs, backend="loky", verbose=10)(
                delayed(run_seed)(filter_seed) for filter_seed in seeds
            )
        )
    stack = np.stack(probability_runs, axis=0)
    total_nll = mean_probability_nll(stack, observed, score_mask)
    return {
        "total_nll": total_nll,
        "mean_trial_nll": total_nll / float(score_context["score_trial_count"]),
        "score_context": score_context,
        "particle_count": int(particle_count),
        "filter_seed_count": int(len(seeds)),
        "filter_seeds": seeds,
        "parallel_n_jobs": int(jobs),
        "probability_aggregation": "mean_probability_then_nll",
        "mean_probability": np.mean(stack, axis=0),
    }


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    proportion = float(successes) / float(total)
    denominator = 1.0 + z * z / float(total)
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / float(total)
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return float(center - half_width), float(center + half_width)


def summarize_module_recovery(
    scores: pd.DataFrame,
    *,
    near_best_delta_nll: float = 2.0,
    gates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize held-out total-NLL architecture recovery."""

    required = {
        "dataset_id", "true_cell", "candidate_cell", "total_nll",
    }
    if not required.issubset(scores.columns):
        raise ValueError("module recovery scores are missing required columns")
    cells = ("P", "PM", "PH", "PMH")
    dataset_rows: list[dict[str, Any]] = []
    for dataset_id, frame in scores.groupby("dataset_id", sort=True):
        if set(frame["candidate_cell"].astype(str)) != set(cells) or len(frame) != 4:
            raise ValueError(f"module dataset {dataset_id} requires four candidate cells")
        if frame["true_cell"].nunique() != 1:
            raise ValueError(f"module dataset {dataset_id} has inconsistent truth")
        ranked = frame.assign(
            _cell_order=frame["candidate_cell"].map(
                {cell: index for index, cell in enumerate(cells)}
            )
        ).sort_values(["total_nll", "_cell_order"])
        if not np.all(np.isfinite(ranked["total_nll"].to_numpy(dtype=float))):
            raise ValueError(f"module dataset {dataset_id} has non-finite NLL")
        winner = ranked.iloc[0]
        true_cell = str(frame["true_cell"].iloc[0])
        true_score = frame.loc[
            frame["candidate_cell"].astype(str).eq(true_cell), "total_nll"
        ]
        if len(true_score) != 1:
            raise ValueError(f"module dataset {dataset_id} lacks one true-cell score")
        best_nll = float(winner["total_nll"])
        true_nll = float(true_score.iloc[0])
        row = {
            "dataset_id": str(dataset_id),
            "true_cell": true_cell,
            "predicted_cell": str(winner["candidate_cell"]),
            "best_total_nll": best_nll,
            "true_total_nll": true_nll,
            "true_delta_nll": true_nll - best_nll,
            "exact_recovery": str(winner["candidate_cell"]) == true_cell,
            "true_within_near_best": (
                true_nll <= best_nll + float(near_best_delta_nll)
            ),
        }
        if "generated_accuracy" in frame:
            row["generated_accuracy"] = float(frame["generated_accuracy"].iloc[0])
        dataset_rows.append(row)
    dataset_frame = pd.DataFrame(dataset_rows)
    confusion_rows = []
    for true_cell in cells:
        for predicted_cell in cells:
            confusion_rows.append(
                {
                    "true_cell": true_cell,
                    "predicted_cell": predicted_cell,
                    "count": int(
                        np.sum(
                            dataset_frame["true_cell"].eq(true_cell)
                            & dataset_frame["predicted_cell"].eq(predicted_cell)
                        )
                    ),
                }
            )
    cell_rows = []
    for cell in cells:
        selected = dataset_frame.loc[dataset_frame["true_cell"].eq(cell)]
        successes = int(selected["exact_recovery"].sum())
        low, high = _wilson_interval(successes, len(selected))
        cell_rows.append(
            {
                "true_cell": cell,
                "dataset_n": int(len(selected)),
                "exact_recovery_count": successes,
                "exact_recovery": float(selected["exact_recovery"].mean()),
                "wilson_low": low,
                "wilson_high": high,
                "near_best_coverage": float(
                    selected["true_within_near_best"].mean()
                ),
            }
        )
    total = len(dataset_frame)
    wrong_absorption = {
        cell: float(
            np.mean(
                dataset_frame["predicted_cell"].eq(cell)
                & ~dataset_frame["true_cell"].eq(cell)
            )
        )
        for cell in cells
    }
    gate_config = {
        "overall_exact_recovery_min": 0.70,
        "per_cell_exact_recovery_min": 0.50,
        "true_cell_near_best_coverage_min": 0.85,
        "maximum_single_wrong_cell_absorption": 0.30,
        **dict(gates or {}),
    }
    overall_exact = float(dataset_frame["exact_recovery"].mean())
    near_best_coverage = float(dataset_frame["true_within_near_best"].mean())
    passes = bool(
        total > 0
        and overall_exact >= float(gate_config["overall_exact_recovery_min"])
        and min(row["exact_recovery"] for row in cell_rows)
        >= float(gate_config["per_cell_exact_recovery_min"])
        and near_best_coverage
        >= float(gate_config["true_cell_near_best_coverage_min"])
        and max(wrong_absorption.values())
        <= float(gate_config["maximum_single_wrong_cell_absorption"])
    )
    return {
        "dataset_n": int(total),
        "near_best_delta_nll": float(near_best_delta_nll),
        "overall_exact_recovery": overall_exact,
        "true_cell_near_best_coverage": near_best_coverage,
        "wrong_cell_absorption": wrong_absorption,
        "passes_pre_registered_gates": passes,
        "gates": gate_config,
        "confusion_rows": confusion_rows,
        "cell_rows": cell_rows,
        "dataset_rows": dataset_frame.to_dict(orient="records"),
    }


def _parameter_support_values(
    parameter_space: Mapping[str, Any],
    parameter: str,
) -> list[float]:
    return [float(value) for value in _declared_support(parameter_space, parameter)]


def _workspace_support_values(
    parameter_space: Mapping[str, Any],
    parameter: str,
) -> list[int]:
    workspace = parameter_space["subject_parameters"]["workspace_execution"]
    values = {
        int(candidate[parameter])
        for candidate in workspace["candidates"]
    }
    if not values:
        raise ValueError(f"workspace parameter {parameter} has empty support")
    return sorted(values)


def _safe_spearman(truth: np.ndarray, estimate: np.ndarray) -> float:
    if truth.size < 2 or np.allclose(truth, truth[0]) or np.allclose(
        estimate, estimate[0]
    ):
        return 1.0 if np.allclose(truth, estimate) else 0.0
    value = float(spearmanr(truth, estimate).statistic)
    return value if np.isfinite(value) else 0.0


def _balanced_accuracy_binary(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth_positive = truth > 0.0
    estimate_positive = estimate > 0.0
    if not np.any(truth_positive) or not np.any(~truth_positive):
        return float("nan")
    sensitivity = float(np.mean(estimate_positive[truth_positive]))
    specificity = float(np.mean(~estimate_positive[~truth_positive]))
    return 0.5 * (sensitivity + specificity)


def summarize_parameter_recovery(
    estimates: pd.DataFrame,
    *,
    parameter_space: Mapping[str, Any],
    gates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize discrete, continuous, and zero-boundary recovery."""

    required = {"dataset_id", "true_M", "estimated_M", "true_chi", "estimated_chi"}
    continuous = (
        "gamma", "E_C", "delta_E", "g_0", "c_A", "c_G",
        "beta_0", "eta_plus", "eta_minus",
    )
    for parameter in continuous:
        required.update({f"true_{parameter}", f"estimated_{parameter}"})
    if not required.issubset(estimates.columns):
        missing = sorted(required - set(estimates.columns))
        raise ValueError(f"parameter recovery estimates are missing: {missing}")
    frame = estimates.copy()
    if "true_within_near_best" not in frame:
        frame["true_within_near_best"] = False
    if frame["dataset_id"].duplicated().any():
        raise ValueError("parameter recovery estimates require one row per dataset")
    gate_config = {
        "chi_exact_recovery_min": 0.70,
        "chi_near_best_coverage_min": 0.85,
        "continuous_spearman_min": 0.60,
        "continuous_normalized_mae_max": 0.20,
        "continuous_near_best_coverage_min": 0.80,
        "zero_positive_balanced_accuracy_min": 0.70,
        **dict(gates or {}),
    }
    near_best_coverage = float(frame["true_within_near_best"].mean())
    parameter_rows = []
    error_columns: dict[str, np.ndarray] = {}
    for parameter in continuous:
        truth = frame[f"true_{parameter}"].to_numpy(dtype=float)
        estimate = frame[f"estimated_{parameter}"].to_numpy(dtype=float)
        if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(estimate)):
            raise ValueError(f"parameter {parameter} contains non-finite values")
        error = estimate - truth
        support = _parameter_support_values(parameter_space, parameter)
        support_span = float(max(support) - min(support))
        if support_span <= 0.0:
            raise ValueError(f"parameter {parameter} has zero support span")
        balanced_accuracy = None
        positive_mae = None
        if parameter in {"delta_E", "c_A", "c_G"}:
            balanced_accuracy = _balanced_accuracy_binary(truth, estimate)
            positive = truth > 0.0
            positive_mae = (
                float(np.mean(np.abs(error[positive])))
                if np.any(positive)
                else None
            )
        spearman = _safe_spearman(truth, estimate)
        normalized_mae = float(np.mean(np.abs(error)) / support_span)
        supported = bool(
            spearman >= float(gate_config["continuous_spearman_min"])
            and normalized_mae
            <= float(gate_config["continuous_normalized_mae_max"])
            and near_best_coverage
            >= float(gate_config["continuous_near_best_coverage_min"])
            and (
                balanced_accuracy is None
                or (
                    np.isfinite(balanced_accuracy)
                    and balanced_accuracy
                    >= float(gate_config["zero_positive_balanced_accuracy_min"])
                )
            )
        )
        parameter_rows.append(
            {
                "parameter": parameter,
                "dataset_n": int(len(frame)),
                "bias": float(np.mean(error)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "spearman": spearman,
                "support_span": support_span,
                "normalized_mae": normalized_mae,
                "near_best_coverage": near_best_coverage,
                "zero_positive_balanced_accuracy": balanced_accuracy,
                "positive_truth_mae": positive_mae,
                "supported": supported,
            }
        )
        error_columns[parameter] = error

    chi_truth = frame["true_chi"].to_numpy(dtype=int)
    chi_estimate = frame["estimated_chi"].to_numpy(dtype=int)
    chi_exact = float(np.mean(chi_truth == chi_estimate))
    chi_successes = int(np.sum(chi_truth == chi_estimate))
    chi_low, chi_high = _wilson_interval(chi_successes, len(frame))
    chi_confusion_rows = [
        {
            "true_chi": truth,
            "estimated_chi": estimate,
            "count": int(np.sum((chi_truth == truth) & (chi_estimate == estimate))),
        }
        for truth in (0, 1)
        for estimate in (0, 1)
    ]
    m_truth = frame["true_M"].to_numpy(dtype=int)
    m_estimate = frame["estimated_M"].to_numpy(dtype=int)
    m_exact = float(np.mean(m_truth == m_estimate))
    m_successes = int(np.sum(m_truth == m_estimate))
    m_low, m_high = _wilson_interval(m_successes, len(frame))
    m_support = _workspace_support_values(parameter_space, "M")
    if not set(m_truth).issubset(m_support) or not set(m_estimate).issubset(
        m_support
    ):
        raise ValueError("M truth or estimate falls outside declared support")
    m_confusion_rows = [
        {
            "true_M": truth,
            "estimated_M": estimate,
            "count": int(np.sum((m_truth == truth) & (m_estimate == estimate))),
        }
        for truth in m_support
        for estimate in m_support
    ]

    correlation_rows: list[dict[str, Any]] = []
    names = list(continuous)
    for left in names:
        for right in names:
            left_error = error_columns[left]
            right_error = error_columns[right]
            if np.std(left_error) == 0.0 or np.std(right_error) == 0.0:
                correlation = 1.0 if left == right else 0.0
            else:
                correlation = float(np.corrcoef(left_error, right_error)[0, 1])
            correlation_rows.append(
                {
                    "left_parameter": left,
                    "right_parameter": right,
                    "error_correlation": correlation,
                }
            )
    chi_supported = bool(
        chi_exact >= float(gate_config["chi_exact_recovery_min"])
        and near_best_coverage
        >= float(gate_config["chi_near_best_coverage_min"])
    )
    return {
        "dataset_n": int(len(frame)),
        "M_exact_recovery": m_exact,
        "M_wilson_low": m_low,
        "M_wilson_high": m_high,
        "chi_exact_recovery": chi_exact,
        "chi_wilson_low": chi_low,
        "chi_wilson_high": chi_high,
        "chi_near_best_coverage": near_best_coverage,
        "chi_supported": chi_supported,
        "gates": gate_config,
        "M_confusion_rows": m_confusion_rows,
        "chi_confusion_rows": chi_confusion_rows,
        "parameter_rows": parameter_rows,
        "error_correlation_rows": correlation_rows,
        "dataset_rows": frame.to_dict(orient="records"),
    }


def _save_png_atomic(figure: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.png"
    )
    try:
        figure.savefig(
            temporary,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _configure_recovery_figure_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 600,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def plot_module_recovery(
    summary: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, str]:
    """Plot the architecture confusion hero panel and supporting diagnostics."""

    import matplotlib.pyplot as plt

    _configure_recovery_figure_style()
    output = Path(output_path)
    confusion = pd.DataFrame(summary["confusion_rows"])
    cells = ["P", "PM", "PH", "PMH"]
    matrix = (
        confusion.pivot(index="true_cell", columns="predicted_cell", values="count")
        .reindex(index=cells, columns=cells)
        .fillna(0)
        .to_numpy(dtype=float)
    )
    cell_frame = pd.DataFrame(summary["cell_rows"])
    dataset_frame = pd.DataFrame(summary["dataset_rows"])
    source_paths = {
        "confusion": output.with_name("module_recovery_confusion_source.csv"),
        "cells": output.with_name("module_recovery_cell_source.csv"),
        "datasets": output.with_name("module_recovery_dataset_source.csv"),
    }
    _atomic_csv(source_paths["confusion"], confusion)
    _atomic_csv(source_paths["cells"], cell_frame)
    _atomic_csv(source_paths["datasets"], dataset_frame)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    ax = axes[0, 0]
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0)
    for row in range(4):
        for column in range(4):
            ax.text(column, row, f"{int(matrix[row, column])}", ha="center", va="center")
    ax.set_xticks(range(4), cells)
    ax.set_yticks(range(4), cells)
    ax.set_xlabel("Recovered architecture")
    ax.set_ylabel("Generating architecture")
    ax.set_title("a  Held-out architecture recovery", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Datasets", fraction=0.046)

    ax = axes[0, 1]
    ordered = cell_frame.set_index("true_cell").reindex(cells)
    values = ordered["exact_recovery"].to_numpy(dtype=float)
    lower = values - ordered["wilson_low"].to_numpy(dtype=float)
    upper = ordered["wilson_high"].to_numpy(dtype=float) - values
    ax.bar(cells, values, color="#5B8DB8", width=0.68)
    ax.errorbar(cells, values, yerr=np.vstack([lower, upper]), fmt="none", color="#263746", capsize=2)
    ax.axhline(0.5, color="#A65E4E", linestyle="--", linewidth=1)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Exact recovery rate")
    ax.set_title("b  Recovery by true cell", loc="left", fontweight="bold")

    ax = axes[1, 0]
    for index, cell in enumerate(cells):
        values = dataset_frame.loc[
            dataset_frame["true_cell"].eq(cell), "true_delta_nll"
        ].to_numpy(dtype=float)
        ax.scatter(
            np.full(values.size, index),
            values,
            color="#5B8DB8",
            edgecolor="white",
            linewidth=0.4,
            s=22,
            zorder=3,
        )
    ax.axhline(2.0, color="#A65E4E", linestyle="--", linewidth=1)
    ax.set_xticks(range(4), cells)
    ax.set_ylabel(r"True-cell $\Delta$ total NLL")
    ax.set_title("c  Near-best coverage", loc="left", fontweight="bold")

    ax = axes[1, 1]
    if "generated_accuracy" in dataset_frame:
        for index, cell in enumerate(cells):
            values = dataset_frame.loc[
                dataset_frame["true_cell"].eq(cell), "generated_accuracy"
            ].to_numpy(dtype=float)
            ax.scatter(
                np.full(values.size, index),
                values,
                color="#8AAE92",
                edgecolor="white",
                linewidth=0.4,
                s=22,
            )
        ax.set_xticks(range(4), cells)
        ax.set_ylabel("Generated choice accuracy")
        ax.set_ylim(0.0, 1.0)
    else:
        ax.text(0.5, 0.5, "Accuracy unavailable", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("d  Synthetic behavior", loc="left", fontweight="bold")
    _save_png_atomic(fig, output)
    plt.close(fig)
    return {name: str(path) for name, path in source_paths.items()}


def plot_parameter_recovery(
    summary: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, str]:
    """Plot readout confusion and parameter-level identifiability diagnostics."""

    import matplotlib.pyplot as plt

    _configure_recovery_figure_style()
    output = Path(output_path)
    m_confusion = pd.DataFrame(summary["M_confusion_rows"])
    confusion = pd.DataFrame(summary["chi_confusion_rows"])
    parameter_frame = pd.DataFrame(summary["parameter_rows"])
    dataset_frame = pd.DataFrame(summary["dataset_rows"])
    source_paths = {
        "M": output.with_name("parameter_recovery_M_source.csv"),
        "chi": output.with_name("parameter_recovery_chi_source.csv"),
        "parameters": output.with_name("parameter_recovery_metric_source.csv"),
        "datasets": output.with_name("parameter_recovery_dataset_source.csv"),
    }
    _atomic_csv(source_paths["M"], m_confusion)
    _atomic_csv(source_paths["chi"], confusion)
    _atomic_csv(source_paths["parameters"], parameter_frame)
    _atomic_csv(source_paths["datasets"], dataset_frame)
    m_levels = sorted(
        set(m_confusion["true_M"].astype(int))
        | set(m_confusion["estimated_M"].astype(int))
    )
    m_matrix = (
        m_confusion.pivot(index="true_M", columns="estimated_M", values="count")
        .reindex(index=m_levels, columns=m_levels)
        .fillna(0)
        .to_numpy(dtype=float)
    )
    chi_matrix = (
        confusion.pivot(index="true_chi", columns="estimated_chi", values="count")
        .reindex(index=[0, 1], columns=[0, 1])
        .fillna(0)
        .to_numpy(dtype=float)
    )
    parameters = parameter_frame["parameter"].astype(str).tolist()
    colors = [
        "#5B8DB8" if bool(value) else "#B7BEC5"
        for value in parameter_frame["supported"]
    ]
    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    axes = fig.subplot_mosaic(
        [["M", "chi", "zero"], ["mae", "mae", "rank"]],
        width_ratios=[1.0, 1.0, 1.0],
    )
    ax = axes["M"]
    image = ax.imshow(m_matrix, cmap="Blues", vmin=0.0)
    for row in range(len(m_levels)):
        for column in range(len(m_levels)):
            ax.text(
                column,
                row,
                f"{int(m_matrix[row, column])}",
                ha="center",
                va="center",
            )
    ax.set_xticks(range(len(m_levels)), m_levels)
    ax.set_yticks(range(len(m_levels)), m_levels)
    ax.set_xlabel("Recovered M")
    ax.set_ylabel("Generating M")
    ax.set_title("a  Capacity recovery", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Datasets", fraction=0.046)

    ax = axes["chi"]
    image = ax.imshow(chi_matrix, cmap="Blues", vmin=0.0)
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{int(chi_matrix[row, column])}", ha="center", va="center")
    ax.set_xticks([0, 1], ["Mixture", "Single rule"])
    ax.set_yticks([0, 1], ["Mixture", "Single rule"])
    ax.set_xlabel("Recovered readout")
    ax.set_ylabel("Generating readout")
    ax.set_title("b  Readout recovery", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Datasets", fraction=0.046)

    ax = axes["mae"]
    ax.barh(parameters, parameter_frame["normalized_mae"], color=colors)
    ax.axvline(0.20, color="#A65E4E", linestyle="--", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized MAE")
    ax.set_title("d  Parameter error", loc="left", fontweight="bold")

    ax = axes["rank"]
    ax.barh(parameters, parameter_frame["spearman"], color=colors)
    ax.axvline(0.60, color="#A65E4E", linestyle="--", linewidth=1)
    ax.set_xlim(-1.0, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("Truth–estimate Spearman")
    ax.set_title("e  Rank recovery", loc="left", fontweight="bold")

    ax = axes["zero"]
    zero_frame = parameter_frame.loc[
        parameter_frame["zero_positive_balanced_accuracy"].notna()
    ]
    ax.bar(
        zero_frame["parameter"],
        zero_frame["zero_positive_balanced_accuracy"],
        color="#8AAE92",
        width=0.68,
    )
    ax.axhline(0.70, color="#A65E4E", linestyle="--", linewidth=1)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("c  Exact-zero detection", loc="left", fontweight="bold")
    _save_png_atomic(fig, output)
    plt.close(fig)
    return {name: str(path) for name, path in source_paths.items()}


__all__ = [
    "CELL_FREE_PARAMETERS",
    "FEATURE_COLUMNS",
    "MODEL_PARAMETER_NAMES",
    "ORDER_COLUMNS",
    "RecoveryDatasetSpec",
    "RecoveryDesign",
    "build_calibration_bank",
    "freeze_smallest_passing_budget",
    "fit_recovery_dataset",
    "generate_synthetic_dataset",
    "load_recovery_design",
    "mean_probability_nll",
    "model_0826_truth_hyperparams",
    "plot_module_recovery",
    "plot_parameter_recovery",
    "resolve_calibration_filter_seeds",
    "resolve_recovery_stage_budgets",
    "schedule_fingerprint",
    "score_frozen_candidate",
    "score_pf_bank",
    "score_pf_bank_parallel",
    "summarize_module_recovery",
    "summarize_parameter_recovery",
    "summarize_pf_calibration",
    "summarize_search_budget_retention",
    "synthetic_dataset_frame",
]
