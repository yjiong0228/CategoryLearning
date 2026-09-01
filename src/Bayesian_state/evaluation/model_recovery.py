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
from scipy.stats import spearmanr

from src.Bayesian_state.inference.backends.particle_filter import (
    run_state_model_particle_filter,
)
from src.Bayesian_state.model.readout import (
    resolve_choice_readout_config,
    resolve_output_noise_config,
)
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


def _named_truth_hyperparams(truth: Mapping[str, Any]) -> dict[str, Any]:
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
    return _named_truth_hyperparams(specification.truth)


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
            _named_truth_hyperparams(truth),
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
                    == candidate_ids[int(np.argmin(right_nll))]
                ),
                "probability_rmse": float(
                    np.sqrt(np.mean(np.concatenate(probability_differences)))
                ),
            }
        )
    return comparisons


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
            and metrics["minimum_adjacent_winner_agreement_count"]
            >= int(gates["adjacent_winner_agreement_min_count"])
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


__all__ = [
    "CELL_FREE_PARAMETERS",
    "FEATURE_COLUMNS",
    "MODEL_PARAMETER_NAMES",
    "ORDER_COLUMNS",
    "RecoveryDatasetSpec",
    "RecoveryDesign",
    "build_calibration_bank",
    "freeze_smallest_passing_budget",
    "generate_synthetic_dataset",
    "load_recovery_design",
    "resolve_calibration_filter_seeds",
    "schedule_fingerprint",
    "score_pf_bank",
    "summarize_pf_calibration",
    "synthetic_dataset_frame",
]
