#!/usr/bin/env python3
"""Run the pre-registered full-trial Model0826 recovery workflow."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Bayesian_state.simulation.recovery import (
    FEATURE_COLUMNS,
    MODEL_PARAMETER_NAMES,
    ORDER_COLUMNS,
    RecoveryDatasetSpec,
    RecoveryDesign,
)
from src.Bayesian_state.workflows.recovery.generation import generate_synthetic_dataset
from src.Bayesian_state.evaluation.recovery import (
    build_calibration_bank,
    freeze_smallest_passing_budget,
    mean_probability_nll,
    plot_module_recovery,
    plot_parameter_recovery,
    resolve_calibration_filter_seeds,
    score_frozen_candidate,
    score_pf_bank,
    score_pf_bank_parallel,
    summarize_module_recovery,
    summarize_parameter_recovery,
    summarize_pf_calibration,
    summarize_search_budget_retention,
)
from src.Bayesian_state.optimization.recovery import fit_recovery_dataset
from src.Bayesian_state.optimization.recovery_parameters import model_0826_truth_hyperparams
from src.Bayesian_state.optimization.artifacts import (  # noqa: E402
    subject_best_hyperparams,
    to_builtin,
)
from src.Bayesian_state.optimization.model_0826 import (  # noqa: E402
    extract_model_0826_parameters,
)
from src.Bayesian_state.optimization.parameter_space import (  # noqa: E402
    load_model_parameter_space,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.utils.seeding import stable_seed  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs/exp123/specific_models/model_0826_recovery_v1.yaml"
)
PHASES = (
    "smoke",
    "generate",
    "calibrate",
    "module-fit",
    "parameter-fit",
    "summarize",
    "priority-all",
    "all",
)


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
                sort_keys=True,
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


def _canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        to_builtin(dict(payload)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser(*, default_config: Path = DEFAULT_CONFIG) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--priority-subject",
        type=int,
        help=(
            "With --phase priority-all, finish this subject's module and "
            "parameter recovery before starting the remaining subjects."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only artifacts whose registered fingerprints match.",
    )
    return parser


def resolve_subject_order(
    available_subjects: Sequence[int],
    priority_subject: int,
) -> tuple[int, ...]:
    """Place one registered subject first without dropping the others."""

    subjects = tuple(dict.fromkeys(int(value) for value in available_subjects))
    priority = int(priority_subject)
    if priority not in subjects:
        raise ValueError(
            f"priority subject {priority} is not registered: {subjects}"
        )
    return (priority, *(value for value in subjects if value != priority))


def prepare_output(
    output_root: str | Path,
    *,
    resume: bool,
    analysis_id: str,
    config_fingerprint: str,
) -> dict[str, Any]:
    """Create a new run root or validate an explicitly resumed one."""

    output = Path(output_root)
    manifest_path = output / "manifest.json"
    if output.exists():
        if not resume:
            raise FileExistsError(
                f"recovery output already exists; use --resume: {output}"
            )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"recovery resume requires manifest.json: {output}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("analysis_id") != str(analysis_id):
            raise ValueError("recovery manifest analysis_id does not match")
        if manifest.get("config_fingerprint") != str(config_fingerprint):
            raise ValueError("recovery manifest config fingerprint does not match")
        return dict(manifest)

    if resume:
        raise FileNotFoundError(f"recovery output does not exist: {output}")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "analysis_id": str(analysis_id),
        "config_fingerprint": str(config_fingerprint),
        "status": "initialized",
        "phases": {},
    }
    _atomic_json(manifest_path, manifest)
    return manifest


def _design_fingerprint(design: RecoveryDesign) -> str:
    paths = {
        "recovery_config": design.source_path,
        "model_engine_config": design.model_engine_config,
        "parameter_space": design.parameter_space_path,
        "base_simulation_config": design.base_simulation_config,
        "recovery_runner": Path(__file__).resolve(),
        "recovery_generation": ROOT / "src/Bayesian_state/workflows/recovery/generation.py",
        "recovery_evaluation_recovery": ROOT / "src/Bayesian_state/evaluation/recovery.py",
        "recovery_optimization_recovery": ROOT / "src/Bayesian_state/optimization/recovery.py",
        "recovery_optimization_recovery_parameters": ROOT / "src/Bayesian_state/optimization/recovery_parameters.py",
        "recovery_simulation_recovery": ROOT / "src/Bayesian_state/simulation/recovery.py",
        "recovery_utils_recovery_artifacts": ROOT / "src/Bayesian_state/utils/recovery_artifacts.py",
        "recovery_workflows_recovery_design": ROOT / "src/Bayesian_state/workflows/recovery/design.py",
        "recovery_library": (
            ROOT / "src/Bayesian_state/evaluation/model_recovery.py"
        ),
        "model_0826_optimization": (
            ROOT / "src/Bayesian_state/optimization/model_0826.py"
        ),
        "hyper_cd": (
            ROOT
            / "src/Bayesian_state/optimization/search/coordinate_descent.py"
        ),
        "particle_filter": (
            ROOT
            / "src/Bayesian_state/inference/backends/particle_filter.py"
        ),
        "choice_readout": ROOT / "src/Bayesian_state/model/readout.py",
    }
    return _canonical_fingerprint(
        {
            "analysis_id": design.analysis_id,
            "files": {
                name: {"path": str(path), "sha256": _file_sha256(path)}
                for name, path in paths.items()
            },
            "subject_trial_counts": design.subject_trial_counts,
        }
    )


def _record_run_provenance(
    design: RecoveryDesign,
    output_root: Path,
) -> None:
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_paths = {
        "recovery_config": design.source_path,
        "model_engine_config": design.model_engine_config,
        "parameter_space": design.parameter_space_path,
        "base_simulation_config": design.base_simulation_config,
        "recovery_runner": Path(__file__).resolve(),
        "recovery_generation": ROOT / "src/Bayesian_state/workflows/recovery/generation.py",
        "recovery_evaluation_recovery": ROOT / "src/Bayesian_state/evaluation/recovery.py",
        "recovery_optimization_recovery": ROOT / "src/Bayesian_state/optimization/recovery.py",
        "recovery_optimization_recovery_parameters": ROOT / "src/Bayesian_state/optimization/recovery_parameters.py",
        "recovery_simulation_recovery": ROOT / "src/Bayesian_state/simulation/recovery.py",
        "recovery_utils_recovery_artifacts": ROOT / "src/Bayesian_state/utils/recovery_artifacts.py",
        "recovery_workflows_recovery_design": ROOT / "src/Bayesian_state/workflows/recovery/design.py",
        "recovery_library": (
            ROOT / "src/Bayesian_state/evaluation/model_recovery.py"
        ),
        "model_0826_optimization": (
            ROOT / "src/Bayesian_state/optimization/model_0826.py"
        ),
        "hyper_cd": (
            ROOT
            / "src/Bayesian_state/optimization/search/coordinate_descent.py"
        ),
        "particle_filter": (
            ROOT
            / "src/Bayesian_state/inference/backends/particle_filter.py"
        ),
        "choice_readout": ROOT / "src/Bayesian_state/model/readout.py",
    }
    provenance = {
        "source_files": {
            name: {"path": str(path), "sha256": _file_sha256(path)}
            for name, path in source_paths.items()
        },
        "subject_trial_counts": {
            str(subject_id): int(count)
            for subject_id, count in design.subject_trial_counts.items()
        },
        "trial_scope": "all_valid_trials_per_subject",
        "seed_registry": {
            "generation": "module_recovery/synthetic and parameter_recovery/synthetic manifests",
            "calibration": "numerical_calibration/score_rows.csv",
            "module_final_score": "module_recovery/fit_scores.csv",
            "parameter_final_score": "parameter_recovery/fit_scores.csv",
            "hyper_cd": "*/search/*/search/subject_*/search_checkpoint.json",
        },
    }
    if manifest.get("provenance") not in (None, provenance):
        raise ValueError("recovery manifest provenance does not match current code")
    manifest["provenance"] = provenance
    _atomic_json(manifest_path, manifest)


def _update_phase_manifest(
    output_root: Path,
    phase: str,
    *,
    status: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phases = dict(manifest.get("phases") or {})
    phases[str(phase)] = {
        "status": str(status),
        **to_builtin(dict(details or {})),
    }
    manifest["phases"] = phases
    if status == "failed":
        manifest["status"] = "failed"
    elif str(phase) == "summarize" and status == "complete":
        manifest["status"] = "complete"
    else:
        manifest["status"] = "running"
    _atomic_json(manifest_path, manifest)


def _phase_is_complete(output_root: Path, phase: str) -> bool:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return (
        ((manifest.get("phases") or {}).get(str(phase)) or {}).get("status")
        == "complete"
    )


def require_frozen_budget(path: str | Path) -> dict[str, int]:
    """Load the numerical PF budget that passed every calibration gate."""

    budget_path = Path(path)
    if not budget_path.is_file():
        raise FileNotFoundError(
            f"PF calibration has not produced a frozen budget: {budget_path}"
        )
    payload = json.loads(budget_path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise ValueError("PF calibration budget is not frozen")
    particle_count = int(payload.get("particle_count", 0))
    filter_seed_count = int(payload.get("filter_seed_count", 0))
    if particle_count < 2 or filter_seed_count < 1:
        raise ValueError("PF calibration frozen an invalid numerical budget")
    return {
        "particle_count": particle_count,
        "filter_seed_count": filter_seed_count,
    }


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return deepcopy(dict(payload))


def load_subject_schedules(
    design: RecoveryDesign,
) -> tuple[dict[int, pd.DataFrame], dict[str, Path]]:
    """Load the three registered condition-1 schedules without truncation."""

    base_simulation = _load_yaml(design.base_simulation_config)
    dataset_paths = resolve_dataset_paths(
        base_simulation,
        design.base_simulation_config.parent,
    )
    learning = pd.read_csv(dataset_paths["learning_data"])
    condition_one = learning.loc[learning["condition"].eq(1)].copy()
    schedules: dict[int, pd.DataFrame] = {}
    for subject_id, expected_trials in design.subject_trial_counts.items():
        frame = (
            condition_one.loc[condition_one["iSub"].eq(int(subject_id))]
            .sort_values(list(ORDER_COLUMNS))
            .reset_index(drop=True)
        )
        if len(frame) != int(expected_trials):
            raise ValueError(
                f"subject {subject_id} has {len(frame)} trials; "
                f"the registered design requires {expected_trials}"
            )
        schedules[int(subject_id)] = frame
    return schedules, dataset_paths


def build_calibration_specs(
    design: RecoveryDesign,
) -> tuple[RecoveryDatasetSpec, ...]:
    """Materialize the six independently generated PF-calibration datasets."""

    calibration = dict(design.config["pf_calibration"])
    anchor = dict(calibration["anchor_truth"])
    templates = [int(value) for value in calibration["templates"]]
    chi_values = [int(value) for value in calibration["chi_values"]]
    if templates != [101, 111, 118] or chi_values != [0, 1]:
        raise ValueError("PF calibration requires templates 101/111/118 and chi 0/1")
    base_seed = int(design.config["generation"]["base_seed"])
    specifications: list[RecoveryDatasetSpec] = []
    for subject_id in templates:
        for chi in chi_values:
            truth = {**deepcopy(anchor), "chi": int(chi)}
            dataset_id = f"calibration_subject_{subject_id}_chi_{chi}"
            generation_seed = stable_seed(
                {
                    "seed_role": "model0826_recovery_calibration_generation",
                    "analysis_id": design.analysis_id,
                    "base_seed": base_seed,
                    "dataset_id": dataset_id,
                    "truth": truth,
                }
            )
            trial_count = int(design.subject_trial_counts[subject_id])
            specifications.append(
                RecoveryDatasetSpec(
                    dataset_id=dataset_id,
                    family="parameter",
                    subject_id=subject_id,
                    trial_count=trial_count,
                    replicate=1,
                    truth_cell="PMH",
                    truth_profile=f"calibration_anchor_chi_{chi}",
                    truth=truth,
                    generation_seed=int(generation_seed),
                    train_trial_count=trial_count,
                    evaluation_trial_count=0,
                )
            )
    return tuple(specifications)


def resolve_final_score_seeds(
    *,
    analysis_id: str,
    dataset_id: str,
    role: str,
    count: int,
) -> list[int]:
    """Return candidate-paired seeds that are disjoint across score roles."""

    seed_count = int(count)
    if seed_count < 1:
        raise ValueError("final score seed count must be positive")
    role_name = str(role).strip()
    if not role_name:
        raise ValueError("final score role cannot be empty")
    return [
        stable_seed(
            {
                "seed_role": "model0826_recovery_final_score",
                "analysis_id": str(analysis_id),
                "dataset_id": str(dataset_id),
                "score_role": role_name,
                "logical_seed_index": int(index),
            }
        )
        for index in range(seed_count)
    ]


def _base_engine(design: RecoveryDesign) -> dict[str, Any]:
    return _load_yaml(design.model_engine_config)


def _synthetic_dir(output_root: Path, family: str) -> Path:
    if family == "module":
        return output_root / "module_recovery" / "synthetic"
    if family == "parameter":
        return output_root / "parameter_recovery" / "synthetic"
    if family == "calibration":
        return output_root / "numerical_calibration" / "synthetic"
    raise ValueError(f"unknown recovery family: {family}")


def _synthetic_paths(
    output_root: Path,
    specification: RecoveryDatasetSpec,
    *,
    family: str | None = None,
) -> tuple[Path, Path, Path]:
    directory = _synthetic_dir(output_root, family or specification.family)
    stem = specification.dataset_id
    return (
        directory / f"{stem}.csv",
        directory / f"{stem}.npz",
        directory / f"{stem}.manifest.json",
    )


def _load_synthetic_arrays(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"synthetic recovery arrays are missing: {path}")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {
            "stimulus": payload["stimulus"].astype(float),
            "categories": payload["categories"].astype(int),
            "choices": payload["choices"].astype(int),
            "feedback": payload["feedback"].astype(float),
        }
    trial_count = int(arrays["choices"].size)
    if arrays["stimulus"].shape != (trial_count, len(FEATURE_COLUMNS)):
        raise ValueError("synthetic recovery stimulus has the wrong shape")
    if arrays["feedback"].size != trial_count:
        raise ValueError("synthetic recovery arrays are misaligned")
    return arrays


def _generate_one(
    design: RecoveryDesign,
    output_root: Path,
    specification: RecoveryDatasetSpec,
    schedules: Mapping[int, pd.DataFrame],
    dataset_paths: Mapping[str, Path],
    *,
    family: str | None = None,
    resume: bool,
) -> dict[str, Any]:
    target_family = family or specification.family
    directory = _synthetic_dir(output_root, target_family)
    return generate_synthetic_dataset(
        specification,
        schedule_frame=schedules[int(specification.subject_id)],
        base_engine_config=_base_engine(design),
        output_dir=directory,
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
        resume=bool(resume),
    )


def run_smoke_phase(
    design: RecoveryDesign,
    output_root: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Exercise every full subject sequence with a tiny numerical budget."""

    if resume and _phase_is_complete(output_root, "smoke"):
        return json.loads(
            (output_root / "smoke" / "smoke_summary.json").read_text(
                encoding="utf-8"
            )
        )
    _update_phase_manifest(output_root, "smoke", status="running")
    schedules, dataset_paths = load_subject_schedules(design)
    specifications = [
        next(
            row
            for row in design.module_datasets
            if row.truth_cell == "PMH"
            and row.subject_id == subject_id
            and row.replicate == 1
        )
        for subject_id in (101, 111, 118)
    ]
    rows: list[dict[str, Any]] = []
    for index, specification in enumerate(specifications, start=1):
        smoke_specification = RecoveryDatasetSpec(
            **{
                **asdict(specification),
                "dataset_id": f"smoke_subject_{specification.subject_id}",
                "generation_seed": stable_seed(
                    {
                        "seed_role": "model0826_recovery_smoke_generation",
                        "analysis_id": design.analysis_id,
                        "subject_id": specification.subject_id,
                    }
                ),
            }
        )
        manifest = generate_synthetic_dataset(
            smoke_specification,
            schedule_frame=schedules[smoke_specification.subject_id],
            base_engine_config=_base_engine(design),
            output_dir=output_root / "smoke" / "synthetic",
            processed_data_dir=dataset_paths["processed_dir"],
            dataset_paths=dataset_paths,
            resume=bool(resume),
        )
        arrays = _load_synthetic_arrays(Path(manifest["npz_path"]))
        seeds = resolve_final_score_seeds(
            analysis_id=design.analysis_id,
            dataset_id=smoke_specification.dataset_id,
            role="smoke",
            count=1,
        )
        score = score_pf_bank(
            dataset_id=smoke_specification.dataset_id,
            subject_id=smoke_specification.subject_id,
            stimulus=arrays["stimulus"],
            choices=arrays["choices"],
            feedback=arrays["feedback"],
            base_engine_config=_base_engine(design),
            candidates=[
                {
                    "candidate_id": "generating_truth",
                    "variant": "generating_truth",
                    "truth": smoke_specification.truth,
                }
            ],
            particle_count=4,
            filter_seeds=seeds,
            ensemble="smoke",
            processed_data_dir=dataset_paths["processed_dir"],
            dataset_paths=dataset_paths,
        )[0]
        rows.append(
            {
                "dataset_id": smoke_specification.dataset_id,
                "subject_id": smoke_specification.subject_id,
                "trial_count": smoke_specification.trial_count,
                "generated_accuracy": manifest["generated_accuracy"],
                "total_nll": score["total_nll"],
                "probability_finite": bool(
                    np.all(np.isfinite(score["mean_probability"]))
                ),
                "probability_normalized": bool(
                    np.allclose(
                        np.asarray(score["mean_probability"]).sum(axis=1),
                        1.0,
                        atol=1e-8,
                    )
                ),
                "particle_count": 4,
                "filter_seeds": seeds,
            }
        )
        print(
            f"[smoke {index}/{len(specifications)}] "
            f"subject {smoke_specification.subject_id}: "
            f"{smoke_specification.trial_count} trials",
            flush=True,
        )
    if not all(
        row["probability_finite"] and row["probability_normalized"]
        for row in rows
    ):
        raise ValueError("Model0826 smoke returned invalid probabilities")
    summary = {
        "status": "complete",
        "full_trial_counts": {
            str(row["subject_id"]): int(row["trial_count"]) for row in rows
        },
        "rows": rows,
    }
    _atomic_csv(output_root / "smoke" / "smoke_summary.csv", pd.DataFrame(rows))
    _atomic_json(output_root / "smoke" / "smoke_summary.json", summary)
    _update_phase_manifest(
        output_root,
        "smoke",
        status="complete",
        details={"subject_trial_counts": summary["full_trial_counts"]},
    )
    return summary


def run_generate_phase(
    design: RecoveryDesign,
    output_root: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Generate all 36 module and 40 parameter-recovery observations."""

    if resume and _phase_is_complete(output_root, "generate"):
        return {"status": "complete", "dataset_count": len(design.all_datasets)}
    _update_phase_manifest(output_root, "generate", status="running")
    schedules, dataset_paths = load_subject_schedules(design)
    manifests: list[dict[str, Any]] = []
    for index, specification in enumerate(design.all_datasets, start=1):
        manifests.append(
            _generate_one(
                design,
                output_root,
                specification,
                schedules,
                dataset_paths,
                resume=resume,
            )
        )
        print(
            f"[generate {index}/{len(design.all_datasets)}] "
            f"{specification.dataset_id}",
            flush=True,
        )
    summary = {
        "status": "complete",
        "dataset_count": len(manifests),
        "module_dataset_count": len(design.module_datasets),
        "parameter_dataset_count": len(design.parameter_datasets),
        "observed_choices_used": False,
    }
    _atomic_json(output_root / "generation_summary.json", summary)
    _update_phase_manifest(output_root, "generate", status="complete", details=summary)
    return summary


def _calibration_bundle_paths(
    output_root: Path,
    dataset_id: str,
    setting_id: str,
) -> tuple[Path, Path]:
    directory = output_root / "numerical_calibration" / "scores" / dataset_id
    return (
        directory / f"{setting_id}.manifest.json",
        directory / f"{setting_id}.npz",
    )


def _save_calibration_bundle(
    manifest_path: Path,
    arrays_path: Path,
    *,
    fingerprint: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    scalar_rows: list[dict[str, Any]] = []
    for row in rows:
        scalar_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {
                    "mean_probability",
                    "probability_runs",
                    "trial_probability_mcse",
                }
            }
        )
    _atomic_npz(
        arrays_path,
        mean_probability=np.stack(
            [np.asarray(row["mean_probability"], dtype=float) for row in rows]
        ),
        probability_runs=np.stack(
            [np.asarray(row["probability_runs"], dtype=float) for row in rows]
        ),
        trial_probability_mcse=np.stack(
            [
                np.asarray(row["trial_probability_mcse"], dtype=float)
                for row in rows
            ]
        ),
    )
    _atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "complete",
            "fingerprint": fingerprint,
            "arrays_path": str(arrays_path),
            "rows": scalar_rows,
        },
    )


def _load_calibration_bundle(
    manifest_path: Path,
    arrays_path: Path,
    *,
    fingerprint: str,
) -> list[dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"calibration cache is incomplete: {manifest_path}")
    if manifest.get("fingerprint") != fingerprint:
        raise ValueError(f"calibration cache fingerprint differs: {manifest_path}")
    if not arrays_path.is_file():
        raise FileNotFoundError(f"calibration arrays are missing: {arrays_path}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        means = arrays["mean_probability"].astype(float)
        runs = arrays["probability_runs"].astype(float)
        mcse = arrays["trial_probability_mcse"].astype(float)
    rows = [dict(row) for row in manifest["rows"]]
    if not (len(rows) == means.shape[0] == runs.shape[0] == mcse.shape[0]):
        raise ValueError("calibration cache metadata and arrays are misaligned")
    for index, row in enumerate(rows):
        row["mean_probability"] = means[index]
        row["probability_runs"] = runs[index]
        row["trial_probability_mcse"] = mcse[index]
    return rows


def _score_calibration_setting(
    design: RecoveryDesign,
    output_root: Path,
    specification: RecoveryDatasetSpec,
    arrays: Mapping[str, np.ndarray],
    candidates: Sequence[Mapping[str, Any]],
    dataset_paths: Mapping[str, Path],
    *,
    particle_count: int,
    filter_seed_count: int,
    ensemble: str,
    resume: bool,
) -> list[dict[str, Any]]:
    seeds = resolve_calibration_filter_seeds(
        dataset_id=specification.dataset_id,
        base_seed=int(design.config["generation"]["base_seed"]),
        ensemble=ensemble,
        count=filter_seed_count,
    )
    setting_id = f"R{particle_count}_B{filter_seed_count}_{ensemble.upper()}"
    manifest_path, arrays_path = _calibration_bundle_paths(
        output_root,
        specification.dataset_id,
        setting_id,
    )
    synthetic_manifest_path = _synthetic_paths(
        output_root,
        specification,
        family="calibration",
    )[2]
    synthetic_manifest = json.loads(
        synthetic_manifest_path.read_text(encoding="utf-8")
    )
    fingerprint = _canonical_fingerprint(
        {
            "analysis_id": design.analysis_id,
            "dataset_id": specification.dataset_id,
            "synthetic_fingerprint": synthetic_manifest["fingerprint"],
            "candidate_bank": list(candidates),
            "particle_count": int(particle_count),
            "filter_seeds": seeds,
            "ensemble": str(ensemble).upper(),
            "model_engine_sha256": _file_sha256(design.model_engine_config),
        }
    )
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"calibration score already exists: {manifest_path}")
        return _load_calibration_bundle(
            manifest_path,
            arrays_path,
            fingerprint=fingerprint,
        )
    rows = score_pf_bank_parallel(
        dataset_id=specification.dataset_id,
        subject_id=specification.subject_id,
        stimulus=arrays["stimulus"],
        choices=arrays["choices"],
        feedback=arrays["feedback"],
        base_engine_config=_base_engine(design),
        candidates=candidates,
        particle_count=particle_count,
        filter_seeds=seeds,
        ensemble=ensemble,
        n_jobs=int(design.config["search"]["cd"]["parallel_budget"]),
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
    )
    _save_calibration_bundle(
        manifest_path,
        arrays_path,
        fingerprint=fingerprint,
        rows=rows,
    )
    return rows


def _prefix_calibration_rows(
    rows: Sequence[Mapping[str, Any]],
    choices: np.ndarray,
    *,
    filter_seed_count: int,
) -> list[dict[str, Any]]:
    prefix_count = int(filter_seed_count)
    derived: list[dict[str, Any]] = []
    for source in rows:
        stack = np.asarray(source["probability_runs"], dtype=float)[:prefix_count]
        if stack.shape[0] != prefix_count:
            raise ValueError("calibration prefix exceeds cached filter seeds")
        if prefix_count > 1:
            mcse = np.std(stack[:, :, 1], axis=0, ddof=1) / np.sqrt(prefix_count)
        else:
            mcse = np.zeros(stack.shape[1], dtype=float)
        row = dict(source)
        row.update(
            {
                "filter_seed_count": prefix_count,
                "filter_seeds": list(source["filter_seeds"][:prefix_count]),
                "total_nll": mean_probability_nll(stack, choices),
                "mean_trial_nll": mean_probability_nll(stack, choices)
                / float(choices.size),
                "mean_probability": np.mean(stack, axis=0),
                "probability_runs": stack,
                "trial_probability_mcse": mcse,
            }
        )
        derived.append(row)
    return derived


def _calibration_scalar_frame(
    rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {
                    "mean_probability",
                    "probability_runs",
                    "trial_probability_mcse",
                }
            }
            for row in rows
        ]
    )


def run_calibrate_phase(
    design: RecoveryDesign,
    output_root: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Freeze the smallest PF particle/repeat budget passing every gate."""

    summary_path = output_root / "numerical_calibration" / "summary.json"
    budget_path = output_root / "numerical_calibration" / "frozen_budget.json"
    if resume and _phase_is_complete(output_root, "calibrate"):
        require_frozen_budget(budget_path)
        return json.loads(summary_path.read_text(encoding="utf-8"))
    _update_phase_manifest(output_root, "calibrate", status="running")
    schedules, dataset_paths = load_subject_schedules(design)
    specifications = build_calibration_specs(design)
    candidates = build_calibration_bank(
        design.config["pf_calibration"]["anchor_truth"]
    )
    for specification in specifications:
        _generate_one(
            design,
            output_root,
            specification,
            schedules,
            dataset_paths,
            family="calibration",
            resume=resume,
        )

    rows: list[dict[str, Any]] = []
    for index, specification in enumerate(specifications, start=1):
        arrays_path = _synthetic_paths(
            output_root,
            specification,
            family="calibration",
        )[1]
        arrays = _load_synthetic_arrays(arrays_path)
        rows.extend(
            _score_calibration_setting(
                design,
                output_root,
                specification,
                arrays,
                candidates,
                dataset_paths,
                particle_count=16,
                filter_seed_count=4,
                ensemble="A",
                resume=resume,
            )
        )
        rows.extend(
            _score_calibration_setting(
                design,
                output_root,
                specification,
                arrays,
                candidates,
                dataset_paths,
                particle_count=32,
                filter_seed_count=4,
                ensemble="A",
                resume=resume,
            )
        )
        r64_a = _score_calibration_setting(
            design,
            output_root,
            specification,
            arrays,
            candidates,
            dataset_paths,
            particle_count=64,
            filter_seed_count=8,
            ensemble="A",
            resume=resume,
        )
        rows.extend(_prefix_calibration_rows(r64_a, arrays["choices"], filter_seed_count=4))
        rows.extend(r64_a)
        rows.extend(
            _score_calibration_setting(
                design,
                output_root,
                specification,
                arrays,
                candidates,
                dataset_paths,
                particle_count=64,
                filter_seed_count=8,
                ensemble="B",
                resume=resume,
            )
        )
        print(
            f"[calibrate primary {index}/{len(specifications)}] "
            f"{specification.dataset_id}",
            flush=True,
        )

    gates = design.config["pf_calibration"]["gates"]
    summary = summarize_pf_calibration(rows, gates)
    budget_validation_policy = design.config["search"].get(
        "budget_validation"
    )
    if summary["status"] != "passed" or budget_validation_policy is not None:
        for index, specification in enumerate(specifications, start=1):
            arrays = _load_synthetic_arrays(
                _synthetic_paths(
                    output_root,
                    specification,
                    family="calibration",
                )[1]
            )
            for ensemble in ("A", "B"):
                rows.extend(
                    _score_calibration_setting(
                        design,
                        output_root,
                        specification,
                        arrays,
                        candidates,
                        dataset_paths,
                        particle_count=128,
                        filter_seed_count=16,
                        ensemble=ensemble,
                        resume=resume,
                    )
                )
            print(
                f"[calibrate escalation {index}/{len(specifications)}] "
                f"{specification.dataset_id}",
                flush=True,
            )
        summary = summarize_pf_calibration(rows, gates)

    calibration_dir = output_root / "numerical_calibration"
    search_budget_validation = None
    if budget_validation_policy is not None:
        search_budget_validation = summarize_search_budget_retention(
            rows,
            budget_validation_policy,
        )
        summary["pf_status"] = summary["status"]
        summary["search_budget_validation"] = search_budget_validation
        if search_budget_validation["status"] != "passed":
            summary["status"] = "failed"
        _atomic_json(
            calibration_dir / "search_budget_validation.json",
            search_budget_validation,
        )
        _atomic_csv(
            calibration_dir / "search_budget_comparisons.csv",
            pd.DataFrame(search_budget_validation["comparisons"]),
        )
    _atomic_csv(calibration_dir / "score_rows.csv", _calibration_scalar_frame(rows))
    _atomic_csv(
        calibration_dir / "comparisons.csv",
        pd.DataFrame(summary["comparisons"]),
    )
    _atomic_csv(
        calibration_dir / "budget_decisions.csv",
        pd.DataFrame(summary["budget_decisions"]),
    )
    _atomic_json(summary_path, summary)
    budget = (
        freeze_smallest_passing_budget(summary, budget_path)
        if summary["status"] == "passed"
        else None
    )
    if budget is None:
        failure_reason = "no PF budget passed every gate"
        if (
            search_budget_validation is not None
            and search_budget_validation["status"] != "passed"
        ):
            failure_reason = (
                "search-stage budgets did not retain the high-budget winner"
            )
        _atomic_json(
            budget_path,
            {"status": "failed", "reason": failure_reason},
        )
        _update_phase_manifest(
            output_root,
            "calibrate",
            status="failed",
            details={"summary_path": str(summary_path)},
        )
        raise RuntimeError(f"Model0826 recovery calibration failed: {failure_reason}")
    details = {"frozen_budget": budget, "dataset_count": len(specifications)}
    _update_phase_manifest(
        output_root,
        "calibrate",
        status="complete",
        details=details,
    )
    return summary


def _require_phase(output_root: Path, phase: str) -> None:
    if not _phase_is_complete(output_root, phase):
        raise RuntimeError(
            f"recovery phase {phase!r} must complete before this phase"
        )


def _fit_root(
    output_root: Path,
    specification: RecoveryDatasetSpec,
    candidate_cell: str,
) -> Path:
    family_dir = (
        "module_recovery"
        if specification.family == "module"
        else "parameter_recovery"
    )
    return (
        output_root
        / family_dir
        / "search"
        / specification.dataset_id
        / str(candidate_cell).upper()
    )


def _selected_hyperparams_path(
    fit_root: Path,
    subject_id: int,
) -> Path:
    return (
        fit_root
        / "search"
        / f"subject_{int(subject_id)}"
        / "best_hyperparams.json"
    )


def _fit_or_load_selected_hyperparams(
    design: RecoveryDesign,
    specification: RecoveryDatasetSpec,
    output_root: Path,
    *,
    candidate_cell: str,
    frozen_budget: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    fit_root = _fit_root(output_root, specification, candidate_cell)
    best_path = _selected_hyperparams_path(
        fit_root,
        specification.subject_id,
    )
    checkpoint_path = best_path.with_name("search_checkpoint.json")
    complete = False
    if best_path.is_file() and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        complete = checkpoint.get("status") == "complete"
    if not complete:
        synthetic_csv = _synthetic_paths(output_root, specification)[0]
        fit_recovery_dataset(
            design,
            specification,
            candidate_cell=candidate_cell,
            synthetic_csv=synthetic_csv,
            frozen_budget=frozen_budget,
            output_dir=fit_root,
            resume=bool(resume and fit_root.exists()),
        )
    if not best_path.is_file():
        raise FileNotFoundError(f"recovery fit lacks selected parameters: {best_path}")
    payload = json.loads(best_path.read_text(encoding="utf-8"))
    selected = subject_best_hyperparams(payload)
    if not isinstance(selected, Mapping):
        raise ValueError(f"recovery fit selected parameters are invalid: {best_path}")
    return deepcopy(dict(selected))


def _load_final_rescore_shortlist(
    fit_root: Path,
    subject_id: int,
) -> list[dict[str, Any]]:
    path = (
        fit_root
        / "search"
        / f"subject_{int(subject_id)}"
        / "final_rescore.jsonl"
    )
    if not path.is_file():
        raise FileNotFoundError(f"parameter fit lacks final shortlist: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records.sort(key=lambda row: int(row.get("shortlist_rank", 10**9)))
    shortlist: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    for record in records:
        hyperparams = record.get("hyperparams")
        if not isinstance(hyperparams, Mapping):
            raise ValueError(f"final shortlist row lacks hyperparams: {path}")
        point = deepcopy(dict(hyperparams))
        fingerprint = _canonical_fingerprint(point)
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            shortlist.append(point)
    if not shortlist:
        raise ValueError(f"parameter fit final shortlist is empty: {path}")
    return shortlist


def _frozen_score_paths(
    output_root: Path,
    specification: RecoveryDatasetSpec,
    score_label: str,
) -> tuple[Path, Path]:
    family_dir = (
        "module_recovery"
        if specification.family == "module"
        else "parameter_recovery"
    )
    directory = (
        output_root
        / family_dir
        / "final_rescore"
        / specification.dataset_id
    )
    return (
        directory / f"{score_label}.manifest.json",
        directory / f"{score_label}.npz",
    )


def _score_or_load_frozen_candidate(
    design: RecoveryDesign,
    output_root: Path,
    specification: RecoveryDatasetSpec,
    arrays: Mapping[str, np.ndarray],
    dataset_paths: Mapping[str, Path],
    *,
    candidate_cell: str,
    fixed_hyperparams: Mapping[str, Any],
    frozen_budget: Mapping[str, Any],
    score_label: str,
    score_role: str,
    evaluation_protocol: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    seeds = resolve_final_score_seeds(
        analysis_id=design.analysis_id,
        dataset_id=specification.dataset_id,
        role=score_role,
        count=int(frozen_budget["filter_seed_count"]),
    )
    manifest_path, arrays_path = _frozen_score_paths(
        output_root,
        specification,
        score_label,
    )
    synthetic_manifest = json.loads(
        _synthetic_paths(output_root, specification)[2].read_text(
            encoding="utf-8"
        )
    )
    fingerprint = _canonical_fingerprint(
        {
            "analysis_id": design.analysis_id,
            "dataset_id": specification.dataset_id,
            "candidate_cell": str(candidate_cell).upper(),
            "fixed_hyperparams": dict(fixed_hyperparams),
            "frozen_budget": dict(frozen_budget),
            "filter_seeds": seeds,
            "evaluation_protocol": dict(evaluation_protocol),
            "synthetic_fingerprint": synthetic_manifest["fingerprint"],
            "model_engine_sha256": _file_sha256(design.model_engine_config),
        }
    )
    if manifest_path.exists():
        if not resume:
            raise FileExistsError(f"frozen score already exists: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"frozen score is incomplete: {manifest_path}")
        if manifest.get("fingerprint") != fingerprint:
            raise ValueError(f"frozen score fingerprint differs: {manifest_path}")
        if not arrays_path.is_file():
            raise FileNotFoundError(f"frozen score probabilities missing: {arrays_path}")
        with np.load(arrays_path, allow_pickle=False) as payload:
            mean_probability = payload["mean_probability"].astype(float)
        return {**dict(manifest["score"]), "mean_probability": mean_probability}

    score = score_frozen_candidate(
        subject_id=specification.subject_id,
        stimulus=arrays["stimulus"],
        choices=arrays["choices"],
        feedback=arrays["feedback"],
        base_engine_config=_base_engine(design),
        candidate_cell=candidate_cell,
        fixed_hyperparams=fixed_hyperparams,
        particle_count=int(frozen_budget["particle_count"]),
        filter_seeds=seeds,
        evaluation_protocol=evaluation_protocol,
        n_jobs=int(design.config["search"]["cd"]["parallel_budget"]),
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
    )
    scalar_score = {
        key: value for key, value in score.items() if key != "mean_probability"
    }
    _atomic_npz(
        arrays_path,
        mean_probability=np.asarray(score["mean_probability"], dtype=float),
    )
    _atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "status": "complete",
            "fingerprint": fingerprint,
            "score_label": str(score_label),
            "score": scalar_score,
            "probabilities_path": str(arrays_path),
        },
    )
    return score


def _generated_manifest(
    output_root: Path,
    specification: RecoveryDatasetSpec,
) -> dict[str, Any]:
    path = _synthetic_paths(output_root, specification)[2]
    if not path.is_file():
        raise FileNotFoundError(f"synthetic manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"synthetic dataset is incomplete: {path}")
    return manifest


def run_module_fit_phase(
    design: RecoveryDesign,
    output_root: Path,
    *,
    resume: bool,
    subject_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Fit all four cells on prefixes and score frozen fits on suffixes."""

    selected_subjects = (
        None
        if subject_ids is None
        else tuple(dict.fromkeys(int(value) for value in subject_ids))
    )
    if selected_subjects is not None and not selected_subjects:
        raise ValueError("module recovery subject_ids cannot be empty")
    phase_name = (
        "module-fit"
        if selected_subjects is None
        else "module-fit-subject-" + "-".join(map(str, selected_subjects))
    )
    score_filename = (
        "fit_scores.csv"
        if selected_subjects is None
        else "fit_scores_subject_" + "_".join(map(str, selected_subjects)) + ".csv"
    )
    score_path = output_root / "module_recovery" / score_filename
    if resume and _phase_is_complete(output_root, phase_name):
        return {"status": "complete", "score_path": str(score_path)}
    _require_phase(output_root, "generate")
    budget = require_frozen_budget(
        output_root / "numerical_calibration" / "frozen_budget.json"
    )
    _update_phase_manifest(output_root, phase_name, status="running")
    _, dataset_paths = load_subject_schedules(design)
    rows: list[dict[str, Any]] = []
    specifications = [
        specification
        for specification in design.module_datasets
        if selected_subjects is None
        or specification.subject_id in selected_subjects
    ]
    if selected_subjects is not None:
        missing = set(selected_subjects) - {
            specification.subject_id for specification in specifications
        }
        if missing:
            raise ValueError(
                f"module recovery subjects are not registered: {sorted(missing)}"
            )
    total = len(specifications) * 4
    completed = 0
    evaluation_protocol = {
        "mode": "sequential_holdout",
        "train_fraction": 0.70,
        "optimization_partition": "train",
        "simulation_partition": "evaluation",
    }
    for specification in specifications:
        arrays = _load_synthetic_arrays(
            _synthetic_paths(output_root, specification)[1]
        )
        generated = _generated_manifest(output_root, specification)
        for candidate_cell in ("P", "PM", "PH", "PMH"):
            selected = _fit_or_load_selected_hyperparams(
                design,
                specification,
                output_root,
                candidate_cell=candidate_cell,
                frozen_budget=budget,
                resume=resume,
            )
            score = _score_or_load_frozen_candidate(
                design,
                output_root,
                specification,
                arrays,
                dataset_paths,
                candidate_cell=candidate_cell,
                fixed_hyperparams=selected,
                frozen_budget=budget,
                score_label=candidate_cell,
                score_role="module_suffix",
                evaluation_protocol=evaluation_protocol,
                resume=resume,
            )
            rows.append(
                {
                    "dataset_id": specification.dataset_id,
                    "subject_id": specification.subject_id,
                    "trial_count": specification.trial_count,
                    "train_trial_count": specification.train_trial_count,
                    "evaluation_trial_count": specification.evaluation_trial_count,
                    "true_cell": specification.truth_cell,
                    "candidate_cell": candidate_cell,
                    "total_nll": score["total_nll"],
                    "mean_trial_nll": score["mean_trial_nll"],
                    "generated_accuracy": generated["generated_accuracy"],
                    "particle_count": budget["particle_count"],
                    "filter_seed_count": budget["filter_seed_count"],
                    "filter_seeds": json.dumps(score["filter_seeds"]),
                    "probability_aggregation": score[
                        "probability_aggregation"
                    ],
                    "selected_hyperparams": json.dumps(
                        to_builtin(selected), sort_keys=True
                    ),
                }
            )
            _atomic_csv(score_path, pd.DataFrame(rows))
            completed += 1
            print(
                f"[module-fit {completed}/{total}] "
                f"{specification.dataset_id} <- {candidate_cell}",
                flush=True,
            )
    frame = pd.DataFrame(rows)
    if len(frame) != total:
        raise ValueError(
            f"module recovery did not produce all {total} fit scores"
        )
    _atomic_csv(score_path, frame)
    details = {
        "dataset_count": len(specifications),
        "subjects": (
            sorted({row["subject_id"] for row in rows}) if rows else []
        ),
        "candidate_score_count": len(frame),
        "score_path": str(score_path),
    }
    _update_phase_manifest(
        output_root,
        phase_name,
        status="complete",
        details=details,
    )
    return {"status": "complete", **details}


def run_parameter_fit_phase(
    design: RecoveryDesign,
    output_root: Path,
    *,
    resume: bool,
    subject_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Recover PMH parameters and audit the true vector with paired seeds."""

    selected_subjects = (
        None
        if subject_ids is None
        else tuple(dict.fromkeys(int(value) for value in subject_ids))
    )
    if selected_subjects is not None and not selected_subjects:
        raise ValueError("parameter recovery subject_ids cannot be empty")
    phase_name = (
        "parameter-fit"
        if selected_subjects is None
        else "parameter-fit-subject-" + "-".join(map(str, selected_subjects))
    )
    score_filename = (
        "fit_scores.csv"
        if selected_subjects is None
        else "fit_scores_subject_" + "_".join(map(str, selected_subjects)) + ".csv"
    )
    score_path = output_root / "parameter_recovery" / score_filename
    if resume and _phase_is_complete(output_root, phase_name):
        return {"status": "complete", "score_path": str(score_path)}
    _require_phase(output_root, "generate")
    budget = require_frozen_budget(
        output_root / "numerical_calibration" / "frozen_budget.json"
    )
    _update_phase_manifest(output_root, phase_name, status="running")
    _, dataset_paths = load_subject_schedules(design)
    rows: list[dict[str, Any]] = []
    specifications = [
        specification
        for specification in design.parameter_datasets
        if selected_subjects is None
        or specification.subject_id in selected_subjects
    ]
    if selected_subjects is not None:
        missing = set(selected_subjects) - {
            specification.subject_id for specification in specifications
        }
        if missing:
            raise ValueError(
                f"parameter recovery subjects are not registered: {sorted(missing)}"
            )
    for index, specification in enumerate(specifications, start=1):
        arrays = _load_synthetic_arrays(
            _synthetic_paths(output_root, specification)[1]
        )
        selected = _fit_or_load_selected_hyperparams(
            design,
            specification,
            output_root,
            candidate_cell="PMH",
            frozen_budget=budget,
            resume=resume,
        )
        estimated = extract_model_0826_parameters(selected)
        if not set(MODEL_PARAMETER_NAMES).issubset(estimated):
            missing = sorted(set(MODEL_PARAMETER_NAMES) - set(estimated))
            raise ValueError(
                f"parameter fit {specification.dataset_id} lacks estimates: {missing}"
            )
        fit_root = _fit_root(output_root, specification, "PMH")
        shortlist = _load_final_rescore_shortlist(
            fit_root,
            specification.subject_id,
        )
        shortlist_scores: list[dict[str, Any]] = []
        selected_fingerprint = _canonical_fingerprint(selected)
        selected_score: dict[str, Any] | None = None
        for shortlist_rank, hyperparams in enumerate(shortlist):
            shortlist_score = _score_or_load_frozen_candidate(
                design,
                output_root,
                specification,
                arrays,
                dataset_paths,
                candidate_cell="PMH",
                fixed_hyperparams=hyperparams,
                frozen_budget=budget,
                score_label=f"shortlist_{shortlist_rank:02d}",
                score_role="parameter_near_best",
                evaluation_protocol={"mode": "all"},
                resume=resume,
            )
            shortlist_scores.append(shortlist_score)
            if _canonical_fingerprint(hyperparams) == selected_fingerprint:
                selected_score = shortlist_score
        if selected_score is None:
            raise ValueError(
                f"selected estimate is absent from final shortlist: "
                f"{specification.dataset_id}"
            )
        truth_hyperparams = model_0826_truth_hyperparams(specification.truth)
        truth_score = _score_or_load_frozen_candidate(
            design,
            output_root,
            specification,
            arrays,
            dataset_paths,
            candidate_cell="PMH",
            fixed_hyperparams=truth_hyperparams,
            frozen_budget=budget,
            score_label="true_vector",
            score_role="parameter_near_best",
            evaluation_protocol={"mode": "all"},
            resume=resume,
        )
        best_nll = min(
            *[float(row["total_nll"]) for row in shortlist_scores],
            float(truth_score["total_nll"]),
        )
        near_best_delta = float(
            design.config["parameter_recovery"]["near_best_delta_total_nll"]
        )
        row: dict[str, Any] = {
            "dataset_id": specification.dataset_id,
            "subject_id": specification.subject_id,
            "trial_count": specification.trial_count,
            "truth_profile": specification.truth_profile,
            "estimated_total_nll": selected_score["total_nll"],
            "true_total_nll": truth_score["total_nll"],
            "true_delta_nll": float(truth_score["total_nll"]) - best_nll,
            "true_within_near_best": bool(
                float(truth_score["total_nll"]) <= best_nll + near_best_delta
            ),
            "final_shortlist_size": len(shortlist_scores),
            "particle_count": budget["particle_count"],
            "filter_seed_count": budget["filter_seed_count"],
            "filter_seeds": json.dumps(selected_score["filter_seeds"]),
            "probability_aggregation": selected_score[
                "probability_aggregation"
            ],
            "selected_hyperparams": json.dumps(
                to_builtin(selected), sort_keys=True
            ),
        }
        for parameter in MODEL_PARAMETER_NAMES:
            row[f"true_{parameter}"] = specification.truth[parameter]
            row[f"estimated_{parameter}"] = estimated[parameter]
        rows.append(row)
        _atomic_csv(score_path, pd.DataFrame(rows))
        print(
            f"[parameter-fit {index}/{len(specifications)}] "
            f"{specification.dataset_id}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    if len(frame) != len(specifications):
        raise ValueError(
            f"parameter recovery did not produce all {len(specifications)} fit rows"
        )
    _atomic_csv(score_path, frame)
    details = {
        "dataset_count": len(frame),
        "subjects": (
            sorted({row["subject_id"] for row in rows}) if rows else []
        ),
        "score_path": str(score_path),
    }
    _update_phase_manifest(
        output_root,
        phase_name,
        status="complete",
        details=details,
    )
    return {"status": "complete", **details}


def run_summarize_phase(
    design: RecoveryDesign,
    output_root: Path,
    *,
    resume: bool,
) -> dict[str, Any]:
    """Rebuild all tables, PNG figures, and gate-derived conclusions."""

    report_path = output_root / "final_report.json"
    if resume and _phase_is_complete(output_root, "summarize"):
        return json.loads(report_path.read_text(encoding="utf-8"))
    _require_phase(output_root, "module-fit")
    _require_phase(output_root, "parameter-fit")
    _update_phase_manifest(output_root, "summarize", status="running")
    module_dir = output_root / "module_recovery"
    parameter_dir = output_root / "parameter_recovery"
    module_scores = pd.read_csv(module_dir / "fit_scores.csv")
    parameter_scores = pd.read_csv(parameter_dir / "fit_scores.csv")
    if len(module_scores) != 144 or len(parameter_scores) != 40:
        raise ValueError("recovery summaries require all pre-registered datasets")

    module_summary = summarize_module_recovery(
        module_scores,
        near_best_delta_nll=float(
            design.config["module_recovery"]["near_best_delta_total_nll"]
        ),
        gates=design.config["module_recovery"]["gates"],
    )
    parameter_space = load_model_parameter_space(
        design.parameter_space_path,
        expected_model_id="model_0826",
    )
    parameter_summary = summarize_parameter_recovery(
        parameter_scores,
        parameter_space=parameter_space,
        gates=design.config["parameter_recovery"]["gates"],
    )

    _atomic_csv(
        module_dir / "recovery_summary.csv",
        pd.DataFrame(module_summary["dataset_rows"]),
    )
    _atomic_csv(
        module_dir / "confusion_matrix.csv",
        pd.DataFrame(module_summary["confusion_rows"]),
    )
    _atomic_csv(
        module_dir / "cell_summary.csv",
        pd.DataFrame(module_summary["cell_rows"]),
    )
    _atomic_json(module_dir / "recovery_summary.json", module_summary)
    plot_module_recovery(
        module_summary,
        module_dir / "module_recovery_overview.png",
    )

    _atomic_csv(
        parameter_dir / "parameter_summary.csv",
        pd.DataFrame(parameter_summary["parameter_rows"]),
    )
    _atomic_csv(
        parameter_dir / "readout_confusion.csv",
        pd.DataFrame(parameter_summary["chi_confusion_rows"]),
    )
    _atomic_csv(
        parameter_dir / "capacity_confusion.csv",
        pd.DataFrame(parameter_summary["M_confusion_rows"]),
    )
    _atomic_csv(
        parameter_dir / "parameter_error_correlations.csv",
        pd.DataFrame(parameter_summary["error_correlation_rows"]),
    )
    _atomic_json(parameter_dir / "parameter_summary.json", parameter_summary)
    plot_parameter_recovery(
        parameter_summary,
        parameter_dir / "parameter_recovery_overview.png",
    )
    supported_parameters = [
        row["parameter"]
        for row in parameter_summary["parameter_rows"]
        if bool(row["supported"])
    ]
    unsupported_parameters = [
        row["parameter"]
        for row in parameter_summary["parameter_rows"]
        if not bool(row["supported"])
    ]
    report = {
        "schema_version": 1,
        "status": "complete",
        "analysis_id": design.analysis_id,
        "pf_budget": require_frozen_budget(
            output_root / "numerical_calibration" / "frozen_budget.json"
        ),
        "module_recovery": {
            "passes_pre_registered_gates": module_summary[
                "passes_pre_registered_gates"
            ],
            "overall_exact_recovery": module_summary[
                "overall_exact_recovery"
            ],
            "true_cell_near_best_coverage": module_summary[
                "true_cell_near_best_coverage"
            ],
        },
        "readout_recovery": {
            "supported": parameter_summary["chi_supported"],
            "exact_recovery": parameter_summary["chi_exact_recovery"],
            "near_best_coverage": parameter_summary[
                "chi_near_best_coverage"
            ],
        },
        "capacity_recovery": {
            "exact_recovery": parameter_summary["M_exact_recovery"],
            "wilson_95": [
                parameter_summary["M_wilson_low"],
                parameter_summary["M_wilson_high"],
            ],
        },
        "continuous_parameters": {
            "supported": supported_parameters,
            "unsupported": unsupported_parameters,
            "interpretation": (
                "Only supported parameters may be reported as stable "
                "individual-difference estimates."
            ),
        },
        "interpretation_boundaries": {
            "particle_filter": "numerical latent-state integration, not an added cognitive operation",
            "generated_trajectories": "independent behavioral observations; never probability-averaged together",
            "module_selection": "held-out suffix total choice NLL after prefix-only fitting",
        },
    }
    _atomic_json(report_path, report)
    _update_phase_manifest(
        output_root,
        "summarize",
        status="complete",
        details={"final_report": str(report_path)},
    )
    return report


def run_subject_summarize_phase(
    design: RecoveryDesign,
    output_root: Path,
    *,
    subject_id: int,
    resume: bool,
) -> dict[str, Any]:
    """Write an explicitly partial recovery summary for one subject template."""

    subject = int(subject_id)
    phase_name = f"summarize-subject-{subject}"
    summary_dir = output_root / "subject_summaries" / f"subject_{subject}"
    report_path = summary_dir / "partial_recovery_report.json"
    if resume and _phase_is_complete(output_root, phase_name):
        return json.loads(report_path.read_text(encoding="utf-8"))
    _require_phase(output_root, f"module-fit-subject-{subject}")
    _require_phase(output_root, f"parameter-fit-subject-{subject}")
    _update_phase_manifest(output_root, phase_name, status="running")

    module_scores = pd.read_csv(
        output_root
        / "module_recovery"
        / f"fit_scores_subject_{subject}.csv"
    )
    parameter_scores = pd.read_csv(
        output_root
        / "parameter_recovery"
        / f"fit_scores_subject_{subject}.csv"
    )
    module_summary = summarize_module_recovery(
        module_scores,
        near_best_delta_nll=float(
            design.config["module_recovery"]["near_best_delta_total_nll"]
        ),
        gates=design.config["module_recovery"]["gates"],
    )
    parameter_space = load_model_parameter_space(
        design.parameter_space_path,
        expected_model_id="model_0826",
    )
    parameter_summary = summarize_parameter_recovery(
        parameter_scores,
        parameter_space=parameter_space,
        gates=design.config["parameter_recovery"]["gates"],
    )
    _atomic_json(summary_dir / "module_recovery_summary.json", module_summary)
    _atomic_json(
        summary_dir / "parameter_recovery_summary.json",
        parameter_summary,
    )
    plot_module_recovery(
        module_summary,
        summary_dir / "module_recovery_overview.png",
    )
    plot_parameter_recovery(
        parameter_summary,
        summary_dir / "parameter_recovery_overview.png",
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "scope": "single_subject_partial_recovery",
        "analysis_id": design.analysis_id,
        "subject_id": subject,
        "module_dataset_count": int(module_scores["dataset_id"].nunique()),
        "parameter_dataset_count": int(parameter_scores["dataset_id"].nunique()),
        "module_recovery": {
            "passes_full_design_gates_on_subject_subset": module_summary[
                "passes_pre_registered_gates"
            ],
            "overall_exact_recovery": module_summary[
                "overall_exact_recovery"
            ],
            "true_cell_near_best_coverage": module_summary[
                "true_cell_near_best_coverage"
            ],
        },
        "parameter_recovery": {
            "chi_exact_recovery": parameter_summary["chi_exact_recovery"],
            "chi_near_best_coverage": parameter_summary[
                "chi_near_best_coverage"
            ],
            "M_exact_recovery": parameter_summary["M_exact_recovery"],
        },
        "interpretation": (
            "This subject-first report is an operational checkpoint. Final "
            "pre-registered conclusions require all 36 module and 40 parameter "
            "datasets."
        ),
    }
    _atomic_json(report_path, report)
    _update_phase_manifest(
        output_root,
        phase_name,
        status="complete",
        details={"report_path": str(report_path)},
    )
    return report


def run_priority_all(
    design: RecoveryDesign,
    output_root: Path,
    *,
    priority_subject: int,
    resume: bool,
) -> dict[str, Any]:
    """Run recovery continuously while completing one subject first."""

    subject_order = resolve_subject_order(
        tuple(design.subject_trial_counts),
        priority_subject,
    )
    _update_phase_manifest(
        output_root,
        "priority-all",
        status="running",
        details={"subject_order": list(subject_order)},
    )
    run_smoke_phase(design, output_root, resume=resume)
    run_calibrate_phase(design, output_root, resume=resume)
    run_generate_phase(design, output_root, resume=resume)
    subject_reports: dict[str, Any] = {}
    for subject in subject_order:
        run_module_fit_phase(
            design,
            output_root,
            resume=resume,
            subject_ids=(subject,),
        )
        run_parameter_fit_phase(
            design,
            output_root,
            resume=resume,
            subject_ids=(subject,),
        )
        subject_reports[str(subject)] = run_subject_summarize_phase(
            design,
            output_root,
            subject_id=subject,
            resume=resume,
        )

    # Consolidate registered all-subject tables from fingerprint-checked
    # artifacts before applying the full-design gates.
    run_module_fit_phase(design, output_root, resume=True)
    run_parameter_fit_phase(design, output_root, resume=True)
    final_report = run_summarize_phase(design, output_root, resume=True)
    details = {
        "subject_order": list(subject_order),
        "subject_reports": subject_reports,
        "final_report": final_report,
    }
    _update_phase_manifest(
        output_root,
        "priority-all",
        status="complete",
        details={"subject_order": list(subject_order)},
    )
    return details


def run(argv: Sequence[str] | None = None, *, default_config: Path = DEFAULT_CONFIG) -> None:
    args = build_parser(default_config=default_config).parse_args(argv)
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"recovery config does not exist: {config_path}")

    from src.Bayesian_state.workflows.recovery.design import load_recovery_design

    design = load_recovery_design(config_path)
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else design.output_root
    )
    prepare_output(
        output,
        resume=bool(args.resume),
        analysis_id=design.analysis_id,
        config_fingerprint=_design_fingerprint(design),
    )
    _record_run_provenance(design, output)
    if args.phase == "priority-all":
        if args.priority_subject is None:
            raise ValueError(
                "--phase priority-all requires --priority-subject"
            )
        try:
            run_priority_all(
                design,
                output,
                priority_subject=args.priority_subject,
                resume=bool(args.resume),
            )
        except Exception as exc:
            _update_phase_manifest(
                output,
                "priority-all",
                status="failed",
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        return
    if args.priority_subject is not None:
        raise ValueError(
            "--priority-subject is only valid with --phase priority-all"
        )
    handlers = {
        "smoke": run_smoke_phase,
        "generate": run_generate_phase,
        "calibrate": run_calibrate_phase,
        "module-fit": run_module_fit_phase,
        "parameter-fit": run_parameter_fit_phase,
        "summarize": run_summarize_phase,
    }
    phases = (
        ("smoke", "generate", "calibrate", "module-fit", "parameter-fit", "summarize")
        if args.phase == "all"
        else (args.phase,)
    )
    for phase in phases:
        print(f"[Model0826 recovery] phase={phase}", flush=True)
        try:
            handlers[phase](design, output, resume=bool(args.resume))
        except Exception as exc:
            _update_phase_manifest(
                output,
                phase,
                status="failed",
                details={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise


def main() -> None:
    run()


if __name__ == "__main__":
    main()
