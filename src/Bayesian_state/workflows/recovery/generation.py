"""Prepare registered truth settings and generate a recovery bundle via simulation."""
from __future__ import annotations
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Callable, Mapping
import numpy as np
import pandas as pd
from src.Bayesian_state.optimization.artifacts import to_builtin
from src.Bayesian_state.optimization.model_0826 import build_model_0826_cell_engine
from src.Bayesian_state.simulation.autonomous import run_autonomous_category_learning
from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
from src.Bayesian_state.optimization.recovery_parameters import model_0826_truth_hyperparams
from src.Bayesian_state.utils.recovery_artifacts import (
    _atomic_csv,
    _atomic_json,
    _atomic_npz,
    _canonical_fingerprint,
)
from src.Bayesian_state.simulation.recovery import (
    RecoveryDatasetSpec,
    FEATURE_COLUMNS,
    schedule_fingerprint,
    synthetic_dataset_frame,
)


def _truth_hyperparams(specification: RecoveryDatasetSpec) -> dict[str, Any]:
    return model_0826_truth_hyperparams(specification.truth)


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
