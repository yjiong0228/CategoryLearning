#!/usr/bin/env python3
"""Run the frozen condition-1, choice-only model_0803 analysis.

The run has two independently resumable phases:

1. fit the predeclared real-data candidate grid for all condition-1 subjects;
2. run H0--H3 model/parameter/trajectory recovery on synthetic choices.

The independent statistical unit is the participant.  Trial-level likelihoods
are used for within-participant fitting, while group comparisons use one
held-out NLL-per-trial difference per participant with equal weighting.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Bayesian_state.reference_models.model_0803 import (  # noqa: E402
    EPS,
    MEMORY_IDS,
    MODEL_IDS,
    ORDER_COLUMNS,
    Model0803Fit,
    TransitionKernels,
    build_partition,
    build_transition_kernels,
    decode_parameters,
    expected_feedback_from_category,
    fit_model0803,
    parameter_definition,
    partition_prior,
    reference_feature_scaling,
    run_model0803,
    score_choice_predictions,
)


DEFAULT_CONFIG = ROOT / "configs/model_0803_cond1_formal.yaml"
DEFAULT_OUTPUT = ROOT / "results/zhuran/model_0803_cond1/formal_20260803_v3"
PRIMARY_PRIOR = "uniform_family"
SENSITIVITY_PRIOR = "uniform_rule"
NR2_ID = "NR2_frozen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=max(1, min(96, os.cpu_count() or 1)))
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--phase", choices=("all", "real", "recovery", "report"), default="all")
    parser.add_argument("--recovery-replicates", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=True)
        stream.write("\n")
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        if fieldnames:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("formal config must be a mapping")
    return payload


def resolve_root_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def frame_fingerprint(frame: pd.DataFrame) -> str:
    columns = [
        "iSub",
        "condition",
        *ORDER_COLUMNS,
        "feature1",
        "feature2",
        "feature3",
        "feature4",
        "choice",
        "feedback",
    ]
    hashed = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def candidate_id(model_id: str, memory_id: str, prior_id: str) -> str:
    return f"{model_id}_{memory_id}_{prior_id}"


def candidate_grid(config: Mapping[str, Any]) -> list[dict[str, str]]:
    models_cfg = config["models"]
    rows: list[dict[str, str]] = []
    for grid_name in ("primary_grid", "prior_sensitivity_grid"):
        grid = models_cfg[grid_name]
        for memory_id in grid["memory"]:
            for model_id in grid["models"]:
                rows.append(
                    {
                        "candidate_id": candidate_id(model_id, memory_id, grid["prior"]),
                        "model_id": str(model_id),
                        "memory_id": str(memory_id),
                        "prior_id": str(grid["prior"]),
                        "grid": grid_name,
                    }
                )
    identifiers = [row["candidate_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate grid contains duplicate identifiers")
    return rows


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unavailable"


def validate_and_load_inputs(
    config: Mapping[str, Any], selected_subjects: set[int] | None
) -> tuple[pd.DataFrame, list[int], dict[str, Any]]:
    data_cfg = config["data"]
    data_path = resolve_root_path(data_cfg["path"])
    observed_hash = sha256_file(data_path)
    if observed_hash != str(data_cfg["expected_sha256"]):
        raise ValueError(
            f"data hash changed: expected {data_cfg['expected_sha256']}, got {observed_hash}"
        )
    data = pd.read_csv(data_path, low_memory=False)
    condition = int(config["condition"])
    frame = data[data["condition"] == condition].copy()
    frame = frame.sort_values(["iSub", *ORDER_COLUMNS], kind="stable").reset_index(drop=True)
    if len(frame) != int(data_cfg["expected_rows"]):
        raise ValueError(f"condition-1 row count changed: {len(frame)}")
    all_subjects = sorted(int(value) for value in frame["iSub"].unique())
    if len(all_subjects) != int(data_cfg["expected_subjects"]):
        raise ValueError(f"condition-1 subject count changed: {len(all_subjects)}")
    if frame[["choice", "feedback"]].isna().any().any():
        raise ValueError("condition-1 choice/feedback contains missing values")
    expected_feedback = expected_feedback_from_category(
        condition,
        frame["choice"].to_numpy(dtype=int),
        frame["category"].to_numpy(dtype=int),
    )
    if not np.allclose(expected_feedback, frame["feedback"].to_numpy(dtype=float)):
        raise ValueError("condition-1 recorded feedback is inconsistent with choice/category")

    if selected_subjects is None:
        subjects = all_subjects
    else:
        unknown = sorted(selected_subjects - set(all_subjects))
        if unknown:
            raise ValueError(f"unknown requested subjects: {unknown}")
        subjects = [value for value in all_subjects if value in selected_subjects]
    if not subjects:
        raise ValueError("no subjects selected")

    trial_counts = frame.groupby("iSub").size()
    audit = {
        "data_path": str(data_path),
        "data_sha256": observed_hash,
        "n_rows_all_condition1": int(len(frame)),
        "n_subjects_all_condition1": int(len(all_subjects)),
        "n_selected_subjects": int(len(subjects)),
        "selected_subjects": subjects,
        "trial_count_min": int(trial_counts.loc[subjects].min()),
        "trial_count_max": int(trial_counts.loc[subjects].max()),
        "missing_choice": int(frame.loc[frame.iSub.isin(subjects), "choice"].isna().sum()),
        "missing_feedback": int(frame.loc[frame.iSub.isin(subjects), "feedback"].isna().sum()),
        "feedback_consistency_rate": 1.0,
        "independent_unit": "subject",
    }
    return frame, subjects, audit


def build_frozen_geometry(
    config: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, TransitionKernels], dict[str, Any]]:
    rule_cfg = config["rule_space"]
    similarity_path = resolve_root_path(rule_cfg["labelled_similarity_cache"])
    observed_hash = sha256_file(similarity_path)
    if observed_hash != str(rule_cfg["expected_similarity_sha256"]):
        raise ValueError(
            "labelled similarity hash changed: "
            f"expected {rule_cfg['expected_similarity_sha256']}, got {observed_hash}"
        )
    similarity = np.load(similarity_path)
    partition = build_partition(1)
    if partition.length != int(rule_cfg["expected_hypotheses"]):
        raise ValueError(f"condition-1 hypothesis count changed: {partition.length}")

    priors: dict[str, np.ndarray] = {}
    kernels: dict[str, TransitionKernels] = {}
    audit: dict[str, Any] = {
        "similarity_path": str(similarity_path),
        "similarity_sha256": observed_hash,
        "similarity_shape": list(similarity.shape),
        "priors": {},
    }
    for prior_id in (PRIMARY_PRIOR, SENSITIVITY_PRIOR):
        prior = partition_prior(partition, prior_id)
        kernel = build_transition_kernels(similarity, prior)
        gap = kernel.expected_global_distance - kernel.expected_local_distance
        if bool(config["numerics"]["require_local_global_distance_gap_every_rule"]):
            if np.any(gap <= 0.0):
                raise ValueError(f"{prior_id} local/global distance ordering fails")
        priors[prior_id] = prior
        kernels[prior_id] = kernel
        audit["priors"][prior_id] = {
            "prior": prior.tolist(),
            "tau_local": float(kernel.tau_local),
            "local_expected_distance_min": float(kernel.expected_local_distance.min()),
            "local_expected_distance_mean": float(kernel.expected_local_distance.mean()),
            "local_expected_distance_max": float(kernel.expected_local_distance.max()),
            "global_expected_distance_min": float(kernel.expected_global_distance.min()),
            "global_expected_distance_mean": float(kernel.expected_global_distance.mean()),
            "global_expected_distance_max": float(kernel.expected_global_distance.max()),
            "distance_gap_min": float(gap.min()),
            "distance_gap_mean": float(gap.mean()),
            "distance_gap_max": float(gap.max()),
            "local_row_sum_error": float(np.max(np.abs(kernel.local.sum(axis=1) - 1.0))),
            "global_row_sum_error": float(np.max(np.abs(kernel.global_.sum(axis=1) - 1.0))),
        }
    return priors, kernels, audit


def subject_paths(config: Mapping[str, Any], subject_id: int) -> tuple[Path, Path]:
    q_dir = resolve_root_path(config["perception"]["source"])
    prediction_dir = resolve_root_path(config["holdout"]["source"])
    return (
        q_dir / f"subject_{int(subject_id)}.npz",
        prediction_dir / f"subject_{int(subject_id)}.npz",
    )


def validate_subject_cache(
    config: Mapping[str, Any], frame: pd.DataFrame, subject_id: int
) -> dict[str, Any]:
    subject_frame = (
        frame[frame.iSub == int(subject_id)]
        .sort_values(list(ORDER_COLUMNS), kind="stable")
        .reset_index(drop=True)
    )
    q_path, prediction_path = subject_paths(config, subject_id)
    if not q_path.exists() or not prediction_path.exists():
        raise ValueError(f"subject {subject_id} is missing q/prediction cache")
    with np.load(q_path, allow_pickle=False) as q_data:
        q = q_data["q"]
        q_metadata = json.loads(str(q_data["metadata_json"].item()))
    with np.load(prediction_path, allow_pickle=False) as prediction:
        cached_choice = prediction["choice"].astype(int)
        cached_feedback = prediction["feedback"].astype(float)
        cached_category = prediction["category"].astype(int)
        holdout_mask = prediction["holdout_mask"].astype(bool)
        split_metadata = json.loads(str(prediction["split_metadata_json"].item()))
        nr2 = prediction["p_NR2"].astype(float)

    expected_fingerprint = frame_fingerprint(subject_frame)
    if q_metadata.get("frame_fingerprint") != expected_fingerprint:
        raise ValueError(f"subject {subject_id} q cache fingerprint is stale")
    if int(q_metadata.get("sobol_points", -1)) != int(config["perception"]["sobol_points"]):
        raise ValueError(f"subject {subject_id} q cache has wrong Sobol precision")
    expected_choice = subject_frame.choice.to_numpy(dtype=int) - 1
    expected_feedback = subject_frame.feedback.to_numpy(dtype=float)
    expected_category = subject_frame.category.to_numpy(dtype=int) - 1
    if q.shape != (len(subject_frame), 38, 2):
        raise ValueError(f"subject {subject_id} q shape changed: {q.shape}")
    if not np.array_equal(cached_choice, expected_choice):
        raise ValueError(f"subject {subject_id} cached choices do not match data")
    if not np.allclose(cached_feedback, expected_feedback):
        raise ValueError(f"subject {subject_id} cached feedback does not match data")
    if not np.array_equal(cached_category, expected_category):
        raise ValueError(f"subject {subject_id} cached category does not match data")
    holdout_rows = np.flatnonzero(holdout_mask)
    if holdout_rows.size == 0 or not np.array_equal(
        holdout_rows, np.arange(holdout_rows[0], len(holdout_mask))
    ):
        raise ValueError(f"subject {subject_id} holdout is not a contiguous suffix")
    if nr2.shape != (len(subject_frame), 2):
        raise ValueError(f"subject {subject_id} NR2 cache has wrong shape")
    return {
        "subject_id": int(subject_id),
        "q_path": str(q_path),
        "prediction_path": str(prediction_path),
        "n_trials": int(len(subject_frame)),
        "n_train": int((~holdout_mask).sum()),
        "n_holdout": int(holdout_mask.sum()),
        "frame_fingerprint": expected_fingerprint,
        "q_metadata": q_metadata,
        "split_metadata": split_metadata,
    }


def _fit_raw_mapping(fit: Model0803Fit) -> dict[str, float]:
    definition = parameter_definition(fit.model_id, fit.memory_id)
    return {
        name: float(fit.raw_vector[index])
        for index, name in enumerate(definition.names)
    }


def _logit_probability(value: float) -> float:
    value = float(np.clip(value, 1e-12, 1.0 - 1e-12))
    return math.log(value / (1.0 - value))


def nested_child_start(parent: Model0803Fit, child_model: str) -> np.ndarray:
    """Embed a parent optimum in the direct child parameterization."""

    if parent.model_id not in MODEL_IDS or child_model not in MODEL_IDS:
        raise ValueError("unknown parent/child model")
    if MODEL_IDS.index(child_model) != MODEL_IDS.index(parent.model_id) + 1:
        raise ValueError("nested warm start requires direct parent and child")
    target = parameter_definition(child_model, parent.memory_id)
    values = _fit_raw_mapping(parent)
    out = target.center.copy()
    for index, name in enumerate(target.names):
        if name in values:
            out[index] = values[name]

    if parent.model_id == "H0" and child_model == "H1":
        out[target.names.index("m")] = 0.0
    elif parent.model_id == "H1" and child_model == "H2":
        out[target.names.index("m")] = values["m"]
        out[target.names.index("g")] = 0.0
    elif parent.model_id == "H2" and child_model == "H3_M":
        out[target.names.index("mu_m")] = _logit_probability(values["m"])
        out[target.names.index("mu_g")] = _logit_probability(values["g"])
        out[target.names.index("phi_m")] = 0.0
        out[target.names.index("b_m_surprise")] = 0.0
        out[target.names.index("b_m_uncertainty")] = 0.0
    elif parent.model_id == "H3_M" and child_model == "H3_MG":
        out[target.names.index("phi_g")] = 0.0
        out[target.names.index("b_g_surprise")] = 0.0
        out[target.names.index("b_g_uncertainty")] = 0.0
    return out


def _load_subject_arrays(task: Mapping[str, Any]) -> dict[str, np.ndarray]:
    with np.load(Path(task["q_path"]), allow_pickle=False) as q_data:
        q_values = q_data["q"].astype(np.float64)
    with np.load(Path(task["prediction_path"]), allow_pickle=False) as prediction:
        return {
            "q": q_values,
            "choice": prediction["choice"].astype(np.int64),
            "feedback": prediction["feedback"].astype(np.float64),
            "category": prediction["category"].astype(np.int64),
            "holdout": prediction["holdout_mask"].astype(bool),
            "nr2": prediction["p_NR2"].astype(np.float64),
            "iSession": prediction["iSession"].astype(np.int32),
            "iBlock": prediction["iBlock"].astype(np.int32),
            "iTrial": prediction["iTrial"].astype(np.int32),
        }


def _kernel_from_task(task: Mapping[str, Any], prior_id: str) -> TransitionKernels:
    payload = task["geometry"][prior_id]
    return TransitionKernels(
        local=np.asarray(payload["local"], dtype=np.float64),
        global_=np.asarray(payload["global"], dtype=np.float64),
        distance=np.asarray(payload["distance"], dtype=np.float64),
        tau_local=float(payload["tau_local"]),
        expected_local_distance=np.asarray(payload["expected_local_distance"], dtype=float),
        expected_global_distance=np.asarray(payload["expected_global_distance"], dtype=float),
    )


def _metric_rows_for_probabilities(
    subject_id: int,
    candidate: Mapping[str, str],
    probabilities: np.ndarray,
    choices: np.ndarray,
    holdout: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for segment, mask in (("train", ~holdout), ("holdout", holdout)):
        rows.append(
            {
                "subject_id": int(subject_id),
                **candidate,
                "segment": segment,
                **score_choice_predictions(probabilities, choices, mask),
            }
        )
    return rows


def _fit_sequence(
    *,
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels: TransitionKernels,
    holdout: np.ndarray,
    memory_id: str,
    prior_id: str,
    config: Mapping[str, Any],
    seed_parts: Sequence[object],
    recovery: bool,
) -> list[Model0803Fit]:
    train = ~holdout
    scaling = reference_feature_scaling(
        q_values, choices, feedback, prior, kernels, train,
        epsilon=float(config["numerics"]["likelihood_epsilon"]),
    )
    starts_cfg = config["optimization"][
        "recovery_n_starts" if recovery else "real_n_starts"
    ]
    fits: list[Model0803Fit] = []
    parent: Model0803Fit | None = None
    for model_id in MODEL_IDS:
        extra = [] if parent is None else [nested_child_start(parent, model_id)]
        fit = fit_model0803(
            q_values,
            choices,
            feedback,
            prior,
            kernels,
            train,
            model_id=model_id,
            memory_id=memory_id,
            feature_scaling=scaling,
            n_starts=int(starts_cfg[model_id]),
            base_seed=int(config["optimization"]["base_seed"]),
            seed_parts=(*seed_parts, prior_id),
            extra_starts=extra,
            maxiter=int(config["optimization"]["max_iterations"]),
            epsilon=float(config["numerics"]["likelihood_epsilon"]),
        )
        if parent is not None and fit.train_nll > parent.train_nll + 1e-6:
            fit.diagnostics["nesting_failure"] = float(fit.train_nll - parent.train_nll)
        else:
            fit.diagnostics["nesting_failure"] = 0.0
        fits.append(fit)
        parent = fit
    return fits


def real_subject_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    started = time.time()
    subject_id = int(task["subject_id"])
    output = Path(task["output"])
    summary_path = output / "real_subjects" / f"subject_{subject_id}.json"
    state_path = output / "trial_states" / f"subject_{subject_id}.npz"
    if summary_path.exists() and state_path.exists() and not bool(task["force"]):
        return json.loads(summary_path.read_text(encoding="utf-8"))

    arrays = _load_subject_arrays(task)
    q_values = arrays["q"]
    choices = arrays["choice"]
    feedback = arrays["feedback"]
    holdout = arrays["holdout"]
    config = task["config"]
    metrics: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    fit_payload: dict[str, Any] = {}
    state_arrays: dict[str, Any] = {
        "subject_id": np.asarray(subject_id),
        "choice": choices.astype(np.int8),
        "feedback": feedback.astype(np.float32),
        "category": arrays["category"].astype(np.int8),
        "holdout_mask": holdout,
        "iSession": arrays["iSession"],
        "iBlock": arrays["iBlock"],
        "iTrial": arrays["iTrial"],
        "p_NR2_frozen": arrays["nr2"].astype(np.float32),
    }
    nr2_candidate = {
        "candidate_id": NR2_ID,
        "model_id": "NR2",
        "memory_id": "none",
        "prior_id": "none",
        "grid": "frozen_baseline",
    }
    metrics.extend(
        _metric_rows_for_probabilities(
            subject_id, nr2_candidate, arrays["nr2"], choices, holdout
        )
    )

    grid = task["candidates"]
    for prior_id in (PRIMARY_PRIOR, SENSITIVITY_PRIOR):
        prior = np.asarray(task["priors"][prior_id], dtype=np.float64)
        kernels = _kernel_from_task(task, prior_id)
        scaling = reference_feature_scaling(
            q_values,
            choices,
            feedback,
            prior,
            kernels,
            ~holdout,
            epsilon=float(config["numerics"]["likelihood_epsilon"]),
        )
        memories = [
            memory
            for memory in MEMORY_IDS
            if any(
                row["prior_id"] == prior_id and row["memory_id"] == memory
                for row in grid
            )
        ]
        for memory_id in memories:
            fits = _fit_sequence(
                q_values=q_values,
                choices=choices,
                feedback=feedback,
                prior=prior,
                kernels=kernels,
                holdout=holdout,
                memory_id=memory_id,
                prior_id=prior_id,
                config=config,
                seed_parts=("real", subject_id, memory_id),
                recovery=False,
            )
            allowed = {
                row["model_id"]: row
                for row in grid
                if row["prior_id"] == prior_id and row["memory_id"] == memory_id
            }
            for fit in fits:
                if fit.model_id not in allowed:
                    continue
                candidate = allowed[fit.model_id]
                trace = run_model0803(
                    q_values,
                    choices,
                    feedback,
                    prior,
                    kernels,
                    model_id=fit.model_id,
                    full_parameters=fit.full_parameters,
                    feature_scaling=scaling,
                    score_mask=np.ones(len(choices), dtype=bool),
                    record_states=True,
                    epsilon=float(config["numerics"]["likelihood_epsilon"]),
                )
                metrics.extend(
                    _metric_rows_for_probabilities(
                        subject_id, candidate, trace.probabilities, choices, holdout
                    )
                )
                parameter_row = {
                    "subject_id": subject_id,
                    **candidate,
                    **fit.parameters,
                    "train_nll_optimizer": float(fit.train_nll),
                    "mean_m": float(np.mean(trace.m)),
                    "sd_m": float(np.std(trace.m)),
                    "mean_g": float(np.mean(trace.g)),
                    "sd_g": float(np.std(trace.g)),
                    "mean_local_weight": float(np.mean(trace.operation_weights[:, 1])),
                    "mean_global_weight": float(np.mean(trace.operation_weights[:, 2])),
                    "max_memory_sync_error": float(np.max(trace.memory_sync_error)),
                    "optimizer_diagnostics_json": json.dumps(
                        fit.diagnostics, sort_keys=True, allow_nan=True
                    ),
                }
                parameters.append(parameter_row)
                fit_payload[candidate["candidate_id"]] = {
                    "parameters": fit.parameters,
                    "raw_vector": fit.raw_vector.tolist(),
                    "full_parameters": fit.full_parameters.tolist(),
                    "diagnostics": fit.diagnostics,
                    "feature_scaling": {
                        "center": scaling.center.tolist(),
                        "scale": scaling.scale.tolist(),
                        "reference": scaling.reference,
                    },
                }
                prefix = candidate["candidate_id"]
                state_arrays[f"p_{prefix}"] = trace.probabilities.astype(np.float32)
                state_arrays[f"m_{prefix}"] = trace.m.astype(np.float32)
                state_arrays[f"g_{prefix}"] = trace.g.astype(np.float32)
                state_arrays[f"weights_{prefix}"] = trace.operation_weights.astype(np.float32)
                state_arrays[f"surprise_{prefix}"] = trace.feedback_surprise.astype(np.float32)
                state_arrays[f"uncertainty_{prefix}"] = trace.rule_uncertainty.astype(np.float32)
                state_arrays[f"pi_minus_{prefix}"] = trace.pi_minus.astype(np.float32)
                state_arrays[f"pi_plus_{prefix}"] = trace.pi_plus.astype(np.float32)
                state_arrays[f"fade_{prefix}"] = trace.fade_state.astype(np.float32)
                state_arrays[f"static_{prefix}"] = trace.static_state.astype(np.float32)

    maximum_sync_error = max(
        float(row.get("max_memory_sync_error", 0.0)) for row in parameters
    )
    nesting_failures = [
        {
            "candidate_id": row["candidate_id"],
            "failure": json.loads(row["optimizer_diagnostics_json"])["nesting_failure"],
        }
        for row in parameters
        if json.loads(row["optimizer_diagnostics_json"])["nesting_failure"] > 1e-6
    ]
    summary = {
        "subject_id": subject_id,
        "status": "complete",
        "runtime_seconds": float(time.time() - started),
        "n_trials": int(len(choices)),
        "n_train": int((~holdout).sum()),
        "n_holdout": int(holdout.sum()),
        "metrics": metrics,
        "parameters": parameters,
        "fits": fit_payload,
        "maximum_memory_sync_error": maximum_sync_error,
        "nesting_failures": nesting_failures,
    }
    state_arrays["metadata_json"] = np.asarray(
        json.dumps(
            {
                "subject_id": subject_id,
                "candidate_ids": sorted(fit_payload),
                "maximum_memory_sync_error": maximum_sync_error,
            },
            sort_keys=True,
        )
    )
    atomic_savez(state_path, **state_arrays)
    atomic_json(summary_path, summary)
    return summary


def _geometry_payload(
    priors: Mapping[str, np.ndarray], kernels: Mapping[str, TransitionKernels]
) -> tuple[dict[str, list[float]], dict[str, dict[str, Any]]]:
    prior_payload = {key: value.tolist() for key, value in priors.items()}
    geometry = {}
    for key, kernel in kernels.items():
        geometry[key] = {
            "local": kernel.local,
            "global": kernel.global_,
            "distance": kernel.distance,
            "tau_local": float(kernel.tau_local),
            "expected_local_distance": kernel.expected_local_distance,
            "expected_global_distance": kernel.expected_global_distance,
        }
    return prior_payload, geometry


def run_real_phase(
    *,
    config: Mapping[str, Any],
    output: Path,
    subjects: Sequence[int],
    cache_audit: Mapping[int, Mapping[str, Any]],
    priors: Mapping[str, np.ndarray],
    kernels: Mapping[str, TransitionKernels],
    jobs: int,
    force: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prior_payload, geometry = _geometry_payload(priors, kernels)
    candidates = candidate_grid(config)
    tasks = []
    for subject_id in subjects:
        audit = cache_audit[int(subject_id)]
        tasks.append(
            {
                "subject_id": int(subject_id),
                "q_path": audit["q_path"],
                "prediction_path": audit["prediction_path"],
                "output": str(output),
                "config": deepcopy(dict(config)),
                "priors": prior_payload,
                "geometry": geometry,
                "candidates": candidates,
                "force": bool(force),
            }
        )

    summaries = []
    errors = []
    with ProcessPoolExecutor(max_workers=min(int(jobs), len(tasks))) as executor:
        futures = {executor.submit(real_subject_worker, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                summary = future.result()
                summaries.append(summary)
                print(
                    f"[real {completed}/{len(tasks)}] subject={summary['subject_id']} "
                    f"runtime={summary['runtime_seconds']:.1f}s",
                    flush=True,
                )
            except Exception as exc:
                error = {
                    "subject_id": int(task["subject_id"]),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(error)
                print(f"[real ERROR] {error}", flush=True)
    atomic_json(output / "real_worker_errors.json", errors)
    if errors:
        raise RuntimeError(f"real-data phase failed for {len(errors)} subjects")

    summaries.sort(key=lambda row: int(row["subject_id"]))
    metrics = [row for summary in summaries for row in summary["metrics"]]
    parameters = [row for summary in summaries for row in summary["parameters"]]
    atomic_csv(output / "subject_metrics.csv", metrics)
    atomic_csv(output / "subject_parameters.csv", parameters)
    return summaries, metrics, parameters


def load_real_summaries(output: Path, subjects: Sequence[int]):
    summaries = []
    for subject_id in subjects:
        path = output / "real_subjects" / f"subject_{int(subject_id)}.json"
        if not path.exists():
            raise ValueError(f"missing real-data summary: {path}")
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    metrics = [row for summary in summaries for row in summary["metrics"]]
    parameters = [row for summary in summaries for row in summary["parameters"]]
    return summaries, metrics, parameters


def write_real_descriptive_audits(
    config: Mapping[str, Any],
    output: Path,
    metrics: Sequence[Mapping[str, Any]],
    parameters: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Write participant-weighted model summaries and optimizer diagnostics."""

    metric_frame = pd.DataFrame(metrics)
    model_summary: list[dict[str, Any]] = []
    group_columns = [
        "candidate_id",
        "model_id",
        "memory_id",
        "prior_id",
        "grid",
        "segment",
    ]
    for keys, group in metric_frame.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "n_subjects": int(group["subject_id"].nunique()),
                "mean_nll_per_trial": float(group["nll_per_trial"].mean()),
                "median_nll_per_trial": float(group["nll_per_trial"].median()),
                "sd_nll_per_trial": float(group["nll_per_trial"].std(ddof=1)),
                "mean_brier": float(group["brier"].mean()),
                "mean_accuracy": float(group["accuracy"].mean()),
                "mean_confidence": float(group["mean_confidence"].mean()),
                "mean_entropy": float(group["mean_entropy"].mean()),
            }
        )
        model_summary.append(row)
    atomic_csv(output / "model_summary.csv", model_summary)

    optimization_rows: list[dict[str, Any]] = []
    for parameter_row in parameters:
        diagnostics = json.loads(str(parameter_row["optimizer_diagnostics_json"]))
        n_starts = int(diagnostics.get("n_starts", 0))
        n_evaluated = n_starts + int(bool(diagnostics.get("rescue_attempted", False)))
        boundary = list(diagnostics.get("boundary_parameters", []))
        optimization_rows.append(
            {
                "subject_id": int(parameter_row["subject_id"]),
                "candidate_id": parameter_row["candidate_id"],
                "model_id": parameter_row["model_id"],
                "memory_id": parameter_row["memory_id"],
                "prior_id": parameter_row["prior_id"],
                "optimizer_success": bool(diagnostics.get("success", False)),
                "n_starts": n_starts,
                "n_evaluated_optimizations": n_evaluated,
                "n_converged": int(diagnostics.get("n_converged", 0)),
                "converged_fraction": (
                    float(diagnostics.get("n_converged", 0)) / n_evaluated
                    if n_evaluated
                    else float("nan")
                ),
                "n_same_optimal_region": int(
                    diagnostics.get("n_same_optimal_region", 0)
                ),
                "rescue_attempted": bool(diagnostics.get("rescue_attempted", False)),
                "rescue_succeeded": bool(diagnostics.get("rescue_succeeded", False)),
                "unresolved_nonconverged_advantage": float(
                    diagnostics.get("unresolved_nonconverged_advantage", 0.0)
                ),
                "n_boundary_parameters": int(len(boundary)),
                "boundary_parameters": "|".join(boundary),
            }
        )
    atomic_csv(output / "optimization_audit.csv", optimization_rows)

    threshold = float(
        config["optimization"]["max_unresolved_nonconverged_advantage"]
    )
    optimization_summary: list[dict[str, Any]] = []
    audit_frame = pd.DataFrame(optimization_rows)
    for candidate, group in audit_frame.groupby("candidate_id", sort=True):
        boundary_counts: dict[str, int] = {}
        for value in group["boundary_parameters"]:
            for name in str(value).split("|"):
                if name:
                    boundary_counts[name] = boundary_counts.get(name, 0) + 1
        optimization_summary.append(
            {
                "candidate_id": candidate,
                "n_fits": int(len(group)),
                "n_successful_selected_fits": int(group["optimizer_success"].sum()),
                "n_rescue_attempted": int(group["rescue_attempted"].sum()),
                "n_rescue_succeeded": int(group["rescue_succeeded"].sum()),
                "n_unresolved_above_tolerance": int(
                    (
                        group["unresolved_nonconverged_advantage"] > threshold
                    ).sum()
                ),
                "maximum_unresolved_nonconverged_advantage": float(
                    group["unresolved_nonconverged_advantage"].max()
                ),
                "minimum_converged_fraction": float(
                    group["converged_fraction"].min()
                ),
                "median_converged_fraction": float(
                    group["converged_fraction"].median()
                ),
                "n_fits_with_boundary_parameters": int(
                    (group["n_boundary_parameters"] > 0).sum()
                ),
                "boundary_parameter_counts_json": json.dumps(
                    boundary_counts, sort_keys=True
                ),
            }
        )
    atomic_csv(output / "optimization_summary.csv", optimization_summary)
    return optimization_rows, optimization_summary


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _bootstrap_mean_interval(
    values: np.ndarray, replicates: int, seed: int
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(replicates), dtype=float)
    chunk = 2000
    for start in range(0, int(replicates), chunk):
        stop = min(int(replicates), start + chunk)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def comparison_rows(
    config: Mapping[str, Any], metrics: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = pd.DataFrame(metrics)
    heldout = frame[frame.segment == "holdout"].copy()
    by_candidate = {
        candidate: group.set_index("subject_id")
        for candidate, group in heldout.groupby("candidate_id")
    }
    primary_specs = config["inference"]["primary_comparisons"]
    bootstrap_replicates = int(config["inference"]["bootstrap_replicates"])
    bootstrap_seed = int(config["inference"]["bootstrap_seed"])

    def one(candidate: str, baseline: str, family: str, index: int) -> dict[str, Any]:
        if candidate not in by_candidate or baseline not in by_candidate:
            raise ValueError(f"missing comparison candidate/baseline: {candidate}, {baseline}")
        a, b = by_candidate[candidate].align(by_candidate[baseline], join="inner", axis=0)
        if len(a) == 0:
            raise ValueError(f"comparison {candidate} vs {baseline} has no paired subjects")
        delta_nll = b["nll_per_trial"].to_numpy(float) - a["nll_per_trial"].to_numpy(float)
        delta_brier = b["brier"].to_numpy(float) - a["brier"].to_numpy(float)
        if np.allclose(delta_nll, 0.0):
            p_value = 1.0
            statistic = 0.0
        else:
            test = wilcoxon(delta_nll, alternative="two-sided", method="auto")
            p_value = float(test.pvalue)
            statistic = float(test.statistic)
        low, high = _bootstrap_mean_interval(
            delta_nll,
            bootstrap_replicates,
            bootstrap_seed + 1009 * int(index),
        )
        return {
            "family": family,
            "candidate": candidate,
            "baseline": baseline,
            "n_subjects": int(len(delta_nll)),
            "mean_delta_nll_per_trial": float(np.mean(delta_nll)),
            "median_delta_nll_per_trial": float(np.median(delta_nll)),
            "bootstrap_ci_low": low,
            "bootstrap_ci_high": high,
            "improved_subjects": int(np.sum(delta_nll > 0.0)),
            "tied_subjects": int(np.sum(np.isclose(delta_nll, 0.0))),
            "mean_delta_brier": float(np.mean(delta_brier)),
            "wilcoxon_statistic": statistic,
            "p_value_raw": p_value,
            "subject_deltas_json": json.dumps(
                {
                    str(int(subject)): float(value)
                    for subject, value in zip(a.index.tolist(), delta_nll)
                },
                sort_keys=True,
            ),
        }

    primary = [
        one(spec["candidate"], spec["baseline"], "primary", index)
        for index, spec in enumerate(primary_specs)
    ]
    adjusted = _holm_adjust([row["p_value_raw"] for row in primary])
    for row, p_adjusted in zip(primary, adjusted):
        row["p_value_holm"] = float(p_adjusted)
        if p_adjusted < 0.05 and row["bootstrap_ci_low"] > 0.0:
            row["decision"] = "supported_improvement"
        elif p_adjusted < 0.05 and row["bootstrap_ci_high"] < 0.0:
            row["decision"] = "supported_worsening"
        else:
            row["decision"] = "inconclusive"

    secondary_specs: list[tuple[str, str]] = []
    for prior_id in (PRIMARY_PRIOR, SENSITIVITY_PRIOR):
        memories = MEMORY_IDS if prior_id == PRIMARY_PRIOR else ("dual",)
        for memory_id in memories:
            ids = [candidate_id(model, memory_id, prior_id) for model in MODEL_IDS]
            secondary_specs.extend(zip(ids[1:], ids[:-1]))
    for model_id in MODEL_IDS:
        secondary_specs.extend(
            [
                (
                    candidate_id(model_id, "fade", PRIMARY_PRIOR),
                    candidate_id(model_id, "bayes", PRIMARY_PRIOR),
                ),
                (
                    candidate_id(model_id, "dual", PRIMARY_PRIOR),
                    candidate_id(model_id, "fade", PRIMARY_PRIOR),
                ),
                (
                    candidate_id(model_id, "dual", PRIMARY_PRIOR),
                    candidate_id(model_id, "bayes", PRIMARY_PRIOR),
                ),
            ]
        )
    seen = set()
    secondary = []
    for index, spec in enumerate(secondary_specs):
        if spec in seen:
            continue
        seen.add(spec)
        secondary.append(one(spec[0], spec[1], "secondary", index + 100))
    secondary_adjusted = _holm_adjust([row["p_value_raw"] for row in secondary])
    for row, value in zip(secondary, secondary_adjusted):
        row["p_value_holm_secondary_family"] = float(value)
    return primary, secondary


def _sample_nonzero(rng: np.random.Generator, low: float, high: float, minimum: float) -> float:
    magnitude = rng.uniform(minimum, max(minimum, high))
    sign = -1.0 if rng.random() < 0.5 else 1.0
    return float(np.clip(sign * magnitude, low, high))


def sample_recovery_parameters(
    model_id: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    definition = parameter_definition(model_id, "dual")
    values: dict[str, float] = {
        "gamma": float(rng.uniform(0.25, 0.90)),
        "w0": float(rng.uniform(0.10, 0.85)),
        "log_kappa": float(rng.uniform(math.log(0.50), math.log(5.0))),
    }
    if model_id in {"H1", "H2"}:
        values["m"] = float(rng.uniform(0.08, 0.45))
    if model_id == "H2":
        values["g"] = float(rng.uniform(0.15, 0.80))
    if model_id in {"H3_M", "H3_MG"}:
        values.update(
            {
                "mu_m": _logit_probability(rng.uniform(0.08, 0.45)),
                "mu_g": _logit_probability(rng.uniform(0.15, 0.80)),
                "phi_m": float(rng.uniform(0.20, 0.82)),
                "b_m_surprise": _sample_nonzero(rng, -1.4, 1.4, 0.30),
                "b_m_uncertainty": _sample_nonzero(rng, -1.4, 1.4, 0.30),
            }
        )
    if model_id == "H3_MG":
        values.update(
            {
                "phi_g": float(rng.uniform(0.20, 0.82)),
                "b_g_surprise": _sample_nonzero(rng, -1.2, 1.2, 0.25),
                "b_g_uncertainty": _sample_nonzero(rng, -1.2, 1.2, 0.25),
            }
        )
    raw = definition.center.copy()
    for index, name in enumerate(definition.names):
        if name in values:
            raw[index] = values[name]
    full, reported = decode_parameters(raw, model_id, "dual")
    return raw, full, reported


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.size < 3 or np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def recovery_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    started = time.time()
    true_model = str(task["true_model"])
    replicate = int(task["replicate"])
    output_path = (
        Path(task["output"])
        / "recovery_replicates"
        / f"true_{true_model}_replicate_{replicate:03d}.json"
    )
    if output_path.exists() and not bool(task["force"]):
        return json.loads(output_path.read_text(encoding="utf-8"))

    arrays = _load_subject_arrays(task)
    q_values = arrays["q"]
    categories = arrays["category"]
    holdout = arrays["holdout"]
    config = task["config"]
    prior = np.asarray(task["priors"][PRIMARY_PRIOR], dtype=np.float64)
    kernels = _kernel_from_task(task, PRIMARY_PRIOR)
    # In binary condition 1, (choice, feedback) always maps back to the true
    # category.  Category choices with correct feedback therefore provide a
    # parameter-free reference history for state generation and scaling.
    reference_choices = categories.copy()
    reference_feedback = np.ones(len(categories), dtype=float)
    scaling = reference_feature_scaling(
        q_values,
        reference_choices,
        reference_feedback,
        prior,
        kernels,
        ~holdout,
        epsilon=float(config["numerics"]["likelihood_epsilon"]),
    )
    rng = np.random.default_rng(int(task["seed"]))
    true_raw, true_full, true_reported = sample_recovery_parameters(true_model, rng)
    true_trace = run_model0803(
        q_values,
        reference_choices,
        reference_feedback,
        prior,
        kernels,
        model_id=true_model,
        full_parameters=true_full,
        feature_scaling=scaling,
        score_mask=np.ones(len(categories), dtype=bool),
        record_states=False,
        epsilon=float(config["numerics"]["likelihood_epsilon"]),
    )
    uniforms = rng.random(len(categories))
    synthetic_choices = (uniforms >= true_trace.probabilities[:, 0]).astype(np.int64)
    synthetic_feedback = (synthetic_choices == categories).astype(float)
    invariance_trace = run_model0803(
        q_values,
        synthetic_choices,
        synthetic_feedback,
        prior,
        kernels,
        model_id=true_model,
        full_parameters=true_full,
        feature_scaling=scaling,
        score_mask=np.ones(len(categories), dtype=bool),
        record_states=False,
        epsilon=float(config["numerics"]["likelihood_epsilon"]),
    )
    generation_invariance_error = float(
        np.max(np.abs(invariance_trace.probabilities - true_trace.probabilities))
    )
    if generation_invariance_error > 1e-10:
        raise AssertionError(
            f"condition-1 generated feedback invariance failed: {generation_invariance_error}"
        )

    fits = _fit_sequence(
        q_values=q_values,
        choices=synthetic_choices,
        feedback=synthetic_feedback,
        prior=prior,
        kernels=kernels,
        holdout=holdout,
        memory_id="dual",
        prior_id=PRIMARY_PRIOR,
        config=config,
        seed_parts=("recovery", true_model, replicate, int(task["subject_id"])),
        recovery=True,
    )
    model_rows = []
    recovered_fit = None
    recovered_trace = None
    for fit in fits:
        trace = run_model0803(
            q_values,
            synthetic_choices,
            synthetic_feedback,
            prior,
            kernels,
            model_id=fit.model_id,
            full_parameters=fit.full_parameters,
            feature_scaling=scaling,
            score_mask=holdout,
            record_states=False,
            epsilon=float(config["numerics"]["likelihood_epsilon"]),
        )
        score = score_choice_predictions(trace.probabilities, synthetic_choices, holdout)
        model_rows.append(
            {
                "true_model": true_model,
                "replicate": replicate,
                "template_subject": int(task["subject_id"]),
                "fit_model": fit.model_id,
                "train_nll": float(fit.train_nll),
                "holdout_nll": float(score["nll"]),
                "holdout_nll_per_trial": float(score["nll_per_trial"]),
                "holdout_brier": float(score["brier"]),
                "optimizer_success": bool(fit.diagnostics["success"]),
                "nesting_failure": float(fit.diagnostics["nesting_failure"]),
            }
        )
        if fit.model_id == true_model:
            recovered_fit = fit
            recovered_trace = trace
    selected = min(
        model_rows,
        key=lambda row: (
            float(row["holdout_nll_per_trial"]),
            MODEL_IDS.index(str(row["fit_model"])),
        ),
    )["fit_model"]
    if recovered_fit is None or recovered_trace is None:
        raise AssertionError("generating-model recovery fit is missing")

    true_mapping = {
        name: float(true_raw[index])
        for index, name in enumerate(parameter_definition(true_model, "dual").names)
    }
    estimated_mapping = _fit_raw_mapping(recovered_fit)
    parameter_rows = [
        {
            "true_model": true_model,
            "replicate": replicate,
            "template_subject": int(task["subject_id"]),
            "parameter": name,
            "true_value": true_mapping[name],
            "estimated_value": estimated_mapping[name],
            "error": estimated_mapping[name] - true_mapping[name],
        }
        for name in true_mapping
    ]
    state_row = {
        "true_model": true_model,
        "replicate": replicate,
        "template_subject": int(task["subject_id"]),
        "m_correlation": _safe_correlation(true_trace.m, recovered_trace.m),
        "g_correlation": _safe_correlation(true_trace.g, recovered_trace.g),
        "m_rmse": float(np.sqrt(np.mean((true_trace.m - recovered_trace.m) ** 2))),
        "g_rmse": float(np.sqrt(np.mean((true_trace.g - recovered_trace.g) ** 2))),
    }
    payload = {
        "status": "complete",
        "true_model": true_model,
        "replicate": replicate,
        "template_subject": int(task["subject_id"]),
        "seed": int(task["seed"]),
        "runtime_seconds": float(time.time() - started),
        "selected_model": selected,
        "true_parameters": true_reported,
        "true_raw_vector": true_raw.tolist(),
        "generation_invariance_error": generation_invariance_error,
        "model_rows": model_rows,
        "parameter_rows": parameter_rows,
        "state_row": state_row,
    }
    atomic_json(output_path, payload)
    return payload


def _stable_seed(base_seed: int, *parts: object) -> int:
    text = ":".join([str(int(base_seed)), *(str(value) for value in parts)])
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


def _wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count <= 0:
        return float("nan"), float("nan")
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def summarize_recovery(
    payloads: Sequence[Mapping[str, Any]], output: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model_rows = [row for payload in payloads for row in payload["model_rows"]]
    parameter_rows = [row for payload in payloads for row in payload["parameter_rows"]]
    state_rows = [payload["state_row"] for payload in payloads]
    selection_rows = [
        {
            "true_model": payload["true_model"],
            "replicate": payload["replicate"],
            "template_subject": payload["template_subject"],
            "selected_model": payload["selected_model"],
        }
        for payload in payloads
    ]
    confusion = []
    for true_model in MODEL_IDS:
        selected = [
            row["selected_model"] for row in selection_rows if row["true_model"] == true_model
        ]
        for fit_model in MODEL_IDS:
            count = int(sum(value == fit_model for value in selected))
            confusion.append(
                {
                    "true_model": true_model,
                    "selected_model": fit_model,
                    "count": count,
                    "proportion": float(count / len(selected)) if selected else float("nan"),
                    "n_replicates": int(len(selected)),
                }
            )

    recovery_summary = []
    model_frame = pd.DataFrame(model_rows)
    for true_model in MODEL_IDS:
        selected = [
            row["selected_model"] for row in selection_rows if row["true_model"] == true_model
        ]
        successes = sum(value == true_model for value in selected)
        low, high = _wilson_interval(successes, len(selected))
        true_rows = model_frame[model_frame.true_model == true_model]
        pivot = true_rows.pivot(index="replicate", columns="fit_model", values="holdout_nll_per_trial")
        parent = MODEL_IDS[MODEL_IDS.index(true_model) - 1] if true_model != "H0" else None
        if parent is None:
            pairwise_rate = float("nan")
            pairwise_mean_delta = float("nan")
            pairwise_successes = 0
            pairwise_low = float("nan")
            pairwise_high = float("nan")
        else:
            delta = pivot[parent] - pivot[true_model]
            pairwise_successes = int(np.sum(delta > 0.0))
            pairwise_rate = float(np.mean(delta > 0.0))
            pairwise_mean_delta = float(np.mean(delta))
            pairwise_low, pairwise_high = _wilson_interval(
                pairwise_successes, len(delta)
            )
        recovery_summary.append(
            {
                "true_model": true_model,
                "n_replicates": int(len(selected)),
                "self_selected": int(successes),
                "self_selection_rate": float(successes / len(selected)) if selected else float("nan"),
                "self_selection_wilson_low": low,
                "self_selection_wilson_high": high,
                "chance_reference": 1.0 / len(MODEL_IDS),
                "direct_parent": parent,
                "pairwise_beats_parent": int(pairwise_successes),
                "pairwise_beats_parent_rate": pairwise_rate,
                "pairwise_parent_wilson_low": pairwise_low,
                "pairwise_parent_wilson_high": pairwise_high,
                "mean_parent_minus_true_nll_per_trial": pairwise_mean_delta,
            }
        )

    parameter_summary = []
    parameter_frame = pd.DataFrame(parameter_rows)
    for (true_model, parameter), group in parameter_frame.groupby(["true_model", "parameter"]):
        true = group.true_value.to_numpy(float)
        estimated = group.estimated_value.to_numpy(float)
        parameter_summary.append(
            {
                "true_model": true_model,
                "parameter": parameter,
                "n": int(len(group)),
                "bias": float(np.mean(estimated - true)),
                "rmse": float(np.sqrt(np.mean((estimated - true) ** 2))),
                "correlation": _safe_correlation(true, estimated),
                "true_mean": float(np.mean(true)),
                "estimated_mean": float(np.mean(estimated)),
            }
        )

    state_summary = []
    state_frame = pd.DataFrame(state_rows)
    for true_model, group in state_frame.groupby("true_model"):
        state_summary.append(
            {
                "true_model": true_model,
                "n": int(len(group)),
                "mean_m_correlation": float(group.m_correlation.mean()),
                "median_m_correlation": float(group.m_correlation.median()),
                "mean_g_correlation": float(group.g_correlation.mean()),
                "median_g_correlation": float(group.g_correlation.median()),
                "mean_m_rmse": float(group.m_rmse.mean()),
                "mean_g_rmse": float(group.g_rmse.mean()),
            }
        )

    atomic_csv(output / "model_recovery_rows.csv", model_rows)
    atomic_csv(output / "model_recovery_confusion.csv", confusion)
    atomic_csv(output / "model_recovery_summary.csv", recovery_summary)
    atomic_csv(output / "parameter_recovery_rows.csv", parameter_rows)
    atomic_csv(output / "parameter_recovery_summary.csv", parameter_summary)
    atomic_csv(output / "state_recovery_rows.csv", state_rows)
    atomic_csv(output / "state_recovery_summary.csv", state_summary)
    return confusion, recovery_summary, parameter_summary, state_summary


def run_recovery_phase(
    *,
    config: Mapping[str, Any],
    output: Path,
    subjects: Sequence[int],
    cache_audit: Mapping[int, Mapping[str, Any]],
    priors: Mapping[str, np.ndarray],
    kernels: Mapping[str, TransitionKernels],
    jobs: int,
    force: bool,
    replicates: int,
):
    prior_payload, geometry = _geometry_payload(priors, kernels)
    tasks = []
    base_seed = int(config["optimization"]["base_seed"])
    for model_index, true_model in enumerate(MODEL_IDS):
        for replicate in range(int(replicates)):
            subject_id = int(subjects[(replicate + 7 * model_index) % len(subjects)])
            audit = cache_audit[subject_id]
            tasks.append(
                {
                    "true_model": true_model,
                    "replicate": int(replicate),
                    "subject_id": subject_id,
                    "seed": _stable_seed(base_seed, "recovery", true_model, replicate, subject_id),
                    "q_path": audit["q_path"],
                    "prediction_path": audit["prediction_path"],
                    "output": str(output),
                    "config": deepcopy(dict(config)),
                    "priors": prior_payload,
                    "geometry": geometry,
                    "force": bool(force),
                }
            )
    payloads = []
    errors = []
    with ProcessPoolExecutor(max_workers=min(int(jobs), len(tasks))) as executor:
        futures = {executor.submit(recovery_worker, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                payload = future.result()
                payloads.append(payload)
                if completed <= 10 or completed % 25 == 0 or completed == len(tasks):
                    print(
                        f"[recovery {completed}/{len(tasks)}] true={payload['true_model']} "
                        f"rep={payload['replicate']} selected={payload['selected_model']} "
                        f"runtime={payload['runtime_seconds']:.1f}s",
                        flush=True,
                    )
            except Exception as exc:
                error = {
                    "true_model": task["true_model"],
                    "replicate": int(task["replicate"]),
                    "subject_id": int(task["subject_id"]),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(error)
                print(f"[recovery ERROR] {error}", flush=True)
    atomic_json(output / "recovery_worker_errors.json", errors)
    if errors:
        raise RuntimeError(f"recovery phase failed for {len(errors)} replicates")
    payloads.sort(key=lambda row: (MODEL_IDS.index(row["true_model"]), row["replicate"]))
    return payloads, summarize_recovery(payloads, output)


def load_recovery_payloads(output: Path, replicates: int):
    payloads = []
    for true_model in MODEL_IDS:
        for replicate in range(int(replicates)):
            path = (
                output
                / "recovery_replicates"
                / f"true_{true_model}_replicate_{replicate:03d}.json"
            )
            if not path.exists():
                raise ValueError(f"missing recovery payload: {path}")
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads, summarize_recovery(payloads, output)


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def write_results_report(
    *,
    output: Path,
    config: Mapping[str, Any],
    input_audit: Mapping[str, Any],
    geometry_audit: Mapping[str, Any],
    primary: Sequence[Mapping[str, Any]],
    recovery_summary: Sequence[Mapping[str, Any]] | None,
    optimization_summary: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    total_fits = int(sum(int(row["n_fits"]) for row in optimization_summary))
    total_rescues = int(
        sum(int(row["n_rescue_attempted"]) for row in optimization_summary)
    )
    total_boundaries = int(
        sum(
            int(row["n_fits_with_boundary_parameters"])
            for row in optimization_summary
        )
    )
    lines = [
        "# model_0803 condition-1 choice analysis",
        "",
        f"> Status: {manifest.get('status', 'unknown')}. This report supports only the frozen condition-1 choice scope.",
        "",
        "## Scope and independent unit",
        "",
        f"- Participants: {input_audit['n_selected_subjects']}; condition-1 rows: {input_audit['n_rows_all_condition1']}.",
        "- The independent statistical unit is the participant. Trial likelihoods were used only for within-participant fitting.",
        "- Parameters were fit on the frozen temporal training prefix and evaluated one step ahead on the frozen suffix.",
        "- RT, oral report, cross-condition generalization, H4/H5, and unique parameter interpretation are outside scope.",
        "",
        "## Frozen geometry",
        "",
        f"- Primary prior: `{config['rule_space']['primary_prior']}`; sensitivity prior: `{config['rule_space']['sensitivity_prior']}`.",
        f"- Primary local scale tau: {_fmt(geometry_audit['priors'][PRIMARY_PRIOR]['tau_local'])}.",
        f"- Minimum global-minus-local expected distance: {_fmt(geometry_audit['priors'][PRIMARY_PRIOR]['distance_gap_min'])}.",
        "",
        "## Optimization audit",
        "",
        f"- All {total_fits} selected real-data fits satisfied the optimizer convergence criterion.",
        f"- Continuation was attempted for {total_rescues} fits whose best initial result had not declared convergence.",
        f"- {total_boundaries}/{total_fits} selected fits had at least one parameter at a fitted bound; these estimates are not treated as uniquely identified mechanisms.",
        "- Candidate-level convergence fractions, continuations, unresolved objective gaps, and boundary counts are in `optimization_summary.csv`.",
        "",
        "## Primary held-out comparisons",
        "",
        "Positive delta NLL/trial favors the candidate. P values are two-sided paired Wilcoxon tests with Holm correction across the frozen primary family.",
        "",
        "| Candidate | Baseline | Mean delta NLL/trial | 95% subject-bootstrap CI | Improved | Holm p | Decision |",
        "|:--|:--|--:|:--|:--|--:|:--|",
    ]
    for row in primary:
        lines.append(
            "| {candidate} | {baseline} | {mean} | [{low}, {high}] | {improved}/{n} | {p} | {decision} |".format(
                candidate=row["candidate"],
                baseline=row["baseline"],
                mean=_fmt(row["mean_delta_nll_per_trial"]),
                low=_fmt(row["bootstrap_ci_low"]),
                high=_fmt(row["bootstrap_ci_high"]),
                improved=row["improved_subjects"],
                n=row["n_subjects"],
                p=_fmt(row["p_value_holm"]),
                decision=row["decision"],
            )
        )
    lines.extend(["", "## Recovery gate", ""])
    if recovery_summary is None:
        lines.append("Recovery was not run in this invocation; mechanism labels are not yet interpretable.")
    else:
        lines.extend(
            [
                "| Generator | Self-selection | Wilson 95% CI | Beats direct parent | Parent-recovery 95% CI |",
                "|:--|:--|:--|:--|:--|",
            ]
        )
        for row in recovery_summary:
            lines.append(
                "| {model} | {success}/{n} ({rate}) | [{low}, {high}] | {parent} | [{parent_low}, {parent_high}] |".format(
                    model=row["true_model"],
                    success=row["self_selected"],
                    n=row["n_replicates"],
                    rate=_fmt(row["self_selection_rate"], 3),
                    low=_fmt(row["self_selection_wilson_low"], 3),
                    high=_fmt(row["self_selection_wilson_high"], 3),
                    parent=_fmt(row["pairwise_beats_parent_rate"], 3),
                    parent_low=_fmt(row["pairwise_parent_wilson_low"], 3),
                    parent_high=_fmt(row["pairwise_parent_wilson_high"], 3),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A held-out predictive improvement is a condition-1 choice result. A psychological interpretation of H labels additionally requires successful model and trajectory recovery. Failure to improve is reported as lack of predictive support; an interval spanning zero is inconclusive rather than evidence of equivalence.",
            "",
            "## Reproducible artifacts",
            "",
            "- `manifest.json`: hashes, versions, frozen scope and run status.",
            "- `subject_metrics.csv`, `subject_parameters.csv`: participant-level evidence and estimates.",
            "- `model_summary.csv`: equal-participant descriptive predictive summaries.",
            "- `optimization_audit.csv`, `optimization_summary.csv`: convergence and boundary audit.",
            "- `primary_comparisons.csv`, `secondary_comparisons.csv`: paired group comparisons.",
            "- `trial_states/`: trialwise predictions, beliefs, memory states and H controls.",
            "- `model_recovery_*`, `parameter_recovery_*`, `state_recovery_*`: recovery evidence.",
            "",
        ]
    )
    path = output / "RESULTS.md"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    if args.smoke:
        config = deepcopy(config)
        for model_id in MODEL_IDS:
            config["optimization"]["real_n_starts"][model_id] = 2
            config["optimization"]["recovery_n_starts"][model_id] = 2
        config["optimization"]["max_iterations"] = 120
        config["inference"]["bootstrap_replicates"] = 500
    selected_subjects = None
    if args.subjects:
        selected_subjects = {
            int(value.strip()) for value in args.subjects.split(",") if value.strip()
        }
    frame, subjects, input_audit = validate_and_load_inputs(config, selected_subjects)
    priors, kernels, geometry_audit = build_frozen_geometry(config)
    cache_audit = {
        subject_id: validate_subject_cache(config, frame, subject_id)
        for subject_id in subjects
    }
    atomic_json(output / "input_audit.json", input_audit)
    atomic_json(output / "geometry_audit.json", geometry_audit)
    atomic_json(output / "cache_audit.json", cache_audit)

    recovery_replicates = int(
        args.recovery_replicates
        if args.recovery_replicates is not None
        else config["recovery"]["replicates_per_generator"]
    )
    if args.smoke and args.recovery_replicates is None:
        recovery_replicates = 1
    manifest = {
        "analysis_id": config["analysis_id"],
        "status": "running",
        "phase": args.phase,
        "smoke": bool(args.smoke),
        "started_unix": float(started),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "module_sha256": sha256_file(
            ROOT / "src/Bayesian_state/reference_models/model_0803.py"
        ),
        "git_commit": _git_commit(),
        "subjects": subjects,
        "n_subjects": len(subjects),
        "jobs": int(args.jobs),
        "recovery_replicates_per_generator": recovery_replicates,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "evidence_scope": (
            "condition-1 choice-only subject-wise bounded maximum-likelihood "
            "with frozen temporal holdout and recovery; no hierarchical posterior"
        ),
    }
    atomic_json(output / "manifest.json", manifest)

    try:
        if args.phase in {"all", "real"}:
            summaries, metrics, parameters = run_real_phase(
                config=config,
                output=output,
                subjects=subjects,
                cache_audit=cache_audit,
                priors=priors,
                kernels=kernels,
                jobs=args.jobs,
                force=args.force,
            )
        else:
            summaries, metrics, parameters = load_real_summaries(output, subjects)

        primary, secondary = comparison_rows(config, metrics)
        atomic_csv(output / "primary_comparisons.csv", primary)
        atomic_csv(output / "secondary_comparisons.csv", secondary)
        optimization_rows, optimization_summary = write_real_descriptive_audits(
            config, output, metrics, parameters
        )

        recovery_summary = None
        recovery_payloads: list[dict[str, Any]] = []
        if args.phase in {"all", "recovery"}:
            recovery_payloads, recovery_outputs = run_recovery_phase(
                config=config,
                output=output,
                subjects=subjects,
                cache_audit=cache_audit,
                priors=priors,
                kernels=kernels,
                jobs=args.jobs,
                force=args.force,
                replicates=recovery_replicates,
            )
            recovery_summary = recovery_outputs[1]
        elif args.phase == "report":
            recovery_payloads, recovery_outputs = load_recovery_payloads(
                output, recovery_replicates
            )
            recovery_summary = recovery_outputs[1]

        maximum_sync_error = max(
            float(summary["maximum_memory_sync_error"]) for summary in summaries
        )
        selected_fit_failures = [
            {
                "subject_id": int(summary["subject_id"]),
                "candidate_id": candidate,
            }
            for summary in summaries
            for candidate, fit_payload in summary.get("fits", {}).items()
            if not bool(fit_payload.get("diagnostics", {}).get("success", False))
        ]
        nesting_failures = [
            item
            for summary in summaries
            for item in summary.get("nesting_failures", [])
        ]
        unresolved_fit_gaps = [
            {
                "subject_id": int(row["subject_id"]),
                "candidate_id": row["candidate_id"],
                "advantage": float(row["unresolved_nonconverged_advantage"]),
            }
            for row in optimization_rows
            if float(row["unresolved_nonconverged_advantage"])
            > float(config["optimization"]["max_unresolved_nonconverged_advantage"])
        ]
        recovery_fit_failures = [
            {
                "true_model": payload["true_model"],
                "replicate": int(payload["replicate"]),
                "fit_model": row["fit_model"],
            }
            for payload in recovery_payloads
            for row in payload["model_rows"]
            if not bool(row["optimizer_success"])
        ]
        recovery_nesting_failures = [
            {
                "true_model": payload["true_model"],
                "replicate": int(payload["replicate"]),
                "fit_model": row["fit_model"],
                "failure": float(row["nesting_failure"]),
            }
            for payload in recovery_payloads
            for row in payload["model_rows"]
            if float(row["nesting_failure"]) > 1e-6
        ]
        maximum_generation_invariance_error = max(
            (
                float(payload["generation_invariance_error"])
                for payload in recovery_payloads
            ),
            default=0.0,
        )
        numerical_audit = {
            "maximum_memory_sync_error": maximum_sync_error,
            "synchronization_tolerance": float(
                config["numerics"]["synchronization_tolerance"]
            ),
            "n_nesting_failures": len(nesting_failures),
            "nesting_failures": nesting_failures,
            "n_selected_fit_failures": len(selected_fit_failures),
            "selected_fit_failures": selected_fit_failures,
            "maximum_allowed_unresolved_nonconverged_advantage": float(
                config["optimization"]["max_unresolved_nonconverged_advantage"]
            ),
            "n_unresolved_fit_gaps": len(unresolved_fit_gaps),
            "unresolved_fit_gaps": unresolved_fit_gaps,
            "maximum_generation_invariance_error": maximum_generation_invariance_error,
            "n_recovery_fit_failures": len(recovery_fit_failures),
            "recovery_fit_failures": recovery_fit_failures,
            "n_recovery_nesting_failures": len(recovery_nesting_failures),
            "recovery_nesting_failures": recovery_nesting_failures,
            "passed": bool(
                maximum_sync_error
                <= float(config["numerics"]["synchronization_tolerance"])
                and not nesting_failures
                and not selected_fit_failures
                and not unresolved_fit_gaps
                and not recovery_fit_failures
                and not recovery_nesting_failures
                and maximum_generation_invariance_error <= 1e-10
            ),
        }
        atomic_json(output / "numerical_audit.json", numerical_audit)
        if not numerical_audit["passed"]:
            raise RuntimeError("formal numerical audit failed")

        manifest.update(
            {
                "status": "complete",
                "runtime_seconds": float(time.time() - started),
                "completed_unix": float(time.time()),
                "n_real_candidates": len(candidate_grid(config)) + 1,
                "numerical_audit_passed": True,
            }
        )
        atomic_json(output / "manifest.json", manifest)
        write_results_report(
            output=output,
            config=config,
            input_audit=input_audit,
            geometry_audit=geometry_audit,
            primary=primary,
            recovery_summary=recovery_summary,
            optimization_summary=optimization_summary,
            manifest=manifest,
        )
        print(
            f"COMPLETE output={output} runtime={manifest['runtime_seconds']:.1f}s",
            flush=True,
        )
    except Exception as exc:
        manifest.update(
            {
                "status": "failed",
                "runtime_seconds": float(time.time() - started),
                "failed_unix": float(time.time()),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
