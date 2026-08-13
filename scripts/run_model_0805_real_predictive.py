#!/usr/bin/env python3
"""Overnight real-choice predictive comparison for model_0805.

The runner keeps the frozen outer temporal holdout closed while it screens and
selects predeclared model variants on an inner validation suffix.  It then
writes the selection manifest before producing any outer-holdout summary.
Latent paths, lapse levels, parameter-grid points, and filter seeds are mixed
sequentially in probability space; no best trajectory is selected.
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

from scripts.run_model_0803_cond1 import (  # noqa: E402
    PRIMARY_PRIOR,
    build_frozen_geometry,
    sha256_file,
    validate_and_load_inputs,
    validate_subject_cache,
)
from src.Bayesian_state.reference_models.model_0803 import TransitionKernels  # noqa: E402
from src.Bayesian_state.reference_models.model_0804.core import (  # noqa: E402
    Model0804Parameters,
    run_model0804_particle_filter,
)


DEFAULT_CONFIG = ROOT / "configs/model_0805_cond1_real_predictive.yaml"
DEFAULT_OUTPUT = (
    ROOT / "results/zhuran/model_0805_cond1/real_predictive_overnight_20260805_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=128)
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument(
        "--phase",
        choices=("all", "audit", "screen", "confirm", "select", "escalate", "report"),
        default="all",
    )
    parser.add_argument("--variants", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fields = sorted({key for row in rows for key in row}) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("model_0805 config must be a mapping")
    return config


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unavailable"


def stable_hash(payload: Mapping[str, Any], length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[: int(length)]


def parse_int_set(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_str_set(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def load_subject_arrays(audit: Mapping[str, Any]) -> dict[str, np.ndarray]:
    with np.load(Path(audit["q_path"]), allow_pickle=False) as payload:
        q = payload["q"].astype(np.float64)
    with np.load(Path(audit["prediction_path"]), allow_pickle=False) as payload:
        return {
            "q": q,
            "choice": payload["choice"].astype(np.int64),
            "feedback": payload["feedback"].astype(np.float64),
            "category": payload["category"].astype(np.int64),
            "outer_holdout": payload["holdout_mask"].astype(bool),
            "nr2": payload["p_NR2"].astype(np.float64),
        }


def split_masks(
    outer_holdout: np.ndarray, inner_fraction: float
) -> dict[str, np.ndarray]:
    outer = np.asarray(outer_holdout, dtype=bool).reshape(-1)
    outer_rows = np.flatnonzero(outer)
    if outer_rows.size == 0 or not np.array_equal(
        outer_rows, np.arange(outer_rows[0], outer.size)
    ):
        raise ValueError("outer holdout must be a contiguous suffix")
    outer_start = int(outer_rows[0])
    inner_count = max(1, int(math.ceil(outer_start * float(inner_fraction))))
    inner_start = outer_start - inner_count
    if inner_start < 2:
        raise ValueError("inner fit prefix is too short")
    inner_fit = np.zeros(outer.size, dtype=bool)
    inner_fit[:inner_start] = True
    inner_validation = np.zeros(outer.size, dtype=bool)
    inner_validation[inner_start:outer_start] = True
    outer_train = ~outer
    return {
        "inner_fit": inner_fit,
        "inner_validation": inner_validation,
        "outer_train": outer_train,
        "outer_holdout": outer,
    }


def structure_payload(
    model_id: str,
    capacity: int,
    gamma: float,
    w0: float,
    kappa: float,
    m: float,
    g: float,
    rho: float,
    memory_id: str,
) -> dict[str, Any]:
    payload = {
        "model_id": str(model_id),
        "capacity": int(capacity),
        "gamma": float(gamma),
        "w0": float(w0),
        "kappa": float(kappa),
        "m": float(m),
        "g": float(g),
        "rho": float(rho),
        "memory_id": str(memory_id),
    }
    payload["structure_id"] = stable_hash(payload)
    return payload


def model_key(structure: Mapping[str, Any]) -> str:
    if structure["model_id"] == "FS_H0":
        return "FS_H0"
    return f"{structure['model_id']}_M{int(structure['capacity'])}"


def base_structures(
    config: Mapping[str, Any], variant: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    if "memory_states" not in variant or "kappa" not in variant:
        raise ValueError(f"variant {variant['id']} requires derived structures")
    capacity = int(config["architecture"]["primary_capacity"])
    m_values = [float(value) for value in config["parameter_support"]["m"]]
    rho_fa2 = [float(value) for value in config["parameter_support"]["rho_FA2"]]
    rho_fa2r = [float(value) for value in config["parameter_support"]["rho_FA2R"]]
    output: dict[str, list[dict[str, Any]]] = {"FS_H0": [], "FA2": [], "FA2R": []}
    for memory in variant["memory_states"]:
        for kappa in variant["kappa"]:
            output["FS_H0"].append(
                structure_payload(
                    "FS_H0", 38, memory["gamma"], memory["w0"], kappa,
                    0.0, 0.0, 0.0, memory["id"],
                )
            )
            for g in variant["g"]:
                for m in m_values:
                    for rho in rho_fa2:
                        output["FA2"].append(
                            structure_payload(
                                "FA2", capacity, memory["gamma"], memory["w0"],
                                kappa, m, g, rho, memory["id"],
                            )
                        )
                    for rho in rho_fa2r:
                        output["FA2R"].append(
                            structure_payload(
                                "FA2R", capacity, memory["gamma"], memory["w0"],
                                kappa, m, g, rho, memory["id"],
                            )
                        )
    return {key: deduplicate_structures(value) for key, value in output.items()}


def deduplicate_structures(
    structures: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for value in structures:
        row = dict(value)
        unique[str(row["structure_id"])] = row
    return sorted(unique.values(), key=lambda row: str(row["structure_id"]))


def lapse_values(config: Mapping[str, Any], variant: Mapping[str, Any]) -> list[float]:
    label = str(variant["lapse_support"])
    key = "lapse_current" if label == "current" else "lapse_robust"
    return [float(value) for value in config["parameter_support"][key]]


def kernel_payload(kernel: TransitionKernels) -> dict[str, Any]:
    return {
        "local": kernel.local,
        "global": kernel.global_,
        "distance": kernel.distance,
        "tau_local": kernel.tau_local,
        "expected_local_distance": kernel.expected_local_distance,
        "expected_global_distance": kernel.expected_global_distance,
    }


def kernel_from_payload(payload: Mapping[str, Any]) -> TransitionKernels:
    return TransitionKernels(
        local=np.asarray(payload["local"], dtype=float),
        global_=np.asarray(payload["global"], dtype=float),
        distance=np.asarray(payload["distance"], dtype=float),
        tau_local=float(payload["tau_local"]),
        expected_local_distance=np.asarray(payload["expected_local_distance"], dtype=float),
        expected_global_distance=np.asarray(payload["expected_global_distance"], dtype=float),
    )


def component_path(
    output: Path,
    variant_id: str,
    stage: str,
    subject_id: int,
    structure: Mapping[str, Any],
    seed: int,
) -> Path:
    return (
        output / "components" / variant_id / stage / f"subject_{int(subject_id)}"
        / model_key(structure) / str(structure["structure_id"]) / f"seed_{int(seed)}.npz"
    )


def evaluate_component_task(task: Mapping[str, Any]) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    path = Path(task["path"])
    if path.exists() and not bool(task["force"]):
        return {"path": str(path), "skipped": True}
    started = time.time()
    with np.load(Path(task["q_path"]), allow_pickle=False) as payload:
        q = payload["q"].astype(np.float64)
    with np.load(Path(task["prediction_path"]), allow_pickle=False) as payload:
        choices = payload["choice"].astype(np.int64)
        feedback = payload["feedback"].astype(np.float64)
    structure = task["structure"]
    actual_model = "FA0" if structure["model_id"] == "FS_H0" else structure["model_id"]
    kernels = kernel_from_payload(task["kernels"])
    probabilities = []
    nll = []
    minimum_pre_ess = []
    minimum_post_ess = []
    resampling_count = []
    for lapse in task["lapse"]:
        parameters = Model0804Parameters(
            gamma=float(structure["gamma"]),
            w0=float(structure["w0"]),
            kappa=float(structure["kappa"]),
            m=float(structure["m"]),
            g=float(structure["g"]),
            rho=float(structure["rho"]),
            lapse=float(lapse),
        )
        trace = run_model0804_particle_filter(
            q,
            choices,
            feedback,
            np.asarray(task["prior"], dtype=float),
            kernels,
            model_id=actual_model,
            parameters=parameters,
            capacity=int(structure["capacity"]),
            particle_count=2 if actual_model == "FA0" else int(task["particle_count"]),
            filter_seed=int(task["seed"]),
            resample_threshold_fraction=(
                0.0 if actual_model == "FA0" else float(task["resample_threshold_fraction"])
            ),
            score_mask=np.ones(choices.size, dtype=bool),
            condition_on_choice_mask=np.ones(choices.size, dtype=bool),
            epsilon=float(task["epsilon"]),
        )
        probabilities.append(trace.probabilities.astype(np.float32))
        nll.append(float(trace.nll))
        minimum_pre_ess.append(float(np.min(trace.pre_choice_ess)))
        minimum_post_ess.append(float(np.min(trace.post_choice_ess)))
        resampling_count.append(int(np.sum(trace.resampled)))
    metadata = {
        "subject_id": int(task["subject_id"]),
        "variant_id": str(task["variant_id"]),
        "stage": str(task["stage"]),
        "structure": structure,
        "model_key": model_key(structure),
        "lapse": [float(value) for value in task["lapse"]],
        "seed": int(task["seed"]),
        "particle_count": int(task["particle_count"]),
        "runtime_seconds": float(time.time() - started),
    }
    atomic_savez(
        path,
        probabilities=np.asarray(probabilities, dtype=np.float32),
        nll=np.asarray(nll, dtype=np.float64),
        lapse=np.asarray(task["lapse"], dtype=np.float64),
        minimum_pre_ess=np.asarray(minimum_pre_ess, dtype=np.float64),
        minimum_post_ess=np.asarray(minimum_post_ess, dtype=np.float64),
        resampling_count=np.asarray(resampling_count, dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {"path": str(path), "skipped": False, "runtime_seconds": metadata["runtime_seconds"]}


def observed_log_probabilities(probabilities: np.ndarray, choices: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(choices, dtype=int).reshape(-1)
    if probs.ndim != 3 or probs.shape[1] != y.size:
        raise ValueError("component probabilities must have shape component x trial x choice")
    selected = np.take_along_axis(probs, y[None, :, None], axis=2)[:, :, 0]
    return np.log(np.clip(selected, 1e-300, 1.0))


def logsumexp(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    maximum = float(np.max(array))
    return maximum + math.log(float(np.sum(np.exp(array - maximum))))


def sequential_mixture(
    probabilities: np.ndarray,
    choices: np.ndarray,
    conditioning_mask: np.ndarray,
    score_mask: np.ndarray,
    log_prior: np.ndarray | None = None,
) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(choices, dtype=int).reshape(-1)
    condition = np.asarray(conditioning_mask, dtype=bool).reshape(-1)
    score = np.asarray(score_mask, dtype=bool).reshape(-1)
    if probs.ndim != 3 or probs.shape[1] != y.size:
        raise ValueError("probabilities must be component x trial x choice")
    if condition.size != y.size or score.size != y.size or np.any(score & ~condition):
        raise ValueError("score trials must be a subset of conditioned trials")
    count = probs.shape[0]
    if log_prior is None:
        log_weights = np.full(count, -math.log(count), dtype=float)
    else:
        log_weights = np.asarray(log_prior, dtype=float).reshape(-1).copy()
        if log_weights.size != count:
            raise ValueError("log_prior size mismatch")
        log_weights -= logsumexp(log_weights)
    mixture = np.full((y.size, probs.shape[2]), np.nan, dtype=float)
    nll = 0.0
    for trial in range(y.size):
        normalized = np.exp(log_weights - logsumexp(log_weights))
        mixture[trial] = normalized @ probs[:, trial, :]
        mixture[trial] /= float(np.sum(mixture[trial]))
        if score[trial]:
            nll -= math.log(max(float(mixture[trial, y[trial]]), 1e-300))
        if condition[trial]:
            log_weights += np.log(np.clip(probs[:, trial, y[trial]], 1e-300, 1.0))
    log_weights -= logsumexp(log_weights)
    return {
        "probabilities": mixture,
        "nll": float(nll),
        "final_weights": np.exp(log_weights),
    }


def score_predictions(
    probabilities: np.ndarray, choices: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(choices, dtype=int)
    selected = np.asarray(mask, dtype=bool)
    observed = probs[np.arange(y.size), y]
    nll = float(-np.sum(np.log(np.clip(observed[selected], 1e-300, 1.0))))
    binary = probs[:, 1]
    brier = float(np.mean(np.square(binary[selected] - y[selected])))
    accuracy = float(np.mean(np.argmax(probs[selected], axis=1) == y[selected]))
    confidence = np.max(probs[selected], axis=1)
    correctness = (np.argmax(probs[selected], axis=1) == y[selected]).astype(float)
    edges = np.linspace(0.5, 1.0, 11)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        include = (confidence >= low) & (confidence <= high if high == 1.0 else confidence < high)
        if np.any(include):
            ece += float(np.mean(include)) * abs(float(np.mean(confidence[include]) - np.mean(correctness[include])))
    return {
        "n_trials": int(np.sum(selected)),
        "nll": nll,
        "nll_per_trial": nll / float(np.sum(selected)),
        "brier": brier,
        "accuracy": accuracy,
        "ece10": float(ece),
        "extreme_error_rate": float(np.mean(observed[selected] < 0.05)),
    }


def load_component(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        probabilities = payload["probabilities"].astype(np.float64)
        metadata = json.loads(str(payload["metadata_json"].item()))
    return probabilities, metadata


def structure_fit_nll(path: Path, choices: np.ndarray, inner_fit: np.ndarray) -> float:
    probabilities, _ = load_component(path)
    logp = observed_log_probabilities(probabilities, choices)
    component_log_likelihood = np.sum(logp[:, inner_fit], axis=1)
    return float(-(logsumexp(component_log_likelihood) - math.log(logp.shape[0])))


def build_tasks(
    *,
    output: Path,
    variant_id: str,
    stage: str,
    subject_audits: Mapping[int, Mapping[str, Any]],
    structures_by_subject: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]],
    lapse: Sequence[float],
    seeds: Sequence[int],
    particle_count: int,
    prior: np.ndarray,
    kernels: TransitionKernels,
    resample_threshold_fraction: float,
    epsilon: float,
    force: bool,
) -> list[dict[str, Any]]:
    tasks = []
    kp = kernel_payload(kernels)
    for subject_id, by_model in structures_by_subject.items():
        audit = subject_audits[int(subject_id)]
        for structures in by_model.values():
            for structure in structures:
                actual_seeds = [0] if structure["model_id"] == "FS_H0" else seeds
                for seed in actual_seeds:
                    path = component_path(output, variant_id, stage, subject_id, structure, seed)
                    tasks.append(
                        {
                            "path": str(path),
                            "variant_id": variant_id,
                            "stage": stage,
                            "subject_id": int(subject_id),
                            "q_path": audit["q_path"],
                            "prediction_path": audit["prediction_path"],
                            "structure": dict(structure),
                            "lapse": list(lapse),
                            "seed": int(seed),
                            "particle_count": int(particle_count),
                            "prior": np.asarray(prior, dtype=float),
                            "kernels": kp,
                            "resample_threshold_fraction": float(resample_threshold_fraction),
                            "epsilon": float(epsilon),
                            "force": bool(force),
                        }
                    )
    return tasks


def run_tasks(
    tasks: Sequence[Mapping[str, Any]], jobs: int, progress_path: Path
) -> None:
    pending = [task for task in tasks if bool(task["force"]) or not Path(task["path"]).exists()]
    completed = len(tasks) - len(pending)
    started = time.time()
    atomic_json(progress_path, {"total": len(tasks), "completed": completed, "status": "running"})
    if not pending:
        atomic_json(progress_path, {"total": len(tasks), "completed": len(tasks), "status": "complete"})
        return
    failures = []
    with ProcessPoolExecutor(max_workers=int(jobs)) as executor:
        futures = {executor.submit(evaluate_component_task, task): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
            except Exception as error:
                failures.append(
                    {
                        "path": task["path"],
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                )
            completed += 1
            if completed % 32 == 0 or completed == len(tasks):
                atomic_json(
                    progress_path,
                    {
                        "total": len(tasks),
                        "completed": completed,
                        "failures": len(failures),
                        "elapsed_seconds": time.time() - started,
                        "status": "running" if completed < len(tasks) else "complete",
                    },
                )
    if failures:
        atomic_json(progress_path.with_name("failures.json"), failures)
        raise RuntimeError(f"{len(failures)} component tasks failed")


def subject_structure_map(
    subjects: Sequence[int], structures: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    return {
        int(subject_id): {
            model: [dict(value) for value in values]
            for model, values in structures.items()
        }
        for subject_id in subjects
    }


def rank_screen_structures(
    *,
    output: Path,
    variant_id: str,
    subjects: Sequence[int],
    structures_by_subject: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]],
    arrays_by_subject: Mapping[int, Mapping[str, np.ndarray]],
    screen_seed: int,
    retain: int,
) -> tuple[dict[int, dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    selected: dict[int, dict[str, list[dict[str, Any]]]] = {}
    rows = []
    for subject_id in subjects:
        masks = arrays_by_subject[int(subject_id)]["masks"]
        choices = arrays_by_subject[int(subject_id)]["choice"]
        selected[int(subject_id)] = {}
        for model, structures in structures_by_subject[int(subject_id)].items():
            scored = []
            for structure in structures:
                seed = 0 if structure["model_id"] == "FS_H0" else int(screen_seed)
                path = component_path(output, variant_id, "screen", subject_id, structure, seed)
                value = structure_fit_nll(path, choices, masks["inner_fit"])
                scored.append((value, dict(structure)))
                rows.append(
                    {
                        "variant_id": variant_id,
                        "subject_id": int(subject_id),
                        "model_key": model_key(structure),
                        "structure_id": structure["structure_id"],
                        "inner_fit_marginal_nll": value,
                        **{key: structure[key] for key in ("capacity", "gamma", "w0", "kappa", "m", "g", "rho", "memory_id")},
                    }
                )
            scored.sort(key=lambda item: (item[0], item[1]["structure_id"]))
            keep = scored if model == "FS_H0" else scored[: int(retain)]
            selected[int(subject_id)][model] = [item[1] for item in keep]
    return selected, rows


def confirmed_component_paths(
    output: Path,
    variant_id: str,
    stage: str,
    subject_id: int,
    structures: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> list[Path]:
    paths = []
    for structure in structures:
        actual_seeds = [0] if structure["model_id"] == "FS_H0" else seeds
        for seed in actual_seeds:
            paths.append(component_path(output, variant_id, stage, subject_id, structure, seed))
    return paths


def variant_dev_summary(
    *,
    output: Path,
    variant_id: str,
    subjects: Sequence[int],
    selected: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]],
    arrays_by_subject: Mapping[int, Mapping[str, np.ndarray]],
    seeds: Sequence[int],
    stage: str = "confirm",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    posterior_rows = []
    for subject_id in subjects:
        arrays = arrays_by_subject[int(subject_id)]
        masks = arrays["masks"]
        for model, structures in selected[int(subject_id)].items():
            component_probabilities = []
            component_metadata = []
            for path in confirmed_component_paths(
                output, variant_id, stage, subject_id, structures, seeds
            ):
                probabilities, metadata = load_component(path)
                component_probabilities.extend(probabilities)
                for lapse in metadata["lapse"]:
                    component_metadata.append({**metadata["structure"], "lapse": float(lapse), "seed": metadata["seed"]})
            stacked = np.asarray(component_probabilities, dtype=float)
            mixture = sequential_mixture(
                stacked,
                arrays["choice"],
                masks["outer_train"],
                masks["inner_validation"],
            )
            score = score_predictions(
                mixture["probabilities"], arrays["choice"], masks["inner_validation"]
            )
            rows.append(
                {
                    "variant_id": variant_id,
                    "subject_id": int(subject_id),
                    "model_key": model if model == "FS_H0" else model_key(structures[0]),
                    "stage": stage,
                    **score,
                }
            )
            weights = mixture["final_weights"]
            for field in ("gamma", "w0", "kappa", "m", "g", "rho", "lapse"):
                posterior_rows.append(
                    {
                        "variant_id": variant_id,
                        "subject_id": int(subject_id),
                        "model_key": model if model == "FS_H0" else model_key(structures[0]),
                        "field": field,
                        "posterior_mean_after_outer_train": float(
                            np.sum(weights * np.asarray([item[field] for item in component_metadata], dtype=float))
                        ),
                    }
                )
    return rows, posterior_rows


def group_summary(rows: Sequence[Mapping[str, Any]], segment: str) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    output = []
    for (variant, model), group in frame.groupby(["variant_id", "model_key"], sort=True):
        output.append(
            {
                "variant_id": str(variant),
                "model_key": str(model),
                "segment": segment,
                "subjects": int(len(group)),
                "mean_nll_per_trial": float(group.nll_per_trial.mean()),
                "median_nll_per_trial": float(group.nll_per_trial.median()),
                "mean_brier": float(group.brier.mean()),
                "mean_accuracy": float(group.accuracy.mean()),
                "mean_ece10": float(group.ece10.mean()),
                "mean_extreme_error_rate": float(group.extreme_error_rate.mean()),
            }
        )
    return output


def bootstrap_mean_interval(
    values: Sequence[float], replicates: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, array.size, size=(int(replicates), array.size))
    means = np.mean(array[draws], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def attach_iteration_diagnosis(
    selection: dict[str, Any], config: Mapping[str, Any]
) -> None:
    explanation = {
        "v0_current_0805": "current_restricted_architecture_was_sufficient_on_inner_validation",
        "v1_robust_readout": "decision_sensitivity_or_response_contamination_was_the_main_correctable_mismatch",
        "v2_memory_readout": "memory_timescale_or_endpoint_was_needed_beyond_readout_adaptation",
        "v3_search_average": "search_range_model_averaging_added_development_prediction",
        "v4_capacity_sensitivity": "workspace_capacity_changed_development_prediction",
    }
    fs = selection["selections"].get("FS_H0")
    diagnosis = {}
    for model, row in selection["selections"].items():
        chosen = str(row["variant_id"])
        item = {"selected_reason": explanation.get(chosen, "predeclared_variant_selected")}
        if model != "FS_H0" and fs is not None:
            advantage = float(fs["mean_inner_validation_nll_per_trial"]) - float(
                row["mean_inner_validation_nll_per_trial"]
            )
            item["mean_inner_validation_advantage_over_selected_FS_H0"] = advantage
            item["finite_model_promising"] = bool(
                advantage
                > float(config["variant_selection"]["finite_model_promising_if_mean_advantage_over_FS_H0"])
            )
        diagnosis[model] = item
    selection["iteration_diagnosis"] = diagnosis


def derive_v3_structures(
    *,
    config: Mapping[str, Any],
    variant: Mapping[str, Any],
    subjects: Sequence[int],
    v2_rank_rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    frame = pd.DataFrame(v2_rank_rows).sort_values(
        ["subject_id", "model_key", "inner_fit_marginal_nll", "structure_id"]
    )
    result: dict[int, dict[str, list[dict[str, Any]]]] = {}
    keep_pairs = int(variant["retained_pairs_per_subject_model"])
    m_values = [float(value) for value in config["parameter_support"]["m"]]
    rho_values = {
        "FA2": [float(value) for value in config["parameter_support"]["rho_FA2"]],
        "FA2R": [float(value) for value in config["parameter_support"]["rho_FA2R"]],
    }
    for subject_id in subjects:
        result[int(subject_id)] = {"FA2": [], "FA2R": []}
        for model in ("FA2", "FA2R"):
            key = f"{model}_M{int(config['architecture']['primary_capacity'])}"
            group = frame[(frame.subject_id == int(subject_id)) & (frame.model_key == key)]
            pairs = []
            seen = set()
            for row in group.itertuples(index=False):
                signature = (row.memory_id, row.gamma, row.w0, row.kappa)
                if signature not in seen:
                    pairs.append(signature)
                    seen.add(signature)
                if len(pairs) >= keep_pairs:
                    break
            for memory_id, gamma, w0, kappa in pairs:
                for g in variant["g"]:
                    for m in m_values:
                        for rho in rho_values[model]:
                            result[int(subject_id)][model].append(
                                structure_payload(
                                    model,
                                    int(config["architecture"]["primary_capacity"]),
                                    gamma,
                                    w0,
                                    kappa,
                                    m,
                                    g,
                                    rho,
                                    memory_id,
                                )
                            )
            result[int(subject_id)][model] = deduplicate_structures(result[int(subject_id)][model])
    return result


def select_variants(
    config: Mapping[str, Any], dev_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    frame = pd.DataFrame(dev_rows)
    threshold = float(
        config["variant_selection"]["minimum_mean_improvement_to_prefer_more_flexible_variant"]
    )
    order = [str(item["id"]) for item in config["variants"]]
    selections = {}
    for model in sorted(frame.model_key.unique()):
        group = frame[frame.model_key == model]
        means = group.groupby("variant_id").nll_per_trial.mean().to_dict()
        available = [value for value in order if value in means]
        chosen = available[0]
        for candidate in available[1:]:
            if float(means[chosen]) - float(means[candidate]) >= threshold:
                chosen = candidate
        selections[str(model)] = {
            "variant_id": chosen,
            "mean_inner_validation_nll_per_trial": float(means[chosen]),
            "all_variant_means": {key: float(value) for key, value in means.items()},
        }
    return {
        "selection_metric": config["variant_selection"]["metric"],
        "complexity_improvement_threshold": threshold,
        "selections": selections,
    }


def derive_v4_structures(
    *,
    config: Mapping[str, Any],
    subjects: Sequence[int],
    selection: Mapping[str, Any],
    selected_structures_by_variant: Mapping[str, Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]]],
) -> dict[int, dict[str, list[dict[str, Any]]]]:
    variant = next(item for item in config["variants"] if item["id"] == "v4_capacity_sensitivity")
    retain = int(variant["retained_structures_per_subject_model"])
    result: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for subject_id in subjects:
        result[int(subject_id)] = {}
        for base_model in ("FA2", "FA2R"):
            selected_key = f"{base_model}_M{int(config['architecture']['primary_capacity'])}"
            selected_variant = selection["selections"][selected_key]["variant_id"]
            source = selected_structures_by_variant[selected_variant][int(subject_id)][base_model][:retain]
            for capacity in variant["capacities"]:
                name = f"{base_model}_M{int(capacity)}"
                changed = []
                for old in source:
                    changed.append(
                        structure_payload(
                            base_model, int(capacity), old["gamma"], old["w0"], old["kappa"],
                            old["m"], old["g"], old["rho"], old["memory_id"],
                        )
                    )
                result[int(subject_id)][name] = deduplicate_structures(changed)
    return result


def outer_summary(
    *,
    output: Path,
    subjects: Sequence[int],
    final_choices: Mapping[str, Mapping[str, Any]],
    structures_by_variant: Mapping[str, Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]]],
    arrays_by_subject: Mapping[int, Mapping[str, np.ndarray]],
    confirmation_seeds: Sequence[int],
    escalation_seeds: Sequence[int],
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for output_model, choice in final_choices.items():
        variant_id = str(choice["variant_id"])
        source_model = str(choice["source_model"])
        stage = str(choice["stage"])
        seeds = escalation_seeds if stage == "escalate" else confirmation_seeds
        for subject_id in subjects:
            arrays = arrays_by_subject[int(subject_id)]
            structures = structures_by_variant[variant_id][int(subject_id)][source_model]
            probabilities = []
            for path in confirmed_component_paths(
                output, variant_id, stage, subject_id, structures, seeds
            ):
                current, _ = load_component(path)
                probabilities.extend(current)
            mixture = sequential_mixture(
                np.asarray(probabilities),
                arrays["choice"],
                np.ones_like(arrays["choice"], dtype=bool),
                arrays["masks"]["outer_holdout"],
            )
            rows.append(
                {
                    "variant_id": variant_id,
                    "subject_id": int(subject_id),
                    "model_key": output_model,
                    "stage": stage,
                    **score_predictions(
                        mixture["probabilities"],
                        arrays["choice"],
                        arrays["masks"]["outer_holdout"],
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    comparisons = []
    baseline = frame[frame.model_key == "FS_H0"].set_index("subject_id")
    for model in sorted(set(frame.model_key) - {"FS_H0"}):
        candidate = frame[frame.model_key == model].set_index("subject_id")
        common = baseline.index.intersection(candidate.index)
        delta = baseline.loc[common, "nll_per_trial"] - candidate.loc[common, "nll_per_trial"]
        low, high = bootstrap_mean_interval(
            delta.to_numpy(dtype=float),
            int(bootstrap_replicates),
            int(bootstrap_seed) + len(comparisons),
        )
        try:
            wilcoxon_result = wilcoxon(
                delta.to_numpy(dtype=float), alternative="two-sided", zero_method="wilcox"
            )
            wilcoxon_p = float(wilcoxon_result.pvalue)
        except ValueError:
            wilcoxon_p = 1.0
        comparisons.append(
            {
                "candidate": model,
                "baseline": "FS_H0",
                "subjects": int(len(common)),
                "mean_delta_nll_per_trial": float(delta.mean()),
                "median_delta_nll_per_trial": float(delta.median()),
                "improved_subjects": int(np.sum(delta > 0.0)),
                "bootstrap_mean_95_interval": [low, high],
                "wilcoxon_two_sided_p": wilcoxon_p,
            }
        )
    return rows, {"group": group_summary(rows, "outer_holdout"), "comparisons": comparisons}


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    selected_subjects = parse_int_set(args.subjects)
    frame, subjects, data_audit = validate_and_load_inputs(config, selected_subjects)
    if args.smoke:
        subjects = subjects[:2]
    priors, kernels, geometry_audit = build_frozen_geometry(config)
    prior = priors[PRIMARY_PRIOR]
    kernel = kernels[PRIMARY_PRIOR]
    subject_audits = {
        int(subject_id): validate_subject_cache(config, frame, subject_id)
        for subject_id in subjects
    }
    arrays_by_subject = {}
    inner_fraction = float(config["holdout"]["inner_validation_fraction_of_outer_training"])
    for subject_id in subjects:
        arrays = load_subject_arrays(subject_audits[int(subject_id)])
        arrays["masks"] = split_masks(arrays["outer_holdout"], inner_fraction)
        arrays_by_subject[int(subject_id)] = arrays
    variant_filter = parse_str_set(args.variants)
    variants = [
        item for item in config["variants"]
        if variant_filter is None or str(item["id"]) in variant_filter
    ]
    if args.smoke:
        config["screening"]["particle_count"] = 64
        config["confirmation"]["particle_count"] = 128
        config["confirmation"]["filter_seeds"] = [2026080521]
        config["escalation"]["enabled"] = False
        for item in variants:
            if "kappa" in item:
                item["kappa"] = item["kappa"][:2]
            if "memory_states" in item:
                item["memory_states"] = item["memory_states"][:2]
    manifest = {
        "analysis_id": config["analysis_id"],
        "started_at_unix": time.time(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "workers": int(args.jobs),
        "subjects": subjects,
        "smoke": bool(args.smoke),
        "data_audit": data_audit,
        "geometry_audit": geometry_audit,
        "guardrails": config["guardrails"],
        "outer_holdout_summary_created": False,
    }
    atomic_json(output / "manifest.json", manifest)
    split_rows = []
    for subject_id in subjects:
        masks = arrays_by_subject[int(subject_id)]["masks"]
        split_rows.append(
            {
                "subject_id": int(subject_id),
                **{f"n_{key}": int(np.sum(value)) for key, value in masks.items()},
            }
        )
    atomic_csv(output / "split_audit.csv", split_rows)
    if args.phase == "audit":
        return

    screening = config["screening"]
    confirmation = config["confirmation"]
    all_dev_rows: list[dict[str, Any]] = []
    all_posterior_rows: list[dict[str, Any]] = []
    rank_rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    screened_by_variant: dict[str, dict[int, dict[str, list[dict[str, Any]]]]] = {}
    confirmed_by_variant: dict[str, dict[int, dict[str, list[dict[str, Any]]]]] = {}
    v2_rank_rows: list[dict[str, Any]] | None = None

    for variant in variants:
        variant_id = str(variant["id"])
        if variant_id == "v4_capacity_sensitivity":
            continue
        if "derive_memory_readout_pairs_from" in variant:
            if v2_rank_rows is None:
                raise RuntimeError("v3 requires v2 screen ranks in the same run")
            structures_by_subject = derive_v3_structures(
                config=config,
                variant=variant,
                subjects=subjects,
                v2_rank_rows=v2_rank_rows,
            )
        else:
            structures_by_subject = subject_structure_map(
                subjects, base_structures(config, variant)
            )
        screened_by_variant[variant_id] = structures_by_subject
        current_lapse = lapse_values(config, variant)
        screen_tasks = build_tasks(
            output=output,
            variant_id=variant_id,
            stage="screen",
            subject_audits=subject_audits,
            structures_by_subject=structures_by_subject,
            lapse=current_lapse,
            seeds=screening["filter_seeds"],
            particle_count=int(screening["particle_count"]),
            prior=prior,
            kernels=kernel,
            resample_threshold_fraction=float(screening["resample_threshold_fraction"]),
            epsilon=float(config["numerics"]["likelihood_epsilon"]),
            force=bool(args.force),
        )
        if args.phase in {"all", "screen"}:
            run_tasks(screen_tasks, args.jobs, output / "progress" / f"{variant_id}_screen.json")
        selected, rank_rows = rank_screen_structures(
            output=output,
            variant_id=variant_id,
            subjects=subjects,
            structures_by_subject=structures_by_subject,
            arrays_by_subject=arrays_by_subject,
            screen_seed=int(screening["filter_seeds"][0]),
            retain=int(confirmation["retain_structures_per_subject_model"]),
        )
        rank_rows_by_variant[variant_id] = rank_rows
        atomic_csv(output / "screen_ranks" / f"{variant_id}.csv", rank_rows)
        if variant_id == "v2_memory_readout":
            v2_rank_rows = rank_rows
        confirmed_by_variant[variant_id] = selected
        confirm_tasks = build_tasks(
            output=output,
            variant_id=variant_id,
            stage="confirm",
            subject_audits=subject_audits,
            structures_by_subject=selected,
            lapse=current_lapse,
            seeds=confirmation["filter_seeds"],
            particle_count=int(confirmation["particle_count"]),
            prior=prior,
            kernels=kernel,
            resample_threshold_fraction=float(confirmation["resample_threshold_fraction"]),
            epsilon=float(config["numerics"]["likelihood_epsilon"]),
            force=bool(args.force),
        )
        if args.phase in {"all", "confirm"}:
            run_tasks(confirm_tasks, args.jobs, output / "progress" / f"{variant_id}_confirm.json")
        dev_rows, posterior_rows = variant_dev_summary(
            output=output,
            variant_id=variant_id,
            subjects=subjects,
            selected=selected,
            arrays_by_subject=arrays_by_subject,
            seeds=confirmation["filter_seeds"],
        )
        all_dev_rows.extend(dev_rows)
        all_posterior_rows.extend(posterior_rows)
        atomic_csv(output / "development" / f"{variant_id}_subject.csv", dev_rows)
        atomic_json(
            output / "development" / f"{variant_id}_group.json",
            group_summary(dev_rows, "inner_validation"),
        )

    atomic_csv(output / "development" / "all_subjects.csv", all_dev_rows)
    atomic_csv(output / "development" / "posterior_means.csv", all_posterior_rows)
    first_selection = select_variants(config, all_dev_rows)
    attach_iteration_diagnosis(first_selection, config)
    atomic_json(output / "development" / "pre_capacity_selection.json", first_selection)

    v4_variants = [item for item in variants if item["id"] == "v4_capacity_sensitivity"]
    if v4_variants:
        v4 = v4_variants[0]
        variant_id = str(v4["id"])
        structures_by_subject = derive_v4_structures(
            config=config,
            subjects=subjects,
            selection=first_selection,
            selected_structures_by_variant=confirmed_by_variant,
        )
        screened_by_variant[variant_id] = structures_by_subject
        current_lapse = lapse_values(config, v4)
        screen_tasks = build_tasks(
            output=output, variant_id=variant_id, stage="screen",
            subject_audits=subject_audits, structures_by_subject=structures_by_subject,
            lapse=current_lapse, seeds=screening["filter_seeds"],
            particle_count=int(screening["particle_count"]), prior=prior, kernels=kernel,
            resample_threshold_fraction=float(screening["resample_threshold_fraction"]),
            epsilon=float(config["numerics"]["likelihood_epsilon"]), force=bool(args.force),
        )
        if args.phase in {"all", "screen"}:
            run_tasks(screen_tasks, args.jobs, output / "progress" / f"{variant_id}_screen.json")
        selected, rank_rows = rank_screen_structures(
            output=output, variant_id=variant_id, subjects=subjects,
            structures_by_subject=structures_by_subject, arrays_by_subject=arrays_by_subject,
            screen_seed=int(screening["filter_seeds"][0]),
            retain=int(confirmation["retain_structures_per_subject_model"]),
        )
        confirmed_by_variant[variant_id] = selected
        atomic_csv(output / "screen_ranks" / f"{variant_id}.csv", rank_rows)
        confirm_tasks = build_tasks(
            output=output, variant_id=variant_id, stage="confirm",
            subject_audits=subject_audits, structures_by_subject=selected,
            lapse=current_lapse, seeds=confirmation["filter_seeds"],
            particle_count=int(confirmation["particle_count"]), prior=prior, kernels=kernel,
            resample_threshold_fraction=float(confirmation["resample_threshold_fraction"]),
            epsilon=float(config["numerics"]["likelihood_epsilon"]), force=bool(args.force),
        )
        if args.phase in {"all", "confirm"}:
            run_tasks(confirm_tasks, args.jobs, output / "progress" / f"{variant_id}_confirm.json")
        dev_rows, posterior_rows = variant_dev_summary(
            output=output, variant_id=variant_id, subjects=subjects, selected=selected,
            arrays_by_subject=arrays_by_subject, seeds=confirmation["filter_seeds"],
        )
        all_dev_rows.extend(dev_rows)
        all_posterior_rows.extend(posterior_rows)
        atomic_csv(output / "development" / f"{variant_id}_subject.csv", dev_rows)
        atomic_json(output / "development" / f"{variant_id}_group.json", group_summary(dev_rows, "inner_validation"))

    selection = select_variants(config, all_dev_rows)
    attach_iteration_diagnosis(selection, config)
    selection["created_before_outer_holdout_summary"] = True
    selection["created_at_unix"] = time.time()
    atomic_json(output / "variant_selection.json", selection)
    if args.phase == "select":
        return

    # For each selected model, retain the top training-evidence structures and
    # recompute them at N=32768.  Selection remains frozen in the file above.
    escalation = config["escalation"]
    final_choices: dict[str, dict[str, Any]] = {}
    escalation_structures: dict[str, dict[int, dict[str, list[dict[str, Any]]]]] = {}
    final_structures = deepcopy(confirmed_by_variant)
    for selected_model, selected_info in selection["selections"].items():
        variant_id = str(selected_info["variant_id"])
        if selected_model == "FS_H0":
            source_model = "FS_H0"
        elif selected_model.startswith("FA2R"):
            source_model = selected_model if variant_id == "v4_capacity_sensitivity" else "FA2R"
        else:
            source_model = selected_model if variant_id == "v4_capacity_sensitivity" else "FA2"
        if selected_model == "FS_H0" or not bool(escalation["enabled"]):
            final_choices[selected_model] = {
                "variant_id": variant_id, "source_model": source_model, "stage": "confirm"
            }
            continue
        per_subject: dict[int, dict[str, list[dict[str, Any]]]] = {}
        for subject_id in subjects:
            source = confirmed_by_variant[variant_id][int(subject_id)][source_model]
            per_subject[int(subject_id)] = {
                source_model: [dict(value) for value in source[: int(escalation["retain_structures_per_subject_model"])]],
            }
        escalation_structures[f"{variant_id}:{source_model}"] = per_subject
        variant = next(item for item in config["variants"] if item["id"] == variant_id)
        tasks = build_tasks(
            output=output, variant_id=variant_id, stage="escalate",
            subject_audits=subject_audits, structures_by_subject=per_subject,
            lapse=lapse_values(config, variant), seeds=confirmation["filter_seeds"],
            particle_count=int(escalation["particle_count"]), prior=prior, kernels=kernel,
            resample_threshold_fraction=float(confirmation["resample_threshold_fraction"]),
            epsilon=float(config["numerics"]["likelihood_epsilon"]), force=bool(args.force),
        )
        if args.phase in {"all", "escalate"}:
            run_tasks(tasks, args.jobs, output / "progress" / f"{variant_id}_{source_model}_escalate.json")
        final_choices[selected_model] = {
            "variant_id": variant_id, "source_model": source_model, "stage": "escalate"
        }

    # Ensure structures used by escalation replace, rather than merge with,
    # the larger confirmation panels for final prediction.
    for key, per_subject in escalation_structures.items():
        variant_id, source_model = key.split(":", 1)
        for subject_id in subjects:
            final_structures[variant_id][int(subject_id)][source_model] = per_subject[int(subject_id)][source_model]

    if args.phase in {"screen", "confirm", "escalate"}:
        return
    outer_rows, outer_report = outer_summary(
        output=output,
        subjects=subjects,
        final_choices=final_choices,
        structures_by_variant=final_structures,
        arrays_by_subject=arrays_by_subject,
        confirmation_seeds=confirmation["filter_seeds"],
        escalation_seeds=confirmation["filter_seeds"],
        bootstrap_replicates=int(config["outer_reporting"]["bootstrap_replicates"]),
        bootstrap_seed=int(config["outer_reporting"]["bootstrap_seed"]),
    )
    atomic_csv(output / "outer_holdout_subjects.csv", outer_rows)
    outer_report["analysis_id"] = config["analysis_id"]
    outer_report["variant_selection_path"] = str(output / "variant_selection.json")
    outer_report["final_choices"] = final_choices
    atomic_json(output / "outer_holdout_report.json", outer_report)
    manifest["outer_holdout_summary_created"] = True
    manifest["completed_at_unix"] = time.time()
    manifest["status"] = "complete"
    atomic_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
