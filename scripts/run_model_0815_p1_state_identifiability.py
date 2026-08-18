#!/usr/bin/env python3
"""Diagnose M1 particle convergence and latent-state identifiability.

The run fixes one M1-generated trajectory for each selected subject, refilters
each trajectory at increasing particle counts with common seeds, and repeats
the highest-count fit while conditioning on the generator's complete
pre-choice orientation-belief vector.  It reports posterior expected switches,
ancestral support, and true-path likelihood replay in addition to MAP metrics.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from itertools import combinations
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0815_p1_mapping_sensitivity import (  # noqa: E402
    _choice_nll,
    _load_subject_arrays,
    _mean_js,
    _probability_rmse,
    _repo_path,
    _resolve_engine_for_model,
)
from src.Bayesian_state.inference.backends.particle_filter import (  # noqa: E402
    run_state_model_particle_filter,
)
from src.Bayesian_state.simulation.autonomous import (  # noqa: E402
    run_autonomous_category_learning,
)
from src.Bayesian_state.simulation.config import load_yaml  # noqa: E402
from src.Bayesian_state.utils.seeding import stable_seed  # noqa: E402
from src.Bayesian_state.utils.subjects import resolve_subject_config  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs/specific_models/model_0815_p1_state_identifiability.yaml"
)
MODES = ("baseline", "orientation_oracle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--reference-cache-dir",
        type=Path,
        help=(
            "Read matching completed fits from an earlier compatible cache "
            "without copying or overwriting that result directory."
        ),
    )
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument(
        "--phase",
        choices=("all", *MODES),
        default="all",
        help="Run both conditions or only one diagnostic condition.",
    )
    parser.add_argument(
        "--particle-counts",
        type=int,
        nargs="+",
        help="Override baseline particle counts; oracle uses the largest count.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse per-fit caches in a compatible, partially completed output directory.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one subject, 12 trials, R=4/8, and two seeds.",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    value = np.asarray(values, dtype=float).reshape(-1)
    weight = np.asarray(weights, dtype=float).reshape(-1)
    keep = np.isfinite(value) & np.isfinite(weight) & (weight >= 0.0)
    if not np.any(keep) or float(np.sum(weight[keep])) <= 0.0:
        return float("nan")
    normalized = weight[keep] / float(np.sum(weight[keep]))
    return float(np.sum(normalized * value[keep]))


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    value = np.asarray(values, dtype=float).reshape(-1)
    weight = np.asarray(weights, dtype=float).reshape(-1)
    keep = np.isfinite(value) & np.isfinite(weight) & (weight >= 0.0)
    value = value[keep]
    weight = weight[keep]
    if value.size == 0 or float(np.sum(weight)) <= 0.0:
        return float("nan")
    order = np.argsort(value)
    value = value[order]
    weight = weight[order] / float(np.sum(weight))
    cumulative = np.cumsum(weight)
    return float(value[min(int(np.searchsorted(cumulative, quantile)), value.size - 1)])


def _rmse(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    keep = np.isfinite(left) & np.isfinite(right)
    if left.shape != right.shape or not np.any(keep):
        return float("nan")
    return float(np.sqrt(np.mean(np.square(left[keep] - right[keep]))))


def _mean_pairwise(arrays: Sequence[np.ndarray], metric: Any) -> float:
    values = [
        metric(arrays[first], arrays[second])
        for first, second in combinations(range(len(arrays)), 2)
    ]
    return float(np.mean(values)) if values else float("nan")


def _binary_scores(probability: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    predicted = np.asarray(probability, dtype=float).reshape(-1)[1:]
    observed = np.asarray(truth, dtype=float).reshape(-1)[1:]
    if predicted.shape != observed.shape or predicted.size == 0:
        return float("nan"), float("nan")
    predicted = np.clip(predicted, 1e-12, 1.0 - 1e-12)
    brier = float(np.mean(np.square(predicted - observed)))
    log_loss = float(
        np.mean(-(observed * np.log(predicted) + (1.0 - observed) * np.log(1.0 - predicted)))
    )
    return brier, log_loss


def _orientation_scores(
    joint: np.ndarray,
    geometry: np.ndarray,
    truth: np.ndarray,
) -> tuple[float, float, float]:
    probability = np.asarray(joint, dtype=float)
    true_geometry = np.asarray(geometry, dtype=int).reshape(-1)
    true_orientation = np.asarray(truth, dtype=float).reshape(-1)
    rows = np.arange(true_geometry.size, dtype=int)
    mass = np.sum(probability[rows, true_geometry, :], axis=1)
    covered = mass > 1e-12
    if not np.any(covered):
        return 0.0, float("nan"), float("nan")
    estimated = probability[rows, true_geometry, 0][covered] / mass[covered]
    error = estimated - true_orientation[covered]
    return (
        float(np.mean(covered)),
        float(np.sqrt(np.mean(np.square(error)))),
        float(np.mean(np.abs(error))),
    )


def _extract_truth(generated: Any) -> dict[str, np.ndarray]:
    steps = list(generated.trajectory.step_log)
    geometry = np.asarray(
        [int(step["executed_hypothesis"]) for step in steps], dtype=int
    )
    orientation = np.stack(
        [np.asarray(step["orientation_probability"], dtype=float) for step in steps]
    )
    orientation_post = np.stack(
        [
            np.asarray(step["orientation_probability_post"], dtype=float)
            for step in steps
        ]
    )
    switch = np.asarray(
        [bool(step.get("execution_switch_event", False)) for step in steps],
        dtype=float,
    )
    geometry_switch = np.zeros_like(switch)
    geometry_switch[1:] = geometry[1:] != geometry[:-1]
    if not np.array_equal(switch[1:].astype(bool), geometry_switch[1:].astype(bool)):
        raise RuntimeError("generator switch events disagree with geometry changes.")
    choices = np.asarray(generated.trajectory.choices, dtype=int)
    observed = np.asarray(generated.trajectory.observed_probabilities, dtype=float)
    rows = np.arange(choices.size, dtype=int)
    selected = observed[rows, choices - 1]
    active = np.zeros((len(steps), orientation.shape[1]), dtype=float)
    for trial_index, step in enumerate(steps):
        active[trial_index, np.asarray(step["active_indices"], dtype=int)] = 1.0
    return {
        "geometry": geometry,
        "orientation": orientation,
        "orientation_post": orientation_post,
        "switch": switch,
        "observed_choice_probability": selected,
        "active": active,
    }


def _build_datasets(
    *,
    subjects: Sequence[int],
    engine_configs: Mapping[int, Mapping[str, Any]],
    simulation_config: Mapping[str, Any],
    simulation_config_path: Path,
    max_trials: int,
    synthetic_repeats: int,
    base_seed: int,
    readout_power: float,
    output_lapse: float,
) -> list[dict[str, Any]]:
    datasets: list[dict[str, Any]] = []
    for subject_id in subjects:
        subject_cfg = resolve_subject_config(simulation_config, int(subject_id))
        arrays, condition, dataset_paths = _load_subject_arrays(
            engine_config=engine_configs[int(subject_id)],
            subject_cfg=subject_cfg,
            simulation_config_path=simulation_config_path,
            subject_id=int(subject_id),
            max_trials=int(max_trials),
        )
        if arrays.categories is None:
            raise ValueError("Synthetic recovery requires hard task categories.")
        for replicate in range(int(synthetic_repeats)):
            generation_seed = stable_seed(
                {
                    "seed_role": "model0815_p1_state_identifiability_generation",
                    "base_seed": int(base_seed),
                    "subject_id": int(subject_id),
                    "replicate": int(replicate),
                }
            )
            generated = run_autonomous_category_learning(
                engine_config=engine_configs[int(subject_id)],
                subject_id=int(subject_id),
                condition=int(condition),
                stimulus=arrays.stimulus,
                categories=arrays.categories,
                trajectory_seed=int(generation_seed),
                choice_readout_config={
                    "method": "expectation",
                    "power": float(readout_power),
                },
                output_noise_config={
                    "enabled": float(output_lapse) > 0.0,
                    "base_lapse": float(output_lapse),
                    "post_error_lapse": 0.0,
                    "low_accuracy_lapse": 0.0,
                    "latent_volatility_lapse": 0.0,
                    "max_lapse": 1.0,
                    "lapse_target": "uniform",
                },
                processed_data_dir=dataset_paths["processed_dir"],
                dataset_paths=dataset_paths,
            )
            truth = _extract_truth(generated)
            datasets.append(
                {
                    "dataset_id": (
                        f"synthetic_m1_s{int(subject_id)}_r{int(replicate):02d}"
                    ),
                    "subject_id": int(subject_id),
                    "condition": int(condition),
                    "replicate": int(replicate),
                    "generation_seed": int(generation_seed),
                    "stimulus": np.asarray(arrays.stimulus, dtype=float),
                    "categories": np.asarray(arrays.categories, dtype=int),
                    "choices": np.asarray(generated.trajectory.choices, dtype=int),
                    "feedback": np.asarray(generated.trajectory.feedback, dtype=float),
                    "dataset_paths": dataset_paths,
                    "truth": truth,
                }
            )
    return datasets


def _cache_path(
    cache_dir: Path,
    *,
    dataset_id: str,
    mode: str,
    particle_count: int,
    seed_index: int,
) -> Path:
    return cache_dir / (
        f"{dataset_id}__{mode}__R{int(particle_count):04d}__"
        f"seed{int(seed_index):02d}.npz"
    )


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def _fit_and_cache(
    *,
    dataset: Mapping[str, Any],
    engine_config: Mapping[str, Any],
    mode: str,
    particle_count: int,
    seed_index: int,
    filter_seed: int,
    readout_power: float,
    output_lapse: float,
    resample_threshold_fraction: float,
    cache_path: Path,
    reference_cache_path: Path | None,
    resume: bool,
) -> dict[str, Any]:
    if resume and cache_path.exists():
        arrays = _load_cache(cache_path)
        loaded_cache_path = cache_path
    elif reference_cache_path is not None and reference_cache_path.exists():
        arrays = _load_cache(reference_cache_path)
        loaded_cache_path = reference_cache_path
    else:
        oracle = (
            np.asarray(dataset["truth"]["orientation"], dtype=float)
            if mode == "orientation_oracle"
            else None
        )
        fitted = run_state_model_particle_filter(
            engine_config=engine_config,
            subject_id=int(dataset["subject_id"]),
            stimulus=dataset["stimulus"],
            choices=dataset["choices"],
            feedback=dataset["feedback"],
            particle_count=int(particle_count),
            choice_readout_power=float(readout_power),
            output_lapse=float(output_lapse),
            resample_threshold_fraction=float(resample_threshold_fraction),
            filter_seed=int(filter_seed),
            choice_transmission_audit=True,
            orientation_oracle_schedule=oracle,
            processed_data_dir=dataset["dataset_paths"]["processed_dir"],
            dataset_paths=dataset["dataset_paths"],
        )
        ancestral = fitted.artifacts["audit_ancestral_paths"]
        arrays = {
            "probabilities": np.asarray(fitted.marginal_probabilities, dtype=float),
            "predictive_geometry": np.asarray(
                fitted.state_probabilities["executed_probability"], dtype=float
            ),
            "filtered_geometry": np.asarray(
                fitted.state_probabilities["filtered_executed_probability"],
                dtype=float,
            ),
            "predictive_orientation_joint": np.asarray(
                fitted.state_probabilities["executed_orientation_joint"],
                dtype=float,
            ),
            "filtered_orientation_joint": np.asarray(
                fitted.state_probabilities[
                    "filtered_executed_orientation_joint"
                ],
                dtype=float,
            ),
            "predictive_switch": np.asarray(
                fitted.latent_summaries[
                    "predictive_execution_switch_event_probability"
                ],
                dtype=float,
            ),
            "filtered_switch": np.asarray(
                fitted.latent_summaries["execution_switch_event_probability"],
                dtype=float,
            ),
            "post_ess": np.asarray(fitted.post_choice_ess, dtype=float),
            "resampled": np.asarray(fitted.resampled, dtype=bool),
            "resampling_unique_ancestors": np.asarray(
                fitted.resampling_unique_ancestors, dtype=int
            ),
            "path_indices": np.asarray(ancestral["particle_indices"], dtype=int),
            "path_weights": np.asarray(ancestral["weights"], dtype=float),
            "path_observed_choice_probability": np.asarray(
                ancestral["observed_choice_probability"], dtype=float
            ),
            "path_geometry": np.asarray(
                ancestral["executed_hypothesis"], dtype=int
            ),
            "path_switch": np.asarray(
                ancestral["execution_switch_event"], dtype=float
            ),
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays)
        temporary.replace(cache_path)
        loaded_cache_path = cache_path
    return {
        "dataset_id": str(dataset["dataset_id"]),
        "subject_id": int(dataset["subject_id"]),
        "replicate": int(dataset["replicate"]),
        "mode": str(mode),
        "particle_count": int(particle_count),
        "seed_index": int(seed_index),
        "filter_seed": int(filter_seed),
        "cache_path": str(loaded_cache_path),
        **arrays,
    }


def _run_metrics(run: Mapping[str, Any], dataset: Mapping[str, Any]) -> dict[str, Any]:
    truth = dataset["truth"]
    geometry = np.asarray(truth["geometry"], dtype=int)
    switch_truth = np.asarray(truth["switch"], dtype=float)
    rows = np.arange(geometry.size, dtype=int)
    predictive = np.asarray(run["predictive_geometry"], dtype=float)
    filtered = np.asarray(run["filtered_geometry"], dtype=float)
    predicted_switch = np.asarray(run["filtered_switch"], dtype=float)
    switch_brier, switch_log_loss = _binary_scores(predicted_switch, switch_truth)
    predictive_orientation = _orientation_scores(
        np.asarray(run["predictive_orientation_joint"], dtype=float),
        geometry,
        np.asarray(truth["orientation"], dtype=float)[rows, geometry],
    )
    filtered_orientation = _orientation_scores(
        np.asarray(run["filtered_orientation_joint"], dtype=float),
        geometry,
        np.asarray(truth["orientation_post"], dtype=float)[rows, geometry],
    )
    path_probability = np.asarray(
        run["path_observed_choice_probability"], dtype=float
    )
    path_nll = np.mean(-np.log(np.clip(path_probability, 1e-12, 1.0)), axis=1)
    path_weights = np.asarray(run["path_weights"], dtype=float)
    true_path_nll = float(
        np.mean(
            -np.log(
                np.clip(
                    np.asarray(truth["observed_choice_probability"], dtype=float),
                    1e-12,
                    1.0,
                )
            )
        )
    )
    path_geometry = np.asarray(run["path_geometry"], dtype=int)
    path_accuracy = np.mean(path_geometry == geometry[None, :], axis=1)
    path_indices = np.asarray(run["path_indices"], dtype=int)
    lineage_fraction = np.asarray(
        [np.unique(path_indices[:, trial]).size for trial in range(geometry.size)],
        dtype=float,
    ) / float(run["particle_count"])
    return {
        "dataset_id": str(run["dataset_id"]),
        "subject_id": int(run["subject_id"]),
        "replicate": int(run["replicate"]),
        "mode": str(run["mode"]),
        "particle_count": int(run["particle_count"]),
        "seed_index": int(run["seed_index"]),
        "filter_seed": int(run["filter_seed"]),
        "choice_nll": _choice_nll(
            np.asarray(run["probabilities"]), np.asarray(dataset["choices"])
        ),
        "predictive_geometry_map_accuracy": float(
            np.mean(np.argmax(predictive, axis=1) == geometry)
        ),
        "filtered_geometry_map_accuracy": float(
            np.mean(np.argmax(filtered, axis=1) == geometry)
        ),
        "predictive_geometry_true_mass": float(
            np.mean(predictive[rows, geometry])
        ),
        "filtered_geometry_true_mass": float(np.mean(filtered[rows, geometry])),
        "predictive_true_geometry_support": float(
            np.mean(predictive[rows, geometry] > 1e-12)
        ),
        "filtered_true_geometry_support": float(
            np.mean(filtered[rows, geometry] > 1e-12)
        ),
        "predictive_orientation_true_geometry_coverage": predictive_orientation[0],
        "predictive_orientation_rmse_given_true_geometry": predictive_orientation[1],
        "predictive_orientation_mae_given_true_geometry": predictive_orientation[2],
        "filtered_orientation_true_geometry_coverage": filtered_orientation[0],
        "filtered_orientation_rmse_given_true_geometry": filtered_orientation[1],
        "filtered_orientation_mae_given_true_geometry": filtered_orientation[2],
        "filtered_expected_switch_count": float(np.sum(predicted_switch[1:])),
        "true_switch_count": int(np.sum(switch_truth[1:])),
        "switch_brier": switch_brier,
        "switch_log_loss": switch_log_loss,
        "median_ess_fraction": float(
            np.median(np.asarray(run["post_ess"], dtype=float))
            / float(run["particle_count"])
        ),
        "resample_fraction": float(np.mean(np.asarray(run["resampled"], dtype=float))),
        "initial_ancestral_lineage_fraction": float(lineage_fraction[0]),
        "minimum_ancestral_lineage_fraction": float(np.min(lineage_fraction)),
        "ancestral_true_geometry_support": float(
            np.mean(np.any(path_geometry == geometry[None, :], axis=0))
        ),
        "ancestral_weighted_geometry_accuracy": _weighted_mean(
            path_accuracy, path_weights
        ),
        "ancestral_best_geometry_accuracy": float(np.max(path_accuracy)),
        "true_path_choice_nll": true_path_nll,
        "ancestral_weighted_choice_nll": _weighted_mean(path_nll, path_weights),
        "ancestral_median_choice_nll": _weighted_quantile(
            path_nll, path_weights, 0.5
        ),
        "true_path_better_than_ancestral_fraction": float(
            np.sum(path_weights[path_nll >= true_path_nll])
        ),
        "ancestral_terminal_expected_switch_count": _weighted_mean(
            np.sum(np.asarray(run["path_switch"], dtype=float)[:, 1:], axis=1),
            path_weights,
        ),
    }


def _aggregate_runs(
    runs: Sequence[Mapping[str, Any]],
    dataset: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    probabilities = [np.asarray(run["probabilities"], dtype=float) for run in runs]
    predictive = [np.asarray(run["predictive_geometry"], dtype=float) for run in runs]
    filtered = [np.asarray(run["filtered_geometry"], dtype=float) for run in runs]
    switches = [np.asarray(run["filtered_switch"], dtype=float) for run in runs]
    predictive_joints = [
        np.asarray(run["predictive_orientation_joint"], dtype=float) for run in runs
    ]
    filtered_joints = [
        np.asarray(run["filtered_orientation_joint"], dtype=float) for run in runs
    ]
    mean_probability = np.mean(np.stack(probabilities), axis=0)
    mean_predictive = np.mean(np.stack(predictive), axis=0)
    mean_filtered = np.mean(np.stack(filtered), axis=0)
    mean_switch = np.mean(np.stack(switches), axis=0)
    mean_predictive_joint = np.mean(np.stack(predictive_joints), axis=0)
    mean_filtered_joint = np.mean(np.stack(filtered_joints), axis=0)
    geometry = np.asarray(dataset["truth"]["geometry"], dtype=int)
    switch_truth = np.asarray(dataset["truth"]["switch"], dtype=float)
    rows = np.arange(geometry.size, dtype=int)
    switch_brier, switch_log_loss = _binary_scores(mean_switch, switch_truth)
    predictive_orientation = _orientation_scores(
        mean_predictive_joint,
        geometry,
        np.asarray(dataset["truth"]["orientation"], dtype=float)[rows, geometry],
    )
    filtered_orientation = _orientation_scores(
        mean_filtered_joint,
        geometry,
        np.asarray(dataset["truth"]["orientation_post"], dtype=float)[rows, geometry],
    )
    individual = [_run_metrics(run, dataset) for run in runs]
    row = {
        "dataset_id": str(dataset["dataset_id"]),
        "subject_id": int(dataset["subject_id"]),
        "replicate": int(dataset["replicate"]),
        "mode": str(runs[0]["mode"]),
        "particle_count": int(runs[0]["particle_count"]),
        "filter_seed_repeats": int(len(runs)),
        "choice_nll": _choice_nll(mean_probability, np.asarray(dataset["choices"])),
        "pf_seed_choice_probability_rmse": _mean_pairwise(
            probabilities, _probability_rmse
        ),
        "pf_seed_predictive_geometry_js": _mean_pairwise(predictive, _mean_js),
        "pf_seed_filtered_geometry_js": _mean_pairwise(filtered, _mean_js),
        "pf_seed_predictive_geometry_orientation_js": _mean_pairwise(
            [value.reshape(value.shape[0], -1) for value in predictive_joints],
            _mean_js,
        ),
        "pf_seed_filtered_geometry_orientation_js": _mean_pairwise(
            [value.reshape(value.shape[0], -1) for value in filtered_joints],
            _mean_js,
        ),
        "pf_seed_filtered_switch_rmse": _mean_pairwise(switches, _rmse),
        "predictive_geometry_map_accuracy": float(
            np.mean(np.argmax(mean_predictive, axis=1) == geometry)
        ),
        "filtered_geometry_map_accuracy": float(
            np.mean(np.argmax(mean_filtered, axis=1) == geometry)
        ),
        "predictive_geometry_true_mass": float(
            np.mean(mean_predictive[rows, geometry])
        ),
        "filtered_geometry_true_mass": float(
            np.mean(mean_filtered[rows, geometry])
        ),
        "predictive_geometry_state_nll": float(
            np.mean(-np.log(np.clip(mean_predictive[rows, geometry], 1e-12, 1.0)))
        ),
        "filtered_geometry_state_nll": float(
            np.mean(-np.log(np.clip(mean_filtered[rows, geometry], 1e-12, 1.0)))
        ),
        "predictive_true_geometry_support": float(
            np.mean(mean_predictive[rows, geometry] > 1e-12)
        ),
        "filtered_true_geometry_support": float(
            np.mean(mean_filtered[rows, geometry] > 1e-12)
        ),
        "predictive_orientation_true_geometry_coverage": predictive_orientation[0],
        "predictive_orientation_rmse_given_true_geometry": predictive_orientation[1],
        "predictive_orientation_mae_given_true_geometry": predictive_orientation[2],
        "filtered_orientation_true_geometry_coverage": filtered_orientation[0],
        "filtered_orientation_rmse_given_true_geometry": filtered_orientation[1],
        "filtered_orientation_mae_given_true_geometry": filtered_orientation[2],
        "filtered_expected_switch_count": float(np.sum(mean_switch[1:])),
        "true_switch_count": int(np.sum(switch_truth[1:])),
        "switch_brier": switch_brier,
        "switch_log_loss": switch_log_loss,
        "map_switch_count": int(
            np.sum(
                np.argmax(mean_filtered, axis=1)[1:]
                != np.argmax(mean_filtered, axis=1)[:-1]
            )
        ),
    }
    for key in (
        "median_ess_fraction",
        "resample_fraction",
        "initial_ancestral_lineage_fraction",
        "minimum_ancestral_lineage_fraction",
        "ancestral_true_geometry_support",
        "ancestral_weighted_geometry_accuracy",
        "ancestral_best_geometry_accuracy",
        "true_path_choice_nll",
        "ancestral_weighted_choice_nll",
        "ancestral_median_choice_nll",
        "true_path_better_than_ancestral_fraction",
        "ancestral_terminal_expected_switch_count",
    ):
        row[key] = float(np.mean([float(item[key]) for item in individual]))
    return row, {
        "probabilities": mean_probability,
        "predictive_geometry": mean_predictive,
        "filtered_geometry": mean_filtered,
        "predictive_orientation_joint": mean_predictive_joint,
        "filtered_orientation_joint": mean_filtered_joint,
        "filtered_switch": mean_switch,
    }


def _convergence_rows(
    aggregate_rows: Sequence[Mapping[str, Any]],
    aggregate_arrays: Mapping[tuple[str, str, int], Mapping[str, np.ndarray]],
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["dataset_id"]), str(row["mode"]), int(row["particle_count"])): row
        for row in aggregate_rows
    }
    rows: list[dict[str, Any]] = []
    dataset_ids = sorted({str(row["dataset_id"]) for row in aggregate_rows})
    for dataset_id in dataset_ids:
        counts = sorted(
            int(row["particle_count"])
            for row in aggregate_rows
            if str(row["dataset_id"]) == dataset_id
            and str(row["mode"]) == "baseline"
        )
        for lower, upper in zip(counts[:-1], counts[1:]):
            low_row = lookup[(dataset_id, "baseline", lower)]
            high_row = lookup[(dataset_id, "baseline", upper)]
            low = aggregate_arrays[(dataset_id, "baseline", lower)]
            high = aggregate_arrays[(dataset_id, "baseline", upper)]
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "subject_id": int(high_row["subject_id"]),
                    "lower_particle_count": int(lower),
                    "upper_particle_count": int(upper),
                    "choice_probability_rmse": _probability_rmse(
                        low["probabilities"], high["probabilities"]
                    ),
                    "predictive_geometry_js": _mean_js(
                        low["predictive_geometry"], high["predictive_geometry"]
                    ),
                    "filtered_geometry_js": _mean_js(
                        low["filtered_geometry"], high["filtered_geometry"]
                    ),
                    "predictive_geometry_orientation_js": _mean_js(
                        low["predictive_orientation_joint"].reshape(
                            low["predictive_orientation_joint"].shape[0], -1
                        ),
                        high["predictive_orientation_joint"].reshape(
                            high["predictive_orientation_joint"].shape[0], -1
                        ),
                    ),
                    "filtered_geometry_orientation_js": _mean_js(
                        low["filtered_orientation_joint"].reshape(
                            low["filtered_orientation_joint"].shape[0], -1
                        ),
                        high["filtered_orientation_joint"].reshape(
                            high["filtered_orientation_joint"].shape[0], -1
                        ),
                    ),
                    "filtered_switch_rmse": _rmse(
                        low["filtered_switch"], high["filtered_switch"]
                    ),
                    "choice_nll_change": float(
                        high_row["choice_nll"] - low_row["choice_nll"]
                    ),
                    "filtered_true_mass_change": float(
                        high_row["filtered_geometry_true_mass"]
                        - low_row["filtered_geometry_true_mass"]
                    ),
                    "filtered_map_accuracy_change": float(
                        high_row["filtered_geometry_map_accuracy"]
                        - low_row["filtered_geometry_map_accuracy"]
                    ),
                    "expected_switch_count_change": float(
                        high_row["filtered_expected_switch_count"]
                        - low_row["filtered_expected_switch_count"]
                    ),
                }
            )
    return rows


def _decision_payload(
    *,
    aggregate_frame: pd.DataFrame,
    convergence_frame: pd.DataFrame,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = aggregate_frame[aggregate_frame["mode"].eq("baseline")]
    counts = sorted(baseline["particle_count"].astype(int).unique().tolist())
    if len(counts) < 2:
        return {
            "status": "insufficient_particle_count_levels",
            "next_action": "Run at least two baseline particle counts.",
        }
    previous, highest = counts[-2], counts[-1]
    largest_step = convergence_frame[
        convergence_frame["lower_particle_count"].eq(previous)
        & convergence_frame["upper_particle_count"].eq(highest)
    ].copy()
    highest_rows = baseline[baseline["particle_count"].eq(highest)].copy()
    gate_rows: list[dict[str, Any]] = []
    for _, convergence in largest_step.iterrows():
        high = highest_rows[
            highest_rows["dataset_id"].eq(convergence["dataset_id"])
        ].iloc[0]
        checks = {
            "choice_probability_rmse": (
                float(convergence["choice_probability_rmse"]),
                float(thresholds["choice_probability_rmse_max"]),
            ),
            "predictive_geometry_js": (
                float(convergence["predictive_geometry_js"]),
                float(thresholds["predictive_geometry_js_max"]),
            ),
            "filtered_geometry_js": (
                float(convergence["filtered_geometry_js"]),
                float(thresholds["filtered_geometry_js_max"]),
            ),
            "predictive_geometry_orientation_js": (
                float(convergence["predictive_geometry_orientation_js"]),
                float(thresholds["predictive_geometry_orientation_js_max"]),
            ),
            "filtered_geometry_orientation_js": (
                float(convergence["filtered_geometry_orientation_js"]),
                float(thresholds["filtered_geometry_orientation_js_max"]),
            ),
            "filtered_true_mass_abs_change": (
                abs(float(convergence["filtered_true_mass_change"])),
                float(thresholds["filtered_true_mass_abs_change_max"]),
            ),
            "expected_switch_count_abs_change": (
                abs(float(convergence["expected_switch_count_change"])),
                float(thresholds["expected_switch_count_abs_change_max"]),
            ),
            "highest_count_seed_choice_rmse": (
                float(high["pf_seed_choice_probability_rmse"]),
                float(thresholds["highest_count_seed_choice_rmse_max"]),
            ),
            "highest_count_seed_filtered_geometry_js": (
                float(high["pf_seed_filtered_geometry_js"]),
                float(thresholds["highest_count_seed_filtered_geometry_js_max"]),
            ),
            "highest_count_seed_filtered_geometry_orientation_js": (
                float(high["pf_seed_filtered_geometry_orientation_js"]),
                float(
                    thresholds[
                        "highest_count_seed_filtered_geometry_orientation_js_max"
                    ]
                ),
            ),
        }
        failed = [name for name, (value, limit) in checks.items() if value > limit]
        gate_rows.append(
            {
                "dataset_id": str(convergence["dataset_id"]),
                "subject_id": int(convergence["subject_id"]),
                "passed": not failed,
                "failed_gates": failed,
                "values_and_limits": {
                    name: {"value": value, "maximum": limit}
                    for name, (value, limit) in checks.items()
                },
            }
        )
    stable = bool(gate_rows) and all(row["passed"] for row in gate_rows)
    oracle = aggregate_frame[
        aggregate_frame["mode"].eq("orientation_oracle")
        & aggregate_frame["particle_count"].eq(highest)
    ]
    oracle_effects: list[dict[str, Any]] = []
    for _, base in highest_rows.iterrows():
        match = oracle[oracle["dataset_id"].eq(base["dataset_id"])]
        if match.empty:
            continue
        conditioned = match.iloc[0]
        oracle_effects.append(
            {
                "dataset_id": str(base["dataset_id"]),
                "subject_id": int(base["subject_id"]),
                "filtered_true_mass_gain": float(
                    conditioned["filtered_geometry_true_mass"]
                    - base["filtered_geometry_true_mass"]
                ),
                "filtered_map_accuracy_gain": float(
                    conditioned["filtered_geometry_map_accuracy"]
                    - base["filtered_geometry_map_accuracy"]
                ),
                "filtered_state_nll_drop": float(
                    base["filtered_geometry_state_nll"]
                    - conditioned["filtered_geometry_state_nll"]
                ),
                "choice_nll_change": float(
                    conditioned["choice_nll"] - base["choice_nll"]
                ),
            }
        )
    mean_effect = {
        key: (
            float(np.mean([row[key] for row in oracle_effects]))
            if oracle_effects
            else float("nan")
        )
        for key in (
            "filtered_true_mass_gain",
            "filtered_map_accuracy_gain",
            "filtered_state_nll_drop",
            "choice_nll_change",
        )
    }
    rescue = bool(oracle_effects) and (
        mean_effect["filtered_true_mass_gain"]
        >= float(thresholds["orientation_rescue_true_mass_gain_min"])
        or mean_effect["filtered_map_accuracy_gain"]
        >= float(thresholds["orientation_rescue_map_accuracy_gain_min"])
        or mean_effect["filtered_state_nll_drop"]
        >= float(thresholds["orientation_rescue_state_nll_drop_min"])
    )
    oracle_was_run = bool(oracle_effects)
    if not stable:
        escalation_limit = int(
            thresholds.get("maximum_escalation_particle_count", 1024)
        )
        if highest < escalation_limit:
            next_count = min(int(highest * 2), escalation_limit)
            status = "pf_not_converged"
            next_action = (
                "Refilter the same frozen trajectories at "
                f"R={next_count}; do not interpret oracle effects until the "
                "frozen convergence gates pass."
            )
        else:
            status = "pf_not_converged_at_escalation_limit"
            next_action = (
                f"Do not brute-force beyond the declared R={escalation_limit} "
                "limit. Reconsider latent-state claims and only then evaluate "
                "a better proposal/filtering method as a separate model change."
            )
    elif not oracle_was_run:
        status = "stable_orientation_oracle_not_run"
        next_action = (
            "Run the true-orientation O1 diagnostic at the highest stable "
            "particle count before interpreting the recovery failure."
        )
    elif rescue:
        status = "stable_with_orientation_rescue"
        next_action = (
            "Treat geometry/orientation confounding as material; inspect the "
            "oracle trajectory differences before adding a deeper state oracle."
        )
    else:
        status = "stable_without_orientation_rescue"
        next_action = (
            "Orientation uncertainty is not the primary recovery bottleneck; "
            "proceed to the true-workspace O2 diagnostic on the same trajectories."
        )
    return {
        "status": status,
        "particle_count_step_evaluated": [int(previous), int(highest)],
        "all_subjects_passed_convergence": stable,
        "convergence_gates": gate_rows,
        "orientation_oracle_effects": oracle_effects,
        "mean_orientation_oracle_effect": mean_effect,
        "orientation_rescue_threshold_crossed": rescue,
        "next_action": next_action,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    resolved = deepcopy(config)
    if args.smoke:
        resolved.update(
            {
                "subjects": [int(config["subjects"][0])],
                "max_trials": 12,
                "synthetic_repeats": 1,
                "particle_counts": [4, 8],
                "oracle_particle_counts": [8],
                "filter_seed_repeats": 2,
                "n_jobs": min(2, int(config.get("n_jobs", 1))),
            }
        )
    if args.particle_counts:
        counts = sorted({int(value) for value in args.particle_counts})
        resolved["particle_counts"] = counts
        resolved["oracle_particle_counts"] = [counts[-1]]
    if args.n_jobs is not None:
        resolved["n_jobs"] = int(args.n_jobs)
    counts = sorted({int(value) for value in resolved["particle_counts"]})
    oracle_counts = sorted(
        {int(value) for value in resolved.get("oracle_particle_counts", [counts[-1]])}
    )
    if not counts or min(counts) < 2:
        raise ValueError("particle_counts must contain values of at least 2.")
    if int(resolved["filter_seed_repeats"]) < 2:
        raise ValueError("At least two PF seeds are required.")
    simulation_config_path = _repo_path(
        resolved["simulation_config_path"], base=config_path.parent
    )
    simulation_config = load_yaml(simulation_config_path)
    model_path = _repo_path(
        resolved["model"]["engine_config_path"], base=config_path.parent
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _repo_path(resolved["output_dir"], base=config_path.parent)
    )
    if args.smoke and args.output_dir is None:
        output_dir = output_dir.parent / f"{output_dir.name}_smoke"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Refusing to overwrite non-empty result directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "fit_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    reference_cache_dir = (
        args.reference_cache_dir.resolve()
        if args.reference_cache_dir is not None
        else None
    )
    if reference_cache_dir is not None:
        if not reference_cache_dir.is_dir():
            raise FileNotFoundError(
                f"Reference cache directory does not exist: {reference_cache_dir}"
            )
        reference_manifest_path = reference_cache_dir.parent / "run_manifest.json"
        if not reference_manifest_path.exists():
            raise FileNotFoundError(
                "Reference cache must have a sibling run_manifest.json."
            )
        reference_manifest = json.loads(
            reference_manifest_path.read_text(encoding="utf-8")
        )
        reference_config = reference_manifest["resolved_run_config"]
        compatibility = {
            "base_seed": int(reference_config["base_seed"])
            == int(resolved["base_seed"]),
            "max_trials": int(reference_config["max_trials"])
            == int(resolved["max_trials"]),
            "subjects": [int(value) for value in reference_config["subjects"]]
            == [int(value) for value in resolved["subjects"]],
            "model_path": Path(reference_manifest["model_path"]).resolve()
            == model_path.resolve(),
        }
        failed = [name for name, passed in compatibility.items() if not passed]
        if failed:
            raise ValueError(
                "Reference cache is incompatible for: " + ", ".join(failed)
            )

    subjects = [int(value) for value in resolved["subjects"]]
    engine_configs = {
        subject_id: _resolve_engine_for_model(
            model_path=model_path,
            simulation_config=simulation_config,
            simulation_config_path=simulation_config_path,
            subject_id=subject_id,
        )
        for subject_id in subjects
    }
    datasets = _build_datasets(
        subjects=subjects,
        engine_configs=engine_configs,
        simulation_config=simulation_config,
        simulation_config_path=simulation_config_path,
        max_trials=int(resolved["max_trials"]),
        synthetic_repeats=int(resolved["synthetic_repeats"]),
        base_seed=int(resolved["base_seed"]),
        readout_power=float(resolved["choice_readout_power"]),
        output_lapse=float(resolved["output_lapse"]),
    )
    dataset_lookup = {str(dataset["dataset_id"]): dataset for dataset in datasets}
    task_modes = MODES if args.phase == "all" else (str(args.phase),)
    tasks: list[dict[str, Any]] = []
    for dataset in datasets:
        for mode in task_modes:
            mode_counts = counts if mode == "baseline" else oracle_counts
            for particle_count in mode_counts:
                for seed_index in range(int(resolved["filter_seed_repeats"])):
                    filter_seed = stable_seed(
                        {
                            "seed_role": "model0815_p1_state_identifiability_filter",
                            "base_seed": int(resolved["base_seed"]),
                            "dataset_id": str(dataset["dataset_id"]),
                            "seed_index": int(seed_index),
                        }
                    )
                    tasks.append(
                        {
                            "dataset": dataset,
                            "mode": mode,
                            "particle_count": int(particle_count),
                            "seed_index": int(seed_index),
                            "filter_seed": int(filter_seed),
                            "cache_path": _cache_path(
                                cache_dir,
                                dataset_id=str(dataset["dataset_id"]),
                                mode=mode,
                                particle_count=int(particle_count),
                                seed_index=int(seed_index),
                            ),
                            "reference_cache_path": (
                                None
                                if reference_cache_dir is None
                                else _cache_path(
                                    reference_cache_dir,
                                    dataset_id=str(dataset["dataset_id"]),
                                    mode=mode,
                                    particle_count=int(particle_count),
                                    seed_index=int(seed_index),
                                )
                            ),
                        }
                    )
    _write_json(
        output_dir / "run_manifest.json",
        {
            "config_path": str(config_path),
            "simulation_config_path": str(simulation_config_path),
            "model_path": str(model_path),
            "resolved_run_config": resolved,
            "phase": str(args.phase),
            "smoke": bool(args.smoke),
            "dataset_ids": sorted(dataset_lookup),
            "pf_fit_count": int(len(tasks)),
            "cache_resume_enabled": bool(args.resume),
            "reference_cache_dir": (
                None
                if reference_cache_dir is None
                else str(reference_cache_dir)
            ),
        },
    )
    print(
        f"Running {len(tasks)} PF fits on {len(datasets)} fixed M1 trajectories...",
        flush=True,
    )
    raw_runs = Parallel(n_jobs=int(resolved["n_jobs"]))(
        delayed(_fit_and_cache)(
            dataset=task["dataset"],
            engine_config=engine_configs[int(task["dataset"]["subject_id"])],
            mode=str(task["mode"]),
            particle_count=int(task["particle_count"]),
            seed_index=int(task["seed_index"]),
            filter_seed=int(task["filter_seed"]),
            readout_power=float(resolved["choice_readout_power"]),
            output_lapse=float(resolved["output_lapse"]),
            resample_threshold_fraction=float(
                resolved["resample_threshold_fraction"]
            ),
            cache_path=task["cache_path"],
            reference_cache_path=task["reference_cache_path"],
            resume=bool(args.resume),
        )
        for task in tasks
    )

    run_rows = [
        _run_metrics(run, dataset_lookup[str(run["dataset_id"])])
        for run in raw_runs
    ]
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for run in raw_runs:
        key = (
            str(run["dataset_id"]),
            str(run["mode"]),
            int(run["particle_count"]),
        )
        grouped.setdefault(key, []).append(run)
    aggregate_rows: list[dict[str, Any]] = []
    aggregate_arrays: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    trajectory_store: dict[str, np.ndarray] = {}
    for dataset_id, dataset in dataset_lookup.items():
        prefix = str(dataset_id)
        trajectory_store[f"{prefix}__choices"] = np.asarray(dataset["choices"])
        trajectory_store[f"{prefix}__feedback"] = np.asarray(dataset["feedback"])
        for truth_key, values in dataset["truth"].items():
            trajectory_store[f"{prefix}__truth_{truth_key}"] = np.asarray(values)
    for key, runs in sorted(grouped.items()):
        runs = sorted(runs, key=lambda item: int(item["seed_index"]))
        dataset = dataset_lookup[key[0]]
        row, arrays = _aggregate_runs(runs, dataset)
        aggregate_rows.append(row)
        aggregate_arrays[key] = arrays
        prefix = f"{key[0]}__{key[1]}__R{key[2]}"
        for array_key, values in arrays.items():
            trajectory_store[f"{prefix}__{array_key}"] = np.asarray(values)

    convergence_rows = _convergence_rows(aggregate_rows, aggregate_arrays)
    run_frame = pd.DataFrame(run_rows)
    aggregate_frame = pd.DataFrame(aggregate_rows)
    convergence_frame = pd.DataFrame(convergence_rows)
    run_frame.to_csv(output_dir / "pf_seed_diagnostics.csv", index=False)
    aggregate_frame.to_csv(output_dir / "aggregate_recovery.csv", index=False)
    convergence_frame.to_csv(output_dir / "particle_convergence.csv", index=False)
    np.savez_compressed(
        output_dir / "aggregated_trajectories.npz", **trajectory_store
    )
    decision = _decision_payload(
        aggregate_frame=aggregate_frame,
        convergence_frame=convergence_frame,
        thresholds=resolved["decision_thresholds"],
    )
    _write_json(output_dir / "decision.json", decision)
    _write_json(
        output_dir / "summary.json",
        {
            "scientific_status": (
                "engineering_smoke_only" if args.smoke else "p1_identifiability_round1"
            ),
            "interpretation_warning": (
                "The orientation oracle conditions on generated latent state and is "
                "a diagnostic upper-control, not a candidate fitted model. Terminal "
                "ancestry is a genealogy diagnostic, not FFBSi/PGAS smoothing."
            ),
            "design": {
                "subjects": subjects,
                "trial_count": int(resolved["max_trials"]),
                "particle_counts": counts,
                "oracle_particle_counts": oracle_counts,
                "filter_seed_repeats": int(resolved["filter_seed_repeats"]),
                "common_filter_seeds_across_counts_and_oracle": True,
            },
            "decision": decision,
            "aggregate_recovery": aggregate_rows,
            "particle_convergence": convergence_rows,
        },
    )
    print(json.dumps(_json_safe(decision), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved diagnostics to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
