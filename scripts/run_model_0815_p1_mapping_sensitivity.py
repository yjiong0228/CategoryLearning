#!/usr/bin/env python3
"""Run the paired Model 0815 M0/M1 mapping sensitivity diagnostic.

M0 uses the historical fixed category labels.  M1 changes only one thing: it
analytically marginalizes a binary label orientation for each active geometry.
The program evaluates real behavior and bidirectionally generated trajectories
with paired PF seeds.  PF repeats are probability-averaged before scoring.
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

from src.Bayesian_state.inference.backends.particle_filter import (  # noqa: E402
    run_state_model_particle_filter,
)
from src.Bayesian_state.simulation.autonomous import (  # noqa: E402
    run_autonomous_category_learning,
)
from src.Bayesian_state.simulation.config import (  # noqa: E402
    load_yaml,
    resolve_engine_config,
)
from src.Bayesian_state.simulation.parameters import (  # noqa: E402
    apply_fixed_hyperparams_to_engine_config,
    infer_fixed_hyperparams_from_engine_config,
)
from src.Bayesian_state.simulation.runner import (  # noqa: E402
    StateModelSimulationRunner,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.utils.seeding import stable_seed  # noqa: E402
from src.Bayesian_state.utils.subjects import resolve_subject_config  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs/specific_models/model_0815_p1_mapping_sensitivity.yaml"
)
MODEL_IDS = ("m0", "m1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Use one subject, 16 trials, 4 particles, one synthetic repeat, "
            "and two PF seeds."
        ),
    )
    return parser.parse_args()


def _repo_path(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


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


def _mean_or_none(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return None if finite.size == 0 else float(np.mean(finite))


def _choice_nll(
    probabilities: np.ndarray,
    choices: np.ndarray,
    *,
    stop: int | None = None,
) -> float:
    probability = np.asarray(probabilities, dtype=float)
    observed = np.asarray(choices, dtype=int).reshape(-1)
    usable = observed.size if stop is None else min(observed.size, int(stop))
    observed = observed[:usable]
    probability = probability[:usable]
    valid = (
        (observed >= 1)
        & (observed <= probability.shape[1])
        & np.all(np.isfinite(probability), axis=1)
    )
    if not np.any(valid):
        return float("nan")
    rows = np.flatnonzero(valid)
    selected = probability[rows, observed[rows] - 1]
    return float(np.mean(-np.log(np.clip(selected, 1e-12, 1.0))))


def _probability_rmse(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.shape != right.shape:
        raise ValueError("Probability arrays must have equal shapes.")
    keep = np.isfinite(left) & np.isfinite(right)
    return (
        float("nan")
        if not np.any(keep)
        else float(np.sqrt(np.mean(np.square(left[keep] - right[keep]))))
    )


def _mean_js(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("State distributions must have equal 2-D shapes.")
    keep = np.all(np.isfinite(left), axis=1) & np.all(np.isfinite(right), axis=1)
    if not np.any(keep):
        return float("nan")
    left = np.clip(left[keep], 0.0, None)
    right = np.clip(right[keep], 0.0, None)
    left_sum = np.sum(left, axis=1, keepdims=True)
    right_sum = np.sum(right, axis=1, keepdims=True)
    valid = (left_sum[:, 0] > 0.0) & (right_sum[:, 0] > 0.0)
    if not np.any(valid):
        return float("nan")
    left = left[valid] / left_sum[valid]
    right = right[valid] / right_sum[valid]
    midpoint = 0.5 * (left + right)

    def kl(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        positive = values > 0.0
        terms = np.zeros_like(values)
        terms[positive] = values[positive] * np.log(
            values[positive] / np.clip(reference[positive], 1e-12, None)
        )
        return np.sum(terms, axis=1)

    return float(np.mean(0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)))


def _mean_pairwise(
    arrays: Sequence[np.ndarray],
    metric: Any,
) -> float:
    values = [
        metric(arrays[first], arrays[second])
        for first, second in combinations(range(len(arrays)), 2)
    ]
    return float(np.mean(values)) if values else float("nan")


def _switch_count(probability: np.ndarray) -> int:
    state = np.argmax(np.asarray(probability, dtype=float), axis=1)
    return int(np.sum(state[1:] != state[:-1]))


def _effect_to_noise(effect: float, first_noise: float, second_noise: float) -> float:
    finite = [value for value in (first_noise, second_noise) if np.isfinite(value)]
    if not finite:
        return float("nan")
    denominator = max(finite)
    if denominator <= 0.0:
        return float("inf") if effect > 0.0 else 0.0
    return float(effect / denominator)


def _resolve_engine_for_model(
    *,
    model_path: Path,
    simulation_config: Mapping[str, Any],
    simulation_config_path: Path,
    subject_id: int,
) -> dict[str, Any]:
    subject_cfg = resolve_subject_config(simulation_config, int(subject_id))
    subject_cfg = deepcopy(subject_cfg)
    subject_cfg["engine_config_path"] = str(model_path)
    engine = resolve_engine_config(
        subject_cfg,
        simulation_config_path.parent,
        subject_id=int(subject_id),
    )
    fixed = {
        **infer_fixed_hyperparams_from_engine_config(engine),
        **dict(subject_cfg.get("fixed_hyperparams") or {}),
    }
    return apply_fixed_hyperparams_to_engine_config(engine, fixed)


def _load_subject_arrays(
    *,
    engine_config: Mapping[str, Any],
    subject_cfg: Mapping[str, Any],
    simulation_config_path: Path,
    subject_id: int,
    max_trials: int,
) -> tuple[Any, int, dict[str, Path]]:
    dataset_paths = resolve_dataset_paths(subject_cfg, simulation_config_path.parent)
    runner = StateModelSimulationRunner(
        engine_config=dict(engine_config),
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
        n_jobs=1,
    )
    runner.prepare_data(dataset_paths["learning_data"])
    frame = runner._get_subject_frame(int(subject_id), 1.0)
    condition = runner._get_condition_value(frame)
    if int(condition) != 1:
        raise ValueError(
            f"Mapping diagnostic requires condition 1, got {condition} for {subject_id}."
        )
    arrays = runner._extract_arrays(frame, int(max_trials))
    if arrays.categories is None:
        raise ValueError("Mapping recovery requires hard task categories.")
    return arrays, int(condition), dataset_paths


def _truth_from_generated(
    generated: Any,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    steps = list(generated.trajectory.step_log)
    geometry = np.asarray(
        [int(step["executed_hypothesis"]) for step in steps], dtype=int
    )
    predictive_values = [
        step.get("executed_orientation_probability") for step in steps
    ]
    filtered_values = [
        step.get("executed_orientation_probability_post") for step in steps
    ]
    if any(value is None for value in predictive_values):
        predictive = None
    else:
        predictive = np.asarray(predictive_values, dtype=float)
    if any(value is None for value in filtered_values):
        filtered = None
    else:
        filtered = np.asarray(filtered_values, dtype=float)
    return geometry, predictive, filtered


def _build_datasets(
    *,
    subjects: Sequence[int],
    model_configs: Mapping[str, Mapping[int, Mapping[str, Any]]],
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
            engine_config=model_configs["m0"][int(subject_id)],
            subject_cfg=subject_cfg,
            simulation_config_path=simulation_config_path,
            subject_id=int(subject_id),
            max_trials=int(max_trials),
        )
        common = {
            "subject_id": int(subject_id),
            "condition": int(condition),
            "stimulus": np.asarray(arrays.stimulus, dtype=float),
            "categories": np.asarray(arrays.categories, dtype=int),
            "dataset_paths": dataset_paths,
        }
        datasets.append(
            {
                **common,
                "dataset_id": f"observed_s{int(subject_id)}",
                "paired_dataset_key": f"observed_s{int(subject_id)}",
                "source": "observed",
                "generator_model": "observed",
                "replicate": -1,
                "choices": np.asarray(arrays.choices, dtype=int),
                "feedback": np.asarray(arrays.feedback, dtype=float),
                "true_geometry": None,
                "true_orientation_probability": None,
                "true_orientation_probability_post": None,
            }
        )
        for generator_model in MODEL_IDS:
            for replicate in range(int(synthetic_repeats)):
                generation_seed = stable_seed(
                    {
                        "seed_role": "model0815_p1_mapping_generation",
                        "base_seed": int(base_seed),
                        "subject_id": int(subject_id),
                        "replicate": int(replicate),
                    }
                )
                generated = run_autonomous_category_learning(
                    engine_config=model_configs[generator_model][int(subject_id)],
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
                geometry, orientation, orientation_post = _truth_from_generated(
                    generated
                )
                datasets.append(
                    {
                        **common,
                        "dataset_id": (
                            f"synthetic_{generator_model}_s{int(subject_id)}_"
                            f"r{int(replicate):02d}"
                        ),
                        "paired_dataset_key": (
                            f"synthetic_s{int(subject_id)}_r{int(replicate):02d}"
                        ),
                        "source": "synthetic",
                        "generator_model": generator_model,
                        "replicate": int(replicate),
                        "generation_seed": int(generation_seed),
                        "choices": generated.trajectory.choices.copy(),
                        "feedback": generated.trajectory.feedback.copy(),
                        "true_geometry": geometry,
                        "true_orientation_probability": orientation,
                        "true_orientation_probability_post": orientation_post,
                        "generated_accuracy": float(
                            np.mean(generated.trajectory.feedback)
                        ),
                    }
                )
    return datasets


def _fit_once(
    *,
    dataset: Mapping[str, Any],
    fit_model: str,
    engine_config: Mapping[str, Any],
    particle_count: int,
    readout_power: float,
    output_lapse: float,
    resample_threshold_fraction: float,
    filter_seed: int,
    seed_index: int,
) -> dict[str, Any]:
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
        processed_data_dir=dataset["dataset_paths"]["processed_dir"],
        dataset_paths=dataset["dataset_paths"],
    )
    predictive_executed = np.asarray(
        fitted.state_probabilities["executed_probability"], dtype=float
    )
    filtered_executed = np.asarray(
        fitted.state_probabilities["filtered_executed_probability"], dtype=float
    )
    predictive_orientation_joint = fitted.state_probabilities.get(
        "executed_orientation_joint"
    )
    filtered_orientation_joint = fitted.state_probabilities.get(
        "filtered_executed_orientation_joint"
    )
    return {
        "dataset_id": str(dataset["dataset_id"]),
        "fit_model": str(fit_model),
        "seed_index": int(seed_index),
        "filter_seed": int(filter_seed),
        "probabilities": np.asarray(fitted.marginal_probabilities, dtype=float),
        "predictive_executed_probability": predictive_executed,
        "filtered_executed_probability": filtered_executed,
        "predictive_orientation_joint": (
            None
            if predictive_orientation_joint is None
            else np.asarray(predictive_orientation_joint, dtype=float)
        ),
        "filtered_orientation_joint": (
            None
            if filtered_orientation_joint is None
            else np.asarray(filtered_orientation_joint, dtype=float)
        ),
        "median_ess_fraction": float(
            np.median(np.asarray(fitted.post_choice_ess, dtype=float))
            / float(particle_count)
        ),
    }


def _aggregate_fit(
    *,
    dataset: Mapping[str, Any],
    fit_model: str,
    runs: Sequence[Mapping[str, Any]],
    early_trials: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray | None]]:
    probabilities = [np.asarray(run["probabilities"], dtype=float) for run in runs]
    predictive_executed = [
        np.asarray(run["predictive_executed_probability"], dtype=float)
        for run in runs
    ]
    filtered_executed = [
        np.asarray(run["filtered_executed_probability"], dtype=float)
        for run in runs
    ]
    mean_probability = np.mean(np.stack(probabilities), axis=0)
    mean_predictive_executed = np.mean(np.stack(predictive_executed), axis=0)
    mean_filtered_executed = np.mean(np.stack(filtered_executed), axis=0)
    predictive_joint_runs = [run["predictive_orientation_joint"] for run in runs]
    filtered_joint_runs = [run["filtered_orientation_joint"] for run in runs]
    if all(value is not None for value in predictive_joint_runs):
        mean_predictive_joint: np.ndarray | None = np.mean(
            np.stack(
                [np.asarray(value, dtype=float) for value in predictive_joint_runs]
            ),
            axis=0,
        )
    else:
        mean_predictive_joint = None
    if all(value is not None for value in filtered_joint_runs):
        mean_filtered_joint: np.ndarray | None = np.mean(
            np.stack(
                [np.asarray(value, dtype=float) for value in filtered_joint_runs]
            ),
            axis=0,
        )
    else:
        mean_filtered_joint = None
    choices = np.asarray(dataset["choices"], dtype=int)
    row: dict[str, Any] = {
        "dataset_id": str(dataset["dataset_id"]),
        "source": str(dataset["source"]),
        "generator_model": str(dataset["generator_model"]),
        "subject_id": int(dataset["subject_id"]),
        "replicate": int(dataset["replicate"]),
        "fit_model": str(fit_model),
        "trial_count": int(choices.size),
        "choice_nll": _choice_nll(mean_probability, choices),
        "early_choice_nll": _choice_nll(
            mean_probability, choices, stop=int(early_trials)
        ),
        "pf_seed_choice_probability_rmse": _mean_pairwise(
            probabilities, _probability_rmse
        ),
        "pf_seed_predictive_geometry_js": _mean_pairwise(
            predictive_executed, _mean_js
        ),
        "pf_seed_filtered_geometry_js": _mean_pairwise(
            filtered_executed, _mean_js
        ),
        "median_ess_fraction": float(
            np.mean([float(run["median_ess_fraction"]) for run in runs])
        ),
        "predictive_map_switch_count": _switch_count(mean_predictive_executed),
        "map_switch_count": _switch_count(mean_filtered_executed),
    }
    truth = dataset.get("true_geometry")
    if truth is not None:
        true_geometry = np.asarray(truth, dtype=int)
        rows = np.arange(true_geometry.size)
        predictive_true_probability = mean_predictive_executed[rows, true_geometry]
        filtered_true_probability = mean_filtered_executed[rows, true_geometry]
        row.update(
            {
                "predictive_geometry_map_accuracy": float(
                    np.mean(
                        np.argmax(mean_predictive_executed, axis=1)
                        == true_geometry
                    )
                ),
                "predictive_geometry_true_state_probability": float(
                    np.mean(predictive_true_probability)
                ),
                "predictive_geometry_state_nll": float(
                    np.mean(
                        -np.log(
                            np.clip(predictive_true_probability, 1e-12, 1.0)
                        )
                    )
                ),
                "geometry_map_accuracy": float(
                    np.mean(
                        np.argmax(mean_filtered_executed, axis=1)
                        == true_geometry
                    )
                ),
                "geometry_true_state_probability": float(
                    np.mean(filtered_true_probability)
                ),
                "geometry_state_nll": float(
                    np.mean(
                        -np.log(
                            np.clip(filtered_true_probability, 1e-12, 1.0)
                        )
                    )
                ),
                "true_switch_count": int(
                    np.sum(true_geometry[1:] != true_geometry[:-1])
                ),
            }
        )
        row["excess_map_switch_count"] = int(
            row["map_switch_count"] - row["true_switch_count"]
        )
        for prefix, joint, truth_key in (
            (
                "predictive_",
                mean_predictive_joint,
                "true_orientation_probability",
            ),
            (
                "",
                mean_filtered_joint,
                "true_orientation_probability_post",
            ),
        ):
            truth_orientation = dataset.get(truth_key)
            if joint is None or truth_orientation is None:
                continue
            true_orientation = np.asarray(truth_orientation, dtype=float)
            geometry_mass = np.sum(joint[rows, true_geometry, :], axis=1)
            covered = geometry_mass > 1e-12
            estimated = np.divide(
                joint[rows, true_geometry, 0],
                geometry_mass,
                out=np.full_like(geometry_mass, np.nan),
                where=covered,
            )
            row[f"{prefix}orientation_true_geometry_coverage"] = float(
                np.mean(covered)
            )
            row[
                f"{prefix}orientation_probability_rmse_given_true_geometry"
            ] = (
                float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                estimated[covered] - true_orientation[covered]
                            )
                        )
                    )
                )
                if np.any(covered)
                else float("nan")
            )
            row[
                f"{prefix}orientation_probability_mae_given_true_geometry"
            ] = (
                float(
                    np.mean(
                        np.abs(estimated[covered] - true_orientation[covered])
                    )
                )
                if np.any(covered)
                else float("nan")
            )
    arrays = {
        "probabilities": mean_probability,
        "predictive_executed_probability": mean_predictive_executed,
        "filtered_executed_probability": mean_filtered_executed,
        "predictive_orientation_joint": mean_predictive_joint,
        "filtered_orientation_joint": mean_filtered_joint,
    }
    return row, arrays


def _pair_models(
    dataset: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    arrays: Mapping[str, Mapping[str, np.ndarray | None]],
) -> dict[str, Any]:
    m0 = rows["m0"]
    m1 = rows["m1"]
    probability_effect = _probability_rmse(
        np.asarray(arrays["m0"]["probabilities"]),
        np.asarray(arrays["m1"]["probabilities"]),
    )
    predictive_geometry_effect = _mean_js(
        np.asarray(arrays["m0"]["predictive_executed_probability"]),
        np.asarray(arrays["m1"]["predictive_executed_probability"]),
    )
    filtered_geometry_effect = _mean_js(
        np.asarray(arrays["m0"]["filtered_executed_probability"]),
        np.asarray(arrays["m1"]["filtered_executed_probability"]),
    )
    row = {
        "dataset_id": str(dataset["dataset_id"]),
        "source": str(dataset["source"]),
        "generator_model": str(dataset["generator_model"]),
        "subject_id": int(dataset["subject_id"]),
        "replicate": int(dataset["replicate"]),
        "delta_nll_m1_minus_m0": float(m1["choice_nll"] - m0["choice_nll"]),
        "delta_early_nll_m1_minus_m0": float(
            m1["early_choice_nll"] - m0["early_choice_nll"]
        ),
        "mapping_choice_probability_rmse": probability_effect,
        "mapping_predictive_geometry_js": predictive_geometry_effect,
        "mapping_filtered_geometry_js": filtered_geometry_effect,
        "mapping_executed_geometry_js": filtered_geometry_effect,
        "m0_map_switch_count": int(m0["map_switch_count"]),
        "m1_map_switch_count": int(m1["map_switch_count"]),
        "map_switch_count_difference_m1_minus_m0": int(
            m1["map_switch_count"] - m0["map_switch_count"]
        ),
        "mapping_probability_effect_to_pf_noise_ratio": _effect_to_noise(
            probability_effect,
            float(m0["pf_seed_choice_probability_rmse"]),
            float(m1["pf_seed_choice_probability_rmse"]),
        ),
        "mapping_geometry_effect_to_pf_noise_ratio": _effect_to_noise(
            filtered_geometry_effect,
            float(m0["pf_seed_filtered_geometry_js"]),
            float(m1["pf_seed_filtered_geometry_js"]),
        ),
        "mapping_predictive_geometry_effect_to_pf_noise_ratio": _effect_to_noise(
            predictive_geometry_effect,
            float(m0["pf_seed_predictive_geometry_js"]),
            float(m1["pf_seed_predictive_geometry_js"]),
        ),
    }
    if dataset["source"] == "synthetic":
        winner = "m1" if float(m1["choice_nll"]) < float(m0["choice_nll"]) else "m0"
        row.update(
            {
                "recovered_model": winner,
                "generator_recovered": bool(winner == dataset["generator_model"]),
                "absolute_nll_separation": abs(float(row["delta_nll_m1_minus_m0"])),
            }
        )
    return row


def _summary_payload(
    *,
    fit_frame: pd.DataFrame,
    pair_frame: pd.DataFrame,
    config: Mapping[str, Any],
    smoke: bool,
) -> dict[str, Any]:
    observed = pair_frame[pair_frame["source"].eq("observed")]
    synthetic = pair_frame[pair_frame["source"].eq("synthetic")]
    state = fit_frame[fit_frame["source"].eq("synthetic")]
    orientation_attempted = (
        state[state["orientation_true_geometry_coverage"].notna()]
        if "orientation_true_geometry_coverage" in state
        else state.iloc[0:0]
    )
    orientation_evaluable = (
        orientation_attempted[
            orientation_attempted[
                "orientation_probability_rmse_given_true_geometry"
            ].notna()
        ]
        if "orientation_probability_rmse_given_true_geometry"
        in orientation_attempted
        else orientation_attempted.iloc[0:0]
    )
    predictive_orientation_attempted = (
        state[state["predictive_orientation_true_geometry_coverage"].notna()]
        if "predictive_orientation_true_geometry_coverage" in state
        else state.iloc[0:0]
    )
    predictive_orientation_evaluable = (
        predictive_orientation_attempted[
            predictive_orientation_attempted[
                "predictive_orientation_probability_rmse_given_true_geometry"
            ].notna()
        ]
        if "predictive_orientation_probability_rmse_given_true_geometry"
        in predictive_orientation_attempted
        else predictive_orientation_attempted.iloc[0:0]
    )
    recovery_by_generator = []
    for generator, frame in synthetic.groupby("generator_model", sort=True):
        recovery_by_generator.append(
            {
                "generator_model": str(generator),
                "dataset_count": int(len(frame)),
                "recovery_accuracy": float(frame["generator_recovered"].mean()),
                "mean_delta_nll_m1_minus_m0": float(
                    frame["delta_nll_m1_minus_m0"].mean()
                ),
                "mean_early_delta_nll_m1_minus_m0": float(
                    frame["delta_early_nll_m1_minus_m0"].mean()
                ),
            }
        )
    return {
        "scientific_status": (
            "engineering_smoke_only" if smoke else "small_subject_p1_pilot"
        ),
        "decision_warning": (
            "Do not retain or reject M1 from this run alone. Inspect effect "
            "size, state recovery, and PF-noise ratios before freezing a larger panel."
        ),
        "recovery_scope": (
            "bidirectional fixed-parameter recovery; other cognitive parameters "
            "are shared and frozen, not re-optimized per synthetic dataset"
        ),
        "design": {
            "subjects": list(config["subjects"]),
            "max_trials": int(config["max_trials"]),
            "early_trials": int(config["early_trials"]),
            "synthetic_repeats": int(config["synthetic_repeats"]),
            "particle_count": int(config["filter_particle_count"]),
            "filter_seed_repeats": int(config["filter_seed_repeats"]),
            "paired_filter_seeds": True,
            "repeat_aggregation": "mean_probability_before_scoring",
        },
        "bidirectional_fixed_parameter_model_recovery": recovery_by_generator,
        "observed_predictive_comparison": {
            "subject_count": int(len(observed)),
            "mean_delta_nll_m1_minus_m0": _mean_or_none(
                observed["delta_nll_m1_minus_m0"].tolist()
            ),
            "mean_early_delta_nll_m1_minus_m0": _mean_or_none(
                observed["delta_early_nll_m1_minus_m0"].tolist()
            ),
        },
        "observed_trajectory_sensitivity": {
            "mean_predictive_executed_geometry_js": _mean_or_none(
                observed["mapping_predictive_geometry_js"].tolist()
            ),
            "mean_filtered_executed_geometry_js": _mean_or_none(
                observed["mapping_filtered_geometry_js"].tolist()
            ),
            "mean_absolute_map_switch_count_difference": _mean_or_none(
                np.abs(
                    observed["map_switch_count_difference_m1_minus_m0"].to_numpy(
                        dtype=float
                    )
                ).tolist()
            ),
        },
        "mapping_effect_vs_pf_seed_noise": {
            "observed_mean_probability_ratio": _mean_or_none(
                observed["mapping_probability_effect_to_pf_noise_ratio"].tolist()
            ),
            "observed_mean_geometry_ratio": _mean_or_none(
                observed["mapping_geometry_effect_to_pf_noise_ratio"].tolist()
            ),
            "observed_mean_predictive_geometry_ratio": _mean_or_none(
                observed[
                    "mapping_predictive_geometry_effect_to_pf_noise_ratio"
                ].tolist()
            ),
        },
        "synthetic_geometry_recovery": [
            {
                "generator_model": str(generator),
                "fit_model": str(fit_model),
                "dataset_count": int(len(frame)),
                "mean_predictive_map_accuracy": float(
                    frame["predictive_geometry_map_accuracy"].mean()
                ),
                "mean_map_accuracy": float(frame["geometry_map_accuracy"].mean()),
                "mean_predictive_true_state_probability": float(
                    frame["predictive_geometry_true_state_probability"].mean()
                ),
                "mean_true_state_probability": float(
                    frame["geometry_true_state_probability"].mean()
                ),
                "mean_excess_map_switch_count": float(
                    frame["excess_map_switch_count"].mean()
                ),
            }
            for (generator, fit_model), frame in state.groupby(
                ["generator_model", "fit_model"], sort=True
            )
        ],
        "synthetic_predictive_orientation_recovery": {
            "attempted_dataset_count": int(
                len(predictive_orientation_attempted)
            ),
            "evaluable_dataset_count": int(
                len(predictive_orientation_evaluable)
            ),
            "mean_true_geometry_coverage": _mean_or_none(
                predictive_orientation_attempted[
                    "predictive_orientation_true_geometry_coverage"
                ].tolist()
            ),
            "mean_probability_rmse_given_true_geometry": _mean_or_none(
                predictive_orientation_evaluable[
                    "predictive_orientation_probability_rmse_given_true_geometry"
                ].tolist()
            ),
            "mean_probability_mae_given_true_geometry": _mean_or_none(
                predictive_orientation_evaluable[
                    "predictive_orientation_probability_mae_given_true_geometry"
                ].tolist()
            ),
        },
        "synthetic_orientation_recovery": {
            "attempted_dataset_count": int(len(orientation_attempted)),
            "evaluable_dataset_count": int(len(orientation_evaluable)),
            "mean_true_geometry_coverage": _mean_or_none(
                orientation_attempted[
                    "orientation_true_geometry_coverage"
                ].tolist()
            ),
            "mean_probability_rmse_given_true_geometry": _mean_or_none(
                orientation_evaluable[
                    "orientation_probability_rmse_given_true_geometry"
                ].tolist()
            ),
            "mean_probability_mae_given_true_geometry": _mean_or_none(
                orientation_evaluable[
                    "orientation_probability_mae_given_true_geometry"
                ].tolist()
            ),
        },
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    simulation_config_path = _repo_path(
        config["simulation_config_path"], base=config_path.parent
    )
    simulation_config = load_yaml(simulation_config_path)
    model_paths = {
        model_id: _repo_path(
            config["models"][model_id]["engine_config_path"],
            base=config_path.parent,
        )
        for model_id in MODEL_IDS
    }
    resolved = deepcopy(config)
    if args.smoke:
        resolved.update(
            {
                "subjects": [int(config["subjects"][0])],
                "max_trials": 16,
                "early_trials": 8,
                "synthetic_repeats": 1,
                "filter_particle_count": 4,
                "filter_seed_repeats": 2,
                "n_jobs": min(2, int(config.get("n_jobs", 1))),
            }
        )
    if args.n_jobs is not None:
        resolved["n_jobs"] = int(args.n_jobs)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _repo_path(config["output_dir"], base=config_path.parent)
    )
    if args.smoke and args.output_dir is None:
        output_dir = output_dir.parent / f"{output_dir.name}_smoke"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty result directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = [int(value) for value in resolved["subjects"]]
    if int(resolved["filter_seed_repeats"]) < 2:
        raise ValueError("At least two PF seeds are required to estimate PF noise.")
    model_configs = {
        model_id: {
            subject_id: _resolve_engine_for_model(
                model_path=model_paths[model_id],
                simulation_config=simulation_config,
                simulation_config_path=simulation_config_path,
                subject_id=subject_id,
            )
            for subject_id in subjects
        }
        for model_id in MODEL_IDS
    }
    datasets = _build_datasets(
        subjects=subjects,
        model_configs=model_configs,
        simulation_config=simulation_config,
        simulation_config_path=simulation_config_path,
        max_trials=int(resolved["max_trials"]),
        synthetic_repeats=int(resolved["synthetic_repeats"]),
        base_seed=int(resolved["base_seed"]),
        readout_power=float(resolved["choice_readout_power"]),
        output_lapse=float(resolved["output_lapse"]),
    )

    tasks = []
    for dataset in datasets:
        paired_seeds = [
            stable_seed(
                {
                    "seed_role": "model0815_p1_mapping_paired_filter",
                    "base_seed": int(resolved["base_seed"]),
                    "paired_dataset_key": str(dataset["paired_dataset_key"]),
                    "seed_index": int(seed_index),
                    "particle_count": int(resolved["filter_particle_count"]),
                }
            )
            for seed_index in range(int(resolved["filter_seed_repeats"]))
        ]
        for fit_model in MODEL_IDS:
            for seed_index, filter_seed in enumerate(paired_seeds):
                tasks.append((dataset, fit_model, seed_index, filter_seed))

    print(
        f"Running {len(tasks)} paired PF fits for {len(datasets)} datasets "
        f"at R={int(resolved['filter_particle_count'])}...",
        flush=True,
    )
    raw_runs = Parallel(n_jobs=int(resolved["n_jobs"]))(
        delayed(_fit_once)(
            dataset=dataset,
            fit_model=fit_model,
            engine_config=model_configs[fit_model][int(dataset["subject_id"])],
            particle_count=int(resolved["filter_particle_count"]),
            readout_power=float(resolved["choice_readout_power"]),
            output_lapse=float(resolved["output_lapse"]),
            resample_threshold_fraction=float(
                resolved["resample_threshold_fraction"]
            ),
            filter_seed=int(filter_seed),
            seed_index=int(seed_index),
        )
        for dataset, fit_model, seed_index, filter_seed in tasks
    )

    run_lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for run in raw_runs:
        key = (str(run["dataset_id"]), str(run["fit_model"]))
        run_lookup.setdefault(key, []).append(run)
    fit_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    array_store: dict[str, np.ndarray] = {}
    for dataset in datasets:
        dataset_prefix = str(dataset["dataset_id"])
        array_store[f"{dataset_prefix}__choices"] = np.asarray(
            dataset["choices"], dtype=int
        )
        array_store[f"{dataset_prefix}__feedback"] = np.asarray(
            dataset["feedback"], dtype=float
        )
        if dataset.get("true_geometry") is not None:
            array_store[f"{dataset_prefix}__true_geometry"] = np.asarray(
                dataset["true_geometry"], dtype=int
            )
        if dataset.get("true_orientation_probability") is not None:
            array_store[
                f"{dataset_prefix}__true_orientation_probability"
            ] = np.asarray(
                dataset["true_orientation_probability"], dtype=float
            )
        if dataset.get("true_orientation_probability_post") is not None:
            array_store[
                f"{dataset_prefix}__true_orientation_probability_post"
            ] = np.asarray(
                dataset["true_orientation_probability_post"], dtype=float
            )
        dataset_rows: dict[str, dict[str, Any]] = {}
        dataset_arrays: dict[str, dict[str, np.ndarray | None]] = {}
        for fit_model in MODEL_IDS:
            runs = sorted(
                run_lookup[(str(dataset["dataset_id"]), fit_model)],
                key=lambda value: int(value["seed_index"]),
            )
            for run in runs:
                seed_rows.append(
                    {
                        "dataset_id": str(dataset["dataset_id"]),
                        "source": str(dataset["source"]),
                        "generator_model": str(dataset["generator_model"]),
                        "subject_id": int(dataset["subject_id"]),
                        "replicate": int(dataset["replicate"]),
                        "fit_model": fit_model,
                        "seed_index": int(run["seed_index"]),
                        "filter_seed": int(run["filter_seed"]),
                        "choice_nll": _choice_nll(
                            np.asarray(run["probabilities"]),
                            np.asarray(dataset["choices"]),
                        ),
                        "early_choice_nll": _choice_nll(
                            np.asarray(run["probabilities"]),
                            np.asarray(dataset["choices"]),
                            stop=int(resolved["early_trials"]),
                        ),
                        "median_ess_fraction": float(
                            run["median_ess_fraction"]
                        ),
                    }
                )
            row, arrays = _aggregate_fit(
                dataset=dataset,
                fit_model=fit_model,
                runs=runs,
                early_trials=int(resolved["early_trials"]),
            )
            fit_rows.append(row)
            dataset_rows[fit_model] = row
            dataset_arrays[fit_model] = arrays
            prefix = f"{dataset['dataset_id']}__{fit_model}"
            array_store[f"{prefix}__probabilities"] = np.asarray(
                arrays["probabilities"]
            )
            array_store[
                f"{prefix}__predictive_executed_probability"
            ] = np.asarray(arrays["predictive_executed_probability"])
            array_store[
                f"{prefix}__filtered_executed_probability"
            ] = np.asarray(arrays["filtered_executed_probability"])
            if arrays["predictive_orientation_joint"] is not None:
                array_store[
                    f"{prefix}__predictive_orientation_joint"
                ] = np.asarray(arrays["predictive_orientation_joint"])
            if arrays["filtered_orientation_joint"] is not None:
                array_store[
                    f"{prefix}__filtered_orientation_joint"
                ] = np.asarray(arrays["filtered_orientation_joint"])
        pair_rows.append(_pair_models(dataset, dataset_rows, dataset_arrays))

    fit_frame = pd.DataFrame(fit_rows)
    seed_frame = pd.DataFrame(seed_rows)
    pair_frame = pd.DataFrame(pair_rows)
    fit_frame.to_csv(output_dir / "fit_summary.csv", index=False)
    seed_frame.to_csv(output_dir / "pf_seed_diagnostics.csv", index=False)
    pair_frame.to_csv(output_dir / "paired_model_comparison.csv", index=False)
    np.savez_compressed(output_dir / "aggregated_trajectories.npz", **array_store)
    summary = _summary_payload(
        fit_frame=fit_frame,
        pair_frame=pair_frame,
        config=resolved,
        smoke=bool(args.smoke),
    )
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "run_manifest.json",
        {
            "config_path": str(config_path),
            "simulation_config_path": str(simulation_config_path),
            "model_paths": {key: str(value) for key, value in model_paths.items()},
            "resolved_run_config": resolved,
            "dataset_count": len(datasets),
            "pf_fit_count": len(tasks),
            "outputs": [
                "fit_summary.csv",
                "pf_seed_diagnostics.csv",
                "paired_model_comparison.csv",
                "aggregated_trajectories.npz",
                "summary.json",
            ],
        },
    )
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2), flush=True)
    print(f"Saved P1 mapping diagnostic to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
