#!/usr/bin/env python3
"""Recover and evaluate the zero-boundary shared-theta B0/D0 model.

The fitted transition strength is theta_s = z_s * theta_plus.  Membership
``z_s`` is subject-specific and exact, while every D0 subject in a cohort
shares one non-zero ``theta_plus``.  A small training-loss penalty for D0
membership is calibrated only on synthetic calibration cohorts, frozen, and
then applied to independent synthetic evaluation cohorts and real held-out
data.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cond1_newplan_factorial import (
    score_probabilities,
    theta_token,
    write_csv,
    write_json,
)
from scripts.run_cond1_newplan_particle_filter import params_for
from scripts.run_cond1_newplan_recovery import time_split_masks
from src.Bayesian_state.run_simulation import (
    apply_fixed_hyperparams_to_engine_config,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.utils.newplan_generation import (
    generate_condition1_trajectory,
)
from src.Bayesian_state.utils.newplan_particle_filter import (
    run_newplan_particle_filter,
)
from src.Bayesian_state.utils.newplan_shared_theta import (
    binary_recovery_metrics,
    choose_membership_penalty,
    select_shared_theta,
)
from src.Bayesian_state.utils.optimization_config import (
    DEFAULT_DATA_PATH,
    load_yaml,
)
from src.Bayesian_state.utils.optimizer_common import stable_seed
from src.Bayesian_state.utils.optimizer_simulation import (
    StateModelSimulationRunner,
)


DEFAULT_REAL_GRIDS = [
    ROOT
    / "results/zhuran/cond1_newplan/particle_filter_dev_r32_64_128"
    / "parameter_grid.csv",
    ROOT
    / "results/zhuran/cond1_newplan/particle_filter_dev_r64_seed2"
    / "parameter_grid.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=[103, 117, 131])
    parser.add_argument(
        "--positive-theta-grid",
        type=float,
        nargs="+",
        default=[0.25, 0.50, 0.75, 1.0],
    )
    parser.add_argument("--particle-counts", type=int, nargs="+", default=[64])
    parser.add_argument("--calibration-particle-count", type=int, default=64)
    parser.add_argument(
        "--evaluation-only-particle-counts",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Particle counts fitted only on frozen evaluation cohorts. "
            "Useful for a confirmation run after calibrating at a smaller R."
        ),
    )
    parser.add_argument("--cohort-replicates", type=int, default=4)
    parser.add_argument("--calibration-replicates", type=int, default=2)
    parser.add_argument(
        "--penalty-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08],
    )
    parser.add_argument("--target-specificity", type=float, default=0.90)
    parser.add_argument("--pass-sensitivity", type=float, default=0.80)
    parser.add_argument("--pass-specificity", type=float, default=0.90)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--resample-threshold", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.55)
    parser.add_argument("--w0", type=float, default=0.10)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--max-trials", type=int, default=128)
    parser.add_argument("--base-seed", type=int, default=20260821)
    parser.add_argument(
        "--real-grids",
        type=Path,
        nargs="*",
        default=DEFAULT_REAL_GRIDS,
    )
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_newplan/shared_theta_hurdle_recovery"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    args.subjects = [int(value) for value in args.subjects]
    if len(args.subjects) < 2 or len(set(args.subjects)) != len(args.subjects):
        raise ValueError("subjects must contain at least two unique IDs.")
    args.positive_theta_grid = sorted(
        {float(value) for value in args.positive_theta_grid}
    )
    if not args.positive_theta_grid or any(
        not np.isfinite(value) or not 0.0 < value <= 1.0
        for value in args.positive_theta_grid
    ):
        raise ValueError("positive theta values must lie in (0, 1].")
    args.particle_counts = sorted({int(value) for value in args.particle_counts})
    if any(value < 2 for value in args.particle_counts):
        raise ValueError("particle counts must be at least 2.")
    if int(args.calibration_particle_count) not in args.particle_counts:
        raise ValueError("calibration particle count must be included in particle counts.")
    args.evaluation_only_particle_counts = sorted(
        {int(value) for value in args.evaluation_only_particle_counts}
    )
    if not set(args.evaluation_only_particle_counts).issubset(
        set(args.particle_counts)
    ):
        raise ValueError(
            "evaluation-only particle counts must be included in particle counts."
        )
    if int(args.calibration_particle_count) in args.evaluation_only_particle_counts:
        raise ValueError(
            "calibration particle count cannot be evaluation-only."
        )
    if (
        args.cohort_replicates < 2
        or args.calibration_replicates <= 0
        or args.calibration_replicates >= args.cohort_replicates
    ):
        raise ValueError(
            "cohort replicates must leave non-empty calibration and evaluation splits."
        )
    args.penalty_grid = sorted({float(value) for value in args.penalty_grid})
    if not args.penalty_grid or any(
        not np.isfinite(value) or value < 0.0
        for value in args.penalty_grid
    ):
        raise ValueError("penalties must be finite and non-negative.")
    for name in (
        "target_specificity",
        "pass_sensitivity",
        "pass_specificity",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")
    if args.n_jobs <= 0 or args.max_trials <= 1:
        raise ValueError("n_jobs and max_trials must be positive.")


def membership_pattern(replicate: int, n_subjects: int) -> tuple[int, ...]:
    if n_subjects == 3:
        patterns = (
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1),
            (0, 1, 0),
        )
        return patterns[int(replicate) % len(patterns)]
    available = [
        tuple((mask >> index) & 1 for index in range(n_subjects))
        for mask in range(1, 2**n_subjects - 1)
    ]
    return available[int(replicate) % len(available)]


def build_cohort_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for theta_plus in args.positive_theta_grid:
        for replicate in range(args.cohort_replicates):
            specs.append(
                {
                    "cohort_id": (
                        f"theta_{theta_token(theta_plus)}_rep_{replicate}"
                    ),
                    "split": (
                        "calibration"
                        if replicate < args.calibration_replicates
                        else "evaluation"
                    ),
                    "cohort_replicate": int(replicate),
                    "true_theta_plus": float(theta_plus),
                    "membership": membership_pattern(
                        replicate, len(args.subjects)
                    ),
                }
            )
    for replicate in range(args.cohort_replicates):
        specs.append(
            {
                "cohort_id": f"all_b0_rep_{replicate}",
                "split": (
                    "calibration"
                    if replicate < args.calibration_replicates
                    else "evaluation"
                ),
                "cohort_replicate": int(replicate),
                "true_theta_plus": 0.0,
                "membership": tuple(0 for _ in args.subjects),
            }
        )
    return sorted(specs, key=lambda item: str(item["cohort_id"]))


def generation_seed_for(
    args: argparse.Namespace,
    cohort_id: str,
    subject_id: int,
) -> int:
    return stable_seed(
        {
            "seed_role": "newplan_shared_theta_generation",
            "base_seed": int(args.base_seed),
            "cohort_id": str(cohort_id),
            "subject_id": int(subject_id),
        }
    )


def filter_seed_for(
    args: argparse.Namespace,
    cohort_id: str,
    subject_id: int,
) -> int:
    return stable_seed(
        {
            "seed_role": "newplan_shared_theta_filter_crn",
            "base_seed": int(args.base_seed),
            "cohort_id": str(cohort_id),
            "subject_id": int(subject_id),
        }
    )


def dataset_path_for(
    args: argparse.Namespace,
    cohort_id: str,
    subject_id: int,
) -> Path:
    return (
        args.output_dir
        / "datasets"
        / str(cohort_id)
        / f"subject_{int(subject_id)}.npz"
    )


def fit_cache_path_for(
    args: argparse.Namespace,
    cohort_id: str,
    subject_id: int,
    particle_count: int,
    fit_theta: float,
) -> Path:
    return (
        args.output_dir
        / "fit_cache"
        / str(cohort_id)
        / f"subject_{int(subject_id)}"
        / f"particles_{int(particle_count)}"
        / f"theta_{theta_token(fit_theta)}.json"
    )


def save_dataset(path: Path, dataset: Mapping[str, Any]) -> None:
    array_keys = (
        "stimulus",
        "choices",
        "feedback",
        "categories",
        "valid_mask",
        "train_mask",
        "test_mask",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{key: np.asarray(dataset[key]) for key in array_keys},
        metadata=np.asarray(
            json.dumps(
                {
                    key: value
                    for key, value in dataset.items()
                    if key not in array_keys
                },
                ensure_ascii=False,
            )
        ),
    )


def load_dataset(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        return {
            **metadata,
            **{
                key: payload[key]
                for key in payload.files
                if key != "metadata"
            },
        }


def generate_subject_dataset(
    *,
    args: argparse.Namespace,
    runner: StateModelSimulationRunner,
    base_engine: Mapping[str, Any],
    dataset_paths: Mapping[str, Path],
    cohort: Mapping[str, Any],
    subject_index: int,
    subject_id: int,
) -> dict[str, Any]:
    frame = runner._get_subject_frame(subject_id, 1.0).iloc[
        : int(args.max_trials)
    ].copy()
    arrays = runner._extract_arrays(frame, None)
    if arrays.categories is None:
        raise ValueError("Shared-theta recovery requires hard categories.")
    true_z = int(cohort["membership"][subject_index])
    true_theta = (
        float(cohort["true_theta_plus"]) if true_z else 0.0
    )
    true_engine = apply_fixed_hyperparams_to_engine_config(
        deepcopy(dict(base_engine)), params_for(args, true_theta)
    )
    generation_seed = generation_seed_for(
        args, str(cohort["cohort_id"]), subject_id
    )
    generated = generate_condition1_trajectory(
        engine_config=true_engine,
        subject_id=int(subject_id),
        stimulus=arrays.stimulus,
        categories=arrays.categories,
        epsilon=0.0,
        rho=float(args.rho),
        trajectory_seed=int(generation_seed),
        processed_data_dir=runner._processed_data_dir,
        dataset_paths=dataset_paths,
    )
    train_mask, test_mask = time_split_masks(frame)
    return {
        "cohort_id": str(cohort["cohort_id"]),
        "split": str(cohort["split"]),
        "cohort_replicate": int(cohort["cohort_replicate"]),
        "true_theta_plus": float(cohort["true_theta_plus"]),
        "subject_id": int(subject_id),
        "true_z": int(true_z),
        "true_theta": float(true_theta),
        "generation_seed": int(generation_seed),
        "generated_accuracy": float(np.mean(generated.feedback)),
        "generated_swap_count": int(
            sum(bool(item["swap_event"]) for item in generated.transition_log)
        ),
        "stimulus": np.asarray(arrays.stimulus, dtype=float),
        "choices": np.asarray(generated.choices, dtype=int),
        "feedback": np.asarray(generated.feedback, dtype=float),
        "categories": np.asarray(arrays.categories, dtype=int),
        "valid_mask": np.asarray(train_mask, dtype=bool)
        | np.asarray(test_mask, dtype=bool),
        "train_mask": np.asarray(train_mask, dtype=bool),
        "test_mask": np.asarray(test_mask, dtype=bool),
    }


def fit_point(
    *,
    args: argparse.Namespace,
    runner: StateModelSimulationRunner,
    base_engine: Mapping[str, Any],
    dataset_paths: Mapping[str, Path],
    dataset: Mapping[str, Any],
    particle_count: int,
    fit_theta: float,
) -> dict[str, Any]:
    engine = apply_fixed_hyperparams_to_engine_config(
        deepcopy(dict(base_engine)), params_for(args, fit_theta)
    )
    filter_seed = filter_seed_for(
        args, str(dataset["cohort_id"]), int(dataset["subject_id"])
    )
    result = run_newplan_particle_filter(
        engine_config=engine,
        subject_id=int(dataset["subject_id"]),
        stimulus=np.asarray(dataset["stimulus"], dtype=float),
        choices=np.asarray(dataset["choices"], dtype=int),
        feedback=np.asarray(dataset["feedback"], dtype=float),
        particle_count=int(particle_count),
        rho=float(args.rho),
        epsilon=0.0,
        filter_seed=int(filter_seed),
        resample_threshold_fraction=float(args.resample_threshold),
        valid_trial_mask=np.asarray(dataset["valid_mask"], dtype=bool),
        processed_data_dir=runner._processed_data_dir,
        dataset_paths=dataset_paths,
    )
    choice_index = np.asarray(dataset["choices"], dtype=int) - 1
    train = score_probabilities(
        result.marginal_probabilities,
        choice_index,
        np.asarray(dataset["train_mask"], dtype=bool),
    )
    test = score_probabilities(
        result.marginal_probabilities,
        choice_index,
        np.asarray(dataset["test_mask"], dtype=bool),
    )
    return {
        **{
            key: dataset[key]
            for key in (
                "cohort_id",
                "split",
                "cohort_replicate",
                "true_theta_plus",
                "subject_id",
                "true_z",
                "true_theta",
                "generation_seed",
                "generated_accuracy",
                "generated_swap_count",
            )
        },
        "fit_theta": float(fit_theta),
        "particle_count": int(particle_count),
        "filter_seed": int(filter_seed),
        "train_n": int(train["n"]),
        "train_brier": float(train["choice_brier"]),
        "train_nll": float(train["choice_nll"]),
        "test_n": int(test["n"]),
        "test_brier": float(test["choice_brier"]),
        "test_nll": float(test["choice_nll"]),
        "mean_pre_choice_ess_fraction": float(
            np.mean(result.pre_choice_ess) / particle_count
        ),
        "mean_post_choice_ess_fraction": float(
            np.mean(result.post_choice_ess) / particle_count
        ),
        "resampling_fraction": float(np.mean(result.resampled)),
    }


def candidate_matrix(
    rows: Sequence[Mapping[str, Any]],
    positive_theta_grid: Sequence[float],
) -> tuple[list[int], np.ndarray, np.ndarray, dict[tuple[int, float], Mapping[str, Any]]]:
    lookup = {
        (int(row["subject_id"]), float(row["fit_theta"])): row
        for row in rows
    }
    subjects = sorted({int(row["subject_id"]) for row in rows})
    b0 = np.asarray(
        [float(lookup[(subject, 0.0)]["train_brier"]) for subject in subjects],
        dtype=float,
    )
    dynamic = np.asarray(
        [
            [
                float(lookup[(subject, float(theta))]["train_brier"])
                for theta in positive_theta_grid
            ]
            for subject in subjects
        ],
        dtype=float,
    )
    return subjects, b0, dynamic, lookup


def select_cohort(
    rows: Sequence[Mapping[str, Any]],
    *,
    positive_theta_grid: Sequence[float],
    penalty: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    subjects, b0, dynamic, lookup = candidate_matrix(
        rows, positive_theta_grid
    )
    selected = select_shared_theta(
        b0_losses=b0,
        dynamic_losses=dynamic,
        positive_theta_grid=positive_theta_grid,
        membership_penalty=float(penalty),
    )
    first = rows[0]
    cohort_row = {
        "cohort_id": str(first["cohort_id"]),
        "split": str(first["split"]),
        "cohort_replicate": int(first["cohort_replicate"]),
        "particle_count": int(first["particle_count"]),
        "penalty": float(penalty),
        "true_theta_plus": float(first["true_theta_plus"]),
        "estimated_theta_plus": float(selected.theta_plus),
        "true_any_dynamic": int(
            any(int(row["true_z"]) for row in rows)
        ),
        "estimated_any_dynamic": int(np.any(selected.membership)),
        "penalized_train_objective": float(selected.objective),
    }
    decisions = []
    for subject_index, subject_id in enumerate(subjects):
        b0_row = lookup[(subject_id, 0.0)]
        estimated_z = bool(selected.membership[subject_index])
        selected_row = (
            lookup[(subject_id, float(selected.theta_plus))]
            if estimated_z
            else b0_row
        )
        dynamic_row = (
            lookup[(subject_id, float(selected.theta_plus))]
            if selected.theta_plus > 0.0
            else b0_row
        )
        decisions.append(
            {
                **{
                    key: b0_row[key]
                    for key in (
                        "cohort_id",
                        "split",
                        "cohort_replicate",
                        "particle_count",
                        "true_theta_plus",
                        "subject_id",
                        "true_z",
                        "true_theta",
                        "generated_accuracy",
                        "generated_swap_count",
                    )
                },
                "penalty": float(penalty),
                "estimated_theta_plus": float(selected.theta_plus),
                "estimated_z": int(estimated_z),
                "b0_train_brier": float(b0_row["train_brier"]),
                "dynamic_train_brier": float(dynamic_row["train_brier"]),
                "selected_train_brier": float(selected_row["train_brier"]),
                "b0_test_brier": float(b0_row["test_brier"]),
                "dynamic_test_brier": float(dynamic_row["test_brier"]),
                "selected_test_brier": float(selected_row["test_brier"]),
                "selected_minus_b0_test_brier": float(
                    selected_row["test_brier"] - b0_row["test_brier"]
                ),
                "selected_minus_b0_test_nll": float(
                    selected_row["test_nll"] - b0_row["test_nll"]
                ),
                "mean_pre_choice_ess_fraction": float(
                    selected_row["mean_pre_choice_ess_fraction"]
                ),
                "resampling_fraction": float(
                    selected_row["resampling_fraction"]
                ),
            }
        )
    return cohort_row, decisions


def all_selections(
    fit_rows: Sequence[Mapping[str, Any]],
    *,
    positive_theta_grid: Sequence[float],
    penalties: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in fit_rows:
        grouped.setdefault(
            (str(row["cohort_id"]), int(row["particle_count"])), []
        ).append(row)
    cohort_rows = []
    decisions = []
    for _, rows in sorted(grouped.items()):
        for penalty in penalties:
            cohort, subject_rows = select_cohort(
                rows,
                positive_theta_grid=positive_theta_grid,
                penalty=float(penalty),
            )
            cohort_rows.append(cohort)
            decisions.extend(subject_rows)
    return cohort_rows, decisions


def penalty_frontier(
    decisions: Sequence[Mapping[str, Any]],
    *,
    calibration_particle_count: int,
    penalties: Sequence[float],
) -> list[dict[str, Any]]:
    frontier = []
    for penalty in penalties:
        rows = [
            row
            for row in decisions
            if str(row["split"]) == "calibration"
            and int(row["particle_count"]) == calibration_particle_count
            and float(row["penalty"]) == float(penalty)
        ]
        metrics = binary_recovery_metrics(
            [bool(row["true_z"]) for row in rows],
            [bool(row["estimated_z"]) for row in rows],
        )
        frontier.append(
            {
                "particle_count": int(calibration_particle_count),
                "penalty": float(penalty),
                **metrics,
            }
        )
    return frontier


def evaluation_frontier(
    decisions: Sequence[Mapping[str, Any]],
    *,
    particle_counts: Sequence[int],
    penalties: Sequence[float],
) -> list[dict[str, Any]]:
    frontier = []
    for particle_count in particle_counts:
        for penalty in penalties:
            rows = [
                row
                for row in decisions
                if str(row["split"]) == "evaluation"
                and int(row["particle_count"]) == int(particle_count)
                and float(row["penalty"]) == float(penalty)
            ]
            if not rows:
                continue
            metrics = binary_recovery_metrics(
                [bool(row["true_z"]) for row in rows],
                [bool(row["estimated_z"]) for row in rows],
            )
            frontier.append(
                {
                    "particle_count": int(particle_count),
                    "penalty": float(penalty),
                    **metrics,
                    "mean_selected_minus_b0_test_brier": float(
                        np.mean(
                            [
                                float(
                                    row["selected_minus_b0_test_brier"]
                                )
                                for row in rows
                            ]
                        )
                    ),
                }
            )
    return frontier


def training_evidence_overlap(
    fit_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, int], list[Mapping[str, Any]]] = {}
    for row in fit_rows:
        if str(row["split"]) != "evaluation":
            continue
        grouped.setdefault(
            (
                int(row["particle_count"]),
                str(row["cohort_id"]),
                int(row["subject_id"]),
            ),
            [],
        ).append(row)
    by_particles: dict[int, list[tuple[bool, float, float]]] = {}
    for (particle_count, _, _), rows in grouped.items():
        b0 = next(row for row in rows if float(row["fit_theta"]) == 0.0)
        dynamic = [row for row in rows if float(row["fit_theta"]) > 0.0]
        best_gain = float(b0["train_brier"]) - min(
            float(row["train_brier"]) for row in dynamic
        )
        true_z = bool(b0["true_z"])
        true_gain = float("nan")
        if true_z:
            true_row = next(
                row
                for row in rows
                if float(row["fit_theta"]) == float(row["true_theta"])
            )
            true_gain = (
                float(b0["train_brier"]) - float(true_row["train_brier"])
            )
        by_particles.setdefault(particle_count, []).append(
            (true_z, best_gain, true_gain)
        )

    summaries = []
    for particle_count, values in sorted(by_particles.items()):
        positive = np.asarray(
            [gain for truth, gain, _ in values if truth], dtype=float
        )
        negative = np.asarray(
            [gain for truth, gain, _ in values if not truth], dtype=float
        )
        true_gains = np.asarray(
            [gain for truth, _, gain in values if truth], dtype=float
        )
        pair_scores = [
            1.0 if pos > neg else 0.5 if pos == neg else 0.0
            for pos in positive
            for neg in negative
        ]
        summaries.append(
            {
                "particle_count": int(particle_count),
                "dynamic_n": int(positive.size),
                "stable_n": int(negative.size),
                "dynamic_best_gain_median": float(np.median(positive)),
                "dynamic_best_gain_q25": float(np.quantile(positive, 0.25)),
                "dynamic_best_gain_q75": float(np.quantile(positive, 0.75)),
                "stable_best_gain_median": float(np.median(negative)),
                "stable_best_gain_q25": float(np.quantile(negative, 0.25)),
                "stable_best_gain_q75": float(np.quantile(negative, 0.75)),
                "true_theta_positive_gain_count": int(
                    np.sum(true_gains > 0.0)
                ),
                "training_gain_auc": float(np.mean(pair_scores)),
            }
        )
    return summaries


def summarize_frozen(
    cohort_rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    *,
    frozen_penalty: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output: dict[str, Any] = {
        "frozen_penalty": float(frozen_penalty),
        "pass_sensitivity": float(args.pass_sensitivity),
        "pass_specificity": float(args.pass_specificity),
        "by_particle_and_split": {},
    }
    for particle_count in args.particle_counts:
        for split in ("calibration", "evaluation"):
            subject_rows = [
                row
                for row in decisions
                if int(row["particle_count"]) == particle_count
                and str(row["split"]) == split
                and float(row["penalty"]) == float(frozen_penalty)
            ]
            selected_cohorts = [
                row
                for row in cohort_rows
                if int(row["particle_count"]) == particle_count
                and str(row["split"]) == split
                and float(row["penalty"]) == float(frozen_penalty)
            ]
            if not subject_rows or not selected_cohorts:
                continue
            membership = binary_recovery_metrics(
                [bool(row["true_z"]) for row in subject_rows],
                [bool(row["estimated_z"]) for row in subject_rows],
            )
            cohort_family = binary_recovery_metrics(
                [bool(row["true_any_dynamic"]) for row in selected_cohorts],
                [bool(row["estimated_any_dynamic"]) for row in selected_cohorts],
            )
            dynamic_cohorts = [
                row
                for row in selected_cohorts
                if bool(row["true_any_dynamic"])
            ]
            theta_errors = np.asarray(
                [
                    float(row["estimated_theta_plus"])
                    - float(row["true_theta_plus"])
                    for row in dynamic_cohorts
                ],
                dtype=float,
            )
            brier_deltas = np.asarray(
                [
                    float(row["selected_minus_b0_test_brier"])
                    for row in subject_rows
                ],
                dtype=float,
            )
            nll_deltas = np.asarray(
                [
                    float(row["selected_minus_b0_test_nll"])
                    for row in subject_rows
                ],
                dtype=float,
            )
            passed = (
                float(membership["sensitivity"]) >= args.pass_sensitivity
                and float(membership["specificity"]) >= args.pass_specificity
                and float(np.mean(brier_deltas)) < 0.0
            )
            output["by_particle_and_split"][
                f"{particle_count}_{split}"
            ] = {
                "n_cohorts": len(selected_cohorts),
                "membership": membership,
                "cohort_family": cohort_family,
                "theta_rmse_dynamic_cohorts": float(
                    np.sqrt(np.mean(np.square(theta_errors)))
                ),
                "theta_exact_count_dynamic_cohorts": int(
                    np.sum(theta_errors == 0.0)
                ),
                "theta_dynamic_cohort_n": len(dynamic_cohorts),
                "mean_selected_minus_b0_test_brier": float(
                    np.mean(brier_deltas)
                ),
                "mean_selected_minus_b0_test_nll": float(
                    np.mean(nll_deltas)
                ),
                "heldout_brier_improved_count": int(
                    np.sum(brier_deltas < 0.0)
                ),
                "mean_pre_choice_ess_fraction": float(
                    np.mean(
                        [
                            float(row["mean_pre_choice_ess_fraction"])
                            for row in subject_rows
                        ]
                    )
                ),
                "mean_resampling_fraction": float(
                    np.mean(
                        [
                            float(row["resampling_fraction"])
                            for row in subject_rows
                        ]
                    )
                ),
                "passes_gate": bool(passed),
            }
    if len(args.particle_counts) >= 2:
        smaller, larger = args.particle_counts[-2:]
        small = {
            str(row["cohort_id"]): row
            for row in cohort_rows
            if int(row["particle_count"]) == smaller
            and str(row["split"]) == "evaluation"
            and float(row["penalty"]) == float(frozen_penalty)
        }
        large = {
            str(row["cohort_id"]): row
            for row in cohort_rows
            if int(row["particle_count"]) == larger
            and str(row["split"]) == "evaluation"
            and float(row["penalty"]) == float(frozen_penalty)
        }
        common = sorted(set(small) & set(large))
        output["particle_convergence"] = {
            "smaller": int(smaller),
            "larger": int(larger),
            "theta_agreement_count": int(
                sum(
                    float(small[key]["estimated_theta_plus"])
                    == float(large[key]["estimated_theta_plus"])
                    for key in common
                )
            ),
            "membership_pattern_agreement_count": int(
                sum(
                    [
                        int(row["estimated_z"])
                        for row in decisions
                        if int(row["particle_count"]) == smaller
                        and str(row["split"]) == "evaluation"
                        and str(row["cohort_id"]) == key
                        and float(row["penalty"]) == float(frozen_penalty)
                    ]
                    == [
                        int(row["estimated_z"])
                        for row in decisions
                        if int(row["particle_count"]) == larger
                        and str(row["split"]) == "evaluation"
                        and str(row["cohort_id"]) == key
                        and float(row["penalty"]) == float(frozen_penalty)
                    ]
                    for key in common
                )
            ),
            "n_cohorts": len(common),
        }
    return output


def real_shared_theta(
    path: Path,
    *,
    frozen_penalty: float,
    positive_theta_grid: Sequence[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = pd.read_csv(path)
    required = {
        "subject_id",
        "theta",
        "particle_count",
        "replicate",
        "train_choice_brier",
        "test_choice_brier",
        "test_choice_nll",
        "test_n",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame[
        frame["theta"].isin([0.0, *positive_theta_grid])
        & frame["replicate"].eq(frame["replicate"].min())
    ].copy()
    frame = frame[frame["test_n"].gt(0)].copy()
    cohort_rows = []
    decisions = []
    for particle_count, particle_frame in frame.groupby("particle_count"):
        rows = particle_frame.to_dict("records")
        subjects = sorted({int(row["subject_id"]) for row in rows})
        lookup = {
            (int(row["subject_id"]), float(row["theta"])): row
            for row in rows
        }
        usable = [
            subject
            for subject in subjects
            if all(
                (subject, float(theta)) in lookup
                for theta in [0.0, *positive_theta_grid]
            )
        ]
        b0 = np.asarray(
            [
                float(lookup[(subject, 0.0)]["train_choice_brier"])
                for subject in usable
            ]
        )
        dynamic = np.asarray(
            [
                [
                    float(
                        lookup[(subject, float(theta))][
                            "train_choice_brier"
                        ]
                    )
                    for theta in positive_theta_grid
                ]
                for subject in usable
            ]
        )
        selected = select_shared_theta(
            b0_losses=b0,
            dynamic_losses=dynamic,
            positive_theta_grid=positive_theta_grid,
            membership_penalty=float(frozen_penalty),
        )
        source_id = str(path.parent.name)
        cohort_rows.append(
            {
                "source": source_id,
                "particle_count": int(particle_count),
                "penalty": float(frozen_penalty),
                "estimated_theta_plus": float(selected.theta_plus),
                "dynamic_count": int(np.sum(selected.membership)),
                "subject_n": len(usable),
            }
        )
        for index, subject in enumerate(usable):
            b0_row = lookup[(subject, 0.0)]
            estimated_z = bool(selected.membership[index])
            selected_row = (
                lookup[(subject, float(selected.theta_plus))]
                if estimated_z
                else b0_row
            )
            decisions.append(
                {
                    "source": source_id,
                    "particle_count": int(particle_count),
                    "subject_id": int(subject),
                    "penalty": float(frozen_penalty),
                    "estimated_theta_plus": float(selected.theta_plus),
                    "estimated_z": int(estimated_z),
                    "selected_minus_b0_test_brier": float(
                        selected_row["test_choice_brier"]
                        - b0_row["test_choice_brier"]
                    ),
                    "selected_minus_b0_test_nll": float(
                        selected_row["test_choice_nll"]
                        - b0_row["test_choice_nll"]
                    ),
                }
            )
    return cohort_rows, decisions


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    frontier: Sequence[Mapping[str, Any]],
    diagnostic_frontier: Sequence[Mapping[str, Any]],
    evidence_overlap: Sequence[Mapping[str, Any]],
    frozen_penalty: float,
    summary: Mapping[str, Any],
    real_cohorts: Sequence[Mapping[str, Any]],
    real_decisions: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Zero-boundary shared-theta B0/D0 recovery",
        "",
        r"Model: `theta_s = z_s * theta_plus`, with exact `z_s in {0,1}`.",
        "",
        f"- subjects per cohort: {', '.join(str(v) for v in args.subjects)}",
        f"- positive theta grid: {', '.join(str(v) for v in args.positive_theta_grid)}",
        f"- particle counts: {', '.join(str(v) for v in args.particle_counts)}",
        f"- calibration/evaluation cohort replicates: "
        f"{args.calibration_replicates}/"
        f"{args.cohort_replicates - args.calibration_replicates}",
        f"- frozen membership penalty: {frozen_penalty:.6f}",
        f"- gate: sensitivity >= {args.pass_sensitivity:.2f}, "
        f"specificity >= {args.pass_specificity:.2f}, "
        "and mean held-out Brier delta < 0",
        "",
        "## Calibration frontier (R="
        f"{args.calibration_particle_count})",
        "",
        "| penalty | sensitivity | specificity | accuracy |",
        "|---:|---:|---:|---:|",
    ]
    for row in frontier:
        lines.append(
            f"| {float(row['penalty']):.6f} | "
            f"{float(row['sensitivity']):.3f} | "
            f"{float(row['specificity']):.3f} | "
            f"{int(row['accuracy_count'])}/{int(row['n'])} |"
        )
    lines.extend(
        [
            "",
            "## Evaluation sensitivity-specificity frontier",
            "",
            "Diagnostic only: evaluation data were not used to select the "
            "frozen penalty.",
            "",
            "| particles | penalty | sensitivity | specificity | "
            "accuracy | mean test Brier delta |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostic_frontier:
        lines.append(
            f"| {int(row['particle_count'])} | "
            f"{float(row['penalty']):.6f} | "
            f"{float(row['sensitivity']):.3f} | "
            f"{float(row['specificity']):.3f} | "
            f"{int(row['accuracy_count'])}/{int(row['n'])} | "
            f"{float(row['mean_selected_minus_b0_test_brier']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen-rule recovery",
            "",
            "| particles | split | z sensitivity | z specificity | "
            "theta RMSE | mean test Brier delta | ESS/R | resample | gate |",
            "|---:|:---|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for key, item in summary["by_particle_and_split"].items():
        particle_count, split = key.split("_", maxsplit=1)
        lines.append(
            f"| {particle_count} | {split} | "
            f"{float(item['membership']['sensitivity']):.3f} | "
            f"{float(item['membership']['specificity']):.3f} | "
            f"{float(item['theta_rmse_dynamic_cohorts']):.4f} | "
            f"{float(item['mean_selected_minus_b0_test_brier']):.6f} | "
            f"{float(item['mean_pre_choice_ess_fraction']):.3f} | "
            f"{float(item['mean_resampling_fraction']):.3f} | "
            f"{'PASS' if item['passes_gate'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "## Training-evidence overlap in evaluation cohorts",
            "",
            "`gain = B0 train Brier - best non-zero-theta train Brier`; "
            "larger values favor D0.",
            "",
            "| particles | dynamic median [IQR] | stable median [IQR] | "
            "true-theta positive | AUC |",
            "|---:|:---|:---|---:|---:|",
        ]
    )
    for row in evidence_overlap:
        lines.append(
            f"| {int(row['particle_count'])} | "
            f"{float(row['dynamic_best_gain_median']):.4f} "
            f"[{float(row['dynamic_best_gain_q25']):.4f}, "
            f"{float(row['dynamic_best_gain_q75']):.4f}] | "
            f"{float(row['stable_best_gain_median']):.4f} "
            f"[{float(row['stable_best_gain_q25']):.4f}, "
            f"{float(row['stable_best_gain_q75']):.4f}] | "
            f"{int(row['true_theta_positive_gain_count'])}/"
            f"{int(row['dynamic_n'])} | "
            f"{float(row['training_gain_auc']):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Real held-out application",
            "",
            "| source | particles | theta_plus | dynamic subjects | "
            "mean Brier delta | mean NLL delta |",
            "|:---|---:|---:|:---|---:|---:|",
        ]
    )
    for cohort in real_cohorts:
        rows = [
            row
            for row in real_decisions
            if str(row["source"]) == str(cohort["source"])
            and int(row["particle_count"]) == int(cohort["particle_count"])
        ]
        dynamic_subjects = [
            str(row["subject_id"]) for row in rows if int(row["estimated_z"])
        ]
        lines.append(
            f"| {cohort['source']} | {cohort['particle_count']} | "
            f"{float(cohort['estimated_theta_plus']):.3f} | "
            f"{', '.join(dynamic_subjects) if dynamic_subjects else 'none'} | "
            f"{np.mean([float(row['selected_minus_b0_test_brier']) for row in rows]):.6f} | "
            f"{np.mean([float(row['selected_minus_b0_test_nll']) for row in rows]):.6f} |"
        )
    highest = max(args.particle_counts)
    evaluation = summary["by_particle_and_split"][f"{highest}_evaluation"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            (
                "**PASS:** the frozen reduced model reaches the predefined "
                "evaluation gate."
                if evaluation["passes_gate"]
                else
                "**FAIL:** the frozen reduced model does not reach the "
                "predefined evaluation gate; do not add further cognitive "
                "states."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cohort_specs = build_cohort_specs(args)

    model_path = ROOT / "configs/model_struct/pmh_model_cond1_newplan.yaml"
    sim_path = ROOT / "configs/simulation_cfg/pmh_cond1_simulation_v14.yaml"
    base_engine = load_yaml(model_path)
    sim_cfg = load_yaml(sim_path)
    dataset_paths = resolve_dataset_paths(
        sim_cfg, sim_path.parent, DEFAULT_DATA_PATH
    )
    runner = StateModelSimulationRunner(
        engine_config=base_engine,
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
        n_jobs=1,
    )
    runner.prepare_data(dataset_paths["learning_data"])

    write_json(
        args.output_dir / "manifest.json",
        {
            "result_type": "newplan_zero_boundary_shared_theta",
            "subjects": args.subjects,
            "positive_theta_grid": args.positive_theta_grid,
            "particle_counts": args.particle_counts,
            "evaluation_only_particle_counts": (
                args.evaluation_only_particle_counts
            ),
            "calibration_particle_count": args.calibration_particle_count,
            "cohort_replicates": args.cohort_replicates,
            "calibration_replicates": args.calibration_replicates,
            "penalty_grid": args.penalty_grid,
            "target_specificity": args.target_specificity,
            "pass_sensitivity": args.pass_sensitivity,
            "pass_specificity": args.pass_specificity,
            "gamma": args.gamma,
            "w0": args.w0,
            "rho": args.rho,
            "max_trials": args.max_trials,
            "base_seed": args.base_seed,
            "cohorts": cohort_specs,
        },
    )

    all_fit_rows: list[dict[str, Any]] = []
    fit_theta_grid = [0.0, *args.positive_theta_grid]
    for cohort in cohort_specs:
        for subject_index, subject_id in enumerate(args.subjects):
            dataset_path = dataset_path_for(
                args, str(cohort["cohort_id"]), subject_id
            )
            if dataset_path.exists() and not args.force:
                dataset = load_dataset(dataset_path)
                print(
                    f"LOAD cohort={cohort['cohort_id']} subject={subject_id}",
                    flush=True,
                )
            else:
                dataset = generate_subject_dataset(
                    args=args,
                    runner=runner,
                    base_engine=base_engine,
                    dataset_paths=dataset_paths,
                    cohort=cohort,
                    subject_index=subject_index,
                    subject_id=subject_id,
                )
                save_dataset(dataset_path, dataset)
                print(
                    f"GENERATE cohort={cohort['cohort_id']} subject={subject_id}",
                    flush=True,
                )
            for particle_count in args.particle_counts:
                if (
                    particle_count in args.evaluation_only_particle_counts
                    and str(cohort["split"]) != "evaluation"
                ):
                    continue
                resolved: dict[float, dict[str, Any]] = {}
                missing = []
                for fit_theta in fit_theta_grid:
                    cache_path = fit_cache_path_for(
                        args,
                        str(cohort["cohort_id"]),
                        subject_id,
                        particle_count,
                        fit_theta,
                    )
                    if cache_path.exists() and not args.force:
                        resolved[float(fit_theta)] = json.loads(
                            cache_path.read_text(encoding="utf-8")
                        )
                    else:
                        missing.append(float(fit_theta))
                computed = Parallel(n_jobs=min(args.n_jobs, len(missing) or 1))(
                    delayed(fit_point)(
                        args=args,
                        runner=runner,
                        base_engine=base_engine,
                        dataset_paths=dataset_paths,
                        dataset=dataset,
                        particle_count=int(particle_count),
                        fit_theta=fit_theta,
                    )
                    for fit_theta in missing
                )
                for fit_theta, row in zip(missing, computed):
                    cache_path = fit_cache_path_for(
                        args,
                        str(cohort["cohort_id"]),
                        subject_id,
                        particle_count,
                        fit_theta,
                    )
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(
                        json.dumps(
                            row,
                            ensure_ascii=False,
                            indent=2,
                            allow_nan=True,
                        ),
                        encoding="utf-8",
                    )
                    resolved[float(fit_theta)] = row
                all_fit_rows.extend(
                    resolved[float(theta)] for theta in fit_theta_grid
                )

    cohort_rows, all_decisions = all_selections(
        all_fit_rows,
        positive_theta_grid=args.positive_theta_grid,
        penalties=args.penalty_grid,
    )
    frontier = penalty_frontier(
        all_decisions,
        calibration_particle_count=int(args.calibration_particle_count),
        penalties=args.penalty_grid,
    )
    penalty_index = choose_membership_penalty(
        penalties=[float(row["penalty"]) for row in frontier],
        specificities=[float(row["specificity"]) for row in frontier],
        sensitivities=[float(row["sensitivity"]) for row in frontier],
        target_specificity=float(args.target_specificity),
    )
    frozen_penalty = float(frontier[penalty_index]["penalty"])
    diagnostic_frontier = evaluation_frontier(
        all_decisions,
        particle_counts=args.particle_counts,
        penalties=args.penalty_grid,
    )
    evidence_overlap = training_evidence_overlap(all_fit_rows)
    summary = summarize_frozen(
        cohort_rows,
        all_decisions,
        frozen_penalty=frozen_penalty,
        args=args,
    )

    frozen_cohorts = [
        row
        for row in cohort_rows
        if float(row["penalty"]) == frozen_penalty
    ]
    frozen_decisions = [
        row
        for row in all_decisions
        if float(row["penalty"]) == frozen_penalty
    ]
    real_cohorts: list[dict[str, Any]] = []
    real_decisions: list[dict[str, Any]] = []
    if not args.skip_real:
        for real_path in args.real_grids:
            if not real_path.exists():
                print(f"SKIP missing real grid={real_path}", flush=True)
                continue
            cohorts, decisions = real_shared_theta(
                real_path,
                frozen_penalty=frozen_penalty,
                positive_theta_grid=args.positive_theta_grid,
            )
            real_cohorts.extend(cohorts)
            real_decisions.extend(decisions)

    write_csv(args.output_dir / "fit_grid.csv", all_fit_rows)
    write_csv(args.output_dir / "penalty_frontier.csv", frontier)
    write_csv(
        args.output_dir / "evaluation_frontier.csv",
        diagnostic_frontier,
    )
    write_csv(
        args.output_dir / "training_evidence_overlap.csv",
        evidence_overlap,
    )
    write_csv(args.output_dir / "cohort_selection.csv", frozen_cohorts)
    write_csv(args.output_dir / "subject_decisions.csv", frozen_decisions)
    write_csv(args.output_dir / "real_cohort_selection.csv", real_cohorts)
    write_csv(args.output_dir / "real_subject_decisions.csv", real_decisions)
    write_json(args.output_dir / "aggregate_summary.json", summary)
    write_json(
        args.output_dir / "diagnostic_summary.json",
        {
            "evaluation_frontier": diagnostic_frontier,
            "training_evidence_overlap": evidence_overlap,
        },
    )
    write_report(
        args.output_dir / "RESULTS.md",
        args=args,
        frontier=frontier,
        diagnostic_frontier=diagnostic_frontier,
        evidence_overlap=evidence_overlap,
        frozen_penalty=frozen_penalty,
        summary=summary,
        real_cohorts=real_cohorts,
        real_decisions=real_decisions,
    )
    print(
        f"COMPLETE output={args.output_dir} frozen_penalty={frozen_penalty}",
        flush=True,
    )


if __name__ == "__main__":
    main()
