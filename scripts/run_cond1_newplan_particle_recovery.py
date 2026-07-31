#!/usr/bin/env python3
"""Closed-loop B0/D0 recovery using the online new-plan particle filter."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
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
from src.Bayesian_state.utils.optimization_config import (
    DEFAULT_DATA_PATH,
    load_yaml,
)
from src.Bayesian_state.utils.optimizer_common import stable_seed
from src.Bayesian_state.utils.optimizer_simulation import (
    StateModelSimulationRunner,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=[103, 117, 131])
    parser.add_argument(
        "--true-theta-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.50, 0.75, 1.0],
    )
    parser.add_argument(
        "--fit-theta-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.50, 0.75, 1.0],
    )
    parser.add_argument(
        "--particle-counts", type=int, nargs="+", default=[32, 64]
    )
    parser.add_argument("--datasets-per-theta", type=int, default=1)
    parser.add_argument("--filter-replicates", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--resample-threshold", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.55)
    parser.add_argument("--w0", type=float, default=0.10)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--max-trials", type=int, default=128)
    parser.add_argument("--base-seed", type=int, default=20260801)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/zhuran/cond1_newplan/particle_recovery",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_probability_grid(values: Sequence[float], name: str) -> list[float]:
    grid = sorted({float(value) for value in values})
    if not grid or any(
        not np.isfinite(value) or not 0.0 <= value <= 1.0
        for value in grid
    ):
        raise ValueError(f"{name} must contain finite values in [0, 1].")
    return grid


def generation_seed_for(
    args: argparse.Namespace,
    subject_id: int,
    true_theta: float,
    dataset_replicate: int,
) -> int:
    return stable_seed(
        {
            "seed_role": "newplan_particle_recovery_generation",
            "base_seed": int(args.base_seed),
            "subject_id": int(subject_id),
            "true_theta": float(true_theta),
            "dataset_replicate": int(dataset_replicate),
        }
    )


def filter_seed_for(
    args: argparse.Namespace,
    subject_id: int,
    true_theta: float,
    dataset_replicate: int,
    filter_replicate: int,
) -> int:
    return stable_seed(
        {
            "seed_role": "newplan_particle_recovery_filter_crn",
            "base_seed": int(args.base_seed),
            "subject_id": int(subject_id),
            "true_theta": float(true_theta),
            "dataset_replicate": int(dataset_replicate),
            "filter_replicate": int(filter_replicate),
        }
    )


def dataset_token(
    subject_id: int,
    true_theta: float,
    dataset_replicate: int,
) -> str:
    return (
        f"subject_{int(subject_id)}/true_theta_{theta_token(true_theta)}"
        f"/dataset_{int(dataset_replicate)}"
    )


def generate_dataset(
    *,
    args: argparse.Namespace,
    runner: StateModelSimulationRunner,
    base_engine: Mapping[str, Any],
    dataset_paths: Mapping[str, Path],
    subject_id: int,
    true_theta: float,
    dataset_replicate: int,
) -> dict[str, Any]:
    subject_frame = runner._get_subject_frame(subject_id, 1.0).iloc[
        : int(args.max_trials)
    ].copy()
    arrays = runner._extract_arrays(subject_frame, None)
    if arrays.categories is None:
        raise ValueError("Closed-loop recovery requires hard categories.")
    true_engine = apply_fixed_hyperparams_to_engine_config(
        deepcopy(dict(base_engine)), params_for(args, true_theta)
    )
    generation_seed = generation_seed_for(
        args, subject_id, true_theta, dataset_replicate
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
    train_mask, test_mask = time_split_masks(subject_frame)
    return {
        "subject_id": int(subject_id),
        "true_theta": float(true_theta),
        "true_model": "B0" if float(true_theta) == 0.0 else "D0",
        "dataset_replicate": int(dataset_replicate),
        "generation_seed": int(generation_seed),
        "stimulus": np.asarray(arrays.stimulus, dtype=float),
        "choices": np.asarray(generated.choices, dtype=int),
        "feedback": np.asarray(generated.feedback, dtype=float),
        "categories": np.asarray(arrays.categories, dtype=int),
        "valid_mask": np.asarray(train_mask, dtype=bool)
        | np.asarray(test_mask, dtype=bool),
        "train_mask": np.asarray(train_mask, dtype=bool),
        "test_mask": np.asarray(test_mask, dtype=bool),
        "generated_accuracy": float(np.mean(generated.feedback)),
        "generated_swap_count": int(
            sum(bool(item["swap_event"]) for item in generated.transition_log)
        ),
    }


def save_dataset(path: Path, dataset: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{
            key: np.asarray(dataset[key])
            for key in (
                "stimulus",
                "choices",
                "feedback",
                "categories",
                "valid_mask",
                "train_mask",
                "test_mask",
            )
        },
        metadata=np.asarray(
            json.dumps(
                {
                    key: value
                    for key, value in dataset.items()
                    if key
                    not in {
                        "stimulus",
                        "choices",
                        "feedback",
                        "categories",
                        "valid_mask",
                        "train_mask",
                        "test_mask",
                    }
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


def fit_point(
    *,
    args: argparse.Namespace,
    runner: StateModelSimulationRunner,
    base_engine: Mapping[str, Any],
    dataset_paths: Mapping[str, Path],
    dataset: Mapping[str, Any],
    fit_theta: float,
    particle_count: int,
    filter_replicate: int,
) -> dict[str, Any]:
    engine = apply_fixed_hyperparams_to_engine_config(
        deepcopy(dict(base_engine)), params_for(args, fit_theta)
    )
    filter_seed = filter_seed_for(
        args,
        int(dataset["subject_id"]),
        float(dataset["true_theta"]),
        int(dataset["dataset_replicate"]),
        int(filter_replicate),
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
    observed_choice = np.asarray(dataset["choices"], dtype=int) - 1
    train = score_probabilities(
        result.marginal_probabilities,
        observed_choice,
        np.asarray(dataset["train_mask"], dtype=bool),
    )
    test = score_probabilities(
        result.marginal_probabilities,
        observed_choice,
        np.asarray(dataset["test_mask"], dtype=bool),
    )
    return {
        "subject_id": int(dataset["subject_id"]),
        "true_theta": float(dataset["true_theta"]),
        "true_model": str(dataset["true_model"]),
        "dataset_replicate": int(dataset["dataset_replicate"]),
        "generation_seed": int(dataset["generation_seed"]),
        "generated_accuracy": float(dataset["generated_accuracy"]),
        "generated_swap_count": int(dataset["generated_swap_count"]),
        "fit_theta": float(fit_theta),
        "particle_count": int(particle_count),
        "filter_replicate": int(filter_replicate),
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
        "min_post_choice_ess_fraction": float(
            np.min(result.post_choice_ess) / particle_count
        ),
        "resampling_count": int(np.sum(result.resampled)),
        "resampling_fraction": float(np.mean(result.resampled)),
    }


def cache_path_for(
    args: argparse.Namespace,
    dataset: Mapping[str, Any],
    fit_theta: float,
    particle_count: int,
    filter_replicate: int,
) -> Path:
    return (
        args.output_dir
        / "fit_cache"
        / dataset_token(
            int(dataset["subject_id"]),
            float(dataset["true_theta"]),
            int(dataset["dataset_replicate"]),
        )
        / f"particles_{int(particle_count)}"
        / f"filter_{int(filter_replicate)}"
        / f"fit_theta_{theta_token(fit_theta)}.json"
    )


def select_dataset(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    b0 = next(row for row in rows if float(row["fit_theta"]) == 0.0)
    d0 = min(
        rows,
        key=lambda row: (
            float(row["train_brier"]),
            float(row["fit_theta"]) > 0.0,
            float(row["fit_theta"]),
        ),
    )
    selected_model = "D0"
    if float(d0["fit_theta"]) == 0.0:
        selected_model = "B0"
    elif float(d0["test_brier"]) >= float(b0["test_brier"]):
        selected_model = "B0"
    return {
        "subject_id": int(b0["subject_id"]),
        "true_theta": float(b0["true_theta"]),
        "true_model": str(b0["true_model"]),
        "dataset_replicate": int(b0["dataset_replicate"]),
        "generation_seed": int(b0["generation_seed"]),
        "generated_accuracy": float(b0["generated_accuracy"]),
        "generated_swap_count": int(b0["generated_swap_count"]),
        "particle_count": int(b0["particle_count"]),
        "filter_replicate": int(b0["filter_replicate"]),
        "filter_seed": int(b0["filter_seed"]),
        "estimated_theta": float(d0["fit_theta"]),
        "selected_model": selected_model,
        "b0_test_brier": float(b0["test_brier"]),
        "d0_test_brier": float(d0["test_brier"]),
        "d0_minus_b0_test_brier": float(
            d0["test_brier"] - b0["test_brier"]
        ),
        "selected_mean_pre_choice_ess_fraction": float(
            d0["mean_pre_choice_ess_fraction"]
        ),
        "selected_resampling_fraction": float(d0["resampling_fraction"]),
    }


def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    particle_counts = sorted(
        {int(row["particle_count"]) for row in selected}
    )
    out: dict[str, Any] = {"by_particle_count": {}}
    for particle_count in particle_counts:
        rows = [
            row
            for row in selected
            if int(row["particle_count"]) == particle_count
        ]
        true_dynamic = np.asarray(
            [str(row["true_model"]) == "D0" for row in rows], dtype=bool
        )
        selected_dynamic = np.asarray(
            [str(row["selected_model"]) == "D0" for row in rows], dtype=bool
        )
        errors = np.asarray(
            [
                float(row["estimated_theta"]) - float(row["true_theta"])
                for row in rows
            ],
            dtype=float,
        )
        positive_n = int(np.sum(true_dynamic))
        negative_n = int(np.sum(~true_dynamic))
        true_positive = int(np.sum(true_dynamic & selected_dynamic))
        true_negative = int(np.sum((~true_dynamic) & (~selected_dynamic)))
        out["by_particle_count"][str(particle_count)] = {
            "n_datasets": len(rows),
            "family_accuracy_count": int(
                np.sum(true_dynamic == selected_dynamic)
            ),
            "dynamic_sensitivity": (
                float(true_positive / positive_n)
                if positive_n
                else float("nan")
            ),
            "dynamic_specificity": (
                float(true_negative / negative_n)
                if negative_n
                else float("nan")
            ),
            "theta_bias": float(np.mean(errors)),
            "theta_rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mean_d0_minus_b0_test_brier": float(
                np.mean(
                    [
                        float(row["d0_minus_b0_test_brier"])
                        for row in rows
                    ]
                )
            ),
            "mean_pre_choice_ess_fraction": float(
                np.mean(
                    [
                        float(row["selected_mean_pre_choice_ess_fraction"])
                        for row in rows
                    ]
                )
            ),
            "mean_resampling_fraction": float(
                np.mean(
                    [
                        float(row["selected_resampling_fraction"])
                        for row in rows
                    ]
                )
            ),
            "confusion": {
                "B0_to_B0": true_negative,
                "B0_to_D0": int(np.sum((~true_dynamic) & selected_dynamic)),
                "D0_to_B0": int(np.sum(true_dynamic & (~selected_dynamic))),
                "D0_to_D0": true_positive,
            },
        }
    if len(particle_counts) >= 2:
        convergence = []
        lookup = {
            (
                int(row["subject_id"]),
                float(row["true_theta"]),
                int(row["dataset_replicate"]),
                int(row["filter_replicate"]),
                int(row["particle_count"]),
            ): row
            for row in selected
        }
        dataset_keys = sorted(
            {
                key[:4]
                for key in lookup
            }
        )
        for smaller, larger in zip(particle_counts[:-1], particle_counts[1:]):
            theta_agreement = []
            family_agreement = []
            for key in dataset_keys:
                small = lookup[(*key, smaller)]
                large = lookup[(*key, larger)]
                theta_agreement.append(
                    float(small["estimated_theta"])
                    == float(large["estimated_theta"])
                )
                family_agreement.append(
                    str(small["selected_model"])
                    == str(large["selected_model"])
                )
            convergence.append(
                {
                    "smaller_particles": smaller,
                    "larger_particles": larger,
                    "theta_agreement_count": int(sum(theta_agreement)),
                    "family_agreement_count": int(sum(family_agreement)),
                    "n": len(dataset_keys),
                }
            )
        out["particle_count_convergence"] = convergence
    return out


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    summary: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Condition-1 B0/D0 particle-filter recovery",
        "",
        f"- subjects: {', '.join(str(value) for value in args.subjects)}",
        f"- true theta: {', '.join(str(value) for value in args.true_theta_grid)}",
        f"- fit theta: {', '.join(str(value) for value in args.fit_theta_grid)}",
        f"- particle counts: {', '.join(str(value) for value in args.particle_counts)}",
        f"- datasets per theta: {args.datasets_per_theta}",
        "",
        "| particles | family accuracy | sensitivity | specificity | theta RMSE | mean ESS/R | resampling fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for particle_count in sorted(
        summary["by_particle_count"], key=lambda value: int(value)
    ):
        item = summary["by_particle_count"][particle_count]
        lines.append(
            f"| {particle_count} | {item['family_accuracy_count']}/"
            f"{item['n_datasets']} | {item['dynamic_sensitivity']:.3f} | "
            f"{item['dynamic_specificity']:.3f} | "
            f"{item['theta_rmse']:.4f} | "
            f"{item['mean_pre_choice_ess_fraction']:.3f} | "
            f"{item['mean_resampling_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Dataset-level recovery",
            "",
            "| subject | true theta | particles | filter rep | estimated theta | selected | D0−B0 test Brier |",
            "|---:|---:|---:|---:|---:|:---:|---:|",
        ]
    )
    for row in selected:
        lines.append(
            f"| {row['subject_id']} | {float(row['true_theta']):.3f} | "
            f"{row['particle_count']} | {row['filter_replicate']} | "
            f"{float(row['estimated_theta']):.3f} | "
            f"{row['selected_model']} | "
            f"{float(row['d0_minus_b0_test_brier']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.true_theta_grid = validate_probability_grid(
        args.true_theta_grid, "true_theta_grid"
    )
    args.fit_theta_grid = validate_probability_grid(
        args.fit_theta_grid, "fit_theta_grid"
    )
    if 0.0 not in args.true_theta_grid or 0.0 not in args.fit_theta_grid:
        raise ValueError("Both theta grids must contain the exact zero boundary.")
    args.particle_counts = sorted({int(value) for value in args.particle_counts})
    if any(value < 2 for value in args.particle_counts):
        raise ValueError("particle counts must be at least 2.")
    if (
        args.datasets_per_theta <= 0
        or args.filter_replicates <= 0
        or args.n_jobs <= 0
    ):
        raise ValueError("replicate counts and n_jobs must be positive.")

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
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "manifest.json",
        {
            "result_type": "cond1_newplan_b0_d0_particle_recovery",
            "subjects": args.subjects,
            "true_theta_grid": args.true_theta_grid,
            "fit_theta_grid": args.fit_theta_grid,
            "particle_counts": args.particle_counts,
            "datasets_per_theta": args.datasets_per_theta,
            "filter_replicates": args.filter_replicates,
            "resample_threshold": args.resample_threshold,
            "gamma": args.gamma,
            "w0": args.w0,
            "rho": args.rho,
            "max_trials": args.max_trials,
            "base_seed": args.base_seed,
        },
    )

    all_fit_rows = []
    for subject_id in args.subjects:
        for true_theta in args.true_theta_grid:
            for dataset_replicate in range(args.datasets_per_theta):
                token = dataset_token(
                    subject_id, true_theta, dataset_replicate
                )
                dataset_path = args.output_dir / "datasets" / f"{token}.npz"
                if dataset_path.exists() and not args.force:
                    dataset = load_dataset(dataset_path)
                    print(f"LOAD dataset={token}", flush=True)
                else:
                    print(f"GENERATE dataset={token}", flush=True)
                    dataset = generate_dataset(
                        args=args,
                        runner=runner,
                        base_engine=base_engine,
                        dataset_paths=dataset_paths,
                        subject_id=int(subject_id),
                        true_theta=float(true_theta),
                        dataset_replicate=int(dataset_replicate),
                    )
                    save_dataset(dataset_path, dataset)
                for particle_count in args.particle_counts:
                    for filter_replicate in range(args.filter_replicates):
                        resolved = {}
                        missing = []
                        for fit_theta in args.fit_theta_grid:
                            cache_path = cache_path_for(
                                args,
                                dataset,
                                fit_theta,
                                particle_count,
                                filter_replicate,
                            )
                            if cache_path.exists() and not args.force:
                                resolved[float(fit_theta)] = json.loads(
                                    cache_path.read_text(encoding="utf-8")
                                )
                            else:
                                missing.append(float(fit_theta))
                        computed = Parallel(
                            n_jobs=min(args.n_jobs, len(missing) or 1)
                        )(
                            delayed(fit_point)(
                                args=args,
                                runner=runner,
                                base_engine=base_engine,
                                dataset_paths=dataset_paths,
                                dataset=dataset,
                                fit_theta=fit_theta,
                                particle_count=int(particle_count),
                                filter_replicate=int(filter_replicate),
                            )
                            for fit_theta in missing
                        )
                        for fit_theta, row in zip(missing, computed):
                            cache_path = cache_path_for(
                                args,
                                dataset,
                                fit_theta,
                                particle_count,
                                filter_replicate,
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
                            resolved[float(theta)]
                            for theta in args.fit_theta_grid
                        )

    grouped: dict[tuple[int, float, int, int, int], list[Mapping[str, Any]]] = {}
    for row in all_fit_rows:
        key = (
            int(row["subject_id"]),
            float(row["true_theta"]),
            int(row["dataset_replicate"]),
            int(row["particle_count"]),
            int(row["filter_replicate"]),
        )
        grouped.setdefault(key, []).append(row)
    selected = [select_dataset(rows) for _, rows in sorted(grouped.items())]
    summary = summarize(selected)
    write_csv(args.output_dir / "fit_grid.csv", all_fit_rows)
    write_csv(args.output_dir / "recovery_rows.csv", selected)
    write_json(args.output_dir / "aggregate_summary.json", summary)
    write_report(
        args.output_dir / "RESULTS.md",
        args=args,
        summary=summary,
        selected=selected,
    )
    print(f"COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
