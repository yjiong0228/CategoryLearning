#!/usr/bin/env python3
"""B0-versus-D0 development comparison using an online particle filter."""

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
    DEFAULT_SUBJECTS,
    score_probabilities,
    split_masks,
    theta_token,
    validate_probability_grid,
    write_csv,
    write_json,
)
from src.Bayesian_state.run_simulation import (
    apply_fixed_hyperparams_to_engine_config,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
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
    parser.add_argument(
        "--subjects", type=int, nargs="+", default=list(DEFAULT_SUBJECTS)
    )
    parser.add_argument(
        "--theta-grid",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.50, 0.75, 1.0],
    )
    parser.add_argument(
        "--particle-counts", type=int, nargs="+", default=[32, 64, 128]
    )
    parser.add_argument("--filter-replicates", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--resample-threshold", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.55)
    parser.add_argument("--w0", type=float, default=0.10)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--base-seed", type=int, default=20260731)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/zhuran/cond1_newplan/particle_filter",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def params_for(args: argparse.Namespace, theta: float) -> dict[str, Any]:
    return {
        "engine.modules.hypo_transitions_mod.kwargs.theta": float(theta),
        "engine.modules.memory_mod.kwargs.gamma": float(args.gamma),
        "engine.modules.memory_mod.kwargs.w0": float(args.w0),
        "engine.choice_readout.kwargs": {
            "method": "sharpened_expectation",
            "power": float(args.rho),
        },
        "engine.output_noise.kwargs.base_lapse": 0.0,
    }


def filter_seed_for(
    args: argparse.Namespace,
    subject_id: int,
    replicate: int,
) -> int:
    return stable_seed(
        {
            "seed_role": "newplan_particle_filter_crn",
            "base_seed": int(args.base_seed),
            "subject_id": int(subject_id),
            "replicate": int(replicate),
            "max_trials": args.max_trials,
        }
    )


def cache_path_for(
    args: argparse.Namespace,
    subject_id: int,
    theta: float,
    particle_count: int,
    replicate: int,
) -> Path:
    return (
        args.output_dir
        / "cache"
        / f"subject_{int(subject_id)}"
        / f"particles_{int(particle_count)}"
        / f"replicate_{int(replicate)}"
        / f"theta_{theta_token(theta)}.npz"
    )


def save_cache(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        key: np.asarray(result[key])
        for key in (
            "marginal_probabilities",
            "pre_choice_ess",
            "post_choice_ess",
            "resampled",
            "resampling_unique_ancestors",
            "filtered_swap_probability",
            "filtered_swap_event_probability",
            "final_weights",
            "particle_swap_counts",
            "observed_choice_index",
            "valid_mask",
            "train_mask",
            "test_mask",
        )
    }
    metadata = {
        key: value
        for key, value in result.items()
        if key not in arrays
    }
    np.savez_compressed(
        path,
        **arrays,
        metadata=np.asarray(
            json.dumps(metadata, ensure_ascii=False, allow_nan=True)
        ),
    )


def load_cache(path: Path) -> dict[str, Any]:
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


def run_filter_point(
    *,
    args: argparse.Namespace,
    runner: StateModelSimulationRunner,
    base_engine: Mapping[str, Any],
    dataset_paths: Mapping[str, Path],
    subject_id: int,
    theta: float,
    particle_count: int,
    replicate: int,
) -> dict[str, Any]:
    subject_frame = runner._get_subject_frame(subject_id, 1.0)
    if args.max_trials is not None:
        subject_frame = subject_frame.iloc[: int(args.max_trials)].copy()
    arrays = runner._extract_arrays(subject_frame, None)
    valid_mask = np.ones(len(subject_frame), dtype=bool)
    valid_mask[0] = False
    train_mask, test_mask, split_status = split_masks(
        subject_frame, valid_mask
    )
    params = params_for(args, theta)
    engine_config = apply_fixed_hyperparams_to_engine_config(
        deepcopy(dict(base_engine)), params
    )
    filter_seed = filter_seed_for(args, subject_id, replicate)
    result = run_newplan_particle_filter(
        engine_config=engine_config,
        subject_id=int(subject_id),
        stimulus=arrays.stimulus,
        choices=arrays.choices,
        feedback=arrays.feedback,
        particle_count=int(particle_count),
        rho=float(args.rho),
        epsilon=0.0,
        filter_seed=int(filter_seed),
        resample_threshold_fraction=float(args.resample_threshold),
        valid_trial_mask=valid_mask,
        processed_data_dir=runner._processed_data_dir,
        dataset_paths=dataset_paths,
    )
    return {
        "subject_id": int(subject_id),
        "theta": float(theta),
        "particle_count": int(particle_count),
        "replicate": int(replicate),
        "filter_seed": int(filter_seed),
        "split_status": str(split_status),
        "resample_threshold": float(args.resample_threshold),
        "marginal_probabilities": result.marginal_probabilities,
        "pre_choice_ess": result.pre_choice_ess,
        "post_choice_ess": result.post_choice_ess,
        "resampled": result.resampled,
        "resampling_unique_ancestors": result.resampling_unique_ancestors,
        "filtered_swap_probability": result.filtered_swap_probability,
        "filtered_swap_event_probability": (
            result.filtered_swap_event_probability
        ),
        "final_weights": result.final_weights,
        "particle_swap_counts": result.particle_swap_counts,
        "observed_choice_index": np.asarray(arrays.choices, dtype=int) - 1,
        "valid_mask": valid_mask,
        "train_mask": train_mask,
        "test_mask": test_mask,
    }


def score_row(result: Mapping[str, Any]) -> dict[str, Any]:
    probabilities = np.asarray(result["marginal_probabilities"], dtype=float)
    choices = np.asarray(result["observed_choice_index"], dtype=int)
    scores = {
        split: score_probabilities(
            probabilities,
            choices,
            np.asarray(result[f"{split}_mask"], dtype=bool),
        )
        for split in ("train", "test")
    }
    post_ess = np.asarray(result["post_choice_ess"], dtype=float)
    pre_ess = np.asarray(result["pre_choice_ess"], dtype=float)
    resampled = np.asarray(result["resampled"], dtype=bool)
    unique = np.asarray(
        result["resampling_unique_ancestors"], dtype=int
    )
    particle_count = int(result["particle_count"])
    row = {
        "subject_id": int(result["subject_id"]),
        "theta": float(result["theta"]),
        "particle_count": particle_count,
        "replicate": int(result["replicate"]),
        "filter_seed": int(result["filter_seed"]),
        "split_status": str(result["split_status"]),
        "mean_pre_choice_ess_fraction": float(
            np.mean(pre_ess) / particle_count
        ),
        "mean_post_choice_ess_fraction": float(
            np.mean(post_ess) / particle_count
        ),
        "min_post_choice_ess_fraction": float(
            np.min(post_ess) / particle_count
        ),
        "resampling_count": int(np.sum(resampled)),
        "resampling_fraction": float(np.mean(resampled)),
        "mean_unique_ancestor_fraction": (
            float(np.mean(unique[resampled]) / particle_count)
            if np.any(resampled)
            else 1.0
        ),
        "mean_filtered_swap_event_probability": float(
            np.mean(result["filtered_swap_event_probability"])
        ),
    }
    for split, item in scores.items():
        for key, value in item.items():
            row[f"{split}_{key}"] = value
    return row


def select_models(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            int(row["subject_id"]),
            int(row["particle_count"]),
            int(row["replicate"]),
        )
        grouped.setdefault(key, []).append(row)
    selected = []
    for key, candidate_rows in sorted(grouped.items()):
        b0 = next(row for row in candidate_rows if float(row["theta"]) == 0.0)
        d0 = min(
            candidate_rows,
            key=lambda row: (
                float(row["train_choice_brier"]),
                float(row["theta"]) > 0.0,
                float(row["theta"]),
            ),
        )
        for model_id, source in (("B0", b0), ("D0", d0)):
            item = dict(source)
            item["model_id"] = model_id
            selected.append(item)
    return selected


def summarize(
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    selected_lookup = {
        (
            int(row["subject_id"]),
            int(row["particle_count"]),
            int(row["replicate"]),
            str(row["model_id"]),
        ): row
        for row in selected
    }
    particle_counts = sorted(
        {int(row["particle_count"]) for row in selected}
    )
    replicates = sorted({int(row["replicate"]) for row in selected})
    subjects = sorted({int(row["subject_id"]) for row in selected})
    by_particles: dict[str, Any] = {}
    for particle_count in particle_counts:
        deltas = []
        theta_values = []
        for replicate in replicates:
            for subject_id in subjects:
                b0 = selected_lookup[
                    (subject_id, particle_count, replicate, "B0")
                ]
                d0 = selected_lookup[
                    (subject_id, particle_count, replicate, "D0")
                ]
                if int(b0["test_n"]) > 0 and int(d0["test_n"]) > 0:
                    delta = (
                        float(d0["test_choice_brier"])
                        - float(b0["test_choice_brier"])
                    )
                    if np.isfinite(delta):
                        deltas.append(delta)
                theta_values.append(float(d0["theta"]))
        by_particles[str(particle_count)] = {
            "n_heldout_pairs": len(deltas),
            "mean_d0_minus_b0_test_brier": (
                float(np.mean(deltas)) if deltas else float("nan")
            ),
            "median_d0_minus_b0_test_brier": (
                float(np.median(deltas)) if deltas else float("nan")
            ),
            "improved_count": int(sum(value < 0.0 for value in deltas)),
            "selected_theta_values": theta_values,
            "mean_pre_choice_ess_fraction": float(
                np.mean(
                    [
                        float(row["mean_pre_choice_ess_fraction"])
                        for row in rows
                        if int(row["particle_count"]) == particle_count
                    ]
                )
            ),
            "mean_resampling_fraction": float(
                np.mean(
                    [
                        float(row["resampling_fraction"])
                        for row in rows
                        if int(row["particle_count"]) == particle_count
                    ]
                )
            ),
        }

    convergence = []
    for smaller, larger in zip(particle_counts[:-1], particle_counts[1:]):
        fixed_differences = []
        selection_agreement = []
        for replicate in replicates:
            for subject_id in subjects:
                for theta in sorted(
                    {
                        float(row["theta"])
                        for row in rows
                        if int(row["subject_id"]) == subject_id
                    }
                ):
                    small = next(
                        row
                        for row in rows
                        if int(row["subject_id"]) == subject_id
                        and int(row["particle_count"]) == smaller
                        and int(row["replicate"]) == replicate
                        and float(row["theta"]) == theta
                    )
                    large = next(
                        row
                        for row in rows
                        if int(row["subject_id"]) == subject_id
                        and int(row["particle_count"]) == larger
                        and int(row["replicate"]) == replicate
                        and float(row["theta"]) == theta
                    )
                    if np.isfinite(
                        float(small["test_choice_brier"])
                    ) and np.isfinite(
                        float(large["test_choice_brier"])
                    ):
                        fixed_differences.append(
                            abs(
                                float(small["test_choice_brier"])
                                - float(large["test_choice_brier"])
                            )
                        )
                small_d0 = selected_lookup[
                    (subject_id, smaller, replicate, "D0")
                ]
                large_d0 = selected_lookup[
                    (subject_id, larger, replicate, "D0")
                ]
                selection_agreement.append(
                    float(small_d0["theta"]) == float(large_d0["theta"])
                )
        convergence.append(
            {
                "smaller_particles": smaller,
                "larger_particles": larger,
                "fixed_cell_mean_absolute_test_brier_change": (
                    float(np.mean(fixed_differences))
                    if fixed_differences
                    else float("nan")
                ),
                "fixed_cell_max_absolute_test_brier_change": (
                    float(np.max(fixed_differences))
                    if fixed_differences
                    else float("nan")
                ),
                "theta_selection_agreement_count": int(
                    sum(selection_agreement)
                ),
                "theta_selection_agreement_n": len(selection_agreement),
            }
        )

    replicate_agreement = {}
    if len(replicates) >= 2:
        for particle_count in particle_counts:
            agreements = []
            for subject_id in subjects:
                values = [
                    float(
                        selected_lookup[
                            (
                                subject_id,
                                particle_count,
                                replicate,
                                "D0",
                            )
                        ]["theta"]
                    )
                    for replicate in replicates
                ]
                agreements.append(len(set(values)) == 1)
            replicate_agreement[str(particle_count)] = {
                "theta_agreement_count": int(sum(agreements)),
                "n": len(agreements),
            }
    return {
        "by_particle_count": by_particles,
        "particle_count_convergence": convergence,
        "independent_replicate_agreement": replicate_agreement,
    }


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    summary: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Condition-1 B0/D0 particle-filter result",
        "",
        f"- subjects: {', '.join(str(value) for value in args.subjects)}",
        f"- particle counts: {', '.join(str(value) for value in args.particle_counts)}",
        f"- independent filter replicates: {args.filter_replicates}",
        f"- resampling threshold: {args.resample_threshold}",
        f"- gamma={args.gamma}, w0={args.w0}, rho={args.rho}",
        f"- max_trials={args.max_trials}",
        "",
        "Negative D0−B0 held-out Brier favors swap-one.",
        "",
        "| particles | mean delta | median delta | improved | mean pre-choice ESS/R | resampling fraction |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for particle_count in sorted(
        summary["by_particle_count"], key=lambda value: int(value)
    ):
        item = summary["by_particle_count"][particle_count]
        lines.append(
            f"| {particle_count} | "
            f"{item['mean_d0_minus_b0_test_brier']:.6f} | "
            f"{item['median_d0_minus_b0_test_brier']:.6f} | "
            f"{item['improved_count']}/{item['n_heldout_pairs']} | "
            f"{item['mean_pre_choice_ess_fraction']:.3f} | "
            f"{item['mean_resampling_fraction']:.3f} |"
        )
    lines.extend(["", "## Particle-count convergence", ""])
    for item in summary["particle_count_convergence"]:
        lines.append(
            f"- {item['smaller_particles']}→{item['larger_particles']}: "
            f"fixed-cell mean |ΔBrier|="
            f"{item['fixed_cell_mean_absolute_test_brier_change']:.6f}, "
            f"theta agreement="
            f"{item['theta_selection_agreement_count']}/"
            f"{item['theta_selection_agreement_n']}"
        )
    lines.extend(
        [
            "",
            "## Selected D0 theta",
            "",
            "| subject | particles | replicate | theta | train Brier | test Brier |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selected:
        if str(row["model_id"]) != "D0":
            continue
        lines.append(
            f"| {row['subject_id']} | {row['particle_count']} | "
            f"{row['replicate']} | {float(row['theta']):.3f} | "
            f"{float(row['train_choice_brier']):.6f} | "
            f"{float(row['test_choice_brier']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.theta_grid = validate_probability_grid(
        args.theta_grid, "theta_grid"
    )
    args.particle_counts = sorted({int(value) for value in args.particle_counts})
    if any(value < 2 for value in args.particle_counts):
        raise ValueError("particle counts must be at least 2.")
    if args.filter_replicates <= 0 or args.n_jobs <= 0:
        raise ValueError("filter_replicates and n_jobs must be positive.")

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
            "result_type": "cond1_newplan_b0_d0_particle_filter",
            "subjects": args.subjects,
            "theta_grid": args.theta_grid,
            "particle_counts": args.particle_counts,
            "filter_replicates": args.filter_replicates,
            "n_jobs": args.n_jobs,
            "resample_threshold": args.resample_threshold,
            "gamma": args.gamma,
            "w0": args.w0,
            "rho": args.rho,
            "max_trials": args.max_trials,
            "base_seed": args.base_seed,
            "common_random_numbers_across_theta": True,
            "parameters_excluded_from_filter_seed": [
                "theta",
                "gamma",
                "w0",
                "rho",
            ],
        },
    )

    detailed = []
    for particle_count in args.particle_counts:
        for replicate in range(args.filter_replicates):
            for subject_id in args.subjects:
                resolved: dict[float, dict[str, Any]] = {}
                missing = []
                for theta in args.theta_grid:
                    cache_path = cache_path_for(
                        args,
                        subject_id,
                        theta,
                        particle_count,
                        replicate,
                    )
                    if cache_path.exists() and not args.force:
                        print(
                            f"LOAD subject={subject_id} theta={theta:.3f} "
                            f"R={particle_count} rep={replicate}",
                            flush=True,
                        )
                        resolved[float(theta)] = load_cache(cache_path)
                    else:
                        print(
                            f"RUN subject={subject_id} theta={theta:.3f} "
                            f"R={particle_count} rep={replicate}",
                            flush=True,
                        )
                        missing.append(float(theta))
                computed = Parallel(n_jobs=min(args.n_jobs, len(missing) or 1))(
                    delayed(run_filter_point)(
                            args=args,
                            runner=runner,
                            base_engine=base_engine,
                            dataset_paths=dataset_paths,
                            subject_id=int(subject_id),
                            theta=float(theta),
                            particle_count=int(particle_count),
                            replicate=int(replicate),
                        )
                    for theta in missing
                )
                for theta, result in zip(missing, computed):
                    save_cache(
                        cache_path_for(
                            args,
                            subject_id,
                            theta,
                            particle_count,
                            replicate,
                        ),
                        result,
                    )
                    resolved[float(theta)] = result
                detailed.extend(
                    resolved[float(theta)] for theta in args.theta_grid
                )

    rows = [score_row(item) for item in detailed]
    selected = select_models(rows)
    summary = summarize(rows, selected)
    write_csv(args.output_dir / "parameter_grid.csv", rows)
    write_csv(args.output_dir / "selected_models.csv", selected)
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
