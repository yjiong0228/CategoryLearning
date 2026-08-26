#!/usr/bin/env python3
"""Run a Model 0818 nested-B particle-filter convergence analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0818_boundary_recovery import (  # noqa: E402
    _atomic_csv,
    _atomic_json,
    _atomic_npz,
    _readout_args,
    _repo_path,
    _sha256,
    _subject_engine,
    build_boundary_profiles,
    resolve_filter_seeds,
)
from src.Bayesian_state.inference.backends.particle_filter import (  # noqa: E402
    run_state_model_particle_filter,
)
from src.Bayesian_state.optimization.parameter_space import (  # noqa: E402
    load_parameter_space,
)
from src.Bayesian_state.optimization.seed_convergence import (  # noqa: E402
    bootstrap_pairwise_delta_nll,
    evaluate_seed_convergence,
    summarize_independent_seed_halves,
    summarize_nested_seed_budgets,
)
from src.Bayesian_state.simulation.config import load_yaml  # noqa: E402
from src.Bayesian_state.simulation.parameters import (  # noqa: E402
    apply_fixed_hyperparams_to_engine_config,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.utils.seeding import stable_seed  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/specific_models/model_0818_seed_convergence.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument(
        "--phase", choices=("run", "summarize", "all"), default="all"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _profile_sha256(profile: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(profile), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seed_cache_path(
    output: Path,
    cache_namespace: str,
    dataset_id: str,
    profile_id: str,
    filter_seed: int,
) -> Path:
    return (
        output
        / "cache"
        / "per_seed"
        / str(cache_namespace)
        / str(dataset_id)
        / str(profile_id)
        / f"seed_{int(filter_seed)}.npz"
    )


def build_seed_jobs(
    *,
    datasets: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    particle_count: int,
    filter_seed_count: int,
    base_seed: int,
    seed_family: str,
) -> list[dict[str, Any]]:
    """Create one candidate-paired job per dataset/profile/filter seed."""

    jobs: list[dict[str, Any]] = []
    for dataset in sorted(datasets, key=lambda value: str(value["dataset_id"])):
        dataset_id = str(dataset["dataset_id"])
        seeds = resolve_filter_seeds(
            dataset_id=dataset_id,
            base_seed=int(base_seed),
            particle_count=int(particle_count),
            filter_seed_count=int(filter_seed_count),
            seed_family=str(seed_family),
        )
        for profile in profiles:
            for seed_index, filter_seed in enumerate(seeds):
                jobs.append(
                    {
                        "dataset": dict(dataset),
                        "profile": dict(profile),
                        "seed_index": int(seed_index),
                        "filter_seed": int(filter_seed),
                    }
                )
    return jobs


def _validate_run_arrays(
    probabilities: np.ndarray,
    pre_choice_ess: np.ndarray,
    resampled: np.ndarray,
    *,
    trial_count: int,
    particle_count: int,
) -> None:
    if probabilities.shape != (trial_count, 2):
        raise ValueError("cached PF probabilities have an unexpected shape")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("PF probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("PF choice probabilities must sum to one")
    if pre_choice_ess.shape != (trial_count,) or not np.all(
        np.isfinite(pre_choice_ess)
    ):
        raise ValueError("pre-choice ESS is invalid")
    if np.any(pre_choice_ess <= 0.0) or np.any(
        pre_choice_ess > float(particle_count) + 1e-8
    ):
        raise ValueError("pre-choice ESS lies outside the particle budget")
    if resampled.shape != (trial_count,):
        raise ValueError("resampling indicator has an unexpected shape")


def _read_seed_cache(
    path: Path, expected_metadata: Mapping[str, Any]
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = dict(json.loads(str(payload["metadata_json"].item())))
        probabilities = payload["marginal_probabilities"].astype(float)
        pre_choice_ess = payload["pre_choice_ess"].astype(float)
        resampled = payload["resampled"].astype(bool)
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"seed cache metadata mismatch for {key}: {path}")
    _validate_run_arrays(
        probabilities,
        pre_choice_ess,
        resampled,
        trial_count=int(metadata["trial_count"]),
        particle_count=int(metadata["particle_count"]),
    )
    return metadata, probabilities, pre_choice_ess, resampled


def run_seed_job(
    *,
    output: Path,
    source_pilot: Path,
    base_config: Mapping[str, Any],
    base_path: Path,
    dataset_paths: Mapping[str, Path],
    job: Mapping[str, Any],
    particle_count: int,
    resample_threshold_fraction: float,
    seed_family: str,
    cache_namespace: str,
    source_synthetic_sha256: str,
    force: bool,
) -> dict[str, Any]:
    dataset = dict(job["dataset"])
    profile = dict(job["profile"])
    dataset_id = str(dataset["dataset_id"])
    profile_id = str(profile["profile_id"])
    subject_id = int(dataset["subject_id"])
    seed_index = int(job["seed_index"])
    filter_seed = int(job["filter_seed"])
    profile_sha256 = _profile_sha256(profile)
    cache_path = seed_cache_path(
        output, cache_namespace, dataset_id, profile_id, filter_seed
    )
    expected_metadata = {
        "format_version": 1,
        "dataset_id": dataset_id,
        "subject_id": subject_id,
        "fit_profile_id": profile_id,
        "seed_index": seed_index,
        "filter_seed": filter_seed,
        "seed_family": str(seed_family),
        "particle_count": int(particle_count),
        "source_synthetic_sha256": str(source_synthetic_sha256),
        "profile_sha256": profile_sha256,
        "probability_aggregation_role": "one_unaggregated_filter_seed",
        "observed_choices_used": False,
    }
    if cache_path.exists() and not force:
        metadata, _, _, _ = _read_seed_cache(cache_path, expected_metadata)
        row = dict(metadata)
        row["cache_hit"] = True
        row["cache_relpath"] = str(cache_path.relative_to(output))
        return row

    synthetic_path = source_pilot / "synthetic" / f"{dataset_id}.npz"
    with np.load(synthetic_path, allow_pickle=False) as payload:
        stimulus = payload["stimulus"].astype(float)
        choices = payload["choices"].astype(int)
        feedback = payload["feedback"].astype(float)
        source_metadata = dict(
            json.loads(str(payload["metadata_json"].item()))
        )
    if bool(source_metadata.get("observed_choices_used", True)):
        raise ValueError("seed convergence requires autonomous synthetic choices")
    if str(source_metadata["dataset_id"]) != dataset_id:
        raise ValueError("source synthetic metadata does not match dataset id")
    engine = apply_fixed_hyperparams_to_engine_config(
        _subject_engine(base_config, base_path, subject_id),
        profile["hyperparams"],
    )
    started = time.perf_counter()
    result = run_state_model_particle_filter(
        engine_config=engine,
        subject_id=subject_id,
        stimulus=stimulus,
        choices=choices,
        feedback=feedback,
        particle_count=int(particle_count),
        filter_seed=filter_seed,
        resample_threshold_fraction=float(resample_threshold_fraction),
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
        **_readout_args(engine),
    )
    elapsed_seconds = float(time.perf_counter() - started)
    probabilities = np.asarray(result.marginal_probabilities, dtype=float)
    pre_choice_ess = np.asarray(result.pre_choice_ess, dtype=float)
    resampled = np.asarray(result.resampled, dtype=bool)
    _validate_run_arrays(
        probabilities,
        pre_choice_ess,
        resampled,
        trial_count=int(choices.size),
        particle_count=int(particle_count),
    )
    selected = probabilities[np.arange(choices.size), choices - 1]
    if np.any(selected <= 0.0):
        raise ValueError("seed run produced an invalid observed-choice probability")
    total_nll = float(-np.log(np.clip(selected, 1e-12, 1.0)).sum())
    metadata = {
        **expected_metadata,
        "true_profile_id": str(dataset["true_profile_id"]),
        "trial_count": int(choices.size),
        "total_nll_single_seed": total_nll,
        "mean_trial_nll_single_seed": total_nll / float(choices.size),
        "mean_pre_choice_ess": float(np.mean(pre_choice_ess)),
        "resampling_fraction": float(np.mean(resampled)),
        "elapsed_seconds": elapsed_seconds,
    }
    _atomic_npz(
        cache_path,
        marginal_probabilities=probabilities.astype(np.float64),
        pre_choice_ess=pre_choice_ess.astype(np.float64),
        resampled=resampled.astype(np.uint8),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    row = dict(metadata)
    row["cache_hit"] = False
    row["cache_relpath"] = str(cache_path.relative_to(output))
    return row


def _load_probability_banks(
    *,
    output: Path,
    cache_namespace: str,
    dataset: Mapping[str, Any],
    profiles: Sequence[Mapping[str, Any]],
    jobs: Sequence[Mapping[str, Any]],
    particle_count: int,
    seed_family: str,
    source_hashes: Mapping[str, str],
) -> tuple[dict[str, np.ndarray], np.ndarray, list[int]]:
    dataset_id = str(dataset["dataset_id"])
    source_path = Path(str(dataset["synthetic_path"]))
    with np.load(source_path, allow_pickle=False) as payload:
        choices = payload["choices"].astype(int)
    dataset_jobs = [
        value
        for value in jobs
        if str(value["dataset"]["dataset_id"]) == dataset_id
    ]
    probabilities_by_candidate: dict[str, np.ndarray] = {}
    reference_seeds: list[int] | None = None
    for profile in profiles:
        profile_id = str(profile["profile_id"])
        profile_jobs = sorted(
            (
                value
                for value in dataset_jobs
                if str(value["profile"]["profile_id"]) == profile_id
            ),
            key=lambda value: int(value["seed_index"]),
        )
        seeds = [int(value["filter_seed"]) for value in profile_jobs]
        if reference_seeds is None:
            reference_seeds = seeds
        elif seeds != reference_seeds:
            raise ValueError("candidate profiles do not share paired filter seeds")
        probability_runs: list[np.ndarray] = []
        for job in profile_jobs:
            expected_metadata = {
                "format_version": 1,
                "dataset_id": dataset_id,
                "subject_id": int(dataset["subject_id"]),
                "fit_profile_id": profile_id,
                "seed_index": int(job["seed_index"]),
                "filter_seed": int(job["filter_seed"]),
                "seed_family": str(seed_family),
                "particle_count": int(particle_count),
                "source_synthetic_sha256": str(source_hashes[dataset_id]),
                "profile_sha256": _profile_sha256(profile),
                "probability_aggregation_role": "one_unaggregated_filter_seed",
                "observed_choices_used": False,
            }
            path = seed_cache_path(
                output,
                cache_namespace,
                dataset_id,
                profile_id,
                int(job["filter_seed"]),
            )
            if not path.exists():
                raise FileNotFoundError(f"missing per-seed cache: {path}")
            _, probabilities, _, _ = _read_seed_cache(path, expected_metadata)
            probability_runs.append(probabilities)
        probabilities_by_candidate[profile_id] = np.stack(
            probability_runs, axis=0
        )
    if reference_seeds is None:
        raise ValueError("no filter seeds were loaded")
    return probabilities_by_candidate, choices, reference_seeds


def _write_report(output: Path, summary: Mapping[str, Any]) -> None:
    observed = dict(summary["observed"])
    halves = dict(summary["independent_half_diagnostic"])
    comparison = dict(summary["comparison"])
    from_count = int(comparison["from_B"])
    to_count = int(comparison["to_B"])
    half_count = int(summary["filter_seed_count"]) // 2
    lines = [
        "# Model 0818 nested-seed convergence",
        "",
        f"- Status: `{summary['budget_status']}`",
        f"- Per-seed PF tasks: {summary['completed_task_count']}",
        f"- R/B maximum: R{summary['particle_count']}/B{summary['filter_seed_count']}",
        (
            f"- Maximum |NLL(B{to_count}) - NLL(B{from_count})|: "
            f"{observed['maximum_absolute_candidate_nll_change']:.6f} "
            f"(gate {summary['gates']['maximum_absolute_candidate_nll_change']:.6f})"
        ),
        (
            "- Maximum pairwise delta-NLL CI half-width: "
            f"{observed['maximum_pairwise_delta_nll_ci_half_width']:.6f} "
            f"(gate {summary['gates']['maximum_pairwise_delta_nll_ci_half_width']:.6f})"
        ),
        (
            f"- Independent B{half_count}-half winner agreement: "
            f"{halves['winner_agreement']:.3f}"
        ),
        f"- Formal recovery authorized: `{str(summary['formal_recovery_authorized']).lower()}`",
        "- Observed-data fitting authorized: `false`",
        "",
        "This analysis isolates particle-filter seed averaging on two fixed autonomous "
        "synthetic trajectories. It is a numerical diagnostic, not recovery evidence.",
    ]
    (output / "seed_convergence_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")

    config_path = args.config.resolve()
    config = load_yaml(config_path)
    parameter_space_path = _repo_path(config["parameter_space"])
    recovery_config_path = _repo_path(config["boundary_recovery_config"])
    base_path = _repo_path(config["base_simulation_config"])
    source_pilot = _repo_path(config["source_pilot_dir"])
    configured_output = _repo_path(config["output_dir"])
    output = args.output_dir.resolve() if args.output_dir else configured_output
    design = dict(config["design"])
    particle_count = int(design["particle_count"])
    filter_seed_count = int(design["filter_seed_count"])
    checkpoints = [int(value) for value in design["nested_checkpoints"]]
    n_jobs = int(args.n_jobs if args.n_jobs is not None else design["n_jobs"])
    if n_jobs < 1 or n_jobs > (os.cpu_count() or 1):
        raise ValueError("n_jobs must lie within the available logical CPU count")
    if checkpoints[-1] != filter_seed_count:
        raise ValueError("the final nested checkpoint must equal filter_seed_count")

    parameter_space = load_parameter_space(parameter_space_path)
    recovery_config = load_yaml(recovery_config_path)
    base_config = load_yaml(base_path)
    profiles = build_boundary_profiles(
        parameter_space, recovery_config["positive_probes"]
    )
    dataset_paths = resolve_dataset_paths(base_config, base_path.parent)
    source_manifest_path = source_pilot / "synthetic_manifest.json"
    source_manifest = list(
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
    )
    requested_ids = [str(value) for value in config["dataset_ids"]]
    manifest_by_id = {
        str(value["dataset_id"]): dict(value) for value in source_manifest
    }
    if len(requested_ids) != len(set(requested_ids)) or not set(
        requested_ids
    ).issubset(manifest_by_id):
        raise ValueError("configured seed-convergence datasets are invalid")
    datasets: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    for dataset_id in requested_ids:
        metadata = dict(manifest_by_id[dataset_id])
        if bool(metadata.get("observed_choices_used", True)):
            raise ValueError("source manifest contains observed human choices")
        synthetic_path = source_pilot / "synthetic" / f"{dataset_id}.npz"
        if not synthetic_path.exists():
            raise FileNotFoundError(synthetic_path)
        metadata["synthetic_path"] = str(synthetic_path)
        datasets.append(metadata)
        source_hashes[dataset_id] = _sha256(synthetic_path)

    base_seed = int(base_config["hyper_base_seed"])
    jobs = build_seed_jobs(
        datasets=datasets,
        profiles=profiles,
        particle_count=particle_count,
        filter_seed_count=filter_seed_count,
        base_seed=base_seed,
        seed_family=str(design["seed_family"]),
    )
    expected_task_count = int(config["execution"]["expected_task_count"])
    if len(jobs) != expected_task_count or len(jobs) != (
        len(datasets) * len(profiles) * filter_seed_count
    ):
        raise ValueError("per-seed task count does not match the declared design")
    unique_jobs = {
        (
            str(value["dataset"]["dataset_id"]),
            str(value["profile"]["profile_id"]),
            int(value["filter_seed"]),
        )
        for value in jobs
    }
    if len(unique_jobs) != len(jobs):
        raise ValueError("per-seed task keys must be unique")

    output.mkdir(parents=True, exist_ok=True)
    manuscript_path = ROOT / "manuscript/model_0818.tex"
    engine_path_raw = Path(str(base_config["engine_config_path"]))
    engine_path = (
        engine_path_raw.resolve()
        if engine_path_raw.is_absolute()
        else (base_path.parent / engine_path_raw).resolve()
    )
    manifest = {
        "analysis_id": str(config["analysis_id"]),
        "scope": str(config["scope"]),
        "status": "configured",
        "resolved_output_dir": str(output),
        "source_pilot_dir": str(source_pilot),
        "source_pilot_manifest_sha256": _sha256(source_manifest_path),
        "source_synthetic_sha256": source_hashes,
        "dataset_ids": requested_ids,
        "profile_ids": [str(value["profile_id"]) for value in profiles],
        "particle_count": particle_count,
        "filter_seed_count": filter_seed_count,
        "nested_checkpoints": checkpoints,
        "seed_family": str(design["seed_family"]),
        "n_jobs": n_jobs,
        "expected_task_count": expected_task_count,
        "parallel_unit": str(config["execution"]["parallel_unit"]),
        "numerical_library_threads_per_process": int(
            config["execution"]["numerical_library_threads_per_process"]
        ),
        "observed_choices_used": False,
        "choices_source": "fixed_autonomous_synthetic_trajectories",
        "probability_aggregation": str(design["probability_aggregation"]),
        "parameter_space_sha256": _sha256(parameter_space_path),
        "boundary_recovery_config_sha256": _sha256(recovery_config_path),
        "base_simulation_config_sha256": _sha256(base_path),
        "engine_config_sha256": _sha256(engine_path),
        "manuscript_sha256": _sha256(manuscript_path),
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }
    _atomic_json(output / "analysis_manifest.json", manifest)
    _atomic_json(output / "analysis_config_snapshot.json", config)
    _atomic_json(output / "candidate_profiles.json", profiles)
    _atomic_json(output / "source_dataset_manifest.json", datasets)

    run_rows: list[dict[str, Any]]
    if args.phase in {"run", "all"}:
        run_rows = Parallel(
            n_jobs=min(n_jobs, len(jobs)),
            backend="loky",
            batch_size=1,
            pre_dispatch="n_jobs",
            verbose=10,
        )(
            delayed(run_seed_job)(
                output=output,
                source_pilot=source_pilot,
                base_config=base_config,
                base_path=base_path,
                dataset_paths=dataset_paths,
                job=job,
                particle_count=particle_count,
                resample_threshold_fraction=float(
                    design["resample_threshold_fraction"]
                ),
                seed_family=str(design["seed_family"]),
                cache_namespace=str(design["cache_namespace"]),
                source_synthetic_sha256=source_hashes[
                    str(job["dataset"]["dataset_id"])
                ],
                force=bool(args.force),
            )
            for job in jobs
        )
        run_index = pd.DataFrame(run_rows).sort_values(
            ["dataset_id", "fit_profile_id", "seed_index"]
        )
        if len(run_index) != expected_task_count:
            raise ValueError("per-seed run index is incomplete")
        _atomic_csv(output / "seed_run_index.csv", run_index)
        print(
            f"[0818 seed convergence] completed_seed_tasks={len(run_index)}",
            flush=True,
        )
        if args.phase == "run":
            return
    elif not (output / "seed_run_index.csv").exists():
        raise FileNotFoundError("seed_run_index.csv is required before summarizing")

    checkpoint_frames: list[pd.DataFrame] = []
    pairwise_frames: list[pd.DataFrame] = []
    half_frames: list[pd.DataFrame] = []
    half_summaries: list[dict[str, Any]] = []
    for dataset in datasets:
        dataset_id = str(dataset["dataset_id"])
        banks, choices, seeds = _load_probability_banks(
            output=output,
            cache_namespace=str(design["cache_namespace"]),
            dataset=dataset,
            profiles=profiles,
            jobs=jobs,
            particle_count=particle_count,
            seed_family=str(design["seed_family"]),
            source_hashes=source_hashes,
        )
        if len(seeds) != filter_seed_count or len(set(seeds)) != len(seeds):
            raise ValueError("nested seed bank is incomplete or duplicated")
        checkpoints_frame = summarize_nested_seed_budgets(
            banks, choices, checkpoints
        )
        checkpoints_frame.insert(0, "dataset_id", dataset_id)
        checkpoints_frame.insert(1, "true_profile_id", dataset["true_profile_id"])
        checkpoints_frame["particle_count"] = particle_count
        checkpoint_frames.append(checkpoints_frame)

        half_frame, half_summary = summarize_independent_seed_halves(
            banks, choices
        )
        half_frame.insert(0, "dataset_id", dataset_id)
        half_frames.append(half_frame)
        half_summaries.append({"dataset_id": dataset_id, **half_summary})

        bootstrap_seed = stable_seed(
            {
                "seed_role": "model0818_seed_convergence_bootstrap",
                "base_seed": int(config["bootstrap"]["base_seed"]),
                "dataset_id": dataset_id,
            }
        )
        pairwise = bootstrap_pairwise_delta_nll(
            banks,
            choices,
            replicates=int(config["bootstrap"]["replicates"]),
            confidence_level=float(config["bootstrap"]["confidence_level"]),
            bootstrap_seed=int(bootstrap_seed),
        )
        pairwise.insert(0, "dataset_id", dataset_id)
        pairwise_frames.append(pairwise)

    checkpoint_scores = pd.concat(checkpoint_frames, ignore_index=True)
    pairwise_intervals = pd.concat(pairwise_frames, ignore_index=True)
    independent_half_scores = pd.concat(half_frames, ignore_index=True)
    half_summary_frame = pd.DataFrame(half_summaries)
    changes, summary = evaluate_seed_convergence(
        checkpoint_scores,
        pairwise_intervals,
        dict(config["convergence_gates"]),
    )
    run_index = pd.read_csv(output / "seed_run_index.csv")
    summary.update(
        {
            "analysis_id": str(config["analysis_id"]),
            "dataset_n": len(datasets),
            "profile_count": len(profiles),
            "particle_count": particle_count,
            "filter_seed_count": filter_seed_count,
            "completed_task_count": int(len(run_index)),
            "all_seed_scores_finite": bool(
                np.all(np.isfinite(run_index["total_nll_single_seed"]))
            ),
            "cache_hit_count_this_invocation": int(
                run_index["cache_hit"].astype(bool).sum()
            ),
            "sum_single_pf_wall_seconds": float(
                run_index["elapsed_seconds"].sum()
            ),
            "maximum_single_pf_wall_seconds": float(
                run_index["elapsed_seconds"].max()
            ),
            "independent_half_diagnostic": {
                "dataset_n": int(len(half_summary_frame)),
                "winner_agreement": float(
                    half_summary_frame["same_winner"].astype(bool).mean()
                ),
                "maximum_absolute_candidate_nll_difference": float(
                    half_summary_frame[
                        "maximum_absolute_candidate_nll_difference"
                    ].max()
                ),
                "maximum_absolute_candidate_delta_nll_difference": float(
                    half_summary_frame[
                        "maximum_absolute_candidate_delta_nll_difference"
                    ].max()
                ),
            },
        }
    )
    checkpoint_winners = (
        checkpoint_scores.sort_values(["total_nll", "fit_profile_id"])
        .groupby(["dataset_id", "filter_seed_count"], as_index=False)
        .first()[
            [
                "dataset_id",
                "filter_seed_count",
                "true_profile_id",
                "fit_profile_id",
                "total_nll",
            ]
        ]
        .rename(columns={"fit_profile_id": "winning_profile_id"})
    )
    _atomic_csv(output / "nested_budget_scores.csv", checkpoint_scores)
    _atomic_csv(output / "nested_budget_winners.csv", checkpoint_winners)
    comparison = dict(summary["comparison"])
    change_filename = (
        f"b{int(comparison['from_B'])}_to_b{int(comparison['to_B'])}"
        "_candidate_changes.csv"
    )
    _atomic_csv(output / change_filename, changes)
    _atomic_csv(output / "pairwise_delta_nll_bootstrap.csv", pairwise_intervals)
    _atomic_csv(output / "independent_half_scores.csv", independent_half_scores)
    _atomic_csv(output / "independent_half_summary.csv", half_summary_frame)
    _atomic_json(output / "seed_convergence_summary.json", summary)
    _write_report(output, summary)
    manifest.update(
        {
            "status": "seed_convergence_complete",
            "budget_status": str(summary["budget_status"]),
            "formal_recovery_authorized": False,
            "observed_data_fit_authorized": False,
        }
    )
    _atomic_json(output / "analysis_manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
