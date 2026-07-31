#!/usr/bin/env python3
"""Hierarchical one-mechanism screen around the frozen condition-1 C1 model.

For every mechanism family, subject, readout mode, and candidate value, the
script conditions a particle filter on the observed prefix and autonomously
generates the suffix.  Development subjects estimate a finite-mixture prior
over candidate values.  Reserved subjects must use that frozen prior.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cond1_b0_trajectory_ppc import (  # noqa: E402
    DEVELOPMENT_SUBJECTS,
    KEY_COLUMNS,
    evaluate_subject,
    load_subject_cache,
    simulate_subject,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.utils.newplan_mechanism_variants import (  # noqa: E402
    MechanismCandidate,
    apply_candidate,
    candidates_for_family,
)
from src.Bayesian_state.utils.optimization_config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    load_yaml,
)
from src.Bayesian_state.utils.optimizer_common import stable_seed  # noqa: E402


READOUT_STATIC = "static"
READOUT_C1 = "c1"
VALID_READOUTS = (READOUT_STATIC, READOUT_C1)
REQUIRED_COLUMNS = (
    "iSub",
    "condition",
    "iSession",
    "iBlock",
    "iTrial",
    "feature1",
    "feature2",
    "feature3",
    "feature4",
    "category",
    "choice",
    "feedback",
    "choRT",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/processed/Task2_processed.csv",
    )
    parser.add_argument(
        "--engine-config",
        type=Path,
        default=(
            ROOT
            / "configs/model_struct/pmh_model_cond1_mechanism_screen.yaml"
        ),
    )
    parser.add_argument(
        "--simulation-config",
        type=Path,
        default=ROOT / "configs/simulation_cfg/pmh_cond1_simulation_v14.yaml",
    )
    parser.add_argument(
        "--cohort",
        choices=("development", "reserved", "all"),
        default="development",
    )
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument(
        "--families",
        choices=("F", "M", "H", "P", "S"),
        nargs="+",
        default=("F", "M", "H", "P"),
    )
    parser.add_argument(
        "--readout-modes",
        choices=VALID_READOUTS,
        nargs="+",
        default=VALID_READOUTS,
    )
    parser.add_argument("--shared-theta", type=float, default=0.75)
    parser.add_argument("--strategy-capacity", type=int, default=5)
    parser.add_argument("--particle-count", type=int, default=64)
    parser.add_argument("--rollout-count", type=int, default=128)
    parser.add_argument("--n-jobs", type=int, default=96)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=20261201)
    parser.add_argument("--em-alpha", type=float, default=1.0)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument(
        "--frozen-priors",
        type=Path,
        help="Development population_priors.csv required for reserved runs.",
    )
    parser.add_argument(
        "--mode",
        choices=("simulate", "summarize", "all"),
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_newplan/mechanism_screen_dev_v1"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=True)
        handle.write("\n")


def _validate_condition1_data(data: pd.DataFrame) -> dict[str, Any]:
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Task2 data are missing required columns: {missing_columns}")
    condition1 = data.loc[data["condition"].eq(1)].copy()
    if condition1.empty:
        raise ValueError("Task2 data contain no condition-1 rows.")
    null_counts = condition1[list(REQUIRED_COLUMNS)].isna().sum()
    duplicate_n = int(condition1.duplicated(list(KEY_COLUMNS)).sum())
    invalid_category_n = int((~condition1["category"].isin([1, 2])).sum())
    invalid_choice_n = int((~condition1["choice"].isin([1, 2])).sum())
    expected_feedback = condition1["choice"].eq(condition1["category"]).astype(float)
    feedback_mismatch_n = int(condition1["feedback"].ne(expected_feedback).sum())
    nonpositive_rt_n = int((condition1["choRT"] <= 0.0).sum())
    blockers = {
        "required_column_null_n": int(null_counts.sum()),
        "duplicate_trial_key_n": duplicate_n,
        "invalid_category_n": invalid_category_n,
        "invalid_choice_n": invalid_choice_n,
        "feedback_mismatch_n": feedback_mismatch_n,
        "nonpositive_rt_n": nonpositive_rt_n,
    }
    if any(value > 0 for value in blockers.values()):
        raise ValueError(f"Condition-1 data quality blockers: {blockers}")
    trial_counts = condition1.groupby("iSub", sort=True).size()
    rt_quantiles = condition1["choRT"].quantile(
        [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0]
    )
    return {
        "assessment": "ready_for_model_screen",
        "row_n": int(len(condition1)),
        "subject_n": int(condition1["iSub"].nunique()),
        "subject_ids": sorted(condition1["iSub"].astype(int).unique().tolist()),
        "trial_count_min": int(trial_counts.min()),
        "trial_count_median": float(trial_counts.median()),
        "trial_count_max": int(trial_counts.max()),
        "null_counts": {key: int(value) for key, value in null_counts.items()},
        **blockers,
        "rt_quantiles_seconds": {
            str(key): float(value) for key, value in rt_quantiles.items()
        },
        "rt_validation_transform": "log_seconds_with_robust_statistics",
    }


def _candidate_args(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    readout_mode: str,
    beta_additive: float,
) -> SimpleNamespace:
    dynamic = str(readout_mode) == READOUT_C1
    return SimpleNamespace(
        output_dir=output_dir,
        particle_count=int(args.particle_count),
        rollout_count=int(args.rollout_count),
        force=bool(args.force),
        split_mode="early_anchor",
        window=int(args.window),
        base_seed=int(args.base_seed),
        beta_correct_additive=float(beta_additive),
        lapse_start=0.0,
        learning_update_probability=1.0,
        beta_additive_grid=None,
        lapse_start_grid=None,
        learning_update_grid=None,
        selection_particle_count=8,
        lapse_half_life=128.0,
        rho=0.5,
        resample_threshold=0.5,
        acquisition_half_life=None,
        pre_acquisition_lapse=0.0,
        dynamic_rho_start=0.5 if dynamic else None,
        dynamic_rho_end=0.5,
        dynamic_rho_volatility=0.2,
        dynamic_rho_persistence=0.95,
        dynamic_rho_start_log_sd=0.35,
        dynamic_rho_gain_log_sd=0.35,
        dynamic_rho_volatility_log_sd=0.50,
        dynamic_rho_reference_trials=128,
    )


def _subject_ids(args: argparse.Namespace, data: pd.DataFrame) -> list[int]:
    available = sorted(data["iSub"].astype(int).unique().tolist())
    development = set(int(value) for value in DEVELOPMENT_SUBJECTS)
    if args.subjects:
        chosen = sorted({int(value) for value in args.subjects})
    elif args.cohort == "development":
        chosen = [value for value in available if value in development]
    elif args.cohort == "reserved":
        chosen = [value for value in available if value not in development]
    else:
        chosen = available
    missing = sorted(set(chosen) - set(available))
    if missing:
        raise ValueError(f"Requested subjects absent from condition 1: {missing}")
    return chosen


def _candidate_grid(args: argparse.Namespace) -> dict[str, list[MechanismCandidate]]:
    return {
        family: candidates_for_family(
            family,
            shared_theta=float(args.shared_theta),
            strategy_capacity=int(args.strategy_capacity),
        )
        for family in dict.fromkeys(str(value).upper() for value in args.families)
    }


def _cache_path(
    args: argparse.Namespace,
    readout: str,
    candidate: MechanismCandidate,
    subject_id: int,
) -> Path:
    return (
        args.output_dir
        / "candidate_runs"
        / readout
        / candidate.family
        / candidate.candidate_id
        / "cache"
        / f"subject_{int(subject_id)}"
        / f"particles_{int(args.particle_count)}"
        / f"rollouts_{int(args.rollout_count)}.npz"
    )


def _simulate_one(
    *,
    args: argparse.Namespace,
    frame: pd.DataFrame,
    base_engine: Mapping[str, Any],
    dataset_paths: Mapping[str, Path],
    readout: str,
    candidate: MechanismCandidate,
) -> dict[str, Any]:
    configured = apply_candidate(base_engine, candidate)
    beta_additive = float(
        configured["modules"]["beta_mod"]["kwargs"]["correct_additive"]
    )
    output_dir = (
        args.output_dir
        / "candidate_runs"
        / readout
        / candidate.family
        / candidate.candidate_id
    )
    path = simulate_subject(
        args=_candidate_args(
            args,
            output_dir=output_dir,
            readout_mode=readout,
            beta_additive=beta_additive,
        ),
        subject_frame=frame,
        engine_config=configured,
        dataset_paths=dataset_paths,
    )
    return {
        "subject_id": int(frame["iSub"].iloc[0]),
        "readout": readout,
        "family": candidate.family,
        "candidate_id": candidate.candidate_id,
        "value": float(candidate.value),
        "is_reference": bool(candidate.is_reference),
        "cache": str(path),
    }


def run_simulations(
    args: argparse.Namespace,
    data: pd.DataFrame,
    base_engine: Mapping[str, Any],
    dataset_paths: Mapping[str, Path],
    grid: Mapping[str, Sequence[MechanismCandidate]],
    subjects: Sequence[int],
) -> list[dict[str, Any]]:
    frames = {
        subject: data.loc[data["iSub"].eq(subject)].copy()
        for subject in subjects
    }
    jobs = [
        (readout, candidate, subject)
        for readout in dict.fromkeys(args.readout_modes)
        for family in grid
        for candidate in grid[family]
        for subject in subjects
    ]
    records = Parallel(
        n_jobs=min(int(args.n_jobs), len(jobs)),
        backend="loky",
        verbose=10,
    )(
        delayed(_simulate_one)(
            args=args,
            frame=frames[subject],
            base_engine=base_engine,
            dataset_paths=dataset_paths,
            readout=readout,
            candidate=candidate,
        )
        for readout, candidate, subject in jobs
    )
    pd.DataFrame(records).to_csv(
        args.output_dir / "candidate_run_index.csv",
        index=False,
    )
    return records


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = np.asarray(values, dtype=float) - float(np.max(values))
    exp = np.exp(shifted)
    return exp / float(np.sum(exp))


def _fit_population_prior(
    log_evidence: np.ndarray,
    *,
    alpha: float,
    max_iter: int = 10000,
    tolerance: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, int]:
    evidence = np.asarray(log_evidence, dtype=float)
    if evidence.ndim != 2 or not np.all(np.isfinite(evidence)):
        raise ValueError("log_evidence must be a finite subject-by-candidate matrix.")
    concentration = float(alpha)
    if not np.isfinite(concentration) or concentration <= 0.0:
        raise ValueError("EM alpha must be positive.")
    n_subjects, n_candidates = evidence.shape
    prior = np.full(n_candidates, 1.0 / n_candidates, dtype=float)
    responsibilities = np.zeros_like(evidence)
    for iteration in range(1, int(max_iter) + 1):
        for subject in range(n_subjects):
            responsibilities[subject] = _softmax(
                np.log(np.clip(prior, 1e-300, 1.0)) + evidence[subject]
            )
        updated = (
            responsibilities.sum(axis=0) + concentration
        ) / (n_subjects + concentration * n_candidates)
        if float(np.max(np.abs(updated - prior))) <= float(tolerance):
            return updated, responsibilities, iteration
        prior = updated
    return prior, responsibilities, int(max_iter)


def _bootstrap_delta(
    values: Sequence[float],
    *,
    seed: int,
    repeats: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": np.nan, "ci025": np.nan, "ci975": np.nan}
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(array, size=(int(repeats), array.size), replace=True).mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci025": float(np.quantile(draws, 0.025)),
        "ci975": float(np.quantile(draws, 0.975)),
    }


def _paired_signflip_p(
    values: Sequence[float],
    *,
    seed: int,
    repeats: int,
) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float("nan")
    observed = abs(float(np.mean(array)))
    if array.size <= 16:
        codes = np.arange(1 << array.size, dtype=np.uint64)[:, None]
        bits = (codes >> np.arange(array.size, dtype=np.uint64)) & 1
        signs = np.where(bits > 0, 1.0, -1.0)
        null = np.mean(signs * array[None, :], axis=1)
        return float(np.mean(np.abs(null) >= observed - 1e-15))
    rng = np.random.default_rng(int(seed))
    signs = rng.choice((-1.0, 1.0), size=(int(repeats), array.size))
    null = np.mean(signs * array[None, :], axis=1)
    return float(
        (1.0 + np.sum(np.abs(null) >= observed - 1e-15))
        / (1.0 + int(repeats))
    )


def _benjamini_hochberg(values: Sequence[float]) -> np.ndarray:
    p_values = np.asarray(values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(p_values))
    if not finite_indices.size:
        return adjusted
    finite = p_values[finite_indices]
    order = np.argsort(finite)
    ranked = finite[order]
    raw = ranked * ranked.size / np.arange(1, ranked.size + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    local = np.empty(ranked.size, dtype=float)
    local[order] = np.clip(monotone, 0.0, 1.0)
    adjusted[finite_indices] = local
    return adjusted


def _mixture_cache(
    *,
    caches: Sequence[Mapping[str, Any]],
    posterior: np.ndarray,
    seed: int,
    output: Path,
) -> dict[str, Any]:
    if not caches:
        raise ValueError("Cannot build a mixture from zero candidate caches.")
    rollout_count = int(np.asarray(caches[0]["choices"]).shape[0])
    rng = np.random.default_rng(int(seed))
    assignments = rng.choice(
        len(caches),
        size=rollout_count,
        replace=True,
        p=np.asarray(posterior, dtype=float),
    )
    choices = np.stack(
        [np.asarray(caches[index]["choices"])[row] for row, index in enumerate(assignments)]
    )
    feedback = np.stack(
        [np.asarray(caches[index]["feedback"])[row] for row, index in enumerate(assignments)]
    )
    probabilities = np.stack(
        [np.asarray(caches[index]["probabilities"])[row] for row, index in enumerate(assignments)]
    )
    generated_rho = np.stack(
        [np.asarray(caches[index]["generated_rho"])[row] for row, index in enumerate(assignments)]
    )
    prefix_pre_choice_ess = np.sum(
        np.asarray(posterior)[:, np.newaxis]
        * np.stack(
            [np.asarray(cache["prefix_pre_choice_ess"], dtype=float) for cache in caches]
        ),
        axis=0,
    )
    prefix_post_choice_ess = np.sum(
        np.asarray(posterior)[:, np.newaxis]
        * np.stack(
            [np.asarray(cache["prefix_post_choice_ess"], dtype=float) for cache in caches]
        ),
        axis=0,
    )
    prefix_rho_posterior_mean = np.sum(
        np.asarray(posterior)[:, np.newaxis]
        * np.stack(
            [np.asarray(cache["prefix_rho_posterior_mean"], dtype=float) for cache in caches]
        ),
        axis=0,
    )
    prefix_resampled_probability = np.sum(
        np.asarray(posterior)[:, np.newaxis]
        * np.stack(
            [np.asarray(cache["prefix_resampled"], dtype=float) for cache in caches]
        ),
        axis=0,
    )
    boundary_weights = np.sum(
        np.asarray(posterior)[:, np.newaxis]
        * np.stack(
            [np.asarray(cache["boundary_weights"], dtype=float) for cache in caches]
        ),
        axis=0,
    )
    boundary_weights /= boundary_weights.sum()
    boundary_rho = np.sum(
        np.asarray(posterior)[:, np.newaxis]
        * np.stack([np.asarray(cache["boundary_rho"], dtype=float) for cache in caches]),
        axis=0,
    )
    boundary_rho_volatility = np.sum(
        np.asarray(posterior)[:, np.newaxis]
        * np.stack(
            [np.asarray(cache["boundary_rho_volatility"], dtype=float) for cache in caches]
        ),
        axis=0,
    )
    first = caches[0]
    metadata = {
        key: first[key]
        for key in (
            "subject_id",
            "cohort",
            "split_index",
            "split_status",
            "split_mode",
            "train_n",
            "test_n",
            "particle_count",
            "rollout_count",
            "filter_seed",
            "rollout_seed",
        )
        if key in first
    }
    metadata["mixture_candidate_posterior"] = [float(value) for value in posterior]
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        choices=choices,
        feedback=feedback,
        probabilities=probabilities,
        generated_rho=generated_rho,
        candidate_assignments=assignments,
        candidate_posterior=np.asarray(posterior, dtype=float),
        prefix_pre_choice_ess=prefix_pre_choice_ess,
        prefix_post_choice_ess=prefix_post_choice_ess,
        prefix_resampled=(prefix_resampled_probability >= 0.5),
        prefix_rho_posterior_mean=prefix_rho_posterior_mean,
        boundary_weights=boundary_weights,
        boundary_rho=boundary_rho,
        boundary_rho_volatility=boundary_rho_volatility,
        observed_test_choices=np.asarray(first["observed_test_choices"]),
        observed_test_feedback=np.asarray(first["observed_test_feedback"]),
        test_iTrial=np.asarray(first["test_iTrial"]),
        test_iSession=np.asarray(first["test_iSession"]),
        test_iBlock=np.asarray(first["test_iBlock"]),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return load_subject_cache(output)


def summarize(
    args: argparse.Namespace,
    data: pd.DataFrame,
    grid: Mapping[str, Sequence[MechanismCandidate]],
    subjects: Sequence[int],
) -> dict[str, Any]:
    frozen = None
    if args.frozen_priors is not None:
        frozen = pd.read_csv(args.frozen_priors)
    if args.cohort == "reserved" and frozen is None:
        raise ValueError("Reserved summarization requires --frozen-priors.")

    prefix_rows: list[dict[str, Any]] = []
    prior_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []
    mixture_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for readout in dict.fromkeys(args.readout_modes):
        for family, candidates in grid.items():
            log_evidence = np.zeros((len(subjects), len(candidates)), dtype=float)
            cache_grid: list[list[dict[str, Any]]] = []
            for subject_index, subject in enumerate(subjects):
                subject_caches: list[dict[str, Any]] = []
                frame = data.loc[data["iSub"].eq(subject)]
                for candidate_index, candidate in enumerate(candidates):
                    cache = load_subject_cache(
                        _cache_path(args, readout, candidate, subject)
                    )
                    subject_caches.append(cache)
                    observed = np.asarray(
                        cache["prefix_observed_choice_probability"],
                        dtype=float,
                    )
                    log_density = float(np.log(np.clip(observed, 1e-12, 1.0)).sum())
                    log_evidence[subject_index, candidate_index] = log_density
                    prefix_rows.append(
                        {
                            "cohort": args.cohort,
                            "subject_id": int(subject),
                            "readout": readout,
                            "family": family,
                            "candidate_id": candidate.candidate_id,
                            "value": float(candidate.value),
                            "is_reference": bool(candidate.is_reference),
                            "prefix_n": int(observed.size),
                            "prefix_log_predictive_density": log_density,
                            "prefix_nll": float(-np.mean(np.log(np.clip(observed, 1e-12, 1.0)))),
                            "prefix_brier": float(np.mean(2.0 * np.square(1.0 - observed))),
                            "test_n": int(len(frame) - int(cache["split_index"])),
                            "split_status": str(cache["split_status"]),
                        }
                    )
                cache_grid.append(subject_caches)

            if frozen is None:
                prior, responsibilities, iterations = _fit_population_prior(
                    log_evidence,
                    alpha=float(args.em_alpha),
                )
                prior_source = "estimated_from_development"
            else:
                selected = frozen.loc[
                    frozen["readout"].eq(readout)
                    & frozen["family"].eq(family)
                ].set_index("candidate_id")
                missing = [
                    candidate.candidate_id
                    for candidate in candidates
                    if candidate.candidate_id not in selected.index
                ]
                if missing:
                    raise ValueError(
                        f"Frozen priors missing {readout}/{family}: {missing}"
                    )
                prior = np.asarray(
                    [
                        float(selected.loc[candidate.candidate_id, "population_prior"])
                        for candidate in candidates
                    ],
                    dtype=float,
                )
                prior /= prior.sum()
                responsibilities = np.stack(
                    [
                        _softmax(np.log(np.clip(prior, 1e-300, 1.0)) + row)
                        for row in log_evidence
                    ]
                )
                iterations = 0
                prior_source = "frozen_development_prior"

            for candidate, weight in zip(candidates, prior):
                prior_rows.append(
                    {
                        "cohort": args.cohort,
                        "readout": readout,
                        "family": family,
                        "candidate_id": candidate.candidate_id,
                        "value": float(candidate.value),
                        "is_reference": bool(candidate.is_reference),
                        "population_prior": float(weight),
                        "prior_source": prior_source,
                        "em_iterations": int(iterations),
                    }
                )

            reference_index = next(
                (index for index, candidate in enumerate(candidates) if candidate.is_reference),
                None,
            )
            mixture_by_subject: dict[int, dict[str, Any]] = {}
            reference_by_subject: dict[int, dict[str, Any]] = {}
            for subject_index, subject in enumerate(subjects):
                for candidate, weight in zip(candidates, responsibilities[subject_index]):
                    posterior_rows.append(
                        {
                            "cohort": args.cohort,
                            "subject_id": int(subject),
                            "readout": readout,
                            "family": family,
                            "candidate_id": candidate.candidate_id,
                            "value": float(candidate.value),
                            "posterior_weight": float(weight),
                            "posterior_mean_value": float(
                                np.sum(
                                    responsibilities[subject_index]
                                    * np.asarray([item.value for item in candidates], dtype=float)
                                )
                            ),
                        }
                    )
                mixture_path = (
                    args.output_dir
                    / "mixtures"
                    / readout
                    / family
                    / f"subject_{int(subject)}.npz"
                )
                mixture_cache = _mixture_cache(
                    caches=cache_grid[subject_index],
                    posterior=responsibilities[subject_index],
                    seed=stable_seed(
                        {
                            "seed_role": "mechanism_screen_mixture",
                            "base_seed": int(args.base_seed),
                            "subject_id": int(subject),
                            "readout": readout,
                            "family": family,
                        }
                    ),
                    output=mixture_path,
                )
                mixture_summary, _, _, _ = evaluate_subject(
                    mixture_cache,
                    window=int(args.window),
                )
                mixture_summary.update(
                    {
                        "cohort": args.cohort,
                        "readout": readout,
                        "family": family,
                        "model": "candidate_bank_mixture",
                    }
                )
                mixture_rows.append(mixture_summary)
                mixture_by_subject[int(subject)] = mixture_summary

                if reference_index is not None:
                    reference_summary, _, _, _ = evaluate_subject(
                        cache_grid[subject_index][reference_index],
                        window=int(args.window),
                    )
                    reference_summary.update(
                        {
                            "cohort": args.cohort,
                            "readout": readout,
                            "family": family,
                            "model": "reference_candidate",
                        }
                    )
                    mixture_rows.append(reference_summary)
                    reference_by_subject[int(subject)] = reference_summary

            if reference_index is not None:
                for metric in (
                    "curve_crps",
                    "summary_discrepancy",
                    "combined_calibration_p",
                ):
                    deltas = [
                        float(mixture_by_subject[int(subject)][metric])
                        - float(reference_by_subject[int(subject)][metric])
                        for subject in subjects
                    ]
                    bootstrap = _bootstrap_delta(
                        deltas,
                        seed=stable_seed(
                            {
                                "seed_role": "mechanism_screen_bootstrap",
                                "base_seed": int(args.base_seed),
                                "readout": readout,
                                "family": family,
                                "metric": metric,
                            }
                        ),
                        repeats=int(args.bootstrap_repeats),
                    )
                    lower_is_better = metric != "combined_calibration_p"
                    comparison_rows.append(
                        {
                            "cohort": args.cohort,
                            "readout": readout,
                            "family": family,
                            "metric": metric,
                            "delta_definition": "mixture_minus_reference",
                            "better_direction": (
                                "negative" if lower_is_better else "positive"
                            ),
                            **bootstrap,
                            "paired_signflip_p": _paired_signflip_p(
                                deltas,
                                seed=stable_seed(
                                    {
                                        "seed_role": "mechanism_screen_signflip",
                                        "base_seed": int(args.base_seed),
                                        "readout": readout,
                                        "family": family,
                                        "metric": metric,
                                    }
                                ),
                                repeats=int(args.bootstrap_repeats),
                            ),
                            "improved_subject_n": int(
                                np.sum(
                                    np.asarray(deltas) < 0.0
                                    if lower_is_better
                                    else np.asarray(deltas) > 0.0
                                )
                            ),
                            "subject_n": int(len(deltas)),
                        }
                    )
                mixture_pass = np.asarray(
                    [
                        bool(mixture_by_subject[int(subject)]["combined_pass_95"])
                        for subject in subjects
                    ]
                )
                reference_pass = np.asarray(
                    [
                        bool(reference_by_subject[int(subject)]["combined_pass_95"])
                        for subject in subjects
                    ]
                )
                improved = int(np.sum(mixture_pass & ~reference_pass))
                worsened = int(np.sum(~mixture_pass & reference_pass))
                discordant = improved + worsened
                coverage_rows.append(
                    {
                        "cohort": args.cohort,
                        "readout": readout,
                        "family": family,
                        "mixture_pass_n": int(mixture_pass.sum()),
                        "reference_pass_n": int(reference_pass.sum()),
                        "subject_n": int(len(subjects)),
                        "improved_n": improved,
                        "worsened_n": worsened,
                        "exact_p": (
                            float(binomtest(improved, discordant, 0.5).pvalue)
                            if discordant
                            else 1.0
                        ),
                    }
                )

    adjusted = _benjamini_hochberg(
        [row["paired_signflip_p"] for row in comparison_rows]
    )
    for row, q_value in zip(comparison_rows, adjusted):
        row["paired_signflip_q"] = float(q_value)
    coverage_adjusted = _benjamini_hochberg(
        [row["exact_p"] for row in coverage_rows]
    )
    for row, q_value in zip(coverage_rows, coverage_adjusted):
        row["exact_q"] = float(q_value)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prefix_rows).to_csv(
        args.output_dir / "prefix_candidate_scores.csv", index=False
    )
    pd.DataFrame(prior_rows).to_csv(
        args.output_dir / "population_priors.csv", index=False
    )
    pd.DataFrame(posterior_rows).to_csv(
        args.output_dir / "subject_candidate_posteriors.csv", index=False
    )
    pd.DataFrame(mixture_rows).to_csv(
        args.output_dir / "mixture_subject_summary.csv", index=False
    )
    pd.DataFrame(comparison_rows).to_csv(
        args.output_dir / "comparison_summary.csv", index=False
    )
    pd.DataFrame(coverage_rows).to_csv(
        args.output_dir / "coverage_comparison.csv", index=False
    )
    summary = {
        "analysis": "condition1_one_mechanism_hierarchical_screen",
        "cohort": args.cohort,
        "subjects": [int(value) for value in subjects],
        "families": list(grid),
        "readout_modes": list(dict.fromkeys(args.readout_modes)),
        "particle_count": int(args.particle_count),
        "rollout_count": int(args.rollout_count),
        "shared_theta": float(args.shared_theta),
        "strategy_capacity": int(args.strategy_capacity),
        "em_alpha": float(args.em_alpha),
        "future_observed_choices_read": False,
        "future_feedback_generated_from_simulated_choices": True,
        "comparison_rows": comparison_rows,
        "coverage_rows": coverage_rows,
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    raw = pd.read_csv(args.data)
    data_quality = _validate_condition1_data(raw)
    data = (
        raw.loc[raw["condition"].eq(1)]
        .sort_values(list(KEY_COLUMNS))
        .reset_index(drop=True)
    )
    subjects = _subject_ids(args, data)
    data = data.loc[data["iSub"].isin(subjects)].copy()
    grid = _candidate_grid(args)
    base_engine = load_yaml(args.engine_config)
    simulation_config = load_yaml(args.simulation_config)
    dataset_paths = resolve_dataset_paths(
        simulation_config,
        args.simulation_config.parent,
        DEFAULT_DATA_PATH,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "data_quality.json", data_quality)
    manifest = {
        "analysis": "condition1_one_mechanism_hierarchical_screen",
        "data": str(args.data),
        "engine_config": str(args.engine_config),
        "cohort": args.cohort,
        "subjects": subjects,
        "families": list(grid),
        "candidate_ids": {
            family: [candidate.candidate_id for candidate in candidates]
            for family, candidates in grid.items()
        },
        "readout_modes": list(dict.fromkeys(args.readout_modes)),
        "particle_count": int(args.particle_count),
        "rollout_count": int(args.rollout_count),
        "n_jobs": int(args.n_jobs),
        "base_seed": int(args.base_seed),
        "frozen_priors": None if args.frozen_priors is None else str(args.frozen_priors),
    }
    _write_json(args.output_dir / "manifest.json", manifest)

    if args.mode in ("simulate", "all"):
        records = run_simulations(
            args,
            data,
            base_engine,
            dataset_paths,
            grid,
            subjects,
        )
        print(
            json.dumps(
                {"stage": "simulation_complete", "run_n": len(records)},
                ensure_ascii=False,
            ),
            flush=True,
        )
    if args.mode in ("summarize", "all"):
        result = summarize(args, data, grid, subjects)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
