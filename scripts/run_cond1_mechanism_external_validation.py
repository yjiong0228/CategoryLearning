#!/usr/bin/env python3
"""Validate frozen mechanism mixtures against suffix choices, RT, and oral reports.

Candidate weights must come from an earlier choice-only prefix screen.  This
script never uses RT or oral reports to select a candidate.  It runs a fresh
sequential filter on each candidate, holds the candidate weights fixed, and
compares the resulting mixture with the family's frozen reference candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cond1_b0_trajectory_ppc import (  # noqa: E402
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    split_for_subject,
)
from scripts.run_cond1_mechanism_screen import (  # noqa: E402
    _benjamini_hochberg,
    _paired_signflip_p,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.active_set.mechanism_variants import (  # noqa: E402
    MechanismCandidate,
    apply_candidate,
    candidates_for_family,
)
from src.Bayesian_state.active_set.particle_filter import (  # noqa: E402
    run_active_set_particle_filter,
)
from src.Bayesian_state.optimization.optimization_config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    load_yaml,
)
from src.Bayesian_state.optimization.optimizer_common import stable_seed  # noqa: E402
from src.Bayesian_state.model_evaluation.oral_model_alignment import (  # noqa: E402
    OralModelAlignmentMixin,
    Oral_center_mapping,
)
from src.Bayesian_state.problems.partitions import Partition  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/processed/Task2_processed.csv"
    )
    parser.add_argument("--posterior-csv", type=Path, required=True)
    parser.add_argument(
        "--engine-config",
        type=Path,
        default=ROOT / "configs/model_struct/pmh_model_cond1_mechanism_screen.yaml",
    )
    parser.add_argument(
        "--simulation-config",
        type=Path,
        default=ROOT / "configs/simulation_cfg/pmh_cond1_simulation_v14.yaml",
    )
    parser.add_argument(
        "--families", choices=("F", "M", "H", "P", "S"), nargs="+", default=("F", "M", "H", "P")
    )
    parser.add_argument("--readout", choices=("static",), default="static")
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--cohort", default="reserved")
    parser.add_argument("--particle-count", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=96)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--shared-theta", type=float, default=0.75)
    parser.add_argument("--strategy-capacity", type=int, default=3)
    parser.add_argument("--oral-beta", type=float, default=10.0)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--base-seed", type=int, default=20261301)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=True)
        + "\n",
        encoding="utf-8",
    )


def _cache_path(
    output_dir: Path, family: str, candidate_id: str, subject_id: int, particles: int
) -> Path:
    return (
        output_dir
        / "candidate_filters"
        / family
        / candidate_id
        / f"subject_{int(subject_id)}_particles_{int(particles)}.npz"
    )


def _run_candidate_filter(
    *,
    args: argparse.Namespace,
    frame: pd.DataFrame,
    candidate: MechanismCandidate,
    base_engine: Mapping[str, Any],
    dataset_paths: Mapping[str, Path],
) -> dict[str, Any]:
    subject_id = int(frame["iSub"].iloc[0])
    output = _cache_path(
        args.output_dir,
        candidate.family,
        candidate.candidate_id,
        subject_id,
        int(args.particle_count),
    )
    if output.exists() and not args.force:
        return {
            "subject_id": subject_id,
            "family": candidate.family,
            "candidate_id": candidate.candidate_id,
            "cache": str(output),
        }
    filter_seed = stable_seed(
        {
            "seed_role": "mechanism_external_validation_paired_filter",
            "base_seed": int(args.base_seed),
            "subject_id": subject_id,
            "particle_count": int(args.particle_count),
        }
    )
    result = run_active_set_particle_filter(
        engine_config=apply_candidate(base_engine, candidate),
        subject_id=subject_id,
        stimulus=frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
        choices=frame["choice"].to_numpy(dtype=int),
        feedback=frame["feedback"].to_numpy(dtype=float),
        particle_count=int(args.particle_count),
        rho=float(args.rho),
        epsilon=float(args.epsilon),
        filter_seed=int(filter_seed),
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        marginal_probabilities=result.marginal_probabilities,
        marginal_hypothesis_prior=result.marginal_hypothesis_prior,
        marginal_active_probability=result.marginal_active_probability,
        post_choice_ess=result.post_choice_ess,
        resampled=result.resampled,
        metadata=np.asarray(
            json.dumps(
                {
                    "subject_id": subject_id,
                    "family": candidate.family,
                    "candidate_id": candidate.candidate_id,
                    "candidate_value": float(candidate.value),
                    "is_reference": bool(candidate.is_reference),
                    "filter_seed": int(filter_seed),
                    "particle_count": int(args.particle_count),
                }
            )
        ),
    )
    return {
        "subject_id": subject_id,
        "family": candidate.family,
        "candidate_id": candidate.candidate_id,
        "cache": str(output),
    }


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            key: np.asarray(payload[key])
            for key in payload.files
            if key != "metadata"
        }


def _oral_center_similarity(
    *,
    partition: Partition,
    hypothesis_prior: np.ndarray,
    frame: pd.DataFrame,
    trial_indices: np.ndarray,
    beta: float,
) -> np.ndarray:
    values = np.full(trial_indices.size, np.nan, dtype=float)
    for output_index, trial_index in enumerate(trial_indices):
        report_text = frame.iloc[int(trial_index)].get("text")
        if pd.isna(report_text) or not str(report_text).strip():
            continue
        oral_center = Oral_center_mapping._parse_center(
            frame.iloc[int(trial_index)]["oral_center"]
        )
        if np.isnan(oral_center).any():
            continue
        choice = int(frame.iloc[int(trial_index)]["choice"])
        conditioned = OralModelAlignmentMixin._choice_conditioned_prior(
            partition=partition,
            prior=hypothesis_prior[int(trial_index)],
            stimulus=frame.iloc[int(trial_index)][list(FEATURE_COLUMNS)].to_numpy(
                dtype=float
            ),
            choice=choice,
            beta=float(beta),
        )
        values[output_index] = OralModelAlignmentMixin._expected_center_similarity(
            partition=partition,
            model_dist=conditioned,
            oral_center=oral_center,
            choice=choice,
        )
    return values


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(keep)) < 4 or np.unique(x[keep]).size < 2 or np.unique(y[keep]).size < 2:
        return float("nan")
    return float(spearmanr(x[keep], y[keep]).statistic)


def _subject_validation(
    *,
    args: argparse.Namespace,
    frame: pd.DataFrame,
    family: str,
    candidates: Sequence[MechanismCandidate],
    posterior_frame: pd.DataFrame,
    partition: Partition,
) -> dict[str, Any]:
    subject_id = int(frame["iSub"].iloc[0])
    posterior = posterior_frame.set_index("candidate_id")["posterior_weight"]
    missing = [candidate.candidate_id for candidate in candidates if candidate.candidate_id not in posterior]
    if missing:
        raise ValueError(f"Posterior weights missing {family}/{subject_id}: {missing}")
    weights = np.asarray(
        [float(posterior.loc[candidate.candidate_id]) for candidate in candidates],
        dtype=float,
    )
    weights /= weights.sum()
    caches = [
        _load_cache(
            _cache_path(
                args.output_dir,
                family,
                candidate.candidate_id,
                subject_id,
                int(args.particle_count),
            )
        )
        for candidate in candidates
    ]
    mixture_choice = np.sum(
        weights[:, None, None]
        * np.stack([cache["marginal_probabilities"] for cache in caches]),
        axis=0,
    )
    mixture_prior = np.sum(
        weights[:, None, None]
        * np.stack([cache["marginal_hypothesis_prior"] for cache in caches]),
        axis=0,
    )
    reference_index = next(
        index for index, candidate in enumerate(candidates) if candidate.is_reference
    )
    reference_choice = caches[reference_index]["marginal_probabilities"]
    reference_prior = caches[reference_index]["marginal_hypothesis_prior"]
    split_index, split_status = split_for_subject(
        frame, mode="early_anchor", window=int(args.window)
    )
    suffix = np.arange(split_index, len(frame), dtype=int)
    observed_index = frame["choice"].to_numpy(dtype=int)[suffix] - 1
    mixture_observed = mixture_choice[suffix, observed_index]
    reference_observed = reference_choice[suffix, observed_index]
    mixture_surprise = -np.log(np.clip(mixture_observed, 1e-12, 1.0))
    reference_surprise = -np.log(np.clip(reference_observed, 1e-12, 1.0))
    log_rt = np.log(frame["choRT"].to_numpy(dtype=float)[suffix])
    oral_mixture = _oral_center_similarity(
        partition=partition,
        hypothesis_prior=mixture_prior,
        frame=frame,
        trial_indices=suffix,
        beta=float(args.oral_beta),
    )
    oral_reference = _oral_center_similarity(
        partition=partition,
        hypothesis_prior=reference_prior,
        frame=frame,
        trial_indices=suffix,
        beta=float(args.oral_beta),
    )
    posterior_mean = float(
        np.sum(weights * np.asarray([candidate.value for candidate in candidates]))
    )
    return {
        "cohort": str(args.cohort),
        "subject_id": subject_id,
        "readout": str(args.readout),
        "family": family,
        "split_status": split_status,
        "prefix_n": int(split_index),
        "suffix_n": int(suffix.size),
        "posterior_mean_value": posterior_mean,
        "max_candidate_posterior": float(np.max(weights)),
        "mixture_suffix_nll": float(np.mean(mixture_surprise)),
        "reference_suffix_nll": float(np.mean(reference_surprise)),
        "delta_suffix_nll": float(np.mean(mixture_surprise - reference_surprise)),
        "mixture_rt_surprise_spearman": _safe_spearman(mixture_surprise, log_rt),
        "reference_rt_surprise_spearman": _safe_spearman(reference_surprise, log_rt),
        "delta_rt_surprise_spearman": (
            _safe_spearman(mixture_surprise, log_rt)
            - _safe_spearman(reference_surprise, log_rt)
        ),
        "oral_valid_n": int(np.sum(np.isfinite(oral_mixture) & np.isfinite(oral_reference))),
        "mixture_oral_center_similarity": float(np.nanmean(oral_mixture)),
        "reference_oral_center_similarity": float(np.nanmean(oral_reference)),
        "delta_oral_center_similarity": float(
            np.nanmean(oral_mixture - oral_reference)
        ),
    }


def _bootstrap(values: np.ndarray, *, seed: int, repeats: int) -> dict[str, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return {"mean": np.nan, "ci025": np.nan, "ci975": np.nan}
    draws = np.random.default_rng(int(seed)).choice(
        finite, size=(int(repeats), finite.size), replace=True
    ).mean(axis=1)
    return {
        "mean": float(np.mean(finite)),
        "ci025": float(np.quantile(draws, 0.025)),
        "ci975": float(np.quantile(draws, 0.975)),
    }


def main() -> None:
    args = parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    raw = pd.read_csv(args.data)
    data = (
        raw.loc[raw["condition"].eq(1)]
        .sort_values(list(KEY_COLUMNS))
        .reset_index(drop=True)
    )
    posteriors = pd.read_csv(args.posterior_csv)
    posteriors = posteriors.loc[posteriors["readout"].eq(args.readout)].copy()
    subjects = (
        sorted({int(value) for value in args.subjects})
        if args.subjects
        else sorted(posteriors["subject_id"].astype(int).unique())
    )
    data = data.loc[data["iSub"].isin(subjects)].copy()
    base_engine = load_yaml(args.engine_config)
    simulation_config = load_yaml(args.simulation_config)
    dataset_paths = resolve_dataset_paths(
        simulation_config, args.simulation_config.parent, DEFAULT_DATA_PATH
    )
    grid = {
        family: candidates_for_family(
            family,
            shared_theta=float(args.shared_theta),
            strategy_capacity=int(args.strategy_capacity),
        )
        for family in dict.fromkeys(str(value).upper() for value in args.families)
    }
    if any(sum(candidate.is_reference for candidate in candidates) != 1 for candidates in grid.values()):
        raise ValueError("Every external-validation family must have exactly one reference.")
    frames = {subject: data.loc[data["iSub"].eq(subject)].copy() for subject in subjects}
    jobs = [
        (family, candidate, subject)
        for family, candidates in grid.items()
        for candidate in candidates
        for subject in subjects
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = Parallel(
        n_jobs=min(int(args.n_jobs), len(jobs)), backend="loky", verbose=10
    )(
        delayed(_run_candidate_filter)(
            args=args,
            frame=frames[subject],
            candidate=candidate,
            base_engine=base_engine,
            dataset_paths=dataset_paths,
        )
        for family, candidate, subject in jobs
    )
    pd.DataFrame(index).to_csv(args.output_dir / "candidate_filter_index.csv", index=False)
    partition = Partition(n_dims=4, n_cats=2, include_label_reversals=True)
    subject_rows = [
        _subject_validation(
            args=args,
            frame=frames[subject],
            family=family,
            candidates=candidates,
            posterior_frame=posteriors.loc[
                posteriors["family"].eq(family)
                & posteriors["subject_id"].eq(subject)
            ],
            partition=partition,
        )
        for family, candidates in grid.items()
        for subject in subjects
    ]
    subject_summary = pd.DataFrame(subject_rows)
    subject_summary.to_csv(args.output_dir / "subject_validation.csv", index=False)
    metric_direction = {
        "delta_suffix_nll": "negative",
        "delta_rt_surprise_spearman": "positive",
        "delta_oral_center_similarity": "positive",
    }
    group_rows = []
    for family, frame in subject_summary.groupby("family", sort=True):
        for metric, direction in metric_direction.items():
            summary = _bootstrap(
                frame[metric].to_numpy(dtype=float),
                seed=stable_seed(
                    {
                        "seed_role": "mechanism_external_validation_bootstrap",
                        "base_seed": int(args.base_seed),
                        "family": family,
                        "metric": metric,
                    }
                ),
                repeats=int(args.bootstrap_repeats),
            )
            group_rows.append(
                {
                    "cohort": str(args.cohort),
                    "readout": str(args.readout),
                    "family": family,
                    "metric": metric,
                    "delta_definition": "mixture_minus_reference",
                    "better_direction": direction,
                    **summary,
                    "paired_signflip_p": _paired_signflip_p(
                        frame[metric].to_numpy(dtype=float),
                        seed=stable_seed(
                            {
                                "seed_role": "mechanism_external_validation_signflip",
                                "base_seed": int(args.base_seed),
                                "family": family,
                                "metric": metric,
                            }
                        ),
                        repeats=int(args.bootstrap_repeats),
                    ),
                    "subject_n": int(frame[metric].notna().sum()),
                }
            )
    adjusted = _benjamini_hochberg(
        [row["paired_signflip_p"] for row in group_rows]
    )
    for row, q_value in zip(group_rows, adjusted):
        row["paired_signflip_q"] = float(q_value)
    group_summary = pd.DataFrame(group_rows)
    group_summary.to_csv(args.output_dir / "group_validation.csv", index=False)
    _write_json(
        args.output_dir / "summary.json",
        {
            "analysis": "condition1_mechanism_external_validation",
            "cohort": str(args.cohort),
            "subjects": subjects,
            "families": list(grid),
            "readout": str(args.readout),
            "particle_count": int(args.particle_count),
            "candidate_weights_source": str(args.posterior_csv),
            "candidate_weights_frozen_during_suffix": True,
            "rt_used_for_fitting": False,
            "oral_report_used_for_fitting": False,
            "rt_transform": "log_seconds_then_within_subject_spearman",
            "oral_metric": "choice_conditioned_expected_center_similarity",
            "group_rows": group_rows,
        },
    )
    print(group_summary.to_json(orient="records", indent=2), flush=True)


if __name__ == "__main__":
    main()
