#!/usr/bin/env python3
"""Recover condition-1 heterogeneity candidates from synthetic trajectories.

The recovery target is deliberately the same early observed prefix that is
used to estimate subject-specific candidate weights in the mechanism screen.
Synthetic choices are generated autonomously on real stimulus/category
schedules.  Every fitted candidate sees the same synthetic choices and uses
paired particle seeds, so differences are attributable to the candidate
mechanism rather than to a different Monte Carlo draw.
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
from scipy.special import logsumexp
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cond1_b0_trajectory_ppc import (  # noqa: E402
    DEVELOPMENT_SUBJECTS,
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    split_for_subject,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.simulation.autonomous_model_execution import (  # noqa: E402
    run_autonomous_category_learning,
)
from src.Bayesian_state.optimization.mechanism_candidates import (  # noqa: E402
    MechanismCandidate,
    apply_candidate,
    candidates_for_family,
)
from src.Bayesian_state.inference_engine.backends.particle_filter import (  # noqa: E402
    run_state_model_particle_filter,
)
from src.Bayesian_state.optimization.optimization_config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    load_yaml,
)
from src.Bayesian_state.utils.seeding import stable_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/processed/Task2_processed.csv"
    )
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
    parser.add_argument("--subjects", type=int, nargs="+", default=[103, 117, 131])
    parser.add_argument(
        "--families", choices=("F", "M", "H", "P"), nargs="+", default=("F", "M", "H", "P")
    )
    parser.add_argument("--scope", choices=("within", "cross"), default="within")
    parser.add_argument("--datasets-per-candidate", type=int, default=3)
    parser.add_argument("--particle-count", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=96)
    parser.add_argument("--rho", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--shared-theta", type=float, default=0.75)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=20261211)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/zhuran/cond1_active_set/mechanism_recovery_within_v1",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _softmax(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    shifted = array - float(np.max(array))
    probabilities = np.exp(shifted)
    return probabilities / float(np.sum(probabilities))


def candidate_grid(
    families: Sequence[str], *, shared_theta: float
) -> dict[str, list[MechanismCandidate]]:
    return {
        family: candidates_for_family(family, shared_theta=float(shared_theta))
        for family in dict.fromkeys(str(value).upper() for value in families)
    }


def cross_candidate_bank(
    grid: Mapping[str, Sequence[MechanismCandidate]],
) -> list[MechanismCandidate]:
    """Return one baseline plus every non-baseline candidate.

    The four family-specific reference labels describe the same engine.  A
    single BASE entry prevents that duplicate parameterization from receiving
    four times the prior mass in cross-family recovery.
    """

    bank = [
        MechanismCandidate(
            family="BASE",
            candidate_id="BASE",
            value=1.0,
            parameters=(),
            is_reference=True,
        )
    ]
    bank.extend(
        candidate
        for candidates in grid.values()
        for candidate in candidates
        if not candidate.is_reference
    )
    ids = [candidate.candidate_id for candidate in bank]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Cross-family recovery candidate IDs are not unique.")
    return bank


def _dataset_cache(
    output_dir: Path,
    scope: str,
    true_candidate: MechanismCandidate,
    subject_id: int,
    replicate: int,
) -> Path:
    return (
        output_dir
        / "cache"
        / str(scope)
        / true_candidate.family
        / true_candidate.candidate_id
        / f"subject_{int(subject_id)}_replicate_{int(replicate):02d}.json"
    )


def _score_probabilities(
    probabilities: np.ndarray, choices: np.ndarray
) -> tuple[float, float, float]:
    observed_index = np.asarray(choices, dtype=int) - 1
    selected = np.asarray(probabilities, dtype=float)[
        np.arange(observed_index.size), observed_index
    ]
    log_density = float(np.log(np.clip(selected, 1e-12, 1.0)).sum())
    nll = float(-np.log(np.clip(selected, 1e-12, 1.0)).mean())
    brier = float(np.mean(2.0 * np.square(1.0 - selected)))
    return log_density, nll, brier


def recover_one_dataset(
    *,
    args: argparse.Namespace,
    subject_frame: pd.DataFrame,
    base_engine: Mapping[str, Any],
    dataset_paths: Mapping[str, Path],
    true_candidate: MechanismCandidate,
    fit_candidates: Sequence[MechanismCandidate],
    replicate: int,
) -> list[dict[str, Any]]:
    subject_id = int(subject_frame["iSub"].iloc[0])
    output = _dataset_cache(
        args.output_dir, args.scope, true_candidate, subject_id, replicate
    )
    if output.exists() and not args.force:
        return list(json.loads(output.read_text(encoding="utf-8")))

    split_index, split_status = split_for_subject(
        subject_frame, mode="early_anchor", window=int(args.window)
    )
    stimulus = subject_frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)[:split_index]
    categories = subject_frame["category"].to_numpy(dtype=int)[:split_index]
    generation_seed = stable_seed(
        {
            "seed_role": "mechanism_recovery_generation",
            "base_seed": int(args.base_seed),
            "scope": str(args.scope),
            "subject_id": subject_id,
            "true_candidate": true_candidate.candidate_id,
            "replicate": int(replicate),
        }
    )
    generated_result = run_autonomous_category_learning(
        engine_config=apply_candidate(base_engine, true_candidate),
        subject_id=subject_id,
        condition=1,
        stimulus=stimulus,
        categories=categories,
        trajectory_seed=int(generation_seed),
        choice_readout_config={
            "method": "sharpened_expectation",
            "power": float(args.rho),
            "weight_floor": 0.0,
        },
        output_noise_config={
            "enabled": float(args.epsilon) > 0.0,
            "base_lapse": float(args.epsilon),
            "post_error_lapse": 0.0,
            "low_accuracy_lapse": 0.0,
            "latent_volatility_lapse": 0.0,
            "max_lapse": 1.0,
            "lapse_target": "uniform",
        },
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
    )
    generated = generated_result.trajectory
    paired_filter_seed = stable_seed(
        {
            "seed_role": "mechanism_recovery_paired_filter",
            "base_seed": int(args.base_seed),
            "scope": str(args.scope),
            "subject_id": subject_id,
            "true_candidate": true_candidate.candidate_id,
            "replicate": int(replicate),
            "particle_count": int(args.particle_count),
        }
    )
    rows: list[dict[str, Any]] = []
    for fit_candidate in fit_candidates:
        fitted = run_state_model_particle_filter(
            engine_config=apply_candidate(base_engine, fit_candidate),
            subject_id=subject_id,
            stimulus=stimulus,
            choices=generated.choices,
            feedback=generated.feedback,
            particle_count=int(args.particle_count),
            choice_readout_power=float(args.rho),
            output_lapse=float(args.epsilon),
            filter_seed=int(paired_filter_seed),
            processed_data_dir=dataset_paths["processed_dir"],
            dataset_paths=dataset_paths,
        )
        log_density, nll, brier = _score_probabilities(
            fitted.marginal_probabilities, generated.choices
        )
        rows.append(
            {
                "scope": str(args.scope),
                "subject_id": subject_id,
                "replicate": int(replicate),
                "split_index": int(split_index),
                "split_status": split_status,
                "generation_seed": int(generation_seed),
                "filter_seed": int(paired_filter_seed),
                "true_family": true_candidate.family,
                "true_candidate_id": true_candidate.candidate_id,
                "true_value": float(true_candidate.value),
                "true_is_reference": bool(true_candidate.is_reference),
                "fit_family": fit_candidate.family,
                "fit_candidate_id": fit_candidate.candidate_id,
                "fit_value": float(fit_candidate.value),
                "fit_is_reference": bool(fit_candidate.is_reference),
                "prefix_log_predictive_density": log_density,
                "prefix_nll": nll,
                "prefix_brier": brier,
                "generated_accuracy": float(np.mean(generated.feedback)),
            }
        )
    _write_json(output, rows)
    return rows


def summarize_recovery(
    scores: pd.DataFrame, *, scope: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_keys = ["subject_id", "replicate", "true_family", "true_candidate_id"]
    recovered_rows: list[dict[str, Any]] = []
    for key, frame in scores.groupby(dataset_keys, sort=True):
        frame = frame.sort_values("fit_candidate_id").reset_index(drop=True)
        candidate_posterior = _softmax(frame["prefix_log_predictive_density"].to_numpy())
        best_index = int(np.argmax(candidate_posterior))
        true_candidate = str(key[3])
        true_family = str(key[2])
        family_names = sorted(frame["fit_family"].astype(str).unique())
        family_log_evidence = {
            family: float(
                logsumexp(
                    frame.loc[
                        frame["fit_family"].eq(family),
                        "prefix_log_predictive_density",
                    ].to_numpy(dtype=float)
                )
                - np.log(int(frame["fit_family"].eq(family).sum()))
            )
            for family in family_names
        }
        family_posterior_values = _softmax(list(family_log_evidence.values()))
        family_posterior = dict(zip(family_names, family_posterior_values))
        row = {
            "scope": scope,
            "subject_id": int(key[0]),
            "replicate": int(key[1]),
            "true_family": true_family,
            "true_candidate_id": true_candidate,
            "true_value": float(frame["true_value"].iloc[0]),
            "predicted_family": max(family_posterior, key=family_posterior.get),
            "predicted_candidate_id": str(frame["fit_candidate_id"].iloc[best_index]),
            "predicted_value": float(frame["fit_value"].iloc[best_index]),
            "exact_candidate_recovered": bool(
                str(frame["fit_candidate_id"].iloc[best_index]) == true_candidate
            ),
            "family_recovered": bool(
                max(family_posterior, key=family_posterior.get) == true_family
            ),
            "true_candidate_posterior": float(
                candidate_posterior[
                    frame["fit_candidate_id"].astype(str).eq(true_candidate).to_numpy()
                ].sum()
            ),
            "true_family_posterior": float(family_posterior.get(true_family, 0.0)),
            "candidate_posterior_entropy": float(
                -np.sum(candidate_posterior * np.log(np.clip(candidate_posterior, 1e-300, 1.0)))
            ),
            "generated_accuracy": float(frame["generated_accuracy"].iloc[0]),
        }
        if scope == "within":
            row["posterior_mean_value"] = float(
                np.sum(candidate_posterior * frame["fit_value"].to_numpy(dtype=float))
            )
            row["absolute_value_error"] = abs(
                float(row["posterior_mean_value"]) - float(row["true_value"])
            )
        recovered_rows.append(row)
    recovered = pd.DataFrame(recovered_rows)

    summary_rows: list[dict[str, Any]] = []
    for family, frame in recovered.groupby("true_family", sort=True):
        summary: dict[str, Any] = {
            "scope": scope,
            "true_family": family,
            "dataset_n": int(len(frame)),
            "exact_candidate_accuracy": float(frame["exact_candidate_recovered"].mean()),
            "family_accuracy": float(frame["family_recovered"].mean()),
            "mean_true_candidate_posterior": float(frame["true_candidate_posterior"].mean()),
            "mean_true_family_posterior": float(frame["true_family_posterior"].mean()),
        }
        if scope == "within":
            correlation = spearmanr(
                frame["true_value"].to_numpy(dtype=float),
                frame["posterior_mean_value"].to_numpy(dtype=float),
            )
            summary.update(
                {
                    "mean_absolute_value_error": float(frame["absolute_value_error"].mean()),
                    "spearman_true_posterior_mean": float(correlation.statistic),
                    "spearman_p": float(correlation.pvalue),
                }
            )
        summary_rows.append(summary)
    summary = pd.DataFrame(summary_rows)
    confusion = (
        recovered.groupby(["true_family", "predicted_family"], sort=True)
        .size()
        .rename("dataset_n")
        .reset_index()
    )
    confusion["row_proportion"] = confusion["dataset_n"] / confusion.groupby(
        "true_family"
    )["dataset_n"].transform("sum")
    return recovered, summary, confusion


def main() -> None:
    args = parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    if int(args.datasets_per_candidate) < 1 or int(args.particle_count) < 2:
        raise ValueError("datasets-per-candidate must be positive and particles >= 2.")

    data = pd.read_csv(args.data)
    data = (
        data.loc[data["condition"].eq(1)]
        .sort_values(list(KEY_COLUMNS))
        .reset_index(drop=True)
    )
    subjects = sorted({int(value) for value in args.subjects})
    available = set(data["iSub"].astype(int).unique())
    if missing := sorted(set(subjects) - available):
        raise ValueError(f"Recovery subjects are absent from condition 1: {missing}")
    base_engine = load_yaml(args.engine_config)
    simulation_config = load_yaml(args.simulation_config)
    dataset_paths = resolve_dataset_paths(
        simulation_config, args.simulation_config.parent, DEFAULT_DATA_PATH
    )
    grid = candidate_grid(args.families, shared_theta=float(args.shared_theta))
    if args.scope == "within":
        jobs = [
            (candidate, list(grid[family]), subject, replicate)
            for family in grid
            for candidate in grid[family]
            for subject in subjects
            for replicate in range(int(args.datasets_per_candidate))
        ]
    else:
        bank = cross_candidate_bank(grid)
        jobs = [
            (candidate, bank, subject, replicate)
            for candidate in bank
            for subject in subjects
            for replicate in range(int(args.datasets_per_candidate))
        ]
    frames = {subject: data.loc[data["iSub"].eq(subject)].copy() for subject in subjects}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        args.output_dir / "manifest.json",
        {
            "analysis": "condition1_mechanism_recovery",
            "scope": args.scope,
            "subjects": subjects,
            "development_subject_pool": [int(value) for value in DEVELOPMENT_SUBJECTS],
            "families": list(grid),
            "dataset_n": len(jobs),
            "datasets_per_candidate": int(args.datasets_per_candidate),
            "particle_count": int(args.particle_count),
            "rho": float(args.rho),
            "epsilon": float(args.epsilon),
            "base_seed": int(args.base_seed),
            "paired_filter_seeds_across_fit_candidates": True,
            "synthetic_schedule": "actual_subject_stimulus_and_category_prefix",
        },
    )
    result = Parallel(
        n_jobs=min(int(args.n_jobs), len(jobs)), backend="loky", verbose=10
    )(
        delayed(recover_one_dataset)(
            args=args,
            subject_frame=frames[subject],
            base_engine=base_engine,
            dataset_paths=dataset_paths,
            true_candidate=true_candidate,
            fit_candidates=fit_candidates,
            replicate=replicate,
        )
        for true_candidate, fit_candidates, subject, replicate in jobs
    )
    scores = pd.DataFrame([row for dataset_rows in result for row in dataset_rows])
    scores.to_csv(args.output_dir / "fit_scores.csv", index=False)
    recovered, summary, confusion = summarize_recovery(scores, scope=str(args.scope))
    recovered.to_csv(args.output_dir / "recovered_datasets.csv", index=False)
    summary.to_csv(args.output_dir / "recovery_summary.csv", index=False)
    confusion.to_csv(args.output_dir / "family_confusion.csv", index=False)
    print(summary.to_json(orient="records", indent=2), flush=True)


if __name__ == "__main__":
    main()
