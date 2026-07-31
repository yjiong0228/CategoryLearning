#!/usr/bin/env python3
"""Combine the screened F/M/H/P candidates into one predictive bank.

This is deliberately a predictive, not mechanistic-label, analysis.  The
development cohort estimates one finite-mixture prior over the unique
candidates.  Target-subject prefix choices update those frozen weights, and
the resulting mixture is evaluated only on the suffix.  Existing particle
filter and rollout caches are reused; no target suffix choice enters the
weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cond1_mechanism_screen import (  # noqa: E402
    _benjamini_hochberg,
    _bootstrap_delta,
    _fit_population_prior,
    _mixture_cache,
    _paired_signflip_p,
    _softmax,
    evaluate_subject,
    load_subject_cache,
    stable_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--readout", nargs="+", default=["static", "c1"])
    parser.add_argument("--em-alpha", type=float, default=1.0)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--base-seed", type=int, default=20261321)
    return parser.parse_args()


def _candidate_bank(scores: pd.DataFrame, readout: str) -> list[tuple[str, str, str]]:
    selected = scores.loc[scores["readout"].eq(readout)].copy()
    available = {
        (str(row.family), str(row.candidate_id)): bool(row.is_reference)
        for row in selected[["family", "candidate_id", "is_reference"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    baseline = ("F", "F_kappa_1")
    if baseline not in available or not available[baseline]:
        raise ValueError(f"Missing canonical baseline candidate for {readout}: {baseline}")
    bank = [("BASE", *baseline)]
    for family in ("F", "M", "H", "P"):
        candidates = sorted(
            candidate_id
            for (candidate_family, candidate_id), is_reference in available.items()
            if candidate_family == family and not is_reference
        )
        bank.extend((family, family, candidate_id) for candidate_id in candidates)
    return bank


def _matrix(
    scores: pd.DataFrame,
    *,
    readout: str,
    subjects: list[int],
    bank: list[tuple[str, str, str]],
) -> np.ndarray:
    lookup = scores.loc[scores["readout"].eq(readout)].set_index(
        ["subject_id", "family", "candidate_id"]
    )["prefix_log_predictive_density"]
    matrix = np.empty((len(subjects), len(bank)), dtype=float)
    for subject_index, subject in enumerate(subjects):
        for candidate_index, (_, source_family, candidate_id) in enumerate(bank):
            matrix[subject_index, candidate_index] = float(
                lookup.loc[(subject, source_family, candidate_id)]
            )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Prefix evidence matrix contains non-finite values.")
    return matrix


def _cache_lookup(index: pd.DataFrame) -> dict[tuple[int, str, str, str], Path]:
    return {
        (int(row.subject_id), str(row.readout), str(row.family), str(row.candidate_id)): Path(row.cache)
        for row in index.itertuples(index=False)
    }


def _discover_cache_lookup(target_dir: Path) -> dict[tuple[int, str, str, str], Path]:
    lookup: dict[tuple[int, str, str, str], Path] = {}
    pattern = "candidate_runs/*/*/*/cache/subject_*/particles_*/rollouts_*.npz"
    for path in target_dir.glob(pattern):
        relative = path.relative_to(target_dir).parts
        readout, family, candidate_id = relative[1:4]
        subject_id = int(relative[5].split("_", 1)[1])
        key = (subject_id, readout, family, candidate_id)
        if key in lookup:
            raise ValueError(f"Multiple candidate caches match {key}: {lookup[key]} and {path}")
        lookup[key] = path
    if not lookup:
        raise FileNotFoundError(f"No candidate caches found below {target_dir}.")
    return lookup


def main() -> None:
    args = parse_args()
    dev_scores = pd.read_csv(args.development_dir / "prefix_candidate_scores.csv")
    target_scores = pd.read_csv(args.target_dir / "prefix_candidate_scores.csv")
    index_path = args.target_dir / "candidate_run_index.csv"
    cache_lookup = (
        _cache_lookup(pd.read_csv(index_path))
        if index_path.exists()
        else _discover_cache_lookup(args.target_dir)
    )
    subjects = sorted(int(value) for value in target_scores["subject_id"].unique())

    prior_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for readout in dict.fromkeys(args.readout):
        bank = _candidate_bank(dev_scores, readout)
        dev_subjects = sorted(
            int(value)
            for value in dev_scores.loc[dev_scores["readout"].eq(readout), "subject_id"].unique()
        )
        dev_evidence = _matrix(
            dev_scores,
            readout=readout,
            subjects=dev_subjects,
            bank=bank,
        )
        prior, _, iterations = _fit_population_prior(
            dev_evidence,
            alpha=float(args.em_alpha),
        )
        target_evidence = _matrix(
            target_scores,
            readout=readout,
            subjects=subjects,
            bank=bank,
        )
        responsibilities = np.stack(
            [
                _softmax(np.log(np.clip(prior, 1e-300, 1.0)) + evidence)
                for evidence in target_evidence
            ]
        )

        for candidate_index, (mechanism, source_family, candidate_id) in enumerate(bank):
            prior_rows.append(
                {
                    "readout": readout,
                    "mechanism": mechanism,
                    "source_family": source_family,
                    "candidate_id": candidate_id,
                    "population_prior": float(prior[candidate_index]),
                    "prior_source": str(args.development_dir),
                    "em_iterations": int(iterations),
                }
            )

        mixture_by_subject: dict[int, dict[str, Any]] = {}
        reference_by_subject: dict[int, dict[str, Any]] = {}
        for subject_index, subject in enumerate(subjects):
            caches = [
                load_subject_cache(
                    cache_lookup[(subject, readout, source_family, candidate_id)]
                )
                for _, source_family, candidate_id in bank
            ]
            for candidate_index, (mechanism, source_family, candidate_id) in enumerate(bank):
                posterior_rows.append(
                    {
                        "cohort": args.cohort,
                        "subject_id": subject,
                        "readout": readout,
                        "mechanism": mechanism,
                        "source_family": source_family,
                        "candidate_id": candidate_id,
                        "posterior_weight": float(responsibilities[subject_index, candidate_index]),
                    }
                )
            mixture_path = (
                args.output_dir / "mixtures" / readout / f"subject_{subject}.npz"
            )
            mixture_cache = _mixture_cache(
                caches=caches,
                posterior=responsibilities[subject_index],
                seed=stable_seed(
                    {
                        "seed_role": "global_mechanism_mixture",
                        "base_seed": int(args.base_seed),
                        "subject_id": subject,
                        "readout": readout,
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
                    "family": "GLOBAL",
                    "model": "global_candidate_bank",
                }
            )
            subject_rows.append(mixture_summary)
            mixture_by_subject[subject] = mixture_summary

            reference_summary, _, _, _ = evaluate_subject(
                caches[0],
                window=int(args.window),
            )
            reference_summary.update(
                {
                    "cohort": args.cohort,
                    "readout": readout,
                    "family": "GLOBAL",
                    "model": "reference_candidate",
                }
            )
            subject_rows.append(reference_summary)
            reference_by_subject[subject] = reference_summary

        for metric in (
            "curve_crps",
            "summary_discrepancy",
            "combined_calibration_p",
            "curve_pointwise_interval_width_95",
        ):
            deltas = np.asarray(
                [
                    float(mixture_by_subject[subject][metric])
                    - float(reference_by_subject[subject][metric])
                    for subject in subjects
                ],
                dtype=float,
            )
            lower_is_better = metric != "combined_calibration_p"
            comparison_rows.append(
                {
                    "cohort": args.cohort,
                    "readout": readout,
                    "family": "GLOBAL",
                    "metric": metric,
                    "delta_definition": "mixture_minus_reference",
                    "better_direction": "negative" if lower_is_better else "positive",
                    **_bootstrap_delta(
                        deltas,
                        seed=stable_seed(
                            {
                                "seed_role": "global_mixture_bootstrap",
                                "base_seed": int(args.base_seed),
                                "readout": readout,
                                "metric": metric,
                            }
                        ),
                        repeats=int(args.bootstrap_repeats),
                    ),
                    "paired_signflip_p": _paired_signflip_p(
                        deltas,
                        seed=stable_seed(
                            {
                                "seed_role": "global_mixture_signflip",
                                "base_seed": int(args.base_seed),
                                "readout": readout,
                                "metric": metric,
                            }
                        ),
                        repeats=int(args.bootstrap_repeats),
                    ),
                    "improved_subject_n": int(
                        np.sum(deltas < 0.0 if lower_is_better else deltas > 0.0)
                    ),
                    "subject_n": len(subjects),
                }
            )

        mixture_pass = np.asarray(
            [bool(mixture_by_subject[subject]["combined_pass_95"]) for subject in subjects]
        )
        reference_pass = np.asarray(
            [bool(reference_by_subject[subject]["combined_pass_95"]) for subject in subjects]
        )
        improved = int(np.sum(mixture_pass & ~reference_pass))
        worsened = int(np.sum(~mixture_pass & reference_pass))
        discordant = improved + worsened
        coverage_rows.append(
            {
                "cohort": args.cohort,
                "readout": readout,
                "mixture_pass_n": int(mixture_pass.sum()),
                "reference_pass_n": int(reference_pass.sum()),
                "subject_n": len(subjects),
                "improved_n": improved,
                "worsened_n": worsened,
                "exact_p": (
                    float(binomtest(improved, discordant, 0.5).pvalue)
                    if discordant
                    else 1.0
                ),
            }
        )

    comparison_q = _benjamini_hochberg(
        [row["paired_signflip_p"] for row in comparison_rows]
    )
    for row, q_value in zip(comparison_rows, comparison_q):
        row["paired_signflip_q"] = float(q_value)
    coverage_q = _benjamini_hochberg([row["exact_p"] for row in coverage_rows])
    for row, q_value in zip(coverage_rows, coverage_q):
        row["exact_q"] = float(q_value)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prior_rows).to_csv(args.output_dir / "population_priors.csv", index=False)
    pd.DataFrame(posterior_rows).to_csv(
        args.output_dir / "subject_candidate_posteriors.csv", index=False
    )
    pd.DataFrame(subject_rows).to_csv(
        args.output_dir / "mixture_subject_summary.csv", index=False
    )
    pd.DataFrame(comparison_rows).to_csv(
        args.output_dir / "comparison_summary.csv", index=False
    )
    pd.DataFrame(coverage_rows).to_csv(
        args.output_dir / "coverage_comparison.csv", index=False
    )
    summary = {
        "analysis": "condition1_global_predictive_mechanism_mixture",
        "cohort": args.cohort,
        "development_dir": str(args.development_dir),
        "target_dir": str(args.target_dir),
        "subjects": subjects,
        "future_observed_choices_read": False,
        "future_feedback_generated_from_simulated_choices": True,
        "mechanistic_label_claim": False,
        "comparisons": comparison_rows,
        "coverage": coverage_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
