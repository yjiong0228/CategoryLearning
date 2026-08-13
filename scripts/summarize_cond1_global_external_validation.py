#!/usr/bin/env python3
"""External validation for the frozen cross-mechanism predictive bank.

The script reuses full-sequence candidate-filter caches from
``run_cond1_mechanism_external_validation.py``.  Candidate weights were fit
from prefix choices only and remain fixed; suffix choices, RT, and oral reports
are evaluation data.  The output is predictive evidence and does not identify
which cognitive mechanism generated a participant's trajectory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cond1_b0_trajectory_ppc import (  # noqa: E402
    KEY_COLUMNS,
    split_for_subject,
)
from scripts.run_cond1_mechanism_external_validation import (  # noqa: E402
    _benjamini_hochberg,
    _bootstrap,
    _cache_path,
    _load_cache,
    _oral_center_similarity,
    _paired_signflip_p,
    _safe_spearman,
)
from src.Bayesian_state.hypothesis_space import ContinuousPartition  # noqa: E402
from src.Bayesian_state.utils.seeding import stable_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data/processed/Task2_processed.csv"
    )
    parser.add_argument("--posterior-csv", type=Path, required=True)
    parser.add_argument("--candidate-filter-dir", type=Path, required=True)
    parser.add_argument("--particle-count", type=int, default=64)
    parser.add_argument("--cohort", default="reserved")
    parser.add_argument("--oral-beta", type=float, default=10.0)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--bootstrap-repeats", type=int, default=20000)
    parser.add_argument("--base-seed", type=int, default=20261303)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _subject_row(
    *,
    args: argparse.Namespace,
    frame: pd.DataFrame,
    posterior: pd.DataFrame,
    partition: ContinuousPartition,
) -> dict[str, Any]:
    subject_id = int(frame["iSub"].iloc[0])
    posterior = posterior.sort_values(
        ["mechanism", "source_family", "candidate_id"]
    ).reset_index(drop=True)
    weights = posterior["posterior_weight"].to_numpy(dtype=float)
    weights /= weights.sum()
    caches = [
        _load_cache(
            _cache_path(
                args.candidate_filter_dir,
                str(row.source_family),
                str(row.candidate_id),
                subject_id,
                int(args.particle_count),
            )
        )
        for row in posterior.itertuples(index=False)
    ]
    reference_indices = np.flatnonzero(
        posterior["source_family"].eq("F").to_numpy()
        & posterior["candidate_id"].eq("F_kappa_1").to_numpy()
    )
    if reference_indices.size != 1:
        raise ValueError(f"Expected one F_kappa_1 reference for subject {subject_id}.")
    reference_index = int(reference_indices[0])
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
    mixture_rt = _safe_spearman(mixture_surprise, log_rt)
    reference_rt = _safe_spearman(reference_surprise, log_rt)
    entropy = float(-np.sum(weights * np.log(np.clip(weights, 1e-300, 1.0))))
    return {
        "cohort": args.cohort,
        "subject_id": subject_id,
        "readout": "static",
        "family": "GLOBAL",
        "split_status": split_status,
        "prefix_n": int(split_index),
        "suffix_n": int(suffix.size),
        "candidate_n": int(weights.size),
        "posterior_entropy": entropy,
        "effective_candidate_n": float(np.exp(entropy)),
        "max_candidate_posterior": float(weights.max()),
        "mixture_suffix_nll": float(np.mean(mixture_surprise)),
        "reference_suffix_nll": float(np.mean(reference_surprise)),
        "delta_suffix_nll": float(np.mean(mixture_surprise - reference_surprise)),
        "mixture_rt_surprise_spearman": mixture_rt,
        "reference_rt_surprise_spearman": reference_rt,
        "delta_rt_surprise_spearman": mixture_rt - reference_rt,
        "oral_valid_n": int(np.sum(np.isfinite(oral_mixture) & np.isfinite(oral_reference))),
        "mixture_oral_center_similarity": float(np.nanmean(oral_mixture)),
        "reference_oral_center_similarity": float(np.nanmean(oral_reference)),
        "delta_oral_center_similarity": float(np.nanmean(oral_mixture - oral_reference)),
    }


def main() -> None:
    args = parse_args()
    posteriors = pd.read_csv(args.posterior_csv)
    posteriors = posteriors.loc[posteriors["readout"].eq("static")].copy()
    subjects = sorted(int(value) for value in posteriors["subject_id"].unique())
    data = pd.read_csv(args.data)
    data = (
        data.loc[data["condition"].eq(1) & data["iSub"].isin(subjects)]
        .sort_values(list(KEY_COLUMNS))
        .reset_index(drop=True)
    )
    frames = {subject: data.loc[data["iSub"].eq(subject)].copy() for subject in subjects}
    partition = ContinuousPartition(n_dims=4, n_cats=2)
    subject_rows = [
        _subject_row(
            args=args,
            frame=frames[subject],
            posterior=posteriors.loc[posteriors["subject_id"].eq(subject)],
            partition=partition,
        )
        for subject in subjects
    ]
    subject_summary = pd.DataFrame(subject_rows)
    metric_direction = {
        "delta_suffix_nll": "negative",
        "delta_rt_surprise_spearman": "positive",
        "delta_oral_center_similarity": "positive",
    }
    group_rows: list[dict[str, Any]] = []
    for metric, direction in metric_direction.items():
        values = subject_summary[metric].to_numpy(dtype=float)
        group_rows.append(
            {
                "cohort": args.cohort,
                "readout": "static",
                "family": "GLOBAL",
                "metric": metric,
                "delta_definition": "mixture_minus_reference",
                "better_direction": direction,
                **_bootstrap(
                    values,
                    seed=stable_seed(
                        {
                            "seed_role": "global_external_bootstrap",
                            "base_seed": int(args.base_seed),
                            "metric": metric,
                        }
                    ),
                    repeats=int(args.bootstrap_repeats),
                ),
                "paired_signflip_p": _paired_signflip_p(
                    values,
                    seed=stable_seed(
                        {
                            "seed_role": "global_external_signflip",
                            "base_seed": int(args.base_seed),
                            "metric": metric,
                        }
                    ),
                    repeats=int(args.bootstrap_repeats),
                ),
                "subject_n": int(np.isfinite(values).sum()),
            }
        )
    q_values = _benjamini_hochberg(
        [row["paired_signflip_p"] for row in group_rows]
    )
    for row, q_value in zip(group_rows, q_values):
        row["paired_signflip_q"] = float(q_value)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    subject_summary.to_csv(args.output_dir / "subject_validation.csv", index=False)
    pd.DataFrame(group_rows).to_csv(args.output_dir / "group_validation.csv", index=False)
    summary = {
        "analysis": "condition1_global_predictive_external_validation",
        "cohort": args.cohort,
        "subjects": subjects,
        "candidate_weights_source": str(args.posterior_csv),
        "candidate_filter_source": str(args.candidate_filter_dir),
        "weights_frozen_during_suffix": True,
        "rt_used_for_fitting": False,
        "oral_report_used_for_fitting": False,
        "mechanistic_label_claim": False,
        "group_rows": group_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
