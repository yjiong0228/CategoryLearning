#!/usr/bin/env python3
"""Summarize Monte Carlo replication of the development mechanism screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_a = pd.read_csv(args.run_a / "comparison_summary.csv")
    comparison_b = pd.read_csv(args.run_b / "comparison_summary.csv")
    comparison_keys = ["readout", "family", "metric"]
    comparison = comparison_a.merge(
        comparison_b, on=comparison_keys, suffixes=("_seed_a", "_seed_b")
    )
    comparison["same_effect_sign"] = np.sign(comparison["mean_seed_a"]) == np.sign(
        comparison["mean_seed_b"]
    )
    comparison["mean_average"] = comparison[
        ["mean_seed_a", "mean_seed_b"]
    ].mean(axis=1)
    comparison["both_ci_exclude_zero"] = (
        (comparison["ci025_seed_a"] > 0) & (comparison["ci025_seed_b"] > 0)
    ) | ((comparison["ci975_seed_a"] < 0) & (comparison["ci975_seed_b"] < 0))

    prior_a = pd.read_csv(args.run_a / "population_priors.csv")
    prior_b = pd.read_csv(args.run_b / "population_priors.csv")
    prior = prior_a.merge(
        prior_b,
        on=["readout", "family", "candidate_id"],
        suffixes=("_seed_a", "_seed_b"),
    )
    prior["absolute_delta"] = np.abs(
        prior["population_prior_seed_a"] - prior["population_prior_seed_b"]
    )
    prior_stability = (
        prior.groupby(["readout", "family"], sort=True)
        .apply(
            lambda frame: pd.Series(
                {
                    "pearson_r": frame["population_prior_seed_a"].corr(
                        frame["population_prior_seed_b"]
                    ),
                    "max_absolute_delta": frame["absolute_delta"].max(),
                    "mean_absolute_delta": frame["absolute_delta"].mean(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    posterior_a = pd.read_csv(args.run_a / "subject_candidate_posteriors.csv")
    posterior_b = pd.read_csv(args.run_b / "subject_candidate_posteriors.csv")
    posterior_columns = ["readout", "family", "subject_id", "posterior_mean_value"]
    posterior_a = posterior_a.drop_duplicates(posterior_columns[:3])[posterior_columns]
    posterior_b = posterior_b.drop_duplicates(posterior_columns[:3])[posterior_columns]
    posterior = posterior_a.merge(
        posterior_b,
        on=posterior_columns[:3],
        suffixes=("_seed_a", "_seed_b"),
    )
    posterior["absolute_delta"] = np.abs(
        posterior["posterior_mean_value_seed_a"]
        - posterior["posterior_mean_value_seed_b"]
    )
    posterior_stability = (
        posterior.groupby(["readout", "family"], sort=True)
        .apply(
            lambda frame: pd.Series(
                {
                    "pearson_r": frame["posterior_mean_value_seed_a"].corr(
                        frame["posterior_mean_value_seed_b"]
                    ),
                    "mean_absolute_delta": frame["absolute_delta"].mean(),
                    "max_absolute_delta": frame["absolute_delta"].max(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "suffix_metric_replication.csv", index=False)
    prior.to_csv(args.output_dir / "population_prior_replication.csv", index=False)
    prior_stability.to_csv(
        args.output_dir / "population_prior_stability.csv", index=False
    )
    posterior.to_csv(
        args.output_dir / "subject_posterior_mean_replication.csv", index=False
    )
    posterior_stability.to_csv(
        args.output_dir / "subject_posterior_mean_stability.csv", index=False
    )
    summary = {
        "analysis": "condition1_mechanism_screen_monte_carlo_replication",
        "run_a": str(args.run_a),
        "run_b": str(args.run_b),
        "comparison_row_n": int(len(comparison)),
        "same_effect_sign_n": int(comparison["same_effect_sign"].sum()),
        "both_ci_exclude_zero_n": int(comparison["both_ci_exclude_zero"].sum()),
        "stable_nonzero_rows": comparison.loc[
            comparison["both_ci_exclude_zero"], comparison_keys + ["mean_average"]
        ].to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
