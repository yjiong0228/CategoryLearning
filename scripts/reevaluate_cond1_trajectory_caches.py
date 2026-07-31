#!/usr/bin/env python3
"""Re-evaluate existing condition-1 rollout caches with the current metrics.

This script does not simulate, fit, or condition on any additional data.  It
only applies the current predeclared trajectory discrepancy vector and
calibration code to caches produced by ``run_cond1_b0_trajectory_ppc.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_cond1_b0_trajectory_ppc import (
    METRIC_SPECS,
    benjamini_hochberg,
    cohort_calibration,
    evaluate_subject,
    load_subject_cache,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Existing result directory containing cache/**/*.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh directory for the re-evaluated summaries.",
    )
    parser.add_argument("--window", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    cache_paths = sorted(source_dir.glob("cache/**/*.npz"))
    if not cache_paths:
        raise FileNotFoundError(f"No rollout caches found under {source_dir}")
    if args.window < 4:
        raise ValueError("window must be at least 4.")

    subject_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    simulated_pass: dict[int, np.ndarray] = {}
    for path in cache_paths:
        cache = load_subject_cache(path)
        subject_row, local_metrics, local_curve, local_sim_pass = (
            evaluate_subject(cache, window=int(args.window))
        )
        subject_rows.append(subject_row)
        metric_rows.extend(local_metrics)
        curve_rows.extend(local_curve)
        simulated_pass[int(subject_row["iSub"])] = local_sim_pass

    subjects = pd.DataFrame(subject_rows).sort_values("iSub")
    if subjects["iSub"].duplicated().any():
        duplicates = subjects.loc[
            subjects["iSub"].duplicated(keep=False), "iSub"
        ].tolist()
        raise ValueError(
            "Expected one cache per subject; duplicate subjects: "
            f"{duplicates}"
        )
    subjects["combined_calibration_fdr_q"] = benjamini_hochberg(
        subjects["combined_calibration_p"].to_numpy(dtype=float)
    )
    metrics = pd.DataFrame(metric_rows)
    curves = pd.DataFrame(curve_rows)
    cohorts = cohort_calibration(subjects, simulated_pass)
    failures = (
        metrics.loc[~metrics["inside_marginal_95"]]
        .groupby(["metric", "metric_label"], as_index=False)
        .agg(
            failed_subject_n=("iSub", "nunique"),
            failed_observation_n=("iSub", "size"),
        )
        .sort_values(
            ["failed_subject_n", "metric"],
            ascending=[False, True],
        )
    )
    if failures.empty:
        failures = pd.DataFrame(
            columns=[
                "metric",
                "metric_label",
                "failed_subject_n",
                "failed_observation_n",
            ]
        )

    all_cohort = cohorts.loc[cohorts["cohort"].eq("all_subjects")].iloc[0]
    reserved = cohorts.loc[
        cohorts["cohort"].eq("reserved_application")
    ]
    cohort_gate = bool(
        all_cohort["lower_tail_calibration_p"] >= 0.05
        and (
            reserved.empty
            or reserved.iloc[0]["lower_tail_calibration_p"] >= 0.05
        )
    )
    fdr_failure_n = int(
        np.sum(subjects["combined_calibration_fdr_q"] <= 0.05)
    )
    decision = {
        "generative_adequacy": (
            "adequate_at_cohort_level"
            if cohort_gate
            else "systematic_coverage_failure"
        ),
        "recommended_model_action": (
            "retain_as_comparator"
            if cohort_gate
            else "do_not_retain_as_adequate_comparator"
        ),
        "subject_level_fdr_failures": fdr_failure_n,
        "interpretation": (
            "This is a metric-only re-evaluation of existing autonomous "
            "suffix rollouts. It adds no fitting, simulation, or observed "
            "future information and does not identify a unique mechanism."
        ),
        "sharpness": {
            "median_90pct_rolling_interval_width": float(
                subjects["curve_pointwise_interval_width_90"].median()
            ),
            "median_95pct_rolling_interval_width": float(
                subjects["curve_pointwise_interval_width_95"].median()
            ),
            "mean_curve_crps": float(subjects["curve_crps"].mean()),
        },
        "cohort_calibration": cohorts.to_dict(orient="records"),
    }
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.exists()
        else {}
    )
    manifest = {
        "analysis": "condition1_cached_rollout_metric_reevaluation",
        "source_dir": str(source_dir.relative_to(ROOT)),
        "source_model": source_manifest.get("model"),
        "source_particle_count": source_manifest.get("particle_count"),
        "source_rollout_count": source_manifest.get("rollout_count"),
        "window": int(args.window),
        "metric_specs": [spec.__dict__ for spec in METRIC_SPECS],
        "metric_n": len(METRIC_SPECS),
        "new_simulation_performed": False,
        "new_parameter_selection_performed": False,
        "future_observed_choices_read_by_rollout": False,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "decision.json", decision)
    subjects.to_csv(output_dir / "subject_summary.csv", index=False)
    metrics.to_csv(output_dir / "metric_summary.csv", index=False)
    curves.to_csv(output_dir / "rolling_curve_summary.csv", index=False)
    cohorts.to_csv(output_dir / "cohort_calibration.csv", index=False)
    failures.to_csv(output_dir / "metric_failures.csv", index=False)
    print(
        json.dumps(
            {
                "source_dir": str(source_dir),
                "subjects": int(len(subjects)),
                "metric_n": len(METRIC_SPECS),
                "combined_pass_n": int(subjects["combined_pass_95"].sum()),
                "fdr_failure_n": fdr_failure_n,
                "cohort_p": float(
                    all_cohort["lower_tail_calibration_p"]
                ),
                "mean_curve_crps": float(subjects["curve_crps"].mean()),
                "output_dir": str(output_dir),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
