#!/usr/bin/env python3
"""Consolidate frozen dynamic-rho results and paired diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
KEY_COLUMNS = ("iSub", "iSession", "iBlock", "iTrial")
FEATURE_COLUMNS = ("feature1", "feature2", "feature3", "feature4")
MORPHOLOGY_METRICS = (
    "accuracy",
    "accuracy_slope",
    "max_adjacent_rise",
    "max_adjacent_drop",
    "trend_reversal_count",
    "event_count",
    "max_event_duration",
    "late_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/zhuran/cond1_active_set",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/processed/Task2_processed.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_active_set/"
            "dynamic_rho_consolidated"
        ),
    )
    parser.add_argument("--bootstrap-count", type=int, default=10000)
    parser.add_argument("--base-seed", type=int, default=20261111)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def all_cohort(path: Path) -> pd.Series:
    frame = pd.read_csv(path)
    return frame.loc[frame["cohort"].eq("all_subjects")].iloc[0]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = pd.Series(np.asarray(x, dtype=float)).rank(
        method="average"
    )
    y_rank = pd.Series(np.asarray(y, dtype=float)).rank(
        method="average"
    )
    return float(x_rank.corr(y_rank))


def subject_bootstrap(
    values: np.ndarray,
    *,
    statistic,
    count: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    array = np.asarray(values)
    estimates: list[float] = []
    for _ in range(int(count)):
        indices = rng.integers(0, len(array), size=len(array))
        estimate = float(statistic(array[indices]))
        if np.isfinite(estimate):
            estimates.append(estimate)
    if not estimates:
        return np.nan, np.nan
    return tuple(
        float(value)
        for value in np.quantile(estimates, [0.025, 0.975])
    )


def model_row(
    *,
    label: str,
    cohort_label: str,
    decision_path: Path,
    cohort_path: Path,
    metric_n: int,
    seed_role: str,
) -> dict[str, Any]:
    decision = read_json(decision_path)
    cohort = all_cohort(cohort_path)
    return {
        "model": label,
        "cohort": cohort_label,
        "metric_n": int(metric_n),
        "seed_role": seed_role,
        "subject_n": int(cohort["subject_n"]),
        "observed_pass_n": int(cohort["observed_pass_n"]),
        "observed_pass_fraction": float(
            cohort["observed_pass_fraction"]
        ),
        "self_expected_pass_mean": float(
            cohort["b0_self_expected_pass_mean"]
        ),
        "self_expected_pass_fraction": float(
            cohort["b0_self_expected_pass_mean"]
            / cohort["subject_n"]
        ),
        "self_expected_pass_q025": float(
            cohort["b0_self_expected_pass_q025"]
        ),
        "self_expected_pass_q975": float(
            cohort["b0_self_expected_pass_q975"]
        ),
        "cohort_calibration_p": float(
            cohort["lower_tail_calibration_p"]
        ),
        "fdr_failure_n": int(decision["subject_level_fdr_failures"]),
        "mean_curve_crps": float(
            decision["sharpness"]["mean_curve_crps"]
        ),
        "median_width_95": float(
            decision["sharpness"][
                "median_95pct_rolling_interval_width"
            ]
        ),
    }


def candidate_model_row(
    *,
    label: str,
    cohort_label: str,
    candidate_path: Path,
    seed_role: str,
) -> dict[str, Any]:
    decision = read_json(candidate_path / "candidate_decision.json")
    return {
        "model": label,
        "cohort": cohort_label,
        "metric_n": 15,
        "seed_role": seed_role,
        "subject_n": int(decision["subject_n"]),
        "observed_pass_n": int(decision["combined_pass_n"]),
        "observed_pass_fraction": float(
            decision["combined_pass_fraction"]
        ),
        "self_expected_pass_mean": float(
            decision["self_expected_pass_mean"]
        ),
        "self_expected_pass_fraction": float(
            decision["self_expected_pass_mean"]
            / decision["subject_n"]
        ),
        "self_expected_pass_q025": float(
            decision["self_expected_pass_q025"]
        ),
        "self_expected_pass_q975": float(
            decision["self_expected_pass_q975"]
        ),
        "cohort_calibration_p": float(
            decision["lower_tail_calibration_p"]
        ),
        "fdr_failure_n": int(decision["fdr_failure_n"]),
        "mean_curve_crps": float(decision["mean_curve_crps"]),
        "median_width_95": float(
            decision["median_curve_interval_width_95"]
        ),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_count < 1000:
        raise ValueError("bootstrap-count must be at least 1000.")
    root = args.results_root
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.base_seed))

    static = (
        root / "fullset_trajectory_ppc_early_anchor_metrics15"
    )
    acquisition = (
        root
        / "acquisition_changepoint_reserved_h128_p256_r1024_metrics15"
    )
    dynamic = (
        root
        / "dynamic_rho_reserved_c1_p256_r1024/"
        "c1_s0p5_e0p5_v0p2_p0p95"
    )
    dynamic_seed = (
        root
        / "dynamic_rho_reserved_c1_seed20261101_p128_r512/"
        "c1_s0p5_e0p5_v0p2_p0p95"
    )

    model_comparison = pd.DataFrame(
        [
            model_row(
                label="静态 full-set｜第一块后",
                cohort_label="全部32人",
                decision_path=static / "decision.json",
                cohort_path=static / "cohort_calibration.csv",
                metric_n=15,
                seed_role="历史缓存重评分",
            ),
            model_row(
                label="单次掌握变点｜冻结保留集",
                cohort_label="保留24人",
                decision_path=acquisition / "decision.json",
                cohort_path=acquisition / "cohort_calibration.csv",
                metric_n=15,
                seed_role="正式",
            ),
            candidate_model_row(
                label="连续动态 readout C1｜冻结保留集",
                cohort_label="保留24人",
                candidate_path=dynamic,
                seed_role="正式",
            ),
            candidate_model_row(
                label="连续动态 readout C1｜独立种子",
                cohort_label="保留24人",
                candidate_path=dynamic_seed,
                seed_role="数值复核",
            ),
        ]
    )
    model_comparison.to_csv(
        output / "model_comparison.csv", index=False
    )

    acquisition_subject = pd.read_csv(
        acquisition / "subject_summary.csv"
    )
    dynamic_subject = pd.read_csv(dynamic / "subject_summary.csv")
    paired = dynamic_subject[
        [
            "iSub",
            "curve_crps",
            "combined_pass_95",
            "combined_calibration_p",
            "combined_calibration_fdr_q",
            "boundary_rho_posterior_mean",
            "boundary_rho_volatility_posterior_mean",
            "suffix_rho_mean",
            "suffix_rho_within_trajectory_sd_mean",
        ]
    ].merge(
        acquisition_subject[
            ["iSub", "curve_crps", "combined_pass_95"]
        ],
        on="iSub",
        suffixes=("_C1", "_acquisition"),
        validate="one_to_one",
    )
    paired["crps_difference_C1_minus_acquisition"] = (
        paired["curve_crps_C1"] - paired["curve_crps_acquisition"]
    )

    dynamic_metrics = pd.read_csv(dynamic / "metric_summary.csv")
    observed_wide = (
        dynamic_metrics.loc[
            dynamic_metrics["metric"].isin(MORPHOLOGY_METRICS),
            ["iSub", "metric", "observed"],
        ]
        .pivot(index="iSub", columns="metric", values="observed")
        .reset_index()
    )
    paired = paired.merge(
        observed_wide,
        on="iSub",
        validate="one_to_one",
    )
    paired.to_csv(
        output / "subject_dynamic_diagnostics.csv", index=False
    )

    crps_difference = paired[
        "crps_difference_C1_minus_acquisition"
    ].to_numpy(dtype=float)
    crps_ci = subject_bootstrap(
        crps_difference[:, None],
        statistic=lambda sample: np.mean(sample[:, 0]),
        count=int(args.bootstrap_count),
        rng=rng,
    )
    paired_crps = {
        "comparison": "C1 minus single-acquisition change-point",
        "subject_n": int(len(paired)),
        "mean_paired_crps_difference": float(
            np.mean(crps_difference)
        ),
        "subject_bootstrap_ci95": list(crps_ci),
        "C1_better_subject_n": int(
            np.sum(crps_difference < 0.0)
        ),
        "acquisition_better_subject_n": int(
            np.sum(crps_difference > 0.0)
        ),
        "tie_subject_n": int(np.sum(np.isclose(crps_difference, 0.0))),
        "interpretation": (
            "Negative favors C1. The comparison is paired by subject and "
            "uses frozen autonomous suffix predictions; it is descriptive "
            "rather than a new model-selection stage."
        ),
    }
    write_json(output / "paired_crps_comparison.json", paired_crps)

    association_rows: list[dict[str, Any]] = []
    predictor = paired[
        "boundary_rho_volatility_posterior_mean"
    ].to_numpy(dtype=float)
    for outcome in (
        "event_count",
        "max_event_duration",
        "trend_reversal_count",
        "max_adjacent_drop",
        "accuracy_slope",
    ):
        target = paired[outcome].to_numpy(dtype=float)
        values = np.column_stack([predictor, target])
        estimate = spearman(predictor, target)
        ci = subject_bootstrap(
            values,
            statistic=lambda sample: spearman(
                sample[:, 0], sample[:, 1]
            ),
            count=int(args.bootstrap_count),
            rng=rng,
        )
        association_rows.append(
            {
                "predictor": (
                    "prefix-conditioned posterior mean volatility"
                ),
                "outcome": outcome,
                "subject_n": int(len(paired)),
                "spearman_r": estimate,
                "bootstrap_ci95_lower": ci[0],
                "bootstrap_ci95_upper": ci[1],
                "confirmatory": False,
            }
        )
    pd.DataFrame(association_rows).to_csv(
        output / "volatility_associations.csv", index=False
    )

    morphology_rows: list[dict[str, Any]] = []
    for model, path in (
        ("单次掌握变点", acquisition),
        ("连续动态 readout C1", dynamic),
    ):
        metrics = pd.read_csv(path / "metric_summary.csv")
        for metric in MORPHOLOGY_METRICS:
            selected = metrics.loc[metrics["metric"].eq(metric)]
            observed = float(selected["observed"].mean())
            simulated = float(selected["sim_mean"].mean())
            morphology_rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "observed_mean": observed,
                    "simulated_mean": simulated,
                    "signed_gap_observed_minus_simulated": (
                        observed - simulated
                    ),
                    "absolute_gap": abs(observed - simulated),
                }
            )
    pd.DataFrame(morphology_rows).to_csv(
        output / "morphology_comparison.csv", index=False
    )

    data = pd.read_csv(args.data)
    condition = data.loc[data["condition"].eq(1)].copy()
    data_quality = {
        "condition1_row_n": int(len(condition)),
        "subject_n": int(condition["iSub"].nunique()),
        "duplicate_trial_key_n": int(
            condition.duplicated(list(KEY_COLUMNS)).sum()
        ),
        "missing_values": {
            column: int(condition[column].isna().sum())
            for column in (
                *KEY_COLUMNS,
                *FEATURE_COLUMNS,
                "category",
                "choice",
                "feedback",
            )
        },
        "feedback_mismatch_n": int(
            np.sum(
                condition["feedback"].to_numpy(dtype=int)
                != (
                    condition["choice"].to_numpy(dtype=int)
                    == condition["category"].to_numpy(dtype=int)
                )
            )
        ),
    }
    write_json(output / "data_quality.json", data_quality)
    write_json(
        output / "manifest.json",
        {
            "analysis": "condition1_dynamic_rho_consolidation",
            "bootstrap_count": int(args.bootstrap_count),
            "base_seed": int(args.base_seed),
            "model_comparison_is_paired_only_for": (
                "single-acquisition versus C1 on 24 reserved subjects"
            ),
            "volatility_associations_are_confirmatory": False,
            "sources": [
                str(path.relative_to(ROOT))
                for path in (
                    static,
                    acquisition,
                    dynamic,
                    dynamic_seed,
                    args.data,
                )
            ],
        },
    )
    print(
        json.dumps(
            {
                "model_rows": int(len(model_comparison)),
                "paired_crps": paired_crps,
                "data_quality": data_quality,
                "output_dir": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
