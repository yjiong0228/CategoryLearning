#!/usr/bin/env python3
"""Development/confirmation sweep for one irreversible acquisition boundary.

The candidate adds one mechanism to the static condition-1 full-set model:
each latent trajectory begins with a fixed novice lapse mixed into the
ordinary readout and crosses at most once to lapse-free readout. Acquisition
time is geometrically distributed. A shared half-life is selected on the
eight development subjects; reserved subjects are evaluated only after that
value is frozen.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cond1_b0_trajectory_ppc import (  # noqa: E402
    DEVELOPMENT_SUBJECTS,
    FEATURE_COLUMNS,
    KEY_COLUMNS,
    METRIC_SPECS,
    benjamini_hochberg,
    cohort_calibration,
    evaluate_subject,
    load_subject_cache,
    simulate_subject,
    write_json,
)
from src.Bayesian_state.run_simulation import (  # noqa: E402
    apply_fixed_hyperparams_to_engine_config,
)
from src.Bayesian_state.utils.datasets import (  # noqa: E402
    resolve_dataset_paths,
)
from src.Bayesian_state.utils.optimization_config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    load_yaml,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/processed/Task2_processed.csv",
    )
    parser.add_argument(
        "--cohort",
        choices=("development", "reserved", "all"),
        default="development",
    )
    parser.add_argument(
        "--half-lives",
        type=float,
        nargs="+",
        default=(4, 8, 16, 32, 64, 128, 256, 512),
    )
    parser.add_argument("--particle-count", type=int, default=128)
    parser.add_argument("--rollout-count", type=int, default=256)
    parser.add_argument("--n-jobs", type=int, default=64)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=20260801)
    parser.add_argument(
        "--pre-acquisition-lapse",
        type=float,
        default=1.0,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_newplan/"
            "fullset_trajectory_ppc_early_anchor/subject_summary.csv"
        ),
    )
    parser.add_argument(
        "--max-median-width-95",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_newplan/"
            "acquisition_changepoint_sweep"
        ),
    )
    return parser.parse_args()


def candidate_name(half_life: float) -> str:
    value = float(half_life)
    if value.is_integer():
        return f"half_life_{int(value):04d}"
    return f"half_life_{value:g}".replace(".", "p")


def build_engine() -> tuple[dict[str, Any], dict[str, Path]]:
    model_path = ROOT / "configs/model_struct/pmh_model_cond1_newplan.yaml"
    simulation_path = (
        ROOT / "configs/simulation_cfg/pmh_cond1_simulation_v14.yaml"
    )
    base_engine = load_yaml(model_path)
    simulation_config = load_yaml(simulation_path)
    dataset_paths = resolve_dataset_paths(
        simulation_config,
        simulation_path.parent,
        DEFAULT_DATA_PATH,
    )
    parameters = {
        "engine.modules.hypo_transitions_mod.kwargs.theta": 0.0,
        "engine.modules.hypo_transitions_mod.kwargs.capacity": 38,
        "engine.modules.memory_mod.kwargs.gamma": 0.55,
        "engine.modules.memory_mod.kwargs.w0": 0.10,
        "engine.modules.beta_mod.kwargs.beta_init": 5.0,
        "engine.modules.beta_mod.kwargs.correct_additive": 0.5,
        "engine.modules.beta_mod.kwargs.decrease_rate": 0.15,
        "engine.choice_readout.kwargs": {
            "method": "sharpened_expectation",
            "power": 2.0,
        },
        "engine.output_noise.kwargs.base_lapse": 0.0,
    }
    return (
        apply_fixed_hyperparams_to_engine_config(
            deepcopy(base_engine),
            parameters,
        ),
        dataset_paths,
    )


def candidate_args(
    args: argparse.Namespace,
    half_life: float,
) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=args.output_dir / candidate_name(half_life),
        particle_count=int(args.particle_count),
        rollout_count=int(args.rollout_count),
        force=bool(args.force),
        split_mode="early_anchor",
        window=int(args.window),
        base_seed=int(args.base_seed),
        beta_correct_additive=0.5,
        lapse_start=0.0,
        learning_update_probability=1.0,
        beta_additive_grid=None,
        lapse_start_grid=None,
        learning_update_grid=None,
        selection_particle_count=8,
        lapse_half_life=128.0,
        rho=2.0,
        resample_threshold=0.5,
        acquisition_half_life=float(half_life),
        pre_acquisition_lapse=float(args.pre_acquisition_lapse),
    )


def simulate_task(
    *,
    args: argparse.Namespace,
    half_life: float,
    subject_frame: pd.DataFrame,
    engine_config: Mapping[str, Any],
    dataset_paths: Mapping[str, Path],
) -> tuple[float, Path]:
    path = simulate_subject(
        args=candidate_args(args, half_life),
        subject_frame=subject_frame,
        engine_config=engine_config,
        dataset_paths=dataset_paths,
    )
    return float(half_life), path


def baseline_development_crps(path: Path) -> float:
    baseline = pd.read_csv(path)
    if "cohort" in baseline:
        baseline = baseline.loc[baseline["cohort"].eq("development")]
    if baseline.empty:
        raise ValueError("Baseline summary has no development rows.")
    return float(baseline["curve_crps"].mean())


def metric_means(
    metric_summary: pd.DataFrame,
    metric: str,
) -> tuple[float, float]:
    selected = metric_summary.loc[metric_summary["metric"].eq(metric)]
    if selected.empty:
        return np.nan, np.nan
    return (
        float(selected["observed"].mean()),
        float(selected["sim_mean"].mean()),
    )


def summarize_candidate(
    *,
    half_life: float,
    paths: list[Path],
    output_dir: Path,
    window: int,
    baseline_crps: float,
    max_median_width_95: float,
) -> dict[str, Any]:
    subject_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    simulated_pass: dict[int, np.ndarray] = {}
    for path in paths:
        cache = load_subject_cache(path)
        subject_row, local_metrics, local_curve, local_sim_pass = (
            evaluate_subject(cache, window=int(window))
        )
        subject_rows.append(subject_row)
        metric_rows.extend(local_metrics)
        curve_rows.extend(local_curve)
        simulated_pass[int(subject_row["iSub"])] = local_sim_pass

    subject_summary = pd.DataFrame(subject_rows).sort_values("iSub")
    subject_summary["combined_calibration_fdr_q"] = benjamini_hochberg(
        subject_summary["combined_calibration_p"].to_numpy(dtype=float)
    )
    metric_summary = pd.DataFrame(metric_rows)
    curve_summary = pd.DataFrame(curve_rows)
    cohort_summary = cohort_calibration(subject_summary, simulated_pass)
    all_row = cohort_summary.loc[
        cohort_summary["cohort"].eq("all_subjects")
    ].iloc[0]
    fdr_failure_n = int(
        np.sum(subject_summary["combined_calibration_fdr_q"] <= 0.05)
    )
    mean_crps = float(subject_summary["curve_crps"].mean())
    median_width = float(
        subject_summary["curve_pointwise_interval_width_95"].median()
    )
    coverage_gate = bool(all_row["lower_tail_calibration_p"] >= 0.05)
    fdr_gate = fdr_failure_n == 0
    sharpness_gate = bool(
        median_width <= float(max_median_width_95)
        and mean_crps < float(baseline_crps)
    )
    observed_accuracy, simulated_accuracy = metric_means(
        metric_summary,
        "accuracy",
    )
    observed_slope, simulated_slope = metric_means(
        metric_summary,
        "accuracy_slope",
    )
    observed_events, simulated_events = metric_means(
        metric_summary,
        "event_count",
    )
    observed_late, simulated_late = metric_means(
        metric_summary,
        "late_accuracy",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    subject_summary.to_csv(output_dir / "subject_summary.csv", index=False)
    metric_summary.to_csv(output_dir / "metric_summary.csv", index=False)
    curve_summary.to_csv(
        output_dir / "rolling_curve_summary.csv",
        index=False,
    )
    cohort_summary.to_csv(
        output_dir / "cohort_calibration.csv",
        index=False,
    )

    row = {
        "acquisition_half_life": float(half_life),
        "acquisition_hazard": float(
            1.0 - np.power(0.5, 1.0 / float(half_life))
        ),
        "subject_n": int(len(subject_summary)),
        "combined_pass_n": int(subject_summary["combined_pass_95"].sum()),
        "combined_pass_fraction": float(
            subject_summary["combined_pass_95"].mean()
        ),
        "self_expected_pass_mean": float(
            all_row["b0_self_expected_pass_mean"]
        ),
        "self_expected_pass_q025": float(
            all_row["b0_self_expected_pass_q025"]
        ),
        "self_expected_pass_q975": float(
            all_row["b0_self_expected_pass_q975"]
        ),
        "lower_tail_calibration_p": float(
            all_row["lower_tail_calibration_p"]
        ),
        "fdr_failure_n": fdr_failure_n,
        "mean_curve_crps": mean_crps,
        "baseline_mean_curve_crps": float(baseline_crps),
        "median_curve_interval_width_95": median_width,
        "max_allowed_median_width_95": float(max_median_width_95),
        "observed_accuracy_mean": observed_accuracy,
        "simulated_accuracy_mean": simulated_accuracy,
        "observed_accuracy_slope_mean": observed_slope,
        "simulated_accuracy_slope_mean": simulated_slope,
        "observed_event_count_mean": observed_events,
        "simulated_event_count_mean": simulated_events,
        "observed_late_accuracy_mean": observed_late,
        "simulated_late_accuracy_mean": simulated_late,
        "boundary_acquired_probability_mean": float(
            subject_summary["boundary_acquired_probability"].mean()
        ),
        "suffix_acquired_fraction_mean": float(
            subject_summary["suffix_acquired_fraction_mean"].mean()
        ),
        "coverage_gate": coverage_gate,
        "fdr_gate": fdr_gate,
        "sharpness_gate": sharpness_gate,
        "development_gate": bool(
            coverage_gate and fdr_gate and sharpness_gate
        ),
    }
    write_json(output_dir / "candidate_decision.json", row)
    return row


def main() -> None:
    args = parse_args()
    if args.particle_count < 2 or args.rollout_count < 20:
        raise ValueError("Require at least 2 particles and 20 rollouts.")
    if args.n_jobs <= 0 or args.window < 4:
        raise ValueError("n_jobs must be positive and window at least 4.")
    if any(
        not np.isfinite(value) or float(value) <= 0.0
        for value in args.half_lives
    ):
        raise ValueError("All acquisition half-lives must be positive.")
    half_lives = sorted({float(value) for value in args.half_lives})
    if not 0.0 < args.max_median_width_95 <= 1.0:
        raise ValueError("max-median-width-95 must lie in (0, 1].")
    if not 0.0 <= args.pre_acquisition_lapse <= 1.0:
        raise ValueError("pre-acquisition-lapse must lie in [0, 1].")

    data = pd.read_csv(args.data)
    data = (
        data.loc[data["condition"].eq(1)]
        .sort_values(list(KEY_COLUMNS))
        .reset_index(drop=True)
    )
    development = set(DEVELOPMENT_SUBJECTS)
    if args.cohort == "development":
        data = data.loc[data["iSub"].isin(development)].copy()
    elif args.cohort == "reserved":
        data = data.loc[~data["iSub"].isin(development)].copy()
    if data.empty:
        raise ValueError("Selected cohort contains no condition-1 rows.")
    required = {
        *KEY_COLUMNS,
        *FEATURE_COLUMNS,
        "category",
        "choice",
        "feedback",
    }
    missing_columns = sorted(required - set(data.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    engine, dataset_paths = build_engine()
    subject_frames = [
        group.copy()
        for _, group in data.groupby("iSub", sort=True)
    ]
    tasks = [
        (half_life, subject_frame)
        for half_life in half_lives
        for subject_frame in subject_frames
    ]
    results = Parallel(
        n_jobs=min(int(args.n_jobs), len(tasks)),
        verbose=10,
        backend="loky",
    )(
        delayed(simulate_task)(
            args=args,
            half_life=half_life,
            subject_frame=subject_frame,
            engine_config=engine,
            dataset_paths=dataset_paths,
        )
        for half_life, subject_frame in tasks
    )
    paths_by_candidate: dict[float, list[Path]] = {
        half_life: [] for half_life in half_lives
    }
    for half_life, path in results:
        paths_by_candidate[half_life].append(path)

    baseline_crps = baseline_development_crps(args.baseline_summary)
    comparison_rows = [
        summarize_candidate(
            half_life=half_life,
            paths=paths_by_candidate[half_life],
            output_dir=args.output_dir / candidate_name(half_life),
            window=int(args.window),
            baseline_crps=baseline_crps,
            max_median_width_95=float(args.max_median_width_95),
        )
        for half_life in half_lives
    ]
    comparison = pd.DataFrame(comparison_rows).sort_values(
        "acquisition_half_life"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "sweep_summary.csv", index=False)

    eligible = comparison.loc[comparison["development_gate"]].copy()
    if eligible.empty:
        selected = None
        status = "development_gate_failed"
        next_action = "do_not_open_reserved_cohort"
    else:
        selected_row = eligible.sort_values(
            [
                "mean_curve_crps",
                "median_curve_interval_width_95",
                "acquisition_half_life",
            ]
        ).iloc[0]
        selected = float(selected_row["acquisition_half_life"])
        status = "development_gate_passed"
        next_action = "freeze_selected_half_life_and_open_reserved_cohort"
    selection = {
        "analysis": "single_irreversible_acquisition_changepoint",
        "cohort": str(args.cohort),
        "subject_ids": sorted(
            int(value) for value in data["iSub"].unique()
        ),
        "particle_count": int(args.particle_count),
        "rollout_count": int(args.rollout_count),
        "n_jobs": int(min(args.n_jobs, len(tasks))),
        "half_lives": half_lives,
        "pre_acquisition_lapse": float(args.pre_acquisition_lapse),
        "selection_status": status,
        "selected_half_life": selected,
        "next_action": next_action,
        "development_gate": {
            "cohort_lower_tail_calibration_p_min": 0.05,
            "subject_level_bh_fdr_failure_n_max": 0,
            "median_rolling_interval_width_95_max": float(
                args.max_median_width_95
            ),
            "mean_curve_crps_must_improve_over_static_fullset": True,
            "static_fullset_development_mean_curve_crps": baseline_crps,
        },
        "interpretation": (
            "Passing means the observed trajectories are non-extreme draws "
            "from this counterfactual generator with useful sharpness. It "
            "does not identify an actual subject-specific change-point."
        ),
    }
    write_json(args.output_dir / "selection.json", selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
