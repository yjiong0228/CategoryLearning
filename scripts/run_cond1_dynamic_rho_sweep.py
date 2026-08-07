#!/usr/bin/env python3
"""Development sweep for continuous deterministic/stochastic rho dynamics.

C0 uses a positive log-linear population trend in the readout concentration
from trial 1 to a shared absolute reference trial, with zero trialwise
innovation. C1 adds one persistent Gaussian deviation around the same trend.
Particle-level start, gain, and volatility random effects are drawn from
shared population distributions and conditioned only on each subject's
observed prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
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
from src.Bayesian_state.optimization.optimization_config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    load_yaml,
)


@dataclass(frozen=True, order=True)
class Candidate:
    family: str
    start: float
    end: float
    volatility: float
    persistence: float


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
        "--families",
        choices=("C0", "C1"),
        nargs="+",
        default=("C0", "C1"),
    )
    parser.add_argument(
        "--starts",
        type=float,
        nargs="+",
        default=(0.25, 0.50, 0.75, 1.00),
    )
    parser.add_argument(
        "--ends",
        type=float,
        nargs="+",
        default=(1.0, 2.0, 4.0),
    )
    parser.add_argument(
        "--volatilities",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.20),
    )
    parser.add_argument(
        "--persistences",
        type=float,
        nargs="+",
        default=(0.95,),
    )
    parser.add_argument("--start-log-sd", type=float, default=0.35)
    parser.add_argument("--gain-log-sd", type=float, default=0.35)
    parser.add_argument(
        "--volatility-log-sd",
        type=float,
        default=0.50,
    )
    parser.add_argument("--reference-trials", type=int, default=128)
    parser.add_argument("--particle-count", type=int, default=64)
    parser.add_argument("--rollout-count", type=int, default=128)
    parser.add_argument("--n-jobs", type=int, default=96)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--base-seed", type=int, default=20261001)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_active_set/"
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
            / "results/zhuran/cond1_active_set/"
            "dynamic_rho_sweep"
        ),
    )
    return parser.parse_args()


def _number_token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def candidate_name(candidate: Candidate) -> str:
    return "_".join(
        [
            candidate.family.lower(),
            f"s{_number_token(candidate.start)}",
            f"e{_number_token(candidate.end)}",
            f"v{_number_token(candidate.volatility)}",
            f"p{_number_token(candidate.persistence)}",
        ]
    )


def build_candidates(args: argparse.Namespace) -> list[Candidate]:
    starts = sorted({float(value) for value in args.starts})
    ends = sorted({float(value) for value in args.ends})
    volatilities = sorted(
        {float(value) for value in args.volatilities if float(value) > 0.0}
    )
    persistences = sorted({float(value) for value in args.persistences})
    candidates: list[Candidate] = []
    for family in sorted(set(args.families)):
        for start in starts:
            for end in ends:
                if end < start:
                    continue
                if family == "C0":
                    candidates.append(
                        Candidate(
                            family="C0",
                            start=start,
                            end=end,
                            volatility=0.0,
                            persistence=persistences[0],
                        )
                    )
                else:
                    for volatility in volatilities:
                        for persistence in persistences:
                            candidates.append(
                                Candidate(
                                    family="C1",
                                    start=start,
                                    end=end,
                                    volatility=volatility,
                                    persistence=persistence,
                                )
                            )
    return sorted(set(candidates))


def build_engine() -> tuple[dict[str, Any], dict[str, Path]]:
    model_path = ROOT / "configs/model_struct/pmh_model_cond1_active_set.yaml"
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
    candidate: Candidate,
) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=args.output_dir / candidate_name(candidate),
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
        acquisition_half_life=None,
        pre_acquisition_lapse=0.0,
        dynamic_rho_start=float(candidate.start),
        dynamic_rho_end=float(candidate.end),
        dynamic_rho_volatility=float(candidate.volatility),
        dynamic_rho_persistence=float(candidate.persistence),
        dynamic_rho_start_log_sd=float(args.start_log_sd),
        dynamic_rho_gain_log_sd=float(args.gain_log_sd),
        dynamic_rho_volatility_log_sd=float(
            args.volatility_log_sd
        ),
        dynamic_rho_reference_trials=int(args.reference_trials),
    )


def simulate_task(
    *,
    args: argparse.Namespace,
    candidate: Candidate,
    subject_frame: pd.DataFrame,
    engine_config: Mapping[str, Any],
    dataset_paths: Mapping[str, Path],
) -> tuple[Candidate, Path]:
    path = simulate_subject(
        args=candidate_args(args, candidate),
        subject_frame=subject_frame,
        engine_config=engine_config,
        dataset_paths=dataset_paths,
    )
    return candidate, path


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
    candidate: Candidate,
    paths: list[Path],
    output_dir: Path,
    cohort: str,
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
    width_gate = bool(
        median_width <= float(max_median_width_95)
    )
    crps_improvement_gate = bool(mean_crps < float(baseline_crps))
    sharpness_gate = bool(width_gate and crps_improvement_gate)
    metrics = {
        metric: metric_means(metric_summary, metric)
        for metric in (
            "accuracy",
            "accuracy_slope",
            "max_adjacent_rise",
            "max_adjacent_drop",
            "trend_reversal_count",
            "event_count",
            "max_event_duration",
            "late_accuracy",
        )
    }

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

    row: dict[str, Any] = {
        **asdict(candidate),
        "candidate": candidate_name(candidate),
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
        "boundary_rho_mean": float(
            subject_summary["boundary_rho_posterior_mean"].mean()
        ),
        "boundary_volatility_mean": float(
            subject_summary[
                "boundary_rho_volatility_posterior_mean"
            ].mean()
        ),
        "suffix_rho_mean": float(
            subject_summary["suffix_rho_mean"].mean()
        ),
        "suffix_rho_sd_mean": float(
            subject_summary[
                "suffix_rho_within_trajectory_sd_mean"
            ].mean()
        ),
        "coverage_gate": coverage_gate,
        "fdr_gate": fdr_gate,
        "width_gate": width_gate,
        "crps_improvement_gate": crps_improvement_gate,
        "sharpness_gate": sharpness_gate,
        "development_gate": bool(
            coverage_gate and fdr_gate and sharpness_gate
        ),
        "frozen_application_gate": bool(
            str(cohort) != "development"
            and coverage_gate
            and width_gate
        ),
        "decision_scope": (
            "development_selection"
            if str(cohort) == "development"
            else "frozen_application"
        ),
    }
    for metric, (observed, simulated) in metrics.items():
        row[f"observed_{metric}_mean"] = observed
        row[f"simulated_{metric}_mean"] = simulated
        row[f"absolute_{metric}_gap"] = float(abs(observed - simulated))
    write_json(output_dir / "candidate_decision.json", row)
    return row


def main() -> None:
    args = parse_args()
    if args.particle_count < 2 or args.rollout_count < 20:
        raise ValueError("Require at least 2 particles and 20 rollouts.")
    if args.n_jobs <= 0 or args.window < 4:
        raise ValueError("n-jobs must be positive and window at least 4.")
    if not 0.0 < args.max_median_width_95 <= 1.0:
        raise ValueError("max-median-width-95 must lie in (0, 1].")
    if min(
        args.start_log_sd,
        args.gain_log_sd,
        args.volatility_log_sd,
    ) < 0.0:
        raise ValueError("Random-effect scales must be non-negative.")
    if args.reference_trials < 2:
        raise ValueError("reference-trials must be at least 2.")
    if any(
        not np.isfinite(value) or float(value) <= 0.0
        for value in [*args.starts, *args.ends]
    ):
        raise ValueError("All rho endpoints must be finite and positive.")
    if any(
        not np.isfinite(value) or float(value) <= 0.0
        for value in args.volatilities
    ):
        raise ValueError("All C1 volatilities must be finite and positive.")
    if any(
        not np.isfinite(value) or not 0.0 <= float(value) < 1.0
        for value in args.persistences
    ):
        raise ValueError("Persistences must lie in [0, 1).")

    candidates = build_candidates(args)
    if not candidates:
        raise ValueError("The candidate grid is empty.")
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
        (candidate, subject_frame)
        for candidate in candidates
        for subject_frame in subject_frames
    ]
    results = Parallel(
        n_jobs=min(int(args.n_jobs), len(tasks)),
        verbose=10,
        backend="loky",
    )(
        delayed(simulate_task)(
            args=args,
            candidate=candidate,
            subject_frame=subject_frame,
            engine_config=engine,
            dataset_paths=dataset_paths,
        )
        for candidate, subject_frame in tasks
    )
    paths_by_candidate: dict[Candidate, list[Path]] = {
        candidate: [] for candidate in candidates
    }
    for candidate, path in results:
        paths_by_candidate[candidate].append(path)

    baseline_crps = baseline_development_crps(args.baseline_summary)
    comparison_rows = [
        summarize_candidate(
            candidate=candidate,
            paths=paths_by_candidate[candidate],
            output_dir=args.output_dir / candidate_name(candidate),
            cohort=str(args.cohort),
            window=int(args.window),
            baseline_crps=baseline_crps,
            max_median_width_95=float(args.max_median_width_95),
        )
        for candidate in candidates
    ]
    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["family", "start", "end", "volatility", "persistence"]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "sweep_summary.csv", index=False)

    selected: dict[str, dict[str, Any] | None] = {
        "C0": None,
        "C1": None,
    }
    frozen_application: list[dict[str, Any]] = []
    if args.cohort == "development":
        for family in ("C0", "C1"):
            family_rows = comparison.loc[
                comparison["family"].eq(family)
                & comparison["development_gate"]
            ]
            if family_rows.empty:
                continue
            selected_row = family_rows.sort_values(
                [
                    "mean_curve_crps",
                    "median_curve_interval_width_95",
                    "absolute_accuracy_slope_gap",
                    "absolute_event_count_gap",
                ]
            ).iloc[0]
            selected[family] = {
                key: (
                    value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                for key, value in selected_row.to_dict().items()
            }

        selected_c0 = selected["C0"]
        selected_c1 = selected["C1"]
        if selected_c1 is None:
            incremental_status = "C1_development_gate_failed"
        elif selected_c0 is None:
            incremental_status = "C1_passed_C0_failed"
        else:
            c1_crps_better = (
                selected_c1["mean_curve_crps"]
                < selected_c0["mean_curve_crps"]
            )
            c1_width_guard = (
                selected_c1["median_curve_interval_width_95"]
                <= selected_c0["median_curve_interval_width_95"] + 0.05
            )
            incremental_status = (
                "C1_increment_supported_on_development"
                if c1_crps_better and c1_width_guard
                else "C1_increment_not_supported_on_development"
            )
    else:
        frozen_application = [
            {
                key: (
                    value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                for key, value in row.items()
            }
            for row in comparison.to_dict(orient="records")
        ]
        passing = [
            row
            for row in frozen_application
            if bool(row["frozen_application_gate"])
        ]
        if not passing:
            incremental_status = "frozen_application_failed"
        else:
            fdr_outliers = int(
                sum(int(row["fdr_failure_n"]) for row in passing)
            )
            incremental_status = (
                "frozen_application_adequate"
                if fdr_outliers == 0
                else (
                    "frozen_application_adequate_with_"
                    f"{fdr_outliers}_individual_fdr_outlier"
                )
            )
    selection = {
        "analysis": "continuous_dynamic_rho_C0_C1",
        "cohort": str(args.cohort),
        "subject_ids": sorted(
            int(value) for value in data["iSub"].unique()
        ),
        "particle_count": int(args.particle_count),
        "rollout_count": int(args.rollout_count),
        "n_jobs": int(min(args.n_jobs, len(tasks))),
        "candidate_n": int(len(candidates)),
        "population_random_effect_scales": {
            "start_log_sd": float(args.start_log_sd),
            "gain_log_sd": float(args.gain_log_sd),
            "volatility_log_sd": float(args.volatility_log_sd),
        },
        "trend_reference_trials": int(args.reference_trials),
        "selected": selected,
        "frozen_application": frozen_application,
        "incremental_status": incremental_status,
        "development_gate": {
            "cohort_lower_tail_calibration_p_min": 0.05,
            "subject_level_bh_fdr_failure_n_max": 0,
            "median_rolling_interval_width_95_max": float(
                args.max_median_width_95
            ),
            "mean_curve_crps_must_improve_over_static_fullset": True,
            "static_fullset_development_mean_curve_crps": baseline_crps,
        },
        "frozen_application_gate": {
            "cohort_lower_tail_calibration_p_min": 0.05,
            "median_rolling_interval_width_95_max": float(
                args.max_median_width_95
            ),
            "individual_bh_fdr_failures": (
                "reported as heterogeneity diagnostics; they do not by "
                "themselves reject cohort-level generative adequacy"
            ),
        },
        "interpretation": (
            "Passing supports a calibrated generator of trajectory shapes. "
            "It does not uniquely identify the trialwise rho path or assign "
            "named cognitive states to observed rises and drops."
        ),
    }
    write_json(args.output_dir / "selection.json", selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
