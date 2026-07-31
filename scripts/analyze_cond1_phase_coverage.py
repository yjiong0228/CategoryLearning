#!/usr/bin/env python3
"""Phase-level posterior-predictive coverage for frozen condition-1 models.

The analysis reuses existing autonomous suffix rollouts.  It does not refit,
retune, or simulate either model.  Observable phase descriptors are evaluated
jointly at 8-, 12-, and 16-trial chunk widths, with 12 trials declared primary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cond1_b0_trajectory_ppc import (  # noqa: E402
    benjamini_hochberg,
    empirical_crps,
    load_subject_cache,
    robust_scales,
    write_json,
)


PHASE_METRICS = (
    "chance_chunk_fraction",
    "longest_chance_run_fraction",
    "stable_high_chunk_fraction",
    "first_high_latency_fraction",
    "abrupt_rise_count",
    "abrupt_drop_count",
    "recovery_count",
    "gradual_change_count",
    "direction_reversal_count",
    "phase_diversity",
)
FRACTION_METRICS = {
    "chance_chunk_fraction",
    "longest_chance_run_fraction",
    "stable_high_chunk_fraction",
    "first_high_latency_fraction",
}
PHASE_LABELS = {
    "chance_chunk_fraction": "接近随机水平窗口占比",
    "longest_chance_run_fraction": "最长混乱段占比",
    "stable_high_chunk_fraction": "稳定高水平窗口占比",
    "first_high_latency_fraction": "首次进入高水平的相对时间",
    "abrupt_rise_count": "陡升次数",
    "abrupt_drop_count": "陡降次数",
    "recovery_count": "陡降后恢复次数",
    "gradual_change_count": "连续渐变段数",
    "direction_reversal_count": "变化方向反转数",
    "phase_diversity": "经历的阶段种类数",
}
MODEL_LABELS = {
    "C1": "连续动态 readout C1",
    "acquisition": "单次掌握变点",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--c1-dir",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_newplan/"
            "dynamic_rho_reserved_c1_p256_r1024/"
            "c1_s0p5_e0p5_v0p2_p0p95"
        ),
    )
    parser.add_argument(
        "--acquisition-dir",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_newplan/"
            "acquisition_changepoint_reserved_h128_p256_r1024"
        ),
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=(8, 12, 16),
    )
    parser.add_argument("--primary-window", type=int, default=12)
    parser.add_argument("--n-jobs", type=int, default=24)
    parser.add_argument("--bootstrap-count", type=int, default=10000)
    parser.add_argument("--base-seed", type=int, default=20261201)
    parser.add_argument(
        "--shared-residual-subject-min",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_newplan/"
            "phase_coverage_frozen_models"
        ),
    )
    return parser.parse_args()


def longest_run(values: np.ndarray) -> int:
    best = 0
    current = 0
    for value in np.asarray(values, dtype=bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def compressed_states(states: Sequence[str]) -> list[str]:
    output: list[str] = []
    for state in states:
        if not output or output[-1] != state:
            output.append(str(state))
    return output


def gradual_change_count(chunk_means: np.ndarray, window: int) -> int:
    values = np.asarray(chunk_means, dtype=float)
    if values.size < 3:
        return 0
    changes = np.diff(values)
    minimum = 1.0 / int(window) - 1e-12
    eligible = (
        (np.abs(changes) >= minimum)
        & (np.abs(changes) < 0.25 - 1e-12)
    )
    signs = np.sign(changes)
    count = 0
    start = 0
    while start < len(changes):
        if not eligible[start]:
            start += 1
            continue
        sign = signs[start]
        end = start + 1
        while (
            end < len(changes)
            and eligible[end]
            and signs[end] == sign
        ):
            end += 1
        run = changes[start:end]
        if len(run) >= 2 and abs(float(np.sum(run))) >= 0.25 - 1e-12:
            count += 1
        start = end
    return int(count)


def phase_metrics(
    feedback: np.ndarray,
    block_ids: np.ndarray,
    *,
    window: int,
) -> dict[str, float]:
    outcomes = np.asarray(feedback, dtype=float).reshape(-1)
    blocks = np.asarray(block_ids).reshape(-1)
    if outcomes.size != blocks.size:
        raise ValueError("block_ids must align with feedback.")
    if int(window) < 4:
        raise ValueError("window must be at least 4.")

    all_means: list[float] = []
    ordered_states: list[str] = []
    abrupt_rise = 0
    abrupt_drop = 0
    recovery = 0
    gradual = 0
    reversals = 0
    longest_chance = 0
    for block in dict.fromkeys(blocks.tolist()):
        local = outcomes[blocks == block]
        starts = range(0, len(local) - int(window) + 1, int(window))
        means = np.asarray(
            [
                np.mean(local[start : start + int(window)])
                for start in starts
            ],
            dtype=float,
        )
        if means.size == 0:
            continue
        all_means.extend(means.tolist())
        states = np.where(
            means <= 0.5 + 1e-12,
            "L",
            np.where(means >= 2.0 / 3.0 - 1e-12, "H", "M"),
        ).tolist()
        ordered_states.extend(states)
        chance = (means >= 1.0 / 3.0 - 1e-12) & (
            means <= 2.0 / 3.0 + 1e-12
        )
        longest_chance = max(longest_chance, longest_run(chance))

        changes = np.diff(means)
        abrupt_rise += int(np.sum(changes >= 0.25 - 1e-12))
        abrupt_drop += int(np.sum(changes <= -0.25 + 1e-12))
        nonzero = np.sign(
            changes[np.abs(changes) >= 1.0 / int(window) - 1e-12]
        )
        if nonzero.size >= 2:
            reversals += int(np.sum(nonzero[1:] != nonzero[:-1]))
        gradual += gradual_change_count(means, int(window))
        compressed = compressed_states(states)
        recovery += sum(
            compressed[index] == "L"
            and "H" in compressed[:index]
            and "H" in compressed[index + 1 :]
            for index in range(1, len(compressed) - 1)
        )

    means = np.asarray(all_means, dtype=float)
    if means.size == 0:
        raise ValueError("No complete phase chunks for this trajectory.")
    states = np.asarray(ordered_states)
    chance = (means >= 1.0 / 3.0 - 1e-12) & (
        means <= 2.0 / 3.0 + 1e-12
    )
    high_indices = np.flatnonzero(states == "H")
    first_high = (
        float(high_indices[0] / max(1, means.size - 1))
        if high_indices.size
        else 1.0
    )
    return {
        "chance_chunk_fraction": float(np.mean(chance)),
        "longest_chance_run_fraction": float(
            longest_chance / means.size
        ),
        "stable_high_chunk_fraction": float(
            np.mean(means >= 5.0 / 6.0 - 1e-12)
        ),
        "first_high_latency_fraction": first_high,
        "abrupt_rise_count": float(abrupt_rise),
        "abrupt_drop_count": float(abrupt_drop),
        "recovery_count": float(recovery),
        "gradual_change_count": float(gradual),
        "direction_reversal_count": float(reversals),
        "phase_diversity": float(len(set(ordered_states))),
    }


def phase_vector(
    feedback: np.ndarray,
    block_ids: np.ndarray,
    windows: Sequence[int],
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    keys = [
        (int(window), metric)
        for window in windows
        for metric in PHASE_METRICS
    ]
    by_window = {
        int(window): phase_metrics(
            feedback,
            block_ids,
            window=int(window),
        )
        for window in windows
    }
    return (
        np.asarray(
            [
                by_window[window][metric]
                for window, metric in keys
            ],
            dtype=float,
        ),
        keys,
    )


def phase_matrix(
    feedback: np.ndarray,
    block_ids: np.ndarray,
    windows: Sequence[int],
) -> tuple[np.ndarray, list[tuple[int, str]]]:
    rows: list[np.ndarray] = []
    keys: list[tuple[int, str]] | None = None
    for trajectory in np.asarray(feedback, dtype=float):
        vector, local_keys = phase_vector(
            trajectory,
            block_ids,
            windows,
        )
        if keys is None:
            keys = local_keys
        elif keys != local_keys:
            raise RuntimeError("Phase vector keys changed across trajectories.")
        rows.append(vector)
    if keys is None:
        raise ValueError("No simulated trajectories.")
    return np.asarray(rows, dtype=float), keys


def phase_resolutions(
    block_ids: np.ndarray,
    windows: Sequence[int],
) -> np.ndarray:
    blocks = np.asarray(block_ids).reshape(-1)
    resolutions: list[float] = []
    for window in windows:
        chunk_n = sum(
            int(np.sum(blocks == block)) // int(window)
            for block in dict.fromkeys(blocks.tolist())
        )
        for metric in PHASE_METRICS:
            resolutions.append(
                1.0 / max(1, chunk_n)
                if metric in FRACTION_METRICS
                else 1.0
            )
    return np.asarray(resolutions, dtype=float)


def cache_paths(directory: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in directory.glob(
        "cache/subject_*/particles_*/rollouts_*.npz"
    ):
        cache = load_subject_cache(path)
        subject = int(cache["subject_id"])
        if subject in paths:
            raise ValueError(
                f"Multiple caches found for subject {subject} in {directory}"
            )
        paths[subject] = path
    if not paths:
        raise FileNotFoundError(f"No caches found under {directory}")
    return paths


def evaluate_distribution(
    actual: np.ndarray,
    simulations: np.ndarray,
    resolutions: np.ndarray,
) -> dict[str, Any]:
    median, scale = robust_scales(simulations, resolutions)
    actual_z = (actual - median) / scale
    simulation_z = (simulations - median[None, :]) / scale[None, :]
    discrepancy = float(np.max(np.abs(actual_z)))
    simulation_discrepancy = np.max(np.abs(simulation_z), axis=1)
    threshold = float(np.quantile(simulation_discrepancy, 0.95))
    p_value = float(
        (
            1.0
            + np.sum(
                simulation_discrepancy >= discrepancy - 1e-12
            )
        )
        / (1.0 + simulations.shape[0])
    )
    simulation_pass = (
        simulation_discrepancy <= threshold + 1e-12
    )
    return {
        "median": median,
        "scale": scale,
        "actual_z": actual_z,
        "discrepancy": discrepancy,
        "threshold": threshold,
        "p_value": p_value,
        "pass_95": bool(discrepancy <= threshold + 1e-12),
        "simulation_pass": simulation_pass,
    }


def metric_rows(
    *,
    model: str,
    subject: int,
    keys: Sequence[tuple[int, str]],
    actual: np.ndarray,
    simulations: np.ndarray,
    evaluation: Mapping[str, Any],
    pooled_scale: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (window, metric) in enumerate(keys):
        samples = simulations[:, index]
        observed = float(actual[index])
        q025, q50, q975 = np.quantile(
            samples,
            [0.025, 0.5, 0.975],
        )
        if observed < q025 - 1e-12:
            direction = "below"
        elif observed > q975 + 1e-12:
            direction = "above"
        else:
            direction = "inside"
        rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "iSub": int(subject),
                "window": int(window),
                "metric": metric,
                "metric_label": PHASE_LABELS[metric],
                "observed": observed,
                "sim_mean": float(np.mean(samples)),
                "sim_median": float(q50),
                "sim_q025": float(q025),
                "sim_q975": float(q975),
                "inside_marginal_95": bool(
                    q025 <= observed <= q975
                ),
                "failure_direction": direction,
                "robust_z": float(evaluation["actual_z"][index]),
                "metric_crps": empirical_crps(
                    np.asarray([observed]),
                    samples[:, None],
                ),
                "scaled_metric_crps": float(
                    empirical_crps(
                        np.asarray([observed]),
                        samples[:, None],
                    )
                    / pooled_scale[index]
                ),
            }
        )
    return rows


def analyze_subject(
    *,
    subject: int,
    c1_path: Path,
    acquisition_path: Path,
    windows: Sequence[int],
    primary_window: int,
) -> dict[str, Any]:
    c1 = load_subject_cache(c1_path)
    acquisition = load_subject_cache(acquisition_path)
    for key in (
        "observed_test_feedback",
        "observed_test_choices",
        "test_iSession",
        "test_iBlock",
        "test_iTrial",
    ):
        if not np.array_equal(c1[key], acquisition[key]):
            raise ValueError(
                f"Subject {subject} differs across models for {key}."
            )
    block_ids = np.asarray(
        [
            f"{int(session)}:{int(block)}"
            for session, block in zip(
                c1["test_iSession"],
                c1["test_iBlock"],
            )
        ]
    )
    actual, keys = phase_vector(
        np.asarray(c1["observed_test_feedback"], dtype=float),
        block_ids,
        windows,
    )
    c1_matrix, c1_keys = phase_matrix(
        np.asarray(c1["feedback"], dtype=float),
        block_ids,
        windows,
    )
    acquisition_matrix, acquisition_keys = phase_matrix(
        np.asarray(acquisition["feedback"], dtype=float),
        block_ids,
        windows,
    )
    if keys != c1_keys or keys != acquisition_keys:
        raise RuntimeError("Phase vector keys do not align.")
    resolutions = phase_resolutions(block_ids, windows)
    _, pooled_scale = robust_scales(
        np.vstack([c1_matrix, acquisition_matrix]),
        resolutions,
    )
    c1_evaluation = evaluate_distribution(
        actual,
        c1_matrix,
        resolutions,
    )
    acquisition_evaluation = evaluate_distribution(
        actual,
        acquisition_matrix,
        resolutions,
    )
    c1_rows = metric_rows(
        model="C1",
        subject=subject,
        keys=keys,
        actual=actual,
        simulations=c1_matrix,
        evaluation=c1_evaluation,
        pooled_scale=pooled_scale,
    )
    acquisition_rows = metric_rows(
        model="acquisition",
        subject=subject,
        keys=keys,
        actual=actual,
        simulations=acquisition_matrix,
        evaluation=acquisition_evaluation,
        pooled_scale=pooled_scale,
    )
    primary = {
        metric: float(
            actual[
                keys.index((int(primary_window), metric))
            ]
        )
        for metric in PHASE_METRICS
    }
    signatures: list[str] = []
    if primary["chance_chunk_fraction"] >= 0.25:
        signatures.append("混乱")
    if primary["abrupt_rise_count"] >= 1:
        signatures.append("陡升")
    if primary["abrupt_drop_count"] >= 1:
        signatures.append("陡降")
    if primary["gradual_change_count"] >= 1:
        signatures.append("渐变")
    if primary["recovery_count"] >= 1:
        signatures.append("恢复")
    if primary["stable_high_chunk_fraction"] >= 0.25:
        signatures.append("稳定高水平")
    if not signatures:
        signatures.append("无预设显著阶段")

    return {
        "subject": int(subject),
        "test_n": int(c1["test_n"]),
        "phase_signature": "、".join(signatures),
        "primary_metrics": primary,
        "C1": {
            "phase_discrepancy": c1_evaluation["discrepancy"],
            "phase_threshold_95": c1_evaluation["threshold"],
            "phase_calibration_p": c1_evaluation["p_value"],
            "phase_pass_95": c1_evaluation["pass_95"],
            "phase_scaled_crps": float(
                np.mean(
                    [
                        row["scaled_metric_crps"]
                        for row in c1_rows
                    ]
                )
            ),
            "simulation_pass": c1_evaluation["simulation_pass"],
        },
        "acquisition": {
            "phase_discrepancy": acquisition_evaluation[
                "discrepancy"
            ],
            "phase_threshold_95": acquisition_evaluation["threshold"],
            "phase_calibration_p": acquisition_evaluation["p_value"],
            "phase_pass_95": acquisition_evaluation["pass_95"],
            "phase_scaled_crps": float(
                np.mean(
                    [
                        row["scaled_metric_crps"]
                        for row in acquisition_rows
                    ]
                )
            ),
            "simulation_pass": acquisition_evaluation[
                "simulation_pass"
            ],
        },
        "metric_rows": c1_rows + acquisition_rows,
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    count: int,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    estimates = np.asarray(
        [
            np.mean(
                array[
                    rng.integers(0, len(array), size=len(array))
                ]
            )
            for _ in range(int(count))
        ],
        dtype=float,
    )
    return tuple(
        float(value)
        for value in np.quantile(estimates, [0.025, 0.975])
    )


def cohort_calibration(
    subject_summary: pd.DataFrame,
    simulation_pass: Mapping[int, np.ndarray],
    *,
    model: str,
) -> dict[str, Any]:
    ordered = subject_summary.sort_values("iSub")
    matrix = np.vstack(
        [simulation_pass[int(subject)] for subject in ordered["iSub"]]
    )
    null_counts = np.sum(matrix, axis=0)
    observed = int(ordered[f"{model}_phase_pass_95"].sum())
    return {
        "model": model,
        "model_label": MODEL_LABELS[model],
        "subject_n": int(len(ordered)),
        "observed_pass_n": observed,
        "observed_pass_fraction": float(observed / len(ordered)),
        "self_expected_pass_mean": float(np.mean(null_counts)),
        "self_expected_pass_q025": float(
            np.quantile(null_counts, 0.025)
        ),
        "self_expected_pass_q975": float(
            np.quantile(null_counts, 0.975)
        ),
        "lower_tail_calibration_p": float(
            (1.0 + np.sum(null_counts <= observed))
            / (1.0 + len(null_counts))
        ),
    }


def shared_residuals(
    metrics: pd.DataFrame,
    *,
    primary_window: int,
    sensitivity_windows: Sequence[int],
    minimum_subjects: int,
) -> pd.DataFrame:
    pivot = metrics.pivot(
        index=["iSub", "window", "metric"],
        columns="model",
        values=["inside_marginal_95", "failure_direction"],
    )
    rows: list[dict[str, Any]] = []
    subjects = sorted(metrics["iSub"].unique())
    for subject in subjects:
        for metric in PHASE_METRICS:
            key = (subject, int(primary_window), metric)
            if key not in pivot.index:
                continue
            primary = pivot.loc[key]
            primary_failure = bool(
                not primary[("inside_marginal_95", "C1")]
                and not primary[
                    ("inside_marginal_95", "acquisition")
                ]
                and primary[("failure_direction", "C1")]
                == primary[("failure_direction", "acquisition")]
                and primary[("failure_direction", "C1")]
                != "inside"
            )
            if not primary_failure:
                continue
            direction = str(primary[("failure_direction", "C1")])
            supported_windows: list[int] = []
            for window in sensitivity_windows:
                local_key = (subject, int(window), metric)
                if local_key not in pivot.index:
                    continue
                local = pivot.loc[local_key]
                if (
                    not local[("inside_marginal_95", "C1")]
                    and not local[
                        ("inside_marginal_95", "acquisition")
                    ]
                    and local[("failure_direction", "C1")] == direction
                    and local[
                        ("failure_direction", "acquisition")
                    ]
                    == direction
                ):
                    supported_windows.append(int(window))
            rows.append(
                {
                    "iSub": int(subject),
                    "metric": metric,
                    "metric_label": PHASE_LABELS[metric],
                    "direction": direction,
                    "primary_window": int(primary_window),
                    "sensitivity_support_windows": ",".join(
                        str(value) for value in supported_windows
                    ),
                    "cross_window_supported": bool(supported_windows),
                }
            )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "metric_label",
                "direction",
                "primary_shared_failure_subject_n",
                "cross_window_supported_subject_n",
                "cross_window_supported_subjects",
                "extension_gate",
            ]
        )
    summary_rows: list[dict[str, Any]] = []
    for (metric, direction), group in detail.groupby(
        ["metric", "direction"]
    ):
        supported = group.loc[group["cross_window_supported"]]
        summary_rows.append(
            {
                "metric": metric,
                "metric_label": PHASE_LABELS[metric],
                "direction": direction,
                "primary_shared_failure_subject_n": int(
                    group["iSub"].nunique()
                ),
                "cross_window_supported_subject_n": int(
                    supported["iSub"].nunique()
                ),
                "cross_window_supported_subjects": ",".join(
                    str(value)
                    for value in sorted(supported["iSub"].unique())
                ),
                "extension_gate": bool(
                    supported["iSub"].nunique() >= int(minimum_subjects)
                ),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(
        [
            "extension_gate",
            "cross_window_supported_subject_n",
            "metric",
        ],
        ascending=[False, False, True],
    )


def main() -> None:
    args = parse_args()
    windows = sorted({int(value) for value in args.windows})
    if (
        args.primary_window not in windows
        or min(windows) < 4
        or args.n_jobs <= 0
        or args.bootstrap_count < 1000
        or args.shared_residual_subject_min < 2
    ):
        raise ValueError("Invalid phase-analysis configuration.")
    sensitivity_windows = [
        value for value in windows if value != args.primary_window
    ]
    c1_paths = cache_paths(args.c1_dir)
    acquisition_paths = cache_paths(args.acquisition_dir)
    subjects = sorted(set(c1_paths) & set(acquisition_paths))
    if subjects != sorted(c1_paths) or subjects != sorted(acquisition_paths):
        raise ValueError("C1 and acquisition subject sets must match.")

    results = Parallel(
        n_jobs=min(int(args.n_jobs), len(subjects)),
        verbose=10,
        backend="loky",
    )(
        delayed(analyze_subject)(
            subject=subject,
            c1_path=c1_paths[subject],
            acquisition_path=acquisition_paths[subject],
            windows=windows,
            primary_window=int(args.primary_window),
        )
        for subject in subjects
    )

    subject_rows: list[dict[str, Any]] = []
    metric_rows_all: list[dict[str, Any]] = []
    simulation_pass: dict[str, dict[int, np.ndarray]] = {
        "C1": {},
        "acquisition": {},
    }
    for result in results:
        subject = int(result["subject"])
        row: dict[str, Any] = {
            "iSub": subject,
            "test_n": int(result["test_n"]),
            "phase_signature": result["phase_signature"],
            **{
                f"observed_{metric}_w{args.primary_window}": value
                for metric, value in result["primary_metrics"].items()
            },
        }
        for model in ("C1", "acquisition"):
            for field in (
                "phase_discrepancy",
                "phase_threshold_95",
                "phase_calibration_p",
                "phase_pass_95",
                "phase_scaled_crps",
            ):
                row[f"{model}_{field}"] = result[model][field]
            simulation_pass[model][subject] = result[model][
                "simulation_pass"
            ]
        row["phase_scaled_crps_C1_minus_acquisition"] = (
            row["C1_phase_scaled_crps"]
            - row["acquisition_phase_scaled_crps"]
        )
        subject_rows.append(row)
        metric_rows_all.extend(result["metric_rows"])

    subjects_frame = pd.DataFrame(subject_rows).sort_values("iSub")
    for model in ("C1", "acquisition"):
        subjects_frame[f"{model}_phase_fdr_q"] = benjamini_hochberg(
            subjects_frame[
                f"{model}_phase_calibration_p"
            ].to_numpy(dtype=float)
        )
    metrics = pd.DataFrame(metric_rows_all).sort_values(
        ["model", "iSub", "window", "metric"]
    )
    cohort_rows = [
        cohort_calibration(
            subjects_frame,
            simulation_pass[model],
            model=model,
        )
        for model in ("C1", "acquisition")
    ]
    cohorts = pd.DataFrame(cohort_rows)

    difference = subjects_frame[
        "phase_scaled_crps_C1_minus_acquisition"
    ].to_numpy(dtype=float)
    difference_ci = bootstrap_mean_ci(
        difference,
        count=int(args.bootstrap_count),
        seed=int(args.base_seed),
    )
    shared = shared_residuals(
        metrics,
        primary_window=int(args.primary_window),
        sensitivity_windows=sensitivity_windows,
        minimum_subjects=int(args.shared_residual_subject_min),
    )
    extension_candidates = (
        []
        if shared.empty
        else shared.loc[shared["extension_gate"]].to_dict(
            orient="records"
        )
    )
    c1_pass = subjects_frame["C1_phase_pass_95"].astype(bool)
    acquisition_pass = subjects_frame[
        "acquisition_phase_pass_95"
    ].astype(bool)
    neither = subjects_frame.loc[
        ~c1_pass & ~acquisition_pass, "iSub"
    ].astype(int).tolist()
    c1_row = cohorts.loc[cohorts["model"].eq("C1")].iloc[0]
    c1_group_adequate = bool(
        c1_row["lower_tail_calibration_p"] >= 0.05
    )
    if extension_candidates:
        next_action = "consider_one_targeted_extension"
    elif c1_group_adequate:
        next_action = "stop_adding_group_level_phase_mechanisms"
    else:
        next_action = "reconsider_C1_phase_generator"

    prevalence_rows = []
    signature_names = (
        "混乱",
        "陡升",
        "陡降",
        "渐变",
        "恢复",
        "稳定高水平",
    )
    for label in signature_names:
        count = int(
            subjects_frame["phase_signature"].str.contains(
                label, regex=False
            ).sum()
        )
        prevalence_rows.append(
            {
                "phase": label,
                "subject_n": count,
                "subject_fraction": count / len(subjects_frame),
            }
        )
    prevalence = pd.DataFrame(prevalence_rows)

    decision = {
        "analysis": "frozen_condition1_phase_level_ppc",
        "subjects": subjects,
        "subject_n": len(subjects),
        "windows": windows,
        "primary_window": int(args.primary_window),
        "models_refit_or_retuned": False,
        "new_rollouts_generated": False,
        "phase_metric_n_per_window": len(PHASE_METRICS),
        "joint_phase_dimension": len(PHASE_METRICS) * len(windows),
        "cohort_calibration": cohort_rows,
        "paired_scaled_phase_crps": {
            "mean_C1_minus_acquisition": float(np.mean(difference)),
            "subject_bootstrap_ci95": list(difference_ci),
            "C1_better_subject_n": int(np.sum(difference < 0.0)),
            "acquisition_better_subject_n": int(
                np.sum(difference > 0.0)
            ),
        },
        "coverage_pattern": {
            "both_pass_n": int(np.sum(c1_pass & acquisition_pass)),
            "C1_only_pass_n": int(
                np.sum(c1_pass & ~acquisition_pass)
            ),
            "acquisition_only_pass_n": int(
                np.sum(~c1_pass & acquisition_pass)
            ),
            "neither_pass_n": int(np.sum(~c1_pass & ~acquisition_pass)),
            "neither_pass_subjects": neither,
        },
        "shared_residual_extension_threshold_subject_n": int(
            args.shared_residual_subject_min
        ),
        "shared_cross_window_residual_candidates": extension_candidates,
        "recommended_action": next_action,
        "interpretation": (
            "Phase labels are observable trajectory descriptions, not "
            "latent cognitive states. Passing means the observed phase "
            "signature is not extreme under the frozen generator."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    subjects_frame.to_csv(
        args.output_dir / "subject_phase_summary.csv", index=False
    )
    metrics.to_csv(
        args.output_dir / "phase_metric_summary.csv", index=False
    )
    cohorts.to_csv(
        args.output_dir / "phase_cohort_calibration.csv", index=False
    )
    shared.to_csv(
        args.output_dir / "shared_residual_summary.csv", index=False
    )
    prevalence.to_csv(
        args.output_dir / "observed_phase_prevalence.csv", index=False
    )
    write_json(args.output_dir / "phase_decision.json", decision)
    write_json(
        args.output_dir / "manifest.json",
        {
            "analysis": "condition1_phase_coverage_frozen_models",
            "source_c1_dir": str(args.c1_dir.relative_to(ROOT)),
            "source_acquisition_dir": str(
                args.acquisition_dir.relative_to(ROOT)
            ),
            "windows": windows,
            "primary_window": int(args.primary_window),
            "phase_metrics": [
                {
                    "metric": metric,
                    "label": PHASE_LABELS[metric],
                    "resolution": (
                        "fraction of non-overlapping chunks"
                        if metric in FRACTION_METRICS
                        else "count"
                    ),
                }
                for metric in PHASE_METRICS
            ],
            "primary_thresholds": {
                "low_chunk_accuracy_max": 0.5,
                "high_chunk_accuracy_min": 2.0 / 3.0,
                "stable_high_accuracy_min": 5.0 / 6.0,
                "chance_band": [1.0 / 3.0, 2.0 / 3.0],
                "abrupt_absolute_change_min": 0.25,
                "gradual_run_min_adjacent_changes": 2,
                "gradual_run_total_change_min": 0.25,
            },
            "joint_test": (
                "maximum robust standardized discrepancy over all phase "
                "metrics and all three windows, calibrated with the "
                "model's own rollout pseudo-observations"
            ),
            "proper_score": (
                "mean univariate empirical CRPS over 30 phase dimensions, "
                "scaled by one pooled C1+acquisition simulation scale per "
                "subject and dimension"
            ),
            "bootstrap_count": int(args.bootstrap_count),
            "base_seed": int(args.base_seed),
            "models_refit_or_retuned": False,
            "new_rollouts_generated": False,
        },
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
