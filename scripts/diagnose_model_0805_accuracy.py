#!/usr/bin/env python3
"""Diagnose the model_0805 NLL-versus-accuracy-trajectory discrepancy.

The script treats subjects as the independent unit.  It reconstructs the
trailing-window accuracy comparison shown in the subject figures, separates
level error from trajectory-shape error, checks multiple rolling windows, and
audits the sequential parameter weights of FS_H0.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    ROOT
    / "results/zhuran/model_0805_cond1/real_predictive_overnight_20260805_v1"
)
DEFAULT_TRIALS = (
    DEFAULT_RESULT / "figures/trialwise_model_fit/trialwise_fit_source_data.csv"
)
DEFAULT_OUTPUT = ROOT / "reports/model_0805_accuracy_diagnostic_20260806"

MODEL_ORDER = [
    "FS_H0",
    "FA2_M3",
    "FA2_M5",
    "FA2_M7",
    "FA2R_M3",
    "FA2R_M5",
    "FA2R_M7",
]
PHASE_ORDER = ["all", "inner_fit", "inner_validation", "outer_holdout"]
WINDOWS = [8, 16, 32]


def stable_softmax(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - float(np.max(log_weights))
    weights = np.exp(shifted)
    return weights / float(np.sum(weights))


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=window, min_periods=1)
        .mean()
        .to_numpy()
    )


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 3 or np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def sustained_onset(curve: np.ndarray, threshold: float, sustain: int = 8) -> int:
    """Return one-indexed onset; n+1 indicates that the criterion was not met."""
    above = np.asarray(curve, dtype=float) >= threshold
    if above.size < sustain:
        return int(above.size + 1)
    hits = np.convolve(above.astype(int), np.ones(sustain, dtype=int), mode="valid")
    indices = np.flatnonzero(hits == sustain)
    return int(indices[0] + 1) if indices.size else int(above.size + 1)


def metric_rows(data: pd.DataFrame, window: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (subject_id, model_key), current in data.groupby(
        ["subject_id", "model_key"], sort=False
    ):
        current = current.sort_values("trial")
        actual = current["feedback_correct"].to_numpy(dtype=float)
        predicted = current["model_probability_of_correct_choice"].to_numpy(
            dtype=float
        )
        actual_curve = trailing_mean(actual, window)
        predicted_curve = trailing_mean(predicted, window)
        phases = current["phase"].to_numpy(dtype=str)
        for phase in PHASE_ORDER:
            mask = np.ones(current.shape[0], dtype=bool) if phase == "all" else phases == phase
            actual_phase = actual_curve[mask]
            predicted_phase = predicted_curve[mask]
            residual = predicted_phase - actual_phase
            centered_residual = (
                predicted_phase - float(np.mean(predicted_phase))
            ) - (actual_phase - float(np.mean(actual_phase)))
            row: dict[str, Any] = {
                "subject_id": int(subject_id),
                "model_key": str(model_key),
                "phase": phase,
                "rolling_window": int(window),
                "n_trials": int(np.sum(mask)),
                "nll_per_trial": float(current.loc[mask, "trial_nll"].mean()),
                "choice_brier": float(
                    np.mean((predicted[mask] - actual[mask]) ** 2)
                ),
                "curve_mae": float(np.mean(np.abs(residual))),
                "curve_rmse": float(np.sqrt(np.mean(residual**2))),
                "curve_bias": float(np.mean(residual)),
                "curve_centered_mae": float(np.mean(np.abs(centered_residual))),
                "curve_correlation": safe_correlation(
                    actual_phase, predicted_phase
                ),
                "actual_curve_sd": float(np.std(actual_phase, ddof=0)),
                "predicted_curve_sd": float(np.std(predicted_phase, ddof=0)),
            }
            if phase == "all":
                for threshold in (0.70, 0.75, 0.80):
                    suffix = str(int(round(threshold * 100)))
                    actual_onset = sustained_onset(actual_curve, threshold)
                    predicted_onset = sustained_onset(predicted_curve, threshold)
                    row[f"actual_onset_{suffix}"] = actual_onset
                    row[f"predicted_onset_{suffix}"] = predicted_onset
                    row[f"onset_error_{suffix}"] = predicted_onset - actual_onset
            rows.append(row)
    return rows


def summarize_subject_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for keys, current in metrics.groupby(
        ["rolling_window", "phase", "model_key"], sort=False
    ):
        window, phase, model_key = keys
        record: dict[str, Any] = {
            "rolling_window": int(window),
            "phase": str(phase),
            "model_key": str(model_key),
            "n_subjects": int(current.shape[0]),
        }
        for column in [
            "nll_per_trial",
            "choice_brier",
            "curve_mae",
            "curve_rmse",
            "curve_bias",
            "curve_centered_mae",
            "curve_correlation",
            "actual_curve_sd",
            "predicted_curve_sd",
        ]:
            record[f"mean_{column}"] = float(current[column].mean())
            record[f"median_{column}"] = float(current[column].median())
        if str(phase) == "all":
            for suffix in ("70", "75", "80"):
                record[f"median_actual_onset_{suffix}"] = float(
                    current[f"actual_onset_{suffix}"].median()
                )
                record[f"median_predicted_onset_{suffix}"] = float(
                    current[f"predicted_onset_{suffix}"].median()
                )
                record[f"median_onset_error_{suffix}"] = float(
                    current[f"onset_error_{suffix}"].median()
                )
                record[f"mean_abs_onset_error_{suffix}"] = float(
                    current[f"onset_error_{suffix}"].abs().mean()
                )
        records.append(record)
    summary = pd.DataFrame(records)
    summary["model_key"] = pd.Categorical(
        summary["model_key"], categories=MODEL_ORDER, ordered=True
    )
    summary["phase"] = pd.Categorical(
        summary["phase"], categories=PHASE_ORDER, ordered=True
    )
    return summary.sort_values(["rolling_window", "phase", "model_key"])


def bootstrap_mean_ci(
    values: np.ndarray, seed: int, replicates: int = 20_000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(replicates, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    selected = metrics[metrics["rolling_window"].eq(16)].copy()
    records: list[dict[str, Any]] = []
    seed = 20260806
    for phase in PHASE_ORDER:
        phase_data = selected[selected["phase"].eq(phase)]
        baseline = phase_data[phase_data["model_key"].eq("FS_H0")].set_index(
            "subject_id"
        )
        for model_key in MODEL_ORDER[1:]:
            candidate = phase_data[phase_data["model_key"].eq(model_key)].set_index(
                "subject_id"
            )
            shared = baseline.index.intersection(candidate.index)
            curve_delta = (
                candidate.loc[shared, "curve_mae"].to_numpy()
                - baseline.loc[shared, "curve_mae"].to_numpy()
            )
            centered_delta = (
                candidate.loc[shared, "curve_centered_mae"].to_numpy()
                - baseline.loc[shared, "curve_centered_mae"].to_numpy()
            )
            nll_delta = (
                candidate.loc[shared, "nll_per_trial"].to_numpy()
                - baseline.loc[shared, "nll_per_trial"].to_numpy()
            )
            curve_ci = bootstrap_mean_ci(curve_delta, seed)
            nll_ci = bootstrap_mean_ci(nll_delta, seed + 1)
            records.append(
                {
                    "phase": phase,
                    "model_key": model_key,
                    "n_subjects": int(shared.size),
                    "mean_curve_mae_delta_vs_FS": float(np.mean(curve_delta)),
                    "curve_mae_delta_ci_low": curve_ci[0],
                    "curve_mae_delta_ci_high": curve_ci[1],
                    "subjects_better_curve_mae_than_FS": int(np.sum(curve_delta < 0)),
                    "mean_centered_curve_mae_delta_vs_FS": float(
                        np.mean(centered_delta)
                    ),
                    "subjects_better_centered_curve_than_FS": int(
                        np.sum(centered_delta < 0)
                    ),
                    "mean_nll_delta_vs_FS": float(np.mean(nll_delta)),
                    "nll_delta_ci_low": nll_ci[0],
                    "nll_delta_ci_high": nll_ci[1],
                    "subjects_better_nll_than_FS": int(np.sum(nll_delta < 0)),
                }
            )
            seed += 2
    return pd.DataFrame(records)


def fs_plateau_diagnostic(data: pd.DataFrame, window: int = 16) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    fs = data[data["model_key"].eq("FS_H0")]
    for subject_id, current in fs.groupby("subject_id", sort=True):
        current = current.sort_values("trial")
        actual = current["feedback_correct"].to_numpy(dtype=float)
        predicted = current["model_probability_of_correct_choice"].to_numpy(
            dtype=float
        )
        actual_curve = trailing_mean(actual, window)
        predicted_curve = trailing_mean(predicted, window)
        later = current["trial"].to_numpy(dtype=int) > 32
        actual_onset = sustained_onset(actual_curve, 0.75)
        fit_end = int(current.loc[current["phase"].eq("inner_fit"), "trial"].max())
        outer_start = int(
            current.loc[current["phase"].eq("outer_holdout"), "trial"].min()
        )
        records.append(
            {
                "subject_id": int(subject_id),
                "n_trials": int(current.shape[0]),
                "fraction_trials_FS_correct_probability_0p89_to_0p91": float(
                    np.mean((predicted >= 0.89) & (predicted <= 0.91))
                ),
                "fraction_later_trials_FS_correct_probability_0p89_to_0p91": float(
                    np.mean((predicted[later] >= 0.89) & (predicted[later] <= 0.91))
                ),
                "later_actual_curve_sd": float(np.std(actual_curve[later])),
                "later_FS_curve_sd": float(np.std(predicted_curve[later])),
                "later_FS_minus_actual_mean": float(
                    np.mean(predicted_curve[later] - actual_curve[later])
                ),
                "actual_onset_75": actual_onset,
                "FS_onset_75": sustained_onset(predicted_curve, 0.75),
                "inner_fit_end": fit_end,
                "outer_holdout_start": outer_start,
                "actual_onset_before_outer": bool(actual_onset < outer_start),
            }
        )
    return pd.DataFrame(records)


def outer_constant_baselines(data: pd.DataFrame) -> pd.DataFrame:
    """Compare FS with simple per-subject constant correctness forecasts."""
    records: list[dict[str, Any]] = []
    fs = data[data["model_key"].eq("FS_H0")]
    for subject_id, current in fs.groupby("subject_id", sort=True):
        current = current.sort_values("trial")
        outer = current[current["phase"].eq("outer_holdout")]
        pre_outer = current[current["phase"].ne("outer_holdout")]
        validation = current[current["phase"].eq("inner_validation")]
        outcome = outer["feedback_correct"].to_numpy(dtype=float)

        def constant_nll(probability: float) -> float:
            probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
            return float(
                np.mean(
                    -outcome * np.log(probability)
                    - (1 - outcome) * np.log(1 - probability)
                )
            )

        outer_accuracy = float(np.mean(outcome))
        records.append(
            {
                "subject_id": int(subject_id),
                "n_outer_trials": int(outer.shape[0]),
                "outer_accuracy": outer_accuracy,
                "FS_outer_nll_per_trial": float(outer["trial_nll"].mean()),
                "oracle_outer_rate_constant_nll": constant_nll(outer_accuracy),
                "pre_outer_rate": float(pre_outer["feedback_correct"].mean()),
                "pre_outer_rate_constant_nll": constant_nll(
                    float(pre_outer["feedback_correct"].mean())
                ),
                "inner_validation_rate": float(
                    validation["feedback_correct"].mean()
                ),
                "inner_validation_rate_constant_nll": constant_nll(
                    float(validation["feedback_correct"].mean())
                ),
            }
        )
    return pd.DataFrame(records)


def load_final_choice(result_root: Path, model_key: str) -> dict[str, Any]:
    report_path = result_root / "outer_holdout_report.json"
    with report_path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    return report["final_choices"][model_key]


def fs_parameter_posterior(
    data: pd.DataFrame, result_root: Path
) -> pd.DataFrame:
    final_choice = load_final_choice(result_root, "FS_H0")
    component_root = (
        result_root
        / "components"
        / str(final_choice["variant_id"])
        / str(final_choice["stage"])
    )
    records: list[dict[str, Any]] = []
    fs = data[data["model_key"].eq("FS_H0")]
    for subject_id, current in fs.groupby("subject_id", sort=True):
        current = current.sort_values("trial")
        choices = current["choice"].to_numpy(dtype=int)
        phases = current["phase"].to_numpy(dtype=str)
        files = sorted(
            (component_root / f"subject_{int(subject_id)}" / "FS_H0").glob(
                "*/*.npz"
            )
        )
        probability_rows: list[np.ndarray] = []
        lapses: list[float] = []
        kappas: list[float] = []
        structure_ids: list[str] = []
        for path in files:
            with np.load(path, allow_pickle=False) as payload:
                probabilities = payload["probabilities"].astype(float)
                lapse_values = payload["lapse"].astype(float)
                metadata = json.loads(str(payload["metadata_json"].item()))
            structure = metadata["structure"]
            if probabilities.shape[0] != lapse_values.size:
                raise ValueError(f"Lapse/probability mismatch in {path}")
            for index, lapse in enumerate(lapse_values):
                probability_rows.append(probabilities[index])
                lapses.append(float(lapse))
                kappas.append(float(structure["kappa"]))
                structure_ids.append(str(structure["structure_id"]))
        probabilities = np.asarray(probability_rows, dtype=float)
        lapse_array = np.asarray(lapses, dtype=float)
        kappa_array = np.asarray(kappas, dtype=float)
        log_weights = np.full(
            probabilities.shape[0], -math.log(probabilities.shape[0]), dtype=float
        )
        phase_end = {
            phase: int(np.flatnonzero(phases == phase)[-1])
            for phase in ("inner_fit", "inner_validation", "outer_holdout")
        }
        for trial in range(choices.size):
            log_weights += np.log(
                np.clip(
                    probabilities[:, trial, choices[trial]], 1e-300, 1.0
                )
            )
            for phase, end_index in phase_end.items():
                if trial != end_index:
                    continue
                weights = stable_softmax(log_weights)
                max_index = int(np.argmax(weights))
                records.append(
                    {
                        "subject_id": int(subject_id),
                        "phase_endpoint": phase,
                        "n_observed_choices": int(trial + 1),
                        "posterior_mean_lapse": float(weights @ lapse_array),
                        "posterior_mass_lapse_0p20": float(
                            np.sum(weights[np.isclose(lapse_array, 0.20)])
                        ),
                        "posterior_mass_lapse_ge_0p10": float(
                            np.sum(weights[lapse_array >= 0.10])
                        ),
                        "posterior_mean_kappa": float(weights @ kappa_array),
                        "map_lapse": float(lapse_array[max_index]),
                        "map_kappa": float(kappa_array[max_index]),
                        "map_component_mass": float(weights[max_index]),
                        "map_structure_id": structure_ids[max_index],
                        "n_parameter_components": int(weights.size),
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=Path, default=DEFAULT_TRIALS)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = pd.read_csv(args.trials)
    required = {
        "subject_id",
        "trial",
        "phase",
        "model_key",
        "choice",
        "feedback_correct",
        "model_probability_of_correct_choice",
        "trial_nll",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing trial columns: {sorted(missing)}")
    if sorted(data["model_key"].unique()) != sorted(MODEL_ORDER):
        raise ValueError("The trial file does not contain the expected seven models")

    all_metrics = pd.DataFrame(
        [row for window in WINDOWS for row in metric_rows(data, window)]
    )
    summary = summarize_subject_metrics(all_metrics)
    paired = paired_comparisons(all_metrics)
    plateau = fs_plateau_diagnostic(data)
    constant_baselines = outer_constant_baselines(data)
    parameter_posterior = fs_parameter_posterior(data, args.result_root)

    args.output.mkdir(parents=True, exist_ok=True)
    all_metrics.to_csv(args.output / "subject_curve_metrics.csv", index=False)
    summary.to_csv(args.output / "group_curve_summary.csv", index=False)
    paired.to_csv(args.output / "paired_model_vs_FS.csv", index=False)
    plateau.to_csv(args.output / "FS_H0_plateau_diagnostic.csv", index=False)
    constant_baselines.to_csv(
        args.output / "outer_constant_baseline_diagnostic.csv", index=False
    )
    parameter_posterior.to_csv(
        args.output / "FS_H0_parameter_posterior.csv", index=False
    )

    window16 = summary[summary["rolling_window"].eq(16)].copy()
    result = {
        "analysis": "model_0805_accuracy_trajectory_diagnostic",
        "subject_count": int(data["subject_id"].nunique()),
        "trial_count": int(
            data[["subject_id", "trial"]].drop_duplicates().shape[0]
        ),
        "model_count": int(data["model_key"].nunique()),
        "rolling_windows_checked": WINDOWS,
        "primary_rolling_window": 16,
        "subject_weighting": "equal weight per subject",
        "onset_definition": (
            "first trial of eight consecutive trailing-window accuracy values "
            "at or above threshold; n_trials+1 means not reached"
        ),
        "window16_group_summary": json.loads(
            window16.to_json(orient="records")
        ),
        "all_window_group_summary": json.loads(
            summary.to_json(orient="records")
        ),
        "paired_model_vs_FS": json.loads(paired.to_json(orient="records")),
        "fs_plateau_group_summary": {
            "median_fraction_later_trials_near_0p90": float(
                plateau[
                    "fraction_later_trials_FS_correct_probability_0p89_to_0p91"
                ].median()
            ),
            "mean_fraction_later_trials_near_0p90": float(
                plateau[
                    "fraction_later_trials_FS_correct_probability_0p89_to_0p91"
                ].mean()
            ),
            "median_later_actual_curve_sd": float(
                plateau["later_actual_curve_sd"].median()
            ),
            "median_later_FS_curve_sd": float(
                plateau["later_FS_curve_sd"].median()
            ),
            "median_FS_onset_minus_actual_onset_75": float(
                (plateau["FS_onset_75"] - plateau["actual_onset_75"]).median()
            ),
            "subjects_actual_onset_before_outer": int(
                plateau["actual_onset_before_outer"].sum()
            ),
        },
        "outer_constant_baseline_group_summary": {
            "mean_subject_outer_accuracy": float(
                constant_baselines["outer_accuracy"].mean()
            ),
            "mean_FS_outer_nll_per_trial": float(
                constant_baselines["FS_outer_nll_per_trial"].mean()
            ),
            "mean_oracle_outer_rate_constant_nll": float(
                constant_baselines["oracle_outer_rate_constant_nll"].mean()
            ),
            "mean_pre_outer_rate_constant_nll": float(
                constant_baselines["pre_outer_rate_constant_nll"].mean()
            ),
            "mean_inner_validation_rate_constant_nll": float(
                constant_baselines["inner_validation_rate_constant_nll"].mean()
            ),
        },
        "fs_parameter_posterior_group_summary": {},
    }
    for phase, current in parameter_posterior.groupby("phase_endpoint"):
        result["fs_parameter_posterior_group_summary"][phase] = {
            "median_posterior_mean_lapse": float(
                current["posterior_mean_lapse"].median()
            ),
            "mean_posterior_mean_lapse": float(
                current["posterior_mean_lapse"].mean()
            ),
            "median_posterior_mass_lapse_0p20": float(
                current["posterior_mass_lapse_0p20"].median()
            ),
            "subjects_map_lapse_0p20": int(
                np.sum(np.isclose(current["map_lapse"], 0.20))
            ),
        }
    with (args.output / "diagnostic_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)

    print(f"Wrote diagnostic outputs to {args.output}")


if __name__ == "__main__":
    main()
