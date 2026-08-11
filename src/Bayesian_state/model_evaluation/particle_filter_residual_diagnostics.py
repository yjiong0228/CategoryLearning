"""Sequential residual diagnostics for particle-filter choice predictions.

This module is an evaluation component of the shared ``run_model_evaluation``
pipeline.  It averages PF repeats before scoring because those repeats estimate
the same marginalized one-step predictive distribution.  The resulting tests
ask whether observed choices leave temporally predictable residual structure;
they do not refit or mutate the cognitive model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Bayesian_state.metrics import (
    benjamini_hochberg,
    bernoulli_calibration_test,
    forward_residual_state_probe,
    logit_intercept_recalibration,
    martingale_lag_tests,
    rolling_martingale_z,
    switch_residual_test,
)
from src.Bayesian_state.utils.stream import StreamList


DEFAULT_MAX_LAG = 8
DEFAULT_STATE_FOLDS = 4
DEFAULT_STATE_RIDGE = 1.0

PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
}


def _subject_json_files(input_dir: Path) -> list[Path]:
    directory = input_dir / "subjects" if input_dir.name != "subjects" else input_dir
    files = sorted(directory.glob("subject_*.json"))
    if not files:
        raise FileNotFoundError(f"No subject_*.json files found under {input_dir}")
    return files


def _load_stream(payload: Mapping[str, Any], subject_path: Path) -> StreamList:
    reference = payload.get("raw_runs_ref") or {}
    relative_path = reference.get("path")
    count = int(reference.get("count", 0) or 0)
    if not relative_path or count <= 0:
        raise ValueError(f"{subject_path.name} has no usable raw PF run stream")
    stream_path = (subject_path.parent / str(relative_path)).resolve()
    if not stream_path.is_file():
        raise FileNotFoundError(f"PF run stream not found: {stream_path}")
    return StreamList(str(stream_path), count)


def _run_metrics(
    run: Mapping[str, Any],
    eval_prediction_mode: str | None,
) -> Mapping[str, Any]:
    metrics = run.get("metrics")
    if isinstance(metrics, Mapping):
        return metrics
    metrics_by_mode = run.get("metrics_by_mode")
    mode = eval_prediction_mode or run.get("selection_prediction_mode")
    if mode is None and isinstance(metrics_by_mode, Mapping) and len(metrics_by_mode) == 1:
        mode = next(iter(metrics_by_mode))
    if isinstance(metrics_by_mode, Mapping) and mode in metrics_by_mode:
        selected = metrics_by_mode[mode]
        if isinstance(selected, Mapping):
            return selected
    return {}


def _same_array(reference: np.ndarray, candidate: np.ndarray, field: str) -> None:
    if reference.shape != candidate.shape or not np.array_equal(reference, candidate):
        raise ValueError(f"PF repeats disagree on observed field {field!r}")


def _collect_marginal_subject(
    subject_path: Path,
    *,
    eval_prediction_mode: str | None,
    max_runs: int | None,
) -> dict[str, Any]:
    with subject_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    stream = _load_stream(payload, subject_path)
    probability_runs: list[np.ndarray] = []
    reference: dict[str, np.ndarray] | None = None
    window_size = int(
        (payload.get("simulation") or {}).get("window_size")
        or ((payload.get("selection") or {}).get("selection_meta") or {}).get(
            "window_size"
        )
        or 16
    )
    for stream_index, run in enumerate(stream):
        if max_runs is not None and stream_index >= int(max_runs):
            break
        if not isinstance(run, Mapping):
            continue
        metrics = _run_metrics(run, eval_prediction_mode)
        if not metrics:
            continue
        probabilities = np.asarray(metrics.get("pred_category_probs"), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != 2:
            raise ValueError(
                "PF sequential residual diagnosis currently requires binary choices"
            )
        current = {
            "choice": np.asarray(metrics.get("observed_choice_index"), dtype=int).reshape(-1),
            "true_category": np.asarray(metrics.get("true_category_index"), dtype=int).reshape(-1),
            "valid": np.asarray(metrics.get("valid_trial_mask"), dtype=bool).reshape(-1),
            "score": np.asarray(metrics.get("score_trial_mask"), dtype=bool).reshape(-1),
        }
        if any(values.size != probabilities.shape[0] for values in current.values()):
            raise ValueError("PF residual inputs do not align at the trial level")
        if reference is None:
            reference = current
        else:
            for field, values in current.items():
                _same_array(reference[field], values, field)
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("PF category probabilities contain non-finite values")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError("PF category probabilities fall outside [0, 1]")
        if not np.allclose(np.sum(probabilities, axis=1), 1.0, atol=1e-6):
            raise ValueError("PF category probabilities are not normalized")
        probability_runs.append(probabilities)
    if reference is None or not probability_runs:
        raise ValueError(f"No usable PF probability runs found for {subject_path.name}")
    marginal_probability = np.mean(np.stack(probability_runs, axis=0), axis=0)
    valid = reference["valid"] & reference["score"]
    valid &= (reference["choice"] >= 0) & (reference["choice"] < 2)
    valid &= (reference["true_category"] >= 0) & (reference["true_category"] < 2)
    return {
        "subject_id": int(payload.get("subject_id", subject_path.stem.replace("subject_", ""))),
        "condition": int(payload.get("condition", -1)),
        "n_runs": int(len(probability_runs)),
        "window_size": int(window_size),
        "probabilities": marginal_probability,
        "choice": reference["choice"],
        "true_category": reference["true_category"],
        "valid": valid,
    }


def _lag_family_summary(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    finite = [row for row in rows if np.isfinite(float(row["z"]))]
    if not finite:
        return {
            "max_abs_z": float("nan"),
            "max_abs_z_lag": -1,
            "familywise_p": float("nan"),
        }
    maximum = max(finite, key=lambda row: abs(float(row["z"])))
    return {
        "max_abs_z": float(abs(float(maximum["z"]))),
        "max_abs_z_lag": int(maximum["lag"]),
        "familywise_p": float(maximum["familywise_p"]),
    }


def _analyse_subject(
    subject: Mapping[str, Any],
    *,
    max_lag: int,
    state_folds: int,
    state_ridge: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    probabilities = np.asarray(subject["probabilities"], dtype=float)
    choices = np.asarray(subject["choice"], dtype=int)
    true_category = np.asarray(subject["true_category"], dtype=int)
    valid = np.asarray(subject["valid"], dtype=bool)
    trial_index = np.arange(choices.size, dtype=int)
    observed_correct = (choices == true_category).astype(float)
    correct_probability = probabilities[trial_index, true_category]
    observed_choice_one = (choices == 1).astype(float)
    choice_one_probability = probabilities[:, 1]

    accuracy_bias = bernoulli_calibration_test(
        observed_correct,
        correct_probability,
        valid,
    )
    choice_bias = bernoulli_calibration_test(
        observed_choice_one,
        choice_one_probability,
        valid,
    )
    calibrated_correct, accuracy_intercept = logit_intercept_recalibration(
        observed_correct,
        correct_probability,
        valid,
    )
    calibrated_choice_one, choice_intercept = logit_intercept_recalibration(
        observed_choice_one,
        choice_one_probability,
        valid,
    )
    calibrated_probabilities = np.column_stack(
        [1.0 - calibrated_choice_one, calibrated_choice_one]
    )
    accuracy_lags = martingale_lag_tests(
        observed_correct,
        calibrated_correct,
        valid,
        max_lag=max_lag,
    )
    choice_lags = martingale_lag_tests(
        observed_choice_one,
        calibrated_choice_one,
        valid,
        max_lag=max_lag,
    )
    rolling_accuracy = rolling_martingale_z(
        observed_correct,
        calibrated_correct,
        valid,
        window_size=int(subject["window_size"]),
    )
    rolling_choice = rolling_martingale_z(
        observed_choice_one,
        calibrated_choice_one,
        valid,
        window_size=int(subject["window_size"]),
    )
    switch_test = switch_residual_test(
        choices,
        calibrated_probabilities,
        valid,
    )
    state_probe = forward_residual_state_probe(
        observed_correct,
        correct_probability,
        valid,
        window_size=int(subject["window_size"]),
        n_folds=state_folds,
        ridge=state_ridge,
    )

    subject_id = int(subject["subject_id"])
    lag_rows: list[dict[str, Any]] = []
    for residual_type, rows in (
        ("accuracy", accuracy_lags),
        ("choice_label", choice_lags),
    ):
        for row in rows:
            lag_rows.append(
                {
                    "subject_id": subject_id,
                    "residual_type": residual_type,
                    **row,
                }
            )
    trial_rows = pd.DataFrame(
        {
            "subject_id": subject_id,
            "trial": trial_index + 1,
            "score_trial": valid,
            "observed_correct": observed_correct,
            "correct_probability": correct_probability,
            "calibrated_correct_probability": calibrated_correct,
            "calibrated_accuracy_residual": observed_correct - calibrated_correct,
            "rolling_accuracy_residual_z": rolling_accuracy["z_values"],
            "observed_choice_index": choices,
            "choice_one_probability": choice_one_probability,
            "calibrated_choice_one_probability": calibrated_choice_one,
            "calibrated_choice_residual": observed_choice_one - calibrated_choice_one,
            "rolling_choice_residual_z": rolling_choice["z_values"],
            "causal_residual_state": state_probe["state_feature"],
            "forward_fold": state_probe["fold_index"],
            "forward_intercept_probability": state_probe["intercept_probability"],
            "forward_state_probability": state_probe["state_probability"],
        }
    )
    accuracy_serial = _lag_family_summary(accuracy_lags)
    choice_serial = _lag_family_summary(choice_lags)
    state_coefficients = np.asarray(state_probe["state_coefficients"], dtype=float)
    summary = {
        "subject_id": subject_id,
        "condition": int(subject["condition"]),
        "n_pf_runs": int(subject["n_runs"]),
        "n_score_trials": int(np.sum(valid)),
        "window_size": int(subject["window_size"]),
        "max_lag": int(max_lag),
        "accuracy_mean_residual": float(accuracy_bias["mean_residual"]),
        "accuracy_bias_z": float(accuracy_bias["z"]),
        "accuracy_bias_p": float(accuracy_bias["p"]),
        "accuracy_intercept_adjustment": float(accuracy_intercept),
        "choice_mean_residual": float(choice_bias["mean_residual"]),
        "choice_bias_z": float(choice_bias["z"]),
        "choice_bias_p": float(choice_bias["p"]),
        "choice_intercept_adjustment": float(choice_intercept),
        "accuracy_serial_max_abs_z": float(accuracy_serial["max_abs_z"]),
        "accuracy_serial_lag": int(accuracy_serial["max_abs_z_lag"]),
        "accuracy_serial_fwer_p": float(accuracy_serial["familywise_p"]),
        "choice_serial_max_abs_z": float(choice_serial["max_abs_z"]),
        "choice_serial_lag": int(choice_serial["max_abs_z_lag"]),
        "choice_serial_fwer_p": float(choice_serial["familywise_p"]),
        "accuracy_local_max_abs_z": float(rolling_accuracy["max_abs_z"]),
        "accuracy_local_end_trial": int(rolling_accuracy["max_end_index"]) + 1,
        "accuracy_local_fwer_p": float(rolling_accuracy["familywise_p"]),
        "choice_local_max_abs_z": float(rolling_choice["max_abs_z"]),
        "choice_local_end_trial": int(rolling_choice["max_end_index"]) + 1,
        "choice_local_fwer_p": float(rolling_choice["familywise_p"]),
        "switch_mean_residual": float(switch_test["mean_residual"]),
        "switch_bias_z": float(switch_test["z"]),
        "switch_bias_p": float(switch_test["p"]),
        "state_probe_folds": int(state_probe["n_folds"]),
        "state_probe_ridge": float(state_probe["ridge"]),
        "state_probe_n_evaluation_trials": int(state_probe["n_evaluation_trials"]),
        "state_probe_baseline_nll": float(state_probe["baseline_nll"]),
        "state_probe_intercept_nll": float(state_probe["intercept_nll"]),
        "state_probe_nll": float(state_probe["state_nll"]),
        "intercept_minus_baseline_nll": float(
            state_probe["intercept_minus_baseline_nll"]
        ),
        "state_minus_intercept_nll": float(state_probe["state_minus_intercept_nll"]),
        "state_minus_baseline_nll": float(state_probe["state_minus_baseline_nll"]),
        "state_probe_improves_over_intercept": bool(
            float(state_probe["state_minus_intercept_nll"]) < 0.0
        ),
        "state_coefficient_min": (
            float(np.min(state_coefficients)) if state_coefficients.size else float("nan")
        ),
        "state_coefficient_max": (
            float(np.max(state_coefficients)) if state_coefficients.size else float("nan")
        ),
        "state_coefficient_sign_consistent": bool(
            state_coefficients.size > 0
            and (
                np.all(state_coefficients > 0.0)
                or np.all(state_coefficients < 0.0)
            )
        ),
    }
    return trial_rows, pd.DataFrame(lag_rows), summary


def _add_across_subject_adjustment(summary: pd.DataFrame) -> pd.DataFrame:
    output = summary.copy()
    families = (
        "accuracy_serial",
        "choice_serial",
        "accuracy_local",
        "choice_local",
        "switch_bias",
    )
    for family in families:
        p_column = f"{family}_fwer_p" if family != "switch_bias" else "switch_bias_p"
        output[f"{family}_across_subject_q"] = benjamini_hochberg(
            pd.to_numeric(output[p_column], errors="coerce").to_numpy(dtype=float)
        )
    output["accuracy_serial_structure_detected"] = (
        output["accuracy_serial_across_subject_q"] < 0.05
    )
    output["choice_serial_structure_detected"] = (
        output["choice_serial_across_subject_q"] < 0.05
    )
    return output


def _plot_diagnostics(
    trial_data: pd.DataFrame,
    lag_data: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    if output_path.suffix.lower() != ".png":
        raise ValueError("PF residual diagnostic figures must use a .png path")
    subjects = sorted(int(value) for value in summary["subject_id"].unique())
    if not subjects:
        raise ValueError("No subjects are available for residual plotting")
    nll_delta_columns = (
        "intercept_minus_baseline_nll",
        "state_minus_intercept_nll",
    )
    nll_limit = float(
        np.nanmax(
            np.abs(summary.loc[:, nll_delta_columns].to_numpy(dtype=float))
        )
    )
    nll_limit = max(0.005, 1.25 * nll_limit)
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(
            len(subjects),
            3,
            figsize=(7.2, max(2.4, 1.75 * len(subjects))),
            squeeze=False,
        )
        for row_index, subject_id in enumerate(subjects):
            trials = trial_data[trial_data["subject_id"] == subject_id]
            lags = lag_data[lag_data["subject_id"] == subject_id]
            subject_summary = summary[summary["subject_id"] == subject_id].iloc[0]

            time_ax = axes[row_index, 0]
            time_ax.axhspan(-1.96, 1.96, color="#E6E6E6", alpha=0.75)
            time_ax.axhline(0.0, color="#555555", linewidth=0.8)
            time_ax.plot(
                trials["trial"],
                trials["rolling_accuracy_residual_z"],
                color="#0072B2",
                linewidth=1.25,
            )
            time_ax.axhline(1.96, color="#999999", linestyle=":", linewidth=0.8)
            time_ax.axhline(-1.96, color="#999999", linestyle=":", linewidth=0.8)
            time_ax.set_ylabel(f"S{subject_id}\nresidual z")
            if row_index == 0:
                time_ax.set_title("Local performance residual")
            if row_index == len(subjects) - 1:
                time_ax.set_xlabel("Trial")

            lag_ax = axes[row_index, 1]
            lag_ax.axhspan(-1.96, 1.96, color="#E6E6E6", alpha=0.75)
            lag_ax.axhline(0.0, color="#555555", linewidth=0.8)
            for residual_type, label, color, marker, offset in (
                ("accuracy", "Accuracy", "#0072B2", "o", -0.08),
                ("choice_label", "Choice label", "#777777", "s", 0.08),
            ):
                subset = lags[lags["residual_type"] == residual_type]
                lag_ax.plot(
                    subset["lag"].to_numpy(dtype=float) + offset,
                    subset["z"],
                    color=color,
                    marker=marker,
                    markersize=3.5,
                    linewidth=1.0,
                    label=label,
                )
                significant = subset[subset["p"] * int(subject_summary["max_lag"]) < 0.05]
                if not significant.empty:
                    lag_ax.scatter(
                        significant["lag"].to_numpy(dtype=float) + offset,
                        significant["z"],
                        facecolors="white",
                        edgecolors=color,
                        marker=marker,
                        s=32,
                        linewidths=1.0,
                        zorder=4,
                    )
            lag_ax.set_xticks(np.arange(1, int(subject_summary["max_lag"]) + 1))
            if row_index == 0:
                lag_ax.set_title("Residual memory by lag")
                lag_ax.legend(loc="upper left", ncol=2, fontsize=6)
            if row_index == len(subjects) - 1:
                lag_ax.set_xlabel("Lag (trials)")

            nll_ax = axes[row_index, 2]
            deltas = np.asarray(
                [
                    subject_summary["intercept_minus_baseline_nll"],
                    subject_summary["state_minus_intercept_nll"],
                ],
                dtype=float,
            )
            colors = ["#999999", "#0072B2"]
            nll_ax.axvline(0.0, color="#333333", linewidth=0.8)
            nll_ax.barh([1, 0], deltas, color=colors, height=0.55)
            nll_ax.set_yticks([1, 0], ["Level", "Past state"])
            nll_ax.set_xlim(-nll_limit, nll_limit)
            for y_position, value in zip([1, 0], deltas):
                alignment = "left" if value >= 0.0 else "right"
                text_x = value + (0.02 * nll_limit if value >= 0.0 else -0.02 * nll_limit)
                nll_ax.text(
                    text_x,
                    y_position,
                    f"{value:+.3f}",
                    ha=alignment,
                    va="center",
                    fontsize=6,
                )
            if row_index == 0:
                nll_ax.set_title("Forward NLL change\n(level vs fitted; state vs level)")
            if row_index == len(subjects) - 1:
                nll_ax.set_xlabel("ΔNLL (negative is better)")

        fig.suptitle(
            "Does the fitted PF leave predictable sequential structure?",
            fontsize=9,
            y=0.995,
        )
        fig.text(
            0.5,
            0.005,
            (
                "Residual tests remove global level bias; open lag markers pass "
                "within-subject Bonferroni correction. The forward state uses only past residuals."
            ),
            ha="center",
            va="bottom",
            fontsize=6,
        )
        fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.97))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=600, bbox_inches="tight")
        plt.close(fig)


def run_particle_filter_residual_diagnostics(
    input_dir: Path,
    output_dir: Path,
    *,
    subjects: Sequence[int] | None = None,
    eval_prediction_mode: str | None = None,
    max_runs_per_subject: int | None = None,
    max_lag: int = DEFAULT_MAX_LAG,
    state_folds: int = DEFAULT_STATE_FOLDS,
    state_ridge: float = DEFAULT_STATE_RIDGE,
) -> dict[str, pd.DataFrame]:
    """Run the PF residual screen and save source tables plus one PNG figure."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    subject_set = {int(value) for value in subjects} if subjects is not None else None
    trial_frames: list[pd.DataFrame] = []
    lag_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for subject_path in _subject_json_files(input_dir):
        subject_id = int(subject_path.stem.replace("subject_", ""))
        if subject_set is not None and subject_id not in subject_set:
            continue
        subject = _collect_marginal_subject(
            subject_path,
            eval_prediction_mode=eval_prediction_mode,
            max_runs=max_runs_per_subject,
        )
        trial_data, lag_data, summary = _analyse_subject(
            subject,
            max_lag=int(max_lag),
            state_folds=int(state_folds),
            state_ridge=float(state_ridge),
        )
        trial_frames.append(trial_data)
        lag_frames.append(lag_data)
        summaries.append(summary)
    if not summaries:
        raise ValueError("No PF subjects were available for sequential residual diagnosis")
    trial_data = pd.concat(trial_frames, ignore_index=True)
    lag_data = pd.concat(lag_frames, ignore_index=True)
    summary_data = _add_across_subject_adjustment(pd.DataFrame(summaries))
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_data.to_csv(output_dir / "sequential_residual_trial_data.csv", index=False)
    lag_data.to_csv(output_dir / "sequential_residual_lag_tests.csv", index=False)
    summary_data.to_csv(output_dir / "sequential_residual_subject_summary.csv", index=False)
    _plot_diagnostics(
        trial_data,
        lag_data,
        summary_data,
        output_dir / "sequential_residual_diagnostics.png",
    )
    return {
        "trial_data": trial_data,
        "lag_tests": lag_data,
        "subject_summary": summary_data,
    }


__all__ = [
    "DEFAULT_MAX_LAG",
    "DEFAULT_STATE_FOLDS",
    "DEFAULT_STATE_RIDGE",
    "run_particle_filter_residual_diagnostics",
]
