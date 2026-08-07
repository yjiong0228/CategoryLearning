#!/usr/bin/env python3
"""Reconstruct and plot every trial for every subject in model_0805.

Figure contract
---------------
Core conclusion:
    Trialwise fit trajectories reveal when each final model tracks or misses
    each subject, without collapsing the data to one score per subject.
Archetype:
    Quantitative atlas plus one detailed page per subject.
Evidence chain:
    The atlas preserves every subject-model-trial probability; each detailed
    page adds behavioral outcome, a trailing-window trend, and cumulative NLL
    change relative to FS_H0, followed by observed/model-predicted accuracy.
    Oral-report compatibility is added only when an explicit source is given.
Backend/output:
    Python only; editable overview SVG/PDF, overview PNG, 32-page detailed PDF,
    one PNG per subject, and complete trial-level source data.
Integrity:
    No subject, model, or trial is excluded. Reconstructed outer-holdout NLL is
    checked against the frozen report before any figure is written.
Reviewer risk:
    Inner-fit and inner-validation trajectories are not independent tests; the
    final outer holdout is marked separately on every panel.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = (
    ROOT
    / "results/zhuran/model_0805_cond1/real_predictive_overnight_20260805_v1"
)
DEFAULT_CONFIG = ROOT / "configs/model_0805_cond1_real_predictive.yaml"
DEFAULT_OUTPUT = DEFAULT_RESULT / "figures/trialwise_model_fit"
DEFAULT_ORAL_TRIALS = None

MODEL_ORDER = [
    "FS_H0",
    "FA2_M3",
    "FA2_M5",
    "FA2_M7",
    "FA2R_M3",
    "FA2R_M5",
    "FA2R_M7",
]
MODEL_LABELS = {
    "FS_H0": "FS_H0",
    "FA2_M3": "FA2 M=3",
    "FA2_M5": "FA2 M=5",
    "FA2_M7": "FA2 M=7",
    "FA2R_M3": "FA2-R M=3",
    "FA2R_M5": "FA2-R M=5",
    "FA2R_M7": "FA2-R M=7",
}
SHORT_LABELS = {
    "FS_H0": "FS",
    "FA2_M3": "A2-3",
    "FA2_M5": "A2-5",
    "FA2_M7": "A2-7",
    "FA2R_M3": "A2R-3",
    "FA2R_M5": "A2R-5",
    "FA2R_M7": "A2R-7",
}
MODEL_COLORS = {
    "FS_H0": "#272727",
    "FA2_M3": "#225E92",
    "FA2_M5": "#5B8FB9",
    "FA2_M7": "#9ABCD5",
    "FA2R_M3": "#8E3D68",
    "FA2R_M5": "#BE6B8B",
    "FA2R_M7": "#DDA8BC",
}
PHASE_COLORS = {
    "inner_fit": "#F2F2F2",
    "inner_validation": "#EAF1F7",
    "outer_holdout": "#FFF1DC",
}
LOG_EPSILON = 1e-300


def apply_style() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Noto Sans CJK JP",
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
    ]
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.major.width"] = 0.7
    plt.rcParams["ytick.major.width"] = 0.7


def stable_logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    maximum = float(np.max(values))
    return maximum + math.log(float(np.sum(np.exp(values - maximum))))


def sequential_mixture_probabilities(
    component_probabilities: np.ndarray, choices: np.ndarray
) -> np.ndarray:
    """Match the formal runner's one-step-ahead sequential model average."""
    probabilities = np.asarray(component_probabilities, dtype=float)
    observed = np.asarray(choices, dtype=int).reshape(-1)
    if probabilities.ndim != 3 or probabilities.shape[1] != observed.size:
        raise ValueError("Expected component x trial x choice probabilities")
    count = int(probabilities.shape[0])
    log_weights = np.full(count, -math.log(count), dtype=float)
    mixture = np.zeros((observed.size, probabilities.shape[2]), dtype=float)
    for trial in range(observed.size):
        weights = np.exp(log_weights - stable_logsumexp(log_weights))
        mixture[trial] = weights @ probabilities[:, trial, :]
        mixture[trial] /= float(np.sum(mixture[trial]))
        # The current response is revealed only after its prediction and then
        # updates component weights for the next trial.
        log_weights += np.log(
            np.clip(probabilities[:, trial, observed[trial]], LOG_EPSILON, 1.0)
        )
    return mixture


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def prediction_root(config_path: Path) -> Path:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    value = Path(str(config["holdout"]["source"]))
    return value if value.is_absolute() else ROOT / value


def component_files(
    result_root: Path,
    final_choices: dict[str, Any],
    subject_id: int,
    model_key: str,
) -> list[Path]:
    choice = final_choices[model_key]
    folder = (
        result_root
        / "components"
        / str(choice["variant_id"])
        / str(choice["stage"])
        / f"subject_{subject_id}"
        / model_key
    )
    files = sorted(folder.glob("*/*.npz"))
    if not files:
        raise FileNotFoundError(f"No final component files under {folder}")
    return files


def reconstruct_subject(
    result_root: Path,
    prediction_dir: Path,
    final_choices: dict[str, Any],
    subject_id: int,
) -> dict[str, Any]:
    prediction_path = prediction_dir / f"subject_{subject_id}.npz"
    with np.load(prediction_path, allow_pickle=False) as payload:
        choices = payload["choice"].astype(np.int64)
        categories = payload["category"].astype(np.int64)
        feedback = payload["feedback"].astype(float)
        holdout = payload["holdout_mask"].astype(bool)
    observed_probabilities: dict[str, np.ndarray] = {}
    choice_probabilities: dict[str, np.ndarray] = {}
    component_counts: dict[str, int] = {}
    for model_key in MODEL_ORDER:
        stacked: list[np.ndarray] = []
        files = component_files(result_root, final_choices, subject_id, model_key)
        for path in files:
            with np.load(path, allow_pickle=False) as payload:
                current = payload["probabilities"].astype(float)
            if current.ndim != 3 or current.shape[1] != choices.size:
                raise ValueError(f"Unexpected probability shape {current.shape} in {path}")
            stacked.extend(current)
        mixture = sequential_mixture_probabilities(np.asarray(stacked), choices)
        choice_probabilities[model_key] = mixture
        observed_probabilities[model_key] = mixture[
            np.arange(choices.size), choices
        ]
        component_counts[model_key] = len(stacked)
    return {
        "subject_id": int(subject_id),
        "choices": choices,
        "categories": categories,
        "feedback": feedback,
        "holdout": holdout,
        "choice_probabilities": choice_probabilities,
        "observed_probabilities": observed_probabilities,
        "component_counts": component_counts,
    }


def attach_oral_target_alignment(
    reconstructed: dict[int, dict[str, Any]], oral_trials_path: Path | None
) -> int:
    """Attach observed oral-target compatibility, leaving unencoded trials NaN."""
    if oral_trials_path is None:
        for payload in reconstructed.values():
            trial_count = int(payload["choices"].size)
            payload["oral_target_compatible"] = np.full(
                trial_count, np.nan, dtype=float
            )
            payload["oral_report_encoded"] = np.zeros(trial_count, dtype=bool)
        return 0
    oral = pd.read_csv(oral_trials_path)
    required = {"subject_id", "condition", "trial", "oral_target_compatible"}
    missing = required - set(oral.columns)
    if missing:
        raise ValueError(f"Oral trial file lacks columns: {sorted(missing)}")
    oral = oral[oral["condition"].eq(1)].copy()
    total_encoded = 0
    for subject_id, payload in reconstructed.items():
        current = oral[oral["subject_id"].eq(subject_id)].copy()
        if current["trial"].duplicated().any():
            raise ValueError(f"Subject {subject_id}: duplicate oral trial rows")
        trial_indices = current["trial"].to_numpy(dtype=int) - 1
        trial_count = int(payload["choices"].size)
        if np.any(trial_indices < 0) or np.any(trial_indices >= trial_count):
            raise ValueError(f"Subject {subject_id}: oral trial index outside task range")
        values = np.full(trial_count, np.nan, dtype=float)
        raw = current["oral_target_compatible"]
        if raw.dtype == bool:
            encoded_values = raw.to_numpy(dtype=float)
        else:
            normalized = raw.astype(str).str.strip().str.lower()
            mapping = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
            encoded_values = normalized.map(mapping).to_numpy(dtype=float)
            if np.isnan(encoded_values).any():
                raise ValueError(f"Subject {subject_id}: invalid oral alignment values")
        values[trial_indices] = encoded_values
        payload["oral_target_compatible"] = values
        payload["oral_report_encoded"] = np.isfinite(values)
        total_encoded += int(np.sum(np.isfinite(values)))
    if total_encoded != len(oral):
        raise ValueError(
            f"Attached {total_encoded} condition-1 oral reports but file contains {len(oral)}"
        )
    return total_encoded


def split_boundaries(split_row: pd.Series) -> tuple[int, int]:
    fit_end = int(split_row["n_inner_fit"])
    holdout_start = fit_end + int(split_row["n_inner_validation"])
    return fit_end, holdout_start


def validate_reconstruction(
    reconstructed: dict[int, dict[str, Any]],
    split_audit: pd.DataFrame,
    frozen_summary: pd.DataFrame,
) -> float:
    maximum_error = 0.0
    split_index = split_audit.set_index("subject_id")
    frozen = frozen_summary.set_index(["subject_id", "model_key"])
    for subject_id, payload in reconstructed.items():
        trial_count = int(payload["choices"].size)
        split_row = split_index.loc[subject_id]
        expected_total = int(
            split_row["n_inner_fit"]
            + split_row["n_inner_validation"]
            + split_row["n_outer_holdout"]
        )
        if trial_count != expected_total:
            raise ValueError(
                f"Subject {subject_id}: {trial_count} trials != split total {expected_total}"
            )
        holdout = payload["holdout"]
        if int(np.sum(holdout)) != int(split_row["n_outer_holdout"]):
            raise ValueError(f"Subject {subject_id}: holdout count mismatch")
        for model_key, probabilities in payload["observed_probabilities"].items():
            nll = float(
                -np.mean(
                    np.log(np.clip(probabilities[holdout], LOG_EPSILON, 1.0))
                )
            )
            expected = float(frozen.loc[(subject_id, model_key), "nll_per_trial"])
            error = abs(nll - expected)
            maximum_error = max(maximum_error, error)
            if not np.isclose(nll, expected, atol=1e-11, rtol=1e-11):
                raise ValueError(
                    f"Subject {subject_id}, {model_key}: rebuilt NLL {nll} != {expected}"
                )
    return maximum_error


def source_rows(
    reconstructed: dict[int, dict[str, Any]], split_audit: pd.DataFrame
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_index = split_audit.set_index("subject_id")
    for subject_id, payload in reconstructed.items():
        fit_end, holdout_start = split_boundaries(split_index.loc[subject_id])
        baseline = payload["observed_probabilities"]["FS_H0"]
        baseline_nll = -np.log(np.clip(baseline, LOG_EPSILON, 1.0))
        for model_key in MODEL_ORDER:
            probability = payload["observed_probabilities"][model_key]
            full_choice_probability = payload["choice_probabilities"][model_key]
            correct_probability = full_choice_probability[
                np.arange(payload["choices"].size), payload["categories"]
            ]
            nll = -np.log(np.clip(probability, LOG_EPSILON, 1.0))
            for index in range(payload["choices"].size):
                if index < fit_end:
                    phase = "inner_fit"
                elif index < holdout_start:
                    phase = "inner_validation"
                else:
                    phase = "outer_holdout"
                rows.append(
                    {
                        "subject_id": int(subject_id),
                        "trial": int(index + 1),
                        "phase": phase,
                        "model_key": model_key,
                        "choice": int(payload["choices"][index]),
                        "feedback_correct": int(payload["feedback"][index] >= 0.5),
                        "probability_assigned_to_observed_choice": float(probability[index]),
                        "model_probability_of_correct_choice": float(correct_probability[index]),
                        "trial_nll": float(nll[index]),
                        "delta_trial_nll_vs_FS_H0": float(nll[index] - baseline_nll[index]),
                        "oral_report_encoded": bool(payload["oral_report_encoded"][index]),
                        "oral_target_compatible": (
                            float(payload["oral_target_compatible"][index])
                            if payload["oral_report_encoded"][index]
                            else np.nan
                        ),
                    }
                )
    return rows


def probability_matrix(payload: dict[str, Any]) -> np.ndarray:
    return np.vstack([payload["observed_probabilities"][model] for model in MODEL_ORDER])


def add_split_lines(ax: mpl.axes.Axes, fit_end: int, holdout_start: int) -> None:
    ax.axvline(fit_end + 0.5, color="#777777", lw=0.8, ls=(0, (3, 2)))
    ax.axvline(holdout_start + 0.5, color="#202020", lw=1.0)


def add_phase_spans(
    ax: mpl.axes.Axes, trial_count: int, fit_end: int, holdout_start: int
) -> None:
    ax.axvspan(0.5, fit_end + 0.5, color=PHASE_COLORS["inner_fit"], zorder=-3)
    ax.axvspan(
        fit_end + 0.5,
        holdout_start + 0.5,
        color=PHASE_COLORS["inner_validation"],
        zorder=-3,
    )
    ax.axvspan(
        holdout_start + 0.5,
        trial_count + 0.5,
        color=PHASE_COLORS["outer_holdout"],
        zorder=-3,
    )
    add_split_lines(ax, fit_end, holdout_start)


def fit_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "observed_choice_probability",
        ["#B64342", "#F4E4E1", "#F7F7F7", "#D5E6F1", "#225E92"],
    )


def make_overview(
    reconstructed: dict[int, dict[str, Any]],
    split_audit: pd.DataFrame,
    output_dir: Path,
) -> None:
    subjects = sorted(reconstructed)
    split_index = split_audit.set_index("subject_id")
    figure, axes = plt.subplots(8, 4, figsize=(15.2, 20.5), facecolor="white")
    axes_flat = axes.ravel()
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    cmap = fit_cmap()
    last_image = None
    for panel_index, (axis, subject_id) in enumerate(zip(axes_flat, subjects)):
        payload = reconstructed[subject_id]
        trial_count = int(payload["choices"].size)
        fit_end, holdout_start = split_boundaries(split_index.loc[subject_id])
        last_image = axis.imshow(
            probability_matrix(payload),
            cmap=cmap,
            norm=norm,
            aspect="auto",
            interpolation="nearest",
            extent=(0.5, trial_count + 0.5, len(MODEL_ORDER) - 0.5, -0.5),
        )
        add_split_lines(axis, fit_end, holdout_start)
        axis.set_title(f"被试 {subject_id}  ·  {trial_count} 试次", fontsize=7.5, pad=3)
        axis.set_xticks([1, trial_count])
        axis.set_xticklabels(["1", str(trial_count)], fontsize=5.5)
        axis.set_yticks(np.arange(len(MODEL_ORDER)))
        if panel_index % 4 == 0:
            axis.set_yticklabels([SHORT_LABELS[item] for item in MODEL_ORDER], fontsize=5.5)
        else:
            axis.set_yticklabels([])
        axis.tick_params(length=0, pad=1)
        for spine in axis.spines.values():
            spine.set_visible(False)
    if last_image is None:
        raise ValueError("No subjects available for overview")
    figure.subplots_adjust(left=0.055, right=0.95, bottom=0.055, top=0.93, wspace=0.14, hspace=0.44)
    cbar_ax = figure.add_axes([0.965, 0.18, 0.012, 0.60])
    colorbar = figure.colorbar(last_image, cax=cbar_ax)
    colorbar.set_label("模型给真实选择的概率", fontsize=8)
    colorbar.ax.tick_params(labelsize=7, length=2)
    figure.suptitle(
        "cond1：32名被试 × 7个模型 × 全部试次的逐试次拟合",
        x=0.055,
        y=0.982,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.055,
        0.958,
        "每个彩色小格都是一个真实试次：蓝色表示模型给被试实际选择较高概率，红色表示模型更意外。",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4D4D4D",
    )
    phase_handles = [
        Line2D([0], [0], color="#777777", lw=1.0, ls=(0, (3, 2)), label="内部拟合 → 内部验证"),
        Line2D([0], [0], color="#202020", lw=1.2, label="内部验证 → 最终留出"),
    ]
    figure.legend(
        handles=phase_handles,
        loc="upper right",
        bbox_to_anchor=(0.95, 0.974),
        ncol=2,
        fontsize=7.5,
    )
    figure.text(
        0.055,
        0.025,
        "模型行顺序：FS、FA2(M=3/5/7)、FA2-R(M=3/5/7)。完整逐被试曲线见32页详细PDF。",
        fontsize=7.5,
        color="#4D4D4D",
    )
    base = output_dir / "trialwise_fit_overview"
    figure.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def trailing_geometric_probability(probability: np.ndarray, window: int) -> np.ndarray:
    nll = -np.log(np.clip(probability, LOG_EPSILON, 1.0))
    rolling = pd.Series(nll).rolling(window=window, min_periods=1).mean().to_numpy()
    return np.exp(-rolling) * 100.0


def trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing task-trial mean; NaN oral trials remain missing, not imputed."""
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window=window, min_periods=1)
        .mean()
        .to_numpy()
    )


def make_subject_figure(
    subject_id: int,
    payload: dict[str, Any],
    split_row: pd.Series,
    rolling_window: int,
) -> plt.Figure:
    trial_count = int(payload["choices"].size)
    trials = np.arange(1, trial_count + 1)
    fit_end, holdout_start = split_boundaries(split_row)
    matrix = probability_matrix(payload)
    nll_matrix = -np.log(np.clip(matrix, LOG_EPSILON, 1.0))

    figure = plt.figure(figsize=(13.0, 13.2), facecolor="white")
    grid = figure.add_gridspec(
        6,
        1,
        height_ratios=[0.16, 1.08, 1.22, 1.22, 1.30, 1.18],
        left=0.09,
        right=0.94,
        bottom=0.055,
        top=0.90,
        hspace=0.42,
    )
    outcome_axis = figure.add_subplot(grid[0, 0])
    heat_axis = figure.add_subplot(grid[1, 0], sharex=outcome_axis)
    rolling_axis = figure.add_subplot(grid[2, 0], sharex=outcome_axis)
    cumulative_axis = figure.add_subplot(grid[3, 0], sharex=outcome_axis)
    accuracy_axis = figure.add_subplot(grid[4, 0], sharex=outcome_axis)
    oral_axis = figure.add_subplot(grid[5, 0], sharex=outcome_axis)

    outcome_cmap = ListedColormap(["#D98B65", "#5B9A78"])
    outcome_axis.imshow(
        (payload["feedback"] >= 0.5)[None, :],
        cmap=outcome_cmap,
        vmin=0,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
        extent=(0.5, trial_count + 0.5, 0.5, -0.5),
    )
    outcome_axis.set_yticks([0])
    outcome_axis.set_yticklabels(["行为反馈"])
    outcome_axis.tick_params(axis="x", labelbottom=False, length=0)
    outcome_axis.tick_params(axis="y", length=0)
    add_split_lines(outcome_axis, fit_end, holdout_start)
    for spine in outcome_axis.spines.values():
        spine.set_visible(False)
    outcome_axis.legend(
        handles=[
            Patch(facecolor="#5B9A78", label="正确"),
            Patch(facecolor="#D98B65", label="错误"),
        ],
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        fontsize=7,
    )

    probability_image = heat_axis.imshow(
        matrix,
        cmap=fit_cmap(),
        norm=TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0),
        aspect="auto",
        interpolation="nearest",
        extent=(0.5, trial_count + 0.5, len(MODEL_ORDER) - 0.5, -0.5),
    )
    heat_axis.set_yticks(np.arange(len(MODEL_ORDER)))
    heat_axis.set_yticklabels([MODEL_LABELS[item] for item in MODEL_ORDER])
    heat_axis.tick_params(axis="x", labelbottom=False, length=0)
    heat_axis.tick_params(axis="y", length=0)
    heat_axis.set_title("a  每个模型在每个试次给被试真实选择的概率", loc="left", fontsize=9, fontweight="bold")
    add_split_lines(heat_axis, fit_end, holdout_start)
    for spine in heat_axis.spines.values():
        spine.set_visible(False)
    cbar = figure.colorbar(probability_image, ax=heat_axis, pad=0.012, fraction=0.025)
    cbar.set_label("真实选择概率", fontsize=7)
    cbar.ax.tick_params(labelsize=6, length=2)

    for model_key in MODEL_ORDER:
        rolling_axis.plot(
            trials,
            trailing_geometric_probability(
                payload["observed_probabilities"][model_key], rolling_window
            ),
            color=MODEL_COLORS[model_key],
            lw=1.15 if model_key != "FS_H0" else 1.7,
            alpha=0.95,
            label=MODEL_LABELS[model_key],
        )
    add_phase_spans(rolling_axis, trial_count, fit_end, holdout_start)
    rolling_axis.set_ylim(0, 100)
    rolling_axis.set_ylabel("真实选择概率（%）")
    rolling_axis.set_title(
        f"b  尾随 {rolling_window} 试次趋势（仅为辅助阅读；原始逐试次值见 a）",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    rolling_axis.grid(axis="y", color="#DDDDDD", lw=0.6)
    rolling_axis.tick_params(axis="x", labelbottom=False)
    rolling_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=7,
        fontsize=6.5,
        handlelength=1.8,
        columnspacing=0.9,
    )

    baseline_nll = nll_matrix[0]
    for model_index, model_key in enumerate(MODEL_ORDER[1:], start=1):
        cumulative_axis.plot(
            trials,
            np.cumsum(nll_matrix[model_index] - baseline_nll),
            color=MODEL_COLORS[model_key],
            lw=1.2,
            label=MODEL_LABELS[model_key],
        )
    add_phase_spans(cumulative_axis, trial_count, fit_end, holdout_start)
    cumulative_axis.axhline(0.0, color="#272727", lw=0.8, ls=(0, (3, 2)))
    cumulative_axis.set_ylabel("累计 ΔNLL\n（复杂模型 − FS_H0）")
    cumulative_axis.set_title(
        "c  相对 FS_H0 的累计拟合差异（低于0：复杂模型累计更好）",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    cumulative_axis.grid(axis="y", color="#DDDDDD", lw=0.6)
    cumulative_axis.tick_params(axis="x", labelbottom=False)
    cumulative_axis.set_xlim(0.5, trial_count + 0.5)

    actual_accuracy = trailing_mean(payload["feedback"], rolling_window) * 100.0
    accuracy_axis.plot(
        trials,
        actual_accuracy,
        color="#009E73",
        lw=2.2,
        label="被试实际正确率",
        zorder=5,
    )
    for model_key in MODEL_ORDER:
        full_probability = payload["choice_probabilities"][model_key]
        probability_correct = full_probability[
            np.arange(trial_count), payload["categories"]
        ]
        accuracy_axis.plot(
            trials,
            trailing_mean(probability_correct, rolling_window) * 100.0,
            color=MODEL_COLORS[model_key],
            lw=1.0 if model_key != "FS_H0" else 1.45,
            alpha=0.92,
            label=MODEL_LABELS[model_key],
        )
    add_phase_spans(accuracy_axis, trial_count, fit_end, holdout_start)
    accuracy_axis.axhline(50.0, color="#777777", lw=0.7, ls=(0, (2, 3)))
    accuracy_axis.set_ylim(0, 100)
    accuracy_axis.set_ylabel("正确率 / 预测正确概率（%）")
    accuracy_axis.set_title(
        f"d  Accuracy curve：实际正确率与模型预测正确率（尾随 {rolling_window} 试次）",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    accuracy_axis.grid(axis="y", color="#DDDDDD", lw=0.6)
    accuracy_axis.tick_params(axis="x", labelbottom=False)
    accuracy_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.19),
        ncol=8,
        fontsize=6.1,
        handlelength=1.7,
        columnspacing=0.75,
    )

    oral_values = np.asarray(payload["oral_target_compatible"], dtype=float)
    oral_encoded = np.isfinite(oral_values)
    oral_rolling = trailing_mean(oral_values, rolling_window) * 100.0
    add_phase_spans(oral_axis, trial_count, fit_end, holdout_start)
    oral_axis.plot(
        trials,
        oral_rolling,
        color="#00796B",
        lw=2.0,
        label=f"尾随{rolling_window}试次一致率",
        zorder=4,
    )
    compatible_rows = oral_encoded & (oral_values >= 0.5)
    incompatible_rows = oral_encoded & (oral_values < 0.5)
    oral_axis.scatter(
        trials[compatible_rows],
        np.full(int(np.sum(compatible_rows)), 100.0),
        s=7,
        color="#5B9A78",
        alpha=0.45,
        linewidths=0,
        label="该试次报告兼容",
        zorder=5,
    )
    oral_axis.scatter(
        trials[incompatible_rows],
        np.zeros(int(np.sum(incompatible_rows))),
        s=7,
        color="#D98B65",
        alpha=0.55,
        linewidths=0,
        label="该试次报告不兼容",
        zorder=5,
    )
    oral_axis.set_ylim(-4, 104)
    oral_axis.set_xlim(0.5, trial_count + 0.5)
    oral_axis.set_xlabel("试次")
    oral_axis.set_ylabel("口头报告与目标规则兼容（%）")
    oral_axis.set_title(
        f"e  Oral report alignment：目标规则兼容率；可编码报告 "
        f"{int(np.sum(oral_encoded))}/{trial_count} 个试次",
        loc="left",
        fontsize=9,
        fontweight="bold",
    )
    oral_axis.grid(axis="y", color="#DDDDDD", lw=0.6)
    oral_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.19),
        ncol=3,
        fontsize=6.5,
        handlelength=1.8,
        columnspacing=1.0,
    )

    phase_handles = [
        Patch(facecolor=PHASE_COLORS["inner_fit"], label=f"内部拟合 1–{fit_end}"),
        Patch(
            facecolor=PHASE_COLORS["inner_validation"],
            label=f"内部验证 {fit_end + 1}–{holdout_start}",
        ),
        Patch(
            facecolor=PHASE_COLORS["outer_holdout"],
            label=f"最终留出 {holdout_start + 1}–{trial_count}",
        ),
    ]
    figure.legend(
        handles=phase_handles,
        loc="upper right",
        bbox_to_anchor=(0.94, 0.965),
        ncol=3,
        fontsize=7,
    )
    figure.suptitle(
        f"被试 {subject_id}：全部 {trial_count} 个试次的模型拟合轨迹",
        x=0.09,
        y=0.982,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.09,
        0.945,
        "选择预测均在看到当前选择之前产生；accuracy 显示正确类别预测；oral 只使用可编码的反馈前报告。",
        fontsize=8,
        color="#4D4D4D",
    )
    return figure


def make_details(
    reconstructed: dict[int, dict[str, Any]],
    split_audit: pd.DataFrame,
    output_dir: Path,
    rolling_window: int,
) -> None:
    subject_dir = output_dir / "subjects"
    subject_dir.mkdir(parents=True, exist_ok=True)
    split_index = split_audit.set_index("subject_id")
    pdf_path = output_dir / "trialwise_fit_all_subjects.pdf"
    with PdfPages(pdf_path) as pdf:
        for subject_id in sorted(reconstructed):
            figure = make_subject_figure(
                subject_id,
                reconstructed[subject_id],
                split_index.loc[subject_id],
                rolling_window,
            )
            pdf.savefig(figure, bbox_inches="tight")
            figure.savefig(
                subject_dir / f"subject_{subject_id}_trialwise_fit.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--oral-trials", type=Path, default=DEFAULT_ORAL_TRIALS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rolling-window", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rolling_window < 1:
        raise ValueError("rolling-window must be positive")
    apply_style()
    args.output.mkdir(parents=True, exist_ok=True)
    report = load_json(args.result_root / "outer_holdout_report.json")
    final_choices = report["final_choices"]
    if set(final_choices) != set(MODEL_ORDER):
        raise ValueError("Final model set does not match the plotting contract")
    split_audit = pd.read_csv(args.result_root / "split_audit.csv").sort_values("subject_id")
    frozen_summary = pd.read_csv(args.result_root / "outer_holdout_subjects.csv")
    prediction_dir = prediction_root(args.config)

    reconstructed = {
        int(subject_id): reconstruct_subject(
            args.result_root,
            prediction_dir,
            final_choices,
            int(subject_id),
        )
        for subject_id in split_audit["subject_id"]
    }
    maximum_error = validate_reconstruction(
        reconstructed, split_audit, frozen_summary
    )
    oral_encoded_reports = attach_oral_target_alignment(
        reconstructed, args.oral_trials
    )
    rows = source_rows(reconstructed, split_audit)
    pd.DataFrame(rows).to_csv(args.output / "trialwise_fit_source_data.csv", index=False)
    make_overview(reconstructed, split_audit, args.output)
    make_details(reconstructed, split_audit, args.output, args.rolling_window)
    manifest = {
        "subjects": len(reconstructed),
        "models": len(MODEL_ORDER),
        "subject_trial_total": int(
            sum(payload["choices"].size for payload in reconstructed.values())
        ),
        "model_subject_trial_rows": len(rows),
        "maximum_outer_nll_reconstruction_error": maximum_error,
        "rolling_window": int(args.rolling_window),
        "oral_encoded_reports": int(oral_encoded_reports),
        "accuracy_curve": "observed rolling correctness versus model probability of correct category",
        "oral_alignment": "encoded report compatible with target rule and current choice; missing reports remain NaN",
        "current_model_specific_oral_distribution_alignment": False,
        "all_subjects_models_trials_retained": True,
    }
    with (args.output / "trialwise_fit_manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
