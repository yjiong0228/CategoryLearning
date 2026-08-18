#!/usr/bin/env python3
"""Plot subject-wise rolling-accuracy bands for the model 0815 H4 screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Bayesian_state.metrics import (  # noqa: E402
    conditional_behavioral_accuracy_band_metrics,
)


DEFAULT_INPUT = ROOT / (
    "results/model_dynamic_adaptive_control/0815_h4/mechanism_audit/"
    "02_nested_subject_screen_coarse"
)
DEFAULT_OUTPUT = ROOT / (
    "results/model_dynamic_adaptive_control/0815_h4/mechanism_audit/"
    "03_accuracy_bands"
)
DATA_PATH = ROOT / "data/processed/Task2_processed.csv"
WINDOW_SIZE = 16
START_INDEX = 1
BEHAVIORAL_DRAWS = 5_000
BEHAVIORAL_SEED = 20260817
SELECTED_VARIANT = "nested_selected"
BOUNDARY_VARIANT = "reactive_boundary"

FIGURE_NAME = "h4_subject_accuracy_bands.png"
TRIAL_DATA_NAME = "accuracy_band_trial_data.csv"
SUMMARY_NAME = "accuracy_band_summary.csv"
README_NAME = "README.md"

SELECTED_COLOR = "#E69F00"
BOUNDARY_COLOR = "#484878"
BAND_90_COLOR = "#B4CDE2"
BAND_50_COLOR = "#6F9FC6"
OBSERVED_COLOR = "#111111"
SPLIT_COLOR = "#7A7A7A"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--behavioral-draws", type=int, default=BEHAVIORAL_DRAWS)
    parser.add_argument("--behavioral-seed", type=int, default=BEHAVIORAL_SEED)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing figure package in the requested output directory.",
    )
    return parser.parse_args()


def _load_manifest(input_dir: Path) -> dict[str, Any]:
    path = input_dir / "run_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing H4 run manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("analysis_id") != "model_0815_h4_nested_subject_parameter_screen_v1":
        raise ValueError("Input directory is not the expected H4 subject screen")
    return payload


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_panel(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        required = {
            "choice_probability",
            "filter_seed",
            "observed_choice_index",
            "valid_trial_mask",
        }
        missing = sorted(required.difference(bundle.files))
        if missing:
            raise ValueError(f"Panel {path} is missing arrays: {missing}")
        panel = {name: np.asarray(bundle[name]) for name in required}
    probabilities = np.asarray(panel["choice_probability"], dtype=float)
    if probabilities.ndim != 3 or probabilities.shape[2] != 2:
        raise ValueError(f"Unexpected probability shape in {path}: {probabilities.shape}")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError(f"Invalid choice probabilities in {path}")
    row_sums = probabilities.sum(axis=2, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"Zero-mass probability row in {path}")
    panel["choice_probability"] = probabilities / row_sums
    panel["filter_seed"] = np.asarray(panel["filter_seed"], dtype=np.uint64).reshape(-1)
    panel["observed_choice_index"] = np.asarray(
        panel["observed_choice_index"], dtype=int
    ).reshape(-1)
    panel["valid_trial_mask"] = np.asarray(
        panel["valid_trial_mask"], dtype=bool
    ).reshape(-1)
    return panel


def _load_subject_trials(
    data: pd.DataFrame,
    *,
    subject_id: int,
    n_trials: int,
) -> pd.DataFrame:
    required = {"iSub", "category", "choice", "feedback"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Behavioral data are missing columns: {missing}")
    subject = data.loc[data["iSub"] == subject_id].iloc[:n_trials].copy()
    if len(subject) != n_trials:
        raise ValueError(
            f"Subject {subject_id} has {len(subject)} rows, expected {n_trials}"
        )
    category = subject["category"].to_numpy(dtype=int)
    choice = subject["choice"].to_numpy(dtype=int)
    feedback = subject["feedback"].to_numpy(dtype=float)
    if np.any((category < 1) | (category > 2)):
        raise ValueError(f"Subject {subject_id} contains a non-binary category")
    if not np.all(np.isin(feedback, [0.0, 1.0])):
        raise ValueError(f"Subject {subject_id} contains non-binary feedback")
    expected_feedback = (choice == category).astype(float)
    if not np.array_equal(feedback, expected_feedback):
        raise ValueError(
            f"Subject {subject_id} feedback disagrees with choice/category coding"
        )
    return subject


def _correct_probability(
    probability: np.ndarray,
    category_index: np.ndarray,
) -> np.ndarray:
    if probability.shape[1] != category_index.size:
        raise ValueError("Category sequence and PF probability panel do not align")
    return np.take_along_axis(
        probability,
        category_index[None, :, None],
        axis=2,
    )[:, :, 0]


def _observed_curve(
    feedback: np.ndarray,
    *,
    window_size: int,
) -> np.ndarray:
    starts = np.arange(
        START_INDEX,
        feedback.size - int(window_size) + 1,
        dtype=int,
    )
    return np.asarray(
        [np.mean(feedback[start : start + window_size]) for start in starts],
        dtype=float,
    )


def _subject_seed(base_seed: int, subject_id: int) -> int:
    return int(
        np.random.SeedSequence([int(base_seed), int(subject_id)]).generate_state(1)[0]
    )


def _run_index(manifest: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    index = {
        (int(row["subject_id"]), str(row["variant_id"])): row
        for row in manifest["evaluation_runs"]
    }
    subjects = [int(value) for value in manifest["design"]["subjects"]]
    expected = {
        (subject_id, variant)
        for subject_id in subjects
        for variant in (BOUNDARY_VARIANT, SELECTED_VARIANT)
    }
    missing = sorted(expected.difference(index))
    if missing:
        raise ValueError(f"Manifest is missing evaluation panels: {missing}")
    return index


def _build_subject_band(
    *,
    subject_id: int,
    manifest: Mapping[str, Any],
    index: Mapping[tuple[int, str], Mapping[str, Any]],
    data: pd.DataFrame,
    selection: pd.Series,
    contrast: pd.Series,
    window_size: int,
    n_draws: int,
    base_seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    panels = {
        variant: _load_panel(
            _resolve_repo_path(index[(subject_id, variant)]["npz_path"])
        )
        for variant in (BOUNDARY_VARIANT, SELECTED_VARIANT)
    }
    selected = panels[SELECTED_VARIANT]
    boundary = panels[BOUNDARY_VARIANT]
    if not np.array_equal(selected["filter_seed"], boundary["filter_seed"]):
        raise ValueError(f"Subject {subject_id} comparison is not seed-paired")
    if not np.array_equal(
        selected["observed_choice_index"], boundary["observed_choice_index"]
    ) or not np.array_equal(
        selected["valid_trial_mask"], boundary["valid_trial_mask"]
    ):
        raise ValueError(f"Subject {subject_id} panels use different observations")

    n_trials = int(selected["choice_probability"].shape[1])
    subject = _load_subject_trials(data, subject_id=subject_id, n_trials=n_trials)
    choice_index = subject["choice"].to_numpy(dtype=int) - 1
    if not np.array_equal(choice_index, selected["observed_choice_index"]):
        raise ValueError(f"Subject {subject_id} PF choices do not align to source data")
    category_index = subject["category"].to_numpy(dtype=int) - 1
    feedback = subject["feedback"].to_numpy(dtype=float)
    observed = _observed_curve(feedback, window_size=window_size)
    selected_p_correct = _correct_probability(
        selected["choice_probability"], category_index
    )
    boundary_p_correct = _correct_probability(
        boundary["choice_probability"], category_index
    )
    score_mask = selected["valid_trial_mask"].copy()
    score_mask[0] = False
    seed = _subject_seed(base_seed, subject_id)
    selected_band = conditional_behavioral_accuracy_band_metrics(
        selected_p_correct,
        observed,
        window_size=window_size,
        n_draws=n_draws,
        seed=seed,
        score_trial_mask=score_mask,
        start_index=START_INDEX,
    )
    boundary_band = conditional_behavioral_accuracy_band_metrics(
        boundary_p_correct,
        observed,
        window_size=window_size,
        n_draws=n_draws,
        seed=seed,
        score_trial_mask=score_mask,
        start_index=START_INDEX,
    )

    x = np.arange(window_size + 1, n_trials + 1, dtype=int)
    if x.size != observed.size:
        raise RuntimeError("Rolling curve coordinates are internally inconsistent")
    trial_frame = pd.DataFrame(
        {
            "subject_id": subject_id,
            "trial": x,
            "window_start_trial": x - window_size + 1,
            "window_end_trial": x,
            "observed_rolling_accuracy": observed,
            "selected_expected_accuracy": selected_band["expected_curve"],
            "reactive_expected_accuracy": boundary_band["expected_curve"],
            "selected_behavioral_q05": selected_band["q05"],
            "selected_behavioral_q25": selected_band["q25"],
            "selected_behavioral_q50": selected_band["q50"],
            "selected_behavioral_q75": selected_band["q75"],
            "selected_behavioral_q95": selected_band["q95"],
            "window_fully_heldout": (x - window_size + 1) >= (
                int(manifest["design"]["train_trials"]) + 1
            ),
        }
    )
    curve_difference = np.abs(
        np.asarray(selected_band["expected_curve"], dtype=float)
        - np.asarray(boundary_band["expected_curve"], dtype=float)
    )
    summary = {
        "subject_id": subject_id,
        "accumulator_logit_gain": float(selection["accumulator_logit_gain"]),
        "n_pf_runs": int(selected_band["n_runs"]),
        "n_behavioral_draws": int(selected_band["n_draws"]),
        "behavioral_seed": int(selected_band["seed"]),
        "window_size": int(window_size),
        "rolling_point_count": int(x.size),
        "selected_coverage_50": float(selected_band["coverage_50"]),
        "selected_coverage_90": float(selected_band["coverage_90"]),
        "selected_mean_width_50": float(selected_band["mean_width_50"]),
        "selected_mean_width_90": float(selected_band["mean_width_90"]),
        "selected_expected_curve_mae": float(selected_band["expected_curve_mae"]),
        "max_abs_selected_minus_reactive_curve": float(np.nanmax(curve_difference)),
        "mean_abs_selected_minus_reactive_curve": float(np.nanmean(curve_difference)),
        "paired_delta_nll_heldout": float(contrast["paired_delta_nll_heldout"]),
        "paired_delta_nll_mcse_heldout": float(
            contrast["paired_delta_nll_mcse_heldout"]
        ),
        "heldout_support": bool(selection["heldout_support"]),
    }
    return trial_frame, summary


def _plot(
    trial_data: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    train_trials: int,
    window_size: int,
    n_draws: int,
    output_path: Path,
) -> None:
    subjects = [int(value) for value in summary["subject_id"]]
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.8, 6.7),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    fully_heldout_end = int(train_trials + window_size)
    for ax, subject_id in zip(axes.flat[:5], subjects):
        values = trial_data[trial_data["subject_id"] == subject_id]
        row = summary.loc[summary["subject_id"] == subject_id].iloc[0]
        x = values["trial"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            values["selected_behavioral_q05"].to_numpy(dtype=float),
            values["selected_behavioral_q95"].to_numpy(dtype=float),
            color=BAND_90_COLOR,
            alpha=0.55,
            linewidth=0,
            zorder=1,
        )
        ax.fill_between(
            x,
            values["selected_behavioral_q25"].to_numpy(dtype=float),
            values["selected_behavioral_q75"].to_numpy(dtype=float),
            color=BAND_50_COLOR,
            alpha=0.62,
            linewidth=0,
            zorder=2,
        )
        ax.axhline(0.5, color="#A8A8A8", linewidth=0.9, linestyle=(0, (2, 2)), zorder=0)
        ax.axvline(
            train_trials + 0.5,
            color=SPLIT_COLOR,
            linewidth=1.0,
            linestyle="--",
            zorder=3,
        )
        ax.axvline(
            fully_heldout_end,
            color="#B0B0B0",
            linewidth=0.9,
            linestyle=":",
            zorder=3,
        )
        ax.plot(
            x,
            values["reactive_expected_accuracy"],
            color=BOUNDARY_COLOR,
            linewidth=1.45,
            linestyle=(0, (4, 2)),
            zorder=4,
        )
        ax.plot(
            x,
            values["selected_expected_accuracy"],
            color=SELECTED_COLOR,
            linewidth=2.0,
            zorder=5,
        )
        ax.plot(
            x,
            values["observed_rolling_accuracy"],
            color=OBSERVED_COLOR,
            linewidth=2.1,
            zorder=6,
        )
        gain = float(row["accumulator_logit_gain"])
        ax.set_title(
            f"Subject {subject_id}  |  $c_{{acc}}$={gain:g}\n"
            f"90% coverage={float(row['selected_coverage_90']):.2f}; "
            f"curve MAE={float(row['selected_expected_curve_mae']):.3f}",
            fontsize=9.2,
            loc="left",
            pad=5,
        )
        ax.set_xlim(window_size + 0.5, 64.5)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xticks([17, 32, 48, 64])
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.tick_params(length=3, width=0.7)
        ax.grid(axis="y", color="#ECECEC", linewidth=0.7, zorder=0)

    for ax in axes[:, 0]:
        ax.set_ylabel("16-trial rolling accuracy")
    for ax in axes[1, :2]:
        ax.set_xlabel("Trial (window end)")

    legend_ax = axes[1, 2]
    legend_ax.axis("off")
    handles = [
        Line2D([0], [0], color=OBSERVED_COLOR, linewidth=2.1, label="Subject"),
        Line2D([0], [0], color=SELECTED_COLOR, linewidth=2.0, label="Selected H4 expectation"),
        Line2D(
            [0],
            [0],
            color=BOUNDARY_COLOR,
            linewidth=1.45,
            linestyle=(0, (4, 2)),
            label="Reactive boundary expectation",
        ),
        Patch(facecolor=BAND_50_COLOR, alpha=0.62, label="Selected H4 50% behavioral PI"),
        Patch(facecolor=BAND_90_COLOR, alpha=0.55, label="Selected H4 90% behavioral PI"),
        Line2D([0], [0], color=SPLIT_COLOR, linewidth=1.0, linestyle="--", label="Train/held-out split"),
        Line2D(
            [0],
            [0],
            color="#B0B0B0",
            linewidth=0.9,
            linestyle=":",
            label="First fully held-out window",
        ),
    ]
    legend_ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=8.2,
        handlelength=2.8,
        borderaxespad=0.0,
        labelspacing=0.75,
    )
    legend_ax.text(
        0.0,
        0.28,
        f"Band definition\n{n_draws:,} Bernoulli draws from the\n"
        "PF-averaged P(correct), conditional\non the observed choice/feedback history.\n\n"
        "Intervals exclude parameter uncertainty\nand are not autonomous rollouts.",
        transform=legend_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color="#4D4D4D",
        linespacing=1.35,
    )

    fig.suptitle(
        "Model 0815 H4: conditional predictive accuracy bands",
        fontsize=12.0,
        fontweight="bold",
        x=0.06,
        ha="left",
        y=0.985,
    )
    fig.text(
        0.06,
        0.943,
        "First 64 trials; parameters selected on trials 1–32; trial 1 is an initialization trial",
        fontsize=8.5,
        color="#4D4D4D",
        ha="left",
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.08, top=0.88, wspace=0.25, hspace=0.42)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_readme(
    path: Path,
    *,
    figure_path: Path,
    trial_path: Path,
    summary_path: Path,
    window_size: int,
    n_draws: int,
    train_trials: int,
) -> None:
    lines = [
        "# Model 0815 H4 accuracy bands",
        "",
        "This package visualizes the five subjects from the nested-accumulator screen without refitting the model.",
        "",
        "## Definition",
        "",
        (
            f"Observed and expected curves use a trailing {window_size}-trial window. "
            "Trial 1 is excluded because the PF treats it as initialization."
        ),
        (
            f"The selected-model 50% and 90% bands use {n_draws:,} Bernoulli draws "
            "from the PF-run-averaged probability of the task-correct category."
        ),
        "They are observed-history-conditional behavioral predictive intervals, not PF seed ranges, autonomous rollouts, or parameter-uncertainty intervals.",
        "",
        "## Split markers",
        "",
        f"The dashed marker is the parameter-selection split after trial {train_trials}.",
        (
            f"Because the curve is trailing, the first window containing only held-out trials "
            f"ends at trial {train_trials + window_size}; this is marked by the dotted line."
        ),
        "",
        "## Files",
        "",
        f"- `{figure_path.name}`: five-panel PNG figure.",
        f"- `{trial_path.name}`: plotted rolling curves and interval bounds.",
        f"- `{summary_path.name}`: coverage, width, MAE, and selected-versus-reactive trajectory differences.",
        "",
        "Only a PNG is emitted, following the repository artifact policy for exploratory scientific plots.",
        "",
        "The plotted band uses the selected H4 model. The reactive boundary is shown as an expectation line only; it is identical to the selected model whenever c_acc=0.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    data_path = args.data_path.resolve()
    window_size = int(args.window_size)
    n_draws = int(args.behavioral_draws)
    if window_size <= 0:
        raise ValueError("window-size must be positive")
    if n_draws < 2:
        raise ValueError("behavioral-draws must be at least 2")

    figure_path = output_dir / FIGURE_NAME
    trial_path = output_dir / TRIAL_DATA_NAME
    summary_path = output_dir / SUMMARY_NAME
    readme_path = output_dir / README_NAME
    targets = (figure_path, trial_path, summary_path, readme_path)
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {names}")

    manifest = _load_manifest(input_dir)
    train_trials = int(manifest["design"]["train_trials"])
    subjects = [int(value) for value in manifest["design"]["subjects"]]
    if len(subjects) != 5:
        raise ValueError(f"Expected five screened subjects, found {subjects}")
    index = _run_index(manifest)
    data = pd.read_csv(data_path, encoding="utf-8-sig")
    selection = pd.read_csv(input_dir / "subject_selection_summary.csv")
    contrast = pd.read_csv(input_dir / "contrast_summary.csv")
    if set(selection["subject_id"].astype(int)) != set(subjects):
        raise ValueError("Subject selection summary and manifest disagree")
    if set(contrast["subject_id"].astype(int)) != set(subjects):
        raise ValueError("Contrast summary and manifest disagree")

    trial_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for subject_id in subjects:
        selection_row = selection.loc[selection["subject_id"] == subject_id].iloc[0]
        contrast_row = contrast.loc[contrast["subject_id"] == subject_id].iloc[0]
        trial_frame, summary_row = _build_subject_band(
            subject_id=subject_id,
            manifest=manifest,
            index=index,
            data=data,
            selection=selection_row,
            contrast=contrast_row,
            window_size=window_size,
            n_draws=n_draws,
            base_seed=int(args.behavioral_seed),
        )
        trial_frames.append(trial_frame)
        summary_rows.append(summary_row)

    trial_data = pd.concat(trial_frames, ignore_index=True)
    summary_frame = pd.DataFrame(summary_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_data.to_csv(trial_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    _plot(
        trial_data,
        summary_frame,
        train_trials=train_trials,
        window_size=window_size,
        n_draws=n_draws,
        output_path=figure_path,
    )
    _write_readme(
        readme_path,
        figure_path=figure_path,
        trial_path=trial_path,
        summary_path=summary_path,
        window_size=window_size,
        n_draws=n_draws,
        train_trials=train_trials,
    )
    print(summary_frame.to_string(index=False))
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
