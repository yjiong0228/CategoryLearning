#!/usr/bin/env python3
"""Plot subject-specific outer-holdout fits for the model_0805 comparison.

Figure contract
---------------
Core conclusion:
    Complex-model performance is heterogeneous across subjects, but the
    full-set baseline has lower outer-holdout NLL for most subjects.
Figure archetype:
    Quantitative grid with a delta heatmap as the hero panel.
Target/output:
    Full-page diagnostic figure; editable SVG/PDF plus high-resolution PNG.
Panel map:
    a, geometric-mean probability assigned to observed choices for every
       subject-model pair (an intuitive transform of NLL/trial);
    b, subject-specific NLL change relative to FS_H0 for every finite model;
    c, direct paired comparison of FS_H0 with the group-best finite model.
Integrity:
    All 32 subjects and all seven final models are retained.  No rows are
    excluded and all panels use the same prespecified subject ordering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    ROOT
    / "results/zhuran/model_0805_cond1/real_predictive_overnight_20260805_v1"
    / "outer_holdout_subjects.csv"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "figures"

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
    "FS_H0": "FS_H0\n全假设集",
    "FA2_M3": "FA2\nM=3",
    "FA2_M5": "FA2\nM=5",
    "FA2_M7": "FA2\nM=7",
    "FA2R_M3": "FA2-R\nM=3",
    "FA2R_M5": "FA2-R\nM=5",
    "FA2R_M7": "FA2-R\nM=7",
}

BASELINE = "FS_H0"
GROUP_BEST_FINITE = "FA2_M3"


def apply_style() -> None:
    # Mandatory editable-text settings for the primary SVG output.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        # The installed Noto CJK collection is registered by Matplotlib under
        # its JP family name, while still containing Simplified-Chinese glyphs.
        "Noto Sans CJK JP",
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
    ]
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams['pdf.fonttype'] = 42
    # Mapping form is kept as well so static figure-audit tools can verify the
    # editable-text contract without executing the script.
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.linewidth"] = 0.7
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["xtick.major.width"] = 0.7
    plt.rcParams["ytick.major.width"] = 0.7


def load_and_validate(path: Path) -> tuple[pd.DataFrame, list[int]]:
    frame = pd.read_csv(path)
    required = {"subject_id", "model_key", "nll_per_trial", "n_trials"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Required figure data contain missing values")
    unknown = sorted(set(frame["model_key"]) - set(MODEL_ORDER))
    absent = sorted(set(MODEL_ORDER) - set(frame["model_key"]))
    if unknown or absent:
        raise ValueError(f"Model mismatch; unknown={unknown}, absent={absent}")
    if frame.duplicated(["subject_id", "model_key"]).any():
        raise ValueError("Duplicate subject-model rows")
    counts = frame.groupby("model_key")["subject_id"].nunique()
    if counts.nunique() != 1 or int(counts.iloc[0]) != 32:
        raise ValueError(f"Expected 32 subjects per model; got {counts.to_dict()}")

    wide = frame.pivot(index="subject_id", columns="model_key", values="nll_per_trial")
    # Sort on a prespecified group-best finite model comparison, not on a
    # per-subject oracle selected after seeing the holdout.
    delta = wide[GROUP_BEST_FINITE] - wide[BASELINE]
    subject_order = delta.sort_values(kind="mergesort").index.astype(int).tolist()
    return frame, subject_order


def annotate_heatmap(
    ax: mpl.axes.Axes,
    matrix: np.ndarray,
    image: mpl.image.AxesImage,
    formatter,
    fontsize: float,
) -> None:
    norm = image.norm
    cmap = image.cmap
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            rgba = cmap(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                col,
                row,
                formatter(value),
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white" if luminance < 0.52 else "#202020",
            )


def add_panel_label(ax: mpl.axes.Axes, label: str, y: float = 1.15) -> None:
    ax.text(
        -0.10,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )


def make_figure(frame: pd.DataFrame, subjects: list[int], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    nll = (
        frame.pivot(index="subject_id", columns="model_key", values="nll_per_trial")
        .loc[subjects, MODEL_ORDER]
    )
    # exp(-mean NLL) is the geometric mean probability assigned to the observed
    # choices.  This is more intuitive than NLL while retaining a one-to-one
    # transformation of the primary metric.
    observed_probability = np.exp(-nll) * 100.0
    finite_models = MODEL_ORDER[1:]
    delta = nll[finite_models].subtract(nll[BASELINE], axis=0)

    # Source data expose exactly what was drawn and the common subject order.
    source = frame.copy()
    source["geometric_mean_observed_choice_probability_pct"] = (
        np.exp(-source["nll_per_trial"]) * 100.0
    )
    source["delta_nll_vs_FS_H0"] = source.apply(
        lambda row: (
            row["nll_per_trial"]
            - float(nll.loc[int(row["subject_id"]), BASELINE])
        ),
        axis=1,
    )
    rank = {subject: index + 1 for index, subject in enumerate(subjects)}
    source["display_order"] = source["subject_id"].map(rank)
    source.sort_values(["display_order", "model_key"]).to_csv(
        output / "subject_level_model_fit_source_data.csv", index=False
    )

    probability_cmap = LinearSegmentedColormap.from_list(
        "fit_probability", ["#F4E7E3", "#D8E7EF", "#3775BA", "#17365D"]
    )
    delta_cmap = LinearSegmentedColormap.from_list(
        "finite_delta", ["#2F6B9A", "#D7E7F2", "#FAFAFA", "#F3D6D2", "#B64342"]
    )
    delta_limit = float(np.max(np.abs(delta.to_numpy())))

    fig = plt.figure(figsize=(15.2, 12.0), facecolor="white")
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.18, 1.02, 1.28],
        left=0.055,
        right=0.985,
        bottom=0.075,
        top=0.82,
        wspace=0.32,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[0, 2])

    # a | Absolute subject-model fit.
    prob_values = observed_probability.to_numpy()
    prob_norm = mpl.colors.Normalize(
        vmin=float(np.floor(prob_values.min() / 5.0) * 5.0),
        vmax=float(np.ceil(prob_values.max() / 5.0) * 5.0),
    )
    im_a = ax_a.imshow(prob_values, cmap=probability_cmap, norm=prob_norm, aspect="auto")
    annotate_heatmap(ax_a, prob_values, im_a, lambda value: f"{value:.1f}", 5.5)
    ax_a.set_xticks(np.arange(len(MODEL_ORDER)))
    ax_a.set_xticklabels([MODEL_LABELS[item] for item in MODEL_ORDER])
    ax_a.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    ax_a.set_yticks(np.arange(len(subjects)))
    ax_a.set_yticklabels([str(item) for item in subjects])
    ax_a.tick_params(axis="y", length=0, pad=3)
    ax_a.set_ylabel("被试编号")
    ax_a.set_title(
        "给真实选择的预测概率（%）\n越高表示拟合越好",
        y=1.115,
        pad=0,
        fontsize=10,
    )
    for spine in ax_a.spines.values():
        spine.set_visible(False)
    cbar_a = fig.colorbar(im_a, ax=ax_a, orientation="horizontal", pad=0.018, fraction=0.035)
    cbar_a.set_label("真实选择的几何平均预测概率（%）", labelpad=3)
    cbar_a.ax.tick_params(labelsize=7, length=2)
    add_panel_label(ax_a, "a")

    # b | Hero delta heatmap.
    delta_values = delta.to_numpy()
    im_b = ax_b.imshow(
        delta_values,
        cmap=delta_cmap,
        norm=TwoSlopeNorm(vmin=-delta_limit, vcenter=0.0, vmax=delta_limit),
        aspect="auto",
    )
    annotate_heatmap(ax_b, delta_values, im_b, lambda value: f"{value:+.2f}", 5.6)
    ax_b.set_xticks(np.arange(len(finite_models)))
    ax_b.set_xticklabels([MODEL_LABELS[item] for item in finite_models])
    ax_b.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    ax_b.set_yticks(np.arange(len(subjects)))
    ax_b.set_yticklabels([str(item) for item in subjects])
    ax_b.tick_params(axis="y", length=0, pad=3)
    ax_b.set_ylabel("被试编号")
    ax_b.set_title(
        "相对 FS_H0 的 NLL 变化\n蓝色：复杂模型更好；红色：复杂模型更差",
        y=1.115,
        pad=0,
        fontsize=10,
    )
    for spine in ax_b.spines.values():
        spine.set_visible(False)
    cbar_b = fig.colorbar(im_b, ax=ax_b, orientation="horizontal", pad=0.018, fraction=0.035)
    cbar_b.set_label("ΔNLL/试次（复杂模型 − FS_H0）", labelpad=3)
    cbar_b.ax.tick_params(labelsize=7, length=2)
    add_panel_label(ax_b, "b")

    # c | Direct paired display for the group-best finite model.
    y = np.arange(len(subjects))
    baseline_values = nll[BASELINE].to_numpy()
    finite_values = nll[GROUP_BEST_FINITE].to_numpy()
    improved = finite_values < baseline_values
    for yi, base, finite, is_improved in zip(y, baseline_values, finite_values, improved):
        color = "#2F6B9A" if is_improved else "#B64342"
        ax_c.plot([base, finite], [yi, yi], color=color, alpha=0.60, lw=1.1, zorder=1)
    ax_c.scatter(
        baseline_values,
        y,
        s=24,
        color="#333333",
        marker="o",
        label="FS_H0",
        zorder=3,
    )
    ax_c.scatter(
        finite_values,
        y,
        s=27,
        facecolor="white",
        edgecolor="#B64342",
        linewidth=1.0,
        marker="s",
        label="FA2, M=3",
        zorder=4,
    )
    # Recolor the rare improvements in blue without changing the model symbol.
    ax_c.scatter(
        finite_values[improved],
        y[improved],
        s=27,
        facecolor="white",
        edgecolor="#2F6B9A",
        linewidth=1.1,
        marker="s",
        zorder=5,
    )
    ax_c.set_yticks(y)
    ax_c.set_yticklabels([str(item) for item in subjects])
    ax_c.set_ylim(len(subjects) - 0.5, -0.5)
    ax_c.set_xlabel("外层留出段 NLL/试次（越低越好）")
    ax_c.set_ylabel("被试编号")
    ax_c.grid(axis="x", color="#DDDDDD", linewidth=0.6, zorder=0)
    ax_c.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.055),
        ncol=2,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    ax_c.set_title(
        f"FS_H0 与组水平最优复杂模型的逐被试比较\n"
        f"FA2, M=3 改善 {int(improved.sum())}/32 名被试",
        y=1.055,
        pad=0,
        fontsize=10,
    )
    add_panel_label(ax_c, "c", y=1.09)

    # Same ordering across panels; top rows are the subjects most favorable to
    # FA2_M3, and the boundary shows where the sign changes.
    improved_count = int(improved.sum())
    if 0 < improved_count < len(subjects):
        boundary = improved_count - 0.5
        for ax in (ax_a, ax_b, ax_c):
            ax.axhline(boundary, color="#202020", lw=0.8, ls=(0, (3, 2)), alpha=0.8)
        ax_c.text(
            0.995,
            boundary - 0.2,
            "上方：FA2, M=3 更好",
            transform=ax_c.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7,
            color="#2F6B9A",
        )

    fig.suptitle(
        "cond1：每名被试在外层留出段上的模型拟合",
        x=0.055,
        y=0.982,
        ha="left",
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.955,
        "每行是一名被试；三个面板使用完全相同的被试顺序。排序依据为 FA2 (M=3) − FS_H0 的 NLL 差。",
        ha="left",
        va="top",
        fontsize=8.5,
        color="#4D4D4D",
    )
    fig.text(
        0.055,
        0.025,
        "注：面板 a 的百分比为 exp(−NLL/试次)，即模型分配给真实选择的几何平均概率；"
        "面板 b 中负值表示复杂模型更好。31名被试有64个外层留出试次；被试105有16个。",
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#4D4D4D",
    )

    base = output / "subject_level_model_fit"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    frame, subjects = load_and_validate(args.input)
    make_figure(frame, subjects, args.output)
    print(f"Saved subject-level model-fit figure to {args.output}")


if __name__ == "__main__":
    main()
