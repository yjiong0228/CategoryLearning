#!/usr/bin/env python3
"""Create the frozen-model phase-coverage summary and individual atlas.

The plotting layer is intentionally read-only: it consumes the already frozen
phase analysis and rolling-curve summaries and performs no refitting, tuning,
or simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS = (
    ROOT / "results/zhuran/cond1_active_set/phase_coverage_frozen_models"
)
DEFAULT_C1 = (
    ROOT
    / "results/zhuran/cond1_active_set/dynamic_rho_reserved_c1_p256_r1024"
    / "c1_s0p5_e0p5_v0p2_p0p95/rolling_curve_summary.csv"
)
DEFAULT_ACQUISITION = (
    ROOT
    / "results/zhuran/cond1_active_set/acquisition_changepoint_reserved_h128_p256_r1024"
    / "rolling_curve_summary.csv"
)

C1_COLOR = "#3568A8"
ACQ_COLOR = "#D28E2F"
OBS_COLOR = "#252525"
MUTED = "#777777"
PASS_COLOR = "#E7EEF6"
FAIL_COLOR = "#B95C50"


def set_style() -> None:
    cjk_regular = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    cjk_bold = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
    for font_path in (cjk_regular, cjk_bold):
        if font_path.exists():
            font_manager.fontManager.addfont(font_path)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                # Matplotlib exposes the multilingual TTC under its JP family
                # name even though the SC glyph set is present.
                "Noto Sans CJK JP",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.14,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
    )


def add_panel_explanation(ax: plt.Axes, text: str) -> None:
    ax.text(
        0,
        -0.27,
        text,
        transform=ax.transAxes,
        fontsize=6.5,
        color="#4A4A4A",
        va="top",
        linespacing=1.25,
    )


def load_inputs(
    analysis_dir: Path, c1_csv: Path, acquisition_csv: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    subject = pd.read_csv(analysis_dir / "subject_phase_summary.csv")
    prevalence = pd.read_csv(analysis_dir / "observed_phase_prevalence.csv")
    residual = pd.read_csv(analysis_dir / "shared_residual_summary.csv")
    decision = json.loads((analysis_dir / "phase_decision.json").read_text())
    c1_curve = pd.read_csv(c1_csv)
    acq_curve = pd.read_csv(acquisition_csv)
    return subject, prevalence, residual, decision, c1_curve, acq_curve


def make_summary_figure(
    subject: pd.DataFrame,
    prevalence: pd.DataFrame,
    residual: pd.DataFrame,
    decision: dict,
    out_dir: Path,
) -> None:
    fig = plt.figure(figsize=(7.1, 6.8), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        2,
        left=0.09,
        right=0.98,
        bottom=0.09,
        top=0.86,
        hspace=0.62,
        wspace=0.36,
    )

    # a: observed cohort coverage against each generator's self-calibration.
    ax = fig.add_subplot(grid[0, 0])
    calibration = pd.DataFrame(decision["cohort_calibration"])
    x = np.arange(len(calibration))
    expected = calibration["self_expected_pass_mean"].to_numpy()
    lower = calibration["self_expected_pass_q025"].to_numpy()
    upper = calibration["self_expected_pass_q975"].to_numpy()
    observed = calibration["observed_pass_n"].to_numpy()
    colors = [C1_COLOR, ACQ_COLOR]
    ax.errorbar(
        x,
        expected,
        yerr=np.vstack([expected - lower, upper - expected]),
        fmt="o",
        color=MUTED,
        ecolor=MUTED,
        capsize=4,
        lw=1.1,
        ms=4,
        label="模型自校准期望（95%区间）",
        zorder=2,
    )
    for xi, obs, color in zip(x, observed, colors):
        ax.scatter(
            xi,
            obs,
            s=56,
            marker="D",
            color=color,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax.text(xi, obs + 0.23, f"{obs}/24", color=color, ha="center", fontsize=8)
    ax.set_xticks(x, ["动态 C1", "单次掌握变点"])
    ax.set_ylim(20.4, 24.7)
    ax.set_yticks([21, 22, 23, 24])
    ax.set_ylabel("通过联合阶段检验的被试数")
    ax.set_title("两种冻结模型都达到自身预期覆盖水平", loc="left")
    ax.legend(frameon=False, loc="lower left", handlelength=1.4)
    panel_label(ax, "a")
    add_panel_explanation(
        ax,
        "菱形为真实轨迹；灰点与误差线为用模型自身模拟轨迹得到的通过数分布。",
    )

    # b: paired proper score.
    ax = fig.add_subplot(grid[0, 1])
    xscore = subject["acquisition_phase_scaled_crps"].to_numpy()
    yscore = subject["C1_phase_scaled_crps"].to_numpy()
    patterns = np.select(
        [
            subject["C1_phase_pass_95"] & subject["acquisition_phase_pass_95"],
            subject["C1_phase_pass_95"] & ~subject["acquisition_phase_pass_95"],
            ~subject["C1_phase_pass_95"] & subject["acquisition_phase_pass_95"],
        ],
        ["both", "C1 only", "acquisition only"],
        default="neither",
    )
    pattern_color = {
        "both": "#8C8C8C",
        "C1 only": C1_COLOR,
        "acquisition only": ACQ_COLOR,
        "neither": FAIL_COLOR,
    }
    for pattern in ["both", "C1 only", "acquisition only", "neither"]:
        mask = patterns == pattern
        if not mask.any():
            continue
        label = {
            "both": "两者均通过",
            "C1 only": "仅 C1 通过",
            "acquisition only": "仅变点通过",
            "neither": "两者均未通过",
        }[pattern]
        ax.scatter(
            xscore[mask],
            yscore[mask],
            s=30,
            color=pattern_color[pattern],
            edgecolor="white",
            linewidth=0.5,
            label=label,
            zorder=3,
        )
    lim = (0, max(xscore.max(), yscore.max()) * 1.08)
    ax.plot(lim, lim, color="#B0B0B0", lw=0.9, ls="--", zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("单次掌握变点：阶段 CRPS（越低越好）")
    ax.set_ylabel("动态 C1：阶段 CRPS（越低越好）")
    ax.set_title("阶段预测精度没有可靠的总体胜者", loc="left")
    ax.legend(frameon=False, loc="upper left")
    panel_label(ax, "b")
    add_panel_explanation(
        ax,
        "点在虚线下方表示 C1 更好。平均配对差为 +0.0045，95% CI [−0.0468, 0.0551]。",
    )

    # c: individual joint coverage status.
    ax = fig.add_subplot(grid[1, 0])
    ordered = subject.sort_values(
        ["C1_phase_pass_95", "acquisition_phase_pass_95", "iSub"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    coverage = ordered[["C1_phase_pass_95", "acquisition_phase_pass_95"]].astype(int)
    ax.imshow(
        coverage.to_numpy(),
        aspect="auto",
        interpolation="none",
        cmap=ListedColormap([FAIL_COLOR, PASS_COLOR]),
        vmin=0,
        vmax=1,
    )
    ax.set_xticks([0, 1], ["动态 C1", "单次掌握变点"])
    ax.set_yticks(np.arange(len(ordered)), ordered["iSub"].astype(str))
    ax.set_ylabel("被试")
    ax.set_title("21 人两者均通过；没有人两者均失败", loc="left")
    for i in range(len(ordered)):
        for j in range(2):
            is_pass = bool(coverage.iloc[i, j])
            ax.text(
                j,
                i,
                "✓" if is_pass else "×",
                ha="center",
                va="center",
                color="#31506F" if is_pass else "white",
                fontsize=7,
                fontweight="bold",
            )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    panel_label(ax, "c")
    add_panel_explanation(
        ax,
        "联合判据同时检查 3 个窗口尺度 × 10 个阶段描述量；红格表示轨迹较该模型极端。",
    )

    # d: observed phase prevalence and extension gate.
    ax = fig.add_subplot(grid[1, 1])
    phase_order = ["混乱", "陡升", "陡降", "渐变", "恢复", "稳定高水平"]
    p = prevalence.set_index("phase").reindex(phase_order)
    y = np.arange(len(p))
    ax.barh(y, p["subject_fraction"] * 100, color="#667C8A", height=0.62)
    for yi, n, fraction in zip(y, p["subject_n"], p["subject_fraction"]):
        ax.text(
            fraction * 100 + 1.5,
            yi,
            f"{int(n)}/24",
            va="center",
            fontsize=7,
        )
    ax.set_yticks(y, p.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("出现该可观察阶段特征的被试比例（%）")
    ax.set_title("复杂阶段广泛存在，但没有共同未解释残差", loc="left")
    supported = (
        int(residual["cross_window_supported_subject_n"].max())
        if len(residual)
        else 0
    )
    ax.text(
        0.98,
        0.05,
        f"跨窗口共同残差：最多 {supported} 人\n预先设定的扩展门槛：≥4 人",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#4A4A4A",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#F4F4F2", "ec": "none"},
    )
    panel_label(ax, "d")
    add_panel_explanation(
        ax,
        "这些词是对正确率轨迹的描述，不等同于已识别出的潜在认知状态。",
    )

    fig.suptitle(
        "冻结模型的阶段级后验预测检验：复杂轨迹已被覆盖，暂无新增机制依据",
        x=0.09,
        y=0.975,
        ha="left",
        fontsize=10.5,
        fontweight="bold",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / "phase_coverage_summary"
    fig.savefig(base.with_suffix(".png"), dpi=300)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def with_gap_rows(
    frame: pd.DataFrame,
    *,
    x_column: str = "test_position",
) -> pd.DataFrame:
    """Order curves chronologically and break them at block boundaries.

    ``iTrial`` resets when a subject enters a new experimental segment, so it
    is not a valid across-segment time axis.  ``test_position`` is the unique,
    monotonically increasing position within the autonomous prediction suffix.
    Missing positions arise because rolling windows are not allowed to cross
    block boundaries; NaN rows keep those boundaries visually disconnected.
    """
    if x_column not in frame.columns:
        raise KeyError(f"Missing chronological axis column: {x_column}")
    if frame[x_column].isna().any():
        raise ValueError(f"{x_column} contains missing values.")
    if frame[x_column].duplicated().any():
        raise ValueError(f"{x_column} must be unique within each subject.")

    frame = frame.sort_values(x_column).reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for i, row in frame.iterrows():
        if i and row[x_column] - frame.loc[i - 1, x_column] > 1:
            gap = pd.DataFrame(
                {
                    c: [np.nan]
                    for c in frame.columns
                }
            )
            gap[x_column] = (
                frame.loc[i - 1, x_column] + row[x_column]
            ) / 2
            pieces.append(gap)
        pieces.append(row.to_frame().T)
    return pd.concat(pieces, ignore_index=True)


def plot_subject(
    ax: plt.Axes,
    sid: int,
    subject_row: pd.Series,
    c1_curve: pd.DataFrame,
    acq_curve: pd.DataFrame,
) -> None:
    c1 = with_gap_rows(c1_curve[c1_curve["iSub"] == sid])
    acq = with_gap_rows(acq_curve[acq_curve["iSub"] == sid])
    x = c1["test_position"].to_numpy(float)
    acq_x = acq["test_position"].to_numpy(float)
    ax.fill_between(
        x,
        c1["pointwise_q025"].to_numpy(float),
        c1["pointwise_q975"].to_numpy(float),
        color=C1_COLOR,
        alpha=0.14,
        lw=0,
    )
    ax.fill_between(
        acq_x,
        acq["pointwise_q025"].to_numpy(float),
        acq["pointwise_q975"].to_numpy(float),
        color=ACQ_COLOR,
        alpha=0.13,
        lw=0,
    )
    ax.plot(
        x,
        c1["sim_median"].to_numpy(float),
        color=C1_COLOR,
        lw=0.9,
        label="动态 C1",
    )
    ax.plot(
        acq_x,
        acq["sim_median"].to_numpy(float),
        color=ACQ_COLOR,
        lw=0.9,
        label="单次掌握变点",
    )
    ax.plot(
        x,
        c1["observed_rolling_accuracy"].to_numpy(float),
        color=OBS_COLOR,
        lw=1.0,
        label="真实轨迹",
        zorder=4,
    )
    ax.axhline(0.5, color="#B5B5B5", lw=0.6, ls=":")
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.5, 1])
    c1_pass = "通过" if subject_row["C1_phase_pass_95"] else "未通过"
    acq_pass = "通过" if subject_row["acquisition_phase_pass_95"] else "未通过"
    ax.set_title(
        f"被试 {sid}｜{subject_row['phase_signature']}\n"
        f"C1 {c1_pass}；变点 {acq_pass}",
        loc="left",
        fontsize=7.4,
    )
    ax.set_xlabel("自主预测后缀位置")
    ax.set_ylabel("12 试次滚动正确率")


def make_individual_atlas(
    subject: pd.DataFrame,
    c1_curve: pd.DataFrame,
    acq_curve: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    subject = subject.sort_values("iSub").reset_index(drop=True)
    pdf_path = out_dir / "individual_phase_atlas.pdf"
    with PdfPages(pdf_path) as pdf:
        for page_i, start in enumerate(range(0, len(subject), 6), start=1):
            rows = subject.iloc[start : start + 6]
            fig, axes = plt.subplots(
                3,
                2,
                figsize=(11.7, 8.3),
                sharey=True,
                constrained_layout=False,
            )
            fig.subplots_adjust(
                left=0.07,
                right=0.985,
                bottom=0.09,
                top=0.91,
                hspace=0.60,
                wspace=0.18,
            )
            for ax, (_, row) in zip(axes.flat, rows.iterrows()):
                plot_subject(
                    ax,
                    int(row["iSub"]),
                    row,
                    c1_curve,
                    acq_curve,
                )
            for ax in axes.flat[len(rows) :]:
                ax.axis("off")
            handles, labels = axes.flat[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(0.985, 0.975),
                ncol=3,
                frameon=False,
            )
            fig.suptitle(
                f"冻结模型逐被试轨迹图谱（第 {page_i}/4 页）",
                x=0.07,
                y=0.975,
                ha="left",
                fontsize=12,
                fontweight="bold",
            )
            fig.text(
                0.07,
                0.025,
                "前 64 试次仅用于条件化；横轴按自主预测后缀的真实时间顺序排列，"
                "空白为 block 边界。实线为真实/模拟中位轨迹；色带为逐点 95% "
                "模拟区间。阶段标签仅描述可观察轨迹，不作为潜在状态判定。",
                fontsize=7,
                color="#4A4A4A",
            )
            pdf.savefig(fig, dpi=300)
            fig.savefig(out_dir / f"individual_phase_atlas_page{page_i}.png", dpi=300)
            plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--c1-curve", type=Path, default=DEFAULT_C1)
    parser.add_argument("--acquisition-curve", type=Path, default=DEFAULT_ACQUISITION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    subject, prevalence, residual, decision, c1_curve, acq_curve = load_inputs(
        args.analysis_dir, args.c1_curve, args.acquisition_curve
    )
    out_dir = args.analysis_dir / "figures"
    make_summary_figure(subject, prevalence, residual, decision, out_dir)
    make_individual_atlas(subject, c1_curve, acq_curve, out_dir)
    print(f"Wrote phase summary and atlas to {out_dir}")


if __name__ == "__main__":
    main()
