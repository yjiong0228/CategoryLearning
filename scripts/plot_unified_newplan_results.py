#!/usr/bin/env python3
"""Create Chinese, publication-grade figures for the unified new-plan results.

The figures deliberately separate the confirmatory static-rule failure from the
adaptive working model.  All quantitative panels use held-out subject-level
observations or the frozen aggregate estimates already reported by the formal
analysis scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/zhuran/unified_newplan"
DEFAULT_OUTPUT = BASE / "figures_20260802"

COLORS = {
    "rule": "#0F4D92",
    "rule_light": "#B9D3EA",
    "feature": "#2A8C82",
    "feature_light": "#BFE1DC",
    "oral": "#8A5A9E",
    "oral_light": "#DDCDE5",
    "positive": "#2E8B57",
    "negative": "#B64342",
    "neutral": "#777777",
    "neutral_light": "#D6D6D6",
    "ink": "#20242A",
    "grid": "#E5E7EB",
    "paper": "#FFFFFF",
    "soft_blue": "#EEF4FA",
    "soft_teal": "#EDF7F5",
    "soft_violet": "#F5EFF7",
    "soft_grey": "#F4F5F6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            json_ready(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def configure_style() -> str:
    chinese_font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not chinese_font.exists():
        candidates = list(Path("/usr/share/fonts").rglob("NotoSansCJK*.ttc"))
        if not candidates:
            raise RuntimeError("No CJK font found; cannot render Chinese labels safely")
        chinese_font = candidates[0]
    font_manager.fontManager.addfont(str(chinese_font))
    family = font_manager.FontProperties(fname=str(chinese_font)).get_name()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "Arial", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return family


def save_figure(fig: plt.Figure, output: Path, stem: str, dpi: int) -> list[Path]:
    svg_path = output / f"{stem}.svg"
    pdf_path = output / f"{stem}.pdf"
    png_path = output / f"{stem}.png"
    tiff_path = output / f"{stem}.tiff"
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(png_path, dpi=max(dpi, 300), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(
        tiff_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    paths = [svg_path, pdf_path, png_path, tiff_path]
    plt.close(fig)
    return paths


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="left",
        va="bottom",
        color=COLORS["ink"],
    )


def soften_axes(ax: plt.Axes, axis: str = "x") -> None:
    ax.grid(axis=axis, color=COLORS["grid"], linewidth=0.7, zorder=0)
    ax.tick_params(length=3, width=0.7, color="#777777")
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")


def deterministic_jitter(n: int, width: float = 0.16) -> np.ndarray:
    """Stable display-only offsets; does not sample or omit observations."""
    index = np.arange(n, dtype=float)
    return width * np.sin(index * math.pi * (3.0 - math.sqrt(5.0)))


def rounded_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    facecolor: str,
    edgecolor: str,
    radius: float = 0.03,
    linewidth: float = 1.2,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.add_patch(patch)
    return patch


def draw_card(
    ax: plt.Axes,
    title: str,
    verdict: str,
    metric: str,
    explanation: str,
    color: str,
    facecolor: str,
    verdict_positive: bool,
) -> None:
    ax.set_axis_off()
    rounded_box(ax, (0.02, 0.03), 0.96, 0.94, facecolor, color, radius=0.04)
    ax.text(0.08, 0.83, title, transform=ax.transAxes, fontsize=15, fontweight="bold", color=color)
    pill_color = COLORS["positive"] if verdict_positive else COLORS["negative"]
    rounded_box(ax, (0.62, 0.79), 0.28, 0.11, "white", pill_color, radius=0.06)
    ax.text(
        0.76,
        0.845,
        verdict,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.5,
        fontweight="bold",
        color=pill_color,
    )
    ax.text(0.08, 0.55, metric, transform=ax.transAxes, fontsize=20, fontweight="bold", color=COLORS["ink"])
    ax.text(
        0.08,
        0.35,
        explanation,
        transform=ax.transAxes,
        fontsize=9.2,
        color="#3B4149",
        ha="left",
        va="top",
        linespacing=1.5,
    )


def read_comparison(
    path: Path,
    comparison: str,
    conditions: tuple[str, ...] = ("1", "2", "3", "all"),
    **filters: object,
) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"condition": str})
    mask = frame["comparison"].eq(comparison)
    for column, value in filters.items():
        mask &= frame[column].eq(value)
    result = frame[mask & frame["condition"].isin(conditions)].copy()
    order = {condition: index for index, condition in enumerate(conditions)}
    result["condition_order"] = result["condition"].map(order)
    result = result.sort_values("condition_order").drop(columns="condition_order")
    if len(result) != len(conditions):
        raise ValueError(f"Expected {len(conditions)} rows for {comparison} in {path}")
    return result


def paired_metric_delta(
    frame: pd.DataFrame,
    candidate: str,
    reference: str,
    metric: str,
    positive_when_lower: bool,
) -> pd.DataFrame:
    subset = frame[
        frame["model"].isin([candidate, reference]) & frame["segment"].eq("holdout")
    ]
    pivot = subset.pivot(index=["subject_id", "condition"], columns="model", values=metric)
    if pivot[[candidate, reference]].isna().any().any():
        raise ValueError(f"Missing paired {metric} values for {candidate} versus {reference}")
    if positive_when_lower:
        delta = pivot[reference] - pivot[candidate]
    else:
        delta = pivot[candidate] - pivot[reference]
    return delta.rename("delta").reset_index()


def draw_forest_pair(
    ax: plt.Axes,
    first: pd.DataFrame,
    second: pd.DataFrame,
    value_column: str,
    labels: tuple[str, str],
    colors: tuple[str, str],
    x_label: str,
    xlim: tuple[float, float],
) -> None:
    condition_labels = ["条件 1", "条件 2", "条件 3", "总体"]
    centers = np.arange(4)[::-1]
    offsets = (0.14, -0.14)
    for frame, label, color, offset in zip((first, second), labels, colors, offsets):
        for center, row in zip(centers, frame.itertuples(index=False)):
            estimate = getattr(row, value_column)
            low = row.bootstrap_mean_ci_low
            high = row.bootstrap_mean_ci_high
            ax.plot([low, high], [center + offset] * 2, color=color, lw=2.0, solid_capstyle="round", zorder=3)
            ax.scatter(estimate, center + offset, s=34, color=color, edgecolor="white", linewidth=0.6, zorder=4)
    ax.axvline(0, color="#555555", linestyle=(0, (3, 3)), linewidth=1.0, zorder=1)
    ax.set_yticks(centers)
    ax.set_yticklabels(condition_labels)
    ax.set_ylim(-0.6, 3.6)
    ax.set_xlim(*xlim)
    ax.set_xlabel(x_label)
    soften_axes(ax, "x")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color=color, lw=2, label=label, markersize=5)
            for label, color in zip(labels, colors)
        ],
        loc="lower right",
    )


def build_sources(output: Path) -> dict[str, Any]:
    core_comparison = read_comparison(
        BASE / "core_sobol512_20260802/model_comparisons.csv", "representation_gate"
    )
    joint_comparison = read_comparison(
        BASE / "joint_dynamic_nr2_20260802/model_comparisons.csv",
        "rule_vs_joint_dynamic_NR2",
    )
    dynamic_parameters = pd.read_csv(
        BASE / "dynamic_readout_20260802/parameters.csv", dtype={"condition": str}
    )
    shared = dynamic_parameters[
        dynamic_parameters["model"].eq("R0KT_GLOBAL")
        & dynamic_parameters["subject_id"].eq(-1)
    ]
    if len(shared) != 1:
        raise ValueError("Expected one R0KT_GLOBAL shared-slope row")
    shared = shared.iloc[0]

    core_subjects = pd.read_csv(BASE / "core_sobol512_20260802/subject_model_metrics.csv")
    joint_subjects = pd.read_csv(BASE / "joint_dynamic_nr2_20260802/subject_metrics.csv")
    static_delta = paired_metric_delta(
        core_subjects, "R_SELECT", "NR_SELECT", "nll_per_trial", True
    )
    dynamic_delta = paired_metric_delta(
        joint_subjects,
        "R0KT_GLOBAL",
        "NR2T_JOINT_INDIVIDUAL",
        "nll_per_trial",
        True,
    )
    accuracy_delta = paired_metric_delta(
        joint_subjects,
        "R0KT_GLOBAL",
        "NR2T_JOINT_INDIVIDUAL",
        "accuracy",
        False,
    )
    accuracy_delta["delta_percentage_points"] = 100 * accuracy_delta["delta"]

    practice = np.linspace(0.0, 1.0, 101)
    slope = float(shared["slope"])
    slope_se = float(shared["slope_se"])
    practice_curve = pd.DataFrame(
        {
            "normalized_practice": practice,
            "relative_readout": np.exp(slope * practice),
            "wald95_low": np.exp((slope - 1.96 * slope_se) * practice),
            "wald95_high": np.exp((slope + 1.96 * slope_se) * practice),
        }
    )

    rt_rule = read_comparison(
        BASE / "rt_external_validation_20260802/model_comparisons.csv",
        "rule_entropy_increment",
        qc_specification="main_qc",
    )
    rt_feature = read_comparison(
        BASE / "rt_external_validation_20260802/model_comparisons.csv",
        "nr_entropy_increment",
        qc_specification="main_qc",
    )
    oral_pure = read_comparison(
        BASE / "oral_external_validation_20260802/model_comparisons.csv",
        "rule_vs_selected_baseline",
    )
    oral_mixture = read_comparison(
        BASE / "oral_mixture_diagnostic_20260802/model_comparisons.csv",
        "global_mixture_vs_baseline",
    )
    mixture_manifest = json.loads(
        (BASE / "oral_mixture_diagnostic_20260802/manifest.json").read_text(encoding="utf-8")
    )

    representation_recovery = pd.read_csv(
        BASE / "representation_recovery_final100_20260802/model_recovery.csv",
        dtype={"condition": str},
    )
    representation_recovery = representation_recovery[
        representation_recovery["condition"].eq("all")
    ].copy()
    readout_recovery = pd.read_csv(
        BASE / "readout_recovery_final100_20260802/model_recovery.csv",
        dtype={"condition": str},
    )
    readout_recovery = readout_recovery[readout_recovery["condition"].eq("all")].copy()

    if len(static_delta) != 96 or len(dynamic_delta) != 96 or len(accuracy_delta) != 96:
        raise ValueError("Choice source data do not contain exactly 96 paired subjects")
    if len(representation_recovery) != 200 or len(readout_recovery) != 200:
        raise ValueError("Final recovery source data do not contain 200 runs")
    expected_representation = {
        "R0KT_GLOBAL": "R0KT_GLOBAL",
        "NR2T_JOINT_INDIVIDUAL": "NR2T_JOINT_INDIVIDUAL",
    }
    correct = representation_recovery.apply(
        lambda row: row["selected_model"] == expected_representation[row["generator"]],
        axis=1,
    )
    if not bool(correct.all()):
        raise ValueError("Representation-recovery labels are not 200/200 correct")
    expected_readout = {"R0K": "R0K", "R0KT_GLOBAL": "R0KT_GLOBAL"}
    correct = readout_recovery.apply(
        lambda row: row["selected_model"] == expected_readout[row["generator"]],
        axis=1,
    )
    if not bool(correct.all()):
        raise ValueError("Readout-recovery labels are not 200/200 correct")

    overview = pd.DataFrame(
        [
            {
                "channel": "choice",
                "comparison": "R0KT_GLOBAL vs joint dynamic NR2",
                "estimate": float(joint_comparison[joint_comparison["condition"].eq("all")]["mean_delta_nll_per_trial"].iloc[0]),
                "ci_low": float(joint_comparison[joint_comparison["condition"].eq("all")]["bootstrap_mean_ci_low"].iloc[0]),
                "ci_high": float(joint_comparison[joint_comparison["condition"].eq("all")]["bootstrap_mean_ci_high"].iloc[0]),
                "n_subjects": 96,
            },
            {
                "channel": "rt",
                "comparison": "dynamic NR2 entropy vs RT baseline",
                "estimate": float(rt_feature[rt_feature["condition"].eq("all")]["mean_delta_log_predictive_density"].iloc[0]),
                "ci_low": float(rt_feature[rt_feature["condition"].eq("all")]["bootstrap_mean_ci_low"].iloc[0]),
                "ci_high": float(rt_feature[rt_feature["condition"].eq("all")]["bootstrap_mean_ci_high"].iloc[0]),
                "n_subjects": 96,
            },
            {
                "channel": "oral",
                "comparison": "global measurement mixture vs oral baseline",
                "estimate": float(oral_mixture[oral_mixture["condition"].eq("all")]["mean_delta_log_score"].iloc[0]),
                "ci_low": float(oral_mixture[oral_mixture["condition"].eq("all")]["bootstrap_mean_ci_low"].iloc[0]),
                "ci_high": float(oral_mixture[oral_mixture["condition"].eq("all")]["bootstrap_mean_ci_high"].iloc[0]),
                "n_subjects": 95,
            },
        ]
    )
    overview["shared_readout_slope"] = slope
    overview["readout_end_factor"] = math.exp(slope)
    overview["oral_rule_weight"] = float(mixture_manifest["global_weight"])

    forest_choice = pd.concat(
        [
            core_comparison.assign(specification="原计划静态规则"),
            joint_comparison.assign(specification="练习依赖规则读出"),
        ],
        ignore_index=True,
    )
    forest_external = pd.concat(
        [
            rt_rule.assign(channel="RT", specification="规则熵"),
            rt_feature.assign(channel="RT", specification="特征学习熵"),
            oral_pure.assign(channel="口头报告", specification="纯规则读出"),
            oral_mixture.assign(channel="口头报告", specification="规则+习惯混合"),
        ],
        ignore_index=True,
    )

    atomic_csv(output / "source_data_fig1_overview.csv", overview)
    atomic_csv(output / "source_data_fig2_choice_forest.csv", forest_choice)
    atomic_csv(output / "source_data_fig2_subject_deltas.csv", dynamic_delta)
    atomic_csv(output / "source_data_fig2_accuracy_deltas.csv", accuracy_delta)
    atomic_csv(output / "source_data_fig2_practice_curve.csv", practice_curve)
    atomic_csv(output / "source_data_fig3_external_forest.csv", forest_external)
    atomic_csv(output / "source_data_fig3_representation_recovery.csv", representation_recovery)
    atomic_csv(output / "source_data_fig3_readout_recovery.csv", readout_recovery)

    return {
        "core_comparison": core_comparison,
        "joint_comparison": joint_comparison,
        "static_delta": static_delta,
        "dynamic_delta": dynamic_delta,
        "accuracy_delta": accuracy_delta,
        "practice_curve": practice_curve,
        "shared_slope": slope,
        "shared_slope_se": slope_se,
        "rt_rule": rt_rule,
        "rt_feature": rt_feature,
        "oral_pure": oral_pure,
        "oral_mixture": oral_mixture,
        "oral_rule_weight": float(mixture_manifest["global_weight"]),
        "representation_recovery": representation_recovery,
        "readout_recovery": readout_recovery,
    }


def figure_overview(data: dict[str, Any], output: Path, dpi: int) -> list[Path]:
    fig = plt.figure(figsize=(13.2, 7.8), constrained_layout=False)
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.08, 0.92],
        left=0.055,
        right=0.975,
        bottom=0.07,
        top=0.80,
        wspace=0.22,
        hspace=0.32,
    )
    fig.text(0.055, 0.965, "数据告诉我们的故事", fontsize=22, fontweight="bold", color=COLORS["ink"], va="top")
    fig.text(
        0.055,
        0.895,
        "选择逐渐规则化，但反应时间和口头报告不是同一规则状态的直接输出",
        fontsize=12,
        color="#4B5563",
        va="top",
    )

    ax_curve = fig.add_subplot(gs[0, 0])
    curve = data["practice_curve"]
    ax_curve.fill_between(
        curve["normalized_practice"].to_numpy(),
        curve["wald95_low"].to_numpy(),
        curve["wald95_high"].to_numpy(),
        color=COLORS["rule_light"],
        alpha=0.75,
        linewidth=0,
    )
    ax_curve.plot(
        curve["normalized_practice"],
        curve["relative_readout"],
        color=COLORS["rule"],
        linewidth=3.0,
    )
    ax_curve.scatter([0, 1], [1, math.exp(data["shared_slope"])], color=COLORS["rule"], s=42, zorder=3)
    ax_curve.text(0.02, 1.12, "1.0×", color=COLORS["rule"], fontweight="bold")
    ax_curve.text(
        0.98,
        math.exp(data["shared_slope"]) + 0.17,
        f"{math.exp(data['shared_slope']):.1f}×",
        color=COLORS["rule"],
        fontweight="bold",
        ha="right",
    )
    ax_curve.set_title("规则对选择的控制随练习增强", loc="left", fontweight="bold", pad=10)
    ax_curve.set_xlabel("训练进程")
    ax_curve.set_ylabel("相对读出强度")
    ax_curve.set_xticks([0, 0.5, 1])
    ax_curve.set_xticklabels(["开始", "中段", "训练末"])
    ax_curve.set_ylim(0.7, 4.65)
    soften_axes(ax_curve, "y")

    joint_all = data["joint_comparison"][data["joint_comparison"]["condition"].eq("all")].iloc[0]
    rt_all = data["rt_feature"][data["rt_feature"]["condition"].eq("all")].iloc[0]
    oral_all = data["oral_mixture"][data["oral_mixture"]["condition"].eq("all")].iloc[0]
    cards = [fig.add_subplot(gs[0, index]) for index in (1, 2, 3)]
    draw_card(
        cards[0],
        "选择",
        "总体支持",
        f"ΔNLL +{joint_all.mean_delta_nll_per_trial:.3f}",
        "动态规则模型优于\n充分调整的特征学习模型\n条件 2/3 明确，条件 1 不确定",
        COLORS["rule"],
        COLORS["soft_blue"],
        True,
    )
    draw_card(
        cards[1],
        "反应时间",
        "规则预测失败",
        f"ΔLPD +{rt_all.mean_delta_log_predictive_density:.3f}",
        "特征学习的不确定性\n能够改善留出 RT 预测\n规则不确定性没有改善",
        COLORS["feature"],
        COLORS["soft_teal"],
        False,
    )
    draw_card(
        cards[2],
        "口头报告",
        "探索性支持",
        f"{100 * data['oral_rule_weight']:.0f}% 规则信息",
        f"规则判断与报告习惯的混合\n留出改善 +{oral_all.mean_delta_log_score:.3f}\n纯规则直接读出未通过",
        COLORS["oral"],
        COLORS["soft_violet"],
        True,
    )

    ax_flow = fig.add_subplot(gs[1, :])
    ax_flow.set_axis_off()
    ax_flow.text(
        0.01,
        0.96,
        "最符合数据的工作模型：三条行为通道由不同成分主导",
        transform=ax_flow.transAxes,
        fontsize=12,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )
    sources = [
        ("规则状态", COLORS["soft_blue"], COLORS["rule"], 0.69),
        ("特征学习", COLORS["soft_teal"], COLORS["feature"], 0.39),
        ("报告习惯", COLORS["soft_grey"], COLORS["neutral"], 0.09),
    ]
    targets = [
        ("选择", "练习后更依赖规则", COLORS["soft_blue"], COLORS["rule"], 0.69),
        ("反应时间", "反映特征学习的不确定性", COLORS["soft_teal"], COLORS["feature"], 0.39),
        ("口头报告", "规则信息 + 表达习惯", COLORS["soft_violet"], COLORS["oral"], 0.09),
    ]
    for text, face, edge, y in sources:
        rounded_box(ax_flow, (0.04, y), 0.18, 0.17, face, edge, radius=0.035)
        ax_flow.text(0.13, y + 0.085, text, transform=ax_flow.transAxes, ha="center", va="center", fontsize=11, fontweight="bold", color=edge)
    for title, subtitle, face, edge, y in targets:
        rounded_box(ax_flow, (0.72, y), 0.24, 0.17, face, edge, radius=0.035)
        ax_flow.text(0.84, y + 0.112, title, transform=ax_flow.transAxes, ha="center", va="center", fontsize=11, fontweight="bold", color=edge)
        ax_flow.text(0.84, y + 0.052, subtitle, transform=ax_flow.transAxes, ha="center", va="center", fontsize=8.2, color="#4B5563")

    def arrow(start: tuple[float, float], end: tuple[float, float], color: str, width: float, alpha: float = 1.0) -> None:
        patch = FancyArrowPatch(
            start,
            end,
            transform=ax_flow.transAxes,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=width,
            color=color,
            alpha=alpha,
            connectionstyle="arc3,rad=0.0",
        )
        ax_flow.add_patch(patch)

    arrow((0.22, 0.775), (0.72, 0.775), COLORS["rule"], 3.0)
    arrow((0.22, 0.475), (0.72, 0.475), COLORS["feature"], 3.0)
    arrow((0.22, 0.775), (0.72, 0.175), COLORS["oral"], 1.6, 0.75)
    arrow((0.22, 0.175), (0.72, 0.175), COLORS["neutral"], 2.6)
    ax_flow.text(0.44, 0.84, "主要通道", transform=ax_flow.transAxes, fontsize=8, color=COLORS["rule"])
    ax_flow.text(0.44, 0.54, "主要通道", transform=ax_flow.transAxes, fontsize=8, color=COLORS["feature"])
    ax_flow.text(0.46, 0.16, "混合测量", transform=ax_flow.transAxes, fontsize=8, color=COLORS["oral"])
    fig.text(0.975, 0.018, "留出检验；n=96（口头报告 n=95）", ha="right", fontsize=7.5, color="#6B7280")
    return save_figure(fig, output, "fig1_chinese_overview", dpi)


def figure_choice(data: dict[str, Any], output: Path, dpi: int) -> list[Path]:
    fig = plt.figure(figsize=(12.0, 7.8))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.25, 1.0],
        height_ratios=[1, 1],
        left=0.085,
        right=0.97,
        bottom=0.095,
        top=0.81,
        hspace=0.42,
        wspace=0.32,
    )
    fig.suptitle("为什么最终保留练习依赖的规则选择模型", fontsize=17, fontweight="bold", y=0.965)
    fig.text(0.5, 0.905, "原计划静态规则失败；加入一个共享练习斜率后，留出选择预测整体转为正向", ha="center", fontsize=10, color="#4B5563")

    ax_a = fig.add_subplot(gs[:, 0])
    draw_forest_pair(
        ax_a,
        data["core_comparison"],
        data["joint_comparison"],
        "mean_delta_nll_per_trial",
        ("原计划静态规则 vs 非规则", "动态规则 vs 动态特征 RL"),
        (COLORS["neutral"], COLORS["rule"]),
        "留出优势 ΔNLL/试次（>0 表示规则模型更好）",
        (-0.125, 0.15),
    )
    ax_a.set_title("静态规则失败，动态读出在总体及条件 2/3 获得支持", loc="left", fontweight="bold", pad=12)
    panel_label(ax_a, "a", x=-0.16)
    ax_a.axvspan(-0.125, 0, color="#FAEEEE", alpha=0.55, zorder=-2)
    ax_a.axvspan(0, 0.15, color="#EEF7F1", alpha=0.65, zorder=-2)
    ax_a.text(-0.118, 3.46, "非规则更好", fontsize=7.5, color=COLORS["negative"], va="top")
    ax_a.text(0.143, 3.46, "规则更好", fontsize=7.5, color=COLORS["positive"], va="top", ha="right")

    ax_b = fig.add_subplot(gs[0, 1])
    curve = data["practice_curve"]
    ax_b.fill_between(
        curve["normalized_practice"].to_numpy(),
        curve["wald95_low"].to_numpy(),
        curve["wald95_high"].to_numpy(),
        color=COLORS["rule_light"],
        alpha=0.8,
        linewidth=0,
        label="Wald 95% 区间",
    )
    ax_b.plot(curve["normalized_practice"], curve["relative_readout"], color=COLORS["rule"], lw=2.8)
    ax_b.scatter([0, 1], [1, math.exp(data["shared_slope"])], color=COLORS["rule"], s=35, zorder=3)
    ax_b.annotate(
        f"训练末 {math.exp(data['shared_slope']):.2f}×",
        xy=(1, math.exp(data["shared_slope"])),
        xytext=(0.56, 3.45),
        arrowprops={"arrowstyle": "->", "color": COLORS["rule"], "lw": 1.2},
        color=COLORS["rule"],
        fontweight="bold",
    )
    ax_b.set_title("规则读出随练习增强", loc="left", fontweight="bold")
    ax_b.set_xlabel("归一化训练进程")
    ax_b.set_ylabel("相对规则读出强度")
    ax_b.set_xticks([0, 0.5, 1])
    ax_b.set_xticklabels(["开始", "中段", "末段"])
    ax_b.set_ylim(0.75, 4.65)
    soften_axes(ax_b, "y")
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 1])
    accuracy = data["accuracy_delta"].copy()
    positions = [1, 2, 3]
    accuracy_means: list[float] = []
    all_accuracy_values = accuracy["delta_percentage_points"].to_numpy()
    y_min = min(-2.0, float(all_accuracy_values.min()) - 1.5)
    y_max = float(all_accuracy_values.max()) + 3.0
    for position, condition in zip(positions, (1, 2, 3)):
        values = accuracy[accuracy["condition"].eq(condition)]["delta_percentage_points"].to_numpy()
        accuracy_means.append(float(values.mean()))
        violin = ax_c.violinplot(values, positions=[position], widths=0.7, showmeans=False, showmedians=False, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(COLORS["rule_light"])
            body.set_edgecolor(COLORS["rule"])
            body.set_alpha(0.75)
        jitter = deterministic_jitter(len(values), 0.16)
        ax_c.scatter(position + jitter, values, s=13, color=COLORS["rule"], alpha=0.48, linewidth=0, zorder=3)
        ax_c.scatter(position, values.mean(), s=62, marker="D", color=COLORS["rule"], edgecolor="white", linewidth=0.8, zorder=4)
    ax_c.set_ylim(y_min, y_max)
    for position, mean in zip(positions, accuracy_means):
        ax_c.text(position, y_max - 0.4, f"均值 {mean:+.1f}", ha="center", va="top", fontsize=7.2, color=COLORS["rule"])
    ax_c.axhline(0, color="#555555", linestyle=(0, (3, 3)), lw=1)
    ax_c.set_xticks(positions)
    ax_c.set_xticklabels(["条件 1", "条件 2", "条件 3"])
    ax_c.set_ylabel("规则模型准确率优势（百分点）")
    ax_c.set_title("被试级准确率差异", loc="left", fontweight="bold")
    soften_axes(ax_c, "y")
    panel_label(ax_c, "c")
    ax_c.text(0.99, 0.03, "每点=1名被试；菱形=均值", transform=ax_c.transAxes, ha="right", fontsize=7, color="#6B7280")

    fig.text(
        0.085,
        0.025,
        "a 均值及被试 bootstrap 95% CI，n=32/条件；b 共享斜率 b=1.433，Wald SE=0.025；c 使用全部 96 名被试的最后完整区块。",
        fontsize=7.5,
        color="#5F6772",
    )
    return save_figure(fig, output, "fig2_choice_model_evidence", dpi)


def figure_external_recovery(data: dict[str, Any], output: Path, dpi: int) -> list[Path]:
    fig = plt.figure(figsize=(12.3, 8.8))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.085,
        right=0.97,
        bottom=0.09,
        top=0.81,
        hspace=0.44,
        wspace=0.32,
    )
    fig.suptitle("外部检验与模型恢复：哪些结论成立，哪些不成立", fontsize=17, fontweight="bold", y=0.965)
    fig.text(0.5, 0.905, "RT 支持特征学习不确定性；纯规则口头读出失败；模拟数据中的模型身份可被稳定辨认", ha="center", fontsize=10, color="#4B5563")

    ax_a = fig.add_subplot(gs[0, 0])
    draw_forest_pair(
        ax_a,
        data["rt_rule"],
        data["rt_feature"],
        "mean_delta_log_predictive_density",
        ("规则熵", "特征学习熵"),
        (COLORS["negative"], COLORS["feature"]),
        "留出 RT 预测改善 ΔLPD/试次（>0 更好）",
        (-0.85, 0.22),
    )
    ax_a.set_title("RT：规则熵失败，特征学习熵总体为正", loc="left", fontweight="bold")
    panel_label(ax_a, "a", x=-0.16)

    ax_b = fig.add_subplot(gs[0, 1])
    draw_forest_pair(
        ax_b,
        data["oral_pure"],
        data["oral_mixture"],
        "mean_delta_log_score",
        ("纯规则直接读出（主检验）", "规则+报告习惯混合（探索性）"),
        (COLORS["neutral"], COLORS["oral"]),
        "留出口头报告改善 Δlog score（>0 更好）",
        (-2.3, 0.75),
    )
    ax_b.set_title("口头报告：纯规则不稳定，简单混合稳健改善", loc="left", fontweight="bold")
    panel_label(ax_b, "b", x=-0.16)
    ax_b.text(
        0.98,
        0.92,
        f"全局规则权重 = {100 * data['oral_rule_weight']:.1f}%",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color=COLORS["oral"],
        fontweight="bold",
    )

    ax_c = fig.add_subplot(gs[1, 0])
    recovery = data["representation_recovery"]
    generator_order = ["NR2T_JOINT_INDIVIDUAL", "R0KT_GLOBAL"]
    y_positions = [0, 1]
    recovery_colors = [COLORS["feature"], COLORS["rule"]]
    for y, generator, color in zip(y_positions, generator_order, recovery_colors):
        values = recovery[recovery["generator"].eq(generator)]["mean_delta_nll_per_trial_rule_vs_nr"].to_numpy()
        violin = ax_c.violinplot(values, positions=[y], vert=False, widths=0.72, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.24)
        jitter = deterministic_jitter(len(values), 0.13)
        ax_c.scatter(values, y + jitter, s=12, color=color, alpha=0.5, linewidth=0)
        ax_c.scatter(values.mean(), y, marker="D", s=60, color=color, edgecolor="white", linewidth=0.8, zorder=4)
    ax_c.axvline(0, color="#555555", linestyle=(0, (3, 3)), lw=1)
    ax_c.set_yticks(y_positions)
    ax_c.set_yticklabels(["特征 RL 生成", "动态规则生成"])
    ax_c.set_xlabel("重新拟合后的规则优势 ΔNLL/试次")
    ax_c.set_title("表示恢复：两类生成机制 200/200 正确识别", loc="left", fontweight="bold")
    soften_axes(ax_c, "x")
    panel_label(ax_c, "c", x=-0.16)
    ax_c.text(0.02, 0.96, "← 特征 RL 更好", transform=ax_c.transAxes, va="top", fontsize=7.5, color=COLORS["feature"])
    ax_c.text(0.98, 0.96, "动态规则更好 →", transform=ax_c.transAxes, ha="right", va="top", fontsize=7.5, color=COLORS["rule"])

    ax_d = fig.add_subplot(gs[1, 1])
    readout = data["readout_recovery"]
    readout_order = ["R0K", "R0KT_GLOBAL"]
    readout_labels = ["静态生成（真值 0）", "动态生成（真值 1.433）"]
    readout_colors = [COLORS["neutral"], COLORS["rule"]]
    for y, generator, color in zip(y_positions, readout_order, readout_colors):
        values = readout[readout["generator"].eq(generator)]["fitted_slope"].to_numpy()
        violin = ax_d.violinplot(values, positions=[y], vert=False, widths=0.72, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.25)
        jitter = deterministic_jitter(len(values), 0.13)
        ax_d.scatter(values, y + jitter, s=12, color=color, alpha=0.48, linewidth=0)
        true_value = float(readout[readout["generator"].eq(generator)]["true_slope"].iloc[0])
        ax_d.scatter(true_value, y, marker="|", s=230, linewidth=3, color="#111111", zorder=5)
        ax_d.scatter(values.mean(), y, marker="D", s=54, color=color, edgecolor="white", linewidth=0.8, zorder=4)
    ax_d.set_yticks(y_positions)
    ax_d.set_yticklabels(readout_labels)
    ax_d.set_xlabel("恢复出的共享练习斜率")
    ax_d.set_title("参数恢复：动态斜率几乎无偏", loc="left", fontweight="bold")
    soften_axes(ax_d, "x")
    panel_label(ax_d, "d", x=-0.16)
    ax_d.legend(
        handles=[
            Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["rule"], markeredgecolor="white", label="100次恢复的均值"),
            Line2D([0], [0], marker="|", color="#111111", linestyle="none", markersize=13, markeredgewidth=2.5, label="生成真值"),
        ],
        loc="lower right",
    )

    fig.text(
        0.085,
        0.025,
        "a RT 主 QC，n=96；b 口头报告 n=95（1名被试无可评分留出报告）；a/b 为被试 bootstrap 95% CI；c/d 每点=1次独立恢复，100次/生成模型。",
        fontsize=7.5,
        color="#5F6772",
    )
    return save_figure(fig, output, "fig3_external_validation_and_recovery", dpi)


def write_guide(output: Path, files: list[Path], data: dict[str, Any]) -> None:
    lines = [
        "# 统一新计划：可视化结果导读",
        "",
        "## 最先看哪张",
        "",
        "先看 `fig1_chinese_overview.png`。它不要求理解模型缩写，直接展示当前最合理的研究故事：选择随练习更依赖规则，RT 更像特征学习不确定性的反映，口头报告则混合了规则判断和报告习惯。",
        "",
        "## 图注",
        "",
        "**图 1 | 当前证据支持的混合行为架构。** 左上显示共享练习斜率对应的相对规则读出强度，阴影为 Wald 95% 区间。三个结果卡片分别总结留出选择、RT 和口头报告检验。下方用箭头表示当前证据支持的主要行为通道。选择和 RT 使用 96 名被试；口头报告使用 95 名具有可评分留出报告的被试。",
        "",
        "**图 2 | 练习依赖规则读出改善留出选择预测。** a 原计划静态规则与训练选择的非规则模型比较，以及动态规则与联合动态特征 RL 比较；点为被试平均，线为被试 bootstrap 95% CI。b 共享斜率带来的规则读出增强，阴影为 Wald 95% 区间。c 动态规则相对联合动态特征 RL 的被试级留出准确率差。每条件 n=32。条件1的 NLL 区间跨零，因此不能声称三个条件分别都支持规则模型。",
        "",
        "**图 3 | 外部检验与模型恢复。** a 规则熵和动态特征学习熵相对 RT 基线的留出预测增量。b 纯规则口头读出和训练拟合的全局混合读出相对口头基线的增量。c 动态规则和动态特征 RL 生成数据的双向表示恢复；200/200 次正确识别。d 静态和动态规则数据的共享斜率恢复；黑色短线为生成真值，菱形为100次恢复均值。a 使用 n=96；b 使用 n=95；区间均为被试 bootstrap 95% CI。c/d 每个生成模型100次独立恢复。",
        "",
        "## 阅读边界",
        "",
        "- 蓝色动态规则结果是见到静态模型失败后实施的预先规划扩展分支，应该表述为探索性工作模型，而不是原始确认性成功。",
        "- 口头报告混合模型是在纯规则读出失败后提出的自适应诊断，尽管留出效果强，仍需要独立数据确认。",
        "- 模型恢复的100%表示模拟数据中的机制可辨识，不表示真人行为预测准确率为100%。",
        "- 所有被试级点均保留；口头报告仅按正式评分规则排除没有可评分留出报告的1名被试，没有为了美化图形额外删点。",
        "",
        "## 文件",
        "",
    ]
    for path in sorted(files):
        lines.append(f"- `{path.name}`")
    lines.extend(
        [
            "",
            "每幅图同时提供 SVG（文字可编辑）、PDF、300 dpi PNG 和 600 dpi TIFF；对应 `source_data_*.csv` 是作图源数据。",
            "",
        ]
    )
    (output / "VISUAL_RESULTS_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    font_family = configure_style()
    data = build_sources(output)
    figure_files = []
    figure_files.extend(figure_overview(data, output, args.dpi))
    figure_files.extend(figure_choice(data, output, args.dpi))
    figure_files.extend(figure_external_recovery(data, output, args.dpi))
    write_guide(output, figure_files, data)

    manifest = {
        "result_type": "unified_newplan_publication_visualization",
        "status": "complete",
        "backend": "python_matplotlib",
        "font_family": font_family,
        "figures": [path.name for path in figure_files],
        "source_data": [path.name for path in sorted(output.glob("source_data_*.csv"))],
        "n_figures": 3,
        "n_exported_figure_files": len(figure_files),
        "subject_counts": {"choice": 96, "rt": 96, "oral": 95},
        "exclusions": {
            "choice": "none beyond the prespecified last-complete-block holdout",
            "rt": "formal main-QC rule only; no figure-specific exclusions",
            "oral": "one subject without an encodable held-out report; no figure-specific exclusions",
        },
        "runtime_seconds": time.time() - started,
        "python": platform.python_version(),
        "matplotlib": mpl.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(output / "manifest.json", manifest)
    print(f"[done] wrote 3 figures in SVG/PDF/PNG/TIFF to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
