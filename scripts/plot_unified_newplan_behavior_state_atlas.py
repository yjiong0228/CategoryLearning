#!/usr/bin/env python3
"""Draw plain-language individual pages for behavior-anchored states."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/zhuran/unified_newplan"
STATE = BASE / "behavior_anchored_state_20260802"
ORAL_VALIDATION = BASE / "behavior_state_oral_validation_20260802"
RT_VALIDATION = BASE / "behavior_state_rt_validation_20260802"
RECOVERY = BASE / "behavior_state_recovery_20260802"
DEFAULT_OUTPUT = BASE / "individual_behavior_state_atlas_20260802"

logging.getLogger("fontTools").setLevel(logging.WARNING)

COLORS = {
    "target": "#1764A5",
    "target_light": "#BBD7EC",
    "feature": "#2B9187",
    "other": "#D49342",
    "guess": "#C9CDD2",
    "actual": "#292D32",
    "ideal": "#8F969E",
    "holdout": "#FFF1D3",
    "boundary": "#C48824",
    "negative": "#B74442",
    "ink": "#20242A",
    "muted": "#626B76",
    "grid": "#E2E6EA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--page-dpi", type=int, default=220)
    parser.add_argument("--subjects", type=str, default=None)
    return parser.parse_args()


def configure_style() -> str:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if not font_path.exists():
        candidates = list(Path("/usr/share/fonts").rglob("NotoSansCJK*.ttc"))
        if not candidates:
            raise RuntimeError("No CJK font found")
        font_path = candidates[0]
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [family, "Arial", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return family


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


def rolling(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(np.asarray(values, dtype=float))
        .rolling(window, min_periods=1)
        .mean()
        .to_numpy()
    )


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7)


def draw_holdout(ax: plt.Axes, start: int, n_trials: int) -> None:
    ax.axvspan(start, n_trials, color=COLORS["holdout"], alpha=0.75, zorder=-8)
    ax.axvline(start, color=COLORS["boundary"], linestyle=(0, (3, 3)), lw=1.0)
    ax.text(
        start + max(1.0, 0.008 * n_trials),
        0.97,
        "最后一段考试",
        transform=ax.get_xaxis_transform(),
        va="top",
        color="#996A18",
        fontsize=6.7,
    )


def draw_sessions(ax: plt.Axes, frame: pd.DataFrame) -> None:
    session = frame["session"].to_numpy(dtype=int)
    for boundary in np.flatnonzero(session[1:] != session[:-1]) + 2:
        ax.axvline(boundary, color="#C8CDD2", lw=0.65, zorder=-6)


def summary_card(
    fig: plt.Figure,
    x: float,
    width: float,
    title: str,
    value: str,
    subtitle: str,
    color: str,
) -> None:
    patch = FancyBboxPatch(
        (x, 0.793),
        width,
        0.115,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=fig.transFigure,
        facecolor="#F7F9FA",
        edgecolor=color,
        linewidth=1.2,
    )
    fig.add_artist(patch)
    fig.text(x + 0.012, 0.883, title, fontsize=7.2, color=COLORS["muted"], va="top")
    fig.text(x + 0.012, 0.850, value, fontsize=12.0, fontweight="bold", color=color, va="top")
    fig.text(x + 0.012, 0.817, subtitle, fontsize=6.5, color=COLORS["muted"], va="top")


def final_state_text(row: pd.Series) -> tuple[str, str, str]:
    masses = {
        "更像正确规则": float(row.target_state_final64_mean),
        "更像特征学习": float(row.feature_state_final64_mean),
        "更像其他规则": float(row.other_rule_state_final64_mean),
        "更像猜测": float(row.guess_state_final64_mean),
    }
    label = max(masses, key=masses.get)
    color = {
        "更像正确规则": COLORS["target"],
        "更像特征学习": COLORS["feature"],
        "更像其他规则": COLORS["other"],
        "更像猜测": COLORS["guess"],
    }[label]
    subtitle = (
        f"正确规则{100*masses['更像正确规则']:.0f}% · "
        f"特征{100*masses['更像特征学习']:.0f}% · "
        f"其他{100*masses['更像其他规则']:.0f}%"
    )
    return label, subtitle, color


def plot_subject_page(frame: pd.DataFrame, row: pd.Series) -> plt.Figure:
    subject_id = int(row.subject_id)
    condition = int(row.condition)
    n_trials = int(row.n_trials)
    holdout_start = int(row.holdout_start_trial)
    chance = 0.5 if condition == 1 else 0.25
    window = min(32, max(8, n_trials // 20))
    trial = frame["trial"].to_numpy(dtype=float)
    delta = float(row.holdout_nll_feature - row.holdout_nll_behavior_oral)
    winner = "新状态模型" if delta > 0 else "特征学习模型"
    winner_color = COLORS["target"] if delta > 0 else COLORS["feature"]
    onset = row.target_state_t50_sustained16
    onset_text = "没有足够证据" if pd.isna(onset) else f"第 {int(onset)} 题"
    onset_subtitle = (
        "从未连续16题超过50%"
        if pd.isna(onset)
        else "连续16题更像正确规则"
    )
    state_value, state_subtitle, state_color = final_state_text(row)

    fig = plt.figure(figsize=(11.5, 8.2))
    fig.text(
        0.055,
        0.966,
        f"被试 {subject_id}｜条件 {condition}",
        fontsize=19,
        fontweight="bold",
        color=COLORS["ink"],
        va="top",
    )
    fig.text(
        0.055,
        0.928,
        f"共{n_trials}题 · 最后{int(row.n_holdout)}题只用于考试模型",
        fontsize=8.5,
        color=COLORS["muted"],
        va="top",
    )
    fig.text(
        0.945,
        0.963,
        "状态由按键和反馈前说法推断；当前反馈不进入状态",
        ha="right",
        va="top",
        fontsize=7.2,
        color=COLORS["negative"],
    )
    summary_card(
        fig,
        0.055,
        0.205,
        "最后64题实际答对",
        f"{100*row.exact_accuracy_final64:.1f}%",
        "这是被试真的做对了多少",
        COLORS["actual"],
    )
    summary_card(
        fig,
        0.273,
        0.205,
        "最后一段谁更会预测按键",
        winner,
        f"新模型每题少罚 {delta:+.3f}",
        winner_color,
    )
    summary_card(
        fig,
        0.491,
        0.205,
        "何时稳定更像正确规则",
        onset_text,
        onset_subtitle,
        COLORS["target"] if not pd.isna(onset) else COLORS["muted"],
    )
    summary_card(
        fig,
        0.709,
        0.235,
        "最后64题主要像什么",
        state_value,
        state_subtitle,
        state_color,
    )

    grid = fig.add_gridspec(
        3,
        2,
        left=0.075,
        right=0.965,
        bottom=0.075,
        top=0.755,
        height_ratios=[0.72, 1.05, 0.82],
        hspace=0.48,
        wspace=0.28,
    )

    ax_behavior = fig.add_subplot(grid[0, :])
    actual = rolling(frame["exact_correct"].to_numpy(dtype=float), window)
    ax_behavior.plot(trial, actual, color=COLORS["actual"], lw=2.0)
    ax_behavior.axhline(chance, color="#999FA6", ls=(0, (2, 3)), lw=0.8)
    ax_behavior.axhline(0.75, color=COLORS["target"], ls=(0, (3, 3)), lw=0.8, alpha=0.75)
    draw_holdout(ax_behavior, holdout_start, n_trials)
    draw_sessions(ax_behavior, frame)
    ax_behavior.set_xlim(1, n_trials)
    ax_behavior.set_ylim(-0.02, 1.02)
    ax_behavior.set_ylabel("最近一段答对的比例")
    ax_behavior.set_xlabel("做到第几题")
    ax_behavior.set_title(
        f"a  他实际上答对得怎么样（附近{window}题取平均）",
        loc="left",
        fontweight="bold",
    )
    ax_behavior.text(
        n_trials,
        0.76,
        "75%参考线",
        ha="right",
        va="bottom",
        fontsize=6.7,
        color=COLORS["target"],
    )
    style_axis(ax_behavior)

    ax_state = fig.add_subplot(grid[1, :])
    target = rolling(frame["state_target_rule_probability"].to_numpy(), window)
    feature = rolling(frame["state_feature_probability"].to_numpy(), window)
    other = rolling(frame["state_other_rule_probability"].to_numpy(), window)
    guess = rolling(frame["state_guess_probability"].to_numpy(), window)
    total = target + feature + other + guess
    target, feature, other, guess = [values / total for values in (target, feature, other, guess)]
    ax_state.stackplot(
        trial,
        guess,
        feature,
        other,
        target,
        colors=[COLORS["guess"], COLORS["feature"], COLORS["other"], COLORS["target"]],
        labels=["像在猜", "像在逐步学特征", "像在用其他规则", "像在用正确规则"],
        alpha=0.92,
        linewidth=0,
    )
    draw_holdout(ax_state, holdout_start, n_trials)
    draw_sessions(ax_state, frame)
    if not pd.isna(onset):
        ax_state.axvline(float(onset), color="#123E67", ls=(0, (3, 3)), lw=1.0)
        ax_state.text(
            float(onset) + max(1.0, 0.006 * n_trials),
            0.52,
            f"从第{int(onset)}题起\n连续16题过半",
            fontsize=6.5,
            color="#123E67",
            va="center",
        )
    ax_state.set_xlim(1, n_trials)
    ax_state.set_ylim(0, 1)
    ax_state.set_ylabel("四种解释合计100%")
    ax_state.set_xlabel("做到第几题")
    ax_state.set_title(
        "b  根据这个人的按键和反馈前说法，哪种解释更像",
        loc="left",
        fontweight="bold",
    )
    ax_state.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=4, fontsize=7)
    style_axis(ax_state)

    ax_fit = fig.add_subplot(grid[2, 0])
    new_fit = rolling(
        frame["observed_choice_probability_behavior_oral"].to_numpy(), window
    )
    feature_fit = rolling(
        frame["observed_choice_probability_feature"].to_numpy(), window
    )
    ax_fit.plot(trial, new_fit, color=COLORS["target"], lw=1.9, label="新状态模型")
    ax_fit.plot(trial, feature_fit, color=COLORS["feature"], lw=1.6, label="只用特征学习")
    ax_fit.axhline(chance, color="#999FA6", ls=(0, (2, 3)), lw=0.8)
    draw_holdout(ax_fit, holdout_start, n_trials)
    draw_sessions(ax_fit, frame)
    ax_fit.set_xlim(1, n_trials)
    ax_fit.set_ylim(-0.02, 1.02)
    ax_fit.set_ylabel("给实际按键的概率")
    ax_fit.set_xlabel("做到第几题")
    ax_fit.set_title("c  模型能不能猜到他下一题会按什么", loc="left", fontweight="bold")
    ax_fit.legend(loc="lower right")
    style_axis(ax_fit)

    ax_audit = fig.add_subplot(grid[2, 1])
    ideal = frame["ideal_observer_target_probability"].to_numpy(dtype=float)
    ax_audit.plot(
        trial,
        ideal,
        color=COLORS["ideal"],
        lw=1.4,
        ls=(0, (4, 2)),
        label="旧线：反馈已经足够让电脑排除错误规则",
    )
    ax_audit.plot(
        trial,
        target,
        color=COLORS["target"],
        lw=2.0,
        label="新线：人的行为和说法支持正确规则",
    )
    draw_holdout(ax_audit, holdout_start, n_trials)
    draw_sessions(ax_audit, frame)
    ax_audit.set_xlim(1, n_trials)
    ax_audit.set_ylim(-0.02, 1.02)
    ax_audit.set_ylabel("正确规则所占比例")
    ax_audit.set_xlabel("做到第几题")
    ax_audit.set_title("d  旧图为什么会假快：电脑有线索 ≠ 人已经学会", loc="left", fontweight="bold")
    ax_audit.legend(loc="lower right", fontsize=6.5)
    style_axis(ax_audit)

    fig.text(
        0.075,
        0.025,
        "读图底线：蓝色状态是相对解释权重，不是读脑结果；如果正确规则与特征学习对下一题给出同样答案，模型会保留不确定，不强行宣布‘学会’。",
        fontsize=6.8,
        color=COLORS["muted"],
    )
    return fig


def prepare_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    states = pd.read_csv(STATE / "trial_states.csv.gz")
    summary = pd.read_csv(STATE / "subject_summary.csv")
    if len(summary) != 96 or len(states) != 62720:
        raise ValueError("Expected 96 subjects and 62,720 state rows")
    return states, summary


def plot_overview(states: pd.DataFrame, summary: pd.DataFrame) -> plt.Figure:
    ordered = summary.sort_values(["condition", "subject_id"]).reset_index(drop=True)
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(11.5, 14.5),
        gridspec_kw={"width_ratios": [1.15, 1.0, 0.82], "wspace": 0.28},
    )
    fig.suptitle("96名被试：任务线索、真实表现和行为状态不再混为一谈", fontsize=16, fontweight="bold", y=0.992)
    fig.text(0.5, 0.972, "每一行是一名被试；三种条件各32人", ha="center", fontsize=8, color=COLORS["muted"])

    ax = axes[0]
    ideal_time = ordered["ideal_observer_t90_sustained16"].clip(lower=1)
    behavioral_time = ordered["behavioral_onset_75_roll32_sustained5"].clip(lower=1)
    ax.scatter(ideal_time, y, s=10, color=COLORS["ideal"], label="电脑排除错误规则")
    ax.scatter(behavioral_time, y, s=12, color=COLORS["actual"], label="实际正确率稳定到75%")
    reached = ordered["target_state_t50_sustained16"].notna()
    ax.scatter(ordered.loc[reached, "target_state_t50_sustained16"], y[reached], s=16, color=COLORS["target"], label="行为状态连续过半")
    ax.scatter(ordered.loc[~reached, "n_trials"], y[~reached], s=16, marker="x", color=COLORS["negative"], label="到结束仍未过半")
    ax.set_xscale("log")
    ax.set_xlim(3, max(2200, ordered.n_trials.max() * 1.08))
    ax.set_xlabel("第几题（横轴为对数）")
    ax.set_title("a  三种‘学会时间’其实回答不同问题", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=6.3)

    ax = axes[1]
    left = np.zeros(len(ordered))
    for column, label, color in (
        ("guess_state_final64_mean", "猜", COLORS["guess"]),
        ("feature_state_final64_mean", "特征学习", COLORS["feature"]),
        ("other_rule_state_final64_mean", "其他规则", COLORS["other"]),
        ("target_state_final64_mean", "正确规则", COLORS["target"]),
    ):
        values = ordered[column].to_numpy(dtype=float)
        ax.barh(y, values, left=left, height=0.78, color=color, edgecolor="none", label=label)
        left += values
    ax.set_xlim(0, 1)
    ax.set_xlabel("最后64题的相对解释权重")
    ax.set_title("b  最后阶段主要像哪种策略", loc="left", fontweight="bold")
    ax.legend(loc="lower right", fontsize=6.3)

    ax = axes[2]
    delta = (ordered["holdout_nll_feature"] - ordered["holdout_nll_behavior_oral"]).to_numpy()
    colors = np.where(delta >= 0, COLORS["target"], COLORS["feature"])
    ax.barh(y, delta, height=0.72, color=colors)
    ax.axvline(0, color=COLORS["ink"], lw=0.8)
    ax.set_xlabel("每题少罚多少\n←特征更好｜新状态更好→")
    ax.set_title("c  谁更会预测最后一段按键", loc="left", fontweight="bold")

    labels = ordered["subject_id"].astype(str).tolist()
    for ax in axes:
        ax.set_ylim(len(ordered) - 0.5, -0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=5.1)
        for boundary in (31.5, 63.5):
            ax.axhline(boundary, color=COLORS["ink"], lw=0.8)
        ax.grid(axis="x", color=COLORS["grid"], lw=0.6)
        ax.set_axisbelow(True)
    fig.text(0.018, 0.825, "条件1", rotation=90, va="center", fontsize=8)
    fig.text(0.018, 0.505, "条件2", rotation=90, va="center", fontsize=8)
    fig.text(0.018, 0.185, "条件3", rotation=90, va="center", fontsize=8)
    fig.text(0.055, 0.012, "灰点只说明实验反馈对完美电脑已经够用；蓝点才是由人的选择和既往说法支持的状态。红叉不是失败，而是诚实保留‘无法区分’。", fontsize=7, color=COLORS["muted"])
    return fig


def write_index(output: Path, summary: pd.DataFrame) -> None:
    lines = [
        "# 逐被试行为状态索引",
        "",
        "点击被试编号查看单页。‘未达到’表示模型从未连续16题把正确规则解释权重放到50%以上；不能擅自改写为已经学会。",
        "",
    ]
    for condition, group in summary.groupby("condition", sort=True):
        lines.extend(
            [
                f"## 条件 {int(condition)}",
                "",
                "| 被试 | 最后64题正确率 | 状态模型比特征模型每题少罚 | 正确规则状态过半 | 最后64题正确规则权重 | 特征权重 |",
                "|:--|--:|--:|--:|--:|--:|",
            ]
        )
        for row in group.sort_values("subject_id").itertuples(index=False):
            onset = "未达到" if pd.isna(row.target_state_t50_sustained16) else f"第{int(row.target_state_t50_sustained16)}题"
            delta = row.holdout_nll_feature - row.holdout_nll_behavior_oral
            lines.append(
                f"| [{int(row.subject_id)}](subjects/subject_{int(row.subject_id)}.png) | "
                f"{100*row.exact_accuracy_final64:.1f}% | {delta:+.3f} | {onset} | "
                f"{100*row.target_state_final64_mean:.1f}% | {100*row.feature_state_final64_mean:.1f}% |"
            )
        lines.append("")
    (output / "INDIVIDUAL_INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def write_guide(output: Path, summary: pd.DataFrame) -> None:
    reached_mask = summary["target_state_t50_sustained16"].notna()
    reached = summary.loc[reached_mask, "target_state_t50_sustained16"]
    oral = pd.read_csv(ORAL_VALIDATION / "model_comparisons.csv")
    oral_all = oral[oral["condition"].astype(str).eq("all")].iloc[0]
    rt = pd.read_csv(RT_VALIDATION / "model_comparisons.csv")
    rt_all = rt[
        rt["qc_specification"].eq("main_qc")
        & rt["comparison"].eq("state_uncertainty_increment")
        & rt["condition"].astype(str).eq("all")
    ].iloc[0]
    lines = [
        "# 新版逐被试图：先读这份大白话说明",
        "",
        "## 一句话结论",
        "",
        "旧图画的是‘反馈足够让完美电脑排除错误规则’，新版画的是‘这个人的选择和反馈前说法更支持哪种策略’。两者不能混用。",
        "",
        "## 新模型怎样工作",
        "",
        "每一题开始前，新模型同时保留四种可能：这个人在猜、在逐步学特征、在用某条错误规则、或在用正确规则。模型先预测他会按什么；看到按键后才调整四种可能；随后可以读取反馈前口头说法；本题反馈绝不直接把状态推向正确规则。",
        "",
        "## 主要结果",
        "",
        f"- 只有{len(reached)}/96人曾连续16题让‘正确规则’解释权重超过50%；其余{96-len(reached)}人保留为‘证据不足’，不再强行给一个学习时刻。",
        f"- 在达到的人中，中位时间是第{reached.median():.1f}题；旧理想电脑曲线的中位时间只有第{summary.ideal_observer_t90_sustained16.median():.1f}题。",
        f"- 只由过去选择得到的状态能够预测下一次反馈前口头报告：留出log score平均改善{oral_all.mean_delta_log_score:.3f}，95% CI [{oral_all.bootstrap_mean_ci_low:.3f}, {oral_all.bootstrap_mean_ci_high:.3f}]，改善{int(oral_all.n_improved)}/{int(oral_all.n_subjects)}人。",
        f"- 反应时没有通过外部检验：加入状态不确定度的留出增量为{rt_all.mean_delta_log_predictive_density:.3f}，95% CI [{rt_all.bootstrap_mean_ci_low:.3f}, {rt_all.bootstrap_mean_ci_high:.3f}]。因此不能说该状态解释了反应速度。",
        "",
        "## 每页怎么看",
        "",
        "- 顶部四个框：真实正确率、最后一段谁更会预测按键、正确规则状态是否稳定过半、最后64题主要像哪种策略。",
        "- a：黑线是真实正确率，是全页最先应该看的线。",
        "- b：四种颜色加起来永远是100%。蓝色多表示行为和说法更像正确规则；绿色多表示正确规则与渐进特征学习仍难区分；橙色是其他规则；灰色是猜测。",
        "- c：两条线比较模型是否能预测被试下一题按什么，不是任务正确率。",
        "- d：灰虚线是旧理想电脑基线，蓝线才是新版行为状态。灰线早到1不再被解释成人学会。",
        "- 黄色区域是最后一段模型考试；参数没有用这段数据调整，但模型仍按顺序读取已经发生的前一题选择和口头报告。",
        "",
        "## 解释底线",
        "",
        "新版状态仍然是模型归因，不是直接读脑。当正确规则和特征学习在某些题上预测相同答案时，数据本来就无法区分，绿色和蓝色可以长期并存。口头报告兼容只是宽泛结构指标；反应时未验证通过；模型目前是条件层面的筛查模型，不是最终层级贝叶斯结论。",
        "",
        "## 文件",
        "",
        "- `all_subjects_behavior_state_atlas.pdf`：96页逐被试图册。",
        "- `INDIVIDUAL_INDEX.md`：点击被试编号打开单页。",
        "- `fig_behavior_state_overview.*`：96人对齐总览。",
        "- `subjects/subject_*.png`：每人一张高清图。",
        "- 新模型的逐试次数据在相邻结果目录的 `trial_states.csv.gz`。",
        "",
    ]
    if (RECOVERY / "RESULTS.md").exists():
        lines.extend(
            [
                "## 恢复检查",
                "",
                "条件模拟恢复已经完成，详细结果见状态恢复目录；它只说明程序能找回自己生成的状态，不能证明真人一定采用这些状态。",
                "",
            ]
        )
    (output / "READ_ME_FIRST_CN.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    subject_output = output / "subjects"
    subject_output.mkdir(exist_ok=True)
    font_family = configure_style()
    states, summary = prepare_summary()
    if args.subjects:
        selected = {int(value.strip()) for value in args.subjects.split(",") if value.strip()}
        summary = summary[summary["subject_id"].isin(selected)].copy()
        states = states[states["subject_id"].isin(selected)].copy()
        if len(summary) != len(selected):
            raise ValueError("At least one requested subject is unavailable")

    overview_paths = []
    if not args.subjects:
        overview = plot_overview(states, summary)
        overview_png = output / "fig_behavior_state_overview.png"
        overview_svg = output / "fig_behavior_state_overview.svg"
        overview_pdf = output / "fig_behavior_state_overview.pdf"
        overview_tiff = output / "fig_behavior_state_overview.tiff"
        overview.savefig(overview_png, dpi=300, bbox_inches="tight", pad_inches=0.08)
        overview.savefig(overview_svg, bbox_inches="tight", pad_inches=0.08)
        overview.savefig(overview_pdf, bbox_inches="tight", pad_inches=0.08)
        overview.savefig(
            overview_tiff,
            dpi=600,
            bbox_inches="tight",
            pad_inches=0.08,
            pil_kwargs={"compression": "tiff_lzw"},
        )
        overview_paths.extend(
            [overview_png, overview_svg, overview_pdf, overview_tiff]
        )
        plt.close(overview)

    atlas_path = output / "all_subjects_behavior_state_atlas.pdf"
    with PdfPages(atlas_path) as pdf:
        for row in summary.sort_values(["condition", "subject_id"]).itertuples(index=False):
            subject_id = int(row.subject_id)
            frame = states[states["subject_id"].eq(subject_id)]
            figure = plot_subject_page(frame, pd.Series(row._asdict()))
            pdf.savefig(figure, bbox_inches="tight", pad_inches=0.08)
            figure.savefig(
                subject_output / f"subject_{subject_id}.png",
                dpi=max(180, int(args.page_dpi)),
                bbox_inches="tight",
                pad_inches=0.08,
            )
            plt.close(figure)

    write_index(output, summary)
    write_guide(output, summary)
    manifest = {
        "result_type": "behavior_anchored_individual_state_atlas",
        "status": "complete",
        "figure_contract": {
            "core_conclusion": (
                "participant choices and feedback-before reports support gradual, "
                "often unresolved strategy attribution rather than universal rapid "
                "target-rule acquisition"
            ),
            "archetype": "quantitative individual diagnostic atlas",
            "hero_panel": "four-way behavior-anchored state composition",
            "backend": "python_matplotlib",
        },
        "n_subjects": int(summary.subject_id.nunique()),
        "n_trial_rows": int(len(states)),
        "atlas_pdf": atlas_path.name,
        "n_subject_png": int(len(list(subject_output.glob("subject_*.png")))),
        "overview_exports": [path.name for path in overview_paths],
        "font_family": font_family,
        "state_manifest_sha256": sha256_file(STATE / "manifest.json"),
        "oral_validation_manifest_sha256": sha256_file(ORAL_VALIDATION / "manifest.json"),
        "rt_validation_manifest_sha256": sha256_file(RT_VALIDATION / "manifest.json"),
        "runtime_seconds": float(time.time() - started),
        "python": platform.python_version(),
        "matplotlib": mpl.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(output / "manifest.json", manifest)
    print(f"[done] wrote {len(summary)} behavior-state pages to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
