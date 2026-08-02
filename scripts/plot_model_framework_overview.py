#!/usr/bin/env python3
"""Draw the manuscript overview for the unified condition 1--3 model.

The figure is intentionally data-free: it summarizes the preregistered causal
order, evidence roles, and model-selection gates defined in model_newplan.tex.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "manuscript" / "figures" / "model_framework_overview"

INK = "#25313C"
MUTED = "#5F6D78"
LINE = "#AAB6C0"
PANEL = "#F7F9FB"
WHITE = "#FFFFFF"

BLUE = "#356D9B"
BLUE_DARK = "#244E73"
BLUE_PALE = "#E7F0F7"
TEAL = "#3A807D"
TEAL_PALE = "#E4F2F0"
GOLD = "#B6812C"
GOLD_PALE = "#FBF2DD"
VIOLET = "#745C96"
VIOLET_PALE = "#F0EBF6"
RED = "#B4514E"
RED_PALE = "#F8E9E7"
GREEN = "#4D7D59"
GREEN_PALE = "#E8F1EA"
GREY_PALE = "#EEF2F5"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "stixsans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7.2,
        "axes.linewidth": 0.8,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
    }
)


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    *,
    facecolor=WHITE,
    edgecolor=LINE,
    linewidth=0.8,
    radius=0.012,
    zorder=2,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.006,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def card(
    ax,
    x,
    y,
    w,
    h,
    title,
    body,
    *,
    facecolor=WHITE,
    edgecolor=LINE,
    title_color=INK,
    title_size=7.5,
    body_size=6.35,
    align="left",
    title_y=0.72,
):
    rounded_box(
        ax,
        x,
        y,
        w,
        h,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.85,
    )
    ha = "center" if align == "center" else "left"
    tx = x + w / 2 if align == "center" else x + 0.012
    ax.text(
        tx,
        y + h * title_y,
        title,
        ha=ha,
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=title_color,
        zorder=4,
    )
    ax.text(
        tx,
        y + h * 0.33,
        body,
        ha=ha,
        va="center",
        fontsize=body_size,
        color=INK,
        linespacing=1.35,
        zorder=4,
    )


def arrow(
    ax,
    start,
    end,
    *,
    color=LINE,
    linewidth=1.1,
    style="-|>",
    connection="arc3,rad=0",
    mutation=9,
    zorder=3,
    dashed=False,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation,
        linewidth=linewidth,
        linestyle="--" if dashed else "-",
        color=color,
        connectionstyle=connection,
        shrinkA=1.5,
        shrinkB=1.5,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def pill(ax, x, y, w, h, text, *, facecolor, color, fontsize=6.0):
    rounded_box(
        ax,
        x,
        y,
        w,
        h,
        facecolor=facecolor,
        edgecolor=facecolor,
        linewidth=0.5,
        radius=h / 2,
        zorder=3,
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=color,
        zorder=4,
    )


def panel_label(ax, letter, title, x, y):
    ax.text(
        x,
        y,
        letter,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color=INK,
        zorder=6,
    )
    ax.text(
        x + 0.035,
        y - 0.001,
        title,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=INK,
        zorder=6,
    )


def draw_distribution(ax, x, y, w, h, values, color):
    gap = w * 0.07
    bar_w = (w - gap * (len(values) - 1)) / len(values)
    for idx, value in enumerate(values):
        bx = x + idx * (bar_w + gap)
        ax.add_patch(
            Rectangle(
                (bx, y),
                bar_w,
                h * value,
                facecolor=color,
                edgecolor="none",
                zorder=5,
            )
        )


def make_figure():
    fig = plt.figure(figsize=(7.2, 8.15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title and the one-sentence contract.
    ax.text(
        0.045,
        0.975,
        "跨条件统一规则学习模型：从独立知觉约束到可证伪预测",
        ha="left",
        va="top",
        fontsize=13.2,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.045,
        0.943,
        "同一反馈前规则信念驱动选择、反应时与口头报告；复杂机制只有通过恢复、时间留出和外部检验才进入解释。",
        ha="left",
        va="top",
        fontsize=7.5,
        color=MUTED,
    )

    # Panel a: task inputs and what is shared versus condition-specific.
    rounded_box(ax, 0.035, 0.785, 0.93, 0.135, facecolor=PANEL, edgecolor="#D7DFE5")
    panel_label(ax, "a", "实验约束：共享流程，条件特异的规则与反馈", 0.050, 0.910)

    a_y, a_h = 0.806, 0.076
    card(
        ax,
        0.055,
        a_y,
        0.205,
        a_h,
        "Task 1b｜独立知觉测量",
        "被试特异误差  $g_s$\n不由 Task 2 选择自由调节",
        facecolor=GOLD_PALE,
        edgecolor=GOLD,
        title_color=GOLD,
        body_size=6.0,
    )
    card(
        ax,
        0.285,
        a_y,
        0.195,
        a_h,
        "条件 1",
        "2 类｜38 条带标签规则\n反馈：0 / 1",
        facecolor=BLUE_PALE,
        edgecolor=BLUE,
        title_color=BLUE_DARK,
        align="center",
        body_size=6.0,
    )
    card(
        ax,
        0.505,
        a_y,
        0.195,
        a_h,
        "条件 2",
        "4 类｜116 条规则\n精确类别反馈：0 / 1",
        facecolor=BLUE_PALE,
        edgecolor=BLUE,
        title_color=BLUE_DARK,
        align="center",
        body_size=6.0,
    )
    card(
        ax,
        0.725,
        a_y,
        0.215,
        a_h,
        "条件 3",
        "4 类｜116 条规则\n家族分级反馈：0 / 0.5 / 1",
        facecolor=BLUE_PALE,
        edgecolor=BLUE,
        title_color=BLUE_DARK,
        align="center",
        body_size=6.0,
    )
    pill(
        ax,
        0.305,
        0.888,
        0.61,
        0.020,
        "共同的四维刺激空间与学习架构；参数跨条件部分汇聚",
        facecolor="#E9EEF2",
        color=MUTED,
        fontsize=5.8,
    )

    # Panel b: the within-trial causal model, the hero panel.
    rounded_box(ax, 0.035, 0.405, 0.93, 0.355, facecolor=WHITE, edgecolor="#C8D2DA", linewidth=1.05)
    panel_label(ax, "b", "一个试次内的生成过程：反馈前状态是三种行为的共同来源", 0.050, 0.748)
    pill(
        ax,
        0.675,
        0.724,
        0.255,
        0.021,
        "因果约束：当前反馈不能解释当前行为",
        facecolor=RED_PALE,
        color=RED,
        fontsize=5.7,
    )

    # Main upper stream.
    card(
        ax,
        0.055,
        0.570,
        0.125,
        0.105,
        "物理刺激",
        "$\\mathbf{x}_{s,t}$",
        facecolor=GREY_PALE,
        edgecolor=LINE,
        align="center",
        title_size=7.1,
        body_size=6.2,
    )
    # Four small feature bars make the input concrete.
    feature_heights = [0.018, 0.031, 0.024, 0.037]
    feature_colors = ["#9EB6C8", "#7FA0B9", "#5F8BAA", "#3F769B"]
    for idx, (bar_h, bar_color) in enumerate(zip(feature_heights, feature_colors)):
        ax.add_patch(
            Rectangle(
                (0.073 + idx * 0.022, 0.580),
                0.012,
                bar_h,
                facecolor=bar_color,
                edgecolor="none",
                zorder=6,
            )
        )
    ax.text(0.117, 0.574, "颈  腿  头  尾", ha="center", va="top", fontsize=5.0, color=MUTED, zorder=7)

    card(
        ax,
        0.215,
        0.570,
        0.145,
        0.105,
        "知觉不确定性",
        "$\\widetilde{\\mathbf{x}}_{s,t} \\sim g_s$\nTask 1b 约束",
        facecolor=GOLD_PALE,
        edgecolor=GOLD,
        title_color=GOLD,
        align="center",
        title_size=7.1,
        body_size=6.2,
    )
    card(
        ax,
        0.395,
        0.570,
        0.155,
        0.105,
        "候选规则库",
        "$q_{s,t,h}(c)$",
        facecolor=TEAL_PALE,
        edgecolor=TEAL,
        title_color=TEAL,
        align="center",
        title_size=7.1,
        body_size=6.2,
    )
    # Minimal rule glyphs.
    for idx in range(3):
        gx = 0.414 + idx * 0.037
        ax.add_patch(Rectangle((gx, 0.580), 0.027, 0.020, facecolor=WHITE, edgecolor=TEAL, linewidth=0.55, zorder=6))
        if idx == 0:
            ax.plot([gx + 0.0135, gx + 0.0135], [0.581, 0.599], color=TEAL, lw=0.75, zorder=7)
        elif idx == 1:
            ax.plot([gx + 0.003, gx + 0.024], [0.582, 0.598], color=TEAL, lw=0.75, zorder=7)
        else:
            ax.plot([gx + 0.003, gx + 0.024], [0.598, 0.582], color=TEAL, lw=0.75, zorder=7)

    card(
        ax,
        0.585,
        0.570,
        0.165,
        0.105,
        "反馈前规则信念",
        "$\\pi^-_{s,t}(h)$\n多条规则保留非零概率",
        facecolor=VIOLET_PALE,
        edgecolor=VIOLET,
        title_color=VIOLET,
        align="center",
        title_size=7.1,
        body_size=6.0,
    )
    draw_distribution(ax, 0.608, 0.580, 0.115, 0.025, [0.25, 0.52, 0.95, 0.38, 0.68, 0.20], VIOLET)

    card(
        ax,
        0.790,
        0.608,
        0.145,
        0.083,
        "选择",
        "$P(c)=\\sum_h \\pi_h^-q_h(c)$",
        facecolor=BLUE_PALE,
        edgecolor=BLUE,
        title_color=BLUE_DARK,
        align="center",
        title_size=7.1,
        body_size=6.0,
        title_y=0.70,
    )
    card(
        ax,
        0.790,
        0.510,
        0.145,
        0.070,
        "反应时 RT",
        "选择熵 $U^{\\rm choice}$",
        facecolor=BLUE_PALE,
        edgecolor=BLUE,
        title_color=BLUE_DARK,
        align="center",
        title_size=7.0,
        body_size=5.9,
        title_y=0.68,
    )
    card(
        ax,
        0.790,
        0.424,
        0.145,
        0.060,
        "反馈前口头报告",
        "$\\pi^{\\rm oral} \\propto \\pi^-q(c)$",
        facecolor=TEAL_PALE,
        edgecolor=TEAL,
        title_color=TEAL,
        align="center",
        title_size=6.6,
        body_size=5.8,
        title_y=0.70,
    )

    # Lower feedback-update stream.
    card(
        ax,
        0.585,
        0.438,
        0.165,
        0.077,
        "反馈 $r$ → 规则更新",
        "$E^+=E^-+\\log L(h)$\n得到 $\\pi^+_{s,t}$",
        facecolor=GREEN_PALE,
        edgecolor=GREEN,
        title_color=GREEN,
        align="center",
        title_size=7.0,
        body_size=5.9,
        title_y=0.70,
    )
    card(
        ax,
        0.395,
        0.438,
        0.155,
        0.077,
        "有限记忆",
        "$E^-_{t+1}=\\lambda E^+_t$\n进入下一试次",
        facecolor=GREY_PALE,
        edgecolor=LINE,
        align="center",
        title_size=7.0,
        body_size=5.9,
        title_y=0.70,
    )

    arrow(ax, (0.180, 0.623), (0.215, 0.623), color=GOLD, linewidth=1.25)
    arrow(ax, (0.360, 0.623), (0.395, 0.623), color=TEAL, linewidth=1.25)
    arrow(ax, (0.550, 0.623), (0.585, 0.623), color=VIOLET, linewidth=1.25)
    arrow(ax, (0.750, 0.640), (0.790, 0.650), color=BLUE, linewidth=1.35)
    arrow(ax, (0.750, 0.600), (0.790, 0.545), color=BLUE, linewidth=1.15)
    arrow(ax, (0.862, 0.608), (0.862, 0.580), color=BLUE, linewidth=1.05)
    arrow(ax, (0.862, 0.510), (0.862, 0.484), color=TEAL, linewidth=1.05)
    arrow(ax, (0.790, 0.454), (0.750, 0.474), color=GREEN, linewidth=1.2)
    arrow(ax, (0.585, 0.476), (0.550, 0.476), color=GREEN, linewidth=1.2)
    arrow(
        ax,
        (0.472, 0.515),
        (0.610, 0.570),
        color=VIOLET,
        linewidth=1.15,
        connection="arc3,rad=-0.20",
    )
    pill(
        ax,
        0.060,
        0.421,
        0.275,
        0.026,
        "关键设计 1｜选择、RT、口头报告都只读取 $\\pi^-_{s,t}$",
        facecolor=VIOLET_PALE,
        color=VIOLET,
        fontsize=5.6,
    )
    pill(
        ax,
        0.060,
        0.455,
        0.275,
        0.026,
        "关键设计 2｜反馈只更新下一试次，不重复计入损失",
        facecolor=GREEN_PALE,
        color=GREEN,
        fontsize=5.5,
    )

    # Panel c: estimation and held-out validation.
    rounded_box(ax, 0.035, 0.125, 0.615, 0.255, facecolor=PANEL, edgecolor="#D7DFE5")
    panel_label(ax, "c", "模型识别：先用选择确定核心，再冻结并检验外部通道", 0.050, 0.368)

    card(
        ax,
        0.055,
        0.260,
        0.135,
        0.075,
        "候选模型阶梯",
        "NR0--NR3 基线\nR0 $\\rightarrow$ R1 $\\rightarrow$ R2/R3",
        facecolor=GREY_PALE,
        edgecolor=LINE,
        title_size=6.8,
        body_size=5.6,
    )
    card(
        ax,
        0.220,
        0.260,
        0.135,
        0.075,
        "参数训练段",
        "只拟合逐试次选择\n个体参数部分汇聚",
        facecolor=BLUE_PALE,
        edgecolor=BLUE,
        title_color=BLUE_DARK,
        title_size=6.8,
        body_size=5.6,
    )
    card(
        ax,
        0.385,
        0.260,
        0.115,
        0.075,
        "时间留出段",
        "一步选择 NLL\n参数不再更新",
        facecolor=BLUE_PALE,
        edgecolor=BLUE,
        title_color=BLUE_DARK,
        title_size=6.8,
        body_size=5.6,
    )
    card(
        ax,
        0.530,
        0.260,
        0.095,
        0.075,
        "冻结核心",
        "$\\lambda,\\kappa,p_0$\n结构与参数",
        facecolor=RED_PALE,
        edgecolor=RED,
        title_color=RED,
        align="center",
        title_size=6.8,
        body_size=5.5,
    )
    arrow(ax, (0.190, 0.298), (0.220, 0.298), color=LINE)
    arrow(ax, (0.355, 0.298), (0.385, 0.298), color=BLUE)
    arrow(ax, (0.500, 0.298), (0.530, 0.298), color=RED)

    card(
        ax,
        0.055,
        0.155,
        0.160,
        0.067,
        "RT 外部检验",
        "选择熵是否提高留出 RT 预测",
        facecolor=BLUE_PALE,
        edgecolor=BLUE,
        title_color=BLUE_DARK,
        title_size=6.7,
        body_size=5.5,
        title_y=0.68,
    )
    card(
        ax,
        0.245,
        0.155,
        0.170,
        0.067,
        "口头报告外部检验",
        "兼容规则质量优于冻结基线",
        facecolor=TEAL_PALE,
        edgecolor=TEAL,
        title_color=TEAL,
        title_size=6.7,
        body_size=5.5,
        title_y=0.68,
    )
    card(
        ax,
        0.445,
        0.155,
        0.180,
        0.067,
        "冻结后的自主生成",
        "前 64 试次预测 + 完整后验检查",
        facecolor=GREEN_PALE,
        edgecolor=GREEN,
        title_color=GREEN,
        title_size=6.7,
        body_size=5.4,
        title_y=0.68,
    )
    pill(
        ax,
        0.190,
        0.232,
        0.300,
        0.020,
        "RT 与口头报告不反向改变核心学习参数",
        facecolor=RED_PALE,
        color=RED,
        fontsize=5.35,
    )
    arrow(ax, (0.578, 0.260), (0.450, 0.252), color=RED, linewidth=1.0, mutation=7)
    arrow(ax, (0.250, 0.232), (0.135, 0.222), color=BLUE, dashed=True, connection="arc3,rad=-0.04", mutation=7)
    arrow(ax, (0.340, 0.232), (0.330, 0.222), color=TEAL, dashed=True, mutation=7)
    arrow(ax, (0.430, 0.232), (0.535, 0.222), color=GREEN, dashed=True, connection="arc3,rad=0.04", mutation=7)

    # Panel d: sequential gates and interpretation boundary.
    rounded_box(ax, 0.675, 0.125, 0.29, 0.255, facecolor=WHITE, edgecolor="#D7DFE5")
    panel_label(ax, "d", "顺序证据门槛", 0.690, 0.368)
    gate_y = [0.321, 0.286, 0.251, 0.216, 0.181]
    gate_titles = [
        "1  实现正确",
        "2  参数/模型可恢复",
        "3  留出选择改善",
        "4  RT 或口头报告支持",
        "5  生成与跨条件合格",
    ]
    gate_colors = [LINE, VIOLET, BLUE, TEAL, GREEN]
    gate_fills = [GREY_PALE, VIOLET_PALE, BLUE_PALE, TEAL_PALE, GREEN_PALE]
    for y, title, edge, fill in zip(gate_y, gate_titles, gate_colors, gate_fills):
        rounded_box(ax, 0.700, y - 0.014, 0.235, 0.026, facecolor=fill, edgecolor=edge, linewidth=0.7, radius=0.010)
        ax.text(0.713, y - 0.001, title, ha="left", va="center", fontsize=5.8, fontweight="bold", color=INK, zorder=5)
    for y1, y2 in zip(gate_y[:-1], gate_y[1:]):
        arrow(ax, (0.817, y1 - 0.014), (0.817, y2 + 0.012), color=LINE, linewidth=0.8, mutation=6, zorder=2)

    rounded_box(ax, 0.700, 0.137, 0.235, 0.023, facecolor=RED_PALE, edgecolor=RED, linewidth=0.7, radius=0.010)
    ax.text(
        0.817,
        0.148,
        "未过门槛：简化、合并等价模型或拒绝机制",
        ha="center",
        va="center",
        fontsize=5.35,
        fontweight="bold",
        color=RED,
        zorder=5,
    )

    # Bottom takeaway.
    rounded_box(ax, 0.035, 0.045, 0.93, 0.052, facecolor=INK, edgecolor=INK, linewidth=0.8, radius=0.012)
    ax.text(
        0.055,
        0.071,
        "可解释结论",
        ha="left",
        va="center",
        fontsize=7.2,
        fontweight="bold",
        color=WHITE,
        zorder=5,
    )
    ax.text(
        0.165,
        0.071,
        "证据支持的是一个透明、可生成、可证伪的最简规则信念模型，而不是对逐试次心理状态的唯一反演。",
        ha="left",
        va="center",
        fontsize=6.25,
        color=WHITE,
        zorder=5,
    )

    return fig


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    fig.savefig(OUTPUT.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTPUT.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        OUTPUT.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.04,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    print(f"Wrote {OUTPUT.with_suffix('.pdf')}")
    print(f"Wrote {OUTPUT.with_suffix('.svg')}")
    print(f"Wrote {OUTPUT.with_suffix('.png')}")
    print(f"Wrote {OUTPUT.with_suffix('.tiff')}")


if __name__ == "__main__":
    main()
