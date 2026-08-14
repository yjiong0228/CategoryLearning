#!/usr/bin/env python3
"""Draw the trial-wise framework for the dynamic rule-learning model."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).with_name("model_0813_framework.png")
CJK_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if CJK_FONT.exists():
    font_manager.fontManager.addfont(CJK_FONT)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Noto Sans CJK JP",
            "Noto Sans CJK SC",
            "Arial",
            "Liberation Sans",
            "DejaVu Sans",
            "sans-serif",
        ],
        "font.size": 7.0,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    }
)

COLORS = {
    "ink": "#2B2B2B",
    "muted": "#6B6B6B",
    "line": "#3E3E3E",
    "predict": "#356F8A",
    "post": "#955A56",
    "next": "#356F8A",
    "struct_fill": "#F4F7F0",
    "struct_edge": "#6F7F67",
    "perception_fill": "#E8E9FA",
    "perception_edge": "#676DA0",
    "control_fill": "#E5F0F5",
    "control_edge": "#3F718A",
    "workspace_fill": "#F6E4EE",
    "workspace_edge": "#9A5E7B",
    "readout_fill": "#FAEDDC",
    "readout_edge": "#A16C35",
    "pf_fill": "#E7F4EF",
    "pf_edge": "#527A69",
    "feedback_fill": "#F5E6E5",
    "feedback_edge": "#955A56",
    "likelihood_fill": "#E4F3F3",
    "likelihood_edge": "#4C8585",
    "memory_fill": "#F6ECDE",
    "memory_edge": "#9B6A37",
    "update_fill": "#ECE8F4",
    "update_edge": "#75678F",
    "state_fill": "#EDF3E9",
    "state_edge": "#647E5D",
}


def add_box(
    ax,
    xy,
    width,
    height,
    title,
    body,
    *,
    fill,
    edge,
    title_size=7.0,
    body_size=6.0,
    body_y=0.45,
    radius=0.9,
):
    """Add a pastel rounded module box with a direct label."""
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.30,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.05,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.8,
        y + height - 1.25,
        title,
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=edge,
        zorder=4,
    )
    ax.text(
        x + width / 2,
        y + height * body_y,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=COLORS["ink"],
        linespacing=1.22,
        zorder=4,
    )
    return patch


def add_arrow(
    ax,
    start,
    end,
    *,
    color=None,
    lw=1.15,
    linestyle="-",
    connectionstyle="arc3",
    mutation_scale=8.5,
    zorder=2,
):
    """Add a directional connector."""
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=linestyle,
        color=color or COLORS["line"],
        connectionstyle=connectionstyle,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def add_arrow_label(ax, xy, text, *, color=None, size=5.5):
    ax.text(
        *xy,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color or COLORS["muted"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.94},
        zorder=8,
    )


def draw_framework():
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 72)
    ax.axis("off")

    # Phase labels keep the causal boundary visible.
    ax.text(
        2.0,
        69.2,
        "作答前：只使用上一试次及更早的信息",
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=COLORS["predict"],
    )
    ax.plot([2.0, 118.0], [67.5, 67.5], color="#C9DCE5", lw=1.0, zorder=0)

    # A light container marks the latent computation repeated in every particle.
    particle_band = FancyBboxPatch(
        (17.8, 42.7),
        79.7,
        22.2,
        boxstyle="round,pad=0.35,rounding_size=1.3",
        facecolor="#FBFCFD",
        edgecolor="#B8C5CC",
        linewidth=0.85,
        linestyle=(0, (3.0, 2.5)),
        zorder=0,
    )
    ax.add_patch(particle_band)
    ax.text(
        19.0,
        63.6,
        "每个粒子的潜在路径",
        ha="left",
        va="center",
        fontsize=5.8,
        color="#78878E",
        zorder=1,
    )

    # Pre-choice path.
    add_box(
        ax,
        (1.8, 47.5),
        13.2,
        13.8,
        "试次输入",
        "物理刺激\n$\\mathbf{x}_t$",
        fill=COLORS["struct_fill"],
        edge=COLORS["struct_edge"],
        body_size=6.4,
    )
    add_box(
        ax,
        (20.0, 47.5),
        13.5,
        13.8,
        "个体化知觉",
        "感知噪声\n$\\widetilde{\\mathbf{x}}_t$",
        fill=COLORS["perception_fill"],
        edge=COLORS["perception_edge"],
        body_size=6.1,
    )
    add_box(
        ax,
        (37.0, 47.5),
        15.5,
        13.8,
        "搜索控制",
        "$F_{t-1},\\ M_{t-1}$\n$\\Downarrow$\n$E_t$：是否搜索　$g_t$：搜索多远",
        fill=COLORS["control_fill"],
        edge=COLORS["control_edge"],
        body_size=5.55,
        body_y=0.43,
    )
    add_box(
        ax,
        (56.0, 45.8),
        19.5,
        17.2,
        "有限规则工作空间",
        "$A_t$：3 或 5 条候选规则\n局部 / 全局替换\n$e_t$：唯一执行规则",
        fill=COLORS["workspace_fill"],
        edge=COLORS["workspace_edge"],
        body_size=5.9,
        body_y=0.43,
    )
    # Small chips make the distinction between candidate rules and the executed rule visual.
    for idx, (cx, label, selected) in enumerate(
        [(58.0, "$h_1$", False), (62.1, "$h_2$", True), (66.2, "$h_3$", False)]
    ):
        chip = FancyBboxPatch(
            (cx, 47.2),
            3.2,
            2.5,
            boxstyle="round,pad=0.15,rounding_size=0.45",
            facecolor="#A95F84" if selected else "#FFF9FC",
            edgecolor="#8D5572",
            linewidth=0.7,
            zorder=5,
        )
        ax.add_patch(chip)
        ax.text(
            cx + 1.6,
            48.45,
            label,
            ha="center",
            va="center",
            fontsize=5.0,
            color="white" if selected else COLORS["ink"],
            zorder=6,
        )
    ax.text(72.0, 48.45, "$\\cdots$", ha="center", va="center", fontsize=6.5, color="#8D5572", zorder=6)

    add_box(
        ax,
        (79.0, 47.5),
        16.2,
        13.8,
        "执行规则读出",
        "$q_{t,e_t}(c)$\n动态 $\\beta_t(e_t)$\n掌握增强 + 2\\% 反应噪声",
        fill=COLORS["readout_fill"],
        edge=COLORS["readout_edge"],
        body_size=5.6,
        body_y=0.42,
    )
    add_box(
        ax,
        (100.0, 46.3),
        17.8,
        16.2,
        "粒子滤波与行为",
        "$\\overline{p}_t(c)=\\sum_r w_{t-1}^{(r)}p_t^{(r)}(c)$\n观察选择 $y_t$\n$w_t^{(r)}\\propto w_{t-1}^{(r)}p_t^{(r)}(y_t)$",
        fill=COLORS["pf_fill"],
        edge=COLORS["pf_edge"],
        body_size=5.25,
        body_y=0.42,
    )

    top_y = 54.4
    add_arrow(ax, (15.0, top_y), (20.0, top_y), color=COLORS["predict"])
    add_arrow(ax, (33.5, top_y), (37.0, top_y), color=COLORS["predict"])
    add_arrow(ax, (52.5, top_y), (56.0, top_y), color=COLORS["predict"])
    add_arrow(ax, (75.5, top_y), (79.0, top_y), color=COLORS["predict"])
    add_arrow(ax, (95.2, top_y), (100.0, top_y), color=COLORS["predict"])
    add_arrow_label(ax, (17.5, 56.2), "$\\mathbf{x}_t$", color=COLORS["predict"])
    add_arrow_label(ax, (54.2, 56.2), "$E_t,g_t$", color=COLORS["predict"])
    add_arrow_label(ax, (77.2, 56.2), "$e_t$", color=COLORS["predict"])

    # Post-choice path.
    ax.text(
        2.0,
        37.9,
        "反馈后：更新状态，供下一试次使用",
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        color=COLORS["post"],
    )
    ax.plot([2.0, 118.0], [36.2, 36.2], color="#E6CFCD", lw=1.0, zorder=0)

    add_box(
        ax,
        (99.8, 15.2),
        18.0,
        15.8,
        "选择与反馈",
        "$(y_t,r_t)$\n本试次结果只进入\n反馈后更新",
        fill=COLORS["feedback_fill"],
        edge=COLORS["feedback_edge"],
        body_size=5.8,
        body_y=0.42,
    )
    add_box(
        ax,
        (78.7, 15.2),
        16.8,
        15.8,
        "反馈支持",
        "$L_t(h)$\n正确：$q_{t,h}(y_t)$\n错误：$1-q_{t,h}(y_t)$",
        fill=COLORS["likelihood_fill"],
        edge=COLORS["likelihood_edge"],
        body_size=5.65,
        body_y=0.43,
    )
    add_box(
        ax,
        (56.2, 13.6),
        18.6,
        18.8,
        "双通道记忆",
        "近期证据 $D_t(h)$\n长期证据 $C_t(h)$\n$\\Downarrow$\n规则信念 $\\pi_t^{+}(h)$",
        fill=COLORS["memory_fill"],
        edge=COLORS["memory_edge"],
        body_size=5.75,
        body_y=0.43,
    )
    add_box(
        ax,
        (33.9, 13.6),
        18.2,
        18.8,
        "反馈状态更新",
        "失败压力 $F_t$\n掌握证据 $M_t$\n仅更新执行规则\n$\\beta_{t+1}(e_t)$",
        fill=COLORS["update_fill"],
        edge=COLORS["update_edge"],
        body_size=5.65,
        body_y=0.42,
    )
    add_box(
        ax,
        (8.2, 13.6),
        20.8,
        18.8,
        "下一试次的完整状态",
        "$\\pi_t^{+},\\ F_t,\\ M_t$\n$e_t,\\ A_t,\\beta_{t+1}$\n$\\Downarrow$\n试次 $t+1$",
        fill=COLORS["state_fill"],
        edge=COLORS["state_edge"],
        body_size=5.9,
        body_y=0.42,
    )

    # Outcome enters only after the pre-choice marginal has been evaluated.
    add_arrow(
        ax,
        (108.9, 46.3),
        (108.9, 31.0),
        color=COLORS["post"],
        connectionstyle="arc3,rad=0.0",
    )
    add_arrow_label(ax, (112.0, 38.0), "获得反馈 $r_t$", color=COLORS["post"], size=5.3)
    add_arrow(ax, (99.8, 23.1), (95.5, 23.1), color=COLORS["post"])
    add_arrow(ax, (78.7, 23.1), (74.8, 23.1), color=COLORS["post"])
    add_arrow_label(ax, (77.0, 25.0), "$L_t(h)$", color=COLORS["post"])
    add_arrow(
        ax,
        (108.8, 15.2),
        (43.0, 13.6),
        color=COLORS["post"],
        connectionstyle="arc3,rad=0.19",
    )
    add_arrow_label(ax, (76.0, 9.0), "反馈更新 $F_t,M_t,\\beta_{t+1}(e_t)$", color=COLORS["post"], size=5.2)
    add_arrow(ax, (56.2, 24.5), (52.1, 24.5), color=COLORS["post"])
    add_arrow_label(ax, (54.1, 26.4), "$\\pi_t^{+}$", color=COLORS["post"])
    add_arrow(ax, (33.9, 23.0), (29.0, 23.0), color=COLORS["post"])

    # Dashed recursion makes the trial boundary explicit and prevents feedback leakage.
    add_arrow(
        ax,
        (18.6, 32.4),
        (44.7, 47.5),
        color=COLORS["next"],
        lw=1.25,
        linestyle=(0, (4.0, 3.0)),
        connectionstyle="arc3,rad=-0.22",
    )
    add_arrow(
        ax,
        (22.5, 32.4),
        (65.7, 45.8),
        color=COLORS["next"],
        lw=1.25,
        linestyle=(0, (4.0, 3.0)),
        connectionstyle="arc3,rad=-0.15",
    )
    add_arrow_label(ax, (30.5, 40.1), "下一试次", color=COLORS["next"], size=5.5)

    # Minimal line key, mirroring the restrained modular style of the reference diagram.
    ax.plot([2.2, 8.0], [6.5, 6.5], color=COLORS["predict"], lw=1.2)
    ax.text(9.2, 6.5, "作答前流程", va="center", ha="left", fontsize=5.4, color=COLORS["muted"])
    ax.plot([27.0, 32.8], [6.5, 6.5], color=COLORS["post"], lw=1.2)
    ax.text(34.0, 6.5, "反馈后更新", va="center", ha="left", fontsize=5.4, color=COLORS["muted"])
    ax.plot([52.8, 58.6], [6.5, 6.5], color=COLORS["next"], lw=1.2, linestyle=(0, (4.0, 3.0)))
    ax.text(59.8, 6.5, "跨试次递归", va="center", ha="left", fontsize=5.4, color=COLORS["muted"])

    fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.02)
    fig.savefig(OUT, dpi=600, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    draw_framework()
