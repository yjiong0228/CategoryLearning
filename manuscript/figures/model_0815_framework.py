#!/usr/bin/env python3
"""Draw the causal architecture of the unfitted Model 0815 P0 candidate."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).with_name("model_0815_framework.png")
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
            "DejaVu Sans",
            "sans-serif",
        ],
        "font.size": 7.0,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
    }
)

COLORS = {
    "ink": "#2B2B2B",
    "muted": "#6F777B",
    "predict": "#356F8A",
    "post": "#955A56",
    "input_fill": "#F2F5EE",
    "input_edge": "#718064",
    "perception_fill": "#E8E9FA",
    "perception_edge": "#676DA0",
    "control_fill": "#E5F0F5",
    "control_edge": "#3F718A",
    "workspace_fill": "#F6E4EE",
    "workspace_edge": "#9A5E7B",
    "action_fill": "#FAEDDC",
    "action_edge": "#A16C35",
    "pf_fill": "#E7F4EF",
    "pf_edge": "#527A69",
    "feedback_fill": "#F5E6E5",
    "feedback_edge": "#955A56",
    "evidence_fill": "#E4F3F3",
    "evidence_edge": "#4C8585",
    "memory_fill": "#F6ECDE",
    "memory_edge": "#9B6A37",
    "update_fill": "#ECE8F4",
    "update_edge": "#75678F",
}


def box(ax, x, y, w, h, title, body, fill, edge, body_size=6.2):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.25,rounding_size=0.8",
        facecolor=fill,
        edgecolor=edge,
        linewidth=1.05,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.65,
        y + h - 0.85,
        title,
        ha="left",
        va="top",
        fontsize=11.2,
        fontweight="bold",
        color=edge,
        zorder=4,
    )
    ax.text(
        x + w / 2,
        y + h * 0.44,
        body,
        ha="center",
        va="center",
        fontsize=body_size * 1.75,
        color=COLORS["ink"],
        linespacing=1.22,
        zorder=4,
    )


def arrow(ax, start, end, color, *, dashed=False, connectionstyle="arc3"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.2,
            linestyle=(0, (4, 3)) if dashed else "-",
            color=color,
            connectionstyle=connectionstyle,
            shrinkA=0,
            shrinkB=0,
            zorder=2,
        )
    )


def label(ax, x, y, text, color):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8.8,
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.94},
        zorder=7,
    )


def draw() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 72)
    ax.axis("off")

    ax.text(
        2,
        69.2,
        "作答前：生成选择概率（不使用本试次选择与反馈）",
        fontsize=11.8,
        fontweight="bold",
        color=COLORS["predict"],
        ha="left",
        va="center",
    )
    ax.plot([2, 118], [67.5, 67.5], color="#C9DCE5", lw=1.0)

    particle_band = FancyBboxPatch(
        (16.5, 40.0),
        81.0,
        24.5,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor="#FBFCFD",
        edgecolor="#B8C5CC",
        linewidth=0.9,
        linestyle=(0, (3, 2.5)),
        zorder=0,
    )
    ax.add_patch(particle_band)
    ax.text(17.8, 63.2, "每个 PF 粒子的潜在认知路径", color="#78878E", fontsize=8.8)

    top_y, top_h = 44.0, 16.5
    box(ax, 1.8, top_y, 12.0, top_h, "试次输入", "物理刺激\n$\\mathbf{x}_t$", COLORS["input_fill"], COLORS["input_edge"])
    box(ax, 19.0, top_y, 12.7, top_h, "个体化知觉", "感知噪声\n$\\widetilde{\\mathbf{x}}_t$", COLORS["perception_fill"], COLORS["perception_edge"])
    box(ax, 35.0, top_y, 14.5, top_h, "错误驱动控制", "$F_{t-1},M_{t-1}$\n$E_t$：是否搜索\n$g_t$：搜索多远", COLORS["control_fill"], COLORS["control_edge"], 5.8)
    box(ax, 53.0, top_y, 15.0, top_h, "有限工作空间", "$A_t$：固定 3 条规则\n局部 / 全局替换\n$e_t$：唯一执行规则", COLORS["workspace_fill"], COLORS["workspace_edge"], 5.8)
    box(ax, 71.5, top_y, 14.0, top_h, "动作发射", "语义边界距离\n动态 $\\beta_t(e_t)$\n$2\\%$ 均匀 lapse", COLORS["action_fill"], COLORS["action_edge"], 5.8)
    box(ax, 101.0, top_y, 17.0, top_h, "PF 边际预测", "$\\bar p_t(c)=\\sum_r w_{t-1}^{(r)}p_t^{(r)}(c)$\n观察选择后更新权重", COLORS["pf_fill"], COLORS["pf_edge"], 5.5)

    arrow(ax, (13.8, 52.2), (19.0, 52.2), COLORS["predict"])
    arrow(ax, (31.7, 52.2), (35.0, 52.2), COLORS["predict"])
    arrow(ax, (49.5, 52.2), (53.0, 52.2), COLORS["predict"])
    arrow(ax, (68.0, 52.2), (71.5, 52.2), COLORS["predict"])
    arrow(ax, (85.5, 52.2), (101.0, 52.2), COLORS["predict"])

    ax.text(
        2,
        36.7,
        "作答后：选择用于 PF 校正；选择与反馈用于认知状态更新",
        fontsize=11.8,
        fontweight="bold",
        color=COLORS["post"],
        ha="left",
        va="center",
    )
    ax.plot([2, 118], [35.0, 35.0], color="#E4CDCB", lw=1.0)

    low_y, low_h = 10.0, 18.0
    box(ax, 101.0, low_y, 17.0, low_h, "观察选择与反馈", "$(y_t,r_t)$\n本试次预测保持不变", COLORS["feedback_fill"], COLORS["feedback_edge"], 6.1)
    box(ax, 78.0, low_y, 17.0, low_h, "固定尺度规则证据", "边界距离 + $\\beta_{\\rm ev}$\n得到 $L_t(h)$\n不读取动态动作 $\\beta_t$", COLORS["evidence_fill"], COLORS["evidence_edge"], 5.7)
    box(ax, 55.0, low_y, 17.0, low_h, "双通道记忆", "近期证据 $D_t(h)$\n长期证据 $C_t(h)$\n规则后验 $\\pi_t^+(h)$", COLORS["memory_fill"], COLORS["memory_edge"], 5.8)
    box(ax, 31.5, low_y, 17.0, low_h, "反馈状态更新", "更新 $F_t,M_t$\n只更新执行规则的\n动作确信度 $\\beta_{t+1}(e_t)$", COLORS["update_fill"], COLORS["update_edge"], 5.7)
    box(ax, 6.5, low_y, 18.5, low_h, "下一试次完整状态", "$A_t,e_t,\\pi_t^+$\n$F_t,M_t,\\beta_{t+1}$\n进入试次 $t+1$", COLORS["input_fill"], COLORS["input_edge"], 5.9)

    arrow(ax, (109.5, 44.0), (109.5, 28.0), COLORS["post"])
    arrow(ax, (101.0, 19.0), (95.0, 19.0), COLORS["post"])
    arrow(ax, (78.0, 19.0), (72.0, 19.0), COLORS["post"])
    arrow(ax, (55.0, 19.0), (48.5, 19.0), COLORS["post"])
    arrow(ax, (31.5, 19.0), (25.0, 19.0), COLORS["post"])

    arrow(
        ax,
        (15.7, 28.0),
        (42.0, 44.0),
        COLORS["predict"],
        dashed=True,
        connectionstyle="arc3,rad=-0.18",
    )
    arrow(
        ax,
        (15.7, 28.0),
        (60.5, 44.0),
        COLORS["predict"],
        dashed=True,
        connectionstyle="arc3,rad=-0.10",
    )
    label(ax, 28.5, 34.0, "跨试次递归", COLORS["predict"])

    ax.text(
        2.2,
        4.7,
        "认知层：每个粒子保存一条完整状态路径；数值层：PF 对路径积分。当前只做 filtering，不做 smoothing。",
        fontsize=9.2,
        color=COLORS["muted"],
        ha="left",
        va="center",
    )
    ax.plot([2.2, 8.0], [1.7, 1.7], color=COLORS["predict"], lw=1.5)
    ax.text(8.8, 1.7, "作答前流程", fontsize=8.5, color=COLORS["muted"], va="center")
    ax.plot([25.0, 30.8], [1.7, 1.7], color=COLORS["post"], lw=1.5)
    ax.text(31.6, 1.7, "作答后更新", fontsize=8.5, color=COLORS["muted"], va="center")
    ax.plot([48.0, 54.0], [1.7, 1.7], color=COLORS["predict"], lw=1.5, linestyle=(0, (4, 3)))
    ax.text(54.8, 1.7, "跨试次递归", fontsize=8.5, color=COLORS["muted"], va="center")

    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
    print(f"Saved: {OUT}")
