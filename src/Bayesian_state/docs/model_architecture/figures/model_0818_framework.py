#!/usr/bin/env python3
"""Draw the trial-level causal architecture of the finite rule-search model."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).with_name("model_0818_framework.png")
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
    "off_fill": "#FFF8ED",
    "on_fill": "#F6EDE1",
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


def box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body,
    fill,
    edge,
    *,
    body_size=9.6,
    title_size=10.6,
):
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
        fontsize=title_size,
        fontweight="bold",
        color=edge,
        zorder=4,
    )
    ax.text(
        x + w / 2,
        y + h * 0.43,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=COLORS["ink"],
        linespacing=1.20,
        zorder=4,
    )


def readout_box(ax, x, y, w, h):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.25,rounding_size=0.8",
        facecolor=COLORS["action_fill"],
        edgecolor=COLORS["action_edge"],
        linewidth=1.05,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.65,
        y + h - 0.85,
        "边界发射与选择读出",
        ha="left",
        va="top",
        fontsize=10.6,
        fontweight="bold",
        color=COLORS["action_edge"],
        zorder=5,
    )
    ax.text(
        x + w / 2,
        y + h - 4.9,
        "边界距离 + 统一动态 $\\beta_t(h)$",
        ha="center",
        va="center",
        fontsize=8.8,
        color=COLORS["ink"],
        zorder=5,
    )

    inner_h = 3.7
    inner_y = [y + 5.2, y + 0.9]
    inner_specs = [
        (
            COLORS["off_fill"],
            "$\\chi_s=0$  信念加权：  $\\sum_h\\pi_t^-(h)p_{t,h}(c)$",
        ),
        (
            COLORS["on_fill"],
            "$\\chi_s=1$  持续执行：  $p_{t,e_t}(c)$",
        ),
    ]
    for y0, (fill, text) in zip(inner_y, inner_specs):
        inner = FancyBboxPatch(
            (x + 0.8, y0),
            w - 1.6,
            inner_h,
            boxstyle="round,pad=0.08,rounding_size=0.42",
            facecolor=fill,
            edgecolor="#C8A678",
            linewidth=0.65,
            zorder=4,
        )
        ax.add_patch(inner)
        ax.text(
            x + w / 2,
            y0 + inner_h / 2,
            text,
            ha="center",
            va="center",
            fontsize=7.7,
            color=COLORS["ink"],
            zorder=5,
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
        fontsize=8.4,
        color=color,
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "pad": 0.35,
            "alpha": 0.94,
        },
        zorder=7,
    )


def draw() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    ax.set_xlim(0, 132)
    ax.set_ylim(0, 78)
    ax.axis("off")

    ax.text(
        2,
        75.0,
        "作答前：生成在线选择概率（不使用本试次选择与反馈）",
        fontsize=11.8,
        fontweight="bold",
        color=COLORS["predict"],
        ha="left",
        va="center",
    )
    ax.plot([2, 130], [73.1, 73.1], color="#C9DCE5", lw=1.0)

    particle_band = FancyBboxPatch(
        (15.4, 42.7),
        87.1,
        27.5,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor="#FBFCFD",
        edgecolor="#B8C5CC",
        linewidth=0.9,
        linestyle=(0, (3, 2.5)),
        zorder=0,
    )
    ax.add_patch(particle_band)
    ax.text(
        16.7,
        68.6,
        "每个 PF 粒子的潜在认知路径",
        color="#78878E",
        fontsize=8.8,
    )

    top_y, top_h = 46.0, 19.4
    box(
        ax,
        1.8,
        top_y,
        11.2,
        top_h,
        "试次输入",
        "物理刺激\n$\\mathbf{x}_t$",
        COLORS["input_fill"],
        COLORS["input_edge"],
        body_size=10.2,
    )
    box(
        ax,
        17.4,
        top_y,
        12.4,
        top_h,
        "个体化知觉",
        "被试特异噪声\n$\\widetilde{\\mathbf{x}}_t$",
        COLORS["perception_fill"],
        COLORS["perception_edge"],
        body_size=9.5,
    )
    box(
        ax,
        33.2,
        top_y,
        15.2,
        top_h,
        "嵌套搜索控制",
        "$F_t$：失败累积\n$E_t$：是否搜索\n$g_t$：局部 / 全局范围\n$c_A=0$ 精确回到 reactive",
        COLORS["control_fill"],
        COLORS["control_edge"],
        body_size=7.7,
    )
    box(
        ax,
        51.9,
        top_y,
        18.0,
        top_h,
        "有限规则工作空间",
        "$M_s$ 个固定槽位\n低信念规则优先离开\n局部 / 全局新人提议\nsimilarity transport 得到 $\\pi_t^-$",
        COLORS["workspace_fill"],
        COLORS["workspace_edge"],
        body_size=7.7,
    )
    readout_box(ax, 73.5, top_y, 25.3, top_h)
    box(
        ax,
        107.0,
        top_y,
        22.5,
        top_h,
        "PF 边际预测",
        "$\\widehat p_t(c)=\\sum_i w_{t-1}^{(i)}p_t^{(i)}(c)$\n\n对不可见路径在线积分",
        COLORS["pf_fill"],
        COLORS["pf_edge"],
        body_size=8.4,
    )

    arrow(ax, (13.0, 55.7), (17.4, 55.7), COLORS["predict"])
    arrow(ax, (29.8, 55.7), (33.2, 55.7), COLORS["predict"])
    arrow(ax, (48.4, 55.7), (51.9, 55.7), COLORS["predict"])
    arrow(ax, (69.9, 55.7), (73.5, 55.7), COLORS["predict"])
    arrow(ax, (98.8, 55.7), (107.0, 55.7), COLORS["predict"])

    ax.text(
        2,
        39.0,
        "作答后：选择校正 PF 权重；选择与反馈更新认知状态",
        fontsize=11.8,
        fontweight="bold",
        color=COLORS["post"],
        ha="left",
        va="center",
    )
    ax.plot([2, 130], [37.1, 37.1], color="#E4CDCB", lw=1.0)

    low_y, low_h = 10.1, 20.3
    box(
        ax,
        101.2,
        low_y,
        28.3,
        low_h,
        "观察选择、反馈并校正 PF",
        "$(y_t,r_t)$\n$y_t:\;\\widetilde w_t^{(i)}\\propto w_{t-1}^{(i)}p_t^{(i)}(y_t)$\nESS 低于阈值时系统重采样\n$r_t$ 不单独改变 PF 权重",
        COLORS["feedback_fill"],
        COLORS["feedback_edge"],
        body_size=7.8,
    )
    box(
        ax,
        78.6,
        low_y,
        18.4,
        low_h,
        "统一规则证据",
        "复用同一 $\\beta_t(h)$\n$\\ell_t(h)=r_tp_{t,h}(y_t)$\n$\quad +(1-r_t)[1-p_{t,h}(y_t)]$\n活跃规则内归一化为 $L_t(h)$",
        COLORS["evidence_fill"],
        COLORS["evidence_edge"],
        body_size=7.4,
    )
    box(
        ax,
        55.5,
        low_y,
        18.9,
        low_h,
        "衰减证据记忆",
        "$\\pi_t^+(h)\\propto$\n$[\\pi_t^-(h)]^{\\gamma_s}L_t(h)$\n静态记忆权重固定为 0",
        COLORS["memory_fill"],
        COLORS["memory_edge"],
        body_size=8.4,
    )
    box(
        ax,
        29.9,
        low_y,
        21.4,
        low_h,
        "确信度与反馈状态",
        "更新全部活跃规则的 $\\beta_{t+1}(h)$\n支持则增加，反驳则降低\n新人重置为 $\\beta_{0,s}$\n登记 $r_t$，供 $F_{t+1}$ 使用",
        COLORS["update_fill"],
        COLORS["update_edge"],
        body_size=7.8,
    )
    box(
        ax,
        3.0,
        low_y,
        22.7,
        low_h,
        "下一试次完整状态",
        "$A_t,\\pi_t^+,\\boldsymbol{\\beta}_{t+1},F_{t+1}$\n$e_t$（仅 execution-on）\n\n进入试次 $t+1$",
        COLORS["input_fill"],
        COLORS["input_edge"],
        body_size=8.4,
    )

    arrow(ax, (118.2, 46.0), (118.2, 30.4), COLORS["post"])
    arrow(ax, (101.2, 20.2), (97.0, 20.2), COLORS["post"])
    arrow(ax, (78.6, 20.2), (74.4, 20.2), COLORS["post"])
    arrow(ax, (55.5, 20.2), (51.3, 20.2), COLORS["post"])
    arrow(ax, (29.9, 20.2), (25.7, 20.2), COLORS["post"])

    arrow(
        ax,
        (14.2, 30.4),
        (40.8, 46.0),
        COLORS["predict"],
        dashed=True,
        connectionstyle="arc3,rad=-0.18",
    )
    arrow(
        ax,
        (14.2, 30.4),
        (60.7, 46.0),
        COLORS["predict"],
        dashed=True,
        connectionstyle="arc3,rad=-0.10",
    )
    label(ax, 30.3, 35.0, "跨试次递归", COLORS["predict"])

    ax.text(
        2.2,
        5.5,
        "认知层：每个粒子保存一条完整随机状态路径；数值层：PF 给出 predictive 与 filtered 在线分布，不使用未来试次做 smoothing。",
        fontsize=9.0,
        color=COLORS["muted"],
        ha="left",
        va="center",
    )
    ax.plot([2.2, 8.0], [2.1, 2.1], color=COLORS["predict"], lw=1.5)
    ax.text(8.8, 2.1, "作答前流程", fontsize=8.3, color=COLORS["muted"], va="center")
    ax.plot([25.0, 30.8], [2.1, 2.1], color=COLORS["post"], lw=1.5)
    ax.text(31.6, 2.1, "作答后更新", fontsize=8.3, color=COLORS["muted"], va="center")
    ax.plot(
        [48.0, 54.0],
        [2.1, 2.1],
        color=COLORS["predict"],
        lw=1.5,
        linestyle=(0, (4, 3)),
    )
    ax.text(54.8, 2.1, "跨试次递归", fontsize=8.3, color=COLORS["muted"], va="center")

    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    draw()
    print(f"Saved: {OUT}")
