#!/usr/bin/env python3
"""Draw the revised Model 0923 trial cycle for the model specification.

The diagram links a prepared candidate state to choice prediction,
parallel feedback updates, and preparation of the next trial. It describes the
specification, with no measured or simulated values. Module names and timing
follow the model document; implementation status remains in that document.

Run from the repository root with Python. The default output is the PNG used by
the Markdown and TeX specifications; --output selects a separate preview path.

Figure contract: show one behavioral trial cycle, with each rule's pre-feedback
prediction feeding both choice and feedback evaluation. Memory and precision
update in parallel; hypothesis updating prepares the next trial. Reuse the
existing native schematic structure and restrained colors. Export a 600-dpi PNG
at 8.8 by 7.4 inches for the document's width-limited figure placement.
"""

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as PlotPath


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "src/Bayesian_state/docs/model_architecture/figures/model_0923_framework.png"
)

COLORS = {
    "ink": "#28343B",
    "muted": "#627179",
    "predict": "#356F8A",
    "feedback": "#955A56",
    "state_fill": "#F2F5EE",
    "state_edge": "#718064",
    "perception_fill": "#EDEEF8",
    "perception_edge": "#676DA0",
    "hypothesis_fill": "#E8F1F5",
    "hypothesis_edge": "#3F718A",
    "readout_fill": "#FAEEDC",
    "readout_edge": "#936B38",
    "memory_fill": "#E8F3F3",
    "memory_edge": "#4C8585",
    "precision_fill": "#EEEAF5",
    "precision_edge": "#75678F",
    "feedback_fill": "#F5E6E5",
}


def box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    fill: str,
    edge: str,
    *,
    body_size: float = 11.0,
) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.15,rounding_size=0.65",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.1,
            zorder=3,
        )
    )
    ax.text(
        x + 0.8,
        y + height - 1.2,
        title,
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        color=edge,
        zorder=4,
    )
    ax.text(
        x + width / 2,
        y + height * 0.40,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        linespacing=1.3,
        color=COLORS["ink"],
        zorder=4,
    )


def arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    color: str,
    *,
    dashed: bool = False,
) -> None:
    # Straight segments keep shared inputs and parallel updates explicit.
    path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1))
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.3,
            linestyle=(0, (4, 3)) if dashed else "-",
            color=color,
            zorder=2,
        )
    )


def label(ax: plt.Axes, x: float, y: float, text: str, *, color: str) -> None:
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=9.7,
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
        zorder=5,
    )


def draw(output: Path) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 11.0,
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "dejavusans",
        }
    )
    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.025, top=0.985)
    ax.set_xlim(-4, 101)
    ax.set_ylim(-8, 99)
    ax.axis("off")

    ax.text(
        2, 96, "作答前：用本题候选形成选择", fontsize=13.4,
        fontweight="bold", color=COLORS["predict"], va="center",
    )
    box(
        ax, (2, 66), 21, 22, "本题输入与状态",
        "$\\mathbf{x}_t$：物理刺激\n"
        "$\\mathcal{A}_t,\\pi_t^-$：候选与信念\n"
        "$\\boldsymbol{\\beta}_t$：规则精度\n"
        "$h_{t-1}^{\\star}$：上一主导规则",
        COLORS["state_fill"], COLORS["state_edge"], body_size=10.5,
    )
    box(
        ax, (27, 66), 20, 22, "知觉（P）",
        "形成内部刺激 $\\widetilde{\\mathbf{x}}_t$\n"
        "计算到各类别的距离\n"
        "结合 $\\beta_t(h)$ 得到\n"
        "类别预测 $p_{t,h}(c)$",
        COLORS["perception_fill"], COLORS["perception_edge"], body_size=10.5,
    )
    box(
        ax, (51, 66), 27, 22, "选择读出（RC）",
        "信念与使用惯性形成 $w_t$\n"
        "汇总各规则的类别预测\n"
        "$P(y_t=c\\mid z_t)=\\sum_h w_t(h)p_{t,h}(c)$\n"
        "记录主导规则 $h_t^{\\star}$",
        COLORS["readout_fill"], COLORS["readout_edge"], body_size=10.5,
    )
    box(
        ax, (82, 66), 16, 22, "本题选择",
        "确定选择 $y_t$\n模拟时抽样\n拟合时读取选择\n用于行为拟合",
        COLORS["state_fill"], COLORS["state_edge"], body_size=10.0,
    )
    for start, end in [((23, 77), (27, 77)), ((47, 77), (51, 77)), ((78, 77), (82, 77))]:
        arrow(ax, [start, end], COLORS["predict"])
    arrow(ax, [(12, 88), (12, 91), (56, 91), (56, 88)], COLORS["predict"])
    label(ax, 34, 91, "候选信念与上一主导规则", color=COLORS["predict"])

    # This annotation expands H without adding another causal-process node.
    ax.plot([2, 2], [39, 54], color=COLORS["hypothesis_edge"], lw=1.0, alpha=0.5)
    ax.text(
        3, 53, "H 的搜索来源与容量", fontsize=11.4, fontweight="bold",
        color=COLORS["hypothesis_edge"], va="top",
    )
    ax.text(
        3, 48, "局部：功能相似性与结构编辑\n"
        "全局：规则基础先验 $p_0$\n"
        "候选数量可变，$N_t\\leq M_s^{\\max}$",
        fontsize=10.5, color=COLORS["ink"], va="top", linespacing=1.4,
    )

    box(
        ax, (51, 37), 47, 17, "反馈评价",
        "结合实际选择 $y_t$ 与外部反馈 $r_t$\n"
        "使用反馈前的逐规则类别预测\n"
        "得到反馈似然 $\\ell_t(h)$ 与支持量 $\\kappa_t(h)$",
        COLORS["feedback_fill"], COLORS["feedback"], body_size=10.5,
    )
    arrow(ax, [(37, 66), (37, 58), (65, 58), (65, 54)], COLORS["predict"])
    label(ax, 51, 58, "$p_{t,h}(c)$：逐规则预测", color=COLORS["predict"])
    arrow(ax, [(90, 66), (90, 54)], COLORS["feedback"])
    label(ax, 90, 60, "接收反馈 $r_t$", color=COLORS["feedback"])

    ax.text(
        2, 35.5, "反馈后：学习，并准备下一题", fontsize=13.4,
        fontweight="bold", color=COLORS["feedback"], va="center",
    )
    box(
        ax, (36, 17), 42, 12, "证据记忆（M）",
        "保留旧证据，再加入反馈似然\n"
        "$\\pi_t^+(h)\\propto[\\pi_t^-(h)]^{\\gamma_s}\\ell_t(h)$",
        COLORS["memory_fill"], COLORS["memory_edge"],
    )
    box(
        ax, (36, 3), 42, 12, "精度更新（B）",
        "按反馈支持提高或降低规则精度\n"
        "全部活跃规则都更新，下一题使用",
        COLORS["precision_fill"], COLORS["precision_edge"],
    )
    box(
        ax, (2, 3), 30, 26, "假设更新（H）",
        "更新下一题的失败与搜索状态\n"
        "弱候选退出；局部／全局提议\n"
        "容量满时移除最弱旧候选\n"
        "按规则关系分配初始信念\n"
        "保留旧精度，新规则设初值\n"
        "$\\mathcal{A}_{t+1},\\pi_{t+1}^-,\\boldsymbol{\\beta}_{t+1}$",
        COLORS["hypothesis_fill"], COLORS["hypothesis_edge"], body_size=10.5,
    )
    arrow(ax, [(90, 37), (90, 23), (78, 23)], COLORS["feedback"])
    arrow(ax, [(90, 23), (90, 9), (78, 9)], COLORS["feedback"])
    arrow(ax, [(90, 32), (18, 32), (18, 29)], COLORS["feedback"])
    label(ax, 55, 32, "反馈与近期失败经历", color=COLORS["feedback"])
    arrow(ax, [(36, 23), (32, 23)], COLORS["feedback"])
    arrow(ax, [(36, 9), (32, 9)], COLORS["feedback"])

    arrow(ax, [(2, 16), (0, 16), (0, 77), (2, 77)], COLORS["predict"], dashed=True)
    ax.text(
        -1.8, 46, "进入下一试次", rotation=90, ha="center", va="center",
        fontsize=10.5, color=COLORS["predict"],
    )
    ax.plot([2, 9], [-1, -1], color=COLORS["predict"], lw=1.3, linestyle=(0, (4, 3)))
    ax.text(10, -1, "跨试次状态传递", fontsize=9.7, va="center", color=COLORS["muted"])
    ax.text(
        40, -1, "主导规则 $h_t^{\\star}$ 随状态保存，用于下一题的使用惯性",
        fontsize=9.7, va="center", color=COLORS["muted"],
    )
    ax.text(
        2, -5, "路径推断（PF）：作答前汇总选择预测；按实际选择更新路径权重。",
        fontsize=9.7, va="center", color=COLORS["muted"],
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    draw(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
