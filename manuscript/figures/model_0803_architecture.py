#!/usr/bin/env python3
"""Create the publication-grade architecture schematic for model_0803."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTDIR = Path(__file__).resolve().parent
STEM = OUTDIR / "model_0803_architecture"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    }
)


COLORS = {
    "ink": "#262626",
    "muted": "#666666",
    "line": "#444444",
    "light_line": "#A8A8A8",
    "struct_fill": "#F7F7F5",
    "struct_edge": "#777777",
    "h_fill": "#E7EFF5",
    "h_edge": "#3D6F8A",
    "memory_fill": "#F5EBDD",
    "memory_edge": "#996735",
    "obs_fill": "#F2F2F2",
    "obs_edge": "#5A5A5A",
    "feedback_fill": "#F4E7E6",
    "feedback_edge": "#925B58",
    "state_fill": "#EDF2EA",
    "state_edge": "#607A59",
    "validation_fill": "#F8F8F8",
    "next_trial": "#2F6D8B",
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
    title_color=None,
    title_size=7.2,
    body_size=6.1,
    body_y=0.52,
    linewidth=1.0,
    radius=0.9,
    zorder=3,
):
    """Draw a rounded module box with a title and compact body."""
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.32,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
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
        color=title_color or edge,
        zorder=zorder + 1,
    )
    ax.text(
        x + width / 2,
        y + height * body_y,
        body,
        ha="center",
        va="center",
        fontsize=body_size,
        color=COLORS["ink"],
        linespacing=1.28,
        zorder=zorder + 1,
    )
    return patch


def add_arrow(
    ax,
    start,
    end,
    *,
    color=None,
    lw=1.15,
    style="-|>",
    linestyle="-",
    connectionstyle="arc3",
    mutation_scale=8.5,
    zorder=2,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
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


def add_arrow_label(ax, xy, text, *, color=None, size=5.6, ha="center"):
    ax.text(
        *xy,
        text,
        ha=ha,
        va="center",
        fontsize=size,
        color=color or COLORS["muted"],
        bbox=dict(facecolor="white", edgecolor="none", pad=0.7, alpha=0.94),
        zorder=8,
    )


def add_panel_heading(ax, x, y, letter, heading):
    ax.text(x, y, letter, ha="left", va="top", fontsize=9.0, fontweight="bold", color=COLORS["ink"])
    ax.text(
        x + 2.3,
        y,
        heading,
        ha="left",
        va="top",
        fontsize=8.2,
        fontweight="bold",
        color=COLORS["ink"],
    )


def draw_figure():
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 82)
    ax.axis("off")

    add_panel_heading(ax, 1.0, 81.0, "a", "Trial-wise generative architecture")

    # Structural inputs -----------------------------------------------------
    add_box(
        ax,
        (2.0, 58.0),
        19.5,
        17.0,
        "Rule structure",
        "Labeled hypotheses  $\\mathcal{H}_k$\n"
        "base prior  $p_{0,k}$\n"
        "rule distance  $d_k$\n"
        "$\\Downarrow$  $K_{L,k},\\ K_{G,k}$",
        fill=COLORS["struct_fill"],
        edge=COLORS["struct_edge"],
        body_size=6.15,
    )
    add_box(
        ax,
        (2.0, 39.0),
        19.5,
        14.0,
        "Trial input",
        "Physical stimulus  $\\mathbf{x}_t$\n"
        "participant perception  $g_s$\n"
        "$\\Downarrow$  $q_{t,h}(c)$\n"
        "task truth / condition  $k$",
        fill=COLORS["struct_fill"],
        edge=COLORS["struct_edge"],
        body_size=6.1,
    )
    ax.text(
        11.8,
        36.1,
        "Frozen rule geometry and perceptual model",
        ha="center",
        va="center",
        fontsize=5.4,
        color=COLORS["muted"],
    )

    # Previous state and H module ------------------------------------------
    add_box(
        ax,
        (26.0, 61.0),
        17.0,
        13.2,
        "Previous state",
        "$\\pi^+_{t-1},\\ F_{t-1},\\ S_{t-1}$\n"
        "$\\mathbf{a}_{t-1}$\n"
        "$S^{\\mathrm{fb}}_{t-1},\\ U^{\\mathrm{rule}}_{t-1}$",
        fill=COLORS["state_fill"],
        edge=COLORS["state_edge"],
        body_size=6.3,
    )
    add_box(
        ax,
        (48.0, 59.0),
        21.5,
        17.2,
        "H strategy dynamics",
        "$\\mathbf{a}_t=\\boldsymbol{\\mu}_H+\\boldsymbol{\\Phi}_H\\mathbf{a}_{t-1}$\n"
        "$\quad+\\mathbf{B}_H(S^{\\mathrm{fb}}_{t-1},U^{\\mathrm{rule}}_{t-1})$\n"
        "$m_t$: revision mass\n"
        "$g_t$: global-search share",
        fill=COLORS["h_fill"],
        edge=COLORS["h_edge"],
        body_size=5.9,
        body_y=0.47,
    )
    add_box(
        ax,
        (74.0, 58.0),
        24.0,
        19.0,
        "H transition and dynamic prior",
        "$T_t=(1-m_t)I+m_t(1-g_t)K_L+m_tg_tK_G$\n"
        "$\\pi^-_t(h')=\\sum_h\\pi^+_{t-1}(h)T_t(h'\\mid h)$\n"
        "Soft redistribution over the full hypothesis space\n"
        "(not hard hypothesis selection)",
        fill=COLORS["h_fill"],
        edge=COLORS["h_edge"],
        body_size=5.65,
        body_y=0.45,
    )
    add_arrow(ax, (43.0, 67.6), (48.0, 67.6), color=COLORS["h_edge"])
    add_arrow(ax, (69.5, 67.6), (74.0, 67.6), color=COLORS["h_edge"])
    add_arrow_label(ax, (45.5, 69.0), "history")

    # Structural arrows to H and initialization.
    add_arrow(
        ax,
        (21.5, 70.8),
        (74.0, 73.4),
        color=COLORS["light_line"],
        lw=0.9,
        linestyle=(0, (2.2, 2.2)),
        connectionstyle="arc3,rad=-0.08",
        mutation_scale=7.0,
    )
    add_arrow(
        ax,
        (21.5, 63.0),
        (26.0, 63.8),
        color=COLORS["light_line"],
        lw=0.85,
        linestyle=(0, (2.2, 2.2)),
        mutation_scale=7.0,
    )
    add_arrow_label(ax, (23.8, 61.8), "$t=1$")

    # Pre-feedback emissions ------------------------------------------------
    add_box(
        ax,
        (102.0, 50.0),
        16.0,
        27.0,
        "Pre-feedback emissions",
        "",
        fill=COLORS["obs_fill"],
        edge=COLORS["obs_edge"],
        body_size=5.7,
    )
    for y in (67.1, 59.5):
        ax.plot([103.0, 117.0], [y, y], color="#C8C8C8", lw=0.7, zorder=4)
    ax.text(110.0, 71.2, "Choice", ha="center", va="center", fontsize=6.5, fontweight="bold", color=COLORS["ink"], zorder=5)
    ax.text(110.0, 68.8, "$\\pi^-_t \\times q_t \\to C_t\\;(\\kappa)$", ha="center", va="center", fontsize=5.7, color=COLORS["ink"], zorder=5)
    ax.text(110.0, 63.9, "RT", ha="center", va="center", fontsize=6.5, fontweight="bold", color=COLORS["ink"], zorder=5)
    ax.text(110.0, 61.4, "$U^{\\rm choice}_t,\\ m_t(1-g_t),\\ m_tg_t$\n$\\to\\;\\mathrm{RT}_t$", ha="center", va="center", fontsize=5.0, color=COLORS["ink"], linespacing=1.15, zorder=5)
    ax.text(110.0, 56.1, "Oral report", ha="center", va="center", fontsize=6.5, fontweight="bold", color=COLORS["ink"], zorder=5)
    ax.text(110.0, 53.3, "$\\pi^-_t \\times q_t(C_t) \\to O_t\\;(\\eta_O)$", ha="center", va="center", fontsize=5.45, color=COLORS["ink"], zorder=5)
    add_arrow(ax, (98.0, 67.6), (102.0, 67.6), color=COLORS["line"])
    add_arrow_label(ax, (100.0, 69.0), "$\\pi^-_t$")

    # Perceptual prediction reaches emissions without changing H.
    add_arrow(
        ax,
        (21.5, 46.0),
        (102.0, 53.2),
        color=COLORS["light_line"],
        lw=0.9,
        linestyle=(0, (2.2, 2.2)),
        connectionstyle="arc3,rad=-0.10",
        mutation_scale=7.0,
    )
    add_arrow_label(ax, (58.0, 48.0), "$q_{t,h}(c)$", color=COLORS["muted"])

    # Feedback event --------------------------------------------------------
    add_box(
        ax,
        (100.5, 32.0),
        17.5,
        13.0,
        "Feedback event",
        "$C_t+$ task truth $\\to r_t$\n"
        "$L_t(h)=P(r_t\\mid h,\\mathbf{x}_t,C_t)$\n"
        "$S^{\\mathrm{fb}}_t=-\\log P(r_t)$",
        fill=COLORS["feedback_fill"],
        edge=COLORS["feedback_edge"],
        body_size=5.65,
    )
    add_arrow(ax, (110.0, 50.0), (110.0, 45.0), color=COLORS["feedback_edge"])
    add_arrow_label(ax, (112.4, 47.4), "$C_t$", ha="left")

    # Dual memory -----------------------------------------------------------
    add_box(
        ax,
        (72.0, 28.0),
        24.0,
        22.0,
        "Fade/static dual memory",
        "Before feedback: synchronize to $\\pi^-_t$\n"
        "$w_0S_{t^-}+(1-w_0)F_{t^-}=\\log\\pi^-_t$\n"
        "After feedback:\n"
        "$F_t=\\gamma F_{t^-}+\\log\\widetilde L_t$\n"
        "$S_t=S_{t^-}+\\log\\widetilde L_t$\n"
        "$w_0S_t+(1-w_0)F_t \\to \\pi^+_t$",
        fill=COLORS["memory_fill"],
        edge=COLORS["memory_edge"],
        body_size=5.55,
        body_y=0.46,
    )
    add_arrow(
        ax,
        (86.0, 58.0),
        (86.0, 50.0),
        color=COLORS["memory_edge"],
        connectionstyle="arc3",
    )
    add_arrow_label(ax, (88.2, 54.0), "sync", color=COLORS["memory_edge"], ha="left")
    add_arrow(
        ax,
        (100.5, 36.5),
        (96.0, 36.5),
        color=COLORS["feedback_edge"],
    )
    add_arrow_label(ax, (98.3, 38.0), "$\\widetilde L_t$", color=COLORS["feedback_edge"])

    # Updated posterior and recursive loop ---------------------------------
    add_box(
        ax,
        (47.0, 31.0),
        19.0,
        14.0,
        "Updated state",
        "$\\pi^+_t,\\ F_t,\\ S_t$\n"
        "$U^{\\mathrm{rule}}_t$\n"
        "$S^{\\mathrm{fb}}_t$",
        fill=COLORS["state_fill"],
        edge=COLORS["state_edge"],
        body_size=6.3,
    )
    add_arrow(ax, (72.0, 38.0), (66.0, 38.0), color=COLORS["memory_edge"])
    add_arrow_label(ax, (69.0, 39.5), "$\\pi^+_t$")
    add_arrow(
        ax,
        (47.0, 38.0),
        (34.5, 61.0),
        color=COLORS["next_trial"],
        lw=1.35,
        linestyle=(0, (4.0, 2.6)),
        connectionstyle="arc3,rad=-0.22",
        mutation_scale=9.0,
    )
    add_arrow_label(
        ax,
        (38.0, 50.0),
        "next trial  $t+1$",
        color=COLORS["next_trial"],
        size=5.9,
    )

    # Descriptive labels only: no causal arrow.
    ax.text(
        58.8,
        56.0,
        "Stable / resistant / local / global are descriptive regions only",
        ha="center",
        va="center",
        fontsize=5.35,
        color=COLORS["h_edge"],
        style="italic",
    )

    # Arrow semantics.
    ax.plot([2.5, 7.0], [29.5, 29.5], color=COLORS["line"], lw=1.15)
    ax.text(7.8, 29.5, "within-trial generative flow", ha="left", va="center", fontsize=5.35, color=COLORS["muted"])
    ax.plot([2.5, 7.0], [26.6, 26.6], color=COLORS["next_trial"], lw=1.25, ls=(0, (4.0, 2.6)))
    ax.text(7.8, 26.6, "next-trial recursion", ha="left", va="center", fontsize=5.35, color=COLORS["muted"])
    ax.plot([2.5, 7.0], [23.7, 23.7], color=COLORS["light_line"], lw=0.9, ls=(0, (2.2, 2.2)))
    ax.text(7.8, 23.7, "fixed structural input", ha="left", va="center", fontsize=5.35, color=COLORS["muted"])

    # Panel b: multichannel identification ---------------------------------
    add_panel_heading(ax, 1.0, 18.7, "b", "Multichannel identification and validation")
    panel = FancyBboxPatch(
        (2.0, 2.0),
        116.0,
        13.5,
        boxstyle="round,pad=0.30,rounding_size=0.9",
        facecolor=COLORS["validation_fill"],
        edgecolor="#B8B8B8",
        linewidth=0.85,
        zorder=1,
    )
    ax.add_patch(panel)

    small_boxes = [
        (5.0, "Choice", "core behavioral\nconstraint", COLORS["obs_edge"]),
        (28.0, "+ RT", "uncertainty and\nsearch cost", COLORS["h_edge"]),
        (51.0, "+ oral report", "rule content and\nchange distance", COLORS["memory_edge"]),
        (78.0, "Predictive equivalence", "$\\mathcal{A}_C \\supseteq \\mathcal{A}_{C,R}$\n$\\supseteq \\mathcal{A}_{C,R,O}$", COLORS["state_edge"]),
    ]
    widths = [17.0, 17.0, 20.5, 23.5]
    for (x, title, body, edge), width in zip(small_boxes, widths):
        add_box(
            ax,
            (x, 6.0),
            width,
            7.0,
            title,
            body,
            fill="white",
            edge=edge,
            title_size=6.25,
            body_size=5.25,
            body_y=0.36,
            linewidth=0.85,
            radius=0.6,
        )
    add_arrow(ax, (22.0, 9.5), (28.0, 9.5), color=COLORS["line"], mutation_scale=7.5)
    add_arrow(ax, (45.0, 9.5), (51.0, 9.5), color=COLORS["line"], mutation_scale=7.5)
    add_arrow(ax, (71.5, 9.5), (78.0, 9.5), color=COLORS["line"], mutation_scale=7.5)
    ax.text(
        114.5,
        9.5,
        "Report the full acceptable set\nand only its stable functions",
        ha="right",
        va="center",
        fontsize=5.4,
        color=COLORS["ink"],
        linespacing=1.3,
    )
    ax.text(
        59.5,
        3.4,
        "External validation: freeze the choice-derived core posterior before testing RT/oral   |   Joint integration: quantify posterior contraction",
        ha="center",
        va="center",
        fontsize=5.15,
        color=COLORS["muted"],
    )

    fig.subplots_adjust(left=0.008, right=0.992, top=0.992, bottom=0.012)
    fig.savefig(f"{STEM}.svg", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{STEM}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{STEM}.png", dpi=400, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        f"{STEM}.tiff",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)


if __name__ == "__main__":
    draw_figure()
