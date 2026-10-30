"""Arrange saved strategy and trajectory panels as wide poster rows.

Reuse the original plotting functions and their saved data. Change only panel
selection, positions, typography and line widths. All three illustrative cases
and all participants in the two cohort comparisons are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np

from . import build_journal_figure as strategies
from .build_poster_font_variants import BOLD, REGULAR, ROOT, supplied_fonts
from ..fig3 import build_journal_figure as dynamics


def data_hash(axes: list) -> str:
    """Hash plotted coordinates and image arrays independently of layout."""
    digest = hashlib.sha256()
    for ax in axes:
        for line in ax.lines:
            for values in (line.get_xdata(), line.get_ydata()):
                digest.update(np.asarray(values, dtype=float).tobytes())
        for image in ax.images:
            arr = np.ma.asarray(image.get_array())
            digest.update(np.asarray(arr.filled(np.nan), dtype=float).tobytes())
            digest.update(np.ma.getmaskarray(arr).tobytes())
        for collection in ax.collections:
            digest.update(np.asarray(collection.get_offsets(), dtype=float).tobytes())
            for path in collection.get_paths():
                digest.update(path.vertices.tobytes())
    return digest.hexdigest()


def format_axis(ax) -> None:
    ax.tick_params(axis="both", labelsize=20, width=1.0, length=4, pad=4)
    ax.xaxis.label.set_fontsize(23)
    ax.yaxis.label.set_fontsize(23)
    ax.title.set_fontsize(23)
    ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=23, pad=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
    for line in ax.lines:
        line.set_linewidth(1.8)
        if line.get_marker() not in (None, "None", "", " "):
            line.set_markersize(max(5.5, line.get_markersize()))
    for text in ax.texts:
        text.set_fontsize(20)


def arrange_strategies(fig: Figure) -> list:
    axes = list(fig.axes)
    keep = [axes[i] for i in (2, 3, 6, 7, 10, 11, 12, 13, 16, 17)]
    for ax in axes:
        if ax not in keep:
            fig.delaxes(ax)
    for text in list(fig.texts):
        text.remove()
    fig.set_size_inches(31, 5.5)
    for column, (subject, task, n, belief, report) in enumerate([
        (102, 1, 448, axes[2], axes[3]),
        (307, 2, 704, axes[6], axes[7]),
        (221, 3, 832, axes[10], axes[12]),
    ]):
        x = .025 + column * .194
        belief.set_position([x, .50, .164, .25])
        report.set_position([x, .18, .164, .25])
        report.set_xticks([1, n // 2, n])
        report.set_xlabel("Trial")
        fig.text(x + .082, .97, f"Task {task} · S{subject}", ha="center", va="top",
                 fontsize=27, weight="bold", color=strategies.TASK_COLORS[task])
    axes[11].set_position([.586, .50, .005, .25])
    axes[13].set_position([.586, .18, .005, .25])
    for ax, x, title in [(axes[16], .655, "Rule-content\ncorrespondence"),
                         (axes[17], .842, "Report-change\ncorrespondence")]:
        ax.set_position([x, .18, .137, .57])
        fig.text(x + .0685, .97, title, ha="center", va="top", fontsize=25, weight="bold")
    for ax in keep:
        format_axis(ax)
    for ax in (axes[11], axes[13]):
        ax.tick_params(labelsize=17, length=2, pad=3)
        ax.set_title(ax.get_title(), fontsize=17, pad=12)
    return keep


def arrange_dynamics(fig: Figure) -> list:
    keep = list(fig.axes[:6])
    for ax in list(fig.axes[6:]):
        fig.delaxes(ax)
    for text in list(fig.texts):
        text.remove()
    for legend in list(fig.legends):
        legend.remove()
    fig.set_size_inches(31, 5)
    for column, (subject, task, title) in enumerate([
        (122, 1, "Earlier improvement"), (206, 3, "Gradual improvement"),
        (215, 3, "Abrupt improvement"),
    ]):
        x = .045 + column * .324
        upper, lower = keep[column * 2:column * 2 + 2]
        upper.set_position([x, .50, .265, .27])
        lower.set_position([x, .13, .265, .27])
        fig.text(x + .1325, .895, f"{title} · S{subject} · Task {task}", ha="center",
                 va="top", fontsize=26, weight="bold")
    for ax in keep:
        format_axis(ax)
    keep[1].set_ylabel("Target-rule\nstate", fontsize=21)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=dynamics.COLORS[key], lw=2, ls=style, label=label)
               for key, label, style in [("observed", "Observed", "-"),
                                        ("predicted", "Predicted", "--"),
                                        ("available", "Considered", "-"),
                                        ("belief", "Belief", "-"),
                                        ("executed", "Executed", "-")]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.52, 1.02),
               ncol=5, fontsize=21, frameon=False, handlelength=2.0, columnspacing=2.4)
    return keep


def build(version: str) -> None:
    base = ROOT / "CategoryLearning_codes/figures/outputs"
    destinations = {"strategies": base / "fig2" / version,
                    "dynamics": base / "fig3" / version}
    if any(path.exists() for path in destinations.values()):
        raise FileExistsError("Use a new output version")
    audits: list[dict] = []
    with tempfile.TemporaryDirectory(prefix=".poster-rows-", dir=base) as temporary:
        temp = Path(temporary)
        for key, layout in [("strategies", arrange_strategies), ("dynamics", arrange_dynamics)]:
            destination = destinations[key]
            destination.mkdir(parents=True)
            checks = {}
            with supplied_fonts(audits):
                save_with_fonts = Figure.savefig

                def save_row(fig, filename, *args, **kwargs):
                    selected = ([fig.axes[i] for i in (2, 3, 6, 7, 10, 11, 12, 13, 16, 17)]
                                if key == "strategies" else list(fig.axes[:6]))
                    before = data_hash(selected)
                    retained = layout(fig)
                    after = data_hash(retained)
                    if before != after:
                        raise ValueError("Plotted data changed during layout")
                    checks.update({"data_sha256_before": before, "data_sha256_after": after,
                                   "size_inches": list(fig.get_size_inches()),
                                   "axes_retained": len(retained), "dpi": 300})
                    return save_with_fonts(fig, destination / f"{key}_row.png", dpi=300,
                                           facecolor="white", format="png")

                with patch.object(Figure, "savefig", save_row):
                    if key == "strategies":
                        strategies.build(ROOT / strategies.DEFAULT_STATES,
                                         ROOT / strategies.DEFAULT_VALIDATION, temp / key)
                    else:
                        dynamics.build(ROOT / "results/model_0826/fig34_twelve_subjects_20260921_v1/analysis",
                                       temp / key)
            source_manifest = json.loads((temp / key / "manifest.json").read_text())
            (destination / "manifest.json").write_text(json.dumps({
                "purpose": "Wide poster row; layout and typography only",
                "selection": ("Three rule/report examples and both complete 12-person cohort panels"
                              if key == "strategies" else "All three existing illustrative trajectories"),
                "omitted_panels": ("Repeated individual accuracy and compatibility traces; model schematic; calibration"
                                   if key == "strategies" else "Separate cohort shape and sustained-belief panels"),
                "original_source_manifest": source_manifest, "layout_checks": checks,
                "font_audit": audits[-1],
                "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in (Path(__file__).resolve(), REGULAR, BOLD)},
                "new_fits": 0, "new_replays": 0, "new_oral_encoding": 0,
            }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    if Path(args.version).name != args.version:
        parser.error("--version must be a single directory name")
    build(args.version)


if __name__ == "__main__":
    main()
