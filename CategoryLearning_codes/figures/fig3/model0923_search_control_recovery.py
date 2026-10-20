"""Plot search controls at generating and recovered parameter points.

Contract: similar choice fits can imply different search-control time courses.
The quantitative grid compares search-trigger probability (top, main evidence)
and global-route probability given search (bottom) for two synthetic sequences.
All five declared parameter points and all 127 post-initialization trials are
shown; parameters were estimated from the entire 128-trial synthetic sequence.
Controls consume only preceding feedback and are evaluated before the current
choice. Curves are deterministic at frozen parameters; no interval or test is
implied. This is a development diagnostic, not a population recovery estimate.

Use --source COMPLETED_SEARCH_CONTROL_AUDIT --output NEW_DIRECTORY.
Python/matplotlib, 183 x 125 mm, PNG only at 600 dpi, exact source CSVs retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(source: Path, output: Path) -> None:
    """Validate the completed audit, then render the full declared comparison."""
    if output.exists():
        raise FileExistsError("Use a new search-control figure directory")
    done = json.loads((source / "completion.json").read_text())
    if (done["status"], done["subject"], done["synthetic_sequences"], done["parameter_points"]) != (
        "complete", 107, 2, 5
    ):
        raise ValueError("Require the completed two-sequence, five-point audit")
    for path, expected in done["inputs_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError("Changed search-control input: " + path)
    for name, expected in done["artifacts"].items():
        if digest(source / name) != expected:
            raise ValueError("Changed search-control artifact: " + name)
    data = pd.read_csv(source / "trials.csv")
    groups = {(int(r), point) for r, point in zip(data.replicate, data.point)}
    expected = {(0, "truth"), (0, "fit_40"), (1, "truth"), (1, "fit_40"), (1, "fit_80")}
    if groups != expected or len(data) != 635:
        raise ValueError("Missing declared parameter groups or trials")
    metrics = ("search_probability", "global_probability_given_search")
    for _, group in data.groupby(["replicate", "point"]):
        np.testing.assert_array_equal(group.trial, np.arange(2, 129))
        values = group[list(metrics)].to_numpy()
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Invalid plotted probability")

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7,
        "axes.titlesize": 8, "axes.labelsize": 7, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": .65,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    styles = {
        "truth": ("#292929", "--", "Generating parameters"),
        "fit_40": ("#BF743D", "-", "40-start estimate"),
        "fit_80": ("#397A9D", "-.", "80-start estimate"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2047244, 4.9212598),
                             sharex=True, sharey=True, height_ratios=[1.35, 1])
    fig.subplots_adjust(left=.105, right=.975, top=.80, bottom=.16, wspace=.18, hspace=.28)
    fig.suptitle("Search-control recovery on the same feedback histories", y=.975, fontsize=10)
    fig.legend(handles=[Line2D([0], [0], color=c, ls=ls, lw=1.3, label=label)
                        for c, ls, label in styles.values()],
               loc="upper center", bbox_to_anchor=(.54, .925), ncol=3, frameon=False)
    for column in range(2):
        group = data[data.replicate.eq(column)]
        axes[0, column].set_title(f"Synthetic sequence {column + 1}", pad=9)
        for point, (color, line, _) in styles.items():
            part = group[group.point.eq(point)]
            if part.empty:
                continue  # The predeclared 80-start extension belongs to sequence 2.
            for row, metric in enumerate(metrics):
                axes[row, column].plot(part.trial, part[metric], color=color, ls=line,
                                       lw=1.1, zorder=4 if point == "truth" else 3)
        for row in range(2):
            ax = axes[row, column]
            ax.set_xlim(2, 128)
            ax.set_ylim(-.025, 1.04)
            ax.set_xticks([2, 32, 64, 96, 128])
            ax.set_yticks([0, .5, 1])
            ax.text(-.16 if column == 0 else -.08, 1.04, "abcd"[row * 2 + column],
                    transform=ax.transAxes, fontsize=9, fontweight="bold")
        axes[1, column].set_xlabel("Trial")
    axes[0, 0].set_ylabel("Search-trigger probability")
    axes[1, 0].set_ylabel("Global-route probability\nconditional on search")
    fig.text(.54, .065, "All estimates use 128 trials. Trial 1 initializes rules; trials 2–128 are shown.",
             ha="center", fontsize=7)
    fig.text(.54, .027, "Controls use preceding feedback, before conditioning on the current choice.",
             ha="center", fontsize=7)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for text in fig.findobj(matplotlib.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < bounds.x0 - 1 or box.y0 < bounds.y0 - 1 or box.x1 > bounds.x1 + 1 or box.y1 > bounds.y1 + 1:
                raise ValueError("Text outside figure bounds: " + text.get_text())
    output.mkdir(parents=True)
    name = "search_control_recovery.png"
    fig.savefig(output / name, dpi=600, facecolor="white")
    plt.close(fig)
    for filename in ("trials.csv", "summary.csv"):
        shutil.copyfile(source / filename, output / filename)
    record = dict(
        status="complete", backend="python", source=str(source.resolve()),
        source_sha256=digest(source / "completion.json"), script_sha256=digest(Path(__file__)),
        dimensions_mm=[183, 125], dpi=600, format="PNG", plotted_rows=635,
        groups=5, sequence_count=2, all_declared_groups_and_trials=True,
        excluded_initialization="Trial 1 has no search draw; excluded by upstream audit.",
        uncertainty="Deterministic controls at frozen fitted/generating points; parameter uncertainty not shown.",
        statistics="Descriptive synthetic recovery; no hypothesis test or independent participant inference.",
        adaptation="New data mapping and layout; repository typography and restrained colors reused.",
        format_policy="PNG only, per repository user instruction.",
        text_bounds="All visible text lies within the figure canvas.",
        artifacts={filename: digest(output / filename) for filename in (name, "trials.csv", "summary.csv")},
    )
    (output / "completion.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.output)
