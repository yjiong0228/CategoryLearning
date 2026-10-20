"""Plot search controls at generating and recovered parameter points.

Contract: compare search-control time courses at generating and recovered points.
The quantitative grid compares search-trigger probability (top, main evidence)
and global-route probability given search (bottom) for two synthetic sequences.
All declared parameter points and post-initialization trials are shown. Each
sequence contains generating parameters and a 40-start estimate, with an
optional 80-start estimate. Sequence length comes from the completed audit;
both sequences must share that length. Parameters were estimated using each
entire synthetic sequence.
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
    if done["status"] != "complete" or done["synthetic_sequences"] != 2:
        raise ValueError("Require a completed two-sequence control audit")
    for path, expected in done["inputs_sha256"].items():
        if digest(Path(path)) != expected:
            raise ValueError("Changed search-control input: " + path)
    for name, expected in done["artifacts"].items():
        if digest(source / name) != expected:
            raise ValueError("Changed search-control artifact: " + name)
    data = pd.read_csv(source / "trials.csv")
    groups = {(int(r), point) for r, point in zip(data.replicate, data.point)}
    required = {(replicate, point) for replicate in (0, 1) for point in ("truth", "fit_40")}
    allowed = required | {(replicate, "fit_80") for replicate in (0, 1)}
    if not required <= groups <= allowed or len(groups) != done["parameter_points"]:
        raise ValueError("Missing or unsupported declared parameter groups")
    summary = pd.read_csv(source / "summary.csv")
    whole = summary[summary.window.eq("whole")]
    if (len(whole) != len(groups) or whole.duplicated(["replicate", "point"]).any()
            or set(zip(whole.replicate, whole.point)) != groups):
        raise ValueError("Each plotted point requires one whole-sequence summary")
    counts = whole.trials.to_numpy(float)
    if not np.isfinite(counts).all() or np.any(counts != np.floor(counts)) or np.any(counts < 3):
        raise ValueError("Invalid post-initialization trial count")
    lengths = set(counts.astype(int) + 1)
    if len(lengths) != 1:
        raise ValueError("Require the same declared length for both synthetic sequences")
    trials = lengths.pop()
    if len(data) != len(groups) * (trials - 1):
        raise ValueError("Missing declared post-initialization trials")
    metrics = ("search_probability", "global_probability_given_search")
    for _, group in data.groupby(["replicate", "point"]):
        np.testing.assert_array_equal(group.trial, np.arange(2, trials + 1))
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
    styles = {point: style for point, style in styles.items() if point in set(data.point)}
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
                continue  # Only the audit's declared optional estimates are shown.
            for row, metric in enumerate(metrics):
                axes[row, column].plot(part.trial, part[metric], color=color, ls=line,
                                       lw=1.1, zorder=4 if point == "truth" else 3)
        for row in range(2):
            ax = axes[row, column]
            ax.set_xlim(2, trials)
            ax.set_ylim(-.025, 1.04)
            ax.set_xticks([2, trials // 4, trials // 2, 3 * trials // 4, trials])
            ax.set_yticks([0, .5, 1])
            ax.text(-.16 if column == 0 else -.08, 1.04, "abcd"[row * 2 + column],
                    transform=ax.transAxes, fontsize=9, fontweight="bold")
        axes[1, column].set_xlabel("Trial")
    axes[0, 0].set_ylabel("Search-trigger probability")
    axes[1, 0].set_ylabel("Global-route probability\nconditional on search")
    fig.text(.54, .065, f"All estimates use {trials} trials. Trial 1 initializes rules; trials 2–{trials} are shown.",
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
        dimensions_mm=[183, 125], dpi=600, format="PNG", plotted_rows=len(data),
        groups=len(groups), sequence_count=2, trials_per_sequence=int(trials),
        all_declared_groups_and_trials=True,
        excluded_initialization="Trial 1 has no search draw; excluded by upstream audit.",
        uncertainty="Deterministic controls at frozen fitted/generating points; parameter uncertainty not shown.",
        statistics="Descriptive synthetic recovery; no hypothesis test or independent participant inference.",
        adaptation="Audit-defined point counts and trial extent; original layout and score definitions retained.",
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
