"""Plot completed fixed-H precision comparisons and every autonomous path.

Run from the repository root:
python -m CategoryLearning_codes.figures.fig3.model0923_fixed_h_precision \
    --source COMPLETED_COMPARISON --audit INDEPENDENT_AUDIT --output NEW_DIRECTORY

Inputs are the frozen four-cell MIX/SELECT × ADD10/LOG25 comparison and its
independent numerical audit. Trial windows, scores and Monte Carlo errors retain
their original definitions. Every path is shown in 32-trial blocks; complete
later-window path accuracies form the empirical distributions. PNG review only.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .model0923_parameter_selection import configure_style

CELLS = ("MIX_ADD10", "MIX_LOG25", "SELECT_ADD10", "SELECT_LOG25")
COLORS = {"ADD10": "#0072B2", "LOG25": "#D55E00"}
BLOCK = 32


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def prepare(source: Path, audit: Path) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    """Bind completed evidence and reproduce all plotted curves from arrays."""
    done, checked, plan = (read(source / "completion.json"), read(audit / "completion.json"),
                           read(source / "protocol.json"))
    if (done["status"] != "complete" or done["smoke"] or done["oral_used"]
            or checked["status"] != "complete"):
        raise ValueError("Require completed full comparisons and independent audit")
    bindings = dict(checked["inputs_sha256"])
    for path in (source / "completion.json", source / "protocol.json", audit / "completion.json",
                 Path(__file__), Path(__file__).with_name("model0923_parameter_selection.py")):
        name, actual = str(path.resolve()), digest(path)
        if name in bindings and bindings[name] != actual:
            raise ValueError(f"Changed audited input: {path}")
        bindings[name] = actual
    for directory, files in ((source, done["summary_sha256"]), (audit, checked["outputs_sha256"])):
        for name, expected in files.items():
            path = directory / name
            if digest(path) != expected:
                raise ValueError(f"Changed summary: {path}")
            bindings[str(path.resolve())] = expected
    for name, expected in bindings.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Changed bound input: {name}")
    cut, trials = plan["training_trials"], plan["total_trials"]
    if not 0 < cut < trials or cut % BLOCK or trials % BLOCK:
        raise ValueError("Require complete 32-trial blocks at the original boundary")
    if set(read(source / "nominees.json")["nominees"]) != set(CELLS):
        raise ValueError("Require all four fixed-H comparison cells")
    reference = pd.read_csv(audit / "curves.csv").set_index(["cell", "first_trial"])
    reference_paths = pd.read_csv(audit / "paths.csv").set_index(["cell", "window", "seed"])
    curves, paths, base = [], [], None
    for cell in CELLS:
        banks = {}
        for phase in ("evaluate", "autonomous"):
            path = source / "banks" / f"{cell}_{phase}.npz"
            if digest(path) != bindings[str(path.resolve())]:
                raise ValueError(f"Changed prediction bank: {path}")
            with np.load(path, allow_pickle=False) as bank:
                banks[phase] = {key: bank[key] for key in bank.files}
        conditional, autonomous = banks["evaluate"], banks["autonomous"]
        observed = conditional["choices"] == conditional["categories"]
        if base is not None:
            np.testing.assert_array_equal(observed, base)
        base = observed
        np.testing.assert_array_equal(autonomous["categories"], conditional["categories"])
        for bank, phase in ((conditional, "evaluation"), (autonomous, "autonomous")):
            np.testing.assert_array_equal(bank["seeds"], plan["seeds"][phase])
            probability = bank["probabilities"]
            assert probability.shape == (len(bank["seeds"]), trials, 4)
            assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
            np.testing.assert_allclose(probability.sum(axis=2), 1, atol=1e-12, rtol=0)
        selected_category = np.take_along_axis(
            conditional["probabilities"], (conditional["categories"] - 1)[None, :, None], axis=2
        )[..., 0]
        np.testing.assert_array_equal(selected_category, conditional["correct_probability"])
        realized = autonomous["choices"] == autonomous["categories"][None, :]
        for first in range(0, trials, BLOCK):
            last = first + BLOCK
            values = realized[:, first:last].mean(axis=1)
            lower, upper = np.quantile(values, [.1, .9])
            row = dict(cell=cell, first_trial=first + 1, last_trial=last,
                       observed=float(observed[first:last].mean()), mean=float(values.mean()),
                       lower=float(lower), upper=float(upper),
                       conditional=float(selected_category[:, first:last].mean()))
            expected = reference.loc[(cell, first + 1)]
            for key in ("last_trial", "observed", "mean", "lower", "upper"):
                np.testing.assert_allclose(row[key], expected[key], atol=1e-12, rtol=0)
            curves.append(row)
            paths.extend(dict(cell=cell, seed=int(seed), first_trial=first + 1,
                              last_trial=last, correct=float(value))
                         for seed, value in zip(autonomous["seeds"], values))
        for seed, value in zip(autonomous["seeds"], realized[:, cut:].mean(axis=1)):
            np.testing.assert_allclose(value, reference_paths.loc[(cell, "later", seed), "correct"],
                                       atol=1e-12, rtol=0)
    return plan, bindings, pd.DataFrame(curves), pd.DataFrame(paths)


def label(ax: plt.Axes, letter: str) -> None:
    ax.text(-.13, 1.09, letter, transform=ax.transAxes, fontsize=8, fontweight="bold")


def learning_figure(plan: dict, curves: pd.DataFrame, paths: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 165 / 25.4))
    fig.subplots_adjust(left=.10, right=.98, top=.89, bottom=.23, hspace=.50, wspace=.25)
    trials, cut = plan["total_trials"], plan["training_trials"]
    for i, (ax, cell) in enumerate(zip(axes.flat, CELLS)):
        policy, family = cell.split("_")
        color = COLORS[family]
        own = curves[curves.cell == cell]
        x = (own.first_trial + own.last_trial) / 2
        for _, path in paths[paths.cell == cell].groupby("seed", sort=False):
            ax.plot((path.first_trial + path.last_trial) / 2, path.correct,
                    color="#777777", alpha=.16, lw=.3, zorder=1)
        ax.fill_between(x, own.lower, own.upper, color=color, alpha=.13, lw=0, zorder=2)
        ax.plot(x, own["mean"], color=color, lw=1.4, zorder=3)
        ax.plot(x, own.conditional, color="#7A5195", ls="--", lw=1.2, zorder=4)
        ax.plot(x, own.observed, color="#222222", marker=".", ms=3, lw=1., zorder=5)
        ax.axvline(cut + .5, color="#444444", ls=":", lw=.8)
        ax.set(title=f"{policy} / {family}", xlabel="Trial", ylabel="Correct response",
               xlim=(1, trials), ylim=(-.02, 1.02),
               xticks=sorted(set([1, cut // 2, cut, trials])))
        label(ax, "abcd"[i])
    fig.suptitle(f"S{plan['subject']}: conditional prediction and autonomous learning", y=.98, fontsize=10)
    fig.legend(handles=[
        Line2D([], [], color="#222222", marker=".", label="Observed", lw=1),
        Line2D([], [], color="#7A5195", ls="--", label="Conditional prediction"),
        Line2D([], [], color="#555555", lw=1.4, label="Autonomous mean"),
        Line2D([], [], color="#777777", alpha=.5, lw=.6, label="Every autonomous path")],
        loc="lower center", bbox_to_anchor=(.5, .125), ncol=2, fontsize=6.5)
    fig.text(.5, .091, f"All {len(plan['seeds']['autonomous'])} paths per cell; 32-trial blocks; "
             "bands: pointwise 10th–90th path percentiles.", ha="center", fontsize=6)
    fig.text(.5, .062, f"Conditional prediction: {len(plan['seeds']['evaluation'])} independent filters; "
             f"dotted line: training ends at trial {cut}.", ha="center", fontsize=6)
    fig.text(.5, .033, "H parameters fixed within each policy; MIX and SELECT retain their own fitted H parameters.",
             ha="center", fontsize=6)
    fig.savefig(output / "learning_paths.png", dpi=300, facecolor="white")
    plt.close(fig)


def tradeoff_figure(plan: dict, source: Path, audit: Path, output: Path) -> None:
    differences = pd.read_csv(source / "contrasts.csv").set_index(["policy", "window"])
    paths = pd.read_csv(audit / "paths.csv")
    scores = pd.read_csv(source / "scores.csv").set_index(["cell", "window"])
    windows = ("training", "later", "whole")
    fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 165 / 25.4))
    fig.subplots_adjust(left=.12, right=.98, top=.88, bottom=.21, hspace=.58, wspace=.37)
    for ax, metric, title in zip(axes[0], ("choice_nll", "energy"),
                               ("Choice prediction", "Autonomous distribution")):
        for i, policy in enumerate(("MIX", "SELECT")):
            own = differences.loc[policy].loc[list(windows)]
            ax.errorbar(np.arange(3) + (i - .5) * .12, own["delta_" + metric],
                        yerr=2 * own["delta_" + metric + "_mc_se"],
                        color=("#0072B2", "#D55E00")[i], fmt=("o", "s")[i], ms=3.5,
                        lw=.8, capsize=2, label=policy)
        ax.axhline(0, color="#777777", lw=.7, ls=":")
        ax.set(title=title, ylabel=("Mean NLL" if metric == "choice_nll" else "Energy score")
               + ": LOG25 − ADD10", xticks=range(3), xticklabels=("Training", "Later", "Whole"))
        ax.margins(x=.15)
        ax.legend(fontsize=6)
    later = plan["total_trials"] - plan["training_trials"]
    for ax, policy in zip(axes[1], ("MIX", "SELECT")):
        for family in ("ADD10", "LOG25"):
            own = paths[(paths.cell == policy + "_" + family) & (paths.window == "later")]
            values = np.sort(own.correct.to_numpy())
            assert len(values) == len(plan["seeds"]["autonomous"])
            ax.step(np.r_[0, values, 1], np.r_[0, np.arange(1, len(values) + 1) / len(values), 1],
                    where="post", color=COLORS[family], lw=1.3, label=family)
        human = scores.loc[(policy + "_ADD10", "later"), "human_correct"]
        ax.axvline(human, color="#222222", ls=":", lw=1, label="Observed")
        ax.set(title=f"{policy}: all later-window paths", xlabel=f"Correct response fraction (last {later} trials)",
               ylabel="Cumulative path fraction", xlim=(0, 1), ylim=(0, 1.02))
        ax.legend(fontsize=6, loc="upper left")
    for ax, letter in zip(axes.flat, "abcd"):
        label(ax, letter)
    fig.suptitle(f"S{plan['subject']}: precision gains and prediction costs", y=.98, fontsize=10)
    fig.text(.5, .115, "a–b: negative differences favor LOG25; bars: ±2 paired Monte Carlo SE.",
             ha="center", fontsize=6)
    fig.text(.5, .082, "Monte Carlo error describes complete filter/path repeats at fixed parameters.",
             ha="center", fontsize=6)
    fig.text(.5, .049, "c–d: every autonomous path; empirical distributions, with no density smoothing.",
             ha="center", fontsize=6)
    fig.text(.5, .016, "Development subject; later-window diagnostics; parameter and population uncertainty are separate.",
             ha="center", fontsize=6)
    fig.savefig(output / "behavior_tradeoff.png", dpi=300, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new figure directory")
    plan, bindings, curves, paths = prepare(args.source, args.audit)
    font = configure_style()
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [font, "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    args.output.mkdir(parents=True)
    contract = dict(
        conclusion="Describe later learning gains together with training prediction costs and complete path distributions.",
        backend="python", archetype="quantitative grid", width_mm=183, height_mm=165,
        export="300 dpi PNG review figures", reuse_level="Style-only; new fixed-H data mapping and panels",
        mapping=dict(curves="32-trial blocks; observed choices, conditional correct probabilities, all realized paths",
                     scores="Original training/later/whole score differences and paired Monte Carlo errors",
                     distributions="All complete-path later-window accuracies, no smoothing"),
        participants=1, subject=plan["subject"], condition=plan["condition"],
        training_trials=plan["training_trials"], total_trials=plan["total_trials"],
        paths_per_cell=len(plan["seeds"]["autonomous"]), filters_per_cell=len(plan["seeds"]["evaluation"]),
        exclusions="None: all four cells, every trial and every declared path retained",
        uncertainty="Path bands are empirical pointwise quantiles; score bars are ±2 MC SE; no population test",
        interpretation="Frozen policy-specific H points; precision coordinates chosen by training behavior; oral unused",
        versions=dict(python=platform.python_version(),
                      **{name: version(name) for name in ("numpy", "pandas", "matplotlib")}),
        font=font, inputs_sha256=bindings)
    write(args.output / "figure_contract.json", contract)
    curves.to_csv(args.output / "curves.csv", index=False)
    paths.to_csv(args.output / "block_paths.csv", index=False)
    for directory, names in ((args.source, ("scores.csv", "contrasts.csv")),
                             (args.audit, ("paths.csv", "process.csv"))):
        for name in names:
            shutil.copy2(directory / name, args.output / name)
    shutil.copy2(__file__, args.output / "plot_source.py")
    learning_figure(plan, curves, paths, args.output)
    tradeoff_figure(plan, args.source, args.audit, args.output)
    for name, expected in bindings.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Input changed while plotting: {name}")
    write(args.output / "completion.json", dict(
        status="complete", subject=plan["subject"], curves=len(curves), path_blocks=len(paths),
        inputs_sha256=bindings,
        artifacts_sha256={path.name: digest(path) for path in args.output.iterdir() if path.is_file()}))


if __name__ == "__main__":
    main()
