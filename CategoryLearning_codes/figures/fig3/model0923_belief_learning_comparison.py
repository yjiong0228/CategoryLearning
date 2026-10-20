"""Plot complete ONLINE/SEPARATE refits and their autonomous learning paths.

Run --comparison COMPLETED_COMPARISON --output NEW_DIRECTORY from the repository
root. Both proposal arms, all participants and every saved path are retained.
Each participant receives a four-cell learning PNG and a phase-specific score
PNG, at 183 mm and 300 dpi. Path bands are 10th–90th percentiles; score bars are
two paired Monte Carlo standard errors at frozen fitted parameter points.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.model_0923_h_validation import checked_arrays, digest, load

MECHANISMS = ("ONLINE", "SEPARATE")
ARMS = ("POOL", "SEARCH")
COLORS = {"ONLINE": "#4678A5", "SEPARATE": "#CC8152"}


def draw(comparison: Path, output: Path) -> None:
    """Validate plot values against full behavior banks, then render all cells."""
    if output.exists():
        raise FileExistsError("Use a new belief-comparison figure directory")
    manifest = load(comparison / "manifest.json")
    if manifest["status"] != "complete" or set(manifest["arms"]) != set(ARMS):
        raise ValueError("Require the complete two-proposal belief comparison")
    bindings = dict(manifest["inputs_sha256"])
    subjects = sorted(manifest["subjects"])
    bindings[str((comparison / "manifest.json").resolve())] = digest(comparison / "manifest.json")
    for name, sha in manifest["tables_sha256"].items():
        bindings[str((comparison / name).resolve())] = sha
    contrasts = pd.read_csv(comparison / "contrasts.csv")
    curves, blocks, source_tables, cells = [], [], {}, []
    checked_values = 0

    def close(actual, expected):
        nonlocal checked_values
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        checked_values += np.asarray(actual).size

    for mechanism in MECHANISMS:
        source = Path(manifest[mechanism.lower() + "_summary"])
        summary = load(source / "manifest.json")
        bindings[str((source / "manifest.json").resolve())] = digest(source / "manifest.json")
        for name, sha in summary["summary_sha256"].items():
            bindings[str((source / name).resolve())] = sha
        own = {name: pd.read_csv(source / (name + ".csv")) for name in ("curves", "path_blocks", "paths", "scores")}
        for name in own:
            source_tables[mechanism.lower() + "_" + name + ".csv"] = source / (name + ".csv")
        evaluation = Path(summary["evaluation"])
        groups = {role: {(row["subject"], row["model"]): row
                         for row in load(evaluation / role / "evaluation.json")["rows"]}
                  for role in ("conditional", "autonomous")}
        if set(groups["autonomous"]) != {(subject, arm) for subject in subjects for arm in ARMS}:
            raise ValueError("Figure must contain every declared participant and proposal arm")
        width = int(summary["block_width"])
        for subject in subjects:
            for arm in ARMS:
                key = subject, arm
                c, a = [checked_arrays(evaluation / role, groups[role][key]) for role in ("conditional", "autonomous")]
                trials = len(c["choices"])
                cut = groups["conditional"][key]["training_trials"]
                human = c["choices"] == c["categories"]
                generated = a["choices"] == a["categories"][None, :]
                ends = np.arange(width, trials + width, width).clip(max=trials)
                x = np.stack([generated[:, first:min(first + width, trials)].mean(axis=1)
                              for first in range(0, trials, width)], axis=1)
                curve = own["curves"].query("subject == @subject and model == @arm").sort_values("trial")
                block = own["path_blocks"].query("subject == @subject and model == @arm")
                paths = own["paths"].query("subject == @subject and model == @arm")
                if len(paths) != len(a["seeds"]) or paths.seed.duplicated().any():
                    raise ValueError("Every unique autonomous path must be present")
                pivot = block.pivot(index="seed", columns="trial", values="accuracy")
                if set(pivot.index) != set(a["seeds"]) or list(pivot.columns) != list(ends):
                    raise ValueError("Path blocks must preserve all seeds and chronological trials")
                close(pivot.loc[a["seeds"]].to_numpy(), x)
                close(curve.trial.to_numpy(), ends)
                close(curve.training_trials.to_numpy(), cut)
                close(curve.autonomous.to_numpy(), x.mean(axis=0))
                close(curve.lower.to_numpy(), np.quantile(x, .1, axis=0))
                close(curve.upper.to_numpy(), np.quantile(x, .9, axis=0))
                close(curve.observed.to_numpy(), [human[first:min(first + width, trials)].mean()
                                                for first in range(0, trials, width)])
                close(curve.conditional.to_numpy(), [c["correct_probability"][:, first:min(first + width, trials)].mean()
                                                   for first in range(0, trials, width)])
                cells.append(dict(subject=subject, mechanism=mechanism, arm=arm, trials=trials,
                                  training_trials=cut, paths=len(a["seeds"]), filters=len(c["seeds"]), block_width=width))
        curves.append(own["curves"].assign(variant=mechanism))
        blocks.append(own["path_blocks"].assign(variant=mechanism))
    for path, expected in bindings.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Figure input changed: {path}")
    curves, blocks = pd.concat(curves), pd.concat(blocks)
    output.mkdir(parents=True)
    contract = dict(
        conclusion="Compare training gains with later learning costs at each mechanism's behavior-fitted parameters.",
        archetype="quantitative grid", backend="python", width_mm=183, dpi=300, formats=["PNG"],
        panels=["All autonomous paths alongside conditional predictions and observed behavior",
                "Training, later and whole-sequence NLL/Energy differences"],
        cells=cells, exclusions=[], uncertainty="10th–90th path percentiles; score bars ±2 paired Monte Carlo SE",
        limits="Development comparison; mechanism and fitted parameters jointly vary; no population or parameter uncertainty",
        export_policy="Repository discussion figures: PNG only, 300 dpi; vector/TIFF submission export is outside this task")
    (output / "figure_contract.json").write_text(json.dumps(contract, indent=2))
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7,
        "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
        "legend.fontsize": 6, "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False, "axes.linewidth": .6, "svg.fonttype": "none", "pdf.fonttype": 42})
    images = []
    for subject in subjects:
        fig, axes = plt.subplots(2, 2, figsize=(183 / 25.4, 145 / 25.4), sharex=True, sharey=True)
        for i, mechanism in enumerate(MECHANISMS):
            for j, arm in enumerate(ARMS):
                ax = axes[i, j]
                c = curves.query("subject == @subject and variant == @mechanism and model == @arm").sort_values("trial")
                b = blocks.query("subject == @subject and variant == @mechanism and model == @arm")
                count = 0
                for _, path in b.groupby("seed"):
                    path = path.sort_values("trial")
                    ax.plot(path.trial, path.accuracy, color=COLORS[mechanism], alpha=.065, lw=.4)
                    count += 1
                ax.fill_between(c.trial, c.lower, c.upper, color=COLORS[mechanism], alpha=.12)
                ax.plot(c.trial, c.autonomous, color=COLORS[mechanism], lw=1.5)
                ax.plot(c.trial, c.conditional, color=COLORS[mechanism], lw=1., ls="--")
                ax.plot(c.trial, c.observed, color="#252A31", lw=1.15)
                ax.axvline(c.training_trials.iloc[0], color="#777777", ls=":", lw=.8)
                ax.set(title=f"{'abcd'[2*i+j]}  {mechanism} · {arm} · {count} paths",
                       xlim=(0, c.trial.max()), ylim=(-.025, 1.025))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
                if j == 0:
                    ax.set_ylabel("Accuracy / probability")
                if i == 1:
                    ax.set_xlabel("Trial")
        handles = [Line2D([], [], color="#252A31", lw=1.2, label="Observed"),
                   Line2D([], [], color="#68778B", lw=1.5, label="Autonomous mean"),
                   Line2D([], [], color="#68778B", ls="--", lw=1., label="Conditional prediction")]
        fig.suptitle(f"S{subject}: full learning distributions after behavior-only fitting", fontsize=8, y=.98)
        fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(.5, .945))
        fig.text(.5, .885, "All paths retained; band: 10th–90th path percentiles; dotted line: training end",
                 ha="center", fontsize=6)
        fig.subplots_adjust(left=.09, right=.985, bottom=.09, top=.82, wspace=.15, hspace=.3)
        name = f"subject_{subject}_learning.png"
        fig.savefig(output / name, dpi=300, facecolor="white")
        plt.close(fig)
        images.append(name)
        fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 112 / 25.4))
        for i, metric in enumerate(("choice_nll", "energy")):
            for j, phase in enumerate(("training", "test", "whole")):
                ax = axes[i, j]
                data = contrasts.query("subject == @subject and phase == @phase").set_index("model").loc[list(ARMS)]
                ax.axvline(0, color="#999999", lw=.7)
                ax.errorbar(data["delta_" + metric], range(2), xerr=2 * data["delta_" + metric + "_mc_se"],
                            fmt="o", color=COLORS["SEPARATE"], ms=3, capsize=2, lw=.8)
                ax.set(yticks=range(2), yticklabels=ARMS, ylim=(1.6, -.6),
                       title=f"{'abcdef'[3*i+j]}  " + {"training": "Training", "test": "Later trials", "whole": "Whole sequence"}[phase],
                       xlabel="Δ NLL / trial" if metric == "choice_nll" else "Δ Energy score")
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        fig.suptitle(f"S{subject}: gains and costs vary across learning phases", fontsize=8, y=.98)
        fig.text(.5, .925, "SEPARATE − ONLINE; negative favors SEPARATE; bars: ±2 paired Monte Carlo SE",
                 ha="center", fontsize=6)
        fig.subplots_adjust(left=.1, right=.98, bottom=.12, top=.82, wspace=.65, hspace=.65)
        name = f"subject_{subject}_scores.png"
        fig.savefig(output / name, dpi=300, facecolor="white")
        plt.close(fig)
        images.append(name)
    for name, path in source_tables.items():
        shutil.copy2(path, output / name)
    shutil.copy2(comparison / "contrasts.csv", output / "paired_contrasts.csv")
    shutil.copy2(__file__, output / "figure_source.py")
    for path, expected in bindings.items():
        if digest(Path(path)) != expected:
            raise ValueError("Figure input changed during rendering")
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>信念机制完整学习比较</title>'
        '<style>body{max-width:1100px;margin:2em auto;font:16px/1.6 sans-serif;padding:0 1em}img{width:100%}</style>'
        '<h1>信念机制：完整学习轨迹与分阶段评分</h1>'
        '<p>各机制分别按训练行为估参，参数冻结后评价全部轨迹。条件预测使用已观察到的历史；'
        '自主路径使用各自生成的回答与反馈。所有路径保留，分位带表示路径分布。</p>'
        '<p>评分差值为SEPARATE减ONLINE；负值表示SEPARATE评分更好。误差线描述固定参数下的计算波动，'
        '个体参数及总体差异的不确定性另行评价。口述作为带噪音的外部支持，结果另行呈现。</p>'
        + "".join(f'<h2>{name}</h2><a href="{name}"><img src="{name}"></a>' for name in images)
        + '<p><a href="figure_contract.json">图表定义</a> · <a href="manifest.json">来源与验证</a></p>',
        encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(dict(
        status="complete", source=str(comparison.resolve()), inputs_sha256=bindings, numeric_checks=checked_values,
        paths=sum(c["paths"] for c in cells), path_blocks=len(blocks), images=images, visual_review="pending",
        source_sha256=digest(Path(__file__)), artifacts_sha256={p.name: digest(p) for p in output.iterdir()},
        versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__)),
        indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    draw(args.comparison, args.output)
