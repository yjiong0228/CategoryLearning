"""Publication-layout Fig. 1 from the complete audited behavioral cohort.

Run from the repository root, with a new --output directory. PNG only.
Scientific definitions and the full legend are in JOURNAL_FIGURE.md.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from .schematics import stimulus_panel, task_panel, draw_animal, _microphone
from .render import report_strips
from ..fig3.bottleneck_analysis import ROOT, sha256


DEFAULT_SOURCE = "CategoryLearning_codes/figures/outputs/fig1/fig1_v10"
TASK_COLORS = {1: "#487DA8", 2: "#A46A8A", 3: "#648B76"}


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 6.5, "axes.titlesize": 7, "axes.labelsize": 6.5,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "lines.linewidth": .85,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2,
        "xtick.major.pad": 2, "ytick.major.pad": 2,
        "legend.frameon": False, "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def panel(fig, letter: str, title: str, x: float, y: float) -> None:
    fig.text(x - .032, y, letter, fontsize=9, weight="bold", va="bottom")
    fig.text(x, y, title, fontsize=7, va="bottom")


def trial_sequence(ax) -> None:
    """Keep the actual five stages, using a quieter native screen diagram."""
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    width, height = .174, .64
    titles = ["Fixation", "Category choice", "Prepare report", "Verbal report", "Feedback"]
    for i, title in enumerate(titles):
        x, y = .01 + .201 * i, .19
        ax.text(x + width / 2, .96, title, ha="center", va="top", fontsize=6.2)
        ax.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=.003,rounding_size=.012",
            fc="#F1F2F3", ec="#C6CACD", lw=.55))
        if i == 0:
            ax.text(x+width/2, y+height*.5, "+", ha="center", va="center", fontsize=15)
        elif i == 1:
            draw_animal(ax, (x+.006, y+.15, width-.012, height-.17), linewidth=.85)
            for dx, text in [(.051, "F"), (.122, "J")]:
                ax.text(x+dx, y+.085, text, ha="center", va="center", fontsize=6,
                        bbox={"boxstyle":"round,pad=.15", "fc":"white", "ec":".6", "lw":.45})
        elif i == 2:
            ax.text(x+width/2, y+height*.5, "Ready…", ha="center", va="center", fontsize=7)
        elif i == 3:
            _microphone(ax, x+width/2, y+.31, .30)
        else:
            ax.text(x+width/2, y+height*.5, "1", ha="center", va="center", fontsize=14)
        if i < 4:
            ax.annotate("", xy=(x+width+.024, y+height/2), xytext=(x+width+.006, y+height/2),
                        arrowprops={"arrowstyle":"->", "lw":.6, "color":".5"})
    ax.annotate("", xy=(.986, .035), xytext=(.012, .035),
                arrowprops={"arrowstyle":"->", "lw":.65, "color":".5"})
    ax.text(.5, -.10, "Choice → report → feedback", ha="center", fontsize=6, color=".35")


def paired_summary(ax, people: pd.DataFrame, tasks: list[dict], metric: str) -> dict:
    """Disjoint windows; each participant is the sampling unit."""
    counts = {}
    for j, task in enumerate(tasks):
        name = "accuracy" if metric == "accuracy" else "feature_count"
        left = "first64_accuracy" if metric == "accuracy" else "first_feature_count"
        right = "last64_accuracy" if metric == "accuracy" else "last_feature_count"
        subset = people.loc[people.condition.eq(task["condition"]) & people.n_trials.ge(128)].copy()
        subset = subset.loc[subset[left].notna() & subset[right].notna()]
        counts[str(task["task"])] = len(subset)
        xx = np.array([j*1.75, j*1.75+.63]); color = task["color"]
        for row in subset.itertuples():
            ax.plot(xx, [getattr(row, left), getattr(row, right)], color=color,
                    alpha=.22, lw=.5, marker="o", ms=1.8, mew=0)
        ax.plot(xx, subset[[left, right]].mean().to_numpy(), color=color,
                lw=1.5, marker="o", ms=3, markeredgecolor="white", markeredgewidth=.4)
        ax.text(np.mean(xx), 1.12, f"Task {task['task']}", ha="center", fontsize=6,
                color=color, transform=ax.get_xaxis_transform())
        ax.text(np.mean(xx), 1.01, f"n = {len(subset)}", ha="center", fontsize=5.5,
                color=".45", transform=ax.get_xaxis_transform())
    ax.set(xticks=[0,.63,1.75,2.38,3.5,4.13], xticklabels=["First", "Last"]*3,
           xlim=(-.28,4.43), xlabel="64-trial periods")
    ax.tick_params(axis="x", labelsize=5.6, pad=2)
    if metric == "accuracy":
        ax.set(ylim=(0,1.03), yticks=[0,.5,1], ylabel="Accuracy")
    else:
        ax.set(ylim=(.8,4.15), yticks=[1,2,3,4], ylabel="Features / report")
    return counts


def render(trials: pd.DataFrame, people: pd.DataFrame, tasks: list[dict], output: Path) -> dict:
    style()
    fig = plt.figure(figsize=(183 / 25.4, 215 / 25.4))
    panel(fig, "a", "Category-learning tasks", .075, .966)
    stimulus_panel(fig.add_axes([.070,.813,.190,.142]))
    for j, task in enumerate(tasks):
        task_panel(fig.add_axes([.285+j*.234,.813,.207,.142]), task)

    panel(fig, "b", "Trial structure", .075, .771)
    trial_sequence(fig.add_axes([.072,.659,.903,.108]))

    panel(fig, "c", "Individual learning trajectories", .075, .606)
    cmap = LinearSegmentedColormap.from_list("learning_accuracy", ["#F7F9FA", "#A9BBC8", "#294B66"])
    cmap.set_bad("#ECEDEE")
    task_counts = {}
    for j, task in enumerate(tasks):
        x = .075 + .314*j
        sub = people.loc[people.condition.eq(task["condition"])].sort_values(["n_trials", "iSub"])
        nmax = int(sub.n_trials.max())
        data = np.full((len(sub), nmax), np.nan)
        for row, sid in enumerate(sub.iSub):
            vals = trials.loc[trials.iSub.eq(sid), "rolling_accuracy"].to_numpy()
            data[row, :len(vals)] = vals
        ax = fig.add_axes([x,.451,.272,.124])
        im = ax.imshow(data, aspect="auto", interpolation="nearest", vmin=0, vmax=1,
                       cmap=cmap, extent=(.5,nmax+.5,32.5,.5))
        ax.set_title(f"Task {task['task']} · n = 32", color=task["color"], pad=5)
        ax.set(yticks=[1,16,32], xticks=[])
        if j == 0: ax.set_ylabel("Participant", labelpad=3)
        else: ax.set_yticklabels([])
        counts = (np.arange(1,nmax+1)[None,:] <= sub.n_trials.to_numpy()[:,None]).sum(0)
        count_ax = fig.add_axes([x,.421,.272,.023])
        count_ax.fill_between(np.arange(1,nmax+1), counts, color=task["color"], alpha=.18, lw=0)
        count_ax.plot(np.arange(1,nmax+1), counts, color=task["color"], lw=.65)
        ticks = [1,256,512,768] if nmax == 768 else ([1,512,1024,1472] if nmax == 1472 else [1,512,1024,1792])
        count_ax.set(xlim=(1,nmax), ylim=(0,34), yticks=[0,32], xticks=ticks, xlabel="Trial")
        count_ax.tick_params(labelsize=5.5, pad=1)
        count_ax.xaxis.labelpad = 2
        if j == 0: count_ax.set_ylabel("n", rotation=0, labelpad=5)
        task_counts[str(task["task"])] = len(sub)
    cax = fig.add_axes([.798,.611,.174,.006])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0,.5,1])
    cbar.ax.tick_params(labelsize=5.5, length=1, pad=1)
    cax.set_title("Accuracy", fontsize=6, pad=3)

    panel(fig, "d", "Behavior and reported features", .075, .360)
    for j, sid in enumerate([122,206,215]):
        x = .075 + .314*j
        g = trials.loc[trials.iSub.eq(sid)]; s = people.set_index("iSub").loc[sid]
        task_id = {1:1,3:2,2:3}[int(s.condition)]; color = TASK_COLORS[task_id]
        ax = fig.add_axes([x,.264,.272,.070])
        ax.plot(g.trial,g.rolling_accuracy,color=color,lw=.9)
        ax.axhline(.5 if task_id==1 else .25,color=".68",lw=.5,ls="--")
        for onset in g.loc[g.iSession.diff().fillna(0).ne(0),"trial"]:
            ax.axvline(onset-.5,color=".7",ls=":",lw=.55)
        ax.set(xlim=(.5,len(g)+.5),ylim=(-.03,1.05),yticks=[0,.5,1],xticks=[])
        ax.text(.01,1.08,f"S{sid} · Task {task_id}",transform=ax.transAxes,fontsize=6.2,color=color)
        if j==0: ax.set_ylabel("Accuracy",labelpad=3)
        else: ax.set_yticklabels([])
        strips = fig.add_axes([x,.224,.272,.031])
        report_strips(strips,g,color)
        ticks = [1,128,256] if sid==122 else ([1,704,1408] if sid==206 else [1,384,768])
        strips.set(xticks=ticks, xlabel="Trial")
        strips.xaxis.labelpad = 2

    panel(fig, "e", "Time to criterion", .075, .162)
    ax = fig.add_axes([.075,.050,.240,.088])
    unreached = {}
    for j, task in enumerate(tasks):
        sub = people.loc[people.condition.eq(task["condition"])].sort_values("iSub")
        hit = sub.first_crossing64.notna().to_numpy()
        xx = j+1+np.linspace(-.22,.22,len(sub))
        ax.scatter(xx[hit],sub.first_crossing64[hit],s=8,color=task["color"],alpha=.7,lw=0)
        ax.scatter(xx[~hit],sub.n_trials[~hit],s=15,marker="^",facecolors="white",edgecolors=task["color"],lw=.7)
        median = sub.first_crossing64.median()
        ax.plot([j+.73,j+1.27],[median]*2,color=task["color"],lw=1.6)
        unreached[str(task["task"])] = int((~hit).sum())
    ax.set(xticks=[1,2,3],xticklabels=["Task 1","Task 2","Task 3"],xlim=(.5,3.5),
           ylim=(0,1840),yticks=[0,800,1600],ylabel="First crossing (trial)")
    ax.scatter([],[],s=14,marker="^",facecolors="white",edgecolors=".4",lw=.7,label="Not reached")
    ax.legend(loc="upper left",bbox_to_anchor=(-.03,1.04),fontsize=5.5,handletextpad=.25,borderpad=0)

    panel(fig, "f", "Behavioral improvement", .396, .162)
    behavior_pairs = paired_summary(fig.add_axes([.396,.050,.247,.088]),people,tasks,"accuracy")
    panel(fig, "g", "Reported rule features", .723, .162)
    report_pairs = paired_summary(fig.add_axes([.723,.050,.247,.088]),people,tasks,"features")

    # The manuscript legend contains definitions; the image contains no report-style footer.
    path = output / "Figure1_journal.png"
    fig.savefig(path,dpi=450,facecolor="white")
    texts = [artist for artist in fig.findobj() if isinstance(artist, matplotlib.text.Text) and artist.get_text()]
    fig.canvas.draw(); renderer = fig.canvas.get_renderer()
    outside = []
    for artist in texts:
        box = artist.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1:
            outside.append(artist.get_text())
    plt.close(fig)
    return {"task_n":task_counts,"unreached_criterion":unreached,
            "paired_accuracy_n":behavior_pairs,"paired_report_n":report_pairs,
            "text_outside_canvas":outside,"dpi":450,"width_mm":183,"height_mm":215}


def build(source: Path, output: Path) -> None:
    if output.exists(): raise FileExistsError(f"Output already exists: {output}")
    trials = pd.read_csv(source/"trial_source.csv")
    people = pd.read_csv(source/"subject_summary.csv")
    config_path = Path(__file__).with_name("config.json")
    tasks = json.loads(config_path.read_text())["tasks"]
    raw_path = ROOT/"data/exp123/processed/Task2_processed.csv"
    keys = ["condition","iSub","iSession","iBlock","iTrial"]
    raw = pd.read_csv(raw_path).sort_values(keys,kind="stable").reset_index(drop=True)
    if len(trials)!=62720 or len(people)!=96: raise ValueError("Unexpected cohort size")
    for field in keys+["choice","feedback"]:
        np.testing.assert_array_equal(trials[field],raw[field])
    for sid, group in trials.groupby("iSub",sort=True):
        np.testing.assert_array_equal(group.trial,np.arange(1,len(group)+1))
        np.testing.assert_array_equal(group.correct,group.feedback.eq(1))
        expected = group.correct.rolling(32,min_periods=32).mean()
        np.testing.assert_allclose(group.rolling_accuracy,expected,equal_nan=True)
    output.mkdir(parents=True)
    checks = render(trials,people,tasks,output)
    if checks["text_outside_canvas"]: raise ValueError(checks["text_outside_canvas"])
    people.to_csv(output/"subject_source.csv",index=False)
    trials.to_csv(output/"trial_source.csv",index=False)
    inputs = [source/"trial_source.csv",source/"subject_summary.csv",raw_path,config_path,
              Path(__file__),Path(__file__).with_name("JOURNAL_FIGURE.md"),
              Path(__file__).with_name("schematics.py"),Path(__file__).with_name("render.py")]
    manifest = {"n_participants":96,"n_trials":len(trials),"examples":[122,206,215],
                "input_sha256":{str(p.resolve().relative_to(ROOT)):sha256(p) for p in inputs},
                "python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
                "matplotlib":matplotlib.__version__,"checks":checks}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(checks,indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path(DEFAULT_SOURCE))
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(); build(args.source.resolve(),args.output.resolve())


if __name__ == "__main__": main()
