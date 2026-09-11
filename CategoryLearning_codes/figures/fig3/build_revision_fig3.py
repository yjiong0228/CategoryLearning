"""Fig3 revision: real report-change excerpts, with event analyses left pending.

Python, 183 x 185 mm, PNG only. Two descriptive examples are selected from raw
oral content, not a search-peak ranking. No population or anticipatory claim.
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

from CategoryLearning_codes.figures.fig3.build_fig3 import CASES, load_case


# Exact report strings preserve polarity and conjunctions; these are illustrative
# excerpts checked against the processed text, not a cohort-wide event detector.
EXCERPTS = (
    dict(start=132, end=144, choice=2, last_old=136, first_new=138,
         contents={"脖子短。": 0, "脖子长。": 1, "尾巴长。": 2},
         labels=["Short neck", "Long neck", "Long tail"],
         old="脖子长。", new="尾巴长。"),
    dict(start=985, end=999, choice=3, last_old=988, first_new=992,
         contents={"腿长，脖子短。": 0, "腿长，尾巴短。": 1},
         labels=["Long leg +\nshort neck", "Long leg +\nshort tail"],
         old="腿长，脖子短。", new="腿长，尾巴短。"),
)


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source_data"
    source.mkdir()
    cases = [load_case(*spec, source) for spec in CASES]
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Noto Sans CJK JP"],
                         "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "legend.fontsize": 7, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(183 / 25.4, 185 / 25.4), facecolor="white")
    fig.text(.07, .964, "Figure 3 | Search around changes in reported rules", fontsize=11, weight="bold")
    fig.text(.07, .938, "Real excerpts from S129 and S229; event-level tests remain pending.", color="#666666")
    fig.text(.02, .893, "a", fontsize=11, weight="bold")
    fig.text(.07, .893, "What happened around a change in report content?", fontsize=9, weight="bold")
    fig.legend(handles=[Line2D([], [], color="#439D91", label="Local search", lw=1.4),
                        Line2D([], [], color="#D88C49", label="Global search", lw=1.4)],
               loc="center", bbox_to_anchor=(.53, .867), ncol=2)
    evidence = []
    for case, spec, excerpt, left in zip(cases, CASES, EXCERPTS, [.14, .63]):
        subject, condition, task, run, _ = spec
        trials_path = Path(f"results/model_0826/cond{condition}/subject_{subject}/{run}/source_data/subject_{subject}_trials.csv")
        trials = pd.read_csv(trials_path)
        assert len(trials) == case["n"]
        assert (trials.iSub == subject).all() and (trials.condition == condition).all()
        trials.insert(0, "sequence_trial", np.arange(1, len(trials) + 1))
        start, end = excerpt["start"], excerpt["end"]
        clip = trials.iloc[start-1:end].copy()
        reports = clip.loc[clip.choice == excerpt["choice"]].copy()
        assert set(reports.text) == set(excerpt["contents"])
        reports["content_index"] = reports.text.map(excerpt["contents"])
        old_t, new_t = excerpt["last_old"], excerpt["first_new"]
        assert trials.iloc[old_t-1].text == excerpt["old"]
        assert trials.iloc[new_t-1].text == excerpt["new"]
        assert reports.loc[reports.sequence_trial.between(old_t + 1, new_t - 1)].empty
        for j, name in [(1, "local_search_probability"), (2, "global_search_probability")]:
            clip[name] = case["mean"][j, start-1:end]
        clip.to_csv(source / f"subject_{subject}_excerpt.csv", index=False)
        reports.to_csv(source / f"subject_{subject}_same_choice_reports.csv", index=False)
        fig.text(left, .826, f"S{subject}  |  condition {condition} / Task {task}", fontsize=8, weight="bold")
        fig.text(left, .806, f"Trials {start}–{end}  ·  reports for choice {excerpt['choice']}", fontsize=6.5, color="#666666")
        x = clip.sequence_trial.to_numpy()
        axes = [fig.add_axes([left, bottom, .325, height])
                for bottom, height in [(.733, .055), (.548, .147), (.405, .103)]]
        for ax in axes:
            ax.axvspan(old_t, new_t, color="#F1E6D7", alpha=.8, zorder=0)
            ax.axvline(new_t, color="#9C8368", lw=.8, ls="--", zorder=1)
            ax.set_xlim(start-.5, end+.5)
        feedback = clip.feedback.to_numpy(dtype=float)
        assert np.isin(feedback, [0, 1]).all()
        axes[0].scatter(x[feedback == 1], feedback[feedback == 1], s=12,
                        facecolor="#5B6570", edgecolor="#5B6570", linewidth=.6, zorder=3)
        axes[0].scatter(x[feedback == 0], feedback[feedback == 0], s=17,
                        marker="x", color="#A7564E", linewidth=.9, zorder=3)
        axes[0].set(ylim=(-.4, 1.4), yticks=[0, 1], yticklabels=["Error", "Correct"])
        axes[0].tick_params(axis="x", bottom=False, labelbottom=False)
        for j, color in [(1, "#439D91"), (2, "#D88C49")]:
            axes[1].plot(x, case["mean"][j, start-1:end], color=color, lw=1.35,
                         marker="o", ms=2, zorder=3)
        axes[1].set(ylim=(0, .5), yticks=[0, .25, .5], ylabel="Pre-choice search\nprobability")
        axes[1].tick_params(axis="x", labelbottom=False)
        # Only real reports in this choice category are plotted; no carried-forward
        # line fabricates the instant when the participant's latent rule changed.
        axes[2].scatter(reports.sequence_trial, reports.content_index, s=20,
                        color="#416787", edgecolor="white", linewidth=.4, zorder=3)
        axes[2].set(ylim=(-.5, len(excerpt["labels"])-.5),
                    yticks=np.arange(len(excerpt["labels"])), yticklabels=excerpt["labels"],
                    xlabel="Trial", xticks=sorted(set([start, old_t, new_t, end])))
        axes[2].text(0, 1.08, "Observed report content", transform=axes[2].transAxes, fontsize=7)
        fig.text(left, .339, f"t{old_t}, choice {excerpt['choice']}  →  t{new_t}, choice {excerpt['choice']}",
                 fontsize=7, color="#666666")
        fig.text(left, .315, f"“{excerpt['old']}” → “{excerpt['new']}”", fontsize=8,
                 fontfamily="Noto Sans CJK JP")
        evidence.append({"subject": subject, **excerpt, "source_sha256": hashlib.sha256(trials_path.read_bytes()).hexdigest(),
                         "source": str(trials_path), "excerpt_trials": len(clip),
                         "same_choice_reports": len(reports), "full_record_trials": len(trials)})
    fig.text(.07, .276, "Shading: gap between the last old report and first new report. Dashed line: first new report.",
             fontsize=6.5, color="#666666")
    for left, letter, title, xlabel, ylabel, note in [
        (.14, "b", "Does this recur across events?", "Trials relative to report change",
         "Global-search\nprobability", "Pending\nVerified event set + matched controls"),
        (.63, "c", "Is learning better afterwards?", "Search probability before report change",
         "Accuracy after\nminus accuracy before", "Pending\nEvent-level behavioral analysis"),
    ]:
        fig.text(left-.12, .237, letter, fontsize=11, weight="bold")
        fig.text(left-.07, .237, title, fontsize=8, weight="bold")
        ax = fig.add_axes([left, .095, .325, .117])
        ax.set_facecolor("#FAFAFA")
        ax.set(xticks=[], yticks=[], xlabel=xlabel, ylabel=ylabel)
        ax.xaxis.label.set_fontsize(6.5)
        for spine in ax.spines.values():
            spine.set_color("#CBD0D5")
        ax.text(.5, .5, note, ha="center", va="center", transform=ax.transAxes,
                color="#92999F", fontsize=7, linespacing=1.8)
    fig.text(.07, .024, "Descriptive examples, not representative event statistics; model curves average 16 saved PF runs.",
             fontsize=6.5, color="#666666")
    fig.savefig(output / "Figure3_draft.png", dpi=450, facecolor="white")
    plt.close(fig)
    manifest = {"status": "descriptive_excerpts_with_pending_event_tests", "width_mm": 183, "height_mm": 185,
                "dpi": 450, "examples": evidence,
                "case_provenance": [{k: c[k] for k in ["subject", "seeds", "input_sha256"]} for c in cases],
                "versions": {"matplotlib": matplotlib.__version__, "numpy": np.__version__, "pandas": pd.__version__},
                "selection": "Direct inspection of oral text; readable same-choice changes, not selected by search peaks",
                "timing": "Report timestamps bracket a content change; cognitive onset unknown. Predictive state excludes current choice/feedback, parameters fit full sequence.",
                "pending": ["cohort event catalogue", "matched controls", "future behavior tests", "near-optimal state stability"]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    for filename in ["build_revision_fig3.py", "build_fig3.py", "README.md"]:
        shutil.copy2(Path(__file__).with_name(filename), output / filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
