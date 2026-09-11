"""Draft Fig3 from saved Model0826 runs; no fitting or simulated placeholder data.

Contract: describe continuous search and belief dynamics in two fitted cases.
Python asymmetric quantitative figure, 183 x 225 mm, PNG only. Event validation
and prospective behavioral effects remain explicitly empty panels.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


CASES = (
    (129, 1, 1, "pipeline_20260908_v1", "evaluation"),
    (229, 2, 3, "pmh_20260909_v1", "evaluation_20260911_v1"),
)
FIELDS = (
    "predictive_strategy_exploit", "predictive_strategy_local_explore",
    "predictive_strategy_global_explore", "predictive_replacement_fraction",
    "predictive_failure_pressure", "predictive_mastery_evidence",
)
COLORS = ("#7696B3", "#439D91", "#D88C49")
LABELS = ("Retain", "Local search", "Global search")


def load_case(subject: int, condition: int, task: int, run: str,
              evaluation: str, source: Path) -> dict:
    """Aggregate saved pre-choice PF marginals equally across independent runs."""
    base = Path(f"results/model_0826/cond{condition}/subject_{subject}/{run}/models/PMH")
    summary_path = base / f"simulation/subjects/subject_{subject}.json"
    summary = json.loads(summary_path.read_text())
    raw_path = summary_path.parent / summary["raw_runs_ref"]["path"]
    oral_path = base / evaluation / "oral_alignment_center_mode/oral_mass_probabilities.npz"
    scalar_runs, beliefs, seeds = [], [], []
    with gzip.open(raw_path, "rb") as stream:
        while True:
            try:
                record = pickle.load(stream)
            except EOFError:
                break
            assert record["subject_id"] == subject and record["condition"] == condition
            state = record["state_log"]
            scalar_runs.append(np.array([state[field] for field in FIELDS], dtype=float))
            beliefs.append(np.array(state["marginal_prior"], dtype=float))
            seeds.append(record["trajectory_seed"])
    values, belief_runs = np.array(scalar_runs), np.array(beliefs)
    assert len(values) == summary["raw_runs_ref"]["count"] == 16
    assert np.isfinite(values).all() and np.isfinite(belief_runs).all()
    assert np.all((values >= -1e-12) & (values <= 1 + 1e-12))
    np.testing.assert_allclose(values[:, :3].sum(axis=1), 1, atol=1e-10)
    np.testing.assert_allclose(belief_runs.sum(axis=2), 1, atol=1e-10)
    mean, belief = values.mean(axis=0), belief_runs.mean(axis=0)
    n, h = belief.shape
    with np.load(oral_path) as archive:
        assert int(archive["subjects"][0]) == subject
        assert int(archive["conditions"][0]) == condition
        assert float(archive["oral_center_sigma"][0]) == 0.05
        oral = archive["oral_mass"][0, :n, :h].copy()
        valid = archive["valid_oral_report"][0, :n].astype(bool)
    assert oral.shape == belief.shape and np.isfinite(oral[valid]).all()
    np.testing.assert_allclose(oral[valid].sum(axis=1), 1, atol=1e-10)
    table = pd.DataFrame({"trial": np.arange(1, n + 1), "valid_oral_report": valid})
    for i, field in enumerate(FIELDS):
        table[field] = mean[i]
        table[field + "_seed_sd"] = values[:, i].std(axis=0, ddof=1)
    table.to_csv(source / f"subject_{subject}_trial_metrics.csv", index=False)
    pd.DataFrame({"trial": np.repeat(np.arange(1, n + 1), h),
                  "rule_id": np.tile(np.arange(h), n),
                  "online_prior": belief.ravel(), "oral_mass": oral.ravel(),
                  "valid_oral_report": np.repeat(valid, h)}).to_csv(
                      source / f"subject_{subject}_rule_mass.csv", index=False)
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in (summary_path, raw_path, oral_path)}
    return dict(subject=subject, condition=condition, task=task, n=n, h=h,
                mean=mean, belief=belief, oral=oral, valid=valid, seeds=seeds,
                input_sha256=provenance)


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source_data"
    source.mkdir()
    cases = [load_case(*spec, source) for spec in CASES]
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "legend.fontsize": 7, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig = plt.figure(figsize=(183 / 25.4, 225 / 25.4), facecolor="white")
    fig.text(.09, .974, "Figure 3 | Search and rule dynamics", fontsize=12, weight="bold")
    fig.text(.09, .951, "Initial draft  ·  two fitted cases  ·  Model 0826 PMH", color="#666666")
    fig.text(.025, .919, "a", fontsize=11, weight="bold")
    fig.text(.09, .919, "Within-subject search and rule evidence", fontsize=9, weight="bold")
    fig.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(COLORS, LABELS)],
               loc="center", bbox_to_anchor=(.52, .897), ncol=3, handlelength=1.2)
    mass_cmap = LinearSegmentedColormap.from_list("rule_mass", ["#FFFFFF", "#21618C"])
    mass_cmap.set_bad("#E8E8E8")
    image_handle = None
    for case, left in zip(cases, [.09, .565]):
        n, h = case["n"], case["h"]
        x, mean = np.arange(1, n + 1), case["mean"]
        fig.text(left, .873, f"S{case['subject']}  |  condition {case['condition']} / Task {case['task']}",
                 fontsize=8, weight="bold")
        fig.text(left, .857, f"{n} trials  ·  {h} rules  ·  16 PF repeats", fontsize=6.5, color="#666666")
        ax = fig.add_axes([left, .742, .375, .099])
        ax.stackplot(x, mean[:3], colors=COLORS, linewidth=0)
        ax.set(ylim=(0, 1), xlim=(.5, n + .5), yticks=[0, .5, 1], ylabel="Search\nprobability")
        ax.tick_params(axis="x", labelbottom=False)
        ax = fig.add_axes([left, .672, .375, .043])
        ax.plot(x, mean[3], color="#525B65", linewidth=.65)
        ax.set(ylim=(0, .2), xlim=(.5, n + .5), yticks=[0, .2], ylabel="Replaced\nfraction")
        ax.tick_params(axis="x", labelbottom=False)
        for bottom, array, label in [(.563, case["belief"], "Model\nrule ID"),
                                     (.454, np.where(case["valid"][:, None], case["oral"], np.nan),
                                      "Oral\nrule ID")]:
            ax = fig.add_axes([left, bottom, .375, .082])
            image_handle = ax.imshow(array.T, aspect="auto", interpolation="nearest", origin="lower",
                                     extent=(.5, n + .5, -.5, h - .5), cmap=mass_cmap, vmin=0, vmax=1)
            ax.set(ylabel=label, yticks=[0, h - 1])
            if bottom < .5:
                ax.set_xlabel("Trial")
                ax.set_xticks([1, n // 2, n])
            else:
                ax.tick_params(axis="x", labelbottom=False)
    color_ax = fig.add_axes([.952, .47, .011, .165])
    cb = fig.colorbar(image_handle, cax=color_ax, ticks=[0, .5, 1])
    fig.text(.942, .648, "P(rule)", fontsize=6.5)
    fig.text(.09, .402, "Online model means; oral state shown at valid report trials only (grey = no valid report).",
             fontsize=6.5, color="#666666")

    fig.text(.025, .377, "b", fontsize=11, weight="bold")
    fig.text(.09, .377, "Continuous search profiles across the two cases", fontsize=9, weight="bold")
    max_trials = max(c["n"] for c in cases)
    for j, (label, color) in enumerate(zip(LABELS, COLORS)):
        cmap = LinearSegmentedColormap.from_list(f"strategy_{j}", ["#FFFFFF", color])
        cmap.set_bad("#E8E8E8")
        data = np.full((len(cases), max_trials), np.nan)
        for i, case in enumerate(cases):
            data[i, :case["n"]] = case["mean"][j]
        ax = fig.add_axes([.09 + j * .302, .288, .247, .060])
        ax.imshow(data, aspect="auto", interpolation="nearest", extent=(.5, max_trials + .5, 1.5, -.5),
                  vmin=0, vmax=1, cmap=cmap)
        ax.set_title(label, fontsize=8, pad=5)
        ax.set(yticks=[0, 1], yticklabels=["S129", "S229"] if j == 0 else ["", ""],
               xticks=[1, 256, 1088], xlabel="Trial")
    fig.text(.09, .236, "White → colour: probability 0 → 1. Grey: recording ended. No time normalization or hard labels.",
             fontsize=6.5, color="#666666")

    for left, letter, title, xlabel, ylabel, note in [
        (.09, "c", "Search around oral revision", "Trial relative to verified oral revision",
         "Search probability", "Pending\nVerified revision events\nand matched control times"),
        (.565, "d", "Search and subsequent learning", "Pre-revision global-search probability",
         "Change in\nobserved accuracy", "Pending\nRevision-event analysis\nand validation of future outcomes"),
    ]:
        fig.text(left - .065, .213, letter, fontsize=11, weight="bold")
        fig.text(left, .213, title, fontsize=8, weight="bold")
        ax = fig.add_axes([left, .081, .375, .110])
        ax.set_facecolor("#FAFAFA")
        ax.set(xticks=[], yticks=[], xlabel=xlabel, ylabel=ylabel)
        ax.xaxis.label.set_fontsize(6.5)
        ax.yaxis.label.set_fontsize(6.5)
        for spine in ax.spines.values():
            spine.set_color("#C6CBD0")
        ax.text(.5, .5, note, ha="center", va="center", transform=ax.transAxes,
                fontsize=7.5, color="#92999F", linespacing=1.7)
    fig.text(.09, .022, "Descriptive fits to complete sequences; no population, causal or held-out prediction claim.",
             fontsize=6.5, color="#666666")
    fig.savefig(output / "Figure3_draft.png", dpi=450, facecolor="white")
    plt.close(fig)
    pd.DataFrame([{"subject": c["subject"], "condition": c["condition"], "trials": c["n"],
                   "valid_reports": int(c["valid"].sum()),
                   **{label: float(c["mean"][j].mean()) for j, label in enumerate(LABELS)}}
                  for c in cases]).to_csv(source / "subject_summary.csv", index=False)
    manifest = {"status": "draft_with_explicit_placeholders", "panels": {"a": "two individual cases",
                "b": "two-case descriptive profiles", "c": "pending", "d": "pending"},
                "width_mm": 183, "height_mm": 225, "dpi": 450, "python_backend": matplotlib.__version__,
                "numpy_version": np.__version__, "pandas_version": pd.__version__,
                "cases": [{k: c[k] for k in ["subject", "condition", "task", "n", "h", "seeds", "input_sha256"]}
                          for c in cases], "checks": "finite probabilities; shapes; normalization; IDs; sigma=0.05; 16 runs"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(__file__, output / "build_fig3.py")
    shutil.copy2(Path(__file__).with_name("README.md"), output / "README.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are refused")
    build(parser.parse_args().output)
