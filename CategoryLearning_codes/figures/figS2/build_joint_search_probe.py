"""Visualize a completed S129 joint-move probe; no model execution.

Contract: distinguish improved candidate fit from a coordinate-search escape.
Quantitative grid, Python, 183 mm wide, PNG only (repository requirement).
All probe evaluations are retained in source CSV; PF repeats are numerical units.
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
import numpy as np
import pandas as pd

from src.Bayesian_state.optimization.diagnostics.search import flatten_hyperparams
from src.Bayesian_state.optimization.search.cd_v2 import canonical_point_key


def build(probe: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source_data"
    source.mkdir()
    inputs = [probe / name for name in ["summary.json", "validation_candidates.json",
              "joint_effects.json", "new_fine_evaluations.jsonl", "anchors.json",
              "trapped_start4_square.json", "coordinate_trap_validation/summary.json",
              "coordinate_trap_validation/scores.json", "pooled_old_candidates_32_repeats.json"]]
    summary = json.loads(inputs[0].read_text())
    candidates = json.loads(inputs[1].read_text())
    effects = json.loads(inputs[2].read_text())
    evaluations = [json.loads(s) for s in inputs[3].read_text().splitlines()]
    anchors = json.loads(inputs[4].read_text())
    assert len(anchors) == 5
    trap = json.loads(inputs[5].read_text())
    trap_validation = json.loads(inputs[6].read_text())
    trap_scores = json.loads(inputs[7].read_text())
    pooled = json.loads(inputs[8].read_text())
    previous_audit = Path(next(p for p in pooled["source_sha256"] if p.endswith("candidate_4_probabilities.npz"))).parent
    old_bootstrap_path = previous_audit / "paired_bootstrap.json"
    old_bootstrap = json.loads(old_bootstrap_path.read_text())
    inputs.append(old_bootstrap_path)
    pd.DataFrame([{k: v for k, v in r.items() if k != "hyperparams"} for r in trap_scores]).to_csv(
        source / "trapped_start4_scores.csv", index=False)
    context = json.loads((probe / "context.json").read_text())
    old_final_path = next(Path(p) for p in context["input_sha256"] if p.endswith("final_rescore.jsonl"))
    old_final = [json.loads(s) for s in old_final_path.read_text().splitlines()]
    inputs.append(old_final_path)
    old_labels = {canonical_point_key(r["hyperparams"]): f"Previous C{i+1}" for i, r in enumerate(old_final)}
    best_new_key = canonical_point_key(min(evaluations, key=lambda r: r["aggregated_error"])["hyperparams"])
    c4_score = summary["C4_fine_mean_nll"]
    rows = []
    for i, candidate in enumerate(candidates):
        key = canonical_point_key(candidate["hyperparams"])
        label = ("Previous best C4" if i == 0 else old_labels.get(key,
                 "Best new point" if key == best_new_key else "Distant-region best"))
        rows.append({"candidate": i + 1, "label": label,
                     "fine_mean_nll": candidate["fine_mean_nll"],
                     "fine_gain_total_from_C4": 255 * (c4_score - candidate["fine_mean_nll"]),
                     "validation_mean_nll": summary["validation_total_nll"][i] / 255,
                     "validation_gain_total_from_C4": summary["validation_gain_from_C4"][i],
                     "paired_mc_lower": summary["paired_mc_gain95_lower"][i],
                     "paired_mc_upper": summary["paired_mc_gain95_upper"][i],
                     **flatten_hyperparams(candidate["hyperparams"])})
    comparison = pd.DataFrame(rows)
    comparison.to_csv(source / "validation_comparison.csv", index=False)
    all_points = pd.DataFrame([{"evaluation_order": i + 1, "combination_index": r["combination_index"],
                               "mean_nll": r["aggregated_error"],
                               "fine_gain_total_from_C4": 255 * (c4_score - r["aggregated_error"]),
                               **flatten_hyperparams(r["hyperparams"])} for i, r in enumerate(evaluations)])
    all_points.to_csv(source / "all_new_points.csv", index=False)
    assert np.isfinite(all_points.mean_nll).all() and len(all_points) == summary["new_fine_points"]
    control_rows = []
    for i, effect in enumerate(effects):
        base = effect["anchor_mean_nll"]
        control_rows.append({"square": i + 1, "anchor": effect["anchor_label"],
                             "block_a": effect["block_a"], "block_b": effect["block_b"],
                             "single_a_gain": 255 * (base - effect["scores"]["single_a"]),
                             "single_b_gain": 255 * (base - effect["scores"]["single_b"]),
                             "joint_gain": effect["joint_gain_total_nll"],
                             "synergy_escape": effect["synergy_escape"]})
    controls = pd.DataFrame(control_rows)
    controls.to_csv(source / "all_paired_squares.csv", index=False)
    for path in inputs[:3] + [inputs[5]]:
        shutil.copy2(path, source / path.name)
    with np.load(probe / "validation/bootstrap.npz") as boot:
        gain = boot["gain_from_C4"]
        np.testing.assert_allclose(np.quantile(gain, [.025, .975], axis=1),
                                   [summary["paired_mc_gain95_lower"], summary["paired_mc_gain95_upper"]])
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                         "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
                         "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": .7})
    blue, orange, grey = "#487DA8", "#C57D47", "#8C949B"
    fig = plt.figure(figsize=(183 / 25.4, 175 / 25.4))
    fig.text(.11, .963, "S2.1 | Can joint moves improve the fit?", fontsize=11, weight="bold")
    fig.text(.11, .92, "S129 · bounded probe within the original parameter grid · 255 scored trials", fontsize=7.4)

    def panel(rect, letter, title):
        ax = fig.add_axes(rect)
        ax.set_title(f"{letter}  {title}", loc="left", weight="bold", pad=10)
        return ax

    ax = panel([.19, .61, .27, .23], "a", "Independent validation")
    for i, row in comparison.iloc[1:].iterrows():
        ax.plot([row.paired_mc_lower, row.paired_mc_upper], [i, i], color=blue, lw=2)
        ax.scatter(row.validation_gain_total_from_C4, i, s=23, color=blue, zorder=4)
    ax.axvline(0, color=grey, lw=.8)
    ax.set(yticks=range(1, len(comparison)), yticklabels=comparison.label.iloc[1:],
           ylim=(len(comparison) - .4, .4), xlabel="Total NLL improvement over C4")
    ax.text(0, -.42, "Positive: alternative fits better\n95% paired PF bootstrap intervals", transform=ax.transAxes, fontsize=6.4)
    ax = panel([.62, .61, .33, .23], "b", "Start 4: single vs joint moves")
    gains = [255 * (trap["points"][0]["aggregated_error"] - r["aggregated_error"]) for r in trap["points"][1:]]
    ax.bar(range(3), gains, color=grey, width=.6, alpha=.6, label="Fine")
    for i in range(3):
        ax.plot([i, i], [trap_validation["paired_mc_gain95_lower"][i+1],
                        trap_validation["paired_mc_gain95_upper"][i+1]], color=blue, lw=1.4)
        ax.scatter(i, trap_validation["gain_from_anchor"][i+1], s=22, color=blue, zorder=5,
                   label="Validation" if i == 0 else None)
    ax.axhline(0, color=".65", lw=.8)
    ax.set(xticks=range(3), xticklabels=["Only E_E", "Only c_G", "Both"], ylabel="Total NLL improvement over Start 4")
    ax.legend(loc="best")
    ax.text(0, -.42, "E_E: 0.413 → 0.610; c_G: 0.625 → 0.875\nAll other parameters fixed at Start 4 endpoint", transform=ax.transAxes, fontsize=6.3)

    ax = panel([.11, .16, .35, .25], "c", "All newly evaluated points")
    ax.scatter(all_points.evaluation_order, all_points.fine_gain_total_from_C4, color=grey, s=11, alpha=.8)
    ax.plot(all_points.evaluation_order, np.maximum.accumulate(all_points.fine_gain_total_from_C4), color=blue, lw=1.2,
            label="Best new so far")
    ax.axhline(0, color=orange, lw=.9, label="C4")
    ax.set(xlabel="New-point evaluation order", ylabel="Total NLL improvement over C4")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(float(all_points.fine_gain_total_from_C4.min()) * 1.1, 1)
    ax.set_yticks([-40, -10, -2, -1, 0], labels=["−40", "−10", "−2", "−1", "0"])
    ax.legend(loc="lower left")
    ax = panel([.70, .16, .25, .25], "d", "C2 versus C4")
    c2 = comparison[comparison.label == "Previous C2"].iloc[0]
    pi = pooled["labels"].index("C2")
    batch_rows = [
        {"batch": "Previous 16", "gain": -old_bootstrap["delta_total_nll"][1],
         "lower": -old_bootstrap["paired_mc_interval95_upper"][1], "upper": -old_bootstrap["paired_mc_interval95_lower"][1]},
        {"batch": "New 16", "gain": c2.validation_gain_total_from_C4,
         "lower": c2.paired_mc_lower, "upper": c2.paired_mc_upper},
        {"batch": "Pooled 32", "gain": pooled["gain_total_from_C4"][pi],
         "lower": pooled["paired_mc_gain95_lower"][pi], "upper": pooled["paired_mc_gain95_upper"][pi]}]
    pd.DataFrame(batch_rows).to_csv(source / "C2_C4_across_seed_batches.csv", index=False)
    for i, row in enumerate(batch_rows):
        color = orange if i == 2 else grey
        ax.plot([row["lower"], row["upper"]], [i, i], color=color, lw=2)
        ax.scatter(row["gain"], i, s=23, color=color, zorder=4)
        ax.text(row["gain"], i - .16, f"{row['gain']:+.3f}", ha="center", va="bottom", fontsize=6.3)
    ax.axvline(0, color=grey, lw=.8)
    ax.set(ylim=(2.5, -.6), yticks=range(3), yticklabels=[r["batch"] for r in batch_rows],
           xlabel="Total NLL improvement of C2 over C4")
    fig.text(.11, .027, "Same fitted sequence throughout. Numerical intervals do not establish parameter certainty or global optimality.", fontsize=6.3)
    fig.savefig(output / "S2_1_joint_search_probe.png", dpi=450, facecolor="white")
    plt.close(fig)
    write = {"archetype": "quantitative grid", "core_question": "Does a bounded joint-move probe improve C4, and do gains survive independent PF seeds?",
             "format": "PNG only per repository instructions", "width_mm": 183, "height_mm": 175, "dpi": 450,
             "displayed_square": "Archived Start 4 endpoint (2476) to C1 (1759), controls 1757 and 1857",
             "square_selection": "A real completed restart with full single-block grid coverage and no single-block improvement, but an observed joint improvement.",
             "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "new_point_count": len(all_points), "square_count": len(controls),
             "panel_c_scale": "Symmetric log; linear from -1 to 1. All 43 points retained.",
             "bootstrap_count": 4000, "scored_trials": 255, "participants": 1,
             "test": "No hypothesis test; percentile bootstrap of paired numerical PF repeats. No multiple-comparison inference.",
             "versions": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__}}
    (output / "manifest.json").write_text(json.dumps(write, indent=2) + "\n")
    (output / "CAPTIONS.md").write_text(
        "# Joint-search probe, S129\n\n"
        "a, Fixed candidates are evaluated on the same 255 scored trials with a fresh common PF seed family (128 particles, 16 repeats). "
        "Dots give total NLL improvement relative to C4. Lines are 95% percentile intervals from 4,000 paired resamples of PF repeats. "
        "These are numerical intervals conditional on the fitted sequence and shortlist, not parameter intervals or out-of-sample evidence. "
        "Alternative identities and complete parameters are in source_data/validation_comparison.csv. "
        "b, The archived Start 4 endpoint, whose full fine-grid single-block axes were evaluated without improvement. "
        "Changing E_E from 0.413010 to 0.610274 and c_G from 0.625 to 0.875 jointly reaches C1. "
        "Grey bars give fine-budget improvements relative to Start 4; blue dots and 95% paired PF intervals give independent validation, each relative to its same-budget anchor. "
        "Positive values improve on Start 4, not C4. This selected example does not describe the frequency of traps across the whole space. "
        "All newly proposed comparisons are supplied in all_paired_squares.csv and joint_effects.json; the archived example is in trapped_start4_square.json. "
        "c, Every newly evaluated point, with running best and C4 reference, at the archived fine budget and common seeds. "
        "The symmetric-log vertical scale is linear between -1 and 1; no points are excluded. "
        "d, C2 relative to C4 with the previous 16 PF repeats, the new independent 16 repeats, and all 32 pooled repeats. "
        "Every repeat uses 128 particles. Probabilities are pooled before taking logs; intervals are paired PF bootstraps within each comparison. "
        "The pooled comparison uses C1/C2/C4 only, each with 32 repeats; the new alternative has 16 and is not mixed into that ranking. "
        "A finite probe can find a counterexample to the previous best, but cannot certify global optimality when none is found.\n")
    print(json.dumps({"output": str(output), "new_points": len(all_points), "squares": len(controls)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.probe, args.output)
