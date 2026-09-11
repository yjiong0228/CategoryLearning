"""Plot S129 search coverage and the numerical advantage of archived finalists.

Reads the completed search and a paired finalist replay; performs no fitting.
Distances describe the configured grid, not a posterior or a confidence region.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.Bayesian_state.optimization.diagnostics.search import flatten_hyperparams
from src.Bayesian_state.simulation.config import expand_profile_candidate_hyperparams


BLUE, ORANGE, INK = "#487DA8", "#C57D47", "#333333"
COLORS = ["#67949A", "#9B7393", "#A19569", BLUE]
PARAMS = ["capacity", "persistent_execution", "gamma", "event_after_correct",
          "event_after_error", "global_search", "global_search_failure_gain",
          "accumulator_logit_gain", "beta_init", "increase_rate", "decrease_rate"]
BLOCK_LABELS = ["Workspace (M, χ)", "Memory γ", "Reactive event (E_C, E_E)",
                "Global search (g₀, c_G)", "Failure gain c_A", "Initial precision β₀",
                "Support update η₊", "Refutation update η₋"]


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def block_distance(point: dict, reference: dict, space: dict) -> float:
    """RMS normalized distance, giving each of the eight search blocks equal weight."""
    point_flat = expand_profile_candidate_hyperparams(point)
    reference_flat = expand_profile_candidate_hyperparams(reference)
    block_ms = []
    for block, spec in space.items():
        alternatives = [expand_profile_candidate_hyperparams({block: v}) for v in spec["values"]]
        squared = []
        for key in alternatives[0]:
            values = np.array([float(v[key]) for v in alternatives])
            span = float(np.ptp(values))
            if span > 0:
                squared.append(((float(point_flat[key]) - float(reference_flat[key])) / span) ** 2)
        block_ms.append(float(np.mean(squared)) if squared else 0.)
    return float(np.sqrt(np.mean(block_ms)))


def build(pipeline: Path, audit: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    sources = output / "source_data"
    sources.mkdir()
    optimization = pipeline / "models/PMH/optimization/subject_129"
    inputs = [optimization / "all_combinations.jsonl", optimization / "final_rescore.jsonl",
              optimization / "restart_summary.json", pipeline / "configs/PMH_hyper.yaml",
              audit / "paired_bootstrap.json", audit / "replay_scores.jsonl"]
    all_rows = [json.loads(s) for s in inputs[0].read_text().splitlines()]
    fine = [r for r in all_rows if r["stage"] == "fine"]
    final = [json.loads(s) for s in inputs[1].read_text().splitlines()]
    restarts = json.loads(inputs[2].read_text())["fine"]
    config = yaml.safe_load(inputs[3].read_text())
    mc = json.loads(inputs[4].read_text())
    replay = [json.loads(s) for s in inputs[5].read_text().splitlines()]
    assert len(fine) == 2499 and len(final) == len(replay) == 4
    assert mc["score_trials"] == 255 and mc["best_candidate"] == 4
    np.testing.assert_allclose([r["difference"] for r in replay], 0, atol=1e-10)
    for stage in ("coarse", "fine"):
        points = [canonical(r["hyperparams"]) for r in all_rows if r["stage"] == stage]
        assert len(set(points)) == len(points)
    lookup = {r["combination_index"]: r for r in fine}
    ref = final[3]["hyperparams"]
    fine_ref = lookup[final[3]["search_combination_index"]]["aggregated_error"]
    space = config["stages"]["fine"]["hyperparam_space"]
    blocks = list(space)
    n = mc["score_trials"]
    table = pd.DataFrame([{
        "combination_index": r["combination_index"], "first_restart": r["restart_id"] + 1,
        "mean_nll": r["aggregated_error"], **flatten_hyperparams(r["hyperparams"]),
        "distance_from_C4": block_distance(r["hyperparams"], ref, space),
    } for r in fine]).sort_values("mean_nll").reset_index(drop=True)
    assert np.isfinite(table[["mean_nll", *PARAMS]].to_numpy()).all()
    table["fine_rank"] = np.arange(1, len(table) + 1)
    table["delta_total_from_fine_min"] = (table.mean_nll - table.mean_nll.min()) * n
    table["delta_total_from_C4_fine"] = (table.mean_nll - fine_ref) * n
    columns = ["combination_index", "first_restart", "fine_rank", "mean_nll",
               "delta_total_from_fine_min", "delta_total_from_C4_fine", "distance_from_C4",
               *PARAMS, "strategy_id", "hyperparam_signature"]
    table[columns].to_csv(sources / "fine_parameters_corrected.csv", index=False)
    summary_rows = []
    for i, row in enumerate(final):
        score = lookup[row["search_combination_index"]]["aggregated_error"]
        assert canonical(lookup[row["search_combination_index"]]["hyperparams"]) == canonical(row["hyperparams"])
        summary_rows.append({"candidate": f"C{i+1}", "search_combination_index": row["search_combination_index"],
                             "fine_mean_nll": score, "final_mean_nll": row["aggregated_error"],
                             "fine_delta_from_C4": (score - fine_ref) * n,
                             "final_delta_from_C4": mc["delta_total_nll"][i],
                             "mc_lower": mc["paired_mc_interval95_lower"][i],
                             "mc_upper": mc["paired_mc_interval95_upper"][i],
                             "bootstrap_win_frequency": mc["bootstrap_selected_frequency"][i],
                             **{k: flatten_hyperparams(row["hyperparams"])[k] for k in PARAMS}})
    scores = pd.DataFrame(summary_rows)
    scores.to_csv(sources / "finalist_comparison.csv", index=False)
    start_rows = []
    for row in restarts:
        initial = lookup[row["initial_combination_index"]]
        start_rows.append({"start": row["restart_id"] + 1,
                           "initial_combination_index": row["initial_combination_index"],
                           "end_combination_index": row["best_combination_index"],
                           "end_fine_mean_nll": row["best_error"], "stopped_by": row["stopped_by"],
                           **{k: flatten_hyperparams(initial["hyperparams"])[k] for k in PARAMS}})
    starts = pd.DataFrame(start_rows)
    starts.to_csv(sources / "fine_start_parameters.csv", index=False)
    coverage = []
    for block, label in zip(blocks, BLOCK_LABELS):
        counts = []
        for rows in (fine, final):
            neighbors = [r for r in rows if canonical(r["hyperparams"][block]) != canonical(ref[block])
                         and all(canonical(r["hyperparams"][k]) == canonical(ref[k]) for k in blocks if k != block)]
            counts.append(len(neighbors))
        coverage.append({"block": block, "label": label, "grid_alternatives": len(space[block]["values"]) - 1,
                         "fine_evaluated": counts[0], "final_evaluated": counts[1]})
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(sources / "C4_one_block_coverage.csv", index=False)
    eta_block = blocks[-1]
    local_ids = [r["combination_index"] for r in fine
                 if all(canonical(r["hyperparams"][k]) == canonical(ref[k]) for k in blocks if k != eta_block)]
    local = table[table.combination_index.isin(local_ids)].sort_values("decrease_rate")
    assert len(local) == 13 and coverage.fine_evaluated.tolist() == [0] * 7 + [12]
    local[columns].to_csv(sources / "C4_eta_minus_profile.csv", index=False)
    thresholds = {str(v): int((table.delta_total_from_fine_min <= v).sum()) for v in (.5, 1., 2.)}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 8,
                         "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "legend.fontsize": 6.2, "legend.frameon": False,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7})
    outputs = []

    def page(title: str, subtitle: str, height: int):
        fig = plt.figure(figsize=(183 / 25.4, height / 25.4))
        fig.text(.10, .96, title, fontsize=11, weight="bold")
        fig.text(.10, .916, subtitle, fontsize=7.4, color=INK)
        return fig

    def panel(fig, rect, label, title):
        ax = fig.add_axes(rect)
        ax.set_title(f"{label}  {title}", loc="left", weight="bold", pad=10)
        return ax

    def save(fig, name, note):
        fig.text(.10, .022, note, fontsize=6.2, color=INK)
        fig.savefig(output / name, dpi=450, facecolor="white")
        plt.close(fig)
        outputs.append(name)

    fig = page("S2.1 | How much does C4 win?",
               "S129 · C1–C4 are the four highest-ranked fine-search points, then rescored.", 176)
    ax = panel(fig, [.11, .59, .35, .25], "a", "Final score differences")
    ax.axvline(0, color=".65", lw=.8)
    for i in range(3):
        ax.plot([scores.mc_lower[i], scores.mc_upper[i]], [i, i], color=COLORS[i], lw=2)
        ax.scatter(scores.final_delta_from_C4[i], i, color=COLORS[i], s=26, zorder=4)
        ax.text(scores.final_delta_from_C4[i], i - .18, f"{scores.final_delta_from_C4[i]:.3f}",
                ha="center", va="bottom", fontsize=6.5)
    ax.set(xlim=(-.5, 1.6), ylim=(2.6, -.65), yticks=range(3), yticklabels=["C1 − C4", "C2 − C4", "C3 − C4"],
           xlabel="Δ total NLL (positive: C4 fits better)")
    ax.text(.0, -.34, "Bars: 95% paired numerical bootstrap interval", transform=ax.transAxes, fontsize=6.1)
    ax = panel(fig, [.61, .59, .34, .25], "b", "Who ranks first on resampling?")
    freq = scores.bootstrap_win_frequency.to_numpy() * 100
    ax.bar(range(4), freq, color=COLORS, width=.6)
    for i, v in enumerate(freq):
        ax.text(i, v + 2.8, f"{v:.1f}%", ha="center", fontsize=7)
    ax.set(xticks=range(4), xticklabels=scores.candidate, ylim=(0, 100), ylabel="Fraction of resamples (%)")
    ax.text(0, -.34, "4,000 resamples of the same 16 PF repeats", transform=ax.transAxes, fontsize=6.1)
    ax = panel(fig, [.11, .15, .35, .25], "c", "Ranking changes on rescoring")
    final_ranks = scores.final_mean_nll.rank(method="first").astype(int).to_numpy()
    for i in range(4):
        ax.plot([0, 1], [i + 1, final_ranks[i]], "o-", color=COLORS[i], lw=1.6 if i == 3 else 1, ms=4)
        ax.text(-.09, i + 1, f"C{i+1}", ha="right", va="center", color=COLORS[i])
        ax.text(1.08, final_ranks[i], f"C{i+1}" + (" selected" if i == 3 else ""), va="center", color=COLORS[i])
    ax.set(xlim=(-.30, 1.58), ylim=(4.5, .5), yticks=range(1, 5), ylabel="Rank among these four candidates",
           xticks=[0, 1], xticklabels=["Fine\n64 particles × 8 seeds", "Final\n128 particles × 16 seeds"])
    ax = panel(fig, [.61, .15, .34, .25], "d", "Changing only η₋ around C4")
    ax.axhline(0, color=".65", lw=.8)
    ax.plot(local.decrease_rate, local.delta_total_from_C4_fine, "o-", color=".60", ms=3, lw=.8,
            label="Fine: 13 grid values")
    for i, label in [(1, "C2"), (3, "C4")]:
        ax.scatter(scores.decrease_rate[i], scores.final_delta_from_C4[i], marker="D", s=30,
                   color=COLORS[i], zorder=5, label=f"Final: {label}")
    ax.set(xlabel="Refutation update η₋", ylabel="Δ total NLL from C4 at same budget")
    ax.legend(loc="upper center", bbox_to_anchor=(.53, 1.))
    save(fig, "S2_1_best_advantage.png",
         "Intervals describe PF numerical variation on the fitted sequence; they are not parameter confidence intervals.")

    fig = page("S2.1 | How much of the search is supported?",
               "Fine support is preserved, but starts and conditional search paths limit what is tested.", 190)
    ax = panel(fig, [.11, .58, .35, .26], "a", "Many evaluated points score closely")
    ax.plot(table.fine_rank, table.delta_total_from_fine_min, color=BLUE, lw=1.2)
    for value, count in thresholds.items():
        ax.axhline(float(value), color=".8", lw=.6, ls=":")
        ax.scatter(count, float(value), s=18, color=ORANGE, zorder=3)
    ax.set(xlabel="Fine-score rank (log scale)", ylabel="Δ total NLL from fine-search minimum",
           xscale="log", yscale="symlog", ylim=(-.025, 30), xlim=(1, 2800))
    ax.set_yticks([0, .5, 1, 2, 5, 10, 25], labels=["0", "0.5", "1", "2", "5", "10", "25"])
    ax.text(.03, .96, "ΔNLL ≤ 0.5: 31 points\nΔNLL ≤ 1: 102 points\nΔNLL ≤ 2: 622 points",
            va="top", transform=ax.transAxes, fontsize=6.7)
    ax = panel(fig, [.62, .58, .33, .26], "b", "Score versus distance from C4")
    near = table.delta_total_from_fine_min <= 2
    ax.scatter(table.distance_from_C4, table.delta_total_from_fine_min, s=3, color=".80", alpha=.65, rasterized=True)
    ax.scatter(table.loc[near, "distance_from_C4"], table.loc[near, "delta_total_from_fine_min"],
               s=4, color=BLUE, alpha=.65, rasterized=True)
    for i, row in scores.iterrows():
        point = table[table.combination_index == row.search_combination_index].iloc[0]
        ax.scatter(point.distance_from_C4, point.delta_total_from_fine_min, s=20, marker="D", color=COLORS[i], zorder=5)
    ax.axhline(2, color=ORANGE, ls=":", lw=.8)
    ax.set(xlabel="Normalized block distance from C4", ylabel="Δ total NLL from fine-search minimum", ylim=(-.7, 26))
    ax.text(.97, .96, "Blue: ΔNLL ≤ 2\nFour diamonds: finalists", ha="right", va="top", transform=ax.transAxes, fontsize=6.5)
    ax = panel(fig, [.10, .15, .39, .27], "c", "The four starts are similar")
    ax.axis("off")
    cells = [[f"Start {int(r.start)}", f"{r.event_after_error:.3f}", f"{r.global_search:.2f}",
              str(int(r.end_combination_index))] for r in starts.itertuples()]
    tab = ax.table(cellText=cells, colLabels=["Fine start", "E_E", "g₀", "End point ID"],
                   cellLoc="center", colWidths=[.28, .20, .20, .32], bbox=[0, .39, 1, .61])
    tab.auto_set_font_size(False)
    tab.set_fontsize(6.6)
    for (ri, ci), cell in tab.get_celld().items():
        cell.set_edgecolor("#DFE3E6")
        cell.set_linewidth(.5)
        if ri == 0:
            cell.set_facecolor("#E8EDF1")
    ax.text(0, .27, "All share M = 4, χ = 0, γ = 0.97, E_C = 0.10,\nc_A = 0, c_G = 1, β₀ = 5, η₊ = 0.04, η₋ = 0.15.",
            va="top", fontsize=6.7, linespacing=1.6)
    ax.text(0, -.04, "Starts 1 and 2 reach the same fine-search point.", color=ORANGE, fontsize=6.7)
    ax = panel(fig, [.77, .15, .18, .27], "d", "Checks around C4")
    y = np.arange(8)
    for i, row in coverage.iterrows():
        ax.barh(i - .13, row.fine_evaluated / row.grid_alternatives, height=.24, color=".68")
        ax.barh(i + .13, row.final_evaluated / row.grid_alternatives, height=.24, color=BLUE)
        ax.text(1.02, i, f"{row.fine_evaluated}/{row.grid_alternatives}", va="center", fontsize=5.8)
    ax.set(yticks=y, yticklabels=BLOCK_LABELS, ylim=(7.6, -.6), xlim=(0, 1.43), xticks=[0, 1],
           xticklabels=["0", "All"], xlabel="Grid alternatives tested")
    ax.tick_params(axis="y", labelsize=6)
    ax.plot([], [], color=".68", lw=4, label="Fine")
    ax.plot([], [], color=BLUE, lw=4, label="Final")
    ax.legend(loc="upper center", bbox_to_anchor=(.45, -.20), ncol=2)
    save(fig, "S2_1_search_coverage.png",
         "Grid distance is descriptive; ΔNLL thresholds do not define confidence regions. Panel d fixes all other blocks at C4.")
    manifest = {"subject": 129, "condition": 1, "model": "PMH Model0826", "score_trials": n,
                "coarse_points": sum(r["stage"] == "coarse" for r in all_rows), "fine_points": len(fine),
                "fine_grid_product": int(np.prod([len(v["values"]) for v in space.values()], dtype=np.int64)),
                "near_score_counts": thresholds, "unique_transition_profiles": int(table.strategy_id.nunique()),
                "memory_gamma_values": sorted(table.gamma.unique().tolist()),
                "block_distance_definition": "Within each configured block, mean squared numeric differences divided by each leaf's fine-grid range; square root of the mean over eight blocks. Chi is coded 0/1. Dependent initial_event_probability is retained in its packed block. Descriptive and encoding-dependent, not a posterior metric.",
                "outputs": outputs, "format": "PNG only; repository override", "width_mm": 183, "dpi": 450,
                "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "versions": {"numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
                "qa": {"four_original_scores_reproduced": True, "unique_within_stage_points": True,
                       "all_model_parameters_finite": True, "all_13_eta_minus_neighbors_present": True}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "CAPTIONS.md").write_text(
        "# S129 parameter-estimation audit\n\n"
        "## Best advantage\n\n"
        "a, Total choice NLL differences relative to C4 on 255 scored trials (trial 1 initializes the state and is excluded from scoring). "
        "Positive values favour C4. Lines show percentile intervals from 4,000 paired resamples of the original 16 particle-filter repeats, with probabilities averaged before taking logs. "
        "They quantify numerical variability conditional on this sequence, shortlist and seed set; not parameter uncertainty, independent validation, or finite-particle bias. "
        "b, The fraction of those resamples selecting each of the four candidates. This is not a probability of global optimality. "
        "c, Candidate IDs follow the fine-search score ordering; the independent final seed family and larger particle/repeat budget change the ordering. "
        "d, One-block grid profile holding all other C4 parameters fixed. Each budget uses its own C4 reference; only C2 and C4 received final-budget evaluation along this profile.\n\n"
        "## Search coverage\n\n"
        "a, Sorted scores of all 2,499 distinct evaluated fine points (log rank; symmetric-log NLL scale, linear near zero). "
        "Threshold counts describe evaluated points, not confidence-region size or parameter-space volume. "
        "b, Those same scores versus descriptive distance from C4: range-normalize each expanded numeric coordinate, average squared distances within each packed block, "
        "then take the RMS across eight blocks. The encoding and grid bounds affect this distance; point density is determined by adaptive coordinate search. "
        "Blue points satisfy ΔNLL ≤ 2 relative to the fine minimum; they are not established to form a connected cluster. "
        "c, The four fine initializations inherited from the coarse shortlist differ only in E_E and g₀. End point IDs are saved combination IDs. "
        "d, Fraction of alternatives evaluated by changing exactly one block while keeping all other blocks at C4. "
        "Grey labels give fine-stage counts/available alternatives; final scoring covers only one alternative in η₋. "
        "This does not describe coverage around earlier incumbents. It shows why final-budget local optimality has not been checked.\n",
        encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.pipeline, args.audit, args.output)
