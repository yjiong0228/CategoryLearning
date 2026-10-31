"""Freeze a Model 1031 structure before examining evaluation-subject outcomes.

This coordinator reads immutable outputs from the bounded study. Two structures
receive an independent numerical check at fixed training-selected parameters.
The resulting decision is a working model choice, with numerical limitations
reported separately from the choice itself.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from .model_1031_finalization import (
    Scheduler, aggregate, digest, fit_jobs, validate_source, write_new,
)


def summarize(rows: list[dict]) -> dict[str, dict]:
    result = {}
    for variant in dict.fromkeys(r["variant"] for r in rows):
        group = [r for r in rows if r["variant"] == variant]
        scores = np.asarray([r["test_nll_per_trial"] for r in group])
        result[variant] = dict(
            subjects=len(group), nll=float(scores.mean()),
            descriptive_subject_se=float(scores.std(ddof=1) / np.sqrt(len(scores))) if len(scores) > 1 else 0.,
            # Subjects reuse seed labels. The triangle bound avoids assuming
            # their Monte Carlo errors are independent.
            numerical_se=float(np.mean([r["mc_se_per_trial"] for r in group])),
            energy=float(np.mean([r["energy"] for r in group])),
            condition_nll={str(c): float(np.mean([r["test_nll_per_trial"] for r in group if r["condition"] == c]))
                           for c in sorted({r["condition"] for r in group})},
            numerically_limited=sum(r["numerical_status"] != "within_working_tolerance" for r in group),
            concentrated_systems=sum(r["max_system_weight"] > .8 for r in group))
    return result


def independent_seed_aggregate(records: list[dict]) -> dict:
    """Pool correlated particle-count estimates within a seed before MC error."""
    grouped = []
    for seed in sorted({r["job"]["seed"] for r in records}):
        cluster = [r for r in records if r["job"]["seed"] == seed]
        grouped.append({key: float(-logsumexp([-r[key] for r in cluster]) + np.log(len(cluster)))
                        for key in ("nll", "train_nll")})
    counts = [sum(r["job"]["seed"] == seed for r in records) for seed in {r["job"]["seed"] for r in records}]
    if len(set(counts)) != 1:
        raise ValueError("Incomplete particle-count pairing across independent seeds")
    return dict(**aggregate(grouped, conditional=True), independent_seeds=len(grouped),
                raw_systems=len(records), particle_counts=sorted({r["job"]["particles"] for r in records}))


def pool_replicates(output: Path, role: str) -> None:
    register = validate_source(output)
    policy = register["policy"]
    original = json.loads((output / f"{role}_evaluation.json").read_text())
    seed_base = policy["seed"] + (2000 if role == "development" else 11000)
    seeds = set(range(seed_base, seed_base + policy["search"]["final_repeats"]))
    records = [json.loads(p.read_text()) for p in (output / "jobs").glob("*.json")]
    records = [r for r in records if r["job"]["kind"] == "fit" and r["job"]["scope"] == "full" and r["job"]["seed"] in seeds]
    rows = []
    for row in original["rows"]:
        matching = [r for r in records if r["job"]["subject"] == row["subject"] and r["job"]["config"] == row["candidate"]["id"]]
        score = independent_seed_aggregate(matching)
        n = row["test_trials"]
        np.testing.assert_allclose(score["nll"] / n, row["test_nll_per_trial"], atol=1e-10)
        rows.append(dict(row, mc_se_per_trial=score["mc_se"] / n,
            max_system_weight=score["max_system_weight"], original_mc_se_per_trial=row["mc_se_per_trial"],
            replicate_pooling={k: score[k] for k in ("independent_seeds", "raw_systems", "particle_counts")}))
    write_new(output / f"{role}_evaluation_grouped.json", dict(rows=rows,
        explanation="Identical configurations can share evaluations at different particle counts. Within-seed systems are pooled before computing Monte Carlo uncertainty; sequence likelihood scores are unchanged.",
        coordinator_sha256=digest(Path(__file__))))


def choose(rows: list[dict], policy: dict) -> dict:
    """Apply a common numerical upper bound and condition/generation safeguards."""
    summary = summarize(rows)
    reference = min(summary, key=lambda v: summary[v]["nll"])
    selection = policy["selection"]
    checks = {}
    for variant, values in summary.items():
        # The identical reference cancels exactly. Adding standard errors is
        # conservative when the initial common random numbers are correlated.
        se = 0. if variant == reference else values["numerical_se"] + summary[reference]["numerical_se"]
        upper = values["nll"] - summary[reference]["nll"] + 2 * se
        regret = max(values["condition_nll"][c] - summary[reference]["condition_nll"][c]
                     for c in values["condition_nll"])
        energy_ratio = values["energy"] / max(summary[reference]["energy"], np.finfo(float).tiny) - 1.
        eligible = (upper <= selection["simplicity_tolerance_nll_per_trial"]
                    and regret <= selection["maximum_condition_regret_nll_per_trial"]
                    and energy_ratio <= selection["maximum_relative_energy_deterioration"])
        checks[variant] = dict(nll_upper_difference=float(upper), maximum_condition_regret=float(regret),
                               relative_energy_deterioration=float(energy_ratio), eligible=bool(eligible))
    eligible = [v for v in summary if checks[v]["eligible"]]
    order = list(policy["variants"])
    complexity = {v: dict(parameters=policy["variants"][v]["parameters"], optional_controls=sum((
        policy["variants"][v]["execution"] == "free", policy["variants"][v]["precision"] == "free",
        policy["variants"][v]["search"] == "accumulate", policy["variants"][v]["retention"] == "outcome",
        policy["variants"][v]["retrieval"] == "similarity"))) for v in summary}
    chosen = min(eligible, key=lambda v: (complexity[v]["parameters"], complexity[v]["optional_controls"],
                                        summary[v]["nll"], order.index(v)))
    return dict(chosen=chosen, reference=reference, summary=summary, checks=checks,
                complexity=complexity,
                inference_scope="finite_search_working_structure_choice")


def shortlist(output: Path) -> None:
    register = validate_source(output)
    rows = json.loads((output / "development_evaluation_grouped.json").read_text())["rows"]
    decision = choose(rows, register["policy"])
    finalists = list(dict.fromkeys([decision["reference"], decision["chosen"]]))
    if len(finalists) == 1:
        other = "compact" if finalists[0] == "full" else "full"
        finalists.append(other)
    write_new(output / "structural_shortlist.json", dict(
        **decision, finalists=finalists, registered_at_utc=datetime.now(timezone.utc).isoformat(),
        coordinator_sha256=digest(Path(__file__)),
        evaluation_outcomes_scored=False,
        precision_scope="Two finalists; fixed parameters selected from development training prefixes"))


def refine(output: Path) -> None:
    register = validate_source(output)
    policy = register["policy"]
    shortlist = json.loads((output / "structural_shortlist.json").read_text())
    parameters = json.loads((output / "development_parameters.json").read_text())["groups"]
    original = json.loads((output / "development_evaluation_grouped.json").read_text())["rows"]
    groups = [g for g in parameters if g["variant"] in shortlist["finalists"]]
    settings = policy["selection"]["leading_model_precision_check"]
    scheduler = Scheduler(output, policy)
    seeds = {v: policy["seed"] + 18000 + i * 100 for i, v in enumerate(shortlist["finalists"])}
    try:
        jobs = [job for g in groups for job in fit_jobs(output, g["subject"],
            [g["candidates"][0]["candidate"]], settings["particles"], settings["repeats"], seeds[g["variant"]], scope="full")]
        records = scheduler.run(jobs, "finalist_independent_precision")
        rows = []
        for g in groups:
            s, c = g["subject"], g["candidates"][0]["candidate"]
            jobs = fit_jobs(output, s, [c], settings["particles"], settings["repeats"], seeds[g["variant"]], scope="full")
            score = aggregate([records[job["id"]] for job in jobs], conditional=True)
            prior = next(r for r in original if r["subject"] == s["subject"] and r["variant"] == g["variant"])
            ntest = prior["test_trials"]
            nll, se = score["nll"] / ntest, score["mc_se"] / ntest
            difference = abs(nll - prior["test_nll_per_trial"]) + 2 * np.hypot(se, prior["mc_se_per_trial"])
            stable = difference <= policy["search"]["numerical_tolerance_nll_per_trial"] and score["max_system_weight"] <= .8
            rows.append(dict(prior, test_nll_per_trial=nll, mc_se_per_trial=se,
                max_system_weight=score["max_system_weight"], particles=settings["particles"],
                repeats=settings["repeats"], precision_difference_per_trial=float(difference),
                numerical_status="within_working_tolerance" if stable else "numerically_limited"))
        write_new(output / "structural_precision.json", dict(rows=rows, cpu_seconds=scheduler.cpu_used()))
    finally:
        scheduler.close(terminate=True)


def freeze(output: Path) -> None:
    register = validate_source(output)
    if (output / "evaluation_evaluation.json").exists() or (output / "evaluation_parameters.json").exists():
        raise ValueError("Structure must be frozen before evaluation-subject fitting or scoring")
    rows = json.loads((output / "structural_precision.json").read_text())["rows"]
    shortlist = json.loads((output / "structural_shortlist.json").read_text())
    decision = choose(rows, register["policy"])
    comparator = "full" if decision["chosen"] != "full" else "compact"
    write_new(output / "structure_selection.json", dict(**decision,
        selected_variant=decision["chosen"], evaluation_variants=[decision["chosen"], comparator],
        coordinator_sha256=digest(Path(__file__)),
        finalists=shortlist["finalists"], frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        evaluation_outcomes_scored=False,
        limitations=[
            "Finite candidate search; global parameter optimum and full acceptable regions are unverified.",
            "Structure selected on development future choices; evaluation-subject future choices remain reserved.",
            "Numerical stability is a separate per-subject diagnostic; selection does not validate a psychological mechanism.",
            "The high-precision comparison concerns the two preselected finalists and fixed training-selected parameters.",
        ]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pool-development", "pool-evaluation", "shortlist", "refine", "freeze"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.phase.startswith("pool-"):
        pool_replicates(args.output.resolve(), args.phase.removeprefix("pool-"))
    else:
        globals()[args.phase](args.output.resolve())


if __name__ == "__main__":
    main()
