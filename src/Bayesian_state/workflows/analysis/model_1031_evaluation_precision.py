"""Check reserved-subject predictions at fixed parameters if the budget permits.

The optional check is registered before evaluation outcomes exist. Admission
uses completed runtimes, remaining CPU allowance and the original deadline.
It cannot change the frozen structure or training-selected parameters.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np

from .model_1031_finalization import Scheduler, digest, fit_jobs, validate_source, write_new
from .model_1031_decision import fixed_configuration_score


def admission(records: list[dict], jobs: list[dict], remaining_seconds: float,
              remaining_cpu: float) -> dict:
    """Bound every new run by twice the observed same-configuration cost."""
    wall, cpu = [], []
    for job in jobs:
        prior = [r for r in records if r["job"]["kind"] == "fit"
                 and r["job"]["scope"] == "full" and r["job"].get("max_trials") is None
                 and r["job"]["subject"] == job["subject"] and r["job"]["config"] == job["config"]]
        if not prior:
            return dict(admitted=False, reason="missing same-configuration full-run timing")
        wall.append(2 * max(r["seconds"] / r["job"]["particles"] for r in prior) * job["particles"])
        cpu.append(2 * max(r["cpu_seconds"] / r["job"]["particles"] for r in prior) * job["particles"])
    estimated_wall, estimated_cpu = max(wall, default=0.) + 300., sum(cpu) + 300.
    return dict(admitted=estimated_wall < remaining_seconds and estimated_cpu < remaining_cpu,
                estimated_wall_seconds=estimated_wall, estimated_cpu_seconds=estimated_cpu,
                remaining_wall_seconds=remaining_seconds, remaining_cpu_seconds=remaining_cpu,
                reason="runtime-only admission; twice prior cost plus five-minute reserve")


def register(output: Path) -> None:
    validate_source(output)
    if any((output / name).exists() for name in ("evaluation_parameters.json", "evaluation_evaluation.json")):
        raise ValueError("Register before evaluation-subject fitting is complete")
    write_new(output / "evaluation_precision_plan.json", dict(
        registered_at_utc=datetime.now(timezone.utc).isoformat(), particles=1024, repeats=4,
        seed_offset=40000, after="registered simulation and state recovery", outcome_inspected=False,
        purpose="Numerical verification only; preserve structure and all training-selected parameters",
        admission="Twice observed same-configuration full-run time and CPU, plus 300 seconds; original compute deadline",
        coordinator_sha256=digest(Path(__file__))))


def run(output: Path) -> None:
    registration = validate_source(output)
    policy = registration["policy"]
    plan = json.loads((output / "evaluation_precision_plan.json").read_text())
    if not (output / "recovery_states.json").exists():
        raise ValueError("Registered recovery takes precedence over this optional check")
    groups = json.loads((output / "evaluation_parameters.json").read_text())["groups"]
    original = json.loads((output / "evaluation_evaluation_grouped.json").read_text())["rows"]
    variants = json.loads((output / "structure_selection.json").read_text())["evaluation_variants"]
    seeds = {v: policy["seed"] + plan["seed_offset"] + 100 * i for i, v in enumerate(variants)}
    jobs = [job for group in groups for job in fit_jobs(output, group["subject"],
        [group["candidates"][0]["candidate"]], plan["particles"], plan["repeats"],
        seeds[group["variant"]], scope="full")]
    records = [json.loads(p.read_text()) for p in (output / "jobs").glob("*.json")]
    recovery_cpu = sum(json.loads(p.read_text())["cpu_seconds"] for p in (output / "recovery").glob("*/jobs/*.json"))
    # Include all elapsed time since registration at the maximum worker count.
    # This also covers external interruptions and process startup conservatively.
    now = time.time()
    cpu_bound = (now - registration["registered_unix"]) * policy["budget"]["workers"] + 3600.
    recorded_cpu = sum(r["cpu_seconds"] for r in records) + recovery_cpu
    scheduler = Scheduler(output, policy)
    check = admission(records, jobs, scheduler.deadline - now,
                      policy["budget"]["cpu_hours"] * 3600 - max(cpu_bound, recorded_cpu))
    workers = min(policy["budget"]["workers"], len(os.sched_getaffinity(0)))
    if len(jobs) > workers:
        check = dict(check, admitted=False, reason="timing bound requires all optional jobs to fit in one worker wave")
    if not check["admitted"]:
        write_new(output / "evaluation_precision.json", dict(status="omitted_by_runtime_budget", admission=check, rows=[]))
        return
    try:
        completed = scheduler.run(jobs, "evaluation_fixed_parameter_precision")
        rows = []
        for group in groups:
            s, c = group["subject"], group["candidates"][0]["candidate"]
            score = fixed_configuration_score(list(completed.values()), s["subject"], c["id"])
            prior = next(r for r in original if r["subject"] == s["subject"] and r["variant"] == group["variant"])
            n = prior["test_trials"]
            nll, se = score["nll"] / n, score["mc_se"] / n
            difference = abs(nll - prior["test_nll_per_trial"]) + 2 * np.hypot(se, prior["mc_se_per_trial"])
            stable = difference <= policy["search"]["numerical_tolerance_nll_per_trial"] and score["max_system_weight"] <= .8
            rows.append(dict(prior, test_nll_per_trial=nll, mc_se_per_trial=se,
                max_system_weight=score["max_system_weight"], particles=plan["particles"], repeats=score["repeats"],
                requested_repeats_per_variant=plan["repeats"], pooled_job_ids=score["pooled_job_ids"],
                original_replicate_pooling=prior.get("replicate_pooling"),
                replicate_pooling=dict(independent_seeds=score["repeats"], raw_systems=score["repeats"],
                                       particle_counts=[plan["particles"]]),
                training_numerical_status=prior["numerical_status"], precision_difference_per_trial=float(difference),
                numerical_status="within_working_tolerance" if stable else "numerically_limited"))
        write_new(output / "evaluation_precision.json", dict(status="completed", admission=check, rows=rows,
            coordinator_sha256=digest(Path(__file__)), structure_and_parameters_changed=False))
    finally:
        scheduler.close(terminate=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("register", "run"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    globals()[args.phase](args.output.resolve())


if __name__ == "__main__":
    main()
