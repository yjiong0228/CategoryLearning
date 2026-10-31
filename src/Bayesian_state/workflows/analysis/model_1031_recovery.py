"""Finite model/parameter and pre-feedback state recovery for Model 1031.

Generating points are scored separately from the optimizer candidate bank.
State recovery conditions on known parameters and the observed choice history;
it therefore measures a different question from individual parameter recovery.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.stats import qmc

from ...inference.backends.particle_filter import run_state_model_particle_filter
from ...model import ModelContext, StateModel
from .model_1031_finalization import (
    Scheduler, candidate, digest, fit_jobs, identity, rank,
    save_arrays, task_feedback, validate_source, write_new,
)


def group_codes(kinds: np.ndarray, masks: np.ndarray) -> np.ndarray:
    return np.where(kinds == 2, 3, np.where(kinds == 1, np.where(masks.sum(axis=-1) == 4, 1, 2), kinds))


def weighted_intervals(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Marginal 5/50/95% weighted quantiles, with finite-particle support."""
    out = []
    for column in values.T:
        order = np.argsort(column)
        cumulative = np.cumsum(weights[order])
        indices = np.minimum(np.searchsorted(cumulative, [.05, .5, .95]), len(order) - 1)
        out.append(column[order[indices]])
    return np.asarray(out)


def _custom_work(job: dict) -> dict:
    output = Path(job["output"])
    path = output / "jobs" / f"{job['id']}.json"
    if path.exists():
        result = json.loads(path.read_text())
        if result["job"] != job or digest(path.with_suffix(".npz")) != result["array_sha256"]:
            raise ValueError("Recovery record changed")
        return result
    started, cpu = time.time(), time.process_time()
    cfg = json.loads((output / "configs" / f"{job['config']}.json").read_text())
    with np.load(output / "inputs" / f"s{job['subject']}.npz") as source:
        data = {key: source[key] for key in source.files}
    if job["kind"] == "recovery_generate":
        cfg["modules"]["hypo_transitions_mod"]["kwargs"].update(
            marginalize_execution=False, module_seed=job["seed"] + 1)
        cfg["modules"]["perception_mod"].setdefault("kwargs", {})["module_seed"] = job["seed"]
        model = StateModel(cfg, context=ModelContext(condition=job["condition"], subject_id=job["subject"]))
        trajectory = model.generate_step_by_step(data["stimulus"],
            lambda t, a: float(task_feedback(a, data["categories"][t], job["condition"])), choice_seed=job["seed"] + 2)
        truth_belief, truth_execution, truth_rule, truth_features = [], [], [], []
        for prior, step in zip(trajectory.prior, trajectory.step_log):
            nodes = step["representations_pre"]
            groups = np.asarray([-1 if n is None else 3 if n["kind"] == "rule" else
                0 if n["kind"] == "uncommitted" else 1 if sum(n["mask"]) == 4 else 2 for n in nodes])
            truth_belief.append([(prior * (groups == g)).sum() for g in range(4)])
            index = int(np.argmax(step["choice_weights"]))
            truth_execution.append(groups[index])
            truth_rule.append(nodes[index].get("rule_id", -1))
            masks = np.asarray([n["mask"] if n is not None and n["kind"] == "template" else [0] * 4 for n in nodes])
            truth_features.append(prior @ masks)
        saved = dict(stimulus=data["stimulus"], categories=data["categories"], choices=trajectory.choices,
            feedback=trajectory.feedback, truth_group_belief=truth_belief, truth_execution_group=truth_execution,
            truth_executed_rule=truth_rule, truth_template_feature_mass=truth_features,
            truth_choice_probabilities=trajectory.observed_probabilities)
        if job["condition"] == 3:
            saved["truth_pairing"] = np.asarray([s["pairing_probability_pre"] for s in trajectory.step_log])
        metrics = dict(trials=len(trajectory.choices))
    elif job["kind"] == "recovery_state":
        with np.load(output / "jobs" / f"{job['generation_id']}.npz") as source:
            truth = {key: source[key] for key in source.files}
        intervals, feature_means, belief_means, actual_rule, pairing_means = [], [], [], [], []
        def observer(t, phase, values):
            if phase != "filtered":
                return
            w = values["weights"]
            group = group_codes(values["representation_kind"], values["template_mask"])
            belief = np.stack([(values["belief"] * (group == g)).sum(axis=1) for g in range(4)], axis=1)
            intervals.append(weighted_intervals(belief, w))
            belief_means.append(w @ belief)
            feature = np.einsum("ph,phd->pd", values["belief"], values["template_mask"])
            feature_means.append(w @ feature)
            if job["condition"] == 3:
                pairing_means.append(w @ values["pairing"])
            rule = truth["truth_executed_rule"][t]
            actual_rule.append(float(w @ ((values["catalogue_rule"] == rule) * values["choice_weights"]).sum(axis=1)) if rule >= 0 else 0.)
        result = run_state_model_particle_filter(engine_config=cfg, subject_id=job["subject"],
            condition=job["condition"], stimulus=truth["stimulus"], choices=truth["choices"], feedback=truth["feedback"],
            particle_count=job["particles"], filter_seed=job["seed"], choice_readout_power=1., state_observer=observer)
        execution = result.state_probabilities["filtered_representation_execution"]
        true_execution = truth["truth_execution_group"].astype(int)
        probability = execution[np.arange(len(execution)), true_execution]
        intervals = np.asarray(intervals)
        covered = ((truth["truth_group_belief"] >= intervals[:, :, 0])
                   & (truth["truth_group_belief"] <= intervals[:, :, 2]))
        rule_mask = truth["truth_executed_rule"] >= 0
        metrics = dict(trials=len(execution), execution_brier=float(np.mean(np.sum(
            (execution - np.eye(4)[true_execution]) ** 2, axis=1))),
            mean_true_execution_probability=float(probability.mean()),
            group_belief_mae=float(np.mean(np.abs(np.asarray(belief_means) - truth["truth_group_belief"]))),
            marginal90_coverage=float(covered.mean()),
            marginal90_width=float(np.mean(intervals[:, :, 2] - intervals[:, :, 0])),
            template_feature_mass_mae=float(np.mean(np.abs(np.asarray(feature_means) - truth["truth_template_feature_mass"]))),
            true_rule_trials=int(rule_mask.sum()),
            true_rule_execution_probability=float(np.asarray(actual_rule)[rule_mask].mean()) if rule_mask.any() else None)
        saved = dict(execution=execution, group_belief=belief_means, group_intervals=intervals,
                     template_feature_mass=feature_means, true_rule_probability=actual_rule,
                     choice_probabilities=result.marginal_probabilities)
        if job["condition"] == 3:
            saved["pairing"] = np.asarray(pairing_means)
            metrics["pairing_mae"] = float(np.mean(np.abs(saved["pairing"] - truth["truth_pairing"])))
    else:
        raise ValueError("Unknown recovery task")
    save_arrays(path.with_suffix(".npz"), **saved)
    result = dict(job=job, **metrics, cpu_seconds=time.process_time() - cpu,
                  seconds=time.time() - started, array_sha256=digest(path.with_suffix(".npz")))
    write_new(path, result)
    return result


def include_case_cpu(scheduler: Scheduler) -> None:
    recorded = sum(json.loads(p.read_text()).get("cpu_seconds", 0.)
                   for p in scheduler.output.glob("jobs/*.json"))
    recorded += sum(json.loads(p.read_text()).get("cpu_seconds", 0.)
                    for p in (scheduler.output / "recovery").glob("*/jobs/*.json"))
    scheduler.previous_cpu = max(scheduler.previous_cpu, recorded)


def run_custom(output: Path, policy: dict, jobs: list[dict]) -> dict:
    """Use the study's absolute deadline and CPU accounting for custom observers."""
    scheduler = Scheduler(output, policy)
    include_case_cpu(scheduler)
    if time.time() >= scheduler.deadline or scheduler.cpu_used() >= policy["budget"]["cpu_hours"] * 3600:
        raise TimeoutError("Recovery cannot start after the compute budget")
    count = min(policy["budget"]["workers"], len(os.sched_getaffinity(0)), len(jobs))
    scheduler.pool = ProcessPoolExecutor(max_workers=count, mp_context=mp.get_context("spawn"))
    futures = {scheduler.pool.submit(_custom_work, job): job for job in jobs}
    records, last_report = {}, 0.
    try:
        while futures:
            if time.time() >= scheduler.deadline or scheduler.cpu_used() >= policy["budget"]["cpu_hours"] * 3600:
                raise TimeoutError("Recovery compute budget reached")
            completed, _ = wait(futures, timeout=1., return_when=FIRST_COMPLETED)
            for future in completed:
                job = futures.pop(future)
                records[job["id"]] = future.result()
            if time.time() - last_report >= 30:
                print(json.dumps(dict(stage=jobs[0]["kind"], completed=len(records), remaining=len(futures))), flush=True)
                last_report = time.time()
    except BaseException:
        write_new(output / "interruptions" / f"recovery_{time.time_ns()}.json", dict(
            pending=list(futures.values()), cpu_seconds=scheduler.cpu_used(), reason=traceback.format_exc()))
        raise
    finally:
        scheduler.close(terminate=True)
    return records


def recovery_bank(policy: dict) -> list[dict]:
    count = policy["recovery"]["candidates"]
    keys, values = list(policy["parameter_values"]), policy["parameter_values"]
    points = qmc.Sobol(len(keys), scramble=True, seed=policy["seed"] + 24000).random_base2(int(np.ceil(np.log2(count))))
    return [{k: values[k][min(int(v * len(values[k])), len(values[k]) - 1)] for k, v in zip(keys, point)} for point in points[:count]]


def prepare_cases(output: Path) -> None:
    register = validate_source(output)
    policy = register["policy"]
    selection = json.loads((output / "structure_selection.json").read_text())
    parameters = json.loads((output / "development_parameters.json").read_text())["groups"]
    jobs, cases = [], []
    for condition in (1, 2, 3):
        subject = next(s for s in register["subjects"] if s["condition"] == condition and s["role"] == "development")
        for variant in selection["evaluation_variants"]:
            group = next(g for g in parameters if g["subject"]["subject"] == subject["subject"] and g["variant"] == variant)
            source = group["candidates"][0]["candidate"]
            for repeat in range(policy["recovery"]["paths_per_source_background"]):
                seed = policy["seed"] + 25000 + len(cases) * 100
                job = dict(output=str(output), kind="recovery_generate", subject=subject["subject"],
                           condition=condition, config=source["id"], seed=seed)
                job["id"] = identity(job)
                jobs.append(job)
                cases.append(dict(id=job["id"], subject=dict(subject, train_trials=subject["trials"]),
                                  source_variant=variant, source=source, seed=seed))
    write_new(output / "recovery_plan.json", dict(cases=cases, bank=recovery_bank(policy),
        variants=selection["evaluation_variants"], coordinator_sha256=digest(Path(__file__)),
        state_conditioning="known parameters; choice included, current feedback excluded; filtering",
        interval_scope="marginal interval diagnostics across correlated trials, not independent binomial coverage"))
    run_custom(output, policy, jobs)
    for case in cases:
        case_output = output / "recovery" / case["id"]
        with np.load(output / "jobs" / f"{case['id']}.npz") as truth:
            save_arrays(case_output / "inputs" / f"s{case['subject']['subject']}.npz",
                        **{k: truth[k] for k in ("stimulus", "choices", "feedback", "categories")})
        cfg = json.loads((output / "configs" / f"{case['source']['id']}.json").read_text())
        write_new(case_output / "configs" / f"{case['source']['id']}.json", cfg)
    write_new(output / "recovery_prepared.json", dict(cases=len(cases), completed_unix=time.time()))


def fit_cases(output: Path) -> None:
    register = validate_source(output)
    policy, settings = register["policy"], register["policy"]["recovery"]
    plan = json.loads((output / "recovery_plan.json").read_text())
    scheduler = Scheduler(output, policy)
    # Nested case jobs share the same worker implementation and are included
    # in the whole-study accounting when this phase is resumed.
    include_case_cpu(scheduler)
    groups, jobs = {}, []
    try:
        for case in plan["cases"]:
            case_output = output / "recovery" / case["id"]
            for variant in plan["variants"]:
                candidates = [candidate(case_output, case["subject"]["condition"], p, variant, policy) for p in plan["bank"]]
                candidates = list({c["id"]: c for c in candidates}.values())
                groups[case["id"], variant] = candidates
                jobs += fit_jobs(case_output, case["subject"], candidates, settings["screening_particles"],
                                 settings["screening_repeats"], policy["seed"] + 27000)
        scheduler.run(jobs, "recovery_candidate_search")
        jobs = []
        for case in plan["cases"]:
            case_output = output / "recovery" / case["id"]
            for variant in plan["variants"]:
                key = case["id"], variant
                rows = rank(case_output, case["subject"], groups[key], settings["screening_particles"],
                            settings["screening_repeats"], policy["seed"] + 27000)
                groups[key] = [row["candidate"] for row in rows[:2]]
            # Containing models receive exact boundary leaders from their
            # simpler competitors, as in the behavioral model comparison.
            leaders = {v: list(groups[case["id"], v]) for v in plan["variants"]}
            for variant in plan["variants"]:
                for source_variant in plan["variants"]:
                    for source in leaders[source_variant]:
                        transferred = candidate(case_output, case["subject"]["condition"],
                                                source["parameters"], variant, policy)
                        if transferred["id"] == source["id"]:
                            groups[case["id"], variant].append(transferred)
                key = case["id"], variant
                groups[key] = list({c["id"]: c for c in groups[key]}.values())
                jobs += fit_jobs(case_output, case["subject"], groups[key], settings["confirmation_particles"],
                                 settings["confirmation_repeats"], policy["seed"] + 28000)
            jobs += fit_jobs(case_output, case["subject"], [case["source"]], settings["confirmation_particles"],
                             settings["confirmation_repeats"], policy["seed"] + 29000)
        scheduler.run(jobs, "recovery_confirmation_and_generating_point")
        results = []
        for case in plan["cases"]:
            case_output = output / "recovery" / case["id"]
            fitted = {v: rank(case_output, case["subject"], groups[case["id"], v], settings["confirmation_particles"],
                              settings["confirmation_repeats"], policy["seed"] + 28000) for v in plan["variants"]}
            oracle = rank(case_output, case["subject"], [case["source"]], settings["confirmation_particles"],
                          settings["confirmation_repeats"], policy["seed"] + 29000)[0]
            winner = min(fitted, key=lambda v: fitted[v][0]["nll"])
            equivalent = [v for v in plan["variants"] if candidate(case_output, case["subject"]["condition"],
                case["source"]["parameters"], v, policy)["id"] == case["source"]["id"]]
            effective_source = min(equivalent, key=lambda v: policy["variants"][v]["parameters"])
            actual_fit = fitted[effective_source][0]
            n = case["subject"]["trials"]
            gap = (actual_fit["nll"] - oracle["nll"]) / n
            gap_se = (actual_fit["mc_se"] + oracle["mc_se"]) / n
            other = next(v for v in plan["variants"] if v != winner)
            margin = (fitted[other][0]["nll"] - fitted[winner][0]["nll"]) / n
            numerical = 2 * (fitted[other][0]["mc_se"] + fitted[winner][0]["mc_se"]) / n
            concentrated = max([oracle["max_system_weight"], *[fitted[v][0]["max_system_weight"] for v in fitted]]) > .8
            status = ("numerically_limited" if concentrated else "search_limited" if gap - 2 * gap_se > .01 else
                      "unresolved" if margin <= max(.01, numerical) else
                      "correct" if winner == effective_source else "incorrect")
            compatible = [v for v in plan["variants"] if (fitted[v][0]["nll"] - fitted[winner][0]["nll"]
                + (0. if v == winner else 2 * (fitted[v][0]["mc_se"] + fitted[winner][0]["mc_se"]))) / n <= .01]
            working_choice = min(compatible, key=lambda v: policy["variants"][v]["parameters"])
            results.append(dict(case=case, fitted=fitted, generating_point=oracle, lower_nll_model=winner,
                effective_source=effective_source, equivalent_source_configurations=equivalent,
                selected_by_parsimony_rule=working_choice,
                classification=status, generating_point_gap_per_trial=float(gap), margin_per_trial=float(margin),
                interpretation="finite-bank diagnostic; candidate counts differ after projection"))
        write_new(output / "recovery_fits.json", dict(rows=results, cpu_seconds=scheduler.cpu_used()))
    finally:
        scheduler.close(terminate=True)


def recover_states(output: Path) -> None:
    register = validate_source(output)
    policy = register["policy"]
    plan = json.loads((output / "recovery_plan.json").read_text())
    jobs = []
    for case in plan["cases"]:
        for repeat in range(2):
            job = dict(output=str(output), kind="recovery_state", subject=case["subject"]["subject"],
                condition=case["subject"]["condition"], config=case["source"]["id"], generation_id=case["id"],
                particles=policy["recovery"]["state_particles"], seed=policy["seed"] + 30000 + repeat)
            job["id"] = identity(job)
            jobs.append(job)
    records = run_custom(output, policy, jobs)
    rows = []
    for case in plan["cases"]:
        pair = [r for r in records.values() if r["job"]["generation_id"] == case["id"]]
        arrays = [np.load(output / "jobs" / f"{r['job']['id']}.npz") for r in pair]
        difference = float(np.mean(np.abs(arrays[0]["execution"] - arrays[1]["execution"])))
        for array in arrays:
            array.close()
        rows.append(dict(case=case, replicates=pair, execution_replication_mae=difference))
    write_new(output / "recovery_states.json", dict(rows=rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "fit", "state"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"prepare": prepare_cases, "fit": fit_cases, "state": recover_states}[args.phase](args.output.resolve())


if __name__ == "__main__":
    main()
