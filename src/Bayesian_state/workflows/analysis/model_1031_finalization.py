"""Bounded three-condition mechanism comparison using the shared StateModel.

All numerical jobs write immutable results. The absolute deadline includes
implementation and manuscript time; the scheduler reserves the final 2.5 hours.
Oral observations are never loaded into scoring or parameter selection.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
import scipy
from scipy.special import logsumexp
from scipy.spatial.distance import cdist, pdist
from scipy.stats import qmc
import yaml

from ...inference.backends.particle_filter import run_state_model_particle_filter
from ...model import ModelContext, StateModel
from ...simulation.data import SubjectTrialDataLoader


ROOT = Path(__file__).resolve().parents[4]
DEFAULT = dict(similarity_width=.25, capacity=4, baseline_probability=.15, error_gain=1.,
               failure_decay=.8, global_search=.2, retention=.8, execution_inertia=1.,
               beta_init=5., learning_rate=.1)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def save_arrays(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def project(parameters: dict, variant: dict) -> dict:
    p = deepcopy(parameters)
    if variant["execution"] == "zero":
        p["execution_inertia"] = 0.
    if variant["precision"] == "fixed":
        p["learning_rate"] = 0.
    if variant["search"] == "constant":
        p["error_gain"] = 0.
    if p["error_gain"] == 0.:
        p["failure_decay"] = 0.
    if variant["retention"] == "one":
        p["retention"] = 1.
    p["capacity"] = int(p["capacity"])
    return p


def engine_config(condition: int, parameters: dict, variant: dict, *, sample=False) -> dict:
    p = project(parameters, variant)
    cfg = yaml.safe_load((ROOT / "configs/exp123/model_struct" /
                         f"model_1031_cond{condition}_unified_v2.yaml").read_text())
    cfg["partition"]["kwargs"].update(capacity=p["capacity"], similarity_width=p["similarity_width"])
    h = cfg["modules"]["hypo_transitions_mod"]["kwargs"]
    for key in ("baseline_probability", "error_gain", "failure_decay", "global_search", "execution_inertia"):
        h[key] = p[key]
    h.update(marginalize_execution=p["execution_inertia"] == 0. and not sample,
             retrieval_mode=variant["retrieval"])
    cfg["modules"]["beta_mod"]["kwargs"].update(beta_init=p["beta_init"], learning_rate=p["learning_rate"])
    memory = cfg["modules"]["memory_mod"]["kwargs"]
    memory.update(gamma=p["retention"], retention_mode=(
        "constant" if variant["retention"] in ("constant", "one") or p["retention"] == 1. else "outcome"))
    return cfg


def candidate(output: Path, condition: int, p: dict, variant_name: str, policy: dict) -> dict:
    variant = policy["variants"][variant_name]
    parameters = project(p, variant)
    cfg = engine_config(condition, parameters, variant)
    key = identity(cfg)
    path = output / "configs" / f"{key}.json"
    if not path.exists():
        write_new(path, cfg)
    elif json.loads(path.read_text()) != cfg:
        raise ValueError("Configuration identity collision")
    return dict(id=key, parameters=parameters, variant=variant_name, condition=condition)


def bank(policy: dict) -> list[dict]:
    values = policy["parameter_values"]
    keys = list(values)
    count = policy["search"]["initial_candidates"]
    points = qmc.Sobol(len(keys), scramble=True, seed=policy["seed"]).random_base2(
        int(np.ceil(np.log2(count * 2))))
    from .model_1031_subject_pilot import bank as pilot_bank

    # These anchors were generated before the earlier pilot outcomes existed.
    previous = yaml.safe_load((ROOT / policy["search"]["previous_prespecified_bank"]).read_text())
    result = [point["parameters"] for point in pilot_bank(previous)]
    for point in points:
        p = {k: values[k][min(int(v * len(values[k])), len(values[k]) - 1)] for k, v in zip(keys, point)}
        if p not in result:
            result.append(p)
        if len(result) == count:
            return result
    raise ValueError("Insufficient initial candidates")


def prepare(config_path: Path, output: Path) -> None:
    policy = yaml.safe_load(config_path.read_text())
    if (output / "registration.json").exists():
        raise FileExistsError("Study is already registered")
    loaders = {}
    for condition in (1, 2, 3):
        loaders[condition] = SubjectTrialDataLoader(engine_config(condition, DEFAULT, policy["variants"]["full"]))
        loaders[condition].prepare_data(ROOT / policy["data"]["source"])
    subjects = []
    for role in ("development", "evaluation"):
        for condition, ids in policy["data"][role].items():
            loader = loaders[condition]
            for sid in ids:
                frame = loader._get_subject_frame(sid, 1.)
                if loader._get_condition_value(frame) != condition:
                    raise ValueError("Subject condition disagrees with registration")
                arrays = loader._extract_arrays(frame, max_trials=None)
                expected = task_feedback(arrays.choices, arrays.categories, condition)
                np.testing.assert_array_equal(expected, arrays.feedback)
                ntrain = int(np.floor(len(arrays.choices) * policy["data"]["train_fraction"]))
                save_arrays(output / "inputs" / f"s{sid}.npz", stimulus=arrays.stimulus,
                    choices=arrays.choices, feedback=arrays.feedback, categories=arrays.categories,
                    presskeys=arrays.presskeys)
                subjects.append(dict(subject=sid, condition=condition, role=role, trials=len(frame),
                    train_trials=ntrain, input_sha256=digest(output / "inputs" / f"s{sid}.npz"),
                    choice_to_presskey=arrays.choice_to_presskey))
    paths = list((ROOT / "src/Bayesian_state").rglob("*.py"))
    paths += list((ROOT / "configs/exp123/model_struct").glob("model_1031*unified_v2.yaml"))
    paths.append(config_path.resolve())
    paths.append(ROOT / policy["search"]["previous_prespecified_bank"])
    perception_paths = [ROOT / "data/exp123/processed" / name for name in
                        ("Task1b_errorsummary_24.csv", "Task1b_errorsummary_72.csv")]
    register = dict(policy=policy, subjects=subjects, bank=bank(policy),
        source_git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in paths},
        data_sha256=digest(ROOT / policy["data"]["source"]),
        perception_sha256={str(p.relative_to(ROOT)): digest(p) for p in perception_paths},
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, yaml=yaml.__version__),
        registered_unix=time.time())
    write_new(output / "registration.json", register)


def task_feedback(choices, categories, condition):
    choices, categories = np.asarray(choices), np.asarray(categories)
    if condition != 3:
        return (choices == categories).astype(float)
    return np.where(choices == categories, 1., np.where((choices - 1) // 2 == (categories - 1) // 2, .5, 0.))


def path_features(choices: np.ndarray, categories: np.ndarray, condition: int, block=32) -> np.ndarray:
    """Fixed blocks and whole-path improvement/maintenance; all paths retained."""
    choices = np.atleast_2d(choices)
    correct = choices == categories[None, :]
    blocks = [correct[:, t:min(t + block, len(categories))].mean(axis=1)
              for t in range(0, len(categories), block)]
    levels = np.stack(blocks, axis=1)
    ncat = 2 if condition == 1 else 4
    counts = []
    for c in range(1, ncat + 1):
        mask = categories == c
        for a in range(1, ncat + 1):
            counts.append((choices[:, mask] == a).mean(axis=1) if mask.any() else np.zeros(len(choices)))
    if levels.shape[1] > 1:
        changes = np.diff(levels, axis=1)
        index = np.argmax(changes, axis=1)
        improvement = changes[np.arange(len(choices)), index]
        timing = (index + 1) / levels.shape[1]
        maintenance = np.asarray([row[min(i + 1, len(row) - 1):].mean() for row, i in zip(levels, index)])
    else:
        improvement, timing, maintenance = np.zeros(len(choices)), np.zeros(len(choices)), levels[:, 0]
    extras = np.column_stack([improvement, timing, maintenance, levels.std(axis=1), *counts])
    if condition == 3:
        family = ((choices - 1) // 2 == (categories[None, :] - 1) // 2).mean(axis=1)
        score = task_feedback(choices, categories[None, :], condition).mean(axis=1)
        extras = np.column_stack([extras, family, score])
    return np.column_stack([levels, extras])


def energy_score(paths: np.ndarray, observed: np.ndarray) -> float:
    scale = np.sqrt(paths.shape[1])
    cross = cdist(paths, observed[None, :]).mean() / scale
    within = pdist(paths).sum() * 2 / (len(paths) ** 2 * scale)
    return float(cross - .5 * within)


def _work(job: dict) -> dict:
    output = Path(job["output"])
    path = output / "jobs" / f"{job['id']}.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record["job"] != job or digest(path.with_suffix(".npz")) != record["array_sha256"]:
            raise ValueError("Completed job identity or arrays changed")
        return record
    started, cpu = time.time(), time.process_time()
    cfg = json.loads((output / "configs" / f"{job['config']}.json").read_text())
    with np.load(output / "inputs" / f"s{job['subject']}.npz") as source:
        arrays = {key: source[key] for key in source.files}
    ntrain = job["train_trials"]
    saved = {}
    if job["kind"] == "fit":
        n = ntrain if job["scope"] == "train" else len(arrays["choices"])
        if job.get("max_trials") is not None:
            n = min(n, job["max_trials"])
        result = run_state_model_particle_filter(engine_config=cfg, subject_id=job["subject"],
            condition=job["condition"], stimulus=arrays["stimulus"][:n], choices=arrays["choices"][:n],
            feedback=arrays["feedback"][:n], particle_count=job["particles"],
            choice_readout_power=1., filter_seed=job["seed"])
        p = result.marginal_probabilities
        if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1), 1.):
            raise ValueError("Invalid filtered probabilities")
        log_chosen = np.log(np.maximum(p[np.arange(n), arrays["choices"][:n] - 1], np.finfo(float).tiny))
        metrics = dict(nll=float(-log_chosen.sum()), train_nll=float(-log_chosen[:ntrain].sum()),
                       trials=n, mean_ess=float(np.mean(result.post_choice_ess)))
        saved.update(choice_probabilities=p, ess=result.post_choice_ess)
        if job["scope"] == "full":
            saved.update({key: value for key, value in result.state_probabilities.items()
                          if "representation" in key or "pairing" in key})
    elif job["kind"] == "generate":
        cfg["modules"]["hypo_transitions_mod"]["kwargs"]["marginalize_execution"] = False
        choices, feedback, groups, rules, pairings = [], [], [], [], []
        for seed in job["seeds"]:
            cfg["modules"]["hypo_transitions_mod"]["kwargs"]["module_seed"] = seed + 1
            cfg["modules"]["perception_mod"].setdefault("kwargs", {})["module_seed"] = seed
            model = StateModel(cfg, context=ModelContext(condition=job["condition"], subject_id=job["subject"]))
            trajectory = model.generate_step_by_step(arrays["stimulus"],
                lambda t, a: float(task_feedback(a, arrays["categories"][t], job["condition"])), choice_seed=seed + 2)
            choices.append(trajectory.choices)
            feedback.append(trajectory.feedback)
            group, rule = [], []
            for step in trajectory.step_log:
                node = step["representations_pre"][int(np.argmax(step["choice_weights"]))]
                group.append(3 if node["kind"] == "rule" else
                             (0 if node["kind"] == "uncommitted" else 1 if sum(node["mask"]) == 4 else 2))
                rule.append(node.get("rule_id", -1))
            groups.append(group)
            rules.append(rule)
            if job["condition"] == 3:
                pairings.append([step["pairing_probability_pre"] for step in trajectory.step_log])
        saved.update(choices=np.asarray(choices), feedback=np.asarray(feedback),
                     representation_group=np.asarray(groups), executed_rule=np.asarray(rules), seeds=job["seeds"])
        if pairings:
            saved["pairing_probability"] = np.asarray(pairings)
        metrics = dict(paths=len(choices), trials=len(arrays["choices"]))
    else:
        raise ValueError("Unknown job kind")
    save_arrays(path.with_suffix(".npz"), **saved)
    record = dict(job=job, **metrics, seconds=time.time() - started, cpu_seconds=time.process_time() - cpu,
                  array_sha256=digest(path.with_suffix(".npz")))
    write_new(path, record)
    return record


class Scheduler:
    def __init__(self, output: Path, policy: dict):
        self.output, self.policy = output, policy
        self.deadline = datetime.fromisoformat(policy["started_at_utc"]).timestamp() + policy["budget"]["computation_stop_hours"] * 3600
        self.previous_cpu = sum(json.loads(p.read_text()).get("cpu_seconds", 0.)
                                for p in (output / "jobs").glob("*.json"))
        interrupted_cpu = [json.loads(p.read_text()).get("cpu_seconds", 0.)
                           for p in (output / "interruptions").glob("*.json")]
        self.previous_cpu = max([self.previous_cpu, *interrupted_cpu])
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.base_child_cpu = usage.ru_utime + usage.ru_stime
        self.pool = None

    def cpu_used(self) -> float:
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        total = self.previous_cpu + usage.ru_utime + usage.ru_stime - self.base_child_cpu
        if self.pool is not None:
            for process in self.pool._processes.values():
                try:
                    fields = Path(f"/proc/{process.pid}/stat").read_text().split(") ", 1)[1].split()
                    total += (float(fields[11]) + float(fields[12])) / os.sysconf("SC_CLK_TCK")
                except (OSError, IndexError):
                    pass
        return total

    def close(self, terminate=False):
        if self.pool is not None:
            if terminate:
                for process in self.pool._processes.values():
                    process.terminate()
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

    def run(self, jobs: list[dict], label: str) -> dict[str, dict]:
        unique = {job["id"]: job for job in jobs}
        records, pending = {}, []
        for key, job in unique.items():
            path = self.output / "jobs" / f"{key}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if record["job"] != job or digest(path.with_suffix(".npz")) != record["array_sha256"]:
                    raise ValueError("Previously completed job changed")
                records[key] = record
            else:
                pending.append(job)
        print(json.dumps(dict(stage=label, pending=len(pending), reused=len(records))), flush=True)
        if not pending:
            return records
        if time.time() >= self.deadline or self.cpu_used() >= self.policy["budget"]["cpu_hours"] * 3600:
            raise TimeoutError("Registered compute budget is exhausted before job admission")
        workers = min(self.policy["budget"]["workers"], len(os.sched_getaffinity(0)), len(pending))
        if self.pool is None:
            self.pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        futures = {self.pool.submit(_work, job): job for job in pending}
        last_report, consecutive_failures = 0., 0
        try:
            while futures:
                if time.time() >= self.deadline or self.cpu_used() >= self.policy["budget"]["cpu_hours"] * 3600:
                    raise TimeoutError("Registered compute deadline/CPU limit reached; preserve manuscript reserve")
                done, _ = wait(futures, timeout=1., return_when=FIRST_COMPLETED)
                for future in done:
                    job = futures.pop(future)
                    try:
                        records[job["id"]] = future.result()
                        consecutive_failures = 0
                    except Exception:
                        consecutive_failures += 1
                        failure = dict(job=job, traceback=traceback.format_exc(), time=time.time())
                        write_new(self.output / "failures" / f"{job['id']}_{time.time_ns()}.json", failure)
                        print(json.dumps(dict(failed=job["id"], error=failure["traceback"])), flush=True)
                        if consecutive_failures >= self.policy["budget"]["failure_stop"]:
                            raise RuntimeError("Consecutive job failure limit reached")
                if time.time() - last_report >= 30:
                    print(json.dumps(dict(stage=label, completed=len(records), remaining=len(futures),
                                          cpu_hours=round(self.cpu_used() / 3600, 3))), flush=True)
                    last_report = time.time()
        except BaseException:
            write_new(self.output / "interruptions" / f"{time.time_ns()}.json",
                      dict(stage=label, pending=[job for job in futures.values()], cpu_seconds=self.cpu_used(),
                           reason=traceback.format_exc(), time=time.time()))
            self.close(terminate=True)
            raise
        if len(records) != len(unique):
            raise RuntimeError(f"{label} contains failed jobs; completed results retained")
        return records


def fit_jobs(output: Path, subject: dict, candidates: list[dict], particles: int,
             repeats: int, seed_base: int, scope="train", max_trials=None) -> list[dict]:
    jobs = []
    for c in candidates:
        for repeat in range(repeats):
            seed = seed_base + repeat
            job = dict(output=str(output.resolve()), kind="fit", subject=subject["subject"],
                condition=subject["condition"], train_trials=subject["train_trials"], config=c["id"],
                particles=particles, seed=seed, scope=scope, max_trials=max_trials)
            job["id"] = identity(job)
            jobs.append(job)
    return jobs


def aggregate(records: list[dict], *, conditional=False) -> dict:
    log_full = -np.asarray([r["nll"] for r in records])
    log_train = -np.asarray([r["train_nll"] for r in records])
    count = len(records)
    nll = float(-logsumexp(log_full) + np.log(count))
    w = np.exp(log_full - logsumexp(log_full))
    variance = float(np.sum((w - 1 / count) ** 2) * count / (count - 1)) if count > 1 else 0.
    if conditional:
        nll += float(logsumexp(log_train) - np.log(count))
        v = np.exp(log_train - logsumexp(log_train))
        variance = float(np.sum((w - v) ** 2) * count / (count - 1)) if count > 1 else 0.
    return dict(nll=nll, mc_se=float(np.sqrt(variance)), repeats=count,
                max_system_weight=float(w.max()), individual_nll=(-log_full).tolist())


def rank(output: Path, subject: dict, candidates: list[dict], particles: int,
         repeats: int, seed_base: int) -> list[dict]:
    rows = []
    for c in candidates:
        jobs = fit_jobs(output, subject, [c], particles, repeats, seed_base)
        records = [json.loads((output / "jobs" / f"{job['id']}.json").read_text()) for job in jobs]
        rows.append(dict(candidate=c, **aggregate(records)))
    return sorted(rows, key=lambda row: row["nll"])


def refinement(output: Path, subject: dict, variant: str, parents: list[dict], policy: dict) -> list[dict]:
    seed = int(identity(dict(subject=subject["subject"], variant=variant)), 16) % (2 ** 32)
    rng = np.random.default_rng(seed)
    values = policy["parameter_values"]
    result = {}
    for _ in range(4096):
        p = deepcopy(parents[int(rng.integers(len(parents)))]["candidate"]["parameters"])
        for key in rng.choice(list(values), size=int(rng.integers(1, 4)), replace=False):
            p[key] = values[key][int(rng.integers(len(values[key])))]
        c = candidate(output, subject["condition"], p, variant, policy)
        result[c["id"]] = c
        if len(result) >= policy["search"]["refinement_candidates"]:
            break
    return list(result.values())


def validate_source(output: Path) -> dict:
    register = json.loads((output / "registration.json").read_text())
    for name, expected in register["source_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen numerical source changed: {name}")
    for subject in register["subjects"]:
        if digest(output / "inputs" / f"s{subject['subject']}.npz") != subject["input_sha256"]:
            raise ValueError("Frozen behavior inputs changed")
    for name, expected in register["perception_sha256"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"Frozen perception input changed: {name}")
    if digest(ROOT / register["policy"]["data"]["source"]) != register["data_sha256"]:
        raise ValueError("Frozen feature-order and behavioral source changed")
    return register


def fit_role(output: Path, role: str, variants: list[str] | None = None) -> None:
    register = validate_source(output)
    policy, search = register["policy"], register["policy"]["search"]
    subjects = [s for s in register["subjects"] if s["role"] == role]
    variants = list(policy["variants"]) if variants is None else variants
    scheduler = Scheduler(output, policy)
    groups, rankings = {}, {}
    seed_base = policy["seed"] + (1000 if role == "development" else 10000)
    try:
        for subject in subjects:
            for variant in variants:
                key = (subject["subject"], variant)
                candidates = [candidate(output, subject["condition"], p, variant, policy) for p in register["bank"]]
                groups[key] = list({c["id"]: c for c in candidates}.values())
        jobs = [job for s in subjects for v in variants
                for job in fit_jobs(output, s, groups[s["subject"], v], search["screening_particles"],
                                    search["screening_repeats"], seed_base)]
        scheduler.run(jobs, f"{role}_screen")
        for s in subjects:
            for v in variants:
                key = s["subject"], v
                rankings[key] = rank(output, s, groups[key], search["screening_particles"], search["screening_repeats"], seed_base)
                added = refinement(output, s, v, rankings[key][:search["refinement_parents"]], policy)
                groups[key] = list({c["id"]: c for c in groups[key] + added}.values())
            # Nested larger models receive smaller-model leaders before confirmation.
            for target in ("full", "no_inertia", "fixed_precision", "compact"):
                if target not in variants:
                    continue
                additions = []
                for source in ("no_inertia", "fixed_precision", "compact", "constant_search", "bayes_retention"):
                    if source not in variants:
                        continue
                    for row in rankings[s["subject"], source][:2]:
                        p = row["candidate"]["parameters"]
                        if (target in ("no_inertia", "compact") and p["execution_inertia"] != 0.
                                or target in ("fixed_precision", "compact") and p["learning_rate"] != 0.):
                            continue
                        additions.append(candidate(output, s["condition"], p, target, policy))
                key = s["subject"], target
                groups[key] = list({c["id"]: c for c in groups[key] + additions}.values())
        jobs = [job for s in subjects for v in variants
                for job in fit_jobs(output, s, groups[s["subject"], v], search["screening_particles"],
                                    search["screening_repeats"], seed_base)]
        scheduler.run(jobs, f"{role}_refine")
        # Refined boundary solutions remain admissible in each containing model.
        # Their canonical job identities reuse the completed likelihood estimates.
        for s in subjects:
            refined = {v: rank(output, s, groups[s["subject"], v], search["screening_particles"],
                               search["screening_repeats"], seed_base) for v in variants}
            for target in ("full", "no_inertia", "fixed_precision", "compact"):
                if target not in variants:
                    continue
                additions = []
                for source in ("no_inertia", "fixed_precision", "compact", "constant_search", "bayes_retention"):
                    for row in refined.get(source, [])[:2]:
                        p = row["candidate"]["parameters"]
                        if (target in ("no_inertia", "compact") and p["execution_inertia"] != 0.
                                or target in ("fixed_precision", "compact") and p["learning_rate"] != 0.):
                            continue
                        additions.append(candidate(output, s["condition"], p, target, policy))
                key = s["subject"], target
                groups[key] = list({c["id"]: c for c in groups[key] + additions}.values())
        for stage, number, particles, repeats, offset in (
                ("confirmation", search["confirmation_candidates"], search["confirmation_particles"], search["confirmation_repeats"], 100),
                ("final", search["final_candidates"], search["final_particles"], search["final_repeats"], 200)):
            previous_particles = search["screening_particles"] if stage == "confirmation" else search["confirmation_particles"]
            previous_repeats = search["screening_repeats"] if stage == "confirmation" else search["confirmation_repeats"]
            previous_offset = 0 if stage == "confirmation" else 100
            for s in subjects:
                for v in variants:
                    key = s["subject"], v
                    rankings[key] = rank(output, s, groups[key], previous_particles, previous_repeats, seed_base + previous_offset)
                    groups[key] = [row["candidate"] for row in rankings[key][:number]]
            jobs = [job for s in subjects for v in variants for job in fit_jobs(
                output, s, groups[s["subject"], v], particles, repeats, seed_base + offset)]
            scheduler.run(jobs, f"{role}_{stage}")
        frozen = []
        for s in subjects:
            for v in variants:
                key = s["subject"], v
                rows = rank(output, s, groups[key], search["final_particles"], search["final_repeats"], seed_base + 200)
                earlier = {row["candidate"]["id"]: row for row in rankings[key]}
                difference = max((abs(row["nll"] - earlier[row["candidate"]["id"]]["nll"])
                    + 2 * np.hypot(row["mc_se"], earlier[row["candidate"]["id"]]["mc_se"])) / s["train_trials"] for row in rows)
                frozen.append(dict(subject=s, variant=v, candidates=rows, particles=search["final_particles"],
                    precision_difference_per_trial=float(difference),
                    numerical_status="within_working_tolerance" if difference <= search["numerical_tolerance_nll_per_trial"] else "requires_extension"))
        extension = [group for group in frozen if group["numerical_status"] == "requires_extension"]
        if extension:
            jobs = [job for group in extension for job in fit_jobs(output, group["subject"],
                [row["candidate"] for row in group["candidates"]], search["maximum_particles"],
                search["maximum_repeats"], seed_base + 300)]
            scheduler.run(jobs, f"{role}_numerical_extension")
            for group in extension:
                previous = {row["candidate"]["id"]: row for row in group["candidates"]}
                rows = rank(output, group["subject"], [row["candidate"] for row in group["candidates"]],
                    search["maximum_particles"], search["maximum_repeats"], seed_base + 300)
                difference = max((abs(row["nll"] - previous[row["candidate"]["id"]]["nll"])
                    + 2 * np.hypot(row["mc_se"], previous[row["candidate"]["id"]]["mc_se"]))
                    / group["subject"]["train_trials"] for row in rows)
                group.update(candidates=rows, particles=search["maximum_particles"],
                    precision_difference_per_trial=float(difference),
                    numerical_status="within_working_tolerance" if difference <= search["numerical_tolerance_nll_per_trial"] else "numerically_limited")
        write_new(output / f"{role}_parameters.json", dict(groups=frozen, cpu_seconds=scheduler.cpu_used(),
                                                          completed_at_utc=datetime.now(timezone.utc).isoformat()))
    finally:
        scheduler.close(terminate=True)


def evaluate_role(output: Path, role: str) -> None:
    register = validate_source(output)
    policy, search = register["policy"], register["policy"]["search"]
    frozen = json.loads((output / f"{role}_parameters.json").read_text())["groups"]
    scheduler = Scheduler(output, policy)
    seed_base = policy["seed"] + (2000 if role == "development" else 11000)
    try:
        jobs = []
        for group in frozen:
            s, c = group["subject"], group["candidates"][0]["candidate"]
            jobs += fit_jobs(output, s, [c], group["particles"], search["final_repeats"], seed_base, scope="full")
            for chunk in range(0, policy["generation"]["paths_per_candidate"], 8):
                job = dict(output=str(output.resolve()), kind="generate", subject=s["subject"],
                    condition=s["condition"], train_trials=s["train_trials"], config=c["id"],
                    seeds=list(range(seed_base + 500 + chunk, seed_base + 500 + min(chunk + 8, policy["generation"]["paths_per_candidate"]))))
                job["id"] = identity(job)
                jobs.append(job)
        records = scheduler.run(jobs, f"{role}_future_and_generation")
        rows = []
        for group in frozen:
            s, c = group["subject"], group["candidates"][0]["candidate"]
            matching = [r for r in records.values() if r["job"]["subject"] == s["subject"] and r["job"]["config"] == c["id"]]
            fitted = [r for r in matching if r["job"]["kind"] == "fit"]
            prediction = aggregate(fitted, conditional=True)
            generated = [r for r in matching if r["job"]["kind"] == "generate"]
            paths = np.concatenate([np.load(output / "jobs" / f"{r['job']['id']}.npz")["choices"] for r in generated])
            with np.load(output / "inputs" / f"s{s['subject']}.npz") as data:
                features = path_features(paths, data["categories"], s["condition"])
                observed = path_features(data["choices"], data["categories"], s["condition"])[0]
            rows.append(dict(subject=s["subject"], condition=s["condition"], variant=group["variant"],
                candidate=c, test_trials=s["trials"] - s["train_trials"],
                test_nll_per_trial=prediction["nll"] / (s["trials"] - s["train_trials"]),
                mc_se_per_trial=prediction["mc_se"] / (s["trials"] - s["train_trials"]),
                max_system_weight=prediction["max_system_weight"], energy=energy_score(features, observed),
                numerical_status=group["numerical_status"], paths=len(paths)))
        write_new(output / f"{role}_evaluation.json", dict(rows=rows, cpu_seconds=scheduler.cpu_used()))
    finally:
        scheduler.close(terminate=True)


def calibrate(output: Path) -> None:
    register = validate_source(output)
    policy = register["policy"]
    rows = []
    for condition in (1, 2, 3):
        s = next(s for s in register["subjects"] if s["condition"] == condition)
        c = candidate(output, condition, DEFAULT, "compact", policy)
        job = fit_jobs(output, s, [c], 8, 1, policy["seed"] + 77, max_trials=24)[0]
        rows.append(_work(job))
    write_new(output / "calibration.json", rows)
    print(json.dumps([dict(condition=r["job"]["condition"], seconds=r["seconds"], cpu_seconds=r["cpu_seconds"]) for r in rows]), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("prepare", "calibrate", "fit-development", "evaluate-development", "fit-evaluation", "evaluate-evaluation"))
    parser.add_argument("--config", type=Path, default=Path("configs/shared/model1031_finalization_20261101.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.phase == "prepare":
        prepare(args.config, output)
    elif args.phase == "calibrate":
        calibrate(output)
    elif args.phase.startswith("fit-"):
        fit_role(output, args.phase.removeprefix("fit-"), args.variants)
    else:
        evaluate_role(output, args.phase.removeprefix("evaluate-"))


if __name__ == "__main__":
    main()
