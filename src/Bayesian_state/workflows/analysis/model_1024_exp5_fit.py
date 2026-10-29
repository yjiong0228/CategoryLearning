"""Bounded full-record Exp5 fitting on the shared Model 1024 mechanisms.

Search proposals, sequence likelihood, numerical checks and process accounting
reuse the existing fitting tools. Every worker imports a frozen source snapshot.
Autonomous evaluation follows candidate freezing and never selects parameters.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import tarfile
import tempfile
import time

if __package__ in (None, ""):
    __package__ = "src.Bayesian_state.workflows.analysis"

import numpy as np
import pandas as pd
import yaml

from .model_1024_execution_comparison import Queue, read, write, digest
from ...evaluation.execution_comparison import precision_check
from ...evaluation.independent_particle_systems import evidence_weights
from ...inference.dispatcher import run_inference_backend
from ...metrics.autonomous_fit import behavior_features, energy_u
from ...metrics.trajectory_shape import sequence_score
from ...model.config import ModelConfig
from ...model.model_1024 import model_1024_runtime
from ...optimization.search.adaptive_proposals import candidate, initial_points, propose_round, select_elites
from ...simulation.exp5_data import load_exp5_subject
from ...simulation.probabilistic import run_autonomous_probabilistic_learning
from ...utils.seeding import stable_seed

ROOT = Path(__file__).resolve().parents[4]
ENTRY = "src/Bayesian_state/workflows/analysis/model_1024_exp5_fit.py"
H = "modules.hypo_transitions_mod.kwargs."
COORDINATES = dict(p0=H+"search_controller.baseline_probability", error_gain=H+"search_controller.error_gain",
                   failure_decay=H+"search_controller.failure_decay", global_search=H+"search_controller.global_search",
                   capacity=H+"max_capacity", retention=H+"intervention.decay", inertia=H+"execution_inertia",
                   beta_init="modules.beta_mod.kwargs.beta_init", learning_rate="modules.beta_mod.kwargs.learning_rate")


def canonical(point: dict, source: str) -> dict:
    values = deepcopy(point)
    if set(values) != set(COORDINATES):
        raise ValueError("Exp5 fitting requires exactly the nine declared coordinates")
    bounds = dict(p0=(0, 1), error_gain=(0, np.inf), failure_decay=(0, 1), global_search=(0, 1),
                  retention=(0, 1), inertia=(0, np.inf), beta_init=(.1, 25), learning_rate=(0, 1), capacity=(2, 61))
    for key, (lo, hi) in bounds.items():
        value = values[key]
        if isinstance(value, bool) or not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"Invalid {key}: {value}")
    if not 0 < values["p0"] < 1 or values["failure_decay"] == 1 or int(values["capacity"]) != values["capacity"]:
        raise ValueError("Invalid search probability, failure decay or capacity")
    values["capacity"] = int(values["capacity"])
    if values["error_gain"] == 0:
        values["failure_decay"] = 0.
    return candidate(values, source)


def engine_at(base: dict, point: dict, particles: int) -> dict:
    engine = deepcopy(base)
    for key, path in COORDINATES.items():
        owner = engine
        parts = path.split(".")
        for part in parts[:-1]:
            owner = owner[part]
        owner[parts[-1]] = point["hyperparams"][key]
    engine["inference"]["particle_count"] = particles
    return ModelConfig.from_mapping(engine).to_dict()


def anchor(base: dict) -> dict:
    point = {}
    for key, path in COORDINATES.items():
        value = base
        for part in path.split("."):
            value = value[part]
        point[key] = value
    return point


def seeds(policy: dict, role: str, count: int) -> list[int]:
    return [stable_seed(dict(protocol=policy["protocol"], seed=policy["seed"], subject=policy["subject"],
                             role=role, repeat=i)) for i in range(count)]


def freeze_sources(output: Path) -> tuple[str, dict]:
    """Archive exact Python bytes and extract them into a private temporary root."""
    files = sorted((ROOT / "src/Bayesian_state").rglob("*.py"))
    if (ROOT / "src/__init__.py").exists():
        files.append(ROOT / "src/__init__.py")
    hashes = {}
    with tarfile.open(output / "source_snapshot.tar", "x") as archive:
        for path in files:
            name = str(path.relative_to(ROOT))
            data = path.read_bytes()
            import hashlib
            hashes[name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    if hashes != {str(path.relative_to(ROOT)): digest(path) for path in files}:
        raise ValueError("Source changed while freezing; retain this incomplete directory and prepare a new one")
    runtime = Path(tempfile.mkdtemp(prefix="model1024_exp5_fit_"))
    with tarfile.open(output / "source_snapshot.tar") as archive:
        archive.extractall(runtime, filter="data")
    return str(runtime), hashes


def prepare(config: Path, output: Path, smoke: bool = False) -> dict:
    if output.exists():
        raise FileExistsError("Use a new fit output directory")
    p = yaml.safe_load(config.read_text())
    if (p.get("protocol") != "model1024_exp5_full_record_fit_v1"
            or p.get("objective") != "mean_sequence_likelihood" or p.get("oral_use") != "excluded"):
        raise ValueError("Unexpected Exp5 fitting protocol or objective")
    if sum(p["budget"]["stage_cpu_hours"].values()) != p["budget"]["cpu_hours"]:
        raise ValueError("Stage CPU allocations must sum to the total")
    case = load_exp5_subject(config.parent / p["input_config"], p["subject"])
    base = yaml.safe_load((config.parent / p["engine_config"]).read_text())
    ModelConfig.from_mapping(base)
    default = anchor(base)
    if set(p["parameter_values"]) != set(COORDINATES):
        raise ValueError("Parameter grid must cover the nine Exp5 coordinates")
    for key, values in p["parameter_values"].items():
        if not values or len(set(values)) != len(values) or default[key] not in values:
            raise ValueError(f"Distinct {key} levels must include the fixed baseline")
        for value in values:
            canonical(dict(default, **{key: value}), "domain_check")
    if smoke:
        p["search"].update(initial=6, rounds=1, elites=1, proposals_per_elite=2, particles=2, repeats=2,
                           confirmation_points=2, confirmation_particles=4, confirmation_repeats=2,
                           audit_points=2, audit_particles=8, audit_repeats=2,
                           escalation_particles=16, escalation_repeats=2)
        p["generation"].update(maximum_candidates=1, paths=4, chunk=2)
        p["budget"].update(workers=1, poll_seconds=.05)
    output.mkdir(parents=True)
    arrays = case.arrays
    n = min(12, len(arrays.choices)) if smoke else len(arrays.choices)
    with (output / "observations.npz").open("xb") as stream:
        np.savez_compressed(stream, stimulus=arrays.stimulus[:n], choices=arrays.choices[:n], feedback=arrays.feedback[:n])
    with (output / "task_metadata.npz").open("xb") as stream:
        np.savez_compressed(stream, categories=arrays.categories[:n], trial_ids=case.analysis_frame[["iSession", "iTrial"]].values[:n])
    write(output / "engine.json", base)
    write(output / "input_metadata.json", case.metadata())
    (output / "registered_config.yaml").write_text(yaml.safe_dump(p, sort_keys=False, allow_unicode=True))
    runtime, hashes = freeze_sources(output)
    protocol = dict(policy=p, smoke=smoke, runtime=runtime, source_sha256=hashes, trials=n,
                    input_sha256={name: digest(output / name) for name in ("observations.npz", "task_metadata.npz", "engine.json", "input_metadata.json", "registered_config.yaml")},
                    snapshot_sha256=digest(output / "source_snapshot.tar"),
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    versions={"python": platform.python_version(), **{
                        name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "PyYAML")}},
                    prepared_unix=time.time(), data_use="full_record_development_no_held_out_claim",
                    calibrated_seconds_128_particles_capacity3=18.)
    write(output / "protocol.json", protocol)
    return protocol


class Exp5Queue(Queue):
    """Use the shared measured-CPU queue with the Exp5 calibration cost."""
    def estimate(self, job: dict) -> float:
        if self.protocol["smoke"]:
            return 5.
        capacity = job["candidate"]["hyperparams"]["capacity"]
        scale = max(1., capacity / 3)
        if job["phase"] == "autonomous":
            return max(10., len(job["seeds"]) * .5 * scale)
        return max(20., self.protocol["calibrated_seconds_128_particles_capacity3"] * job["particles"] / 128 * scale * 1.5)


def worker(file: Path) -> None:
    job = read(file)
    output = Path(job["output"])
    protocol = read(output / "protocol.json")
    if any(digest(ROOT / name) != value for name, value in protocol["source_sha256"].items()):
        raise ValueError("Frozen source identity mismatch")
    if any(digest(output / name) != value for name, value in protocol["input_sha256"].items()):
        raise ValueError("Registered input identity mismatch")
    limit = int(job["cpu_limit_seconds"])
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit+1))
    def exhausted(*_):
        raise TimeoutError("Worker CPU allocation exhausted")
    signal.signal(signal.SIGXCPU, exhausted)
    status = "failed"
    try:
        engine = engine_at(read(output / "engine.json"), job["candidate"], job["particles"] or 2)
        with np.load(output / "observations.npz") as z:
            x, choices, feedback = (z[k] for k in ("stimulus", "choices", "feedback"))
        payload = {}
        with model_1024_runtime():
            if job["phase"] == "autonomous":
                with np.load(output / "task_metadata.npz") as z:
                    labels = z["categories"]
                paths = []
                for seed in job["seeds"]:
                    result = run_autonomous_probabilistic_learning(engine_config=engine, subject_id=job["subject"],
                        stimulus=x, target_probabilities=np.eye(2)[labels-1], trajectory_seed=seed)
                    np.testing.assert_array_equal(result.trajectory.feedback, result.trajectory.choices == labels)
                    paths.append(result.trajectory.choices)
                payload["choices"] = np.asarray(paths)
            else:
                probabilities, ess = [], []
                for seed in job["seeds"]:
                    result = run_inference_backend(engine_config=engine, subject_id=job["subject"], condition=1,
                        stimulus=x, choices=choices, feedback=feedback, inference_seed=seed)
                    probabilities.append(result.observation_probabilities["prior_t"])
                    ess.append(result.pre_choice_ess)
                values = np.asarray(probabilities)
                payload.update(probabilities=values, observed_probability=values[:, np.arange(len(choices)), choices-1],
                               pre_choice_ess=np.asarray(ess))
                sequence_score(payload["observed_probability"])
        target = Path(job["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.with_suffix(".npz").open("xb") as stream:
            np.savez_compressed(stream, **payload, seeds=job["seeds"])
        write(target.with_suffix(".json"), dict(status="complete", array_sha256=digest(target.with_suffix(".npz")),
              trials=len(x), candidate=job["candidate"], particles=job["particles"], seeds=job["seeds"]))
        status = "complete"
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        write(file.with_suffix(".usage.json"), dict(status=status, cpu_seconds=usage.ru_utime+usage.ru_stime))


def jobs_for(protocol: dict, output: Path, points: list[dict], role: str, stage: str,
             particles: int = 0, repeats: int = 0, paths: int = 0) -> list[dict]:
    bank = seeds(protocol["policy"], role, paths or repeats)
    chunk = protocol["policy"]["generation"]["chunk"] if paths else 1
    jobs = []
    for point in points:
        point = {key: point[key] for key in ("id", "hyperparams", "sources", "origin")}
        for offset in range(0, len(bank), chunk):
            name = f'{role}_{point["id"]}_{offset:03d}'
            jobs.append(dict(name=name, subject=protocol["policy"]["subject"], candidate=point,
                seeds=bank[offset:offset+chunk], particles=particles, phase="autonomous" if paths else "filter",
                runtime=protocol["runtime"], worker_entry=str(Path(protocol["runtime"]) / ENTRY),
                output=str(output.resolve()), target=str((output / "numerical" / name).resolve()), budget_stage=stage))
    return jobs


def collect(jobs: list[dict], key: str) -> np.ndarray | None:
    values = []
    for job in jobs:
        target = Path(job["target"])
        receipt = Path(job["output"]) / "ledger" / (job["name"] + ".json")
        if not receipt.exists() or read(receipt)["status"] != "complete":
            return None
        if digest(target.with_suffix(".npz")) != read(target.with_suffix(".json"))["array_sha256"]:
            raise ValueError("Numerical output identity changed")
        with np.load(target.with_suffix(".npz")) as z:
            values.append(z[key])
    return np.concatenate(values) if values else None


def evaluate(protocol: dict, output: Path, queue: Exp5Queue, points: list[dict], role: str,
             stage: str, particles: int, repeats: int) -> list[dict]:
    points = [{key: point[key] for key in ("id", "hyperparams", "sources", "origin")} for point in points]
    jobs = jobs_for(protocol, output, points, role, stage, particles, repeats)
    queue.execute(jobs, role)
    rows = []
    for point in points:
        own = [job for job in jobs if job["candidate"]["id"] == point["id"]]
        values = collect(own, "observed_probability")
        if values is not None:
            score = sequence_score(values)
            rows.append(dict(**point, mean_nll=score["nll"], score=score, jobs=own, particles=particles))
    write(output / "scores" / (role + ".json"), rows)
    return rows


def pooled_predictions(probabilities: np.ndarray, choices: np.ndarray) -> np.ndarray:
    """Weight each filter by evidence preceding the current choice."""
    own = probabilities[:, np.arange(len(choices)), choices-1]
    cumulative = np.cumsum(np.log(own), axis=1)
    prior = np.c_[np.zeros(len(own)), cumulative[:, :-1]]
    return np.einsum("rt,rtc->tc", evidence_weights(prior), probabilities)


def run(output: Path) -> dict:
    if (output / "completion.json").exists():
        raise FileExistsError("This fitting run already has a completion record")
    protocol = read(output / "protocol.json")
    p, base = protocol["policy"], read(output / "engine.json")
    s = p["search"]
    if any(digest(Path(protocol["runtime"]) / name) != value for name, value in protocol["source_sha256"].items()):
        raise ValueError("Frozen runtime is missing or changed")
    queue = Exp5Queue(output, protocol)
    bank = {}
    for point in initial_points(p["parameter_values"], anchor(base), s["initial"], seeds(p, "initial_proposals", 1)[0], cover_key="capacity"):
        item = canonical(point, "initial_spread")
        bank[item["id"]] = item
    for capacity in p["parameter_values"]["capacity"]:
        item = canonical(dict(anchor(base), capacity=capacity, inertia=0), "zero_boundary_anchor")
        bank[item["id"]] = item
    rows = evaluate(protocol, output, queue, list(bank.values()), "screen_0", "search", s["particles"], s["repeats"])
    for index in range(s["rounds"]):
        elites = select_elites(rows, s["elites"], exclude_key="capacity")
        proposals = propose_round(elites, p["parameter_values"], set(bank), s["proposals_per_elite"],
            seeds(p, f"proposals_{index}", 1)[0], s["local_fraction"], s["jump_fraction"])
        added = {}
        for point in proposals:
            item = canonical(point["hyperparams"], f"search_round_{index+1}")
            if item["id"] not in bank:
                added[item["id"]] = item
        bank.update(added)
        rows.extend(evaluate(protocol, output, queue, list(added.values()), f"screen_{index+1}", "search", s["particles"], s["repeats"]))
    write(output / "candidate_bank.json", list(bank.values()))
    # Half the confirmation quota follows NLL rank, half preserves distinct alternatives.
    ordered = sorted(rows, key=lambda row: (row["mean_nll"], row["id"]))
    diverse = select_elites(rows, s["confirmation_points"], exclude_key="capacity")
    selected = {r["id"]: r for r in ordered[:max(1, s["confirmation_points"]//2)]}
    for row in diverse + ordered:
        if len(selected) >= s["confirmation_points"]:
            break
        selected.setdefault(row["id"], row)
    confirmed = evaluate(protocol, output, queue, list(selected.values()), "confirm", "confirmation", s["confirmation_particles"], s["confirmation_repeats"])
    nominees = sorted(confirmed, key=lambda row: (row["mean_nll"], row["id"]))[:s["audit_points"]]
    audited = evaluate(protocol, output, queue, nominees, "audit", "audit", s["audit_particles"], s["audit_repeats"])
    previous = {r["id"]: r for r in confirmed}
    for row in audited:
        row["precision"] = precision_check(previous[row["id"]]["score"], row["score"], s)
    final = audited
    escalated = any(not row["precision"]["passed"] for row in audited)
    if escalated:
        final = evaluate(protocol, output, queue, nominees, "escalation", "audit", s["escalation_particles"], s["escalation_repeats"])
        previous = {r["id"]: r for r in audited}
        for row in final:
            row["precision"] = precision_check(previous[row["id"]]["score"], row["score"], s) if row["id"] in previous else {"passed": False}
    final.sort(key=lambda row: (row["mean_nll"], row["id"]))
    frozen = dict(status="frozen" if final else "incomplete", candidates=final, frozen_unix=time.time(),
                  evaluated_candidates=len(rows), proposed_candidates=len(bank), escalated=escalated,
                  data_use=p["data_use"], selection="choice_sequence_likelihood_only")
    if final:
        best = final[0]
        frozen["best_id"] = best["id"]
        frozen["working_sets"] = {str(t): [r["id"] for r in final if r["mean_nll"]-best["mean_nll"] <= t]
                                  for t in s["working_nll_tolerances"]}
        frozen["numerically_checked_working_sets"] = {str(t): [r["id"] for r in final
            if r["mean_nll"]-best["mean_nll"] <= t and r["precision"]["passed"]] for t in s["working_nll_tolerances"]}
        engine = engine_at(base, best, best["particles"])
        engine["provenance"]["scientific_status"] = "full_record_development_estimate"
        (output / "best_engine.yaml").write_text(yaml.safe_dump(engine, sort_keys=False, allow_unicode=True))
        with np.load(output / "observations.npz") as z:
            choices = z["choices"]
        probability = collect(best["jobs"], "probabilities")
        pooled = pooled_predictions(probability, choices)
        np.testing.assert_allclose(-np.log(pooled[np.arange(len(choices)), choices-1]).mean(), best["mean_nll"], atol=1e-12)
        frame = pd.DataFrame(dict(trial=np.arange(1, len(choices)+1), choice=choices,
                                  probability_category1=pooled[:, 0], probability_category2=pooled[:, 1]))
        frame.to_csv(output / "best_predictions.csv", index=False)
    write(output / "frozen.json", frozen)
    generation_points = [r for r in final if r["mean_nll"]-final[0]["mean_nll"] <= s["primary_tolerance"]][:p["generation"]["maximum_candidates"]]
    generation = jobs_for(protocol, output, generation_points, "autonomous", "generation", paths=p["generation"]["paths"])
    queue.execute(generation, "autonomous")
    completions = [read(path) for path in (output / "batches").glob("*.completion.json")]
    complete = bool(final) and len(final) == len(nominees) and all(r["succeeded"] == r["expected"] and r["status"] == "complete" for r in completions)
    result = dict(status="completed" if complete else "bounded_partial", stop_reason=queue.stop,
                  cpu_hours=queue.used()/3600, elapsed_hours=(time.time()-queue.started)/3600,
                  peak_workers=queue.peak, successful_jobs=sum(r["status"] == "complete" for r in queue.records.values()),
                  failed_jobs=sum(r["status"] == "failed" for r in queue.records.values()),
                  numerical_status="checked" if final and all(r["precision"]["passed"] for r in final) else "unresolved",
                  formal_state_recovery="not_run", fitted_subject=p["subject"], trials=protocol["trials"])
    write(output / "completion.json", result)
    report(output)
    return result


def report(output: Path) -> Path:
    """Create a fresh, self-contained fit report from saved numerical artifacts."""
    protocol, frozen = read(output / "protocol.json"), read(output / "frozen.json")
    completion = read(output / "completion.json")
    index = 1
    while (output / f"report_v{index}").exists():
        index += 1
    folder = output / f"report_v{index}"
    folder.mkdir()
    final, p = frozen["candidates"], protocol["policy"]
    with np.load(output / "observations.npz") as z:
        choices, feedback = z["choices"], z["feedback"]
    with np.load(output / "task_metadata.npz") as z:
        labels = z["categories"]
    scope = "短序列流程试跑" if protocol["smoke"] else "完整记录拟合"
    trial_scope = "前" if protocol["smoke"] else "全部"
    lines = [f'# Exp5 被试 {p["subject"]}：Model 1024 {scope}', '',
             f'使用{trial_scope} {len(choices)} 题，实际正确率 {feedback.mean():.1%}。计算状态：`{completion["status"]}`；数值状态：`{completion["numerical_status"]}`。', '',
             '参数依据完整选择序列的似然估计，潜在学习路径由粒子过滤积分。口述未进入选参。全部记录用于开发拟合，后续自主生成使用冻结参数。', '',
             f'搜索提出 {frozen["proposed_candidates"]} 个不同组合，{frozen["evaluated_candidates"]} 个完成初筛；最终高粒子档评价 {len(final)} 个组合。搜索为登记范围内的有限搜索。', '',
             '| 候选 | 每题 NLL | 蒙特卡洛标准误 | 粒子数 | 独立重复 | 数值检查 |', '|---|---:|---:|---:|---:|---|']
    records = []
    for row in final:
        lines.append(f'| {row["id"]} | {row["mean_nll"]:.6f} | {row["score"]["mcse"]:.6f} | {row["particles"]} | {row["score"]["repeats"]} | {"通过" if row["precision"]["passed"] else "受限"} |')
        records.append(dict(candidate=row["id"], nll=row["mean_nll"], mcse=row["score"]["mcse"], precision_passed=row["precision"]["passed"], **row["hyperparams"]))
    pd.DataFrame(records).to_csv(folder / "parameters.csv", index=False)
    if final:
        best = final[0]
        lines += ['', '## 最低 NLL 候选', '', '| 参数 | 数值 | 登记范围 |', '|---|---:|---|']
        for key, value in best["hyperparams"].items():
            levels = p["parameter_values"][key]
            lines.append(f'| {key} | {value:g} | {min(levels):g}—{max(levels):g} |')
        lines += ['', 'NLL为负对数似然，越低表示模型给予实际选择序列的概率越高。标准误仅衡量有限计算的误差。',
                  '相邻粒子档的NLL差加两倍合并标准误须不超过0.005；两倍单点标准误须不超过0.005，最大独立系统权重须不超过0.9。',
                  '候选按相对最低点的每题NLL差0.005、0.01、0.02分别保留。工作集合的统计置信覆盖尚未校准，参数没有后验权重。', '',
                  '完整参数表见[parameters.csv](parameters.csv)，冻结候选见[原始记录](../frozen.json)。',
                  '可运行配置见[best_engine.yaml](../best_engine.yaml)，拟合内逐题预测见[best_predictions.csv](../best_predictions.csv)。', '', '## 冻结参数后的自主生成', '']
    batch = read(output / "batches/autonomous.json")["jobs"]
    generation_rows = []
    for row in final:
        own = [j for j in batch if j["candidate"]["id"] == row["id"]]
        if not own:
            continue
        paths = collect(own, "choices")
        if paths is None:
            lines.append(f'{row["id"]} 的自主生成未完整完成。')
            continue
        generation = p["generation"]
        features = behavior_features(paths, labels, n_categories=2, block_width=generation["block_width"], temporal_weight=generation["temporal_weight"])
        target = behavior_features(choices[None], labels, n_categories=2, block_width=generation["block_width"], temporal_weight=generation["temporal_weight"])[0]
        energy = energy_u(features, target)
        correct = paths == labels
        own_summary = dict(candidate=row["id"], **energy, overall_mean=float(correct.mean()),
                           overall_q025=float(np.quantile(correct.mean(axis=1), .025)),
                           overall_q975=float(np.quantile(correct.mean(axis=1), .975)))
        generation_rows.append(own_summary)
        lines += [f'候选 `{row["id"]}`：全部 {len(paths)} 条路径的平均正确率 {correct.mean():.1%}，',
                  f'整段正确率中央95%范围 {own_summary["overall_q025"]:.1%}—{own_summary["overall_q975"]:.1%}；Energy分数 {energy["value"]:.5f}，计算标准误 {energy["mc_se"]:.5f}。', '',
                  '| 题目范围 | 真人正确率 | 模拟均值 | 模拟中央95%范围 |', '|---|---:|---:|---|']
        block_records = []
        width = generation["block_width"]
        for a in range(0, len(labels), width):
            b = min(a+width, len(labels))
            values = correct[:, a:b].mean(axis=1)
            low, high = np.quantile(values, [.025, .975])
            observed = float(feedback[a:b].mean())
            lines.append(f'| {a+1}—{b} | {observed:.1%} | {values.mean():.1%} | {low:.1%}—{high:.1%} |')
            block_records.append(dict(first_trial=a+1, last_trial=b, observed=observed, simulated_mean=float(values.mean()), q025=float(low), q975=float(high)))
        pd.DataFrame(block_records).to_csv(folder / f'{row["id"]}_blocks.csv', index=False)
        lines += ['']
    write(folder / "generation_summary.json", generation_rows)
    lines += ['Energy分数比较全部自主路径的32题块正确率及类别条件反应频率与真人记录，越小表示分布更相容。上述范围表示模型的过程变异。', '',
              '## 计算与适用范围', '',
              f'成功任务 {completion["successful_jobs"]} 项，失败 {completion["failed_jobs"]} 项；CPU消耗 {completion["cpu_hours"]:.3f} 小时，运行 {completion["elapsed_hours"]:.3f} 小时，峰值 {completion["peak_workers"]} 个单线程进程。',
              '使用62条奇偶规则目录、准确黑白感知及实际刺激顺序与记录长度。规则目录覆盖、个体参数可辨识性和正式状态恢复仍需后续验证。',
              '数值稳定、搜索覆盖、拟合表现和心理解释分别评价；边界参数与候选分歧保留在参数表中。',
              f'源码起点提交 `{protocol["git_head"]}`；实际运行的Python源码完整冻结在`source_snapshot.tar`，逐文件哈希与依赖版本保存在[协议](../protocol.json)。',
              '分阶段评分位于`scores/`，每项任务记录、CPU账本及未完成批次位于`jobs/`、`ledger/`和`batches/`。']
    (folder / "report.md").write_text('\n'.join(lines)+'\n')
    return folder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "smoke", "run", "report"))
    parser.add_argument("--config", type=Path, default=Path("configs/exp5/model1024_fit_501.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.output is None or args.phase is None:
        parser.error("--phase and --output are required outside worker mode")
    if args.phase in ("prepare", "smoke"):
        prepare(args.config, args.output, smoke=args.phase == "smoke")
    if args.phase in ("run", "smoke"):
        print(json.dumps(run(args.output)), flush=True)
    if args.phase == "report":
        print(report(args.output))


if __name__ == "__main__":
    main()
