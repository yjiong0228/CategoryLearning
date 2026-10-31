"""Independently verify saved probabilities and summarize the completed study.

This command launches no model fits. It recomputes likelihoods from saved arrays
and pools state summaries using the appropriate prefix likelihood weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from ..analysis.model_1031_decision import summarize
from ..analysis.model_1031_finalization import digest, energy_score, path_features, validate_source, write_new


LABELS = dict(full="完整参考", no_inertia="取消额外执行维持", fixed_precision="固定共享精度",
              compact="共同简化", constant_search="恒定搜索", constant_retention="恒定保持",
              bayes_retention="完全保持", similarity_retrieval="相似度检索")


def verify_jobs(output: Path) -> tuple[list[dict], dict]:
    records, inputs = [], {}
    maximum_error = 0.
    paths = list((output / "jobs").glob("*.json")) + list((output / "recovery").glob("*/jobs/*.json"))
    for path in paths:
        record = json.loads(path.read_text())
        job = record["job"]
        if digest(path.with_suffix(".npz")) != record["array_sha256"]:
            raise ValueError(f"Saved array hash changed: {path}")
        if job["kind"] == "fit":
            input_path = Path(job["output"]) / "inputs" / f"s{job['subject']}.npz"
            if input_path not in inputs:
                with np.load(input_path) as data:
                    inputs[input_path] = data["choices"].astype(int)
            with np.load(path.with_suffix(".npz")) as data:
                p = data["choice_probabilities"]
                if not np.isfinite(p).all() or np.any(p < 0.) or not np.allclose(p.sum(axis=1), 1.):
                    raise ValueError(f"Invalid choice probabilities: {path}")
                chosen = p[np.arange(len(p)), inputs[input_path][:len(p)] - 1]
                log_chosen = np.log(np.maximum(chosen, np.finfo(float).tiny))
                actual = (-float(log_chosen.sum()), -float(log_chosen[:job["train_trials"]].sum()))
                expected = (record["nll"], record["train_nll"])
                maximum_error = max(maximum_error, float(np.max(np.abs(np.subtract(actual, expected)))))
                np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-12)
        records.append(record)
    orphan = [str(path.relative_to(output)) for path in (output / "jobs").glob("*.npz") if not path.with_suffix(".json").exists()]
    return records, dict(records=len(records), counts_by_kind=dict(Counter(r["job"]["kind"] for r in records)),
        max_recomputed_nll_error=maximum_error, orphan_arrays=orphan,
        recorded_worker_cpu_hours=sum(r["cpu_seconds"] for r in records) / 3600)


def merge_state_records(output: Path, records: list[dict], subject: int) -> dict:
    """Condition pooled particle-system state means on the same choice prefix."""
    with np.load(output / "inputs" / f"s{subject}.npz") as data:
        choices = data["choices"].astype(int)
    log_z, states = [], []
    for record in records:
        path = Path(record["job"]["output"]) / "jobs" / f"{record['job']['id']}.npz"
        with np.load(path) as data:
            p = data["choice_probabilities"]
            chosen = p[np.arange(len(p)), choices - 1]
            log_z.append(np.cumsum(np.log(np.maximum(chosen, np.finfo(float).tiny))))
            states.append({key: data[key].copy() for key in data.files if key.startswith("filtered_representation_")})
    log_z = np.stack(log_z)
    weight = np.exp(log_z - logsumexp(log_z, axis=0, keepdims=True))
    result = {}
    for key in states[0]:
        values = np.stack([s[key] for s in states])
        merged = np.einsum("rt,rtg->tg", weight, values)
        np.testing.assert_allclose(merged.sum(axis=1), 1., atol=1e-10)
        result[key] = dict(whole=merged.mean(axis=0).tolist(), first64=merged[:64].mean(axis=0).tolist(),
                           last64=merged[-64:].mean(axis=0).tolist(),
                           repeat_whole_min=values.mean(axis=1).min(axis=0).tolist(),
                           repeat_whole_max=values.mean(axis=1).max(axis=0).tolist())
    return dict(subject=subject, state_conditioning="after current choice, before current feedback; fixed parameters",
                group_order=["uncommitted", "example", "generalization", "rule"], summaries=result)


def parameter_boundaries(groups: list[dict]) -> list[dict]:
    rows = []
    for group in groups:
        best = group["candidates"][0]
        p = best["candidate"]["parameters"]
        redundant = []
        if p["capacity"] == 1:
            redundant.append("execution_inertia has no behavioral effect at K=1")
            if group["subject"]["condition"] != 3:
                redundant.append("content retention has no behavioral effect with one active content")
        if p["error_gain"] == 0:
            redundant.append("failure_decay has no behavioral effect at error_gain=0")
        rows.append(dict(subject=group["subject"]["subject"], condition=group["subject"]["condition"],
            variant=group["variant"], parameters=p, training_nll=best["nll"],
            particles=group["particles"],
            numerical_status=group["numerical_status"], boundary_notes=redundant,
            confirmed_alternatives=group["candidates"]))
    return rows


def nested_search_diagnostics(groups: list[dict], tolerance: float) -> list[dict]:
    """Check whether a supported boundary point exposes finite-search loss."""
    rows = []
    nested = {"no_inertia", "fixed_precision", "compact", "constant_search", "bayes_retention"}
    for full in (g for g in groups if g["variant"] == "full"):
        for smaller in (g for g in groups if g["subject"]["subject"] == full["subject"]["subject"]
                        and g["variant"] in nested):
            own, other = full["candidates"][0], smaller["candidates"][0]
            n = full["subject"]["train_trials"]
            same_precision = full["particles"] == smaller["particles"] and own["repeats"] == other["repeats"]
            difference = (own["nll"] - other["nll"]) / n
            lower = difference - 2 * (own["mc_se"] + other["mc_se"]) / n
            status = ("different_numerical_precision" if not same_precision else
                      "finite_search_warning" if lower > tolerance else "no_detected_dominance_failure")
            rows.append(dict(subject=full["subject"]["subject"], condition=full["subject"]["condition"],
                boundary_variant=smaller["variant"], full_minus_boundary_training_nll_per_trial=difference,
                conservative_lower_difference=lower, full_particles=full["particles"],
                boundary_particles=smaller["particles"], status=status,
                scope="A better simpler point belongs to the full model; this diagnostic concerns finite search and numerical precision"))
    return rows


def parameter_recovery_diagnostics(recovery: list[dict], policy: dict) -> list[dict]:
    """Report errors at generating points, excluding structurally inactive coordinates."""
    rows = []
    for result in recovery:
        case, variant_name = result["case"], result["effective_source"]
        variant = policy["variants"][variant_name]
        truth = case["source"]["parameters"]
        fitted = result["fitted"][variant_name][0]["candidate"]["parameters"]
        inactive = {}
        if variant["execution"] == "zero":
            inactive["execution_inertia"] = "fixed by model structure"
        if variant["precision"] == "fixed":
            inactive["learning_rate"] = "fixed by model structure"
        if variant["search"] == "constant":
            inactive["error_gain"] = "fixed by model structure"
            inactive["failure_decay"] = "constant search"
        if variant["retention"] == "one":
            inactive["retention"] = "fixed by model structure"
        if truth["capacity"] == 1:
            inactive["execution_inertia"] = "K=1 at generating point"
            if case["subject"]["condition"] != 3:
                inactive["retention"] = "K=1 in a binary-feedback condition at generating point"
        if truth["error_gain"] == 0:
            inactive["failure_decay"] = "error_gain=0 at generating point"
        coordinates = {key: dict(generating=value, recovered=fitted[key],
            signed_error=fitted[key] - value, absolute_error=abs(fitted[key] - value),
            inactive_reason=inactive.get(key)) for key, value in truth.items()}
        rows.append(dict(case=case["id"], subject=case["subject"]["subject"],
            condition=case["subject"]["condition"], effective_source=variant_name,
            numerical_and_search_classification=result["classification"], coordinates=coordinates,
            active_coordinates=[key for key in coordinates if key not in inactive],
            scope="Descriptive finite-bank point recovery under the generating structure; fixed or inactive coordinates are excluded from recovery interpretation"))
    return rows


def generation_summaries(output: Path, records: list[dict], evaluation_rows: list[dict],
                         policy: dict, role: str, variants: list[str]) -> list[dict]:
    """Retain every registered autonomous path in marginal and joint summaries."""
    summaries = []
    base = policy["seed"] + (2500 if role == "development" else 11500)
    expected_seeds = set(range(base, base + policy["generation"]["paths_per_candidate"]))
    for row in evaluation_rows:
        if row["variant"] not in variants:
            continue
        source = [r for r in records if r["job"]["kind"] == "generate"
                  and r["job"]["subject"] == row["subject"] and r["job"]["config"] == row["candidate"]["id"]
                  and set(r["job"]["seeds"]).issubset(expected_seeds)]
        seeds = [seed for r in source for seed in r["job"]["seeds"]]
        if set(seeds) != expected_seeds or len(seeds) != len(expected_seeds):
            raise ValueError("Autonomous summaries require every registered path exactly once")
        choices, groups = [], []
        for record in source:
            with np.load(output / "jobs" / f"{record['job']['id']}.npz") as arrays:
                choices.append(arrays["choices"])
                groups.append(arrays["representation_group"])
        choices, groups = np.concatenate(choices), np.concatenate(groups)
        order = np.argsort(seeds)
        choices, groups = choices[order], groups[order]
        seeds = np.asarray(seeds)[order]
        with np.load(output / "inputs" / f"s{row['subject']}.npz") as data:
            features = path_features(choices, data["categories"], row["condition"])
            observed = path_features(data["choices"], data["categories"], row["condition"])[0]
            accuracy = (choices == data["categories"]).mean(axis=1)
            observed_accuracy = float((data["choices"] == data["categories"]).mean())
        block_count = (choices.shape[1] + 31) // 32
        categories = 2 if row["condition"] == 1 else 4
        names = ([f"accuracy_block_{i + 1}" for i in range(block_count)]
                 + ["maximum_block_improvement", "improvement_time_fraction", "post_improvement_level", "block_sd"]
                 + [f"response_{a}_given_category_{c}" for c in range(1, categories + 1) for a in range(1, categories + 1)])
        if row["condition"] == 3:
            names += ["family_accuracy", "feedback_score"]
        if features.shape != (len(expected_seeds), len(names)) or not np.isfinite(features).all():
            raise ValueError("Autonomous feature dimensions or values are invalid")
        intervals = np.quantile(features, [.05, .5, .95], axis=0)
        center = features.mean(axis=0)
        radius = np.linalg.norm(features - center, axis=1) / np.sqrt(features.shape[1])
        observed_radius = float(np.linalg.norm(observed - center) / np.sqrt(features.shape[1]))
        cutoff = float(np.quantile(radius, .9))
        group_occupancy = np.stack([(groups == g).mean(axis=1) for g in range(4)], axis=1)
        halves = [indices for indices in np.array_split(np.arange(len(seeds)), 2) if len(indices)]
        stability = [dict(paths=len(indices), seeds=seeds[indices].tolist(),
            energy=energy_score(features[indices], observed), mean_accuracy=float(accuracy[indices].mean()),
            mean_group_occupancy=group_occupancy[indices].mean(axis=0).tolist()) for indices in halves]
        summaries.append(dict(subject=row["subject"], condition=row["condition"], variant=row["variant"], role=role,
            paths=len(seeds), source_jobs=[r["job"]["id"] for r in source],
            feature_names=names, observed_features=observed.tolist(), feature_quantiles_05_50_95=intervals.tolist(),
            marginal_feature_coverage=float(np.mean((observed >= intervals[0]) & (observed <= intervals[2]))),
            joint_feature_region=dict(center=center.tolist(), radius90=cutoff, observed_radius=observed_radius,
                observed_inside=observed_radius <= cutoff, simulated_mass=float(np.mean(radius <= cutoff))),
            observed_accuracy=observed_accuracy, accuracy_quantiles_05_50_95=np.quantile(accuracy, [.05, .5, .95]).tolist(),
            representation_occupancy_quantiles_05_50_95=np.quantile(group_occupancy, [.05, .5, .95], axis=0).tolist(),
            disjoint_seed_half_diagnostics=stability,
            stability_scope="Two non-overlapping seed blocks; descriptive finite-simulation sensitivity, with structure selection still using all registered paths",
            scope="Fixed-parameter whole recorded schedule, including training; descriptive empirical feature-region coverage, not a confidence region for parameters"))
    return summaries


def audit(output: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("Choose a new audit destination")
    register = validate_source(output)
    if not (output / "pipeline_completed.json").exists():
        raise ValueError("The registered numerical pipeline has not completed")
    selected = json.loads((output / "structure_selection.json").read_text())
    dev = json.loads((output / "development_evaluation_grouped.json").read_text())["rows"]
    evaluation = json.loads((output / "evaluation_evaluation_grouped.json").read_text())["rows"]
    refined = json.loads((output / "structural_precision.json").read_text())["rows"]
    precision_path = output / "evaluation_precision.json"
    evaluation_precision = (json.loads(precision_path.read_text()) if precision_path.exists()
                            else dict(status="not_available", rows=[]))
    records, verification = verify_jobs(output)
    parameters = [group for role in ("development", "evaluation") for group in
                  json.loads((output / f"{role}_parameters.json").read_text())["groups"]]
    paired = []
    chosen, comparator = selected["evaluation_variants"]
    for subject in sorted({r["subject"] for r in evaluation}):
        own = next(r for r in evaluation if r["subject"] == subject and r["variant"] == chosen)
        other = next(r for r in evaluation if r["subject"] == subject and r["variant"] == comparator)
        paired.append(dict(subject=subject, condition=own["condition"], selected_nll=own["test_nll_per_trial"],
            comparator_nll=other["test_nll_per_trial"], nll_difference=own["test_nll_per_trial"]-other["test_nll_per_trial"],
            selected_energy=own["energy"], comparator_energy=other["energy"],
            energy_difference=own["energy"] - other["energy"],
            selected_numerical_status=own["numerical_status"], comparator_numerical_status=other["numerical_status"],
            numerical_status_scope="training parameter check; future precision is reported separately"))
    states = []
    shortlist = json.loads((output / "structural_shortlist.json").read_text())["finalists"]
    for role, rows in (("development", refined), ("evaluation", evaluation)):
        base = register["policy"]["seed"] + (18000 + 100 * shortlist.index(chosen) if role == "development" else 11000)
        for row in [r for r in rows if r["variant"] == chosen]:
            pooled_ids = set(row.get("pooled_job_ids", []))
            runs = [r for r in records if r["job"]["id"] in pooled_ids] if pooled_ids else [r for r in records if r["job"]["kind"] == "fit" and r["job"]["scope"] == "full"
                    and r["job"]["subject"] == row["subject"] and r["job"]["config"] == row["candidate"]["id"]
                    and r["job"]["seed"] in range(base, base + 4)]
            states.append(dict(merge_state_records(output, runs, row["subject"]), role=role,
                               numerical_status=row["numerical_status"]))
    recovery = json.loads((output / "recovery_fits.json").read_text())["rows"]
    state_recovery = json.loads((output / "recovery_states.json").read_text())["rows"]
    generation = [row for role, rows in (("development", dev), ("evaluation", evaluation))
                  for row in generation_summaries(output, records, rows, register["policy"], role, [chosen, comparator])]
    result = dict(selected_variant=chosen, comparator=comparator, names=LABELS,
        uniform_choice_reference={str(c): float(np.log(2 if c == 1 else 4)) for c in (1, 2, 3)},
        development_summary=summarize(dev), refined_summary=summarize(refined), evaluation_summary=summarize(evaluation),
        evaluation_precision=evaluation_precision,
        evaluation_precision_summary=summarize(evaluation_precision["rows"]),
        paired_evaluation=paired, mean_evaluation_nll_difference=float(np.mean([r["nll_difference"] for r in paired])),
        mean_evaluation_energy_difference=float(np.mean([r["energy_difference"] for r in paired])),
        parameters=parameter_boundaries(parameters), selected_model_states=states,
        nested_search_diagnostics=nested_search_diagnostics(parameters, register["policy"]["search"]["numerical_tolerance_nll_per_trial"]),
        autonomous_distributions=generation,
        parameter_recovery=parameter_recovery_diagnostics(recovery, register["policy"]),
        recovery_classification=dict(Counter(r["classification"] for r in recovery)),
        recovery_rows=recovery, state_recovery=state_recovery, verification=verification,
        frozen_model_source_head=register["source_git_head"], audit_source_sha256=digest(Path(__file__)),
        created_unix=time.time(), interpretation="Finite-search working-model selection; fixed schedules and parameters for state summaries; numerical limitations retained")
    destination.mkdir(parents=True)
    write_new(destination / "analysis.json", result)
    write_new(destination / "numeric_audit.json", verification)
    lines = ["# Model 1031 数值核验与版本比较", "", f"所选工作版本：**{LABELS[chosen]}**；比较版本：{LABELS[comparator]}。", "",
        "开发后段参与结构选择；评价被试后段在结构冻结后使用。以下NLL均按被试等权汇总，每题值越低越好。", "",
        "评分尺度参照：逐题均匀随机选择的每题NLL为log(C)，条件1为0.693147，条件2、3为1.386294。", "",
        "|版本|名义参数数|开发后段NLL|自主Energy|训练数值受限条目|", "|---|---:|---:|---:|---:|"]
    for variant, row in result["development_summary"].items():
        lines.append(f"|{LABELS[variant]}|{register['policy']['variants'][variant]['parameters']}|{row['nll']:.6f}|{row['energy']:.6f}|{row['numerically_limited']} / {row['subjects']}|")
    lines += ["", "两个预选结构的独立高精度复核：", "", "|版本|后段NLL|保守合并的数值标准误|数值受限条目|", "|---|---:|---:|---:|"]
    for variant, row in result["refined_summary"].items():
        lines.append(f"|{LABELS[variant]}|{row['nll']:.6f}|{row['numerical_se']:.6f}|{row['numerically_limited']} / {row['subjects']}|")
    lines += ["", "结构冻结后的逐人评价。差值为所选版本减比较版本，负值支持所选版本。", "",
              "|被试|条件|所选NLL|比较NLL|NLL差|所选Energy|比较Energy|", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in paired:
        lines.append(f"|{row['subject']}|{row['condition']}|{row['selected_nll']:.6f}|{row['comparator_nll']:.6f}|{row['nll_difference']:+.6f}|{row['selected_energy']:.6f}|{row['comparator_energy']:.6f}|")
    lines += ["", f"平均NLL差：{result['mean_evaluation_nll_difference']:+.6f}；平均Energy差：{result['mean_evaluation_energy_difference']:+.6f}。",
        "", "原评价记录的数值限制标记来自训练选参检查；后段预测的独立精度检查另列。",
        "", f"核验了{verification['records']}项完整任务的数组哈希；由保存概率重算NLL的最大差为{verification['max_recomputed_nll_error']:.3g}。",
        f"任务内记录CPU合计{verification['recorded_worker_cpu_hours']:.3f}小时；进程启动、外部中断余量及总预算以监管器记录为准。", "",
        "模拟恢复分类：" + "、".join(f"{key}={value}" for key, value in result["recovery_classification"].items()) + "。", "",
        "参数点恢复逐坐标保存生成值、恢复值和误差，同时标注结构固定项及生成点上的无效坐标；该检查仍受有限搜索与数值精度限制。", "",
        "完整逐人参数、复核候选、固定参数的内容状态和两次状态恢复结果见[analysis.json](analysis.json)。",
        "这是数值核验表。模型决定的心理解释、优化与可辨识性边界由最终说明分别陈述。"]
    lines += ["", f"评价被试固定参数的额外精度检查：{evaluation_precision['status']}。"]
    if evaluation_precision["rows"]:
        lines += ["", "|版本|1024粒子后段NLL|保守数值标准误|后段数值受限条目|", "|---|---:|---:|---:|"]
        for variant, row in result["evaluation_precision_summary"].items():
            lines.append(f"|{LABELS[variant]}|{row['nll']:.6f}|{row['numerical_se']:.6f}|{row['numerically_limited']} / {row['subjects']}|")
    with (destination / "tables.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.study.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
