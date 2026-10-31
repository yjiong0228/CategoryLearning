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
from ..analysis.model_1031_finalization import digest, validate_source, write_new


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
            numerical_status=group["numerical_status"], boundary_notes=redundant,
            confirmed_alternatives=group["candidates"]))
    return rows


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
            runs = [r for r in records if r["job"]["kind"] == "fit" and r["job"]["scope"] == "full"
                    and r["job"]["subject"] == row["subject"] and r["job"]["config"] == row["candidate"]["id"]
                    and r["job"]["seed"] in range(base, base + 4)]
            states.append(dict(merge_state_records(output, runs, row["subject"]), role=role,
                               numerical_status=row["numerical_status"]))
    recovery = json.loads((output / "recovery_fits.json").read_text())["rows"]
    state_recovery = json.loads((output / "recovery_states.json").read_text())["rows"]
    result = dict(selected_variant=chosen, comparator=comparator, names=LABELS,
        development_summary=summarize(dev), refined_summary=summarize(refined), evaluation_summary=summarize(evaluation),
        evaluation_precision=evaluation_precision,
        evaluation_precision_summary=summarize(evaluation_precision["rows"]),
        paired_evaluation=paired, mean_evaluation_nll_difference=float(np.mean([r["nll_difference"] for r in paired])),
        mean_evaluation_energy_difference=float(np.mean([r["energy_difference"] for r in paired])),
        parameters=parameter_boundaries(parameters), selected_model_states=states,
        recovery_classification=dict(Counter(r["classification"] for r in recovery)),
        recovery_rows=recovery, state_recovery=state_recovery, verification=verification,
        frozen_model_source_head=register["source_git_head"], audit_source_sha256=digest(Path(__file__)),
        created_unix=time.time(), interpretation="Finite-search working-model selection; fixed schedules and parameters for state summaries; numerical limitations retained")
    destination.mkdir(parents=True)
    write_new(destination / "analysis.json", result)
    write_new(destination / "numeric_audit.json", verification)
    lines = ["# Model 1031 数值核验与版本比较", "", f"所选工作版本：**{LABELS[chosen]}**；比较版本：{LABELS[comparator]}。", "",
        "开发后段参与结构选择；评价被试后段在结构冻结后使用。以下NLL均按被试等权汇总，每题值越低越好。", "",
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
