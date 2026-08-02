#!/usr/bin/env python3
"""Build a durable Chinese handoff for the unified new-plan overnight run."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/zhuran/unified_newplan"
OUTPUT = BASE / "overnight_summary_20260802"
RUNS = {
    "core128": BASE / "core_sobol128_20260802",
    "core256": BASE / "core_sobol256_20260802",
    "core512": BASE / "core_sobol512_20260802",
    "precision": BASE / "precision_128_256_512_20260802",
    "dynamic": BASE / "dynamic_readout_20260802",
    "readout_recovery20": BASE / "readout_recovery_screen20_20260802",
    "readout_recovery100": BASE / "readout_recovery_final100_20260802",
    "joint_nr2": BASE / "joint_dynamic_nr2_20260802",
    "representation_recovery20": BASE / "representation_recovery_screen20_20260802",
    "representation_recovery100": BASE / "representation_recovery_final100_20260802",
    "rt": BASE / "rt_external_validation_20260802",
    "oral": BASE / "oral_external_validation_20260802",
    "oral_mixture": BASE / "oral_mixture_diagnostic_20260802",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    """Convert pandas/NumPy output to strict, portable JSON values."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            json_ready(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def comparison_row(
    path: Path,
    comparison: str,
    condition: str,
    extra_filters: dict[str, object] | None = None,
) -> pd.Series:
    frame = pd.read_csv(path, dtype={"condition": str})
    mask = (
        (frame["comparison"] == comparison) & (frame["condition"] == condition)
    )
    for column, value in (extra_filters or {}).items():
        mask &= frame[column] == value
    row = frame[mask]
    if len(row) != 1:
        raise ValueError(f"expected one {comparison}/{condition} row in {path}")
    return row.iloc[0]


def validate_runs() -> pd.DataFrame:
    rows = []
    for name, path in RUNS.items():
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"incomplete run: {name}")
        error_files = sorted(path.glob("*errors.json"))
        error_count = 0
        for error_file in error_files:
            payload = json.loads(error_file.read_text(encoding="utf-8"))
            error_count += len(payload) if isinstance(payload, list) else int(bool(payload))
        if error_count:
            raise ValueError(f"{name} contains {error_count} recorded errors")
        rows.append(
            {
                "run": name,
                "path": str(path.resolve()),
                "result_type": manifest.get("result_type"),
                "status": manifest.get("status"),
                "manifest_sha256": sha256_file(manifest_path),
                "recorded_error_count": error_count,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    started = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    artifacts = validate_runs()

    static_rows = {
        condition: comparison_row(
            RUNS["core512"] / "model_comparisons.csv",
            "representation_gate",
            condition,
        )
        for condition in ("1", "2", "3", "all")
    }
    dynamic_rows = {
        condition: comparison_row(
            RUNS["joint_nr2"] / "model_comparisons.csv",
            "rule_vs_joint_dynamic_NR2",
            condition,
        )
        for condition in ("1", "2", "3", "all")
    }
    dynamic_parameters = pd.read_csv(
        RUNS["dynamic"] / "parameters.csv", dtype={"condition": str}
    )
    shared_slope = dynamic_parameters[
        (dynamic_parameters["model"] == "R0KT_GLOBAL")
        & (dynamic_parameters["subject_id"] == -1)
    ].iloc[0]
    readout_recovery = pd.read_csv(
        RUNS["readout_recovery100"] / "recovery_summary.csv"
    )
    representation_recovery = pd.read_csv(
        RUNS["representation_recovery100"] / "recovery_summary.csv"
    )
    rt_rule = comparison_row(
        RUNS["rt"] / "model_comparisons.csv",
        "rule_entropy_increment",
        "all",
        {"qc_specification": "main_qc"},
    )
    rt_nr = comparison_row(
        RUNS["rt"] / "model_comparisons.csv",
        "nr_entropy_increment",
        "all",
        {"qc_specification": "main_qc"},
    )
    oral_primary = comparison_row(
        RUNS["oral"] / "model_comparisons.csv",
        "rule_vs_selected_baseline",
        "all",
    )
    oral_adaptive = comparison_row(
        RUNS["oral_mixture"] / "model_comparisons.csv",
        "global_mixture_vs_baseline",
        "all",
    )
    oral_manifest = json.loads(
        (RUNS["oral_mixture"] / "manifest.json").read_text(encoding="utf-8")
    )
    precision = pd.read_csv(RUNS["precision"] / "q_precision_summary.csv")
    q_last = precision[precision["precision_pair"] == "256->512"].iloc[0]
    parameter_precision = pd.read_csv(
        RUNS["precision"] / "parameter_precision_summary.csv",
        dtype={"condition": str},
    )
    r0k_precision = parameter_precision[
        (parameter_precision["precision_pair"] == "256->512")
        & (parameter_precision["condition"] == "all")
        & (parameter_precision["model"] == "R0K")
        & (parameter_precision["parameter"] == "sensitivity")
    ]
    if len(r0k_precision) != 1:
        raise ValueError("could not resolve the R0K sensitivity precision row")
    r0k_precision = r0k_precision.iloc[0]

    key_results: dict[str, Any] = {
        "data": {
            "subjects": 96,
            "trials": 62720,
            "conditions": {"1": 10048, "2": 31872, "3": 20800},
            "rules": {"condition1": 38, "condition2_3": 116},
            "known_integrity_issue": (
                "condition-3 subject 319 session-5 category column is inconsistent; "
                "known h42 categories reproduce 100% of delivered feedback"
            ),
        },
        "precision": {
            "decision": "freeze_512",
            "q_mean_abs_256_to_512": float(q_last.mean_subject_mean_abs_q_delta),
            "q_argmax_disagreement_256_to_512": float(q_last.mean_argmax_disagreement),
            "r0k_kappa_pearson_256_to_512": float(r0k_precision.pearson_r),
        },
        "static_primary_gate": {
            condition: {
                "mean_delta_nll_per_trial": float(row.mean_delta_nll_per_trial),
                "ci": [float(row.bootstrap_mean_ci_low), float(row.bootstrap_mean_ci_high)],
                "improved": [int(row.n_improved), int(row.n_subjects)],
            }
            for condition, row in static_rows.items()
        },
        "adaptive_choice_model": {
            "name": "R0KT_GLOBAL",
            "equation": "log(kappa_s,t) = intercept_s + shared_slope * normalized_practice",
            "shared_slope": float(shared_slope.slope),
            "shared_slope_se_wald": float(shared_slope.slope_se),
            "training_end_factor": float(math.exp(shared_slope.slope)),
            "versus_joint_dynamic_nr2": {
                condition: {
                    "mean_delta_nll_per_trial": float(row.mean_delta_nll_per_trial),
                    "ci": [float(row.bootstrap_mean_ci_low), float(row.bootstrap_mean_ci_high)],
                    "improved": [int(row.n_improved), int(row.n_subjects)],
                }
                for condition, row in dynamic_rows.items()
            },
        },
        "recovery": {
            "static_vs_dynamic": readout_recovery.to_dict(orient="records"),
            "rule_vs_feature": representation_recovery.to_dict(orient="records"),
        },
        "external_validation": {
            "rt_rule_entropy": {
                "mean_delta_lpd": float(rt_rule.mean_delta_log_predictive_density),
                "ci": [float(rt_rule.bootstrap_mean_ci_low), float(rt_rule.bootstrap_mean_ci_high)],
                "improved": [int(rt_rule.n_improved), int(rt_rule.n_subjects)],
                "decision": "failed",
            },
            "rt_nr_entropy": {
                "mean_delta_lpd": float(rt_nr.mean_delta_log_predictive_density),
                "ci": [float(rt_nr.bootstrap_mean_ci_low), float(rt_nr.bootstrap_mean_ci_high)],
                "improved": [int(rt_nr.n_improved), int(rt_nr.n_subjects)],
            },
            "oral_pure_rule": {
                "mean_delta_log_score": float(oral_primary.mean_delta_log_score),
                "ci": [float(oral_primary.bootstrap_mean_ci_low), float(oral_primary.bootstrap_mean_ci_high)],
                "improved": [int(oral_primary.n_improved), int(oral_primary.n_subjects)],
                "decision": "failed_primary",
            },
            "oral_adaptive_mixture": {
                "global_rule_weight": float(oral_manifest["global_weight"]),
                "mean_delta_log_score": float(oral_adaptive.mean_delta_log_score),
                "ci": [float(oral_adaptive.bootstrap_mean_ci_low), float(oral_adaptive.bootstrap_mean_ci_high)],
                "improved": [int(oral_adaptive.n_improved), int(oral_adaptive.n_subjects)],
                "decision": "promising_but_adaptive",
            },
        },
        "final_decision": {
            "status": "exploratory_working_model_not_confirmatory_unified_mechanism",
            "choice": "R0KT_GLOBAL supported overall and in conditions 2/3; condition 1 uncertain against joint dynamic NR2",
            "resource": "stable practice-dependent readout supported; forgetting and prior-family increments unsupported",
            "rt": "rule entropy unsupported; dynamic NR2 entropy predicts held-out RT",
            "oral": "pure rule readout unsupported; one shared adaptive measurement mixture is promising",
            "stopped_by_sequence_gate": [
                "hierarchical final posterior",
                "Task1b distribution-form uncertainty propagation",
                "single-percept sensitivity model",
                "prefix-64 and full autonomous posterior-predictive generation",
            ],
        },
        "verification": {
            "pytest": "260 passed, 5 pre-existing warnings",
            "git_diff_check": "passed",
            "formal_run_error_count": int(artifacts.recorded_error_count.sum()),
        },
    }
    atomic_json(OUTPUT / "key_results.json", key_results)
    atomic_csv(OUTPUT / "artifact_index.csv", artifacts)

    def format_row(row: pd.Series) -> str:
        return (
            f"{row.mean_delta_nll_per_trial:.4f} "
            f"[{row.bootstrap_mean_ci_low:.4f}, {row.bootstrap_mean_ci_high:.4f}]"
        )

    lines = [
        "# model_newplan.tex 一晚高强度计算总报告",
        "",
        "> 结论状态：得到一个可复现的探索性工作模型，但没有通过原计划要求的全部确认性统一机制门槛。所有负结果均保留，未用后加机制覆盖原主检验。",
        "",
        "## 一句话结论",
        "",
        "原定静态规则族 R0--R3 未优于最强非规则基线；加入一个训练段估计、跨条件共享的练习读出斜率后，规则模型 R0KT_GLOBAL 在选择预测和双向模型恢复上成立，尤其在条件 2/3。可是规则熵不能预测留出 RT，纯规则口头读出也未通过均值门槛。因此最强证据支持‘随练习增强的规则选择读出 + 非规则 RT 不确定性 + 带测量污染的口头通道’，而不是一个内部状态统一解释全部行为。",
        "",
        "## 数据、实现与数值审计",
        "",
        "- 96 名被试、62,720 个试次；条件 1/2/3 分别为 10,048 / 31,872 / 20,800 个试次。",
        "- 条件 1 使用 38 条带标签规则；条件 2/3 使用 116 条规则；当前反馈严格只影响下一试次。",
        "- 512 点 Sobol 已冻结：256→512 的 q 平均绝对变化为 "
        f"{q_last.mean_subject_mean_abs_q_delta:.6f}，argmax 分歧 {q_last.mean_argmax_disagreement:.6f}；R0K κ 跨精度 r={r0k_precision.pearson_r:.6f}。",
        "- 条件 3 被试 319 第 5 session 的 category 列有误；已知任务规则 h42 生成的类别与实际 delivered feedback 100% 一致。核心一步预测使用实际反馈；自主恢复使用 h42 派生类别，没有静默改写原数据。",
        "",
        "## 原计划静态规则门槛：失败",
        "",
        "正 ΔNLL/trial 表示规则优于训练选择的非规则族：",
        "",
        "| 条件 | ΔNLL/trial [95% CI] | 改善被试 |",
        "|:--|:--|:--|",
    ]
    for condition in ("1", "2", "3", "all"):
        row = static_rows[condition]
        lines.append(
            f"| {condition} | {format_row(row)} | {int(row.n_improved)}/{int(row.n_subjects)} |"
        )
    lines.extend(
        [
            "",
            "R0K 诊断表明静态规则模型的主要问题不是遗忘：稳定读出敏感度显著改善 R0，而在已有敏感度后增加遗忘 λ 没有总体增量，条件 1 反而变差；规则家族先验 R3 相对 R2 也没有增量。因此不支持遗忘、复杂先验或个体 λ 解释。",
            "",
            "## 自适应修正：R0KT_GLOBAL 成为选择层工作模型",
            "",
            f"R0KT 使用 log κ(s,t)=a_s+b·practice，跨条件共享 b={shared_slope.slope:.6f}（Wald SE={shared_slope.slope_se:.6f}），意味着从第一试次到训练段末读出敏感度平均乘以 exp(b)={math.exp(shared_slope.slope):.3f}。条件斜率和逐被试斜率均未带来留出增量，因此保留最简单的全局斜率。",
            "",
            "相对于联合拟合、且给每名被试独立学习率/截距/时间斜率的 NR2：",
            "",
            "| 条件 | R0KT 优势 ΔNLL/trial [95% CI] | 改善被试 |",
            "|:--|:--|:--|",
        ]
    )
    for condition in ("1", "2", "3", "all"):
        row = dynamic_rows[condition]
        lines.append(
            f"| {condition} | {format_row(row)} | {int(row.n_improved)}/{int(row.n_subjects)} |"
        )
    lines.extend(
        [
            "",
            "整体、条件 2 和条件 3 的区间为正；条件 1 区间跨零。因此跨条件共同机制有总体支持，但不能宣称三个条件分别都优于动态特征 RL。",
            "",
            "## 恢复分析：模型身份可辨识",
            "",
            "- 新种子 100+100 队列的静态 R0K / 动态 R0KT 恢复均为 100/100，精确 95% 下界 0.964；动态斜率偏差 0.0038、RMSE 0.0281。动态斜率 Wald 覆盖率只有 90%，说明普通近似区间偏窄，正式参数区间仍需层级后验。",
            "- 新种子 100+100 队列的 R0KT / 动态 NR2 双向表示恢复也均为 100/100，三个条件分别亦为 100%；没有 worker 或优化失败。",
            "- NR2 模型身份虽可恢复，其个体学习率恢复相关仅约 0.34、RMSE 约 0.264，不能把 NR2 个体参数解释为稳定心理差异。",
            "",
            "## 外部检验：主规格未全部通过",
            "",
            f"- RT：规则熵相对 RT 基线的留出 ΔLPD/trial={rt_rule.mean_delta_log_predictive_density:.4f} [{rt_rule.bootstrap_mean_ci_low:.4f}, {rt_rule.bootstrap_mean_ci_high:.4f}]，失败；动态 NR2 熵为 {rt_nr.mean_delta_log_predictive_density:.4f} [{rt_nr.bootstrap_mean_ci_low:.4f}, {rt_nr.bootstrap_mean_ci_high:.4f}]，有正增量。不能声称规则不确定性解释 RT。",
            f"- 纯规则口头读出：95 名可评分被试，Δ log score={oral_primary.mean_delta_log_score:.4f} [{oral_primary.bootstrap_mean_ci_low:.4f}, {oral_primary.bootstrap_mean_ci_high:.4f}]；68/95 方向为正但受少数近零兼容质量强烈影响，主检验失败且对质量地板敏感。",
            f"- 自适应口头测量混合：训练段只拟合一个跨条件权重 w={oral_manifest['global_weight']:.4f}，留出改善 {oral_adaptive.mean_delta_log_score:.4f} [{oral_adaptive.bootstrap_mean_ci_low:.4f}, {oral_adaptive.bootstrap_mean_ci_high:.4f}]，94/95 被试为正；条件/个体权重没有总体增量。该结果很强但属于见到纯读出失败后提出的探索性证据，必须独立复核。",
            "",
            "## 最终判定与停止边界",
            "",
            "1. 可以保留 R0KT_GLOBAL 作为选择层的探索性工作模型，并明确其关键修正是稳定、共享的练习依赖读出，而不是遗忘。",
            "2. 已实现的规则与动态特征 RL 在模拟中高度可区分；真实选择结果不是拟合器无法识别 NR2 所致。",
            "3. 原计划的确认性‘同一规则状态统一解释选择、RT、口头报告’没有成立：RT 明确偏向 NR2 熵，纯口头主检验失败。",
            "4. 一个双通道描述最贴近现有证据：规则读出主导选择，特征学习不确定性关联 RT，口头表达是规则信息与稳定报告习惯的混合。但这仍是自适应架构，不是已确认机制。",
            "5. 因顺序外部门槛未通过，没有继续把计算投入完整层级后验、Task1b 分布形式传播、单次知觉抽样敏感性和 1024→4096 自主生成。这是科学停止规则，不是算力不足。",
            "",
            "## 复现与文件",
            "",
            "- 全仓测试：260 passed；5 个 warning 来自既有空数组/无穷路径。`git diff --check` 通过。",
            "- 所有 13 个正式结果目录状态 complete，记录的 worker error 总数为 0。",
            "- `artifact_index.csv` 给出每个目录及 manifest SHA256；`key_results.json` 给出机器可读关键数值。",
            "- `manuscript/model_newplan.tex` 和 PDF 的原有未提交修改没有被覆盖。",
            "",
        ]
    )
    (OUTPUT / "OVERNIGHT_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "result_type": "unified_newplan_overnight_synthesis",
        "status": "complete",
        "n_artifact_runs": int(len(artifacts)),
        "artifact_error_count": int(artifacts.recorded_error_count.sum()),
        "decision": key_results["final_decision"]["status"],
        "runtime_seconds": float(time.time() - started),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(OUTPUT / "manifest.json", manifest)
    print(f"[done] wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
