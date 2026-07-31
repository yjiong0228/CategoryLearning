#!/usr/bin/env python3
"""Build the canonical portable-report artifact for the RT validation."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TITLE = "条件1：冻结 B0/D0 的 RT 外部机制检验"

PRIMARY_SQL = """
SELECT
    source_id,
    particle_count,
    train_n,
    test_n,
    subject_n,
    dynamic_effect_percent_per_train_sd,
    cluster_ci_lower,
    cluster_ci_upper,
    wild_cluster_p_two_sided,
    baseline_rmse_log_rt,
    dynamic_rmse_log_rt,
    mean_subject_baseline_minus_dynamic_mse,
    bootstrap_dmse_ci_lower,
    bootstrap_dmse_ci_upper,
    dynamic_subject_win_count,
    dynamic_subject_n
FROM model_comparison
WHERE source_id = 'main_r128' AND analysis_id = 'primary'
""".strip()

SUBJECT_SQL = """
SELECT
    CAST(iSub AS TEXT) AS subject,
    test_rt_n,
    b0_minus_d0_test_choice_brier AS choice_brier_gain,
    baseline_minus_dynamic_mse AS rt_mse_gain,
    CASE WHEN dynamic_wins_mse = 1 THEN 'D0特征改善' ELSE 'D0特征变差' END AS outcome
FROM subject_heldout_metrics
WHERE source_id = 'main_r128' AND analysis_id = 'primary'
ORDER BY rt_mse_gain ASC
""".strip()

SENSITIVITY_SQL = """
SELECT
    CASE analysis_id
        WHEN 'primary' THEN '主规格'
        WHEN 'all_positive_rt' THEN '保留全部正RT'
        WHEN 'subject_practice_slopes' THEN '被试特异练习斜率'
        WHEN 'total_variation' THEN '总变差距离'
        WHEN 'omit_current_outcome' THEN '不控制当前结果'
        ELSE analysis_id
    END AS specification,
    dynamic_effect_percent_per_train_sd AS rt_effect_percent,
    wild_cluster_p_two_sided,
    baseline_rmse_log_rt - dynamic_rmse_log_rt AS heldout_rmse_gain,
    dynamic_subject_win_count,
    dynamic_subject_n,
    bootstrap_dmse_ci_lower,
    bootstrap_dmse_ci_upper
FROM model_comparison
WHERE source_id = 'main_r128'
ORDER BY
    CASE analysis_id
        WHEN 'primary' THEN 1
        WHEN 'all_positive_rt' THEN 2
        WHEN 'subject_practice_slopes' THEN 3
        WHEN 'total_variation' THEN 4
        WHEN 'omit_current_outcome' THEN 5
        ELSE 99
    END
""".strip()

STABILITY_SQL = """
SELECT
    CASE comparison
        WHEN 'particle_count' THEN '64 vs 128 particles'
        WHEN 'independent_seed' THEN 'independent PF seed'
        ELSE comparison
    END AS comparison,
    trial_n,
    jsd_pearson_r,
    jsd_spearman_rho,
    jsd_mean_absolute_difference
FROM seed_particle_stability
ORDER BY comparison
""".strip()

RECOVERY_SQL = """
SELECT
    CASE noise_model
        WHEN 'gaussian' THEN 'Gaussian'
        WHEN 'empirical_residual' THEN '经验残差重采样'
        ELSE noise_model
    END AS noise_model,
    CASE effect_id
        WHEN 'null' THEN '零效应'
        WHEN 'cost_10pct' THEN '每SD慢10%'
        ELSE effect_id
    END AS injected_effect,
    true_percent_effect_per_jsd_sd,
    cluster_95pct_coverage,
    inference_pass_rate,
    heldout_improvement_rate,
    full_gate_rate
FROM conditional_rt_recovery
WHERE effect_id IN ('null', 'cost_10pct')
ORDER BY effect_id, noise_model
""".strip()

SCOPE_SQL = """
SELECT
    data_rows,
    condition1_rows,
    target8_rows,
    heldout7_rows,
    aligned_choice_mismatches,
    qc_excluded_target8,
    target8_rt_median_seconds,
    target8_rt_p95_seconds,
    target8_rt_max_seconds
FROM scope_summary
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_newplan/rt_validation_theta075"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_newplan/rt_validation_theta075/report/artifact.json"
        ),
    )
    return parser.parse_args()


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def source(
    source_id: str,
    label: str,
    sql: str,
    description: str,
    *,
    tables_used: list[str],
    filters: list[str],
    definitions: list[str],
    generated_at: str,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": (
            "results/zhuran/cond1_newplan/rt_validation_theta075/"
            "report/report_queries.sql"
        ),
        "query": {
            "engine": "sqlite",
            "sql": sql,
            "description": description,
            "language": "SQL",
            "executed_at": generated_at,
            "tables_used": tables_used,
            "filters": filters,
            "metric_definitions": definitions,
        },
    }


def main() -> None:
    args = parse_args()
    comparison = pd.read_csv(args.results_dir / "model_comparison.csv")
    subjects = pd.read_csv(args.results_dir / "subject_heldout_metrics.csv")
    stability = pd.read_csv(args.results_dir / "seed_particle_stability.csv")
    recovery = pd.read_csv(args.results_dir / "conditional_rt_recovery.csv")
    data_quality = json.loads(
        (args.results_dir / "data_quality.json").read_text(encoding="utf-8")
    )
    decision = json.loads(
        (args.results_dir / "decision.json").read_text(encoding="utf-8")
    )
    cross_modal = json.loads(
        (args.results_dir / "cross_modal_summary.json").read_text(
            encoding="utf-8"
        )
    )
    generated_at = datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")

    scope = pd.DataFrame(
        [
            {
                "data_rows": data_quality["row_n"],
                "condition1_rows": data_quality["condition_1"]["n"],
                "target8_rows": data_quality["target_8_subjects"]["n"],
                "heldout7_rows": data_quality["heldout_7_subjects"]["n"],
                "aligned_choice_mismatches": data_quality[
                    "aligned_choice_mismatch_n"
                ],
                "qc_excluded_target8": data_quality[
                    "main_r128_qc_excluded_n"
                ],
                "target8_rt_median_seconds": data_quality[
                    "target_8_subjects"
                ]["median_seconds"],
                "target8_rt_p95_seconds": data_quality[
                    "target_8_subjects"
                ]["p95_seconds"],
                "target8_rt_max_seconds": data_quality[
                    "target_8_subjects"
                ]["max_seconds"],
            }
        ]
    )
    connection = sqlite3.connect(":memory:")
    comparison.to_sql(
        "model_comparison", connection, index=False, if_exists="replace"
    )
    subjects.to_sql(
        "subject_heldout_metrics",
        connection,
        index=False,
        if_exists="replace",
    )
    stability.to_sql(
        "seed_particle_stability",
        connection,
        index=False,
        if_exists="replace",
    )
    recovery.to_sql(
        "conditional_rt_recovery",
        connection,
        index=False,
        if_exists="replace",
    )
    scope.to_sql(
        "scope_summary", connection, index=False, if_exists="replace"
    )
    datasets = {
        "primary_summary": records(
            pd.read_sql_query(PRIMARY_SQL, connection)
        ),
        "subject_effects": records(
            pd.read_sql_query(SUBJECT_SQL, connection)
        ),
        "sensitivity_results": records(
            pd.read_sql_query(SENSITIVITY_SQL, connection)
        ),
        "stability_results": records(
            pd.read_sql_query(STABILITY_SQL, connection)
        ),
        "recovery_results": records(
            pd.read_sql_query(RECOVERY_SQL, connection)
        ),
        "scope_summary": records(
            pd.read_sql_query(SCOPE_SQL, connection)
        ),
    }
    connection.close()

    primary = datasets["primary_summary"][0]
    source_objects = [
        source(
            "primary_model",
            "主规格 held-out 模型比较",
            PRIMARY_SQL,
            "读取冻结 B0/D0 的主规格 log-RT 比较结果。",
            tables_used=["model_comparison"],
            filters=["source_id = main_r128", "analysis_id = primary"],
            definitions=[
                "dynamic_effect_percent_per_train_sd = exp(beta_JSD)-1，以百分比表示。",
                "bootstrap_dmse_ci = 7名被试等权重采样的 baseline MSE - dynamic MSE 区间。",
            ],
            generated_at=generated_at,
        ),
        source(
            "subject_model",
            "逐被试 held-out RT 与选择收益",
            SUBJECT_SQL,
            "读取 7 名具有末块 held-out 的被试级动态增量结果。",
            tables_used=["subject_heldout_metrics"],
            filters=["source_id = main_r128", "analysis_id = primary"],
            definitions=[
                "rt_mse_gain = baseline held-out log-RT MSE - dynamic held-out log-RT MSE；正值支持动态增量。",
                "choice_brier_gain = B0 held-out Brier - D0 held-out Brier；正值支持固定 theta=0.75 的 D0。",
            ],
            generated_at=generated_at,
        ),
        source(
            "sensitivity_model",
            "RT 规格敏感性分析",
            SENSITIVITY_SQL,
            "读取主 PF 运行下五种 RT 模型规格的效应和 held-out 指标。",
            tables_used=["model_comparison"],
            filters=["source_id = main_r128"],
            definitions=[
                "rt_effect_percent = 冻结 B0-D0 分歧每增加一个训练集标准差时的条件 RT 百分比变化。",
                "heldout_rmse_gain = baseline RMSE - dynamic RMSE；正值支持动态增量。",
            ],
            generated_at=generated_at,
        ),
        source(
            "stability_model",
            "粒子数与随机种子稳定性",
            STABILITY_SQL,
            "读取逐试次 JSD 特征在粒子数和独立种子间的一致性。",
            tables_used=["seed_particle_stability"],
            filters=["all aligned valid trials from the eight target subjects"],
            definitions=[
                "jsd_pearson_r = 两次 PF 运行逐试次 B0-D0 Jensen-Shannon 分歧的 Pearson 相关。",
            ],
            generated_at=generated_at,
        ),
        source(
            "recovery_model",
            "条件 RT emission 恢复",
            RECOVERY_SQL,
            "读取零效应和每标准差 10% RT 成本的条件模拟恢复结果。",
            tables_used=["conditional_rt_recovery"],
            filters=["effect_id in (null, cost_10pct)"],
            definitions=[
                "full_gate_rate 同时要求方向与聚类推断正确、总体 held-out 改善且至少 4/7 被试改善。",
                "此恢复以冻结的观测设计矩阵为条件，不是完整 choice-state-RT 联合恢复。",
            ],
            generated_at=generated_at,
        ),
        source(
            "scope_model",
            "RT 数据质量与分析范围",
            SCOPE_SQL,
            "读取 Task2 choRT 的范围、对齐和稳健 QC 汇总。",
            tables_used=["scope_summary"],
            filters=[
                "condition = 1 for target analyses",
                "target subjects = 103,105,111,112,117,118,127,131",
            ],
            definitions=[
                "QC = 相对各被试训练集 log-RT 中位数的绝对偏差不超过 4 个 Gaussian-consistent MAD。",
                "每块首试次因 lag RT 未定义而不进入 RT 回归。",
            ],
            generated_at=generated_at,
        ),
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": (
                "使用 choRT 对冻结的 condition-1 B0/D0 选择预测进行独立机制检验。"
            ),
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "subject_rt_gain",
                    "title": "逐被试 held-out RT MSE 增量",
                    "subtitle": (
                        "末块 held-out；正值表示加入冻结 B0-D0 分歧后预测改善，n=7"
                    ),
                    "type": "horizontalBar",
                    "dataset": "subject_effects",
                    "sourceId": "subject_model",
                    "intent": "comparison",
                    "question": "动态分歧特征是否在多数被试的未见 RT 中改善预测？",
                    "rationale": "7 个有标签的同单位效应适合以零线水平条形图比较。",
                    "comparisonContext": {
                        "baseline": "不含 B0-D0 分歧的 log-RT 基线",
                        "grain": "subject",
                        "unit": "log-RT MSE difference",
                    },
                    "encodings": {
                        "x": {
                            "field": "subject",
                            "type": "nominal",
                            "label": "被试",
                        },
                        "y": {
                            "field": "rt_mse_gain",
                            "type": "quantitative",
                            "label": "Baseline − dynamic MSE",
                        },
                        "tooltip": [
                            {
                                "field": "choice_brier_gain",
                                "type": "quantitative",
                                "label": "D0 选择 Brier 收益",
                            },
                            {
                                "field": "test_rt_n",
                                "type": "quantitative",
                                "label": "held-out RT n",
                            },
                        ],
                    },
                    "referenceLines": [
                        {
                            "axis": "y",
                            "value": 0,
                            "label": "无增量收益",
                            "color": "neutral",
                        }
                    ],
                    "palette": {"kind": "diverging", "midpoint": 0},
                    "labels": {"values": "all"},
                    "settings": {
                        "orientation": "horizontal",
                        "sort": "ascending",
                        "showValues": True,
                    },
                    "layout": "full",
                    "surface": {
                        "surface": "explorer",
                        "viewMode": "both",
                    },
                },
                {
                    "id": "sensitivity_effect",
                    "title": "不同 RT 规格下的 B0-D0 分歧效应",
                    "subtitle": (
                        "每增加一个训练集 SD 的分歧所对应的条件 RT 百分比变化"
                    ),
                    "type": "horizontalBar",
                    "dataset": "sensitivity_results",
                    "sourceId": "sensitivity_model",
                    "intent": "comparison",
                    "question": "负向效应是否依赖单一 RT 预处理或基线规格？",
                    "rationale": "五个离散规格和同一百分比单位适合水平条形比较。",
                    "comparisonContext": {
                        "baseline": "0% RT change",
                        "grain": "analysis specification",
                        "unit": "%",
                    },
                    "encodings": {
                        "x": {
                            "field": "specification",
                            "type": "nominal",
                            "label": "RT 规格",
                        },
                        "y": {
                            "field": "rt_effect_percent",
                            "type": "quantitative",
                            "label": "RT 变化",
                            "unit": "%",
                        },
                        "tooltip": [
                            {
                                "field": "wild_cluster_p_two_sided",
                                "type": "quantitative",
                                "label": "wild-cluster p",
                            },
                            {
                                "field": "heldout_rmse_gain",
                                "type": "quantitative",
                                "label": "held-out RMSE 收益",
                            },
                            {
                                "field": "dynamic_subject_win_count",
                                "type": "quantitative",
                                "label": "改善被试数",
                            },
                        ],
                    },
                    "referenceLines": [
                        {
                            "axis": "y",
                            "value": 0,
                            "label": "无条件效应",
                            "color": "neutral",
                        }
                    ],
                    "palette": {"kind": "sequential", "name": "blue"},
                    "labels": {"values": "all"},
                    "settings": {
                        "orientation": "horizontal",
                        "sort": "ascending",
                        "showValues": True,
                    },
                    "layout": "full",
                    "surface": {
                        "surface": "explorer",
                        "viewMode": "both",
                    },
                },
            ],
            "tables": [
                {
                    "id": "stability_table",
                    "title": "PF 数值稳定性",
                    "subtitle": "所有对齐有效试次上的 B0-D0 JSD 一致性",
                    "dataset": "stability_results",
                    "sourceId": "stability_model",
                    "defaultSort": {
                        "field": "jsd_pearson_r",
                        "direction": "desc",
                    },
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {
                            "field": "comparison",
                            "label": "比较",
                            "type": "text",
                        },
                        {
                            "field": "trial_n",
                            "label": "试次 n",
                            "format": "number",
                        },
                        {
                            "field": "jsd_pearson_r",
                            "label": "Pearson r",
                            "format": "number",
                        },
                        {
                            "field": "jsd_spearman_rho",
                            "label": "Spearman rho",
                            "format": "number",
                        },
                        {
                            "field": "jsd_mean_absolute_difference",
                            "label": "JSD 平均绝对差",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "recovery_table",
                    "title": "条件 RT emission 恢复与零效应校准",
                    "subtitle": "2,000 次模拟；冻结实际设计矩阵",
                    "dataset": "recovery_results",
                    "sourceId": "recovery_model",
                    "defaultSort": {
                        "field": "full_gate_rate",
                        "direction": "desc",
                    },
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {
                            "field": "noise_model",
                            "label": "噪声模型",
                            "type": "text",
                        },
                        {
                            "field": "injected_effect",
                            "label": "注入效应",
                            "type": "text",
                        },
                        {
                            "field": "cluster_95pct_coverage",
                            "label": "95% CI coverage",
                            "format": "percent",
                        },
                        {
                            "field": "inference_pass_rate",
                            "label": "推断通过率",
                            "format": "percent",
                        },
                        {
                            "field": "heldout_improvement_rate",
                            "label": "held-out 改善率",
                            "format": "percent",
                        },
                        {
                            "field": "full_gate_rate",
                            "label": "完整门槛率",
                            "format": "percent",
                        },
                    ],
                },
            ],
            "sources": [
                {
                    "id": item["id"],
                    "label": item["label"],
                    "path": item["path"],
                }
                for item in source_objects
            ],
            "blocks": [
                {
                    "id": "report_title",
                    "type": "markdown",
                    "body": f"# {TITLE}",
                    "layout": "full",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "primary_model",
                    "body": (
                        "## 技术结论：RT 不足以救回 D0\n\n"
                        f"**确认性结论仍是保留 B0、停止推进 D0。** 冻结的 B0–D0 "
                        f"分歧每增加一个训练集标准差，条件 RT 约变化 "
                        f"{primary['dynamic_effect_percent_per_train_sd']:.2f}%；"
                        "方向是变快而非预先可解释的重构成本。虽然总体 held-out "
                        f"log-RT RMSE 从 {primary['baseline_rmse_log_rt']:.4f} "
                        f"降至 {primary['dynamic_rmse_log_rt']:.4f}，但仅 "
                        f"{int(primary['dynamic_subject_win_count'])}/"
                        f"{int(primary['dynamic_subject_n'])} 名被试改善，"
                        f"等权被试 MSE 收益的 95% bootstrap 区间为 "
                        f"[{primary['bootstrap_dmse_ci_lower']:.4f}, "
                        f"{primary['bootstrap_dmse_ci_upper']:.4f}]。\n\n"
                        "这构成一个探索性的双侧 RT 关联，不构成动态重构机制验证。"
                        "RT 从未用于选择 B0、D0 或 theta。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "subject_heading",
                    "type": "markdown",
                    "sourceId": "subject_model",
                    "body": (
                        "## 总体改善由少数被试驱动\n\n"
                        "图中正值才表示动态分歧特征在末块未见 RT 上优于基线。"
                        "111、118 和 127 的收益明显，而 103 与 131 明显变差；"
                        "因此总体 RMSE 的下降不能外推成共享机制。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "subject_chart_block",
                    "type": "chart",
                    "chartId": "subject_rt_gain",
                    "layout": "full",
                },
                {
                    "id": "cross_modal_note",
                    "type": "markdown",
                    "sourceId": "subject_model",
                    "body": (
                        "### 选择收益与 RT 收益没有形成可靠的跨模态收敛\n\n"
                        f"7 名被试的 Pearson r={cross_modal['pearson_r']:.3f}"
                        f"（p={cross_modal['pearson_p']:.3f}），Spearman "
                        f"rho={cross_modal['spearman_rho']:.3f}"
                        f"（p={cross_modal['spearman_p']:.3f}）。样本很小，"
                        "该结果只能说明目前没有足够证据把两种收益视为同一潜在动态机制。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "scope_heading",
                    "type": "markdown",
                    "sourceId": "scope_model",
                    "body": (
                        "## 数据范围与指标定义\n\n"
                        "Task2 共 62,736 行；条件1有 10,064 行。目标 8 名被试有 "
                        "1,856 个 RT，7 名具有末块 held-out 的被试有 1,792 个 RT。"
                        "choRT 无缺失、非有限值或非正值，逐试次选择与 PF 缓存对齐错误为 0。"
                        "主 RT 回归排除每块首试次，并用各被试训练集 log-RT 中位数 ± "
                        "4 个 Gaussian-consistent MAD 做稳健 QC；不做 QC 的结果另作敏感性分析。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "method_heading",
                    "type": "markdown",
                    "sourceId": "primary_model",
                    "body": (
                        "## 设计保证 RT 是外部检验，而不是再次选模\n\n"
                        "D0 固定 theta=0.75，来自先前仅使用选择训练集的共享强度结果。"
                        "主基线以 log RT 为因变量，包含被试固定截距、lag log RT、"
                        "上一试次错误、当前反应错误、刺激歧义、B0 预测熵、被试内试次位置"
                        "和块内位置；动态模型只额外加入预选择时刻的 B0–D0 "
                        "Jensen–Shannon 分歧。系数不确定性按被试聚类，并用 7 个 cluster "
                        "的 128 种符号翻转穷举 wild-cluster 检查；预测比较完全使用各被试最后一块。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "sensitivity_heading",
                    "type": "markdown",
                    "sourceId": "sensitivity_model",
                    "body": (
                        "## 负向关联重复出现，但显著性和跨被试泛化不稳\n\n"
                        "五种规格的点估计均为负；然而 wild-cluster 双侧证据会随 QC、"
                        "被试特异练习斜率、分歧度量和当前结果控制而变化。更重要的是，"
                        "各规格都没有达到至少 5/7 被试改善且 bootstrap 下界大于 0 的 held-out 门槛。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "sensitivity_chart_block",
                    "type": "chart",
                    "chartId": "sensitivity_effect",
                    "layout": "full",
                },
                {
                    "id": "stability_heading",
                    "type": "markdown",
                    "sourceId": "stability_model",
                    "body": (
                        "## Monte Carlo 误差不是唯一问题，独立种子仍未过严格门槛\n\n"
                        "64 与 128 粒子的逐试次分歧 Pearson r=0.902；独立 PF 种子为 "
                        "r=0.882，低于预设的 0.90 门槛。三次运行的 RT 系数方向和总体 "
                        "RMSE 方向一致，说明结果具有相当程度的数值重复性，但还不足以把"
                        "逐试次动态量视为完全稳定。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "stability_table_block",
                    "type": "table",
                    "tableId": "stability_table",
                    "layout": "full",
                },
                {
                    "id": "recovery_heading",
                    "type": "markdown",
                    "sourceId": "recovery_model",
                    "body": (
                        "## 10% RT 效应本来可以被当前设计检出\n\n"
                        "在 Gaussian 与经验残差重采样两种噪声下，注入每标准差 10% "
                        "RT 成本时完整门槛通过率分别为 97.3% 和 96.0%；零效应误报率"
                        "分别为 0.75% 和 0.60%。因此现实结果没有通过机制门槛，不能主要"
                        "归咎于当前条件 RT emission 完全不可估计。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "recovery_table_block",
                    "type": "table",
                    "tableId": "recovery_table",
                    "layout": "full",
                },
                {
                    "id": "limitations_heading",
                    "type": "markdown",
                    "body": (
                        "## 结果能说明什么、不能说明什么\n\n"
                        "D0 选择模型本身没有定义有符号 RT emission，所以负向系数不能"
                        "单独形式化证伪 D0；它同样不能被事后解释为“切换更高效”。"
                        "条件恢复固定了观测设计矩阵和 lag RT，只验证附加 RT 系数的可估计性，"
                        "不是完整的 choice–latent-state–RT 联合恢复。被试数只有 7，任何"
                        "被试亚型解释都必须保持探索性。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 建议：在 B0 停住，不加入 RT emission\n\n"
                        "- 条件1确认模型继续冻结为 B0；D0 仅作探索性预测基准。\n"
                        "- 不因总体 RMSE 小幅下降而重开容量、三状态控制器或更复杂转移。\n"
                        "- 若未来建立联合 choice–RT 模型，必须先规定 RT 映射的方向和函数形式，"
                        "并增加足够的 held-out 被试来估计异质斜率。\n"
                        "- 现阶段可把 111、118、127 的模式作为新数据收集假设，不能用本批 RT "
                        "反向给被试分类或选择 D0。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 后续真正需要回答的问题\n\n"
                        "1. 动态重构理论到底预言 RT 成本、RT 加速，还是只预言方差变化？\n"
                        "2. 这种 RT 映射能否在新被试上预先指定并恢复，而不是由本数据决定方向？\n"
                        "3. 若允许异质斜率，选择与 RT 是否能共同识别一个可重复的少数动态群体？"
                    ),
                    "layout": "full",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": source_objects,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(args.output)
    print(decision["decision"])


if __name__ == "__main__":
    main()
