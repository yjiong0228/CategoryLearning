#!/usr/bin/env python3
"""Build the portable technical report for the condition-1 event atlas."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TITLE = "条件1异常表现事件图谱：机制诊断与止损结论"
REPORT_QUERY_PATH = (
    "results/zhuran/cond1_newplan/behavior_event_atlas/"
    "report/report_queries.sql"
)

SEQUENCE_SQL = """
SELECT level, label, metric, subject_n, observed, null_mean,
       null_ci_lower, null_ci_upper, randomization_p_greater_equal
FROM sequence_null_summary
WHERE level IN ('cohort', 'overall')
ORDER BY level, label, metric
""".strip()

EVENT_TYPE_SQL = """
SELECT cohort, event_type, event_n, subject_n, robust_event_n
FROM event_type_counts
WHERE cohort = 'reserved_application'
ORDER BY event_n DESC
""".strip()

GATE_SQL = """
SELECT mechanism, event_n, subject_n, reserved_event_share,
       robust_event_fraction, common_prevalence_gate,
       defining_effect_gate, b0_residual_gate, passed
FROM mechanism_gate_summary
ORDER BY event_n DESC
""".strip()

SUBJECT_NULL_SQL = """
SELECT label AS subject, observed, null_mean, null_ci_lower, null_ci_upper,
       randomization_p_greater_equal, randomization_fdr_q
FROM sequence_null_summary
WHERE level = 'subject' AND metric = 'event_count'
ORDER BY randomization_p_greater_equal ASC
LIMIT 10
""".strip()

EFFECT_SQL = """
SELECT column, subject_n, mean, ci_lower, ci_upper,
       positive_subject_fraction, negative_subject_fraction
FROM decision_overall_reserved_effects
""".strip()


EVENT_TYPE_LABELS = {
    "candidate_wrong_rule": "替代规则（FDR后）",
    "choice_bias_or_perseveration": "选择偏置/固着",
    "candidate_engagement_speedup": "投入下降候选：加速",
    "candidate_engagement_slowdown": "投入下降候选：减速",
    "mixed_unresolved": "混合/未解析",
}

MECHANISM_LABELS = {
    "wrong_rule_switch": "替代规则切换",
    "perseveration": "选择固着",
    "engagement_speedup": "投入变化：加速",
    "engagement_slowdown": "投入变化：减速",
}

EFFECT_LABELS = {
    "delta_accuracy": "正确率变化",
    "delta_rt_robust_z_mean": "RT稳健z变化",
    "delta_b0_brier_mean": "B0 Brier残差变化",
    "delta_dominant_choice_rate": "优势反应比例变化",
    "delta_lose_stay_rate": "错误后保持率变化",
    "wrong_rule_gain_over_target": "最佳错误规则相对目标规则增益",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_newplan/behavior_event_atlas"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_newplan/behavior_event_atlas/"
            "report/artifact.json"
        ),
    )
    return parser.parse_args()


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def query_source(
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
        "path": REPORT_QUERY_PATH,
        "query": {
            "engine": "csv+pandas",
            "sql": sql,
            "description": description,
            "language": "SQL-like",
            "executed_at": generated_at,
            "tables_used": tables_used,
            "filters": filters,
            "metric_definitions": definitions,
        },
    }


def main() -> None:
    args = parse_args()
    result_dir = args.results_dir
    decision = json.loads(
        (result_dir / "decision.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (result_dir / "metadata.json").read_text(encoding="utf-8")
    )
    quality = json.loads(
        (result_dir / "data_quality.json").read_text(encoding="utf-8")
    )
    events = pd.read_csv(result_dir / "events_primary.csv")
    event_types = pd.read_csv(result_dir / "event_type_counts.csv")
    gates = pd.read_csv(result_dir / "mechanism_gate_summary.csv")
    cohort = pd.read_csv(result_dir / "cohort_summary.csv")
    sequence = pd.read_csv(result_dir / "sequence_null_summary.csv")

    generated_at = datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    reserved_events = events.loc[
        events["cohort"].eq("reserved_application")
    ].copy()
    reserved_sequence = sequence.loc[
        sequence["level"].eq("cohort")
        & sequence["label"].eq("reserved_application")
    ]
    reserved_event_null = reserved_sequence.loc[
        reserved_sequence["metric"].eq("event_count")
    ].iloc[0]
    reserved_subject_null = reserved_sequence.loc[
        reserved_sequence["metric"].eq("affected_subject_count")
    ].iloc[0]

    sequence_chart_rows: list[dict[str, Any]] = []
    for label, display in (
        ("development", "开发集"),
        ("reserved_application", "保留应用集"),
    ):
        row = sequence.loc[
            sequence["level"].eq("cohort")
            & sequence["label"].eq(label)
            & sequence["metric"].eq("event_count")
        ].iloc[0]
        sequence_chart_rows.extend(
            [
                {
                    "comparison": f"{display}｜实际",
                    "event_count": float(row["observed"]),
                    "kind": "实际",
                },
                {
                    "comparison": f"{display}｜块内打乱均值",
                    "event_count": float(row["null_mean"]),
                    "kind": "序列零假设",
                },
            ]
        )
    sequence_chart = pd.DataFrame(sequence_chart_rows)

    reserved_type_counts = (
        event_types.loc[event_types["cohort"].eq("reserved_application")]
        .set_index("event_type")["event_n"]
        .to_dict()
    )
    event_type_chart = pd.DataFrame(
        [
            {
                "event_type": label,
                "event_n": int(reserved_type_counts.get(event_type, 0)),
            }
            for event_type, label in EVENT_TYPE_LABELS.items()
        ]
    ).sort_values("event_n", ascending=False)

    gate_table = gates.copy()
    gate_table["mechanism_label"] = gate_table["mechanism"].map(
        MECHANISM_LABELS
    )
    for column in (
        "common_prevalence_gate",
        "defining_effect_gate",
        "b0_residual_gate",
        "passed",
    ):
        gate_table[column] = gate_table[column].map(
            {True: "通过", False: "未通过"}
        )

    effects = pd.DataFrame(decision["overall_reserved_effects"])
    effects["effect_label"] = effects["column"].map(EFFECT_LABELS)

    subject_null = (
        sequence.loc[
            sequence["level"].eq("subject")
            & sequence["metric"].eq("event_count")
        ]
        .sort_values("randomization_p_greater_equal")
        .head(10)
        .rename(columns={"label": "subject"})
    )

    raw_rule_flags = int(
        np.sum(reserved_events["wrong_rule_permutation_p"] <= 0.05)
    )
    fdr_rule_flags = int(
        np.sum(reserved_events["wrong_rule_fdr_q"] <= 0.05)
    )
    min_rule_q = float(reserved_events["wrong_rule_fdr_q"].min())
    unrecovered = int(np.sum(~reserved_events["recovered"].astype(bool)))
    rt_effect = next(
        item
        for item in decision["overall_reserved_effects"]
        if item["column"] == "delta_rt_robust_z_mean"
    )
    b0_effect = next(
        item
        for item in decision["overall_reserved_effects"]
        if item["column"] == "delta_b0_brier_mean"
    )
    rt_excluded = int(
        sum(item["rt_qc_excluded"] for item in quality["rt_qc"])
    )

    scope_table = cohort.copy()
    scope_table["cohort_label"] = scope_table["cohort"].map(
        {
            "development": "开发集",
            "reserved_application": "保留应用集",
        }
    )
    sequence_table = sequence.loc[
        sequence["level"].isin(["cohort", "overall"])
    ].copy()
    sequence_table["label"] = sequence_table["label"].replace(
        {
            "development": "开发集",
            "reserved_application": "保留应用集",
            "all_subjects": "全部被试",
        }
    )
    sequence_table["metric"] = sequence_table["metric"].replace(
        {
            "event_count": "事件数",
            "affected_subject_count": "至少一事件的被试数",
        }
    )

    datasets = {
        "sequence_chart": records(sequence_chart),
        "event_type_chart": records(event_type_chart),
        "sequence_table": records(sequence_table),
        "gate_table": records(gate_table),
        "effect_table": records(effects),
        "subject_null_table": records(subject_null),
        "scope_table": records(scope_table),
    }

    source_objects = [
        query_source(
            "sequence_null",
            "块内序列随机化基准",
            SEQUENCE_SQL,
            "读取实际事件数与块内反馈顺序打乱后的零分布。",
            tables_used=["sequence_null_summary"],
            filters=["2,000 within-block permutations", "condition = 1"],
            definitions=[
                "每次打乱保留每个实验块的试次数和正确率，只破坏错误的序列聚集。",
                "randomization p 是零分布事件数大于或等于实际值的单侧比例，加一校正。",
            ],
            generated_at=generated_at,
        ),
        query_source(
            "event_types",
            "保留集事件表型",
            EVENT_TYPE_SQL,
            "读取保留应用集中经过事件间多重比较控制后的表型计数。",
            tables_used=["event_type_counts", "events_primary"],
            filters=[
                "cohort = reserved_application",
                "primary detector window = 12",
            ],
            definitions=[
                "替代规则检验先在每个事件内用1,000次选择置换控制38条规则择优，再在事件间做BH-FDR。",
                "表型标签是描述性筛查，不是被试的潜在状态诊断。",
            ],
            generated_at=generated_at,
        ),
        query_source(
            "mechanism_gates",
            "预设单机制进入门槛",
            GATE_SQL,
            "读取四个候选机制家族的流行度、定义效应和B0残差门槛。",
            tables_used=["mechanism_gate_summary"],
            filters=["cohort = reserved_application"],
            definitions=[
                "共同门槛要求至少4名被试、至少25%保留集事件、且至少60%事件被相邻窗口支持。",
                "通过还要求机制定义效应的被试bootstrap区间和正向B0残差区间。",
            ],
            generated_at=generated_at,
        ),
        query_source(
            "reserved_effects",
            "冻结B0事件前后效应",
            EFFECT_SQL,
            "读取保留应用集中按被试等权bootstrap的事件前后变化。",
            tables_used=["decision_overall_reserved_effects"],
            filters=["cohort = reserved_application"],
            definitions=[
                "变化量均为事件段均值减去事件前12试次均值。",
                "B0 Brier残差增加是必要但非充分证据，因为事件由错误聚集定义。",
            ],
            generated_at=generated_at,
        ),
        query_source(
            "subject_null",
            "逐被试序列随机化",
            SUBJECT_NULL_SQL,
            "读取最小随机化p值的10名被试并报告32人FDR。",
            tables_used=["sequence_null_summary"],
            filters=["level = subject", "metric = event_count"],
            definitions=[
                "逐被试FDR在32名条件1被试的随机化p值上使用BH方法。",
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
                "在冻结B0上检验条件1异常正确率突降是否支持一个可识别的"
                "最小动态机制。"
            ),
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "sequence_event_count",
                    "title": "实际突降事件并未超过块内序列零假设",
                    "subtitle": "零假设保留每块正确率，只随机打乱反馈顺序",
                    "type": "horizontalBar",
                    "dataset": "sequence_chart",
                    "sourceId": "sequence_null",
                    "intent": "comparison",
                    "question": "实际错误是否比相同块正确率下更成簇？",
                    "rationale": "实际计数与零分布均值使用同一事件单位。",
                    "comparisonContext": {
                        "baseline": "块内打乱反馈顺序",
                        "grain": "cohort",
                        "unit": "events",
                    },
                    "encodings": {
                        "x": {
                            "field": "comparison",
                            "type": "nominal",
                            "label": "队列与基准",
                        },
                        "y": {
                            "field": "event_count",
                            "type": "quantitative",
                            "label": "事件数",
                        },
                        "tooltip": [
                            {
                                "field": "kind",
                                "type": "nominal",
                                "label": "类型",
                            }
                        ],
                    },
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
                {
                    "id": "reserved_event_types",
                    "title": "保留应用集事件以“混合/未解析”为主",
                    "subtitle": "替代规则标签已经过事件内置换与事件间BH-FDR",
                    "type": "horizontalBar",
                    "dataset": "event_type_chart",
                    "sourceId": "event_types",
                    "intent": "composition",
                    "question": "哪一种候选机制能覆盖足够多的保留集事件？",
                    "rationale": "五类互斥描述标签适合用计数条形图比较。",
                    "comparisonContext": {
                        "baseline": "50个保留集主检测器事件",
                        "grain": "event phenotype",
                        "unit": "events",
                    },
                    "encodings": {
                        "x": {
                            "field": "event_type",
                            "type": "nominal",
                            "label": "事件表型",
                        },
                        "y": {
                            "field": "event_n",
                            "type": "quantitative",
                            "label": "事件数",
                        },
                    },
                    "palette": {"kind": "sequential", "name": "blue"},
                    "labels": {"values": "all"},
                    "settings": {
                        "orientation": "horizontal",
                        "sort": "descending",
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
                    "id": "sequence_table_def",
                    "title": "序列零假设检验",
                    "subtitle": "2,000次块内打乱；p为实际事件更多的单侧检验",
                    "dataset": "sequence_table",
                    "sourceId": "sequence_null",
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "label", "label": "范围", "type": "text"},
                        {"field": "metric", "label": "指标", "type": "text"},
                        {"field": "observed", "label": "实际", "format": "number"},
                        {"field": "null_mean", "label": "零均值", "format": "number"},
                        {"field": "null_ci_lower", "label": "零95%下界", "format": "number"},
                        {"field": "null_ci_upper", "label": "零95%上界", "format": "number"},
                        {
                            "field": "randomization_p_greater_equal",
                            "label": "随机化p",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "gate_table_def",
                    "title": "单机制进入门槛",
                    "subtitle": "没有候选同时通过流行度、定义效应与B0残差门槛",
                    "dataset": "gate_table",
                    "sourceId": "mechanism_gates",
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "mechanism_label", "label": "机制", "type": "text"},
                        {"field": "event_n", "label": "事件", "format": "number"},
                        {"field": "subject_n", "label": "被试", "format": "number"},
                        {
                            "field": "reserved_event_share",
                            "label": "事件占比",
                            "format": "percent",
                        },
                        {
                            "field": "common_prevalence_gate",
                            "label": "流行度门槛",
                            "type": "text",
                        },
                        {
                            "field": "defining_effect_gate",
                            "label": "定义效应",
                            "type": "text",
                        },
                        {
                            "field": "b0_residual_gate",
                            "label": "B0残差",
                            "type": "text",
                        },
                        {"field": "passed", "label": "总结果", "type": "text"},
                    ],
                },
                {
                    "id": "subject_null_table_def",
                    "title": "逐被试事件计数随机化检验（最小p的10人）",
                    "subtitle": "32人BH-FDR后无人通过q≤0.05",
                    "dataset": "subject_null_table",
                    "sourceId": "subject_null",
                    "density": "compact",
                    "layout": "full",
                    "columns": [
                        {"field": "subject", "label": "被试", "type": "text"},
                        {"field": "observed", "label": "实际事件", "format": "number"},
                        {"field": "null_mean", "label": "零均值", "format": "number"},
                        {
                            "field": "randomization_p_greater_equal",
                            "label": "p",
                            "format": "number",
                        },
                        {
                            "field": "randomization_fdr_q",
                            "label": "BH q",
                            "format": "number",
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
                    "sourceId": "sequence_null",
                    "body": (
                        "## 技术结论：当前数据不支持新增动态机制\n\n"
                        "**条件1确认模型继续冻结为 B0；不加入切换、固着、"
                        "投入状态或新的三状态控制器。** 保留应用集的24名被试"
                        f"出现50个突降事件、涉及18人，但块内随机打乱反馈顺序后"
                        f"平均会出现{reserved_event_null['null_mean']:.2f}个事件"
                        f"（95%零区间[{reserved_event_null['null_ci_lower']:.0f}, "
                        f"{reserved_event_null['null_ci_upper']:.0f}]；"
                        f"单侧随机化p={reserved_event_null['randomization_p_greater_equal']:.3f}）。"
                        "因此“看起来突然掉下去”本身没有超过相同块正确率下随机错误"
                        "聚集的预期。\n\n"
                        "这不是说所有被试都理性、也不是说B0解释了全部异常；结论是"
                        "现有外显序列没有给出足够特异的证据，去选择某一个额外心理机制。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "sequence_heading",
                    "type": "markdown",
                    "sourceId": "sequence_null",
                    "body": (
                        "## 先问“事件是否超出随机聚集”，再问“它是什么机制”\n\n"
                        f"保留应用集中实际有18人至少出现一次事件；块内打乱的均值为"
                        f"{reserved_subject_null['null_mean']:.2f}人，95%零区间为"
                        f"[{reserved_subject_null['null_ci_lower']:.0f}, "
                        f"{reserved_subject_null['null_ci_upper']:.0f}]，"
                        f"p={reserved_subject_null['randomization_p_greater_equal']:.3f}。"
                        "开发集与全体被试得到相同方向。检测器在每个实验块重新开始，"
                        "以事件前12试次正确率至少2/3、随后12试次至多1/2且下降至少"
                        "0.25定义起点；8与16试次窗口仅作邻近敏感性支持。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "sequence_chart_block",
                    "type": "chart",
                    "chartId": "sequence_event_count",
                    "layout": "full",
                },
                {
                    "id": "sequence_table_block",
                    "type": "table",
                    "tableId": "sequence_table_def",
                    "layout": "full",
                },
                {
                    "id": "phenotype_heading",
                    "type": "markdown",
                    "sourceId": "event_types",
                    "body": (
                        "## 事件的后续表现没有收敛到同一种机制\n\n"
                        f"50个保留集事件中，42个为混合/未解析，6个达到描述性的"
                        f"选择偏置或错误后固着阈值，2个表现为RT减慢候选；其中"
                        f"{unrecovered}/50在当前块结束前未达到恢复标准。替代规则搜索"
                        f"在未校正时标出{raw_rule_flags}个事件，但事件间BH-FDR后为"
                        f"{fdr_rule_flags}个，最小q={min_rule_q:.3f}。因此不能把"
                        "短暂低正确率自动解释成规则切换。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "event_type_chart_block",
                    "type": "chart",
                    "chartId": "reserved_event_types",
                    "layout": "full",
                },
                {
                    "id": "frozen_b0_heading",
                    "type": "markdown",
                    "sourceId": "reserved_effects",
                    "body": (
                        "## 冻结B0确实在事件段失配，但没有指向唯一修复\n\n"
                        f"保留集事件段的B0逐试次Brier残差平均增加"
                        f"{b0_effect['mean']:.3f}，被试bootstrap 95% CI "
                        f"[{b0_effect['ci_lower']:.3f}, {b0_effect['ci_upper']:.3f}]。"
                        "这说明B0没有吸收这些错误段，但由于事件本身就是用正确率下降"
                        "定义，残差增加只能作为必要条件，不能识别错误背后的原因。"
                        f"RT稳健z的平均变化为{rt_effect['mean']:.3f}，95% CI "
                        f"[{rt_effect['ci_lower']:.3f}, {rt_effect['ci_upper']:.3f}]，"
                        "也没有形成共享的投入下降信号。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "gate_heading",
                    "type": "markdown",
                    "sourceId": "mechanism_gates",
                    "body": (
                        "## 预设门槛拒绝所有单模块扩展\n\n"
                        "候选模块必须在保留应用集中至少覆盖4名被试和25%的事件，"
                        "至少60%的事件得到相邻窗口支持，并同时通过机制定义效应与"
                        "正向B0残差的被试bootstrap区间。替代规则在FDR后无事件；"
                        "固着只有6人/6事件且B0残差区间跨零；RT减慢只有2人/2事件。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "gate_table_block",
                    "type": "table",
                    "tableId": "gate_table_def",
                    "layout": "full",
                },
                {
                    "id": "scope_method",
                    "type": "markdown",
                    "body": (
                        "## 范围、方法与数据质量\n\n"
                        f"分析覆盖条件1的{quality['row_count']:,}个试次、"
                        f"{quality['subject_count']}名被试；8人为既有开发集，24人为"
                        "保留应用集。choRT完整，按被试log RT中位数与Gaussian-consistent "
                        f"MAD排除绝对稳健z超过4的{rt_excluded}个试次。32名被试的"
                        "冻结B0选择预测全部成功对齐，选择索引不匹配为0。每个事件还"
                        "计算恢复、反应偏置、win/lose-stay、RT、刺激歧义、B0残差，"
                        "并在38条完整带标签规则空间中寻找替代规则。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "subject_heading",
                    "type": "markdown",
                    "sourceId": "subject_null",
                    "body": (
                        "## 个别被试可作为探索案例，但不能成为模型选择依据\n\n"
                        "逐被试序列随机化中只有127号被试达到未经校正的p<0.05；"
                        "在32名被试之间控制BH-FDR后无人达到q≤0.05。该被试可以"
                        "用于生成新任务假设，不能在当前数据内据此开放专属状态机制。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "subject_null_table_block",
                    "type": "table",
                    "tableId": "subject_null_table_def",
                    "layout": "full",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 局限性与稳健性边界\n\n"
                        "保留应用集没有参与先前B0/D0开发，但在冻结本检测器前已经"
                        "看过一次粗略的全体行为扫描，因此这里是准确认性应用，不是"
                        "真正盲法确认。块内打乱检验的是序列聚集，而非完整认知过程；"
                        "它保留块正确率，却不保留学习趋势或反馈自相关。窗口敏感性"
                        "100%支持主事件并不独立于同一滑窗定义，不能抵消序列零假设"
                        "未通过。事件表型也不是生成模型，不应用来给被试命名。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 下一步：停止往当前数据里加状态，转向可区分的测量设计\n\n"
                        "- 条件1确认模型保持B0；D0与事件标签只留作探索性附录。\n"
                        "- 不再用同一批选择/RT数据继续搜索更多隐藏状态或被试专属机制。\n"
                        "- 若要检验规则切换，应在新实验中加入可预先指定的规则探针、"
                        "反转操纵或口头报告时点；若要检验投入波动，应加入独立的注意/"
                        "生理指标，并预先规定RT方向。\n"
                        "- 新数据中先恢复“是否有事件”和“事件属于何种机制”，再只"
                        "实现一个最小模块，并要求真实试次数下的模型家族恢复与held-out泛化。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 后续真正需要回答的问题\n\n"
                        "1. 哪一种独立测量能把规则切换、固着、短暂lapse与疲劳分开？\n"
                        "2. 新任务能否让这些机制对下一试次、恢复速度和RT产生方向不同的"
                        "预注册预测？\n"
                        "3. 被试异质性应由层级混合比例表达，还是由可重复的外部协变量"
                        "预测，而不是由同一行为序列事后命名？"
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
    query_path = args.output.parent / "report_queries.sql"
    query_path.write_text(
        "\n\n".join(
            [
                "-- Sequence-null benchmark\n" + SEQUENCE_SQL + ";",
                "-- Reserved event phenotypes\n" + EVENT_TYPE_SQL + ";",
                "-- Mechanism gates\n" + GATE_SQL + ";",
                "-- Subject-level sequence null\n" + SUBJECT_NULL_SQL + ";",
                "-- Reserved event effects\n" + EFFECT_SQL + ";",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    args.output.write_text(
        json.dumps(
            artifact,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "artifact": str(args.output),
                "datasets": {
                    key: len(value) for key, value in datasets.items()
                },
                "recommended_module": decision["recommended_module"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
