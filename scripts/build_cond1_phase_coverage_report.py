#!/usr/bin/env python3
"""Build a portable technical report for frozen phase-level coverage."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TITLE = "条件1阶段级生成检验：复杂轨迹覆盖与机制停止规则"
REPORT_QUERY_PATH = (
    "results/zhuran/cond1_newplan/phase_coverage_report/report/"
    "report_queries.sql"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_newplan/"
            "phase_coverage_frozen_models"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_newplan/"
            "phase_coverage_report/report/artifact.json"
        ),
    )
    return parser.parse_args()


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def source_object(
    source_id: str,
    label: str,
    query: str,
    description: str,
    generated_at: str,
    tables_used: list[str],
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": REPORT_QUERY_PATH,
        "query": {
            "engine": "csv+json+pandas",
            "sql": query,
            "description": description,
            "language": "SQL-like",
            "executed_at": generated_at,
            "tables_used": tables_used,
            "filters": [
                "condition = 1",
                "cohort = 24 frozen reserved subjects",
                "models refit or retuned = false",
                "new rollouts generated = false",
                "phase windows = 8, 12, 16 trials",
            ],
            "metric_definitions": [
                "阶段词汇是可观察正确率轨迹描述，不是潜在认知状态。",
                "联合通过检验取30个阶段量的最大稳健标准化偏差。",
                "每个模型用自身模拟轨迹校准95%联合阈值。",
                "scaled phase CRPS越低越好；共享尺度不读取真实结果。",
                "扩展门槛要求至少4名被试出现同方向、跨窗口、双模型残差。",
            ],
        },
    }


def main() -> None:
    args = parse_args()
    analysis = args.analysis_dir
    generated_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    decision = json.loads(
        (analysis / "phase_decision.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (analysis / "manifest.json").read_text(encoding="utf-8")
    )
    subject = pd.read_csv(analysis / "subject_phase_summary.csv")
    prevalence = pd.read_csv(
        analysis / "observed_phase_prevalence.csv"
    )
    residual = pd.read_csv(
        analysis / "shared_residual_summary.csv"
    )
    calibration = pd.DataFrame(decision["cohort_calibration"])

    calibration_chart_rows: list[dict[str, Any]] = []
    short_model = {
        "C1": "动态 C1",
        "acquisition": "单次掌握变点",
    }
    for row in calibration.to_dict(orient="records"):
        calibration_chart_rows.extend(
            [
                {
                    "model_and_reference": (
                        f"{short_model[row['model']]}｜真实"
                    ),
                    "pass_n": float(row["observed_pass_n"]),
                    "series": "真实轨迹",
                },
                {
                    "model_and_reference": (
                        f"{short_model[row['model']]}｜模型自期望"
                    ),
                    "pass_n": float(row["self_expected_pass_mean"]),
                    "series": "模型自期望",
                },
            ]
        )
    calibration_chart = pd.DataFrame(calibration_chart_rows)

    prevalence_chart = prevalence.copy()
    prevalence_chart["subject_percent"] = (
        prevalence_chart["subject_fraction"] * 100
    )

    crps_scatter = subject[
        [
            "iSub",
            "C1_phase_scaled_crps",
            "acquisition_phase_scaled_crps",
            "C1_phase_pass_95",
            "acquisition_phase_pass_95",
            "phase_signature",
        ]
    ].copy()
    crps_scatter["coverage_pattern"] = crps_scatter.apply(
        lambda row: (
            "两者均通过"
            if row["C1_phase_pass_95"]
            and row["acquisition_phase_pass_95"]
            else (
                "仅C1通过"
                if row["C1_phase_pass_95"]
                else "仅单变点通过"
            )
        ),
        axis=1,
    )

    coverage = decision["coverage_pattern"]
    coverage_chart = pd.DataFrame(
        [
            {"pattern": "两者均通过", "subject_n": coverage["both_pass_n"]},
            {"pattern": "仅 C1 通过", "subject_n": coverage["C1_only_pass_n"]},
            {
                "pattern": "仅单变点通过",
                "subject_n": coverage["acquisition_only_pass_n"],
            },
            {"pattern": "两者均失败", "subject_n": coverage["neither_pass_n"]},
        ]
    )

    metric_definitions = pd.DataFrame(manifest["phase_metrics"])
    thresholds = manifest["primary_thresholds"]
    paired = decision["paired_scaled_phase_crps"]
    c1 = calibration.loc[calibration["model"] == "C1"].iloc[0]
    acq = calibration.loc[
        calibration["model"] == "acquisition"
    ].iloc[0]

    source_sql = {
        "cohort_calibration": (
            "SELECT model, observed_pass_n, subject_n, "
            "self_expected_pass_mean, self_expected_pass_q025, "
            "self_expected_pass_q975, lower_tail_calibration_p "
            "FROM phase_cohort_calibration"
        ),
        "subject_phase": (
            "SELECT iSub, phase_signature, C1_phase_pass_95, "
            "acquisition_phase_pass_95, C1_phase_scaled_crps, "
            "acquisition_phase_scaled_crps, C1_phase_fdr_q, "
            "acquisition_phase_fdr_q FROM subject_phase_summary"
        ),
        "phase_prevalence": (
            "SELECT phase, subject_n, subject_fraction "
            "FROM observed_phase_prevalence"
        ),
        "shared_residual": (
            "SELECT metric, metric_label, direction, "
            "primary_shared_failure_subject_n, "
            "cross_window_supported_subject_n, extension_gate "
            "FROM shared_residual_summary"
        ),
        "metric_definition": (
            "SELECT metric, label, resolution "
            "FROM manifest.phase_metrics"
        ),
    }
    source_meta = {
        "cohort_calibration": (
            "冻结模型的阶段联合覆盖与模型自身校准。",
            [
                "phase_coverage_frozen_models/phase_cohort_calibration.csv",
                "phase_coverage_frozen_models/phase_decision.json",
            ],
        ),
        "subject_phase": (
            "逐被试阶段签名、联合通过和配对proper score。",
            ["phase_coverage_frozen_models/subject_phase_summary.csv"],
        ),
        "phase_prevalence": (
            "24名保留被试的可观察阶段描述出现率。",
            [
                "phase_coverage_frozen_models/"
                "observed_phase_prevalence.csv"
            ],
        ),
        "shared_residual": (
            "双模型同方向失败及跨窗口扩展门槛。",
            [
                "phase_coverage_frozen_models/"
                "shared_residual_summary.csv",
                "phase_coverage_frozen_models/phase_decision.json",
            ],
        ),
        "metric_definition": (
            "阶段指标、窗口和阈值的冻结定义。",
            ["phase_coverage_frozen_models/manifest.json"],
        ),
    }
    labels = {
        "cohort_calibration": "群体阶段校准",
        "subject_phase": "逐被试阶段结果",
        "phase_prevalence": "阶段特征出现率",
        "shared_residual": "共同残差门槛",
        "metric_definition": "阶段指标定义",
    }
    sources = [
        source_object(
            source_id,
            labels[source_id],
            query,
            source_meta[source_id][0],
            generated_at,
            source_meta[source_id][1],
        )
        for source_id, query in source_sql.items()
    ]

    subject_table = subject[
        [
            "iSub",
            "phase_signature",
            "C1_phase_pass_95",
            "acquisition_phase_pass_95",
            "C1_phase_calibration_p",
            "acquisition_phase_calibration_p",
            "C1_phase_scaled_crps",
            "acquisition_phase_scaled_crps",
            "phase_scaled_crps_C1_minus_acquisition",
        ]
    ].copy()

    datasets = {
        "calibration_chart": records(calibration_chart),
        "calibration_table": records(calibration),
        "prevalence_chart": records(prevalence_chart),
        "crps_scatter": records(crps_scatter),
        "coverage_chart": records(coverage_chart),
        "subject_table": records(subject_table),
        "metric_definitions": records(metric_definitions),
        "shared_residual": records(residual),
    }

    charts = [
        {
            "id": "calibration_comparison",
            "title": "阶段联合通过人数：真实轨迹与模型自期望",
            "subtitle": "同一24名冻结保留被试；每模型各自校准",
            "type": "horizontalBar",
            "dataset": "calibration_chart",
            "sourceId": "cohort_calibration",
            "intent": "comparison",
            "question": "两种冻结生成器能否覆盖复杂阶段组合？",
            "rationale": (
                "把真实通过人数与模型自身伪观测期望并列，"
                "直接判断真实队列是否比模型为真时更极端。"
            ),
            "comparisonContext": {
                "baseline": "每个模型自身的伪观测通过数",
                "grain": "model × reference",
                "unit": "subjects passing 30-dimensional joint test",
            },
            "encodings": {
                "x": {
                    "field": "model_and_reference",
                    "type": "nominal",
                    "label": "模型与参照",
                },
                "y": {
                    "field": "pass_n",
                    "type": "quantitative",
                    "label": "通过人数",
                },
            },
            "labels": {"values": "all"},
            "settings": {
                "orientation": "horizontal",
                "sort": "ascending",
                "showValues": True,
            },
            "layout": "full",
            "surface": {"surface": "explorer", "viewMode": "both"},
        },
        {
            "id": "phase_prevalence",
            "title": "真实轨迹中的复杂阶段描述广泛存在",
            "subtitle": "同一被试可同时具有多种轨迹特征",
            "type": "horizontalBar",
            "dataset": "prevalence_chart",
            "sourceId": "phase_prevalence",
            "intent": "comparison",
            "question": "复杂阶段是否只来自少数异常被试？",
            "rationale": (
                "显示每种可观察阶段特征的被试覆盖率，"
                "但不把描述性标签升级为潜在状态。"
            ),
            "comparisonContext": {
                "baseline": "24名冻结保留被试",
                "grain": "observable phase descriptor",
                "unit": "percent of subjects",
            },
            "encodings": {
                "x": {
                    "field": "phase",
                    "type": "nominal",
                    "label": "阶段描述",
                },
                "y": {
                    "field": "subject_percent",
                    "type": "quantitative",
                    "label": "被试比例（%）",
                },
                "tooltip": [
                    {
                        "field": "subject_n",
                        "type": "quantitative",
                        "label": "被试数",
                    }
                ],
            },
            "labels": {"values": "all"},
            "settings": {
                "orientation": "horizontal",
                "sort": "ascending",
                "showValues": True,
            },
            "layout": "full",
            "surface": {"surface": "explorer", "viewMode": "both"},
        },
        {
            "id": "paired_phase_crps",
            "title": "逐被试阶段proper score：C1对单变点",
            "subtitle": "越低越好；两轴使用同一冻结的共享尺度",
            "type": "scatter",
            "dataset": "crps_scatter",
            "sourceId": "subject_phase",
            "intent": "relationship",
            "question": "动态C1是否在阶段预测精度上形成可靠优势？",
            "rationale": (
                "逐被试配对显示模型优势的异质性；"
                "总体均值必须结合bootstrap区间解释。"
            ),
            "comparisonContext": {
                "baseline": "identity line: equal phase CRPS",
                "grain": "reserved subject",
                "unit": "scaled phase CRPS",
            },
            "encodings": {
                "x": {
                    "field": "acquisition_phase_scaled_crps",
                    "type": "quantitative",
                    "label": "单次掌握变点阶段CRPS",
                },
                "y": {
                    "field": "C1_phase_scaled_crps",
                    "type": "quantitative",
                    "label": "动态C1阶段CRPS",
                },
                "color": {
                    "field": "coverage_pattern",
                    "type": "nominal",
                    "label": "联合覆盖",
                },
                "tooltip": [
                    {
                        "field": "iSub",
                        "type": "quantitative",
                        "label": "被试",
                    },
                    {
                        "field": "phase_signature",
                        "type": "nominal",
                        "label": "阶段签名",
                    },
                ],
            },
            "layout": "full",
            "surface": {"surface": "explorer", "viewMode": "both"},
        },
        {
            "id": "coverage_pattern",
            "title": "逐被试联合覆盖模式",
            "subtitle": "没有任何被试同时被两种模型判为极端",
            "type": "horizontalBar",
            "dataset": "coverage_chart",
            "sourceId": "subject_phase",
            "intent": "composition",
            "question": "是否存在两种最小生成器都无法覆盖的个案？",
            "rationale": (
                "交叉分类比单独报告两套通过数更直接，"
                "并为是否开放个体例外机制提供门槛证据。"
            ),
            "comparisonContext": {
                "baseline": "24名冻结保留被试",
                "grain": "joint model coverage pattern",
                "unit": "subjects",
            },
            "encodings": {
                "x": {
                    "field": "pattern",
                    "type": "nominal",
                    "label": "覆盖模式",
                },
                "y": {
                    "field": "subject_n",
                    "type": "quantitative",
                    "label": "被试数",
                },
            },
            "labels": {"values": "all"},
            "settings": {
                "orientation": "horizontal",
                "sort": "ascending",
                "showValues": True,
            },
            "layout": "full",
            "surface": {"surface": "explorer", "viewMode": "both"},
        },
    ]

    tables = [
        {
            "id": "calibration_table",
            "title": "模型级阶段联合校准",
            "subtitle": "30维最大偏差；模型自身伪观测阈值",
            "dataset": "calibration_table",
            "sourceId": "cohort_calibration",
            "density": "spacious",
            "layout": "full",
            "columns": [
                {"field": "model_label", "label": "模型", "type": "text"},
                {
                    "field": "observed_pass_n",
                    "label": "真实通过",
                    "format": "number",
                },
                {"field": "subject_n", "label": "人数", "format": "number"},
                {
                    "field": "self_expected_pass_mean",
                    "label": "自期望",
                    "format": "number",
                },
                {
                    "field": "self_expected_pass_q025",
                    "label": "自期望2.5%",
                    "format": "number",
                },
                {
                    "field": "self_expected_pass_q975",
                    "label": "自期望97.5%",
                    "format": "number",
                },
                {
                    "field": "lower_tail_calibration_p",
                    "label": "群体p",
                    "format": "number",
                },
            ],
        },
        {
            "id": "subject_table",
            "title": "逐被试阶段签名与模型比较",
            "subtitle": "阶段签名仅为可观察描述；CRPS越低越好",
            "dataset": "subject_table",
            "sourceId": "subject_phase",
            "density": "compact",
            "layout": "full",
            "columns": [
                {"field": "iSub", "label": "被试", "format": "number"},
                {
                    "field": "phase_signature",
                    "label": "阶段签名",
                    "type": "text",
                },
                {
                    "field": "C1_phase_pass_95",
                    "label": "C1通过",
                    "type": "boolean",
                },
                {
                    "field": "acquisition_phase_pass_95",
                    "label": "变点通过",
                    "type": "boolean",
                },
                {
                    "field": "C1_phase_scaled_crps",
                    "label": "C1 CRPS",
                    "format": "number",
                },
                {
                    "field": "acquisition_phase_scaled_crps",
                    "label": "变点CRPS",
                    "format": "number",
                },
                {
                    "field": "phase_scaled_crps_C1_minus_acquisition",
                    "label": "C1−变点",
                    "format": "number",
                },
            ],
        },
        {
            "id": "metric_table",
            "title": "冻结的阶段描述量",
            "subtitle": "每个量同时在8、12和16试次窗口计算",
            "dataset": "metric_definitions",
            "sourceId": "metric_definition",
            "density": "compact",
            "layout": "full",
            "columns": [
                {"field": "metric", "label": "变量", "type": "text"},
                {"field": "label", "label": "中文定义", "type": "text"},
                {
                    "field": "resolution",
                    "label": "度量",
                    "type": "text",
                },
            ],
        },
        {
            "id": "residual_table",
            "title": "双模型共同残差检查",
            "subtitle": "跨窗口支持人数须达到4人才开放新机制",
            "dataset": "shared_residual",
            "sourceId": "shared_residual",
            "density": "compact",
            "layout": "full",
            "columns": [
                {
                    "field": "metric_label",
                    "label": "阶段量",
                    "type": "text",
                },
                {
                    "field": "direction",
                    "label": "方向",
                    "type": "text",
                },
                {
                    "field": "primary_shared_failure_subject_n",
                    "label": "主尺度共同失败",
                    "format": "number",
                },
                {
                    "field": "cross_window_supported_subject_n",
                    "label": "跨窗口支持",
                    "format": "number",
                },
                {
                    "field": "extension_gate",
                    "label": "开放扩展",
                    "type": "boolean",
                },
            ],
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": (
                "在不增加新机制的前提下，直接检验两种冻结模型"
                "能否生成混乱、陡升、陡降、渐变、恢复等复杂轨迹。"
            ),
            "generatedAt": generated_at,
            "charts": charts,
            "tables": tables,
            "sources": [
                {
                    "id": source["id"],
                    "label": source["label"],
                    "path": source["path"],
                }
                for source in sources
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": f"# {TITLE}",
                    "layout": "full",
                },
                {
                    "id": "summary",
                    "type": "markdown",
                    "sourceId": "cohort_calibration",
                    "body": (
                        "## 技术摘要\n\n"
                        "阶段级检验已经给出停止扩展的结果。动态C1有"
                        f"{int(c1['observed_pass_n'])}/24人通过，模型自身期望"
                        f"{c1['self_expected_pass_mean']:.2f}"
                        f"（95%范围[{c1['self_expected_pass_q025']:.0f},"
                        f"{c1['self_expected_pass_q975']:.0f}]），"
                        f"群体p={c1['lower_tail_calibration_p']:.3f}；"
                        "单次掌握变点有"
                        f"{int(acq['observed_pass_n'])}/24人通过，自期望"
                        f"{acq['self_expected_pass_mean']:.2f}"
                        f"（95%范围[{acq['self_expected_pass_q025']:.0f},"
                        f"{acq['self_expected_pass_q975']:.0f}]），"
                        f"p={acq['lower_tail_calibration_p']:.3f}。"
                        "21人两者均通过，2人仅C1通过，1人仅变点通过，"
                        "没有人两者均失败。\n\n"
                        "C1并未取得可靠的阶段预测优势。scaled phase CRPS"
                        "的平均配对差（C1减单变点）为"
                        f"{paired['mean_C1_minus_acquisition']:.4f}，"
                        f"bootstrap 95% CI [{paired['subject_bootstrap_ci95'][0]:.4f},"
                        f"{paired['subject_bootstrap_ci95'][1]:.4f}]；"
                        f"C1在{paired['C1_better_subject_n']}人中更低，"
                        f"单变点在{paired['acquisition_better_subject_n']}人中更低。"
                        "没有任何共同残差达到至少4人的跨窗口扩展门槛。"
                        "故C1保留为阶段型工作模型，单变点保留为简约基准，"
                        "当前停止增加群体层机制。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "coverage_explanation",
                    "type": "markdown",
                    "sourceId": "cohort_calibration",
                    "body": (
                        "## 主要发现一：复杂阶段组合不是模型下的边缘事件\n\n"
                        "联合通过不是要求真实轨迹逐点贴近模拟中位数，而是检查"
                        "真实轨迹在30个阶段描述量中的最大偏差，是否超过模型自身"
                        "重复样本通常出现的范围。两种冻结模型的真实通过数均落在"
                        "自身95%通过数范围内，因此都具备阶段层面的群体生成充分性。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "calibration_chart_block",
                    "type": "chart",
                    "chartId": "calibration_comparison",
                    "layout": "full",
                },
                {
                    "id": "calibration_table_block",
                    "type": "table",
                    "tableId": "calibration_table",
                    "layout": "full",
                },
                {
                    "id": "prevalence_explanation",
                    "type": "markdown",
                    "sourceId": "phase_prevalence",
                    "body": (
                        "## 主要发现二：被试确实呈现多样、非单调的学习时程\n\n"
                        "混乱、陡升、陡降、渐变和恢复分别见于13、16、14、"
                        "10和7名被试；稳定高水平在24人中均出现。"
                        "这证实仅用“学会前/学会后”描述数据是不充分的。"
                        "但阶段词汇只是对轨迹的可复核描述，不能据此声称"
                        "模型已经识别出同名的内部状态。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "prevalence_chart_block",
                    "type": "chart",
                    "chartId": "phase_prevalence",
                    "layout": "full",
                },
                {
                    "id": "proper_score_explanation",
                    "type": "markdown",
                    "sourceId": "subject_phase",
                    "body": (
                        "## 主要发现三：C1有表达优势，但没有阶段预测优势\n\n"
                        "C1能自然地产生多次可逆升降，因此更贴合本研究的"
                        "理论表达目标；然而proper score比较的是完整预测分布"
                        "离真实阶段量有多近，并惩罚过宽分布。配对均值几乎为零、"
                        "区间跨0，逐被试胜负也接近对半。故把C1作为工作模型"
                        "不能写成数据已证明持续波动机制优于单次掌握。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "crps_chart_block",
                    "type": "chart",
                    "chartId": "paired_phase_crps",
                    "layout": "full",
                },
                {
                    "id": "coverage_pattern_explanation",
                    "type": "markdown",
                    "sourceId": "subject_phase",
                    "body": (
                        "## 个体异质性如何处理\n\n"
                        "不同被试可以由不同的最小机制获得更好预测；"
                        "本分析不强迫一套机制对所有人占优。关键判据是是否有人"
                        "同时超出两种冻结生成器，以及这些失败是否形成共同方向。"
                        "实际没有双模型共同失败者，因此目前没有需要用新机制"
                        "才能纳入生成空间的个体。119号虽在旧15项联合检验留下"
                        "个案残差，在本阶段联合检验中同时被两模型覆盖。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "coverage_chart_block",
                    "type": "chart",
                    "chartId": "coverage_pattern",
                    "layout": "full",
                },
                {
                    "id": "subject_table_block",
                    "type": "table",
                    "tableId": "subject_table",
                    "layout": "full",
                },
                {
                    "id": "scope_method",
                    "type": "markdown",
                    "sourceId": "metric_definition",
                    "body": (
                        "## 范围、数据、定义与模型规格\n\n"
                        "分析对象是未参与候选选择的24名条件1保留被试。"
                        "C1固定ρ0=ρ128=0.5、σ+=0.20、φ=0.95；"
                        "单次掌握模型固定Hacq=128 trials与λ0=0.5。"
                        "本分析直接复用两模型各1024条自主后缀轨迹，"
                        "没有重新拟合、调参或生成新rollout。每个实验块分别"
                        "切成8、12和16试次非重叠窗口，不跨块计算变化；"
                        "12试次为主尺度。每尺度10个描述量，共30维。"
                        f"接近随机水平定义为窗口正确率[{thresholds['chance_band'][0]:.3f},"
                        f"{thresholds['chance_band'][1]:.3f}]；"
                        f"稳定高水平至少{thresholds['stable_high_accuracy_min']:.3f}；"
                        f"陡升/陡降的绝对变化至少"
                        f"{thresholds['abrupt_absolute_change_min']:.2f}；"
                        "渐变要求至少两个连续同方向小变化，累计至少0.25。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "metric_table_block",
                    "type": "table",
                    "tableId": "metric_table",
                    "layout": "full",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "sourceId": "shared_residual",
                    "body": (
                        "## 局限性与稳健性\n\n"
                        "窗口化阶段描述会受切分对齐影响，因此结果同时要求"
                        "8、12、16试次尺度的一致支持。三个窗口也不是三套"
                        "独立证据；它们共同进入一个最大偏差联合检验。"
                        "阶段CRPS依赖预先定义的描述量与共享模拟尺度，"
                        "不能证明潜在状态或心理原因。当前唯一的主尺度双模型"
                        "共同边际失败是“方向反转数偏低”1人，但在另外两个尺度"
                        "没有支持；所有扩展门槛均为false。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "residual_table_block",
                    "type": "table",
                    "tableId": "residual_table",
                    "layout": "full",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 决策与下一步\n\n"
                        "1. 冻结C1与单次掌握变点；不再为条件1增加群体层"
                        "跳跃、burst lapse、三状态控制器或stacking。\n"
                        "2. C1作为能够表达反复升降的工作生成器；单变点作为"
                        "不可删除的简约基准。论文同时报告两者，不能写成唯一胜者。\n"
                        "3. 保留24名逐被试轨迹图谱，作为读者检查异质性与"
                        "个案边界的审计材料，而不据此给被试做机制分型。\n"
                        "4. 只有未来在同一描述量上出现至少4名被试、同方向、"
                        "双模型且跨窗口的冻结残差，才一次开放一个特殊机制。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 仍待回答的问题\n\n"
                        "- 条件2和3能否复用同一最小动态集合，必须分别开发和冻结，"
                        "不能从条件1外推。\n"
                        "- C1的ρ波动能否获得选择以外的独立过程证据，"
                        "当前RT结果不支持正向机制解释。\n"
                        "- 若以后出现共同残差，首先应比较哪一个单机制扩展，"
                        "必须由残差方向决定，而非提前罗列复杂状态。"
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
        "sources": sources,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    query_path = args.output.parent / "report_queries.sql"
    query_path.write_text(
        "\n\n".join(
            f"-- {source_meta[source_id][0]}\n{query};"
            for source_id, query in source_sql.items()
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
                "recommended_action": decision["recommended_action"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
