#!/usr/bin/env python3
"""Build the portable technical report for continuous dynamic readout."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TITLE = "条件1连续动态 readout：生成充分性、比较边界与个体限制"
REPORT_QUERY_PATH = (
    "results/zhuran/cond1_newplan/dynamic_rho_report/report/"
    "report_queries.sql"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/zhuran/cond1_newplan",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_newplan/dynamic_rho_report/"
            "report/artifact.json"
        ),
    )
    return parser.parse_args()


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(
        frame.to_json(orient="records", force_ascii=False)
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
                "prediction boundary = after first block",
                "future observed choices not read",
                "feedback generated from simulated choices",
            ],
            "metric_definitions": [
                "联合通过同时约束15项预设轨迹摘要与块内滚动正确率曲线。",
                "群体p把实际通过数与模型自身伪观测通过数分布比较。",
                "CRPS越低越好；95%滚动带宽不得超过0.50。",
                "生成充分性不等于唯一机制识别或真实潜在路径恢复。",
            ],
        },
    }


def main() -> None:
    args = parse_args()
    root = args.results_root
    consolidated = root / "dynamic_rho_consolidated"
    formal = (
        root
        / "dynamic_rho_reserved_c1_p256_r1024/"
        "c1_s0p5_e0p5_v0p2_p0p95"
    )
    seed = (
        root
        / "dynamic_rho_reserved_c1_seed20261101_p128_r512/"
        "c1_s0p5_e0p5_v0p2_p0p95"
    )
    generated_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    model_comparison = pd.read_csv(
        consolidated / "model_comparison.csv"
    )
    morphology = pd.read_csv(
        consolidated / "morphology_comparison.csv"
    )
    associations = pd.read_csv(
        consolidated / "volatility_associations.csv"
    )
    subjects = pd.read_csv(
        consolidated / "subject_dynamic_diagnostics.csv"
    )
    dev_sweep = pd.read_csv(
        root
        / "dynamic_rho_shortlist_dev_p128_r384/"
        "sweep_summary.csv"
    )
    separability = read_json(
        root
        / "dynamic_rho_separability_dev/"
        "separability_summary.json"
    )
    paired = read_json(
        consolidated / "paired_crps_comparison.json"
    )
    data_quality = read_json(consolidated / "data_quality.json")
    development = read_json(
        root / "dynamic_rho_shortlist_dev_p128_r384/selection.json"
    )["selected"]["C1"]
    formal_decision = read_json(formal / "candidate_decision.json")
    seed_decision = read_json(seed / "candidate_decision.json")

    pass_chart_rows: list[dict[str, Any]] = []
    for row in model_comparison.to_dict(orient="records"):
        pass_chart_rows.extend(
            [
                {
                    "model_and_reference": f"{row['model']}｜实际",
                    "pass_fraction": float(
                        row["observed_pass_fraction"]
                    ),
                    "series": "实际轨迹",
                    "cohort": row["cohort"],
                    "pass_n": float(row["observed_pass_n"]),
                },
                {
                    "model_and_reference": (
                        f"{row['model']}｜模型自期望"
                    ),
                    "pass_fraction": float(
                        row["self_expected_pass_fraction"]
                    ),
                    "series": "模型自期望",
                    "cohort": row["cohort"],
                    "pass_n": float(
                        row["self_expected_pass_mean"]
                    ),
                },
            ]
        )
    pass_chart = pd.DataFrame(pass_chart_rows)

    crps_chart = model_comparison[
        ["model", "mean_curve_crps", "cohort", "seed_role"]
    ].copy()
    morphology_chart = morphology.copy()
    morphology_chart["model_and_metric"] = (
        morphology_chart["model"] + "｜" + morphology_chart["metric"]
    )
    dev_sweep = dev_sweep.copy()
    dev_sweep["candidate_label"] = dev_sweep.apply(
        lambda row: (
            f"{row['family']}｜σ={row['volatility']:.2f}, "
            f"φ={row['persistence']:.2f}"
        ),
        axis=1,
    )
    subject_table = subjects[
        [
            "iSub",
            "combined_pass_95_C1",
            "combined_calibration_fdr_q",
            "boundary_rho_posterior_mean",
            "boundary_rho_volatility_posterior_mean",
            "suffix_rho_mean",
            "suffix_rho_within_trajectory_sd_mean",
            "event_count",
            "max_event_duration",
            "trend_reversal_count",
            "curve_crps_C1",
            "curve_crps_acquisition",
            "crps_difference_C1_minus_acquisition",
        ]
    ].copy()

    source_sql = {
        "model_comparison": (
            "SELECT model, cohort, observed_pass_n, subject_n, "
            "self_expected_pass_mean, cohort_calibration_p, "
            "fdr_failure_n, mean_curve_crps, median_width_95 "
            "FROM model_comparison"
        ),
        "development_sweep": (
            "SELECT family, volatility, persistence, combined_pass_n, "
            "lower_tail_calibration_p, fdr_failure_n, mean_curve_crps, "
            "median_curve_interval_width_95, development_gate "
            "FROM development_shortlist"
        ),
        "morphology": (
            "SELECT model, metric, observed_mean, simulated_mean, "
            "absolute_gap FROM morphology_comparison"
        ),
        "paired_and_subject": (
            "SELECT iSub, curve_crps_C1, curve_crps_acquisition, "
            "boundary_rho_volatility_posterior_mean, event_count, "
            "max_event_duration, trend_reversal_count "
            "FROM subject_dynamic_diagnostics"
        ),
        "identifiability": (
            "SELECT predictor, outcome, spearman_r, "
            "bootstrap_ci95_lower, bootstrap_ci95_upper "
            "FROM volatility_associations"
        ),
    }
    source_meta = {
        "model_comparison": (
            "冻结模型的群体覆盖、锐度与数值复核。",
            [
                "dynamic_rho_consolidated/model_comparison.csv",
                "dynamic_rho_reserved_c1_p256_r1024/"
                "c1_s0p5_e0p5_v0p2_p0p95/candidate_decision.json",
                "dynamic_rho_reserved_c1_seed20261101_p128_r512/"
                "c1_s0p5_e0p5_v0p2_p0p95/candidate_decision.json",
            ],
        ),
        "development_sweep": (
            "C0/C1高精度开发短名单及预设门槛。",
            [
                "dynamic_rho_shortlist_dev_p128_r384/"
                "sweep_summary.csv",
                "dynamic_rho_shortlist_dev_p128_r384/selection.json",
            ],
        ),
        "morphology": (
            "同一24名保留被试上，单变点与C1的轨迹形态均值。",
            ["dynamic_rho_consolidated/morphology_comparison.csv"],
        ),
        "paired_and_subject": (
            "逐被试配对CRPS与C1后验/轨迹诊断。",
            [
                "dynamic_rho_consolidated/"
                "subject_dynamic_diagnostics.csv",
                "dynamic_rho_consolidated/"
                "paired_crps_comparison.json",
            ],
        ),
        "identifiability": (
            "前缀波动强度后验与真实后缀表型的描述性关联。",
            [
                "dynamic_rho_consolidated/"
                "volatility_associations.csv",
                "dynamic_rho_separability_dev/"
                "separability_summary.json",
            ],
        ),
    }
    sources = [
        source_object(
            source_id,
            {
                "model_comparison": "模型级生成充分性",
                "development_sweep": "C0/C1开发短名单",
                "morphology": "轨迹形态比较",
                "paired_and_subject": "逐被试配对诊断",
                "identifiability": "机制可识别性诊断",
            }[source_id],
            query,
            source_meta[source_id][0],
            generated_at,
            source_meta[source_id][1],
        )
        for source_id, query in source_sql.items()
    ]

    datasets = {
        "pass_chart": records(pass_chart),
        "crps_chart": records(crps_chart),
        "model_comparison": records(model_comparison),
        "development_sweep": records(dev_sweep),
        "morphology_chart": records(morphology_chart),
        "morphology": records(morphology),
        "associations": records(associations),
        "subject_table": records(subject_table),
    }

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": (
                "检验持续随机readout是否能生成复杂学习时程，并与单次掌握基准公平比较。"
            ),
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "pass_comparison",
                    "title": "联合轨迹通过比例：实际与模型自期望",
                    "subtitle": (
                        "静态模型为32人；两种动态模型为同一24人保留集"
                    ),
                    "type": "horizontalBar",
                    "dataset": "pass_chart",
                    "sourceId": "model_comparison",
                    "intent": "comparison",
                    "question": "实际通过数是否符合模型自身校准？",
                    "rationale": (
                        "把实际通过比例与每个模型自身伪观测期望并列，"
                        "避免把95%区域误当作人人必须通过。"
                    ),
                    "comparisonContext": {
                        "baseline": "模型自身重复抽样",
                        "grain": "model × seed role",
                        "unit": "subject pass fraction",
                    },
                    "encodings": {
                        "x": {
                            "field": "model_and_reference",
                            "type": "nominal",
                            "label": "模型与参照",
                        },
                        "y": {
                            "field": "pass_fraction",
                            "type": "quantitative",
                            "label": "联合通过比例",
                        },
                        "tooltip": [
                            {
                                "field": "cohort",
                                "type": "nominal",
                                "label": "队列",
                            },
                            {
                                "field": "pass_n",
                                "type": "quantitative",
                                "label": "通过人数/期望",
                            },
                        ],
                    },
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
                    "id": "crps_comparison",
                    "title": "滚动正确率曲线 CRPS",
                    "subtitle": "越低越好；动态正式比较使用同一24名保留被试",
                    "type": "horizontalBar",
                    "dataset": "crps_chart",
                    "sourceId": "model_comparison",
                    "intent": "comparison",
                    "question": "动态C1是否在整体曲线proper score上优于单变点？",
                    "rationale": (
                        "CRPS同时惩罚位置偏差与过宽预测；群体覆盖通过"
                        "并不自动意味着更低CRPS。"
                    ),
                    "comparisonContext": {
                        "baseline": "单次掌握变点",
                        "grain": "model × seed role",
                        "unit": "mean curve CRPS",
                    },
                    "encodings": {
                        "x": {
                            "field": "model",
                            "type": "nominal",
                            "label": "模型",
                        },
                        "y": {
                            "field": "mean_curve_crps",
                            "type": "quantitative",
                            "label": "平均曲线CRPS",
                        },
                        "tooltip": [
                            {
                                "field": "seed_role",
                                "type": "nominal",
                                "label": "运行角色",
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
                    "surface": {
                        "surface": "explorer",
                        "viewMode": "both",
                    },
                },
                {
                    "id": "morphology_gaps",
                    "title": "单变点与 C1 的轨迹形态绝对偏差",
                    "subtitle": "不同统计量单位不同，只在同一统计量内比较两模型",
                    "type": "horizontalBar",
                    "dataset": "morphology_chart",
                    "sourceId": "morphology",
                    "intent": "comparison",
                    "question": "C1具体改善了哪些非单调学习形态？",
                    "rationale": (
                        "逐项显示真实均值与模拟均值的绝对差，避免只用"
                        "一个通过数概括全部行为结构。"
                    ),
                    "comparisonContext": {
                        "baseline": "单次掌握变点",
                        "grain": "model × morphology statistic",
                        "unit": "metric-specific absolute gap",
                    },
                    "encodings": {
                        "x": {
                            "field": "model_and_metric",
                            "type": "nominal",
                            "label": "模型与统计量",
                        },
                        "y": {
                            "field": "absolute_gap",
                            "type": "quantitative",
                            "label": "绝对均值偏差",
                        },
                        "tooltip": [
                            {
                                "field": "observed_mean",
                                "type": "quantitative",
                                "label": "真实均值",
                            },
                            {
                                "field": "simulated_mean",
                                "type": "quantitative",
                                "label": "模拟均值",
                            },
                        ],
                    },
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
                    "id": "model_table",
                    "title": "模型级充分性与锐度",
                    "subtitle": "15项轨迹摘要＋滚动曲线联合校准",
                    "dataset": "model_comparison",
                    "sourceId": "model_comparison",
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "model", "label": "模型", "type": "text"},
                        {"field": "cohort", "label": "队列", "type": "text"},
                        {"field": "observed_pass_n", "label": "实际通过", "format": "number"},
                        {"field": "subject_n", "label": "人数", "format": "number"},
                        {"field": "self_expected_pass_mean", "label": "模型自期望", "format": "number"},
                        {"field": "self_expected_pass_q025", "label": "自期望2.5%", "format": "number"},
                        {"field": "self_expected_pass_q975", "label": "自期望97.5%", "format": "number"},
                        {"field": "cohort_calibration_p", "label": "群体p", "format": "number"},
                        {"field": "fdr_failure_n", "label": "FDR离群", "format": "number"},
                        {"field": "mean_curve_crps", "label": "CRPS", "format": "number"},
                        {"field": "median_width_95", "label": "95%带宽", "format": "number"},
                    ],
                },
                {
                    "id": "development_table",
                    "title": "高精度开发短名单",
                    "subtitle": "8名开发被试；C0/C1同一门槛",
                    "dataset": "development_sweep",
                    "sourceId": "development_sweep",
                    "density": "compact",
                    "layout": "full",
                    "columns": [
                        {"field": "family", "label": "家族", "type": "text"},
                        {"field": "volatility", "label": "σ+", "format": "number"},
                        {"field": "persistence", "label": "φ", "format": "number"},
                        {"field": "combined_pass_n", "label": "通过", "format": "number"},
                        {"field": "self_expected_pass_mean", "label": "自期望", "format": "number"},
                        {"field": "lower_tail_calibration_p", "label": "群体p", "format": "number"},
                        {"field": "fdr_failure_n", "label": "FDR离群", "format": "number"},
                        {"field": "mean_curve_crps", "label": "CRPS", "format": "number"},
                        {"field": "median_curve_interval_width_95", "label": "95%带宽", "format": "number"},
                        {"field": "development_gate", "label": "开发通过", "type": "boolean"},
                    ],
                },
                {
                    "id": "association_table",
                    "title": "个体波动强度的外部描述性关联",
                    "subtitle": "24名保留被试；所有区间均跨0",
                    "dataset": "associations",
                    "sourceId": "identifiability",
                    "density": "compact",
                    "layout": "full",
                    "columns": [
                        {"field": "outcome", "label": "真实后缀表型", "type": "text"},
                        {"field": "spearman_r", "label": "Spearman r", "format": "number"},
                        {"field": "bootstrap_ci95_lower", "label": "95%下界", "format": "number"},
                        {"field": "bootstrap_ci95_upper", "label": "95%上界", "format": "number"},
                        {"field": "confirmatory", "label": "确认性", "type": "boolean"},
                    ],
                },
                {
                    "id": "subject_table",
                    "title": "逐被试动态诊断",
                    "subtitle": "后验波动只作诊断，不作机制分型",
                    "dataset": "subject_table",
                    "sourceId": "paired_and_subject",
                    "density": "compact",
                    "layout": "full",
                    "columns": [
                        {"field": "iSub", "label": "被试", "format": "number"},
                        {"field": "combined_pass_95_C1", "label": "C1通过", "type": "boolean"},
                        {"field": "combined_calibration_fdr_q", "label": "C1 FDR q", "format": "number"},
                        {"field": "boundary_rho_posterior_mean", "label": "边界ρ", "format": "number"},
                        {"field": "boundary_rho_volatility_posterior_mean", "label": "波动后验", "format": "number"},
                        {"field": "event_count", "label": "真实事件数", "format": "number"},
                        {"field": "curve_crps_C1", "label": "C1 CRPS", "format": "number"},
                        {"field": "curve_crps_acquisition", "label": "单变点CRPS", "format": "number"},
                        {"field": "crps_difference_C1_minus_acquisition", "label": "C1-单变点", "format": "number"},
                    ],
                },
            ],
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
                    "sourceId": "model_comparison",
                    "body": (
                        "## 技术摘要\n\n"
                        "持续随机 readout C1 在冻结保留集中达到群体生成充分性："
                        f"{int(formal_decision['combined_pass_n'])}/"
                        f"{int(formal_decision['subject_n'])}人联合通过，"
                        f"模型自身期望{formal_decision['self_expected_pass_mean']:.2f}"
                        f"（95%范围[{formal_decision['self_expected_pass_q025']:.0f},"
                        f"{formal_decision['self_expected_pass_q975']:.0f}]），"
                        f"群体p={formal_decision['lower_tail_calibration_p']:.3f}，"
                        f"CRPS={formal_decision['mean_curve_crps']:.4f}，"
                        f"95%带宽={formal_decision['median_curve_interval_width_95']:.3f}。"
                        "独立种子为"
                        f"{int(seed_decision['combined_pass_n'])}/"
                        f"{int(seed_decision['subject_n'])}、"
                        f"p={seed_decision['lower_tail_calibration_p']:.3f}、"
                        f"CRPS={seed_decision['mean_curve_crps']:.4f}。"
                        "这支持C1作为复杂时程的随机生成器，但不识别真实逐试次ρ路径。\n\n"
                        "C1并未全面优于更简单的单次掌握变点。逐被试配对CRPS差"
                        f"（C1减单变点）为{paired['mean_paired_crps_difference']:.4f}，"
                        f"bootstrap 95%区间"
                        f"[{paired['subject_bootstrap_ci95'][0]:.4f},"
                        f"{paired['subject_bootstrap_ci95'][1]:.4f}]；"
                        f"仅{paired['C1_better_subject_n']}/24人由C1获得更低CRPS。"
                        "因此当前证据支持一个最小动态解释集合，而非唯一胜者。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "pass_explanation",
                    "type": "markdown",
                    "sourceId": "model_comparison",
                    "body": (
                        "## 群体生成充分性\n\n"
                        "静态full-set从第一块后自主生成仍只有8/32人通过，"
                        "而单变点与C1均回到模型自身约95%的通过率。"
                        "119号在两次C1种子中均联合未通过，但只有正式高精度"
                        "运行达到FDR；个体离群作为异质性诊断报告，不推翻群体校准。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "pass_chart_block",
                    "type": "chart",
                    "chartId": "pass_comparison",
                    "layout": "full",
                },
                {
                    "id": "model_table_block",
                    "type": "table",
                    "tableId": "model_table",
                    "layout": "full",
                },
                {
                    "id": "crps_explanation",
                    "type": "markdown",
                    "sourceId": "paired_and_subject",
                    "body": (
                        "## C1能生成复杂阶段，但proper score没有胜出\n\n"
                        "C1把真实总体正确率的均值偏差从单变点的0.0216降到"
                        "0.0024，把学习斜率偏差从0.1022降到0.0827，"
                        "把最长突降持续时间偏差从3.80 trials降到1.81。"
                        "但它过度生成最大下降，平均曲线CRPS也略高；配对区间跨0。"
                        "所以选择C1作为工作模型来自研究问题需要非单调时程，"
                        "不是数据已证明它比单变点更真实。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "crps_chart_block",
                    "type": "chart",
                    "chartId": "crps_comparison",
                    "layout": "full",
                },
                {
                    "id": "morphology_chart_block",
                    "type": "chart",
                    "chartId": "morphology_gaps",
                    "layout": "full",
                },
                {
                    "id": "model_spec",
                    "type": "markdown",
                    "sourceId": "development_sweep",
                    "body": (
                        "## 模型规格与开发冻结\n\n"
                        "证据学习继续使用静态full-set核心；只有readout concentration"
                        "随试次变化。C0只允许确定性趋势，C1增加 "
                        "$u_t=0.95u_{t-1}+\\sigma_s\\xi_t$。"
                        "冻结候选的群体中位起点与绝对第128试次均为0.5，"
                        "所以没有共享单调趋势；群体创新尺度为0.20，"
                        "起点、趋势和波动强度的log随机效应尺度为0.35、0.35、0.50，"
                        "ρ截断在[0.05,20]。高精度开发结果为"
                        f"{int(development['combined_pass_n'])}/"
                        f"{int(development['subject_n'])}、"
                        f"p={development['lower_tail_calibration_p']:.4f}、"
                        f"CRPS={development['mean_curve_crps']:.4f}、"
                        f"带宽={development['median_curve_interval_width_95']:.3f}；"
                        "C0没有候选通过。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "development_table_block",
                    "type": "table",
                    "tableId": "development_table",
                    "layout": "full",
                },
                {
                    "id": "scope_method",
                    "type": "markdown",
                    "body": (
                        "## 范围、数据与方法\n\n"
                        f"条件1共有{data_quality['condition1_row_n']}行、"
                        f"{data_quality['subject_n']}名被试；试次键重复"
                        f"{data_quality['duplicate_trial_key_n']}行，反馈与"
                        f"choice/category不一致{data_quality['feedback_mismatch_n']}行，"
                        "所需字段无缺失。8名开发被试用于候选选择，24名保留被试"
                        "只接受冻结应用。每名被试用第一块形成粒子后验；后缀固定"
                        "实际物理刺激和类别，却自主抽取知觉、选择与由选择产生的反馈。"
                        "正式C1使用256粒子、1024条后缀；独立种子复核使用128粒子、"
                        "512条后缀。旧基线缓存均以相同15项统计重新评分。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "identifiability",
                    "type": "markdown",
                    "sourceId": "identifiability",
                    "body": (
                        "## 生成通过不等于个体机制可识别\n\n"
                        "开发集C0/C1条件后缀分布的平衡识别率为"
                        f"{separability['curve_balanced_accuracy']:.3f}"
                        "（曲线）和"
                        f"{separability['summary_balanced_accuracy']:.3f}"
                        "（摘要）；C1自身伪数据的恢复率只有"
                        f"{separability['family_results'][1]['curve_recovery_rate']:.3f}"
                        "和"
                        f"{separability['family_results'][1]['summary_recovery_rate']:.3f}。"
                        "保留集中，前缀条件化波动强度与后缀五项起伏表型的相关"
                        "bootstrap区间全部跨0。故不能按后验波动强度给被试分型。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "association_table_block",
                    "type": "table",
                    "tableId": "association_table",
                    "layout": "full",
                },
                {
                    "id": "subject_table_block",
                    "type": "table",
                    "tableId": "subject_table",
                    "layout": "full",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 局限性与稳健性\n\n"
                        "C1的开发p=0.0545且带宽正好等于0.50上限，是边界性而非"
                        "富余通过；正式与独立种子保持群体结论，但个体FDR边界变化。"
                        "C1与单变点的配对CRPS差区间跨0，且开发候选扫描多于单变点，"
                        "因此不能声称预测优势。ρ波动只是一种功能性readout解释；"
                        "同样轨迹也可能来自注意、规则切换、记忆或反应偏置。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 建议的下一步\n\n"
                        "1. 冻结C1与单变点，不再调参；用层级stacking或预注册的"
                        "out-of-sample混合权重比较两者，而不强迫给每人指定唯一机制。\n"
                        "2. 把119号保留为明确个案残差，先检查一个最小的个体观测层"
                        "是否改善其预测；不得据此为全体加入三状态控制器。\n"
                        "3. 只有组合预测仍留下跨被试、可重复的突降持续时间或学习"
                        "斜率残差时，才一次加入一个跳跃或burst机制。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 仍待回答的问题\n\n"
                        "- C1的形态改善能否在不增加带宽的条件下降低独立预测CRPS？\n"
                        "- 119号的残差是反应偏置、偶发lapse还是证据学习异常，当前"
                        "单条轨迹不能区分。\n"
                        "- 条件2和3是否需要动态readout，必须各自开发与冻结，不能"
                        "从条件1外推。"
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
                "dataset_rows": {
                    key: len(value) for key, value in datasets.items()
                },
                "formal_pass_n": int(
                    formal_decision["combined_pass_n"]
                ),
                "paired_crps_difference": float(
                    paired["mean_paired_crps_difference"]
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
