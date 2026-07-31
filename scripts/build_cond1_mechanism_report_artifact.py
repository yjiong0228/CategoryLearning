#!/usr/bin/env python3
"""Build the canonical portable-report artifact for mechanism heterogeneity."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


FAMILY_LABELS = {
    "F": "反馈敏感性 F",
    "M": "记忆稳定性 M",
    "H": "有限假设集 H",
    "P": "规则可塑性 P",
    "S": "交换敏感性 S",
    "BASE": "基础模型",
    "GLOBAL": "跨机制预测库",
}
METRIC_LABELS = {
    "curve_crps": "曲线 CRPS",
    "summary_discrepancy": "轨迹摘要偏差",
    "combined_calibration_p": "联合校准 p",
    "curve_pointwise_interval_width_95": "点状 95% 区间宽度",
    "delta_suffix_nll": "后缀 NLL 差",
    "delta_rt_surprise_spearman": "RT—惊讶度相关差",
    "delta_oral_center_similarity": "口头中心相似度差",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-a", type=Path, required=True)
    parser.add_argument("--dev-b", type=Path, required=True)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--within-recovery", type=Path, required=True)
    parser.add_argument("--cross-recovery", type=Path, nargs="+", required=True)
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--reserved", type=Path, required=True)
    parser.add_argument("--reserved-confirm", type=Path, required=True)
    parser.add_argument("--global-reserved", type=Path, required=True)
    parser.add_argument("--global-external", type=Path, required=True)
    parser.add_argument("--static-global-vs-c1", type=Path, required=True)
    parser.add_argument("--external-dev", type=Path, required=True)
    parser.add_argument("--external-reserved", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records"))


def _beneficial(frame: pd.DataFrame) -> pd.Series:
    direction = np.where(
        frame["better_direction"].eq("negative"),
        frame["ci975"] < 0.0,
        frame["ci025"] > 0.0,
    )
    if "paired_signflip_q" in frame.columns:
        direction = direction & frame["paired_signflip_q"].lt(0.05)
    return direction


def _harmful(frame: pd.DataFrame) -> pd.Series:
    direction = np.where(
        frame["better_direction"].eq("negative"),
        frame["ci025"] > 0.0,
        frame["ci975"] < 0.0,
    )
    if "paired_signflip_q" in frame.columns:
        direction = direction & frame["paired_signflip_q"].lt(0.05)
    return direction


def _source(source_id: str, label: str, path: Path) -> dict[str, Any]:
    source_path = str(path)
    sql_path = source_path.replace("'", "''")
    return {
        "id": source_id,
        "label": label,
        "path": source_path,
        "query": {
            "engine": "duckdb",
            "sql": f"SELECT * FROM read_csv_auto('{sql_path}')",
            "description": f"读取报告所用的冻结结果文件：{source_path}",
            "tables_used": [source_path],
        },
    }


def main() -> None:
    args = parse_args()
    generated_at = datetime.now(timezone.utc).isoformat()

    replication = pd.read_csv(args.replication / "suffix_metric_replication.csv")
    dev_a = pd.read_csv(args.dev_a / "comparison_summary.csv")
    strategy = pd.read_csv(args.strategy / "comparison_summary.csv")
    within = pd.read_csv(args.within_recovery / "recovery_summary.csv")
    cross = pd.concat(
        [pd.read_csv(path / "recovered_datasets.csv") for path in args.cross_recovery],
        ignore_index=True,
    )
    reserved = pd.read_csv(args.reserved / "comparison_summary.csv")
    reserved_subject = pd.read_csv(args.reserved / "mixture_subject_summary.csv")
    reserved_coverage_test = pd.read_csv(args.reserved / "coverage_comparison.csv")
    reserved_confirm = pd.read_csv(args.reserved_confirm / "comparison_summary.csv")
    global_reserved = pd.read_csv(args.global_reserved / "comparison_summary.csv")
    global_coverage = pd.read_csv(args.global_reserved / "coverage_comparison.csv")
    global_external = pd.read_csv(args.global_external / "group_validation.csv")
    direct_comparison = pd.read_csv(
        args.static_global_vs_c1 / "metric_comparison.csv"
    )
    direct_coverage = pd.read_csv(
        args.static_global_vs_c1 / "coverage_comparison.csv"
    )
    external_dev = pd.read_csv(args.external_dev / "group_validation.csv")
    external_reserved = pd.read_csv(args.external_reserved / "group_validation.csv")

    reserved["beneficial"] = _beneficial(reserved)
    reserved["harmful"] = _harmful(reserved)
    reserved_confirm["beneficial"] = _beneficial(reserved_confirm)
    global_reserved["beneficial"] = _beneficial(global_reserved)
    global_reserved["harmful"] = _harmful(global_reserved)
    global_external["beneficial"] = _beneficial(global_external)
    global_external["harmful"] = _harmful(global_external)
    external_reserved["beneficial"] = _beneficial(external_reserved)
    external_reserved["harmful"] = _harmful(external_reserved)

    dev_summary = replication.loc[
        replication["metric"].eq("summary_discrepancy")
    ].copy()
    dev_chart = pd.concat(
        [
            dev_summary.assign(
                seed="随机种子 A",
                delta=dev_summary["mean_seed_a"],
            ),
            dev_summary.assign(
                seed="随机种子 B",
                delta=dev_summary["mean_seed_b"],
            ),
        ],
        ignore_index=True,
    )
    dev_chart["candidate"] = (
        dev_chart["readout"].map({"static": "静态", "c1": "C1"})
        + " / "
        + dev_chart["family"].map(FAMILY_LABELS)
    )
    dev_chart = dev_chart[["candidate", "seed", "delta"]]

    family_candidate_n = {"F": 5, "M": 5, "H": 4, "P": 5}
    within_chart = []
    for row in within.to_dict(orient="records"):
        family = str(row["true_family"])
        within_chart.extend(
            [
                {
                    "family": FAMILY_LABELS[family],
                    "series": "观察恢复率",
                    "accuracy": float(row["exact_candidate_accuracy"]),
                },
                {
                    "family": FAMILY_LABELS[family],
                    "series": "机会水平",
                    "accuracy": 1.0 / family_candidate_n[family],
                },
            ]
        )
    within_chart = pd.DataFrame(within_chart)

    cross_nonbase = cross.loc[cross["true_family"].ne("BASE")].copy()
    cross_summary = (
        cross_nonbase.groupby("true_family", sort=True)["family_recovered"]
        .agg([("dataset_n", "size"), ("family_accuracy", "mean")])
        .reset_index()
    )
    cross_chart = []
    for row in cross_summary.to_dict(orient="records"):
        cross_chart.extend(
            [
                {
                    "family": FAMILY_LABELS[str(row["true_family"])],
                    "series": "观察分类率",
                    "accuracy": float(row["family_accuracy"]),
                    "dataset_n": int(row["dataset_n"]),
                },
                {
                    "family": FAMILY_LABELS[str(row["true_family"])],
                    "series": "机会水平",
                    "accuracy": 0.20,
                    "dataset_n": int(row["dataset_n"]),
                },
            ]
        )
    cross_chart = pd.DataFrame(cross_chart)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cross_summary.to_csv(
        args.output.parent / "cross_recovery_combined.csv", index=False
    )
    cross_confusion = (
        pd.crosstab(
            cross["true_family"],
            cross["predicted_family"],
            normalize="index",
        )
        .rename_axis(index="true_family", columns="predicted_family")
        .stack(future_stack=True)
        .rename("proportion")
        .reset_index()
    )
    cross_confusion["true_family"] = cross_confusion["true_family"].map(
        FAMILY_LABELS
    )
    cross_confusion["predicted_family"] = cross_confusion[
        "predicted_family"
    ].map(FAMILY_LABELS)
    cross_confusion.to_csv(
        args.output.parent / "cross_recovery_confusion.csv", index=False
    )

    strategy_table = strategy.copy()
    strategy_table["readout_label"] = strategy_table["readout"].map(
        {"static": "静态", "c1": "C1"}
    )
    strategy_table["metric_label"] = strategy_table["metric"].map(METRIC_LABELS)
    strategy_table["interpretation"] = np.where(
        _beneficial(strategy_table),
        "显著改善",
        np.where(_harmful(strategy_table), "显著恶化", "不确定"),
    )

    reserved_table = reserved.copy()
    reserved_table["readout_label"] = reserved_table["readout"].map(
        {"static": "静态", "c1": "C1"}
    )
    reserved_table["family_label"] = reserved_table["family"].map(FAMILY_LABELS)
    reserved_table["metric_label"] = reserved_table["metric"].map(METRIC_LABELS)
    reserved_table["interpretation"] = np.where(
        reserved_table["beneficial"],
        "显著改善",
        np.where(reserved_table["harmful"], "显著恶化", "不确定"),
    )
    reserved_summary_chart = reserved_table.loc[
        reserved_table["metric"].eq("summary_discrepancy"),
        ["readout_label", "family_label", "mean"],
    ].rename(columns={"readout_label": "readout", "family_label": "family", "mean": "delta"})
    reserved_coverage = (
        reserved_subject.groupby(["readout", "family", "model"], sort=True)[
            "combined_pass_95"
        ]
        .agg([("pass_n", "sum"), ("subject_n", "size"), ("pass_rate", "mean")])
        .reset_index()
    )
    reserved_coverage["candidate"] = (
        reserved_coverage["readout"].map({"static": "静态", "c1": "C1"})
        + " / "
        + reserved_coverage["family"].map(FAMILY_LABELS)
    )
    reserved_coverage["model_label"] = reserved_coverage["model"].map(
        {
            "candidate_bank_mixture": "个体候选混合",
            "reference_candidate": "参考模型",
        }
    )
    reserved_coverage_test["readout_label"] = reserved_coverage_test["readout"].map(
        {"static": "静态", "c1": "C1"}
    )
    reserved_coverage_test["family_label"] = reserved_coverage_test["family"].map(
        FAMILY_LABELS
    )

    confirm_table = reserved_confirm.copy()
    confirm_table["readout_label"] = confirm_table["readout"].map(
        {"static": "静态", "c1": "C1"}
    )
    confirm_table["metric_label"] = confirm_table["metric"].map(METRIC_LABELS)
    confirm_table["interpretation"] = np.where(
        confirm_table["beneficial"], "显著改善", "未确认改善"
    )

    global_table = global_reserved.copy()
    global_table["readout_label"] = global_table["readout"].map(
        {"static": "静态", "c1": "C1"}
    )
    global_table["metric_label"] = global_table["metric"].map(METRIC_LABELS)
    global_table["interpretation"] = np.where(
        global_table["beneficial"],
        "显著改善",
        np.where(global_table["harmful"], "显著恶化", "不确定"),
    )
    global_coverage_chart = pd.concat(
        [
            global_coverage.assign(
                model_label="跨机制预测库",
                pass_n=global_coverage["mixture_pass_n"],
            ),
            global_coverage.assign(
                model_label="C1/静态参考",
                pass_n=global_coverage["reference_pass_n"],
            ),
        ],
        ignore_index=True,
    )
    global_coverage_chart["readout_label"] = global_coverage_chart["readout"].map(
        {"static": "静态", "c1": "C1"}
    )
    global_coverage_chart["pass_rate"] = (
        global_coverage_chart["pass_n"] / global_coverage_chart["subject_n"]
    )
    global_external_table = global_external.copy()
    global_external_table["metric_label"] = global_external_table["metric"].map(
        METRIC_LABELS
    )
    global_external_table["interpretation"] = np.where(
        global_external_table["beneficial"],
        "显著改善",
        np.where(global_external_table["harmful"], "显著恶化", "不确定"),
    )
    direct_comparison["metric_label"] = direct_comparison["metric"].map(
        METRIC_LABELS
    )
    direct_comparison["interpretation"] = np.where(
        (
            np.where(
                direct_comparison["better_direction"].eq("negative"),
                direct_comparison["ci975"] < 0.0,
                direct_comparison["ci025"] > 0.0,
            )
            & direct_comparison["paired_signflip_q"].lt(0.05)
        ),
        "静态跨机制库更好",
        "没有可靠差异",
    )

    external_table = external_reserved.copy()
    external_table["family_label"] = external_table["family"].map(FAMILY_LABELS)
    external_table["metric_label"] = external_table["metric"].map(METRIC_LABELS)
    external_table["interpretation"] = np.where(
        external_table["beneficial"],
        "显著改善",
        np.where(external_table["harmful"], "显著恶化", "不确定"),
    )
    external_chart = external_table.loc[
        external_table["metric"].isin(
            ["delta_suffix_nll", "delta_oral_center_similarity"]
        ),
        ["family_label", "metric_label", "mean"],
    ].rename(columns={"family_label": "family", "metric_label": "metric", "mean": "delta"})
    external_nll_chart = external_chart.loc[
        external_chart["metric"].eq(METRIC_LABELS["delta_suffix_nll"]),
        ["family", "delta"],
    ]
    external_oral_chart = external_chart.loc[
        external_chart["metric"].eq(
            METRIC_LABELS["delta_oral_center_similarity"]
        ),
        ["family", "delta"],
    ]

    stable_dev_n = int(replication["both_ci_exclude_zero"].sum())
    capacity_recovery = float(
        within.loc[within["true_family"].eq("H"), "exact_candidate_accuracy"].iloc[0]
    )
    pooled_cross_accuracy = float(cross_nonbase["family_recovered"].mean())
    reserved_beneficial_n = int(reserved["beneficial"].sum())
    external_beneficial_n = int(external_reserved["beneficial"].sum())
    global_c1 = global_coverage.loc[global_coverage["readout"].eq("c1")].iloc[0]
    global_c1_gain = int(global_c1["mixture_pass_n"] - global_c1["reference_pass_n"])
    global_c1_q = float(global_c1["exact_q"])
    global_c1_calibration = global_reserved.loc[
        global_reserved["readout"].eq("c1")
        & global_reserved["metric"].eq("combined_calibration_p")
    ].iloc[0]
    global_c1_width = global_reserved.loc[
        global_reserved["readout"].eq("c1")
        & global_reserved["metric"].eq("curve_pointwise_interval_width_95")
    ].iloc[0]
    direct_width = direct_comparison.loc[
        direct_comparison["metric"].eq("curve_pointwise_interval_width_95")
    ].iloc[0]
    direct_coverage_row = direct_coverage.iloc[0]
    global_external_lookup = global_external.set_index("metric")
    global_external_nll = global_external_lookup.loc["delta_suffix_nll"]
    global_external_rt = global_external_lookup.loc["delta_rt_surprise_spearman"]
    global_external_oral = global_external_lookup.loc[
        "delta_oral_center_similarity"
    ]

    sources = [
        _source("dev_screen", "开发集双随机种子机制筛查", args.replication / "suffix_metric_replication.csv"),
        _source("within_recovery", "家族内参数恢复", args.within_recovery / "recovery_summary.csv"),
        _source("cross_recovery", "两次跨机制恢复的合并混淆矩阵", args.output.parent / "cross_recovery_confusion.csv"),
        _source("strategy_screen", "容量 3 下交换敏感性筛查", args.strategy / "comparison_summary.csv"),
        _source("reserved_screen", "24 名保留被试冻结先验检验", args.reserved / "comparison_summary.csv"),
        _source("reserved_coverage_test", "24 名保留被试成对覆盖检验", args.reserved / "coverage_comparison.csv"),
        _source("reserved_confirm", "有限容量高精度确认运行", args.reserved_confirm / "comparison_summary.csv"),
        _source("global_reserved", "保留集跨机制预测库比较", args.global_reserved / "comparison_summary.csv"),
        _source("global_coverage", "保留集跨机制预测库覆盖", args.global_reserved / "coverage_comparison.csv"),
        _source("global_external", "跨机制预测库的 RT 与口头报告外部验证", args.global_external / "group_validation.csv"),
        _source("direct_global_c1", "静态跨机制预测库与冻结 C1 的直接比较", args.static_global_vs_c1 / "metric_comparison.csv"),
        _source("direct_global_c1_coverage", "静态跨机制预测库与冻结 C1 的覆盖比较", args.static_global_vs_c1 / "coverage_comparison.csv"),
        _source("external_validation", "保留集 RT 与口头报告外部验证", args.external_reserved / "group_validation.csv"),
    ]

    datasets = {
        "headline": [
            {
                "stable_dev_effects": stable_dev_n,
                "capacity_recovery": capacity_recovery,
                "cross_family_accuracy": pooled_cross_accuracy,
                "reserved_beneficial_effects": reserved_beneficial_n,
                "external_beneficial_effects": external_beneficial_n,
                "global_c1_coverage_gain": global_c1_gain,
            }
        ],
        "dev_summary_effects": _records(dev_chart),
        "within_recovery": _records(within_chart),
        "cross_recovery": _records(cross_chart),
        "cross_recovery_confusion": _records(cross_confusion),
        "within_recovery_table": _records(within),
        "strategy_table": _records(
            strategy_table[
                ["readout_label", "metric_label", "mean", "ci025", "ci975", "paired_signflip_q", "interpretation"]
            ]
        ),
        "reserved_summary_effects": _records(reserved_summary_chart),
        "reserved_coverage": _records(
            reserved_coverage[
                ["candidate", "model_label", "pass_n", "subject_n", "pass_rate"]
            ]
        ),
        "reserved_table": _records(
            reserved_table[
                [
                    "readout_label",
                    "family_label",
                    "metric_label",
                    "mean",
                    "ci025",
                    "ci975",
                    "paired_signflip_q",
                    "improved_subject_n",
                    "subject_n",
                    "interpretation",
                ]
            ]
        ),
        "reserved_coverage_test": _records(
            reserved_coverage_test[
                [
                    "readout_label",
                    "family_label",
                    "mixture_pass_n",
                    "reference_pass_n",
                    "subject_n",
                    "improved_n",
                    "worsened_n",
                    "exact_q",
                ]
            ]
        ),
        "confirm_table": _records(
            confirm_table[
                ["readout_label", "metric_label", "mean", "ci025", "ci975", "paired_signflip_q", "interpretation"]
            ]
        ),
        "global_coverage": _records(
            global_coverage_chart[
                ["readout_label", "model_label", "pass_n", "subject_n", "pass_rate"]
            ]
        ),
        "global_table": _records(
            global_table[
                [
                    "readout_label",
                    "metric_label",
                    "mean",
                    "ci025",
                    "ci975",
                    "paired_signflip_q",
                    "improved_subject_n",
                    "subject_n",
                    "interpretation",
                ]
            ]
        ),
        "global_external_table": _records(
            global_external_table[
                [
                    "metric_label",
                    "mean",
                    "ci025",
                    "ci975",
                    "paired_signflip_q",
                    "subject_n",
                    "interpretation",
                ]
            ]
        ),
        "direct_global_c1_table": _records(
            direct_comparison[
                [
                    "metric_label",
                    "mean",
                    "ci025",
                    "ci975",
                    "global_better_subject_n",
                    "subject_n",
                    "paired_signflip_q",
                    "interpretation",
                ]
            ]
        ),
        "direct_global_c1_coverage": _records(direct_coverage),
        "external_nll_effects": _records(external_nll_chart),
        "external_oral_effects": _records(external_oral_chart),
        "external_table": _records(
            external_table[
                ["family_label", "metric_label", "mean", "ci025", "ci975", "paired_signflip_q", "subject_n", "interpretation"]
            ]
        ),
    }

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "条件 1 个体异质性机制：分阶段预测与可识别性检验",
            "description": "反馈敏感性、记忆稳定性、有限假设容量、规则可塑性与策略交换的开发筛查、恢复、保留集和外部验证。",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "dev_effects",
                    "description": "24 个开发集比较中，在两个蒙特卡洛种子下均排除零的效应数。",
                    "dataset": "headline",
                    "sourceId": "dev_screen",
                    "metrics": [{"label": "稳定开发效应", "field": "stable_dev_effects", "format": "number"}],
                },
                {
                    "id": "capacity_recovery",
                    "description": "有限容量候选的精确参数恢复率；四候选机会水平同为 25%。",
                    "dataset": "headline",
                    "sourceId": "within_recovery",
                    "metrics": [{"label": "容量精确恢复率", "field": "capacity_recovery", "format": "percent"}],
                },
                {
                    "id": "cross_accuracy",
                    "description": "两次跨机制恢复中，排除基础模型后的加权机制家族分类率；机会水平为 20%。",
                    "dataset": "headline",
                    "sourceId": "cross_recovery",
                    "metrics": [{"label": "跨机制分类率", "field": "cross_family_accuracy", "format": "percent"}],
                },
                {
                    "id": "reserved_effects",
                    "description": "保留集上相对家族参考模型达到预设有利方向、bootstrap 区间排除零且 sign-flip FDR q<0.05 的比较数。",
                    "dataset": "headline",
                    "sourceId": "reserved_screen",
                    "metrics": [{"label": "保留集有利效应", "field": "reserved_beneficial_effects", "format": "number"}],
                },
                {
                    "id": "external_effects",
                    "description": "保留集一步预测、RT 和口头报告中达到有利方向、区间排除零且 sign-flip FDR q<0.05 的比较数。",
                    "dataset": "headline",
                    "sourceId": "external_validation",
                    "metrics": [{"label": "外部验证有利效应", "field": "external_beneficial_effects", "format": "number"}],
                },
                {
                    "id": "global_c1_gain",
                    "description": "保留集上跨机制预测库相对冻结 C1 的联合 95% 覆盖人数差。",
                    "dataset": "headline",
                    "sourceId": "global_coverage",
                    "metrics": [{"label": "C1 覆盖人数差", "field": "global_c1_coverage_gain", "format": "number", "signed": True}],
                },
            ],
            "charts": [
                {
                    "id": "dev_summary_chart",
                    "title": "开发集轨迹摘要偏差：双随机种子",
                    "subtitle": "候选混合减参考模型；负值表示改善，8 名开发被试。",
                    "type": "bar",
                    "dataset": "dev_summary_effects",
                    "sourceId": "dev_screen",
                    "encodings": {
                        "x": {"field": "candidate", "type": "nominal", "label": "读出 / 机制"},
                        "y": {"field": "delta", "type": "quantitative", "label": "平均差值"},
                        "color": {"field": "seed", "type": "nominal", "label": "蒙特卡洛复核"},
                    },
                    "yAxisTitle": "混合 − 参考",
                    "valueFormat": "number",
                    "palette": {"kind": "categorical", "name": "blue-orange"},
                    "labels": {"values": "auto"},
                    "referenceLines": [{"value": 0, "axis": "y", "color": "neutral", "lineStyle": "dashed", "label": "无差异"}],
                    "settings": {"categoryLabelPolicy": "wrap"},
                    "layout": "full",
                },
                {
                    "id": "within_recovery_chart",
                    "title": "家族内候选参数恢复",
                    "subtitle": "在真实刺激顺序生成合成轨迹；观察精确恢复率与离散候选机会水平。",
                    "type": "bar",
                    "dataset": "within_recovery",
                    "sourceId": "within_recovery",
                    "encodings": {
                        "x": {"field": "family", "type": "nominal", "label": "机制家族"},
                        "y": {"field": "accuracy", "type": "quantitative", "label": "精确恢复率", "format": "percent"},
                        "color": {"field": "series", "type": "nominal", "label": "比较"},
                    },
                    "yAxisTitle": "恢复率",
                    "valueFormat": "percent",
                    "palette": {"kind": "categorical", "name": "blue-orange"},
                    "labels": {"values": "auto"},
                    "layout": "full",
                },
                {
                    "id": "cross_recovery_chart",
                    "title": "跨机制家族恢复混淆矩阵",
                    "subtitle": "合并 32 与 64 粒子独立运行；每行归一化为 100%，对角线才是正确分类。",
                    "type": "heatmap",
                    "dataset": "cross_recovery_confusion",
                    "sourceId": "cross_recovery",
                    "encodings": {
                        "x": {"field": "predicted_family", "type": "nominal", "label": "恢复出的机制"},
                        "y": {"field": "proportion", "type": "quantitative", "label": "行比例", "format": "percent"},
                        "color": {"field": "true_family", "type": "nominal", "label": "真实生成机制"},
                    },
                    "yAxisTitle": "真实生成机制",
                    "valueFormat": "percent",
                    "palette": {"kind": "sequential", "name": "blue"},
                    "labels": {"values": "all"},
                    "layout": "full",
                },
                {
                    "id": "reserved_coverage_chart",
                    "title": "保留集联合 95% 轨迹通过率",
                    "subtitle": "逐被试联合摘要与曲线门槛；每个读出/机制均为 24 名冻结保留被试。",
                    "type": "bar",
                    "dataset": "reserved_coverage",
                    "sourceId": "reserved_screen",
                    "encodings": {
                        "x": {"field": "candidate", "type": "nominal", "label": "读出 / 机制"},
                        "y": {"field": "pass_rate", "type": "quantitative", "label": "通过率", "format": "percent"},
                        "color": {"field": "model_label", "type": "nominal", "label": "模型"},
                        "tooltip": [
                            {"field": "pass_n", "type": "quantitative", "label": "通过人数"},
                            {"field": "subject_n", "type": "quantitative", "label": "总人数"},
                        ],
                    },
                    "yAxisTitle": "联合通过率",
                    "valueFormat": "percent",
                    "palette": {"kind": "categorical", "name": "blue-orange"},
                    "labels": {"values": "auto"},
                    "settings": {"categoryLabelPolicy": "wrap"},
                    "layout": "full",
                },
                {
                    "id": "reserved_summary_chart",
                    "title": "保留集轨迹摘要偏差",
                    "subtitle": "冻结开发集先验；候选混合减参考模型，负值表示改善，24 名保留被试。",
                    "type": "bar",
                    "dataset": "reserved_summary_effects",
                    "sourceId": "reserved_screen",
                    "encodings": {
                        "x": {"field": "family", "type": "nominal", "label": "机制家族"},
                        "y": {"field": "delta", "type": "quantitative", "label": "平均差值"},
                        "color": {"field": "readout", "type": "nominal", "label": "选择读出"},
                    },
                    "yAxisTitle": "混合 − 参考",
                    "valueFormat": "number",
                    "palette": {"kind": "categorical", "name": "blue-orange"},
                    "labels": {"values": "auto"},
                    "referenceLines": [{"value": 0, "axis": "y", "color": "neutral", "lineStyle": "dashed", "label": "无差异"}],
                    "layout": "full",
                },
                {
                    "id": "global_coverage_chart",
                    "title": "保留集跨机制预测库覆盖",
                    "subtitle": "开发集群体先验冻结；每名保留被试只用前缀更新候选权重。",
                    "type": "bar",
                    "dataset": "global_coverage",
                    "sourceId": "global_coverage",
                    "encodings": {
                        "x": {"field": "readout_label", "type": "nominal", "label": "选择读出"},
                        "y": {"field": "pass_rate", "type": "quantitative", "label": "联合通过率", "format": "percent"},
                        "color": {"field": "model_label", "type": "nominal", "label": "模型"},
                        "tooltip": [
                            {"field": "pass_n", "type": "quantitative", "label": "通过人数"},
                            {"field": "subject_n", "type": "quantitative", "label": "总人数"},
                        ],
                    },
                    "yAxisTitle": "联合通过率",
                    "valueFormat": "percent",
                    "palette": {"kind": "categorical", "name": "blue-orange"},
                    "labels": {"values": "all"},
                    "layout": "full",
                },
                {
                    "id": "external_nll_chart",
                    "title": "保留集一步选择预测",
                    "subtitle": "后缀 NLL：候选混合减参考模型；负值表示改善。",
                    "type": "bar",
                    "dataset": "external_nll_effects",
                    "sourceId": "external_validation",
                    "encodings": {
                        "x": {"field": "family", "type": "nominal", "label": "机制家族"},
                        "y": {"field": "delta", "type": "quantitative", "label": "平均差值"},
                    },
                    "yAxisTitle": "混合 − 参考",
                    "valueFormat": "number",
                    "palette": {"kind": "sequential", "name": "blue"},
                    "labels": {"values": "all"},
                    "referenceLines": [{"value": 0, "axis": "y", "color": "neutral", "lineStyle": "dashed", "label": "无差异"}],
                    "layout": "full",
                },
                {
                    "id": "external_oral_chart",
                    "title": "保留集口头规则表征一致性",
                    "subtitle": "口头中心相似度：候选混合减参考模型；正值表示改善。",
                    "type": "bar",
                    "dataset": "external_oral_effects",
                    "sourceId": "external_validation",
                    "encodings": {
                        "x": {"field": "family", "type": "nominal", "label": "机制家族"},
                        "y": {"field": "delta", "type": "quantitative", "label": "平均差值"},
                    },
                    "yAxisTitle": "混合 − 参考",
                    "valueFormat": "number",
                    "palette": {"kind": "sequential", "name": "blue"},
                    "labels": {"values": "all"},
                    "referenceLines": [{"value": 0, "axis": "y", "color": "neutral", "lineStyle": "dashed", "label": "无差异"}],
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "within_table",
                    "title": "家族内恢复统计",
                    "dataset": "within_recovery_table",
                    "sourceId": "within_recovery",
                    "columns": [
                        {"field": "true_family", "label": "家族", "type": "text"},
                        {"field": "dataset_n", "label": "合成数据数", "format": "number"},
                        {"field": "exact_candidate_accuracy", "label": "精确恢复率", "format": "percent"},
                        {"field": "spearman_true_posterior_mean", "label": "真实值—后验均值 ρ", "format": "number"},
                        {"field": "mean_true_candidate_posterior", "label": "真实候选平均后验", "format": "percent"},
                    ],
                },
                {
                    "id": "strategy_table",
                    "title": "容量 3 下交换敏感性个体化",
                    "dataset": "strategy_table",
                    "sourceId": "strategy_screen",
                    "columns": [
                        {"field": "readout_label", "label": "读出", "type": "text"},
                        {"field": "metric_label", "label": "指标", "type": "text"},
                        {"field": "mean", "label": "平均差", "format": "number"},
                        {"field": "ci025", "label": "CI 2.5%", "format": "number"},
                        {"field": "ci975", "label": "CI 97.5%", "format": "number"},
                        {"field": "paired_signflip_q", "label": "FDR q", "format": "number"},
                        {"field": "interpretation", "label": "判定", "type": "text"},
                    ],
                },
                {
                    "id": "reserved_table",
                    "title": "保留集全部冻结比较",
                    "dataset": "reserved_table",
                    "sourceId": "reserved_screen",
                    "columns": [
                        {"field": "readout_label", "label": "读出", "type": "text"},
                        {"field": "family_label", "label": "机制", "type": "text"},
                        {"field": "metric_label", "label": "指标", "type": "text"},
                        {"field": "mean", "label": "平均差", "format": "number"},
                        {"field": "ci025", "label": "CI 2.5%", "format": "number"},
                        {"field": "ci975", "label": "CI 97.5%", "format": "number"},
                        {"field": "improved_subject_n", "label": "改善人数", "format": "number"},
                        {"field": "subject_n", "label": "总人数", "format": "number"},
                        {"field": "paired_signflip_q", "label": "FDR q", "format": "number"},
                        {"field": "interpretation", "label": "判定", "type": "text"},
                    ],
                    "layout": "full",
                },
                {
                    "id": "reserved_coverage_test",
                    "title": "保留集逐被试成对覆盖检验",
                    "dataset": "reserved_coverage_test",
                    "sourceId": "reserved_coverage_test",
                    "columns": [
                        {"field": "readout_label", "label": "读出", "type": "text"},
                        {"field": "family_label", "label": "机制", "type": "text"},
                        {"field": "mixture_pass_n", "label": "混合通过", "format": "number"},
                        {"field": "reference_pass_n", "label": "参考通过", "format": "number"},
                        {"field": "subject_n", "label": "总人数", "format": "number"},
                        {"field": "improved_n", "label": "失败→通过", "format": "number"},
                        {"field": "worsened_n", "label": "通过→失败", "format": "number"},
                        {"field": "exact_q", "label": "覆盖 FDR q", "format": "number"},
                    ],
                    "layout": "full",
                },
                {
                    "id": "confirm_table",
                    "title": "有限容量高精度确认",
                    "dataset": "confirm_table",
                    "sourceId": "reserved_confirm",
                    "columns": [
                        {"field": "readout_label", "label": "读出", "type": "text"},
                        {"field": "metric_label", "label": "指标", "type": "text"},
                        {"field": "mean", "label": "平均差", "format": "number"},
                        {"field": "ci025", "label": "CI 2.5%", "format": "number"},
                        {"field": "ci975", "label": "CI 97.5%", "format": "number"},
                        {"field": "paired_signflip_q", "label": "FDR q", "format": "number"},
                        {"field": "interpretation", "label": "判定", "type": "text"},
                    ],
                },
                {
                    "id": "external_table",
                    "title": "保留集 RT 与口头报告外部验证",
                    "dataset": "external_table",
                    "sourceId": "external_validation",
                    "columns": [
                        {"field": "family_label", "label": "机制", "type": "text"},
                        {"field": "metric_label", "label": "指标", "type": "text"},
                        {"field": "mean", "label": "平均差", "format": "number"},
                        {"field": "ci025", "label": "CI 2.5%", "format": "number"},
                        {"field": "ci975", "label": "CI 97.5%", "format": "number"},
                        {"field": "paired_signflip_q", "label": "FDR q", "format": "number"},
                        {"field": "subject_n", "label": "被试数", "format": "number"},
                        {"field": "interpretation", "label": "判定", "type": "text"},
                    ],
                    "layout": "full",
                },
                {
                    "id": "global_table",
                    "title": "保留集跨机制预测库的后缀比较",
                    "dataset": "global_table",
                    "sourceId": "global_reserved",
                    "columns": [
                        {"field": "readout_label", "label": "读出", "type": "text"},
                        {"field": "metric_label", "label": "指标", "type": "text"},
                        {"field": "mean", "label": "平均差", "format": "number"},
                        {"field": "ci025", "label": "CI 2.5%", "format": "number"},
                        {"field": "ci975", "label": "CI 97.5%", "format": "number"},
                        {"field": "improved_subject_n", "label": "改善人数", "format": "number"},
                        {"field": "subject_n", "label": "总人数", "format": "number"},
                        {"field": "paired_signflip_q", "label": "FDR q", "format": "number"},
                        {"field": "interpretation", "label": "判定", "type": "text"},
                    ],
                    "layout": "full",
                },
                {
                    "id": "global_external_table",
                    "title": "跨机制预测库的独立外部约束（静态读出）",
                    "dataset": "global_external_table",
                    "sourceId": "global_external",
                    "columns": [
                        {"field": "metric_label", "label": "指标", "type": "text"},
                        {"field": "mean", "label": "平均差", "format": "number"},
                        {"field": "ci025", "label": "CI 2.5%", "format": "number"},
                        {"field": "ci975", "label": "CI 97.5%", "format": "number"},
                        {"field": "paired_signflip_q", "label": "FDR q", "format": "number"},
                        {"field": "subject_n", "label": "被试数", "format": "number"},
                        {"field": "interpretation", "label": "判定", "type": "text"},
                    ],
                    "layout": "full",
                },
                {
                    "id": "direct_global_c1_table",
                    "title": "静态跨机制预测库与冻结 C1：同一保留集直接比较",
                    "dataset": "direct_global_c1_table",
                    "sourceId": "direct_global_c1",
                    "columns": [
                        {"field": "metric_label", "label": "指标", "type": "text"},
                        {"field": "mean", "label": "静态库 − C1", "format": "number"},
                        {"field": "ci025", "label": "CI 2.5%", "format": "number"},
                        {"field": "ci975", "label": "CI 97.5%", "format": "number"},
                        {"field": "global_better_subject_n", "label": "静态库更好人数", "format": "number"},
                        {"field": "subject_n", "label": "总人数", "format": "number"},
                        {"field": "paired_signflip_q", "label": "FDR q", "format": "number"},
                        {"field": "interpretation", "label": "判定", "type": "text"},
                    ],
                    "layout": "full",
                },
                {
                    "id": "direct_global_c1_coverage_table",
                    "title": "静态跨机制预测库与冻结 C1：联合覆盖",
                    "dataset": "direct_global_c1_coverage",
                    "sourceId": "direct_global_c1_coverage",
                    "columns": [
                        {"field": "global_pass_n", "label": "静态库通过", "format": "number"},
                        {"field": "c1_pass_n", "label": "C1 通过", "format": "number"},
                        {"field": "subject_n", "label": "总人数", "format": "number"},
                        {"field": "global_only_pass_n", "label": "仅静态库通过", "format": "number"},
                        {"field": "c1_only_pass_n", "label": "仅 C1 通过", "format": "number"},
                        {"field": "exact_p", "label": "精确 p", "format": "number"},
                    ],
                    "layout": "full",
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# 条件 1 个体异质性机制：分阶段预测与可识别性检验"},
                {
                    "id": "answer",
                    "type": "markdown",
                    "body": "## 结论先行\n\n这轮分析回答的不是‘哪套机制能把某条真实轨迹逐点复刻’，而是两个分开的判断：候选异质性是否改善冻结后的自主预测，以及这种改善能否唯一归因于反馈、记忆、容量或可塑性。预测改善并不自动等于机制识别；最终路线必须同时尊重这两个层面。",
                },
                {"id": "headline_cards", "type": "metric-strip", "cardIds": ["dev_effects", "capacity_recovery", "cross_accuracy", "reserved_effects", "external_effects", "global_c1_gain"]},
                {
                    "id": "design",
                    "type": "markdown",
                    "body": "## 设计与判定标准\n\n8 名开发被试只用于估计有限候选混合的群体先验；每名被试的候选权重只读取第一块（单块被试读取第一四分之一）。后缀保持真实刺激与正确类别顺序，但选择、反馈和后续学习全部自主生成。24 名保留被试使用冻结先验。每类机制均与该家族的基础参考值比较；模型恢复另问‘模型为真时能否认回生成候选’。RT 与口头报告从未进入候选选择。",
                },
                {
                    "id": "development_heading",
                    "type": "markdown",
                    "body": "## 开发集筛查：可复制的预测信号很少\n\n反馈敏感性、记忆稳定性和规则可塑性的被试权重对蒙特卡洛种子较稳定，但大多数后缀预测效应的区间仍跨零。双种子中唯一让 bootstrap 区间都排除零的比较，是静态读出下有限容量对轨迹摘要偏差的改善；该效应未通过全体比较的 sign-flip FDR，逐时点曲线和 C1 动态读出也没有给出同样证据，因此只作为进入恢复与保留集的筛查信号。",
                    "sourceId": "dev_screen",
                },
                {"id": "dev_chart_block", "type": "chart", "chartId": "dev_summary_chart", "layout": "full"},
                {
                    "id": "recovery_heading",
                    "type": "markdown",
                    "body": "## 恢复检验：有限假设集信号不是可解释的容量测量\n\nH 候选同时改变活跃集合容量，并在 $K<38$ 时使用固定的反馈驱动单条替换；它不是一个纯容量参数。家族内恢复显示，反馈敏感性和记忆稳定性存在排序信息；H 的精确候选恢复率等于机会水平，真实候选值与后验均值没有相关。把四个机制放在同一候选库时，机制家族分类整体接近 20% 机会水平，并在独立粒子数下重复。因而不能把真实被试的候选权重命名为唯一心理机制。",
                },
                {"id": "within_chart_block", "type": "chart", "chartId": "within_recovery_chart", "layout": "full"},
                {"id": "within_table_block", "type": "table", "tableId": "within_table"},
                {"id": "cross_chart_block", "type": "chart", "chartId": "cross_recovery_chart", "layout": "full"},
                {
                    "id": "strategy_heading",
                    "type": "markdown",
                    "body": "## 动态策略没有通过增量门槛\n\n在开发集最偏好的容量 3 下，让交换敏感性在被试间取不同候选值，并未改善固定 θ=0.75 的参考模型；多个 bootstrap 区间指向恶化，sign-flip FDR 后没有出现可靠的有利效应。因此当前不把策略交换强度继续开放为个体参数，也不再叠加更复杂的隐藏策略状态。",
                    "sourceId": "strategy_screen",
                },
                {"id": "strategy_table_block", "type": "table", "tableId": "strategy_table", "layout": "full"},
                {
                    "id": "reserved_heading",
                    "type": "markdown",
                    "body": "## 保留集：冻结候选库是否真的增加预测覆盖\n\n下表是本轮最关键的确认性检验。候选混合的权重来自冻结开发先验与保留被试前缀；后缀没有重新调参。有限容量另以 128 粒子、512 条后缀作高精度确认。",
                },
                {"id": "reserved_coverage_chart_block", "type": "chart", "chartId": "reserved_coverage_chart", "layout": "full"},
                {"id": "reserved_coverage_test_block", "type": "table", "tableId": "reserved_coverage_test", "layout": "full"},
                {"id": "reserved_chart_block", "type": "chart", "chartId": "reserved_summary_chart", "layout": "full"},
                {"id": "reserved_table_block", "type": "table", "tableId": "reserved_table", "layout": "full"},
                {"id": "confirm_table_block", "type": "table", "tableId": "confirm_table"},
                {
                    "id": "global_heading",
                    "type": "markdown",
                    "body": f"## 跨机制预测库：只作预测，不作机制贴标签\n\n该候选库把 F/M/H/P 的非重复候选放在同一个有限混合中。开发集估计群体先验，保留被试仍只用前缀更新。把候选库加到 C1 后，联合覆盖只增加 {global_c1_gain:+d} 人（FDR q={global_c1_q:.4f}）；联合校准提高 {float(global_c1_calibration['mean']):.3f}（q={float(global_c1_calibration['paired_signflip_q']):.4f}），但点状 95% 区间同时增宽 {float(global_c1_width['mean']):.3f}（q={float(global_c1_width['paired_signflip_q']):.4f}），且 CRPS 没有改善。因此这不是足以替换 C1 的净增量。无论是否改善，跨机制恢复接近机会水平都意味着这些权重只能表达预测不确定性，不能命名为被试的真实机制。",
                    "sourceId": "global_coverage",
                },
                {"id": "global_coverage_chart_block", "type": "chart", "chartId": "global_coverage_chart", "layout": "full"},
                {"id": "global_table_block", "type": "table", "tableId": "global_table", "layout": "full"},
                {"id": "global_external_table_block", "type": "table", "tableId": "global_external_table", "layout": "full"},
                {"id": "direct_global_c1_table_block", "type": "table", "tableId": "direct_global_c1_table", "layout": "full"},
                {"id": "direct_global_c1_coverage_block", "type": "table", "tableId": "direct_global_c1_coverage_table"},
                {
                    "id": "external_heading",
                    "type": "markdown",
                    "body": "## 外部约束：一步选择、RT 与口头规则报告\n\n后缀 NLL 使用每次选择发生前的顺序预测；RT 检验比较 log RT 与模型惊讶度的被试内 Spearman 相关；口头报告检验把选择条件化的假设分布投影为四维期望中心。候选权重在整个后缀保持冻结，RT 与口头报告均未参与拟合。",
                    "sourceId": "external_validation",
                },
                {"id": "external_nll_chart_block", "type": "chart", "chartId": "external_nll_chart", "layout": "full"},
                {"id": "external_oral_chart_block", "type": "chart", "chartId": "external_oral_chart", "layout": "full"},
                {"id": "external_table_block", "type": "table", "tableId": "external_table", "layout": "full"},
                {
                    "id": "decision",
                    "type": "markdown",
                    "body": f"## 建模决策\n\n1. **正式主模型继续保留 C1。** 静态跨机制库与 C1 在 CRPS、摘要偏差和联合校准上均无可靠差异；覆盖仅为 {int(direct_coverage_row['global_pass_n'])}/24 对 {int(direct_coverage_row['c1_pass_n'])}/24（精确 p={float(direct_coverage_row['exact_p']):.4f}），但静态库的点状 95% 区间反而宽 {float(direct_width['mean']):.3f}（q={float(direct_width['paired_signflip_q']):.4f}）。\n2. **把 H/P 与跨机制库保留为预测等价性审计，而不是主模型自由参数。** 静态跨机制库相对静态参考的一步 NLL 改善 {float(global_external_nll['mean']):.3f}、RT--惊讶度相关提高 {float(global_external_rt['mean']):.3f}、口头中心相似度提高 {float(global_external_oral['mean']):.4f}，三者均经 FDR；这说明深层异质性有真实预测信息，但不足以证明具体机制。\n3. **不把候选后验当作被试标签。** 有限容量精确恢复等于机会水平，跨机制家族恢复也接近机会水平；‘H 权重大’不能写成‘该被试容量低’。\n4. **固定交换敏感性。** 不引入被试特异 $\\theta$ 或额外动态策略状态，因为开发筛查没有有利增量。\n5. **下一次扩展必须增加新观测约束。** 若以后构造联合 choice--RT--口头报告模型，应预先规定机制对三类观测的方向性预测；在此之前，不同时自由化反馈、记忆、容量和可塑性。",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": "## 解释边界与复现\n\n每名被试只有一次学习轨迹，因此本研究的首要目标是生成分布覆盖，而非逐点潜变量真值恢复。模型恢复使用相同物理刺激/类别序列和新抽样选择轨迹，仍显示机制混淆；这正是本报告不作唯一机制归因的依据。所有正式运行固定单进程数值线程并行，保存候选配置、种子、粒子数、轨迹数、前缀分数、群体先验、被试后验和后缀评价。",
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
    (args.output.parent / "chart_map.md").write_text(
        """# Chart map

| Section | Analytical question | Form | Fields | Takeaway supported | Palette |
|---|---|---|---|---|---|
| Development | Does the suffix effect replicate across Monte Carlo seeds? | Grouped bar | candidate, seed, delta | Stability and direction of summary-discrepancy deltas | hard two-root blue/orange + zero line |
| Within recovery | Can each family recover its own candidate value? | Grouped bar | family, observed/chance, accuracy | Exact recovery relative to discrete chance | hard two-root blue/orange |
| Cross recovery | Can generated mechanisms be distinguished across families? | Heatmap | true family, predicted family, row proportion | Correct diagonal and off-diagonal confusions | single-root sequential blue |
| Reserved coverage | Do frozen mixtures place more real trajectories inside the joint 95% region? | Grouped bar | readout/family, model, pass rate | Absolute generative coverage against each reference | hard two-root blue/orange |
| Reserved score | Do frozen mixtures improve held-out trajectory summaries? | Grouped bar | family, readout, delta | Confirmatory summary-discrepancy effects | hard two-root blue/orange + zero line |
| Global coverage | Does one cross-mechanism predictive bank improve coverage over the matching static/C1 reference? | Grouped bar | readout, global/reference, pass rate | Prediction-only mixture gain and its dependence on readout | hard two-root blue/orange |
| External choice | Do frozen mixtures improve sequential suffix NLL? | Single-series bar | family, delta | One-step-ahead choice validation | single-root blue + zero line |
| External oral | Do frozen mixtures better align with oral centers? | Single-series bar | family, delta | Independent representational validation | single-root blue + zero line |

All charts use subject-level means; uncertainty intervals and denominators are retained in adjacent tables. RT is table-only because its effect sizes share neither unit nor decision threshold with NLL or oral similarity.
""",
        encoding="utf-8",
    )
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"artifact": str(args.output), "datasets": list(datasets)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
