#!/usr/bin/env python3
"""Build a portable technical report for the condition-1 trajectory PPC."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TITLE = "条件1随机生成充分性：从静态全规则集到单次掌握变点"
REPORT_QUERY_PATH = (
    "results/zhuran/cond1_active_set/trajectory_ppc_report/report/"
    "report_queries.sql"
)

MODEL_SPECS = (
    (
        "M=5｜末块",
        "有限容量静态B0",
        "b0_trajectory_ppc",
    ),
    (
        "full-set｜末块",
        "静态全规则集边界",
        "fullset_trajectory_ppc",
    ),
    (
        "full-set｜第一块后",
        "静态全规则集长时程",
        "fullset_trajectory_ppc_early_anchor",
    ),
    (
        "full-set｜第一块后（个体β）",
        "前缀选择个体稳定学习速度",
        "fullset_trajectory_ppc_early_anchor_subject_beta",
    ),
    (
        "衰减lapse｜开发集",
        "个体初始探索＋共享128试次半衰期",
        "decay_lapse_dev_h128",
    ),
    (
        "随机更新｜开发集",
        "个体反馈写入概率",
        "stochastic_update_dev",
    ),
    (
        "单次掌握变点｜保留集",
        "掌握前50%探索；共享128试次半衰期",
        "acquisition_changepoint_reserved_h128_p256_r1024",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=ROOT / "results/zhuran/cond1_active_set",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_active_set/trajectory_ppc_report/"
            "report/artifact.json"
        ),
    )
    return parser.parse_args()


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def source_object(
    source_id: str,
    label: str,
    sql: str,
    description: str,
    generated_at: str,
    tables_used: list[str] | None = None,
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
            "tables_used": tables_used
            or [
                "decision.json",
                "cohort_calibration.csv",
                "subject_summary.csv",
                "metric_failures.csv",
            ],
            "filters": [
                "condition = 1",
                "autonomous suffix rollout",
                "fixed physical stimulus/category schedule",
            ],
            "metric_definitions": [
                "联合通过同时要求预设轨迹统计与滚动正确率曲线不超过模型自身95%离群阈值。",
                "模型自期望通过数由模型生成轨迹轮流作为伪观测重新评分得到。",
                "生成通过只表示实际轨迹不是模型分布中的极端实现，不识别唯一心理机制。",
            ],
        },
    }


def main() -> None:
    args = parse_args()
    generated_at = datetime.now(timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")

    model_rows: list[dict[str, Any]] = []
    cohort_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    decisions: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}

    for model_label, model_description, directory in MODEL_SPECS:
        result_dir = args.results_root / directory
        decision = json.loads(
            (result_dir / "decision.json").read_text(encoding="utf-8")
        )
        manifest = json.loads(
            (result_dir / "manifest.json").read_text(encoding="utf-8")
        )
        cohort = pd.read_csv(result_dir / "cohort_calibration.csv")
        subjects = pd.read_csv(result_dir / "subject_summary.csv")
        failures = pd.read_csv(result_dir / "metric_failures.csv")
        decisions[model_label] = decision
        manifests[model_label] = manifest

        all_subjects = cohort.loc[cohort["cohort"].eq("all_subjects")].iloc[0]
        model_rows.append(
            {
                "model": model_label,
                "description": model_description,
                "capacity": int(manifest.get("capacity", 5)),
                "particle_count": int(manifest["particle_count"]),
                "rollout_count": int(manifest["rollout_count"]),
                "prediction_boundary": manifest.get(
                    "split_mode", "last_block"
                ),
                "observed_pass_n": int(all_subjects["observed_pass_n"]),
                "subject_n": int(all_subjects["subject_n"]),
                "observed_pass_fraction": float(
                    all_subjects["observed_pass_fraction"]
                ),
                "self_expected_pass_mean": float(
                    all_subjects["b0_self_expected_pass_mean"]
                ),
                "self_expected_pass_q025": float(
                    all_subjects["b0_self_expected_pass_q025"]
                ),
                "self_expected_pass_q975": float(
                    all_subjects["b0_self_expected_pass_q975"]
                ),
                "cohort_calibration_p": float(
                    all_subjects["lower_tail_calibration_p"]
                ),
                "subject_fdr_failures": int(
                    decision["subject_level_fdr_failures"]
                ),
                "curve_crps": float(decision["sharpness"]["mean_curve_crps"]),
                "rolling_width_95": float(
                    decision["sharpness"][
                        "median_95pct_rolling_interval_width"
                    ]
                ),
                "decision": decision["generative_adequacy"],
            }
        )

        for row in cohort.to_dict(orient="records"):
            cohort_rows.append(
                {
                    "model": model_label,
                    "cohort": row["cohort"],
                    "subject_n": int(row["subject_n"]),
                    "observed_pass_n": int(row["observed_pass_n"]),
                    "observed_pass_fraction": float(
                        row["observed_pass_fraction"]
                    ),
                    "self_expected_pass_mean": float(
                        row["b0_self_expected_pass_mean"]
                    ),
                    "self_expected_pass_q025": float(
                        row["b0_self_expected_pass_q025"]
                    ),
                    "self_expected_pass_q975": float(
                        row["b0_self_expected_pass_q975"]
                    ),
                    "calibration_p": float(row["lower_tail_calibration_p"]),
                }
            )

        if not failures.empty:
            for failure in failures.to_dict(orient="records"):
                failure_rows.append(
                    {
                        "model": model_label,
                        "metric": str(failure["metric"]),
                        "marginal_failure_n": int(
                            failure["failed_subject_n"]
                        ),
                    }
                )

        for row in subjects.to_dict(orient="records"):
            subject_rows.append(
                {
                    "model": model_label,
                    "iSub": int(row["iSub"]),
                    "cohort": row["cohort"],
                    "actual_accuracy": float(row["observed_test_accuracy"]),
                    "simulated_accuracy_mean": float(
                        row["simulated_test_accuracy_mean"]
                    ),
                    "simulated_accuracy_q025": float(
                        row["simulated_test_accuracy_q025"]
                    ),
                    "simulated_accuracy_q975": float(
                        row["simulated_test_accuracy_q975"]
                    ),
                    "joint_pass": bool(row["combined_pass_95"]),
                    "joint_p": float(row["combined_calibration_p"]),
                    "joint_fdr_q": float(
                        row["combined_calibration_fdr_q"]
                    ),
                }
            )

    model_summary = pd.DataFrame(model_rows)
    cohort_summary = pd.DataFrame(cohort_rows)
    failure_summary = pd.DataFrame(failure_rows)
    subject_summary = pd.DataFrame(subject_rows)
    pure_development = pd.read_csv(
        args.results_root
        / "acquisition_changepoint_dev_p128_r256/sweep_summary.csv"
    ).assign(novice_regime="掌握前100%探索")
    partial_development = pd.read_csv(
        args.results_root
        / "acquisition_changepoint_partial_dev_p128_r256/"
        "sweep_summary.csv"
    ).assign(novice_regime="掌握前50%探索")
    acquisition_development = pd.concat(
        [pure_development, partial_development],
        ignore_index=True,
    )
    acquisition_development["half_life_label"] = (
        acquisition_development["acquisition_half_life"]
        .astype(int)
        .map(lambda value: f"{value} trials")
    )
    acquisition_development["pass_fraction"] = (
        acquisition_development["combined_pass_n"]
        / acquisition_development["subject_n"]
    )
    partial_pass_chart = acquisition_development.loc[
        acquisition_development["novice_regime"].eq("掌握前50%探索"),
        [
            "half_life_label",
            "pass_fraction",
            "combined_pass_n",
            "lower_tail_calibration_p",
            "mean_curve_crps",
            "median_curve_interval_width_95",
        ],
    ].copy()

    pass_chart_rows: list[dict[str, Any]] = []
    for row in model_rows:
        pass_chart_rows.extend(
            [
                {
                    "comparison": f"{row['model']}｜实际",
                    "pass_fraction": (
                        row["observed_pass_n"] / row["subject_n"]
                    ),
                    "kind": "实际轨迹",
                },
                {
                    "comparison": f"{row['model']}｜模型自期望",
                    "pass_fraction": (
                        row["self_expected_pass_mean"] / row["subject_n"]
                    ),
                    "kind": "模型重复抽样",
                },
            ]
        )
    pass_chart = pd.DataFrame(pass_chart_rows)
    sharpness_chart = model_summary[
        ["model", "curve_crps"]
    ].copy()
    last_block_subjects = subject_summary.loc[
        subject_summary["model"].isin(["M=5｜末块", "full-set｜末块"])
    ].copy()
    last_block_subjects["accuracy_error"] = (
        last_block_subjects["actual_accuracy"]
        - last_block_subjects["simulated_accuracy_mean"]
    )
    largest_errors = (
        last_block_subjects.sort_values(
            ["model", "accuracy_error"],
            ascending=[True, False],
        )
        .groupby("model", as_index=False)
        .head(10)
    )

    source_sql = {
        "model_comparison": (
            "SELECT model, capacity, prediction_boundary, observed_pass_n, "
            "subject_n, self_expected_pass_mean, self_expected_pass_q025, "
            "self_expected_pass_q975, cohort_calibration_p, "
            "subject_fdr_failures, curve_crps, rolling_width_95, decision "
            "FROM model_summary ORDER BY capacity, prediction_boundary"
        ),
        "cohort_calibration": (
            "SELECT * FROM cohort_summary "
            "ORDER BY model, cohort"
        ),
        "subject_accuracy": (
            "SELECT model, iSub, actual_accuracy, simulated_accuracy_mean, "
            "simulated_accuracy_q025, simulated_accuracy_q975, joint_pass, "
            "joint_fdr_q FROM subject_summary "
            "WHERE model IN ('M=5｜末块', 'full-set｜末块')"
        ),
        "metric_failures": (
            "SELECT model, metric, marginal_failure_n "
            "FROM failure_summary ORDER BY model, marginal_failure_n DESC"
        ),
        "acquisition_development": (
            "SELECT novice_regime, acquisition_half_life, combined_pass_n, "
            "subject_n, lower_tail_calibration_p, fdr_failure_n, "
            "mean_curve_crps, median_curve_interval_width_95, "
            "development_gate FROM acquisition_development "
            "ORDER BY novice_regime, acquisition_half_life"
        ),
        "acquisition_stability": (
            "SELECT cohort, subject_n, observed_pass_n, "
            "b0_self_expected_pass_mean, b0_self_expected_pass_q025, "
            "b0_self_expected_pass_q975, lower_tail_calibration_p "
            "FROM independent_seed_cohort_calibration "
            "WHERE cohort = 'all_subjects'"
        ),
    }
    source_descriptions = {
        "model_comparison": "比较七个自主轨迹检验的联合覆盖、锐度与决策。",
        "cohort_calibration": "读取开发集、保留应用集和全样本的模型自校准通过数。",
        "subject_accuracy": "比较同一被试在两种静态容量边界下的实际与模拟正确率。",
        "metric_failures": "汇总每个模型在哪些单项轨迹统计上出现边缘覆盖失败。",
        "acquisition_development": (
            "比较纯随机新手与固定50%探索边界下的开发集掌握半衰期扫描。"
        ),
        "acquisition_stability": "读取冻结掌握变点的独立随机种子保留集稳健性结果。",
    }
    source_labels = {
        "model_comparison": "模型级生成充分性对照",
        "cohort_calibration": "队列级重复抽样校准",
        "subject_accuracy": "逐被试正确率覆盖",
        "metric_failures": "边缘统计失败",
        "acquisition_development": "单次掌握变点开发扫描",
        "acquisition_stability": "独立随机种子稳健性",
    }
    source_tables = {
        "acquisition_development": [
            "acquisition_changepoint_dev_p128_r256/sweep_summary.csv",
            (
                "acquisition_changepoint_partial_dev_p128_r256/"
                "sweep_summary.csv"
            ),
        ],
        "acquisition_stability": [
            (
                "acquisition_changepoint_reserved_h128_"
                "seed20260901_p128_r512/decision.json"
            ),
            (
                "acquisition_changepoint_reserved_h128_"
                "seed20260901_p128_r512/subject_summary.csv"
            ),
        ],
    }
    sources = [
        source_object(
            source_id,
            source_labels[source_id],
            sql,
            source_descriptions[source_id],
            generated_at,
            tables_used=source_tables.get(source_id),
        )
        for source_id, sql in source_sql.items()
    ]

    finite = model_summary.loc[
        model_summary["model"].eq("M=5｜末块")
    ].iloc[0]
    full_last = model_summary.loc[
        model_summary["model"].eq("full-set｜末块")
    ].iloc[0]
    full_long = model_summary.loc[
        model_summary["model"].eq("full-set｜第一块后")
    ].iloc[0]
    full_long_beta = model_summary.loc[
        model_summary["model"].eq("full-set｜第一块后（个体β）")
    ].iloc[0]
    lapse_dev = model_summary.loc[
        model_summary["model"].eq("衰减lapse｜开发集")
    ].iloc[0]
    update_dev = model_summary.loc[
        model_summary["model"].eq("随机更新｜开发集")
    ].iloc[0]
    acquisition_reserved = model_summary.loc[
        model_summary["model"].eq("单次掌握变点｜保留集")
    ].iloc[0]
    acquisition_dev = partial_development.loc[
        partial_development["acquisition_half_life"].eq(128.0)
    ].iloc[0]
    independent_decision = json.loads(
        (
            args.results_root
            / "acquisition_changepoint_reserved_h128_seed20260901_p128_r512/"
            "decision.json"
        ).read_text(encoding="utf-8")
    )
    independent_all = next(
        row
        for row in independent_decision["cohort_calibration"]
        if row["cohort"] == "all_subjects"
    )
    long_is_adequate = (
        full_long["decision"] == "adequate_at_cohort_level"
    )
    long_conclusion = (
        "更严格的第一块后长时程生成也通过群体校准，因此目前没有生成层面的理由加入动态策略状态。"
        if long_is_adequate
        else (
            "第一块后长时程生成未通过群体校准；静态全规则集能解释末块，"
            "但尚不能覆盖完整学习过程。只用前缀选择个体稳定β后通过数由"
            f"{int(full_long['observed_pass_n'])}/"
            f"{int(full_long['subject_n'])}提高到"
            f"{int(full_long_beta['observed_pass_n'])}/"
            f"{int(full_long_beta['subject_n'])}，仍未达到群体校准。"
        )
    )

    datasets = {
        "pass_chart": records(pass_chart),
        "sharpness_chart": records(sharpness_chart),
        "model_summary": records(model_summary),
        "cohort_summary": records(cohort_summary),
        "largest_errors": records(largest_errors),
        "failure_summary": records(failure_summary),
        "acquisition_development": records(acquisition_development),
        "partial_acquisition_pass_chart": records(partial_pass_chart),
    }
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": (
                "以模型自身重复抽样为标尺，检验固定刺激序列下实际学习轨迹是否为非边缘实现。"
            ),
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "pass_count_comparison",
                    "title": "联合轨迹通过比例：实际与模型自期望",
                    "subtitle": "各模型使用其对应开发、保留或全样本队列",
                    "type": "horizontalBar",
                    "dataset": "pass_chart",
                    "sourceId": "model_comparison",
                    "intent": "comparison",
                    "question": "实际轨迹通过数是否与模型自身95%标准相容？",
                    "rationale": "所有模型使用相同联合通过定义，并与各自模型自校准比较。",
                    "comparisonContext": {
                        "baseline": "模型重复抽样",
                        "grain": "model × horizon",
                        "unit": "subjects",
                    },
                    "encodings": {
                        "x": {
                            "field": "comparison",
                            "type": "nominal",
                            "label": "模型与基准",
                        },
                        "y": {
                            "field": "pass_fraction",
                            "type": "quantitative",
                            "label": "联合通过比例",
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
                    "surface": {"surface": "explorer", "viewMode": "both"},
                },
                {
                    "id": "acquisition_development_pass",
                    "title": "开发集半衰期扫描的联合通过比例",
                    "subtitle": "固定掌握前50%探索；8名开发被试",
                    "type": "horizontalBar",
                    "dataset": "partial_acquisition_pass_chart",
                    "sourceId": "acquisition_development",
                    "intent": "comparison",
                    "question": "哪个共享掌握时标同时改善开发集轨迹覆盖？",
                    "rationale": "半衰期是有序候选，实际选择还同时受FDR、CRPS和带宽门槛约束。",
                    "comparisonContext": {
                        "baseline": "静态full-set长时程",
                        "grain": "shared acquisition half-life",
                        "unit": "development-subject pass fraction",
                    },
                    "encodings": {
                        "x": {
                            "field": "half_life_label",
                            "type": "nominal",
                            "label": "共享掌握半衰期",
                        },
                        "y": {
                            "field": "pass_fraction",
                            "type": "quantitative",
                            "label": "联合通过比例",
                        },
                        "tooltip": [
                            {
                                "field": "lower_tail_calibration_p",
                                "type": "quantitative",
                                "label": "群体校准p",
                            },
                            {
                                "field": "mean_curve_crps",
                                "type": "quantitative",
                                "label": "曲线CRPS",
                            },
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
                    "surface": {"surface": "explorer", "viewMode": "both"},
                },
                {
                    "id": "sharpness_comparison",
                    "title": "滚动正确率曲线CRPS对照",
                    "subtitle": "滚动正确率曲线CRPS；越低越好",
                    "type": "horizontalBar",
                    "dataset": "sharpness_chart",
                    "sourceId": "model_comparison",
                    "intent": "comparison",
                    "question": "覆盖改善是否伴随更差的预测锐度？",
                    "rationale": "CRPS同时惩罚位置偏差和过宽分布。",
                    "comparisonContext": {
                        "baseline": "M=5有限容量静态B0",
                        "grain": "model × horizon",
                        "unit": "CRPS",
                    },
                    "encodings": {
                        "x": {
                            "field": "model",
                            "type": "nominal",
                            "label": "模型",
                        },
                        "y": {
                            "field": "curve_crps",
                            "type": "quantitative",
                            "label": "曲线CRPS",
                        },
                    },
                    "palette": {"kind": "sequential", "name": "blue"},
                    "labels": {"values": "all"},
                    "settings": {
                        "orientation": "horizontal",
                        "sort": "ascending",
                        "showValues": True,
                    },
                    "layout": "full",
                    "surface": {"surface": "explorer", "viewMode": "both"},
                },
            ],
            "tables": [
                {
                    "id": "model_table",
                    "title": "模型级生成充分性结果",
                    "subtitle": "联合通过同时约束轨迹统计和滚动正确率曲线",
                    "dataset": "model_summary",
                    "sourceId": "model_comparison",
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "model", "label": "模型/时程", "type": "text"},
                        {"field": "capacity", "label": "容量", "format": "number"},
                        {"field": "observed_pass_n", "label": "实际通过", "format": "number"},
                        {"field": "subject_n", "label": "总人数", "format": "number"},
                        {"field": "self_expected_pass_mean", "label": "自期望", "format": "number"},
                        {"field": "self_expected_pass_q025", "label": "自校准2.5%", "format": "number"},
                        {"field": "self_expected_pass_q975", "label": "自校准97.5%", "format": "number"},
                        {"field": "cohort_calibration_p", "label": "群体p", "format": "number"},
                        {"field": "subject_fdr_failures", "label": "FDR失败人数", "format": "number"},
                        {"field": "curve_crps", "label": "曲线CRPS", "format": "number"},
                        {"field": "rolling_width_95", "label": "95%带宽中位数", "format": "number"},
                    ],
                },
                {
                    "id": "subject_table",
                    "title": "末块正确率偏差最大的被试",
                    "subtitle": "同一真实轨迹分别与M=5和full-set自主模拟比较",
                    "dataset": "largest_errors",
                    "sourceId": "subject_accuracy",
                    "density": "compact",
                    "layout": "full",
                    "columns": [
                        {"field": "model", "label": "模型", "type": "text"},
                        {"field": "iSub", "label": "被试", "format": "number"},
                        {"field": "actual_accuracy", "label": "实际正确率", "format": "number"},
                        {"field": "simulated_accuracy_mean", "label": "模拟均值", "format": "number"},
                        {"field": "simulated_accuracy_q025", "label": "模拟2.5%", "format": "number"},
                        {"field": "simulated_accuracy_q975", "label": "模拟97.5%", "format": "number"},
                        {"field": "accuracy_error", "label": "实际-模拟", "format": "number"},
                        {"field": "joint_pass", "label": "联合通过", "type": "boolean"},
                    ],
                },
                {
                    "id": "acquisition_development_table",
                    "title": "单次掌握变点开发集扫描",
                    "subtitle": "两种掌握前探索边界；每种边界只选择共享半衰期",
                    "dataset": "acquisition_development",
                    "sourceId": "acquisition_development",
                    "density": "compact",
                    "layout": "full",
                    "columns": [
                        {"field": "novice_regime", "label": "掌握前边界", "type": "text"},
                        {"field": "acquisition_half_life", "label": "半衰期", "format": "number"},
                        {"field": "combined_pass_n", "label": "通过人数", "format": "number"},
                        {"field": "subject_n", "label": "开发人数", "format": "number"},
                        {"field": "lower_tail_calibration_p", "label": "群体p", "format": "number"},
                        {"field": "fdr_failure_n", "label": "FDR失败", "format": "number"},
                        {"field": "mean_curve_crps", "label": "曲线CRPS", "format": "number"},
                        {"field": "median_curve_interval_width_95", "label": "95%带宽", "format": "number"},
                        {"field": "development_gate", "label": "开发门槛", "type": "boolean"},
                    ],
                },
            ],
            "sources": [
                {"id": item["id"], "label": item["label"], "path": item["path"]}
                for item in sources
            ],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": f"# {TITLE}",
                    "layout": "full",
                },
                {
                    "id": "conclusion",
                    "type": "markdown",
                    "sourceId": "model_comparison",
                    "body": (
                        "## 技术结论：单次掌握变点达到群体生成充分性，但不是普适机制\n\n"
                        f"固定容量 $M=5$ 的静态B0仅覆盖 {int(finite['observed_pass_n'])}/"
                        f"{int(finite['subject_n'])} 名被试，而模型自身期望覆盖"
                        f"{finite['self_expected_pass_mean']:.2f}人"
                        f"（95%范围[{finite['self_expected_pass_q025']:.0f}, "
                        f"{finite['self_expected_pass_q975']:.0f}]；"
                        f"群体校准p={finite['cohort_calibration_p']:.4f}）。"
                        "它系统性低估了多数被试已经学会后的高正确率。\n\n"
                        f"不增加任何切换、lapse或隐藏状态，只把活跃集合放宽到全部38条"
                        f"带标签规则后，末块覆盖提高到{int(full_last['observed_pass_n'])}/"
                        f"{int(full_last['subject_n'])}，模型自期望为"
                        f"{full_last['self_expected_pass_mean']:.2f}"
                        f"（95%范围[{full_last['self_expected_pass_q025']:.0f}, "
                        f"{full_last['self_expected_pass_q975']:.0f}]；"
                        f"p={full_last['cohort_calibration_p']:.3f}；"
                        f"FDR失败{int(full_last['subject_fdr_failures'])}人）。"
                        f"{long_conclusion}\n\n"
                        "冻结的单次掌握变点在24名保留被试中联合通过"
                        f"{int(acquisition_reserved['observed_pass_n'])}/"
                        f"{int(acquisition_reserved['subject_n'])}，模型自期望"
                        f"{acquisition_reserved['self_expected_pass_mean']:.2f}"
                        f"（95%范围[{acquisition_reserved['self_expected_pass_q025']:.0f}, "
                        f"{acquisition_reserved['self_expected_pass_q975']:.0f}]；"
                        f"群体p={acquisition_reserved['cohort_calibration_p']:.3f}；"
                        f"FDR失败{int(acquisition_reserved['subject_fdr_failures'])}人）。"
                        "因此它获得的是群体层面的生成支持，不是“每名被试都经历同一真实"
                        "切换”的机制证明。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "meaning",
                    "type": "markdown",
                    "body": (
                        "## 检验回答的是什么\n\n"
                        "每名被试的真实轨迹只被视为同一稳定特征、同一物理刺激序列下众多"
                        "可能轨迹中的一次实现。模型只读取预测边界以前的真实选择与反馈；"
                        "边界以后自主抽取知觉、选择和由该选择产生的反馈，绝不读取未来真实"
                        "行为。通过意味着真实轨迹在联合统计意义上不是模型生成分布的极端"
                        "样本；它不意味着模型复原了被试实际的潜在路径，也不证明full-set"
                        "是唯一认知机制。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "acquisition_development",
                    "type": "markdown",
                    "sourceId": "acquisition_development",
                    "body": (
                        "## 完全随机的新手边界过宽，固定50%探索后才通过开发门槛\n\n"
                        "纯随机新手版本在64-trial半衰期时虽达到7/8开发被试联合通过、"
                        "FDR失败0，但95%滚动带宽中位数为0.651，未过0.50锐度上限。"
                        "因此只修正这一极端边界：掌握前以0.5概率按现有认知readout作答、"
                        "以0.5概率探索，掌握后探索率一次性降为0；没有增加可反复切换状态。"
                        f"共享半衰期128 trials由开发集选中："
                        f"{int(acquisition_dev['combined_pass_n'])}/"
                        f"{int(acquisition_dev['subject_n'])}通过，群体"
                        f"p={acquisition_dev['lower_tail_calibration_p']:.4f}，"
                        f"FDR失败{int(acquisition_dev['fdr_failure_n'])}，"
                        f"CRPS={acquisition_dev['mean_curve_crps']:.4f}，"
                        f"95%带宽={acquisition_dev['median_curve_interval_width_95']:.3f}。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "acquisition_development_chart_block",
                    "type": "chart",
                    "chartId": "acquisition_development_pass",
                    "layout": "full",
                },
                {
                    "id": "acquisition_development_table_block",
                    "type": "table",
                    "tableId": "acquisition_development_table",
                    "layout": "full",
                },
                {
                    "id": "pass_chart_block",
                    "type": "chart",
                    "chartId": "pass_count_comparison",
                    "layout": "full",
                },
                {
                    "id": "model_table_block",
                    "type": "table",
                    "tableId": "model_table",
                    "layout": "full",
                },
                {
                    "id": "sharpness",
                    "type": "markdown",
                    "sourceId": "model_comparison",
                    "body": (
                        "## 覆盖与锐度必须同时成立\n\n"
                        f"M=5末块的平均曲线CRPS为{finite['curve_crps']:.4f}，"
                        f"95%滚动正确率区间宽度中位数为{finite['rolling_width_95']:.3f}；"
                        f"full-set末块分别为{full_last['curve_crps']:.4f}和"
                        f"{full_last['rolling_width_95']:.3f}。覆盖改善的同时CRPS与带宽"
                        "都下降，说明full-set不是靠把预测分布无限放宽来蒙中真实轨迹，"
                        "而是修正了M=5对已掌握行为的系统性低预测。\n\n"
                        "单次掌握变点保留集的CRPS为"
                        f"{acquisition_reserved['curve_crps']:.4f}，95%带宽中位数为"
                        f"{acquisition_reserved['rolling_width_95']:.3f}。它比静态长时程"
                        f"full-set的CRPS {full_long['curve_crps']:.4f}更低，但预测带更宽；"
                        "该宽度恰好达到预设0.50上限，所以应解释为足够锐利的边界结果，"
                        "而不是非常稳健的富余通过。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "sharpness_chart_block",
                    "type": "chart",
                    "chartId": "sharpness_comparison",
                    "layout": "full",
                },
                {
                    "id": "subject_table_block",
                    "type": "table",
                    "tableId": "subject_table",
                    "layout": "full",
                },
                {
                    "id": "acquisition_stability",
                    "type": "markdown",
                    "sourceId": "acquisition_stability",
                    "body": (
                        "## 独立随机种子保持群体结论，个体FDR边界不稳定\n\n"
                        "冻结同一结构与参数、仅更换随机种子，并以128粒子和512条轨迹"
                        "重跑24名保留被试后，联合通过"
                        f"{int(independent_all['observed_pass_n'])}/"
                        f"{int(independent_all['subject_n'])}，模型自期望"
                        f"{independent_all['b0_self_expected_pass_mean']:.2f}"
                        f"（95%范围[{independent_all['b0_self_expected_pass_q025']:.0f}, "
                        f"{independent_all['b0_self_expected_pass_q975']:.0f}]；"
                        f"群体p={independent_all['lower_tail_calibration_p']:.3f}；"
                        f"FDR失败{int(independent_decision['subject_level_fdr_failures'])}）。"
                        "CRPS为"
                        f"{independent_decision['sharpness']['mean_curve_crps']:.4f}，"
                        "95%带宽中位数为"
                        f"{independent_decision['sharpness']['median_95pct_rolling_interval_width']:.3f}。"
                        "119号仍是唯一联合未通过者，但其FDR q由主运行0.0468变为"
                        "0.0936；因此群体充分性稳定，单名被试是否跨过校正阈值不稳定。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "decision_rule",
                    "type": "markdown",
                    "body": (
                        "## 由结果推出的模型路线\n\n"
                        "1. 条件1已掌握阶段继续使用静态full-set；完整学习时程则在其"
                        "外增加一个不可逆的探索终止/掌握变点。两者是同一简单核心的不"
                        "同时程边界，不是三个可反复切换的策略状态。\n"
                        "2. $M=5$ 不能再作为已支持的心理事实；有限工作记忆若仍是理论"
                        "重点，必须另做容量参数化和恢复，而不能由当前结果默认。\n"
                        "3. 变点候选使真实总体正确率、学习斜率和事件暴露更接近模型分布，"
                        "但仍低估平均学习斜率与低正确率段持续时间；不能把群体通过改写成"
                        "对每次突然下降的心理解释。\n"
                        "4. 119号在两次种子运行中均联合未通过，但FDR显著性只出现一次；"
                        "这保留为个案残差，不据此给全体增加response-bias或三状态控制器。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 局限性\n\n"
                        "这里固定了现有的共享参数 "
                        "$\\gamma=0.55, w_0=0.10, \\rho=2$，比较的是结构边界而非重新"
                        "估计的层级个体参数。末块正式分析使用64粒子、256条自主轨迹；"
                        "长时程敏感性使用32粒子、128条轨迹；个体β诊断使用16粒子、"
                        "64条轨迹，开发集的衰减lapse和随机更新筛查使用低预算，因此"
                        "只承担止损和候选排序作用。单次掌握变点在开发集用128粒子、"
                        "256条轨迹选择共享半衰期，在24名保留被试上用256粒子、1024条"
                        "轨迹作一次性确认。其掌握前50%探索是诊断完全随机边界过宽后固定"
                        "的结构边界，不是从保留集估计的个体参数。full-set是"
                        "预测上有效的边界，不自动获得“所有38条规则都同时进入工作记忆”"
                        "的字面心理解释；单次变点通过也不自动获得唯一心理真实性。"
                    ),
                    "layout": "full",
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 后续需要回答的问题\n\n"
                        "- 较大但有限的容量是否能在不损失末块覆盖的前提下逼近full-set？\n"
                        "- 119号持续的反应偏置残差是否只需个体观测层处理，仍不能由当前"
                        "一次轨迹区分其心理来源。\n"
                        "- 条件2和3是否需要同一种单次掌握边界，必须各自重新开发和确认，"
                        "不能从条件1外推。"
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
    (args.output.parent / "report_queries.sql").write_text(
        "\n\n".join(
            f"-- {source_descriptions[source_id]}\n{sql};"
            for source_id, sql in source_sql.items()
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
                "models": len(model_summary),
                "datasets": {key: len(value) for key, value in datasets.items()},
                "long_horizon_adequate": long_is_adequate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
