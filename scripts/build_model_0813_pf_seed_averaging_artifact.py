#!/usr/bin/env python3
"""Build the technical report artifact for 0813 PF seed averaging."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_RESULTS = ROOT / (
    "results/model_dynamic_adaptive_control/0813_pf/mechanism_audit/"
    "01_pf_calibration/seed_averaging"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _source(
    source_id: str,
    label: str,
    path: Path,
    description: str,
    metric_definitions: list[str],
) -> dict[str, Any]:
    return {
        "id": source_id,
        "label": label,
        "path": _repo_relative(path),
        "query": {
            "language": "python",
            "description": description,
            "tables_used": [_repo_relative(path)],
            "metric_definitions": metric_definitions,
        },
    }


def main() -> None:
    args = parse_args()
    results = args.results_dir.resolve()
    output = (args.output or results / "artifact.json").resolve()
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    seed_scores = pd.read_csv(results / "seed_scores.csv")
    split = pd.read_csv(results / "split_half_stability.csv")
    running = pd.read_csv(results / "running_prefix_stability.csv")
    datasets = pd.read_csv(results / "dataset_summary.csv")
    equivalence = pd.read_csv(results / "candidate_equivalence_sets.csv")
    sensitivity = pd.read_csv(results / "aggregation_method_sensitivity.csv")

    split_chart = (
        split.groupby("comparison_seed_count")["candidate_rank_spearman"]
        .agg(["median", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "comparison_seed_count": "total_seeds",
                "median": "median_rank_rho",
                "min": "minimum_rank_rho",
                "max": "maximum_rank_rho",
            }
        )
    )
    split_chart["total_seeds"] = split_chart["total_seeds"].astype(str)
    running_chart = (
        running.groupby("comparison_seed_count")["candidate_rank_spearman"]
        .agg(["median", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "comparison_seed_count": "ending_seed_count",
                "median": "median_rank_rho",
                "min": "minimum_rank_rho",
                "max": "maximum_rank_rho",
            }
        )
    )
    running_chart["comparison"] = running_chart["ending_seed_count"].map(
        lambda value: f"{int(value) // 2}->{int(value)}"
    )
    dataset_chart = datasets[
        [
            "dataset_id",
            "training_validation_rank_spearman",
            "prefix_8_to_16_rank_spearman",
            "equivalence_set_size",
        ]
    ].copy()
    dataset_chart["dataset"] = dataset_chart["dataset_id"].map(
        lambda value: str(value).split("_subject")[0]
    )
    gate_table = pd.DataFrame(
        [
            {
                "gate": key,
                "passed": "yes" if value else "no",
            }
            for key, value in summary["gate_checks"].items()
        ]
    )
    equivalence_table = equivalence[
        [
            "dataset_id",
            "best_profile_id",
            "fit_profile_id",
            "delta_nll_from_selected_best",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "equivalent_to_selected_best",
        ]
    ].copy()
    equivalence_table["equivalent_to_selected_best"] = equivalence_table[
        "equivalent_to_selected_best"
    ].map({True: "yes", False: "no"})
    headline = [
        {
            "split_rank_rho": float(summary["final_split_median_rank_spearman"]),
            "winner_agreement": float(summary["final_split_winner_agreement"]),
            "median_equivalence_size": float(
                np.median(list(summary["equivalence_set_sizes"].values()))
            ),
            "effective_seed_fraction": float(
                summary["median_effective_seed_fraction"]
            ),
            "aggregate_mcse": float(
                summary["median_aggregate_log_likelihood_mcse"]
            ),
        }
    ]
    snapshot_datasets = {
        "headline": headline,
        "split_chart": _records(split_chart),
        "running_chart": _records(running_chart),
        "dataset_chart": _records(dataset_chart),
        "dataset_table": _records(datasets),
        "gate_table": _records(gate_table),
        "equivalence_table": _records(equivalence_table),
        "sensitivity_table": _records(sensitivity),
    }
    sources = [
        _source(
            "summary",
            "Replicated PF likelihood summary",
            results / "summary.json",
            "Applies the frozen 16-seed training/validation, convergence, effective-seed, MCSE, and noise-to-signal gates.",
            [
                "Primary score is log of the arithmetic mean of 16 independent PF likelihood estimates.",
                "PF seeds are Monte Carlo repeats and are not independent behavioral observations.",
            ],
        ),
        _source(
            "seed_scores",
            "Seed-level PF likelihood estimates",
            results / "seed_scores.csv",
            "Contains the complete three-dataset by nine-candidate by sixteen-seed score bank.",
            [
                "Seeds 0-7 are the fixed training panel and 8-15 are the disjoint validation panel.",
                "Candidate comparisons are paired by dataset and filter seed.",
            ],
        ),
        _source(
            "split_stability",
            "Independent split-half stability",
            results / "split_half_stability.csv",
            "Compares candidate ranks, winners and top-three sets between disjoint halves at total panels of 2, 4, 8 and 16 seeds.",
            [
                "The final split compares seeds 0-7 with seeds 8-15.",
                "Top-three overlap is intersection over union.",
            ],
        ),
        _source(
            "dataset_summary",
            "Dataset-level replicated-likelihood summary",
            results / "dataset_summary.csv",
            "Collects each fixed trajectory's selected candidate, equivalence-set size, independent and nested rank checks, and Monte Carlo diagnostics.",
            [
                "Each row is one fixed technical calibration trajectory, not an independent behavioral subject.",
                "Training/validation comparisons use disjoint filter-seed panels.",
            ],
        ),
        _source(
            "equivalence",
            "Paired-bootstrap candidate equivalence sets",
            results / "candidate_equivalence_sets.csv",
            "Resamples the same seed indices for each candidate and the selected 16-seed winner.",
            [
                "A candidate is numerically equivalent when the paired bootstrap interval for candidate NLL minus selected-winner NLL includes zero.",
                "The selected-best reference makes these descriptive numerical intervals, not post-selection confidence intervals for psychological truth.",
            ],
        ),
        _source(
            "design",
            "Frozen replicated-likelihood design",
            ROOT / "configs/specific_models/model_0813_pf_seed_averaging.yaml",
            "Defines fixed datasets, candidates, particle count, seeds, split, aggregation stages, bootstrap and gates before seeds 3-15 were inspected.",
            [
                "All nine gate conditions must pass before using the replicated likelihood for candidate selection or recovery.",
            ],
        ),
    ]
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    title = "0813 PF 多 seed 似然平均校准"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "以16个独立128-particle PF重复检验聚合似然、训练/验证排序与候选等价集合。",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": [
                {
                    "id": "split_rank",
                    "description": "独立8-seed训练与8-seed验证面板的九候选排名相关中位数；冻结门槛为0.80。",
                    "dataset": "headline",
                    "sourceId": "summary",
                    "metrics": [
                        {
                            "label": "8-vs-8 rank rho",
                            "field": "split_rank_rho",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "winner_agreement",
                    "description": "三个固定校准数据集的训练/验证最佳候选一致比例；冻结门槛为0.75。",
                    "dataset": "headline",
                    "sourceId": "summary",
                    "metrics": [
                        {
                            "label": "赢家一致率",
                            "field": "winner_agreement",
                            "format": "percent",
                        }
                    ],
                },
                {
                    "id": "equivalence_size",
                    "description": "每条轨迹与16-seed赢家无法由配对bootstrap明确区分的候选数中位数，候选总数为9。",
                    "dataset": "headline",
                    "sourceId": "equivalence",
                    "metrics": [
                        {
                            "label": "等价候选数中位数",
                            "field": "median_equivalence_size",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "effective_seeds",
                    "description": "likelihood权重换算的有效seed数占16个名义seeds的比例中位数；冻结门槛为0.50。",
                    "dataset": "headline",
                    "sourceId": "summary",
                    "metrics": [
                        {
                            "label": "有效 seed 比例",
                            "field": "effective_seed_fraction",
                            "format": "percent",
                        }
                    ],
                },
                {
                    "id": "aggregate_mcse",
                    "description": "16-seed聚合log-likelihood的delta-method Monte Carlo标准误中位数。",
                    "dataset": "headline",
                    "sourceId": "summary",
                    "metrics": [
                        {
                            "label": "聚合 MCSE",
                            "field": "aggregate_mcse",
                            "format": "number",
                        }
                    ],
                },
            ],
            "charts": [
                {
                    "id": "split_chart",
                    "title": "独立 seed 面板的候选排名相关",
                    "subtitle": "2、4、8、16个总seeds；每个阶段分为两个不重叠等大面板。",
                    "type": "bar",
                    "dataset": "split_chart",
                    "sourceId": "split_stability",
                    "xField": "total_seeds",
                    "xAxisTitle": "可供分半的总 seed 数",
                    "yAxisTitle": "候选排名 Spearman rho",
                    "series": [
                        {"field": "median_rank_rho", "label": "中位数"},
                        {"field": "minimum_rank_rho", "label": "最小值"},
                    ],
                    "valueFormat": "number",
                    "layout": "full",
                },
                {
                    "id": "dataset_chart",
                    "title": "逐校准数据集的候选排序稳定性",
                    "subtitle": "独立8-vs-8面板与嵌套8-vs-16聚合的Spearman rho。",
                    "type": "bar",
                    "dataset": "dataset_chart",
                    "sourceId": "dataset_summary",
                    "xField": "dataset",
                    "xAxisTitle": "固定校准轨迹",
                    "yAxisTitle": "候选排名 Spearman rho",
                    "series": [
                        {
                            "field": "training_validation_rank_spearman",
                            "label": "独立8-vs-8",
                        },
                        {
                            "field": "prefix_8_to_16_rank_spearman",
                            "label": "嵌套8-vs-16",
                        },
                    ],
                    "valueFormat": "number",
                    "layout": "full",
                },
                {
                    "id": "equivalence_chart",
                    "title": "16-seed数值等价候选集合大小",
                    "subtitle": "每条轨迹共九个候选；区间包含0即保留在等价集合。",
                    "type": "bar",
                    "dataset": "dataset_chart",
                    "sourceId": "equivalence",
                    "xField": "dataset",
                    "xAxisTitle": "固定校准轨迹",
                    "yAxisTitle": "等价候选数",
                    "series": [
                        {
                            "field": "equivalence_set_size",
                            "label": "等价候选数",
                        }
                    ],
                    "valueFormat": "number",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "dataset_results",
                    "title": "逐数据集的16-seed校准结果",
                    "subtitle": "三个预先固定的自主合成轨迹；subjects和seeds均不作总体推断。",
                    "dataset": "dataset_table",
                    "sourceId": "dataset_summary",
                    "layout": "full",
                    "density": "spacious",
                    "defaultSort": {"field": "dataset_id", "direction": "asc"},
                    "columns": [
                        {"field": "dataset_id", "label": "dataset", "type": "text"},
                        {"field": "selected_profile_id", "label": "16-seed winner", "type": "text"},
                        {"field": "equivalence_set_size", "label": "等价候选数", "format": "number"},
                        {"field": "training_validation_rank_spearman", "label": "8-vs-8 rho", "format": "number"},
                        {"field": "training_validation_winner_agreement", "label": "8-vs-8同赢家", "type": "text"},
                        {"field": "prefix_8_to_16_rank_spearman", "label": "8-vs-16 rho", "format": "number"},
                        {"field": "median_effective_seed_fraction", "label": "有效seed比例", "format": "percent"},
                        {"field": "median_aggregate_log_likelihood_mcse", "label": "聚合MCSE", "format": "number"},
                        {"field": "noise_to_signal_ratio", "label": "noise/signal", "format": "number"},
                    ],
                },
                {
                    "id": "gate_results",
                    "title": "冻结的稳定性门槛结果",
                    "subtitle": "全部九项必须通过；本表不按结果重新加权或删减。",
                    "dataset": "gate_table",
                    "sourceId": "design",
                    "layout": "full",
                    "density": "dense",
                    "defaultSort": {"field": "gate", "direction": "asc"},
                    "columns": [
                        {"field": "gate", "label": "gate", "type": "text"},
                        {"field": "passed", "label": "通过", "type": "text"},
                    ],
                },
                {
                    "id": "method_sensitivity",
                    "title": "聚合方法敏感性",
                    "subtitle": "主指标log-mean-exp与次要mean-log候选排序及赢家比较。",
                    "dataset": "sensitivity_table",
                    "sourceId": "seed_scores",
                    "layout": "full",
                    "density": "spacious",
                    "defaultSort": {"field": "dataset_id", "direction": "asc"},
                    "columns": [
                        {"field": "dataset_id", "label": "dataset", "type": "text"},
                        {"field": "logmeanexp_vs_meanlog_rank_spearman", "label": "方法间rho", "format": "number"},
                        {"field": "logmeanexp_winner", "label": "log-mean-exp winner", "type": "text"},
                        {"field": "mean_log_winner", "label": "mean-log winner", "type": "text"},
                        {"field": "aggregation_method_winner_agreement", "label": "同赢家", "type": "text"},
                    ],
                },
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## 技术结论\n\n"
                        "**16-seed平均降低了PF likelihood的Monte Carlo误差，但仍没有产生可复现的唯一超参数排序。** "
                        f"有效seed比例中位数为 {summary['median_effective_seed_fraction']:.3f}，聚合MCSE为 "
                        f"{summary['median_aggregate_log_likelihood_mcse']:.3f}，noise/signal为 "
                        f"{summary['median_noise_to_signal_ratio']:.3f}，三项数值误差门槛均通过。"
                        f"然而独立8-vs-8排序rho中位数只有 {summary['final_split_median_rank_spearman']:.3f}，"
                        f"三个数据集的赢家一致率为 {summary['final_split_winner_agreement']:.0%}，"
                        "因此不能把16-seed赢家当成稳定超参数估计。"
                    ),
                    "sourceId": "summary",
                },
                {
                    "id": "headline_cards",
                    "type": "metric-strip",
                    "cardIds": [
                        "split_rank",
                        "winner_agreement",
                        "equivalence_size",
                        "effective_seeds",
                        "aggregate_mcse",
                    ],
                },
                {
                    "id": "split_finding",
                    "type": "markdown",
                    "body": (
                        "## 独立8-seed面板仍选出不同赢家\n\n"
                        "随着seed数增加，分半排序相关由接近0提高到中等水平，但最终三个轨迹的训练/验证赢家全部不同，"
                        "top-3集合Jaccard也都只有0.5。嵌套8-vs-16相关更高，是因为16-seed结果包含前8个seeds；"
                        "它不能替代真正独立的8-vs-8验证。"
                    ),
                    "sourceId": "split_stability",
                },
                {"id": "split_chart_block", "type": "chart", "chartId": "split_chart", "layout": "full"},
                {"id": "dataset_chart_block", "type": "chart", "chartId": "dataset_chart", "layout": "full"},
                {
                    "id": "equivalence_finding",
                    "type": "markdown",
                    "body": (
                        "## 每条轨迹仍有5–6个数值等价候选\n\n"
                        "16-seed完整聚合虽然能给出一个数值最小值，但配对bootstrap显示多数近优候选与该赢家的Delta NLL区间仍包含0。"
                        "这意味着当前证据更适合报告候选集合，而不是报告一个精确超参数组合；继续细调九候选全局排序的收益有限。"
                    ),
                    "sourceId": "equivalence",
                },
                {"id": "equivalence_chart_block", "type": "chart", "chartId": "equivalence_chart", "layout": "full"},
                {"id": "dataset_table_block", "type": "table", "tableId": "dataset_results", "layout": "full"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "body": (
                        "## 范围、数据与指标定义\n\n"
                        "分析固定三条128-trial自主合成轨迹、九个联合参数候选、128 particles和16个独立PF seeds，共432条候选得分。"
                        "主分数是16个独立likelihood estimates的算术平均取log；有效seed比例衡量该平均是否被极少数seeds主导；"
                        "MCSE使用相对likelihood的delta method。轨迹是技术覆盖案例，seeds和particles都不是行为样本量。"
                    ),
                    "sourceId": "seed_scores",
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## 冻结的训练、验证与等价集合方法\n\n"
                        "Seeds 0–7在运行前指定为训练面板，8–15指定为验证面板；同时保存2、4、8、16 seeds的嵌套前缀结果。"
                        "所有候选在同一数据集内按seed编号配对。等价集合用2000次配对seed bootstrap重算候选与16-seed赢家的Delta NLL；"
                        "区间包含0则保留。稳定性门槛在查看seeds 3–15之前写入配置，九项必须全部通过。"
                    ),
                    "sourceId": "design",
                },
                {"id": "gate_table_block", "type": "table", "tableId": "gate_results", "layout": "full"},
                {
                    "id": "robustness",
                    "type": "markdown",
                    "body": (
                        "## 结果不是单纯由一两个极端seed造成\n\n"
                        f"有效seed比例中位数 {summary['median_effective_seed_fraction']:.3f} 相当于约"
                        f" {16 * summary['median_effective_seed_fraction']:.1f}/16 个有效重复，最小比例也有"
                        f" {summary['minimum_effective_seed_fraction']:.3f}。但是log-mean-exp与mean-log的"
                        f"候选排序rho中位数只有 {summary['median_logmeanexp_vs_meanlog_rank_spearman']:.3f}，"
                        f"赢家一致率为 {summary['aggregation_method_winner_agreement']:.1%}。"
                        "因此剩余不确定性同时包含候选近等价和聚合目标敏感性，不能只归因于少数异常seed。"
                    ),
                    "sourceId": "summary",
                },
                {"id": "sensitivity_table_block", "type": "table", "tableId": "method_sensitivity", "layout": "full"},
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 解释边界\n\n"
                        "标准bootstrap PF的序列normalizing-constant estimate通常作为likelihood estimate使用，本报告据此采用log-mean-exp；"
                        "但没有为本项目全部自适应内部状态给出单独的无偏性证明。等价区间又以同一16 seeds选择出的赢家为参照，"
                        "所以它们是数值筛查区间而非校正选择偏差后的统计置信区间。三条固定轨迹也不能支持总体参数可恢复性结论。"
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 推荐下一步：停止全局九候选精排，转向有理论目标的配对比较\n\n"
                        "1. 不把当前16-seed赢家用于个体超参数报告或参数恢复真值判断。\n"
                        "2. 将候选集缩小到单参数或单机制的宽间距对照，直接检验paired Delta NLL的符号与Monte Carlo区间。\n"
                        "3. 对可在同一particle states上重读出的choice-layer机制先做低噪声审计。\n"
                        "4. 只有特定配对效应仍被PF噪声覆盖时，才针对该比较增加seeds，而不是继续给全部九候选平均加算力。\n"
                        "5. 在稳定的结构候选上再加入RT和oral report，检验它们是否缩小等价集合。"
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 后续需要回答的问题\n\n"
                        "特定机制的paired Delta NLL是否比全局超参数名次更稳定？"
                        "哪些参数块在拉大候选间距后仍形成相同choice predictions？"
                        "RT和oral report分别能排除等价集合中的哪些候选？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": snapshot_datasets,
        },
        "sources": sources,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    notes = f"""# Report source notes

- Audience: technical. Intended delivery mode: portable HTML from canonical `artifact.json`.
- The canonical artifact was generated successfully. Portable HTML rendering is blocked on this host because Node v12.22.9 cannot parse the packaged builder's nullish-coalescing syntax; see `HTML_BUILD_BLOCKER.md`.
- Required structure mapping: title -> `title`; technical summary -> `technical_summary`; key findings with visual evidence -> `split_finding`, `split_chart_block`, `dataset_chart_block`, `equivalence_finding`, `equivalence_chart_block`; scope/data/definitions -> `scope_definitions`; methodology -> `methodology`; limitations/uncertainty/robustness -> `robustness`, `limitations`, `gate_table_block`, `sensitivity_table_block`; recommended next steps -> `next_steps`; further questions -> `further_questions`.
- Chart map: split-half progression uses a grouped comparison bar; per-dataset independent versus nested stability uses grouped bars; equivalence-set size uses a single-series bar. Exact candidate intervals remain in the source CSV because a 27-row interval table is audit detail rather than the primary reading path.
- Source-row audit: seed scores={len(seed_scores)}; split comparisons={len(split)}; running comparisons={len(running)}; dataset summaries={len(datasets)}; equivalence rows={len(equivalence)}; sensitivity rows={len(sensitivity)}.
- The static scientific PNG is retained as a separate source-traceable QA figure and is not used as the report renderer.
"""
    (output.parent / "report_source_notes.md").write_text(notes, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
