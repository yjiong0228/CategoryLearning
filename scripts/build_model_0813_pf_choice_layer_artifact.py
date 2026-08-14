#!/usr/bin/env python3
"""Build the canonical technical-report artifact for the choice-layer audit."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_RESULTS = ROOT / (
    "results/model_dynamic_adaptive_control/0813_pf/mechanism_audit/"
    "02_choice_layer"
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
    contrasts = pd.read_csv(results / "contrast_summary.csv")
    subjects = pd.read_csv(results / "subject_contrast_summary.csv")
    seeds = pd.read_csv(results / "contrast_seed_scores.csv")
    runs = pd.read_csv(results / "run_summary.csv")

    labels = {
        "active_weight_sharpening": "Active weighting power",
        "active_strategy_confidence": "Active strategy confidence",
        "persistent_execution": "Persistent execution",
        "strategy_confidence_under_execution": "Strategy confidence | execution",
        "uniform_output_lapse": "Uniform output lapse",
    }
    contrasts["contrast_label"] = contrasts["contrast_id"].map(labels)
    subjects["contrast_label"] = subjects["contrast_id"].map(labels)
    contrast_order = list(labels)
    order_lookup = {value: index for index, value in enumerate(contrast_order)}
    contrasts["display_order"] = contrasts["contrast_id"].map(order_lookup)
    contrasts = contrasts.sort_values("display_order")
    subjects["display_order"] = subjects["contrast_id"].map(order_lookup)
    subjects = subjects.sort_values(["display_order", "subject_id"])
    stability_long = pd.concat(
        [
            contrasts[["contrast_id", "contrast_label"]]
            .assign(
                metric="Train-validation subject-rank rho",
                value=contrasts["train_validation_subject_spearman"].to_numpy(),
                gate=float(summary["stability_gates"]["minimum_train_validation_subject_spearman"]),
            ),
            contrasts[["contrast_id", "contrast_label"]]
            .assign(
                metric="Subject sign agreement",
                value=contrasts["subject_split_sign_agreement"].to_numpy(),
                gate=float(summary["stability_gates"]["minimum_subject_sign_agreement"]),
            ),
        ],
        ignore_index=True,
    )
    lookup = contrasts.set_index("contrast_id")
    headline = [
        {
            "stable_contrast_fraction": float(
                summary["numerically_stable_contrast_n"] / summary["contrast_n"]
            ),
            "persistent_execution_effect": float(
                lookup.loc["persistent_execution", "mean_subject_delta_nll"]
            ),
            "execution_strategy_effect": float(
                lookup.loc[
                    "strategy_confidence_under_execution",
                    "mean_subject_delta_nll",
                ]
            ),
            "uniform_lapse_effect": float(
                lookup.loc["uniform_output_lapse", "mean_subject_delta_nll"]
            ),
        }
    ]
    snapshots = {
        "headline": headline,
        "contrast_chart": _records(
            contrasts[
                [
                    "contrast_id",
                    "contrast_label",
                    "mean_subject_delta_nll",
                    "subject_bootstrap_ci_low",
                    "subject_bootstrap_ci_high",
                    "practical_effect_threshold",
                    "conditional_triage",
                ]
            ]
        ),
        "split_scatter": _records(
            subjects[
                [
                    "subject_id",
                    "contrast_id",
                    "contrast_label",
                    "training_mean_delta_nll",
                    "validation_mean_delta_nll",
                    "full_mean_delta_nll",
                ]
            ]
        ),
        "stability_long": _records(stability_long),
        "contrast_table": _records(contrasts),
        "subject_table": _records(
            subjects[
                [
                    "contrast_id",
                    "contrast_label",
                    "subject_id",
                    "training_mean_delta_nll",
                    "validation_mean_delta_nll",
                    "full_mean_delta_nll",
                    "paired_mean_delta_nll_mcse",
                    "split_sign_agreement",
                ]
            ]
        ),
    }
    sources = [
        _source(
            "summary",
            "Choice-layer audit summary",
            results / "summary.json",
            "Applies the frozen numerical gates and interpretation boundary to five common-state readout contrasts.",
            [
                "Positive delta mean NLL is comparator NLL minus mechanism NLL.",
                "Alternative readouts share baseline pre-choice particle states and do not update future weights.",
            ],
        ),
        _source(
            "contrast_summary",
            "Contrast-level paired-effect summary",
            results / "contrast_summary.csv",
            "Aggregates eight PF seeds within each of eight pilot subjects, then reports disjoint-panel stability and descriptive subject bootstrap intervals.",
            [
                "Seeds 0-3 are training and 4-7 are validation.",
                "Practical threshold is max(1% comparator mean NLL, 2 times median paired-seed effect SD).",
            ],
        ),
        _source(
            "subject_summary",
            "Subject-level paired effects",
            results / "subject_contrast_summary.csv",
            "Contains all eight pilot subjects for every contrast without effect-based exclusion.",
            [
                "Subject intervals resample PF seeds and quantify numerical rather than population uncertainty.",
            ],
        ),
        _source(
            "seed_scores",
            "Seed-level paired readout scores",
            results / "contrast_seed_scores.csv",
            "Contains all 320 subject-seed-contrast paired scores from 64 baseline PF runs.",
            [
                "Every contrast within a subject-seed run uses the same particle states and weights.",
            ],
        ),
        _source(
            "design",
            "Frozen choice-layer audit design",
            ROOT / "configs/specific_models/model_0813_pf_choice_layer_audit.yaml",
            "Defines subjects, common trial window, particles, seed panels, contrasts, stability gates and the practical-effect rule.",
            [
                "The common window was reduced from 128 to 64 trials before any formal effect was computed because subject 105 has only 64 eligible trials.",
            ],
        ),
    ]
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    title = "0813 common-state choice-layer 机制审计"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "在完全相同的 pre-choice PF states 上分解 choice-layer 机制的即时预测贡献与数值稳定性。",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": [
                {
                    "id": "stable_fraction",
                    "description": "五个冻结 contrasts 中同时通过 split-rank、sign、aggregate-sign 和 MCSE 门槛的比例。",
                    "dataset": "headline",
                    "sourceId": "summary",
                    "metrics": [
                        {
                            "label": "数值稳定 contrasts",
                            "field": "stable_contrast_fraction",
                            "format": "percent",
                        }
                    ],
                },
                {
                    "id": "persistent_effect",
                    "description": "active strategy mixture 到 persistent executed-rule readout 的 subject-mean paired NLL gain。",
                    "dataset": "headline",
                    "sourceId": "contrast_summary",
                    "metrics": [
                        {
                            "label": "Persistent execution delta NLL",
                            "field": "persistent_execution_effect",
                            "format": "number",
                            "signed": True,
                        }
                    ],
                },
                {
                    "id": "strategy_effect",
                    "description": "persistent execution 条件下 strategy confidence gain=2 相对 gain=0 的即时 paired NLL gain。",
                    "dataset": "headline",
                    "sourceId": "contrast_summary",
                    "metrics": [
                        {
                            "label": "Strategy-confidence delta NLL",
                            "field": "execution_strategy_effect",
                            "format": "number",
                            "signed": True,
                        }
                    ],
                },
                {
                    "id": "lapse_effect",
                    "description": "2% uniform lapse 相对 no-lapse executed-strategy readout 的即时 paired NLL gain。",
                    "dataset": "headline",
                    "sourceId": "contrast_summary",
                    "metrics": [
                        {
                            "label": "Uniform-lapse delta NLL",
                            "field": "uniform_lapse_effect",
                            "format": "number",
                            "signed": True,
                        }
                    ],
                },
            ],
            "charts": [
                {
                    "id": "effect_chart",
                    "title": "五个 frozen choice-layer contrasts 的 paired mean NLL gain",
                    "subtitle": "正值表示 mechanism-side readout 更好；完整区间与实用阈值见下表。",
                    "type": "bar",
                    "dataset": "contrast_chart",
                    "sourceId": "contrast_summary",
                    "xField": "contrast_label",
                    "xAxisTitle": "Choice-layer contrast",
                    "yAxisTitle": "Comparator - mechanism mean NLL",
                    "series": [
                        {
                            "field": "mean_subject_delta_nll",
                            "label": "Subject-mean delta NLL",
                        }
                    ],
                    "valueFormat": "number",
                    "layout": "full",
                },
                {
                    "id": "split_chart",
                    "title": "Disjoint seed panels 的 subject-level paired effects",
                    "subtitle": "每点为一个 pilot subject x contrast；training seeds 0-3，validation seeds 4-7。",
                    "type": "scatter",
                    "dataset": "split_scatter",
                    "sourceId": "subject_summary",
                    "xField": "training_mean_delta_nll",
                    "yField": "validation_mean_delta_nll",
                    "xAxisTitle": "Training paired delta NLL",
                    "yAxisTitle": "Validation paired delta NLL",
                    "seriesField": "contrast_label",
                    "layout": "full",
                },
                {
                    "id": "stability_chart",
                    "title": "逐 contrast 的 split-panel 稳定性",
                    "subtitle": "Rank rho 门槛0.70；subject sign agreement门槛0.75。",
                    "type": "bar",
                    "dataset": "stability_long",
                    "sourceId": "contrast_summary",
                    "xField": "contrast_label",
                    "xAxisTitle": "Choice-layer contrast",
                    "yAxisTitle": "Stability metric",
                    "series": [
                        {"field": "value", "label": "Stability value"}
                    ],
                    "seriesField": "metric",
                    "valueFormat": "number",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "contrast_results",
                    "title": "Contrast-level conditional triage",
                    "subtitle": "八名固定 pilot subjects、128 particles、八个 PF seeds；区间为描述性 subject bootstrap。",
                    "dataset": "contrast_table",
                    "sourceId": "contrast_summary",
                    "layout": "full",
                    "density": "spacious",
                    "defaultSort": {"field": "display_order", "direction": "asc"},
                    "columns": [
                        {"field": "contrast_label", "label": "contrast", "type": "text"},
                        {"field": "mechanism_id", "label": "mechanism", "type": "text"},
                        {"field": "mean_subject_delta_nll", "label": "mean delta NLL", "format": "number", "movement": True},
                        {"field": "subject_bootstrap_ci_low", "label": "CI low", "format": "number"},
                        {"field": "subject_bootstrap_ci_high", "label": "CI high", "format": "number"},
                        {"field": "train_validation_subject_spearman", "label": "split rho", "format": "number"},
                        {"field": "subject_split_sign_agreement", "label": "sign agreement", "format": "percent"},
                        {"field": "median_paired_mean_delta_nll_mcse", "label": "median MCSE", "format": "number"},
                        {"field": "practical_effect_threshold", "label": "practical threshold", "format": "number"},
                        {"field": "conditional_triage", "label": "conditional triage", "type": "text"},
                    ],
                },
                {
                    "id": "subject_results",
                    "title": "Subject-level conditional effects",
                    "subtitle": "每行保留一个 subject x contrast；用于检查 aggregate 是否被少数轨迹驱动。",
                    "dataset": "subject_table",
                    "sourceId": "subject_summary",
                    "layout": "full",
                    "density": "dense",
                    "defaultSort": {"field": "contrast_id", "direction": "asc"},
                    "columns": [
                        {"field": "contrast_label", "label": "contrast", "type": "text"},
                        {"field": "subject_id", "label": "subject", "format": "number"},
                        {"field": "training_mean_delta_nll", "label": "training delta", "format": "number", "movement": True},
                        {"field": "validation_mean_delta_nll", "label": "validation delta", "format": "number", "movement": True},
                        {"field": "full_mean_delta_nll", "label": "full delta", "format": "number", "movement": True},
                        {"field": "paired_mean_delta_nll_mcse", "label": "MCSE", "format": "number"},
                        {"field": "split_sign_agreement", "label": "same sign", "type": "text"},
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
                        "**首批 common-state 审计把 uniform lapse 与另外两个优先机制分开了。** "
                        "五个 contrasts 中三个通过全部 numerical gates。Persistent execution 的平均即时收益最大，"
                        "但 MCSE 与 subject heterogeneity 仍不允许直接保留；execution 条件下的 strategy confidence 数值稳定，"
                        "但 pilot interval 跨0；uniform lapse 数值稳定且没有 practical benefit，应降低后续优先级。"
                    ),
                    "sourceId": "contrast_summary",
                },
                {
                    "id": "headline_cards",
                    "type": "metric-strip",
                    "cardIds": [
                        "stable_fraction",
                        "persistent_effect",
                        "strategy_effect",
                        "lapse_effect",
                    ],
                },
                {
                    "id": "effect_finding",
                    "type": "markdown",
                    "body": (
                        "## Persistent execution 最大，但仍需要针对性增算力\n\n"
                        "Persistent execution 的 subject-mean delta NLL 为 +0.0221，明显大于其他读出层；"
                        "然而它的 95% pilot interval 为 [-0.0028, 0.0557]，median paired MCSE 为0.00306，未通过0.001门槛。"
                        "这不是删除证据，而是下一轮应优先对它做 full counterfactual common-seed comparator 的理由。"
                    ),
                    "sourceId": "contrast_summary",
                },
                {"id": "effect_chart_block", "type": "chart", "chartId": "effect_chart", "layout": "full"},
                {
                    "id": "split_finding",
                    "type": "markdown",
                    "body": (
                        "## Paired effects 的相对个体排序已高度复现\n\n"
                        "五个 contrasts 的 training-validation subject-rank rho 均在0.952到0.976之间，sign agreement为0.875到1.0。"
                        "这与九候选全局排序不稳定形成对照：缩小到理论明确且共享 particle states 的 paired effects 后，"
                        "数值复现性显著提高；剩余问题主要是特定 effect 的 MCSE 和跨被试异质性。"
                    ),
                    "sourceId": "contrast_summary",
                },
                {"id": "split_chart_block", "type": "chart", "chartId": "split_chart", "layout": "full"},
                {"id": "stability_chart_block", "type": "chart", "chartId": "stability_chart", "layout": "full"},
                {"id": "contrast_table_block", "type": "table", "tableId": "contrast_results", "layout": "full"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "body": (
                        "## 范围、数据与指标\n\n"
                        "分析保留八名预注册 pilot subjects，每人前64 trials、128 particles和八个 PF seeds。"
                        "正向 delta mean NLL 定义为 comparator NLL 减 mechanism NLL。Subject是描述性异质性单位；"
                        "PF seeds只是技术重复。原计划的128-trial窗口在任何正式效应计算前因subject 105只有64 trials而统一缩短。"
                    ),
                    "sourceId": "design",
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## Common-state 设计隔离即时读出效应\n\n"
                        "每个 subject-seed 只运行一次 fitted baseline PF，然后在同一 pre-choice particles和weights上重读"
                        "active expectation、weight sharpening、strategy confidence、persistent execution和uniform lapse。"
                        "Seeds 0-3与4-7构成不重叠面板。该设计最大化随机流配对，但替代读出不改变后续 filtering。"
                    ),
                    "sourceId": "seed_scores",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 条件分解不能代替完整机制拟合\n\n"
                        "结果不包含替代 readout 对后续 importance weights、resampling、latent paths 或 learning 的反馈。"
                        "八人 interval 也不是正式总体推断。Phase 0 已证明 OBS-01 在当前 persistent execution 路径下 exact no-op；"
                        "active-weighting contrast 仅用于 execution-off interaction，不应被解读为当前模型收益。"
                    ),
                    "sourceId": "summary",
                },
                {"id": "subject_table_block", "type": "table", "tableId": "subject_results", "layout": "full"},
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 下一步聚焦两个 full counterfactual comparators\n\n"
                        "1. 优先比较 persistent execution on/off，并使用 linked all-active beta scope comparator。\n"
                        "2. 在 execution on 条件下比较 strategy confidence gain 2 vs 0。\n"
                        "3. 两个比较继续使用 common seeds，并分别增加 seeds 直到 paired MCSE 过门槛；不恢复九候选全局排序。\n"
                        "4. Uniform lapse 暂不进入高成本结构筛选；只在 held-out calibration 或极端概率诊断显示必要时复核。\n"
                        "5. 稳定结构确定后再依次加入 RT 与 oral report。"
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 后续需要回答的问题\n\n"
                        "Persistent execution 的大平均效应是否主要由subject 103驱动？完整off-comparator是否保留其方向？"
                        "Strategy confidence 的正效应能否在32人确认样本和held-out trials中稳定？"
                        "Uniform lapse 是否只在少数极端-probability trials上改善校准？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": snapshots,
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
- Required structure mapping: title -> `title`; technical summary -> `technical_summary`; key findings with visual evidence -> `effect_finding`, `effect_chart_block`, `split_finding`, `split_chart_block`, `stability_chart_block`; scope/data/definitions -> `scope_definitions`; methodology -> `methodology`; limitations/uncertainty/robustness -> `limitations`, `contrast_table_block`, `subject_table_block`; recommended next steps -> `next_steps`; further questions -> `further_questions`.
- Chart map: category bars summarize signed conditional effects; a 40-point scatter checks disjoint seed panels at subject x contrast grain; grouped stability bars compare the two frozen panel-replication metrics. Exact intervals, MCSE and practical thresholds remain in the adjacent table.
- Source-row audit: runs={len(runs)}; seed contrasts={len(seeds)}; subject contrasts={len(subjects)}; contrast summaries={len(contrasts)}.
- The static scientific PNG is a separate source-traceable QA figure and is not used as the report renderer.
- On this host, portable HTML rendering is expected to remain blocked by Node v12.22.9; the exact packaged-builder result is recorded separately after the canonical artifact is built.
"""
    (output.parent / "report_source_notes.md").write_text(notes, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
