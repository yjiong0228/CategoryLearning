#!/usr/bin/env python3
"""Build the portable Phase-1 report for the 0813 PF calibration audit."""

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
    "01_pf_calibration"
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
    ranking = json.loads(
        (results / "ranking_summary.json").read_text(encoding="utf-8")
    )
    decomposition = json.loads(
        (results / "filter_decomposition_summary.json").read_text(
            encoding="utf-8"
        )
    )
    particles = pd.read_csv(results / "particle_count_summary.csv")
    subject_modes = pd.read_csv(
        results / "filter_decomposition_by_subject.csv"
    )
    contrasts = pd.read_csv(results / "filter_decomposition_contrasts.csv")
    ranking_scores = pd.read_csv(results / "ranking_scores.csv")
    decomposition_scores = pd.read_csv(
        results / "filter_decomposition_scores.csv"
    )

    mode_labels = {
        "unweighted_mixture": "均匀轨迹混合",
        "importance_no_resampling": "仅 choice weighting",
        "full_particle_filter": "完整 PF",
    }
    subject_chart = (
        subject_modes[
            ["subject_id", "mode_id", "mean_nll_mean"]
        ]
        .pivot(index="subject_id", columns="mode_id", values="mean_nll_mean")
        .reset_index()
        .rename(
            columns={
                "subject_id": "subject",
                "unweighted_mixture": "uniform_mixture",
                "importance_no_resampling": "choice_weighting",
                "full_particle_filter": "full_pf",
            }
        )
    )
    subject_chart["subject"] = subject_chart["subject"].map(
        lambda value: f"S{int(value)}"
    )
    particle_chart = particles[
        [
            "particle_count",
            "median_seed_candidate_rank_spearman",
            "minimum_seed_candidate_rank_spearman",
        ]
    ].rename(
        columns={
            "particle_count": "particles",
            "median_seed_candidate_rank_spearman": "median_rank_rho",
            "minimum_seed_candidate_rank_spearman": "minimum_rank_rho",
        }
    )
    particle_chart["particles"] = particle_chart["particles"].astype(str)
    particle_table = particles[
        [
            "particle_count",
            "seed_repeat_n",
            "median_seed_candidate_rank_spearman",
            "minimum_seed_candidate_rank_spearman",
            "mean_modal_winner_agreement",
            "median_candidate_total_nll_sd",
            "noise_to_signal_ratio",
            "median_cross_count_rank_spearman",
            "cross_count_winner_agreement",
            "all_stability_gates_pass",
        ]
    ].copy()
    particle_table["all_stability_gates_pass"] = particle_table[
        "all_stability_gates_pass"
    ].map({True: "yes", False: "no"})
    subject_table = subject_modes[
        [
            "subject_id",
            "mode_id",
            "mean_nll_mean",
            "mean_nll_std",
            "mean_post_choice_ess_fraction_mean",
            "resampling_fraction_mean",
        ]
    ].copy()
    subject_table["mode"] = subject_table["mode_id"].map(mode_labels)
    subject_table = subject_table.drop(columns="mode_id")

    maximum_count = int(max(ranking["particle_counts_evaluated"]))
    maximum_row = particles.loc[
        particles["particle_count"].astype(int).eq(maximum_count)
    ].iloc[0]
    weighting_gain = decomposition["contrasts"]["choice_weighting_gain"]
    resampling_gain = decomposition["contrasts"]["resampling_gain"]
    full_gain = decomposition["contrasts"]["full_filter_gain"]
    headline = [
        {
            "maximum_particles": maximum_count,
            "median_rank_rho": float(
                maximum_row["median_seed_candidate_rank_spearman"]
            ),
            "median_candidate_nll_sd": float(
                maximum_row["median_candidate_total_nll_sd"]
            ),
            "resampling_gain": float(resampling_gain["mean_of_subject_means"]),
        }
    ]
    datasets = {
        "headline": headline,
        "particle_chart": _records(particle_chart),
        "particle_table": _records(particle_table),
        "subject_chart": _records(subject_chart),
        "subject_table": _records(subject_table),
        "contrast_table": _records(contrasts),
    }
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    sources = [
        _source(
            "ranking_summary",
            "PF ranking-calibration summary",
            results / "ranking_summary.json",
            "Applied the frozen numerical stability gates to the reused 32/64-particle scores and the new 128/256-particle paired-seed scores.",
            [
                "Within-count stability is Spearman correlation of the nine candidate total NLL values between PF seeds.",
                "A particle count passes only when every preregistered repeat, ranking, winner, variance, noise-to-signal, and adjacent-count gate passes.",
            ],
        ),
        _source(
            "particle_summary",
            "Particle-count stability metrics",
            results / "particle_count_summary.csv",
            "One row per evaluated particle count with all component gates retained.",
            [
                "PF seeds and particles are technical repeats, not independent behavioral samples.",
                "Candidate NLL SD is the across-seed standard deviation for each dataset-candidate pair, summarized by its median.",
            ],
        ),
        _source(
            "decomposition_summary",
            "Filtering contribution summary",
            results / "filter_decomposition_summary.json",
            "Paired three outer-inference modes at equal particle, subject, trial, and seed budgets.",
            [
                "Positive gain means the later filtering layer has lower prequential mean choice NLL.",
                "The unweighted control disables only outer choice importance weighting; each StateModel still processes its observed task history.",
            ],
        ),
        _source(
            "calibration_config",
            "Frozen calibration design and gates",
            ROOT / "configs/specific_models/model_0813_pf_calibration.yaml",
            "Defines the fixed datasets, candidate bank, particle progression, paired seeds, decomposition modes, and stopping thresholds.",
            [
                "The design proceeds from 128 to 256 particles only when the previous stage fails all-gates stopping.",
            ],
        ),
    ]

    stable_text = (
        f"最小稳定设置为 {int(ranking['minimum_stable_particle_count'])} particles。"
        if ranking["minimum_stable_particle_count"] is not None
        else f"截至 {maximum_count} particles，仍没有设置同时通过全部数值门槛。"
    )
    title = "0813 粒子滤波的数值校准与贡献分解"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "机制审计 Phase 1：校准候选似然排序的数值稳定性，并分离 choice weighting 与 resampling 的贡献。",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": [
                {
                    "id": "maximum_particles",
                    "description": "按冻结的渐进规则实际评估到的最大粒子数。",
                    "dataset": "headline",
                    "sourceId": "ranking_summary",
                    "metrics": [
                        {
                            "label": "最大粒子数",
                            "field": "maximum_particles",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "median_rank_rho",
                    "description": "最大粒子数下，不同 PF seeds 的候选 NLL 排名相关中位数。",
                    "dataset": "headline",
                    "sourceId": "particle_summary",
                    "metrics": [
                        {
                            "label": "seed-rank ρ 中位数",
                            "field": "median_rank_rho",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "candidate_sd",
                    "description": "最大粒子数下，单候选跨 seed 总 NLL SD 的中位数；门槛为 0.50。",
                    "dataset": "headline",
                    "sourceId": "particle_summary",
                    "metrics": [
                        {
                            "label": "候选 NLL SD",
                            "field": "median_candidate_nll_sd",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "resampling_gain",
                    "description": "在 choice weighting 之上增加 resampling 的 subject-mean prequential mean-NLL 改善。",
                    "dataset": "headline",
                    "sourceId": "decomposition_summary",
                    "metrics": [
                        {
                            "label": "resampling gain",
                            "field": "resampling_gain",
                            "format": "number",
                        }
                    ],
                },
            ],
            "charts": [
                {
                    "id": "rank_chart",
                    "title": "增加粒子数没有稳定候选排序",
                    "subtitle": "点为不同 PF seeds 的候选排名相关；正式中位数门槛为 0.80。",
                    "type": "bar",
                    "dataset": "particle_chart",
                    "sourceId": "particle_summary",
                    "xField": "particles",
                    "xAxisTitle": "粒子数",
                    "yAxisTitle": "候选排名 Spearman ρ",
                    "series": [
                        {"field": "median_rank_rho", "label": "中位数"},
                        {"field": "minimum_rank_rho", "label": "最小值"},
                    ],
                    "valueFormat": "number",
                    "layout": "full",
                },
                {
                    "id": "decomposition_chart",
                    "title": "choice weighting 的收益不一致；resampling 修复权重退化",
                    "subtitle": "同一 subject、128 trials、128 particles 和配对 seeds 下的 mean NLL。",
                    "type": "bar",
                    "dataset": "subject_chart",
                    "sourceId": "decomposition_summary",
                    "xField": "subject",
                    "xAxisTitle": "覆盖案例",
                    "yAxisTitle": "Prequential mean choice NLL",
                    "series": [
                        {"field": "uniform_mixture", "label": "均匀轨迹混合"},
                        {"field": "choice_weighting", "label": "仅 choice weighting"},
                        {"field": "full_pf", "label": "完整 PF"},
                    ],
                    "valueFormat": "number",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "particle_results",
                    "title": "逐粒子数的完整数值门槛",
                    "subtitle": "32/64 粒子复用既有结果；128/256 粒子为本阶段新增。",
                    "dataset": "particle_table",
                    "sourceId": "particle_summary",
                    "layout": "full",
                    "density": "dense",
                    "defaultSort": {"field": "particle_count", "direction": "asc"},
                    "columns": [
                        {"field": "particle_count", "label": "particles", "format": "number"},
                        {"field": "seed_repeat_n", "label": "seeds", "format": "number"},
                        {"field": "median_seed_candidate_rank_spearman", "label": "rank ρ 中位数", "format": "number"},
                        {"field": "minimum_seed_candidate_rank_spearman", "label": "rank ρ 最小值", "format": "number"},
                        {"field": "mean_modal_winner_agreement", "label": "胜者一致率", "format": "percent"},
                        {"field": "median_candidate_total_nll_sd", "label": "候选 NLL SD", "format": "number"},
                        {"field": "noise_to_signal_ratio", "label": "noise/signal", "format": "number"},
                        {"field": "median_cross_count_rank_spearman", "label": "跨粒子 rank ρ", "format": "number"},
                        {"field": "cross_count_winner_agreement", "label": "跨粒子胜者一致率", "format": "percent"},
                        {"field": "all_stability_gates_pass", "label": "全部通过", "type": "text"},
                    ],
                },
                {
                    "id": "subject_results",
                    "title": "按覆盖案例分解过滤模式",
                    "subtitle": "seeds 是技术重复；subject 仅用于展示异质性，不做总体推断。",
                    "dataset": "subject_table",
                    "sourceId": "decomposition_summary",
                    "layout": "full",
                    "density": "spacious",
                    "defaultSort": {"field": "subject_id", "direction": "asc"},
                    "columns": [
                        {"field": "subject_id", "label": "subject", "format": "number"},
                        {"field": "mode", "label": "过滤模式", "type": "text"},
                        {"field": "mean_nll_mean", "label": "mean NLL", "format": "number"},
                        {"field": "mean_nll_std", "label": "seed SD", "format": "number"},
                        {"field": "mean_post_choice_ess_fraction_mean", "label": "post-choice ESS 比例", "format": "percent"},
                        {"field": "resampling_fraction_mean", "label": "重采样 trial 比例", "format": "percent"},
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
                        f"**{stable_text}** 最大设置下 seed-rank ρ 中位数为 "
                        f"{float(maximum_row['median_seed_candidate_rank_spearman']):.3f}，"
                        f"单候选总 NLL SD 中位数为 {float(maximum_row['median_candidate_total_nll_sd']):.3f}，"
                        f"相对 128 粒子的候选胜者一致率为 {float(maximum_row['cross_count_winner_agreement']):.1%}。"
                        "因此 Phase 1 的停止结论不是继续优化机制，而是先处理 PF 似然估计与候选可分性。"
                    ),
                },
                {
                    "id": "headline_cards",
                    "type": "metric-strip",
                    "cardIds": [
                        "maximum_particles",
                        "median_rank_rho",
                        "candidate_sd",
                        "resampling_gain",
                    ],
                },
                {
                    "id": "ranking_interpretation",
                    "type": "markdown",
                    "body": (
                        "## 粒子更多，但参数排名仍不可复现\n\n"
                        "从 32 增加到 256 粒子确实降低了单候选似然方差，但不同 seeds 给出的九候选排序仍接近无相关，"
                        "相邻粒子数的胜者也不一致。这说明当前固定超参看似能给出不错行为拟合，不能自动推出这些超参已被数据确认。"
                    ),
                    "sourceId": "ranking_summary",
                },
                {"id": "rank_chart_block", "type": "chart", "chartId": "rank_chart", "layout": "full"},
                {"id": "particle_table_block", "type": "table", "tableId": "particle_results", "layout": "full"},
                {
                    "id": "decomposition_interpretation",
                    "type": "markdown",
                    "body": (
                        "## 完整 PF 的平均优势主要来自重采样，而不是单独的 choice weighting\n\n"
                        f"均匀混合到仅 choice weighting 的 subject-mean gain 为 {float(weighting_gain['mean_of_subject_means']):.4f}，"
                        f"只在 {float(weighting_gain['positive_pair_fraction']):.1%} 的 subject-seed 配对中为正；"
                        f"在 weighting 之上增加重采样的 gain 为 {float(resampling_gain['mean_of_subject_means']):.4f}，"
                        f"9/9 配对均为正。完整 PF 相对均匀混合的 gain 为 {float(full_gain['mean_of_subject_means']):.4f}。"
                        "这里的 resampling 是防止 importance weights 退化的数值步骤，不应作为额外心理机制计入复杂性收益。"
                    ),
                    "sourceId": "decomposition_summary",
                },
                {"id": "decomposition_chart_block", "type": "chart", "chartId": "decomposition_chart", "layout": "full"},
                {"id": "subject_table_block", "type": "table", "tableId": "subject_results", "layout": "full"},
                {
                    "id": "scope_method",
                    "type": "markdown",
                    "body": (
                        "## 范围与方法\n\n"
                        "排名校准固定三条自主合成轨迹、九个联合参数候选和候选内配对 PF seeds。32/64 粒子复用 Phase 0 前已有的数值结果；"
                        "128/256 粒子按冻结门槛渐进增加。过滤分解使用三个真实行为历史覆盖案例的前 128 trials，"
                        "在相同 128 粒子与配对 seeds 下依次关闭或开启外层 choice importance weighting 与 resampling。"
                    ),
                    "sourceId": "calibration_config",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 解释边界\n\n"
                        "这些轨迹和 subjects 是技术覆盖案例，不是总体推断样本；PF seeds 也不是独立 n。"
                        "分解中的 NLL 是使用过去 observed history 的 prequential 指标，不是 held-out 泛化。"
                        "均匀混合只关闭外层选择权重，每个 StateModel 仍接收观察到的任务历史。"
                        "因此结果既不证明 choice conditioning 是心理机制，也不能把排序不稳完全归因于 PF 噪声；候选本身近等价也可能造成低相关。"
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 下一步\n\n"
                        "1. 暂停基于 choice-only PF likelihood 的机制去留与细粒度超参排名。\n"
                        "2. 先比较多-seed likelihood averaging、固定随机数或其他低方差估计，判断能否稳定候选排序。\n"
                        "3. 同时扩大候选间距或做单参数/单机制局部恢复，以区分数值噪声与结构等价。\n"
                        "4. 只有形成稳定 likelihood 基线后，才依次加入 RT 和 oral report，检验它们是否真正打破参数补偿。"
                    ),
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    notes = f"""# Report source notes

- Delivery target: portable HTML built from canonical `artifact.json`.
- Required-structure mapping: title → `title`; answer-first summary → `technical_summary`; visual evidence → `rank_chart_block` and `decomposition_chart_block`; definitions/method → `scope_method`; uncertainty and limitations → `limitations`; recommendations → `next_steps`.
- The scientific PNG remains a separate source-traceable figure; it is not used as an opaque dashboard screenshot.
- Source-row audit: ranking scores={len(ranking_scores)}; particle summaries={len(particles)}; decomposition scores={len(decomposition_scores)}; subject-mode summaries={len(subject_modes)}; technical contrasts={len(contrasts)}.
- Portable HTML blocker: the packaged report builder was invoked, but the available Node v12.22.9 cannot parse its required nullish-coalescing syntax. No ad hoc second renderer was created; `artifact.json` remains the canonical portable-report input and `HTML_BUILD_BLOCKER.md` records the exact failure.
"""
    (output.parent / "report_source_notes.md").write_text(
        notes, encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
