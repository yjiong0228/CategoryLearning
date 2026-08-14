#!/usr/bin/env python3
"""Build the portable technical report for the 0813 PF recovery audit."""

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

DEFAULT_RESULTS = (
    ROOT / "results/model_dynamic_adaptive_control/0813_pf/parameter_recovery"
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
    summary = json.loads(
        (results / "recovery_summary.json").read_text(encoding="utf-8")
    )
    stability = json.loads(
        (results / "stability_summary.json").read_text(encoding="utf-8")
    )
    parameters = pd.read_csv(results / "parameter_recovery_summary.csv")
    by_subject = pd.read_csv(results / "recovery_by_subject.csv")
    recovered = pd.read_csv(results / "recovered_datasets.csv")
    stability_recovered = pd.read_csv(
        results / "stability_recovered_settings.csv"
    )

    parameter_labels = {
        "Memory gamma": "记忆更新率 γ",
        "Exploration failure threshold": "探索失败阈值",
        "Execution switch scale": "执行切换尺度",
    }
    parameters["metric"] = parameters["label"].map(parameter_labels)
    recovery_rates = pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "metric": "九候选联合剖面",
                        "observed_rate": float(summary["exact_profile_recovery_rate"]),
                        "chance_rate": float(summary["profile_chance_rate"]),
                        "wilson_95_low": float(
                            summary["exact_profile_recovery_wilson_95"][0]
                        ),
                        "wilson_95_high": float(
                            summary["exact_profile_recovery_wilson_95"][1]
                        ),
                        "dataset_n": int(summary["dataset_n"]),
                    }
                ]
            ),
            parameters[
                [
                    "metric",
                    "exact_recovery_rate",
                    "chance_rate",
                    "wilson_95_low",
                    "wilson_95_high",
                    "dataset_n",
                ]
            ].rename(columns={"exact_recovery_rate": "observed_rate"}),
        ],
        ignore_index=True,
    )
    parameter_table = parameters[
        [
            "metric",
            "dataset_n",
            "exact_recovery_count",
            "exact_recovery_rate",
            "wilson_95_low",
            "wilson_95_high",
            "chance_rate",
            "mean_absolute_error",
            "spearman_true_posterior_mean",
        ]
    ].copy()
    subject_table = by_subject.rename(
        columns={
            "subject_id": "schedule_subject",
            "exact_profile_recovery_rate": "joint_profile_rate",
            "exact_memory_gamma_recovery_rate": "gamma_rate",
            "exact_exploration_failure_threshold_recovery_rate": "threshold_rate",
            "exact_execution_switch_scale_recovery_rate": "switch_scale_rate",
        }
    )
    subject_chart = subject_table[
        ["schedule_subject", "joint_profile_rate", "dataset_n"]
    ].copy()
    subject_chart["schedule_subject"] = subject_chart["schedule_subject"].map(
        lambda value: f"Schedule {int(value)}"
    )
    ambiguity = pd.DataFrame(
        [
            {
                "dataset_n": int(summary["dataset_n"]),
                "exact_profile_rate": float(summary["exact_profile_recovery_rate"]),
                "true_within_delta2_rate": float(
                    summary["true_profile_within_delta_nll_rate"]
                ),
                "median_true_rank": float(summary["median_true_profile_rank"]),
                "mean_true_posterior": float(
                    summary["mean_true_profile_posterior"]
                ),
                "median_effective_profiles": float(
                    summary["median_effective_profile_count"]
                ),
                "median_near_best_profiles": float(
                    summary["median_near_best_profile_count"]
                ),
                "median_runner_up_delta_nll": float(
                    summary["median_runner_up_delta_nll"]
                ),
            }
        ]
    )
    stability_table = stability_recovered[
        [
            "true_profile_id",
            "setting",
            "predicted_profile_id",
            "true_profile_recovered",
        ]
    ].copy()
    stability_table["true_profile_recovered"] = stability_table[
        "true_profile_recovered"
    ].map({True: "yes", False: "no"})

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    sources = [
        _source(
            "recovery_summary",
            "0813 PF joint parameter-recovery summary",
            results / "recovery_summary.json",
            "Summarized nine-profile recovery over 72 autonomous synthetic trajectories using the primary 32-particle paired-seed fits.",
            [
                "Exact profile recovery means the maximum-likelihood L9 candidate equals the generating profile.",
                "The near-best set contains candidates whose total NLL is within 2 of the best candidate.",
                "The effective profile count is exp of the discrete candidate-posterior entropy under a uniform profile prior.",
            ],
        ),
        _source(
            "parameter_summary",
            "Parameter-level recovery estimates",
            results / "parameter_recovery_summary.csv",
            "Computed parameter-level exact recovery, posterior-mean error, rank ordering, and Wilson intervals from the saved recovered-dataset table.",
            [
                "Each parameter has three discrete levels and therefore a one-third chance rate.",
                "Wilson intervals use synthetic trajectories as simulation replicates conditional on four fixed schedules.",
            ],
        ),
        _source(
            "subject_blocks",
            "Recovery by fixed stimulus/category schedule",
            results / "recovery_by_subject.csv",
            "Stratified recovery by the four real condition-1 schedules used as fixed design blocks.",
            [
                "Each schedule contributes 18 trajectories: nine profiles times two autonomous replicates.",
            ],
        ),
        _source(
            "stability_summary",
            "Particle-filter numerical stability audit",
            results / "stability_summary.json",
            "Compared candidate rankings for three predeclared datasets under 32 and 64 particles and two independent filter seeds.",
            [
                "Stability runs are numerical checks and do not increase the recovery sample size.",
                "Candidate-rank stability is the Spearman correlation of the nine candidate NLL values between numerical settings.",
            ],
        ),
        _source(
            "recovery_config",
            "Predeclared recovery design and parameter support",
            ROOT / "configs/specific_models/model_0813_pf_parameter_recovery.yaml",
            "Defines the L9 factor levels, fixed schedules, trial count, particle budget, paired-seed policy, and stability subset.",
            [
                "The L9 bank varies memory gamma, exploration failure threshold, and execution switch scale while fixing every other v2f setting.",
            ],
        ),
    ]

    headline_primary = [
        {
            "exact_profile_rate": float(summary["exact_profile_recovery_rate"]),
            "true_within_delta2_rate": float(
                summary["true_profile_within_delta_nll_rate"]
            ),
            "effective_profile_count": float(
                summary["median_effective_profile_count"]
            ),
        }
    ]
    headline_stability = [
        {
            "median_rank_spearman": float(
                stability["median_pairwise_candidate_nll_spearman"]
            )
        }
    ]
    datasets = {
        "headline_primary": headline_primary,
        "headline_stability": headline_stability,
        "recovery_rates": _records(recovery_rates),
        "parameter_table": _records(parameter_table),
        "subject_chart": _records(subject_chart),
        "subject_table": _records(subject_table),
        "ambiguity": _records(ambiguity),
        "stability_table": _records(stability_table),
    }

    exact_count = int(summary["exact_profile_recovery_count"])
    dataset_n = int(summary["dataset_n"])
    exact_rate = float(summary["exact_profile_recovery_rate"])
    exact_low, exact_high = summary["exact_profile_recovery_wilson_95"]
    within_count = int(summary["true_profile_within_delta_nll_count"])
    within_rate = float(summary["true_profile_within_delta_nll_rate"])
    effective = float(summary["median_effective_profile_count"])
    runner_up = float(summary["median_runner_up_delta_nll"])
    median_rank_rho = float(stability["median_pairwise_candidate_nll_spearman"])
    modal_agreement = float(stability["mean_within_dataset_modal_winner_agreement"])
    rho32 = float(
        stability["by_particle_count"]["32"][
            "median_seed_candidate_rank_spearman"
        ]
    )
    rho64 = float(
        stability["by_particle_count"]["64"][
            "median_seed_candidate_rank_spearman"
        ]
    )

    title = "0813 粒子滤波模型的联合参数恢复"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "三项 v2f 超参数的平衡联合恢复、参数混淆和粒子滤波数值稳定性审计。",
            "generatedAt": generated_at,
            "sources": sources,
            "cards": [
                {
                    "id": "exact_profile",
                    "description": "九个联合参数剖面中，最大似然候选精确等于生成剖面的比例；机会水平为 11.1%。",
                    "dataset": "headline_primary",
                    "sourceId": "recovery_summary",
                    "metrics": [
                        {
                            "label": "联合剖面精确恢复",
                            "field": "exact_profile_rate",
                            "format": "percent",
                        }
                    ],
                },
                {
                    "id": "true_near_best",
                    "description": "真实生成剖面仍落在距最佳总 NLL 不超过 2 的候选集合中的比例。",
                    "dataset": "headline_primary",
                    "sourceId": "recovery_summary",
                    "metrics": [
                        {
                            "label": "真实剖面仍近优",
                            "field": "true_within_delta2_rate",
                            "format": "percent",
                        }
                    ],
                },
                {
                    "id": "effective_profiles",
                    "description": "按离散候选后验熵换算的有效候选数中位数，候选总数为 9。",
                    "dataset": "headline_primary",
                    "sourceId": "recovery_summary",
                    "metrics": [
                        {
                            "label": "有效候选数中位数",
                            "field": "effective_profile_count",
                            "format": "number",
                        }
                    ],
                },
                {
                    "id": "rank_stability",
                    "description": "三条预先指定轨迹在 32/64 粒子和独立 PF 种子之间的候选 NLL 排名相关中位数。",
                    "dataset": "headline_stability",
                    "sourceId": "stability_summary",
                    "metrics": [
                        {
                            "label": "PF 排名稳定性",
                            "field": "median_rank_spearman",
                            "format": "number",
                        }
                    ],
                },
            ],
            "charts": [
                {
                    "id": "recovery_rate_chart",
                    "title": "联合剖面与单参数精确恢复率",
                    "subtitle": "72 条独立合成轨迹；观察值与相应离散机会水平并列，精确区间见下表。",
                    "type": "bar",
                    "dataset": "recovery_rates",
                    "sourceId": "parameter_summary",
                    "xField": "metric",
                    "xAxisTitle": "恢复目标",
                    "yAxisTitle": "精确恢复率",
                    "series": [
                        {"field": "observed_rate", "label": "观察恢复率"},
                        {"field": "chance_rate", "label": "离散机会水平"},
                    ],
                    "valueFormat": "percent",
                    "layout": "full",
                },
                {
                    "id": "subject_rate_chart",
                    "title": "联合剖面恢复依赖刺激序列",
                    "subtitle": "每个固定序列 18 条轨迹；区间不解释为被试总体不确定性。",
                    "type": "bar",
                    "dataset": "subject_chart",
                    "sourceId": "subject_blocks",
                    "xField": "schedule_subject",
                    "xAxisTitle": "固定刺激/类别序列",
                    "yAxisTitle": "联合剖面精确恢复率",
                    "series": [
                        {"field": "joint_profile_rate", "label": "精确恢复率"}
                    ],
                    "valueFormat": "percent",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "parameter_results",
                    "title": "单参数恢复与等级排序",
                    "subtitle": "每个参数三个离散水平；95% Wilson 区间以合成轨迹为模拟重复。",
                    "dataset": "parameter_table",
                    "sourceId": "parameter_summary",
                    "layout": "full",
                    "density": "spacious",
                    "defaultSort": {"field": "exact_recovery_rate", "direction": "desc"},
                    "columns": [
                        {"field": "metric", "label": "参数", "type": "text"},
                        {"field": "dataset_n", "label": "合成轨迹 n", "format": "number"},
                        {"field": "exact_recovery_count", "label": "精确命中", "format": "number"},
                        {"field": "exact_recovery_rate", "label": "恢复率", "format": "percent"},
                        {"field": "wilson_95_low", "label": "95% CI 下限", "format": "percent"},
                        {"field": "wilson_95_high", "label": "95% CI 上限", "format": "percent"},
                        {"field": "chance_rate", "label": "机会水平", "format": "percent"},
                        {"field": "mean_absolute_error", "label": "后验均值 MAE", "format": "number"},
                        {"field": "spearman_true_posterior_mean", "label": "真实值–后验均值 ρ", "format": "number"},
                    ],
                },
                {
                    "id": "ambiguity_results",
                    "title": "候选等价性的整体诊断",
                    "subtitle": "近优阈值为总 NLL 距最佳不超过 2。",
                    "dataset": "ambiguity",
                    "sourceId": "recovery_summary",
                    "layout": "full",
                    "density": "spacious",
                    "defaultSort": {"field": "dataset_n", "direction": "desc"},
                    "columns": [
                        {"field": "dataset_n", "label": "合成轨迹 n", "format": "number"},
                        {"field": "exact_profile_rate", "label": "联合精确恢复", "format": "percent"},
                        {"field": "true_within_delta2_rate", "label": "真实剖面仍近优", "format": "percent"},
                        {"field": "median_true_rank", "label": "真实剖面名次中位数", "format": "number"},
                        {"field": "mean_true_posterior", "label": "真实剖面平均后验", "format": "percent"},
                        {"field": "median_effective_profiles", "label": "有效候选数中位数", "format": "number"},
                        {"field": "median_near_best_profiles", "label": "近优候选数中位数", "format": "number"},
                        {"field": "median_runner_up_delta_nll", "label": "次优–最优 ΔNLL 中位数", "format": "number"},
                    ],
                },
                {
                    "id": "subject_results",
                    "title": "按固定刺激/类别序列分层的恢复率",
                    "subtitle": "序列是固定设计块，不是四名总体被试的推断样本。",
                    "dataset": "subject_table",
                    "sourceId": "subject_blocks",
                    "layout": "full",
                    "density": "spacious",
                    "defaultSort": {"field": "schedule_subject", "direction": "asc"},
                    "columns": [
                        {"field": "schedule_subject", "label": "序列 ID", "format": "number"},
                        {"field": "dataset_n", "label": "轨迹 n", "format": "number"},
                        {"field": "joint_profile_rate", "label": "联合剖面", "format": "percent"},
                        {"field": "gamma_rate", "label": "记忆 γ", "format": "percent"},
                        {"field": "threshold_rate", "label": "探索阈值", "format": "percent"},
                        {"field": "switch_scale_rate", "label": "切换尺度", "format": "percent"},
                    ],
                },
                {
                    "id": "stability_results",
                    "title": "PF 数值设置下的胜出候选",
                    "subtitle": "三条预先指定轨迹 × 32/64 粒子 × 两个独立 PF 种子。",
                    "dataset": "stability_table",
                    "sourceId": "stability_summary",
                    "layout": "full",
                    "density": "dense",
                    "defaultSort": {"field": "true_profile_id", "direction": "asc"},
                    "columns": [
                        {"field": "true_profile_id", "label": "生成剖面", "type": "text"},
                        {"field": "setting", "label": "PF 设置", "type": "text"},
                        {"field": "predicted_profile_id", "label": "胜出剖面", "type": "text"},
                        {"field": "true_profile_recovered", "label": "命中生成剖面", "type": "text"},
                    ],
                },
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## 技术摘要\n\n"
                        f"**当前 32 粒子设置不足以支持把这三项固定超参数解释为可可靠识别的个体参数。** "
                        f"九个联合剖面仅精确恢复 {exact_count}/{dataset_n}（{exact_rate:.1%}；"
                        f"95% Wilson CI {float(exact_low):.1%}–{float(exact_high):.1%}），"
                        "对应机会水平为 11.1%。三个单参数的恢复率为 37.5%–43.1%，"
                        "其区间均覆盖各自 33.3% 的机会水平。\n\n"
                        f"候选并没有收缩到唯一解释：有效候选数中位数为 {effective:.2f}/9，"
                        f"真实生成剖面仅在 {within_count}/{dataset_n}（{within_rate:.1%}）"
                        "的数据中落入 ΔNLL≤2 的近优集合。数值复核进一步显示，"
                        f"候选 NLL 排名在 32/64 粒子与独立种子之间的 Spearman ρ 中位数仅 {median_rank_rho:.3f}。"
                        "因此，这一结果证明的是‘现有推断配置下不可可靠恢复’，而不是已经把失败唯一归因于心理模型的结构等价。"
                    ),
                },
                {
                    "id": "headline_cards",
                    "type": "metric-strip",
                    "cardIds": [
                        "exact_profile",
                        "true_near_best",
                        "effective_profiles",
                        "rank_stability",
                    ],
                },
                {
                    "id": "key_findings",
                    "type": "markdown",
                    "body": (
                        "## 精确恢复没有与离散机会水平清楚分离\n\n"
                        "下图将联合剖面和三个组成参数分别与其机会水平比较。联合剖面命中率略高于 1/9，"
                        "但区间下限为 10.9%，仍覆盖 11.1% 的机会水平；单参数也只有弱的等级信息。"
                        "因此不能因为真实被试拟合较好，就把当前固定 γ、探索阈值或切换尺度当作已经被行为数据单独确认的参数。"
                    ),
                },
                {"id": "rate_chart", "type": "chart", "chartId": "recovery_rate_chart", "layout": "full"},
                {"id": "parameter_table_block", "type": "table", "tableId": "parameter_results", "layout": "full"},
                {
                    "id": "equivalence",
                    "type": "markdown",
                    "body": (
                        "## 多个参数组合仍给出相近的选择似然\n\n"
                        f"最佳候选与次优候选的总 NLL 差中位数只有 {runner_up:.2f}；"
                        f"每条轨迹的近优候选数中位数为 {float(summary['median_near_best_profile_count']):.1f}，"
                        f"真实剖面的名次中位数为 {float(summary['median_true_profile_rank']):.1f}。"
                        "这正是参数补偿的表现：不同记忆速度、失败门槛和外显切换尺度，可以通过不同潜在轨迹产生相近的逐试次选择概率。"
                        "但由于后面的 PF 数值稳定性门槛没有通过，此处应称为‘候选等价与数值噪声共同造成的歧义’，不能只归因于参数补偿。"
                    ),
                    "sourceId": "recovery_summary",
                },
                {"id": "ambiguity_table_block", "type": "table", "tableId": "ambiguity_results", "layout": "full"},
                {
                    "id": "schedule_dependence",
                    "type": "markdown",
                    "body": (
                        "## 可恢复性明显依赖具体刺激序列\n\n"
                        "四个固定序列的联合剖面恢复率从 0% 到 33.3%。这不是被试间参数差异，因为生成参数支持完全相同；"
                        "它说明同样长度的 128 个试次，因刺激和类别顺序不同，对参数提供的信息量可以差很多。"
                        "所以后续正式恢复不能只用一条便利的刺激序列，也不能把所有试次当作独立 n。"
                    ),
                    "sourceId": "subject_blocks",
                },
                {"id": "subject_chart_block", "type": "chart", "chartId": "subject_rate_chart", "layout": "full"},
                {"id": "subject_table_block", "type": "table", "tableId": "subject_results", "layout": "full"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "body": (
                        "## 范围、数据与指标定义\n\n"
                        "独立模拟单位是一条由模型自主作答产生的合成选择轨迹；每条轨迹在选择之后才由任务环境返回正确/错误反馈。"
                        "设计包含 9 个平衡 L9 联合剖面、4 个真实 condition-1 刺激/类别序列和每剖面–序列 2 次独立重复，共 72 条轨迹，"
                        "每条 128 试次。粒子与试次不是独立样本。Wilson 区间描述的是在这四个固定序列条件下的模拟重复不确定性，"
                        "不是总体被试置信区间。"
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## 联合恢复怎样实施\n\n"
                        "L9 正交设计联合改变记忆 γ（0.60/0.80/0.95）、探索失败阈值（0.40/0.55/0.70）和执行切换尺度（0.10/0.20/0.40），"
                        "并包含 0813 的全中心基线。其余 v2f 参数、29 规则空间、读出和 lapse 均固定。"
                        "每条合成数据由单一潜在认知轨迹生成；拟合时，九个候选分别用普通 bootstrap PF 对潜在轨迹积分。"
                        "同一数据集内所有候选共用配对 PF 种子，以降低不相关的蒙特卡洛差异。"
                        "候选后验使用九候选均匀先验归一化完整序列似然，仅作为离散可识别性诊断。"
                    ),
                    "sourceId": "recovery_config",
                },
                {
                    "id": "numerical_stability",
                    "type": "markdown",
                    "body": (
                        "## PF 数值稳定性没有达到参数比较所需水平\n\n"
                        f"三条预先指定轨迹在四种数值设置中的模态胜者一致率平均只有 {modal_agreement:.1%}；"
                        f"同一粒子数下两个 PF 种子的候选排序相关中位数在 32 粒子时为 {rho32:.3f}，在 64 粒子时为 {rho64:.3f}。"
                        "增加到 64 粒子降低了单候选跨种子总 NLL 标准差中位数，但没有稳定候选排序。"
                        "所以当前 32 粒子的似然适合做行为预测与潜在状态积分，不足以直接作为细粒度超参数排名已经收敛的证据。"
                    ),
                    "sourceId": "stability_summary",
                },
                {"id": "stability_table_block", "type": "table", "tableId": "stability_results", "layout": "full"},
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## 局限与当前可下的结论\n\n"
                        "本轮只变化三项参数，L9 也不是完整的 3×3×3 网格；beta、lapse、读出增益、容量和其余控制器参数都固定。"
                        "因此它不能证明所有 v2f 超参数都不可恢复。反过来，由于数值稳定性失败，也不能把低恢复率完全解释为心理机制的结构不可识别。"
                        "最稳妥的结论是：**在 128 试次、32 粒子和当前三因素联合候选库下，恢复结果不足以支持唯一参数解释。**"
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## 推荐的下一步\n\n"
                        "1. **先校准 PF 似然，再做大规模超参优化。** 在小型预声明数据集上提高粒子数并平均多个独立 PF 似然，直到候选排序稳定；"
                        "可在运行前规定排名相关和胜者一致率门槛。\n"
                        "2. **分块恢复，不要一次开放所有参数。** 先分别恢复记忆、探索控制、执行切换和 choice readout，再只联合那些单独可恢复的块。\n"
                        "3. **增加独立观测约束。** 选择数据不足时，把 RT 或口头规则报告作为预先规定的联合观测，而不是继续增加选择层自由参数。\n"
                        "4. **延长并优化任务信息。** 用更长轨迹、多种刺激顺序，或基于预期信息增益设计刺激；当前四个序列的差异表明试次顺序本身决定可识别性。\n"
                        "5. **主文中区分预测与参数解释。** 当前 PF 模型可以继续作为预测/潜在状态边际化工具，但固定参数应表述为工作设定，不能写成已从个体行为精确估计。"
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## 后续需要回答的问题\n\n"
                        "第一，候选排序要在多少粒子和多少独立 PF 重复后才达到预设稳定门槛？"
                        "第二，把轨迹从 128 延长到每名被试的完整长度，恢复提高的是哪一类参数？"
                        "第三，RT 与口头报告能否打破记忆–探索–执行切换之间的等价，而不是只提高整体拟合？"
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
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    source_notes = f"""# Report source notes

- Delivery mode: portable HTML.
- Audience: technical.
- Required-structure mapping: title → `title`; technical summary → `technical_summary`; key findings with visual evidence → `key_findings`, `equivalence`, and `schedule_dependence`; scope/data/definitions → `scope_definitions`; methodology/model specification → `methodology`; limitations/uncertainty/robustness → `numerical_stability` and `limitations`; recommended next steps → `next_steps`; further questions → `further_questions`.
- The static four-panel PNG is retained as a supporting scientific figure, not embedded as a second report renderer.
- Profile-confusion detail remains in `profile_confusion.csv`; the report uses the recovery-rate chart and exact audit tables to keep the reading path compact.
- Portable HTML blocker: the packaged builder was invoked, but the available Node v12.22.9 cannot parse its required nullish-coalescing/optional-chaining syntax. No parallel ad hoc HTML renderer was created; `artifact.json` remains the canonical report input.
- Source-row audit: recovered datasets={len(recovered)}; parameter rows={len(parameters)}; schedule rows={len(by_subject)}; stability setting rows={len(stability_recovered)}.
"""
    (output.parent / "report_source_notes.md").write_text(
        source_notes, encoding="utf-8"
    )
    print(json.dumps({"artifact": str(output), "datasets": list(datasets)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
