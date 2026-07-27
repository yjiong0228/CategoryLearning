#!/usr/bin/env python3
"""Build a canonical portable report for the model architecture effect audit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "results/model_architecture_effect_audit"
REPORT_DIR = AUDIT_DIR / "report"


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    summary = json.loads(
        (AUDIT_DIR / "effect_audit_summary.json").read_text(encoding="utf-8")
    )
    effects = pd.DataFrame(summary["paired_corrected_effects"])
    subject_effects = pd.read_csv(AUDIT_DIR / "subject_mechanism_effects.csv")
    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def effect(name: str) -> pd.Series:
        matches = effects[effects.mechanism.eq(name)]
        if len(matches) != 1:
            raise ValueError(f"Expected one effect row for {name!r}, found {len(matches)}")
        return matches.iloc[0]

    beta = effect("dynamic hypothesis-specific beta versus static beta=5")
    readout = effect("flexible readout versus expectation")
    state = effect("latent volatility state (frozen selected mode)")
    perception = effect("calibrated perception noise")
    labels = effect("label-reversed hypothesis copies (unconfounded 7)")
    active = effect("dynamic active-set selection versus full hypothesis set")
    stable = effect("controller stable profile")
    aggressive = effect("controller aggressive profile")
    conservative = effect("controller conservative profile")
    stubborn = effect("controller stubborn profile")
    choice = effect("choice-informed transition scoring")
    dual_static = effect("dual memory versus static-only")
    dual_fade = effect("dual memory versus fade-only")

    decision_rows = [
        {
            "mechanism": "Dynamic hypothesis-specific beta",
            "decision": "保留并重构",
            "mean_brier_benefit": beta.mean_brier_benefit,
            "brier_wins": f"{int(beta.brier_helped_subjects)}/{int(beta.subject_count)}",
            "mean_crps_benefit": beta.mean_crps_benefit,
            "evidence": "A*",
            "reason": "最大且最稳定的局部效应；仍需与最优静态 beta 而非固定 5 做最终比较。",
        },
        {
            "mechanism": "Flexible choice readout",
            "decision": "保留但参数化",
            "mean_brier_benefit": readout.mean_brier_benefit,
            "brier_wins": f"{int(readout.brier_helped_subjects)}/{int(readout.subject_count)}",
            "mean_crps_benefit": readout.mean_crps_benefit,
            "evidence": "A* + B",
            "reason": "修正后 8/8 改善；用一个连续 concentration/temperature 取代四套分支。",
        },
        {
            "mechanism": "Dynamic active set / controller",
            "decision": "保留为可选容量分支",
            "mean_brier_benefit": active.mean_brier_benefit,
            "brier_wins": f"{int(active.brier_helped_subjects)}/{int(active.subject_count)}",
            "mean_crps_benefit": active.mean_crps_benefit,
            "evidence": "A* + B",
            "reason": "平均有益但高度异质；3 名被试完整假设集更好，正式搜索必须加入 full-set 分支。",
        },
        {
            "mechanism": "Stable controller profile",
            "decision": "保留",
            "mean_brier_benefit": stable.mean_brier_benefit,
            "brier_wins": f"{int(stable.brier_helped_subjects)}/{int(stable.subject_count)}",
            "mean_crps_benefit": stable.mean_crps_benefit,
            "evidence": "A*",
            "reason": "四个 profile 中效应最大，尤其解释高表现被试。",
        },
        {
            "mechanism": "Conservative profile",
            "decision": "合并后复测",
            "mean_brier_benefit": conservative.mean_brier_benefit,
            "brier_wins": f"{int(conservative.brier_helped_subjects)}/{int(conservative.subject_count)}",
            "mean_crps_benefit": conservative.mean_crps_benefit,
            "evidence": "A*",
            "reason": "平均收益小且个体方向不一致；不支持原样独立成一套复杂 profile。",
        },
        {
            "mechanism": "Stubborn profile",
            "decision": "暂留／尝试合并",
            "mean_brier_benefit": stubborn.mean_brier_benefit,
            "brier_wins": f"{int(stubborn.brier_helped_subjects)}/{int(stubborn.subject_count)}",
            "mean_crps_benefit": stubborn.mean_crps_benefit,
            "evidence": "A*",
            "reason": "6/8 有益但平均效应不大；优先与 conservative 合为低探索状态。",
        },
        {
            "mechanism": "Aggressive profile",
            "decision": "移出主模型",
            "mean_brier_benefit": aggressive.mean_brier_benefit,
            "brier_wins": f"{int(aggressive.brier_helped_subjects)}/{int(aggressive.subject_count)}",
            "mean_crps_benefit": aggressive.mean_crps_benefit,
            "evidence": "A*",
            "reason": "仅 2/8 Brier 受益，平均上删除后更好；可作为亚组扩展而非核心。",
        },
        {
            "mechanism": "Persistent latent-volatility state",
            "decision": "降级为亚组扩展",
            "mean_brier_benefit": state.mean_brier_benefit,
            "brier_wins": f"{int(state.brier_helped_subjects)}/{int(state.subject_count)}",
            "mean_crps_benefit": state.mean_crps_benefit,
            "evidence": "A* + B",
            "reason": "平均 Brier 约 0.001、CRPS 无收益；且主要驱动的 aggressive profile 本身不稳。",
        },
        {
            "mechanism": "Choice-informed transition scoring",
            "decision": "不进通用主模型",
            "mean_brier_benefit": choice.mean_brier_benefit,
            "brier_wins": f"{int(choice.brier_helped_subjects)}/{int(choice.subject_count)}",
            "mean_crps_benefit": choice.mean_crps_benefit,
            "evidence": "A*",
            "reason": "仅 4 名相关被试，2/4 Brier 受益而平均 CRPS 变差；只保留为候选亚组。",
        },
        {
            "mechanism": "Dual memory versus static-only",
            "decision": "先以 static-only 为简约基线",
            "mean_brier_benefit": dual_static.mean_brier_benefit,
            "brier_wins": f"{int(dual_static.brier_helped_subjects)}/{int(dual_static.subject_count)}",
            "mean_crps_benefit": dual_static.mean_crps_benefit,
            "evidence": "A*",
            "reason": "dual 未胜过 static-only；当前搜索又未包含 w0=1，需补端点后再决定是否保留双通道。",
        },
        {
            "mechanism": "Dual memory versus fade-only",
            "decision": "不采用 fade-only",
            "mean_brier_benefit": dual_fade.mean_brier_benefit,
            "brier_wins": f"{int(dual_fade.brier_helped_subjects)}/{int(dual_fade.subject_count)}",
            "mean_crps_benefit": dual_fade.mean_crps_benefit,
            "evidence": "A*",
            "reason": "fade-only 在 6/8 被试更差；如果简化，方向应是 static-only 而非只保留衰减轨道。",
        },
        {
            "mechanism": "Calibrated perception noise",
            "decision": "移出主行为拟合",
            "mean_brier_benefit": perception.mean_brier_benefit,
            "brier_wins": f"{int(perception.brier_helped_subjects)}/{int(perception.subject_count)}",
            "mean_crps_benefit": perception.mean_crps_benefit,
            "evidence": "A*",
            "reason": "关闭后 7/8 Brier 更好；作为外部校准 robustness model 保留，而不宣称提升行为拟合。",
        },
        {
            "mechanism": "Label-reversed hypothesis copies",
            "decision": "移出主模型",
            "mean_brier_benefit": labels.mean_brier_benefit,
            "brier_wins": f"{int(labels.brier_helped_subjects)}/{int(labels.subject_count)}",
            "mean_crps_benefit": labels.mean_crps_benefit,
            "evidence": "A*",
            "reason": "排除初始化混淆后，关闭反转副本在 6/7 被试改善 Brier；同时解决论文写 19、代码跑 38 的冲突。",
        },
    ]
    decisions = pd.DataFrame(decision_rows)

    chart_mechanisms = [
        "dynamic hypothesis-specific beta versus static beta=5",
        "flexible readout versus expectation",
        "dynamic active-set selection versus full hypothesis set",
        "controller stable profile",
        "controller stubborn profile",
        "latent volatility state (frozen selected mode)",
        "controller aggressive profile",
        "calibrated perception noise",
        "label-reversed hypothesis copies (unconfounded 7)",
    ]
    chart = effects[effects.mechanism.isin(chart_mechanisms)][
        [
            "mechanism",
            "mean_brier_benefit",
            "brier_ci_low",
            "brier_ci_high",
            "brier_helped_subjects",
            "subject_count",
        ]
    ].copy()
    chart["mechanism_short"] = chart.mechanism.map(
        {
            "dynamic hypothesis-specific beta versus static beta=5": "dynamic beta",
            "flexible readout versus expectation": "readout",
            "dynamic active-set selection versus full hypothesis set": "active set",
            "controller stable profile": "stable profile",
            "controller stubborn profile": "stubborn profile",
            "latent volatility state (frozen selected mode)": "latent state",
            "controller aggressive profile": "aggressive profile",
            "calibrated perception noise": "perception noise",
            "label-reversed hypothesis copies (unconfounded 7)": "label reversals",
        }
    )
    chart = chart.sort_values("mean_brier_benefit", ascending=False)

    active_subjects = subject_effects[
        subject_effects.mechanism.eq(
            "dynamic active-set selection versus full hypothesis set"
        )
    ].copy()
    active_subjects["subject_id"] = active_subjects.subject_id.astype(str)

    headline = pd.DataFrame(
        [
            {
                "beta_brier_benefit": beta.mean_brier_benefit,
                "readout_brier_benefit": readout.mean_brier_benefit,
                "state_brier_benefit": state.mean_brier_benefit,
                "label_brier_benefit": labels.mean_brier_benefit,
            }
        ]
    )

    sources = [
        {
            "id": "paired_rows",
            "label": "Corrected paired architecture ablations",
            "path": "results/model_architecture_effect_audit/paired_ablations/paired_rows.csv",
        },
        {
            "id": "effect_summary",
            "label": "Paired effect estimates and audit limitations",
            "path": "results/model_architecture_effect_audit/effect_audit_summary.json",
        },
        {
            "id": "mechanism_effects",
            "label": "Mechanism-level effect table",
            "path": "results/model_architecture_effect_audit/mechanism_effects.csv",
        },
        {
            "id": "subject_effects",
            "label": "Subject-level paired deltas",
            "path": "results/model_architecture_effect_audit/subject_mechanism_effects.csv",
        },
        {
            "id": "structural_pilot",
            "label": "Pre-fix controller/readout structural pilot",
            "path": "results/cond1_v14/pilot_state_readout/pilot_rows.csv",
        },
        {
            "id": "readout_confirmation",
            "label": "Independent-seed readout confirmation",
            "path": "results/cond1_v14/confirm_gain_readout/pilot_rows.csv",
        },
        {
            "id": "frozen_confirmation",
            "label": "Pre-fix frozen state confirmation",
            "path": "results/cond1_v14/frozen_confirmation/pilot_rows.csv",
        },
        {
            "id": "v14_search",
            "label": "Formal V14 coordinate-descent search configuration",
            "path": "configs/hyper_cd_cfg/pmh_cond1_hyper_cd_v14.yaml",
        },
        {
            "id": "manuscript_model",
            "label": "Current model-method manuscript draft",
            "path": "manuscript/model.tex",
        },
        {
            "id": "headline_sql",
            "label": "Headline paired-effect metrics",
            "path": "results/model_architecture_effect_audit/effect_audit_summary.json",
            "query": {
                "engine": "sqlite",
                "sql": "SELECT * FROM headline",
                "description": "Selects the four headline local paired effects.",
                "executed_at": generated,
                "tables_used": ["headline"],
                "metric_definitions": [
                    "benefit = loss(ablated model) - loss(corrected frozen baseline)"
                ],
            },
        },
        {
            "id": "mechanism_chart_sql",
            "label": "Corrected mechanism-level Brier effects",
            "path": "results/model_architecture_effect_audit/mechanism_effects.csv",
            "query": {
                "engine": "sqlite",
                "sql": "SELECT * FROM mechanism_chart ORDER BY mean_brier_benefit DESC",
                "description": "Selects corrected paired mechanism effects for the overview chart.",
                "executed_at": generated,
                "tables_used": ["mechanism_chart"],
            },
        },
        {
            "id": "active_subjects_sql",
            "label": "Subject-level active-set effects",
            "path": "results/model_architecture_effect_audit/subject_mechanism_effects.csv",
            "query": {
                "engine": "sqlite",
                "sql": "SELECT * FROM active_subjects ORDER BY CAST(subject_id AS INTEGER)",
                "description": "Selects paired full-set versus dynamic-active-set effects by subject.",
                "executed_at": generated,
                "tables_used": ["active_subjects"],
            },
        },
        {
            "id": "decisions_sql",
            "label": "Architecture decisions derived from paired effects",
            "path": "results/model_architecture_effect_audit/architecture_decisions.csv",
            "query": {
                "engine": "sqlite",
                "sql": "SELECT * FROM decisions",
                "description": "Selects the reviewed retain, simplify, demote, and remove decisions.",
                "executed_at": generated,
                "tables_used": ["decisions"],
            },
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Category-learning 模型架构：效果优先的再核验",
            "description": "修正因果有效性问题后，对关键机制进行同被试、同随机数的局部成对消融。",
            "generatedAt": generated,
            "cards": [
                {
                    "id": "headline_effects",
                    "description": "benefit = ablation loss − baseline loss；正值表示保留机制改善拟合。",
                    "dataset": "headline",
                    "sourceId": "headline_sql",
                    "metrics": [
                        {"label": "dynamic beta ΔBrier benefit", "field": "beta_brier_benefit", "format": "number", "signed": True},
                        {"label": "readout ΔBrier benefit", "field": "readout_brier_benefit", "format": "number", "signed": True},
                        {"label": "latent state ΔBrier benefit", "field": "state_brier_benefit", "format": "number", "signed": True},
                        {"label": "label reversal ΔBrier benefit", "field": "label_brier_benefit", "format": "number", "signed": True},
                    ],
                }
            ],
            "charts": [
                {
                    "id": "mechanism_brier",
                    "title": "修正后局部消融：各机制的平均 Brier 贡献",
                    "subtitle": "正值支持保留；负值表示关闭该机制后拟合更好。区间为代表被试 bootstrap，仅描述异质性。",
                    "type": "bar",
                    "dataset": "mechanism_chart",
                    "sourceId": "mechanism_chart_sql",
                    "valueFormat": "number",
                    "layout": "full",
                    "encodings": {
                        "x": {"field": "mechanism_short", "type": "nominal", "label": "机制"},
                        "y": {"field": "mean_brier_benefit", "type": "quantitative", "label": "Brier benefit"},
                        "tooltip": [
                            {"field": "brier_ci_low", "type": "quantitative", "label": "CI low", "format": "number"},
                            {"field": "brier_ci_high", "type": "quantitative", "label": "CI high", "format": "number"},
                            {"field": "brier_helped_subjects", "type": "quantitative", "label": "受益人数"},
                            {"field": "subject_count", "type": "quantitative", "label": "样本人数"},
                        ],
                    },
                    "referenceLines": [{"value": 0, "label": "无局部贡献"}],
                },
                {
                    "id": "active_subjects",
                    "title": "Active set 不是人人适用：被试级 Brier 贡献",
                    "subtitle": "正值表示动态 active set 优于更新完整假设集；三名被试方向相反。",
                    "type": "bar",
                    "dataset": "active_subjects",
                    "sourceId": "active_subjects_sql",
                    "valueFormat": "number",
                    "layout": "full",
                    "encodings": {
                        "x": {"field": "subject_id", "type": "nominal", "label": "被试"},
                        "y": {"field": "brier_benefit", "type": "quantitative", "label": "Brier benefit"},
                        "tooltip": [
                            {"field": "crps_benefit", "type": "quantitative", "label": "CRPS benefit", "format": "number"}
                        ],
                    },
                    "referenceLines": [{"value": 0, "label": "full-set 与 active-set 持平"}],
                },
            ],
            "tables": [
                {
                    "id": "decision_table",
                    "title": "重新裁决：保留、降级或移出主模型",
                    "subtitle": "A* 为修正实现后的成对局部消融；B 为修正前但结构可比的成对证据。",
                    "dataset": "decisions",
                    "sourceId": "decisions_sql",
                    "layout": "full",
                    "columns": [
                        {"field": "mechanism", "label": "机制", "type": "text"},
                        {"field": "decision", "label": "裁决", "type": "text"},
                        {"field": "mean_brier_benefit", "label": "Brier benefit", "format": "number", "movement": True},
                        {"field": "brier_wins", "label": "受益人数", "type": "text"},
                        {"field": "mean_crps_benefit", "label": "CRPS benefit", "format": "number", "movement": True},
                        {"field": "evidence", "label": "证据级", "type": "text"},
                        {"field": "reason", "label": "理由", "type": "text"},
                    ],
                }
            ],
            "sources": sources,
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "# Category-learning 模型架构：效果优先的再核验\n\n"
                        "**修正后的结论：不能按“看起来复杂”来删机制，但也不能把曾经入选搜索当作有效性证据。** "
                        "dynamic beta 与 flexible readout 有大而一致的拟合贡献；active set 与 controller 结构存在很强的被试异质性；"
                        "latent state、aggressive profile、知觉噪声与标签反转不应继续作为通用主模型组件。"
                    ),
                },
                {
                    "id": "scope",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "paired_rows",
                    "body": (
                        "## 本轮到底核验了什么\n\n"
                        "先修复两处会污染 trial-level prediction 的实现：`beta_log[t]` 改为记录反馈前、实际用于第 t 试次预测的 beta；"
                        "双通道记忆每个试次与 transition prior 同步，同时保留每个假设的长短时轨道差值。随后冻结 8 名代表被试的 V14 参数，"
                        "每次只关闭一个机制，并为同一被试的所有变体使用完全相同的 256 组 trajectory seeds。\n\n"
                        "共得到 **108 行配置结果**，8 名被试每人 13–14 个变体；所有主指标完整。"
                        "这里的 benefit 定义为 `loss(ablated) − loss(baseline)`，所以正数表示该机制有拟合贡献。"
                    ),
                },
                {"id": "headline", "type": "metric-strip", "layout": "full", "cardIds": ["headline_effects"]},
                {"id": "mechanism_chart_block", "type": "chart", "layout": "full", "chartId": "mechanism_brier"},
                {"id": "decisions_block", "type": "table", "layout": "full", "tableId": "decision_table"},
                {
                    "id": "keep",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 哪些设计应当留下\n\n"
                        f"**Dynamic beta 是当前最强机制。** 固定为 beta=5 后，平均 Brier 变差 **{beta.mean_brier_benefit:.4f}**、"
                        f"CRPS 变差 **{beta.mean_crps_benefit:.4f}**，8/8 被试方向一致。它应保留，但代码上应变成一个拥有明确 "
                        "`pre-trial beta → likelihood → feedback update` 接口的独立状态模块；正式论文还需与“每名被试优化一个静态 beta”比较。\n\n"
                        f"**Flexible readout 也必须保留。** 强制 expectation 后平均 Brier 变差 **{readout.mean_brier_benefit:.4f}**，"
                        "同样为 8/8。建议把 expectation、sharp-2、sharp-4、MAP 收拢成一个连续 concentration 参数；MAP 可视为极限，而不是四套分叉代码。\n\n"
                        "**Stable profile 保留。** 删除它的平均 Brier 代价约 "
                        f"**{stable.mean_brier_benefit:.4f}**。controller 异质性也不能强制压成单一 stable-dominant；旧 pilot 中统一 controller "
                        "的平均 Brier 代价约 0.015–0.018。"
                    ),
                },
                {
                    "id": "heterogeneity",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "subject_effects",
                    "body": (
                        "## Active set 要保留，但必须允许 full-set\n\n"
                        f"平均上动态 active set 的 Brier benefit 为 **{active.mean_brier_benefit:.4f}**，但区间跨零，且只有 "
                        f"**{int(active.brier_helped_subjects)}/{int(active.subject_count)}** 被试受益。被试 105、117、127 在更新完整假设集时明显更好；"
                        "103、111、112、118、131 则需要容量限制或动态迁移。\n\n"
                        "因此正确的结构不是“六套 controller 全删”或“六套全部原样留下”，而是把 **inference capacity** 明确写成模型选择轴："
                        "`full-set` 与一个精简的 `limited active-set` 分支并列。当前六个家族可暂时作为候选库，但不能在论文中宣称六类心理策略已被证实。"
                    ),
                },
                {"id": "active_chart_block", "type": "chart", "layout": "full", "chartId": "active_subjects"},
                {
                    "id": "remove",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 哪些设计应移出通用主模型\n\n"
                        f"- **标签反转副本：** 排除 subject 103 的初始化混淆后，关闭它平均改善 Brier **{abs(labels.mean_brier_benefit):.4f}**，6/7 被试改善。主模型回到论文所写的 19 个几何假设；反转版本只做 robustness。\n"
                        f"- **知觉噪声：** 关闭后平均改善 Brier **{abs(perception.mean_brier_benefit):.4f}**，7/8 改善。它有独立测量意义，但不应再被描述为提高行为拟合；放到外部校准/敏感性分析。\n"
                        f"- **Aggressive profile：** 平均 contribution 为 **{aggressive.mean_brier_benefit:.4f}**，仅 2/8 受益。移出主 controller；若后续 held-out 证实某亚组需要，再作为扩展。\n"
                        f"- **Persistent state：** 平均 Brier contribution 仅 **{state.mean_brier_benefit:.4f}**，CRPS contribution 为 **{state.mean_crps_benefit:.4f}**。降级为亚组模型，不能作为论文核心创新承担主要拟合提升。\n"
                        f"- **Choice-informed scoring：** 只在 2/4 相关被试改善 Brier，平均 CRPS contribution 为 **{choice.mean_crps_benefit:.4f}**；不进入通用主模型。"
                    ),
                },
                {
                    "id": "memory",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 记忆模块：简化方向是 static，而不是 fade-only\n\n"
                        f"双通道相对 fade-only 的 Brier benefit 为 **{dual_fade.mean_brier_benefit:.4f}**，说明纯衰减轨道不够；"
                        f"但双通道相对 static-only 的 benefit 为 **{dual_static.mean_brier_benefit:.4f}**，并没有显示双通道优于长期累积。\n\n"
                        "更关键的是，当前正式网格的 w0 最大只到 0.50，根本没有让优化器选择 static-only 的 w0=1。"
                        "因此暂时不应把双通道写成已获支持的机制。下一版先加入 w0∈{0,1} 边界并重新优化；如果 static-only 仍持平或更好，主模型直接删去 fade 轨道和 gamma。"
                    ),
                },
                {
                    "id": "manuscript_alignment",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "manuscript_model",
                    "body": (
                        "## 发表前必须统一的三处方法描述\n\n"
                        "1. 当前稿件写 19 个假设，但配置启用了 label reversals，实际是 38 个带标签映射的状态；按本轮结果，建议代码回到 19，而不是把稿件改成 38。\n"
                        "2. 稿件写的是两种 `random_M / opp_random_M` 策略，当前代码实际是 conservative、stable、aggressive、stubborn 四 profile 的 feedback-gated softmax；两者不是同一个模型。\n"
                        "3. 稿件称 active-set 大小、初始数量和 beta 参数等联合优化，但 V14 正式搜索实际只覆盖 25 个 memory 组合、24 个 controller/state 配置与 4 个 readout。Methods 必须按最终冻结的搜索空间重写。\n\n"
                        "这三处在架构冻结前不要靠文字补丁掩盖；先定主模型，再让配置、代码、公式和报告四者一致。"
                    ),
                },
                {
                    "id": "evidence",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 证据边界\n\n"
                        "- **A\\***：修正后的实现、同被试 common random numbers、单机制局部消融；但参数是在旧实现上选择后冻结的，尚未对每个消融模型重新优化。\n"
                        "- **B**：修正前的结构可比成对实验，例如 controller 异质性与 readout 独立种子复核。\n"
                        "- **C**：32 人 V13 入选频次，只说明搜索空间有使用价值，不等于机制有因果贡献。\n\n"
                        "8 名被试不是 held-out 总体样本；bootstrap 区间只描述跨被试分散。独立 Monte Carlo seed 也不等于独立行为数据。"
                        "所以本报告足以决定“哪些结构现在不该删除、哪些结构必须降级”，但正式发表还需在修正后的候选空间上重优化，并做 held-out trial 或 held-out subject 比较。"
                    ),
                },
                {
                    "id": "next",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 建议冻结的下一版候选架构\n\n"
                        "1. **共同核心：** 19 个几何假设、因果 dynamic beta、一个连续 readout concentration、static memory 基线。\n"
                        "2. **容量分支：** full-set 与精简 limited active-set 并列；limited 分支至少保留 stable，并把 conservative/stubborn 尝试合并。\n"
                        "3. **扩展而非核心：** persistent state、aggressive、choice-informed scoring、知觉噪声、label reversal。\n"
                        "4. **公平比较：** static beta 要优化，w0 必须含 0 与 1，所有模型族都要在同样预算下重优化；之后再用 held-out prediction、口头报告对齐和参数恢复共同裁决。\n\n"
                        "这一步完成后，再重写 `manuscript/model.tex`；否则现在改 Methods 很快还会再次失配。"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "headline": records(headline),
                "mechanism_chart": records(chart),
                "active_subjects": records(active_subjects),
                "decisions": records(decisions),
            },
            "accessIssues": [],
        },
        "sources": sources,
        "package_info": {
            "originUrl": "artifact://category-learning-architecture-effect-audit",
            "controls": {"edit": False, "refresh": False},
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(AUDIT_DIR / "architecture_decisions.csv", index=False)
    output = REPORT_DIR / "artifact.json"
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
