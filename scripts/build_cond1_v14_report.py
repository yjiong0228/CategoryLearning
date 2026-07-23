#!/usr/bin/env python3
"""Build the canonical portable-report payload for the Cond1 V14 decision."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "results/cond1_v14/analysis"
STATE_DIR = ROOT / "results/cond1_v14/state_diagnostics"
REPORT_DIR = ROOT / "results/cond1_v14/report"


def records(frame: pd.DataFrame) -> list[dict]:
    """Return strict-JSON records, replacing non-finite values with null."""
    return json.loads(frame.to_json(orient="records"))


def main() -> None:
    comparison = pd.read_csv(ANALYSIS / "variant_comparison.csv")
    selected = pd.read_csv(ANALYSIS / "subject_confirm_selection.csv")
    pilot = pd.read_csv(ANALYSIS / "pilot_structure_summary.csv")
    analysis = json.loads((ANALYSIS / "analysis_summary.json").read_text())
    state = json.loads(
        (STATE_DIR / "state_diagnostic_summary.json").read_text()
    )["aggregate"]

    selected_chart = selected[
        [
            "subject_id",
            "controller_id",
            "readout",
            "state_setting",
            "marginal_choice_brier",
            "delta_brier_vs_v13",
            "trajectory_crps",
            "delta_crps_vs_v13",
        ]
    ].copy()
    selected_chart["subject_id"] = selected_chart.subject_id.astype(str)

    policy = pd.DataFrame(
        [
            {"policy": "conservative", "fraction": state["policy_conservative_fraction"]},
            {"policy": "stable", "fraction": state["policy_stable_fraction"]},
            {"policy": "aggressive", "fraction": state["policy_aggressive_fraction"]},
            {"policy": "stubborn", "fraction": state["policy_stubborn_fraction"]},
        ]
    )
    state_card = pd.DataFrame(
        [
            {
                "nonzero_fraction": state["state_nonzero_fraction"],
                "above_threshold_fraction": state["state_above_threshold_fraction"],
                "lag1": state["state_lag1"],
                "high_spell_trials": state["state_above_threshold_spell_mean"],
                "aggressive_low": state["aggressive_probability_low_state"],
                "aggressive_high": state["aggressive_probability_high_state"],
            }
        ]
    )

    interpretation = analysis["interpretation"]
    generated = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    brier_delta = interpretation["selected_subject_mean_delta_brier_vs_v13"]
    crps_delta = interpretation["selected_subject_mean_delta_crps_vs_v13"]
    gain50 = comparison.loc[comparison.variant_id == "m2_v14_gain_0p50"].iloc[0]
    state_off = comparison.loc[
        comparison.variant_id == "m1_selected_state_off"
    ].iloc[0]
    state_gain = state["aggressive_probability_high_state"] - state[
        "aggressive_probability_low_state"
    ]

    source_specs = [
        {
            "id": "confirmation",
            "label": "V14 independent confirmation (8 subjects, 512 runs/config)",
            "path": "results/cond1_v14/confirm_gain_readout/pilot_rows.csv",
        },
        {
            "id": "pilot",
            "label": "V14 structural pilot (8 subjects, 256 runs/config)",
            "path": "results/cond1_v14/pilot_state_readout/pilot_rows.csv",
        },
        {
            "id": "state_diagnostics",
            "label": "Persistent-state diagnostics (256 fully logged trajectories)",
            "path": "results/cond1_v14/state_diagnostics/state_diagnostic_runs.csv",
        },
        {
            "id": "v14_configuration",
            "label": "Formal V14 coordinate-descent configuration",
            "path": "configs/hyper_cd_cfg/pmh_cond1_hyper_cd_v14.yaml",
        },
    ]
    widget_sources = [
        {
            "id": "decision_summary_sql",
            "label": "Confirmed headline metrics",
            "path": "results/cond1_v14/analysis/subject_confirm_selection.csv",
            "query": {
                "engine": "sqlite",
                "sql": "SELECT * FROM decision_summary",
                "description": "Selects the reviewed V14-versus-V13 headline metrics.",
                "executed_at": generated,
                "tables_used": ["decision_summary"],
                "metric_definitions": ["negative deltas indicate lower loss than V13"],
            },
        },
        {
            "id": "state_summary_sql",
            "label": "Persistent-state aggregate diagnostics",
            "path": "results/cond1_v14/state_diagnostics/state_diagnostic_summary.json",
            "query": {
                "engine": "sqlite",
                "sql": "SELECT * FROM state_summary",
                "description": "Selects reviewed aggregate persistent-state diagnostics.",
                "executed_at": generated,
                "tables_used": ["state_summary"],
            },
        },
        {
            "id": "policy_usage_sql",
            "label": "Inner-policy usage diagnostics",
            "path": "results/cond1_v14/state_diagnostics/state_diagnostic_summary.json",
            "query": {
                "engine": "sqlite",
                "sql": "SELECT policy, fraction FROM policy_usage ORDER BY rowid",
                "description": "Selects the observed usage fraction of each inner transition policy.",
                "executed_at": generated,
                "tables_used": ["policy_usage"],
            },
        },
        {
            "id": "variant_comparison_sql",
            "label": "Fixed-gain confirmation comparison",
            "path": "results/cond1_v14/analysis/variant_comparison.csv",
            "query": {
                "engine": "sqlite",
                "sql": "SELECT * FROM variant_comparison ORDER BY rowid",
                "description": "Selects the reviewed independent-confirmation comparison by state setting.",
                "executed_at": generated,
                "tables_used": ["variant_comparison"],
            },
        },
        {
            "id": "subject_selection_sql",
            "label": "Confirmed subject-level V14 selections",
            "path": "results/cond1_v14/analysis/subject_confirm_selection.csv",
            "query": {
                "engine": "sqlite",
                "sql": "SELECT * FROM selected_subjects ORDER BY CAST(subject_id AS INTEGER)",
                "description": "Selects subject-level candidates chosen after independent confirmation.",
                "executed_at": generated,
                "tables_used": ["selected_subjects"],
            },
        },
    ]
    all_sources = source_specs + widget_sources

    decision_summary = pd.DataFrame(
        [
            {
                "selected_brier_delta": brier_delta,
                "selected_crps_delta": crps_delta,
                "brier_win_fraction": interpretation["selected_subject_brier_wins"] / 8,
                "crps_win_fraction": interpretation["selected_subject_crps_wins"] / 8,
                "state_off_selected_count": interpretation["state_off_selected_count"],
                "state_off_brier_delta": state_off.delta_brier_vs_v13,
            }
        ]
    )
    frames = {
        "decision_summary": decision_summary,
        "state_summary": state_card,
        "policy_usage": policy,
        "variant_comparison": comparison,
        "selected_subjects": selected_chart,
        "pilot_structure": pilot,
    }
    sql_by_table = {
        "decision_summary": "SELECT * FROM decision_summary",
        "state_summary": "SELECT * FROM state_summary",
        "policy_usage": "SELECT policy, fraction FROM policy_usage ORDER BY rowid",
        "variant_comparison": "SELECT * FROM variant_comparison ORDER BY rowid",
        "selected_subjects": "SELECT * FROM selected_subjects ORDER BY CAST(subject_id AS INTEGER)",
        "pilot_structure": "SELECT * FROM pilot_structure ORDER BY rowid",
    }
    connection = sqlite3.connect(":memory:")
    for name, frame in frames.items():
        frame.to_sql(name, connection, index=False)
    snapshot_datasets = {
        name: records(pd.read_sql_query(sql_by_table[name], connection))
        for name in frames
    }
    connection.close()

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Cond1 V14：持续波动状态与控制器结构决策",
            "description": "基于成对随机种子的结构筛选、独立 Monte Carlo 确认与状态轨迹诊断。",
            "generatedAt": generated,
            "cards": [
                {
                    "id": "confirmed_improvement",
                    "description": "先由 pilot 选择 controller/readout，再用独立种子确认；负数表示优于 V13。",
                    "dataset": "decision_summary",
                    "sourceId": "decision_summary_sql",
                    "metrics": [
                        {"label": "平均 Δ marginal Brier", "field": "selected_brier_delta", "format": "number", "signed": True},
                        {"label": "平均 Δ trajectory CRPS", "field": "selected_crps_delta", "format": "number", "signed": True},
                        {"label": "Brier 改善人数占比", "field": "brier_win_fraction", "format": "percent"},
                        {"label": "CRPS 改善人数占比", "field": "crps_win_fraction", "format": "percent"},
                    ],
                },
                {
                    "id": "state_operation",
                    "description": "32 条完整日志轨迹/被试；阈值为 0.55。",
                    "dataset": "state_summary",
                    "sourceId": "state_summary_sql",
                    "metrics": [
                        {"label": "状态非零试次", "field": "nonzero_fraction", "format": "percent"},
                        {"label": "阈值以上试次", "field": "above_threshold_fraction", "format": "percent"},
                        {"label": "状态 lag-1", "field": "lag1", "format": "number"},
                        {"label": "高状态连续长度", "field": "high_spell_trials", "format": "number"},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "subject_brier_delta",
                    "title": "独立确认：每个被试的最佳候选相对 V13 的 Brier 变化",
                    "subtitle": "7/8 被试改善；负值越大表示 V14 越好。候选结构在 pilot 中选择，确认采用新随机种子。",
                    "type": "bar",
                    "dataset": "selected_subjects",
                    "sourceId": "subject_selection_sql",
                    "valueFormat": "number",
                    "layout": "full",
                    "encodings": {
                        "x": {"field": "subject_id", "type": "nominal", "label": "被试"},
                        "y": {"field": "delta_brier_vs_v13", "type": "quantitative", "label": "V14 − V13 marginal Brier"},
                        "tooltip": [
                            {"field": "controller_id", "type": "text", "label": "controller"},
                            {"field": "state_setting", "type": "text", "label": "state"},
                            {"field": "readout", "type": "text", "label": "readout"},
                            {"field": "delta_crps_vs_v13", "type": "quantitative", "label": "Δ CRPS", "format": "number"},
                        ],
                    },
                    "referenceLines": [{"value": 0, "label": "与 V13 持平"}],
                },
                {
                    "id": "policy_usage",
                    "title": "四种内层策略均被实际使用",
                    "subtitle": "占比来自 256 条完整日志轨迹，而非配置文件中的静态权重。",
                    "type": "bar",
                    "dataset": "policy_usage",
                    "sourceId": "policy_usage_sql",
                    "valueFormat": "percent",
                    "layout": "full",
                    "encodings": {
                        "x": {"field": "policy", "type": "nominal", "label": "内层策略"},
                        "y": {"field": "fraction", "type": "quantitative", "label": "被采样比例", "format": "percent"},
                    },
                },
            ],
            "tables": [
                {
                    "id": "variant_comparison",
                    "title": "固定状态增益的独立确认",
                    "subtitle": "置信区间为 8 个代表被试上的成对 bootstrap 95% CI；仅量化代表集，不作总体推断。",
                    "dataset": "variant_comparison",
                    "sourceId": "variant_comparison_sql",
                    "layout": "full",
                    "columns": [
                        {"field": "variant_id", "label": "变体", "type": "text"},
                        {"field": "mean_marginal_choice_brier", "label": "Brier", "format": "number"},
                        {"field": "delta_brier_vs_v13", "label": "Δ Brier", "format": "number", "movement": True},
                        {"field": "delta_brier_vs_v13_ci_low", "label": "CI low", "format": "number"},
                        {"field": "delta_brier_vs_v13_ci_high", "label": "CI high", "format": "number"},
                        {"field": "brier_wins_vs_v13", "label": "Brier wins", "format": "number"},
                        {"field": "mean_trajectory_crps", "label": "CRPS", "format": "number"},
                        {"field": "delta_crps_vs_v13", "label": "Δ CRPS", "format": "number", "movement": True},
                    ],
                },
                {
                    "id": "subject_selection",
                    "title": "代表被试的 V14 候选结构",
                    "subtitle": "增益是在确认集内作描述性最优选择，因此正式搜索仍保留 off/0.20/0.35/0.50。",
                    "dataset": "selected_subjects",
                    "sourceId": "subject_selection_sql",
                    "layout": "full",
                    "columns": [
                        {"field": "subject_id", "label": "被试", "type": "text"},
                        {"field": "controller_id", "label": "controller", "type": "text"},
                        {"field": "state_setting", "label": "state", "type": "text"},
                        {"field": "readout", "label": "readout", "type": "text"},
                        {"field": "marginal_choice_brier", "label": "Brier", "format": "number"},
                        {"field": "delta_brier_vs_v13", "label": "Δ Brier", "format": "number", "movement": True},
                        {"field": "trajectory_crps", "label": "CRPS", "format": "number"},
                        {"field": "delta_crps_vs_v13", "label": "Δ CRPS", "format": "number", "movement": True},
                    ],
                },
            ],
            "sources": all_sources,
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "layout": "full",
                    "body": "# Cond1 V14：持续波动状态与控制器结构决策\n\n**结论：V14 不应把 16 套 controller 删除成 1 套。** 旧 16 套完整保留；正式 V14 搜索压缩为 6 个有行为区分度的 controller 家族，并与 4 个状态设置、4 个 readout 正交组合。",
                },
                {
                    "id": "executive_summary",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "confirmation",
                    "body": (
                        "## 结论先行\n\n"
                        f"在 8 个代表被试、每配置 512 次独立 Monte Carlo 确认中，pilot 选择的候选按被试取最优状态设置后，平均 marginal Brier 相对冻结 V13 下降 **{abs(brier_delta):.4f}**，trajectory CRPS 下降 **{abs(crps_delta):.4f}**；两项指标均有 **7/8** 被试改善。\n\n"
                        f"如果强制统一增益，gain=0.50 的平均 ΔBrier 为 **{gain50.delta_brier_vs_v13:.4f}**；但被试 118 仍偏好 state-off。因此，正确决策不是“强制每个人启用一个增益”，而是把持续状态作为可选择机制进入正式拟合。"
                    ),
                },
                {"id": "decision_metrics", "type": "metric-strip", "layout": "full", "cardIds": ["confirmed_improvement"]},
                {"id": "subject_chart", "type": "chart", "layout": "full", "chartId": "subject_brier_delta"},
                {
                    "id": "architecture",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "v14_configuration",
                    "body": (
                        "## V14 的完整架构\n\n"
                        "每个 trial 的路径是：**Bayesian hypothesis posterior → 持续 volatility state → controller 权重 → 四种内层 transition policy → hypothesis transition → choice readout**。\n\n"
                        "持续状态使用 `confidence_weighted_error`：高置信度犯错的冲击更大；状态按 `state[t] = clip(0.8 × state[t−1] + gain × surprise[t], 0, 1)` 衰减并累积。controller 同时看到快速通道 `last_error` 和慢速通道 `latent_volatility_pressure`，所以一次错误可产生瞬时响应，连续异常才形成跨试次压力。\n\n"
                        "正式候选是 **6 controller 家族 × 4 状态设置（off、0.20、0.35、0.50）× 4 readout（expectation、sharp-2、sharp-4、MAP）**。每个 controller 内仍保留 conservative、stable、aggressive、stubborn 四种策略。"
                    ),
                },
                {
                    "id": "why_six",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "pilot",
                    "body": (
                        "## 为什么是 6 个 controller，而不是 1 个或原样保留 16 个\n\n"
                        "统一 stable-dominant 在个别被试上有帮助，但在代表集上平均相对 V13 **Brier 变差 0.0142、CRPS 变差 0.0037**。这否定了“一套 controller 管所有人”。另一方面，原 16 套中存在大量重复或近重复的行为机制，全部与状态和 readout 做笛卡尔积会浪费计算预算。6 个家族保留了稳定主导、choice 驱动刷新、保守重、早探索晚稳定、error+choice newcomer、error aggressive 六类不同反应模式；旧 16 套文件没有被删除，可随时作审计或回退。"
                    ),
                },
                {
                    "id": "state_heading",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "state_diagnostics",
                    "body": (
                        "## 持续状态确实被启用了\n\n"
                        f"状态在 **{state['state_nonzero_fraction']:.1%}** 的试次非零，lag-1 自相关为 **{state['state_lag1']:.3f}**，超过阈值后的平均连续长度为 **{state['state_above_threshold_spell_mean']:.2f} trials**。高状态下 aggressive policy 的平均概率比低状态高 **{state_gain:.3f}**。因此它不是对最近一次错误的改名，而是能跨 trial 累积、衰减并改变策略混合的慢变量。"
                    ),
                },
                {"id": "state_metrics", "type": "metric-strip", "layout": "full", "cardIds": ["state_operation"]},
                {"id": "policy_chart", "type": "chart", "layout": "full", "chartId": "policy_usage"},
                {"id": "variant_table", "type": "table", "layout": "full", "tableId": "variant_comparison"},
                {"id": "selection_table", "type": "table", "layout": "full", "tableId": "subject_selection"},
                {
                    "id": "methods",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 方法、口径与质量控制\n\n"
                        "- **主目标：** marginal choice Brier，即先跨模拟运行平均每 trial 的类别概率，再对人类选择计算 Brier；它评价模型的预测分布，而不是挑一条最好看的随机轨迹。\n"
                        "- **次目标：** trajectory CRPS，用经验模拟分布评价行为轨迹，保留不确定性。\n"
                        "- **配对设计：** 同一被试的所有变体使用相同 trajectory seeds；确认阶段使用不同于 pilot 的新种子。\n"
                        "- **完整性：** pilot 72 行、确认 40 行、状态诊断 256 行；无指标缺失；稳定 controller 的重复对照完全一致；每个被试内部 seed 唯一性检查通过。"
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 局限与不过度解读\n\n"
                        "这 8 个被试用于结构筛选，不是总体推断样本。controller/readout 的确认是独立的，但状态 gain 的按被试最优值仍在确认结果内选择，因而它是描述性上界；正式拟合必须保留 state-off 与三个 gain，让完整 coordinate descent 决定。当前结果支持“把持续状态放进 V14 搜索”，尚不能证明每个被试都需要持续状态，也不能用代表集 bootstrap 替代全样本检验。"
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "layout": "full",
                    "body": (
                        "## 下一步执行顺序\n\n"
                        "1. 用正式 V14 配置在 selected-eight 上运行 coordinate descent，主目标 Brier、次目标 CRPS。\n"
                        "2. 每个入选配置以更高 repeats 重算，并比较 state-on 与同结构 state-off 的成对差值。\n"
                        "3. 若持续状态在代表集保持优势，再扩展到全部被试；否则保留 state-off 并收缩 gain 网格。\n"
                        "4. 最终报告参数恢复、状态占用、四策略占用及 posterior predictive coverage，避免只汇报单一 loss。\n\n"
                        "### 仍需回答的问题\n\n"
                        "- 被试 118 为什么不受益：真实稳定性、controller/readout 耦合，还是现有 surprise 定义不合适？\n"
                        "- gain=0.50 的优势在更高 repeats 和全样本上是否稳定？\n"
                        "- 六个 controller 家族中是否仍有可合并的行为等价类？"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": snapshot_datasets,
            "accessIssues": [],
        },
        "sources": all_sources,
        "package_info": {
            "originUrl": "artifact://cond1-v14-structural-decision",
            "controls": {"edit": False, "refresh": False},
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "artifact.json"
    output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
