#!/usr/bin/env python3
"""Build the canonical portable report artifact for the 0805 accuracy audit."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/model_0805_accuracy_diagnostic_20260806"
TITLE = "FS_H0 的 accuracy 失败对模型设计的启示"
MODEL_ORDER = [
    "FS_H0",
    "FA2_M3",
    "FA2_M5",
    "FA2_M7",
    "FA2R_M3",
    "FA2R_M5",
    "FA2R_M7",
]
MODEL_LABELS = {
    "FS_H0": "FS_H0",
    "FA2_M3": "FA2 M=3",
    "FA2_M5": "FA2 M=5",
    "FA2_M7": "FA2 M=7",
    "FA2R_M3": "FA2-R M=3",
    "FA2R_M5": "FA2-R M=5",
    "FA2R_M7": "FA2-R M=7",
}
PHASE_LABELS = {
    "inner_fit": "训练段",
    "inner_validation": "内部验证段",
    "outer_holdout": "最终留出段",
}


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def metric_row(
    summary: pd.DataFrame, model_key: str, phase: str, window: int = 16
) -> pd.Series:
    selected = summary[
        summary["model_key"].eq(model_key)
        & summary["phase"].eq(phase)
        & summary["rolling_window"].eq(window)
    ]
    if selected.shape[0] != 1:
        raise ValueError(f"Expected one summary row for {model_key}/{phase}/{window}")
    return selected.iloc[0]


def build_artifact() -> dict[str, Any]:
    summary = pd.read_csv(REPORT_DIR / "group_curve_summary.csv")
    plateau = pd.read_csv(REPORT_DIR / "FS_H0_plateau_diagnostic.csv")
    parameters = pd.read_csv(REPORT_DIR / "FS_H0_parameter_posterior.csv")
    constants = pd.read_csv(REPORT_DIR / "outer_constant_baseline_diagnostic.csv")

    comparison_rows: list[dict[str, Any]] = []
    onset_rows: list[dict[str, Any]] = [
        {
            "model": "被试实际",
            "model_key": "Observed",
            "median_onset_trial": float(
                metric_row(summary, "FS_H0", "all")["median_actual_onset_75"]
            ),
            "kind": "实际",
        }
    ]
    for model_key in MODEL_ORDER:
        all_row = metric_row(summary, model_key, "all")
        outer_row = metric_row(summary, model_key, "outer_holdout")
        comparison_rows.append(
            {
                "model": MODEL_LABELS[model_key],
                "model_key": model_key,
                "all_curve_mae_pp": 100 * float(all_row["mean_curve_mae"]),
                "all_centered_curve_mae_pp": 100
                * float(all_row["mean_curve_centered_mae"]),
                "all_curve_correlation": float(all_row["mean_curve_correlation"]),
                "all_predicted_curve_sd_pp": 100
                * float(all_row["mean_predicted_curve_sd"]),
                "actual_curve_sd_pp": 100
                * float(all_row["mean_actual_curve_sd"]),
                "all_nll_per_trial": float(all_row["mean_nll_per_trial"]),
                "outer_curve_mae_pp": 100 * float(outer_row["mean_curve_mae"]),
                "outer_centered_curve_mae_pp": 100
                * float(outer_row["mean_curve_centered_mae"]),
                "outer_curve_correlation": float(
                    outer_row["mean_curve_correlation"]
                ),
                "outer_nll_per_trial": float(outer_row["mean_nll_per_trial"]),
                "median_onset_trial": float(all_row["median_predicted_onset_75"]),
            }
        )
        onset_rows.append(
            {
                "model": MODEL_LABELS[model_key],
                "model_key": model_key,
                "median_onset_trial": float(
                    all_row["median_predicted_onset_75"]
                ),
                "kind": "模型",
            }
        )

    phase_rows: list[dict[str, Any]] = []
    for phase in ("inner_fit", "inner_validation", "outer_holdout"):
        for model_key in ("FS_H0", "FA2_M3"):
            current = metric_row(summary, model_key, phase)
            phase_rows.append(
                {
                    "phase": PHASE_LABELS[phase],
                    "phase_key": phase,
                    "model": MODEL_LABELS[model_key],
                    "model_key": model_key,
                    "nll_per_trial": float(current["mean_nll_per_trial"]),
                    "curve_mae_pp": 100 * float(current["mean_curve_mae"]),
                    "curve_correlation": float(
                        current["mean_curve_correlation"]
                    ),
                }
            )

    sensitivity_rows: list[dict[str, Any]] = []
    for window in (8, 16, 32):
        for model_key in ("FS_H0", "FA2_M3"):
            current = metric_row(summary, model_key, "all", window)
            sensitivity_rows.append(
                {
                    "rolling_window": int(window),
                    "model": MODEL_LABELS[model_key],
                    "curve_mae_pp": 100 * float(current["mean_curve_mae"]),
                }
            )

    fs_all = metric_row(summary, "FS_H0", "all")
    fa2m3_all = metric_row(summary, "FA2_M3", "all")
    fs_outer = metric_row(summary, "FS_H0", "outer_holdout")
    fa2m3_outer = metric_row(summary, "FA2_M3", "outer_holdout")
    final_parameters = parameters[parameters["phase_endpoint"].eq("outer_holdout")]
    headline = [
        {
            "fs_full_curve_mae": float(fs_all["mean_curve_mae"]),
            "best_finite_full_curve_mae": float(fa2m3_all["mean_curve_mae"]),
            "subjects_onset_before_outer": int(
                plateau["actual_onset_before_outer"].sum()
            ),
            "subject_count": int(plateau.shape[0]),
            "subjects_map_lapse_0p20": int(
                final_parameters["map_lapse"].eq(0.20).sum()
            ),
            "median_lapse_0p20_mass": float(
                final_parameters["posterior_mass_lapse_0p20"].median()
            ),
            "fs_outer_nll": float(constants["FS_outer_nll_per_trial"].mean()),
            "oracle_constant_outer_nll": float(
                constants["oracle_outer_rate_constant_nll"].mean()
            ),
            "outer_nll_gap_to_oracle": float(
                constants["FS_outer_nll_per_trial"].mean()
                - constants["oracle_outer_rate_constant_nll"].mean()
            ),
            "mean_outer_accuracy": float(constants["outer_accuracy"].mean()),
        }
    ]

    diagnostic_source = {
        "id": "accuracy_diagnostic",
        "label": "model_0805 accuracy 轨迹诊断汇总",
        "path": "reports/model_0805_accuracy_diagnostic_20260806/diagnostic_summary.json",
    }

    def csv_source(source_id: str, label: str, filename: str) -> dict[str, Any]:
        path = f"reports/model_0805_accuracy_diagnostic_20260806/{filename}"
        return {
            "id": source_id,
            "label": label,
            "path": path,
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": f"SELECT * FROM read_csv_auto('{path}')",
                "description": f"读取诊断脚本生成并复核的 {filename}。",
                "tables_used": [path],
                "filters": ["condition = 1", "equal weight per subject"],
                "executed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            },
        }

    group_source = csv_source(
        "group_curve_summary",
        "逐模型 accuracy 曲线汇总",
        "group_curve_summary.csv",
    )
    plateau_source = csv_source(
        "fs_plateau_summary",
        "FS_H0 平台与学习起点诊断",
        "FS_H0_plateau_diagnostic.csv",
    )
    parameter_source = csv_source(
        "fs_parameter_summary",
        "FS_H0 参数后验诊断",
        "FS_H0_parameter_posterior.csv",
    )
    constant_source = csv_source(
        "outer_constant_summary",
        "最终留出常数概率基线诊断",
        "outer_constant_baseline_diagnostic.csv",
    )
    all_sources = [
        diagnostic_source,
        group_source,
        plateau_source,
        parameter_source,
        constant_source,
    ]

    charts = [
        {
            "id": "full_curve_mae",
            "title": "全序列 accuracy 曲线误差",
            "subtitle": "32 名被试等权平均；16-trial 尾随窗口；越低越好",
            "type": "bar",
            "dataset": "model_comparison",
            "sourceId": "group_curve_summary",
            "intent": "comparison",
            "question": "哪个模型最能重现每名被试随试次变化的正确率轨迹？",
            "rationale": "七个离散模型的一项主误差指标适合直接柱状比较。",
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "模型"},
                "y": {
                    "field": "all_curve_mae_pp",
                    "type": "quantitative",
                    "label": "曲线 MAE",
                    "unit": "百分点",
                },
                "tooltip": [
                    {
                        "field": "all_curve_correlation",
                        "type": "quantitative",
                        "label": "曲线相关",
                    },
                    {
                        "field": "all_nll_per_trial",
                        "type": "quantitative",
                        "label": "全序列 NLL/试次",
                    },
                ],
            },
            "yAxisTitle": "平均绝对误差（百分点）",
            "valueFormat": "number",
            "unit": "百分点",
            "palette": {"kind": "sequential", "name": "blue"},
            "settings": {"showValues": True, "sort": "descending"},
            "layout": "full",
        },
        {
            "id": "onset_trial",
            "title": "稳定达到 75% 正确率的中位试次",
            "subtitle": "16-trial 曲线连续八点达到阈值；被试实际与七个模型",
            "type": "bar",
            "dataset": "onset_comparison",
            "sourceId": "group_curve_summary",
            "intent": "comparison",
            "question": "各模型把稳定学习发生的时间预测在何处？",
            "rationale": "学习起点是离散类别比较，柱状图比短折线更诚实。",
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "实际/模型"},
                "y": {
                    "field": "median_onset_trial",
                    "type": "quantitative",
                    "label": "中位起点",
                    "unit": "试次",
                },
                "color": {"field": "kind", "type": "nominal", "label": "类型"},
            },
            "yAxisTitle": "试次",
            "valueFormat": "number",
            "unit": "试次",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "settings": {"showValues": True, "sort": "none"},
            "legend": {"position": "bottom", "sort": "spec"},
            "layout": "full",
        },
        {
            "id": "phase_nll",
            "title": "FS_H0 与 FA2 M=3 的分阶段 NLL",
            "subtitle": "训练、内部验证与最终留出分别计算；越低越好",
            "type": "bar",
            "dataset": "phase_comparison",
            "sourceId": "group_curve_summary",
            "intent": "comparison",
            "question": "FS_H0 的 NLL 优势在哪个阶段出现？",
            "rationale": "三个离散阶段使用分组柱状图，避免把阶段误读为连续时间趋势。",
            "encodings": {
                "x": {"field": "phase", "type": "ordinal", "label": "阶段"},
                "y": {
                    "field": "nll_per_trial",
                    "type": "quantitative",
                    "label": "NLL/试次",
                },
                "color": {"field": "model", "type": "nominal", "label": "模型"},
                "tooltip": [
                    {
                        "field": "curve_mae_pp",
                        "type": "quantitative",
                        "label": "曲线 MAE",
                        "unit": "百分点",
                    },
                    {
                        "field": "curve_correlation",
                        "type": "quantitative",
                        "label": "曲线相关",
                    },
                ],
            },
            "yAxisTitle": "NLL/试次",
            "valueFormat": "number",
            "palette": {"kind": "categorical", "name": "neutral-blue"},
            "settings": {"groupMode": "grouped", "showValues": True, "sort": "none"},
            "legend": {"position": "bottom", "sort": "spec"},
            "layout": "full",
        },
    ]

    table = {
        "id": "model_detail",
        "title": "七个模型的 accuracy 轨迹与 NLL 对照",
        "subtitle": "全序列用于过程诊断；最终留出用于一步预测比较；32 名被试等权",
        "dataset": "model_comparison",
        "sourceId": "group_curve_summary",
        "defaultSort": {"field": "all_curve_mae_pp", "direction": "asc"},
        "density": "compact",
        "layout": "full",
        "columns": [
            {"field": "model", "label": "模型", "type": "text"},
            {
                "field": "all_curve_mae_pp",
                "label": "全序列曲线 MAE",
                "format": "number",
                "unit": "pp",
            },
            {
                "field": "all_curve_correlation",
                "label": "全序列曲线相关",
                "format": "number",
            },
            {
                "field": "median_onset_trial",
                "label": "75% 起点",
                "format": "number",
                "unit": "trial",
            },
            {
                "field": "all_nll_per_trial",
                "label": "全序列 NLL/试次",
                "format": "number",
            },
            {
                "field": "outer_curve_mae_pp",
                "label": "最终留出曲线 MAE",
                "format": "number",
                "unit": "pp",
            },
            {
                "field": "outer_centered_curve_mae_pp",
                "label": "留出中心化 MAE",
                "format": "number",
                "unit": "pp",
            },
            {
                "field": "outer_curve_correlation",
                "label": "留出曲线相关",
                "format": "number",
            },
            {
                "field": "outer_nll_per_trial",
                "label": "最终留出 NLL/试次",
                "format": "number",
            },
        ],
    }

    cards = [
        {
            "id": "full_curve_error",
            "description": "FS_H0 与全序列表现最好的有限模型的 16-trial accuracy 曲线 MAE。",
            "dataset": "headline",
            "sourceId": "group_curve_summary",
            "metrics": [
                {"label": "FS_H0 全序列曲线 MAE", "field": "fs_full_curve_mae", "format": "percent"},
                {"label": "FA2 M=3", "field": "best_finite_full_curve_mae", "format": "percent"},
            ],
        },
        {
            "id": "onset_outside_holdout",
            "description": "实际稳定达到 75% 的时点早于最终留出起点的被试数。",
            "dataset": "headline",
            "sourceId": "fs_plateau_summary",
            "metrics": [
                {"label": "学习转折已发生", "field": "subjects_onset_before_outer", "format": "number"},
                {"label": "总被试", "field": "subject_count", "format": "number"},
            ],
        },
        {
            "id": "lapse_boundary",
            "description": "最终参数权重中，MAP lapse 落在 0.20 网格上界的被试数。",
            "dataset": "headline",
            "sourceId": "fs_parameter_summary",
            "metrics": [
                {"label": "MAP lapse = 0.20", "field": "subjects_map_lapse_0p20", "format": "number"},
                {"label": "总被试", "field": "subject_count", "format": "number"},
            ],
        },
        {
            "id": "outer_nll_oracle",
            "description": "FS_H0 最终留出 NLL 与事后按每名被试最终正确率设定的常数概率下界。",
            "dataset": "headline",
            "sourceId": "outer_constant_summary",
            "metrics": [
                {"label": "FS_H0 留出 NLL/试次", "field": "fs_outer_nll", "format": "number"},
                {"label": "事后常数概率下界", "field": "oracle_constant_outer_nll", "format": "number"},
            ],
        },
    ]

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "sourceId": "accuracy_diagnostic",
            "body": (
                "## 技术结论：FS_H0 是强预测基线，但不是合格的学习过程模型\n\n"
                "FS_H0 的问题不是完全不会预测 choice，而是把学习过程压缩成了“很早确定正确规则 + 一个几乎固定的高 lapse”。"
                "全序列上，它的 16-trial accuracy 曲线 MAE 为 **15.45 个百分点**，而 FA2 M=3 为 **8.72 个百分点**；"
                "FS_H0 的预测曲线相关仅 **0.25**，FA2/FA2-R 为 **0.74–0.82**。\n\n"
                "最终留出 NLL 仍由 FS_H0 获胜，主要因为 **31/32** 名被试的学习转折早已发生在最终留出之前，"
                "该段平均正确率已经达到 **93.6%**。因此，最终后缀更像在检验“能否预测稳定高正确率”，而不是检验“如何学会”。"
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "full_curve_error",
                "onset_outside_holdout",
                "lapse_boundary",
                "outer_nll_oracle",
            ],
        },
        {
            "id": "whole_sequence_finding",
            "type": "markdown",
            "sourceId": "accuracy_diagnostic",
            "body": (
                "## FS_H0 没有重现学习轨迹，有限集模型保留了明显更多动态\n\n"
                "在 32 名被试中，FA2 M=3 有 **29 名**的全序列曲线 MAE 低于 FS_H0；"
                "去掉各段平均水平、只比较曲线形状后，则有 **31 名**更好。"
                "这说明差异不是少数异常被试造成，也不只是一个整体高估或低估。\n\n"
                "图中的 MAE 以百分点计。它是描述性过程诊断，不替代逐试次 NLL，但直接回答模型是否重现了被试随试次变化的正确率。"
            ),
        },
        {"id": "whole_sequence_chart", "type": "chart", "chartId": "full_curve_mae", "layout": "full"},
        {
            "id": "onset_finding",
            "type": "markdown",
            "sourceId": "accuracy_diagnostic",
            "body": (
                "## FS_H0 学得过早，当前有限集模型又普遍学得偏晚\n\n"
                "被试实际稳定达到 75% 的中位时点是第 **48.5** 次；FS_H0 预测为第 **12** 次，提前约 **36** 次。"
                "FA2 M=5 最接近，但仍在第 **73.5** 次；FA2 M=3 是第 **83.5** 次。\n\n"
                "因此证据并不是“当前 FA2 已经正确”。更准确的结论是：FS 缺少延迟搜索/承诺过程，而当前 FA2 的搜索与进入正确规则又过慢、过平滑。下一版应修转折机制，而不是在 FS 与当前 FA2 之间二选一。"
            ),
        },
        {"id": "onset_chart", "type": "chart", "chartId": "onset_trial", "layout": "full"},
        {
            "id": "outer_holdout_finding",
            "type": "markdown",
            "sourceId": "accuracy_diagnostic",
            "body": (
                "## 最终留出主要奖励稳定的高正确率，而非学习机制\n\n"
                "FS_H0 在训练段和内部验证段的 NLL 都差于 FA2 M=3，但在最终留出段反转为 **0.230 vs 0.257 NLL/试次**。"
                "同一最终段中，FS_H0 的曲线水平 MAE 较低（**7.34 vs 9.21 个百分点**），但曲线相关仍只有 **0.19**，"
                "低于 FA2 M=3 的 **0.50**；去掉平均水平后，FA2 M=3 的形状误差反而更小。\n\n"
                "这说明 FS_H0 赢的是最终阶段的平均校准，不是动态轨迹。它的 0.230 NLL 还非常接近事后为每名被试只设一个固定正确率的理论下界 0.226；"
                "这个下界使用了留出答案，不能作为合法竞争模型，但说明该后缀中剩余的动态可预测信息很少。"
            ),
        },
        {"id": "phase_chart", "type": "chart", "chartId": "phase_nll", "layout": "full"},
        {
            "id": "mechanism_finding",
            "type": "markdown",
            "sourceId": "accuracy_diagnostic",
            "body": (
                "## 机制诊断指向“过早规则塌缩 + 静态 lapse 代偿”\n\n"
                "FS_H0 在第 32 次之后给正确类别约 0.89–0.91 概率的试次比例，中位数达到 **94.3%**。"
                "此后实际 accuracy 曲线的被试内标准差中位数为 **14.3 个百分点**，FS_H0 只有 **1.37 个百分点**。"
                "到最终留出结束时，32 名被试中 **28 名**的最高权重参数组合位于 lapse 上界 0.20，"
                "且 lapse=0.20 的后验质量中位数为 **1.00**。\n\n"
                "静态 lapse 本应解释孤立误按或走神；现在它同时承担了尚未学会、错误规则坚持、阶段性退步和真正偶发错误。"
                "这会提高平均 choice 概率，却必然抹平学习曲线，也使 lapse 的心理解释失效。"
            ),
        },
        {"id": "model_table", "type": "table", "tableId": "model_detail", "layout": "full"},
        {
            "id": "scope_definitions",
            "type": "markdown",
            "sourceId": "accuracy_diagnostic",
            "body": (
                "## 数据范围与指标定义\n\n"
                "分析覆盖 condition 1 的 **32 名被试、10,048 个独特试次和 7 个冻结模型**，被试等权。"
                "绿色实际曲线是 0/1 正确性的尾随平均；模型曲线是每次在观察当前 choice 之前给正确类别的概率，再作同样尾随平均。"
                "主窗口为 16 次，8 与 32 次用于敏感性检查。\n\n"
                "NLL 评估模型给**真实 choice**的概率，accuracy 曲线评估模型给**正确类别**的概率如何随学习变化。"
                "二者都是有效信息，但回答不同问题；分类 accuracy（argmax 是否正确）在接近天花板的后缀里尤其不敏感。"
            ),
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "accuracy_diagnostic",
            "body": (
                "## 诊断方法与稳健性检查\n\n"
                "每名被试、每个模型分别计算曲线 MAE、中心化 MAE、相关、动态幅度、75% 稳定起点、NLL 和 Brier，"
                "再对被试等权汇总。FA2 M=3 相对 FS_H0 的全序列曲线改善为 **6.73 个百分点**，"
                "被试 bootstrap 95% 区间为 **5.30–8.14 个百分点**。窗口改为 8 或 32 次，FS_H0 的全序列 MAE 仍分别为 "
                "**16.79/14.88**，FA2 M=3 为 **11.17/7.31** 个百分点，方向不变。"
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## 结论边界：这还不能直接证明有限工作空间\n\n"
                "全序列曲线包含训练段，所以可用于发现过程失配，不能当作新的独立模型选择结果。当前曲线还是一步预测："
                "模型在每次预测后继续读取真实 choice/feedback 来更新粒子或参数权重，并非从切点开始自由生成整条行为轨迹。"
                "因此它比真正的自主生成更容易。\n\n"
                "有限模型的曲线更好，只说明离散搜索/有限激活这类动力学值得保留；它不证明被试意识里确实只有 M 条规则。"
                "当前 oral 图又只是实际报告与目标规则的一致性，不是模型特异的报告似然。需要自由生成、RT 或正式 oral readout 才能获得认知层解释。"
            ),
        },
        {
            "id": "recommendations",
            "type": "markdown",
            "body": (
                "## 推荐的下一版模型与评价顺序\n\n"
                "1. **把 FS_H0 降级为预测基线。** 它保留作数值边界和校准参照，但不能凭最终后缀 NLL 被解释为学习过程。\n\n"
                "2. **采用“背景证据库 + 小型活跃集 + 承诺状态”的混合架构。** 全部规则可在背景层缓慢积累证据；"
                "choice 与 report 只读取少量活跃规则；当证据达到阈值时进入 committed 状态，负反馈或高 surprise 再触发 reopen/search。"
                "这既保留旧模型的有限 hypothesis set transition，也能产生晚学、突然学会和阶段性退步。\n\n"
                "3. **把结构性错误与污染分开。** 错误串、重复负反馈和与 oral rule 一致的错误，应由错误假设状态解释；"
                "孤立且前后不一致的 choice 才交给 lapse。不要先开放任意逐试次 lapse，否则它会再次吞掉学习转折。\n\n"
                "4. **修订评价设计。** 使用覆盖早、中、晚学习阶段的 rolling-origin folds；在预测段冻结静态参数后验；"
                "同时报告一步 NLL 与不读取真实后续 choice 的自由轨迹。模型先通过 accuracy onset、曲线形状、错误串和校准的基本 adequacy gate，再比较 NLL。\n\n"
                "5. **不要直接最小化平滑 accuracy 曲线。** 这会丢掉具体选择信息并重复使用相邻窗口。NLL 仍保留为 proper score；"
                "accuracy/生成统计作为独立的过程充分性约束。"
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## 下一轮需要回答的三个问题\n\n"
                "- 被试的错误是孤立噪声，还是成串地指向同一个错误规则？\n\n"
                "- 口头报告或 RT 是否在行为转折前后提供 active-set/commitment 的独立证据？\n\n"
                "- 冻结参数并自由生成后，混合模型能否同时保住最终 NLL、学习起点和个体曲线，而不是只改善其中一个？"
            ),
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "condition 1 中 FS_H0 的 NLL 与 accuracy 轨迹矛盾诊断，以及下一版模型设计建议。",
            "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
            "cards": cards,
            "charts": charts,
            "tables": [table],
            "sources": all_sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "ready",
            "datasets": {
                "headline": headline,
                "model_comparison": comparison_rows,
                "onset_comparison": onset_rows,
                "phase_comparison": phase_rows,
                "window_sensitivity": sensitivity_rows,
            },
        },
        "sources": all_sources,
    }


def main() -> None:
    artifact = build_artifact()
    output = REPORT_DIR / "artifact.json"
    with output.open("w", encoding="utf-8") as stream:
        json.dump(artifact, stream, ensure_ascii=False, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
