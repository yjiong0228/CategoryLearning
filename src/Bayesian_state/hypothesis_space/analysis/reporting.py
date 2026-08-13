"""Tables, figures, concise report, and provenance for the oral audit."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .oral_evidence import (
    COVERAGE_PRIMITIVES,
    KEY_COLUMNS,
    PRIMITIVES,
    PRIMITIVE_BY_KEY,
    ROOT,
    git_revision,
    sha256_file,
)


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_prevalence_plot(prevalence: pd.DataFrame, output_path: Path) -> None:
    _configure_plot_style()
    data = prevalence[prevalence["level_kind"].eq("task_arity")]
    order = [primitive.key for primitive in PRIMITIVES]
    labels = [PRIMITIVE_BY_KEY[key].label_en for key in order]
    binary = data[data["task_arity"].eq("binary")].set_index("primitive")
    four = data[data["task_arity"].eq("four_category")].set_index("primitive")
    y = np.arange(len(order))
    height = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 6.4))
    ax.barh(y - height / 2, [100 * binary.loc[key, "subject_rate"] for key in order], height, color="#1473e6", label="Binary")
    ax.barh(y + height / 2, [100 * four.loc[key, "subject_rate"] for key in order], height, color="#d88916", label="Four-category")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Subjects mentioning the primitive at least once (%)")
    ax.set_title("Oral relation primitives by task arity")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_complexity_plot(inventory: pd.DataFrame, output_path: Path) -> None:
    _configure_plot_style()
    counts = inventory.groupby(["task_arity", "cognitive_predicates"]).size().rename("hypotheses").reset_index()
    levels = sorted(counts["cognitive_predicates"].unique())
    x = np.arange(len(levels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for offset, (arity, color, label) in enumerate((("binary", "#1473e6", "Binary"), ("four_category", "#d88916", "Four-category"))):
        lookup = counts[counts["task_arity"].eq(arity)].set_index("cognitive_predicates")["hypotheses"]
        values = [int(lookup.get(level, 0)) for level in levels]
        bars = ax.bar(x + (offset - 0.5) * width, values, width, color=color, label=label)
        ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_xticks(x, [str(level) for level in levels])
    ax.set_xlabel("Named cognitive predicates per hypothesis")
    ax.set_ylabel("Concrete hypotheses")
    ax.set_title("Current hypothesis-space complexity")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_coverage_plot(coverage: pd.DataFrame, output_path: Path) -> None:
    _configure_plot_style()
    scenarios = ["current", "core_additions", "extended_pilot"]
    x = np.arange(len(scenarios))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for offset, (arity, color, label) in enumerate((("binary", "#1473e6", "Binary"), ("four_category", "#d88916", "Four-category"))):
        lookup = coverage[coverage["task_arity"].eq(arity)].set_index("scenario")
        values = [100 * float(lookup.loc[item, "full_trial_coverage_rate"]) for item in scenarios]
        bars = ax.bar(x + (offset - 0.5) * width, values, width, color=color, label=label)
        ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3, fontsize=8)
    ax.set_xticks(x, ["Current", "Core additions", "Extended pilot"])
    ax.set_ylabel("Structured oral reports fully covered (%)")
    ax.set_title("Coverage of decision-relevant oral primitives")
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _rate_row(prevalence: pd.DataFrame, arity: str, primitive: str) -> pd.Series:
    return prevalence[
        prevalence["level_kind"].eq("task_arity")
        & prevalence["task_arity"].eq(arity)
        & prevalence["primitive"].eq(primitive)
    ].iloc[0]


def _format_rate(row: pd.Series) -> str:
    return f"{int(row.subjects_ever)}/{int(row.subjects)} ({100 * row.subject_rate:.1f}%)"


def render_markdown_report(
    prevalence: pd.DataFrame,
    inventory: pd.DataFrame,
    coverage: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write an answer-first report without duplicating all CSV detail."""
    pair_binary = _rate_row(prevalence, "binary", "pairwise_near_equal")
    center_binary = _rate_row(prevalence, "binary", "center_band")
    pair_four = _rate_row(prevalence, "four_category", "pairwise_near_equal")
    current_binary = coverage.query("task_arity == 'binary' and scenario == 'current'").iloc[0]
    core_binary = coverage.query("task_arity == 'binary' and scenario == 'core_additions'").iloc[0]
    current_four = coverage.query("task_arity == 'four_category' and scenario == 'current'").iloc[0]
    core_four = coverage.query("task_arity == 'four_category' and scenario == 'core_additions'").iloc[0]
    binary_count = int((inventory.task_arity == "binary").sum())
    four_count = int((inventory.task_arity == "four_category").sum())
    text = f"""# Task2 hypothesis space：结论与证据

## 结论

二分类最值得补的是成对近似带和中心带；当前实现已经加入这两类。因此二分类空间由原来的 19 条扩为 {binary_count} 条。四分类目前仍保持 {four_count} 条，下一步最值得检验的是同一对维度的 similarity quartet，而不是立刻展开任意谓词组合。

## 为什么这样补

- 成对近似相等在二分类中覆盖 {_format_rate(pair_binary)} 名被试，在四分类中覆盖 {_format_rate(pair_four)}。
- 中心/适中描述在二分类中覆盖 {_format_rate(center_binary)} 名被试。
- 当前所谓 pairwise order 只表示 `x_i < x_j` 与 `x_i > x_j`；近似带才表示 `|x_i-x_j| <= delta`，二者不是同一个 hypothesis。
- band 的补集可能不连通，所以 prototype 实现按连通分量自动求质心；boundary 实现直接使用同一 space 中的多面体区域。

## 口述覆盖变化

这里的覆盖率指“结构化口述中的全部关系原语是否可由 grammar 表达”，不是行为拟合优度。

- 二分类：{100 * current_binary.full_trial_coverage_rate:.1f}% → {100 * core_binary.full_trial_coverage_rate:.1f}%
- 四分类候选 grammar：{100 * current_four.full_trial_coverage_rate:.1f}% → {100 * core_four.full_trial_coverage_rate:.1f}%

四分类的后一个数字仍是候选 grammar 的覆盖能力，不表示 similarity quartet 已经进入正式模型。

## 合理性与完备性的边界

不应声称覆盖所有数学上可能的分割。更可检验的定义是：

1. 实验真实生成规则必须可表示；
2. 高频口述原语必须进入 core 或明确记录为 pilot；
3. 规则复杂度用软先验惩罚，而不是设置武断的硬上限；
4. 新 family 只有在被试或 session 留出预测中有增益，才从候选 grammar 升为正式 hypothesis。

完整 trial 标记、被试频率、候选 family、容差敏感性和可复现信息都在本目录的 CSV 与 `analysis_manifest.json` 中。
"""
    output_path.write_text(text, encoding="utf-8")


def audit_sample(frame: pd.DataFrame, per_primitive: int = 20) -> pd.DataFrame:
    rows = []
    for arity in ("binary", "four_category"):
        subset = frame[frame["task_arity"].eq(arity)]
        for primitive in PRIMITIVE_BY_KEY:
            sample = subset[subset[primitive]].sample(
                n=min(per_primitive, int(subset[primitive].sum())),
                random_state=20260812,
            ).copy()
            sample["primitive"] = primitive
            rows.append(sample)
    columns = [*KEY_COLUMNS, "condition", "task_arity", "primitive", "text", "fidelity"]
    result = pd.concat(rows, ignore_index=True)[columns]
    result["human_relation_present"] = ""
    result["human_decision_rule_evidence"] = ""
    result["reviewer_notes"] = ""
    return result


def _artifact_payload(
    prevalence: pd.DataFrame,
    inventory: pd.DataFrame,
    candidates: pd.DataFrame,
    coverage: pd.DataFrame,
    generated_at: str,
) -> dict:
    """Build the portable report snapshot consumed by artifact viewers."""
    def records(table: pd.DataFrame) -> list[dict]:
        return json.loads(table.to_json(orient="records"))

    prevalence_chart = prevalence[prevalence["level_kind"].eq("task_arity")].copy()
    prevalence_chart["task"] = prevalence_chart["task_arity"].map(
        {"binary": "Binary", "four_category": "Four-category"}
    )
    coverage_chart = coverage.copy()
    coverage_chart["task"] = coverage_chart["task_arity"].map(
        {"binary": "Binary", "four_category": "Four-category"}
    )
    complexity = (
        inventory.groupby(["task_arity", "cognitive_predicates"])
        .size()
        .rename("hypotheses")
        .reset_index()
    )
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Task2 hypothesis-space audit",
            "description": "Evidence for bounded continuous-space extensions.",
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "primitive_prevalence",
                    "title": "Oral relation primitives by task arity",
                    "type": "bar",
                    "dataset": "primitive_prevalence",
                },
                {
                    "id": "coverage_scenarios",
                    "title": "Structured reports covered by each grammar",
                    "type": "bar",
                    "dataset": "coverage_scenarios",
                },
            ],
            "tables": [
                {
                    "id": "candidate_families",
                    "title": "Candidate hypothesis families",
                    "dataset": "candidate_families",
                }
            ],
            "sources": [
                {"id": "task2_data", "path": "data/processed/Task2_processed.csv"},
                {"id": "oral_diagnostics", "path": "results/oral_analysis/Task2_oral_trial_diagnostics.csv"},
                {"id": "partition_code", "path": "src/Bayesian_state/hypothesis_space/observation_model/continuous_partition.py"},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "primitive_prevalence": records(prevalence_chart),
                "candidate_families": records(candidates),
                "complexity": records(complexity),
                "coverage_scenarios": records(coverage_chart),
            },
            "accessIssues": [],
        },
    }


def write_outputs(
    *,
    frame: pd.DataFrame,
    subject_counts: pd.DataFrame,
    prevalence: pd.DataFrame,
    inventory: pd.DataFrame,
    family_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    coverage: pd.DataFrame,
    cooccurrence: pd.DataFrame,
    sensitivity: pd.DataFrame,
    examples: pd.DataFrame,
    sample: pd.DataFrame,
    output_dir: Path,
    data_path: Path,
    diagnostics_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    flags = [*KEY_COLUMNS, "condition", "task_arity", "text", "nonempty_text", "fidelity", "fidelity_status", *PRIMITIVE_BY_KEY, "n_structural_primitives", "pair_equality_claim_count", "middle_part_count", "unparsed_item_count"]
    outputs = {
        "oral_primitive_trial_flags.csv": frame[flags],
        "subject_primitive_counts.csv": subject_counts,
        "primitive_prevalence.csv": prevalence,
        "existing_partition_inventory.csv": inventory,
        "existing_partition_family_summary.csv": family_summary,
        "candidate_families.csv": candidates,
        "coverage_scenarios.csv": coverage,
        "primitive_cooccurrence.csv": cooccurrence,
        "equality_tolerance_sensitivity.csv": sensitivity,
        "evidence_examples.csv": examples,
        "audit_sample.csv": sample,
    }
    for filename, table in outputs.items():
        table.to_csv(output_dir / filename, index=False)
    save_prevalence_plot(prevalence, output_dir / "primitive_subject_prevalence.png")
    save_complexity_plot(inventory, output_dir / "existing_space_complexity.png")
    save_coverage_plot(coverage, output_dir / "coverage_scenarios.png")
    render_markdown_report(prevalence, inventory, coverage, output_dir / "report.md")

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    (output_dir / "artifact.json").write_text(
        json.dumps(
            _artifact_payload(
                prevalence,
                inventory,
                candidates,
                coverage,
                generated_at,
            ),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    partition_path = ROOT / "src/Bayesian_state/hypothesis_space/observation_model/continuous_partition.py"
    manifest = {
        "generated_at_utc": generated_at,
        "analysis_module": "src.Bayesian_state.hypothesis_space.analysis",
        "git_revision": git_revision(),
        "inputs": {
            str(data_path.relative_to(ROOT)): {"sha256": sha256_file(data_path), "rows": int(len(frame))},
            str(diagnostics_path.relative_to(ROOT)): {"sha256": sha256_file(diagnostics_path), "rows": int(len(frame))},
            str(partition_path.relative_to(ROOT)): {"sha256": sha256_file(partition_path)},
            "src/oral_coding.py": {"sha256": sha256_file(ROOT / "src/oral_coding.py")},
        },
        "population": {
            "trials": int(len(frame)),
            "subjects": int(frame["iSub"].nunique()),
            "binary_subjects": int(frame.loc[frame["task_arity"].eq("binary"), "iSub"].nunique()),
            "four_category_subjects": int(frame.loc[frame["task_arity"].eq("four_category"), "iSub"].nunique()),
            "nonempty_oral_reports": int(frame["nonempty_text"].sum()),
        },
        "analysis_parameters": {
            "repeated_subject_threshold_trials": 3,
            "semantic_equality_tolerance_context": 0.10,
            "proposed_model_sensitivity_grid": [0.06, 0.10, 0.15],
            "coverage_primitives": list(COVERAGE_PRIMITIVES),
        },
        "validation": {
            "trial_keys_one_to_one": True,
            "raw_data_edited": False,
            "oral_reports_are_not_treated_as_decision_rule_labels": True,
            "human_audit_status": "pending; fill audit_sample.csv",
        },
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = ["audit_sample", "render_markdown_report", "write_outputs"]
