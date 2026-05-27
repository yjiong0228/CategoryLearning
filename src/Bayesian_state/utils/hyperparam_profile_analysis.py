"""Subject-wise hyperparameter profile analysis.

This script combines outer subject-wise hyper-opt choices, inner memory-grid
choices, and oral/model alignment summaries for PMH condition 1 and 3.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_EXCLUDED_SUBJECTS = {102, 108, 123, 313, 317, 319, 321, 328}

DEFAULT_CONDITIONS = {
    "cond1": Path("results/state-based-grid-result/pmh/cond1_subjectwise_hyper_best"),
    "cond3": Path("results/state-based-grid-result/pmh/cond3_subjectwise_hyper_cd_best"),
}

DEFAULT_OUTPUT_DIR = Path("results/state-based-grid-result/pmh/hyperparam_profile_analysis")


def _safe_float(value: Any) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _format_value(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _strategy_atom(strategy: dict[str, Any]) -> str:
    method = str(strategy.get("method", "unknown"))
    amount = strategy.get("amount", "unknown")
    if amount == "fixed":
        value = strategy.get("value", "?")
        if method == "top_posterior":
            suffix = f"top{value}"
            if "top_p" in strategy:
                suffix += f"_p{strategy['top_p']}"
            return suffix
        if method == "random":
            return f"random{value}"
        if method == "ksimilar_centers":
            proto = strategy.get("proto_hypo_amount", "?")
            return f"ksim{value}_proto{proto}"
        return f"{method}{value}"
    amount_text = str(amount)
    if method == "random_posterior":
        return f"randpost_{amount_text}"
    if method == "random" and amount_text.startswith("opp_"):
        return amount_text
    return f"{method}_{amount_text}"


def _strategy_family(strategies: list[dict[str, Any]]) -> str:
    methods = [str(s.get("method", "")) for s in strategies]
    amounts = [str(s.get("amount", "")) for s in strategies]
    if "top_posterior" in methods and "ksimilar_centers" in methods:
        return "top_ksim_random"
    if "top_posterior" in methods and "random" in methods:
        return "top_random"
    if "random_posterior" in methods and any("entropy" in a for a in amounts):
        return "entropy_random_posterior"
    if "random_posterior" in methods and any("confidence" in a for a in amounts):
        return "confidence_random_posterior"
    if "random_posterior" in methods:
        return "random_random_posterior"
    return "+".join(methods) if methods else "unknown"


def _flatten_outer_hyperparams(hyperparams: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    strategies: list[dict[str, Any]] = []
    for key, value in hyperparams.items():
        if key.endswith(".strategies"):
            strategies = list(value)
            continue
        short = key.split(".")[-1]
        out[short] = value

    out["strategy_signature"] = "+".join(_strategy_atom(s) for s in strategies)
    out["strategy_family"] = _strategy_family(strategies)
    out["n_strategies"] = len(strategies)
    for idx, strategy in enumerate(strategies, start=1):
        out[f"strategy{idx}_method"] = strategy.get("method")
        out[f"strategy{idx}_amount"] = strategy.get("amount")
        out[f"strategy{idx}_value"] = strategy.get("value")
        out[f"strategy{idx}_top_p"] = strategy.get("top_p")
        out[f"strategy{idx}_proto_hypo_amount"] = strategy.get("proto_hypo_amount")
        out[f"strategy{idx}_atom"] = _strategy_atom(strategy)
    return out


def _gamma_bin(gamma: float) -> str:
    if gamma < 0.4:
        return "fast_recent"
    if gamma < 0.7:
        return "moderate"
    return "long_history"


def _w0_bin(w0: float) -> str:
    if w0 <= 0.02:
        return "low_floor"
    if w0 <= 0.08:
        return "moderate_floor"
    return "high_floor"


def _memory_interpretation(gamma: float, w0: float) -> str:
    gamma_label = _gamma_bin(gamma)
    w0_label = _w0_bin(w0)
    if gamma_label == "fast_recent" and w0_label == "low_floor":
        return "strong_recency_low_remote_trace"
    if gamma_label == "fast_recent":
        return "strong_recency_with_remote_baseline"
    if gamma_label == "moderate" and w0_label == "low_floor":
        return "moderate_integration_low_remote_trace"
    if gamma_label == "moderate":
        return "moderate_integration_with_remote_baseline"
    if w0_label == "low_floor":
        return "long_history_but_low_remote_floor"
    return "long_history_with_remote_baseline"


def _summarize_memory_grid(grid_summary: list[dict[str, Any]]) -> dict[str, Any]:
    if not grid_summary:
        return {}
    ranked = sorted(grid_summary, key=lambda item: _safe_float(item.get("mean_error")))
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    best_mean = _safe_float(best.get("mean_error"))
    top2_margin = (
        _safe_float(second.get("mean_error")) - best_mean if second is not None else float("nan")
    )
    tol_abs = 0.005
    tol_rel = 0.05 * best_mean
    tol = max(tol_abs, tol_rel)
    near = [item for item in ranked if _safe_float(item.get("mean_error")) <= best_mean + tol]
    near_gammas = sorted({_safe_float(item["params"].get("gamma")) for item in near})
    near_w0s = sorted({_safe_float(item["params"].get("w0")) for item in near})

    if len(near) <= 2 and (pd.isna(top2_margin) or top2_margin >= 0.002):
        stability = "sharp"
    elif len(near) <= 4:
        stability = "moderate"
    else:
        stability = "broad"

    return {
        "grid_best_mean_error": best_mean,
        "grid_best_std_error": _safe_float(best.get("std_error")),
        "grid_best_error": _safe_float(best.get("best_error")),
        "grid_top2_mean_error_margin": top2_margin,
        "near_optimal_tolerance": tol,
        "near_optimal_n": len(near),
        "near_optimal_gamma_values": ";".join(f"{x:g}" for x in near_gammas),
        "near_optimal_w0_values": ";".join(f"{x:g}" for x in near_w0s),
        "near_optimal_gamma_span": max(near_gammas) - min(near_gammas) if near_gammas else float("nan"),
        "near_optimal_w0_span": max(near_w0s) - min(near_w0s) if near_w0s else float("nan"),
        "memory_identifiability": stability,
    }


def _load_alignment(root: Path, oral_mode: str) -> pd.DataFrame:
    mode_dir = root / "plots" / oral_mode
    frames: list[pd.DataFrame] = []

    hit_path = mode_dir / "hit_based_alignment_subject_metrics.csv"
    if hit_path.exists():
        hit = pd.read_csv(hit_path)
        keep = [
            "subject",
            "model_hit_rate",
            "oral_hit_rate",
            "joint_hit_rate",
            "active_set_size_mean",
            "oral_topn_mass_mean",
            "active_oral_mass_mean",
            "phi_correlation",
            "cohen_kappa",
            "hit_agreement_rate",
            "positive_hit_jaccard",
        ]
        hit = hit[[c for c in keep if c in hit.columns]].copy()
        hit = hit.rename(columns={c: f"center_hit_{c}" for c in hit.columns if c != "subject"})
        frames.append(hit)

    coverage_path = mode_dir / "coverage_based_alignment_subject_means.csv"
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        keep = [
            "subject",
            "active_capture_ratio",
            "active_topn_overlap",
            "active_oral_mass",
            "oracle_topn_oral_mass",
            "random_expected_mass",
            "n_active",
            "active_fraction",
        ]
        coverage = coverage[[c for c in keep if c in coverage.columns]].copy()
        coverage = coverage.rename(
            columns={c: f"center_coverage_{c}" for c in coverage.columns if c != "subject"}
        )
        frames.append(coverage)

    distribution_path = mode_dir / "distribution_based_alignment_subject_means.csv"
    if distribution_path.exists():
        dist = pd.read_csv(distribution_path)
        if {"subject", "alignment_space", "js_similarity"}.issubset(dist.columns):
            pivot = dist.pivot_table(
                index="subject",
                columns="alignment_space",
                values="js_similarity",
                aggfunc="mean",
            ).reset_index()
            pivot = pivot.rename(
                columns={c: f"center_distribution_js_{c}" for c in pivot.columns if c != "subject"}
            )
            frames.append(pivot)

    oral_path = mode_dir / "oral_based_alignment_subject_means.csv"
    if oral_path.exists():
        oral = pd.read_csv(oral_path)
        keep = [
            "subject",
            "oral_based_similarity",
            "expected_center_similarity",
            "fuzzy_iou_similarity",
            "fuzzy_cosine_similarity",
        ]
        oral = oral[[c for c in keep if c in oral.columns]].copy()
        oral = oral.rename(columns={c: f"center_oral_{c}" for c in oral.columns if c != "subject"})
        frames.append(oral)

    target_path = mode_dir / "target_based_alignment_subject_metrics.csv"
    if target_path.exists():
        target = pd.read_csv(target_path)
        metrics = [
            "model_target_prior_mean",
            "oral_target_mass_mean",
            "pearson_r",
            "spearman_rho",
            "cosine_similarity",
        ]
        if {"subject", "alignment_space"}.issubset(target.columns):
            pieces = []
            for metric in metrics:
                if metric not in target.columns:
                    continue
                pivot = target.pivot_table(
                    index="subject",
                    columns="alignment_space",
                    values=metric,
                    aggfunc="mean",
                ).reset_index()
                pivot = pivot.rename(
                    columns={c: f"center_target_{metric}_{c}" for c in pivot.columns if c != "subject"}
                )
                pieces.append(pivot)
            if pieces:
                merged = pieces[0]
                for piece in pieces[1:]:
                    merged = merged.merge(piece, on="subject", how="outer")
                frames.append(merged)

    if not frames:
        return pd.DataFrame(columns=["subject"])
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="subject", how="outer")
    return merged


def build_subject_profiles(
    condition_roots: dict[str, Path],
    excluded_subjects: set[int],
    oral_mode: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition, root in condition_roots.items():
        source = _load_json(root / "hyper_best_source.json")
        outer = source.get("per_subject_best", {})
        alignment = _load_alignment(root, oral_mode)

        for sid_text, record in sorted(outer.items(), key=lambda item: int(item[0])):
            subject = int(sid_text)
            if subject in excluded_subjects:
                continue
            subject_path = root / "subjects" / f"subject_{subject}.json"
            if not subject_path.exists():
                continue
            subject_payload = _load_json(subject_path)
            best_params = subject_payload.get("best_params", {})
            gamma = _safe_float(best_params.get("gamma"))
            w0 = _safe_float(best_params.get("w0"))
            hyperparams = record.get("best_hyperparams", {})
            row = {
                "condition": condition,
                "condition_id": subject_payload.get("condition"),
                "subject": subject,
                "excluded": False,
                "outer_best_stage": record.get("best_stage"),
                "outer_combination_index": record.get("best_combination_index"),
                "outer_mean_error": _safe_float(record.get("mean_error")),
                "outer_best_error": _safe_float(record.get("best_error")),
                "outer_random_seed": record.get("random_seed"),
                "inner_best_error": _safe_float(subject_payload.get("best_error")),
                "inner_mean_error": _safe_float(subject_payload.get("mean_error")),
                "inner_refit_mean_error": _safe_float(subject_payload.get("refit_mean_error")),
                "inner_std_error": _safe_float(subject_payload.get("std_error")),
                "gamma": gamma,
                "w0": w0,
                "gamma_exact": f"{gamma:g}",
                "w0_exact": f"{w0:g}",
                "gamma_bin": _gamma_bin(gamma),
                "w0_bin": _w0_bin(w0),
                "memory_profile": f"{_gamma_bin(gamma)}+{_w0_bin(w0)}",
                "memory_interpretation": _memory_interpretation(gamma, w0),
            }
            row.update(_flatten_outer_hyperparams(hyperparams))
            row.update(_summarize_memory_grid(subject_payload.get("grid_summary", [])))
            rows.append(row)

        if rows and not alignment.empty:
            current_subjects = {r["subject"] for r in rows if r["condition"] == condition}
            alignment = alignment[alignment["subject"].isin(current_subjects)].copy()
            for idx, row in enumerate(rows):
                if row["condition"] != condition:
                    continue
                matched = alignment[alignment["subject"] == row["subject"]]
                if matched.empty:
                    continue
                values = matched.iloc[0].to_dict()
                values.pop("subject", None)
                rows[idx].update(values)

    return pd.DataFrame(rows).sort_values(["condition", "subject"]).reset_index(drop=True)


def value_counts_table(profiles: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for condition, group in profiles.groupby("condition"):
        n = len(group)
        for feature in features:
            if feature not in group.columns:
                continue
            counts = group[feature].fillna("<NA>").map(_format_value).value_counts(dropna=False)
            for value, count in counts.items():
                rows.append(
                    {
                        "condition": condition,
                        "feature": feature,
                        "value": value,
                        "count": int(count),
                        "proportion": count / n if n else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _cramers_v(x: Iterable[Any], y: Iterable[Any]) -> tuple[float, int, int, int]:
    table = pd.crosstab(pd.Series(list(x), dtype="object"), pd.Series(list(y), dtype="object"))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return float("nan"), int(table.values.sum()), table.shape[0], table.shape[1]
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.values.sum()
    k = min(table.shape)
    return math.sqrt(chi2 / (n * (k - 1))), int(n), table.shape[0], table.shape[1]


def _permutation_p_cramers(
    x: pd.Series,
    y: pd.Series,
    observed: float,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    if pd.isna(observed):
        return float("nan")
    x_vals = x.to_numpy(dtype=object)
    y_vals = y.to_numpy(dtype=object)
    ge = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(y_vals)
        v, _, _, _ = _cramers_v(x_vals, shuffled)
        if not pd.isna(v) and v >= observed - 1e-12:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def pairwise_associations(
    profiles: pd.DataFrame,
    features: list[str],
    n_perm: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for condition, group in profiles.groupby("condition"):
        for left, right in combinations(features, 2):
            if left not in group.columns or right not in group.columns:
                continue
            tmp = group[[left, right]].dropna()
            if len(tmp) < 5:
                continue
            left_values = tmp[left].map(_format_value)
            right_values = tmp[right].map(_format_value)
            if left_values.nunique() < 2 or right_values.nunique() < 2:
                continue
            observed, n, left_levels, right_levels = _cramers_v(left_values, right_values)
            p_perm = _permutation_p_cramers(left_values, right_values, observed, n_perm, rng)
            rows.append(
                {
                    "condition": condition,
                    "feature_x": left,
                    "feature_y": right,
                    "n": n,
                    "levels_x": left_levels,
                    "levels_y": right_levels,
                    "cramers_v": observed,
                    "p_perm": p_perm,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["condition", "p_perm", "cramers_v"], ascending=[True, True, False]
    )


def association_rules(
    profiles: pd.DataFrame,
    features: list[str],
    min_support_count: int = 3,
    min_confidence: float = 0.6,
    min_lift: float = 1.1,
) -> pd.DataFrame:
    rows = []
    for condition, group in profiles.groupby("condition"):
        n = len(group)
        items = {
            feature: group[feature].map(_format_value)
            for feature in features
            if feature in group.columns and group[feature].nunique(dropna=True) > 1
        }
        for left, right in combinations(items.keys(), 2):
            left_series = items[left]
            right_series = items[right]
            left_counts = left_series.value_counts()
            right_counts = right_series.value_counts()
            pair_counts = pd.crosstab(left_series, right_series)
            for left_value in pair_counts.index:
                for right_value in pair_counts.columns:
                    count = int(pair_counts.loc[left_value, right_value])
                    if count < min_support_count:
                        continue
                    for antecedent_feature, antecedent_value, consequent_feature, consequent_value in [
                        (left, left_value, right, right_value),
                        (right, right_value, left, left_value),
                    ]:
                        antecedent_count = (
                            int(left_counts[left_value])
                            if antecedent_feature == left
                            else int(right_counts[right_value])
                        )
                        consequent_count = (
                            int(right_counts[right_value])
                            if consequent_feature == right
                            else int(left_counts[left_value])
                        )
                        confidence = count / antecedent_count
                        baseline = consequent_count / n
                        lift = confidence / baseline if baseline else float("nan")
                        if confidence >= min_confidence and lift >= min_lift:
                            rows.append(
                                {
                                    "condition": condition,
                                    "antecedent": f"{antecedent_feature}={antecedent_value}",
                                    "consequent": f"{consequent_feature}={consequent_value}",
                                    "support_count": count,
                                    "support": count / n,
                                    "confidence": confidence,
                                    "lift": lift,
                                }
                            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "condition",
                "antecedent",
                "consequent",
                "support_count",
                "support",
                "confidence",
                "lift",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["condition", "lift", "confidence", "support_count"],
        ascending=[True, False, False, False],
    )


def alignment_spearman(profiles: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    metric_prefixes = (
        "center_hit_",
        "center_coverage_",
        "center_distribution_js_",
        "center_oral_",
        "center_target_",
    )
    metrics = [
        c
        for c in profiles.columns
        if c.startswith(metric_prefixes) and pd.api.types.is_numeric_dtype(profiles[c])
    ]
    rows = []
    for condition, group in profiles.groupby("condition"):
        for predictor in predictors:
            if predictor not in group.columns:
                continue
            for metric in metrics:
                tmp = group[[predictor, metric]].dropna()
                if len(tmp) < 6 or tmp[predictor].nunique() < 2 or tmp[metric].nunique() < 2:
                    continue
                rho, p_value = stats.spearmanr(tmp[predictor], tmp[metric])
                rows.append(
                    {
                        "condition": condition,
                        "predictor": predictor,
                        "metric": metric,
                        "n": len(tmp),
                        "spearman_rho": rho,
                        "p_value": p_value,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["condition", "predictor", "metric", "n", "spearman_rho", "p_value"])
    return pd.DataFrame(rows).sort_values(
        ["condition", "p_value", "spearman_rho"], ascending=[True, True, False]
    )


def _top_counts_markdown(df: pd.DataFrame, condition: str, feature: str, n: int = 8) -> str:
    sub = df[(df["condition"] == condition) & (df["feature"] == feature)].copy()
    if sub.empty:
        return "_No data._"
    sub = sub.sort_values(["count", "value"], ascending=[False, True]).head(n)
    return sub[["value", "count", "proportion"]].to_markdown(index=False, floatfmt=".3f")


def _top_rows_markdown(df: pd.DataFrame, condition: str, columns: list[str], n: int = 10) -> str:
    sub = df[df["condition"] == condition].head(n)
    if sub.empty:
        return "_No data._"
    return sub[columns].to_markdown(index=False, floatfmt=".3f")


def _write_report(
    output_dir: Path,
    profiles: pd.DataFrame,
    counts: pd.DataFrame,
    pairwise: pd.DataFrame,
    rules: pd.DataFrame,
    alignment: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Subject-wise Hyperparameter Profile Analysis")
    lines.append("")
    lines.append("本报告合并外层 subject-wise hyper-opt、内层 memory grid (`gamma`, `w0`) 与 `oral_center_mode` 对齐指标。")
    lines.append(f"排除被试: {', '.join(str(x) for x in sorted(DEFAULT_EXCLUDED_SUBJECTS))}。")
    lines.append("")

    n_by_cond = profiles.groupby("condition")["subject"].count()
    lines.append("## 样本")
    lines.append("")
    lines.append(n_by_cond.rename("n_subjects").reset_index().to_markdown(index=False))
    lines.append("")
    lines.append(
        "注意: cond1 与 cond3 的内层 memory 网格不同，跨 condition 的数值差异应先作为描述性结果，"
        "更稳妥的解释应结合近似最优区域与外层策略差异。"
    )
    lines.append("")

    for condition in ["cond1", "cond3"]:
        if condition not in set(profiles["condition"]):
            continue
        lines.append(f"## {condition} 画像概览")
        lines.append("")
        lines.append("### Memory 参数")
        lines.append("")
        lines.append("`gamma` 取值分布:")
        lines.append("")
        lines.append(_top_counts_markdown(counts, condition, "gamma_exact"))
        lines.append("")
        lines.append("`w0` 取值分布:")
        lines.append("")
        lines.append(_top_counts_markdown(counts, condition, "w0_exact"))
        lines.append("")
        lines.append("组合画像:")
        lines.append("")
        lines.append(_top_counts_markdown(counts, condition, "memory_profile"))
        lines.append("")

        stability = counts[(counts["condition"] == condition) & (counts["feature"] == "memory_identifiability")]
        lines.append("近似最优区域稳定性:")
        lines.append("")
        lines.append(stability[["value", "count", "proportion"]].to_markdown(index=False, floatfmt=".3f"))
        lines.append("")

        lines.append("### 外层策略与 beta 动态")
        lines.append("")
        for feature in ["strategy_family", "strategy_signature", "max_active_hypotheses", "init_num", "beta_init", "decrease_rate", "prior_beta_scale"]:
            lines.append(f"`{feature}`:")
            lines.append("")
            lines.append(_top_counts_markdown(counts, condition, feature))
            lines.append("")

    lines.append("## 群体层面关联")
    lines.append("")
    lines.append("Cramér's V 使用 condition 内 permutation p 值，主要用于探索性排序。")
    lines.append("")
    for condition in ["cond1", "cond3"]:
        lines.append(f"### {condition} strongest pairwise associations")
        lines.append("")
        lines.append(
            _top_rows_markdown(
                pairwise,
                condition,
                ["feature_x", "feature_y", "n", "levels_x", "levels_y", "cramers_v", "p_perm"],
                n=12,
            )
        )
        lines.append("")

    lines.append("## Association Rules")
    lines.append("")
    lines.append("规则格式为 `A -> B`，报告 support、confidence、lift；仅保留 support_count >= 3。")
    lines.append("")
    for condition in ["cond1", "cond3"]:
        lines.append(f"### {condition}")
        lines.append("")
        lines.append(
            _top_rows_markdown(
                rules,
                condition,
                ["antecedent", "consequent", "support_count", "support", "confidence", "lift"],
                n=15,
            )
        )
        lines.append("")

    lines.append("## Oral Alignment 探索")
    lines.append("")
    lines.append("下面列出 `gamma`, `w0` 与 oral/model alignment 指标的 Spearman 相关中 p 值最小的条目。")
    lines.append("")
    for condition in ["cond1", "cond3"]:
        lines.append(f"### {condition}")
        lines.append("")
        lines.append(
            _top_rows_markdown(
                alignment,
                condition,
                ["predictor", "metric", "n", "spearman_rho", "p_value"],
                n=15,
            )
        )
        lines.append("")

    lines.append("## 认知解释建议")
    lines.append("")
    lines.append(
        "1. 先把被试分为 memory-identifiability 为 `sharp/moderate/broad` 三类。"
        "`broad` 被试不宜解释为某一个唯一机制，更像是一组等价参数都能解释行为。"
    )
    lines.append(
        "2. 对 `sharp` 或 `moderate` 被试，再解释 memory profile: "
        "`fast_recent` 偏近因/快更新，`long_history` 偏长期整合，`w0` 越高表示旧经验仍有底线影响。"
    )
    lines.append(
        "3. 外层 strategy 与 active-set 参数解释的是假设空间搜索风格；"
        "`top_random` 偏保留高 posterior 假设并少量探索，"
        "`entropy_random_posterior` 更像不确定性驱动的候选假设补充。"
    )
    lines.append(
        "4. beta 参数解释决策锐度动态；高 `beta_init`/高 `prior_beta_scale` 表示对候选规则的初始区分更尖锐，"
        "高 `decrease_rate` 表示错误反馈后更强烈地下调不一致假设。"
    )
    lines.append(
        "5. oral alignment 指标可作为认知解释的约束: 若某类超参画像同时有更高 hit agreement、active capture 或 active-space JS similarity，"
        "它比单纯拟合误差更有资格被解释为被试的显性策略。"
    )
    lines.append("")

    (output_dir / "hyperparam_profile_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    output_dir: Path,
    excluded_subjects: set[int],
    oral_mode: str,
    n_perm: int,
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = build_subject_profiles(DEFAULT_CONDITIONS, excluded_subjects, oral_mode)

    count_features = [
        "gamma_exact",
        "w0_exact",
        "gamma_bin",
        "w0_bin",
        "memory_profile",
        "memory_identifiability",
        "memory_interpretation",
        "strategy_family",
        "strategy_signature",
        "max_active_hypotheses",
        "init_num",
        "beta_init",
        "decrease_rate",
        "prior_beta_scale",
    ]
    counts = value_counts_table(profiles, count_features)

    association_features = [
        "strategy_family",
        "strategy_signature",
        "max_active_hypotheses",
        "init_num",
        "beta_init",
        "decrease_rate",
        "prior_beta_scale",
        "gamma_exact",
        "w0_exact",
        "gamma_bin",
        "w0_bin",
        "memory_profile",
        "memory_identifiability",
    ]
    pairwise = pairwise_associations(profiles, association_features, n_perm=n_perm, seed=seed)
    rules = association_rules(
        profiles,
        [
            "strategy_family",
            "strategy_signature",
            "max_active_hypotheses",
            "init_num",
            "beta_init",
            "decrease_rate",
            "prior_beta_scale",
            "gamma_exact",
            "w0_exact",
            "memory_identifiability",
        ],
    )
    alignment = alignment_spearman(
        profiles,
        ["gamma", "w0", "beta_init", "decrease_rate", "prior_beta_scale", "max_active_hypotheses", "init_num"],
    )

    profiles.to_csv(output_dir / "subject_hyperparam_profiles.csv", index=False)
    counts.to_csv(output_dir / "hyperparam_value_counts.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_hyperparam_associations.csv", index=False)
    rules.to_csv(output_dir / "association_rules.csv", index=False)
    alignment.to_csv(output_dir / "alignment_spearman_associations.csv", index=False)
    _write_report(output_dir, profiles, counts, pairwise, rules, alignment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--oral-mode", default="oral_center_mode")
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument(
        "--exclude",
        type=int,
        nargs="*",
        default=sorted(DEFAULT_EXCLUDED_SUBJECTS),
        help="Subject IDs to exclude.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analysis(
        output_dir=args.output_dir,
        excluded_subjects=set(args.exclude),
        oral_mode=args.oral_mode,
        n_perm=args.n_perm,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
