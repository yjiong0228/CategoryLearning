"""Subject-wise hyperparameter profile analysis.

This script combines outer subject-wise hyper-opt choices, inner memory-grid
choices, and oral/model alignment summaries for PMH condition 1 and 3.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _factorize(values: Iterable[Any]) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(pd.Series(list(values), dtype="object"), sort=True)
    return codes.astype(int), len(uniques)


def _cramers_v_from_codes(
    x_codes: np.ndarray,
    y_codes: np.ndarray,
    n_x: int,
    n_y: int,
) -> tuple[float, int, int, int]:
    if n_x < 2 or n_y < 2:
        return float("nan"), int(len(x_codes)), n_x, n_y
    table = np.zeros((n_x, n_y), dtype=float)
    np.add.at(table, (x_codes, y_codes), 1.0)
    n = int(table.sum())
    row_sum = table.sum(axis=1, keepdims=True)
    col_sum = table.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    valid = expected > 0
    chi2 = float(np.sum(((table - expected) ** 2)[valid] / expected[valid]))
    k = min(n_x, n_y)
    return math.sqrt(chi2 / (n * (k - 1))), n, n_x, n_y


def _cramers_v(x: Iterable[Any], y: Iterable[Any]) -> tuple[float, int, int, int]:
    x_codes, n_x = _factorize(x)
    y_codes, n_y = _factorize(y)
    return _cramers_v_from_codes(x_codes, y_codes, n_x, n_y)


def _permutation_p_cramers(
    x: pd.Series,
    y: pd.Series,
    observed: float,
    n_perm: int,
    rng: np.random.Generator,
) -> float:
    if pd.isna(observed) or n_perm <= 0:
        return float("nan")
    x_codes, n_x = _factorize(x)
    y_codes, n_y = _factorize(y)
    ge = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(y_codes)
        v, _, _, _ = _cramers_v_from_codes(x_codes, shuffled, n_x, n_y)
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
    lines.append("下面列出超参与 oral/model alignment 指标的 Spearman 相关中 p 值最小的条目。")
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

    lines.append("## 综合认知解释")
    lines.append("")
    lines.append(
        "下面的解释把模型参数当成认知加工的不同层次: `gamma/w0` 描述经验证据在时间上的保留方式，"
        "`strategy_signature/max_active_hypotheses/init_num` 描述被试可能在假设空间里如何搜索规则，"
        "`beta_init/decrease_rate/prior_beta_scale` 描述规则信心如何初始化和被反馈更新，"
        "oral alignment 指标则作为外显报告对这些隐变量解释的约束。"
    )
    lines.append("")

    lines.append("### 1. 条件差异: cond1 更像近因驱动的规则利用，cond3 更像不确定性驱动的结构搜索")
    lines.append("")
    lines.append(
        "`cond1` 的主导画像是低到中等 `gamma` 加低 `w0`: `gamma=0.2/0.4` 覆盖 26/29 个被试，"
        "`w0=0.01/0.03` 覆盖 24/29 个被试。这意味着模型通常只需要较强地依赖近期反馈，"
        "远期试次保留很弱，就能解释被试行为。外层策略也高度一致，27/29 个被试属于 `top_random`: "
        "保留当前 posterior 较高的少数候选规则，再加一点随机探索。认知上，这更像一个相对紧凑的规则空间: "
        "被试主要在少数高可信规则附近更新，而不是持续广泛搜索。"
    )
    lines.append(
        "`cond3` 的画像明显不同。`gamma=0.6/0.7/0.8` 占 17/27，最大组合是 `long_history+low_floor`。"
        "外层策略中 20/27 是 `entropy_random_posterior`，并且多数还组合 `ksimilar_centers`。"
        "这说明 cond3 中行为更像是在较大的、结构更复杂的假设空间里工作: 被试需要保留更长历史，"
        "同时在不确定性较高时补充候选假设，并利用原型或相似中心来组织规则。"
    )
    lines.append(
        "因此，两个 condition 的差异不应只解释为记忆强弱差异。更合理的图景是: cond1 主要负荷在快速利用和局部更新，"
        "cond3 同时负荷在历史整合、候选假设维护和结构化搜索。"
    )
    lines.append("")

    lines.append("### 2. Memory 参数: `gamma` 是历史整合，`w0` 是远期经验的底线影响")
    lines.append("")
    lines.append(
        "`gamma` 越高，越说明较早试次仍能影响当前似然或规则信念；`w0` 越高，越说明即使很久以前的经验也不会衰减到接近零。"
        "这两个参数的认知含义不同: 高 `gamma` 是连续的历史整合，高 `w0` 更像远期经验的背景偏置或稳定先验。"
    )
    lines.append(
        "在 cond1 中，低 `w0` 与 `sharp` identifiability 共现较明显: association rule 显示 "
        "`memory_identifiability=sharp -> w0=0.01` 的 confidence 为 0.727，lift 为 1.622。"
        "这说明对一部分 cond1 被试，模型能比较明确地识别出一个近因主导、远期记忆底线很低的加工方式。"
    )
    lines.append(
        "在 cond3 中，`gamma=0.7` 与 `w0=0.01` 是最清楚的 memory 共现规则: "
        "`gamma=0.7 -> w0=0.01` 的 confidence 为 0.750，反向 `w0=0.01 -> gamma=0.7` 的 confidence 为 0.857，"
        "lift 均约 2.893。这个组合很有认知意义: 被试整合较长历史，但远期经验的最低权重仍很低。"
        "换句话说，他们不是简单地把所有旧经验都平均保留，而是保留一条较长的证据轨迹，同时允许很旧的信息逐渐退出主导地位。"
    )
    lines.append("")

    lines.append("### 3. Identifiability 本身也是认知信号: broad 不是噪声，而可能是多策略等价")
    lines.append("")
    lines.append(
        "`memory_identifiability` 衡量的是误差面是否尖锐。`sharp` 表示只有少数组合能解释行为，"
        "`broad` 表示很多 memory 组合都差不多好。这个指标不只是技术诊断，也能反映行为是否足够约束某种单一机制。"
    )
    lines.append(
        "cond1 中 `broad` 只有 6/29，但它和 `max_active_hypotheses=3` 强关联: "
        "Cramer's V=0.538, permutation p=0.014；列联表中 active=3 包含全部 6 个 broad，被试 active=4 时没有 broad。"
        "一个可能解释是，较小 active set 会把行为压缩到少数候选规则上，导致不同记忆衰减曲线都能产生类似预测；"
        "而 active=4 时模型保留更多竞争规则，行为中的细微差异反而更能揭示具体 memory profile。"
    )
    lines.append(
        "cond3 中 `broad` 高达 13/27，应更加谨慎。这里 broad 不一定是坏拟合，而可能说明被试在复杂任务里采用了混合加工: "
        "有时依赖长期结构，有时依赖近期反馈，有时受显性策略或局部原型吸引。单个 `gamma/w0` 点只是这个混合过程的一个等价投影。"
    )
    lines.append("")

    lines.append("### 4. 假设空间搜索: active set、init_num 与 strategy 反映探索-利用权衡")
    lines.append("")
    lines.append(
        "cond1 的 `top_random` 策略说明多数被试像是在做窄范围的 exploitation: posterior 高的规则被持续保留，"
        "随机候选只提供少量探索。association rules 中 `top1+random2 -> decrease_rate=0.3` "
        "和 `top2_p0.7+random1 -> decrease_rate=0.2` 提示搜索策略和反馈敏感性是耦合的: "
        "当模型更依赖极少数 top 规则时，错误反馈后需要更强地惩罚不一致规则，才能跟上被试的转向。"
    )
    lines.append(
        "cond1 里 `init_num` 与 `w0` 的关联也值得注意: Cramer's V=0.562, p=0.026。"
        "`init_num=2` 更多对应 `w0=0.03/0.08`，而 `init_num=3` 更多对应 `w0=0.01`。"
        "这可以理解为一种替代关系: 初始假设更少时，模型需要给旧经验保留一点底线影响来稳定行为；"
        "初始探索稍多时，模型可以通过候选假设本身吸收不确定性，不必让远期记忆维持较高权重。"
    )
    lines.append(
        "cond3 的搜索机制更开放。主要策略 `randpost_entropy_7+ksim...` 表示候选补充由 entropy 驱动，"
        "且通过 `ksimilar_centers` 组织相似规则。这和人类在复杂分类任务中的一种常见加工方式相符: "
        "不是枚举所有规则，而是在不确定时围绕当前可解释的原型或相似中心扩展候选空间。"
        "虽然 `strategy_signature ~ gamma` 的 permutation p=0.109 未达到常规阈值，但 Cramer's V=0.654，"
        "模式上显示不同搜索策略倾向对应不同历史整合程度，值得作为后续假设。"
    )
    lines.append("")

    lines.append("### 5. Beta 和反馈动态: 信心不是单独参数，而是和搜索/记忆共同塑形")
    lines.append("")
    lines.append(
        "`beta_init` 可以理解为新候选规则一开始的决策锐度，`prior_beta_scale` 是 prior 对初始锐度的放大，"
        "`decrease_rate` 是错误反馈后对不一致规则的惩罚强度。"
    )
    lines.append(
        "cond1 中 `max_active_hypotheses ~ prior_beta_scale` 关联较强，V=0.515, p=0.023；"
        "association rule 显示 `prior_beta_scale=10 -> max_active_hypotheses=3` 的 confidence 为 1.000。"
        "这提示当 active set 较小的时候，模型更依赖 prior 来快速区分候选规则；当 active set 较大时，"
        "则可以通过保留更多候选来表达不确定性。也就是说，人类可能有两种等价方式处理不确定性: "
        "一种是少候选但强信心筛选，另一种是多候选但让后续反馈慢慢筛。"
    )
    lines.append(
        "cond3 中 `decrease_rate ~ memory_identifiability` 是最强 pairwise 结果之一，V=0.450, p=0.025。"
        "`decrease_rate=0.2 -> memory_identifiability=moderate` 的 confidence 为 0.750，lift 为 3.375；"
        "`decrease_rate=0.1` 则更容易落在 broad。认知上可以理解为: 过弱的错误惩罚让多种 memory 轨迹都能解释行为，"
        "中等惩罚反而让模型更容易识别出较稳定的更新模式；而强惩罚 0.3 在 cond3 中同时出现在 broad 和 sharp，"
        "说明有些被试对反馈非常敏感，但这种敏感性可以服务于不同策略。"
    )
    lines.append("")

    lines.append("### 6. Oral alignment: 哪些模型机制更像被试自己说出来的策略")
    lines.append("")
    lines.append(
        "oral alignment 结果很关键，因为它帮助区分“能拟合选择”的机制和“接近被试显性报告”的机制。"
        "如果一个超参只改善选择拟合，但和 oral report 不一致，它可能是模型补偿项；"
        "如果它同时提高 hit agreement、active capture 或 target correlation，则更可能反映被试真实使用的策略。"
    )
    lines.append(
        "cond1 中最稳定的 oral 关联是 `prior_beta_scale` 的负相关: 它与 active/union_topn 空间中的 target Pearson/Spearman "
        "均为负相关，其中 `center_target_pearson_r_union_topn` rho=-0.662, p<0.001，"
        "`center_target_pearson_r_active` rho=-0.623, p=0.001；它也与 `cohen_kappa` 和 `phi_correlation` 负相关。"
        "这说明 prior 放大越强，模型的 target belief 越不贴近被试口述中心所指向的假设。"
        "认知解释上，cond1 被试的显性策略可能更依赖近期可观察反馈，而不是模型内部的强 prior 初始化；"
        "过强 prior 虽能帮助选择拟合，却可能把模型推向被试没有口头报告的隐性规则偏置。"
    )
    lines.append(
        "cond1 中 `init_num` 与 oral alignment 也多为负相关: `init_num` 与 `center_hit_phi_correlation` rho=-0.464, p=0.022，"
        "与 active capture rho=-0.396, p=0.034。这个结果支持一个简单解释: cond1 的显性策略比较集中，"
        "初始候选过多会让模型 active set 包含更多被试没有报告的规则，从而降低和口述策略的重合。"
    )
    lines.append(
        "cond1 的 `gamma` 和 `w0` 都与 active-space JS similarity 负相关。也就是说，在这个条件下，"
        "历史整合越强或远期底线越高，模型在 active set 内的分布形状越不像 oral distribution。"
        "这进一步支持 cond1 是近因主导、显性策略较局部的任务状态。"
    )
    lines.append(
        "cond3 的 oral 结果呈现另一种机制。`w0` 与 hit/kappa/phi/target correlation 多为正相关: "
        "`w0 ~ cohen_kappa` rho=0.495, p=0.009，`w0 ~ phi_correlation` rho=0.483, p=0.011，"
        "`w0 ~ target_pearson_r_active` rho=0.491, p=0.009。"
        "这说明在复杂任务里，保留一定远期经验底线反而更接近被试口述策略。"
        "一种解释是，被试在 cond3 中会把较早形成的结构性假设维持为背景框架，即使近期反馈有波动，也不会完全丢掉。"
    )
    lines.append(
        "同时，cond3 的 `w0` 与 full-space JS similarity 为负相关 rho=-0.515, p=0.006。"
        "这看似矛盾，但其实很有信息量: 高 `w0` 让模型更容易在 target hit 层面和口述报告一致，"
        "但它不一定让整个假设空间分布形状一致。换言之，被试口述可能抓住了目标或局部结构，"
        "而模型的全局 posterior 仍包含许多未被口述的竞争规则。"
    )
    lines.append(
        "cond3 中 `gamma` 与 oral_topn_mass、active_oral_mass、active_set_size 等指标为正相关，"
        "例如 `gamma ~ oral_topn_mass_mean` rho=0.515, p=0.006，"
        "`gamma ~ active_set_size_mean` rho=0.496, p=0.009。"
        "这提示长期历史整合会让模型保留更大的候选集合，并使 oral top-N 所覆盖的质量更高。"
        "但 `gamma ~ center_distribution_js_active` 为负相关 rho=-0.446, p=0.020，说明高 gamma 捕获了更多口述相关假设，"
        "却未必在 active set 内按同样比例分配概率。认知上，这像是“保持多个可解释结构”而不是“锁定一个口述规则”。"
    )
    lines.append("")

    lines.append("### 7. 可以形成的被试加工类型")
    lines.append("")
    lines.append(
        "基于这些结果，可以先把被试粗分为几类，而不是逐个解释孤立参数。"
    )
    lines.append(
        "第一类是 cond1 中的近因利用型: `top_random`、低 `w0`、`sharp/moderate` memory identifiability。"
        "这类被试可能主要维护少数当前有效规则，近期反馈快速改变 posterior，口述策略也更集中。"
    )
    lines.append(
        "第二类是 cond1 中的压缩/等价型: `max_active_hypotheses=3` 且 `broad`。"
        "这些被试的行为可由多种 memory 曲线解释，可能说明他们的选择主要由少数规则切换或局部启发式驱动，"
        "而不是由稳定的时间衰减机制唯一决定。"
    )
    lines.append(
        "第三类是 cond3 中的不确定性结构搜索型: `entropy_random_posterior + ksimilar_centers`，较大 active set，"
        "中高 `gamma`。这类被试可能持续维护多个候选分类结构，并在不确定时围绕相似中心扩展规则。"
    )
    lines.append(
        "第四类是 cond3 中的长期轨迹低底线型: 典型组合为 `gamma=0.7,w0=0.01`。"
        "他们整合较长历史，但不会让很旧的经验保持高权重，适合解释一种“有历史感但仍能更新”的策略。"
    )
    lines.append(
        "第五类是 cond3 中的远期框架保留型: 较高 `w0`，oral hit/kappa/phi 更好，但 full-space JS 更差。"
        "这类被试可能有一个稳定的显性结构框架，口述时能命中目标，但模型内部仍需要许多竞争假设来解释完整选择轨迹。"
    )
    lines.append("")

    lines.append("### 8. 解释限制和下一步分析")
    lines.append("")
    lines.append(
        "这些结果是探索性的。pairwise association 使用 permutation p 值，但没有作为确认性假设做多重比较校正；"
        "oral alignment 的 Spearman p 值也应主要作为排序线索，而不是最终显著性结论。"
    )
    lines.append(
        "此外，cond1 和 cond3 的 memory grid 不同，所以跨条件比较应看整体画像和分箱，不应过度比较某一个精确取值。"
        "尤其是 `broad` 被试，不适合写成“这个人就是 gamma=某值”，更适合写成“这个人的行为不能唯一约束记忆衰减机制”。"
    )
    lines.append(
        "下一步最有价值的分析是: 对上述五类被试分别画 trial-level posterior、active-set size、oral hit trace 和 feedback 后 beta 更新，"
        "看这些机制是否真的在时间序列上表现为不同的学习阶段。若这些类别在 trial-level 动态中也分离，"
        "就可以更有把握地把它们解释为人类任务加工机制，而不仅是参数共现。"
    )
    lines.append("")
    lines.append("## 输出图")
    lines.append("")
    lines.append("- `figures/memory_pair_heatmap_cond1.png`: cond1 的 `gamma x w0` 组合频数。")
    lines.append("- `figures/memory_pair_heatmap_cond3.png`: cond3 的 `gamma x w0` 组合频数。")
    lines.append("- `figures/gamma_w0_scatter.png`: 两个 condition 的 memory 组合散点，点大小表示人数。")
    lines.append("- `figures/memory_identifiability_by_condition.png`: 近似最优区域稳定性分布。")
    lines.append("- `figures/strategy_family_by_condition.png`: 外层策略家族分布。")
    lines.append("- `subject_profile_table.md`: 便于人工浏览的被试级精简画像表。")
    lines.append("")

    (output_dir / "hyperparam_profile_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_subject_profile_markdown(output_dir: Path, profiles: pd.DataFrame) -> None:
    columns = [
        "condition",
        "subject",
        "strategy_signature",
        "max_active_hypotheses",
        "init_num",
        "beta_init",
        "decrease_rate",
        "prior_beta_scale",
        "gamma",
        "w0",
        "memory_identifiability",
        "near_optimal_n",
        "inner_best_error",
        "center_hit_phi_correlation",
        "center_coverage_active_capture_ratio",
        "center_distribution_js_active",
    ]
    existing = [c for c in columns if c in profiles.columns]
    compact = profiles[existing].copy()
    compact = compact.where(pd.notna(compact), "")
    text = [
        "# Subject Hyperparameter Profile Table",
        "",
        compact.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    (output_dir / "subject_profile_table.md").write_text("\n".join(text), encoding="utf-8")


def _sorted_numeric_labels(labels: Iterable[Any]) -> list[str]:
    return sorted([str(x) for x in labels], key=lambda value: float(value))


def _write_figures(output_dir: Path, profiles: pd.DataFrame) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    for condition, group in profiles.groupby("condition"):
        pivot = pd.crosstab(group["gamma_exact"], group["w0_exact"])
        row_labels = _sorted_numeric_labels(pivot.index)
        col_labels = _sorted_numeric_labels(pivot.columns)
        pivot = pivot.reindex(index=row_labels, columns=col_labels, fill_value=0)

        fig_width = max(5.0, 0.55 * len(col_labels) + 2.0)
        fig_height = max(4.0, 0.45 * len(row_labels) + 1.7)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("w0")
        ax.set_ylabel("gamma")
        ax.set_title(f"{condition}: gamma x w0 best-combo counts")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = int(pivot.iat[i, j])
                if value:
                    ax.text(j, i, str(value), ha="center", va="center", color="black", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(fig_dir / f"memory_pair_heatmap_{condition}.png", dpi=180)
        plt.close(fig)

    agg = (
        profiles.groupby(["condition", "gamma", "w0"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    colors = {"cond1": "#3b82f6", "cond3": "#ef4444"}
    for condition, group in agg.groupby("condition"):
        ax.scatter(
            group["gamma"],
            group["w0"],
            s=70 + 45 * group["count"],
            alpha=0.72,
            label=condition,
            color=colors.get(condition),
            edgecolor="white",
            linewidth=0.8,
        )
        for _, row in group.iterrows():
            if row["count"] > 1:
                ax.text(row["gamma"], row["w0"], str(int(row["count"])), ha="center", va="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("gamma")
    ax.set_ylabel("w0 (log scale)")
    ax.set_title("Memory best-combo landscape")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "gamma_w0_scatter.png", dpi=180)
    plt.close(fig)

    for feature, filename, title in [
        ("memory_identifiability", "memory_identifiability_by_condition.png", "Memory identifiability"),
        ("strategy_family", "strategy_family_by_condition.png", "Outer strategy family"),
    ]:
        table = pd.crosstab(profiles["condition"], profiles[feature])
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        table.plot(kind="bar", stacked=True, ax=ax)
        ax.set_xlabel("condition")
        ax.set_ylabel("subject count")
        ax.set_title(title)
        ax.legend(title=feature, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / filename, dpi=180)
        plt.close(fig)


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
        "strategy_signature",
        "max_active_hypotheses",
        "init_num",
        "beta_init",
        "decrease_rate",
        "prior_beta_scale",
        "gamma_exact",
        "w0_exact",
        "memory_identifiability",
    ]
    pairwise = pairwise_associations(profiles, association_features, n_perm=n_perm, seed=seed)
    rules = association_rules(
        profiles,
        [
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
    _write_figures(output_dir, profiles)
    _write_subject_profile_markdown(output_dir, profiles)
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
