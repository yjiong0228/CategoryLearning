#!/usr/bin/env python3
"""Held-out gate for previous rule uncertainty beyond static FA2 choice."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0806_targeted_diagnostics import (  # noqa: E402
    atomic_frame,
    atomic_json,
    bernoulli_log_density,
    bootstrap_summary,
    build_predictor_table,
    clipped_logit,
    fit_offset_logistic,
    read_yaml,
    standardize,
    subject_design,
)


DEFAULT_CONFIG = ROOT / "configs/model_0806_uncertainty_gate.yaml"
DEFAULT_OUTPUT = (
    ROOT / "results/zhuran/model_0806_cond1/uncertainty_gate_20260806_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def choice_designs(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    lag_column: str,
    previous_error_column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    subjects = sorted(int(value) for value in evaluation["subject_id"].unique())
    train_base, evaluation_base = subject_design(train, evaluation, subjects)
    train_log_trial, evaluation_log_trial, _, _ = standardize(
        train["log_trial"], evaluation["log_trial"]
    )
    (
        train_uncertainty,
        evaluation_uncertainty,
        uncertainty_mean,
        uncertainty_scale,
    ) = standardize(train[lag_column], evaluation[lag_column])
    train_controls = np.column_stack([
        train_log_trial,
        train["ambiguous"].to_numpy(dtype=float),
        train[previous_error_column].to_numpy(dtype=float),
    ])
    evaluation_controls = np.column_stack([
        evaluation_log_trial,
        evaluation["ambiguous"].to_numpy(dtype=float),
        evaluation[previous_error_column].to_numpy(dtype=float),
    ])
    baseline_train = np.column_stack([train_base, train_controls])
    baseline_evaluation = np.column_stack([evaluation_base, evaluation_controls])
    candidate_train = np.column_stack([baseline_train, train_uncertainty])
    candidate_evaluation = np.column_stack([
        baseline_evaluation, evaluation_uncertainty
    ])
    return (
        baseline_train,
        baseline_evaluation,
        candidate_train,
        candidate_evaluation,
        {
            "uncertainty_mean": float(uncertainty_mean),
            "uncertainty_scale": float(uncertainty_scale),
        },
    )


def run_choice_gate(
    table: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    lag_policy: str,
    ridge_penalty: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if lag_policy == "within_block":
        lag_column = "lag_uncertainty_within"
        previous_error_column = "previous_error_within"
    elif lag_policy == "all_transitions":
        lag_column = "lag_uncertainty_all"
        previous_error_column = "previous_error_all"
    else:
        raise ValueError(f"unknown lag policy: {lag_policy}")
    first_block = int(config["design"]["first_evaluation_block"])
    fold_rows: list[dict[str, Any]] = []
    trial_frames: list[pd.DataFrame] = []
    maximum_block = int(table["evaluation_block"].max())
    for evaluation_block in range(first_block, maximum_block + 1):
        eligible_subjects = sorted(
            int(value)
            for value in table.loc[
                table["evaluation_block"] == evaluation_block, "subject_id"
            ].unique()
        )
        if not eligible_subjects:
            continue
        include_subject = table["subject_id"].isin(eligible_subjects)
        valid = table[[lag_column, previous_error_column]].notna().all(axis=1)
        train = table[
            include_subject & (table["evaluation_block"] < evaluation_block) & valid
        ].copy()
        evaluation = table[
            include_subject & (table["evaluation_block"] == evaluation_block) & valid
        ].copy()
        if len(train) < 100 or len(evaluation) < 10:
            continue
        (
            baseline_train,
            baseline_evaluation,
            candidate_train,
            candidate_evaluation,
            scaling,
        ) = choice_designs(
            train,
            evaluation,
            lag_column=lag_column,
            previous_error_column=previous_error_column,
        )
        train_offset = clipped_logit(train["static_correct_probability"])
        evaluation_offset = clipped_logit(
            evaluation["static_correct_probability"]
        )
        baseline_beta, baseline_diagnostics = fit_offset_logistic(
            baseline_train,
            train["correct"].to_numpy(dtype=float),
            train_offset,
            float(ridge_penalty),
        )
        candidate_beta, candidate_diagnostics = fit_offset_logistic(
            candidate_train,
            train["correct"].to_numpy(dtype=float),
            train_offset,
            float(ridge_penalty),
        )
        baseline_probability = expit(
            evaluation_offset + baseline_evaluation @ baseline_beta
        )
        candidate_probability = expit(
            evaluation_offset + candidate_evaluation @ candidate_beta
        )
        outcome = evaluation["correct"].to_numpy(dtype=float)
        scored = evaluation[[
            "subject_id",
            "trial_index",
            "evaluation_block",
            "block_position",
        ]].copy()
        scored["lag_policy"] = lag_policy
        scored["ridge_penalty"] = float(ridge_penalty)
        scored["baseline_log_density"] = bernoulli_log_density(
            outcome, baseline_probability
        )
        scored["candidate_log_density"] = bernoulli_log_density(
            outcome, candidate_probability
        )
        scored["delta_log_density"] = (
            scored["candidate_log_density"] - scored["baseline_log_density"]
        )
        scored["baseline_probability_correct"] = baseline_probability
        scored["candidate_probability_correct"] = candidate_probability
        scored["correct"] = outcome
        trial_frames.append(scored)
        fold_rows.append({
            "lag_policy": lag_policy,
            "ridge_penalty": float(ridge_penalty),
            "evaluation_block": int(evaluation_block),
            "subjects": len(eligible_subjects),
            "train_trials": len(train),
            "evaluation_trials": len(evaluation),
            "delta_lpd_per_trial": float(np.mean(scored["delta_log_density"])),
            "uncertainty_coefficient_standardized": float(candidate_beta[-1]),
            "uncertainty_mean_train": scaling["uncertainty_mean"],
            "uncertainty_scale_train": scaling["uncertainty_scale"],
            "baseline_optimizer_success": bool(baseline_diagnostics["success"]),
            "candidate_optimizer_success": bool(candidate_diagnostics["success"]),
        })
    if not trial_frames:
        raise RuntimeError("uncertainty gate produced no evaluable folds")
    return pd.concat(trial_frames, ignore_index=True), pd.DataFrame(fold_rows)


def coefficient_summary(folds: pd.DataFrame) -> dict[str, Any]:
    values = folds["uncertainty_coefficient_standardized"].to_numpy(dtype=float)
    return {
        "folds": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "positive_folds": int(np.sum(values > 0.0)),
        "negative_folds": int(np.sum(values < 0.0)),
        "all_optimizers_successful": bool(
            folds["baseline_optimizer_success"].all()
            and folds["candidate_optimizer_success"].all()
        ),
    }


def phase_summary(
    folds: pd.DataFrame, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    phase_config = config["learning_phases"]
    masks = {
        "early": folds["evaluation_block"].isin(
            [int(value) for value in phase_config["early"]]
        ),
        "middle": folds["evaluation_block"].isin(
            [int(value) for value in phase_config["middle"]]
        ),
        "late": folds["evaluation_block"]
        >= int(phase_config["late_minimum_block"]),
    }
    result: dict[str, dict[str, Any]] = {}
    for phase, mask in masks.items():
        selected = folds.loc[mask, "uncertainty_coefficient_standardized"].to_numpy(
            dtype=float
        )
        if selected.size == 0:
            raise RuntimeError(f"learning phase {phase} has no evaluable folds")
        result[phase] = {
            "folds": int(selected.size),
            "mean": float(np.mean(selected)),
            "median": float(np.median(selected)),
            "positive_folds": int(np.sum(selected > 0.0)),
            "negative_folds": int(np.sum(selected < 0.0)),
        }
    return result


def data_quality_summary(table: pd.DataFrame) -> dict[str, Any]:
    selected = table[[
        "subject_id",
        "static_feedback_uncertainty",
        "static_feedback_surprise",
        "correct",
    ]].dropna()
    within_subject_sd = selected.groupby("subject_id")[
        "static_feedback_uncertainty"
    ].std(ddof=0)
    return {
        "trials": int(len(selected)),
        "subjects": int(selected["subject_id"].nunique()),
        "uncertainty_mean": float(selected["static_feedback_uncertainty"].mean()),
        "uncertainty_sd": float(selected["static_feedback_uncertainty"].std(ddof=0)),
        "uncertainty_minimum": float(selected["static_feedback_uncertainty"].min()),
        "uncertainty_maximum": float(selected["static_feedback_uncertainty"].max()),
        "median_within_subject_uncertainty_sd": float(within_subject_sd.median()),
        "correlation_with_surprise": float(
            selected["static_feedback_uncertainty"].corr(
                selected["static_feedback_surprise"]
            )
        ),
        "correlation_with_current_correct": float(
            selected["static_feedback_uncertainty"].corr(selected["correct"])
        ),
    }


def evaluate_support_gate(
    summary: Mapping[str, Any], phases: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    interval_pass = float(summary["bootstrap_mean_95_interval"][0]) > 0.0
    median = float(summary["coefficient"]["median"])
    nonzero_median = not np.isclose(median, 0.0, atol=1e-12, rtol=0.0)
    direction = int(np.sign(median))
    phase_directions = {
        phase: int(np.sign(float(values["mean"])))
        for phase, values in phases.items()
    }
    stable_direction = bool(
        direction != 0
        and all(value == direction for value in phase_directions.values())
    )
    return {
        "passed": bool(interval_pass and nonzero_median and stable_direction),
        "bootstrap_interval_lower_above_zero": bool(interval_pass),
        "coefficient_median_nonzero": bool(nonzero_median),
        "coefficient_direction": (
            "positive" if direction > 0 else "negative" if direction < 0 else "zero"
        ),
        "phase_directions": phase_directions,
        "same_direction_in_all_learning_phases": bool(stable_direction),
    }


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    primary = summary["primary"]
    sensitivity = summary["all_transition_sensitivity"]
    gate = primary["support_gate"]
    phases = primary["coefficient_by_learning_phase"]
    lines = [
        "# 0806 规则不确定性门控检验",
        "",
        "## 问题",
        "",
        "在静态 FA2 的正确选择概率、被试差异、练习、刺激歧义性和上一试次错误之外，上一试次的规则不确定性能否稳定改善下一试次选择的滚动留出预测？",
        "",
        "## 主结果",
        "",
        f"- 块内相邻试次：ΔLPD/试次={primary['mean_delta_lpd_per_trial']:.6f}，"
        f"被试 bootstrap 95% 区间 "
        f"[{primary['bootstrap_mean_95_interval'][0]:.6f}, "
        f"{primary['bootstrap_mean_95_interval'][1]:.6f}]。",
        f"- {primary['improved_subjects']}/{primary['subjects']} 名被试改善。",
        f"- 标准化不确定性系数中位数={primary['coefficient']['median']:.6f}，"
        f"{primary['coefficient']['positive_folds']}/{primary['coefficient']['folds']} 个折叠为正。",
        f"- 早期/中期/晚期系数均值分别为 "
        f"{phases['early']['mean']:.6f}、{phases['middle']['mean']:.6f}、"
        f"{phases['late']['mean']:.6f}。",
        f"- 包含跨块相邻试次的敏感性：ΔLPD/试次="
        f"{sensitivity['mean_delta_lpd_per_trial']:.6f}。",
        "",
        "## 判定",
        "",
        f"- 规则不确定性门槛：{'通过' if gate['passed'] else '未通过'}。",
        f"- 留出区间下界大于 0：{gate['bootstrap_interval_lower_above_zero']}。",
        f"- 早期、中期和晚期方向一致：{gate['same_direction_in_all_learning_phases']}。",
        "- 这个检查只是进入完整 FA3-M-U 的低成本门槛；即使通过，也不能单独证明不确定性会提高替换率。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    table = build_predictor_table(config)
    atomic_frame(args.output / "predictor_table.csv", table)
    primary_policy = str(config["design"]["primary_lag_policy"])
    sensitivity_policy = str(config["design"]["sensitivity_lag_policy"])
    ridge_penalty = float(config["choice_residual"]["ridge_penalty"])
    bootstrap_replicates = int(config["design"]["bootstrap_replicates"])
    bootstrap_seed = int(config["design"]["bootstrap_seed"])
    summary: dict[str, Any] = {
        "analysis_id": str(config["analysis_id"]),
        "data_quality": data_quality_summary(table),
        "guardrails": {
            "static_FA2_states_only": True,
            "block_frozen_candidate_weights": True,
            "uncertainty_lagged_one_trial": True,
            "current_feedback_never_predicts_current_choice": True,
            "independent_bootstrap_unit": "subject",
            "primary_lag_policy": primary_policy,
            "primary_ridge_penalty": ridge_penalty,
        },
    }
    results: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for index, (result_id, lag_policy) in enumerate((
        ("primary", primary_policy),
        ("all_transition_sensitivity", sensitivity_policy),
    )):
        trials, folds = run_choice_gate(
            table,
            config,
            lag_policy=lag_policy,
            ridge_penalty=ridge_penalty,
        )
        subjects, values = bootstrap_summary(
            trials,
            bootstrap_replicates=bootstrap_replicates,
            seed=bootstrap_seed + index,
        )
        values["coefficient"] = coefficient_summary(folds)
        values["coefficient_by_learning_phase"] = phase_summary(folds, config)
        if result_id == "primary":
            values["support_gate"] = evaluate_support_gate(
                values, values["coefficient_by_learning_phase"]
            )
        summary[result_id] = values
        results[result_id] = (trials, folds)
        atomic_frame(args.output / f"{result_id}_trial_scores.csv", trials)
        atomic_frame(args.output / f"{result_id}_folds.csv", folds)
        atomic_frame(args.output / f"{result_id}_subjects.csv", subjects)

    ridge_rows: list[dict[str, Any]] = []
    for index, sensitivity_ridge in enumerate(
        config["choice_residual"]["ridge_penalty_sensitivity"]
    ):
        trials, folds = run_choice_gate(
            table,
            config,
            lag_policy=primary_policy,
            ridge_penalty=float(sensitivity_ridge),
        )
        _, values = bootstrap_summary(
            trials,
            bootstrap_replicates=bootstrap_replicates,
            seed=bootstrap_seed + 10 + index,
        )
        coefficients = coefficient_summary(folds)
        ridge_rows.append({
            "ridge_penalty": float(sensitivity_ridge),
            "mean_delta_lpd_per_trial": values["mean_delta_lpd_per_trial"],
            "bootstrap_lower": values["bootstrap_mean_95_interval"][0],
            "bootstrap_upper": values["bootstrap_mean_95_interval"][1],
            "improved_subjects": values["improved_subjects"],
            "subjects": values["subjects"],
            "coefficient_median": coefficients["median"],
            "positive_folds": coefficients["positive_folds"],
            "folds": coefficients["folds"],
        })
    ridge_frame = pd.DataFrame(ridge_rows)
    summary["ridge_sensitivity"] = ridge_frame.to_dict(orient="records")
    atomic_frame(args.output / "ridge_sensitivity.csv", ridge_frame)
    atomic_json(args.output / "uncertainty_gate_summary.json", summary)
    write_report(args.output / "uncertainty_gate_report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
