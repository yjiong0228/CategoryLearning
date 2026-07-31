#!/usr/bin/env python3
"""Independent RT validation of frozen condition-1 B0 and D0 predictions.

The script never uses RT to select or refit the choice model.  It compares a
hierarchical/fixed-subject log-RT baseline containing B0 uncertainty with one
additional frozen B0-versus-D0 disagreement feature, evaluates the increment
on each subject's last block, checks particle/seed stability, and runs a
conditional synthetic recovery analysis for the RT emission coefficient.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Bayesian_state.utils.newplan_rt_validation import (
    cr1_standard_error,
    entropy_rows,
    jensen_shannon_rows,
    normalized_probabilities,
    robust_location_scale,
    subject_bootstrap_interval,
    total_variation_rows,
)


ALL_SUBJECTS = (103, 105, 111, 112, 117, 118, 127, 131)
HELDOUT_SUBJECTS = (103, 111, 112, 117, 118, 127, 131)
PRIMARY_THETA = 0.75
KEY_COLUMNS = ["iSub", "iSession", "iBlock", "iTrial"]


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    result_dir: str
    particle_count: int


@dataclass(frozen=True)
class AnalysisSpec:
    analysis_id: str
    use_rt_qc: bool
    dynamic_feature: str
    practice_model: str
    include_current_error: bool


SOURCES = (
    SourceSpec(
        "main_r128",
        "results/zhuran/cond1_newplan/particle_filter_dev_r32_64_128",
        128,
    ),
    SourceSpec(
        "main_r64",
        "results/zhuran/cond1_newplan/particle_filter_dev_r32_64_128",
        64,
    ),
    SourceSpec(
        "seed2_r64",
        "results/zhuran/cond1_newplan/particle_filter_dev_r64_seed2",
        64,
    ),
)

ANALYSES = (
    AnalysisSpec("primary", True, "jsd", "common", True),
    AnalysisSpec("all_positive_rt", False, "jsd", "common", True),
    AnalysisSpec("subject_practice_slopes", True, "jsd", "subject", True),
    AnalysisSpec("total_variation", True, "total_variation", "common", True),
    AnalysisSpec("omit_current_outcome", True, "jsd", "common", False),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/processed/Task2_processed.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_newplan/rt_validation_theta075"
        ),
    )
    parser.add_argument("--theta", type=float, default=PRIMARY_THETA)
    parser.add_argument("--qc-threshold", type=float, default=4.0)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    parser.add_argument("--recovery-draws", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=20260819)
    return parser.parse_args()


def theta_token(theta: float) -> str:
    return f"{float(theta):g}".replace("-", "m").replace(".", "p")


def cache_path(source: SourceSpec, subject: int, theta: float) -> Path:
    return (
        ROOT
        / source.result_dir
        / "cache"
        / f"subject_{int(subject)}"
        / f"particles_{source.particle_count}"
        / "replicate_0"
        / f"theta_{theta_token(theta)}.npz"
    )


def load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"particle-filter cache not found: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {
            key: np.asarray(payload[key])
            for key in payload.files
            if key != "metadata"
        }


def validate_input_data(data: pd.DataFrame) -> None:
    required = {
        "iSub",
        "condition",
        "iSession",
        "iBlock",
        "iTrial",
        "choice",
        "feedback",
        "ambiguous",
        "choRT",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"data are missing required columns: {missing}")
    rt = data["choRT"].to_numpy(dtype=float)
    if not np.all(np.isfinite(rt)):
        raise ValueError("choRT contains non-finite values")
    if np.any(rt <= 0.0):
        raise ValueError("choRT contains non-positive values")
    duplicates = data.duplicated(KEY_COLUMNS, keep=False)
    if bool(duplicates.any()):
        raise ValueError(
            "trial key is not unique; duplicate rows cannot be aligned safely"
        )


def add_trial_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["log_rt"] = np.log(result["choRT"].to_numpy(dtype=float))
    blocks = result.groupby(["iSession", "iBlock"], sort=False)
    result["lag_log_rt"] = blocks["log_rt"].shift(1)
    result["prev_error"] = 1.0 - blocks["feedback"].shift(1)
    result["current_error"] = 1.0 - result["feedback"].to_numpy(dtype=float)
    result["block_position"] = blocks.cumcount() + 1
    result["subject_trial"] = np.arange(1, len(result) + 1, dtype=int)
    result["log_trial"] = np.log1p(result["subject_trial"])
    result["log_block_position"] = np.log1p(result["block_position"])
    return result


def build_source_frame(
    data: pd.DataFrame,
    source: SourceSpec,
    *,
    theta: float,
    qc_threshold: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    subject_frames: list[pd.DataFrame] = []
    qc_rows: list[dict[str, Any]] = []
    for subject in ALL_SUBJECTS:
        frame = data[
            (data["condition"] == 1) & (data["iSub"] == subject)
        ].copy()
        frame = frame.reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"condition-1 subject {subject} is missing")
        b0 = load_cache(cache_path(source, subject, 0.0))
        d0 = load_cache(cache_path(source, subject, theta))
        b0_prob = normalized_probabilities(b0["marginal_probabilities"])
        d0_prob = normalized_probabilities(d0["marginal_probabilities"])
        if len(frame) != b0_prob.shape[0] or b0_prob.shape != d0_prob.shape:
            raise ValueError(
                f"cache/data length mismatch for {source.source_id}, subject {subject}"
            )
        observed = frame["choice"].to_numpy(dtype=int) - 1
        if not np.array_equal(observed, b0["observed_choice_index"]):
            raise ValueError(
                f"choice order mismatch for {source.source_id}, subject {subject}"
            )
        for mask_name in ("valid_mask", "train_mask", "test_mask"):
            if not np.array_equal(b0[mask_name], d0[mask_name]):
                raise ValueError(
                    f"B0/D0 {mask_name} mismatch for {source.source_id}, "
                    f"subject {subject}"
                )

        frame["source_id"] = source.source_id
        frame["particle_count"] = source.particle_count
        frame["theta"] = float(theta)
        frame["valid_mask"] = b0["valid_mask"].astype(bool)
        frame["train_mask"] = b0["train_mask"].astype(bool)
        frame["test_mask"] = b0["test_mask"].astype(bool)
        frame["heldout_eligible"] = bool(frame["test_mask"].any())
        frame["b0_prob_choice1"] = b0_prob[:, 0]
        frame["d0_prob_choice1"] = d0_prob[:, 0]
        frame["b0_entropy"] = entropy_rows(b0_prob)
        frame["d0_entropy"] = entropy_rows(d0_prob)
        frame["jsd"] = jensen_shannon_rows(b0_prob, d0_prob)
        frame["total_variation"] = total_variation_rows(b0_prob, d0_prob)
        one_hot = np.eye(b0_prob.shape[1], dtype=float)[observed]
        frame["b0_choice_brier_trial"] = np.sum(
            (b0_prob - one_hot) ** 2, axis=1
        )
        frame["d0_choice_brier_trial"] = np.sum(
            (d0_prob - one_hot) ** 2, axis=1
        )
        frame = add_trial_covariates(frame)

        qc_reference = (
            frame["train_mask"]
            & frame["valid_mask"]
            & frame["lag_log_rt"].notna()
        )
        location, scale = robust_location_scale(
            frame.loc[qc_reference, "log_rt"]
        )
        robust_z = np.abs((frame["log_rt"] - location) / scale)
        frame["rt_qc_keep"] = robust_z <= float(qc_threshold)
        frame["rt_robust_z"] = robust_z
        qc_rows.append(
            {
                "source_id": source.source_id,
                "iSub": int(subject),
                "train_log_rt_median": location,
                "train_log_rt_mad_scale": scale,
                "qc_threshold": float(qc_threshold),
                "all_n": int(len(frame)),
                "qc_excluded_n": int((~frame["rt_qc_keep"]).sum()),
                "train_qc_excluded_n": int(
                    (frame["train_mask"] & ~frame["rt_qc_keep"]).sum()
                ),
                "test_qc_excluded_n": int(
                    (frame["test_mask"] & ~frame["rt_qc_keep"]).sum()
                ),
            }
        )
        subject_frames.append(frame)

    combined = pd.concat(subject_frames, ignore_index=True)
    if bool(combined.duplicated(KEY_COLUMNS).any()):
        raise ValueError(f"duplicate trial keys after loading {source.source_id}")
    return combined, qc_rows


def formula_for(spec: AnalysisSpec) -> tuple[str, str]:
    terms = [
        "lag_log_rt",
        "prev_error",
        "ambiguous",
        "b0_entropy_z",
        "log_trial",
        "log_block_position",
        "C(iSub)",
    ]
    if spec.include_current_error:
        terms.insert(2, "current_error")
    if spec.practice_model == "subject":
        terms.append("C(iSub):log_trial")
    elif spec.practice_model != "common":
        raise ValueError(f"unknown practice model: {spec.practice_model}")
    baseline = "log_rt ~ " + " + ".join(terms)
    dynamic = baseline + f" + {spec.dynamic_feature}_z"
    return baseline, dynamic


def prepare_analysis_frame(
    frame: pd.DataFrame,
    spec: AnalysisSpec,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    working = frame[frame["iSub"].isin(HELDOUT_SUBJECTS)].copy()
    base_valid = working["valid_mask"] & working["lag_log_rt"].notna()
    if spec.use_rt_qc:
        base_valid &= working["rt_qc_keep"]
    train = base_valid & working["train_mask"]
    test = base_valid & working["test_mask"]
    if not bool(train.any()) or not bool(test.any()):
        raise ValueError(f"analysis {spec.analysis_id} has an empty split")
    for column in ("b0_entropy", spec.dynamic_feature):
        mean = float(working.loc[train, column].mean())
        standard_deviation = float(
            working.loc[train, column].std(ddof=0)
        )
        if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
            raise ValueError(
                f"{column} is degenerate in {spec.analysis_id}"
            )
        working[f"{column}_z"] = (
            working[column] - mean
        ) / standard_deviation
        working[f"{column}_train_mean"] = mean
        working[f"{column}_train_sd"] = standard_deviation
    return working, train, test


def cr1_many_standard_errors(
    exog: np.ndarray,
    residuals: np.ndarray,
    groups: np.ndarray,
    coefficient_index: int,
) -> np.ndarray:
    x = np.asarray(exog, dtype=float)
    errors = np.asarray(residuals, dtype=float)
    if errors.ndim == 1:
        errors = errors[:, None]
    unique = np.unique(groups)
    bread = np.linalg.pinv(x.T @ x)
    influence = x @ bread[:, int(coefficient_index)]
    cluster_scores = []
    for group in unique:
        mask = groups == group
        cluster_scores.append(
            np.sum(influence[mask, None] * errors[mask], axis=0)
        )
    scores = np.vstack(cluster_scores)
    correction = (unique.size / (unique.size - 1.0)) * (
        (x.shape[0] - 1.0) / max(x.shape[0] - x.shape[1], 1.0)
    )
    return np.sqrt(np.maximum(correction * np.sum(scores**2, axis=0), 0.0))


def exact_wild_cluster_pvalues(
    baseline_fit: Any,
    dynamic_fit: Any,
    groups: np.ndarray,
    coefficient_name: str,
) -> tuple[float, float, int]:
    x = np.asarray(dynamic_fit.model.exog, dtype=float)
    coefficient_index = list(dynamic_fit.model.exog_names).index(
        coefficient_name
    )
    observed_beta = float(dynamic_fit.params[coefficient_name])
    observed_se = cr1_standard_error(
        x,
        np.asarray(dynamic_fit.resid, dtype=float),
        groups,
        coefficient_index=coefficient_index,
    )
    observed_t = observed_beta / observed_se
    unique = np.unique(groups)
    patterns = np.asarray(
        list(itertools.product((-1.0, 1.0), repeat=unique.size)),
        dtype=float,
    )
    row_signs = np.empty((groups.size, patterns.shape[0]), dtype=float)
    for index, group in enumerate(unique):
        row_signs[groups == group] = patterns[:, index]
    y_star = (
        np.asarray(baseline_fit.fittedvalues, dtype=float)[:, None]
        + np.asarray(baseline_fit.resid, dtype=float)[:, None] * row_signs
    )
    pseudo_inverse = np.linalg.pinv(x)
    beta_star = pseudo_inverse @ y_star
    residual_star = y_star - x @ beta_star
    se_star = cr1_many_standard_errors(
        x, residual_star, groups, coefficient_index
    )
    t_star = beta_star[coefficient_index] / np.maximum(se_star, 1e-12)
    two_sided = float(
        (1 + np.sum(np.abs(t_star) >= abs(observed_t)))
        / (patterns.shape[0] + 1)
    )
    positive = float(
        (1 + np.sum(t_star >= observed_t)) / (patterns.shape[0] + 1)
    )
    return two_sided, positive, int(patterns.shape[0])


def gaussian_nll(y: np.ndarray, prediction: np.ndarray, sigma: float) -> float:
    variance = max(float(sigma) ** 2, 1e-12)
    errors = np.asarray(y, dtype=float) - np.asarray(prediction, dtype=float)
    return float(
        np.mean(0.5 * np.log(2.0 * np.pi * variance) + 0.5 * errors**2 / variance)
    )


def run_model_comparison(
    frame: pd.DataFrame,
    source: SourceSpec,
    spec: AnalysisSpec,
    *,
    bootstrap_draws: int,
    seed: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    pd.DataFrame,
]:
    working, train, test = prepare_analysis_frame(frame, spec)
    baseline_formula, dynamic_formula = formula_for(spec)
    train_frame = working.loc[train].copy()
    test_frame = working.loc[test].copy()
    baseline_fit = smf.ols(baseline_formula, data=train_frame).fit()
    dynamic_fit = smf.ols(dynamic_formula, data=train_frame).fit()
    feature_name = f"{spec.dynamic_feature}_z"
    feature_index = list(dynamic_fit.model.exog_names).index(feature_name)
    groups = train_frame["iSub"].to_numpy(dtype=int)
    coefficient = float(dynamic_fit.params[feature_name])
    standard_error = cr1_standard_error(
        np.asarray(dynamic_fit.model.exog, dtype=float),
        np.asarray(dynamic_fit.resid, dtype=float),
        groups,
        coefficient_index=feature_index,
    )
    n_clusters = int(np.unique(groups).size)
    degrees_freedom = n_clusters - 1
    t_statistic = coefficient / standard_error
    p_two_sided = float(
        2.0 * stats.t.sf(abs(t_statistic), df=degrees_freedom)
    )
    p_positive = float(stats.t.sf(t_statistic, df=degrees_freedom))
    critical = float(stats.t.ppf(0.975, df=degrees_freedom))
    ci_lower = coefficient - critical * standard_error
    ci_upper = coefficient + critical * standard_error
    wild_two_sided, wild_positive, wild_patterns = (
        exact_wild_cluster_pvalues(
            baseline_fit,
            dynamic_fit,
            groups,
            feature_name,
        )
    )

    baseline_prediction = np.asarray(
        baseline_fit.predict(test_frame), dtype=float
    )
    dynamic_prediction = np.asarray(
        dynamic_fit.predict(test_frame), dtype=float
    )
    observed = test_frame["log_rt"].to_numpy(dtype=float)
    baseline_error = observed - baseline_prediction
    dynamic_error = observed - dynamic_prediction
    baseline_sigma = float(np.sqrt(np.mean(baseline_fit.resid**2)))
    dynamic_sigma = float(np.sqrt(np.mean(dynamic_fit.resid**2)))

    scored = test_frame[
        KEY_COLUMNS
        + [
            "source_id",
            "jsd",
            "total_variation",
            "b0_entropy",
            "b0_choice_brier_trial",
            "d0_choice_brier_trial",
        ]
    ].copy()
    scored["analysis_id"] = spec.analysis_id
    scored["baseline_prediction_log_rt"] = baseline_prediction
    scored["dynamic_prediction_log_rt"] = dynamic_prediction
    scored["observed_log_rt"] = observed
    scored["baseline_squared_error"] = baseline_error**2
    scored["dynamic_squared_error"] = dynamic_error**2

    subject_rows: list[dict[str, Any]] = []
    for subject, group in scored.groupby("iSub", sort=True):
        baseline_mse = float(group["baseline_squared_error"].mean())
        dynamic_mse = float(group["dynamic_squared_error"].mean())
        choice_test = working[
            (working["iSub"] == subject) & working["test_mask"]
        ]
        subject_rows.append(
            {
                "source_id": source.source_id,
                "particle_count": source.particle_count,
                "analysis_id": spec.analysis_id,
                "iSub": int(subject),
                "test_rt_n": int(len(group)),
                "baseline_rmse_log_rt": float(np.sqrt(baseline_mse)),
                "dynamic_rmse_log_rt": float(np.sqrt(dynamic_mse)),
                "baseline_mae_log_rt": float(
                    np.mean(
                        np.abs(
                            group["observed_log_rt"]
                            - group["baseline_prediction_log_rt"]
                        )
                    )
                ),
                "dynamic_mae_log_rt": float(
                    np.mean(
                        np.abs(
                            group["observed_log_rt"]
                            - group["dynamic_prediction_log_rt"]
                        )
                    )
                ),
                "baseline_minus_dynamic_mse": (
                    baseline_mse - dynamic_mse
                ),
                "dynamic_wins_mse": bool(dynamic_mse < baseline_mse),
                "test_mean_jsd": float(group["jsd"].mean()),
                "test_mean_total_variation": float(
                    group["total_variation"].mean()
                ),
                "b0_test_choice_brier": float(
                    choice_test["b0_choice_brier_trial"].mean()
                ),
                "d0_test_choice_brier": float(
                    choice_test["d0_choice_brier_trial"].mean()
                ),
                "b0_minus_d0_test_choice_brier": float(
                    (
                        choice_test["b0_choice_brier_trial"]
                        - choice_test["d0_choice_brier_trial"]
                    ).mean()
                ),
            }
        )
    subject_values = np.asarray(
        [row["baseline_minus_dynamic_mse"] for row in subject_rows],
        dtype=float,
    )
    interval_lower, interval_upper = subject_bootstrap_interval(
        subject_values,
        n_bootstrap=bootstrap_draws,
        seed=seed,
    )
    win_count = int(
        sum(bool(row["dynamic_wins_mse"]) for row in subject_rows)
    )
    sign_test = stats.binomtest(
        win_count, n=len(subject_rows), p=0.5, alternative="two-sided"
    )
    summary = {
        "source_id": source.source_id,
        "particle_count": source.particle_count,
        "analysis_id": spec.analysis_id,
        "use_rt_qc": spec.use_rt_qc,
        "dynamic_feature": spec.dynamic_feature,
        "practice_model": spec.practice_model,
        "include_current_error": spec.include_current_error,
        "train_n": int(train.sum()),
        "test_n": int(test.sum()),
        "subject_n": n_clusters,
        "dynamic_feature_train_mean": float(
            working.loc[train, spec.dynamic_feature].mean()
        ),
        "dynamic_feature_train_sd": float(
            working.loc[train, spec.dynamic_feature].std(ddof=0)
        ),
        "dynamic_coefficient_per_train_sd": coefficient,
        "dynamic_effect_percent_per_train_sd": float(
            100.0 * np.expm1(coefficient)
        ),
        "cluster_standard_error": standard_error,
        "cluster_df": degrees_freedom,
        "cluster_t": t_statistic,
        "cluster_p_two_sided": p_two_sided,
        "cluster_p_positive_cost": p_positive,
        "cluster_ci_lower": ci_lower,
        "cluster_ci_upper": ci_upper,
        "wild_cluster_p_two_sided": wild_two_sided,
        "wild_cluster_p_positive_cost": wild_positive,
        "wild_cluster_patterns": wild_patterns,
        "baseline_rmse_log_rt": float(
            np.sqrt(np.mean(baseline_error**2))
        ),
        "dynamic_rmse_log_rt": float(
            np.sqrt(np.mean(dynamic_error**2))
        ),
        "baseline_mae_log_rt": float(np.mean(np.abs(baseline_error))),
        "dynamic_mae_log_rt": float(np.mean(np.abs(dynamic_error))),
        "baseline_gaussian_nll": gaussian_nll(
            observed, baseline_prediction, baseline_sigma
        ),
        "dynamic_gaussian_nll": gaussian_nll(
            observed, dynamic_prediction, dynamic_sigma
        ),
        "mean_subject_baseline_minus_dynamic_mse": float(
            subject_values.mean()
        ),
        "bootstrap_dmse_ci_lower": interval_lower,
        "bootstrap_dmse_ci_upper": interval_upper,
        "dynamic_subject_win_count": win_count,
        "dynamic_subject_n": int(len(subject_rows)),
        "subject_win_sign_test_p": float(sign_test.pvalue),
        "baseline_formula": baseline_formula,
        "dynamic_formula": dynamic_formula,
    }
    return summary, subject_rows, scored


def describe_rt(values: pd.Series) -> dict[str, Any]:
    array = values.to_numpy(dtype=float)
    return {
        "n": int(array.size),
        "missing_n": int(values.isna().sum()),
        "nonpositive_n": int(np.sum(array <= 0.0)),
        "mean_seconds": float(np.mean(array)),
        "median_seconds": float(np.median(array)),
        "p95_seconds": float(np.quantile(array, 0.95)),
        "p99_seconds": float(np.quantile(array, 0.99)),
        "max_seconds": float(np.max(array)),
        "over_30_seconds_n": int(np.sum(array > 30.0)),
        "over_60_seconds_n": int(np.sum(array > 60.0)),
        "over_120_seconds_n": int(np.sum(array > 120.0)),
    }


def build_data_quality_summary(
    data: pd.DataFrame,
    main_frame: pd.DataFrame,
    qc_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    condition_one = data[data["condition"] == 1]
    target = condition_one[condition_one["iSub"].isin(ALL_SUBJECTS)]
    eligible = condition_one[condition_one["iSub"].isin(HELDOUT_SUBJECTS)]
    by_subject = {
        str(int(subject)): describe_rt(group["choRT"])
        for subject, group in target.groupby("iSub", sort=True)
    }
    main_qc = [
        row for row in qc_rows if row["source_id"] == "main_r128"
    ]
    return {
        "file": "data/processed/Task2_processed.csv",
        "row_n": int(len(data)),
        "column_n": int(data.shape[1]),
        "exact_duplicate_row_n": int(data.duplicated().sum()),
        "duplicate_trial_key_n": int(data.duplicated(KEY_COLUMNS).sum()),
        "all_conditions": describe_rt(data["choRT"]),
        "condition_1": describe_rt(condition_one["choRT"]),
        "target_8_subjects": describe_rt(target["choRT"]),
        "heldout_7_subjects": describe_rt(eligible["choRT"]),
        "by_target_subject": by_subject,
        "qc_definition": (
            "absolute log-RT deviation <= 4 Gaussian-consistent MADs "
            "from the subject's training median"
        ),
        "main_r128_qc_excluded_n": int(
            sum(row["qc_excluded_n"] for row in main_qc)
        ),
        "main_r128_qc_rows": main_qc,
        "aligned_trial_n": int(len(main_frame)),
        "aligned_choice_mismatch_n": 0,
    }


def seed_feature_stability(
    frames: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    comparisons = (
        ("main_r128", "main_r64", "particle_count"),
        ("main_r64", "seed2_r64", "independent_seed"),
    )
    rows: list[dict[str, Any]] = []
    columns = KEY_COLUMNS + ["valid_mask", "jsd", "total_variation"]
    for left_id, right_id, comparison in comparisons:
        left = frames[left_id][columns].copy()
        right = frames[right_id][columns].copy()
        merged = left.merge(
            right,
            on=KEY_COLUMNS,
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        valid = merged["valid_mask_left"] & merged["valid_mask_right"]
        merged = merged.loc[valid]
        pearson = stats.pearsonr(
            merged["jsd_left"], merged["jsd_right"]
        )
        spearman = stats.spearmanr(
            merged["jsd_left"], merged["jsd_right"]
        )
        rows.append(
            {
                "comparison": comparison,
                "left_source": left_id,
                "right_source": right_id,
                "trial_n": int(len(merged)),
                "jsd_pearson_r": float(pearson.statistic),
                "jsd_pearson_p": float(pearson.pvalue),
                "jsd_spearman_rho": float(spearman.statistic),
                "jsd_spearman_p": float(spearman.pvalue),
                "jsd_mean_absolute_difference": float(
                    np.mean(
                        np.abs(
                            merged["jsd_left"] - merged["jsd_right"]
                        )
                    )
                ),
                "tv_mean_absolute_difference": float(
                    np.mean(
                        np.abs(
                            merged["total_variation_left"]
                            - merged["total_variation_right"]
                        )
                    )
                ),
            }
        )
    return rows


def design_matrices(
    formula: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    y, x = patsy.dmatrices(
        formula, train_frame, return_type="dataframe"
    )
    test_x = patsy.build_design_matrices(
        [x.design_info], test_frame, return_type="dataframe"
    )[0]
    return (
        np.asarray(y).reshape(-1),
        np.asarray(x, dtype=float),
        np.asarray(test_x, dtype=float),
        list(x.columns),
    )


def empirical_noise(
    rng: np.random.Generator,
    residuals: np.ndarray,
    train_groups: np.ndarray,
    target_groups: np.ndarray,
    n_draws: int,
) -> np.ndarray:
    noise = np.empty((target_groups.size, n_draws), dtype=float)
    for group in np.unique(target_groups):
        pool = residuals[train_groups == group]
        if pool.size == 0:
            pool = residuals
        mask = target_groups == group
        noise[mask] = rng.choice(
            pool,
            size=(int(mask.sum()), n_draws),
            replace=True,
        )
    return noise


def conditional_rt_recovery(
    frame: pd.DataFrame,
    *,
    n_draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    primary = next(item for item in ANALYSES if item.analysis_id == "primary")
    working, train, test = prepare_analysis_frame(frame, primary)
    baseline_formula, dynamic_formula = formula_for(primary)
    train_frame = working.loc[train].copy()
    test_frame = working.loc[test].copy()
    observed, x_base, x_base_test, _ = design_matrices(
        baseline_formula, train_frame, test_frame
    )
    _, x_dynamic, x_dynamic_test, dynamic_names = design_matrices(
        dynamic_formula, train_frame, test_frame
    )
    coefficient_index = dynamic_names.index("jsd_z")
    jsd_train = train_frame["jsd_z"].to_numpy(dtype=float)
    jsd_test = test_frame["jsd_z"].to_numpy(dtype=float)
    base_inverse = np.linalg.pinv(x_base)
    dynamic_inverse = np.linalg.pinv(x_dynamic)
    base_parameters = base_inverse @ observed
    base_mean_train = x_base @ base_parameters
    base_mean_test = x_base_test @ base_parameters
    residuals = observed - base_mean_train
    sigma = float(np.sqrt(np.mean(residuals**2)))
    train_groups = train_frame["iSub"].to_numpy(dtype=int)
    test_groups = test_frame["iSub"].to_numpy(dtype=int)
    unique_groups = np.unique(train_groups)
    degrees_freedom = unique_groups.size - 1
    effect_levels = {
        "null": 0.0,
        "cost_5pct": float(np.log(1.05)),
        "cost_10pct": float(np.log(1.10)),
        "cost_20pct": float(np.log(1.20)),
        "speedup_10pct": float(np.log(0.90)),
    }
    rows: list[dict[str, Any]] = []
    for noise_index, noise_model in enumerate(
        ("gaussian", "empirical_residual")
    ):
        for effect_index, (effect_id, effect) in enumerate(
            effect_levels.items()
        ):
            rng = np.random.default_rng(
                int(seed + 10_000 * noise_index + 101 * effect_index)
            )
            if noise_model == "gaussian":
                train_noise = rng.normal(
                    0.0, sigma, size=(len(train_frame), n_draws)
                )
                test_noise = rng.normal(
                    0.0, sigma, size=(len(test_frame), n_draws)
                )
            else:
                train_noise = empirical_noise(
                    rng,
                    residuals,
                    train_groups,
                    train_groups,
                    n_draws,
                )
                test_noise = empirical_noise(
                    rng,
                    residuals,
                    train_groups,
                    test_groups,
                    n_draws,
                )
            y_train = (
                base_mean_train[:, None]
                + effect * jsd_train[:, None]
                + train_noise
            )
            y_test = (
                base_mean_test[:, None]
                + effect * jsd_test[:, None]
                + test_noise
            )
            base_estimates = base_inverse @ y_train
            dynamic_estimates = dynamic_inverse @ y_train
            dynamic_residuals = (
                y_train - x_dynamic @ dynamic_estimates
            )
            standard_errors = cr1_many_standard_errors(
                x_dynamic,
                dynamic_residuals,
                train_groups,
                coefficient_index,
            )
            estimates = dynamic_estimates[coefficient_index]
            t_values = estimates / np.maximum(standard_errors, 1e-12)
            if effect == 0.0:
                inference_pass = (
                    2.0
                    * stats.t.sf(np.abs(t_values), df=degrees_freedom)
                    < 0.05
                )
                sign_correct = np.ones(n_draws, dtype=bool)
            else:
                oriented_t = np.sign(effect) * t_values
                inference_pass = (
                    stats.t.sf(oriented_t, df=degrees_freedom) < 0.05
                )
                sign_correct = np.sign(estimates) == np.sign(effect)
            base_prediction = x_base_test @ base_estimates
            dynamic_prediction = x_dynamic_test @ dynamic_estimates
            base_squared_error = (y_test - base_prediction) ** 2
            dynamic_squared_error = (y_test - dynamic_prediction) ** 2
            heldout_improves = (
                np.mean(
                    base_squared_error - dynamic_squared_error, axis=0
                )
                > 0.0
            )
            subject_wins = np.zeros(n_draws, dtype=int)
            for group in np.unique(test_groups):
                mask = test_groups == group
                subject_wins += (
                    np.mean(
                        base_squared_error[mask]
                        - dynamic_squared_error[mask],
                        axis=0,
                    )
                    > 0.0
                )
            majority_improves = subject_wins >= 4
            full_gate = (
                inference_pass
                & sign_correct
                & heldout_improves
                & majority_improves
            )
            critical = float(
                stats.t.ppf(0.975, df=degrees_freedom)
            )
            coverage = (
                (estimates - critical * standard_errors <= effect)
                & (effect <= estimates + critical * standard_errors)
            )
            rows.append(
                {
                    "noise_model": noise_model,
                    "effect_id": effect_id,
                    "draw_n": int(n_draws),
                    "true_log_effect_per_jsd_sd": effect,
                    "true_percent_effect_per_jsd_sd": float(
                        100.0 * np.expm1(effect)
                    ),
                    "mean_estimated_log_effect": float(
                        np.mean(estimates)
                    ),
                    "estimation_bias": float(
                        np.mean(estimates) - effect
                    ),
                    "cluster_95pct_coverage": float(
                        np.mean(coverage)
                    ),
                    "inference_pass_rate": float(
                        np.mean(inference_pass & sign_correct)
                    ),
                    "heldout_improvement_rate": float(
                        np.mean(heldout_improves)
                    ),
                    "majority_subject_improvement_rate": float(
                        np.mean(majority_improves)
                    ),
                    "full_gate_rate": float(np.mean(full_gate)),
                    "baseline_sigma_log_rt": sigma,
                    "interpretation": (
                        "false-positive rate"
                        if effect == 0.0
                        else "conditional recovery rate"
                    ),
                }
            )
    return rows


def cross_modal_summary(
    primary_subject_rows: pd.DataFrame,
) -> dict[str, Any]:
    pearson = stats.pearsonr(
        primary_subject_rows["b0_minus_d0_test_choice_brier"],
        primary_subject_rows["baseline_minus_dynamic_mse"],
    )
    spearman = stats.spearmanr(
        primary_subject_rows["b0_minus_d0_test_choice_brier"],
        primary_subject_rows["baseline_minus_dynamic_mse"],
    )
    return {
        "subject_n": int(len(primary_subject_rows)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def make_decision(
    comparisons: pd.DataFrame,
    stability: pd.DataFrame,
    recovery: pd.DataFrame,
) -> dict[str, Any]:
    primary = comparisons[
        (comparisons["source_id"] == "main_r128")
        & (comparisons["analysis_id"] == "primary")
    ].iloc[0]
    seed_rows = comparisons[
        comparisons["analysis_id"].eq("primary")
    ].copy()
    independent = stability[
        stability["comparison"].eq("independent_seed")
    ].iloc[0]
    directional_cost_pass = bool(
        primary["cluster_ci_lower"] > 0.0
        and primary["wild_cluster_p_positive_cost"] < 0.05
    )
    predictive_pass = bool(
        primary["bootstrap_dmse_ci_lower"] > 0.0
        and primary["dynamic_subject_win_count"] >= 5
    )
    numerical_stability_pass = bool(
        independent["jsd_pearson_r"] >= 0.90
        and len(
            set(
                np.sign(
                    seed_rows[
                        "dynamic_coefficient_per_train_sd"
                    ].to_numpy(dtype=float)
                )
            )
        )
        == 1
        and np.all(
            seed_rows[
                "mean_subject_baseline_minus_dynamic_mse"
            ].to_numpy(dtype=float)
            > 0.0
        )
    )
    recovery_10 = recovery[
        recovery["effect_id"].eq("cost_10pct")
    ]
    recovery_null = recovery[recovery["effect_id"].eq("null")]
    recovery_pass = bool(
        np.all(recovery_10["full_gate_rate"] >= 0.80)
        and np.all(recovery_null["full_gate_rate"] <= 0.10)
    )
    agnostic_association = bool(
        primary["wild_cluster_p_two_sided"] < 0.05
        and primary["dynamic_rmse_log_rt"]
        < primary["baseline_rmse_log_rt"]
    )
    continue_dynamic = bool(
        directional_cost_pass
        and predictive_pass
        and numerical_stability_pass
        and recovery_pass
    )
    return {
        "decision": (
            "CONTINUE_D0_CONFIRMATORY"
            if continue_dynamic
            else "STOP_D0_CONFIRMATORY_RETAIN_B0"
        ),
        "continue_dynamic": continue_dynamic,
        "directional_switch_cost_gate": directional_cost_pass,
        "heldout_generalization_gate": predictive_pass,
        "particle_seed_stability_gate": numerical_stability_pass,
        "conditional_recovery_gate": recovery_pass,
        "agnostic_two_sided_association": agnostic_association,
        "decision_rule": (
            "Continue only if the frozen D0 disagreement has the preregisterable "
            "positive switch-cost direction, improves equal-subject held-out MSE "
            "with a positive bootstrap lower bound and at least 5/7 subject wins, "
            "is particle/seed stable, and a 10% RT effect passes conditional "
            "recovery with controlled null false positives."
        ),
        "interpretation": (
            "The RT data do not rescue D0 as a confirmatory cognitive mechanism. "
            "Any two-sided association is exploratory because D0 currently has "
            "no signed RT emission and subject-level held-out effects are "
            "heterogeneous."
            if not continue_dynamic
            else (
                "The frozen D0 statistic passed all external RT validation gates "
                "and can proceed to a preregistered joint choice-RT model."
            )
        ),
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    subset = frame[columns].copy()
    return subset.to_markdown(index=False, floatfmt=".4f")


def write_results_markdown(
    path: Path,
    comparisons: pd.DataFrame,
    subject_metrics: pd.DataFrame,
    stability: pd.DataFrame,
    recovery: pd.DataFrame,
    cross_modal: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    primary = comparisons[
        (comparisons["source_id"] == "main_r128")
        & (comparisons["analysis_id"] == "primary")
    ].iloc[0]
    primary_subjects = subject_metrics[
        (subject_metrics["source_id"] == "main_r128")
        & (subject_metrics["analysis_id"] == "primary")
    ]
    sensitivity = comparisons[
        comparisons["source_id"].eq("main_r128")
    ]
    lines = [
        "# Condition-1 frozen B0/D0 RT validation",
        "",
        "## Decision",
        "",
        f"**{decision['decision']}**.",
        "",
        decision["interpretation"],
        "",
        "RT was never used to select or refit B0, D0, or theta. The tested "
        "dynamic candidate was frozen at theta = 0.75 from the previous "
        "choice-only development analysis.",
        "",
        "## Primary held-out result",
        "",
        (
            "A one-training-SD increase in B0–D0 Jensen–Shannon disagreement "
            f"was associated with {primary['dynamic_effect_percent_per_train_sd']:.2f}% "
            "change in RT after baseline adjustment "
            f"(clustered 95% CI on log RT "
            f"[{primary['cluster_ci_lower']:.4f}, {primary['cluster_ci_upper']:.4f}]; "
            f"exact wild-cluster two-sided p = "
            f"{primary['wild_cluster_p_two_sided']:.4f})."
        ),
        "",
        (
            f"Held-out log-RT RMSE changed from "
            f"{primary['baseline_rmse_log_rt']:.4f} to "
            f"{primary['dynamic_rmse_log_rt']:.4f}. The dynamic increment "
            f"won in {int(primary['dynamic_subject_win_count'])}/"
            f"{int(primary['dynamic_subject_n'])} subjects; the equal-subject "
            f"mean MSE improvement had bootstrap 95% CI "
            f"[{primary['bootstrap_dmse_ci_lower']:.4f}, "
            f"{primary['bootstrap_dmse_ci_upper']:.4f}]."
        ),
        "",
        "The coefficient was negative rather than a positive reconfiguration "
        "cost, and held-out gains were not sufficiently general across subjects. "
        "Because the choice-only D0 model has no signed RT emission, the negative "
        "coefficient is not itself a formal falsification; it is also not a "
        "mechanistic validation.",
        "",
        "## Subject-level held-out evidence",
        "",
        markdown_table(
            primary_subjects,
            [
                "iSub",
                "b0_minus_d0_test_choice_brier",
                "baseline_minus_dynamic_mse",
                "dynamic_wins_mse",
            ],
        ),
        "",
        (
            "Across seven subjects, convergence between D0's held-out choice "
            "gain and its RT increment was weak/uncertain: "
            f"Pearson r = {cross_modal['pearson_r']:.3f} "
            f"(p = {cross_modal['pearson_p']:.3f}); "
            f"Spearman rho = {cross_modal['spearman_rho']:.3f} "
            f"(p = {cross_modal['spearman_p']:.3f})."
        ),
        "",
        "## Robustness and numerical stability",
        "",
        markdown_table(
            sensitivity,
            [
                "analysis_id",
                "dynamic_effect_percent_per_train_sd",
                "wild_cluster_p_two_sided",
                "baseline_rmse_log_rt",
                "dynamic_rmse_log_rt",
                "dynamic_subject_win_count",
                "bootstrap_dmse_ci_lower",
                "bootstrap_dmse_ci_upper",
            ],
        ),
        "",
        markdown_table(
            stability,
            [
                "comparison",
                "trial_n",
                "jsd_pearson_r",
                "jsd_spearman_rho",
                "jsd_mean_absolute_difference",
            ],
        ),
        "",
        "The frozen disagreement feature was highly similar across particle "
        "counts, but its independent-seed trial-level Pearson correlation was "
        f"{stability.loc[stability['comparison'].eq('independent_seed'), 'jsd_pearson_r'].iloc[0]:.3f}, "
        "below the predeclared 0.90 stability gate. The negative coefficient and "
        "aggregate RMSE direction nevertheless repeated in all three runs. This "
        "is substantial but incomplete numerical stability, and in any event "
        "numerical stability is not psychological validity.",
        "",
        "## Conditional recovery check",
        "",
        markdown_table(
            recovery,
            [
                "noise_model",
                "effect_id",
                "true_percent_effect_per_jsd_sd",
                "cluster_95pct_coverage",
                "inference_pass_rate",
                "heldout_improvement_rate",
                "full_gate_rate",
            ],
        ),
        "",
        "Recovery is conditional on the observed frozen choice-derived design "
        "matrix (including observed lag RT); it tests whether an RT emission "
        "coefficient is estimable at this sample size. It is not a full joint "
        "choice-state-RT recovery experiment.",
        "",
        "## Next action",
        "",
        "Retain B0 as the condition-1 confirmatory model and keep D0 exploratory. "
        "Do not add an RT emission or reopen capacity/three-state mechanisms from "
        "this result. A future joint choice-RT model should be attempted only "
        "after specifying the sign and functional form of its RT prediction in "
        "advance and obtaining enough held-out subjects to estimate heterogeneous "
        "effects rather than letting three subjects drive the aggregate gain.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if args.recovery_draws <= 0 or args.bootstrap_draws <= 0:
        raise ValueError("draw counts must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data, encoding="utf-8-sig")
    validate_input_data(data)

    frames: dict[str, pd.DataFrame] = {}
    all_qc_rows: list[dict[str, Any]] = []
    for source in SOURCES:
        frame, qc_rows = build_source_frame(
            data,
            source,
            theta=args.theta,
            qc_threshold=args.qc_threshold,
        )
        frames[source.source_id] = frame
        all_qc_rows.extend(qc_rows)

    comparison_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    scored_frames: list[pd.DataFrame] = []
    for source in SOURCES:
        specs = (
            ANALYSES
            if source.source_id == "main_r128"
            else (ANALYSES[0],)
        )
        for analysis_index, analysis in enumerate(specs):
            summary, subjects, scored = run_model_comparison(
                frames[source.source_id],
                source,
                analysis,
                bootstrap_draws=args.bootstrap_draws,
                seed=args.seed + 97 * analysis_index,
            )
            comparison_rows.append(summary)
            subject_rows.extend(subjects)
            scored_frames.append(scored)

    comparisons = pd.DataFrame(comparison_rows)
    subject_metrics = pd.DataFrame(subject_rows)
    scored_trials = pd.concat(scored_frames, ignore_index=True)
    stability = pd.DataFrame(seed_feature_stability(frames))
    recovery = pd.DataFrame(
        conditional_rt_recovery(
            frames["main_r128"],
            n_draws=args.recovery_draws,
            seed=args.seed,
        )
    )
    primary_subjects = subject_metrics[
        (subject_metrics["source_id"] == "main_r128")
        & (subject_metrics["analysis_id"] == "primary")
    ].copy()
    cross_modal = cross_modal_summary(primary_subjects)
    decision = make_decision(comparisons, stability, recovery)
    data_quality = build_data_quality_summary(
        data,
        frames["main_r128"],
        all_qc_rows,
    )

    feature_columns = [
        "source_id",
        "particle_count",
        "theta",
        *KEY_COLUMNS,
        "choice",
        "feedback",
        "ambiguous",
        "choRT",
        "log_rt",
        "lag_log_rt",
        "prev_error",
        "current_error",
        "subject_trial",
        "block_position",
        "valid_mask",
        "train_mask",
        "test_mask",
        "heldout_eligible",
        "rt_qc_keep",
        "rt_robust_z",
        "b0_prob_choice1",
        "d0_prob_choice1",
        "b0_entropy",
        "d0_entropy",
        "jsd",
        "total_variation",
        "b0_choice_brier_trial",
        "d0_choice_brier_trial",
    ]
    trial_features = pd.concat(
        [frames[source.source_id][feature_columns] for source in SOURCES],
        ignore_index=True,
    )
    trial_features.to_csv(
        args.output_dir / "trial_features.csv", index=False
    )
    pd.DataFrame(all_qc_rows).to_csv(
        args.output_dir / "rt_qc_by_subject.csv", index=False
    )
    comparisons.to_csv(
        args.output_dir / "model_comparison.csv", index=False
    )
    subject_metrics.to_csv(
        args.output_dir / "subject_heldout_metrics.csv", index=False
    )
    scored_trials.to_csv(
        args.output_dir / "heldout_trial_predictions.csv", index=False
    )
    stability.to_csv(
        args.output_dir / "seed_particle_stability.csv", index=False
    )
    recovery.to_csv(
        args.output_dir / "conditional_rt_recovery.csv", index=False
    )
    write_json(args.output_dir / "data_quality.json", data_quality)
    write_json(args.output_dir / "cross_modal_summary.json", cross_modal)
    write_json(args.output_dir / "decision.json", decision)
    write_json(
        args.output_dir / "manifest.json",
        {
            "result_type": "cond1_frozen_b0_d0_rt_validation",
            "data": str(args.data.relative_to(ROOT)),
            "theta": float(args.theta),
            "subjects": list(ALL_SUBJECTS),
            "heldout_subjects": list(HELDOUT_SUBJECTS),
            "sources": [asdict(source) for source in SOURCES],
            "analyses": [asdict(analysis) for analysis in ANALYSES],
            "qc_threshold": float(args.qc_threshold),
            "bootstrap_draws": int(args.bootstrap_draws),
            "recovery_draws": int(args.recovery_draws),
            "seed": int(args.seed),
            "choice_model_frozen": True,
            "rt_used_for_choice_selection": False,
            "primary_split": "last block per heldout-eligible subject",
            "first_trial_of_each_block_excluded_for_lag_rt": True,
            "recovery_scope": "conditional RT emission, not full joint choice-state-RT",
        },
    )
    write_results_markdown(
        args.output_dir / "RESULTS.md",
        comparisons,
        subject_metrics,
        stability,
        recovery,
        cross_modal,
        decision,
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    primary = comparisons[
        (comparisons["source_id"] == "main_r128")
        & (comparisons["analysis_id"] == "primary")
    ].iloc[0]
    print(
        "Primary: beta={:.6f}, wild two-sided p={:.4f}, "
        "held-out RMSE {:.6f}->{:.6f}, wins={}/{}".format(
            primary["dynamic_coefficient_per_train_sd"],
            primary["wild_cluster_p_two_sided"],
            primary["baseline_rmse_log_rt"],
            primary["dynamic_rmse_log_rt"],
            int(primary["dynamic_subject_win_count"]),
            int(primary["dynamic_subject_n"]),
        )
    )
    print(f"Wrote results to {args.output_dir}")


if __name__ == "__main__":
    main()
