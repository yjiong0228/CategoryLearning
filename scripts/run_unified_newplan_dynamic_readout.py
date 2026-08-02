#!/usr/bin/env python3
"""Screen a pre-specified time-varying decision readout after the core gate.

The frozen R0 probabilities contain the rule belief recursion but no fitted
choice sensitivity.  This script fits

    log(kappa_s,t) = a_s + b_g * x_s,t,

using training trials only, where x runs from zero to one over the training
prefix and g is either global or condition-specific.  The fitted trajectory is
then extrapolated into the frozen last-block holdout.  A matched calibration of
the already-fitted NR2 probabilities is included as a conservative fairness
check.  It is a diagnostic screen, not a hierarchical posterior or recovery
analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE = ROOT / "results/zhuran/unified_newplan/core_sobol512_20260802"
DEFAULT_OUTPUT = ROOT / "results/zhuran/unified_newplan/dynamic_readout_20260802"
LOG_KAPPA_MIN = math.log(0.01)
LOG_KAPPA_MAX = math.log(20.0)
SLOPE_BOUNDS = (-4.0, 4.0)
EPS = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), EPS, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def score(probabilities: np.ndarray, choices: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    rows = np.flatnonzero(mask)
    p = probabilities[rows]
    y = choices[rows]
    observed = np.clip(p[np.arange(len(rows)), y], EPS, 1.0)
    one_hot = np.eye(p.shape[1])[y]
    return {
        "n_trials": int(len(rows)),
        "nll": float(-np.log(observed).sum()),
        "nll_per_trial": float(-np.log(observed).mean()),
        "brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "accuracy": float(np.mean(np.argmax(p, axis=1) == y)),
        "mean_confidence": float(np.mean(np.max(p, axis=1))),
        "mean_entropy": float(np.mean(entropy(p))),
    }


def load_subjects(core: Path) -> list[dict[str, Any]]:
    subjects = []
    for path in sorted((core / "subject_predictions").glob("subject_*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            holdout = archive["holdout_mask"].astype(bool)
            train_rows = np.flatnonzero(~holdout)
            if not len(train_rows):
                raise ValueError(f"subject archive has no training rows: {path}")
            if not np.all(~holdout[: train_rows[-1] + 1]):
                raise ValueError(f"training trials are not a temporal prefix: {path}")
            parameters = json.loads(str(archive["parameters_json"].item()))
            n_trials = len(holdout)
            subjects.append(
                {
                    "subject_id": int(archive["subject_id"]),
                    "condition": int(archive["condition"]),
                    "choices": archive["choice"].astype(np.int64),
                    "feedback": archive["feedback"].astype(float),
                    "category": archive["category"].astype(np.int64),
                    "holdout": holdout,
                    "practice": np.arange(n_trials, dtype=float) / max(1, int(train_rows[-1])),
                    "log_R0": np.log(np.clip(archive["p_R0"].astype(float), EPS, 1.0)),
                    "p_R0K": archive["p_R0K"].astype(float),
                    "log_NR2": np.log(np.clip(archive["p_NR2"].astype(float), EPS, 1.0)),
                    "parameters": parameters,
                    "iSession": archive["iSession"].astype(np.int32),
                    "iBlock": archive["iBlock"].astype(np.int32),
                    "iTrial": archive["iTrial"].astype(np.int32),
                }
            )
    if len(subjects) != 96:
        raise ValueError(f"expected 96 completed subject archives, found {len(subjects)}")
    return subjects


def probabilities_from_readout(
    log_base_probability: np.ndarray,
    practice: np.ndarray,
    intercept: float,
    slope: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_log_kappa = float(intercept) + float(slope) * np.asarray(practice, dtype=float)
    log_kappa = np.clip(raw_log_kappa, LOG_KAPPA_MIN, LOG_KAPPA_MAX)
    kappa = np.exp(log_kappa)
    logits = kappa[:, None] * log_base_probability
    logits -= logsumexp(logits, axis=1)[:, None]
    return np.exp(logits), kappa, raw_log_kappa


def _loss_gradient(
    parameters: np.ndarray,
    subjects: list[dict[str, Any]],
    base_model: str,
    group_codes: np.ndarray,
    n_groups: int,
) -> tuple[float, np.ndarray]:
    n_subjects = len(subjects)
    intercepts = parameters[:n_subjects]
    slopes = parameters[n_subjects:]
    gradient = np.zeros_like(parameters)
    loss = 0.0
    for index, subject in enumerate(subjects):
        mask = ~subject["holdout"]
        x = subject["practice"][mask]
        log_probability = subject[f"log_{base_model}"][mask]
        choices = subject["choices"][mask]
        group = int(group_codes[index])
        raw_log_kappa = intercepts[index] + slopes[group] * x
        log_kappa = np.clip(raw_log_kappa, LOG_KAPPA_MIN, LOG_KAPPA_MAX)
        kappa = np.exp(log_kappa)
        logits = kappa[:, None] * log_probability
        normalizer = logsumexp(logits, axis=1)
        observed_log_probability = log_probability[np.arange(len(choices)), choices]
        loss += float(np.sum(-kappa * observed_log_probability + normalizer))

        calibrated = np.exp(logits - normalizer[:, None])
        derivative = kappa * (
            -observed_log_probability + np.sum(calibrated * log_probability, axis=1)
        )
        derivative *= (
            (raw_log_kappa > LOG_KAPPA_MIN) & (raw_log_kappa < LOG_KAPPA_MAX)
        )
        gradient[index] = float(np.sum(derivative))
        gradient[n_subjects + group] += float(np.sum(derivative * x))
    return loss, gradient


def _observed_hessian(
    parameters: np.ndarray,
    subjects: list[dict[str, Any]],
    base_model: str,
    group_codes: np.ndarray,
    n_groups: int,
) -> np.ndarray:
    """Observed training Hessian for intercept/slope Wald diagnostics."""

    n_subjects = len(subjects)
    intercepts = parameters[:n_subjects]
    slopes = parameters[n_subjects:]
    hessian = np.zeros((n_subjects + n_groups, n_subjects + n_groups), dtype=float)
    for index, subject in enumerate(subjects):
        mask = ~subject["holdout"]
        x = subject["practice"][mask]
        log_probability = subject[f"log_{base_model}"][mask]
        choices = subject["choices"][mask]
        group = int(group_codes[index])
        raw_log_kappa = intercepts[index] + slopes[group] * x
        log_kappa = np.clip(raw_log_kappa, LOG_KAPPA_MIN, LOG_KAPPA_MAX)
        kappa = np.exp(log_kappa)
        logits = kappa[:, None] * log_probability
        logits -= logsumexp(logits, axis=1)[:, None]
        calibrated = np.exp(logits)
        observed = log_probability[np.arange(len(choices)), choices]
        mean_log_probability = np.sum(calibrated * log_probability, axis=1)
        variance_log_probability = np.sum(
            calibrated * (log_probability - mean_log_probability[:, None]) ** 2,
            axis=1,
        )
        curvature = (
            kappa * (mean_log_probability - observed)
            + kappa**2 * variance_log_probability
        )
        curvature *= (
            (raw_log_kappa > LOG_KAPPA_MIN) & (raw_log_kappa < LOG_KAPPA_MAX)
        )
        slope_index = n_subjects + group
        hessian[index, index] += float(np.sum(curvature))
        cross = float(np.sum(curvature * x))
        hessian[index, slope_index] += cross
        hessian[slope_index, index] += cross
        hessian[slope_index, slope_index] += float(np.sum(curvature * x**2))
    return hessian


def fit_shared_readout(
    subjects: list[dict[str, Any]],
    base_model: str,
    grouping: str,
) -> dict[str, Any]:
    if grouping == "global":
        group_labels = ["all"]
        group_codes = np.zeros(len(subjects), dtype=int)
    elif grouping == "condition":
        group_labels = ["1", "2", "3"]
        group_codes = np.asarray([subject["condition"] - 1 for subject in subjects], dtype=int)
    else:
        raise ValueError(f"unknown grouping: {grouping}")
    n_groups = len(group_labels)

    if base_model == "R0":
        initial_intercepts = np.asarray(
            [math.log(subject["parameters"]["R0K"]["sensitivity"]) for subject in subjects]
        )
    elif base_model == "NR2":
        initial_intercepts = np.zeros(len(subjects), dtype=float)
    else:
        raise ValueError(f"unsupported base model: {base_model}")

    fits = []
    for intercept_shift in (-0.5, 0.0, 0.5):
        for slope_start in (-0.5, 0.0, 1.0, 2.0):
            start = np.concatenate(
                [
                    np.clip(
                        initial_intercepts + intercept_shift,
                        LOG_KAPPA_MIN,
                        LOG_KAPPA_MAX,
                    ),
                    np.full(n_groups, slope_start, dtype=float),
                ]
            )
            fit = minimize(
                lambda values: _loss_gradient(
                    values, subjects, base_model, group_codes, n_groups
                )[0],
                start,
                jac=lambda values: _loss_gradient(
                    values, subjects, base_model, group_codes, n_groups
                )[1],
                method="L-BFGS-B",
                bounds=[(LOG_KAPPA_MIN, LOG_KAPPA_MAX)] * len(subjects)
                + [SLOPE_BOUNDS] * n_groups,
            )
            fits.append(fit)
    best = min(fits, key=lambda result: float(result.fun))
    values = np.asarray(best.x, dtype=float)
    slopes = values[len(subjects) :]
    hessian = _observed_hessian(
        values, subjects, base_model, group_codes, n_groups
    )
    covariance = np.linalg.pinv(hessian, hermitian=True)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    minimum_hessian_eigenvalue = float(np.linalg.eigvalsh(hessian).min())
    predictions = {}
    kappa = {}
    raw_log_kappa = {}
    for index, subject in enumerate(subjects):
        probability, subject_kappa, subject_raw = probabilities_from_readout(
            subject[f"log_{base_model}"],
            subject["practice"],
            values[index],
            slopes[group_codes[index]],
        )
        subject_id = subject["subject_id"]
        predictions[subject_id] = probability
        kappa[subject_id] = subject_kappa
        raw_log_kappa[subject_id] = subject_raw
    return {
        "base_model": base_model,
        "grouping": grouping,
        "group_labels": group_labels,
        "group_codes": group_codes,
        "intercepts": values[: len(subjects)],
        "slopes": slopes,
        "standard_errors": standard_errors,
        "minimum_hessian_eigenvalue": minimum_hessian_eigenvalue,
        "predictions": predictions,
        "kappa": kappa,
        "raw_log_kappa": raw_log_kappa,
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "train_nll": float(best.fun),
        "n_starts": len(fits),
        "n_same_optimal_region": int(
            sum(abs(float(result.fun) - float(best.fun)) <= 1e-4 for result in fits)
        ),
    }


def fit_individual_readouts(
    subjects: list[dict[str, Any]],
    base_model: str,
    shared_fit: dict[str, Any],
) -> dict[str, Any]:
    predictions = {}
    kappa = {}
    intercepts = []
    slopes = []
    diagnostics = []
    for index, subject in enumerate(subjects):
        group = int(shared_fit["group_codes"][index])

        def objective_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
            mask = ~subject["holdout"]
            x = subject["practice"][mask]
            log_probability = subject[f"log_{base_model}"][mask]
            choices = subject["choices"][mask]
            raw = values[0] + values[1] * x
            log_kappa = np.clip(raw, LOG_KAPPA_MIN, LOG_KAPPA_MAX)
            subject_kappa = np.exp(log_kappa)
            logits = subject_kappa[:, None] * log_probability
            normalizer = logsumexp(logits, axis=1)
            observed = log_probability[np.arange(len(choices)), choices]
            loss = float(np.sum(-subject_kappa * observed + normalizer))
            calibrated = np.exp(logits - normalizer[:, None])
            derivative = subject_kappa * (
                -observed + np.sum(calibrated * log_probability, axis=1)
            )
            derivative *= (raw > LOG_KAPPA_MIN) & (raw < LOG_KAPPA_MAX)
            gradient = np.asarray([np.sum(derivative), np.sum(derivative * x)])
            return loss, gradient

        shared_start = np.asarray(
            [
                shared_fit["intercepts"][index],
                shared_fit["slopes"][group],
            ]
        )
        starts = [
            shared_start,
            np.asarray([shared_start[0], 0.0]),
            np.asarray([shared_start[0] - 0.5, 1.0]),
            np.asarray([shared_start[0] + 0.5, 2.0]),
        ]
        fits = [
            minimize(
                lambda values: objective_gradient(values)[0],
                start,
                jac=lambda values: objective_gradient(values)[1],
                method="L-BFGS-B",
                bounds=[(LOG_KAPPA_MIN, LOG_KAPPA_MAX), SLOPE_BOUNDS],
            )
            for start in starts
        ]
        best = min(fits, key=lambda result: float(result.fun))
        probability, subject_kappa, _ = probabilities_from_readout(
            subject[f"log_{base_model}"],
            subject["practice"],
            best.x[0],
            best.x[1],
        )
        subject_id = subject["subject_id"]
        predictions[subject_id] = probability
        kappa[subject_id] = subject_kappa
        intercepts.append(float(best.x[0]))
        slopes.append(float(best.x[1]))
        diagnostics.append(
            {
                "subject_id": subject_id,
                "optimizer_success": bool(best.success),
                "train_nll": float(best.fun),
                "n_same_optimal_region": int(
                    sum(abs(float(result.fun) - float(best.fun)) <= 1e-5 for result in fits)
                ),
            }
        )
    return {
        "base_model": base_model,
        "grouping": "individual",
        "intercepts": np.asarray(intercepts),
        "slopes": np.asarray(slopes),
        "predictions": predictions,
        "kappa": kappa,
        "diagnostics": diagnostics,
    }


def bootstrap_interval(values: np.ndarray, seed: int, n_boot: int = 10000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def compare_models(metrics: pd.DataFrame, seed: int) -> pd.DataFrame:
    specifications = [
        ("R0KT_GLOBAL", "R0K", "time_trend_increment_global"),
        ("R0KT_CONDITION", "R0K", "time_trend_increment_condition"),
        ("R0KT_INDIVIDUAL", "R0K", "time_trend_increment_individual"),
        ("R0KT_GLOBAL", "NR2", "dynamic_rule_vs_NR2"),
        ("R0KT_CONDITION", "NR2", "dynamic_rule_condition_vs_NR2"),
        ("R0KT_INDIVIDUAL", "NR2", "dynamic_rule_individual_vs_NR2"),
        ("R0KT_GLOBAL", "NR2T_GLOBAL", "matched_dynamic_gate_global"),
        ("R0KT_CONDITION", "NR2T_CONDITION", "matched_dynamic_gate_condition"),
        ("R0KT_INDIVIDUAL", "NR2T_INDIVIDUAL", "matched_dynamic_gate_individual"),
        ("NR2T_GLOBAL", "NR2", "NR2_time_calibration_global"),
        ("NR2T_CONDITION", "NR2", "NR2_time_calibration_condition"),
        ("R0KT_CONDITION", "R0KT_GLOBAL", "condition_slope_increment"),
        ("R0KT_INDIVIDUAL", "R0KT_CONDITION", "individual_slope_increment"),
    ]
    holdout = metrics[metrics["segment"] == "holdout"]
    rows = []
    for candidate, reference, label in specifications:
        for condition_label, group in [
            (str(condition), holdout[holdout["condition"] == condition])
            for condition in (1, 2, 3)
        ] + [("all", holdout)]:
            candidate_rows = group[group["model"] == candidate]
            reference_rows = group[group["model"] == reference]
            paired = candidate_rows.merge(
                reference_rows,
                on=["subject_id", "condition", "segment"],
                suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            delta = (
                paired["nll_per_trial_reference"].to_numpy()
                - paired["nll_per_trial_candidate"].to_numpy()
            )
            if not len(delta):
                continue
            lower, upper = bootstrap_interval(
                delta,
                seed + sum(ord(character) for character in f"{label}{condition_label}"),
            )
            nonzero = delta[~np.isclose(delta, 0.0)]
            try:
                p_value = float(wilcoxon(nonzero).pvalue) if len(nonzero) else 1.0
            except ValueError:
                p_value = float("nan")
            rows.append(
                {
                    "comparison": label,
                    "candidate": candidate,
                    "reference": reference,
                    "condition": condition_label,
                    "n_subjects": int(len(delta)),
                    "mean_delta_nll_per_trial": float(np.mean(delta)),
                    "median_delta_nll_per_trial": float(np.median(delta)),
                    "bootstrap_mean_ci_low": lower,
                    "bootstrap_mean_ci_high": upper,
                    "n_improved": int(np.sum(delta > 0)),
                    "proportion_improved": float(np.mean(delta > 0)),
                    "wilcoxon_p_uncorrected": p_value,
                }
            )
    return pd.DataFrame(rows)


def calibration_rows(
    subject: dict[str, Any], model: str, probabilities: np.ndarray
) -> list[dict[str, Any]]:
    mask = subject["holdout"]
    p = probabilities[mask]
    choices = subject["choices"][mask]
    observed = np.eye(p.shape[1])[choices]
    flat_p = p.ravel()
    flat_y = observed.ravel()
    bins = np.minimum((flat_p * 10).astype(int), 9)
    rows = []
    for bin_index in range(10):
        selected = bins == bin_index
        if selected.any():
            rows.append(
                {
                    "subject_id": subject["subject_id"],
                    "condition": subject["condition"],
                    "model": model,
                    "bin": bin_index,
                    "n": int(selected.sum()),
                    "sum_probability": float(flat_p[selected].sum()),
                    "sum_observed": float(flat_y[selected].sum()),
                }
            )
    return rows


def trial_diagnostics(
    subjects: list[dict[str, Any]], model_predictions: dict[str, dict[int, np.ndarray]]
) -> pd.DataFrame:
    rows = []
    for subject in subjects:
        mask = subject["holdout"]
        for trial in np.flatnonzero(mask):
            choice = subject["choices"][trial]
            base = {
                "subject_id": subject["subject_id"],
                "condition": subject["condition"],
                "trial_index": int(trial),
                "iSession": int(subject["iSession"][trial]),
                "iBlock": int(subject["iBlock"][trial]),
                "iTrial": int(subject["iTrial"][trial]),
                "choice": int(choice + 1),
                "category": int(subject["category"][trial] + 1),
                "feedback": float(subject["feedback"][trial]),
            }
            for model in ("R0K", "R0KT_GLOBAL", "NR2", "NR2T_GLOBAL"):
                probability = model_predictions[model][subject["subject_id"]][trial]
                base[f"observed_p_{model}"] = float(probability[choice])
                base[f"argmax_hit_{model}"] = bool(np.argmax(probability) == choice)
            base["log_score_advantage_R0K_vs_NR2"] = float(
                math.log(max(base["observed_p_R0K"], EPS))
                - math.log(max(base["observed_p_NR2"], EPS))
            )
            base["log_score_advantage_R0KT_vs_NR2T"] = float(
                math.log(max(base["observed_p_R0KT_GLOBAL"], EPS))
                - math.log(max(base["observed_p_NR2T_GLOBAL"], EPS))
            )
            rows.append(base)
    return pd.DataFrame(rows)


def residual_summary(trials: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparison, column in (
        ("R0K_vs_NR2", "log_score_advantage_R0K_vs_NR2"),
        ("R0KT_vs_NR2T", "log_score_advantage_R0KT_vs_NR2T"),
    ):
        for condition_label, group in [
            (str(condition), trials[trials["condition"] == condition])
            for condition in (1, 2, 3)
        ] + [("all", trials)]:
            values = np.sort(group[column].to_numpy())
            rows.append(
                {
                    "comparison": comparison,
                    "condition": condition_label,
                    "n_trials": int(len(values)),
                    "sum_log_score_advantage": float(values.sum()),
                    "mean_log_score_advantage": float(values.mean()),
                    "worst_1pct_sum": float(values[: max(1, len(values) // 100)].sum()),
                    "worst_5pct_sum": float(values[: max(1, len(values) // 20)].sum()),
                    "q01": float(np.quantile(values, 0.01)),
                    "median": float(np.median(values)),
                    "q99": float(np.quantile(values, 0.99)),
                }
            )
    return pd.DataFrame(rows)


def render_report(
    path: Path,
    comparisons: pd.DataFrame,
    parameters: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    relevant = comparisons[
        comparisons["comparison"].isin(
            [
                "time_trend_increment_global",
                "dynamic_rule_vs_NR2",
                "matched_dynamic_gate_global",
                "NR2_time_calibration_global",
                "condition_slope_increment",
                "individual_slope_increment",
            ]
        )
    ]
    slopes = parameters[
        (parameters["model"] == "R0KT_GLOBAL") & (parameters["subject_id"] == -1)
    ]
    global_slope = float(slopes["slope"].iloc[0])
    global_gate = comparisons[
        (comparisons["comparison"] == "matched_dynamic_gate_global")
        & (comparisons["condition"] == "all")
    ].iloc[0]
    vs_nr2 = comparisons[
        (comparisons["comparison"] == "dynamic_rule_vs_NR2")
        & (comparisons["condition"] == "all")
    ].iloc[0]
    increment = comparisons[
        (comparisons["comparison"] == "time_trend_increment_global")
        & (comparisons["condition"] == "all")
    ].iloc[0]
    accuracy = (
        metrics[(metrics["segment"] == "holdout") & (metrics["model"].isin(["R0KT_GLOBAL", "NR2T_GLOBAL"]))]
        .groupby("model")["accuracy"]
        .mean()
    )

    lines = [
        "# Unified new-plan dynamic-readout screen",
        "",
        "> Status: exploratory, training-only mechanism repair after the frozen static representation gate failed. The candidate is not accepted until parameter/model recovery and hierarchical posterior validation are complete.",
        "",
        "## Result",
        "",
        f"The global practice slope was b={global_slope:.6f}, implying a multiplicative increase exp(b)={math.exp(global_slope):.3f} in decision sensitivity from the first trial to the end of the training prefix, before subject-specific temporal extrapolation.",
        "",
        f"Relative to static R0K, R0KT_GLOBAL improved held-out NLL/trial by {increment.mean_delta_nll_per_trial:.6f} on average (95% subject-bootstrap CI [{increment.bootstrap_mean_ci_low:.6f}, {increment.bootstrap_mean_ci_high:.6f}]; {int(increment.n_improved)}/{int(increment.n_subjects)} subjects).",
        "",
        f"Relative to the original NR2, its mean advantage was {vs_nr2.mean_delta_nll_per_trial:.6f} (95% CI [{vs_nr2.bootstrap_mean_ci_low:.6f}, {vs_nr2.bootstrap_mean_ci_high:.6f}]; {int(vs_nr2.n_improved)}/{int(vs_nr2.n_subjects)} subjects).",
        "",
        f"After applying the same training-only time calibration to frozen NR2 probabilities, the matched gate remained {global_gate.mean_delta_nll_per_trial:.6f} (95% CI [{global_gate.bootstrap_mean_ci_low:.6f}, {global_gate.bootstrap_mean_ci_high:.6f}]; {int(global_gate.n_improved)}/{int(global_gate.n_subjects)} subjects). Mean held-out argmax accuracy was {accuracy['R0KT_GLOBAL']:.4f} for dynamic rule and {accuracy['NR2T_GLOBAL']:.4f} for calibrated NR2.",
        "",
        "## Paired comparisons",
        "",
        "Positive ΔNLL/trial favors the candidate. Bootstrap intervals condition on the training-fitted shared slope; a later hierarchical posterior must propagate its uncertainty.",
        "",
        "| Comparison | Condition | Mean ΔNLL/trial | 95% CI | Improved |",
        "|:--|:--|--:|:--|:--|",
    ]
    for row in relevant.sort_values(["comparison", "condition"]).itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {row.condition} | {row.mean_delta_nll_per_trial:.6f} | "
            f"[{row.bootstrap_mean_ci_low:.6f}, {row.bootstrap_mean_ci_high:.6f}] | "
            f"{int(row.n_improved)}/{int(row.n_subjects)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The static conclusion is unchanged: R0--R3 do not pass the planned representation gate. The new result instead identifies a specific model misspecification: a stationary choice readout averages early noisy responding and later stable rule use, then underpredicts confidence in the final block. A smooth practice-dependent readout was named in the analysis plan, but testing it after observing the static failure makes this an adaptive/exploratory result.",
            "",
            "The matched NR2 calibration is deliberately conservative but not a full joint refit of feature-RL learning and time-varying sensitivity. Thus the screen justifies recovery and a joint/hierarchical follow-up; it does not yet prove a rule representation or a unique psychological resource mechanism.",
            "",
            "Condition-specific and individual slopes are retained only as complexity checks. They should be rejected unless their direct held-out increments over the global/shared alternative are clear. No RT, oral-report, or autonomous-generation evidence is used in this screen.",
            "",
            "## Artifacts",
            "",
            "- `subject_metrics.csv`, `model_comparisons.csv`: frozen temporal-holdout scores.",
            "- `parameters.csv`, `optimizer_diagnostics.json`: training-only readout fits and convergence.",
            "- `trial_diagnostics.csv`, `residual_summary.csv`, `calibration.csv`: localization of the static failure and repair.",
            "- `subject_predictions/`: probabilities and κ trajectories for downstream recovery/RT/oral diagnostics.",
            "- `manifest.json`: provenance and evidence boundary.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    core = args.core.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_output = output / "subject_predictions"
    prediction_output.mkdir(exist_ok=True)
    core_manifest = json.loads((core / "manifest.json").read_text(encoding="utf-8"))
    if core_manifest.get("status") != "complete" or int(core_manifest.get("sobol_points", 0)) != 512:
        raise ValueError("dynamic screen requires the complete frozen 512-point core run")
    subjects = load_subjects(core)

    fits: dict[str, dict[str, Any]] = {}
    for base_model, prefix in (("R0", "R0KT"), ("NR2", "NR2T")):
        for grouping in ("global", "condition"):
            name = f"{prefix}_{grouping.upper()}"
            print(f"[fit] {name}", flush=True)
            fits[name] = fit_shared_readout(subjects, base_model, grouping)
        print(f"[fit] {prefix}_INDIVIDUAL", flush=True)
        fits[f"{prefix}_INDIVIDUAL"] = fit_individual_readouts(
            subjects, base_model, fits[f"{prefix}_CONDITION"]
        )

    model_predictions: dict[str, dict[int, np.ndarray]] = {
        "R0K": {subject["subject_id"]: subject["p_R0K"] for subject in subjects},
        "NR2": {subject["subject_id"]: np.exp(subject["log_NR2"]) for subject in subjects},
    }
    for name, fit in fits.items():
        model_predictions[name] = fit["predictions"]

    metric_rows = []
    parameter_rows = []
    calibration_payload = []
    for index, subject in enumerate(subjects):
        subject_id = subject["subject_id"]
        for model, predictions in model_predictions.items():
            probability = predictions[subject_id]
            for segment, mask in (
                ("train", ~subject["holdout"]),
                ("holdout", subject["holdout"]),
            ):
                metric_rows.append(
                    {
                        "subject_id": subject_id,
                        "condition": subject["condition"],
                        "model": model,
                        "segment": segment,
                        **score(probability, subject["choices"], mask),
                    }
                )
            calibration_payload.extend(calibration_rows(subject, model, probability))

        output_arrays: dict[str, Any] = {
            "subject_id": np.asarray(subject_id),
            "condition": np.asarray(subject["condition"]),
            "choice": subject["choices"].astype(np.int8),
            "holdout_mask": subject["holdout"],
            "practice": subject["practice"].astype(np.float32),
        }
        for model in model_predictions:
            probability = model_predictions[model][subject_id]
            output_arrays[f"p_{model}"] = probability.astype(np.float32)
            output_arrays[f"choice_entropy_{model}"] = entropy(probability).astype(np.float32)
        for name, fit in fits.items():
            output_arrays[f"kappa_{name}"] = fit["kappa"][subject_id].astype(np.float32)
            slope = (
                fit["slopes"][index]
                if fit["grouping"] == "individual"
                else fit["slopes"][int(fit["group_codes"][index])]
            )
            parameter_rows.append(
                {
                    "model": name,
                    "subject_id": subject_id,
                    "condition": subject["condition"],
                    "intercept": float(fit["intercepts"][index]),
                    "intercept_se": float(fit["standard_errors"][index])
                    if "standard_errors" in fit
                    else float("nan"),
                    "slope": float(slope),
                    "slope_se": float(
                        fit["standard_errors"][
                            len(subjects) + int(fit["group_codes"][index])
                        ]
                    )
                    if "standard_errors" in fit
                    else float("nan"),
                    "kappa_first": float(fit["kappa"][subject_id][0]),
                    "kappa_train_end": float(
                        fit["kappa"][subject_id][np.flatnonzero(~subject["holdout"])[-1]]
                    ),
                    "kappa_holdout_end": float(fit["kappa"][subject_id][-1]),
                    "holdout_cap_hit": bool(
                        np.any(fit["raw_log_kappa"].get(subject_id, np.asarray([])) >= LOG_KAPPA_MAX)
                        if "raw_log_kappa" in fit
                        else np.any(fit["kappa"][subject_id] >= np.exp(LOG_KAPPA_MAX) - 1e-6)
                    ),
                }
            )
        atomic_savez(prediction_output / f"subject_{subject_id}.npz", **output_arrays)

    for name, fit in fits.items():
        if fit["grouping"] == "individual":
            continue
        for group_label, slope in zip(fit["group_labels"], fit["slopes"]):
            parameter_rows.append(
                {
                    "model": name,
                    "subject_id": -1,
                    "condition": group_label,
                    "intercept": float("nan"),
                    "intercept_se": float("nan"),
                    "slope": float(slope),
                    "slope_se": float(
                        fit["standard_errors"][
                            len(subjects) + fit["group_labels"].index(group_label)
                        ]
                    ),
                    "kappa_first": float("nan"),
                    "kappa_train_end": float("nan"),
                    "kappa_holdout_end": float("nan"),
                    "holdout_cap_hit": False,
                }
            )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["condition", "subject_id", "segment", "model"]
    )
    parameters = pd.DataFrame(parameter_rows).sort_values(
        ["model", "subject_id"]
    )
    comparisons = compare_models(metrics, args.seed)
    calibration_subject = pd.DataFrame(calibration_payload)
    calibration = (
        calibration_subject.groupby(["condition", "model", "bin"], as_index=False)
        .agg(
            n=("n", "sum"),
            sum_probability=("sum_probability", "sum"),
            sum_observed=("sum_observed", "sum"),
        )
    )
    calibration["mean_probability"] = calibration["sum_probability"] / calibration["n"]
    calibration["observed_frequency"] = calibration["sum_observed"] / calibration["n"]
    trials = trial_diagnostics(subjects, model_predictions)
    residuals = residual_summary(trials)

    atomic_csv(output / "subject_metrics.csv", metrics)
    atomic_csv(output / "model_comparisons.csv", comparisons)
    atomic_csv(output / "parameters.csv", parameters)
    atomic_csv(output / "calibration.csv", calibration)
    atomic_csv(output / "calibration_subject_sufficient_stats.csv", calibration_subject)
    atomic_csv(output / "trial_diagnostics.csv", trials)
    atomic_csv(output / "residual_summary.csv", residuals)
    optimizer_diagnostics = {
        name: {
            key: value
            for key, value in fit.items()
            if key
            in {
                "base_model",
                "grouping",
                "group_labels",
                "optimizer_success",
                "optimizer_message",
                "train_nll",
                "n_starts",
                "n_same_optimal_region",
                "minimum_hessian_eigenvalue",
                "diagnostics",
            }
        }
        for name, fit in fits.items()
    }
    atomic_json(output / "optimizer_diagnostics.json", optimizer_diagnostics)
    render_report(output / "RESULTS.md", comparisons, parameters, metrics)
    manifest = {
        "result_type": "unified_newplan_dynamic_readout_screen",
        "status": "complete",
        "evidence_status": "exploratory_adaptive_screen",
        "core_run": str(core),
        "core_manifest_sha256": sha256_file(core / "manifest.json"),
        "core_sobol_points": int(core_manifest["sobol_points"]),
        "data_sha256": core_manifest["data_sha256"],
        "n_subjects": len(subjects),
        "models": list(model_predictions),
        "readout_equation": "log(kappa_s,t) = intercept_s + shared_slope * normalized_training_practice",
        "kappa_bounds": [math.exp(LOG_KAPPA_MIN), math.exp(LOG_KAPPA_MAX)],
        "slope_bounds": list(SLOPE_BOUNDS),
        "holdout_use": "scoring only; no readout parameter was estimated on holdout trials",
        "fairness_control": (
            "NR2T applies the same post-hoc dynamic power calibration to frozen NR2 probabilities; "
            "it is not a full joint re-fit of feature-RL learning dynamics"
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy_version,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(output / "manifest.json", manifest)
    print(f"[done] wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
