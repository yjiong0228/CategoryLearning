#!/usr/bin/env python3
"""Targeted held-out diagnostics for FA3-M-S choice residuals and real RT."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, gammaln
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0806_dynamic_m_recovery import softmax  # noqa: E402
from scripts.run_model_0806_real_rolling import (  # noqa: E402
    candidate_grid,
    component_path,
)


DEFAULT_CONFIG = ROOT / "configs/model_0806_targeted_diagnostics.yaml"
DEFAULT_OUTPUT = (
    ROOT / "results/zhuran/model_0806_cond1/targeted_diagnostics_20260806_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("diagnostic config must be a mapping")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def atomic_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def clipped_logit(probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(values / (1.0 - values))


def bernoulli_log_density(outcome: np.ndarray, probability: np.ndarray) -> np.ndarray:
    y = np.asarray(outcome, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    return y * np.log(p) + (1.0 - y) * np.log1p(-p)


def student_log_density(
    response: np.ndarray,
    location: np.ndarray,
    scale: float,
    degrees_of_freedom: float,
) -> np.ndarray:
    y = np.asarray(response, dtype=float)
    mean = np.asarray(location, dtype=float)
    sigma = float(scale)
    nu = float(degrees_of_freedom)
    standardized = (y - mean) / sigma
    constant = (
        gammaln((nu + 1.0) / 2.0)
        - gammaln(nu / 2.0)
        - 0.5 * math.log(nu * math.pi)
        - math.log(sigma)
    )
    return constant - 0.5 * (nu + 1.0) * np.log1p(
        np.square(standardized) / nu
    )


def fit_offset_logistic(
    design: np.ndarray,
    outcome: np.ndarray,
    offset: np.ndarray,
    ridge_penalty: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = np.asarray(design, dtype=float)
    y = np.asarray(outcome, dtype=float)
    fixed = np.asarray(offset, dtype=float)
    penalty = np.ones(x.shape[1], dtype=float)
    penalty[0] = 0.0

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = fixed + x @ beta
        probability = expit(eta)
        value = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        value += 0.5 * float(ridge_penalty) * float(np.sum(penalty * beta**2))
        gradient = x.T @ (probability - y)
        gradient += float(ridge_penalty) * penalty * beta
        return value, gradient

    fit = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(x.shape[1], dtype=float),
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
    )
    return np.asarray(fit.x, dtype=float), {
        "success": bool(fit.success),
        "message": str(fit.message),
        "train_nll_penalized": float(fit.fun),
    }


def fit_student_regression(
    design: np.ndarray,
    response: np.ndarray,
    *,
    degrees_of_freedom: float,
    ridge_penalty: float,
    minimum_scale: float,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    x = np.asarray(design, dtype=float)
    y = np.asarray(response, dtype=float)
    beta_start = np.linalg.lstsq(x, y, rcond=None)[0]
    residual = y - x @ beta_start
    scale_start = max(float(np.sqrt(np.mean(residual**2))), float(minimum_scale))
    penalty = np.ones(x.shape[1], dtype=float)
    penalty[0] = 0.0

    def objective(values: np.ndarray) -> float:
        beta = values[:-1]
        scale = float(np.exp(values[-1]))
        nll = float(
            -np.sum(
                student_log_density(y, x @ beta, scale, degrees_of_freedom)
            )
        )
        return nll + 0.5 * float(ridge_penalty) * float(
            np.sum(penalty * beta**2)
        )

    bounds = [(None, None)] * x.shape[1] + [
        (math.log(float(minimum_scale)), math.log(5.0))
    ]
    starts = [
        np.concatenate([beta_start, [math.log(scale_start * multiplier)]])
        for multiplier in (0.75, 1.0, 1.5)
    ]
    fits = [
        minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        for start in starts
    ]
    converged = [fit for fit in fits if bool(fit.success)]
    best = min(converged if converged else fits, key=lambda fit: float(fit.fun))
    return np.asarray(best.x[:-1]), float(np.exp(best.x[-1])), {
        "success": bool(converged),
        "message": str(best.message),
        "converged_starts": int(len(converged)),
        "train_nll_penalized": float(best.fun),
    }


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    location = float(np.median(array))
    scale = float(1.4826 * np.median(np.abs(array - location)))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(array, ddof=0))
    return location, max(scale, 1e-6)


def standardize(
    train: np.ndarray, evaluation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    train_values = np.asarray(train, dtype=float)
    evaluation_values = np.asarray(evaluation, dtype=float)
    mean = float(np.mean(train_values))
    scale = float(np.std(train_values, ddof=0))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return (
        (train_values - mean) / scale,
        (evaluation_values - mean) / scale,
        mean,
        scale,
    )


def subject_design(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    subjects: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    identifiers = [int(value) for value in subjects]
    train_columns = [np.ones(len(train), dtype=float)]
    evaluation_columns = [np.ones(len(evaluation), dtype=float)]
    for subject in identifiers[1:]:
        train_columns.append((train["subject_id"].to_numpy() == subject).astype(float))
        evaluation_columns.append(
            (evaluation["subject_id"].to_numpy() == subject).astype(float)
        )
    return np.column_stack(train_columns), np.column_stack(evaluation_columns)


def load_family_arrays(
    rolling_output: Path,
    rolling_config: Mapping[str, Any],
    subject_id: int,
    family: str,
) -> dict[str, np.ndarray]:
    candidates = [
        value for value in candidate_grid(rolling_config) if value["family"] == family
    ]
    seeds = [int(value) for value in rolling_config["design"]["filter_seeds"]]
    primary_seed = seeds[0]
    probabilities: list[np.ndarray] = []
    correct: list[np.ndarray] = []
    replacement: list[np.ndarray] = []
    surprise: list[np.ndarray] = []
    uncertainty: list[np.ndarray] = []
    for candidate in candidates:
        for seed in seeds:
            path = component_path(
                rolling_output,
                int(subject_id),
                str(candidate["candidate_id"]),
                int(seed),
                primary_seed,
            )
            with np.load(path, allow_pickle=False) as payload:
                probabilities.append(payload["probabilities"].astype(float))
                correct.append(payload["correct_probability"].astype(float))
                replacement.append(payload["replacement_fraction"].astype(float))
                if family == "static":
                    required = {"feedback_surprise", "feedback_uncertainty"}
                    if not required.issubset(payload.files):
                        missing = sorted(required.difference(payload.files))
                        raise ValueError(
                            f"static component lacks diagnostic traces {missing}: {path}"
                        )
                    surprise.append(payload["feedback_surprise"].astype(float))
                    uncertainty.append(payload["feedback_uncertainty"].astype(float))
    result = {
        "probabilities": np.stack(probabilities),
        "correct_probability": np.stack(correct),
        "replacement_fraction": np.stack(replacement),
    }
    if family == "static":
        result["feedback_surprise"] = np.stack(surprise)
        result["feedback_uncertainty"] = np.stack(uncertainty)
    return result


def block_frozen_family_predictors(
    arrays: Mapping[str, np.ndarray],
    choices: np.ndarray,
    block_size: int,
) -> dict[str, np.ndarray]:
    probabilities = np.asarray(arrays["probabilities"], dtype=float)
    y = np.asarray(choices, dtype=int)
    component_count, n_trials, _ = probabilities.shape
    observed = probabilities[:, np.arange(n_trials), y]
    log_observed = np.log(np.clip(observed, 1e-300, 1.0))
    cumulative_log_likelihood = np.zeros(component_count, dtype=float)
    output = {
        "choice_probability": np.zeros((n_trials, 2), dtype=float),
        "correct_probability": np.zeros(n_trials, dtype=float),
        "replacement_fraction": np.zeros(n_trials, dtype=float),
    }
    diagnostic_traces = (
        "feedback_surprise",
        "feedback_uncertainty",
    )
    for trace_name in diagnostic_traces:
        if trace_name in arrays:
            output[trace_name] = np.zeros(n_trials, dtype=float)
    for start in range(0, n_trials, int(block_size)):
        stop = min(start + int(block_size), n_trials)
        block = slice(start, stop)
        weights = softmax(cumulative_log_likelihood)
        output["choice_probability"][block] = np.tensordot(
            weights, probabilities[:, block, :], axes=(0, 0)
        )
        output["correct_probability"][block] = np.tensordot(
            weights,
            np.asarray(arrays["correct_probability"])[:, block],
            axes=(0, 0),
        )
        output["replacement_fraction"][block] = np.tensordot(
            weights,
            np.asarray(arrays["replacement_fraction"])[:, block],
            axes=(0, 0),
        )
        for trace_name in diagnostic_traces:
            if trace_name in arrays:
                output[trace_name][block] = np.tensordot(
                    weights,
                    np.asarray(arrays[trace_name])[:, block],
                    axes=(0, 0),
                )
        cumulative_log_likelihood += np.sum(log_observed[:, block], axis=1)
    choice_probability = np.clip(output["choice_probability"], 1e-12, 1.0)
    output["choice_entropy"] = -np.sum(
        choice_probability * np.log(choice_probability), axis=1
    )
    return output


def build_predictor_table(config: Mapping[str, Any]) -> pd.DataFrame:
    rolling_config_path = ROOT / str(config["rolling_config"])
    rolling_config = read_yaml(rolling_config_path)
    rolling_output = ROOT / str(config["rolling_output"])
    data = pd.read_csv(ROOT / str(config["data_path"]), low_memory=False)
    data = (
        data[data["condition"] == 1]
        .sort_values(["iSub", "iSession", "iBlock", "iTrial"], kind="stable")
        .reset_index(drop=True)
    )
    block_size = int(config["design"]["block_size"])
    rows: list[pd.DataFrame] = []
    for subject_id, subject_frame in data.groupby("iSub", sort=True):
        frame = subject_frame.reset_index(drop=True).copy()
        choices = frame["choice"].to_numpy(dtype=int) - 1
        static_arrays = load_family_arrays(
            rolling_output, rolling_config, int(subject_id), "static"
        )
        dynamic_arrays = load_family_arrays(
            rolling_output, rolling_config, int(subject_id), "dynamic"
        )
        static = block_frozen_family_predictors(static_arrays, choices, block_size)
        dynamic = block_frozen_family_predictors(dynamic_arrays, choices, block_size)
        n_trials = len(frame)
        if static["correct_probability"].size != n_trials:
            raise ValueError(f"subject {subject_id} predictor length mismatch")
        result = pd.DataFrame({
            "subject_id": int(subject_id),
            "trial_index": np.arange(1, n_trials + 1, dtype=int),
            "evaluation_block": np.arange(n_trials, dtype=int) // block_size + 1,
            "block_position": np.arange(n_trials, dtype=int) % block_size + 1,
            "correct": frame["feedback"].to_numpy(dtype=float),
            "ambiguous": frame["ambiguous"].to_numpy(dtype=float),
            "log_rt": np.log(frame["choRT"].to_numpy(dtype=float)),
            "static_correct_probability": static["correct_probability"],
            "static_choice_entropy": static["choice_entropy"],
            "static_replacement_fraction": static["replacement_fraction"],
            "static_feedback_surprise": static["feedback_surprise"],
            "static_feedback_uncertainty": static["feedback_uncertainty"],
            "dynamic_correct_probability": dynamic["correct_probability"],
            "dynamic_choice_entropy": dynamic["choice_entropy"],
            "dynamic_replacement_fraction": dynamic["replacement_fraction"],
        })
        result["log_trial"] = np.log1p(result["trial_index"].to_numpy(dtype=float))
        result["log_block_position"] = np.log1p(
            result["block_position"].to_numpy(dtype=float)
        )
        result["previous_error_within"] = (
            1.0 - result.groupby("evaluation_block", sort=False)["correct"].shift(1)
        )
        result["lag_log_rt_within"] = result.groupby(
            "evaluation_block", sort=False
        )["log_rt"].shift(1)
        result["lag_surprise_within"] = result.groupby(
            "evaluation_block", sort=False
        )["static_feedback_surprise"].shift(1)
        result["lag_uncertainty_within"] = result.groupby(
            "evaluation_block", sort=False
        )["static_feedback_uncertainty"].shift(1)
        result["previous_error_all"] = 1.0 - result["correct"].shift(1)
        result["lag_surprise_all"] = result["static_feedback_surprise"].shift(1)
        result["lag_uncertainty_all"] = result[
            "static_feedback_uncertainty"
        ].shift(1)
        rows.append(result)
    table = pd.concat(rows, ignore_index=True)
    if not np.all(np.isfinite(table["log_rt"])):
        raise ValueError("condition-1 log RT contains non-finite values")
    return table


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
    train_surprise, evaluation_surprise, surprise_mean, surprise_scale = standardize(
        train[lag_column], evaluation[lag_column]
    )
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
    candidate_train = np.column_stack([baseline_train, train_surprise])
    candidate_evaluation = np.column_stack([
        baseline_evaluation, evaluation_surprise
    ])
    return baseline_train, baseline_evaluation, candidate_train, candidate_evaluation, {
        "surprise_mean": surprise_mean,
        "surprise_scale": surprise_scale,
    }


def run_choice_diagnostic(
    table: pd.DataFrame,
    config: Mapping[str, Any],
    lag_policy: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if lag_policy == "within_block":
        lag_column = "lag_surprise_within"
        previous_error_column = "previous_error_within"
    elif lag_policy == "all_transitions":
        lag_column = "lag_surprise_all"
        previous_error_column = "previous_error_all"
    else:
        raise ValueError(f"unknown lag policy: {lag_policy}")
    first_block = int(config["design"]["first_evaluation_block"])
    fold_rows: list[dict[str, Any]] = []
    trial_frames: list[pd.DataFrame] = []
    for evaluation_block in range(first_block, int(table["evaluation_block"].max()) + 1):
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
            float(config["choice_residual"]["ridge_penalty"]),
        )
        candidate_beta, candidate_diagnostics = fit_offset_logistic(
            candidate_train,
            train["correct"].to_numpy(dtype=float),
            train_offset,
            float(config["choice_residual"]["ridge_penalty"]),
        )
        baseline_probability = expit(
            evaluation_offset + baseline_evaluation @ baseline_beta
        )
        candidate_probability = expit(
            evaluation_offset + candidate_evaluation @ candidate_beta
        )
        y = evaluation["correct"].to_numpy(dtype=float)
        scored = evaluation[[
            "subject_id", "trial_index", "evaluation_block", "block_position"
        ]].copy()
        scored["lag_policy"] = lag_policy
        scored["baseline_log_density"] = bernoulli_log_density(
            y, baseline_probability
        )
        scored["candidate_log_density"] = bernoulli_log_density(
            y, candidate_probability
        )
        scored["delta_log_density"] = (
            scored["candidate_log_density"] - scored["baseline_log_density"]
        )
        scored["baseline_probability_correct"] = baseline_probability
        scored["candidate_probability_correct"] = candidate_probability
        scored["correct"] = y
        trial_frames.append(scored)
        fold_rows.append({
            "lag_policy": lag_policy,
            "evaluation_block": int(evaluation_block),
            "subjects": len(eligible_subjects),
            "train_trials": len(train),
            "evaluation_trials": len(evaluation),
            "delta_lpd_per_trial": float(np.mean(scored["delta_log_density"])),
            "surprise_coefficient_standardized": float(candidate_beta[-1]),
            "surprise_mean_train": float(scaling["surprise_mean"]),
            "surprise_scale_train": float(scaling["surprise_scale"]),
            "baseline_optimizer_success": bool(baseline_diagnostics["success"]),
            "candidate_optimizer_success": bool(candidate_diagnostics["success"]),
        })
    return pd.concat(trial_frames, ignore_index=True), pd.DataFrame(fold_rows)


def rt_qc_masks(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_keep = np.zeros(len(train), dtype=bool)
    evaluation_keep = np.zeros(len(evaluation), dtype=bool)
    for subject_id in evaluation["subject_id"].unique():
        train_subject = train["subject_id"].to_numpy() == subject_id
        evaluation_subject = evaluation["subject_id"].to_numpy() == subject_id
        location, scale = robust_location_scale(
            train.loc[train_subject, "log_rt"].to_numpy(dtype=float)
        )
        train_keep[train_subject] = (
            np.abs(train.loc[train_subject, "log_rt"].to_numpy(dtype=float) - location)
            <= float(threshold) * scale
        )
        evaluation_keep[evaluation_subject] = (
            np.abs(
                evaluation.loc[evaluation_subject, "log_rt"].to_numpy(dtype=float)
                - location
            )
            <= float(threshold) * scale
        )
    return train_keep, evaluation_keep


def rt_designs(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    family: str,
    practice_model: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    subjects = sorted(int(value) for value in evaluation["subject_id"].unique())
    train_base, evaluation_base = subject_design(train, evaluation, subjects)
    continuous = [
        "log_trial",
        "log_block_position",
        "lag_log_rt_within",
        f"{family}_choice_entropy",
        f"{family}_replacement_fraction",
    ]
    train_values: dict[str, np.ndarray] = {}
    evaluation_values: dict[str, np.ndarray] = {}
    scaling: dict[str, float] = {}
    for column in continuous:
        train_z, evaluation_z, mean, scale = standardize(
            train[column], evaluation[column]
        )
        train_values[column] = train_z
        evaluation_values[column] = evaluation_z
        scaling[f"{column}_mean"] = mean
        scaling[f"{column}_scale"] = scale
    train_controls = np.column_stack([
        train_values["log_trial"],
        train_values["log_block_position"],
        train["ambiguous"].to_numpy(dtype=float),
        train["previous_error_within"].to_numpy(dtype=float),
        train_values["lag_log_rt_within"],
        train_values[f"{family}_choice_entropy"],
    ])
    evaluation_controls = np.column_stack([
        evaluation_values["log_trial"],
        evaluation_values["log_block_position"],
        evaluation["ambiguous"].to_numpy(dtype=float),
        evaluation["previous_error_within"].to_numpy(dtype=float),
        evaluation_values["lag_log_rt_within"],
        evaluation_values[f"{family}_choice_entropy"],
    ])
    if practice_model == "subject_slopes":
        train_slope_columns = [
            train_values["log_trial"]
            * (train["subject_id"].to_numpy() == subject).astype(float)
            for subject in subjects[1:]
        ]
        evaluation_slope_columns = [
            evaluation_values["log_trial"]
            * (evaluation["subject_id"].to_numpy() == subject).astype(float)
            for subject in subjects[1:]
        ]
        if train_slope_columns:
            train_controls = np.column_stack(
                [train_controls, *train_slope_columns]
            )
            evaluation_controls = np.column_stack(
                [evaluation_controls, *evaluation_slope_columns]
            )
    elif practice_model != "common":
        raise ValueError(f"unknown RT practice model: {practice_model}")
    baseline_train = np.column_stack([train_base, train_controls])
    baseline_evaluation = np.column_stack([evaluation_base, evaluation_controls])
    candidate_train = np.column_stack([
        baseline_train, train_values[f"{family}_replacement_fraction"]
    ])
    candidate_evaluation = np.column_stack([
        baseline_evaluation,
        evaluation_values[f"{family}_replacement_fraction"],
    ])
    return baseline_train, baseline_evaluation, candidate_train, candidate_evaluation, scaling


def run_rt_diagnostic(
    table: pd.DataFrame,
    config: Mapping[str, Any],
    family: str,
    practice_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_block = int(config["design"]["first_evaluation_block"])
    rt_config = config["rt"]
    fold_rows: list[dict[str, Any]] = []
    trial_frames: list[pd.DataFrame] = []
    required = [
        "lag_log_rt_within",
        "previous_error_within",
        f"{family}_choice_entropy",
        f"{family}_replacement_fraction",
    ]
    for evaluation_block in range(first_block, int(table["evaluation_block"].max()) + 1):
        eligible_subjects = sorted(
            int(value)
            for value in table.loc[
                table["evaluation_block"] == evaluation_block, "subject_id"
            ].unique()
        )
        if not eligible_subjects:
            continue
        include_subject = table["subject_id"].isin(eligible_subjects)
        valid = table[required].notna().all(axis=1)
        train = table[
            include_subject & (table["evaluation_block"] < evaluation_block) & valid
        ].copy()
        evaluation = table[
            include_subject & (table["evaluation_block"] == evaluation_block) & valid
        ].copy()
        train_keep, evaluation_keep = rt_qc_masks(
            train,
            evaluation,
            float(rt_config["qc_mad_threshold"]),
        )
        train = train.loc[train_keep].reset_index(drop=True)
        evaluation = evaluation.loc[evaluation_keep].reset_index(drop=True)
        if len(train) < 100 or len(evaluation) < 10:
            continue
        (
            baseline_train,
            baseline_evaluation,
            candidate_train,
            candidate_evaluation,
            scaling,
        ) = rt_designs(train, evaluation, family, practice_model)
        replacement_scale = float(
            scaling[f"{family}_replacement_fraction_scale"]
        )
        if replacement_scale < float(rt_config["minimum_predictor_train_sd"]):
            continue
        fit_kwargs = {
            "degrees_of_freedom": float(rt_config["student_degrees_of_freedom"]),
            "ridge_penalty": float(rt_config["ridge_penalty"]),
            "minimum_scale": float(rt_config["minimum_scale"]),
        }
        baseline_beta, baseline_scale, baseline_diagnostics = fit_student_regression(
            baseline_train, train["log_rt"].to_numpy(dtype=float), **fit_kwargs
        )
        candidate_beta, candidate_scale, candidate_diagnostics = fit_student_regression(
            candidate_train, train["log_rt"].to_numpy(dtype=float), **fit_kwargs
        )
        response = evaluation["log_rt"].to_numpy(dtype=float)
        baseline_log_density = student_log_density(
            response,
            baseline_evaluation @ baseline_beta,
            baseline_scale,
            float(rt_config["student_degrees_of_freedom"]),
        )
        candidate_log_density = student_log_density(
            response,
            candidate_evaluation @ candidate_beta,
            candidate_scale,
            float(rt_config["student_degrees_of_freedom"]),
        )
        scored = evaluation[[
            "subject_id", "trial_index", "evaluation_block", "block_position"
        ]].copy()
        scored["family"] = family
        scored["practice_model"] = practice_model
        scored["baseline_log_density"] = baseline_log_density
        scored["candidate_log_density"] = candidate_log_density
        scored["delta_log_density"] = candidate_log_density - baseline_log_density
        scored["log_rt"] = response
        trial_frames.append(scored)
        fold_rows.append({
            "family": family,
            "practice_model": practice_model,
            "evaluation_block": int(evaluation_block),
            "subjects": len(eligible_subjects),
            "train_trials_after_qc": len(train),
            "evaluation_trials_after_qc": len(evaluation),
            "delta_lpd_per_trial": float(np.mean(scored["delta_log_density"])),
            "replacement_coefficient_per_train_sd": float(candidate_beta[-1]),
            "replacement_coefficient_raw": float(
                candidate_beta[-1] / replacement_scale
            ),
            "replacement_scale_train": replacement_scale,
            "baseline_scale": float(baseline_scale),
            "candidate_scale": float(candidate_scale),
            "baseline_optimizer_success": bool(baseline_diagnostics["success"]),
            "candidate_optimizer_success": bool(candidate_diagnostics["success"]),
        })
    return pd.concat(trial_frames, ignore_index=True), pd.DataFrame(fold_rows)


def bootstrap_summary(
    trial_scores: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    subject = (
        trial_scores.groupby("subject_id", sort=True)
        .agg(
            trials=("delta_log_density", "size"),
            delta_lpd=("delta_log_density", "sum"),
        )
        .reset_index()
    )
    subject["delta_lpd_per_trial"] = subject["delta_lpd"] / subject["trials"]
    values = subject["delta_lpd_per_trial"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    draws = np.mean(
        rng.choice(
            values,
            size=(int(bootstrap_replicates), values.size),
            replace=True,
        ),
        axis=1,
    )
    summary = {
        "subjects": int(len(subject)),
        "trials": int(len(trial_scores)),
        "mean_delta_lpd_per_trial": float(np.mean(values)),
        "median_delta_lpd_per_trial": float(np.median(values)),
        "bootstrap_mean_95_interval": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
        "improved_subjects": int(np.sum(values > 0.0)),
    }
    return subject, summary


def fold_coefficient_summary(
    folds: pd.DataFrame, coefficient_column: str
) -> dict[str, Any]:
    coefficients = folds[coefficient_column].to_numpy(dtype=float)
    return {
        "folds": int(len(folds)),
        "mean": float(np.mean(coefficients)),
        "median": float(np.median(coefficients)),
        "minimum": float(np.min(coefficients)),
        "maximum": float(np.max(coefficients)),
        "positive_folds": int(np.sum(coefficients > 0.0)),
        "all_optimizers_successful": bool(
            folds["baseline_optimizer_success"].all()
            and folds["candidate_optimizer_success"].all()
        ),
    }


def stable_fold_summary(
    trial_scores: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    minimum_subjects: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    stable_blocks = folds.loc[
        folds["subjects"] >= int(minimum_subjects), "evaluation_block"
    ].to_numpy(dtype=int)
    selected = trial_scores[
        trial_scores["evaluation_block"].isin(stable_blocks)
    ]
    _, summary = bootstrap_summary(
        selected,
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    summary["evaluation_blocks"] = stable_blocks.tolist()
    summary["minimum_subjects_per_fold"] = int(minimum_subjects)
    return summary


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    choice = summary["choice"]["within_block"]
    choice_sensitivity = summary["choice"]["all_transitions"]
    rt_dynamic = summary["rt"]["dynamic"]
    rt_dynamic_subject = summary["rt"]["dynamic_subject_practice"]
    rt_static = summary["rt"]["static"]
    lines = [
        "# 0806 两项定向诊断",
        "",
        "所有状态和候选权重只由 choice 历史确定。每个自然块开始时冻结候选权重，预测下一块；RT 从未反过来选择或调整 choice 模型。",
        "",
        "## Surprise 是否解释下一试次 choice 残差",
        "",
        f"- 块内相邻试次：ΔLPD/试次={choice['mean_delta_lpd_per_trial']:.5f}，"
        f"95% 区间 [{choice['bootstrap_mean_95_interval'][0]:.5f}, "
        f"{choice['bootstrap_mean_95_interval'][1]:.5f}]，"
        f"{choice['improved_subjects']}/{choice['subjects']} 名被试改善。",
        f"- 标准化 surprise 系数中位数={choice['coefficient']['median']:.4f}，"
        f"{choice['coefficient']['positive_folds']}/{choice['coefficient']['folds']} 个预测块为正。",
        f"- 第 2--3 块的系数平均为 "
        f"{choice['coefficient_by_learning_stage']['blocks_2_to_3_mean']:.4f}，"
        f"第 4 块以后为 "
        f"{choice['coefficient_by_learning_stage']['blocks_4_plus_mean']:.4f}。",
        f"- 包含跨块相邻试次的敏感性结果：ΔLPD/试次="
        f"{choice_sensitivity['mean_delta_lpd_per_trial']:.5f}。",
        "",
        "## 预测替换量是否解释真实 RT",
        "",
        f"- 动态模型替换量：ΔLPD/试次={rt_dynamic['mean_delta_lpd_per_trial']:.5f}，"
        f"95% 区间 [{rt_dynamic['bootstrap_mean_95_interval'][0]:.5f}, "
        f"{rt_dynamic['bootstrap_mean_95_interval'][1]:.5f}]，"
        f"{rt_dynamic['improved_subjects']}/{rt_dynamic['subjects']} 名被试改善。",
        f"- 动态替换量标准化系数中位数="
        f"{rt_dynamic['coefficient']['median']:.4f}，"
        f"{rt_dynamic['coefficient']['positive_folds']}/"
        f"{rt_dynamic['coefficient']['folds']} 个预测块为正。",
        f"- 仅保留至少 5 名被试的折叠时，系数中位数="
        f"{rt_dynamic['stable_fold_coefficient']['median']:.4f}，"
        f"ΔLPD/试次="
        f"{rt_dynamic['stable_fold_sensitivity']['mean_delta_lpd_per_trial']:.5f}。",
        f"- 允许每名被试有自己的练习斜率后：ΔLPD/试次="
        f"{rt_dynamic_subject['mean_delta_lpd_per_trial']:.5f}，"
        f"95% 区间 [{rt_dynamic_subject['bootstrap_mean_95_interval'][0]:.5f}, "
        f"{rt_dynamic_subject['bootstrap_mean_95_interval'][1]:.5f}]。",
        f"- 静态替换量负对照：ΔLPD/试次="
        f"{rt_static['mean_delta_lpd_per_trial']:.5f}，"
        f"系数中位数={rt_static['coefficient']['median']:.4f}。",
        "",
        "## 判定",
        "",
        f"- Surprise--choice 连接："
        f"{'通过' if choice['support_gate'] else '未通过'}。",
        f"- 动态替换量--RT 连接："
        f"{'通过' if rt_dynamic['support_gate'] else '未通过'}。",
        "- 通过要求同时满足：留出预测增量的被试 bootstrap 区间下界大于 0，且预测方向的折叠系数中位数为正。",
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
    bootstrap_replicates = int(config["design"]["bootstrap_replicates"])
    bootstrap_seed = int(config["design"]["bootstrap_seed"])

    summary: dict[str, Any] = {
        "analysis_id": config["analysis_id"],
        "choice": {},
        "rt": {},
        "guardrails": {
            "choice_only_state_weights": True,
            "block_frozen_candidate_weights": True,
            "rt_never_updates_choice_state": True,
            "current_feedback_excluded_from_rt_controls": True,
            "independent_unit": "subject",
        },
    }
    for index, lag_policy in enumerate(("within_block", "all_transitions")):
        choice_trials, choice_folds = run_choice_diagnostic(
            table, config, lag_policy
        )
        choice_subjects, choice_summary = bootstrap_summary(
            choice_trials,
            bootstrap_replicates=bootstrap_replicates,
            seed=bootstrap_seed + index,
        )
        coefficient = fold_coefficient_summary(
            choice_folds, "surprise_coefficient_standardized"
        )
        choice_summary["coefficient"] = coefficient
        early_coefficients = choice_folds.loc[
            choice_folds["evaluation_block"].isin([2, 3]),
            "surprise_coefficient_standardized",
        ].to_numpy(dtype=float)
        later_coefficients = choice_folds.loc[
            choice_folds["evaluation_block"] >= 4,
            "surprise_coefficient_standardized",
        ].to_numpy(dtype=float)
        choice_summary["coefficient_by_learning_stage"] = {
            "blocks_2_to_3_mean": float(np.mean(early_coefficients)),
            "blocks_4_plus_mean": float(np.mean(later_coefficients)),
        }
        choice_summary["support_gate"] = bool(
            choice_summary["bootstrap_mean_95_interval"][0] > 0.0
            and coefficient["median"] > 0.0
        )
        summary["choice"][lag_policy] = choice_summary
        atomic_frame(
            args.output / f"choice_{lag_policy}_trial_scores.csv", choice_trials
        )
        atomic_frame(
            args.output / f"choice_{lag_policy}_folds.csv", choice_folds
        )
        atomic_frame(
            args.output / f"choice_{lag_policy}_subjects.csv", choice_subjects
        )

    rt_specifications = (
        ("dynamic", "common", "dynamic"),
        (
            "dynamic",
            str(config["rt"]["sensitivity_practice_model"]),
            "dynamic_subject_practice",
        ),
        ("static", "common", "static"),
    )
    for index, (family, practice_model, result_id) in enumerate(rt_specifications):
        rt_trials, rt_folds = run_rt_diagnostic(
            table, config, family, practice_model
        )
        rt_subjects, rt_summary = bootstrap_summary(
            rt_trials,
            bootstrap_replicates=bootstrap_replicates,
            seed=bootstrap_seed + 10 + index,
        )
        coefficient = fold_coefficient_summary(
            rt_folds, "replacement_coefficient_per_train_sd"
        )
        rt_summary["coefficient"] = coefficient
        stable_fold_mask = (
            rt_folds["subjects"]
            >= int(config["rt"]["stable_fold_minimum_subjects"])
        )
        rt_summary["stable_fold_coefficient"] = fold_coefficient_summary(
            rt_folds.loc[stable_fold_mask].reset_index(drop=True),
            "replacement_coefficient_per_train_sd",
        )
        rt_summary["stable_fold_sensitivity"] = stable_fold_summary(
            rt_trials,
            rt_folds,
            minimum_subjects=int(config["rt"]["stable_fold_minimum_subjects"]),
            bootstrap_replicates=bootstrap_replicates,
            seed=bootstrap_seed + 20 + index,
        )
        rt_summary["support_gate"] = bool(
            rt_summary["bootstrap_mean_95_interval"][0] > 0.0
            and coefficient["median"] > 0.0
        )
        summary["rt"][result_id] = rt_summary
        atomic_frame(args.output / f"rt_{result_id}_trial_scores.csv", rt_trials)
        atomic_frame(args.output / f"rt_{result_id}_folds.csv", rt_folds)
        atomic_frame(args.output / f"rt_{result_id}_subjects.csv", rt_subjects)

    atomic_json(args.output / "diagnostic_summary.json", summary)
    write_report(args.output / "diagnostic_report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
