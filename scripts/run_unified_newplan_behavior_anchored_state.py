#!/usr/bin/env python3
"""Fit a behavior-anchored strategy filter for Task-2 trajectories.

The old rule posterior answers an ideal-observer question: which task rule is
compatible with the feedback delivered so far?  It must not be interpreted as
the participant's learning state.  This script instead filters a latent
strategy distribution from the participant's choices and, in a secondary
model, their feedback-before oral report.  Current or future feedback never
updates the strategy filter.

The competing states are:

* uniform exploration/guessing;
* the frozen, training-fitted feature-RL predictor;
* every explicit rule in the condition's rule library.

State persistence has the structured transition

    p(z_t) = rho * p(z_{t-1} | history) + (1-rho) * base_prior.

Only condition-level base masses and persistence are fitted.  The oral model
adds one condition-level reliability parameter.  All parameters are estimated
on each subject's training prefix pooled within condition, then frozen for the
last-block one-step-ahead holdout.  The state at trial t is updated by choice t
and oral report t only after scoring choice t.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from numba import njit
from scipy import __version__ as scipy_version
from scipy.optimize import minimize


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_unified_newplan_oral_validation import (  # noqa: E402
    hypothesis_assignments,
    parse_center,
)
from src.Bayesian_state.utils.unified_newplan import (  # noqa: E402
    ORDER_COLUMNS,
    build_partition,
    rule_predictions,
)


BASE = ROOT / "results/zhuran/unified_newplan"
DEFAULT_DATA = ROOT / "data/processed/Task2_processed.csv"
DEFAULT_CORE = BASE / "core_sobol512_20260802"
DEFAULT_DYNAMIC = BASE / "dynamic_readout_20260802"
DEFAULT_JOINT = BASE / "joint_dynamic_nr2_20260802"
DEFAULT_OUTPUT = BASE / "behavior_anchored_state_20260802"
SCORE_EPS = 1e-7
STATE_EPS = 1e-12
TARGET_BY_CONDITION = {1: 0, 2: 42, 3: 42}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--dynamic", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--joint", type=Path, default=DEFAULT_JOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            json_ready(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


@njit(cache=False)
def _softplus(value: float) -> float:
    if value > 30.0:
        return value
    if value < -30.0:
        return math.exp(value)
    return math.log1p(math.exp(value))


@njit(cache=False)
def _choice_filter_objective(
    parameters: np.ndarray,
    observed_rule_probability: np.ndarray,
    observed_feature_probability: np.ndarray,
    chance_probability: np.ndarray,
    sequence_start: np.ndarray,
) -> float:
    feature_exp = math.exp(parameters[0])
    rule_exp = math.exp(parameters[1])
    denominator = 1.0 + feature_exp + rule_exp
    guess_weight = 1.0 / denominator
    feature_weight = feature_exp / denominator
    rule_weight = rule_exp / denominator
    persistence = 0.001 + 0.998 / (1.0 + math.exp(-parameters[2]))
    n_hypotheses = observed_rule_probability.shape[1]
    posterior = np.empty(n_hypotheses + 2, dtype=np.float64)
    nll = 0.0

    for trial in range(len(observed_feature_probability)):
        if sequence_start[trial]:
            prior_guess = guess_weight
            prior_feature = feature_weight
            observed_probability = (
                prior_guess * chance_probability[trial]
                + prior_feature * observed_feature_probability[trial]
            )
            for hypothesis in range(n_hypotheses):
                observed_probability += (
                    rule_weight
                    / n_hypotheses
                    * observed_rule_probability[trial, hypothesis]
                )
            observed_probability = max(observed_probability, STATE_EPS)
            posterior[0] = (
                prior_guess * chance_probability[trial] / observed_probability
            )
            posterior[1] = (
                prior_feature
                * observed_feature_probability[trial]
                / observed_probability
            )
            for hypothesis in range(n_hypotheses):
                posterior[hypothesis + 2] = (
                    rule_weight
                    / n_hypotheses
                    * observed_rule_probability[trial, hypothesis]
                    / observed_probability
                )
        else:
            prior_guess = (
                persistence * posterior[0]
                + (1.0 - persistence) * guess_weight
            )
            prior_feature = (
                persistence * posterior[1]
                + (1.0 - persistence) * feature_weight
            )
            observed_probability = (
                prior_guess * chance_probability[trial]
                + prior_feature * observed_feature_probability[trial]
            )
            for hypothesis in range(n_hypotheses):
                prior_rule = (
                    persistence * posterior[hypothesis + 2]
                    + (1.0 - persistence) * rule_weight / n_hypotheses
                )
                observed_probability += (
                    prior_rule * observed_rule_probability[trial, hypothesis]
                )
            observed_probability = max(observed_probability, STATE_EPS)
            posterior[0] = (
                prior_guess * chance_probability[trial] / observed_probability
            )
            posterior[1] = (
                prior_feature
                * observed_feature_probability[trial]
                / observed_probability
            )
            for hypothesis in range(n_hypotheses):
                prior_rule = (
                    persistence * posterior[hypothesis + 2]
                    + (1.0 - persistence) * rule_weight / n_hypotheses
                )
                posterior[hypothesis + 2] = (
                    prior_rule
                    * observed_rule_probability[trial, hypothesis]
                    / observed_probability
                )
        nll -= math.log(observed_probability)
    return nll


@njit(cache=False)
def _oral_filter_objective(
    parameters: np.ndarray,
    observed_rule_probability: np.ndarray,
    observed_feature_probability: np.ndarray,
    chance_probability: np.ndarray,
    sequence_start: np.ndarray,
    oral_compatible: np.ndarray,
    oral_encoded: np.ndarray,
) -> float:
    feature_exp = math.exp(parameters[0])
    rule_exp = math.exp(parameters[1])
    denominator = 1.0 + feature_exp + rule_exp
    guess_weight = 1.0 / denominator
    feature_weight = feature_exp / denominator
    rule_weight = rule_exp / denominator
    persistence = 0.001 + 0.998 / (1.0 + math.exp(-parameters[2]))
    oral_strength = _softplus(parameters[3])
    oral_multiplier = math.exp(oral_strength)
    n_hypotheses = observed_rule_probability.shape[1]
    posterior = np.empty(n_hypotheses + 2, dtype=np.float64)
    nll = 0.0

    for trial in range(len(observed_feature_probability)):
        if sequence_start[trial]:
            prior_guess = guess_weight
            prior_feature = feature_weight
            observed_probability = (
                prior_guess * chance_probability[trial]
                + prior_feature * observed_feature_probability[trial]
            )
            for hypothesis in range(n_hypotheses):
                observed_probability += (
                    rule_weight
                    / n_hypotheses
                    * observed_rule_probability[trial, hypothesis]
                )
            observed_probability = max(observed_probability, STATE_EPS)
            posterior[0] = (
                prior_guess * chance_probability[trial] / observed_probability
            )
            posterior[1] = (
                prior_feature
                * observed_feature_probability[trial]
                / observed_probability
            )
            for hypothesis in range(n_hypotheses):
                posterior[hypothesis + 2] = (
                    rule_weight
                    / n_hypotheses
                    * observed_rule_probability[trial, hypothesis]
                    / observed_probability
                )
        else:
            prior_guess = (
                persistence * posterior[0]
                + (1.0 - persistence) * guess_weight
            )
            prior_feature = (
                persistence * posterior[1]
                + (1.0 - persistence) * feature_weight
            )
            observed_probability = (
                prior_guess * chance_probability[trial]
                + prior_feature * observed_feature_probability[trial]
            )
            for hypothesis in range(n_hypotheses):
                prior_rule = (
                    persistence * posterior[hypothesis + 2]
                    + (1.0 - persistence) * rule_weight / n_hypotheses
                )
                observed_probability += (
                    prior_rule * observed_rule_probability[trial, hypothesis]
                )
            observed_probability = max(observed_probability, STATE_EPS)
            posterior[0] = (
                prior_guess * chance_probability[trial] / observed_probability
            )
            posterior[1] = (
                prior_feature
                * observed_feature_probability[trial]
                / observed_probability
            )
            for hypothesis in range(n_hypotheses):
                prior_rule = (
                    persistence * posterior[hypothesis + 2]
                    + (1.0 - persistence) * rule_weight / n_hypotheses
                )
                posterior[hypothesis + 2] = (
                    prior_rule
                    * observed_rule_probability[trial, hypothesis]
                    / observed_probability
                )
        nll -= math.log(observed_probability)

        # The oral report happens after the scored choice and before feedback.
        # Normalize the report multiplier so a report that is uninformative
        # across a uniform rule prior does not mechanically inflate total rule
        # mass merely because its compatible set is broad.
        if oral_encoded[trial]:
            compatible_count = 0.0
            for hypothesis in range(n_hypotheses):
                compatible_count += oral_compatible[trial, hypothesis]
            compatible_fraction = compatible_count / n_hypotheses
            report_normalizer = (
                compatible_fraction * oral_multiplier
                + 1.0
                - compatible_fraction
            )
            state_normalizer = posterior[0] + posterior[1]
            for hypothesis in range(n_hypotheses):
                if oral_compatible[trial, hypothesis]:
                    posterior[hypothesis + 2] *= (
                        oral_multiplier / report_normalizer
                    )
                else:
                    posterior[hypothesis + 2] /= report_normalizer
                state_normalizer += posterior[hypothesis + 2]
            for state in range(n_hypotheses + 2):
                posterior[state] /= state_normalizer
    return nll


def unpack_parameters(values: np.ndarray, include_oral: bool) -> dict[str, float]:
    exponentiated = np.exp(np.asarray([0.0, values[0], values[1]], dtype=float))
    weights = exponentiated / exponentiated.sum()
    output = {
        "base_guess_mass": float(weights[0]),
        "base_feature_mass": float(weights[1]),
        "base_rule_mass": float(weights[2]),
        "state_persistence": float(
            0.001 + 0.998 / (1.0 + np.exp(-float(values[2])))
        ),
        "oral_log_likelihood_ratio": 0.0,
    }
    if include_oral:
        output["oral_log_likelihood_ratio"] = float(
            np.log1p(np.exp(float(values[3])))
        )
    return output


def build_oral_compatibility(
    frame: pd.DataFrame,
    choices: np.ndarray,
    condition: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    partition = build_partition(condition)
    n_hypotheses = int(partition.length)
    centers = [parse_center(value) for value in frame["oral_center"]]
    encoded = np.asarray([center is not None for center in centers], dtype=bool)
    compatible = np.zeros((len(frame), n_hypotheses), dtype=bool)
    if encoded.any():
        encoded_rows = np.flatnonzero(encoded)
        points = np.vstack([centers[index] for index in encoded_rows])
        unique_points, inverse = np.unique(points, axis=0, return_inverse=True)
        assignments = hypothesis_assignments(partition, unique_points)
        compatible[encoded_rows] = (
            assignments[inverse] == choices[encoded_rows, None]
        )
    set_size = compatible.sum(axis=1).astype(np.int16)
    encoded &= set_size > 0
    return compatible, encoded, set_size


def load_subjects(
    data_path: Path,
    core: Path,
    dynamic: Path,
    joint: Path,
) -> list[dict[str, Any]]:
    data = pd.read_csv(data_path, low_memory=False).sort_values(
        ["condition", "iSub", *ORDER_COLUMNS], kind="stable"
    )
    subjects: list[dict[str, Any]] = []
    for subject_id, frame in data.groupby("iSub", sort=True):
        subject_id = int(subject_id)
        frame = frame.reset_index(drop=True)
        condition = int(frame["condition"].iloc[0])
        name = f"subject_{subject_id}.npz"
        with np.load(core / "subject_predictions" / name, allow_pickle=False) as archive:
            choices = archive["choice"].astype(np.int64)
            feedback = archive["feedback"].astype(float)
            holdout = archive["holdout_mask"].astype(bool)
            session = archive["iSession"].astype(np.int16)
            block = archive["iBlock"].astype(np.int16)
            within_trial = archive["iTrial"].astype(np.int16)
            category = archive["category"].astype(np.int64)
        if not np.array_equal(choices + 1, frame["choice"].to_numpy(dtype=int)):
            raise ValueError(f"Choice mismatch for subject {subject_id}")
        if not np.allclose(feedback, frame["feedback"].to_numpy(dtype=float)):
            raise ValueError(f"Feedback mismatch for subject {subject_id}")
        with np.load(core / "q_cache" / name, allow_pickle=False) as archive:
            q_values = archive["q"].astype(np.float64)
        with np.load(dynamic / "subject_predictions" / name, allow_pickle=False) as archive:
            probability_dynamic_rule = archive["p_R0KT_GLOBAL"].astype(np.float64)
        with np.load(joint / "subject_predictions" / name, allow_pickle=False) as archive:
            probability_feature = archive["p_NR2T_JOINT_INDIVIDUAL"].astype(
                np.float64
            )
        if q_values.shape[0] != len(frame):
            raise ValueError(f"Q-cache length mismatch for subject {subject_id}")
        compatible, oral_encoded, oral_set_size = build_oral_compatibility(
            frame, choices, condition
        )
        subjects.append(
            {
                "subject_id": subject_id,
                "condition": condition,
                "frame": frame,
                "choices": choices,
                "category": category,
                "feedback": feedback,
                "holdout": holdout,
                "session": session,
                "block": block,
                "within_trial": within_trial,
                "q": q_values,
                "probability_feature": probability_feature,
                "probability_dynamic_rule": probability_dynamic_rule,
                "oral_compatible": compatible,
                "oral_encoded": oral_encoded,
                "oral_set_size": oral_set_size,
            }
        )
    if len(subjects) != 96:
        raise ValueError(f"Expected 96 subjects, found {len(subjects)}")
    return subjects


def concatenate_training(
    subjects: list[dict[str, Any]], condition: int
) -> dict[str, np.ndarray]:
    observed_rule = []
    observed_feature = []
    chance = []
    sequence_start = []
    oral_compatible = []
    oral_encoded = []
    for subject in subjects:
        if int(subject["condition"]) != int(condition):
            continue
        train_length = int(np.flatnonzero(subject["holdout"])[0])
        choices = subject["choices"][:train_length]
        q_values = subject["q"][:train_length]
        feature = subject["probability_feature"][:train_length]
        observed_rule.append(
            np.take_along_axis(
                q_values, choices[:, None, None], axis=2
            )[:, :, 0]
        )
        observed_feature.append(feature[np.arange(train_length), choices])
        chance.append(
            np.full(train_length, 1.0 / feature.shape[1], dtype=np.float64)
        )
        start = np.zeros(train_length, dtype=bool)
        start[0] = True
        sequence_start.append(start)
        oral_compatible.append(subject["oral_compatible"][:train_length])
        oral_encoded.append(subject["oral_encoded"][:train_length])
    return {
        "observed_rule": np.concatenate(observed_rule),
        "observed_feature": np.concatenate(observed_feature),
        "chance": np.concatenate(chance),
        "sequence_start": np.concatenate(sequence_start),
        "oral_compatible": np.concatenate(oral_compatible),
        "oral_encoded": np.concatenate(oral_encoded),
    }


def fit_condition(
    arrays: dict[str, np.ndarray], include_oral: bool
) -> tuple[np.ndarray, dict[str, Any]]:
    starts_choice = [
        (0.0, 0.0, 2.0),
        (0.0, 1.0, 2.0),
        (1.0, 0.0, 2.0),
        (-1.0, 1.0, 3.0),
        (1.0, -1.0, 3.0),
        (0.0, 0.0, 4.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, -2.0),
    ]
    if include_oral:
        starts = [
            np.asarray((*start, oral), dtype=float)
            for start in starts_choice[:6]
            for oral in (-3.0, 0.0, 2.0)
        ]

        def objective(values: np.ndarray) -> float:
            return float(
                _oral_filter_objective(
                    values,
                    arrays["observed_rule"],
                    arrays["observed_feature"],
                    arrays["chance"],
                    arrays["sequence_start"],
                    arrays["oral_compatible"],
                    arrays["oral_encoded"],
                )
            )

        bounds = [(-8.0, 8.0)] * 4
    else:
        starts = [np.asarray(start, dtype=float) for start in starts_choice]

        def objective(values: np.ndarray) -> float:
            return float(
                _choice_filter_objective(
                    values,
                    arrays["observed_rule"],
                    arrays["observed_feature"],
                    arrays["chance"],
                    arrays["sequence_start"],
                )
            )

        bounds = [(-8.0, 8.0)] * 3

    # Warm the numba specialization before timing the multi-start fits.
    objective(starts[0])
    fits = [
        minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 400, "ftol": 1e-11},
        )
        for start in starts
    ]
    best = min(fits, key=lambda result: float(result.fun))
    diagnostics = {
        **unpack_parameters(best.x, include_oral),
        "train_nll": float(best.fun),
        "optimizer_success": bool(best.success),
        "optimizer_message": str(best.message),
        "n_starts": int(len(fits)),
        "n_converged_starts": int(sum(bool(result.success) for result in fits)),
        "n_same_optimal_region": int(
            sum(abs(float(result.fun) - float(best.fun)) <= 1e-4 for result in fits)
        ),
        "raw_parameters": [float(value) for value in best.x],
    }
    return np.asarray(best.x, dtype=float), diagnostics


def apply_oral_update(
    posterior: np.ndarray,
    compatible: np.ndarray,
    encoded: bool,
    oral_strength: float,
) -> np.ndarray:
    output = posterior.copy()
    if not encoded or oral_strength <= 0.0:
        return output
    n_hypotheses = len(compatible)
    fraction = float(np.mean(compatible))
    multiplier = float(np.exp(oral_strength))
    normalizer = fraction * multiplier + 1.0 - fraction
    output[2:][compatible] *= multiplier / normalizer
    output[2:][~compatible] /= normalizer
    output /= output.sum()
    return output


def run_filter(
    subject: dict[str, Any], values: np.ndarray, include_oral: bool
) -> dict[str, np.ndarray]:
    parameters = unpack_parameters(values, include_oral)
    q_values = subject["q"]
    feature = subject["probability_feature"]
    choices = subject["choices"]
    compatible = subject["oral_compatible"]
    encoded = subject["oral_encoded"]
    n_trials, n_hypotheses, n_categories = q_values.shape
    base = np.concatenate(
        [
            [parameters["base_guess_mass"], parameters["base_feature_mass"]],
            np.full(
                n_hypotheses,
                parameters["base_rule_mass"] / n_hypotheses,
                dtype=float,
            ),
        ]
    )
    persistence = parameters["state_persistence"]
    oral_strength = parameters["oral_log_likelihood_ratio"]
    predictive = np.empty((n_trials, n_categories), dtype=float)
    prior_state = np.empty((n_trials, n_hypotheses + 2), dtype=np.float32)
    post_choice = np.empty_like(prior_state)
    post_oral = np.empty_like(prior_state)
    posterior = base.copy()
    for trial in range(n_trials):
        prior = (
            base.copy()
            if trial == 0
            else persistence * posterior + (1.0 - persistence) * base
        )
        probability = (
            prior[0] / n_categories
            + prior[1] * feature[trial]
            + prior[2:] @ q_values[trial]
        )
        probability = np.maximum(probability, STATE_EPS)
        probability /= probability.sum()
        choice = int(choices[trial])
        emissions = np.concatenate(
            [[1.0 / n_categories, feature[trial, choice]], q_values[trial, :, choice]]
        )
        choice_posterior = prior * emissions
        choice_posterior /= max(float(choice_posterior.sum()), STATE_EPS)
        posterior = apply_oral_update(
            choice_posterior,
            compatible[trial],
            bool(encoded[trial]),
            oral_strength,
        )
        predictive[trial] = probability
        prior_state[trial] = prior
        post_choice[trial] = choice_posterior
        post_oral[trial] = posterior
    return {
        "predictive_probability": predictive,
        "prior_state": prior_state.astype(float),
        "post_choice_state": post_choice.astype(float),
        "post_oral_state": post_oral.astype(float),
    }


def score(
    probabilities: np.ndarray, choices: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    rows = np.flatnonzero(mask)
    p = probabilities[rows]
    y = choices[rows]
    observed = np.clip(p[np.arange(len(rows)), y], SCORE_EPS, 1.0)
    one_hot = np.eye(p.shape[1])[y]
    return {
        "n_trials": int(len(rows)),
        "nll": float(-np.log(observed).sum()),
        "nll_per_trial": float(-np.log(observed).mean()),
        "brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "accuracy": float(np.mean(np.argmax(p, axis=1) == y)),
        "mean_observed_choice_probability": float(observed.mean()),
    }


def first_sustained(
    values: np.ndarray, threshold: float, window: int = 16
) -> float:
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return np.nan
    above = values >= float(threshold)
    for start in range(len(values) - window + 1):
        if np.all(above[start : start + window]):
            return float(start + 1)
    return np.nan


def behavioral_onset(
    exact_correct: np.ndarray,
    rolling_window: int = 32,
    threshold: float = 0.75,
    sustained_windows: int = 5,
) -> float:
    series = pd.Series(np.asarray(exact_correct, dtype=float))
    rolling = series.rolling(rolling_window, min_periods=rolling_window).mean()
    above = rolling.to_numpy() >= threshold
    for start in range(len(above) - sustained_windows + 1):
        if np.all(above[start : start + sustained_windows]):
            return float(start + 1)
    return np.nan


def bootstrap_interval(
    values: np.ndarray, seed: int, n_boot: int = 10000
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def summarize_comparisons(metrics: pd.DataFrame, seed: int) -> pd.DataFrame:
    specifications = [
        ("BEHAVIOR_ORAL", "BEHAVIOR_CHOICE", "past_oral_increment"),
        ("BEHAVIOR_ORAL", "FEATURE_RL", "behavior_oral_vs_feature"),
        ("BEHAVIOR_CHOICE", "FEATURE_RL", "behavior_choice_vs_feature"),
        ("BEHAVIOR_ORAL", "DYNAMIC_RULE", "behavior_oral_vs_old_dynamic_rule"),
    ]
    holdout = metrics[metrics["segment"].eq("holdout")]
    rows = []
    for candidate, reference, label in specifications:
        for condition_label, group in [
            (str(condition), holdout[holdout["condition"].eq(condition)])
            for condition in (1, 2, 3)
        ] + [("all", holdout)]:
            paired = group[group["model"].eq(candidate)].merge(
                group[group["model"].eq(reference)],
                on=["subject_id", "condition", "segment"],
                suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            delta = (
                paired["nll_per_trial_reference"].to_numpy(dtype=float)
                - paired["nll_per_trial_candidate"].to_numpy(dtype=float)
            )
            low, high = bootstrap_interval(
                delta,
                seed + sum(ord(character) for character in label + condition_label),
            )
            rows.append(
                {
                    "comparison": label,
                    "candidate": candidate,
                    "reference": reference,
                    "condition": condition_label,
                    "n_subjects": int(len(delta)),
                    "mean_delta_nll_per_trial": float(delta.mean()),
                    "median_delta_nll_per_trial": float(np.median(delta)),
                    "bootstrap_mean_ci_low": low,
                    "bootstrap_mean_ci_high": high,
                    "n_improved": int((delta > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def render_report(
    output: Path,
    parameters: pd.DataFrame,
    comparisons: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    oral_increment = comparisons[
        comparisons["comparison"].eq("past_oral_increment")
        & comparisons["condition"].eq("all")
    ].iloc[0]
    feature_gate = comparisons[
        comparisons["comparison"].eq("behavior_oral_vs_feature")
        & comparisons["condition"].eq("all")
    ].iloc[0]
    reached = int(summary["target_state_t50_sustained16"].notna().sum())
    lines = [
        "# Behavior-anchored strategy-state filter",
        "",
        "> Status: training-fitted conditional state-space diagnostic. The state is inferred from choices and past feedback-before oral reports; feedback never directly updates strategy identity.",
        "",
        "## What changed",
        "",
        "The old feedback posterior has been demoted to an ideal-observer task-identifiability benchmark. It is not used as a participant cognitive state. The new filter lets uniform exploration, the frozen feature-RL predictor, and every explicit rule compete to explain each observed choice.",
        "",
        "## Held-out prediction",
        "",
        f"Adding past oral reports to the choice-only strategy filter changed held-out NLL/trial by {oral_increment.mean_delta_nll_per_trial:.6f} (95% subject-bootstrap CI [{oral_increment.bootstrap_mean_ci_low:.6f}, {oral_increment.bootstrap_mean_ci_high:.6f}]; improved {int(oral_increment.n_improved)}/{int(oral_increment.n_subjects)} subjects).",
        f"Relative to the frozen individual feature-RL predictor, the behavior+oral filter changed held-out NLL/trial by {feature_gate.mean_delta_nll_per_trial:.6f} (95% CI [{feature_gate.bootstrap_mean_ci_low:.6f}, {feature_gate.bootstrap_mean_ci_high:.6f}]; improved {int(feature_gate.n_improved)}/{int(feature_gate.n_subjects)} subjects).",
        "",
        "Positive values favor the first-named model. Choice t is scored before choice t or oral report t updates the state.",
        "",
        "| Comparison | Condition | Mean ΔNLL/trial | 95% CI | Improved |",
        "|:--|:--|--:|:--|:--|",
    ]
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {row.condition} | {row.mean_delta_nll_per_trial:.6f} | "
            f"[{row.bootstrap_mean_ci_low:.6f}, {row.bootstrap_mean_ci_high:.6f}] | "
            f"{int(row.n_improved)}/{int(row.n_subjects)} |"
        )
    lines.extend(
        [
            "",
            "## State interpretation",
            "",
            f"Only {reached}/96 subjects reached a behavior+oral target-rule state probability above 0.50 for 16 consecutive trials. Non-attainment is retained rather than forced into an early acquisition time.",
            "",
            "The target-rule state means that the participant's choices and prior oral reports are more consistent with the known target rule than with guessing, feature-RL, or another explicit rule. It is still a model attribution, not direct access to a private mental state. When feature-RL and the target rule make the same predictions, the data may remain unable to distinguish them.",
            "",
            "## Parameters",
            "",
            "Parameters are condition-level and trained only on subject training prefixes. Rule emissions have no fitted learning rate or sensitivity; perceptual q probabilities are frozen. The oral log-likelihood ratio is fitted only by whether past reports improve later choice prediction.",
            "",
            "| Model | Condition | Guess base | Feature base | Rule base | Persistence | Oral log LR | Same optimum |",
            "|:--|:--|--:|--:|--:|--:|--:|--:|",
        ]
    )
    for row in parameters.itertuples(index=False):
        lines.append(
            f"| {row.model} | {int(row.condition)} | {row.base_guess_mass:.4f} | "
            f"{row.base_feature_mass:.4f} | {row.base_rule_mass:.4f} | "
            f"{row.state_persistence:.4f} | {row.oral_log_likelihood_ratio:.4f} | "
            f"{int(row.n_same_optimal_region)}/{int(row.n_starts)} |"
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Oral reports are structured compatible sets, not a generative likelihood over Chinese sentences; their fitted reliability must be judged by held-out next-choice gain.",
            "- The feature-RL state is flexible and individually fitted on the training prefix. If it and a rule predict the same choices, the filter should remain uncertain rather than declare rule knowledge.",
            "- RT is not used to fit the state. A separate frozen-state RT test is required before claiming that state uncertainty explains response time.",
            "- The model is a conditional strategy filter, not a final hierarchical posterior and not evidence that every participant uses a verbal rule.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    data_path = args.data.resolve()
    core = args.core.resolve()
    dynamic = args.dynamic.resolve()
    joint = args.joint.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_output = output / "subject_predictions"
    prediction_output.mkdir(exist_ok=True)

    subjects = load_subjects(data_path, core, dynamic, joint)
    fitted: dict[tuple[str, int], np.ndarray] = {}
    parameter_rows = []
    for condition in (1, 2, 3):
        arrays = concatenate_training(subjects, condition)
        for model, include_oral in (
            ("BEHAVIOR_CHOICE", False),
            ("BEHAVIOR_ORAL", True),
        ):
            fit_started = time.time()
            values, diagnostics = fit_condition(arrays, include_oral)
            fitted[(model, condition)] = values
            parameter_rows.append(
                {
                    "model": model,
                    "condition": condition,
                    **diagnostics,
                    "runtime_seconds": float(time.time() - fit_started),
                }
            )
            print(
                f"[fit] {model} condition {condition}: "
                f"NLL={diagnostics['train_nll']:.3f}, "
                f"rho={diagnostics['state_persistence']:.3f}, "
                f"same={diagnostics['n_same_optimal_region']}/{diagnostics['n_starts']}",
                flush=True,
            )

    metric_rows = []
    trial_frames = []
    summary_rows = []
    for subject in subjects:
        subject_id = int(subject["subject_id"])
        condition = int(subject["condition"])
        choices = subject["choices"]
        holdout = subject["holdout"]
        n_trials = len(choices)
        target = TARGET_BY_CONDITION[condition]
        choice_result = run_filter(
            subject, fitted[("BEHAVIOR_CHOICE", condition)], False
        )
        oral_result = run_filter(
            subject, fitted[("BEHAVIOR_ORAL", condition)], True
        )
        model_probabilities = {
            "FEATURE_RL": subject["probability_feature"],
            "DYNAMIC_RULE": subject["probability_dynamic_rule"],
            "BEHAVIOR_CHOICE": choice_result["predictive_probability"],
            "BEHAVIOR_ORAL": oral_result["predictive_probability"],
        }
        for model, probability in model_probabilities.items():
            for segment, mask in (("train", ~holdout), ("holdout", holdout)):
                metric_rows.append(
                    {
                        "subject_id": subject_id,
                        "condition": condition,
                        "model": model,
                        "segment": segment,
                        **score(probability, choices, mask),
                    }
                )

        state = oral_result["post_oral_state"]
        prior_state = oral_result["prior_state"]
        rule_state = state[:, 2:]
        prior_rule_state = prior_state[:, 2:]
        target_probability = rule_state[:, target]
        prior_target_probability = prior_rule_state[:, target]
        wrong_rule_probability = rule_state.sum(axis=1) - target_probability
        prior_wrong_rule_probability = (
            prior_rule_state.sum(axis=1) - prior_target_probability
        )
        top_wrong_rule = np.argmax(
            np.where(
                np.arange(rule_state.shape[1])[None, :] == target,
                -np.inf,
                rule_state,
            ),
            axis=1,
        )
        strongest_wrong_probability = rule_state[
            np.arange(n_trials), top_wrong_rule
        ]
        mode_matrix = np.column_stack(
            [
                state[:, 0],
                state[:, 1],
                target_probability,
                wrong_rule_probability,
            ]
        )
        mode_names = np.asarray(
            ["guess", "feature_rl", "target_rule", "other_rule"], dtype=object
        )
        dominant_mode = mode_names[np.argmax(mode_matrix, axis=1)]
        prior_mode_matrix = np.column_stack(
            [
                prior_state[:, 0],
                prior_state[:, 1],
                prior_target_probability,
                prior_wrong_rule_probability,
            ]
        )
        prior_strategy_entropy = -np.sum(
            np.clip(prior_mode_matrix, STATE_EPS, 1.0)
            * np.log(np.clip(prior_mode_matrix, STATE_EPS, 1.0)),
            axis=1,
        )

        ideal = rule_predictions(
            subject["q"],
            choices,
            subject["feedback"],
            condition,
            retention=1.0,
            sensitivity=1.0,
            return_beliefs=True,
        )
        if ideal.beliefs is None:
            raise RuntimeError("Ideal-observer rule beliefs were not returned")
        ideal_target = ideal.beliefs[:, target]
        exact_correct = choices == subject["category"]
        observed_oral = model_probabilities["BEHAVIOR_ORAL"][
            np.arange(n_trials), choices
        ]
        observed_feature = model_probabilities["FEATURE_RL"][
            np.arange(n_trials), choices
        ]
        frame = pd.DataFrame(
            {
                "subject_id": subject_id,
                "condition": condition,
                "trial": np.arange(n_trials) + 1,
                "session": subject["session"],
                "block": subject["block"],
                "within_experiment_trial": subject["within_trial"],
                "segment": np.where(holdout, "holdout", "train"),
                "choice": choices + 1,
                "category": subject["category"] + 1,
                "exact_correct": exact_correct,
                "feedback": subject["feedback"],
                "oral_encoded": subject["oral_encoded"],
                "oral_compatible_set_size": subject["oral_set_size"],
                "oral_target_compatible": subject["oral_compatible"][:, target],
                "state_guess_probability": state[:, 0],
                "state_feature_probability": state[:, 1],
                "state_target_rule_probability": target_probability,
                "state_other_rule_probability": wrong_rule_probability,
                "prior_state_guess_probability": prior_state[:, 0],
                "prior_state_feature_probability": prior_state[:, 1],
                "prior_state_target_rule_probability": prior_target_probability,
                "prior_state_other_rule_probability": prior_wrong_rule_probability,
                "prior_strategy_entropy": prior_strategy_entropy,
                "strongest_wrong_rule_probability": strongest_wrong_probability,
                "top_wrong_rule": top_wrong_rule,
                "dominant_state": dominant_mode,
                "ideal_observer_target_probability": ideal_target,
                "observed_choice_probability_behavior_oral": observed_oral,
                "observed_choice_probability_feature": observed_feature,
                "choice_only_target_rule_probability": choice_result[
                    "post_choice_state"
                ][:, 2 + target],
                "choice_only_prior_target_rule_probability": choice_result[
                    "prior_state"
                ][:, 2 + target],
            }
        )
        trial_frames.append(frame)

        holdout_rows = np.flatnonzero(holdout)
        final_window = min(64, n_trials)
        target_t50 = first_sustained(target_probability, 0.50, 16)
        target_t80 = first_sustained(target_probability, 0.80, 16)
        ideal_t90 = first_sustained(ideal_target, 0.90, 16)
        summary_rows.append(
            {
                "subject_id": subject_id,
                "condition": condition,
                "n_trials": n_trials,
                "n_holdout": int(holdout.sum()),
                "holdout_start_trial": int(holdout_rows[0] + 1),
                "holdout_nll_behavior_oral": score(
                    model_probabilities["BEHAVIOR_ORAL"], choices, holdout
                )["nll_per_trial"],
                "holdout_nll_behavior_choice": score(
                    model_probabilities["BEHAVIOR_CHOICE"], choices, holdout
                )["nll_per_trial"],
                "holdout_nll_feature": score(
                    model_probabilities["FEATURE_RL"], choices, holdout
                )["nll_per_trial"],
                "holdout_nll_old_dynamic_rule": score(
                    model_probabilities["DYNAMIC_RULE"], choices, holdout
                )["nll_per_trial"],
                "target_state_t50_sustained16": target_t50,
                "target_state_t80_sustained16": target_t80,
                "ideal_observer_t90_sustained16": ideal_t90,
                "behavioral_onset_75_roll32_sustained5": behavioral_onset(
                    exact_correct
                ),
                "target_state_initial32_mean": float(
                    target_probability[: min(32, n_trials)].mean()
                ),
                "target_state_final64_mean": float(
                    target_probability[-final_window:].mean()
                ),
                "target_state_max": float(target_probability.max()),
                "feature_state_final64_mean": float(
                    state[-final_window:, 1].mean()
                ),
                "other_rule_state_final64_mean": float(
                    wrong_rule_probability[-final_window:].mean()
                ),
                "guess_state_final64_mean": float(
                    state[-final_window:, 0].mean()
                ),
                "exact_accuracy_final64": float(
                    exact_correct[-final_window:].mean()
                ),
                "oral_target_compatible_final64": float(
                    subject["oral_compatible"][-final_window:, target][
                        subject["oral_encoded"][-final_window:]
                    ].mean()
                )
                if subject["oral_encoded"][-final_window:].any()
                else np.nan,
                "final_dominant_state": str(dominant_mode[-1]),
            }
        )
        atomic_savez(
            prediction_output / f"subject_{subject_id}.npz",
            subject_id=np.asarray(subject_id),
            condition=np.asarray(condition),
            choice=choices.astype(np.int8),
            holdout_mask=holdout,
            predictive_probability_behavior_choice=choice_result[
                "predictive_probability"
            ].astype(np.float32),
            predictive_probability_behavior_oral=oral_result[
                "predictive_probability"
            ].astype(np.float32),
            post_choice_state_choice_model=choice_result[
                "post_choice_state"
            ].astype(np.float32),
            prior_state_choice_model=choice_result["prior_state"].astype(
                np.float32
            ),
            post_oral_state=oral_result["post_oral_state"].astype(np.float32),
            prior_state_behavior_oral=oral_result["prior_state"].astype(np.float32),
            ideal_observer_target_probability=ideal_target.astype(np.float32),
            oral_encoded=subject["oral_encoded"],
            oral_set_size=subject["oral_set_size"],
        )

    parameters = pd.DataFrame(parameter_rows).sort_values(["model", "condition"])
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["condition", "subject_id", "segment", "model"]
    )
    trials = pd.concat(trial_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["condition", "subject_id"]
    )
    comparisons = summarize_comparisons(metrics, args.seed)
    atomic_csv(output / "parameters.csv", parameters)
    atomic_csv(output / "subject_metrics.csv", metrics)
    atomic_csv(output / "model_comparisons.csv", comparisons)
    atomic_csv(output / "subject_summary.csv", summary)
    atomic_csv_gzip(output / "trial_states.csv.gz", trials)
    render_report(output, parameters, comparisons, summary)
    manifest = {
        "result_type": "unified_newplan_behavior_anchored_strategy_filter",
        "status": "complete",
        "evidence_scope": (
            "condition-level training-fitted conditional state filter with "
            "subject temporal holdout; not final hierarchical posterior"
        ),
        "states": ["guess", "feature_rl", "one state per explicit rule"],
        "transition": "rho * previous posterior + (1-rho) * condition base prior",
        "state_update_order": (
            "predict current choice; update with current choice; update with current "
            "feedback-before oral compatible set; never update strategy with feedback"
        ),
        "oral_mapping": (
            "candidate rule is compatible when it assigns the just-made choice to "
            "the structured oral center; broad-set normalization preserves mean "
            "rule mass under a uniform rule prior"
        ),
        "target_hypothesis": TARGET_BY_CONDITION,
        "n_subjects": int(summary.subject_id.nunique()),
        "n_trials": int(len(trials)),
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "input_manifest_sha256": {
            "core": sha256_file(core / "manifest.json"),
            "dynamic": sha256_file(dynamic / "manifest.json"),
            "joint": sha256_file(joint / "manifest.json"),
        },
        "runtime_seconds": float(time.time() - started),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy_version,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(output / "manifest.json", manifest)
    print(
        f"[done] wrote behavior-anchored states for {len(summary)} subjects to {output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
