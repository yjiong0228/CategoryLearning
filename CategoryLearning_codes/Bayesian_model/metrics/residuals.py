"""Causal residual diagnostics for sequential Bernoulli predictions.

The functions in this module operate on one-step-ahead probabilities.  Trial
``t`` outcomes are never used to construct their own predictions: observed
outcomes enter only lagged residual tests and the state carried into later
trials.  This makes the diagnostics suitable for asking whether a fitted
sequential model leaves predictable temporal structure behind.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import norm


_PROBABILITY_EPSILON = 1e-8


def _binary_inputs(
    observed: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    valid_mask: Sequence[bool] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outcome = np.asarray(observed, dtype=float).reshape(-1)
    probability = np.asarray(predicted, dtype=float).reshape(-1)
    if outcome.shape != probability.shape:
        raise ValueError(
            "observed and predicted Bernoulli arrays must have the same shape"
        )
    if valid_mask is None:
        mask = np.ones(outcome.size, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if mask.shape != outcome.shape:
            raise ValueError("valid_mask does not align with Bernoulli arrays")
    mask &= np.isfinite(outcome) & np.isfinite(probability)
    if np.any((outcome[mask] < 0.0) | (outcome[mask] > 1.0)):
        raise ValueError("observed Bernoulli outcomes must lie in [0, 1]")
    if np.any((probability[mask] < 0.0) | (probability[mask] > 1.0)):
        raise ValueError("predicted Bernoulli probabilities must lie in [0, 1]")
    probability = np.clip(probability, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
    return outcome, probability, mask


def bernoulli_calibration_test(
    observed: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    valid_mask: Sequence[bool] | np.ndarray | None = None,
) -> dict[str, float | int]:
    """Test aggregate observed-minus-predicted Bernoulli calibration."""

    outcome, probability, mask = _binary_inputs(observed, predicted, valid_mask)
    residual = outcome - probability
    variance = probability * (1.0 - probability)
    numerator = float(np.sum(residual[mask]))
    denominator = float(np.sqrt(np.sum(variance[mask])))
    z_value = numerator / denominator if denominator > 0.0 else float("nan")
    p_value = (
        float(2.0 * norm.sf(abs(z_value)))
        if np.isfinite(z_value)
        else float("nan")
    )
    return {
        "n_observations": int(np.sum(mask)),
        "mean_residual": (
            float(np.mean(residual[mask])) if np.any(mask) else float("nan")
        ),
        "z": float(z_value),
        "p": float(p_value),
    }


def logit_intercept_recalibration(
    observed: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    valid_mask: Sequence[bool] | np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Remove only global level bias while preserving trialwise probability shape.

    The fitted intercept is a nuisance adjustment for residual diagnosis, not a
    replacement model and not an out-of-sample performance estimate.
    """

    outcome, probability, mask = _binary_inputs(observed, predicted, valid_mask)
    if not np.any(mask):
        raise ValueError("intercept recalibration requires at least one valid trial")
    target_total = float(np.sum(outcome[mask]))
    offset = logit(probability)
    lower = -40.0
    upper = 40.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        fitted_total = float(np.sum(expit(offset[mask] + midpoint)))
        if fitted_total < target_total:
            lower = midpoint
        else:
            upper = midpoint
    intercept = 0.5 * (lower + upper)
    return np.asarray(expit(offset + intercept), dtype=float), float(intercept)


def martingale_lag_tests(
    observed: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    valid_mask: Sequence[bool] | np.ndarray | None = None,
    *,
    max_lag: int = 8,
) -> list[dict[str, float | int]]:
    """Test whether past one-step residuals predict later residuals.

    For each lag, the numerator is ``sum(e_t * e_(t-lag))``.  Its denominator
    uses the Bernoulli conditional variance at ``t`` and treats the lagged
    residual as information already available before trial ``t``.
    """

    outcome, probability, mask = _binary_inputs(observed, predicted, valid_mask)
    max_lag = int(max_lag)
    if max_lag < 1:
        raise ValueError("max_lag must be positive")
    residual = outcome - probability
    variance = probability * (1.0 - probability)
    rows: list[dict[str, float | int]] = []
    for lag in range(1, max_lag + 1):
        pair_mask = mask[lag:] & mask[:-lag]
        current = residual[lag:]
        previous = residual[:-lag]
        conditional_variance = variance[lag:]
        numerator = float(np.sum(current[pair_mask] * previous[pair_mask]))
        denominator = float(
            np.sqrt(
                np.sum(
                    conditional_variance[pair_mask]
                    * np.square(previous[pair_mask])
                )
            )
        )
        z_value = numerator / denominator if denominator > 0.0 else float("nan")
        p_value = (
            float(2.0 * norm.sf(abs(z_value)))
            if np.isfinite(z_value)
            else float("nan")
        )
        rows.append(
            {
                "lag": int(lag),
                "n_pairs": int(np.sum(pair_mask)),
                "z": float(z_value),
                "p": float(p_value),
            }
        )
    finite_p = np.asarray([row["p"] for row in rows], dtype=float)
    finite_p = finite_p[np.isfinite(finite_p)]
    family_p = (
        float(min(1.0, float(np.min(finite_p)) * max_lag))
        if finite_p.size
        else float("nan")
    )
    for row in rows:
        row["familywise_p"] = family_p
    return rows


def rolling_martingale_z(
    observed: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    valid_mask: Sequence[bool] | np.ndarray | None = None,
    *,
    window_size: int,
) -> dict[str, Any]:
    """Return causal-window residual z scores and a scan-adjusted p value."""

    outcome, probability, mask = _binary_inputs(observed, predicted, valid_mask)
    window_size = int(window_size)
    if window_size < 1:
        raise ValueError("window_size must be positive")
    residual = outcome - probability
    variance = probability * (1.0 - probability)
    z_values = np.full(outcome.size, np.nan, dtype=float)
    for end in range(window_size, outcome.size + 1):
        start = end - window_size
        window = slice(start, end)
        if not bool(np.all(mask[window])):
            continue
        denominator = float(np.sqrt(np.sum(variance[window])))
        if denominator > 0.0:
            z_values[end - 1] = float(np.sum(residual[window]) / denominator)
    finite_indices = np.flatnonzero(np.isfinite(z_values))
    if finite_indices.size:
        maximum_index = int(
            finite_indices[np.argmax(np.abs(z_values[finite_indices]))]
        )
        maximum_z = float(z_values[maximum_index])
        raw_p = float(2.0 * norm.sf(abs(maximum_z)))
        family_p = float(min(1.0, raw_p * finite_indices.size))
    else:
        maximum_index = -1
        maximum_z = float("nan")
        raw_p = float("nan")
        family_p = float("nan")
    return {
        "z_values": z_values,
        "n_windows": int(finite_indices.size),
        "max_end_index": int(maximum_index),
        "max_abs_z": float(abs(maximum_z)) if np.isfinite(maximum_z) else float("nan"),
        "max_signed_z": float(maximum_z),
        "max_raw_p": float(raw_p),
        "familywise_p": float(family_p),
    }


def switch_residual_test(
    observed_choice_index: Sequence[int] | np.ndarray,
    category_probabilities: Sequence[Sequence[float]] | np.ndarray,
    valid_mask: Sequence[bool] | np.ndarray | None = None,
) -> dict[str, float | int]:
    """Test observed switch count against its one-step predictive expectation."""

    choices = np.asarray(observed_choice_index, dtype=int).reshape(-1)
    probabilities = np.asarray(category_probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] != choices.size:
        raise ValueError("choice probabilities must have shape (trials, categories)")
    if valid_mask is None:
        mask = np.ones(choices.size, dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if mask.shape != choices.shape:
            raise ValueError("valid_mask does not align with choice trials")
    pair_mask = mask[1:] & mask[:-1]
    pair_mask &= (choices[:-1] >= 0) & (choices[:-1] < probabilities.shape[1])
    pair_mask &= (choices[1:] >= 0) & (choices[1:] < probabilities.shape[1])
    pair_mask &= np.all(np.isfinite(probabilities[1:]), axis=1)
    rows = np.arange(1, choices.size)
    previous = choices[:-1]
    predicted_switch = np.full(choices.size - 1, np.nan, dtype=float)
    valid_rows = np.flatnonzero(pair_mask)
    predicted_switch[valid_rows] = 1.0 - probabilities[rows[valid_rows], previous[valid_rows]]
    observed_switch = (choices[1:] != choices[:-1]).astype(float)
    return bernoulli_calibration_test(
        observed_switch,
        predicted_switch,
        pair_mask,
    )


def causal_residual_state_feature(
    observed: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    valid_mask: Sequence[bool] | np.ndarray | None = None,
    *,
    window_size: int,
) -> np.ndarray:
    """Build an EWMA state from residuals available strictly before each trial."""

    outcome, probability, mask = _binary_inputs(observed, predicted, valid_mask)
    window_size = int(window_size)
    if window_size < 1:
        raise ValueError("window_size must be positive")
    alpha = 2.0 / float(window_size + 1)
    state = 0.0
    feature = np.zeros(outcome.size, dtype=float)
    for trial_index in range(outcome.size):
        feature[trial_index] = state
        if mask[trial_index]:
            state = (
                (1.0 - alpha) * state
                + alpha * float(outcome[trial_index] - probability[trial_index])
            )
    return feature


def _fit_offset_logistic(
    offset: np.ndarray,
    feature: np.ndarray,
    outcome: np.ndarray,
    train_indices: np.ndarray,
    *,
    include_state: bool,
    ridge: float,
) -> np.ndarray:
    if include_state:
        design = np.column_stack(
            [np.ones(train_indices.size, dtype=float), feature[train_indices]]
        )
    else:
        design = np.ones((train_indices.size, 1), dtype=float)
    train_offset = offset[train_indices]
    train_outcome = outcome[train_indices]

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        eta = train_offset + design @ coefficient
        probability = expit(eta)
        penalty = float(ridge) * float(np.sum(np.square(coefficient[1:])))
        loss = float(
            np.sum(np.logaddexp(0.0, eta) - train_outcome * eta) + penalty
        )
        gradient = design.T @ (probability - train_outcome)
        if coefficient.size > 1:
            gradient[1:] += 2.0 * float(ridge) * coefficient[1:]
        return loss, np.asarray(gradient, dtype=float)

    result = minimize(
        objective,
        np.zeros(design.shape[1], dtype=float),
        jac=True,
        method="BFGS",
    )
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("offset logistic residual probe did not converge to finite values")
    return np.asarray(result.x, dtype=float)


def forward_residual_state_probe(
    observed: Sequence[float] | np.ndarray,
    predicted: Sequence[float] | np.ndarray,
    valid_mask: Sequence[bool] | np.ndarray | None = None,
    *,
    window_size: int,
    n_folds: int = 4,
    ridge: float = 1.0,
) -> dict[str, Any]:
    """Compare baseline, intercept-only, and causal residual-state predictions.

    Chronological expanding folds prevent future outcomes from entering either
    coefficient fitting or the causal residual feature used for a test trial.
    The state model adds one coefficient to an offset-logistic recalibration of
    the fitted model probability.
    """

    outcome, probability, mask = _binary_inputs(observed, predicted, valid_mask)
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    ridge = float(ridge)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
    valid_indices = np.flatnonzero(mask)
    if valid_indices.size < n_folds * 4:
        raise ValueError(
            "forward residual-state probe requires at least four valid trials per fold"
        )
    chunks = [chunk for chunk in np.array_split(valid_indices, n_folds) if chunk.size]
    if len(chunks) < 2:
        raise ValueError("forward residual-state probe has no evaluation fold")
    feature = causal_residual_state_feature(
        outcome,
        probability,
        mask,
        window_size=int(window_size),
    )
    offset = logit(probability)
    fold_index = np.full(outcome.size, -1, dtype=int)
    intercept_probability = np.full(outcome.size, np.nan, dtype=float)
    state_probability = np.full(outcome.size, np.nan, dtype=float)
    intercept_coefficients: list[float] = []
    state_coefficients: list[float] = []
    evaluation_indices: list[np.ndarray] = []
    for fold in range(1, len(chunks)):
        train_indices = np.concatenate(chunks[:fold])
        test_indices = chunks[fold]
        intercept_fit = _fit_offset_logistic(
            offset,
            feature,
            outcome,
            train_indices,
            include_state=False,
            ridge=ridge,
        )
        state_fit = _fit_offset_logistic(
            offset,
            feature,
            outcome,
            train_indices,
            include_state=True,
            ridge=ridge,
        )
        intercept_probability[test_indices] = expit(
            offset[test_indices] + intercept_fit[0]
        )
        state_probability[test_indices] = expit(
            offset[test_indices]
            + state_fit[0]
            + state_fit[1] * feature[test_indices]
        )
        fold_index[test_indices] = int(fold)
        intercept_coefficients.append(float(intercept_fit[0]))
        state_coefficients.append(float(state_fit[1]))
        evaluation_indices.append(test_indices)
    evaluation = np.concatenate(evaluation_indices)

    def mean_nll(values: np.ndarray) -> float:
        clipped = np.clip(
            values[evaluation],
            _PROBABILITY_EPSILON,
            1.0 - _PROBABILITY_EPSILON,
        )
        target = outcome[evaluation]
        return float(
            np.mean(-(target * np.log(clipped) + (1.0 - target) * np.log(1.0 - clipped)))
        )

    baseline_nll = mean_nll(probability)
    intercept_nll = mean_nll(intercept_probability)
    state_nll = mean_nll(state_probability)
    return {
        "state_feature": feature,
        "fold_index": fold_index,
        "intercept_probability": intercept_probability,
        "state_probability": state_probability,
        "n_evaluation_trials": int(evaluation.size),
        "baseline_nll": float(baseline_nll),
        "intercept_nll": float(intercept_nll),
        "state_nll": float(state_nll),
        "intercept_minus_baseline_nll": float(intercept_nll - baseline_nll),
        "state_minus_intercept_nll": float(state_nll - intercept_nll),
        "state_minus_baseline_nll": float(state_nll - baseline_nll),
        "intercept_coefficients": intercept_coefficients,
        "state_coefficients": state_coefficients,
        "window_size": int(window_size),
        "n_folds": int(len(chunks)),
        "ridge": float(ridge),
    }


__all__ = [
    "bernoulli_calibration_test",
    "causal_residual_state_feature",
    "forward_residual_state_probe",
    "logit_intercept_recalibration",
    "martingale_lag_tests",
    "rolling_martingale_z",
    "switch_residual_test",
]
