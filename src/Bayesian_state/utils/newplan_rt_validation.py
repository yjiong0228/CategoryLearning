"""Shared helpers for the condition-1 new-plan RT validation.

The RT analysis deliberately treats the already-computed B0 and D0 choice
predictions as frozen inputs.  Nothing in this module refits or selects the
choice model from reaction times.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


PROBABILITY_EPS = 1e-12


def normalized_probabilities(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Validate and normalize a two-dimensional probability matrix."""

    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2:
        raise ValueError(
            f"probabilities must be two-dimensional, got {values.shape}"
        )
    if values.shape[1] < 2:
        raise ValueError("probabilities must contain at least two categories")
    if not np.all(np.isfinite(values)):
        raise ValueError("probabilities contain non-finite values")
    if np.any(values < 0.0):
        raise ValueError("probabilities contain negative values")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("probability rows must have positive mass")
    return values / totals


def entropy_rows(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return Shannon entropy (natural-log units) for every probability row."""

    values = normalized_probabilities(probabilities)
    safe = np.clip(values, PROBABILITY_EPS, 1.0)
    return -np.sum(values * np.log(safe), axis=1)


def jensen_shannon_rows(
    first: Sequence[Sequence[float]] | np.ndarray,
    second: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return bounded, symmetric Jensen-Shannon divergence for paired rows."""

    left = normalized_probabilities(first)
    right = normalized_probabilities(second)
    if left.shape != right.shape:
        raise ValueError(
            f"probability matrices must match, got {left.shape} and {right.shape}"
        )
    mixture = 0.5 * (left + right)
    safe_left = np.clip(left, PROBABILITY_EPS, 1.0)
    safe_right = np.clip(right, PROBABILITY_EPS, 1.0)
    safe_mixture = np.clip(mixture, PROBABILITY_EPS, 1.0)
    return 0.5 * np.sum(
        left * (np.log(safe_left) - np.log(safe_mixture)), axis=1
    ) + 0.5 * np.sum(
        right * (np.log(safe_right) - np.log(safe_mixture)), axis=1
    )


def total_variation_rows(
    first: Sequence[Sequence[float]] | np.ndarray,
    second: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Return total-variation distance for paired predictive distributions."""

    left = normalized_probabilities(first)
    right = normalized_probabilities(second)
    if left.shape != right.shape:
        raise ValueError(
            f"probability matrices must match, got {left.shape} and {right.shape}"
        )
    return 0.5 * np.sum(np.abs(left - right), axis=1)


def robust_location_scale(
    values: Sequence[float] | np.ndarray,
) -> tuple[float, float]:
    """Return median and Gaussian-consistent MAD, with a safe fallback."""

    array = np.asarray(values, dtype=float).reshape(-1)
    array = array[np.isfinite(array)]
    if array.size == 0:
        raise ValueError("robust location/scale requires finite observations")
    location = float(np.median(array))
    scale = float(1.4826 * np.median(np.abs(array - location)))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.std(array, ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return location, scale


def subject_bootstrap_interval(
    values: Sequence[float] | np.ndarray,
    *,
    n_bootstrap: int = 20_000,
    seed: int = 20260819,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval for an equal-subject mean."""

    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be non-empty and finite")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(array, size=(int(n_bootstrap), array.size), replace=True)
    means = draws.mean(axis=1)
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def cr1_standard_error(
    exog: np.ndarray,
    residuals: np.ndarray,
    groups: Sequence[int] | np.ndarray,
    *,
    coefficient_index: int,
) -> float:
    """Small-sample corrected one-way cluster-robust standard error."""

    x = np.asarray(exog, dtype=float)
    u = np.asarray(residuals, dtype=float).reshape(-1)
    cluster = np.asarray(groups).reshape(-1)
    if x.ndim != 2 or x.shape[0] != u.size or u.size != cluster.size:
        raise ValueError("exog, residuals, and groups have incompatible shapes")
    n_observations, n_parameters = x.shape
    unique_groups = np.unique(cluster)
    n_groups = unique_groups.size
    if n_groups < 2:
        raise ValueError("cluster-robust inference requires at least two groups")
    bread = np.linalg.pinv(x.T @ x)
    meat = np.zeros((n_parameters, n_parameters), dtype=float)
    for value in unique_groups:
        mask = cluster == value
        score = x[mask].T @ u[mask]
        meat += np.outer(score, score)
    correction = (n_groups / (n_groups - 1.0)) * (
        (n_observations - 1.0) / max(n_observations - n_parameters, 1.0)
    )
    covariance = correction * bread @ meat @ bread
    variance = float(covariance[coefficient_index, coefficient_index])
    return float(np.sqrt(max(variance, 0.0)))
