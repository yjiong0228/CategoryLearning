"""Compare predicted and empirical distributions over complete oral reports.

The caller supplies a shared report vocabulary containing both model and observed
codes. An observed code outside the model vocabulary receives a zero prediction
column and remains in the empirical denominator. Each rolling comparison uses
the same scored reports on both sides, within a trailing window of actual trials.
The scores describe frequency agreement within windows; report order inside each
window is not represented.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _distribution(values: ArrayLike, *, name: str) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must contain real probabilities")
    array = np.asarray(values, dtype=float)
    if array.ndim == 0 or array.shape[-1] == 0:
        raise ValueError(f"{name} must have a nonempty vocabulary axis")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must contain finite, nonnegative probabilities")
    if not np.allclose(array.sum(axis=-1), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError(f"{name} must be normalized along its vocabulary axis")
    return array


def _distribution_pair(p: ArrayLike, q: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    predicted = _distribution(p, name="p")
    observed = _distribution(q, name="q")
    if predicted.shape != observed.shape:
        raise ValueError("p and q must have the same shape and vocabulary order")
    return predicted, observed


def _metric_result(values: np.ndarray) -> float | np.ndarray:
    return float(values) if values.ndim == 0 else values


def distribution_overlap(p: ArrayLike, q: ArrayLike) -> float | np.ndarray:
    """Return shared probability mass, ``sum(min(p, q)) = 1 - TV``.

    The last axis indexes complete report codes. One-dimensional inputs return a
    float; batched inputs return one value per distribution. The result is 1 for
    identical distributions and 0 for distributions with disjoint support.
    """
    predicted, observed = _distribution_pair(p, q)
    overlap = np.minimum(predicted, observed).sum(axis=-1)
    return _metric_result(np.clip(overlap, 0.0, 1.0))


def _js_similarity(predicted: np.ndarray, observed: np.ndarray) -> np.ndarray:
    log_predicted = np.full_like(predicted, -np.inf)
    log_observed = np.full_like(observed, -np.inf)
    np.log2(predicted, out=log_predicted, where=predicted > 0.0)
    np.log2(observed, out=log_observed, where=observed > 0.0)
    # A log-space mixture also handles the smallest positive floating values.
    log_mixture = np.logaddexp2(log_predicted, log_observed) - 1.0
    predicted_terms = np.zeros_like(predicted)
    observed_terms = np.zeros_like(observed)
    positive = predicted > 0.0
    predicted_terms[positive] = predicted[positive] * (
        log_predicted[positive] - log_mixture[positive]
    )
    positive = observed > 0.0
    observed_terms[positive] = observed[positive] * (
        log_observed[positive] - log_mixture[positive]
    )
    divergence = 0.5 * (
        predicted_terms.sum(axis=-1) + observed_terms.sum(axis=-1)
    )
    return np.clip(1.0 - divergence, 0.0, 1.0)


def js_similarity(p: ArrayLike, q: ArrayLike) -> float | np.ndarray:
    """Return ``1 - JSD(p, q)`` with base-2 Jensen--Shannon divergence.

    JSD is the mean KL divergence from each distribution to their midpoint.
    Zero-probability entries contribute zero. Identical distributions have
    similarity 1; disjoint supports have similarity 0. Inputs follow the same
    vocabulary-axis convention as :func:`distribution_overlap`.
    """
    predicted, observed = _distribution_pair(p, q)
    return _metric_result(_js_similarity(predicted, observed))


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def rolling_report_distributions(
    probabilities: ArrayLike,
    observed_indices: ArrayLike,
    score_mask: ArrayLike,
    *,
    window: int = 64,
    min_reports: int = 16,
) -> dict[str, np.ndarray]:
    """Compare distributions in each trailing window of actual trials.

    ``probabilities`` has shape ``(T, V)`` and every row is a normalized report
    distribution. ``observed_indices`` contains vocabulary indices or -1 for
    unencoded reports. ``score_mask`` is a Boolean vector selecting reports to
    compare; selected reports require a valid vocabulary index. Encoded reports
    may also be excluded by the mask, for example an excluded first trial.

    At trial ``t`` the window spans ``max(0, t-window+1)`` through ``t``. Both the
    model mean and observed frequencies use exactly its selected rows. Windows
    with fewer than ``min_reports`` have NaN distributions and similarities;
    ``n_reports`` records their count and ``valid`` records sufficient coverage.
    The observed distribution is an empirical estimate from these reports.
    A window can be valid even when its final trial has no encoded report.

    Returns ``model`` and ``observed`` arrays of shape ``(T, V)``, plus length-T
    ``n_reports``, ``valid``, ``overlap`` and ``js_similarity`` arrays. Similarities
    use every supplied vocabulary column, including observed-only report codes.
    """
    window = _positive_integer(window, name="window")
    min_reports = _positive_integer(min_reports, name="min_reports")
    if min_reports > window:
        raise ValueError("min_reports cannot exceed the trial window")
    predicted = _distribution(probabilities, name="probabilities")
    if predicted.ndim != 2:
        raise ValueError("probabilities must have shape (trials, report codes)")
    n_trials, n_codes = predicted.shape
    indices = np.asarray(observed_indices)
    mask = np.asarray(score_mask)
    if indices.shape != (n_trials,) or mask.shape != (n_trials,):
        raise ValueError("observed_indices and score_mask must match the trial count")
    if mask.dtype.kind != "b":
        raise ValueError("score_mask must be Boolean")
    if indices.dtype.kind not in "iuf" or not np.all(np.isfinite(indices)):
        raise ValueError("observed_indices must contain finite integer indices")
    if not np.all(indices == np.floor(indices)):
        raise ValueError("observed_indices must contain integer indices")
    if np.any(indices < -1) or np.any(indices >= n_codes):
        raise ValueError("observed_indices must be -1 or a vocabulary index")
    if np.any(mask & (indices < 0)):
        raise ValueError("selected reports must have an encoded vocabulary index")
    indices = indices.astype(np.intp)

    end = np.arange(1, n_trials + 1)
    start = np.maximum(0, end - window)
    count_prefix = np.concatenate(([0], np.cumsum(mask, dtype=np.int64)))
    n_reports = count_prefix[end] - count_prefix[start]
    valid = n_reports >= min_reports
    model = np.full((n_trials, n_codes), np.nan)
    observed = np.full((n_trials, n_codes), np.nan)
    overlap = np.full(n_trials, np.nan)
    similarity = np.full(n_trials, np.nan)

    if np.any(valid):
        model_prefix = np.zeros((n_trials + 1, n_codes))
        np.cumsum(predicted * mask[:, None], axis=0, out=model_prefix[1:])
        report_counts = np.zeros((n_trials + 1, n_codes), dtype=np.int64)
        rows = np.flatnonzero(mask)
        report_counts[rows + 1, indices[rows]] = 1
        np.cumsum(report_counts, axis=0, out=report_counts)
        denominator = n_reports[valid, None]
        model[valid] = (
            model_prefix[end[valid]] - model_prefix[start[valid]]
        ) / denominator
        observed[valid] = (
            report_counts[end[valid]] - report_counts[start[valid]]
        ) / denominator
        overlap[valid] = distribution_overlap(model[valid], observed[valid])
        similarity[valid] = js_similarity(model[valid], observed[valid])

    return {
        "model": model,
        "observed": observed,
        "n_reports": n_reports,
        "valid": valid,
        "overlap": overlap,
        "js_similarity": similarity,
    }


__all__ = ["distribution_overlap", "js_similarity", "rolling_report_distributions"]
