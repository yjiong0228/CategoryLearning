"""Small, explicit diagnostics for provisional oral-reason development.

The split depends only on trial order. Oral observation helpers can reweight
fixed cognitive paths and their local execution distributions. Geometric
ceilings inspect the catalogue and describe its possible expression range.
"""
from __future__ import annotations

import json
from typing import Sequence

import numpy as np


def oral_split(length: int, block_trials: int = 32, buffer_trials: int = 4) -> np.ndarray:
    """Alternate calibration and assessment blocks, excluding both edges."""
    if length < 1 or block_trials < 2 or not 0 <= buffer_trials < block_trials / 2:
        raise ValueError('Invalid oral block split')
    index = np.arange(length)
    block, offset = np.divmod(index, block_trials)
    result = np.where(block % 2 == 0, 'calibration', 'assessment')
    # The final partial block also has an actual, rather than nominal, edge.
    width = np.minimum(block_trials, length - block * block_trials)
    result[(offset < buffer_trials) | (offset >= width - buffer_trials)] = 'buffer'
    return result


def review_indices(length: int, count: int) -> np.ndarray:
    """Sample evenly in trial order before looking at alignment scores."""
    if length < 1 or count < 1:
        raise ValueError('Require positive record and sample sizes')
    return np.unique(np.rint(np.linspace(0, length - 1, min(count, length))).astype(int))


def facet_rows(token: str) -> np.ndarray:
    """Decode the shared canonical numeric inequality format A x <= b."""
    if not token.startswith('facet:'):
        raise ValueError('A numeric facet is required')
    rows = np.asarray(json.loads(token.split(':', 1)[1]), float)
    if rows.ndim != 2 or rows.shape[1] != 5 or not len(rows) or not np.isfinite(rows).all():
        raise ValueError('Invalid four-dimensional facet')
    return rows


def token_truth(tokens: Sequence[str], stimulus: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    """Check coded statements against physical stimuli, preserving ambiguity."""
    x = np.asarray(stimulus, float)
    if x.shape != (4,) or not np.isfinite(x).all():
        raise ValueError('Require a finite four-feature stimulus')
    return np.array([bool(np.all(facet_rows(t)[:, :4] @ x <= facet_rows(t)[:, 4] + tolerance))
                     if t.startswith('facet:') else np.nan for t in tokens], float)


def token_feature_mask(token: str) -> np.ndarray:
    """Features explicitly involved in a numeric statement, without its truth."""
    return np.any(np.abs(facet_rows(token)[:, :4]) > 1e-12, axis=0)


def categorical_log_score(probability: np.ndarray, observed: Sequence[Sequence[int]]) -> np.ndarray:
    """Mean log probability of a randomly selected coded atom per report.

    Each row is one categorical prediction over the fixed atom vocabulary.
    Several atoms from one report are averaged so a verbose report has the
    same weight as a brief report. This is not a joint sentence likelihood.
    """
    p = np.asarray(probability, float)
    if p.ndim != 2 or len(p) != len(observed) or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError('Require positive aligned categorical probabilities')
    np.testing.assert_allclose(p.sum(1), 1., atol=1e-10, rtol=0)
    result = np.full(len(p), np.nan)
    for i, ids in enumerate(observed):
        if len(ids):
            if min(ids) < 0 or max(ids) >= p.shape[1]:
                raise ValueError('Observed atom is outside the vocabulary')
            result[i] = np.log(p[i, list(ids)]).mean()
    return result


def normalized_exponential(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Normalize scores across the last axis; temperature zero is uniform."""
    scores = np.asarray(scores, float)
    if not np.isfinite(scores).all() or not np.isfinite(temperature) or temperature < 0:
        raise ValueError('Invalid categorical scores or temperature')
    logits = temperature * scores
    logits -= logits.max(axis=-1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=-1, keepdims=True)


def subset_log_probability(weights: np.ndarray, observed: Sequence[int]) -> np.ndarray:
    """Probability of a distinct unordered subset conditional on its size.

    P(S | |S|=k) is product(weights[S]) divided by the kth elementary
    symmetric polynomial of all vocabulary weights. The dynamic program
    sums every size-k subset, avoiding an unnormalized product score.
    """
    w = np.asarray(weights, float)
    ids = np.asarray(observed, int)
    if (w.ndim != 2 or not len(ids) or len(set(ids)) != len(ids)
            or min(ids) < 0 or max(ids) >= w.shape[1]
            or not np.isfinite(w).all() or (w <= 0).any()):
        raise ValueError('Require positive weights and a nonempty distinct subset')
    w = w / w.sum(1, keepdims=True)
    polynomial = np.zeros((len(w), len(ids)+1))
    polynomial[:, 0] = 1.
    for j in range(w.shape[1]):
        polynomial[:, 1:] += w[:, j, None] * polynomial[:, :-1]
    result = np.log(w[:, ids]).sum(1) - np.log(polynomial[:, -1])
    if not np.isfinite(result).all() or (result > 1e-10).any():
        raise ValueError('Invalid normalized subset probability')
    return result


def anchored_weights(weights: np.ndarray, likelihood: np.ndarray) -> tuple[np.ndarray, float]:
    """Reweight coherent terminal paths by one normalized oral likelihood."""
    w, likelihood = np.asarray(weights, float), np.asarray(likelihood, float)
    if w.ndim != 1 or likelihood.shape != w.shape or (w < 0).any() or (likelihood <= 0).any():
        raise ValueError('Invalid terminal path weights or oral likelihood')
    np.testing.assert_allclose(w.sum(), 1., atol=1e-10, rtol=0)
    if not np.isfinite(w).all() or not np.isfinite(likelihood).all() or (likelihood > 1+1e-10).any():
        raise ValueError('Non-probability oral observation')
    evidence = float(w @ likelihood)
    return w*likelihood/evidence, evidence


def condition_component_distribution(component: np.ndarray, likelihood: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Condition both mixture identity and particle evidence on an oral event."""
    component, likelihood = np.asarray(component, float), np.asarray(likelihood, float)
    if (component.ndim != 3 or likelihood.shape != component.shape or
            not np.isfinite(component).all() or not np.isfinite(likelihood).all() or
            (component < 0).any() or (likelihood <= 0).any() or (likelihood > 1+1e-10).any()):
        raise ValueError('Invalid component distribution or oral likelihood')
    np.testing.assert_allclose(component.sum((1, 2)), 1., atol=1e-10, rtol=0)
    joint = component*likelihood
    evidence = joint.sum((1, 2))
    return joint/evidence[:, None, None], evidence
