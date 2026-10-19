"""Evaluate stimulus contrasts with each recorded decision state held fixed.

Rule identities, response contributions, precision and particle weights keep
their original values. Only the supplied perceived stimulus changes. Callers
provide the source model's category-distance and softmax implementations; the
utility performs no learning, rule selection, perception sampling or reporting.
The supported readout is fixed-label, boundary-distance choice without lapse.
Reconstruct the saved unperturbed predictions before interpreting any contrast.

Single-feature curves describe sensitivity under the specified interventions.
They do not by themselves score a verbal explanation: redundant reasons and
interactions require joint contrasts, and effect size depends on the contrast.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def fixed_rule_readout(
    rule_ids: np.ndarray,
    contributions: np.ndarray,
    beta: np.ndarray,
    stimuli: np.ndarray,
    *,
    category_distances: Callable,
    softmax: Callable,
) -> np.ndarray:
    """Return particle × probe × category probabilities at fixed rule states.

    ``stimuli`` has particle × probe × feature axes. The other arrays retain
    particle × active-slot axes; padding has rule ID -1 and zero contribution.
    ``category_distances(rule, stimuli)`` returns stimulus × category distances.
    The supplied softmax is the original model's numerical implementation.
    """
    ids = np.asarray(rule_ids)
    u, b, x = (np.asarray(value, float) for value in (contributions, beta, stimuli))
    if (ids.ndim != 2 or not np.issubdtype(ids.dtype, np.integer)
            or u.shape != ids.shape or b.shape != ids.shape or x.ndim != 3
            or x.shape[0] != len(ids) or not x.shape[1] or not x.shape[2]
            or not all(np.isfinite(value).all() for value in (u, b, x))
            or (ids < -1).any() or (u < 0).any() or (b < 0).any()
            or (u[ids < 0] != 0).any() or (x < 0).any() or (x > 1).any()):
        raise ValueError('Require aligned finite decision states and unit-cube probes')
    np.testing.assert_allclose(u.sum(axis=1), 1., atol=1e-12, rtol=0)
    if any(len(set(row[row >= 0])) != (row >= 0).sum() for row in ids):
        raise ValueError('Active slots must contain distinct rule identities')
    result = None
    for rule in np.unique(ids[u > 0]):
        particles, slots = np.nonzero((ids == rule) & (u > 0))
        probes = x[particles].reshape(-1, x.shape[-1])
        # Reuse the same boundary geometry; distinct betas share its distances.
        unique, inverse = np.unique(probes, axis=0, return_inverse=True)
        distances = np.asarray(category_distances(int(rule), unique), float)[inverse]
        if (distances.ndim != 2 or distances.shape[0] != len(probes) or distances.shape[1] < 2
                or not np.isfinite(distances).all() or (distances < 0).any()):
            raise ValueError('Invalid category distances from the source model')
        precision = np.repeat(b[particles, slots], x.shape[1])
        probabilities = softmax(distances.T, -precision[None, :], axis=0).T
        probabilities = probabilities.reshape(len(particles), x.shape[1], distances.shape[1])
        if result is None:
            result = np.zeros((*x.shape[:2], distances.shape[1]))
        if probabilities.shape[-1] != result.shape[-1]:
            raise ValueError('Rules disagree on the category coordinate')
        result[particles] += u[particles, slots, None, None] * probabilities
    if result is None or not np.isfinite(result).all() or (result < 0).any():
        raise ValueError('Invalid fixed-state prediction')
    np.testing.assert_allclose(result.sum(axis=-1), 1., atol=1e-12, rtol=0)
    return result


def feature_grid(stimuli: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Keep probe zero unchanged, then vary each feature across the full grid."""
    x, grid = np.asarray(stimuli, float), np.asarray(values, float)
    if (x.ndim != 2 or grid.ndim != 1 or len(grid) < 2
            or not np.isfinite(x).all() or not np.isfinite(grid).all()
            or (x < 0).any() or (x > 1).any() or grid[0] != 0. or grid[-1] != 1.
            or (np.diff(grid) <= 0).any()):
        raise ValueError('Require unit-cube stimuli and an increasing grid from zero to one')
    probes = np.repeat(x[:, None, :], 1 + x.shape[1]*len(grid), axis=1)
    for feature in range(x.shape[1]):
        probes[:, 1+feature*len(grid):1+(feature+1)*len(grid), feature] = grid
    return probes


def summarize_feature_sweep(probability: np.ndarray, particle_weights: np.ndarray) -> dict[str, np.ndarray]:
    """Summarize particle × feature × grid choice probabilities without refitting.

    Particle-wise ranges are averaged before marginalization. ``pooled_range``
    also retains the range after pooling, which may be smaller when plausible
    states predict opposing effects. Positive/negative effects refer only to
    the zero-to-one endpoint contrast; nonmonotonic curves remain available.
    """
    q, weights = np.asarray(probability, float), np.asarray(particle_weights, float)
    if (q.ndim != 3 or q.shape[-1] < 2 or weights.shape != (q.shape[0],)
            or not np.isfinite(q).all() or not np.isfinite(weights).all()
            or (q < 0).any() or (q > 1+1e-12).any() or (weights < 0).any()):
        raise ValueError('Require finite aligned probability curves and particle weights')
    np.testing.assert_allclose(weights.sum(), 1., atol=1e-12, rtol=0)
    curve = np.einsum('p,pfg->fg', weights, q)
    difference = q[:, :, -1] - q[:, :, 0]
    return dict(curve=curve, mean_particle_range=weights @ np.ptp(q, axis=-1),
                pooled_range=np.ptp(curve, axis=-1), endpoint_effect=weights @ difference,
                positive_endpoint_effect=weights @ np.maximum(difference, 0),
                negative_endpoint_effect=weights @ np.maximum(-difference, 0))
