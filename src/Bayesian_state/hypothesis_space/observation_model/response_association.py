"""Probability primitives for geometry-to-response associations.

Each geometrical rule retains its own identity. Its conditional association
distribution covers the 24 bijections from four geometrical regions to four
response categories. Condition 3 additionally retains the existing three
response-family pairings. Summing conditional evidence integrates these latent
associations while leaving the prior mass assigned to each geometry unchanged.

This experimental module is not installed in an engine configuration. Choosing
retention, transporting associations on rule entry, and using marginal evidence
in search/admission require an explicit model integration. The API consumes no
oral reports or true task categories.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np

from .pairing import pairing_feedback_kernel


# Entry [a, g] is the zero-based response associated with geometrical region g.
ASSOCIATIONS = np.asarray(list(permutations(range(4))), dtype=np.int64)
INVERSE_ASSOCIATIONS = np.argsort(ASSOCIATIONS, axis=1)
ASSOCIATIONS.setflags(write=False)
INVERSE_ASSOCIATIONS.setflags(write=False)


def _probabilities(values: np.ndarray, *, label: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if (not np.isfinite(result).all() or np.any(result < 0.)
            or not np.allclose(result.sum(axis=-1), 1., rtol=1e-10, atol=1e-12)):
        raise ValueError(f'{label} must be finite, nonnegative and normalized.')
    return result


def associated_probabilities(geometry_probabilities: np.ndarray) -> np.ndarray:
    """Map (rules, 4 regions) to (rules, 24 associations, 4 responses)."""
    p = np.asarray(geometry_probabilities, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] != 4:
        raise ValueError('Need probabilities shaped (rules, 4 geometrical regions).')
    p = _probabilities(p, label='Geometry probabilities')
    return p[:, INVERSE_ASSOCIATIONS]


def association_feedback_kernel(
    geometry_probabilities: np.ndarray, choice: int, feedback: float, *,
    hierarchical: bool = False,
) -> np.ndarray:
    """Return event probabilities shaped (rules, 24, 1 or 3 pairings).

Choice is one-indexed. Binary feedback uses the selected response or its
complement. Hierarchical feedback delegates to the existing pairing kernel,
preserving separate probabilities for feedback 0, 0.5 and 1.
"""
    p = associated_probabilities(geometry_probabilities)
    if hierarchical:
        return pairing_feedback_kernel(p.reshape(-1, 4), choice, feedback).reshape(p.shape[:2] + (3,))
    if (not np.isfinite(choice) or choice != int(choice) or not 1 <= int(choice) <= 4
            or feedback not in (0., 1.)):
        raise ValueError('Binary feedback needs choice in 1..4 and feedback 0 or 1.')
    selected = int(choice) - 1
    # Direct complement summation preserves tiny probabilities near certainty.
    event = p[..., selected] if feedback == 1. else p[..., np.arange(4) != selected].sum(axis=-1)
    return event[..., None]


def _conditional(values: np.ndarray) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    if (weights.ndim != 3 or weights.shape[0] == 0 or weights.shape[1] != 24
            or weights.shape[2] not in (1, 3)):
        raise ValueError('Conditional weights must have shape (rules, 24, 1 or 3).')
    _probabilities(weights.reshape(weights.shape[0], -1), label='Conditional association weights')
    return weights


def association_choice_probabilities(
    geometry_probabilities: np.ndarray, conditional_weights: np.ndarray,
) -> np.ndarray:
    """Integrate associations and pairings, returning one response vector per rule."""
    p = associated_probabilities(geometry_probabilities)
    weights = _conditional(conditional_weights)
    if weights.shape[0] != p.shape[0]:
        raise ValueError('Rule counts of probabilities and conditional weights differ.')
    return np.einsum('rac,ra->rc', p, weights.sum(axis=2))


def update_associations(
    conditional_weights: np.ndarray, kernel: np.ndarray, *, retention: float,
    association_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conditional posteriors and marginal evidence for each geometry.

Retention is applied to each geometry's normalized joint association/pairing
distribution. The effective distribution is normalized within that geometry;
its dispersion therefore does not reweight the geometry prior. Feedback is
applied once. Zero-evidence geometries retain their effective conditional state
and return evidence zero so that the caller can update the geometry marginal.
The input arrays are never mutated. This helper does not update rule beliefs.
    An optional Boolean mask of length 24 fixes the allowed associations. At
    retention zero only allowed states are reset, preserving a fixed-label
    comparison even at the memoryless boundary.
"""
    weights = _conditional(conditional_weights)
    likelihood = np.asarray(kernel, dtype=float)
    if (likelihood.shape != weights.shape or not np.isfinite(likelihood).all()
            or np.any(likelihood < 0.) or np.any(likelihood > 1.)):
        raise ValueError('Feedback kernel must match conditional weights and lie in [0, 1].')
    if not np.isfinite(retention) or not 0. <= retention <= 1.:
        raise ValueError('Association retention must lie in [0, 1].')
    allowed = np.ones(24, dtype=bool) if association_mask is None else np.asarray(association_mask)
    if allowed.shape != (24,) or allowed.dtype != np.bool_ or not allowed.any():
        raise ValueError('Association mask must be a nonempty Boolean vector of length 24.')
    if np.any(weights[:, ~allowed, :] != 0.):
        raise ValueError('Conditional mass outside the declared association support.')
    effective = (np.broadcast_to(allowed[None, :, None], weights.shape).astype(float)
                 if retention == 0. else weights ** retention)
    effective /= effective.sum(axis=(1, 2), keepdims=True)
    weighted = effective * likelihood
    evidence = weighted.sum(axis=(1, 2))
    posterior = np.divide(weighted, evidence[:, None, None], out=effective.copy(),
                          where=evidence[:, None, None] > 0.)
    return posterior, evidence
