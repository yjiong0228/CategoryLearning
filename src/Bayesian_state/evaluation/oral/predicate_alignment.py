"""Equal-predicate content coverage under a behavior-derived basis distribution.

Each reported predicate receives the existing structural and functional verdict.
Their fractions describe partial content coverage, not a report probability.
The complete conjunction remains a separate, unchanged evaluation target.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .constraint_alignment import evaluate_component, validate_report_constraints
from .decision_basis_scoring import DOMAINS, LABELS


def predicate_fractions(tokens: Sequence[str], rule_A: np.ndarray,
                        rule_b: np.ndarray, *, tolerance: float = 1e-9) -> np.ndarray:
    """Return domain-by-verdict fractions for unique, jointly valid predicates."""
    if isinstance(tokens, str) or not tokens or len(set(tokens)) != len(tokens):
        raise ValueError('Require unique nonempty report predicates')
    validate_report_constraints(tokens, np.asarray(rule_A).shape[1], tolerance=tolerance)
    matches = [evaluate_component((token,), rule_A, rule_b, tolerance=tolerance)
               for token in tokens]
    return np.array([[sum(getattr(match, domain) == label for match in matches) / len(tokens)
                      for label in LABELS] for domain in DOMAINS])


def fractional_masses(mass: np.ndarray, fractions: np.ndarray) -> np.ndarray:
    """Apply the same component weights to every predicate, then average.

    ``fractions[t,h,m,d,k]`` sums to one over verdicts at each represented
    component and domain. Padding and unscored reports use NaN in every cell.
    Components carrying mass must be covered for both domains on scored trials.
    """
    mass, fractions = np.asarray(mass, float), np.asarray(fractions, float)
    if mass.ndim != 3 or fractions.shape != (*mass.shape, 2, 3):
        raise ValueError('Expected trial/rule/component masses and two domains, three verdicts')
    if not np.isfinite(mass).all() or (mass < 0).any():
        raise ValueError('Component masses must be finite and nonnegative')
    covered = np.isfinite(fractions).all(axis=(-1, -2))
    empty = np.isnan(fractions).all(axis=(-1, -2))
    if not (covered | empty).all() or (fractions[covered] < 0).any():
        raise ValueError('Each component needs complete fractions or NaN padding')
    np.testing.assert_allclose(fractions[covered].sum(-1), 1., atol=1e-12, rtol=0)
    valid = covered.any(axis=(1, 2))
    if (mass[valid][~covered[valid]] != 0).any():
        raise ValueError('A component carrying mass lacks predicate verdicts')
    result = np.einsum('trc,trcdk->tdk', mass, np.nan_to_num(fractions), optimize=True)
    np.testing.assert_allclose(result[valid].sum(-1),
                               np.repeat(mass[valid].sum((1, 2))[:, None], 2, axis=1),
                               atol=1e-12, rtol=0)
    result[~valid] = np.nan
    return result
