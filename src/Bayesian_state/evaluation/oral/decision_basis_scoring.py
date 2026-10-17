"""Score observed report constraints using behavior-only local-basis masses.

Independent filters are pooled as joint masses before conditioning on the
current choice. Uncertainty here measures finite-filter simulation error.
"""
from __future__ import annotations

import numpy as np

DOMAINS = ('structural', 'functional')
LABELS = ('support', 'conflict', 'undetermined')


def pool_joint(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Pool seed-by-trial joint masses, then condition once per trial."""
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    if (denominator.ndim != 2 or numerator.shape[:2] != denominator.shape
            or numerator.ndim < 2 or not np.isfinite(numerator).all()
            or not np.isfinite(denominator).all() or np.any(numerator < 0)
            or np.any(denominator <= 0)):
        raise ValueError('Expected finite seed/trial masses and positive choice probabilities')
    scale = denominator.sum(axis=0).reshape(denominator.shape[1], *([1] * (numerator.ndim - 2)))
    return numerator.sum(axis=0) / scale


def label_masses(mass: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Return trial/domain/verdict mass; undefined report rows remain NaN.

    labels[t,h,m,d] is -1 for padding or unscored reports, else 0/1/2.
    Both domains must cover every component carrying nonzero mass.
    """
    mass, labels = np.asarray(mass, float), np.asarray(labels)
    if mass.ndim != 3 or labels.shape != (*mass.shape, 2):
        raise ValueError('Expected trial/rule/component mass and two match domains')
    if not np.isfinite(mass).all() or np.any(mass < 0) or not np.isin(labels, [-1, 0, 1, 2]).all():
        raise ValueError('Invalid component mass or match label')
    valid = np.any(labels >= 0, axis=(1, 2, 3))
    result = np.stack([np.stack([(mass * (labels[..., d] == k)).sum(axis=(1, 2))
                                 for k in range(3)], axis=1) for d in range(2)], axis=1)
    np.testing.assert_allclose(result[valid].sum(axis=2),
                               np.repeat(mass[valid].sum(axis=(1, 2))[:, None], 2, axis=1),
                               atol=1e-12, rtol=0)
    result[~valid] = np.nan
    return result


def support_rule_masks(labels: np.ndarray, domain: int) -> tuple[np.ndarray, np.ndarray]:
    """Lower/upper availability masks using all/any category components."""
    values = np.asarray(labels)[..., domain]
    valid_components = values >= 0
    any_component = np.any(values == 0, axis=2)
    all_components = np.all((values == 0) | ~valid_components, axis=2) & np.any(valid_components, axis=2)
    return all_components, any_component


def prefix_baseline(basis: np.ndarray, choices: np.ndarray, training_trials: int) -> np.ndarray:
    """Freeze a category-specific mean local basis after the training prefix.

    Each training trial contributes equally to its category. Test reports and
    test states never determine this baseline. Undefined categories stay NaN.
    It is a static representation benchmark, not a behavioral simulator.
    """
    basis, choices = np.asarray(basis, float), np.asarray(choices)
    if basis.ndim != 3 or choices.shape != (len(basis),) or not 0 <= training_trials <= len(basis):
        raise ValueError('Invalid prefix or choice identities')
    np.testing.assert_allclose(basis.sum(axis=(1, 2)), 1., atol=1e-12, rtol=0)
    result = np.full_like(basis, np.nan)
    for choice in np.unique(choices[:training_trials]):
        value = basis[:training_trials][choices[:training_trials] == choice].mean(axis=0)
        selected = (np.arange(len(choices)) >= training_trials) & (choices == choice)
        result[selected] = value
    return result


def pooled_mean_and_mcse(numerator: np.ndarray, denominator: np.ndarray,
                         mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean trial score and delete-one-filter jackknife Monte Carlo SE.

    Trials are retained together inside each independent filter replicate.
    The returned jackknife estimates support paired mechanism contrasts.
    """
    numerator, denominator, mask = np.asarray(numerator), np.asarray(denominator), np.asarray(mask, bool)
    if mask.shape != (denominator.shape[1],):
        raise ValueError('Trial mask differs from the probability sequence')
    shape = numerator.shape[2:]
    if not mask.any():
        return np.full(shape, np.nan), np.full(shape, np.nan), np.full((len(denominator), *shape), np.nan)
    selected, probabilities = numerator[:, mask], denominator[:, mask]
    point = pool_joint(selected, probabilities).mean(axis=0)
    if len(probabilities) < 2:
        return point, np.full(shape, np.nan), np.full((len(probabilities), *shape), np.nan)
    estimates = np.stack([pool_joint(np.delete(selected, i, axis=0), np.delete(probabilities, i, axis=0)).mean(axis=0)
                          for i in range(len(probabilities))])
    centered = estimates - estimates.mean(axis=0)
    se = np.sqrt((len(probabilities) - 1) / len(probabilities) * np.sum(centered ** 2, axis=0))
    return point, se, estimates
