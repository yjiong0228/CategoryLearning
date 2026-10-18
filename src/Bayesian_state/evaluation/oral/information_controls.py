"""Compare oral support across weighting and frozen-representation controls.

All three distributions use the observed choice to identify category regions.
The unweighted control removes only current-choice likelihood reweighting.
The prefix control also removes later state and stimulus-specific basis changes.
Neither control is an autonomous behavioral model.
"""
from __future__ import annotations

import numpy as np

from .decision_basis_scoring import label_masses, pool_joint, prefix_baseline

METRICS = ('conditioned', 'unweighted', 'prefix_static',
           'current_choice_gain', 'dynamic_static_gain')


def jackknife_error(values: np.ndarray) -> np.ndarray:
    """Monte Carlo SE from deleting one independent filter at a time."""
    values = np.asarray(values, float)
    if len(values) < 2:
        raise ValueError('Need at least two independent filters')
    return np.sqrt((len(values)-1)/len(values) *
                   np.sum((values-values.mean(axis=0))**2, axis=0))


def information_controls(joint: np.ndarray, probability: np.ndarray,
                         unweighted: np.ndarray, labels: np.ndarray,
                         choices: np.ndarray, cut: int,
                         masks: dict[str, np.ndarray]) -> dict:
    """Recompute the prefix benchmark within every delete-filter replicate.

The common report mask excludes categories absent from the prefix and records
that count explicitly. Requested reports must have complete catalogue labels
and occur after the prefix. Trial differences retain their sign.
"""
    joint, probability, unweighted = map(lambda x: np.asarray(x, float),
                                        (joint, probability, unweighted))
    if joint.ndim != 4 or joint.shape != unweighted.shape or len(joint) < 2:
        raise ValueError('Need matching seed/trial/rule/component arrays')
    if (probability.shape != joint.shape[:2] or not 0 < cut < joint.shape[1]
            or not np.isfinite(unweighted).all() or np.any(unweighted < 0)):
        raise ValueError('Invalid probability, prefix or unweighted masses')
    np.testing.assert_allclose(joint.sum(axis=(2, 3)), probability, atol=1e-12, rtol=0)
    np.testing.assert_allclose(unweighted.sum(axis=(2, 3)), 1., atol=1e-12, rtol=0)

    def distributions(keep):
        basis = pool_joint(joint[keep], probability[keep])
        current = label_masses(basis, labels)
        prior = label_masses(unweighted[keep].mean(axis=0), labels)
        static = prefix_baseline(basis, choices, cut)
        valid = np.isfinite(static).all(axis=(1, 2))
        frozen = np.full_like(current, np.nan)
        if valid.any():
            frozen[valid] = label_masses(static[valid], labels[valid])
        return np.stack([current, prior, frozen, current-prior, current-frozen])

    seeds = np.arange(len(joint))
    values = distributions(seeds)
    available = np.isfinite(values).all(axis=(0, 2, 3))
    windows = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, bool)
        if mask.shape != (joint.shape[1],) or mask[:cut].any():
            raise ValueError('Static comparisons require later report masks')
        if not np.isfinite(values[:2, mask]).all():
            raise ValueError('Requested reports lack catalogue labels')
        common = mask & available
        windows[name] = dict(mask=common, requested_reports=int(mask.sum()),
                             reports=int(common.sum()), excluded_reports=int((mask & ~available).sum()))
    deleted = [distributions(seeds[seeds != i]) for i in seeds]
    for result in windows.values():
        mask = result['mask']
        if mask.any():
            result['point'] = values[:, mask].mean(axis=1)
            result['jackknife'] = np.stack([item[:, mask].mean(axis=1) for item in deleted])
            result['mcse'] = jackknife_error(result['jackknife'])
        else:
            result['point'] = np.full((5, 2, 3), np.nan)
            result['jackknife'] = np.full((len(seeds), 5, 2, 3), np.nan)
            result['mcse'] = np.full((5, 2, 3), np.nan)
    return dict(trials=values, windows=windows)
