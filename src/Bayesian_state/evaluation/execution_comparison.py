"""Sequence likelihood and numerical checks for nested execution models."""
from __future__ import annotations

import numpy as np

from ..metrics.trajectory_shape import sequence_score


def conditional_score(probabilities: np.ndarray, prefix: int = 0) -> dict:
    """Score a suffix using the ratio of full and prefix mean likelihoods.

    Independent particle systems are weighted by their prefix evidence. A
    simple average of suffix likelihoods discards that conditioning. The
    complete system is the delete-one jackknife unit in both terms.
    """
    full = sequence_score(probabilities)
    if not isinstance(prefix, int) or not 0 <= prefix < full['trials']:
        raise ValueError('Prefix must leave at least one evaluation trial')
    if prefix == 0:
        return full
    early = sequence_score(np.asarray(probabilities)[:, :prefix])
    n = full['trials'] - prefix
    deleted = (full['trials'] * np.array(full['deleted_nll'])
               - prefix * np.array(early['deleted_nll'])) / n
    r = full['repeats']
    se = float(np.sqrt((r - 1) / r * np.square(deleted - deleted.mean()).sum())) if r > 1 else None
    return dict(nll=(full['trials'] * full['nll'] - prefix * early['nll']) / n,
                mcse=se, trials=n, repeats=r, prefix_trials=prefix,
                deleted_nll=deleted.tolist(),
                maximum_system_weight=max(full['maximum_system_weight'], early['maximum_system_weight']),
                effective_systems=min(full['effective_systems'], early['effective_systems']),
                full_score=full, prefix_score=early)


def score_difference(a: np.ndarray, b: np.ndarray, prefix: int = 0) -> dict:
    """NLL(a) minus NLL(b), pairing entire independent filter systems."""
    if np.shape(a) != np.shape(b) or len(a) < 2:
        raise ValueError('Require paired repeat-by-trial arrays')
    x, y = conditional_score(a, prefix), conditional_score(b, prefix)
    deleted = np.array(x['deleted_nll']) - np.array(y['deleted_nll'])
    r = len(deleted)
    return dict(difference=x['nll'] - y['nll'],
                mcse=float(np.sqrt((r - 1) / r * np.square(deleted - deleted.mean()).sum())),
                trials=x['trials'], repeats=r)


def precision_check(low: dict, high: dict, policy: dict) -> dict:
    """Apply the registered numerical tolerances without a significance claim."""
    gap = abs(high['nll'] - low['nll'])
    bound = gap + 2 * float(np.hypot(low['mcse'], high['mcse']))
    passed = (bound <= policy['particle_tolerance']
              and 2 * max(low['mcse'], high['mcse']) <= policy['maximum_twice_mcse']
              and max(low['maximum_system_weight'], high['maximum_system_weight'])
              <= policy['maximum_system_weight'])
    return dict(passed=bool(passed), absolute_difference=gap, upper_numerical_difference=bound,
                tolerance=policy['particle_tolerance'])
