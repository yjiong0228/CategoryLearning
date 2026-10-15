"""Exact capacity competition after at most one new rule is proposed.

All rules are retained while the pool fits capacity. With K+1 candidates and
capacity K, retain a K-member set with weight exp(sum(member scores)). This
reduces exactly to dropping one rule with probability proportional to exp(-score).
An admitted proposal therefore replaces one old rule; a rejected proposal
leaves the old set unchanged. The pool cannot shrink during this competition.

This experimental sampler is installed only by the dedicated research worker.
The regular H implementation and its configuration remain unchanged pending
structural comparison. Scores, evidence updates, proposals and belief transfer
remain owned by the shared mechanistic transition.
"""
from __future__ import annotations

import numpy as np
from scipy.special import softmax


def drop_probabilities(scores: np.ndarray, capacity: int) -> np.ndarray:
    """Return one-drop probabilities, or all zeros when every rule fits."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Need finite nonempty candidate scores')
    if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
        raise ValueError('Capacity must be a positive integer')
    if len(values) > capacity + 1:
        raise ValueError('Competition supports at most one excess candidate')
    return np.zeros(len(values)) if len(values) <= capacity else softmax(-values)


def sample_competing_set(scores: np.ndarray, capacity: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Return retained indices and the first (new) candidate's inclusion chance."""
    probability = drop_probabilities(scores, capacity)
    indices = np.arange(len(probability), dtype=int)
    if len(indices) > capacity:
        dropped = int(rng.choice(len(indices), p=probability))
        indices = indices[indices != dropped]
    return indices, float(1 - probability[0])


def install_worker_intervention(transition_module) -> None:
    """Replace the J selector's two sampler dependencies in this process only.

The existing J transition consumes only the third distribution return value;
the unused partition table and cardinality vector are represented by None.
Research workers enforce refresh='search' and record this source separately.
"""
    def distribution(scores, capacity):
        probability = drop_probabilities(scores, capacity)
        return None, None, float(1 - probability[0])
    transition_module.subset_distribution = distribution
    transition_module.sample_candidate_set = sample_competing_set
