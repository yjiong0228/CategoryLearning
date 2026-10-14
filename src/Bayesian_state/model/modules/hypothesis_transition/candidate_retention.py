"""Exact sampling of nonempty candidate sets with a finite capacity.

A set has weight exp(sum of its member scores). Dynamic programming sums over
all allowed sets without enumerating them; storage is O(candidate_count * capacity).
The returned marginal describes inclusion of the first candidate before sampling.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp


def subset_distribution(scores: np.ndarray, capacity: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Return suffix log partitions, cardinality probabilities and first marginal."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError('Candidate scores must be a finite nonempty vector')
    if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
        raise ValueError('Capacity must be a positive integer')
    n, cap = len(scores), min(int(capacity), len(scores))
    suffix = np.full((n+1, cap+1), -np.inf)
    suffix[:, 0] = 0.
    for i in range(n-1, -1, -1):
        suffix[i, 1:] = np.logaddexp(suffix[i+1, 1:], scores[i]+suffix[i+1, :-1])
    normalizer = logsumexp(suffix[0, 1:])
    cardinality = np.exp(suffix[0, 1:]-normalizer)
    marginal = float(np.clip(np.exp(scores[0]+logsumexp(suffix[1, :cap])-normalizer), 0., 1.))
    return suffix, cardinality, marginal


def sample_candidate_set(scores: np.ndarray, capacity: int, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Sample a set exactly, conditional on size in [1, capacity]."""
    scores = np.asarray(scores, dtype=float)
    suffix, cardinality, marginal = subset_distribution(scores, capacity)
    remaining = int(rng.choice(len(cardinality), p=cardinality))+1
    selected = []
    for i in range(len(scores)):
        if remaining == 0:
            break
        if len(scores)-i == remaining:
            selected.extend(range(i, len(scores)))
            break
        probability = float(np.clip(np.exp(scores[i]+suffix[i+1, remaining-1]-suffix[i, remaining]), 0., 1.))
        if rng.random() < probability:
            selected.append(i)
            remaining -= 1
    return np.asarray(selected, dtype=int), marginal
