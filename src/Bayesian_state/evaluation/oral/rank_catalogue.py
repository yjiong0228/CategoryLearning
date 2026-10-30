"""Candidate rank partitions for expression diagnostics, outside active fits.

The 48 candidates include all label permutations of the longest and shortest
feature. They reuse the shared polytope representation. The default catalogue,
configuration validation, and cognitive learning code remain unchanged.
"""
from itertools import permutations

import numpy as np

from ...hypothesis_space.spaces.continuous import ContinuousHypothesisSpec
from ...hypothesis_space.spaces.regions import CategoryRegion, Polytope


def rank_hypotheses(offset: int = 152) -> tuple[ContinuousHypothesisSpec, ...]:
    """Four labels identify the extremal feature under each permutation."""
    axes = np.eye(4)
    result = []
    for kind, sign in [('longest', 1.), ('shortest', -1.)]:
        for order in permutations(range(4)):
            regions = tuple(CategoryRegion((Polytope(
                np.array([sign*(axes[j]-axes[i]) for j in range(4) if j != i]), np.zeros(3)),)) for i in order)
            index = offset+len(result)
            result.append(ContinuousHypothesisSpec(index=index, family='oral_probe_'+kind,
                hyperplanes=tuple((tuple(axes[i]-axes[j]), 0.) for i in range(4) for j in range(i+1, 4)),
                categories=regions, parameters={'kind':kind, 'feature_for_category':order,
                    'role':'expression_probe_not_adopted_cognitive_catalogue'},
                feedback_neighbors=((), (), (), ())))
    return tuple(result)
