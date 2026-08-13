"""Canonical continuous and discrete hypothesis-space definitions."""

from .continuous import (
    BINARY_FAMILY_LABELS,
    FAMILY_PAIRED_SUM_ORDER,
    FAMILY_PAIRWISE_ORDER,
    FAMILY_PAIRWISE_SIMILARITY_BAND,
    FAMILY_PAIRWISE_SUM_THRESHOLD,
    FAMILY_UNIVARIATE_CENTER_BAND,
    FAMILY_UNIVARIATE_THRESHOLD,
    ContinuousHypothesisSpace,
    ContinuousHypothesisSpec,
    build_continuous_hypothesis_space,
)
from .discrete import (
    DiscreteHypothesisSpace,
    DiscreteHypothesisSpec,
    FAMILY_PARITY_RULE,
    build_discrete_hypothesis_space,
)
from .common import FrozenParameters
from .regions import CategoryRegion, Hyperplane, Polytope

__all__ = [
    "BINARY_FAMILY_LABELS",
    "CategoryRegion",
    "ContinuousHypothesisSpace",
    "ContinuousHypothesisSpec",
    "DiscreteHypothesisSpace",
    "DiscreteHypothesisSpec",
    "FAMILY_PAIRED_SUM_ORDER",
    "FAMILY_PAIRWISE_ORDER",
    "FAMILY_PAIRWISE_SIMILARITY_BAND",
    "FAMILY_PAIRWISE_SUM_THRESHOLD",
    "FAMILY_PARITY_RULE",
    "FAMILY_UNIVARIATE_CENTER_BAND",
    "FAMILY_UNIVARIATE_THRESHOLD",
    "FrozenParameters",
    "Hyperplane",
    "Polytope",
    "build_continuous_hypothesis_space",
    "build_discrete_hypothesis_space",
]
