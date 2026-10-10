"""Versioned 128- and 152-rule structural catalogues for Model 0923."""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations

import numpy as np

from .continuous import (
    ContinuousHypothesisSpace,
    ContinuousHypothesisSpec,
    DEFAULT_PAIRWISE_SIMILARITY_TOLERANCE,
    FAMILY_PAIRWISE_SIMILARITY_BAND,
    _SpaceBuilder,
    build_continuous_hypothesis_space,
)
from .regions import CategoryRegion, Polytope

AXIS_PAIR_OVERLAP_EXTENSION = "axis_pair_overlap_0923"
AXIS_PAIR_OVERLAP_VERSION = "continuous_axis_pair_overlap_0923_v1"
# Preserve the original structural-probe import and catalogue signature.
OVERLAP_PROBE_VERSION = AXIS_PAIR_OVERLAP_VERSION
RELATIONAL_EXTENSION = "relational_152_0923"
RELATIONAL_VERSION = "continuous_relational_152_0923_v1"
FAMILY_SHARED_PAIR_COMPARISONS = "shared_pair_comparisons"
FAMILY_SIMILARITY_AXIS = "similarity_axis_disjoint"


def build_axis_pair_overlap_space() -> ContinuousHypothesisSpace:
    """Append exactly 12 four-category rules that reuse the threshold feature.

    Each new rule combines x_axis < .5 with x_i < x_j, where axis is one
    member of (i, j). Comparator orientation follows the existing catalogue's
    i < j dimension order, never a participant's report or selected category.
    Labels retain the ordinary axis-equality sign order. Model 0923 B0 v2
    adopts this catalogue without adding the B1 structural-search mechanism.
    """
    base = build_continuous_hypothesis_space(4, 4)
    builder = _SpaceBuilder(4, 4)
    axes = np.eye(4)
    for axis in range(4):
        for other in range(4):
            if other == axis:
                continue
            first, second = sorted((axis, other))
            # Reuse the canonical region builder and label convention. Only
            # the formerly disjoint feature sets are permitted to overlap.
            builder.add(
                "3d_axis_equality",
                [(axes[axis], 0.5), (axes[first] - axes[second], 0.0)],
                axis_dimension=axis,
                related_dimensions=(first, second),
                feature_overlap=True,
                source_family="3d_axis_equality",
            )
    offset = len(base)
    additions = tuple(
        replace(hypothesis, index=offset + index,
                base_hypothesis_index=offset + index, family="axis_pair_overlap")
        for index, hypothesis in enumerate(builder.hypotheses)
    )
    return ContinuousHypothesisSpace(
        n_dims=4, n_cats=4, hypotheses=base.hypotheses + additions,
        version=AXIS_PAIR_OVERLAP_VERSION,
        parameters={**dict(base.parameters), "parent_version": base.version,
                    "structural_extension": "axis_pair_overlap",
                    "added_rule_count": len(additions)},
    )


def build_axis_pair_overlap_probe_space() -> ContinuousHypothesisSpace:
    """Compatibility entry for the frozen R2 structural-check workflow."""
    return build_axis_pair_overlap_space()


def build_relational_0923_space(
    *, pairwise_similarity_tolerance: float = DEFAULT_PAIRWISE_SIMILARITY_TOLERANCE,
) -> ContinuousHypothesisSpace:
    """Preserve rules 0--127 and append two approved twelve-rule families.

    Shared comparisons use canonical i<j orientation and the existing paired
    comparison labels. Similarity is exactly the binary near/far region, crossed
    with an unused dimension's .5 threshold: low-near, low-far, high-near,
    high-far. The two far components are one category, not extra rules.
    """
    binary = build_continuous_hypothesis_space(
        4, 2, pairwise_similarity_tolerance=pairwise_similarity_tolerance,
    )
    tolerance = binary.parameters["pairwise_similarity_tolerance"]
    base = build_axis_pair_overlap_space()
    axes = np.eye(4)
    builder = _SpaceBuilder(4, 4)
    for center in range(4):
        for neighbors in combinations([d for d in range(4) if d != center], 2):
            pairs = [tuple(sorted((center, other))) for other in neighbors]
            builder.add(
                "4d_equality_pair",
                [(axes[i] - axes[j], 0.) for i, j in pairs],
                shared_dimension=center, comparison_dimensions=tuple(pairs),
            )
    hypotheses = list(base.hypotheses)
    for hypothesis in builder.hypotheses:
        hypotheses.append(replace(
            hypothesis, index=len(hypotheses), base_hypothesis_index=len(hypotheses),
            family=FAMILY_SHARED_PAIR_COMPARISONS,
        ))
    for band in binary:
        if band.family != FAMILY_PAIRWISE_SIMILARITY_BAND:
            continue
        pair = tuple(band.parameters["dimensions"])
        for axis in (d for d in range(4) if d not in pair):
            categories = []
            for sign in (1., -1.):
                for category in band.categories:
                    categories.append(CategoryRegion(tuple(
                        Polytope(np.vstack((sign * axes[axis], part.A)),
                                 np.r_[sign * .5, part.b])
                        for part in category.components
                    )))
            index = len(hypotheses)
            hypotheses.append(ContinuousHypothesisSpec(
                index=index, family=FAMILY_SIMILARITY_AXIS,
                hyperplanes=((tuple(axes[axis]), .5), *band.hyperplanes),
                categories=tuple(categories),
                parameters={"axis_dimension": axis, "related_dimensions": pair,
                            "threshold": .5, "tolerance": tolerance},
                # Four-class graded feedback uses learned pairing, never this
                # historical topology. New families infer no privileged pairing.
                feedback_neighbors=((), (), (), ()),
            ))
    return ContinuousHypothesisSpace(
        n_dims=4, n_cats=4, hypotheses=tuple(hypotheses), version=RELATIONAL_VERSION,
        parameters={**dict(base.parameters), "parent_version": base.version,
                    "structural_extension": RELATIONAL_EXTENSION,
                    "pairwise_similarity_tolerance": tolerance, "added_rule_count": 24},
    )
