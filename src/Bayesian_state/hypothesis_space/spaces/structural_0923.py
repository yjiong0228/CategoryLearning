"""The adopted, finite feature-overlap extension for Model 0923."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .continuous import (
    ContinuousHypothesisSpace,
    _SpaceBuilder,
    build_continuous_hypothesis_space,
)

AXIS_PAIR_OVERLAP_EXTENSION = "axis_pair_overlap_0923"
AXIS_PAIR_OVERLAP_VERSION = "continuous_axis_pair_overlap_0923_v1"
# Preserve the original structural-probe import and catalogue signature.
OVERLAP_PROBE_VERSION = AXIS_PAIR_OVERLAP_VERSION


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
