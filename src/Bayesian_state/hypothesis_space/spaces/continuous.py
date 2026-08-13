"""Define and build the canonical continuous hypothesis space for Task2."""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Iterator, Sequence

import numpy as np

from .common import FrozenParameters
from .regions import CategoryRegion, Hyperplane, Polytope


@dataclass(frozen=True, eq=False)
class ContinuousHypothesisSpec:
    """One rule with fixed category labels and complete category geometry."""

    index: int
    family: str
    hyperplanes: tuple[Hyperplane, ...]
    categories: tuple[CategoryRegion, ...]
    parameters: Mapping[str, object]
    feedback_neighbors: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.parameters, FrozenParameters):
            object.__setattr__(
                self,
                "parameters",
                FrozenParameters.from_mapping(self.parameters),
            )


@dataclass(frozen=True, eq=False)
class ContinuousHypothesisSpace:
    """Ordered space shared by prototype and boundary implementations."""

    n_dims: int
    n_cats: int
    hypotheses: tuple[ContinuousHypothesisSpec, ...]
    version: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        hypotheses = tuple(self.hypotheses)
        if tuple(item.index for item in hypotheses) != tuple(range(len(hypotheses))):
            raise ValueError("Hypothesis indices must be contiguous and match list order.")
        for hypothesis in hypotheses:
            if len(hypothesis.categories) != int(self.n_cats):
                raise ValueError(
                    f"Hypothesis {hypothesis.index} has {len(hypothesis.categories)} "
                    f"categories; expected {self.n_cats}."
                )
        object.__setattr__(self, "hypotheses", hypotheses)
        if not isinstance(self.parameters, FrozenParameters):
            object.__setattr__(
                self,
                "parameters",
                FrozenParameters.from_mapping(self.parameters),
            )

    def __len__(self) -> int:
        return len(self.hypotheses)

    def __iter__(self) -> Iterator[ContinuousHypothesisSpec]:
        return iter(self.hypotheses)

    def __getitem__(self, index: int) -> ContinuousHypothesisSpec:
        return self.hypotheses[index]

    @property
    def signature(self) -> tuple:
        """Stable in-process cache key for derived geometry."""
        return (
            self.version,
            int(self.n_dims),
            int(self.n_cats),
            tuple(
                sorted(
                    (str(key), repr(value))
                    for key, value in self.parameters.items()
                )
            ),
        )


FAMILY_UNIVARIATE_THRESHOLD = "univariate_threshold"
FAMILY_PAIRWISE_ORDER = "pairwise_order"
FAMILY_PAIRWISE_SUM_THRESHOLD = "pairwise_sum_threshold"
FAMILY_PAIRED_SUM_ORDER = "paired_sum_order"
FAMILY_PAIRWISE_SIMILARITY_BAND = "pairwise_similarity_band"
FAMILY_UNIVARIATE_CENTER_BAND = "univariate_center_band"

BINARY_FAMILY_LABELS = {
    FAMILY_UNIVARIATE_THRESHOLD: "single feature below/above 0.5",
    FAMILY_PAIRWISE_ORDER: "one feature smaller/larger than another",
    FAMILY_PAIRWISE_SUM_THRESHOLD: "two-feature sum below/above 1",
    FAMILY_PAIRED_SUM_ORDER: "one feature-pair sum smaller/larger than another",
    FAMILY_PAIRWISE_SIMILARITY_BAND: "two features near/not near each other",
    FAMILY_UNIVARIATE_CENTER_BAND: "one feature near/not near 0.5",
}

SPACE_VERSION = "continuous_fixed_labels_v1"


def _plane(coefficients: Sequence[float], intercept: float) -> Hyperplane:
    return tuple(float(value) for value in coefficients), float(intercept)


def _polytope(rows: Iterable[Sequence[float]], bounds: Iterable[float]) -> Polytope:
    return Polytope(np.asarray(list(rows), dtype=float), np.asarray(list(bounds), dtype=float))


def _category(rows: Iterable[Sequence[float]], bounds: Iterable[float]) -> CategoryRegion:
    return CategoryRegion((_polytope(rows, bounds),))


def _feedback_neighbors(family: str, n_cats: int) -> tuple[tuple[int, ...], ...]:
    """Return the historical family-feedback topology without using prototypes.

    This is label topology, not metric geometry.  Keeping it explicit prevents
    automatic prototype placement from silently changing 0.5-feedback behavior.
    """
    if n_cats == 2:
        connected = family in {
            FAMILY_UNIVARIATE_THRESHOLD,
            FAMILY_UNIVARIATE_CENTER_BAND,
        }
        return ((1,), (0,)) if connected else ((), ())

    maps = {
        "2d_axis_pair": ((1, 2), (0, 3), (0, 3), (1, 2)),
        "2d_equality_sum": ((1,), (0,), (3,), (2,)),
        "3d_axis_equality": ((2,), (3,), (0,), (1,)),
        "3d_axis_sum": ((2,), (3,), (0,), (1,)),
        "3d_axis_triple": ((1,), (0,), (3,), (2,)),
        "4d_equality_axis_pair": ((1,), (0,), (3,), (2,)),
        "4d_sum_axis_pair": ((1,), (0,), (3,), (2,)),
    }
    return maps.get(family, ((), (), (), ()))


def _build_category_regions(
    family: str,
    hyperplanes: tuple[Hyperplane, ...],
    n_dims: int,
) -> tuple[CategoryRegion, ...]:
    """Translate one rule definition into its fixed-label category regions."""
    three_plane_families = {
        "3d_axis_triple",
        "3d_axis_equality_sum",
        "4d_equality_axis_pair",
        "4d_sum_axis_pair",
    }
    band_families = {
        FAMILY_PAIRWISE_SIMILARITY_BAND,
        FAMILY_UNIVARIATE_CENTER_BAND,
    }

    if family in band_families:
        difference_vector = np.asarray(hyperplanes[0][0], dtype=float)
        upper_bound = float(hyperplanes[0][1])
        negated_lower_bound = float(hyperplanes[1][1])
        lower_bound = -negated_lower_bound
        near = _category(
            [difference_vector, -difference_vector],
            [upper_bound, negated_lower_bound],
        )
        far = CategoryRegion(
            (
                _polytope([difference_vector], [lower_bound]),
                _polytope([-difference_vector], [-upper_bound]),
            )
        )
        return near, far

    if family not in three_plane_families and family not in {
        "dimension_max",
        "dimension_min",
    }:
        sign_orders = {
            FAMILY_UNIVARIATE_THRESHOLD: [(1,), (-1,)],
            FAMILY_PAIRWISE_ORDER: [(1,), (-1,)],
            FAMILY_PAIRWISE_SUM_THRESHOLD: [(1,), (-1,)],
            FAMILY_PAIRED_SUM_ORDER: [(1,), (-1,)],
            "2d_axis_pair": [(1, 1), (-1, 1), (1, -1), (-1, -1)],
            "2d_equality_sum": [(-1, 1), (1, -1), (-1, -1), (1, 1)],
            "3d_axis_equality": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
            "3d_axis_sum": [(1, 1), (1, -1), (-1, 1), (-1, -1)],
            "4d_equality_pair": [(1, 1), (-1, 1), (1, -1), (-1, -1)],
            "4d_sum_pair": [(1, 1), (-1, 1), (1, -1), (-1, -1)],
        }
        signs_iter = sign_orders.get(
            family,
            list(product([1, -1], repeat=len(hyperplanes))),
        )
        categories = []
        for signs in signs_iter:
            rows, bounds = [], []
            for (coefficients, intercept), sign in zip(hyperplanes, signs):
                rows.append(sign * np.asarray(coefficients, dtype=float))
                bounds.append(sign * intercept)
            categories.append(_category(rows, bounds))
        return tuple(categories)

    if family in three_plane_families:
        (a1, b1), (a2, b2), (a3, b3) = hyperplanes
        a1, a2, a3 = map(lambda value: np.asarray(value, dtype=float), (a1, a2, a3))
        return (
            _category([a1, a2], [b1, b2]),
            _category([a1, -a2], [b1, -b2]),
            _category([-a1, a3], [-b1, b3]),
            _category([-a1, -a3], [-b1, -b3]),
        )

    categories = []
    for category_index in range(n_dims):
        rows, bounds = [], []
        for other_index in range(n_dims):
            if category_index == other_index:
                continue
            coefficients = np.zeros(n_dims, dtype=float)
            if family == "dimension_max":
                coefficients[category_index], coefficients[other_index] = -1.0, 1.0
            else:
                coefficients[category_index], coefficients[other_index] = 1.0, -1.0
            rows.append(coefficients)
            bounds.append(0.0)
        categories.append(_category(rows, bounds))
    return tuple(categories)


class _SpaceBuilder:
    def __init__(self, n_dims: int, n_cats: int) -> None:
        self.n_dims = int(n_dims)
        self.n_cats = int(n_cats)
        self.hypotheses: list[ContinuousHypothesisSpec] = []

    def add(
        self,
        family: str,
        hyperplanes: Sequence[tuple[Sequence[float], float]],
        **parameters: object,
    ) -> None:
        normalized = tuple(_plane(coefficients, intercept) for coefficients, intercept in hyperplanes)
        categories = _build_category_regions(family, normalized, self.n_dims)
        self.hypotheses.append(
            ContinuousHypothesisSpec(
                index=len(self.hypotheses),
                family=family,
                hyperplanes=normalized,
                categories=categories,
                parameters=dict(parameters),
                feedback_neighbors=_feedback_neighbors(family, self.n_cats),
            )
        )


def _build_binary_space(
    builder: _SpaceBuilder,
    pairwise_similarity_tolerance: float,
    center_band_tolerance: float,
) -> None:
    n_dims = builder.n_dims

    for dimension in range(n_dims):
        coefficients = [0.0] * n_dims
        coefficients[dimension] = 1.0
        builder.add(
            FAMILY_UNIVARIATE_THRESHOLD,
            [(coefficients, 0.5)],
            dimension=dimension,
            threshold=0.5,
        )

    for first, second in itertools.combinations(range(n_dims), 2):
        coefficients = [0.0] * n_dims
        coefficients[first], coefficients[second] = 1.0, -1.0
        builder.add(
            FAMILY_PAIRWISE_ORDER,
            [(coefficients, 0.0)],
            dimensions=(first, second),
        )

    for first, second in itertools.combinations(range(n_dims), 2):
        coefficients = [0.0] * n_dims
        coefficients[first] = coefficients[second] = 1.0
        builder.add(
            FAMILY_PAIRWISE_SUM_THRESHOLD,
            [(coefficients, 1.0)],
            dimensions=(first, second),
            threshold=1.0,
        )

    if n_dims >= 4:
        for first, second, third, fourth in (
            (0, 1, 2, 3),
            (0, 2, 1, 3),
            (0, 3, 1, 2),
        ):
            coefficients = [0.0] * n_dims
            coefficients[first] = coefficients[second] = 1.0
            coefficients[third] = coefficients[fourth] = -1.0
            builder.add(
                FAMILY_PAIRED_SUM_ORDER,
                [(coefficients, 0.0)],
                dimension_pairs=((first, second), (third, fourth)),
            )

    # Append new families so historical indices 0--18 remain stable.
    for first, second in itertools.combinations(range(n_dims), 2):
        difference = [0.0] * n_dims
        difference[first], difference[second] = 1.0, -1.0
        builder.add(
            FAMILY_PAIRWISE_SIMILARITY_BAND,
            [(difference, pairwise_similarity_tolerance),
             ([-value for value in difference], pairwise_similarity_tolerance)],
            dimensions=(first, second),
            tolerance=pairwise_similarity_tolerance,
        )

    for dimension in range(n_dims):
        direction = [0.0] * n_dims
        direction[dimension] = 1.0
        builder.add(
            FAMILY_UNIVARIATE_CENTER_BAND,
            [(direction, 0.5 + center_band_tolerance),
             ([-value for value in direction], center_band_tolerance - 0.5)],
            dimension=dimension,
            center=0.5,
            tolerance=center_band_tolerance,
        )


def _build_four_category_space(builder: _SpaceBuilder) -> None:
    n_dims = builder.n_dims

    for first, second in itertools.combinations(range(n_dims), 2):
        plane1, plane2 = [0.0] * n_dims, [0.0] * n_dims
        plane1[first], plane2[second] = 1.0, 1.0
        builder.add("2d_axis_pair", [(plane1, 0.5), (plane2, 0.5)], dimensions=(first, second))

    for first, second in itertools.combinations(range(n_dims), 2):
        equality, summed = [0.0] * n_dims, [0.0] * n_dims
        equality[first], equality[second] = 1.0, -1.0
        summed[first] = summed[second] = 1.0
        builder.add("2d_equality_sum", [(equality, 0.0), (summed, 1.0)], dimensions=(first, second))

    if n_dims >= 3:
        for axis in range(n_dims):
            remaining = [dimension for dimension in range(n_dims) if dimension != axis]
            for first, second in itertools.combinations(remaining, 2):
                axis_plane, equality = [0.0] * n_dims, [0.0] * n_dims
                axis_plane[axis] = 1.0
                equality[first], equality[second] = 1.0, -1.0
                builder.add(
                    "3d_axis_equality",
                    [(axis_plane, 0.5), (equality, 0.0)],
                    axis_dimension=axis,
                    related_dimensions=(first, second),
                )

        for axis in range(n_dims):
            remaining = [dimension for dimension in range(n_dims) if dimension != axis]
            for first, second in itertools.combinations(remaining, 2):
                axis_plane, summed = [0.0] * n_dims, [0.0] * n_dims
                axis_plane[axis] = 1.0
                summed[first] = summed[second] = 1.0
                builder.add(
                    "3d_axis_sum",
                    [(axis_plane, 0.5), (summed, 1.0)],
                    axis_dimension=axis,
                    related_dimensions=(first, second),
                )

    if n_dims >= 4:
        pairings = ((0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2))
        for first, second, third, fourth in pairings:
            plane1, plane2 = [0.0] * n_dims, [0.0] * n_dims
            plane1[first], plane1[second] = 1.0, -1.0
            plane2[third], plane2[fourth] = 1.0, -1.0
            builder.add(
                "4d_equality_pair",
                [(plane1, 0.0), (plane2, 0.0)],
                dimension_pairs=((first, second), (third, fourth)),
            )

        for first, second, third, fourth in pairings:
            plane1, plane2 = [0.0] * n_dims, [0.0] * n_dims
            plane1[first] = plane1[second] = 1.0
            plane2[third] = plane2[fourth] = 1.0
            builder.add(
                "4d_sum_pair",
                [(plane1, 1.0), (plane2, 1.0)],
                dimension_pairs=((first, second), (third, fourth)),
            )

    if n_dims >= 3:
        for dimensions in itertools.combinations(range(n_dims), 3):
            for first, second, third in itertools.permutations(dimensions):
                planes = []
                for dimension in (first, second, third):
                    coefficients = [0.0] * n_dims
                    coefficients[dimension] = 1.0
                    planes.append((coefficients, 0.5))
                builder.add("3d_axis_triple", planes, ordered_dimensions=(first, second, third))

        for axis in range(n_dims):
            remaining = [dimension for dimension in range(n_dims) if dimension != axis]
            for first, second in itertools.combinations(remaining, 2):
                axis_plane, equality, summed = ([0.0] * n_dims for _ in range(3))
                axis_plane[axis] = 1.0
                equality[first], equality[second] = 1.0, -1.0
                summed[first] = summed[second] = 1.0
                builder.add(
                    "3d_axis_equality_sum",
                    [(axis_plane, 0.5), (equality, 0.0), (summed, 1.0)],
                    axis_dimension=axis,
                    related_dimensions=(first, second),
                    low_branch="equality",
                )
                builder.add(
                    "3d_axis_equality_sum",
                    [(axis_plane, 0.5), (summed, 1.0), (equality, 0.0)],
                    axis_dimension=axis,
                    related_dimensions=(first, second),
                    low_branch="sum",
                )

    if n_dims >= 4:
        for first, second in itertools.combinations(range(n_dims), 2):
            remaining = [dimension for dimension in range(n_dims) if dimension not in (first, second)]
            for third, fourth in itertools.combinations(remaining, 2):
                equality = [0.0] * n_dims
                equality[first], equality[second] = 1.0, -1.0
                axis1, axis2 = [0.0] * n_dims, [0.0] * n_dims
                axis1[third], axis2[fourth] = 1.0, 1.0
                builder.add(
                    "4d_equality_axis_pair",
                    [(equality, 0.0), (axis1, 0.5), (axis2, 0.5)],
                    related_dimensions=(first, second),
                    ordered_axis_dimensions=(third, fourth),
                )
                builder.add(
                    "4d_equality_axis_pair",
                    [(equality, 0.0), (axis2, 0.5), (axis1, 0.5)],
                    related_dimensions=(first, second),
                    ordered_axis_dimensions=(fourth, third),
                )

        for first, second in itertools.combinations(range(n_dims), 2):
            remaining = [dimension for dimension in range(n_dims) if dimension not in (first, second)]
            for third, fourth in itertools.combinations(remaining, 2):
                summed = [0.0] * n_dims
                summed[first] = summed[second] = 1.0
                axis1, axis2 = [0.0] * n_dims, [0.0] * n_dims
                axis1[third], axis2[fourth] = 1.0, 1.0
                builder.add(
                    "4d_sum_axis_pair",
                    [(summed, 1.0), (axis1, 0.5), (axis2, 0.5)],
                    related_dimensions=(first, second),
                    ordered_axis_dimensions=(third, fourth),
                )
                builder.add(
                    "4d_sum_axis_pair",
                    [(summed, 1.0), (axis2, 0.5), (axis1, 0.5)],
                    related_dimensions=(first, second),
                    ordered_axis_dimensions=(fourth, third),
                )

    if n_dims == builder.n_cats:
        equality_planes = []
        for first, second in itertools.combinations(range(n_dims), 2):
            coefficients = [0.0] * n_dims
            coefficients[first], coefficients[second] = 1.0, -1.0
            equality_planes.append((coefficients, 0.0))
        builder.add("dimension_max", equality_planes)
        builder.add("dimension_min", equality_planes)


def build_continuous_hypothesis_space(
    n_dims: int,
    n_cats: int,
    *,
    pairwise_similarity_tolerance: float = 0.10,
    center_band_tolerance: float = 0.10,
) -> ContinuousHypothesisSpace:
    """Build the ordered, fixed-label continuous hypothesis space."""
    n_dims = int(n_dims)
    n_cats = int(n_cats)
    if n_dims <= 0:
        raise ValueError(f"n_dims must be positive, got {n_dims}.")
    if n_cats not in {2, 4}:
        raise ValueError(f"Continuous hypothesis space supports 2 or 4 categories, got {n_cats}.")

    pair_tolerance = float(pairwise_similarity_tolerance)
    if not np.isfinite(pair_tolerance) or not 0.0 < pair_tolerance < 1.0:
        raise ValueError(
            "pairwise_similarity_tolerance must be finite and in (0, 1), "
            f"got {pairwise_similarity_tolerance!r}."
        )
    center_tolerance = float(center_band_tolerance)
    if not np.isfinite(center_tolerance) or not 0.0 < center_tolerance < 0.5:
        raise ValueError(
            "center_band_tolerance must be finite and in (0, 0.5), "
            f"got {center_band_tolerance!r}."
        )

    builder = _SpaceBuilder(n_dims, n_cats)
    if n_cats == 2:
        _build_binary_space(builder, pair_tolerance, center_tolerance)
    else:
        _build_four_category_space(builder)

    return ContinuousHypothesisSpace(
        n_dims=n_dims,
        n_cats=n_cats,
        hypotheses=tuple(builder.hypotheses),
        version=SPACE_VERSION,
        parameters={
            "pairwise_similarity_tolerance": pair_tolerance if n_cats == 2 else None,
            "center_band_tolerance": center_tolerance if n_cats == 2 else None,
        },
    )
