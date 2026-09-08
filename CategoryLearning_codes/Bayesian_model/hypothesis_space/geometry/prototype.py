"""Automatically derived prototype realization of a hypothesis space."""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Iterable

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ...utils.numeric import softmax
from ..spaces import ContinuousHypothesisSpace, Polytope
from .stimuli import as_stimuli


class PrototypeGeometry:
    """Use one volume-centroid prototype per connected category component."""

    METHOD_COMPONENT_VOLUME_CENTROID = "component_volume_centroid"
    VALID_METHODS = (METHOD_COMPONENT_VOLUME_CENTROID,)
    _cache: dict[tuple, tuple[tuple[tuple[np.ndarray, ...], ...], ...]] = {}

    def __init__(
        self,
        hypothesis_space: ContinuousHypothesisSpace,
        method: str = METHOD_COMPONENT_VOLUME_CENTROID,
    ) -> None:
        self.space = hypothesis_space
        self.method = str(method).strip().lower()
        if self.method not in self.VALID_METHODS:
            raise ValueError(
                f"Unsupported prototype method '{method}'. "
                f"Expected one of: {self.VALID_METHODS}."
            )
        cache_key = (self.space.signature, self.method)
        cached = self._cache.get(cache_key)
        if cached is None:
            cached = self._build_component_prototypes()
            self._cache[cache_key] = cached
        self.component_prototypes = cached
        self.prototypes, self.prototype_mask = self._build_padded_array(cached)

    @staticmethod
    def _bounded_constraints(polytope: Polytope) -> tuple[np.ndarray, np.ndarray]:
        identity = np.eye(polytope.n_dims, dtype=float)
        A = np.vstack([polytope.A, identity, -identity])
        b = np.concatenate([polytope.b, np.ones(polytope.n_dims), np.zeros(polytope.n_dims)])
        return A, b

    @classmethod
    def _vertices(cls, polytope: Polytope, tol: float = 1e-9) -> np.ndarray:
        """Enumerate vertices by intersecting active constraint sets."""
        A, b = cls._bounded_constraints(polytope)
        n_dims = polytope.n_dims
        vertices = []
        for active in combinations(range(A.shape[0]), n_dims):
            active_A = A[list(active)]
            if np.linalg.matrix_rank(active_A, tol=tol) < n_dims:
                continue
            try:
                point = np.linalg.solve(active_A, b[list(active)])
            except np.linalg.LinAlgError:
                continue
            if np.all(A @ point <= b + tol):
                vertices.append(point)
        if not vertices:
            raise ValueError("Cannot derive a prototype from an empty category component.")
        return np.unique(np.round(np.asarray(vertices, dtype=float), decimals=12), axis=0)

    @classmethod
    def component_centroid(cls, polytope: Polytope) -> np.ndarray:
        """Return the exact centroid from a boundary-simplex decomposition."""
        vertices = cls._vertices(polytope)
        n_dims = polytope.n_dims
        if n_dims == 1:
            return np.asarray([(vertices[:, 0].min() + vertices[:, 0].max()) / 2.0])
        if vertices.shape[0] == n_dims + 1:
            return vertices.mean(axis=0)

        reference = vertices.mean(axis=0)
        try:
            hull = ConvexHull(vertices)
        except QhullError as exc:
            raise ValueError("Category component is not full-dimensional.") from exc

        weighted_centroid = np.zeros(n_dims, dtype=float)
        total_volume = 0.0
        for facet in hull.simplices:
            facet_vertices = vertices[np.asarray(facet, dtype=int)]
            edge_matrix = (facet_vertices - reference).T
            volume = abs(float(np.linalg.det(edge_matrix))) / factorial(n_dims)
            if volume <= 1e-15:
                continue
            simplex_centroid = (reference + facet_vertices.sum(axis=0)) / (n_dims + 1)
            weighted_centroid += volume * simplex_centroid
            total_volume += volume
        if total_volume <= 0.0:
            raise ValueError("Could not obtain positive volume for a category component.")
        return np.clip(weighted_centroid / total_volume, 0.0, 1.0)

    def _build_component_prototypes(
        self,
    ) -> tuple[tuple[tuple[np.ndarray, ...], ...], ...]:
        return tuple(
            tuple(
                tuple(self.component_centroid(component) for component in category.components)
                for category in hypothesis.categories
            )
            for hypothesis in self.space
        )

    def _build_padded_array(
        self,
        component_prototypes: tuple[tuple[tuple[np.ndarray, ...], ...], ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        max_prototypes = max(
            len(category)
            for hypothesis in component_prototypes
            for category in hypothesis
        )
        values = np.empty(
            (len(self.space), max_prototypes, self.space.n_cats, self.space.n_dims),
            dtype=float,
        )
        mask = np.zeros((len(self.space), max_prototypes, self.space.n_cats), dtype=bool)
        for hypothesis_index, hypothesis in enumerate(component_prototypes):
            for category_index, category in enumerate(hypothesis):
                for prototype_index in range(max_prototypes):
                    source_index = min(prototype_index, len(category) - 1)
                    values[hypothesis_index, prototype_index, category_index] = category[source_index]
                    mask[hypothesis_index, prototype_index, category_index] = prototype_index < len(category)
        values.setflags(write=False)
        mask.setflags(write=False)
        return values, mask

    def get_category_prototypes(self, hypo: int, category: int) -> np.ndarray:
        return np.asarray(
            self.component_prototypes[int(hypo)][int(category)],
            dtype=float,
        ).copy()

    def category_distances(self, hypo: int, stimuli: np.ndarray | Iterable) -> np.ndarray:
        values = as_stimuli(stimuli, self.space.n_dims)
        distances = np.empty((values.shape[0], self.space.n_cats), dtype=float)
        for category_index in range(self.space.n_cats):
            prototypes = self.get_category_prototypes(hypo, category_index)
            distances[:, category_index] = np.linalg.norm(
                values[:, None, :] - prototypes[None, :, :],
                axis=2,
            ).min(axis=1)
        return distances

    def category_probabilities(
        self,
        hypo: int,
        stimuli: np.ndarray | Iterable,
        beta: float,
    ) -> np.ndarray:
        return softmax(self.category_distances(hypo, stimuli).T, -float(beta), axis=0)

    def category_assignments(self, hypo: int, stimuli: np.ndarray | Iterable) -> np.ndarray:
        return np.argmin(self.category_distances(hypo, stimuli), axis=1).astype(int)
