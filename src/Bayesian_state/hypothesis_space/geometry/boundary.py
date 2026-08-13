"""Boundary-distance realization of a shared continuous hypothesis space."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ...utils.numeric import softmax
from ..spaces import CategoryRegion, ContinuousHypothesisSpace, Polytope
from .stimuli import as_stimuli


class BoundaryGeometry:
    """Compute category distance by projecting onto category regions."""

    def __init__(self, hypothesis_space: ContinuousHypothesisSpace) -> None:
        self.space = hypothesis_space

    @staticmethod
    def _project_to_halfspace(point: np.ndarray, normal: np.ndarray, bound: float) -> np.ndarray:
        normal = np.asarray(normal, dtype=float)
        excess = float(np.dot(normal, point) - bound)
        if excess <= 0.0:
            return point
        return point - excess / (float(np.dot(normal, normal)) + 1e-9) * normal

    @classmethod
    def _project_to_polytope(
        cls,
        point: np.ndarray,
        polytope: Polytope,
        n_iter: int = 100,
    ) -> np.ndarray:
        projected = np.asarray(point, dtype=float).copy()
        corrections = [np.zeros_like(projected) for _ in range(len(polytope.A))]
        for _ in range(int(n_iter)):
            for index, (normal, bound) in enumerate(zip(polytope.A, polytope.b)):
                candidate = projected + corrections[index]
                updated = cls._project_to_halfspace(candidate, normal, float(bound))
                corrections[index] = candidate - updated
                projected = updated
            projected = np.clip(projected, 0.0, 1.0)
        return projected

    @classmethod
    def distance_to_polytope(cls, point: np.ndarray, polytope: Polytope) -> float:
        point = np.asarray(point, dtype=float)
        if np.all(polytope.A @ point - polytope.b <= 1e-9):
            return 0.0
        projection = cls._project_to_polytope(point, polytope)
        return float(np.linalg.norm(point - projection))

    @classmethod
    def distance_to_category(cls, point: np.ndarray, category: CategoryRegion) -> float:
        return min(
            cls.distance_to_polytope(point, component)
            for component in category.components
        )

    def category_distances(self, hypo: int, stimuli: np.ndarray | Iterable) -> np.ndarray:
        values = as_stimuli(stimuli, self.space.n_dims)
        categories = self.space[int(hypo)].categories
        distances = np.empty((values.shape[0], len(categories)), dtype=float)
        for category_index, category in enumerate(categories):
            for trial_index, point in enumerate(values):
                distances[trial_index, category_index] = self.distance_to_category(
                    point,
                    category,
                )
        return distances

    def category_probabilities(
        self,
        hypo: int,
        stimuli: np.ndarray | Iterable,
        beta: float,
    ) -> np.ndarray:
        distances = self.category_distances(hypo, stimuli)
        return softmax(distances.T, -float(beta), axis=0)

    def category_assignments(
        self,
        hypo: int,
        stimuli: np.ndarray | Iterable,
        tol: float = 1e-9,
    ) -> np.ndarray:
        """Assign by membership, with deterministic handling of boundary ties."""
        values = as_stimuli(stimuli, self.space.n_dims)
        categories = self.space[int(hypo)].categories
        assignments = np.full(values.shape[0], -1, dtype=int)

        for row_index, point in enumerate(values):
            inside_categories: list[int] = []
            violation_scores: list[float] = []
            for category_index, category in enumerate(categories):
                component_scores = []
                inside = False
                for component in category.components:
                    violations = component.A @ point - component.b
                    component_scores.append(float(np.clip(violations, 0.0, None).sum()))
                    inside = inside or bool(np.all(violations <= tol))
                violation_scores.append(min(component_scores))
                if inside:
                    inside_categories.append(category_index)

            if inside_categories:
                assignments[row_index] = min(
                    inside_categories,
                    key=lambda category_index: violation_scores[category_index],
                )
            else:
                assignments[row_index] = int(
                    np.argmin(
                        [self.distance_to_category(point, category) for category in categories]
                    )
                )
        return assignments
