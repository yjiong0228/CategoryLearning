"""Boundary-distance realization of a shared continuous hypothesis space."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from typing import Iterable

import numpy as np

from ...utils.numeric import softmax
from ..spaces import CategoryRegion, ContinuousHypothesisSpace, Polytope
from .stimuli import as_stimuli


class BoundaryProjectionError(RuntimeError):
    """Raised when a boundary solver cannot find a feasible projection."""


@dataclass(frozen=True)
class _ActiveSetGroup:
    constraints: np.ndarray
    bounds: np.ndarray
    gram_inverses: np.ndarray


@dataclass(frozen=True)
class _CompiledPolytope:
    constraints: np.ndarray
    bounds: np.ndarray
    active_set_groups: tuple[_ActiveSetGroup, ...]
    signature: str


class BoundaryGeometry:
    """Compute Euclidean distance to category regions inside the unit cube."""

    METHOD_DYKSTRA = "dykstra_iterative_projection"
    METHOD_KKT_ACTIVE_SET = "kkt_active_set_projection"
    VALID_METHODS = (METHOD_DYKSTRA, METHOD_KKT_ACTIVE_SET)
    CACHE_VERSION = "boundary_geometry_v2"
    _compilation_cache: dict[str, _CompiledPolytope] = {}

    def __init__(
        self,
        hypothesis_space: ContinuousHypothesisSpace,
        method: str = METHOD_DYKSTRA,
        tolerance: float = 1e-9,
        projection_iterations: int = 100,
    ) -> None:
        self.space = hypothesis_space
        self.method = self.resolve_method(method)
        self.tolerance = float(tolerance)
        self.projection_iterations = int(projection_iterations)
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("boundary tolerance must be positive and finite.")
        if self.projection_iterations <= 0:
            raise ValueError("boundary projection_iterations must be positive.")

    @classmethod
    def resolve_method(cls, method: str) -> str:
        resolved = str(method).strip().lower()
        if resolved not in cls.VALID_METHODS:
            raise ValueError(
                f"Unsupported boundary distance method '{method}'. "
                f"Expected one of: {cls.VALID_METHODS}."
            )
        return resolved

    @staticmethod
    def _bounded_constraints(polytope: Polytope) -> tuple[np.ndarray, np.ndarray]:
        identity = np.eye(polytope.n_dims, dtype=float)
        constraints = np.vstack((polytope.A, -identity, identity))
        bounds = np.concatenate(
            (polytope.b, np.zeros(polytope.n_dims), np.ones(polytope.n_dims))
        )
        return np.ascontiguousarray(constraints), np.ascontiguousarray(bounds)

    @classmethod
    def _geometry_signature(
        cls,
        constraints: np.ndarray,
        bounds: np.ndarray,
        tolerance: float,
    ) -> str:
        digest = sha256()
        digest.update(cls.CACHE_VERSION.encode("ascii"))
        digest.update(repr(constraints.shape).encode("ascii"))
        digest.update(constraints.tobytes())
        digest.update(repr(bounds.shape).encode("ascii"))
        digest.update(bounds.tobytes())
        digest.update(np.asarray([tolerance], dtype=np.float64).tobytes())
        return digest.hexdigest()

    @classmethod
    def _compile_polytope(
        cls,
        polytope: Polytope,
        tolerance: float,
    ) -> _CompiledPolytope:
        constraints, bounds = cls._bounded_constraints(polytope)
        signature = cls._geometry_signature(constraints, bounds, tolerance)
        cached = cls._compilation_cache.get(signature)
        if cached is not None:
            return cached

        grouped: list[_ActiveSetGroup] = []
        max_size = min(polytope.n_dims, constraints.shape[0])
        for size in range(1, max_size + 1):
            active_constraints: list[np.ndarray] = []
            active_bounds: list[np.ndarray] = []
            gram_inverses: list[np.ndarray] = []
            for indices in combinations(range(constraints.shape[0]), size):
                selected = np.asarray(indices, dtype=int)
                active = constraints[selected]
                if np.linalg.matrix_rank(active, tol=tolerance) != size:
                    continue
                gram = active @ active.T
                try:
                    gram_inverse = np.linalg.inv(gram)
                except np.linalg.LinAlgError:
                    continue
                active_constraints.append(active)
                active_bounds.append(bounds[selected])
                gram_inverses.append(gram_inverse)
            if active_constraints:
                grouped.append(
                    _ActiveSetGroup(
                        constraints=np.stack(active_constraints),
                        bounds=np.stack(active_bounds),
                        gram_inverses=np.stack(gram_inverses),
                    )
                )

        compiled = _CompiledPolytope(
            constraints=constraints,
            bounds=bounds,
            active_set_groups=tuple(grouped),
            signature=signature,
        )
        cls._compilation_cache[signature] = compiled
        return compiled

    @classmethod
    def clear_compilation_cache(cls) -> None:
        cls._compilation_cache.clear()

    @classmethod
    def compilation_cache_info(cls) -> dict[str, int]:
        return {"size": len(cls._compilation_cache)}

    @staticmethod
    def _project_to_halfspace(
        point: np.ndarray,
        normal: np.ndarray,
        bound: float,
    ) -> np.ndarray:
        excess = float(np.dot(normal, point) - bound)
        if excess <= 0.0:
            return point
        return point - excess / float(np.dot(normal, normal)) * normal

    @classmethod
    def _project_dykstra(
        cls,
        point: np.ndarray,
        polytope: Polytope,
        projection_iterations: int,
    ) -> np.ndarray:
        # Preserve the historical solver: Dykstra corrections apply to the
        # declared halfspaces and the unit-box projection follows each cycle.
        projected = np.asarray(point, dtype=float).copy()
        corrections = [np.zeros_like(projected) for _ in range(len(polytope.A))]
        for _ in range(projection_iterations):
            for index, (normal, bound) in enumerate(zip(polytope.A, polytope.b)):
                candidate = projected + corrections[index]
                updated = cls._project_to_halfspace(candidate, normal, float(bound))
                corrections[index] = candidate - updated
                projected = updated
            projected = np.clip(projected, 0.0, 1.0)
        return projected

    def _distances_dykstra(
        self,
        stimuli: np.ndarray,
        polytope: Polytope,
    ) -> np.ndarray:
        constraints, bounds = self._bounded_constraints(polytope)
        violations = stimuli @ constraints.T - bounds[None, :]
        outside = np.any(violations > self.tolerance, axis=1)
        distances = np.zeros(stimuli.shape[0], dtype=float)
        for row in np.flatnonzero(outside):
            projection = self._project_dykstra(
                stimuli[row], polytope, self.projection_iterations
            )
            distances[row] = float(np.linalg.norm(stimuli[row] - projection))
        return distances

    def _distances_kkt_active_set(
        self,
        stimuli: np.ndarray,
        polytope: Polytope,
        *,
        context: str,
    ) -> np.ndarray:
        compiled = self._compile_polytope(polytope, self.tolerance)
        violations = stimuli @ compiled.constraints.T - compiled.bounds[None, :]
        inside = np.all(violations <= self.tolerance, axis=1)
        distances = np.full(stimuli.shape[0], np.inf, dtype=float)
        distances[inside] = 0.0
        rows = np.flatnonzero(~inside)
        if rows.size == 0:
            return distances

        points = stimuli[rows]
        best = np.full(points.shape[0], np.inf, dtype=float)
        for start in range(0, points.shape[0], 256):
            chunk = points[start : start + 256]
            chunk_best = np.full(chunk.shape[0], np.inf, dtype=float)
            for group in compiled.active_set_groups:
                rhs = (
                    np.einsum("pd,skd->psk", chunk, group.constraints, optimize=True)
                    - group.bounds[None, :, :]
                )
                multipliers = np.einsum(
                    "psk,skl->psl", rhs, group.gram_inverses, optimize=True
                )
                valid_multiplier = np.all(multipliers >= -self.tolerance, axis=2)
                projected = chunk[:, None, :] - np.einsum(
                    "psk,skd->psd", multipliers, group.constraints, optimize=True
                )
                feasible = np.all(
                    np.einsum(
                        "psd,md->psm", projected, compiled.constraints, optimize=True
                    )
                    <= compiled.bounds[None, None, :] + 10.0 * self.tolerance,
                    axis=2,
                )
                valid = valid_multiplier & feasible
                candidate = np.linalg.norm(chunk[:, None, :] - projected, axis=2)
                candidate[~valid] = np.inf
                chunk_best = np.minimum(chunk_best, np.min(candidate, axis=1))
            best[start : start + chunk.shape[0]] = chunk_best

        missing = ~np.isfinite(best)
        if np.any(missing):
            raise BoundaryProjectionError(
                f"KKT active-set projection found no feasible candidate for "
                f"{int(np.sum(missing))} stimulus/stimuli ({context}); "
                f"geometry={compiled.signature[:12]}."
            )
        distances[rows] = best
        return distances

    def distances_to_polytope(
        self,
        stimuli: np.ndarray | Iterable,
        polytope: Polytope,
        *,
        context: str = "polytope",
    ) -> np.ndarray:
        values = as_stimuli(stimuli, self.space.n_dims)
        if self.method == self.METHOD_DYKSTRA:
            return self._distances_dykstra(values, polytope)
        return self._distances_kkt_active_set(values, polytope, context=context)

    def distances_to_category(
        self,
        stimuli: np.ndarray | Iterable,
        category: CategoryRegion,
        *,
        context: str = "category",
    ) -> np.ndarray:
        component_distances = [
            self.distances_to_polytope(
                stimuli,
                component,
                context=f"{context}, component={component_index}",
            )
            for component_index, component in enumerate(category.components)
        ]
        return np.min(np.stack(component_distances, axis=1), axis=1)

    def category_distances(
        self,
        hypo: int,
        stimuli: np.ndarray | Iterable,
    ) -> np.ndarray:
        values = as_stimuli(stimuli, self.space.n_dims)
        categories = self.space[int(hypo)].categories
        return np.column_stack(
            [
                self.distances_to_category(
                    values,
                    category,
                    context=f"hypothesis={hypo}, category={category_index}",
                )
                for category_index, category in enumerate(categories)
            ]
        )

    def category_probabilities(
        self,
        hypo: int,
        stimuli: np.ndarray | Iterable,
        beta: float,
    ) -> np.ndarray:
        return softmax(self.category_distances(hypo, stimuli).T, -float(beta), axis=0)

    def category_assignments(
        self,
        hypo: int,
        stimuli: np.ndarray | Iterable,
        tol: float | None = None,
    ) -> np.ndarray:
        """Assign by region membership, then nearest-region distance."""
        values = as_stimuli(stimuli, self.space.n_dims)
        tolerance = self.tolerance if tol is None else float(tol)
        categories = self.space[int(hypo)].categories
        distances = self.category_distances(hypo, values)
        memberships = np.zeros((values.shape[0], len(categories)), dtype=bool)
        violation_scores = np.full_like(distances, np.inf)
        for category_index, category in enumerate(categories):
            for component in category.components:
                constraints, bounds = self._bounded_constraints(component)
                violations = values @ constraints.T - bounds[None, :]
                memberships[:, category_index] |= np.all(
                    violations <= tolerance, axis=1
                )
                violation_scores[:, category_index] = np.minimum(
                    violation_scores[:, category_index],
                    np.clip(violations, 0.0, None).sum(axis=1),
                )

        assignments = np.argmin(distances, axis=1).astype(int)
        for row in np.flatnonzero(np.any(memberships, axis=1)):
            candidates = np.flatnonzero(memberships[row])
            assignments[row] = int(
                candidates[np.argmin(violation_scores[row, candidates])]
            )
        return assignments


__all__ = ["BoundaryGeometry", "BoundaryProjectionError"]
