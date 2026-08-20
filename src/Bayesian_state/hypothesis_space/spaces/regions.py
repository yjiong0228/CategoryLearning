"""Geometric region objects used by continuous hypothesis spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


Hyperplane = tuple[tuple[float, ...], float]


def _readonly_array(value: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(value, dtype=float).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class Polytope:
    """One convex component represented as ``A @ x <= b`` in ``[0, 1]^d``."""

    A: np.ndarray
    b: np.ndarray

    def __post_init__(self) -> None:
        A = _readonly_array(self.A)
        b = np.asarray(self.b, dtype=float).reshape(-1).copy()
        b.setflags(write=False)
        if A.ndim != 2 or A.shape[0] != b.size:
            raise ValueError(
                f"Polytope expects A[m, d] and b[m], got {A.shape} and {b.shape}."
            )
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
            raise ValueError("Polytope constraints must be finite.")
        zero_rows = np.linalg.norm(A, axis=1) == 0.0
        if np.any(zero_rows & (b < 0.0)):
            raise ValueError("Polytope contains an infeasible zero-normal constraint.")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", b)

    @property
    def n_dims(self) -> int:
        return int(self.A.shape[1])


@dataclass(frozen=True, eq=False)
class CategoryRegion:
    """A category region represented by one or more convex components."""

    components: tuple[Polytope, ...]

    def __post_init__(self) -> None:
        components = tuple(self.components)
        if not components:
            raise ValueError("CategoryRegion requires at least one component.")
        n_dims = components[0].n_dims
        if any(component.n_dims != n_dims for component in components):
            raise ValueError("All category components must use the same dimension count.")
        object.__setattr__(self, "components", components)


__all__ = ["CategoryRegion", "Hyperplane", "Polytope"]
