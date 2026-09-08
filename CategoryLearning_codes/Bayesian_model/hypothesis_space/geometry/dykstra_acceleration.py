"""Optional machine-code acceleration for the historical Dykstra solver."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised in environments without Numba.
    njit = None


def _dykstra_distances_impl(
    stimuli: np.ndarray,
    outside_rows: np.ndarray,
    constraints: np.ndarray,
    bounds: np.ndarray,
    projection_iterations: int,
) -> np.ndarray:
    """Run the legacy projection loop without changing its update order."""

    distances = np.zeros(stimuli.shape[0], dtype=np.float64)
    n_constraints = constraints.shape[0]
    n_dims = stimuli.shape[1]
    for outside_index in range(outside_rows.size):
        row = outside_rows[outside_index]
        projected = stimuli[row].copy()
        corrections = np.zeros((n_constraints, n_dims), dtype=np.float64)
        for _ in range(projection_iterations):
            for constraint_index in range(n_constraints):
                excess = -bounds[constraint_index]
                normal_norm = 0.0
                for dim in range(n_dims):
                    candidate_value = (
                        projected[dim] + corrections[constraint_index, dim]
                    )
                    excess += constraints[constraint_index, dim] * candidate_value
                    normal_norm += (
                        constraints[constraint_index, dim]
                        * constraints[constraint_index, dim]
                    )
                if excess > 0.0:
                    scale = excess / normal_norm
                    for dim in range(n_dims):
                        candidate_value = (
                            projected[dim] + corrections[constraint_index, dim]
                        )
                        updated = (
                            candidate_value
                            - scale * constraints[constraint_index, dim]
                        )
                        corrections[constraint_index, dim] = (
                            candidate_value - updated
                        )
                        projected[dim] = updated
                else:
                    for dim in range(n_dims):
                        projected[dim] += corrections[constraint_index, dim]
                        corrections[constraint_index, dim] = 0.0
            for dim in range(n_dims):
                if projected[dim] < 0.0:
                    projected[dim] = 0.0
                elif projected[dim] > 1.0:
                    projected[dim] = 1.0

        squared_distance = 0.0
        for dim in range(n_dims):
            difference = stimuli[row, dim] - projected[dim]
            squared_distance += difference * difference
        distances[row] = np.sqrt(squared_distance)
    return distances


if njit is None:
    _dykstra_distances_numba = None
else:
    _dykstra_distances_numba = njit(cache=True, fastmath=False)(
        _dykstra_distances_impl
    )


def dykstra_numba_available() -> bool:
    """Return whether the optional compiled backend can be used."""

    return _dykstra_distances_numba is not None


def dykstra_distances_numba(
    stimuli: np.ndarray,
    outside_rows: np.ndarray,
    constraints: np.ndarray,
    bounds: np.ndarray,
    projection_iterations: int,
) -> np.ndarray:
    """Evaluate batched distances with the compiled historical algorithm."""

    if _dykstra_distances_numba is None:
        raise RuntimeError(
            "The Numba Dykstra backend was requested, but Numba is unavailable."
        )
    return _dykstra_distances_numba(
        np.ascontiguousarray(stimuli, dtype=np.float64),
        np.ascontiguousarray(outside_rows, dtype=np.int64),
        np.ascontiguousarray(constraints, dtype=np.float64),
        np.ascontiguousarray(bounds, dtype=np.float64),
        int(projection_iterations),
    )


def warmup_dykstra_numba() -> bool:
    """Compile or load the cached kernel before process-level parallelism."""

    if _dykstra_distances_numba is None:
        return False
    dykstra_distances_numba(
        np.asarray([[0.75]], dtype=np.float64),
        np.asarray([0], dtype=np.int64),
        np.asarray([[1.0]], dtype=np.float64),
        np.asarray([0.5], dtype=np.float64),
        1,
    )
    return True


__all__ = [
    "dykstra_distances_numba",
    "dykstra_numba_available",
    "warmup_dykstra_numba",
]
