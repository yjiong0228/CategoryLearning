"""Equivalence and contract of the optional exact active-set distance solver.

The active-set backend replaces 100 Dykstra projection cycles with an exact
enumeration of the declared halfspaces.  These tests pin the properties the
shared task space relies on: it agrees with the historical Numba loop, it is
opt-in, and it keeps the historical solver as a fallback for rows that no
halfspace subset can resolve.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.Bayesian_state.hypothesis_space.geometry import (
    BoundaryGeometry,
    active_set_numba_available,
)
from src.Bayesian_state.hypothesis_space.spaces import (
    build_continuous_hypothesis_space,
)

pytestmark = pytest.mark.skipif(
    not active_set_numba_available(),
    reason="Numba is unavailable",
)


def _geometries(n_dims: int, n_cats: int):
    space = build_continuous_hypothesis_space(n_dims, n_cats)
    historical = BoundaryGeometry(
        space,
        method="dykstra_iterative_projection",
        dykstra_backend="numba",
    )
    active = BoundaryGeometry(
        space,
        method="dykstra_iterative_projection",
        dykstra_backend="active_set_numba",
    )
    return space, historical, active


def _stimuli(n_dims: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.vstack(
        [
            rng.random((512, n_dims)),
            np.clip(rng.normal(0.5, 0.30, (256, n_dims)), 0.0, 1.0),
            np.clip(rng.normal(0.5, 0.05, (256, n_dims)), 0.0, 1.0),
            np.zeros((1, n_dims)),
            np.ones((1, n_dims)),
            rng.integers(0, 2, (64, n_dims)).astype(float),
        ]
    ).astype(np.float64)


def _polytopes(space):
    return [
        polytope
        for hypothesis in range(len(space))
        for category in space[hypothesis].categories
        for polytope in category.components
    ]


@pytest.mark.parametrize("n_dims,n_cats", [(2, 2), (4, 2), (4, 4)])
def test_active_set_matches_historical_solver_on_task_geometry(n_dims, n_cats):
    space, historical, active = _geometries(n_dims, n_cats)
    stimuli = _stimuli(n_dims)

    for polytope in _polytopes(space):
        expected = historical.distances_to_polytope(stimuli, polytope)
        actual = active.distances_to_polytope(stimuli, polytope)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("n_dims,n_cats", [(4, 2), (4, 4)])
def test_single_stimulus_fast_path_matches_batched_path(n_dims, n_cats):
    """The fused one-stimulus kernel must equal the batched enumeration."""

    space, historical, active = _geometries(n_dims, n_cats)
    stimuli = _stimuli(n_dims, seed=3)[:64]

    for polytope in _polytopes(space):
        batched = active.distances_to_polytope(stimuli, polytope)
        for row in range(stimuli.shape[0]):
            single = active.distances_to_polytope(stimuli[row : row + 1], polytope)
            assert single.shape == (1,)
            np.testing.assert_allclose(single[0], batched[row], atol=0.0, rtol=0.0)


def test_active_set_backend_is_opt_in_and_auto_stays_historical():
    space = build_continuous_hypothesis_space(2, 2)

    automatic = BoundaryGeometry(space, dykstra_backend="auto")
    assert automatic.dykstra_backend == "numba"

    explicit = BoundaryGeometry(space, dykstra_backend="active_set_numba")
    assert explicit.dykstra_backend == "active_set_numba"


def test_active_set_candidates_are_cached_per_polytope():
    space, _historical, active = _geometries(4, 2)
    polytope = _polytopes(space)[0]

    BoundaryGeometry.clear_active_sets_cache()
    first = active._active_sets_for(polytope)
    second = active._active_sets_for(polytope)
    assert first is second
    assert first[0].shape[0] == first[2].shape[0]
    # The empty subset is always available, so every polytope gets candidates.
    assert first[0].shape[0] >= 1


def test_active_set_backend_requires_numba():
    """Requesting the compiled backend without Numba must fail loudly."""

    space = build_continuous_hypothesis_space(2, 2)
    import src.Bayesian_state.hypothesis_space.geometry.boundary as boundary_module

    original = boundary_module.active_set_numba_available
    boundary_module.active_set_numba_available = lambda: False
    try:
        with pytest.raises(RuntimeError, match="active-set Dykstra backend"):
            BoundaryGeometry(space, dykstra_backend="active_set_numba")
    finally:
        boundary_module.active_set_numba_available = original
