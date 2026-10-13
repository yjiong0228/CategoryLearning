"""Contract of the shared boundary distance solver.

The solver enumerates the halfspace subsets of a category region instead of
repeating projection cycles.  These tests pin the properties the shared task
space relies on: it agrees with an independent optimiser on unit-cube stimuli,
it agrees with the alternative active-set projection method, the fused
single-stimulus path equals the batched path, out-of-cube stimuli are
rejected, and candidate active sets are cached per polytope.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import Bounds, LinearConstraint, minimize

from src.Bayesian_state.hypothesis_space.geometry import (
    BoundaryGeometry,
    active_set_numba_available,
    warmup_active_set_numba,
)
from src.Bayesian_state.hypothesis_space.spaces import (
    Polytope,
    build_continuous_hypothesis_space,
)

pytestmark = pytest.mark.skipif(
    not active_set_numba_available(),
    reason="Numba is unavailable",
)


def _geometries(n_dims: int, n_cats: int):
    space = build_continuous_hypothesis_space(n_dims, n_cats)
    exact = BoundaryGeometry(space, method="dykstra_iterative_projection")
    kkt = BoundaryGeometry(space, method="kkt_active_set_projection")
    return space, exact, kkt


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
def test_exact_solver_matches_kkt_active_set_projection(n_dims, n_cats):
    """Two independent exact methods must agree on the task geometry."""

    space, exact, kkt = _geometries(n_dims, n_cats)
    stimuli = _stimuli(n_dims)

    for polytope in _polytopes(space):
        expected = kkt.distances_to_polytope(stimuli, polytope)
        actual = exact.distances_to_polytope(stimuli, polytope)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("n_dims,n_cats", [(2, 2), (4, 2)])
def test_exact_solver_matches_slsqp_oracle(n_dims, n_cats):
    """Spot-check against a general-purpose constrained optimiser."""

    space, exact, _kkt = _geometries(n_dims, n_cats)
    stimuli = _stimuli(n_dims, seed=5)[:32]

    for polytope in _polytopes(space)[:6]:
        observed = exact.distances_to_polytope(stimuli, polytope)
        constraint = LinearConstraint(polytope.A, -np.inf, polytope.b)
        for row, point in enumerate(stimuli):
            result = minimize(
                lambda value: 0.5 * float(np.sum((value - point) ** 2)),
                x0=np.clip(point, 0.0, 1.0),
                method="SLSQP",
                bounds=Bounds(0.0, 1.0),
                constraints=(constraint,),
                options={"ftol": 1e-12, "maxiter": 500},
            )
            assert result.success, result.message
            expected = np.linalg.norm(point - result.x)
            np.testing.assert_allclose(
                observed[row], expected, atol=1e-7, rtol=1e-6
            )


@pytest.mark.parametrize("n_dims,n_cats", [(4, 2), (4, 4)])
def test_single_stimulus_fast_path_matches_batched_path(n_dims, n_cats):
    """The fused one-stimulus kernel must equal the batched enumeration."""

    space, exact, _kkt = _geometries(n_dims, n_cats)
    stimuli = _stimuli(n_dims, seed=3)[:64]

    for polytope in _polytopes(space):
        batched = exact.distances_to_polytope(stimuli, polytope)
        for row in range(stimuli.shape[0]):
            single = exact.distances_to_polytope(stimuli[row : row + 1], polytope)
            assert single.shape == (1,)
            np.testing.assert_allclose(single[0], batched[row], atol=0.0, rtol=0.0)


def test_out_of_cube_stimuli_are_rejected_by_every_entry_point():
    space, exact, _kkt = _geometries(2, 2)
    polytope = _polytopes(space)[0]
    for bad in (np.asarray([[1.5, 0.5]]), np.asarray([[-0.5, 0.5]])):
        with pytest.raises(ValueError, match="unit cube"):
            exact.distances_to_polytope(bad, polytope)
    with pytest.raises(ValueError, match="unit cube"):
        exact.category_distances(0, np.asarray([[1.5, 0.5]]))


def test_active_set_candidates_are_cached_per_polytope():
    space, exact, _kkt = _geometries(4, 2)
    polytope = _polytopes(space)[0]

    BoundaryGeometry.clear_active_sets_cache()
    first = exact._active_sets_for(polytope)
    second = exact._active_sets_for(polytope)
    assert first is second
    cheap, full = first
    for family in (cheap, full):
        assert family[0].shape[0] == family[2].shape[0]
        # The empty subset is always available, so every polytope gets candidates.
        assert family[0].shape[0] >= 1
    # The exact family is a superset: it also enumerates the unit-cube faces.
    assert full[0].shape[0] >= cheap[0].shape[0]


def test_solver_requires_numba(monkeypatch):
    """Without Numba the geometry must fail loudly rather than silently."""

    space = build_continuous_hypothesis_space(2, 2)
    import src.Bayesian_state.hypothesis_space.geometry.boundary as boundary_module

    monkeypatch.setattr(boundary_module, "active_set_numba_available", lambda: False)
    with pytest.raises(RuntimeError, match="require Numba"):
        BoundaryGeometry(space)


def test_warmup_reports_compiled_availability():
    assert warmup_active_set_numba() is active_set_numba_available()
