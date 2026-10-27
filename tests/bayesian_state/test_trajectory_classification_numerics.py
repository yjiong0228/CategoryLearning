"""Implementation checks only; no statistical classification validation."""
from itertools import combinations

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.special import xlogy

from src.Bayesian_state.metrics.trajectory_classification import _piecewise_fits, _ramp_loss


def test_segment_solver_matches_exhaustive_boundaries():
    y = np.array([1,0,1,0,0,0,1,1,0,1,1,1,0,0,0,1,1,1,0,1,0,1,1,1])
    for result in _piecewise_fits(y, 4, 4):
        losses = []
        for cuts in combinations(range(4, len(y)-3), result['segments']-1):
            edges = (0, *cuts, len(y))
            if min(np.diff(edges))<4:
                continue
            loss = 0.
            for a, b in zip(edges[:-1], edges[1:]):
                k, n = y[a:b].sum(), b-a
                loss -= xlogy(k, k/n)+xlogy(n-k, 1-k/n)
            losses.append(loss)
        assert result['nll']==pytest.approx(min(losses), abs=1e-10)


def test_ramp_gradient_matches_finite_differences():
    y = np.array([0,1,0,1,0,0,1,1,0,1,1,1], dtype=float)
    x = np.linspace(0, 1, len(y))
    z = np.array([.2, .8, .13, .77])
    _, gradient = _ramp_loss(z, y, x)
    numeric = approx_derivative(lambda a:_ramp_loss(a,y,x)[0], z)
    np.testing.assert_allclose(gradient, numeric.ravel(), rtol=1e-5, atol=1e-6)
