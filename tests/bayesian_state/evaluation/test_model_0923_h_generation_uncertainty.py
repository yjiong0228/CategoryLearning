"""Whole-path deletion retains the empirical distribution scoring convention."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_h_generation_uncertainty import path_jackknife
from src.Bayesian_state.evaluation.model_0923_h_validation import jackknife_se


def test_identical_paths_have_zero_simulation_error():
    scores=path_jackknife(np.tile([.1,.9],(4,1)),np.array([.2,.7]))
    assert jackknife_se(scores['energy'])==pytest.approx(0.)
    assert jackknife_se(scores['crps'])==pytest.approx(0.)


def test_single_block_deletions_have_known_empirical_energy_scores():
    scores=path_jackknife(np.array([[0.],[1.],[1.]]),np.array([0.]))
    np.testing.assert_allclose(scores['energy'],[1.,.25,.25])
    np.testing.assert_allclose(scores['crps'],scores['energy'])
    assert jackknife_se(scores['energy'])==pytest.approx(.5)
