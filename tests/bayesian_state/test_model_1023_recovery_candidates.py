"""Check paired recovery errors without treating absent rules as zero precision."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_1023_recovery_candidates import paired_changes


def traces(presence, contribution, beta):
    return dict(presence_probability=np.array(presence, float),
                contribution_mean=np.array(contribution, float),
                conditional_beta_mean=np.array(beta, float))


def test_paired_errors_use_the_same_truth_present_trials():
    truth = dict(present=np.array([1, 0]), contribution=np.array([.8, 0]), beta=np.array([4, 0]))
    a = traces([[1, 0]]*3, [[.8, 0]]*3, [[4, 20]]*3)
    b = traces([[.5, .5]]*3, [[.4, .2]]*3, [[6, 1]]*3)
    result = paired_changes(a, b, truth)
    assert result['membership_brier_delta'] == -.25
    assert result['contribution_rmse_delta'] == pytest.approx(-np.sqrt(.1))
    assert result['conditional_beta_rmse_delta'] == -2
    assert result['conditional_beta_rmse_trials'] == 1
    assert result['conditional_beta_rmse_mc_se'] == 0


def test_missing_replica_keeps_point_error_but_undefined_monte_carlo_error():
    truth = dict(present=np.array([1, 0]), contribution=np.array([.8, 0]), beta=np.array([4, 0]))
    a = traces([[1, 0]]*3, [[.8, 0]]*3, [[4, np.nan], [np.nan, np.nan], [4, np.nan]])
    b = traces([[1, 0]]*3, [[.8, 0]]*3, [[6, np.nan]]*3)
    result = paired_changes(a, b, truth)
    assert result['conditional_beta_rmse_delta'] == -2
    assert np.isnan(result['conditional_beta_rmse_mc_se'])
    truth['present'][:] = 0
    result = paired_changes(a, b, truth)
    assert result['conditional_beta_rmse_trials'] == 0
    assert np.isnan(result['conditional_beta_rmse_delta'])
