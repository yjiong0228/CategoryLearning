"""The error decomposition preserves probability mass and missing states."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_readout_channels import error_channels


def test_exact_error_allocation_and_instantaneous_replacement():
    rows = error_channels([[0, 1, 1]], [[0, .8, 1]],
                          [[np.nan, .9, .7]], [[.3, .76, .7]])
    np.testing.assert_allclose(rows['missing_target_error'], [[.7, 0, 0]])
    np.testing.assert_allclose(rows['target_emission_error'], [[0, .08, .3]])
    np.testing.assert_allclose(rows['other_rules_error'], [[0, .16, 0]])
    np.testing.assert_allclose(rows['force_present_target_correct'], [[.3, .9, .7]])
    np.testing.assert_allclose(sum(rows[k] for k in
        ('missing_target_error', 'target_emission_error', 'other_rules_error')),
        rows['expected_error'])


def test_present_but_unused_target_keeps_other_rule_errors():
    rows = error_channels([[1]], [[0]], [[.2]], [[.8]])
    assert rows['other_rules_error'].item() == pytest.approx(.2)
    # Forcing the target can reduce accuracy; this is not an upper bound.
    assert rows['force_present_target_correct'].item() == .2


@pytest.mark.parametrize('present,contribution,q,y', [
    ([[0]], [[.1]], [[.5]], [[.5]]),
    ([[1]], [[.9]], [[.9]], [[.5]]),
    ([[1]], [[.9]], [[.1]], [[.9]]),
    ([[1]], [[1]], [[np.nan]], [[.5]]),
    ([[.5]], [[0]], [[.5]], [[.5]]),
    ([[1]], [[0]], [[.5]], [[1.1]]),
    ([[1]], [[0, 0]], [[.5]], [[.5]]),
])
def test_incompatible_probabilities_are_rejected(present, contribution, q, y):
    with pytest.raises(ValueError):
        error_channels(present, contribution, q, y)
