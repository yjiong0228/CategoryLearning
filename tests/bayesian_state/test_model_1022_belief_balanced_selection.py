"""Training eligibility and numerical uncertainty determine the frozen decision."""
import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import model_1022_belief_balanced_selection as selection


def test_generation_choice_respects_likelihood_tolerance():
    rows = [dict(candidate='nll', confirmation_nll=.5, training_energy=.2),
            dict(candidate='balanced', confirmation_nll=.515, training_energy=.1),
            dict(candidate='outside', confirmation_nll=.53, training_energy=.01)]
    assert selection.choose_energy(rows, .02)['candidate'] == 'balanced'
    assert selection.choose_energy(rows, .01)['candidate'] == 'nll'


def test_energy_ties_use_likelihood_then_stable_identity():
    rows = [dict(candidate='b', confirmation_nll=.5, training_energy=.1),
            dict(candidate='a', confirmation_nll=.5, training_energy=.1),
            dict(candidate='c', confirmation_nll=.51, training_energy=.1)]
    assert selection.choose_energy(rows, .02)['candidate'] == 'a'


def test_identical_points_pass_with_zero_paired_uncertainty():
    p = np.array([[.6, .8], [.7, .9], [.5, .7], [.65, .85]])
    decision = selection.likelihood_guard(p, p.copy(), .02, 2.)
    assert decision['accepted']
    assert decision['nll_difference'] == decision['difference_mcse'] == 0


def test_numerical_uncertainty_can_reject_an_equal_point_estimate():
    original = np.full((2, 3), .8)
    proposed = np.array([[.6, .8, .8], [1., .8, .8]])
    decision = selection.likelihood_guard(original, proposed, .02, 2.)
    assert decision['nll_difference'] == pytest.approx(0, abs=1e-14)
    assert decision['difference_mcse'] > .01
    assert not decision['accepted']


def test_guard_accepts_small_stable_loss_and_rejects_large_loss():
    original = np.full((4, 3), .8)
    accepted = selection.likelihood_guard(original, np.full((4, 3), .79), .02, 2.)
    rejected = selection.likelihood_guard(original, np.full((4, 3), .7), .02, 2.)
    assert accepted['nll_difference'] == pytest.approx(np.log(.8/.79))
    assert accepted['accepted'] and not rejected['accepted']


@pytest.mark.parametrize('proposed', [np.ones((3, 3)), np.full((4, 3), 1.1), np.full((4, 3), np.nan)])
def test_invalid_guard_arrays_fail(proposed):
    with pytest.raises(ValueError):
        selection.likelihood_guard(np.full((4, 3), .8), proposed, .02, 2.)
