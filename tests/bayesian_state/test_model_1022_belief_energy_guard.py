"""Independent generation gains preserve pairing and the earlier NLL guard."""
import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import model_1022_belief_energy_guard as guard


def test_identical_paths_do_not_establish_a_generation_improvement():
    x = np.array([[.2], [.3], [.4], [.5]])
    result = guard.paired_energy_guard(x, x.copy(), np.array([0.]), 2.)
    assert result['energy_difference'] == result['difference_mcse'] == 0
    assert not result['energy_accepted']


def test_matched_constant_shift_has_exact_paired_gain():
    x = np.array([[.4], [.5], [.6], [.7]])
    result = guard.paired_energy_guard(x, x-.2, np.array([0.]), 2.)
    assert result['energy_difference'] == pytest.approx(-.2)
    assert result['difference_mcse'] == pytest.approx(0, abs=1e-14)
    assert result['energy_accepted']
    worse = guard.paired_energy_guard(x, x+.2, np.array([0.]), 2.)
    assert not worse['energy_accepted']


def test_likelihood_failure_always_retains_original_point():
    decision = dict(original='old', proposed='new', accepted=False)
    assert guard.selected_candidate(decision, dict(energy_accepted=True)) == 'old'
    decision['accepted'] = True
    assert guard.selected_candidate(decision, dict(energy_accepted=False)) == 'old'
    assert guard.selected_candidate(decision, dict(energy_accepted=True)) == 'new'


def test_uncertain_apparent_generation_gain_is_rejected():
    original = np.full((4, 1), .5)
    proposed = np.array([[.1], [.6], [.6], [.6]])
    result = guard.paired_energy_guard(original, proposed, np.array([0.]), 2.)
    assert result['energy_difference'] == pytest.approx(-.15)
    assert result['difference_mcse'] == pytest.approx(.25)
    assert not result['energy_accepted']
    assert guard.paired_energy_guard(original, proposed, np.array([0.]), 0.)['energy_accepted']


@pytest.mark.parametrize('seeds,used', [([1, 1], set()), ([1, 2], {2})])
def test_reused_or_duplicate_seeds_fail(seeds, used):
    with pytest.raises(ValueError, match='independent'):
        guard.validate_fresh_seeds(seeds, used)


def test_fresh_seed_bank_is_accepted():
    guard.validate_fresh_seeds([4, 5, 6], {1, 2, 3})


@pytest.mark.parametrize('other,multiplier', [
    (np.ones((3, 1)), 2.), (np.full((4, 1), np.nan), 2.),
    (np.ones((4, 1)), -1.), (np.ones((4, 1)), np.nan),
])
def test_invalid_guard_input_fails(other, multiplier):
    with pytest.raises(ValueError):
        guard.paired_energy_guard(np.ones((4, 1)), other, np.array([0.]), multiplier)
