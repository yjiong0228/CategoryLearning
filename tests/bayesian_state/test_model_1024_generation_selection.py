"""Acceptance and future-data boundaries for the new selection workflow."""
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_1024_generation_selection import (
    final_decision, seed_bank, training_boundary)
from src.Bayesian_state.workflows.analysis.model_1024_generation_selection_summary import pareto_mask


@pytest.mark.parametrize('likelihood,energy,expected', [
    (True, True, 'new'), (False, True, 'old'), (True, False, 'old'), (False, False, 'old')])
def test_replacement_requires_both_independent_guards(likelihood, energy, expected):
    result = final_decision('old', 'new', {'accepted': likelihood}, {'energy_accepted': energy})
    assert result['selected'] == expected
    assert result['replaced'] == (expected == 'new')


def test_same_point_does_not_claim_an_improvement():
    result = final_decision('a', 'a', {}, {})
    assert not result['replaced'] and result['reason'] == 'original_is_energy_leader'


def test_training_boundary_blocks_pathlib_builtin_and_numpy_and_restores(tmp_path):
    future = tmp_path/'sealed_test.npz'
    training = tmp_path/'training.npz'
    np.savez(future, choices=[1, 2])
    np.savez(training, choices=[2, 1])
    with training_boundary():
        with np.load(training) as z:
            np.testing.assert_array_equal(z['choices'], [2, 1])
        for action in (lambda: open(future, 'rb'), lambda: future.read_bytes(), lambda: np.load(future),
                       lambda: open(bytes(future), 'rb')):
            with pytest.raises(PermissionError, match='Training workers'):
                action()
    assert future.read_bytes()


def test_paired_seeds_match_across_parameters_but_are_disjoint_across_stages():
    policy = dict(protocol='fixture', seed=42)
    rank = seed_bank(policy, 129, 'ranking', 64, set())
    confirm = seed_bank(policy, 129, 'energy_confirmation', 128, set(rank))
    assert rank == seed_bank(policy, 129, 'ranking', 64, set())
    assert not set(rank).intersection(confirm)
    with pytest.raises(ValueError, match='independent'):
        seed_bank(policy, 129, 'ranking', 64, set(rank))


def test_competing_candidates_retain_tradeoffs_and_identical_scores():
    mask = pareto_mask(np.array([1., 1.01, 1.02, 1.]), np.array([.2, .1, .3, .2]))
    np.testing.assert_array_equal(mask, [True, True, False, True])


def test_candidate_frontier_rejects_invalid_scores():
    with pytest.raises(ValueError, match='finite'):
        pareto_mask(np.array([np.nan]), np.array([.1]))
