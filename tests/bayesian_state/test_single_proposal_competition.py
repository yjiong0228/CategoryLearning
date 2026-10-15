"""Enumerated fixed-cardinality reference and capacity-one retention check."""
from itertools import combinations

import numpy as np
import pytest
from scipy.special import softmax

from src.Bayesian_state.model.modules.hypothesis_transition.single_proposal_competition import (
    drop_probabilities, sample_competing_set,
)


@pytest.mark.parametrize('scores', [[0, 0, 0], [-3, .2, 2], [1000, -1000, 1]])
def test_single_drop_matches_exact_fixed_size_set_enumeration(scores):
    sets = list(combinations(range(3), 2))
    probability = softmax([sum(scores[i] for i in s) for s in sets])
    expected = [sum(p for s, p in zip(sets, probability) if i not in s) for i in range(3)]
    np.testing.assert_allclose(drop_probabilities(np.array(scores), 2), expected, atol=1e-13, rtol=1e-13)


def test_spare_capacity_keeps_every_candidate_without_random_draw():
    class NoDraw:
        def choice(self, *args, **kwargs):
            raise AssertionError('Spare capacity needs no competition draw')
    selected, marginal = sample_competing_set(np.array([-1000., 0.]), 3, NoDraw())
    np.testing.assert_array_equal(selected, [0, 1]); assert marginal == 1


def test_capacity_one_can_reject_new_rule_and_retain_strong_incumbent():
    selected, marginal = sample_competing_set(np.array([0., 1000.]), 1, np.random.default_rng(4))
    np.testing.assert_array_equal(selected, [1]); assert marginal == 0


def test_empirical_frequency_and_seed_replay():
    scores = np.array([-.7, .4, 1.2]); expected = drop_probabilities(scores, 2)
    rng = np.random.default_rng(15); counts = np.zeros(3)
    for _ in range(6000):
        selected, marginal = sample_competing_set(scores, 2, rng)
        assert len(selected) == 2
        counts[list(set(range(3)) - set(selected))[0]] += 1
        assert marginal == pytest.approx(1 - expected[0])
    np.testing.assert_allclose(counts / 6000, expected, atol=.025, rtol=0)
    a, b = np.random.default_rng(2), np.random.default_rng(2)
    for _ in range(20):
        np.testing.assert_array_equal(sample_competing_set(scores, 2, a)[0], sample_competing_set(scores, 2, b)[0])


@pytest.mark.parametrize('scores,capacity', [([], 1), ([np.nan], 1), ([0], 0), ([0], True), ([0], 1.5), ([0,0,0], 1)])
def test_invalid_pool_and_capacity_are_rejected(scores, capacity):
    with pytest.raises(ValueError):
        drop_probabilities(np.array(scores), capacity)
