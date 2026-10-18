"""Boundary and random-stream checks for the recalled-search forecast wrapper."""
import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_recalled_prefix_forecast import (
    branch_seeds, forecast_inputs,
)


class FutureTaskOnly(dict):
    def __getitem__(self, key):
        assert key in ('stimulus', 'categories'), 'Observed future behavior was accessed'
        return super().__getitem__(key)


def test_scientific_forecast_does_not_read_future_responses():
    training = dict(stimulus=np.ones((16, 4)), choices=np.ones(16), feedback=np.zeros(16))
    future = FutureTaskOnly(stimulus=np.zeros((4, 4)), categories=np.full(4, 2))
    prefix, task = forecast_inputs(training, future, False)
    assert set(prefix) == {'stimulus', 'choices', 'feedback'}
    assert set(task) == {'stimulus', 'categories'}
    assert len(prefix['choices']) == 16
    np.testing.assert_array_equal(task['categories'], [2]*4)


def test_smoke_uses_disjoint_short_training_slices():
    training = {name: np.arange(20) for name in ('stimulus', 'choices', 'feedback', 'categories')}
    prefix, future = forecast_inputs(training, {}, True)
    np.testing.assert_array_equal(prefix['choices'], np.arange(8))
    np.testing.assert_array_equal(future['categories'], np.arange(8, 16))
    with pytest.raises(ValueError, match='sixteen'):
        forecast_inputs({key: value[:12] for key, value in training.items()}, {}, True)


def test_branch_seeds_are_reproducible_distinct_and_filter_specific():
    settings = dict(protocol='test', seed=123, branches_per_filter=16)
    seeds = branch_seeds(settings, 308, 1)
    assert seeds == branch_seeds(settings, 308, 1)
    assert len(set(seeds)) == 16
    assert not set(seeds).intersection(branch_seeds(settings, 308, 2))
    assert not set(seeds).intersection(branch_seeds(settings, 207, 1))
