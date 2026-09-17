"""Ownership and stream invariants for PF/workspace engineering shortcuts."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.inference.backends import particle_filter as pf
from src.Bayesian_state.model.modules.hypothesis_transition.contracts import (
    HypothesisSelection, _membership_mask,
)


@pytest.mark.parametrize('before,after', [
    ([], []), ([], [4, 1]), ([3, 2], []), ([5, 1, 3], [3, 5, 7]),
    ([1, 1, -5], [1, -5, -5, 10**12]), ([10**12, -10**12], [-10**12, 2]),
])
def test_integer_membership_preserves_empty_duplicate_and_order_contract(before, after):
    before, after = np.asarray(before, dtype=int), np.asarray(after, dtype=int)
    np.testing.assert_array_equal(_membership_mask(before, after), np.isin(before, after))
    result = HypothesisSelection.from_active_sets(before, after)
    np.testing.assert_array_equal(result.survivors, after[np.isin(after, before)])
    np.testing.assert_array_equal(result.dropped, before[~np.isin(before, after)])
    np.testing.assert_array_equal(result.newcomers, after[~np.isin(after, before)])


@pytest.mark.parametrize('probability', [0., 1., .2, .8])
def test_update_gate_keeps_independent_seed_decisions(probability):
    for trial in range(5):
        for particle in range(8):
            seed = pf._future_seed(8326, trial, particle, 'active_set_learning_update_gate')
            expected = np.random.default_rng(seed).random() < probability
            assert pf._learning_update_occurs(probability, 8326, trial, particle) == expected


def test_deterministic_gate_does_not_construct_rng(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('RNG should not be needed')
    monkeypatch.setattr(pf, '_future_seed', forbidden)
    assert pf._learning_update_occurs(1., 8326, 0, 0)
    assert not pf._learning_update_occurs(0., 8326, 0, 0)


def test_unique_ancestor_snapshots_preserve_child_ownership_and_future_seeds():
    class Module:
        def reseed_future(self, seed):
            self.seed = seed
    class Engine:
        def __init__(self, value):
            self.payload = {'nested': {'array': np.array([value], dtype=float)}}
            self.snapshots = 0
            self.module = Module()
        def state_dict(self):
            self.snapshots += 1
            return self.payload
        def load_state_dict(self, payload):
            self.payload = payload
        def get_module(self, role):
            return self.module
    models = [SimpleNamespace(engine=Engine(i)) for i in range(3)]
    snapshots = pf._snapshot_ancestors(models, np.array([2, 2, 0]))
    assert [model.engine.snapshots for model in models] == [1, 0, 1]
    assert snapshots[0] is snapshots[1]
    for i, snapshot in enumerate(snapshots):
        pf._restore(models[i], snapshot, filter_seed=8326, trial_index=2, particle_index=i)
    children = [model.engine.payload['nested']['array'] for model in models]
    np.testing.assert_array_equal([child[0] for child in children], [2., 2., 0.])
    assert not np.shares_memory(children[0], children[1])
    children[0][0] = 100.
    assert children[1][0] == 2.
    assert snapshots[0].payload['nested']['array'][0] == 2.
    assert len({model.engine.module.seed for model in models}) == 3
