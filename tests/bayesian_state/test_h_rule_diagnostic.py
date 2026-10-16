"""Oracle timing, intervention isolation, and replayed admission decisions."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.hypothesis_space.observation_model.continuous_partition import ContinuousPartition
from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
from src.Bayesian_state.simulation.h_rule_diagnostic import include_target, intervention


@pytest.fixture
def make(tmp_path, monkeypatch):
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    partition = ContinuousPartition(4, 2)
    def build():
        n = len(partition.hypothesis_space)
        engine = SimpleNamespace(partition=partition, set_size=n, distance_mode='boundary',
            prior=np.full(n, 1/n), beta=np.ones(n), get_module=lambda *a, **kw: None)
        module = Transition(engine, mechanism_version='0923-full-v3', max_capacity=2, exit_rate=0.,
            transfer_scale=.1, execution_inertia=0., relation_samples=128, relation_seed=0, module_seed=23,
            edit_costs=dict(feature=1., relation=1.5, reference=1., branch=2.),
            search_controller=dict(baseline_probability=.5, error_gain=0., failure_decay=.6,
                global_search=1., global_search_failure_gain=0.),
            intervention=dict(kind='E', decay=1., capacity='evidence_mean', search='accumulate'))
        module.active = np.array([0, 1]); module.executed_hypothesis = 0; module.trial_index = 1
        module.engine.posterior = np.zeros(n); module.engine.posterior[:2] = [.8, .2]
        module.engine.prior = module.engine.posterior.copy(); module._apply_mask()
        module.retention_evidence[:2] = [2., 1.]; module.retention_exposure[:2] = 1.
        return module
    return build


def test_protection_preserves_order_and_capacity():
    final, dropped = include_target(np.array([0, 1]), np.array([1, 2]), 0, 2, np.array([2., 0.]))
    assert final.tolist() == [0, 1] and dropped == 2


@pytest.mark.parametrize('arm', ['ONCE', 'ONCE_EQUAL', 'KEEP', 'KEEP_EQUAL', 'ONLY'])
def test_oracle_admitted_at_onset_and_prior_normalized(make, arm):
    model = make()
    with intervention(Transition, arm=arm, target=2, onset=1): model.process()
    assert 2 in model.active and len(model.active) <= model.capacity
    assert model.engine.prior.sum() == pytest.approx(1.)
    if arm == 'ONLY': assert model.active.tolist() == [2]
    if arm.endswith('EQUAL'): np.testing.assert_allclose(model.engine.prior[model.active], 1 / len(model.active))


def test_no_oracle_effect_before_onset_and_class_restored(make):
    original = Transition.select_hypotheses
    plain, oracle = make(), make()
    plain.process()
    with intervention(Transition, arm='KEEP_EQUAL', target=2, onset=8): oracle.process()
    np.testing.assert_array_equal(plain.engine.prior, oracle.engine.prior)
    assert plain.trial_rng.bit_generator.state == oracle.trial_rng.bit_generator.state
    assert Transition.select_hypotheses is original
    with pytest.raises(RuntimeError):
        with intervention(Transition, arm='ONLY', target=2, onset=1): raise RuntimeError()
    assert Transition.select_hypotheses is original


def test_rejection_compares_zero_new_evidence_with_positive_incumbents(make):
    forced, rejected = make(), make()
    tape = [None, dict(proposed=2, rng=deepcopy(forced.trial_rng.bit_generator.state))]
    with intervention(Transition, arm='REPLAY', target=2, onset=1, tape=tape): forced.process()
    with intervention(Transition, arm='REJECT', target=2, onset=1, tape=tape): rejected.process()
    assert forced.active.tolist() == [0, 2]
    assert rejected.active.tolist() == [0, 1]
    assert rejected.transition_log[-1]['newcomer_rejected']
    np.testing.assert_array_equal(rejected.engine.prior[:2], [.8, .2])


def test_rejection_accepts_when_new_beats_negative_old_evidence(make):
    model = make(); model.retention_evidence[1] = -1.
    tape = [None, dict(proposed=2, rng=deepcopy(model.trial_rng.bit_generator.state))]
    with intervention(Transition, arm='REJECT', target=2, onset=1, tape=tape): model.process()
    assert model.active.tolist() == [0, 2]


def test_tape_candidate_already_present_is_not_duplicated(make):
    model = make()
    tape = [None, dict(proposed=1, rng=deepcopy(model.trial_rng.bit_generator.state))]
    with intervention(Transition, arm='REJECT', target=2, onset=1, tape=tape): model.process()
    assert model.active.tolist() == [0, 1]
    assert model.transition_log[-1]['tape_already_available']


def test_keep_restores_target_after_capacity_eviction(make):
    model = make(); model.current_event_probability = 1.; model.retention_evidence[0] = -.5
    with intervention(Transition, arm='KEEP', target=0, onset=1): model.process()
    assert 0 in model.active
    assert model.transition_log[-1]['oracle_prevented_capacity']
    assert model.engine.prior.sum() == pytest.approx(1.)
