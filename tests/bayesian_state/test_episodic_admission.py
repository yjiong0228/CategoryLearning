"""Causal recall, comparable evidence, stochastic retrieval and particle resume."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
from src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission import (
    Episode, recalled_episodes, recalled_log_evidence, comparison_weights, episodic_admission)


def config(condition):
    cfg = yaml.safe_load(Path(f'configs/exp123/model_struct/model_0923_cond{condition}_full_v3.yaml').read_text())
    cfg['oral_readout'] = None
    cfg['modules']['perception_mod']['kwargs'] = dict(features=4, mean=[0.]*4, std=[0.]*4, module_seed=2)
    h = cfg['modules']['hypo_transitions_mod']['kwargs']
    h.update(intervention=dict(kind='E', decay=.8, capacity='evidence_mean', search='accumulate'),
             relation_samples=128, module_seed=3, max_capacity=2, exit_rate=0.)
    h['search_controller']['baseline_probability'] = .999
    cfg['provenance']['hypothesis_similarity']['sample_count'] = 128
    return cfg


def test_history_is_immutable_and_has_no_fixed_window():
    head = None
    for i in range(40):head = Episode((float(i),), 1, float(i % 2), None, head)
    assert deepcopy(head) is head
    with pytest.raises(FrozenInstanceError):head.choice = 2
    episodes, weights = recalled_episodes(head, .8)
    assert len(episodes) == 40 and episodes[-1].stimulus == (0.,)
    np.testing.assert_allclose(weights, .8**np.arange(40), rtol=1e-14)
    assert len(recalled_episodes(head, 0.)[0]) == 1
    np.testing.assert_array_equal(recalled_episodes(head, 1.)[1], np.ones(40))
    for decay in (-1., 1.01, np.nan):
        with pytest.raises(ValueError):recalled_episodes(head, decay)


def test_dirichlet_retrieval_preserves_mean_and_shared_episode_covariation():
    weights = np.array([1., .5, .25])
    rng = np.random.default_rng(20261017)
    draws = np.array([comparison_weights(weights, 'DRAW', rng) for _ in range(20000)])
    np.testing.assert_allclose(draws.sum(axis=1), 1., rtol=0, atol=1e-14)
    np.testing.assert_allclose(draws.mean(axis=0), weights/weights.sum(), atol=.01)
    # Identical rule evidence receives identical sampled scores, even when
    # retrieval weights fluctuate substantially across judgments.
    evidence = np.array([[.3,.3],[-2.,-2.],[.5,.5]])
    np.testing.assert_array_equal((draws @ evidence)[:,0], (draws @ evidence)[:,1])
    state = deepcopy(rng.bit_generator.state)
    np.testing.assert_array_equal(comparison_weights([1.], 'DRAW', rng), [1.])
    assert rng.bit_generator.state == state


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_batched_recall_matches_original_one_trial_emissions(condition, monkeypatch, tmp_path):
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    model = StateModel(config(condition), context=ModelContext(condition=condition))
    likelihood = model.engine.observation_likelihood
    pairings = [(1.,0.,0.),(.2,.3,.5),(.1,.8,.1)] if condition == 3 else [None]*3
    feedbacks = [1., .5, 0.] if condition == 3 else [1., 0., 1.]
    episodes = [Episode(tuple(x), y, f, w) for x,y,f,w in zip(
        [[.8,.1,.5,.3],[.2,.7,.3,.8],[.5,.4,.3,.6]], [1,2,1], feedbacks, pairings)]
    rules = np.array([0, 4, 7]); beta = 9.
    actual, maximum = recalled_log_evidence(likelihood, episodes, rules, beta)
    expected = []
    for e in episodes:
        observation = (e.stimulus,e.choice,e.feedback)
        if condition == 3:
            value = likelihood.pairing_kernel(observation,rules,beta) @ e.pairing
            chance = .5 if e.feedback == 0 else .25
        else:
            value = likelihood.binary_feedback_likelihood(observation,rules,beta)
            n = model.engine.partition.n_cats
            chance = 1/n if e.feedback == 1 else 1-1/n
        expected.append(np.log(np.maximum(value,np.finfo(float).tiny))-np.log(chance))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert maximum <= 1+8*np.finfo(float).eps


@pytest.mark.parametrize('condition', [1, 2, 3])
@pytest.mark.parametrize('mode', ['NEW', 'POOL', 'DRAW'])
def test_particle_restore_preserves_feedback_history_and_future_choices(condition, mode, monkeypatch, tmp_path):
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    original = Transition.record_outcome
    with episodic_admission(Transition, mode=mode):
        a = StateModel(config(condition), context=ModelContext(condition=condition))
        a.begin_trial([.8,.3,.2,.4])
        memory = a.engine.get_module(ModuleRole.MEMORY)
        pairing = tuple(memory.pairing_marginal()) if condition == 3 else None
        a.complete_trial(1,.5 if condition == 3 else 0.)
        ha = a.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        assert ha.admission_history.pairing == pairing
        saved = a.engine.state_dict()
        b = StateModel(config(condition), context=ModelContext(condition=condition))
        b.engine.load_state_dict(deepcopy(saved))
        hb = b.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        assert hb.admission_history is ha.admission_history
        for t in range(5):
            pa,pb = [m.begin_trial([.2,.7,.8,.6]) for m in (a,b)]
            np.testing.assert_array_equal(pa.prior,pb.prior)
            np.testing.assert_array_equal(ha.active,hb.active)
            assert ha.trial_rng.bit_generator.state == hb.trial_rng.bit_generator.state
            event = ha.transition_log[-1]
            assert event['recalled_episodes'] == t+1
            assert np.isnan(event['admission_probability_given_search'])
            if ha.last_transition_result.selection.newcomers.size:
                h = ha.last_transition_result.selection.newcomers[0]
                assert ha.retention_exposure[h] == pytest.approx(sum(.8**np.arange(t+1)))
                assert ha.retention_evidence[h] == pytest.approx(event['candidate_initial_evidence'])
            for m in (a,b):m.complete_trial(1,float(t % 2))
        # Extending one particle's immutable history cannot mutate another.
        old = hb.admission_history
        a.begin_trial([.1,.2,.3,.4]);a.complete_trial(1,0.)
        assert hb.admission_history is old and ha.admission_history is not old
    assert Transition.record_outcome is original


@pytest.mark.parametrize('candidate_score,accepted', [(-1.,False),(1.,True)])
def test_pool_admission_can_reject_and_preserves_nonempty_capacity(candidate_score,accepted,monkeypatch,tmp_path):
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    import src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission as helper
    with episodic_admission(Transition,mode='POOL'):
        model=StateModel(config(1),context=ModelContext(condition=1))
        h=model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        h.active=np.array([0,1]);h.executed_hypothesis=0;h.trial_index=1
        model.engine.posterior=np.zeros(h.total_hypo);model.engine.posterior[:2]=[.8,.2]
        model.engine.prior=model.engine.posterior.copy();h._apply_mask()
        h.record_outcome(([.8,.3,.2,.4],1,1.));h.outcome_pending=False
        h.current_event_probability=1.
        monkeypatch.setattr(helper,'recalled_log_evidence',lambda *args:(np.array([[2.,0.,candidate_score]]),.9))
        h.process();event=h.transition_log[-1]
        assert len(h.active)==2
        assert (event['proposed_hypothesis'] in h.active)==accepted
        assert event.get('newcomer_rejected',False)==(not accepted)
