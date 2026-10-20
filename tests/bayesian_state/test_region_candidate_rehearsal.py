from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import softmax

from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.active_rule_associations import ActiveRuleAssociations
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission import Episode, recalled_episodes
from src.Bayesian_state.model.modules.hypothesis_transition.finite_region_learning import (
    finite_region_learning, frozen_association_recall,
)
from src.Bayesian_state.model.modules.hypothesis_transition.region_candidate_rehearsal import rehearse_candidate
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule as Transition,
)
from src.Bayesian_state.model.modules.readout.choice import read_choice_probabilities_from_model
from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
from tests.bayesian_state.test_finite_region_learning import configuration


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)


@pytest.mark.parametrize('decay', [0., .5, 1.])
def test_chronological_discounted_replay_matches_independent_updates(decay):
    sequence = [((.1,.2,.3,.4),1,1.), ((.7,.1,.1,.1),2,0.), ((.2,.3,.4,.1),4,1.)]
    head = None
    for stimulus, choice, feedback in sequence: head = Episode(stimulus, choice, feedback, None, head)
    bank = ActiveRuleAssociations(3,4,4,learning_rate=.7)
    bank.commit_feedback(bank.evaluate_feedback([0],[[.1,.2,.3,.4]],3,1.))
    before = bank.state_dict()
    partition = SimpleNamespace(get_category_probabilities=lambda h,data,beta,distance_mode:np.asarray(data[0]).T)
    evaluator = SimpleNamespace(partition=partition,distance_mode='boundary')
    transition = SimpleNamespace(region_associations=bank,reentry_precision_seen=np.array([True,False,False]),
        admission_history=head,retention_decay=decay,
        engine=SimpleNamespace(observation_likelihood=evaluator,get_module=lambda role:SimpleNamespace(beta_init=5.)))
    result = rehearse_candidate(transition,1)
    logits = np.zeros((4,4));count=0;exposure=0.
    for age,(stimulus,choice,feedback) in zip((2,1,0),sequence):
        weight = decay**age
        if weight==0: continue
        a=softmax(logits,axis=-1);allowed=np.arange(4)==choice-1
        if feedback==0:allowed=~allowed
        responsibilities=np.asarray(stimulus)[:,None]*a*allowed
        responsibilities/=responsibilities.sum()
        logits+=.7*weight*(responsibilities-responsibilities.sum(-1,keepdims=True)*a)
        logits-=logits.mean(-1,keepdims=True);count+=1;exposure+=weight
    np.testing.assert_allclose(result.scratch.state_dict()['logits'][1],logits,atol=1e-14,rtol=0)
    assert result.count==count and result.exposure==exposure
    np.testing.assert_array_equal(bank.state_dict()['logits'],before['logits'])
    np.testing.assert_array_equal(result.scratch.state_dict()['logits'][[0,2]],before['logits'][[0,2]])
    transition.reentry_precision_seen[1]=True
    with pytest.raises(ValueError,match='unseen'):rehearse_candidate(transition,1)


def test_admitted_candidate_keeps_rehearsal_and_gamma_score_uses_predictive_beta():
    with finite_region_learning(Transition,learning_rate=.7,rehearse_candidates=True):
        model=StateModel(configuration(),context=ModelContext(condition=2))
        h=model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        model.begin_trial([.8,.3,.2,.4]);model.complete_trial(1,1.)
        before=h.region_associations.state_dict()
        model.begin_trial([.2,.3,.8,.4])
        candidate=int(h.last_transition_result.selection.newcomers[0])
        assert h.transition_log[-1]['candidate_rehearsal_committed']
        assert h.region_rehearsal_count[candidate]==1
        assert h.region_rehearsal_exposure[candidate]==1.
        assert not np.all(h.region_associations.associations([candidate])==.25)
        beta=model.engine.get_module(ModuleRole.BETA)
        assert model.engine.beta[candidate]==beta.beta_init
        episodes,weights=recalled_episodes(h.admission_history,.3)
        expected,_=frozen_association_recall(model.engine.observation_likelihood,episodes,np.array([candidate]),beta.beta_init)
        np.testing.assert_allclose(h.belief_evidence[candidate],weights@expected[:,0],atol=1e-12,rtol=0)
        others=np.setdiff1d(np.arange(h.total_hypo),[candidate])
        np.testing.assert_array_equal(h.region_associations.state_dict()['logits'][others],before['logits'][others])
        assert h._candidate_rehearsal is None


def test_rejected_scratch_bank_is_discarded(monkeypatch):
    with finite_region_learning(Transition,learning_rate=.7,rehearse_candidates=True):
        model=StateModel(configuration(),context=ModelContext(condition=2))
        h=model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for _ in range(2):model.begin_trial([.8,.3,.2,.4]);model.complete_trial(1,1.)
        original=Transition._retention_scores
        def reject(self,rules,mean=False):
            values=original(self,rules,mean);values[-1]=values.min()-1.;return values
        monkeypatch.setattr(Transition,'_retention_scores',reject)
        bank=h.region_associations;before=bank.state_dict();counts=h.region_rehearsal_count.copy()
        model.begin_trial([.2,.3,.8,.4])
        event=h.transition_log[-1]
        assert event['candidate_rehearsal'] and event['newcomer_rejected']
        assert not event['candidate_rehearsal_committed']
        assert h.region_associations is bank
        np.testing.assert_array_equal(bank.state_dict()['logits'],before['logits'])
        np.testing.assert_array_equal(bank.state_dict()['feedback_counts'],before['feedback_counts'])
        np.testing.assert_array_equal(h.region_rehearsal_count,counts)


def test_reentry_restores_previous_associations_without_rehearsing_again(monkeypatch):
    def draw(self,before,remaining,gamma,diagnostics):
        candidate=next(rule for rule in (0,1,2) if rule not in before)
        diagnostics.update(proposal_source='test_schedule',proposed_hypothesis=candidate)
        return candidate
    monkeypatch.setattr(Transition,'_draw_v3_proposal',draw)
    with finite_region_learning(Transition,learning_rate=.5,rehearse_candidates=True):
        ordinary=Transition._retention_scores
        def admit(self,rules,mean=False):
            values=ordinary(self,rules,mean);values[0]=values.min()-1.;return values
        monkeypatch.setattr(Transition,'_retention_scores',admit)
        model=StateModel(configuration(),context=ModelContext(condition=2));h=model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        returns=0
        for trial in range(9):
            before=h.region_associations.state_dict();counts=h.region_rehearsal_count.copy()
            model.begin_trial([.8,.3,.2,.4]);event=h.transition_log[-1]
            if event.get('candidate_previously_active'):
                returns+=1;candidate=event['proposed_hypothesis']
                assert not event['candidate_rehearsal'] and not event['candidate_rehearsal_committed']
                np.testing.assert_array_equal(h.region_associations.state_dict()['logits'][candidate],before['logits'][candidate])
                np.testing.assert_array_equal(h.region_rehearsal_count,counts)
            model.complete_trial(1,float(trial%2))
        assert returns>=2


def test_snapshot_continues_exactly_and_rehearsal_fields_are_validated():
    cfg=configuration()
    with finite_region_learning(Transition,learning_rate=.5,rehearse_candidates=True):
        a=StateModel(cfg,context=ModelContext(condition=2))
        for trial in range(4):a.begin_trial([.8,.3,.2,.4]);a.complete_trial(1,float(trial%2))
        b=StateModel(cfg,context=ModelContext(condition=2));b.engine.load_state_dict(deepcopy(a.engine.state_dict()))
        ha,hb=[m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION) for m in (a,b)]
        assert ha.region_associations is not hb.region_associations
        for trial in range(3):
            for m in (a,b):m.begin_trial([.2,.3,.8,.4])
            pa,pb=[read_choice_probabilities_from_model(m,[.2,.3,.8,.4]) for m in (a,b)]
            np.testing.assert_array_equal(pa,pb)
            np.testing.assert_array_equal(ha.region_rehearsal_count,hb.region_rehearsal_count)
            np.testing.assert_array_equal(ha.region_rehearsal_exposure,hb.region_rehearsal_exposure)
            for m in (a,b):m.complete_trial(3,float(trial%2))
        for key,value in [('region_rehearsal_count',np.full(ha.total_hypo,-1)),
                           ('region_rehearsal_exposure',np.full(ha.total_hypo,np.nan))]:
            state=ha.state_dict();state[key]=value
            with pytest.raises(ValueError,match='rehearsal snapshot'):ha.load_state_dict(state)


def test_particle_filter_rehearsal_is_causal_and_reproducible():
    kwargs=dict(engine_config=configuration(),subject_id=207,condition=2,
        stimulus=np.random.default_rng(78).uniform(.05,.95,(6,4)),choices=np.tile([1,2],3),
        particle_count=4,choice_readout_power=1.,filter_seed=78,resample_threshold_fraction=1.)
    feedback=np.tile([1.,0.],3)
    with finite_region_learning(Transition,learning_rate=.5,rehearse_candidates=True):
        a=run_state_model_particle_filter(feedback=feedback,**kwargs)
        b=run_state_model_particle_filter(feedback=feedback,**kwargs)
        altered=feedback.copy();altered[-1]=1.
        c=run_state_model_particle_filter(feedback=altered,**kwargs)
    np.testing.assert_array_equal(a.marginal_probabilities,b.marginal_probabilities)
    np.testing.assert_array_equal(a.marginal_probabilities,c.marginal_probabilities)
    np.testing.assert_array_equal(a.final_weights,b.final_weights)
    assert a.resampled.any()


def test_invalid_rehearsal_switch_rejected():
    with pytest.raises(ValueError,match='explicitly'):
        with finite_region_learning(Transition,learning_rate=.5,rehearse_candidates=1):pass


def test_zero_association_rate_preserves_behavior_and_proposal_stream():
    histories=[]
    for enabled in (False,True):
        rows=[]
        with finite_region_learning(Transition,learning_rate=0.,rehearse_candidates=enabled):
            model=StateModel(configuration(),context=ModelContext(condition=2))
            h=model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
            for trial in range(6):
                model.begin_trial([.8,.3,.2,.4])
                rows.append((h.active.copy(),model.engine.prior.copy(),model.engine.beta.copy(),
                             read_choice_probabilities_from_model(model,[.8,.3,.2,.4])))
                model.complete_trial(1,float(trial%2))
        histories.append(rows)
    for first,second in zip(*histories):
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
