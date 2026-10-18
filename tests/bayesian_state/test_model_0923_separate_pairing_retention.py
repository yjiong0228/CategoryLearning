"""Independent two-rate Bayes updates and particle-continuation safeguards."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule as Memory
from src.Bayesian_state.model.modules.pairing_retention import pairing_retention
from tests.bayesian_state.test_model_0923_precision_memory import _memory


JOINT = np.array([[.24,.12,.04],[.06,.18,.36],[0.,0.,0.],[0.,0.,0.]])
KERNEL = np.array([[.1,.2,.7],[.5,.3,.2],[.3,.4,.3],[.2,.2,.2]])


@pytest.mark.parametrize('rule_gamma', [0., .6, 1.])
@pytest.mark.parametrize('pairing_gamma', [0., .8, 1.])
def test_two_rates_match_independent_factorized_update(rule_gamma, pairing_gamma):
    conditional = np.array([[.6,.3,.1],[.1,.3,.6],[1/3]*3])
    prior = np.array([.4,.6,0.])
    rules = np.ones(3) if rule_gamma==0 else prior**rule_gamma
    rules /= rules.sum()
    pairings = conditional**pairing_gamma
    pairings /= pairings.sum(axis=1,keepdims=True)
    joint = rules[:,None]*pairings*KERNEL[:3]
    joint /= joint.sum()
    with pairing_retention(Memory,rate=pairing_gamma) as counts:
        m=_memory(JOINT,gamma=rule_gamma,active=[1,1,1,0])
        m.prepare_feedback(KERNEL)
        np.testing.assert_allclose(m.feedback_pairing_weights[:3],pairings,atol=1e-15)
        np.testing.assert_allclose(m.feedback_evidence[:3],(pairings*KERNEL[:3]).sum(axis=1),atol=1e-15)
        m.process()
        np.testing.assert_allclose(m.joint[:3],joint,atol=1e-15)
        np.testing.assert_array_equal(m.joint[3],0.)
        assert m.gamma==rule_gamma and counts['updates']==1
        assert counts['recombined_updates']==int(rule_gamma!=pairing_gamma)


@pytest.mark.parametrize('gamma', [0., .6, 1.])
def test_equal_rates_reproduce_original_update_bit_exactly(gamma):
    old=_memory(JOINT,gamma=gamma,active=[1,1,1,0])
    old.prepare_feedback(KERNEL);old.process()
    with pairing_retention(Memory,rate=gamma):
        new=_memory(JOINT,gamma=gamma,active=[1,1,1,0])
        new.prepare_feedback(KERNEL);new.process()
        for field in ('joint','feedback_evidence','feedback_pairing_weights'):
            np.testing.assert_array_equal(getattr(new,field),getattr(old,field))


def test_pending_feedback_clone_continues_exactly_and_protocol_is_enforced():
    plain=_memory(JOINT,gamma=0.,active=[1,1,1,0]).state_dict()
    with pairing_retention(Memory,rate=.8):
        a=_memory(JOINT,gamma=0.,active=[1,1,1,0]);a.prepare_feedback(KERNEL)
        state=deepcopy(a.state_dict())
        b=_memory(JOINT,gamma=0.,active=[1,1,1,0]);b.load_state_dict(state)
        for m in (a,b):m.process()
        np.testing.assert_array_equal(a.joint,b.joint)
        with pytest.raises(ValueError,match='protocol'):
            b.load_state_dict(plain)
        for m in (a,b):
            m.engine.prior=m.engine.posterior.copy()
            m.prepare_feedback(KERNEL[:,::-1]);m.process()
        np.testing.assert_array_equal(a.joint,b.joint)
    with pytest.raises(ValueError,match='forgetting_mode'):
        _memory(JOINT).load_state_dict(state)
    with pairing_retention(Memory,rate=1.):
        with pytest.raises(ValueError,match='protocol'):
            _memory(JOINT).load_state_dict(state)


def test_disabled_context_and_error_leave_original_methods_and_gamma_intact():
    original=Memory.prepare_feedback
    with pairing_retention(Memory) as counts:
        assert Memory.prepare_feedback is original and counts['updates']==0
    with pairing_retention(Memory,rate=.8):
        m=_memory(JOINT,gamma=0.,active=[1,1,1,0])
        with pytest.raises(ValueError,match='kernel'):
            m.prepare_feedback(KERNEL*100)
        assert m.gamma==0.
        with pytest.raises(ValueError,match='already'):
            with pairing_retention(Memory,rate=.5):pass
        with pytest.raises(ValueError,match='factorized'):
            _memory(JOINT,mode='joint').prepare_feedback(KERNEL)
    assert Memory.prepare_feedback is original


@pytest.mark.parametrize('rate', [True,'0.8',-.1,1.1,np.nan])
def test_invalid_rates_are_rejected(rate):
    with pytest.raises(ValueError,match='rate'):
        with pairing_retention(Memory,rate=rate):pass


def test_actual_online_engine_preserves_pairing_cache_and_clone(monkeypatch,tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.base_module import ModuleRole
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as H
    from src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission import episodic_admission
    from src.Bayesian_state.model.modules.hypothesis_transition.episodic_proposal import episodic_proposal
    from src.Bayesian_state.model.modules.hypothesis_transition.online_evidence_belief import online_evidence_belief
    from src.Bayesian_state.model.modules.hypothesis_transition.reentry_precision import reentry_precision
    from tests.bayesian_state.test_model_0923_continuous_recalled_belief import config
    monkeypatch.setattr(ContinuousSimilarity,'RUNTIME_CACHE_DIR',tmp_path)
    cfg=config(3);cfg['modules']['memory_mod']['kwargs']['gamma']=0.
    with episodic_admission(H,mode='POOL'),episodic_proposal(H,proposal=True),reentry_precision(H),online_evidence_belief(H),pairing_retention(Memory,rate=.95):
        a=StateModel(cfg,context=ModelContext(condition=3))
        memory=a.engine.get_module(ModuleRole.MEMORY)
        for t in range(4):
            a.begin_trial([.8,.3,.2,.4]);a.complete_trial(1,.5 if t%2 else 0.)
        assert not np.allclose(memory.feedback_pairing_weights,1/3)
        state=deepcopy(a.engine.state_dict())
        b=StateModel(cfg,context=ModelContext(condition=3));b.engine.load_state_dict(state)
        for t in range(4):
            first,second=[m.begin_trial([.2,.7,.8,.6]) for m in (a,b)]
            np.testing.assert_array_equal(first.prior,second.prior)
            np.testing.assert_array_equal(first.beta,second.beta)
            for m in (a,b):m.complete_trial(2,.5 if t%2 else 1.)
            np.testing.assert_array_equal(a.engine.posterior,b.engine.posterior)
            np.testing.assert_array_equal(memory.joint,b.engine.get_module(ModuleRole.MEMORY).joint)
            assert memory.gamma==0.
