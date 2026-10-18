"""Check a removed belief-transfer channel before interpreting fitted parameters."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission import episodic_admission
from src.Bayesian_state.model.modules.hypothesis_transition.episodic_proposal import episodic_proposal
from src.Bayesian_state.model.modules.hypothesis_transition.online_evidence_belief import online_evidence_belief
from src.Bayesian_state.model.modules.hypothesis_transition.reentry_precision import reentry_precision
from tests.bayesian_state.test_model_0923_continuous_recalled_belief import config


def trajectory(condition, proposal, transfer, enabled):
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
    cfg=config(condition)
    cfg['modules']['hypo_transitions_mod']['kwargs']['transfer_scale']=transfer
    rows=[];admissions=0
    with episodic_admission(Transition, mode='POOL'), episodic_proposal(Transition, proposal=proposal), reentry_precision(Transition), online_evidence_belief(Transition, enabled=enabled):
        model=StateModel(cfg,context=ModelContext(condition=condition))
        h=model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for trial in range(16):
            pending=model.begin_trial([.8,.3,.2,.4] if trial%2 else [.2,.7,.8,.6])
            admissions+=len(h.last_transition_result.selection.newcomers)
            row=dict(prior=pending.prior.copy(),beta=pending.beta.copy(),active=h.active.copy(),
                     execution=h.execution_probabilities.copy(),evidence=h.retention_evidence.copy(),
                     rng=deepcopy(h.trial_rng.bit_generator.state))
            if condition==3:row['pairing']=pending.log['conditional_pairing_pre'].copy()
            model.complete_trial(1,.5 if condition==3 and trial%3==0 else float(trial%2))
            row['posterior']=model.engine.posterior.copy();rows.append(row)
    return rows,admissions


@pytest.mark.parametrize('condition',[1,2,3])
@pytest.mark.parametrize('proposal',[False,True])
def test_online_transfer_extremes_leave_actual_rule_process_unchanged(condition,proposal,monkeypatch,tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    monkeypatch.setattr(ContinuousSimilarity,'RUNTIME_CACHE_DIR',tmp_path)
    a,admitted=trajectory(condition,proposal,.01,True)
    b,other=trajectory(condition,proposal,1.,True)
    assert admitted==other and admitted>0
    for x,y in zip(a,b):
        for key in x:
            if key=='rng':assert x[key]==y[key]
            else:np.testing.assert_array_equal(x[key],y[key])


def test_original_belief_transfer_responds_to_the_same_manipulation(monkeypatch,tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    monkeypatch.setattr(ContinuousSimilarity,'RUNTIME_CACHE_DIR',tmp_path)
    a,first=trajectory(2,False,.01,False)
    b,second=trajectory(2,False,1.,False)
    assert first and second
    assert any(not np.array_equal(x['prior'],y['prior']) for x,y in zip(a,b))
