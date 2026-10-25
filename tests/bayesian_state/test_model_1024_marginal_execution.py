"""Execution marginalization, shared-state independence and genealogy timing."""
from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from src.Bayesian_state.evaluation.marginal_execution import (
    execution_posterior,MarginalExecutionCollector,marginal_smoothing)
from src.Bayesian_state.inference.backends.particle_filter import _trace_ancestral_indices
from src.Bayesian_state.model import ModelContext,StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.beta import BetaModule
from src.Bayesian_state.model.modules.precision_interventions import precision_intervention
from src.Bayesian_state.model.modules.readout.choice import (
    read_choice_probabilities_from_model,resolve_choice_readout_config,resolve_output_noise_config)
from tests.bayesian_state.test_model_1023_belief_execution import installed,compact_config


@pytest.fixture(autouse=True)
def isolated_similarity(monkeypatch,tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    monkeypatch.setattr(ContinuousSimilarity,'RUNTIME_CACHE_DIR',tmp_path)


def test_posterior_is_current_choice_conditioned_not_the_execution_prior():
    prior=np.array([[.25,.75],[.6,.4]])
    likelihood=np.array([[[.9,.1],[.1,.9]],[[.2,.8],[.8,.2]]])
    mass,q=execution_posterior(prior,likelihood,1)
    np.testing.assert_allclose(mass,[.3,.44])
    np.testing.assert_allclose(q,[[.75,.25],[3/11,8/11]])
    # Explicit categorical enumeration gives the same choice and event joint.
    for p in range(2):
        joint=np.array([prior[p,h]*likelihood[p,h,0] for h in range(2)])
        np.testing.assert_allclose(q[p],joint/joint.sum())


def test_zero_mass_rule_is_ignored_and_impossible_choice_fails():
    mass,q=execution_posterior(np.array([[1.,0.]]),np.array([[[1.,0.],[0.,0.]]]),1)
    np.testing.assert_array_equal(q,[[1.,0.]])
    with pytest.raises(ValueError,match='impossible'):
        execution_posterior(np.array([[1.,0.]]),np.array([[[1.,0.],[0.,0.]]]),2)


def shared_history(condition,force):
    cfg=compact_config(condition);cfg['modules']['hypo_transitions_mod']['kwargs']['max_capacity']=2
    rows=[]
    with precision_intervention(BetaModule,mode='fast',upper=25.), installed('MIX' if force is None else 'STICKY'):
        model=StateModel(cfg,context=ModelContext(condition=condition));h=model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for t in range(8):
            x=[.15+.08*t,.75-.05*t,.25,.65];prepared=model.begin_trial(x)
            prior=h.execution_probabilities.copy() if force is None else h.execution_selection_probabilities.copy()
            # The counterfactual changes the overt rule and its cached choice
            # distribution. Feedback, physical inputs and learning remain fixed.
            if force is not None:
                selected=int(h.active[(t+force)%len(h.active)])
                h.executed_hypothesis=selected;h.execution_probabilities[:]=0.;h.execution_probabilities[selected]=1.
            _,snapshot=read_choice_probabilities_from_model(model,prepared.perceived_stimulus,return_snapshot=True)
            ids=np.array(snapshot.hypothesis_indices);likelihood=snapshot.category_probabilities
            marginal=prior[ids]@likelihood
            choice=t%model.n_cats+1;feedback=(0.,.5,1.)[t%3] if condition==3 else float(t%2)
            model.predict_choice(choices=[],feedback=[],
                choice_readout_config=resolve_choice_readout_config(None,model.engine_config),
                output_noise_config=resolve_output_noise_config(None,model.engine_config),
                rng=np.random.default_rng(71+t),sticky_state={},post_error_lapse_state=0.)
            row=dict(active=h.active.copy(),belief=prepared.prior.copy(),beta=prepared.beta.copy(),
                perceived=prepared.perceived_stimulus.copy(),search_rng=deepcopy(h.trial_rng.bit_generator.state),
                probability=marginal,prior=prior[ids],likelihood=likelihood[:,choice-1],choice=choice,
                actual_choice_probability=model._oral_choice_prediction.observed_probabilities.copy())
            model.complete_trial(choice,feedback)
            row.update(posterior=model.engine.posterior.copy(),beta_after=model.engine.beta.copy(),
                failure=h.accumulated_failure,search_probability=h.current_event_probability)
            memory=model.engine.get_module(ModuleRole.MEMORY)
            if condition==3:row['pairing']=memory.conditional_pairing().copy()
            rows.append(row)
    return rows


@pytest.mark.parametrize('condition',[1,2,3])
def test_execution_identity_and_cached_readout_do_not_change_shared_learning(condition):
    mixture=shared_history(condition,None);first=shared_history(condition,0);second=shared_history(condition,1)
    for a,b,c in zip(mixture,first,second):
        for key in ('active','belief','beta','perceived','posterior','beta_after','probability','failure','search_probability'):
            np.testing.assert_allclose(a[key],b[key],atol=1e-14,rtol=0)
            np.testing.assert_allclose(a[key],c[key],atol=1e-14,rtol=0)
        assert a['search_rng']==b['search_rng']==c['search_rng']
        if condition==3:
            np.testing.assert_array_equal(a['pairing'],b['pairing']);np.testing.assert_array_equal(a['pairing'],c['pairing'])
        np.testing.assert_allclose(a['actual_choice_probability'],a['probability'],atol=1e-14,rtol=0)


@pytest.mark.parametrize('condition',[1,2,3])
def test_complete_execution_enumeration_matches_shared_model_sequence_likelihood(condition):
    rows=shared_history(condition,None)
    paths=list(product(*[range(len(r['prior'])) for r in rows]));weights=[]
    for path in paths:weights.append(np.prod([r['prior'][j]*r['likelihood'][j] for r,j in zip(rows,path)]))
    weights=np.array(weights);exact=float(weights.sum())
    expected=float(np.prod([r['probability'][r['choice']-1] for r in rows]))
    assert exact==pytest.approx(expected,rel=1e-13,abs=1e-18)
    for t,row in enumerate(rows):
        posterior=np.bincount([p[t] for p in paths],weights=weights/exact,minlength=len(row['prior']))
        local=row['prior']*row['likelihood'];local/=local.sum()
        np.testing.assert_allclose(posterior,local,atol=1e-13,rtol=0)


def test_collector_and_smoothing_keep_conditional_execution_mass():
    c=MarginalExecutionCollector(np.array([1,1]),[0]);prior=np.array([[.25,.75],[0.,1.]])
    likelihood=np.array([[[.9,.1],[.1,.9]],[[0.,0.],[.2,.8]]])
    snapshot=dict(active=np.array([[1,1],[0,1]]),belief=prior,choice_weights=prior,
        rule_choice_probabilities=likelihood,prediction=np.einsum('ph,phc->pc',prior,likelihood),weights=np.array([.5,.5]))
    for t in range(2):
        c(t,'predictive',snapshot)
        c(t,'filtered',dict(weights=np.array([.6,.4])))
    arrays=c.arrays();weights=arrays.pop('filtered_weights');arrays.pop('predictive_target_execution')
    out=marginal_smoothing(arrays,np.array([[1,1],[0,0]]),weights,_trace_ancestral_indices,1,0)
    np.testing.assert_allclose(out['filtered_target_execution'],[.45,.45])
    np.testing.assert_allclose(out['filtered_source'],[[.4,.15,.45],[.4,.15,.45]])
    np.testing.assert_allclose(out['lag_target_execution'],out['filtered_target_execution'])
    np.testing.assert_allclose(out['full_target_execution'],[0.,.45])
    np.testing.assert_allclose(out['prefix_target_execution'],[.45])
    # Readout-prior averaging would have yielded .6*.25=.15, not .45.
    assert c.predictive==pytest.approx([.125,.125])


@pytest.mark.parametrize('change',[
    'inertia','executed_learning','mapping','readout_power','lapse'])
def test_exact_marginalizer_rejects_unsupported_couplings(change):
    from src.Bayesian_state.workflows.analysis.model_1024_marginal_execution import validate_carrier
    values=dict(engine_config=dict(modules=dict(
        hypo_transitions_mod=dict(kwargs=dict(execution_inertia=0.,exit_rate=0.,mechanism_version='0923-full-v3')),
        beta_mod=dict(kwargs=dict(update_scope='active_hypotheses')))),choice_readout_power=1.,output_lapse=0.)
    validate_carrier(values)
    if change=='inertia':values['engine_config']['modules']['hypo_transitions_mod']['kwargs']['execution_inertia']=1.
    elif change=='executed_learning':values['engine_config']['modules']['beta_mod']['kwargs']['update_scope']='executed_hypothesis'
    elif change=='mapping':values['engine_config']['modules']['mapping_mod']={}
    elif change=='readout_power':values['choice_readout_power']=2.
    else:values['output_lapse']=.1
    with pytest.raises(ValueError,match='zero inertia'):validate_carrier(values)


def test_execution_count_variance_retains_uncertainty_within_each_latent_path():
    from src.Bayesian_state.workflows.analysis.model_1024_marginal_execution_summary import count_moments
    p=np.array([[.2,.7],[.3,.8],[.6,.1]]);w=np.array([.4,.6])
    mean,variance,within=count_moments(p,w,integrated=True)
    counts=[];mass=[]
    for j in range(2):
        for path in product([0,1],repeat=3):
            counts.append(sum(path));mass.append(w[j]*np.prod([p[t,j] if y else 1-p[t,j] for t,y in enumerate(path)]))
    counts=np.array(counts);mass=np.array(mass)
    assert mean==pytest.approx(mass@counts)
    assert variance==pytest.approx(mass@np.square(counts-mean))
    assert within>0 and variance>np.var(p.sum(0))
    mean,var,within=count_moments(np.array([[1,0],[1,0]]),w,integrated=False)
    assert mean==pytest.approx(.8) and var==pytest.approx(.96) and within==0
