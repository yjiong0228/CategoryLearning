"""Keep mechanism changes, behavioral selection and feedback units explicit."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis import model_0923_online_evidence_refit_oral as oral


def identities():
    before=dict(scientific={'particles':512,'trials':576},arm_settings={'precision':'fast'},
        mechanism={'kind':'E','retain_learned_precision':True},archived_pf_sha256='same',
        task_sha256={'training':'same'},oral_manifest_sha256='same',model_sources={'original':'hash'},
        candidate={'id':'old','hyperparams':{'capacity':6}},structural_intervention='retained_rule_precision',
        parameter_estimation_scope='prefix',parameter_candidate_policy='common_search_policy_behavior_refit')
    after=deepcopy(before)
    after['mechanism']['online_belief']=True
    after['candidate']={'id':'new','hyperparams':{'capacity':4}}
    after['parameter_candidate_policy']='projected_common_search_policy_behavior_refit'
    after['model_sources']['online']='new_hash'
    return before,after


def test_allows_reestimation_only_alongside_declared_online_change():
    before,after=identities();oral.validate_identities(before,after)
    with pytest.raises(ValueError):oral.comparison.validate_comparison(before,after,'parameters')
    assert after['mechanism']['online_belief'] is True


def test_replaced_wrapper_requires_its_actual_bound_execution_hash():
    before,after=identities();del after['model_sources']['original']
    with pytest.raises(ValueError,match='dependency'):oral.validate_identities(before,after)
    oral.validate_identities(before,after,executed_sources={'original':'hash'})
    with pytest.raises(ValueError,match='dependency'):
        oral.validate_identities(before,after,executed_sources={'original':'wrong'})
    after['model_sources']['original']='changed'
    with pytest.raises(ValueError,match='dependency'):
        oral.validate_identities(before,after,executed_sources={'original':'hash'})


@pytest.mark.parametrize('fault',['reversed','no_online','no_retention','full_fit','oral_selected',
    'budget','oral','task','other_mechanism','source'])
def test_rejects_undeclared_mechanism_or_data_use_changes(fault):
    before,after=identities()
    if fault=='reversed':before,after=after,before
    elif fault=='no_online':after['mechanism']['online_belief']=False
    elif fault=='no_retention':before['mechanism']['retain_learned_precision']=False
    elif fault=='full_fit':after['parameter_estimation_scope']='full'
    elif fault=='oral_selected':after['parameter_candidate_policy']='oral_selected'
    elif fault=='budget':after['scientific']['particles']=128
    elif fault=='oral':after['oral_manifest_sha256']='changed'
    elif fault=='task':after['task_sha256']['training']='changed'
    elif fault=='other_mechanism':after['mechanism']['kind']='new'
    elif fault=='source':after['model_sources']['original']='changed'
    with pytest.raises(ValueError):oral.validate_identities(before,after)


def nomination():
    identity=identities()[1]
    frozen=dict(status='frozen_before_test',smoke=False,settings={'seed':19},source_sha256=identity['model_sources'],
                nominees=[dict(subject=208,model='SEARCH',candidate=deepcopy(identity['candidate']))])
    protocol={key:deepcopy(frozen[key]) for key in ('smoke','settings','source_sha256')}
    done=dict(status='complete',smoke=False,test_choices_used=False,nomination_sha256='nomination')
    manifest=dict(refit_nomination_sha256='nomination')
    return identity,manifest,frozen,protocol,done,'nomination',208,'SEARCH'


def test_accepts_the_exact_behavior_nomination():
    oral.validate_nomination(*nomination())


@pytest.mark.parametrize('fault',['test_choices','incomplete','point','digest','scope_settings','source','duplicates'])
def test_rejects_oral_points_unbound_to_the_original_prefix_nomination(fault):
    args=nomination();identity,manifest,frozen,protocol,done=args[:5]
    if fault=='test_choices':done['test_choices_used']=True
    elif fault=='incomplete':done['status']='running'
    elif fault=='point':identity['candidate']['id']='different'
    elif fault=='digest':manifest['refit_nomination_sha256']='other'
    elif fault=='scope_settings':protocol['settings']['seed']=20
    elif fault=='source':identity['model_sources']['online']='changed'
    elif fault=='duplicates':frozen['nominees']*=2
    with pytest.raises(ValueError):oral.validate_nomination(*args)


def test_feedback_comparison_preserves_empty_strata_and_complete_filter_pairing():
    p=np.array([[.1,.3,.2,.6],[.9,.7,.8,.4]])
    support=np.array([[.9,.2,.8,.3],[.1,.6,.2,.7]])
    fractions=np.stack((support,np.zeros_like(support),1-support),axis=-1)
    joint=np.repeat(fractions[:,:,None,:],2,axis=2)*p[:,:,None,None]
    prior=np.broadcast_to([.4,.1,.5],joint.shape).copy()
    estimates=oral.strata.stratified_estimates(joint,p,prior,np.array([1,1,0,1],bool),np.array([0.,1.,.5,1.]),2)
    trials=pd.DataFrame({key:[str(key)]*4 for key in oral.comparison.IDENTITY})
    bank=dict(seeds=np.array([3,5]),trials=trials,feedback=np.array([0.,1.,.5,1.]),categories=np.array([1,1,2,1]),estimates=estimates)
    scores,deleted=oral.feedback_comparison(bank,deepcopy(bank),208,'SEARCH')
    assert len(scores)==72 and len(deleted)==24
    nonempty=scores.complete_reports>0
    assert np.all(scores.loc[nonempty].filter(regex='difference').to_numpy()==0)
    assert scores.loc[~nonempty].filter(regex='difference').isna().all().all()
    changed=deepcopy(bank);changed['seeds']=changed['seeds'][::-1]
    with pytest.raises(AssertionError):oral.feedback_comparison(bank,changed,208,'SEARCH')
