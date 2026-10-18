"""Verify ONLINE execution counts and additive feedback loss at pooled filters."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_online_refit import check_online_execution, feedback_scores


def execution(phase):
    fixed = {'transfer':.01}
    job = dict(phase=phase,seeds=[3,7],particles=5,online_belief=True,
        belief_intervention='online_retention_evidence',fixed_coordinates=fixed,
        candidate={'id':'frozen','hyperparams':dict(fixed)})
    record = dict(candidate=deepcopy(job['candidate']),trials=4)
    record.update(paths=len(job['seeds'])) if phase=='autonomous' else record.update(seeds=job['seeds'])
    sidecar = dict(enabled=True,belief_intervention=job['belief_intervention'],fixed_coordinates=fixed,
        exact_transition_timing=True,evidence_matches_retained_state=True,
        reconstructions=6 if phase=='autonomous' else 30)
    return job,record,sidecar,fixed


@pytest.mark.parametrize('phase',['evaluate','autonomous'])
def test_counts_filters_and_generated_paths_separately(phase):
    check_online_execution(*execution(phase))


@pytest.mark.parametrize('fault',['disabled','timing','state','count','coordinate','candidate','seed','phase'])
def test_rejects_changed_execution_or_nominated_point(fault):
    job,record,sidecar,fixed = execution('autonomous')
    if fault=='disabled':job['online_belief']=False
    elif fault=='timing':sidecar['exact_transition_timing']=False
    elif fault=='state':sidecar['evidence_matches_retained_state']=False
    elif fault=='count':sidecar['reconstructions']=30
    elif fault=='coordinate':job['candidate']['hyperparams']['transfer']=.2
    elif fault=='candidate':record['candidate']['id']='other'
    elif fault=='seed':record['paths']=1
    elif fault=='phase':job['phase']='training'
    with pytest.raises(ValueError):check_online_execution(job,record,sidecar,fixed)


def banks():
    old = dict(seeds=np.array([1,2,3]),choices=np.array([1,2,3,4]),
               observed_probability=np.array([[.1,.9,.2,.3],[.9,.1,.4,.6],[.2,.8,.3,.9]]))
    new = deepcopy(old)
    new['observed_probability'] = np.array([[.3,.8,.7,.3],[.7,.4,.9,.4],[.5,.6,.8,.8]])
    return old,new,np.array([0.,.5,1.,1.]),2


def test_feedback_parts_sum_to_whole_window_and_delete_complete_filters():
    old,new,feedback,cut = banks()
    scores,trials = feedback_scores(old,new,feedback,cut)
    direct = np.log(old['observed_probability'].mean(0))-np.log(new['observed_probability'].mean(0))
    np.testing.assert_allclose(trials.delta_choice_nll,direct)
    for phase,indices in [('training',[0,1]),('test',[2,3]),('whole',[0,1,2,3])]:
        table = scores[scores.phase==phase]
        whole = table[table.feedback_group=='all'].iloc[0]
        parts = table[table.feedback_group!='all']
        assert whole.delta_choice_nll == pytest.approx(direct[indices].mean())
        assert parts.delta_total_nll.sum() == pytest.approx(direct[indices].sum())
        assert parts.contribution_to_window_mean.sum() == pytest.approx(whole.delta_choice_nll)
        estimates=[]
        for i in range(3):
            p,q=(np.delete(bank['observed_probability'],i,axis=0)[:,indices].mean(0) for bank in (new,old))
            estimates.append(np.mean(np.log(q)-np.log(p)))
        assert whole.delta_choice_nll_mc_se == pytest.approx(np.sqrt(2/3*np.sum((estimates-np.mean(estimates))**2)))
    empty = scores[scores.trials==0]
    assert len(empty)==3
    assert empty.delta_choice_nll.isna().all()
    assert (empty.delta_total_nll==0).all()


def test_feedback_identity_and_seed_pairing_are_required():
    old,new,feedback,cut = banks()
    identical,_ = feedback_scores(old,old,feedback,cut)
    assert (identical.loc[identical.trials>0,'delta_choice_nll_mc_se']==0).all()
    new['seeds']=new['seeds'][::-1]
    with pytest.raises(AssertionError):feedback_scores(old,new,feedback,cut)
    with pytest.raises(ValueError):feedback_scores(old,old,np.array([0,.3,1,1]),cut)
