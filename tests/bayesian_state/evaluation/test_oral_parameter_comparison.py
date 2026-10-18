"""Parameter contrasts retain report identities and paired simulation errors."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_oral_parameter_comparison import paired_scores


def scores():
    trials = pd.DataFrame(dict(iSub=[208,208],arm=['SEARCH']*2,iSession=[1,1],iTrial=[1,2],
        state_trial=[0,1],choice=[1,2],text=['neck','head'],tokens=['[]','[]'],query_id=['q1','q2'],
        scope=['complete']*2,catalogue_version=['v1']*2,training_trials=[1,1],phase=['training','test']))
    for domain in ('structural','functional'):
        for label,values in [('support',[.3,.7]),('conflict',[.4,0]),('undetermined',[.3,.3])]:
            trials[domain+'_'+label]=values
    row=dict(subject=208,condition=2,arm='SEARCH',window='whole',domain='structural',subset='complete',
             window_trials=2,reports=2,full_reports=2,encoded_fragments=0,filter_seeds=3,
             support=.5,conflict=.2,undetermined=.3)
    estimates={('whole','structural','complete'):dict(seeds=np.array([3,5,7]),point=np.array([.5,.2,.3]),
        values=np.array([[.4,.3,.3],[.5,.2,.3],[.6,.1,.3]]))}
    return trials,[row],estimates


def test_same_parameter_scores_have_exactly_zero_differences_and_errors():
    original=scores();summary,trials=paired_scores(original,deepcopy(original))
    assert np.all(summary.filter(regex='difference').to_numpy()==0)
    assert np.all(trials.filter(regex='difference').to_numpy()==0)
    assert len(trials)==2 and summary.reports.iloc[0]==2


def test_paired_common_filter_variation_cancels_in_parameter_difference():
    original=scores();revised=deepcopy(original)
    delta=np.array([.1,-.1,0])
    for k,name in enumerate(('support','conflict','undetermined')):
        revised[1][0][name]+=delta[k]
        for domain in ('structural','functional'):revised[0][domain+'_'+name]+=delta[k]
    estimate=next(iter(revised[2].values()));estimate['point']+=delta;estimate['values']+=delta
    summary,_=paired_scores(original,revised)
    assert summary.support_difference.iloc[0]==pytest.approx(.1)
    assert summary.support_difference_mcse.iloc[0]==pytest.approx(0,abs=1e-15)
    assert summary.conflict_difference.iloc[0]==pytest.approx(-.1)


@pytest.mark.parametrize('fault',['trial','text','query','scope','boundary','reports','seed_order','duplicate_seed','missing_estimate','short_estimate','point'])
def test_misaligned_reports_or_random_units_are_rejected(fault):
    original=scores();revised=deepcopy(original);estimate=next(iter(revised[2].values()))
    if fault=='trial':revised[0].loc[0,'iTrial']=7
    elif fault=='text':revised[0].loc[0,'text']='changed'
    elif fault=='query':revised[0].loc[0,'query_id']='changed'
    elif fault=='scope':revised[0].loc[0,'scope']='parsed_fragment'
    elif fault=='boundary':revised[0]['training_trials']=2
    elif fault=='reports':revised[1][0]['reports']=1
    elif fault=='seed_order':estimate['seeds']=estimate['seeds'][::-1]
    elif fault=='duplicate_seed':
        estimate['seeds'][1]=3;next(iter(original[2].values()))['seeds'][1]=3
    elif fault=='missing_estimate':revised[2].clear()
    elif fault=='short_estimate':estimate['values']=estimate['values'][:2]
    else:estimate['point'][0]+=.1
    with pytest.raises((ValueError,AssertionError)):paired_scores(original,revised)
