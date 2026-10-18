"""Parameter contrasts retain report identities and paired simulation errors."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_oral_parameter_comparison import paired_scores
from src.Bayesian_state.workflows.analysis import model_0923_oral_parameter_comparison as comparison


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


def test_independent_errors_add_variances_without_cancelling_equal_fluctuations():
    original=scores();revised=deepcopy(original)
    estimate=next(iter(revised[2].values()));estimate['seeds']+=100
    summary,_=paired_scores(original,revised,seed_relation='independent')
    assert summary.support_difference.iloc[0]==0
    # Each bank has jackknife variance (3-1)/3 * (.1**2 + .1**2).
    expected=np.sqrt(2*(2/3)*.02)
    assert summary.support_difference_mcse.iloc[0]==pytest.approx(expected)
    assert summary.conflict_difference_mcse.iloc[0]==pytest.approx(expected)
    assert summary.undetermined_difference_mcse.iloc[0]==pytest.approx(0)
    estimate['seeds']=estimate['seeds'][::-1]
    estimate['values']=estimate['values'][::-1]
    reordered,_=paired_scores(original,revised,seed_relation='independent')
    pd.testing.assert_frame_equal(summary,reordered)


@pytest.mark.parametrize('seeds',[[3,5,7],[3,105,107],[103,103,107]])
def test_independent_mode_rejects_overlap_or_duplicates(seeds):
    original=scores();revised=deepcopy(original)
    next(iter(revised[2].values()))['seeds']=np.array(seeds)
    with pytest.raises(ValueError):paired_scores(original,revised,seed_relation='independent')


def test_declaring_independence_does_not_relax_report_boundaries_or_budget():
    original=scores();revised=deepcopy(original)
    next(iter(revised[2].values()))['seeds']+=100
    revised[0].loc[0,'training_trials']=2
    with pytest.raises(AssertionError):paired_scores(original,revised,seed_relation='independent')
    revised[0].loc[0,'training_trials']=1
    revised[1][0]['filter_seeds']=4
    with pytest.raises(AssertionError):paired_scores(original,revised,seed_relation='independent')


def test_unknown_seed_relation_is_rejected():
    with pytest.raises(ValueError,match='Seed relation'):
        paired_scores(scores(),scores(),seed_relation='automatic')


def intervention_identities():
    before=dict(scientific={'particles':512,'trials':2},arm_settings={'precision':'fast'},
        mechanism={'kind':'E','retain_learned_precision':False},archived_pf_sha256='same',
        task_sha256={'training':'same'},oral_manifest_sha256='same',model_sources={'original.py':'original'},
        candidate={'id':'point','hyperparams':{'gamma':0}},structural_intervention=None,
        parameter_candidate_policy='original_evaluation_protocol',selected_parameter_id=None)
    after=deepcopy(before)
    after['mechanism']['retain_learned_precision']=True
    after['structural_intervention']='retained_rule_precision'
    after['parameter_candidate_policy']='declared_point_no_new_selection'
    after['selected_parameter_id']='point';after['model_sources']['retention.py']='retention'
    return before,after


def test_precision_contrast_accepts_only_its_declared_mechanism_change():
    before,after=intervention_identities()
    comparison.validate_comparison(before,after,'precision_retention')
    with pytest.raises(ValueError):comparison.validate_comparison(before,after,'parameters')


@pytest.mark.parametrize('fault',['task','budget','oral','parameter','parameter_id','other_mechanism',
    'disabled','reversed','selection','point_binding','source'])
def test_precision_contrast_rejects_confounding_changes(fault):
    before,after=intervention_identities()
    if fault=='task':after['task_sha256']['training']='changed'
    elif fault=='budget':after['scientific']['particles']=256
    elif fault=='oral':after['oral_manifest_sha256']='changed'
    elif fault=='parameter':after['candidate']['hyperparams']['gamma']=.8
    elif fault=='parameter_id':after['candidate']['id']='different'
    elif fault=='other_mechanism':after['mechanism']['kind']='R1'
    elif fault=='disabled':after['mechanism']['retain_learned_precision']=False
    elif fault=='reversed':before,after=after,before
    elif fault=='selection':after['parameter_candidate_policy']='oral_selected'
    elif fault=='point_binding':after['selected_parameter_id']='different'
    elif fault=='source':after['model_sources']['original.py']='changed'
    with pytest.raises(ValueError):comparison.validate_comparison(before,after,'precision_retention')


def test_precision_contrast_exports_mechanism_labels_without_parameter_selection(tmp_path,monkeypatch):
    before,after=intervention_identities()
    def load(directory,subject,arm):
        return scores(),before if directory.name=='original' else after,{}
    monkeypatch.setattr(comparison,'load_group',load)
    output=tmp_path/'output'
    comparison.run(tmp_path/'original',tmp_path/'revised',output,208,'SEARCH',contrast_kind='precision_retention')
    manifest=comparison.alignment.read(output/'manifest.json')
    assert manifest['contrast_kind']=='precision_retention'
    assert 'RESET versus RETAIN' in manifest['data_use']
    assert '同一参数下保留已学精度' in (output/'index.html').read_text()
    summary=pd.read_csv(output/'scores.csv')
    assert np.all(summary.filter(regex='difference').to_numpy()==0)


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
