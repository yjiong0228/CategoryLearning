"""Two-coordinate isolation, fixed-reference eligibility and seed boundaries."""
from copy import deepcopy
import numpy as np
import pytest
from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
from src.Bayesian_state.workflows.analysis.model_1024_retention_capacity import (
    CAPACITY, DELTA, choose_profile, extract_seeds, profile, clean_template)


def anchor():
    return candidate({DELTA:0.8,CAPACITY:6,'precision_rate':.06,'execution_inertia':2.},'fixture')


def test_grid_preserves_anchor_and_all_nuisance_parameters():
    old=anchor();before=deepcopy(old);points=profile(old,[0.,.3,.6,.8,1.],[2,3,6])
    assert len(points)==15 and len({p['id'] for p in points})==15
    assert next(p for p in points if p['id']==old['id'])==old and old==before
    for p in points:
        assert {k:v for k,v in p['hyperparams'].items() if k not in (DELTA,CAPACITY)}=={
            k:v for k,v in old['hyperparams'].items() if k not in (DELTA,CAPACITY)}


@pytest.mark.parametrize('retention,capacity', [([0.,.8],[2,True]),([.8,.8],[6]),([np.nan,.8],[6]),([1.1,.8],[6]),([0.,1.],[2,3])])
def test_invalid_or_anchor_excluding_grids_are_rejected(retention,capacity):
    with pytest.raises(ValueError):profile(anchor(),retention,capacity)


def test_likelihood_tolerance_does_not_accumulate_against_latest_anchor():
    rows=[dict(candidate='anchor',training_nll=1.01,training_energy=.2),
          dict(candidate='tempting',training_nll=1.03,training_energy=.01)]
    assert choose_profile(rows,1.,.02)['candidate']=='anchor'
    assert choose_profile(rows,.98,.02) is None


def test_ranking_resolves_ties_without_future_or_parameter_order():
    rows=[dict(candidate='b',training_nll=1.,training_energy=.1),
          dict(candidate='a',training_nll=1.,training_energy=.1),
          dict(candidate='c',training_nll=.99,training_energy=.1)]
    assert choose_profile(rows,1.,.02)['candidate']=='c'
    assert choose_profile(rows[:2][::-1],1.,.02)['candidate']=='a'
    with pytest.raises(ValueError):choose_profile([dict(candidate='bad',training_nll=np.nan,training_energy=0)],1.,.02)


def test_prior_seed_roles_are_collected_but_hyperparameter_values_are_not():
    data={'seed':42,'cases':[{'seeds':[1,2],'branch_seeds':[3],'forbidden':[4]},
        {'candidate':{'hyperparams':{'retention':1}}}]}
    assert extract_seeds(data)=={1,2,3,4}


def test_old_freeze_does_not_mislabel_new_profile_jobs():
    case={'autonomous_jobs':[dict(subject=317,freeze_file='old.json',freeze_sha256='abc',
          training_generation=False,selection_sources={},generation_trials=16,model='MIX')]}
    result=clean_template(case)
    assert result==dict(subject=317,model='MIX')
    assert 'freeze_file' in case['autonomous_jobs'][0]


def test_timing_requires_gain_maintenance_and_prespecified_trial_window():
    from src.Bayesian_state.workflows.analysis.model_1024_retention_capacity_summary import timed_events
    metrics={'complete_center':np.array([68,69,36,35])};truth={'complete_center':36}
    base={'jump_and_maintenance':np.array([True,True,True,False]),
          'jump_maintenance_and_late':np.array([True,True,False,False])}
    result=timed_events(metrics,truth,base,32)
    np.testing.assert_array_equal(result['jump_maintenance_and_timing'],[True,False,True,False])
    np.testing.assert_array_equal(result['jump_maintenance_timing_and_late'],[True,False,False,False])


def test_paired_contrasts_keep_shared_simulation_uncertainty():
    from src.Bayesian_state.workflows.analysis.model_1024_retention_capacity_summary import linear_contrast,energy_contrast
    x=np.array([0.,1.,2.,3.]);value,se=linear_contrast([x,x],[1,-1])
    assert value==0. and se==0.
    features=x[:,None]
    value,se=energy_contrast([features,features,features,features],[1,-1,-1,1],np.array([1.]))
    assert value==0. and se==0.
    with pytest.raises(ValueError):linear_contrast([x,x[:2]],[1,-1])
