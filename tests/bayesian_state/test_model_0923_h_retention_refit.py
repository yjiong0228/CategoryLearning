"""Retention-policy restrictions, preserved anchors and equal search budgets."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.Bayesian_state.workflows.analysis.model_0923_h_retention_refit import (
    DECAY, model_space, project, shortlist, starting_banks, validate_settings,
)


@pytest.fixture
def settings():
    return yaml.safe_load(Path('configs/shared/model0923_h_retention_refit.yaml').read_text())


def test_policies_have_identical_common_support_and_distinct_retention_constraints(settings):
    spaces={m:model_space(settings,m)[0] for m in settings['models']}
    common=spaces['SHARED']
    assert len(common)==13 and DECAY not in common
    assert spaces['CUMULATIVE'][DECAY]==[1.]
    assert spaces['FREE'][DECAY]==settings['evidence_decay_values']
    for model in ('FREE','CUMULATIVE'):
        assert {k:v for k,v in spaces[model].items() if k!=DECAY}==common


def test_project_changes_only_the_retention_coordinate(settings):
    _,anchor=model_space(settings,'FREE');before=deepcopy(anchor)
    for model in settings['models']:
        point=project(anchor,model,settings)
        assert {k:v for k,v in point.items() if k!=DECAY}=={k:v for k,v in anchor.items() if k!=DECAY}
    assert anchor==before
    wrong=dict(anchor,unexpected=1.)
    with pytest.raises(ValueError):project(wrong,'FREE',settings)


def test_model_checks_reject_confounding_structural_changes(settings):
    validate_settings(settings)
    settings['models']['CUMULATIVE']['capacity']='posterior'
    with pytest.raises(ValueError):validate_settings(settings)


def test_initial_budgets_keep_old_nominees_only_for_development_subjects(settings):
    if not Path(settings['warm_start_fit']).exists():pytest.skip('Archived candidate bank unavailable')
    banks,warm=starting_banks(settings,[107,109],False)
    assert len(banks)==6 and all(len(rows)==48 for rows in banks.values())
    for model in settings['models']:
        assert len(warm[107,model])==2 and not warm[109,model]
        assert {r['id'] for r in warm[107,model]}<={r['id'] for r in banks[107,model]}
    # New participants receive exactly the same common-coordinate starts.
    common=lambda model:[{k:v for k,v in r['hyperparams'].items() if k!=DECAY} for r in banks[109,model]]
    assert common('FREE')==common('CUMULATIVE')==common('SHARED')


def test_shortlist_retains_the_required_original_points():
    rows=[dict(id=str(i),mean_nll=i) for i in range(10)]
    chosen=shortlist(rows,[rows[-1]],3)
    assert [r['id'] for r in chosen]==['9','0','1']
    with pytest.raises(ValueError):shortlist(rows,[],11)


def test_free_support_contains_both_constrained_policies(settings):
    free,_=model_space(settings,'FREE')
    assert 1. in free[DECAY]
    assert set(free['engine.modules.memory_mod.kwargs.gamma'])<=set(free[DECAY])


def test_prefix_particle_overrides_preserve_default_and_smoke_budgets():
    from src.Bayesian_state.workflows.analysis.model_0923_h_prefix_forecast import configured_particles
    settings=dict(particles=1024,particle_overrides={305:4096})
    assert configured_particles(settings,207)==1024
    assert configured_particles(settings,305)==4096
    assert configured_particles(settings,305,True)==4
    with pytest.raises(ValueError):configured_particles(dict(particles=True),207)


def test_prefix_metadata_preserves_new_subject_roles_and_explicit_development_status(settings):
    from src.Bayesian_state.workflows.analysis.model_0923_h_prefix_forecast import data_use_roles
    frozen=dict(status='frozen_before_test',settings=settings)
    overall,roles=data_use_roles(frozen,[107,109],dict(data_use_status='mixed'))
    assert overall=='mixed' and roles=={107:'development',109:'new_validation'}
    frozen['status']='frozen_for_development_evaluation'
    assert data_use_roles(frozen,[107,109],{})==('development',{107:'development',109:'development'})
    assert data_use_roles(dict(status='frozen_before_test',settings={}),[1],{})==('development',{1:'development'})


def test_evaluation_plan_pairs_seeds_and_shares_one_queue(settings,tmp_path):
    import json
    from src.Bayesian_state.workflows.analysis.model_0923_h_retention_evaluation import plan
    from src.Bayesian_state.workflows.analysis.model_0923_h_validation import read
    fit=Path('results/model_0923/h_retention_refit_smoke_20261016_v1/nominees.json')
    if not fit.exists():pytest.skip('Archived smoke candidates unavailable')
    frozen=read(fit);frozen['settings']=settings
    groups,chunks=plan(frozen,tmp_path/'unused')
    assert len(chunks)==3*(8+512//16)
    assert len({j['name'] for j in chunks})==len(chunks)
    for role,items in groups.items():
        assert len(items)==3 and all(job['phase']==('autonomous' if role=='autonomous' else 'evaluate') for job,_ in items)
        assert all(job['seeds']==items[0][0]['seeds'] for job,_ in items)
    assert not (tmp_path/'unused').exists()
    # The actual handoff reloads JSON, whose subject-ID dictionary keys are strings.
    frozen=json.loads(json.dumps(frozen))
    for nominee in frozen['nominees']:nominee['subject']=305
    groups,_=plan(frozen,tmp_path/'subject_override')
    assert all(job['particles']==4096 for items in groups.values() for job,_ in items)


def test_aggregate_keeps_new_participants_separate_and_weights_people_equally(settings):
    import numpy as np
    import pandas as pd
    from src.Bayesian_state.evaluation.model_0923_h_retention_refit import aggregate_contrasts,cohort_for
    rows=[];differences=[]
    for subject,value,trials,se in [(107,.2,64,.03),(108,.6,192,.04),(109,.9,160,.07)]:
        tag=dict(cohort=cohort_for(subject,settings),phase='test',model='CUMULATIVE')
        rows.append(dict(tag,trials=trials,choice_nll=value,feature_energy=value))
        differences.append(dict(tag,delta_choice_nll=value,delta_feature_energy=value,
            choice_nll_paired_mc_se=se,feature_energy_paired_mc_se=se))
    result=aggregate_contrasts(pd.DataFrame(rows),pd.DataFrame(differences)).set_index('cohort')
    assert result.loc['development','choice_nll']==pytest.approx(.4)
    assert result.loc['development','delta_choice_nll_mc_se']==pytest.approx(np.hypot(.03,.04)/2)
    assert result.loc['new_validation','choice_nll']==pytest.approx(.9)
    assert result.loc['new_validation','subjects']==1
    with pytest.raises(ValueError):cohort_for(999,settings)


def test_prefix_cohort_summary_rejects_missing_or_duplicated_cases(settings):
    import pandas as pd
    from src.Bayesian_state.evaluation.model_0923_h_retention_refit import aggregate_prefix_cohorts
    rows=[];contrasts=[]
    for subject,value in [(107,.2),(108,.6),(109,.9)]:
        tag=dict(subject=subject,model='CUMULATIVE',start='filtered_prefix')
        rows.append(dict(tag,energy=value,crps=value))
        contrasts.append(dict(tag,delta_energy=-value,delta_crps=-value,
                              delta_energy_mc_se=.03,delta_crps_mc_se=.03))
    scores,diffs=pd.DataFrame(rows),pd.DataFrame(contrasts)
    result=aggregate_prefix_cohorts(scores,diffs,settings).set_index('cohort')
    assert result.loc['development','energy']==pytest.approx(.4)
    assert result.loc['new_validation','energy']==pytest.approx(.9)
    assert result.loc['development','improved_energy']==2
    with pytest.raises(ValueError):aggregate_prefix_cohorts(scores,diffs.iloc[:2],settings)
    with pytest.raises(ValueError):aggregate_prefix_cohorts(pd.concat([scores,scores]),diffs,settings)


def test_training_diagnostics_read_the_completed_smoke_artifact_format(tmp_path):
    import pandas as pd
    from src.Bayesian_state.evaluation.model_0923_h_retention_refit import load,training_diagnostics
    fit=Path('results/model_0923/h_retention_refit_smoke_20261016_v1')
    if not (fit/'nominees.json').exists():pytest.skip('Archived smoke candidates unavailable')
    training_diagnostics(fit,load(fit/'nominees.json'),tmp_path)
    precision=pd.read_csv(tmp_path/'training_precision_change.csv')
    assert len(precision)==6 and set(precision.model)=={'FREE','CUMULATIVE','SHARED'}
    assert precision[['low_nll','high_nll','independent_mc_se']].notna().all().all()
    differences=pd.read_csv(tmp_path/'training_finalist_differences.csv')
    assert len(differences)==6 and differences.selected.sum()==3
    budgets=pd.read_csv(tmp_path/'training_budgets.csv')
    assert set(budgets.stage)=={'screen','refine1','selection','confirmation'}


def test_pooled_reference_uses_training_scores_and_preserves_free_support(settings):
    from src.Bayesian_state.workflows.analysis.model_0923_h_retention_evaluation import pooled_training_reference
    _,anchor=model_space(settings,'FREE')
    nominees=[dict(subject=109,model=model,candidate=dict(id=model,hyperparams=project(anchor,model,settings),
        mean_nll=score)) for model,score in [('FREE',1.1),('CUMULATIVE',1.),('SHARED',.9)]]
    frozen=dict(settings=settings,nominees=nominees);before=deepcopy(frozen)
    chosen=pooled_training_reference(frozen)[0]
    assert chosen['source_model']=='SHARED'
    point=chosen['equivalent_free_hyperparams']
    assert point[DECAY]==point['engine.modules.memory_mod.kwargs.gamma']
    assert frozen==before
    for row in nominees:row['candidate']['mean_nll']=1.
    assert pooled_training_reference(frozen)[0]['source_model']=='FREE'


def test_reference_feasibility_distinguishes_constraints_from_search_results(settings):
    from src.Bayesian_state.evaluation.model_0923_h_retention_refit import retention_constraint_accepts
    _,point=model_space(settings,'FREE')
    point[DECAY]=1.;point['engine.modules.memory_mod.kwargs.gamma']=.3
    assert retention_constraint_accepts(point,'FREE')
    assert retention_constraint_accepts(point,'CUMULATIVE')
    assert not retention_constraint_accepts(point,'SHARED')
    point[DECAY]=.3
    assert retention_constraint_accepts(point,'SHARED')
    assert not retention_constraint_accepts(point,'CUMULATIVE')
    with pytest.raises(ValueError):retention_constraint_accepts(point,'unknown')


@pytest.mark.parametrize('condition',[1,2,3])
@pytest.mark.parametrize('gamma',[0.,.6,1.])
def test_shared_and_explicit_equal_decay_preserve_predictions_and_internal_states(condition,gamma,tmp_path,monkeypatch):
    """Justify prediction reuse when embedding SHARED points in FREE support."""
    import numpy as np
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import ModelContext,StateModel
    from src.Bayesian_state.model.modules.base_module import ModuleRole
    from src.Bayesian_state.model.modules.readout.choice import resolve_choice_readout_config,resolve_output_noise_config
    monkeypatch.setattr(ContinuousSimilarity,'RUNTIME_CACHE_DIR',tmp_path)
    config=yaml.safe_load(Path(f'configs/exp123/model_struct/model_0923_cond{condition}_full_v3.yaml').read_text())
    config['modules']['perception_mod']['kwargs']=dict(features=4,mean=[0.]*4,std=[0.]*4,module_seed=2)
    config['modules']['memory_mod']['kwargs']['gamma']=gamma
    transition=config['modules']['hypo_transitions_mod']['kwargs']
    transition.update(max_capacity=2,exit_rate=.1,relation_samples=128,module_seed=3)
    transition['search_controller']['baseline_probability']=.8
    config['provenance']['hypothesis_similarity']['sample_count']=128
    models=[]
    for decay in ['memory',gamma]:
        current=deepcopy(config)
        current['modules']['hypo_transitions_mod']['kwargs']['intervention']=dict(
            kind='E',decay=decay,capacity='evidence_mean',search='accumulate')
        models.append(StateModel(current,context=ModelContext(condition=condition)))
    stimuli=np.random.default_rng(91).uniform(.1,.9,(12,4))
    choices=np.tile([1,2,1,2],3)
    feedback=np.tile([0.,.5 if condition==3 else 0.,1.,0.],3)
    modules=[m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION) for m in models]
    for trial,stimulus in enumerate(stimuli):
        predictions=[]
        for model in models:
            model.begin_trial(stimulus)
            prediction=model.predict_choice(choices=choices[:trial],feedback=feedback[:trial],
                choice_readout_config=resolve_choice_readout_config(None,model.engine_config),
                output_noise_config=resolve_output_noise_config(None,model.engine_config),
                rng=np.random.default_rng(12),sticky_state={},post_error_lapse_state=0.)
            predictions.append(prediction.observed_probabilities)
            model.complete_trial(int(choices[trial]),float(feedback[trial]))
        np.testing.assert_array_equal(*predictions)
        for field in ['prior','posterior','beta']:
            np.testing.assert_array_equal(getattr(models[0].engine,field),getattr(models[1].engine,field))
        for field in ['active','retention_evidence','retention_exposure']:
            np.testing.assert_array_equal(getattr(modules[0],field),getattr(modules[1],field))
        assert modules[0].trial_rng.bit_generator.state==modules[1].trial_rng.bit_generator.state
    assert any(event['search_event'] for event in modules[0].transition_log)
