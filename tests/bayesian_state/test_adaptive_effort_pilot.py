"""Search coverage and numerical precision contracts of the follow-up pilot."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.optimization.model_0826 import (
    WORKSPACE_PROFILE_KEY, REACTIVE_PROFILE_KEY, build_model_0826_hyper_config,
    extract_model_0826_parameters,
)
from src.Bayesian_state.optimization.parameter_space import load_model_parameter_space
from src.Bayesian_state.workflows.analysis.pilot_model_0826_simplified_fit import (
    STATE_KEYS, point_id,
)
from src.Bayesian_state.workflows.analysis.pilot_model_0826_adaptive_effort import (
    budget_diagnostics, effort_action, initial_points, neighbor_values, propose_round,
    select_elites, verify_files, verify_reuse_context,
)


def configuration():
    cfg = yaml.safe_load(Path('configs/exp123/specific_models/model_0826_adaptive_effort_pilot.yaml').read_text())
    space = load_model_parameter_space('configs/exp123/specific_models/model_0826_cond1_parameter_space.yaml',
                                      expected_model_id='model_0826')
    budget = {'particle_count':2,'filter_seed_count':2,'seed_family':'test'}
    hyper = build_model_0826_hyper_config({'analysis_id':'test','subjects':[129],'hyper_base_seed':7},space,'PMH','base','out',
                                        {'coarse':budget,'fine':budget,'final_rescore':budget})
    return cfg, {k:v['values'] for k,v in hyper['stages']['fine']['hyperparam_space'].items()}, hyper['cd']['initial_points'][0]


def test_stratified_starts_cover_all_cells_and_do_not_collapse_to_workspace_variants():
    _,space,anchor = configuration()
    points = initial_points(space,anchor,36,8726)
    assert points == initial_points(space,anchor,36,8726)
    assert len({point_id(p) for p in points}) == 36
    assert {point_id({WORKSPACE_PROFILE_KEY:p[WORKSPACE_PROFILE_KEY]}) for p in points} == {
        point_id({WORKSPACE_PROFILE_KEY:v}) for v in space[WORKSPACE_PROFILE_KEY]}
    rows = [{'id':point_id(p),'hyperparams':p,'mean_nll':i/10} for i,p in enumerate(points)]
    elites = select_elites(rows,4)
    assert len(elites) == 4
    for i,a in enumerate(elites):
        for b in elites[:i]:
            assert sum(a['hyperparams'][k] != b['hyperparams'][k] for k in space if k != WORKSPACE_PROFILE_KEY) >= 2


def test_profile_neighbors_change_one_primitive_coordinate_not_flat_cartesian_index():
    _,space,_ = configuration()
    values = space[REACTIVE_PROFILE_KEY]
    anchor = values[len(values)//2]
    named = extract_model_0826_parameters({REACTIVE_PROFILE_KEY:anchor})
    neighbors = neighbor_values(REACTIVE_PROFILE_KEY,anchor,values)
    assert len(neighbors) >= 3
    changed = set()
    for value in neighbors:
        other = extract_model_0826_parameters({REACTIVE_PROFILE_KEY:value})
        keys = {k for k in ('E_C','delta_E') if abs(other[k]-named[k]) > 1e-9}
        assert len(keys) == 1
        changed.update(keys)
    assert changed == {'E_C','delta_E'}


def test_local_joint_proposals_respect_support_uniqueness_and_compute_cap():
    _,space,anchor = configuration()
    starts = initial_points(space,anchor,36,8726)
    rows = [{'id':point_id(p),'hyperparams':p,'mean_nll':i} for i,p in enumerate(starts)]
    seen = {r['id'] for r in rows}
    proposed = propose_round(select_elites(rows,4),space,seen,24,8727,.5,.25)
    assert 0 < len(proposed) <= 96
    assert len({r['id'] for r in proposed}) == len(proposed)
    assert not seen.intersection(r['id'] for r in proposed)
    assert {r['origin']['kind'] for r in proposed} == {'local','jump','joint'}
    anchors = {r['id']:r['hyperparams'] for r in rows}
    for row in proposed:
        assert all(row['hyperparams'][key] in values for key,values in space.items())
        if row['origin']['kind'] == 'joint':
            anchor_point = anchors[row['origin']['anchor']]
            assert sum(row['hyperparams'][k] != anchor_point[k] for k in space) == 2
    small = propose_round(rows[:1],space,seen,1,8727,0.,0.)
    assert len(small) <= 1


def test_effort_recommendation_separates_noise_from_search_and_never_certifies_optimality():
    assert effort_action(False,True,0.,0.,.005,.0001) == 'calibrate_numerical_budget'
    assert effort_action(True,False,.01,0.,.005,.0001) == 'expand_search_coverage'
    assert effort_action(True,False,0.,.001,.005,.0001) == 'expand_search_coverage'
    assert effort_action(True,True,.001,0.,.005,.0001) == 'provisional_stop_within_tested_scope'
    assert effort_action(True,True,None,0.,.005,.0001) == 'insufficient_search_diagnostics'
    assert effort_action(True,False,.01,0.,.005,.0001,output_ok=False) == 'expand_search_coverage'
    assert effort_action(True,True,.001,0.,.005,.0001,output_ok=False) == 'calibrate_output_precision'


def test_numerical_diagnostic_detects_seed_noise_without_requiring_unique_winner():
    cfg,_,_ = configuration()
    p = np.tile([[[.4,.6],[.3,.7]]],(4,1,1))
    a = {'probabilities':p,'observed':np.array([0,1]),'mask':np.array([False,True]),'seeds':np.arange(4)}
    for key in STATE_KEYS:
        a[key] = p.copy() if key in STATE_KEYS[:2] else p[:,:,0].copy()
    bank = [{'id':'a'},{'id':'b'}]
    arrays = {'a':a,'b':deepcopy(a)}
    stable = budget_diagnostics(bank,arrays,4,cfg['criteria'],99)
    assert stable['prediction_numerically_stable']
    assert stable['ranking_numerically_stable']
    assert stable['states_numerically_stable']
    assert stable['bootstrap_regret95'] == pytest.approx(0.)
    assert stable['disjoint_group_count'] == 1
    arrays['b']['probabilities'][:2] = [.99,.01]
    noisy = budget_diagnostics(bank,arrays,4,cfg['criteria'],99)
    assert not noisy['prediction_numerically_stable']
    assert noisy['worst_split_seed_difference']['choice_probability_rmse'] > .1
    # Shared seed noise can cancel in pairwise score differences while absolute
    # trial predictions still vary: search and output need separate decisions.
    arrays['a'] = deepcopy(arrays['b'])
    shared_noise = budget_diagnostics(bank,arrays,4,cfg['criteria'],99)
    assert shared_noise['ranking_numerically_stable']
    assert not shared_noise['prediction_numerically_stable']


def test_reuse_rejects_changed_dependencies(tmp_path):
    import hashlib
    f = tmp_path/'source.py'
    f.write_text('old')
    fingerprints = {str(f):hashlib.sha256(f.read_bytes()).hexdigest()}
    verify_files(fingerprints)
    f.write_text('changed')
    with pytest.raises(ValueError,match='dependencies changed'):
        verify_files(fingerprints)


def test_reuse_rejects_library_changes_even_when_files_are_identical():
    previous = {'input_sha256':{},'versions':{'numpy':'original'}}
    verify_reuse_context(previous, {'numpy':'original'})
    with pytest.raises(ValueError,match='library versions changed'):
        verify_reuse_context(previous, {'numpy':'changed'})


def test_extension_reuses_only_verified_seed_prefix_and_evaluates_missing_seeds(tmp_path, monkeypatch):
    import json
    from src.Bayesian_state.workflows.analysis import pilot_model_0826_adaptive_effort as module
    old = tmp_path/'old'
    target = tmp_path/'new'
    old.mkdir()
    target.mkdir()
    seeds = module.validation_seeds(77,129,'confirmation',4)
    p = np.tile([[[.4,.6],[.3,.7]]],(2,1,1))
    a = {'probabilities':p,'observed':np.array([0,1]),'mask':np.array([False,True]),'seeds':seeds[:2]}
    for key in STATE_KEYS:
        a[key] = p.copy() if key in STATE_KEYS[:2] else p[:,:,0].copy()
    bank = [{'id':'test','hyperparams':{'x':1},'sources':['test']}]
    np.savez_compressed(old/'test.npz',**a)
    before = (old/'test.npz').read_bytes()
    (old/'scores.json').write_text(json.dumps({'budget':{'particle_count':8},'seeds':seeds[:2],'rows':bank}))
    seen = []
    def evaluate(ctx,point,particles,seed):
        seen.append(seed)
        return {**{k:a[k][0].copy() for k in ('probabilities',*STATE_KEYS)},
                'observed':a['observed'],'mask':a['mask'],'seconds':0.}
    monkeypatch.setattr(module,'score_one',evaluate)
    result = module.extend_confirmation({'subject':129},bank,{'particle_count':8,'filter_seed_count':4},
                                        old,77,1,target)
    assert seen == seeds[2:]
    assert result['reused_seed_count_per_candidate'] == 2
    with np.load(target/'confirmation/test.npz') as z:
        np.testing.assert_array_equal(z['seeds'],seeds)
        np.testing.assert_array_equal(z['probabilities'][:2],p)
    assert (old/'test.npz').read_bytes() == before
