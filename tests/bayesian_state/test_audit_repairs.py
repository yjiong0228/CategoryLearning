"""Regression cases for the confirmed 2026-09-17 audit failures."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
from src.Bayesian_state.optimization.search.cd_v2 import search_context_fingerprint
from src.Bayesian_state.simulation.data import SubjectTrialDataLoader, _coerce_trial_arrays, prepare_trial_sequence
from src.Bayesian_state.utils.provenance import DependencySnapshot
from src.Bayesian_state.workflows.recovery.run import prepare_output


def _frame():
    return pd.DataFrame(dict(iSub=[401]*4, condition=[1]*4, iSession=[1,1,2,2],
        iBlock=[1]*4, iTrial=[1,2,1,2], feature1=[.1,.2,.3,.4],
        choice=[1,2,1,2], feedback=[1,0,1,0]))


def _loader(frame, **data):
    loader = SubjectTrialDataLoader({'data': {'feature_columns':['feature1'], **data}})
    loader.learning_data = frame
    return loader


@pytest.mark.parametrize('condition', [1,2,3])
@pytest.mark.parametrize('bad', [1.25, np.nan, np.inf, 0, -1, 5])
def test_pf_rejects_raw_invalid_choices(condition, bad):
    with pytest.raises(ValueError, match='choices.*finite integers'):
        run_state_model_particle_filter(engine_config={}, subject_id=401,
            stimulus=[[.1]], choices=[bad], feedback=[1], condition=condition,
            particle_count=2, choice_readout_power=1)


@pytest.mark.parametrize('bad', [1.25, np.nan, np.inf, 0, 3])
def test_loader_checks_choices_without_presskey_before_truncation(bad):
    frame = _frame().astype({'choice':float})
    frame.loc[3,'choice'] = bad
    with pytest.raises(ValueError, match='choices.*finite integers'):
        _loader(frame)._get_subject_frame(401, .25)
    with pytest.raises(ValueError, match='choices.*finite integers'):
        _loader(frame)._extract_arrays(frame, 1)


def test_legacy_tuple_cannot_truncate_fractional_choices_or_unequal_trials():
    with pytest.raises(ValueError, match='finite integers'):
        _coerce_trial_arrays(([[.1]], [1.25], [1]))
    with pytest.raises(ValueError, match='finite integers'):
        prepare_trial_sequence(np.array([[.1]]), np.array([1.25]), np.array([1]))
    with pytest.raises(ValueError, match='equal trial counts'):
        prepare_trial_sequence(np.array([[.1]]), np.array([1,2]), np.array([1,0]))


def test_order_checks_full_subject_and_preserves_continuous_sessions():
    frame = _frame()
    arrays = _loader(frame)._extract_arrays(_loader(frame)._get_subject_frame(401, 1), None)
    np.testing.assert_array_equal(arrays.stimulus[:,0], frame.feature1)
    for bad in (frame.iloc[::-1], frame.iloc[[0,1,3,2]]):
        with pytest.raises(ValueError, match='chronological'):
            _loader(bad)._get_subject_frame(401, .25)
    duplicate = pd.concat([frame, frame.iloc[-1:]])
    with pytest.raises(ValueError, match='Duplicate trial keys'):
        _loader(duplicate)._get_subject_frame(401, .25)
    mixed = frame.copy()
    mixed.loc[3,'condition'] = 2
    with pytest.raises(ValueError, match='consistent condition'):
        _loader(mixed)._get_subject_frame(401, .25)
    pd.testing.assert_frame_equal(frame, _frame())


def test_declared_task_order_keys_are_required_and_checked():
    frame = _frame().rename(columns={'iSession':'session','iTrial':'trial'})
    keys = ['session','iBlock','trial']
    _loader(frame, trial_order_columns=keys)._get_subject_frame(401, 1)
    with pytest.raises(ValueError, match='chronological'):
        _loader(frame.iloc[::-1], trial_order_columns=keys)._get_subject_frame(401, .25)
    with pytest.raises(ValueError, match='Missing trial order columns'):
        _loader(frame, trial_order_columns=['missing'])._get_subject_frame(401, 1)


def _fingerprint_case(tmp_path):
    engine = tmp_path/'engine.yaml'
    engine.write_text(yaml.safe_dump({'modules':{}}))
    for name in ('learn.csv','perception.csv','uniform.csv','features.csv'):
        (tmp_path/name).write_text('value\n1\n')
    base = {'engine_config_path':'engine.yaml', 'dataset':{
        'processed_dir':str(tmp_path), 'learning_data':'learn.csv',
        'perception_summary':'perception.csv','perception_summary_72':'uniform.csv',
        'feature_order_data':'features.csv'}}
    search = {'base_sim_config_path':str(tmp_path/'base.yaml')}
    (tmp_path/'base.yaml').write_text(yaml.safe_dump(base))
    def fingerprint():
        return search_context_fingerprint(search, base, [401], 'coarse', config_dir=tmp_path)
    return search, base, fingerprint


@pytest.mark.parametrize('name', ['engine.yaml','learn.csv','perception.csv','uniform.csv','features.csv'])
def test_resume_hash_changes_when_input_content_changes_in_place(tmp_path, name):
    _, _, fingerprint = _fingerprint_case(tmp_path)
    before = fingerprint()
    assert fingerprint() == before
    path = tmp_path/name
    path.write_text(path.read_text()+'\n# changed in place\n')
    assert fingerprint() != before


def test_resume_includes_stage_and_subject_resolved_inputs(tmp_path):
    search, base, fingerprint = _fingerprint_case(tmp_path)
    special = tmp_path/'special.csv'
    special.write_text('value\n2\n')
    search['stages'] = {'fine':{'simulation_overrides':{
        'subject_overrides':{401:{'dataset':{'learning_data':'special.csv'}}}}}}
    before = fingerprint()
    special.write_text('value\n3\n')
    assert fingerprint() != before


def test_resume_includes_complete_core_and_rule_resources():
    files = DependencySnapshot().payload()['files']
    for relative in ('model/modules/memory.py','model/modules/beta.py',
                     'model/modules/hypothesis_transition/prior_assignment.py'):
        assert any(path.endswith('/'+relative) and digest for path,digest in files.items())
    assert any('/resources/similarity/' in path and path.endswith('.npy') and digest
               for path,digest in files.items())


def test_recovery_rejects_old_fingerprint_without_mutating_manifest(tmp_path):
    output = tmp_path/'run'
    prepare_output(output, resume=False, analysis_id='audit', config_fingerprint='abc')
    prepare_output(output, resume=True, analysis_id='audit', config_fingerprint='abc')
    manifest = output/'manifest.json'
    old = json.loads(manifest.read_text())
    old.pop('fingerprint_schema_version')
    manifest.write_text(json.dumps(old))
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match='fingerprint schema is obsolete'):
        prepare_output(output, resume=True, analysis_id='audit', config_fingerprint='abc')
    assert manifest.read_bytes() == before


def test_effective_similarity_cache_content_is_fingerprinted(tmp_path, monkeypatch):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model.assembly import build_partition
    monkeypatch.setattr(ContinuousSimilarity, '_memory_cache', {})
    engine = {'partition':{'class':'src.Bayesian_state.hypothesis_space.observation_model.continuous_partition.ContinuousPartition',
        'kwargs':{'n_dims':4,'n_cats':2,'similarity_cache_dir':str(tmp_path)}},
        'likelihood':{'distance_mode':'boundary'}}
    first = DependencySnapshot()
    first._similarity(engine)
    partition = build_partition(engine, 1)
    similarity = partition.similarity
    key = similarity._cache_key(distance_mode='boundary', n_samples=100000,
        random_state=0, sample_distribution='uniform')
    matrix = similarity.get_matrix(distance_mode='boundary').copy()
    matrix *= .9
    np.fill_diagonal(matrix, 1)
    np.save(similarity._cache_path(key), matrix)
    with pytest.raises(ValueError, match='changed while cached'):
        DependencySnapshot()._similarity(engine)
    monkeypatch.setattr(ContinuousSimilarity, '_memory_cache', {})
    second = DependencySnapshot()
    second._similarity(engine)
    assert first.similarities != second.similarities


def test_core_edits_change_provenance_without_changing_config(tmp_path, monkeypatch):
    from src.Bayesian_state.utils import provenance
    core = tmp_path/'core'
    core.mkdir()
    source = core/'memory.py'
    source.write_text('gamma = 0.8\n')
    monkeypatch.setattr(provenance, 'BAYESIAN_STATE_DIR', core)
    before = DependencySnapshot().payload()
    source.write_text('gamma = 0.9\n')
    assert DependencySnapshot().payload() != before


@pytest.mark.parametrize('coordinate', [
    'engine.partition.kwargs.n_cats', 'engine.likelihood.distance_mode',
    'engine.modules.perception_mod.kwargs', 'simulation.dataset.learning_data',
])
def test_resource_changing_coordinates_cannot_bypass_resume_dependencies(tmp_path, coordinate):
    search, _, fingerprint = _fingerprint_case(tmp_path)
    search['hyperparam_space'] = {coordinate: {'values': [1]}}
    with pytest.raises(ValueError, match='fixed input/geometry dependencies'):
        fingerprint()
