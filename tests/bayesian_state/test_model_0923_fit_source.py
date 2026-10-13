"""Saved-fit adapter checks using synthetic source snapshots and PF arrays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.Bayesian_state.evaluation import model_0923_fit_source as source
from src.Bayesian_state.evaluation import model_0923_postfit as postfit
from src.Bayesian_state.optimization.adaptive_runtime import cache_path, phase_seeds
from src.Bayesian_state.optimization.search.adaptive_proposals import point_id
from src.Bayesian_state.simulation.data import TrialArrays


POINT = {
    'engine.modules.hypo_transitions_mod.kwargs.search_controller.baseline_probability': .05,
    'engine.modules.hypo_transitions_mod.kwargs.search_controller.error_gain': 1.,
    'engine.modules.hypo_transitions_mod.kwargs.search_controller.failure_decay': .6,
    'engine.modules.hypo_transitions_mod.kwargs.search_controller.global_search': .2,
    'engine.modules.hypo_transitions_mod.kwargs.search_controller.global_search_failure_gain': .5,
    'engine.modules.hypo_transitions_mod.kwargs.max_capacity': 6,
    'engine.modules.hypo_transitions_mod.kwargs.exit_rate': .1,
    'engine.modules.hypo_transitions_mod.kwargs.local_structure_weight': .5,
    'engine.modules.hypo_transitions_mod.kwargs.transfer_scale': 1.,
    'engine.modules.hypo_transitions_mod.kwargs.execution_inertia': 2.,
    'engine.modules.memory_mod.kwargs.gamma': .7,
    'engine.modules.beta_mod.kwargs.beta_init': 5.,
    'engine.modules.beta_mod.kwargs.learning_rate': .1,
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initial_fit(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / 'repository'
    monkeypatch.setattr(source, 'ROOT_DIR', root)
    fit = root / 'results' / 'initial'
    name = 'src/Bayesian_state/model/model_0923.py'
    current = root / name
    current.parent.mkdir(parents=True)
    current.write_text('scientific-core\n')
    archived = fit / 'source' / name
    archived.parent.mkdir(parents=True)
    archived.write_bytes(current.read_bytes())
    runtime = fit / 'runtime'
    runtime.mkdir()
    (runtime / 'simulation.yaml').write_text(yaml.safe_dump({'engine_config_path': '/old/initial/source/configs/model.yaml'}))
    hyper = {'hyper_base_seed': 41,
             'hyperparam_space': {key: {'values': [value]} for key, value in POINT.items()},
             'final_rescore': {'seed_family': 'final', 'simulation_overrides': {'engine_config': {'inference': {'particle_count': 128}}}}}
    (runtime / 'hyper.yaml').write_text(yaml.safe_dump(hyper))
    write_json(fit / 'run_manifest.json', {'snapshot_files_sha256': {name: digest(archived)},
                                         'runtime_files_sha256': {f'runtime/{p.name}': digest(p) for p in runtime.iterdir()},
                                         'subjects': [{'subject': 307}], 'working_directory': '/old/initial/source'})
    write_json(fit / 'run_status.json', {'status': 'completed'})
    directory = fit / 'hyper_cd' / 'subject_307'
    write_json(directory / 'best_hyperparams.json', {'final_rescore_best': {'hyperparams': POINT, 'aggregated_error': .4}})
    write_json(directory / 'restart_summary.json', {})
    row = {'hyperparams': POINT, 'aggregated_error': .4,
           'subject_metrics': {'307': {'simulation': {'sample_errors': [.5, .6]}}}}
    (directory / 'final_rescore.jsonl').write_text(json.dumps(row) + '\n')
    return fit


def adaptive_fit(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / 'repository'
    monkeypatch.setattr(source, 'ROOT_DIR', root)
    fit = root / 'results' / 'adaptive'
    numeric = root / 'src/Bayesian_state/model/model_0923.py'
    analysis = root / 'src/Bayesian_state/evaluation/evaluator.py'
    for path in (numeric, analysis):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('original\n')
    config = {'backend': 'model0923_adaptive', 'base_seed': 51,
              'seed_namespace': 'model0923_adaptive_full_v1', 'base_sim_config_path': str(root / 'simulation.yaml')}
    manifest = {'config': config, 'subjects': [{'subject': 335}],
                'input_sha256': {str(p): digest(p) for p in (numeric, analysis)}}
    write_json(fit / 'manifest.json', manifest)
    pid = point_id(POINT)
    result = {'subject': 335, 'condition': 3, 'status': 'unresolved', 'issues': ['search_budget_without_plateau'],
              'selected': pid, 'hyperparams': POINT, 'selected_mean_nll': .4,
              'candidate_bank': [{'id': pid, 'hyperparams': POINT, 'mean_nll': .5}],
              'independent_audit': {'scores': {pid: .4}, 'status': 'unresolved'},
              'challenge_diagnostic': {'status': 'acceptable_within_bank'},
              'boundary': {'review_required': True}, 'audit_family': 'cycle_0/independent_audit',
              'audit_budget': {'particle_count': 4, 'filter_seed_count': 2}, 'search_trace': []}
    write_json(fit / 'subjects/335/fit_result.json', result)
    write_json(fit / 'fit_results.json', {'backend': 'model0923_adaptive', 'subjects': {'335': result}})
    write_json(fit / 'effective_parameter_support.json', {'parameters': {}})
    paths = []
    for seed in phase_seeds(config['base_seed'], 335, result['audit_family'], 2, namespace=config['seed_namespace']):
        path = cache_path(fit, 335, pid, 4, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            np.savez_compressed(stream, particles=4, seed=seed, point_id=pid,
                                probabilities=np.array([[.1,.9],[.8,.2],[.3,.7]]),
                                observed=np.array([0,0,1]), mask=np.array([False,True,True]))
        paths.append(path)
    directory = fit / 'batches/cycle_0/audit_0'
    write_json(directory / 'plan.json', {'family': result['audit_family'], 'budget': result['audit_budget'],
                                        'banks': {'335': result['candidate_bank']}})
    write_json(directory / 'scores.json', {'cache_sha256': {str(p.relative_to(fit)): digest(p) for p in paths}})
    return fit


def test_initial_source_checks_snapshot_and_current_scientific_code(tmp_path, monkeypatch):
    fit = initial_fit(tmp_path, monkeypatch)
    metadata = source.validate_fit_source(fit)
    assert metadata['subject_acceptance'] == {'307': 'initial_search_only'}
    assert metadata['process_status'] == 'completed'
    current = source.ROOT_DIR / 'src/Bayesian_state/model/model_0923.py'
    current.write_text('changed mechanism\n')
    with pytest.raises(ValueError, match='scientific input hash'):
        source.validate_fit_source(fit)


def test_initial_snapshot_corruption_is_always_rejected(tmp_path, monkeypatch):
    fit = initial_fit(tmp_path, monkeypatch)
    (fit / 'source/src/Bayesian_state/model/model_0923.py').write_text('changed archive')
    with pytest.raises(ValueError, match='scientific input hash'):
        source.validate_fit_source(fit)


def test_archived_paths_are_resolved_inside_the_requested_fit(tmp_path, monkeypatch):
    fit = initial_fit(tmp_path, monkeypatch)
    path, config = source._base_config(fit, 'initial_cd')
    assert path == fit / 'runtime/simulation.yaml'
    assert config['engine_config_path'] == str(fit / 'source/configs/model.yaml')


def test_adaptive_completion_preserves_unresolved_status_and_analysis_changes(tmp_path, monkeypatch):
    fit = adaptive_fit(tmp_path, monkeypatch)
    analysis = source.ROOT_DIR / 'src/Bayesian_state/evaluation/evaluator.py'
    analysis.write_text('new report implementation\n')
    metadata = source.validate_fit_source(fit)
    assert metadata['process_status'] == 'completed'
    assert metadata['subject_acceptance'] == {'335': 'unresolved'}
    assert metadata['evaluation_source_changes'][0]['path'] == str(analysis)
    assert metadata['input_sha256'][str(analysis)] == digest(analysis)
    numeric = source.ROOT_DIR / 'src/Bayesian_state/model/model_0923.py'
    numeric.write_text('changed mechanism\n')
    with pytest.raises(ValueError, match='scientific input hash'):
        source.validate_fit_source(fit)


def test_queued_snapshot_checks_current_executing_source_and_keeps_archive_immutable(tmp_path, monkeypatch):
    fit = adaptive_fit(tmp_path, monkeypatch)
    manifest_path = fit / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    frozen_hashes = {}
    for name,expected in manifest['input_sha256'].items():
        current = Path(name)
        archived = fit / 'source' / current.relative_to(source.ROOT_DIR)
        archived.parent.mkdir(parents=True, exist_ok=True)
        archived.write_bytes(current.read_bytes())
        frozen_hashes[str(archived)] = expected
    manifest['input_sha256'] = frozen_hashes
    write_json(manifest_path, manifest)
    write_json(fit.parent / 'queue_plan.json', {'jobs': [{'output': str(fit)}], 'source_commit': 'source-commit'})
    analysis = source.ROOT_DIR / 'src/Bayesian_state/evaluation/evaluator.py'
    analysis.write_text('new current analysis\n')
    metadata = source.validate_fit_source(fit)
    assert metadata['source_fit_commit'] == 'source-commit'
    assert metadata['evaluation_source_changes'][0]['path'] == str(analysis)
    archived_analysis = fit / 'source/src/Bayesian_state/evaluation/evaluator.py'
    archived_analysis.write_text('corrupted frozen source\n')
    with pytest.raises(ValueError, match='scientific input hash'):
        source.validate_fit_source(fit)
    archived_analysis.write_text('original\n')
    current_numeric = source.ROOT_DIR / 'src/Bayesian_state/model/model_0923.py'
    current_numeric.write_text('changed current mechanism\n')
    with pytest.raises(ValueError, match='scientific input hash'):
        source.validate_fit_source(fit)


def test_incomplete_adaptive_process_is_rejected(tmp_path, monkeypatch):
    fit = adaptive_fit(tmp_path, monkeypatch)
    (fit / 'fit_results.json').unlink()
    with pytest.raises(ValueError, match='process is not complete'):
        source.validate_fit_source(fit)


def test_smoke_source_metadata_is_preserved(tmp_path, monkeypatch):
    fit = adaptive_fit(tmp_path, monkeypatch)
    path = fit / 'fit_results.json'
    record = json.loads(path.read_text())
    record['smoke_only'] = True
    write_json(path, record)
    assert source.validate_fit_source(fit)['smoke_only']


def test_selected_nll_is_checked_against_the_final_audit(tmp_path, monkeypatch):
    fit = adaptive_fit(tmp_path, monkeypatch)
    path = fit / 'subjects/335/fit_result.json'
    record = json.loads(path.read_text())
    record['selected_mean_nll'] = .2
    write_json(path, record)
    with pytest.raises(ValueError, match='selection and audit scores differ'):
        source.fit_subject_record(fit, 335)


def test_adaptive_subject_record_must_equal_the_completed_result(tmp_path, monkeypatch):
    fit = adaptive_fit(tmp_path, monkeypatch)
    path = fit / 'subjects/335/fit_result.json'
    row = json.loads(path.read_text())
    row['status'] = 'provisional_stop_within_tested_scope'
    write_json(path, row)
    with pytest.raises(ValueError, match='records differ'):
        source.validate_fit_source(fit)


def test_initial_replay_uses_original_budget_and_seed(tmp_path, monkeypatch):
    fit = initial_fit(tmp_path, monkeypatch)
    spec = source.original_replay_spec(fit, 307)
    assert spec['particles'] == 128
    assert spec['expected_mean_choice_nll'] == .5
    assert spec['source_probability_file'] is None
    assert spec['seed'] == source.original_replay_spec(fit, 307)['seed']


def test_adaptive_replay_uses_final_audit_cache_and_mask(tmp_path, monkeypatch):
    fit = adaptive_fit(tmp_path, monkeypatch)
    record = source.fit_subject_record(fit, 335)
    result = record['result']
    config = json.loads((fit / 'manifest.json').read_text())['config']
    seed = phase_seeds(config['base_seed'], 335, result['audit_family'], 2, namespace=config['seed_namespace'])[0]
    path = cache_path(fit, 335, result['selected'], 4, seed)
    spec = source.original_replay_spec(fit, 335)
    assert spec['seed'] == seed
    assert spec['particles'] == 4
    assert spec['expected_mean_choice_nll'] == pytest.approx(-np.log([.8,.7]).mean())
    assert spec['source_probability_file'] == str(path)
    assert record['best']['final_rescore_best']['hyperparams'] == POINT
    assert record['status'] == 'unresolved'
    assert record['issues'] == ['search_budget_without_plateau']


def test_adaptive_cache_change_is_rejected_by_the_saved_receipt(tmp_path, monkeypatch):
    fit = adaptive_fit(tmp_path, monkeypatch)
    path = next((fit / 'cache').rglob('*.npz'))
    path.write_bytes(b'changed scientific probability cache')
    with pytest.raises(ValueError, match='scientific input hash'):
        source.validate_fit_source(fit)


def test_audit_retains_selected_scores_status_and_configured_condition(tmp_path, monkeypatch):
    fit = initial_fit(tmp_path, monkeypatch)
    monkeypatch.setattr(source, 'resolve_engine_config', lambda *args,**kwargs: {'provenance': {'condition': 1}})
    output = tmp_path/'audit'
    output.mkdir()
    summary = source.audit_fit_source(fit, [307], output)
    assert summary.iloc[0]['condition'] == 1
    assert summary.iloc[0]['selected_fit_nll'] == .4
    assert summary.iloc[0]['fit_acceptance_status'] == 'initial_search_only'
    assert summary.iloc[0]['fit_issues'] == '[]'
    assert summary.iloc[0]['independent_audit_status'] == 'not_performed'
    finalists = pd.read_csv(output/'finalists.csv')
    assert finalists.iloc[0]['selected']
    assert finalists.iloc[0]['mean_paired_single_run_nll_delta'] == 0.
    assert finalists.iloc[0]['candidate_id'] == point_id(POINT)
    assert finalists.iloc[0]['candidate_id_source'] == 'derived_parameter_point'


def test_adaptive_audit_preserves_original_issues_checks_and_candidate_id(tmp_path, monkeypatch):
    fit = adaptive_fit(tmp_path, monkeypatch)
    specs = {}
    for path,value in POINT.items():
        name = path.split('.')[-1]
        lower,upper = source.PARAMETER_DOMAINS[name]
        specs[name] = dict(path=path, values=[value], domain=dict(lower=lower,upper=upper,
                           lower_closed=lower is not None,upper_closed=upper is not None))
    write_json(fit/'effective_parameter_support.json', {'parameters': specs})
    monkeypatch.setattr(source, '_base_config', lambda *args: (Path('/tmp/unused.yaml'), {}))
    monkeypatch.setattr(source, 'resolve_engine_config', lambda *args,**kwargs: {'provenance': {'condition': 3}})
    output = tmp_path/'audit'
    output.mkdir()
    summary = source.audit_fit_source(fit, [335], output).iloc[0]
    assert json.loads(summary['fit_issues']) == ['search_budget_without_plateau']
    assert summary['independent_audit_status'] == 'unresolved'
    assert summary['challenge_status'] == 'acceptable_within_bank'
    assert summary['boundary_review_required']
    finalist = pd.read_csv(output/'finalists.csv').iloc[0]
    assert finalist['candidate_id'] == point_id(POINT)
    assert finalist['candidate_id_source'] == 'saved'


def test_subject_summary_preserves_seed_scores_and_trial_ess(tmp_path, monkeypatch):
    arrays = TrialArrays(stimulus=np.zeros((3,4)), choices=np.array([1,1,2]),
                         feedback=np.array([1.,1.,0.]), categories=np.array([1,1,1]))
    inputs = dict(subject=102, condition=1, arrays=arrays,
                  engine={'modules': {'hypo_transitions_mod': {'kwargs': {'max_capacity': 2}}}},
                  frame=pd.DataFrame({'choice': arrays.choices, 'category': arrays.categories, 'feedback': arrays.feedback}),
                  best={'final_rescore_best': {'aggregated_error': .4}})
    monkeypatch.setattr(postfit, 'rule_catalog', lambda _: pd.DataFrame({'rule': [0,1]}))
    files = []
    probabilities = [np.array([[.5,.5],[.8,.2],[.3,.7]]), np.array([[.5,.5],[.6,.4],[.4,.6]])]
    for i,p in enumerate(probabilities):
        values = {'probabilities': p, 'seed': 11+i, 'particles': 2,
                  'pre_choice_ess': np.array([2.,1.+.5*i,1.5+.5*i]),
                  'post_choice_ess': np.array([2.,1.+.5*i,1.+.5*i]),
                  'resampled': np.array([False,True,bool(i)]),
                  'mean_choice_nll': postfit.score_probabilities(p, arrays.choices)['mean_choice_nll']}
        for phase in postfit.PHASES:
            values[phase+'_weights'] = np.full((3,2), .5)
            for name in postfit.SCALARS:
                values[phase+'_'+name] = np.ones((3,2))
            for name in (*postfit.RULE_FIELDS, 'active_beta'):
                values[phase+'_'+name] = np.full((3,2), .5)
            for name in ('search','global_search','failure','search_event','admission_event'):
                values[phase+'_'+name] = np.zeros(3)
        path = tmp_path/f'seed_{i}.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, **values)
        files.append(str(path))
    output = tmp_path/'summary'
    output.mkdir()
    summary, _ = postfit.summarize_subject(inputs, files, output, 2)
    assert summary['mean_choice_nll'] == pytest.approx(-np.log([.7,.65]).mean())
    trial = pd.read_csv(output/'trials.csv')
    np.testing.assert_allclose(trial['predictive_ess_fraction'], [1.,.625,.875])
    np.testing.assert_allclose(trial['filtered_ess_fraction'], [1.,.625,.625])
    np.testing.assert_allclose(trial['resampled_fraction'], [0.,1.,.5])
    assert pd.read_csv(output/'seed_nll.csv')['seed'].tolist() == [11,12]
    with pytest.raises(ValueError, match='distinct'):
        postfit.summarize_subject(inputs, [files[0],files[0]], output, 2)
