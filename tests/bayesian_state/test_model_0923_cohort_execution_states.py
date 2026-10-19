"""Cohort SELECT oral evidence retains its execution, fit and trial boundaries."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.Bayesian_state.workflows.analysis import model_0923_cohort_execution_states as states


def evaluation(arm='POOL', condition=1):
    separate = states.comparison_helpers().cohort.separate
    fixed = {separate.TRANSFER: .01}
    point = dict(id='point', hyperparams={separate.GAMMA: .6, separate.PREFIX + 'intervention.decay': .8, **fixed})
    settings = dict(arms={arm: dict(precision='learned')}, mechanism=dict(kind='E'), fixed_coordinates=fixed)
    folder = Path('/tmp/cohort_execution_states_test')
    planned = dict(phase='evaluate', smoke=False, model=arm, candidate=point, seeds=[12], particles=4,
                   folder=str(folder), arm_settings=settings['arms'][arm], mechanism=settings['mechanism'])
    metadata = dict(subject=107, condition=condition, trials=8, training_trials=4, test_trials=4)
    record = dict(metadata, **{key: deepcopy(planned[key]) for key in ('candidate', 'model', 'seeds', 'particles')})
    intervention = 'separate_gamma_belief_evidence'
    actual = dict(deepcopy(planned), structural_intervention='retained_rule_precision', retain_precision=True,
        retention_reference_inputs={}, belief_intervention=intervention, online_belief=False,
        probe_mechanism='SEPARATE', cohort_belief_mechanism='SEPARATE', fixed_coordinates=fixed, execution_policy='SELECT')
    precision = dict(structural_intervention='retained_rule_precision', retained_precision=True, transitions=32)
    belief = dict(enabled=True, belief_intervention=intervention, fixed_coordinates=fixed, gamma=.6, delta=[.8],
                  updates=28, completed_feedback_timing_exact=True, belief_matches_saved_state=True)
    execution = dict(enabled=True, policy='SELECT', rule_execution_policy=states.POLICY,
                     uses_existing_dominant=True, task_categories_used_for_selection=False, updates=32)
    return [planned, actual, point, settings, 'SEPARATE', 107, arm, 12, 4, folder, metadata,
            record, precision, belief, {}], execution


@pytest.mark.parametrize('arm', ['POOL', 'SEARCH'])
@pytest.mark.parametrize('condition', [1, 2, 3])
def test_inherits_cohort_filter_identity_and_checks_select_execution(arm, condition):
    args, execution = evaluation(arm, condition)
    states.check_filter(*args, execution=execution)


@pytest.mark.parametrize('index,key,value', [(1, 'execution_policy', 'MIX'), (1, 'seeds', [13]),
    (1, 'candidate', {'id': 'other'}), (0, 'smoke', True), (10, 'subject', 207),
    (11, 'training_trials', 5), (12, 'transitions', 28), (13, 'updates', 32)])
def test_rejects_changed_execution_parameter_or_boundary(index, key, value):
    args, execution = evaluation()
    args[index][key] = value
    with pytest.raises(ValueError):
        states.check_filter(*args, execution=execution)


@pytest.mark.parametrize('key,value', [('enabled', False), ('policy', 'MIX'), ('updates', 28),
    ('uses_existing_dominant', False), ('task_categories_used_for_selection', True)])
def test_rejects_wrong_actual_select_record(key, value):
    args, execution = evaluation(); execution[key] = value
    with pytest.raises(ValueError):
        states.check_filter(*args, execution=execution)


def context():
    return dict(parameter_estimation_scope='prefix', evaluation_use=states.USE,
        parameter_candidate_policy='projected_common_search_policy_behavior_refit',
        structural_intervention='retained_rule_precision', belief_mechanism='SEPARATE',
        belief_intervention='separate_gamma_belief_evidence', cohort_fit='fit', cohort_summary='summary',
        cohort_manifest_sha256='cohort-hash', refit_nomination_sha256='nomination-hash',
        fixed_coordinates={'transfer': .01}, scientific_scope=True,
        selected_parameters=[dict(subject=107, arm=arm, parameter_id='point') for arm in states.CELLS],
        execution_policy='SELECT', rule_execution_policy=states.POLICY, execution_fit='fit',
        execution_summary='summary', execution_manifest_sha256='execution-hash')


def test_oral_context_identifies_both_mechanisms_and_original_timing():
    metadata, note = states.estimation_context(context())
    assert metadata['execution_policy'] == 'SELECT' and metadata['belief_mechanism'] == 'SEPARATE'
    assert metadata['window_definition'] == 'original_chronological_boundary'
    assert '训练段的选择似然' in note and '反馈前' in note


@pytest.mark.parametrize('key,value', [('scientific_scope', False), ('execution_policy', 'MIX'),
    ('parameter_estimation_scope', 'full'), ('execution_fit', 'other'), ('execution_summary', 'other'),
    ('execution_manifest_sha256', ''), ('cohort_manifest_sha256', ''), ('selected_parameters', []),
    ('selected_parameters', [dict(subject=107, arm='SELECT_POOL', parameter_id='point')])])
def test_incomplete_or_mixed_fit_cannot_enter_scientific_oral_alignment(key, value):
    protocol = context(); protocol[key] = value
    with pytest.raises(ValueError):
        states.estimation_context(protocol)


def test_record_check_keeps_select_updates_and_after_choice_timing():
    job = dict(model='POOL', candidate={'id': 'point'}, particles=4, seeds=[12])
    record = dict(job, arm='SELECT_POOL', trials=8, execution_policy='SELECT', rule_execution_policy=states.POLICY,
                  execution_updates=32, belief_updates=28, evaluation_use=states.USE, exact_behavior_replay=True,
                  timing='after_choice_before_current_feedback', data_use='behavior_only; no_oral_input')
    states.check_record(job, record)
    with pytest.raises(ValueError):
        states.check_record(job, dict(record, timing='after_feedback'))


def summary_fixture(tmp_path):
    from src.Bayesian_state.evaluation import model_0923_cohort_execution_refit as scores

    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    source, fit, evaluation_path = [tmp_path / name for name in ('summary', 'fit', 'evaluation')]
    save(source / 'cohort_manifest.json', {'status': 'complete'})
    save(fit / 'nominees.json', {'settings': {'execution_policy': 'SELECT', 'scope': states.USE}})
    bound = {}
    for directory, role in ((fit, 'training'), (evaluation_path, 'evaluate')):
        target = directory / 'chunks' / 'job'
        save(directory / 'jobs' / 'job.json', dict(target=str(target), phase=role, particles=4, seeds=[12], execution_policy='SELECT'))
        save(target.with_suffix('.json'), dict(trials=8))
        save(target.with_suffix('.execution.json'), evaluation()[1])
        for file in directory.glob('**/*.json'):
            bound[str(file.resolve())] = states.digest(file)
    code = {str(Path(module.__file__).resolve()): states.digest(Path(module.__file__)) for module in (scores, scores.driver)}
    declaration = dict(status='complete', execution_policy='SELECT', rule_execution_policy=states.POLICY,
        scope=states.USE, cohort_manifest_sha256=states.digest(source / 'cohort_manifest.json'), inputs_sha256=code)
    save(source / 'execution_manifest.json', declaration)
    manifest = dict(fit=str(fit), evaluation=str(evaluation_path))
    return source, manifest, bound, declaration, save


def test_summary_binding_keeps_actual_execution_and_analysis_sources(tmp_path):
    source, manifest, bound, _, _ = summary_fixture(tmp_path)
    result = states.bind_summary(source, manifest, 'SEPARATE', lambda *args: dict(bound))
    assert str((source / 'execution_manifest.json').resolve()) in result
    assert all(result[path] == sha for path, sha in bound.items())


@pytest.mark.parametrize('fault', ['scope', 'cohort_hash', 'analysis_source', 'unbound_execution', 'wrong_updates', 'different_fit_policy'])
def test_summary_rejects_inconsistent_policy_or_execution_evidence(tmp_path, fault):
    source, manifest, bound, declaration, save = summary_fixture(tmp_path)
    if fault == 'scope':
        declaration['scope'] = 'fixed_parameters'
    elif fault == 'cohort_hash':
        declaration['cohort_manifest_sha256'] = 'changed'
    elif fault == 'analysis_source':
        declaration['inputs_sha256'] = {}
    elif fault == 'different_fit_policy':
        file = Path(manifest['fit']) / 'nominees.json'
        save(file, {'settings': {'execution_policy': 'MIX', 'scope': states.USE}})
        bound[str(file)] = states.digest(file)
    else:
        file = Path(manifest['evaluation']) / 'chunks' / 'job.execution.json'
        if fault == 'unbound_execution':
            del bound[str(file)]
        else:
            save(file, dict(evaluation()[1], updates=28))
            bound[str(file)] = states.digest(file)
    save(source / 'execution_manifest.json', declaration)
    with pytest.raises(ValueError):
        states.bind_summary(source, manifest, 'SEPARATE', lambda *args: dict(bound))
