"""Refitted SELECT oral evidence must keep its own policy and fit provenance."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import yaml

from src.Bayesian_state.evaluation.model_0923_h_validation import digest
from src.Bayesian_state.workflows.analysis import model_0923_dominant_execution_refit_states as states


def record():
    job = dict(model='POOL', candidate={'id': 'candidate'}, particles=4, seeds=[12])
    row = dict(job, arm='SELECT_POOL', trials=8, execution_policy='SELECT', rule_execution_policy=states.POLICY,
               execution_updates=32, belief_updates=28, evaluation_use=states.USE, exact_behavior_replay=True,
               timing='after_choice_before_current_feedback', data_use='behavior_only; no_oral_input')
    return job, row


def test_accepts_actual_refitted_policy_counts():
    states.check_record(*record())


@pytest.mark.parametrize('field,value', [('arm', 'MIX_POOL'), ('execution_policy', 'MIX'),
    ('rule_execution_policy', 'mixture'), ('execution_updates', 28), ('belief_updates', 32),
    ('evaluation_use', 'fixed_parameters'), ('exact_behavior_replay', False), ('particles', 8),
    ('seeds', [13]), ('candidate', {'id': 'other'}), ('timing', 'after_feedback')])
def test_rejects_wrong_oral_state_identity(field, value):
    job, row = record()
    row[field] = value
    with pytest.raises(ValueError):
        states.check_record(job, row)


def context():
    return dict(parameter_estimation_scope='prefix', evaluation_use=states.USE,
        parameter_candidate_policy='projected_common_search_policy_behavior_refit', execution_fit='fit',
        refit_nomination_sha256='nomination', selected_parameters=[], execution_policy='SELECT',
        rule_execution_policy=states.POLICY, scientific_scope=True, fixed_coordinates={'transfer': .01},
        belief_intervention='separate_gamma_belief_evidence', structural_intervention='retained_rule_precision')


def test_context_identifies_behavior_refitting_and_choice_timing():
    metadata, note = states.estimation_context(context())
    assert metadata['execution_policy'] == 'SELECT' and metadata['parameter_estimation_scope'] == 'prefix'
    assert '训练选择似然' in note and '反馈前' in note


@pytest.mark.parametrize('key,value', [('scientific_scope', False), ('parameter_estimation_scope', 'full'),
    ('execution_policy', 'MIX'), ('execution_fit', ''), ('refit_nomination_sha256', ''),
    ('parameter_candidate_policy', 'fixed_parameters'), ('evaluation_use', 'fixed_parameters')])
def test_short_or_mismatched_selection_cannot_be_scientific_oral_evidence(key, value):
    protocol = context()
    protocol[key] = value
    with pytest.raises(ValueError):
        states.estimation_context(protocol)


def references(tmp_path, monkeypatch):
    config = dict(reference_fit='reference_fit', execution_policy='SELECT', parallel_budget=16, scope='development')
    original = dict(execution_reference_fit=config['reference_fit'], execution_policy='SELECT', parallel_budget=16,
                    scope='development')
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(config))
    reference = tmp_path / 'reference.json'
    reference.write_text('{}')
    evaluation = {str(reference): digest(reference)}
    search = dict(evaluation, **{str(config_path): digest(config_path)})
    probe = 'src/Bayesian_state/workflows/analysis/model_0923_dominant_execution_probe.py'
    execution = 'src/Bayesian_state/model/modules/hypothesis_transition/dominant_rule_execution.py'
    frozen = dict(settings=original, source_sha256={name: digest(states.ROOT / name)
                                                  for name in (states.REFIT, probe, execution)})
    fit = tmp_path / 'fit'
    fit.mkdir()
    declaration = dict(execution_policy='SELECT', rule_execution_policy=states.POLICY,
                       execution_reference_fit='reference_fit', inputs_sha256=search)
    (fit / 'execution_refit_protocol.json').write_text(json.dumps(declaration))
    refitter = SimpleNamespace(helpers=lambda: None,
        prepare_settings=lambda config, modules: (deepcopy(original), deepcopy(evaluation)),
        PROBE=probe, EXECUTION=execution)
    monkeypatch.setattr(states, 'helper', lambda name, relative: refitter)
    return fit, frozen, search, evaluation, config_path


def test_search_config_and_reconstructed_evaluation_references_both_bound(tmp_path, monkeypatch):
    fit, frozen, search, evaluation, _ = references(tmp_path, monkeypatch)
    actual = states.reference_bindings(fit, frozen)
    assert actual == (search, evaluation)
    assert len(search) == len(evaluation) + 1


@pytest.mark.parametrize('change', ['missing_config', 'changed_config', 'changed_reference', 'changed_policy', 'changed_source'])
def test_rejects_changed_reference_configuration_or_execution_source(tmp_path, monkeypatch, change):
    fit, frozen, search, _, config = references(tmp_path, monkeypatch)
    file = fit / 'execution_refit_protocol.json'
    declaration = json.loads(file.read_text())
    if change == 'missing_config':
        del declaration['inputs_sha256'][str(config)]
    elif change == 'changed_config':
        config.write_text('execution_policy: MIX\n')
    elif change == 'changed_reference':
        declaration['inputs_sha256'][str(tmp_path / 'reference.json')] = 'wrong'
    elif change == 'changed_policy':
        declaration['rule_execution_policy'] = 'mixture'
    else:
        frozen['source_sha256'][states.REFIT] = 'wrong'
    file.write_text(json.dumps(declaration))
    with pytest.raises(ValueError):
        states.reference_bindings(fit, frozen)
