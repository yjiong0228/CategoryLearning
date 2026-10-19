"""Execution provenance and common labels for matched oral comparisons."""
from copy import deepcopy

import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis import model_0923_execution_refit_oral as oral


def state(policy):
    job = dict(model='SEARCH', retain_precision=True, belief_intervention='separate_gamma_belief_evidence',
               candidate={'id': 'one', 'hyperparams': {'engine.modules.memory_mod.kwargs.gamma': .99}},
               seeds=[17], particles=4, evaluation_use='development_scope')
    record = dict(job, arm='SEARCH' if policy == 'MIX' else 'SELECT_SEARCH', separate_belief=True,
                  online_belief=False, belief_matches_saved_state=True, belief_gamma=.99,
                  belief_updates=28, particle_transitions=32, trials=8,
                  timing='after_choice_before_current_feedback', data_use='behavior_only; no_oral_input',
                  exact_behavior_replay=True)
    if policy == 'SELECT':
        job.update(execution_policy='SELECT', rule_execution_policy=oral.POLICY)
        record.update(execution_policy='SELECT', rule_execution_policy=oral.POLICY, execution_updates=32)
    return job, record


@pytest.mark.parametrize('policy', ['MIX', 'SELECT'])
def test_completed_executed_identity_is_accepted(policy):
    oral.check_state(*state(policy), policy, 'SEARCH')


@pytest.mark.parametrize('field,value', [('separate_belief', False), ('belief_gamma', .6),
    ('belief_updates', 24), ('particle_transitions', 28), ('timing', 'after_feedback'),
    ('data_use', 'oral_conditioned'), ('seeds', [18]), ('execution_updates', 28),
    ('rule_execution_policy', 'other'), ('evaluation_use', 'changed')])
def test_changed_state_is_rejected(field, value):
    job, record = state('SELECT')
    record[field] = value
    with pytest.raises(ValueError):
        oral.check_state(job, record, 'SELECT', 'SEARCH')


def test_select_cannot_be_called_a_mix_reference():
    job, record = state('SELECT')
    record['arm'] = 'SEARCH'
    with pytest.raises(ValueError, match='MIX reference'):
        oral.check_state(job, record, 'MIX', 'SEARCH')


def test_common_arm_changes_only_copied_presentation_labels():
    trials = pd.DataFrame({'arm': ['SELECT_SEARCH'], 'state_trial': [0], 'choice': [1]})
    rows = [{'arm': 'SELECT_SEARCH', 'support': .4}]
    estimates = {'window': {'values': [.4]}}
    mapped = oral.common_arm((trials, rows, estimates), 'SEARCH')
    assert mapped[0].arm.tolist() == ['SEARCH'] and mapped[1][0]['arm'] == 'SEARCH'
    assert trials.arm.tolist() == ['SELECT_SEARCH'] and rows[0]['arm'] == 'SELECT_SEARCH'
    assert mapped[0].choice.tolist() == [1] and mapped[2] is estimates


def identity():
    return dict(parameter_estimation_scope='prefix',
        parameter_candidate_policy='projected_common_search_policy_behavior_refit',
        mechanism={'retain_learned_precision': True, 'kind': 'E'}, structural_intervention='retained_rule_precision',
        scientific={'particles': 512}, arm_settings={'precision': 'fast'}, archived_pf_sha256='pf',
        task_sha256={'training': 'data'}, oral_manifest_sha256='oral', model_sources={'old_wrapper.py': 'old'})


def test_reference_hash_must_prove_missing_original_dependency():
    old, new = identity(), identity()
    new['model_sources'] = {'select_wrapper.py': 'select'}
    with pytest.raises(ValueError):
        oral.match_identities(old, new, {})
    references = {str((oral.ROOT / 'old_wrapper.py').resolve()): 'old'}
    oral.match_identities(old, new, references)
    assert 'old_wrapper.py' not in new['model_sources']
    changed = deepcopy(new); changed['task_sha256'] = {'training': 'changed'}
    with pytest.raises(ValueError):
        oral.match_identities(old, changed, references)


def test_existing_output_is_preserved(tmp_path):
    output = tmp_path / 'kept'; output.mkdir()
    with pytest.raises(FileExistsError):
        oral.run(tmp_path / 'missing1', tmp_path / 'missing2', output, 107, 'SEARCH')
