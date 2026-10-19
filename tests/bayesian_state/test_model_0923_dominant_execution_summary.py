"""Check actual SELECT execution identity and avoid counting sidecars as runs."""
import json

import pytest

from src.Bayesian_state.evaluation.model_0923_dominant_execution_refit import (
    POLICY, check_selection, separate_summary, training_chunk_budgets)


def execution(phase):
    job = dict(phase=phase, seeds=[7, 8], particles=4, execution_policy='SELECT')
    result = dict(trials=9)
    proof = dict(enabled=True, policy='SELECT', rule_execution_policy=POLICY,
                 updates=18 if phase == 'autonomous' else 72,
                 uses_existing_dominant=True, task_categories_used_for_selection=False)
    return job, result, proof


@pytest.mark.parametrize('phase', ['training', 'evaluate', 'autonomous'])
def test_execution_includes_trial_zero_for_every_particle_or_path(phase):
    job, result, proof = execution(phase)
    check_selection(job, result, proof)
    proof['updates'] -= len(job['seeds']) * (1 if phase == 'autonomous' else job['particles'])
    with pytest.raises(ValueError, match='per-choice count'):
        check_selection(job, result, proof)


@pytest.mark.parametrize('field,value', [('enabled', False), ('policy', 'MIX'),
    ('rule_execution_policy', 'mixture'), ('uses_existing_dominant', False),
    ('task_categories_used_for_selection', True)])
def test_rejects_wrong_execution(field, value):
    job, result, proof = execution('training')
    proof[field] = value
    with pytest.raises(ValueError):
        check_selection(job, result, proof)


def test_rejects_wrong_job_policy():
    job, result, proof = execution('training')
    job['execution_policy'] = 'MIX'
    with pytest.raises(ValueError):
        check_selection(job, result, proof)


def test_counts_one_worker_with_all_intervention_sidecars(tmp_path):
    chunk = tmp_path / 'subject_208' / 'POOL' / 'chunks'
    chunk.mkdir(parents=True)
    row = dict(subject=208, model='POOL', numeric_pools=[dict(num_threads=1)],
               particles=4, seeds=[7, 8], trials=9, cpu_seconds=1.5)
    (chunk / 'point.json').write_text(json.dumps(row))
    for suffix in ('experiment', 'precision', 'online', 'belief', 'execution'):
        (chunk / ('point.' + suffix + '.json')).write_text('{}')
    rows = training_chunk_budgets(tmp_path)
    assert len(rows) == 1 and rows[0]['state_updates'] == 72
    assert rows[0]['filter_runs'] == 2 and rows[0]['cpu_seconds'] == 1.5


def test_summary_bindings_do_not_mutate_original_module():
    from src.Bayesian_state.evaluation import model_0923_separate_belief_refit as original
    before = original.SIDECARS
    module = separate_summary()
    assert '.execution.json' in module.SIDECARS
    assert original.SIDECARS == before and '.execution.json' not in before
