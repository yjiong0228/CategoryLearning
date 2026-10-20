"""Protect the bounded recovery batch and the boundary around generating truth."""
from copy import deepcopy
from pathlib import Path

import pytest

from src.Bayesian_state.workflows.analysis import model_0923_finalization_recovery as recovery


def config(smoke=False):
    return dict(scope='software_validation' if smoke else 'finalization_recovery',
        recommendation=None if smoke else 'frozen/recommendation.json',
        software_candidate='SEARCH_ADD10' if smoke else None,
        evidence_config='registered/evidence.yaml', subjects=[205] if smoke else [205, 306],
        replicates=1 if smoke else 2, parallel_budget=1 if smoke else 32, seed=2026102102)


@pytest.mark.parametrize('smoke', [False, True])
def test_scientific_and_software_scopes_are_distinct(smoke):
    assert recovery.validate_config(config(smoke)) is smoke


@pytest.mark.parametrize('field,value', [
    ('subjects', [108, 205]), ('subjects', [205, 306, 308]), ('replicates', 3),
    ('replicates', True), ('parallel_budget', 128), ('recommendation', None),
    ('software_candidate', 'SEARCH_ADD10'), ('scope', 'unbounded_recovery'),
])
def test_formal_batch_cannot_expand_or_skip_recommendation(field, value):
    own = config()
    own[field] = value
    with pytest.raises(ValueError):
        recovery.validate_config(own)


@pytest.mark.parametrize('field,value', [
    ('subjects', [205, 306]), ('replicates', 2), ('parallel_budget', 2),
    ('recommendation', 'formal/recommendation.json'), ('software_candidate', None),
])
def test_software_case_stays_small_and_explicit(field, value):
    own = config(True)
    own[field] = value
    with pytest.raises(ValueError):
        recovery.validate_config(own)


def job(candidate='SEARCH_ADD10', smoke=False):
    arm, mode, scale = recovery.CANDIDATES[candidate]
    return dict(recovery_candidate=candidate,
        recovery_scope='software_validation' if smoke else 'finalization_recovery',
        model=arm, arm_settings=dict(precision=mode, rate_scale=scale, upper=25.),
        smoke=smoke, seeds=[12, 13], recovery_generation_seeds=[1, 2])


@pytest.mark.parametrize('candidate', list(recovery.CANDIDATES))
@pytest.mark.parametrize('smoke', [False, True])
def test_scoring_preserves_the_selected_proposal_and_precision(candidate, smoke):
    recovery.check_job(job(candidate, smoke))


@pytest.mark.parametrize('fault', ['seed_overlap', 'mode', 'arm', 'scope', 'smoke'])
def test_worker_rejects_crossed_mechanism_or_simulation_roles(fault):
    own = job()
    if fault == 'seed_overlap':
        own['seeds'] = [1, 13]
    elif fault == 'mode':
        own['arm_settings']['precision'] = 'log'
    elif fault == 'arm':
        own['model'] = 'POOL'
    elif fault == 'scope':
        own['recovery_scope'] = 'development_fit'
    else:
        own['smoke'] = True
    with pytest.raises(ValueError):
        recovery.check_job(own)


def test_truth_parameters_are_not_opened_before_estimates_freeze(tmp_path, monkeypatch):
    settings = dict(scope='synthetic_online_joint_recovery')
    monkeypatch.setattr(recovery, 'case_settings', lambda *args: (settings, {}, {}, {}))
    opened = []

    def read(path):
        opened.append(Path(path).name)
        if Path(path).name == 'nominees.json':
            return dict(status='provisional', settings=deepcopy(settings), smoke=False)
        if Path(path).name == 'completion.json':
            return dict(status='complete', smoke=False)
        raise AssertionError('Generating truth was opened before nomination freeze')

    monkeypatch.setattr(recovery, 'read', read)
    with pytest.raises(ValueError, match='Freeze matching'):
        recovery.score_truth(tmp_path / 'prepared', 205, 0, tmp_path / 'fit', tmp_path / 'new', False)
    assert opened == ['nominees.json', 'completion.json']
    assert not (tmp_path / 'new').exists()


def test_existing_research_output_is_preserved(tmp_path):
    protected = tmp_path / 'existing'
    protected.mkdir()
    marker = protected / 'saved.txt'
    marker.write_text('retained')
    with pytest.raises(FileExistsError):
        recovery.search(tmp_path / 'unused', 205, 0, protected, False)
    assert marker.read_text() == 'retained'
