"""Protect inherited cohort policy and provenance when changing rule execution."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from src.Bayesian_state.workflows.analysis import model_0923_cohort_execution_refit as driver
from src.Bayesian_state.evaluation import model_0923_cohort_execution_refit as scores
from src.Bayesian_state.evaluation import model_0923_cohort_belief_refit as original_scores


def reference():
    return dict(belief_mechanism='SEPARATE', belief_intervention='separate_gamma_belief_evidence',
        subjects=[107, 207, 306], arms=['POOL', 'SEARCH'], parallel_budget=128, scope='original',
        data='original_prefixes', seed=42, parameter_support='original.yaml',
        budgets={'screen': {'particles': 128, 'seeds': 2}}, initial_candidates=40)


def config():
    return dict(reference_fit='completed/cohort', execution_policy='SELECT', parallel_budget=32, scope=driver.SCOPE)


def test_execution_change_preserves_all_subjects_supports_seeds_and_stage_budgets():
    before = reference(); original = deepcopy(before)
    settings = driver.expanded_settings(config(), before)
    assert {k: v for k, v in settings.items() if k in before and k not in ('parallel_budget', 'scope')} == {
        k: v for k, v in before.items() if k not in ('parallel_budget', 'scope')}
    assert settings['execution_policy'] == 'SELECT' and settings['parallel_budget'] == 32
    settings['subjects'].append(110); settings['budgets']['screen']['particles'] = 1
    assert before == original


@pytest.mark.parametrize('key,value', [('subjects', [110]), ('seed', 99), ('budgets', {}),
    ('scope', 'held_out'), ('parallel_budget', True), ('parallel_budget', 0), ('parallel_budget', 129),
    ('execution_policy', 'MIX'), ('reference_fit', '')])
def test_config_cannot_change_the_reference_policy_or_development_scope(key, value):
    own = config(); own[key] = value
    with pytest.raises(ValueError):
        driver.expanded_settings(own, reference())


@pytest.mark.parametrize('key,value', [('belief_mechanism', 'ONLINE'),
    ('belief_intervention', 'online_retention_evidence'), ('execution_policy', 'SELECT')])
def test_reference_requires_separate_belief_and_mixture_execution(key, value):
    old = reference(); old[key] = value
    with pytest.raises(ValueError):
        driver.expanded_settings(config(), old)


def binding_fixture(tmp_path):
    digest = lambda file: hashlib.sha256(Path(file).read_bytes()).hexdigest()
    source = tmp_path / 'source'; source.write_text('bound input')
    file = tmp_path / 'config.yaml'; file.write_text(yaml.safe_dump(config()))
    expected = {str(source): digest(source)}
    declared = dict(expected, **{str(file): digest(file)})
    return expected, declared, SimpleNamespace(digest=digest), source, file


def test_input_binding_requires_the_original_yaml_and_rechecks_all_hashes(tmp_path):
    expected, declared, validation, source, _ = binding_fixture(tmp_path)
    assert driver.checked_config_inputs(expected, declared, config(), validation) == declared
    source.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        driver.checked_config_inputs(expected, declared, config(), validation)


@pytest.mark.parametrize('corruption', ['missing_source', 'wrong_source', 'missing_yaml', 'extra_file', 'changed_yaml'])
def test_binding_rejects_missing_extra_or_semantically_changed_sources(tmp_path, corruption):
    expected, declared, validation, source, file = binding_fixture(tmp_path)
    if corruption == 'missing_source':
        del declared[str(source)]
    elif corruption == 'wrong_source':
        declared[str(source)] = 'wrong'
    elif corruption == 'missing_yaml':
        del declared[str(file)]
    elif corruption == 'extra_file':
        declared[str(tmp_path / 'extra')] = 'unexpected'
    else:
        file.write_text(yaml.safe_dump(dict(config(), subjects=[110])))
        declared[str(file)] = validation.digest(file)
    with pytest.raises(ValueError):
        driver.checked_config_inputs(expected, declared, config(), validation)


def test_private_summary_adds_execution_checks_without_changing_existing_cohort_scorer():
    before = original_scores.separate.SIDECARS
    adapted = scores.cohort_summary()
    assert '.execution.json' in adapted.separate.SIDECARS
    assert original_scores.separate.SIDECARS == before
    assert original_scores.driver.CONFIG_KEYS != adapted.driver.CONFIG_KEYS
    with pytest.raises(ValueError, match='SEPARATE'):
        adapted.audit_job(Path('unused'), 'ONLINE', lambda p: p)


def test_frozen_configuration_keeps_the_original_reference_and_scope():
    settings = driver.expanded_settings(config(), reference())
    assert scores.frozen_config(settings) == config()
