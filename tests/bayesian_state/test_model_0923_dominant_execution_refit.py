"""A SELECT refit cannot silently expand its reference search policy."""
from copy import deepcopy

import pytest

from src.Bayesian_state.workflows.analysis.model_0923_dominant_execution_refit import expanded_settings


def reference():
    return dict(belief_intervention='separate_gamma_belief_evidence', parallel_budget=112,
                subjects=[208], arms=['POOL', 'SEARCH'], parameter_support='original.yaml',
                fixed_coordinates={'transfer': .01}, stages={'screen': {'particles': 128, 'seeds': [1, 2]}},
                seed=42, scope='original', initial_candidates=40)


def config():
    return dict(reference_fit='completed/fit', execution_policy='SELECT', parallel_budget=96, scope='SELECT')


def test_only_declared_identity_workers_and_scope_change():
    before = reference()
    expected = deepcopy(before)
    result = expanded_settings(config(), before)
    for key, value in expected.items():
        if key not in ('parallel_budget', 'scope'):
            assert result[key] == value
    assert result['execution_policy'] == 'SELECT'
    assert result['execution_reference_fit'] == 'completed/fit'
    assert result['parallel_budget'] == 96
    result['subjects'].append(999)
    result['stages']['screen']['seeds'].append(3)
    assert before == expected


@pytest.mark.parametrize('key,value', [
    ('particles', 1024), ('subjects', [208, 308]), ('initial_candidates', 80),
    ('execution_policy', 'MIX'), ('parallel_budget', 0), ('parallel_budget', 129),
    ('parallel_budget', True), ('reference_fit', ''),
])
def test_reject_changed_policy_or_invalid_allocation(key, value):
    cfg = config(); cfg[key] = value
    with pytest.raises(ValueError):
        expanded_settings(cfg, reference())


@pytest.mark.parametrize('key,value', [('belief_intervention', 'online_retention_evidence'),
                                     ('execution_policy', 'SELECT')])
def test_reference_must_be_separate_mixture(key, value):
    old = reference(); old[key] = value
    with pytest.raises(ValueError):
        expanded_settings(config(), old)
