"""Prospective v2 stopping, independent audit, and mandatory-score contracts."""
from copy import deepcopy
import json

import numpy as np
import pytest
import yaml

from src.Bayesian_state.optimization import adaptive_fit as fit
from src.Bayesian_state.optimization import adaptive_runtime as runtime
from src.Bayesian_state.optimization.adaptive_boundary import adaptive_support
from src.Bayesian_state.optimization.parameter_space import load_model_parameter_space


@pytest.fixture
def setup(monkeypatch):
    config = fit.load_fit_config(fit.DEFAULT_CONFIG, smoke=True)
    config['search']['max_cycles'] = 2
    original = load_model_parameter_space(config['parameter_space'], expected_model_id='model_0826')
    space, anchor, support = adaptive_support(original, {})

    def fake_seed(context, point, particles, seed, path):
        runtime.publish(path, lambda h: np.savez_compressed(
            h, probabilities=np.array([[.5, .5], [.8, .2], [.7, .3]]),
            observed=np.zeros(3, dtype=int), mask=np.array([False, True, True]),
            particles=particles, seed=seed, point_id=runtime.point_id(point), seconds=0.))
        return {'computed': True}

    monkeypatch.setattr(runtime, 'score_seed', fake_seed)
    class Arrays:
        choices = np.zeros(3)
    contexts = {103: {'subject': 103, 'condition': 1, 'arrays': Arrays()}}
    return config, contexts, space, anchor, support


def test_v2_default_and_legacy_schema(tmp_path):
    new = fit.load_fit_config(fit.DEFAULT_CONFIG)
    old = fit.load_fit_config(fit.DEFAULT_CONFIG.with_name('model_0826_adaptive_fit_v1.yaml'))
    assert new['schema_version'] == 2 and old['schema_version'] == 1
    assert 'audit' not in new['precision'] and 'audit' in old['precision']
    assert new['boundary']['extensions'] == old['boundary']['extensions'] == {}
    assert new['parallel_budget'] == 128
    raw = yaml.safe_load(fit.DEFAULT_CONFIG.read_text())
    raw['precision']['audit'] = raw['precision']['tiers'][-1]
    bad = tmp_path/'bad.yaml'; bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='Unknown or missing'):
        fit.load_fit_config(bad)


def test_direct_boundary_skips_discovery_and_resume_preserves_result(tmp_path, setup):
    config, contexts, space, anchor, support = setup
    config['boundary']['max_candidates'] = 100  # Include tied boundary anchors.
    first = fit.fit_subjects(config, contexts, space, anchor, support, tmp_path, True)
    second = fit.fit_subjects(config, contexts, space, anchor, support, tmp_path, True)
    assert first == second
    assert first['schema_version'] == 2
    assert not list(tmp_path.glob('batches/*/boundary/discovery/*'))
    assert list(tmp_path.glob('batches/*/boundary/guide/scores.json'))
    assert not list(tmp_path.glob('batches/*/decision*'))
    assert len(list(tmp_path.glob('subjects/103/cycle_*.json'))) == 1
    row = first['subjects']['103']
    assert row['precision_policy'] == 'single_independent_audit'
    assert len(row['tiers']) == 1
    assert row['challenge_diagnostic']['status'] == 'acceptable_within_bank'
    assert row['state_precision'] == row['parameter_recovery'] == 'not_checked'
    assert row['independent_audit']['alpha'] == pytest.approx(.05 / 4)
    assert 'selection_precision_unresolved' not in row['issues']
    if row['boundary']['review_required']:
        assert 'boundary_review_required' in row['issues']
    legacy = deepcopy(config)
    legacy['schema_version'] = 1
    legacy['precision']['audit'] = deepcopy(legacy['precision']['tiers'][-1])
    old_root = tmp_path/'legacy'
    fit.fit_subjects(legacy, contexts, space, anchor, support, old_root, True)
    old = json.loads((old_root/'batches/cycle_0/boundary/guide/scores.json').read_text())
    new = json.loads((tmp_path/'batches/cycle_0/boundary/guide/scores.json').read_text())
    assert old['rows'] == new['rows']
    assert old['cache_sha256'] == new['cache_sha256']


def test_unconfirmed_guide_gain_does_not_restart_search(tmp_path, setup, monkeypatch):
    config, contexts, space, anchor, support = setup
    config['search']['min_improvement'] = .001
    original = runtime.Scorer.batch
    def noisy_guide(self, banks, budget, family, name):
        rows = original(self, banks, budget, family, name)
        if '/challenge_' in name and family == 'guide':
            for bank in rows.values():
                for row in bank:
                    row['mean_nll'] -= .002  # Guide-only noise; audit probabilities agree.
        return rows
    monkeypatch.setattr(runtime.Scorer, 'batch', noisy_guide)
    result = fit.fit_subjects(config, contexts, space, anchor, support, tmp_path, True)['subjects']['103']
    assert result['guide_challenge_improved']
    assert result['challenge_baseline'] != result['challenge_nominee']
    assert result['challenge_diagnostic']['status'] == 'acceptable_within_bank'
    assert 'challenge_improved' not in result['issues']
    assert len(list(tmp_path.glob('subjects/103/cycle_*.json'))) == 1


@pytest.mark.parametrize('trigger', ['primary', 'challenge'])
def test_confirmed_inferiority_restarts_with_fresh_seeds(tmp_path, setup, monkeypatch, trigger):
    config, contexts, space, anchor, support = setup
    calls = []
    def diagnostics(arrays, selected, tolerance, alpha, replicates, seed):
        # Each cycle checks the nominated point, then the predeclared challenge.
        index = len(calls)
        other = next((p for p in arrays if p != selected), selected)
        scores = {pid: .3 for pid in arrays}; scores[other] = .1
        fail = index == (0 if trigger == 'primary' else 1)
        calls.append((selected, other, set(next(iter(arrays.values()))['seeds'])))
        return {'selected': selected, 'scores': scores,
                'status': 'selected_point_inferior' if fail else 'acceptable_within_bank'}
    monkeypatch.setattr(fit, 'decision_diagnostics', diagnostics)
    result = fit.fit_subjects(config, contexts, space, anchor, support, tmp_path, True)['subjects']['103']
    first = json.loads((tmp_path/'subjects/103/cycle_0.json').read_text())
    assert first['selected'] == calls[0][0]
    assert first['audit_family'] != result['audit_family']
    assert not calls[0][2] & calls[-1][2]
    if trigger == 'primary':
        assert result['selected'] == calls[0][1]
    else:
        assert 'challenge_improved' in first['issues']


@pytest.mark.parametrize('first_status,expected_tiers', [('unresolved', 2), ('acceptable_within_bank', 1), ('selected_point_inferior', 1)])
def test_audit_upgrades_only_when_a_decision_is_still_needed(tmp_path, setup, monkeypatch, first_status, expected_tiers):
    config, contexts, space, anchor, support = setup
    config['search']['max_cycles'] = 1
    config['precision']['tiers'] += [{'particle_count': 3, 'filter_seed_count': 3}]
    calls = []
    def diagnostics(arrays, selected, tolerance, alpha, replicates, seed):
        calls.append((len(next(iter(arrays.values()))['seeds']), alpha))
        return {'selected': selected, 'scores': {pid: .3 for pid in arrays},
                'status': first_status if len(calls) == 1 else 'acceptable_within_bank'}
    monkeypatch.setattr(fit, 'decision_diagnostics', diagnostics)
    row = fit.fit_subjects(config, contexts, space, anchor, support, tmp_path, True)['subjects']['103']
    assert len(row['tiers']) == expected_tiers
    assert row['audit_budget']['filter_seed_count'] == (3 if expected_tiers == 2 else 2)
    assert all(alpha == pytest.approx(.05 / 4) for _, alpha in calls)
    assert len(calls) == 2 * expected_tiers


def test_unresolved_challenge_cannot_claim_stop_or_expand_search_forever(tmp_path, setup, monkeypatch):
    config, contexts, space, anchor, support = setup
    count = 0
    def diagnostics(arrays, selected, *args):
        nonlocal count
        count += 1
        return {'selected': selected, 'scores': {pid: .3 for pid in arrays},
                'status': 'unresolved' if count % 2 == 0 else 'acceptable_within_bank'}
    monkeypatch.setattr(fit, 'decision_diagnostics', diagnostics)
    row = fit.fit_subjects(config, contexts, space, anchor, support, tmp_path, True)['subjects']['103']
    assert row['status'] == 'unresolved'
    assert 'challenge_precision_unresolved' in row['issues']
    assert len(list(tmp_path.glob('subjects/103/cycle_*.json'))) == 1


def test_streamlined_status_separates_search_numerical_and_boundary_gates():
    good = 'acceptable_within_bank'
    assert fit.streamlined_stop_status(True, good, good, False) == ('provisional_stop_within_tested_scope', [])
    for args in [(False, good, good, False), (True, 'unresolved', good, False),
                 (True, good, 'unresolved', False), (True, good, 'selected_point_inferior', False),
                 (True, good, good, True)]:
        assert fit.streamlined_stop_status(*args)[0] == 'unresolved'
