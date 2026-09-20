"""Preserve the estimator, preregistration and exact historical parameters."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import diagnose_model_0826_numerics as pilot


def arrays():
    return {'probabilities': np.array([[[.9, .1], [.2, .8], [.1, .9]],
                                      [[.1, .9], [.6, .4], [.9, .1]]]),
            'observed': np.array([0, 1, 0]), 'mask': np.array([False, True, True]),
            'seeds': np.array([10, 20])}


def test_mean_probability_before_log_and_mask():
    a = arrays()
    result = pilot.numerical_summary(a)
    assert result['mean_nll'] == pytest.approx(-np.log([.6, .5]).mean())
    assert result['scored_trials'] == 2
    assert result['mean_nll'] != pytest.approx(np.mean(result['half_mean_nll']))
    b = deepcopy(a)
    b['probabilities'][:, 0] = [1., 0.]
    assert pilot.numerical_summary(b) == result


def test_loss_contrast_preserves_trial_indices_and_total():
    a, b = arrays(), arrays()
    b['probabilities'][:, 2] = [.25, .75]
    result = pilot.trial_contrast(a, b)
    assert result['right_minus_left_mean_nll'] == pytest.approx(np.log(2)/2)
    assert sum(result['block64_mean_contributions']) == pytest.approx(result['right_minus_left_mean_nll'])
    assert result['top_trials'][0]['trial'] == 3
    b['mask'][0] = True
    with pytest.raises(AssertionError):
        pilot.trial_contrast(a, b)


@pytest.mark.parametrize('first_status', ['acceptable_within_bank', 'selected_point_inferior', 'unresolved'])
def test_only_final_budget_decides_without_reselection(tmp_path, monkeypatch, first_status):
    calls, selections = [], []
    class Scorer:
        def batch(self, bank, budget, family, name):
            calls.append(budget['particle_count'])
            return {1: [{'id': 'nominee', 'mean_nll': 1.}, {'id': 'other', 'mean_nll': 0.}]}
        def arrays(self, *args):
            return arrays()
    def diagnostic(data, nominee, *args):
        selections.append(nominee)
        return {'status': first_status if len(selections) == 1 else 'unresolved'}
    monkeypatch.setattr(pilot, 'decision_diagnostics', diagnostic)
    protocol = {'cases': [{'subject': 1, 'kind': 'numerical', 'nominee': 'nominee', 'budgets': [
        {'particle_count': 8, 'filter_seed_count': 2}, {'particle_count': 16, 'filter_seed_count': 2}]}],
        'alpha': .01, 'tolerance': .005, 'bootstrap_replicates': 100, 'base_seed': 42}
    result = pilot.execute(Scorer(), protocol, {1: []}, tmp_path)['subjects']['1']
    assert calls == [8, 16] and selections == ['nominee', 'nominee']
    assert result['fixed_bank_status'] == 'unresolved'
    assert result['whole_fit_status'] == 'not_reassessed'
    assert [look['primary'] for look in result['looks']] == [False, True]


def test_declared_banks_budget_and_coupled_boundary_probes():
    config, protocol = pilot.load_protocol(pilot.DEFAULT_CONFIG)
    if not (Path(protocol['previous'])/'fit_results.json').is_file():
        pytest.skip('Requires archived acceptance candidate file')
    banks, support, excluded = pilot.fixed_banks(protocol, config)
    assert {sid: len(rows) for sid, rows in banks.items()} == {307: 3, 221: 2, 314: 2, 102: 5, 206: 6}
    assert sum(len(banks[c['subject']])*sum(b['filter_seed_count'] for b in c['budgets']) for c in protocol['cases']) == 896
    assert config['parallel_budget'] == 128 and all(excluded.values())
    assert all('mean_nll' not in row for rows in banks.values() for row in rows)
    for sid in (102, 206):
        rows = {row['sources'][0]: row for row in banks[sid] if row['sources'][0] != 'historical_reference'}
        case = next(c for c in protocol['cases'] if c['subject'] == sid)
        baseline = next(row for row in banks[sid] if row['id'] == case['nominee'])
        original = pilot.extract_model_0826_parameters(baseline['hyperparams'])
        fixed = pilot.extract_model_0826_parameters(rows['ec_outer_fixed_error']['hyperparams'])
        coupled = pilot.extract_model_0826_parameters(rows['ec_outer']['hyperparams'])
        assert fixed['E_C'] == .01 and fixed['E_E'] == original['E_E']
        assert coupled['delta_E'] == pytest.approx(original['delta_E'])
        assert coupled['E_E'] < original['E_E']
    small, smoke = pilot.load_protocol(pilot.DEFAULT_CONFIG, smoke=True)
    assert small['parallel_budget'] == 1 and len(smoke['cases']) == 1
    assert smoke['cases'][0]['budgets'] == [{'particle_count': 2, 'filter_seed_count': 2}]
