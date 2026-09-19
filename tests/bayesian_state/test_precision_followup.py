"""Fixed nominations, primary-budget discipline and trialwise ambiguity."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import followup_model_0826_precision as pilot


def arrays(p):
    p = np.array(p, dtype=float)
    return {'probabilities': np.stack([p, p]), 'observed': np.array([0, 1, 0]),
            'mask': np.array([False, True, True]), 'seeds': np.array([1, 2])}


def test_equal_total_score_does_not_imply_equal_predictions():
    a = arrays([[.99, .01], [.2, .8], [.4, .6]])
    b = arrays([[.01, .99], [.6, .4], [.8, .2]])
    result = pilot.prediction_comparison(a, b)
    assert result['mean_nll'][0] == pytest.approx(result['mean_nll'][1])
    assert result['mean_total_variation'] == pytest.approx(.4)
    assert result['argmax_disagreements'] == 2
    assert result['scored_trials'] == 2
    assert result['split_half_probability_rmse'] == [0., 0.]


def test_prediction_comparison_rejects_misaligned_trials():
    a = arrays([[.5, .5]]*3)
    b = deepcopy(a)
    b['mask'][0] = True
    with pytest.raises(AssertionError):
        pilot.prediction_comparison(a, b)


@pytest.mark.parametrize('first_status', ['acceptable_within_bank', 'selected_point_inferior', 'unresolved'])
def test_earlier_budget_never_stops_or_reselects(tmp_path, monkeypatch, first_status):
    a = arrays([[.5, .5]]*3)
    calls, selections = [], []
    class Scorer:
        def batch(self, bank, budget, family, name):
            calls.append(budget['particle_count'])
            return {1: [{'id': 'nominee', 'mean_nll': 1.}, {'id': 'challenger', 'mean_nll': 0.}]}
        def arrays(self, *args):
            return a
    def diagnostic(data, nominee, *args):
        selections.append(nominee)
        return {'status': first_status if len(selections) == 1 else 'unresolved'}
    monkeypatch.setattr(pilot, 'decision_diagnostics', diagnostic)
    protocol = {'cases': [{'subject': 1, 'nominee': 'nominee', 'budgets': [
        {'particle_count': 8, 'filter_seed_count': 2}, {'particle_count': 16, 'filter_seed_count': 2}]}],
        'alpha': .05, 'tolerance': .005, 'bootstrap_replicates': 100, 'base_seed': 42}
    result = pilot.execute(Scorer(), protocol, {1: []}, tmp_path)['subjects']['1']
    assert calls == [8, 16] and selections == ['nominee', 'nominee']
    assert result['pair_status'] == 'unresolved'
    assert result['whole_bank_status'] == 'not_reassessed'
    assert [r['primary'] for r in result['looks']] == [False, True]


def test_declared_bank_and_budget_cap():
    config, protocol = pilot.load_protocol(pilot.DEFAULT_CONFIG)
    if not (Path(protocol['previous'])/'shortlists.json').is_file():
        pytest.skip('Requires archived boundary candidate file')
    banks, excluded = pilot.fixed_banks(protocol)
    assert {s: len(rows) for s, rows in banks.items()} == {203: 2, 301: 2}
    assert {s: len(ids) for s, ids in excluded.items()} == {'203': 2, '301': 4}
    assert all('mean_nll' not in r for rows in banks.values() for r in rows)
    assert sum(len(c['candidates'])*sum(b['filter_seed_count'] for b in c['budgets']) for c in protocol['cases']) == 384
    assert config['parallel_budget'] == 128
    small, smoke = pilot.load_protocol(pilot.DEFAULT_CONFIG, smoke=True)
    assert small['parallel_budget'] == 1
    assert smoke['cases'][0]['budgets'] == [{'particle_count': 2, 'filter_seed_count': 2}]
