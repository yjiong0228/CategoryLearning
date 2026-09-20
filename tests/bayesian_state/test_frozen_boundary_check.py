"""Boundary attribution must retain good inner alternatives and coupled rates."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.optimization.model_0826 import ETA_PLUS_PATH, extract_model_0826_parameters
from src.Bayesian_state.workflows.analysis import probe_model_0826_frozen_boundaries as probe


def arrays(probabilities):
    values = np.broadcast_to(np.asarray(probabilities, dtype=float), (2,))
    p = np.array([[[.5, .5], [v, 1-v]] for v in values])
    return {'probabilities': p, 'observed': np.array([0, 0]),
            'mask': np.array([False, True]), 'seeds': np.array([11, 22])}


@pytest.mark.parametrize('inside,outer,status', [
    ([.2, .9], .8, 'no_material_gain_in_tested_bank'),
    ([.2, .3], .8, 'outer_gain_confirmed_in_tested_bank'),
    ([.5, .5], [.1, .9], 'unresolved'),
])
def test_range_decision_uses_all_inner_controls(inside, outer, status):
    data = {'a': arrays(inside[0]), 'b': arrays(inside[1]), 'outer': arrays(outer)}
    protocol = {'alpha': .01, 'tolerance': .005, 'bootstrap_replicates': 200, 'base_seed': 41}
    result = probe.boundary_decision(data, ['a', 'b'], protocol, 1)
    assert result['status'] == status
    assert result['per_inner_alpha'] == .005
    assert set(result['inner_comparisons']) == {'a', 'b'}
    assert all(set(x['scores']) == {pid, 'outer'} for pid, x in result['inner_comparisons'].items())


def test_declared_points_preserve_all_finalists_and_fixed_error_paths():
    config, protocol = probe.load_protocol(probe.DEFAULT_CONFIG)
    if not (Path(protocol['previous'])/'fit_results.json').is_file():
        pytest.skip('Requires the archived three-subject fit')
    banks, support = probe.fixed_banks(protocol, config)
    assert {sid: len(rows) for sid, rows in banks.items()} == {122: 22, 222: 18, 315: 15}
    assert sum(len(rows)*32 for rows in banks.values()) == 1760
    assert config['parallel_budget'] == 128
    for case in protocol['cases']:
        rows = banks[case['subject']]
        assert len(case['references']) == 8
        assert all('mean_nll' not in row for row in rows)
        anchor = next(r for r in rows if r['id'] == case['nominee'])['hyperparams']
        before = deepcopy(anchor)
        changed = probe.outer_point(anchor, {'eta_plus': .0025})
        assert changed == {**anchor, ETA_PLUS_PATH: .0025}
        assert anchor == before
        fixed = probe.outer_point(anchor, {'E_C': .01, 'eta_plus': .0025}, True)
        values = extract_model_0826_parameters(fixed)
        assert values['E_E'] == extract_model_0826_parameters(anchor)['E_E']
        assert values['E_C'] == .01 and values['eta_plus'] == .0025
        for bad in (0, -1, True, float('nan'), 1.1):
            with pytest.raises(ValueError):
                probe.outer_point(anchor, {'eta_plus': bad})
        with pytest.raises(ValueError):
            probe.outer_point(anchor, {'E_C': .01, 'delta_E': 4.8}, True)


def test_removing_historical_reference_is_rejected():
    config, protocol = probe.load_protocol(probe.DEFAULT_CONFIG)
    if not (Path(protocol['previous'])/'fit_results.json').is_file():
        pytest.skip('Requires the archived three-subject fit')
    altered = deepcopy(protocol)
    altered['cases'][0]['references'].pop()
    with pytest.raises(ValueError, match='entire historical bank'):
        probe.fixed_banks(altered, config)
