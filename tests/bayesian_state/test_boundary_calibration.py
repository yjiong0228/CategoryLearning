"""Scientific contracts for boundary probes and frozen audit decisions."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis import calibrate_model_0826_boundaries as pilot
from src.Bayesian_state.optimization.adaptive_boundary import adaptive_support
from src.Bayesian_state.optimization.parameter_space import load_model_parameter_space
from src.Bayesian_state.optimization.model_0826 import extract_model_0826_parameters


@pytest.fixture
def prepared():
    config, protocol = pilot.load_protocol(pilot.DEFAULT_CONFIG)
    if any(not Path(protocol[k]).is_file() for k in ('historical_banks', 'historical_stop')):
        pytest.skip('Boundary integration checks require the archived decision-precision candidate files')
    banks, extensions = pilot.build_banks(protocol)
    original = load_model_parameter_space(config['parameter_space'], expected_model_id='model_0826')
    _, _, support = adaptive_support(original, extensions)
    return config, protocol, banks, support


def test_banks_preserve_controls_and_valid_joint_probes(prepared):
    _, protocol, banks, support = prepared
    assert {s: len(r) for s, r in banks.items()} == {103: 12, 129: 30, 203: 5, 301: 12}
    pilot.validate_banks(banks, support)
    for case in protocol['cases']:
        rows = banks[case['subject']]
        assert set(case['references']) <= {r['id'] for r in rows}
        assert all('mean_nll' not in r for r in rows)
    assert any(len(r['origin'].get('changes', {})) == 4 for r in banks[129])
    for r in banks[103]:
        p = extract_model_0826_parameters(r['hyperparams'])
        assert p['chi'] == 1 and p['E_C'] == p['E_E'] == .1


def test_matched_error_probe_does_not_confuse_ec_with_ee(prepared):
    rows = prepared[2][203]
    baseline = extract_model_0826_parameters(rows[0]['hyperparams'])
    matched = [r for r in rows if 'matched_error_probability' in r['sources']]
    assert len(matched) == 2
    for r in matched:
        p = extract_model_0826_parameters(r['hyperparams'])
        assert p['E_C'] < baseline['E_C']
        assert p['E_E'] == baseline['E_E']
        assert p['delta_E'] > baseline['delta_E']
    with pytest.raises(ValueError, match='Preserving'):
        pilot.change_point(rows[0]['hyperparams'], {'E_C': .01, 'delta_E': .8}, preserve_error=True)


def test_shortlist_keeps_family_coverage_when_one_family_dominates(prepared):
    _, protocol, banks, _ = prepared
    case = next(c for c in protocol['cases'] if c['subject'] == 129)
    rows = [{**r, 'mean_nll': i*.001 + (0 if 'capacity_memory' in r['sources'] else 10)} for i, r in enumerate(banks[129])]
    picked = pilot.select_bank(rows, case)
    assert len(picked) == case['shortlist_size']
    assert set(case['references']) <= {r['id'] for r in picked}
    assert all(any(f['name'] in r['sources'] for r in picked) for f in case['families'])


@pytest.mark.parametrize('first_status', ['unresolved', 'selected_point_inferior', 'acceptable_within_bank'])
def test_audit_never_reselects_and_caps_optional_second_look(prepared, tmp_path, monkeypatch, first_status):
    config, protocol, banks, support = deepcopy(prepared)
    protocol['cases'] = [c for c in protocol['cases'] if c['subject'] == 301]
    banks = {301: banks[301]}
    original = protocol['cases'][0]['anchor']
    calls, diagnostics = [], []
    class FakeScorer:
        def __init__(self, *args):
            pass
        def batch(self, bank, budget, family, name):
            calls.append((family, budget.copy()))
            return {s: [{**r, 'mean_nll': float((r['id'] != original) if family != 'boundary_audit' else (r['id'] == original))}
                         for r in rows] for s, rows in bank.items()}
    def fake_audit(scorer, sid, rows, budget, nominee, baseline, protocol, condition):
        diagnostics.append(nominee)
        return {'budget': budget, 'decision': {'status': first_status}, 'scored_rows': rows}
    monkeypatch.setattr(pilot, 'Scorer', FakeScorer)
    monkeypatch.setattr(pilot, 'audit_result', fake_audit)
    result = pilot.calibrate(config, protocol, {301: {'condition': 3}}, banks, support, tmp_path)
    assert result['subjects']['301']['nominee'] == original
    assert diagnostics == [original]*(2 if first_status == 'unresolved' else 1)
    audit_budgets = [b['filter_seed_count'] for f, b in calls if f == 'boundary_audit']
    assert audit_budgets == ([32, 64] if first_status == 'unresolved' else [32])
    assert json.loads((tmp_path/'condition3/nominations.json').read_text()) == {'301': original}


def test_effective_support_requires_declaration(prepared):
    _, _, banks, support = deepcopy(prepared)
    banks[103][0]['hyperparams'] = pilot.change_point(banks[103][0]['hyperparams'], {'gamma': .999})
    with pytest.raises(ValueError, match='Undeclared'):
        pilot.validate_banks(banks, support)


def test_independent_stages_and_nested_audit_extension(prepared):
    from src.Bayesian_state.optimization.adaptive_runtime import phase_seeds
    base = prepared[0]['base_seed']
    screen = phase_seeds(base, 301, 'boundary_screen', 16)
    select = phase_seeds(base, 301, 'boundary_selection', 32)
    audit = phase_seeds(base, 301, 'boundary_audit', 32)
    extended = phase_seeds(base, 301, 'boundary_audit', 64)
    assert not set(screen) & set(select) and not set(select) & set(audit)
    np.testing.assert_array_equal(audit, extended[:32])
