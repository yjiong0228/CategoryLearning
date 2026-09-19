"""Default-fit contracts: joint escape, bounds, precision, resume and data scope."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.optimization import adaptive_fit as fit
from src.Bayesian_state.optimization import adaptive_runtime as runtime
from src.Bayesian_state.optimization.adaptive_boundary import adaptive_support, boundary_report
from src.Bayesian_state.optimization.parameter_space import load_model_parameter_space
from src.Bayesian_state.optimization.model_0826 import GAMMA_PATH, BETA_PATH
from src.Bayesian_state.optimization.search.adaptive_proposals import candidate, propose_round


@pytest.fixture
def setup():
    config = fit.load_fit_config(fit.DEFAULT_CONFIG, smoke=True)
    original = load_model_parameter_space(config['parameter_space'], expected_model_id='model_0826')
    space, anchor, support = adaptive_support(original, {})
    return config, original, space, anchor, support


def test_joint_proposal_escapes_single_coordinate_trap():
    anchor = {GAMMA_PATH: .8, BETA_PATH: 5.}
    space = {GAMMA_PATH: [.8, .9], BETA_PATH: [5., 10.]}
    row = candidate(anchor, 'initial')
    proposals = propose_round([row], space, {row['id']}, 4, 93, 0, 0)
    def score(p):
        a, b = p[GAMMA_PATH] == .9, p[BETA_PATH] == 10
        return 8 if a and b else 10 + int(a) + 2*int(b)
    assert score({GAMMA_PATH: .9, BETA_PATH: 5.}) > score(anchor)
    assert score({GAMMA_PATH: .8, BETA_PATH: 10.}) > score(anchor)
    assert min(score(r['hyperparams']) for r in proposals) == 8
    assert all(r['origin']['kind'] == 'joint' for r in proposals)


def test_boundary_distinguishes_zero_and_artificial_upper(setup):
    config, original, space, anchor, support = setup
    point = deepcopy(anchor); point[GAMMA_PATH] = .97
    row = {**candidate(point, 'test'), 'mean_nll': .4}
    report = boundary_report([row], support, .005, 4)
    hits = {(h['parameter'], h['kind']) for h in report['hits']}
    assert ('gamma', 'artificial_boundary') in hits
    assert ('c_A', 'mechanism_boundary') in hits
    assert not any(h['parameter'] in ('chi', 'E_E') for h in report['hits'])
    assert report['review_required']


def test_extension_is_explicit_does_not_mutate_frozen_support(setup):
    config, original, _, anchor, _ = setup
    before = deepcopy(original)
    space, _, support = adaptive_support(original, {'gamma': [.985, 1.], 'E_C': [.01], 'delta_E': [4.8]})
    assert original == before
    assert 1. in space[GAMMA_PATH]
    point = deepcopy(anchor); point[GAMMA_PATH] = 1.
    row = {**candidate(point, 'test'), 'mean_nll': .4}
    report = boundary_report([row], support, .005, 4)
    assert any(h['parameter'] == 'gamma' and h['kind'] == 'mechanism_boundary' for h in report['hits'])
    assert not report['review_required']


@pytest.mark.parametrize('extension', [{'gamma': [1.1]}, {'E_C': [0]}, {'c_G': [-.1]},
                                       {'eta_plus': [0]}, {'beta_0': [26]},
                                       {'workspace_execution': [{'M': 1, 'chi': 1}]}, {'unknown': [2]}])
def test_invalid_extensions_rejected(setup, extension):
    with pytest.raises(ValueError):
        adaptive_support(setup[1], extension)


def test_budget_cap_noise_boundary_cannot_pass():
    good = 'acceptable_within_bank'
    assert fit.stop_status(True, False, good, good, False) == ('provisional_stop_within_tested_scope', [])
    cases = [(False, False, good, good, False), (True, True, good, good, False),
             (True, False, 'unresolved', good, False), (True, False, good, 'unresolved', False),
             (True, False, good, good, True)]
    for case in cases:
        status, issues = fit.stop_status(*case)
        assert status == 'unresolved' and issues


def test_independent_families_and_nested_seed_prefix():
    decision = runtime.phase_seeds(20260919, 103, 'cycle_0/decision', 32)
    assert decision[:16] == runtime.phase_seeds(20260919, 103, 'cycle_0/decision', 16)
    audit = runtime.phase_seeds(20260919, 103, 'cycle_0/independent_audit', 32)
    next_audit = runtime.phase_seeds(20260919, 103, 'cycle_1/independent_audit', 32)
    assert not set(decision) & set(audit)
    assert not set(audit) & set(next_audit)


def test_proposal_dedup_preserves_joint_origin(setup):
    point = setup[3]
    row = candidate(point, 'new_search', kind='joint', anchor='previous_point')
    merged = fit.merge_proposals([('challenge', [row]), ('guide', [row])])
    assert len(merged) == 1
    assert merged[0]['origin'] == {'kind': 'joint', 'anchor': 'previous_point'}
    assert merged[0]['sources'] == ['new_search', 'challenge', 'guide']
    assert row['sources'] == ['new_search']


def test_unified_cli_uses_configured_default(monkeypatch, capsys):
    import sys
    from src.Bayesian_state.optimization import cli
    calls = []
    def fake_run(*args, **kwargs):
        calls.append((args, kwargs)); return {'dry': True}
    monkeypatch.setattr(fit, 'run_fit', fake_run)
    monkeypatch.setattr(sys, 'argv', ['cli', '--config', str(fit.DEFAULT_CONFIG), '--subjects', '103', '--dry-run'])
    cli.main()
    assert len(calls) == 1 and calls[0][0][2] == [103]
    assert calls[0][1]['dry_run']
    assert json.loads(capsys.readouterr().out) == {'dry': True}


def test_default_subject_selection_and_condition_mismatch(setup):
    config = setup[0]
    subjects = fit.select_subjects(config, None, None, False)
    assert len(subjects) == 96
    assert {c: sum(s['condition'] == c for s in subjects) for c in (1, 2, 3)} == {1: 32, 2: 32, 3: 32}
    with pytest.raises(ValueError, match='mismatched'):
        fit.select_subjects(config, [301], [1], False)
    with pytest.raises(ValueError, match='exactly one'):
        fit.select_subjects(config, [103, 129], None, True)


def test_config_does_not_ignore_unknown_scientific_options(tmp_path):
    config = yaml.safe_load(fit.DEFAULT_CONFIG.read_text())
    config['score_trial_mask'] = [True]
    path = tmp_path/'bad.yaml'; path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='unknown keys'):
        fit.load_fit_config(path)


def fake_seed(context, point, particles, seed, path):
    p = np.array([[.5, .5], [.8, .2], [.7, .3]])
    data = dict(probabilities=p, observed=np.zeros(3, dtype=int), mask=np.array([False, True, True]),
                particles=particles, seed=seed, point_id=runtime.point_id(point), seconds=0.)
    runtime.publish(path, lambda h: np.savez_compressed(h, **data))
    return {'computed': True}


def test_interrupted_batch_reuses_completed_seeds_and_rejects_changed_plan(tmp_path, setup, monkeypatch):
    config, _, _, anchor, _ = setup
    scorer = runtime.Scorer(tmp_path, config, {103: {'subject': 103}})
    row = candidate(anchor, 'test'); budget = {'particle_count': 2, 'filter_seed_count': 2}
    called = []
    def interrupt(*args):
        called.append(args[3])
        if len(called) == 2:
            raise RuntimeError('simulated interruption')
        return fake_seed(*args)
    monkeypatch.setattr(runtime, 'score_seed', interrupt)
    with pytest.raises(RuntimeError):
        scorer.batch({103: [row]}, budget, 'test', 'interrupted')
    saved = list(tmp_path.rglob('*.npz'))
    assert len(saved) == 1
    first_bytes = saved[0].read_bytes()
    called.clear()
    def resumed(*args):
        called.append(args[3]); return fake_seed(*args)
    monkeypatch.setattr(runtime, 'score_seed', resumed)
    result = scorer.batch({103: [row]}, budget, 'test', 'interrupted')
    assert len(called) == 1 and saved[0].read_bytes() == first_bytes
    assert result[103][0]['mean_nll'] == pytest.approx(-np.log([.8, .7]).mean())
    called.clear()
    assert scorer.batch({103: [row]}, budget, 'test', 'interrupted') == result
    assert not called
    with pytest.raises(ValueError, match='differs'):
        scorer.batch({103: [row]}, budget, 'different_family', 'interrupted')
    saved[0].write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='changed'):
        scorer.batch({103: [row]}, budget, 'test', 'interrupted')


def test_controller_resume_and_no_false_boundary_success(tmp_path, setup, monkeypatch):
    config, _, space, anchor, support = setup
    monkeypatch.setattr(runtime, 'score_seed', fake_seed)
    class Arrays:
        choices = np.zeros(3)
    contexts = {103: {'subject': 103, 'condition': 1, 'arrays': Arrays()}}
    first = fit.fit_subjects(config, contexts, space, anchor, support, tmp_path, True)
    second = fit.fit_subjects(config, contexts, space, anchor, support, tmp_path, True)
    assert first == second
    result = first['subjects']['103']
    assert result['state_precision'] == 'not_checked'
    assert result['parameter_recovery'] == 'not_checked'
    if result['boundary']['review_required']:
        assert 'boundary_review_required' in result['issues']
    assert len(result['near_candidates']) >= 2


def test_inferior_audit_restarts_with_new_family_not_same_audit_reselection(tmp_path, setup, monkeypatch):
    config, _, space, anchor, support = setup
    config['search']['max_cycles'] = 2
    monkeypatch.setattr(runtime, 'score_seed', fake_seed)
    calls = []
    def diagnostics(arrays, selected, *args):
        ids = sorted(arrays)
        winner = next(pid for pid in ids if pid != selected)
        scores = {pid: .3 for pid in ids}; scores[winner] = .1
        status = 'selected_point_inferior' if len(calls) == 1 else 'acceptable_within_bank'
        calls.append((selected, winner, set(next(iter(arrays.values()))['seeds'])))
        return {'selected': selected, 'scores': scores, 'status': status}
    monkeypatch.setattr(fit, 'decision_diagnostics', diagnostics)
    class Arrays:
        choices = np.zeros(3)
    contexts = {103: {'subject': 103, 'condition': 1, 'arrays': Arrays()}}
    result = fit.fit_subjects(config, contexts, space, anchor, support, tmp_path, True)
    first = json.loads((tmp_path/'subjects/103/cycle_0.json').read_text())
    second = result['subjects']['103']
    assert first['selected'] == calls[1][0]  # Failed audit never changes its nominee.
    assert first['independent_audit']['status'] == 'selected_point_inferior'
    assert second['selected'] == calls[1][1]
    assert first['audit_family'] != second['audit_family']
    assert not calls[1][2] & calls[-1][2]


def test_parallel_compact_scoring_matches_serial(tmp_path, setup):
    config, _, _, anchor, _ = setup
    spec = {'subject': 301, 'condition': 3, 'engine': config['conditions']['3']}
    context = fit.make_context(spec, config, True)
    row = candidate(anchor, 'parallel_check')
    budget = {'particle_count': 2, 'filter_seed_count': 2}
    serial = runtime.Scorer(tmp_path/'serial', config, {301: context})
    serial.batch({301: [row]}, budget, 'same_seed_family', 'check')
    parallel_config = deepcopy(config); parallel_config['parallel_budget'] = 2
    parallel = runtime.Scorer(tmp_path/'parallel', parallel_config, {301: context})
    parallel.batch({301: [row]}, budget, 'same_seed_family', 'check')
    left, right = serial.arrays(301, row, budget, 'same_seed_family'), parallel.arrays(301, row, budget, 'same_seed_family')
    for key in left:
        np.testing.assert_array_equal(left[key], right[key])


@pytest.mark.parametrize('subject,condition', [(103, 1), (203, 2), (301, 3)])
def test_compact_scoring_exactly_matches_existing_logged_scorer(tmp_path, setup, subject, condition):
    # Independent reference is the established pilot call (logs=True), not two
    # aliases of the new runtime. The scientific core itself is unchanged.
    from src.Bayesian_state.workflows.analysis.pilot_model_0826_simplified_fit import score_one
    config, _, _, anchor, _ = setup
    spec = {'subject': subject, 'condition': condition, 'engine': config['conditions'][str(condition)]}
    context = fit.make_context(spec, config, True)
    old = score_one(context, anchor, 2, 5143)
    target = tmp_path/'run.npz'
    runtime.score_seed(context, anchor, 2, 5143, target)
    new = runtime.read_seed(target, 2, 5143, runtime.point_id(anchor))
    for key in ('probabilities', 'observed', 'mask'):
        np.testing.assert_array_equal(old[key], new[key])
