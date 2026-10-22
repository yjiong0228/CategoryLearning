"""Guard search budgets, information boundaries and independent fixed audits."""
from copy import deepcopy
from pathlib import Path
from threading import Event

import pytest
import yaml

from src.Bayesian_state.optimization.search import compact_joint as proposals
from src.Bayesian_state.workflows.analysis import model_1023_estimation_reliability as study


def settings():
    return yaml.safe_load(Path('configs/shared/model1023_estimation_reliability.yaml').read_text())


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_joint_population_budget_support_and_behavior_duplicates(condition):
    original = yaml.safe_load(Path('configs/shared/model1021_simplification.yaml').read_text())
    space = study.driver.model_space(original, 'COMPACT', condition)
    policy = settings()
    starts = proposals.initial_bank(space, [], 64, 71)
    assert starts == proposals.initial_bank(space, [], 64, 71)
    assert {r['hyperparams'][proposals.CAPACITY] for r in starts} == set(space[proposals.CAPACITY])
    seen = {proposals.behavior_key(r['hyperparams']) for r in starts}
    frozen = deepcopy(starts)
    rows = [dict(r, mean_nll=i) for i, r in enumerate(starts)]
    for round_index in range(3):
        new = proposals.population_round(rows, space, seen, seed=20 + round_index, **policy['proposals'])
        assert new == proposals.population_round(rows, space, seen, seed=20 + round_index, **policy['proposals'])
        assert len(new) == 80
        keys = {proposals.behavior_key(r['hyperparams']) for r in new}
        assert len(keys) == 80 and not keys & seen
        for row in new:
            proposals.validate_point(row['hyperparams'], space)
            if row['sources'] == ['multi_coordinate']:
                parent = next(r for r in rows if r['id'] == row['origin']['anchor'])
                assert sum(row['hyperparams'][k] != parent['hyperparams'][k] for k in space) >= 3
        seen |= keys
        rows = [dict(r, mean_nll=i) for i, r in enumerate(new)]
    assert len(seen) == 304 and starts == frozen


def test_only_proven_inactive_failure_decay_is_collapsed():
    c = proposals.CONTROL
    a = {c+'error_gain': 0., c+'global_search_failure_gain': 0., c+'failure_decay': .3}
    b = dict(a, **{c+'failure_decay': .99})
    assert proposals.behavior_key(a) == proposals.behavior_key(b)
    a[c+'error_gain'] = b[c+'error_gain'] = .25
    assert proposals.behavior_key(a) != proposals.behavior_key(b)


def test_independent_recheck_preserves_previous_stage_leader():
    def row(value, nll):
        return dict(proposals.candidate({'x': value}, 'example'), mean_nll=nll)
    first = [row(1, .1), row(2, .3), row(3, .4)]
    later = [row(4, .2), row(5, .4), row(6, .5)]
    result = proposals.stage_archive_shortlist([first, later], first[2], 4, 1)
    assert {first[0]['id'], later[0]['id'], first[2]['id']} <= {r['id'] for r in result}


def test_selection_audit_and_evaluation_seed_banks_are_disjoint():
    policy = settings()
    banks = study.independent_seed_banks(policy, 306)
    assert len(banks['post_freeze_audit']) == 8
    with pytest.raises(ValueError, match='seeds overlap'):
        study.independent_seed_banks(policy, 306, banks['confirmation'][0])


def test_numeric_queue_measures_and_bounds_ready_jobs(tmp_path):
    ready, release = Event(), Event()
    def work():
        ready.set()
        assert release.wait(5)
    with study.NumericQueue(2, tmp_path) as queue:
        jobs = [queue.submit(work) for _ in range(6)]
        assert ready.wait(5)
        release.set()
        for future in jobs:
            future.result()
    record = study.read(tmp_path/'resource_status.json')
    assert record['completed'] == 6 and record['failed'] == 0
    assert 1 <= record['peak'] <= 2 and record['active'] == 0


def test_independent_audit_preserves_choice_when_reference_scores_better(monkeypatch, tmp_path):
    original = tmp_path/'original'
    original.mkdir()
    chosen = dict(proposals.candidate({'x': 1}, 'chosen'), mean_nll=.2)
    reference = dict(proposals.candidate({'x': 2}, 'reference'), mean_nll=.3)
    nominee = dict(case_id='case', subject=306, candidate=chosen,
                   finalists=[chosen, reference], reference_id=reference['id'])
    study.write(original/'nominees.json', dict(status='frozen_before_truth', nominee=nominee,
        source_sha256={}, data='unused'))
    case = dict(output=str(original), case_id='case', subject=306,
                settings=dict(seed=31, data='unused', audit_candidates=2))
    def audit(*args):
        return {(306, 'COMPACT'): [dict(reference, mean_nll=.1), dict(chosen, mean_nll=.4)]}
    monkeypatch.setattr(study.driver.base, 'stage', audit)
    _, result = study.audit_case(case, tmp_path/'audit', True)
    assert result['candidate']['id'] == chosen['id']
    assert result['candidate']['mean_nll'] == .4
    assert result['finalists'][0]['id'] == reference['id']
