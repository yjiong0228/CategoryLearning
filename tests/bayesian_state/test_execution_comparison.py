"""Data boundaries, nested model comparisons and complete-system scoring."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.evaluation.execution_comparison import (
    conditional_score, precision_check, score_difference,
)
from src.Bayesian_state.workflows.analysis.model_1024_execution_comparison import (
    KEYS, Queue, jobs_for, neutral_point, paired_candidates, propose, freeze_shortlist,
)

POLICY = yaml.safe_load(Path('configs/shared/model1024_execution_comparison.yaml').read_text())


def case(group='prefix'):
    parameters = {KEYS[k]: v[len(v)//2] for k, v in POLICY['parameter_values'].items()}
    template = dict(candidate={'hyperparams': parameters}, mechanism={}, folder='unused', runtime='unused')
    own = neutral_point(template, POLICY, 2)
    template['candidate'] = own
    return dict(id='test_'+group, group=group, subject=205, condition=2, trials=64, cut=32,
                template=template, warm=[own])


def test_suffix_uses_prefix_evidence_weights():
    p = np.array([[.9, .9], [.1, .1]])
    # P(prefix,suffix)/P(prefix) = (.81+.01)/(.9+.1) = .82.
    score = conditional_score(p, 1)
    assert score['nll'] == pytest.approx(-np.log(.82))
    assert score['nll'] != pytest.approx(-np.log(.5))
    assert conditional_score(np.full((4, 8), .7), 3)['nll'] == pytest.approx(-np.log(.7))


def test_identical_boundary_reuse_has_exact_zero_difference():
    p = np.array([[.9, .4, .8], [.3, .7, .6], [.6, .5, .7]])
    for prefix in (0, 1, 2):
        result = score_difference(p, p, prefix)
        assert result['difference'] == 0 and result['mcse'] == 0


@pytest.mark.parametrize('prefix', [-1, 2, 0.5])
def test_invalid_evaluation_boundaries_are_rejected(prefix):
    with pytest.raises(ValueError):
        conditional_score(np.full((4, 2), .5), prefix)


def test_numerical_checks_reject_concentrated_systems():
    low = conditional_score(np.full((6, 32), .7))
    high = deepcopy(low)
    assert precision_check(low, high, POLICY['search'])['passed']
    high['maximum_system_weight'] = .95
    assert not precision_check(low, high, POLICY['search'])['passed']


def test_positive_candidates_include_exact_zero_projection():
    c = case()
    points = paired_candidates(c['warm'], c['condition'], [0, 1, 3])
    assert len(points) == 2
    zero, positive = [r['hyperparams'] for r in points]
    assert zero[KEYS['inertia']] == 0 and positive[KEYS['inertia']] > 0
    assert {k: v for k, v in zero.items() if k != KEYS['inertia']} == {
        k: v for k, v in positive.items() if k != KEYS['inertia']}


def test_dispersed_starts_reproducible_and_paired():
    c = case()
    a = propose(POLICY, c, 12, 0)
    assert a == propose(POLICY, c, 12, 0)
    parameters = [r['hyperparams'] for r in a]
    for point in parameters:
        if point[KEYS['inertia']] > 0:
            assert dict(point, **{KEYS['inertia']: 0.}) in parameters


def test_confirm_shortlist_preserves_both_mechanism_boundaries():
    c = case()
    points = paired_candidates(c['warm'], 2, [0, 2])
    rows = [dict(candidate=p, score={'nll': i}) for i, p in enumerate(points)]
    assert {p['hyperparams'][KEYS['inertia']] for p in freeze_shortlist(rows, 1)} == {
        0, c['warm'][0]['hyperparams'][KEYS['inertia']]}


def test_confirm_protects_prior_anchor_from_coarse_ranking_noise():
    rows = [dict(candidate=dict(id=str(i), hyperparams={KEYS['inertia']: 0}), score={'nll': i}) for i in range(5)]
    selected = freeze_shortlist(rows, 3, [rows[-1]['candidate']])
    assert [p['id'] for p in selected] == ['0', '1', '4']


def test_prefix_fit_never_requests_the_sealed_suffix(tmp_path):
    protocol = dict(policy=POLICY, region_sources={})
    c = case('prefix')
    jobs = jobs_for(protocol, c, c['warm'], tmp_path, 'confirm', particles=16, repeats=2)
    assert all(j['phase'] == 'training' and j['expected_trials'] == 32 for j in jobs)
    evaluation = jobs_for(protocol, c, c['warm'], tmp_path, 'evaluation', particles=16, repeats=2, evaluate=True)
    assert all(j['phase'] == 'evaluate' and j['expected_trials'] == 64 for j in evaluation)
    assert {j['seeds'][0] for j in jobs}.isdisjoint(j['seeds'][0] for j in evaluation)


def test_zero_filter_integrates_identity_and_generation_keeps_sampling(tmp_path):
    c = case('full')
    point = paired_candidates(c['warm'], 2, [0, 2])[0]
    protocol = dict(policy=POLICY, region_sources={})
    filtered = jobs_for(protocol, c, [point], tmp_path, 'audit', particles=16, repeats=2)
    generated = jobs_for(protocol, c, [point], tmp_path, 'generation', paths=8)
    assert all(j['execution_mode'] == 'MIX' for j in filtered)
    assert all(j['execution_mode'] == 'STICKY' for j in generated)


def test_recovery_starts_do_not_retain_template_truth_coordinates():
    c = case('recovery')
    altered = deepcopy(c['template'])
    for name in KEYS:
        altered['candidate']['hyperparams'][KEYS[name]] = POLICY['parameter_values'][name][0]
    assert neutral_point(altered, POLICY, 2)['hyperparams'] == neutral_point(c['template'], POLICY, 2)['hyperparams']


def test_expired_wall_budget_leaves_jobs_unstarted(tmp_path):
    import json
    protocol = dict(policy=deepcopy(POLICY), smoke=True)
    protocol['policy']['budget']['wall_hours'] = 0
    queue = Queue(tmp_path, protocol)
    queue.execute([{'name': 'not_launched', 'budget_stage': 'full'}], 'expired')
    assert queue.stop == 'wall_budget'
    assert not (tmp_path/'jobs/not_launched.log').exists()
    receipt = json.loads((tmp_path/'batches/expired.completion.json').read_text())
    assert receipt['unsubmitted'] == ['not_launched'] and receipt['succeeded'] == 0


def test_single_worker_queue_continues_after_each_completed_job(tmp_path, monkeypatch):
    import subprocess
    import sys
    from src.Bayesian_state.workflows.analysis import model_1024_execution_comparison as workflow
    protocol = dict(policy=deepcopy(POLICY), smoke=True)
    protocol['policy']['budget'].update(workers=1, poll_seconds=.01)
    original = subprocess.Popen

    def lightweight_worker(command, **kwargs):
        program = ('import json,sys;from pathlib import Path;'
                   'f=Path(sys.argv[1]);'
                   'f.with_suffix(".usage.json").write_text(json.dumps({"status":"complete"}))')
        return original([sys.executable, '-c', program, command[-1]], **kwargs)

    monkeypatch.setattr(workflow.subprocess, 'Popen', lightweight_worker)
    queue = Queue(tmp_path, protocol)
    queue.execute([dict(name=f'job{i}', budget_stage='full', runtime='unused') for i in range(3)], 'serial')
    assert len(queue.records) == 3
    assert all(r['status'] == 'complete' for r in queue.records.values())
