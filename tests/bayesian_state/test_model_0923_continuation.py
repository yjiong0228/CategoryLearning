"""Bounded continuation preserves nomination, domains and report delivery."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.optimization import model_0923_continuation as continuation
from src.Bayesian_state.optimization.model_0923_adaptive import load_config, parameter_support
from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
from src.Bayesian_state.workflows.runs import run_model_0923_complete_pipeline as pipeline


ROOT = Path(__file__).resolve().parents[2]


def fixture():
    support = yaml.safe_load((ROOT / 'configs/shared/model0923_adaptive_parameter_space.yaml').read_text())
    point = {v['path']: v['anchor'] for v in support['parameters'].values()}
    baseline = support['parameters']['baseline_probability']['path']
    point[baseline] = .005
    first = candidate(point, 'old')
    second_point = deepcopy(point)
    second_point[support['parameters']['max_capacity']['path']] = 8
    second = candidate(second_point, 'old')
    result = dict(subject=335, condition=3, selected=first['id'], candidate_bank=[first, second],
        issues=['boundary_review_required'], status='unresolved', search_trace=[],
        boundary=dict(hits=[dict(candidate=first['id'], parameter='baseline_probability',
                                value=.005, side='lower', kind='artificial_boundary'),
                           dict(candidate=second['id'], parameter='max_capacity',
                                value=8, side='upper', kind='artificial_boundary'),
                           dict(candidate=first['id'], parameter='global_search',
                                value=0, side='lower', kind='mechanism_boundary')]))
    policy = continuation.load_policy(ROOT / 'configs/shared/model0923_continuation.yaml')
    return support, result, policy


def test_near_alternative_edges_are_checked_and_natural_edges_preserved():
    support, result, policy = fixture()
    before = deepcopy(result)
    rows, extensions = continuation.boundary_candidates(result, support, policy)
    assert result == before
    assert len(rows) == 6
    assert extensions == {'baseline_probability': [.001, .0025], 'max_capacity': [10, 12]}
    assert {r['id'] for r in result['candidate_bank']} <= {r['id'] for r in rows}
    assert all('mean_nll' not in r for r in rows)


def test_budget_exceeded_is_rejected_without_silently_dropping_probes():
    support, result, policy = fixture()
    policy['max_new_candidates'] = 1
    with pytest.raises(ValueError, match='fixed candidate budget'):
        continuation.boundary_candidates(result, support, policy)


def test_open_probability_domain_and_integer_capacity_are_enforced():
    config = load_config(ROOT / 'configs/meg/model0923_adaptive_335.yaml')
    for name, values in [('baseline_probability', [0]), ('transfer_scale', [0]), ('max_capacity', [10.5])]:
        config['boundary']['extensions'] = {name: values}
        with pytest.raises(ValueError):
            parameter_support(config)


def test_screening_keeps_original_bank_and_freezes_new_nominee():
    support, result, policy = fixture()
    rows, _ = continuation.boundary_candidates(result, support, policy)
    original_ids = {r['id'] for r in result['candidate_bank']}
    rows = [{**r, 'mean_nll': 1 if r['id'] in original_ids else .5 + i*.01} for i, r in enumerate(rows)]
    selected, bank = continuation.freeze_nomination(rows, original_ids, 2)
    assert original_ids <= {r['id'] for r in bank}
    assert selected not in original_ids and len(bank) == 4


def test_audit_does_not_reselect_and_unresolved_is_a_completed_result(tmp_path, monkeypatch):
    support, result, policy = fixture()
    rows, _ = continuation.boundary_candidates(result, support, policy)
    screen_winner = rows[-1]['id']
    audit_winner = rows[0]['id']
    calls = []

    class Scorer:
        def batch(self, banks, budget, family, name):
            calls.append(family)
            winner = screen_winner if family == continuation.SCREEN_FAMILY else audit_winner
            return {335: [{**r, 'mean_nll': .1 if r['id'] == winner else .2} for r in banks[335]]}

        def arrays(self, *args):
            return dict(mask=np.array([False, True, True]))

    def diagnostic(arrays, selected, *args):
        assert selected == screen_winner
        return dict(status='unresolved', scores={pid: .1 if pid == audit_winner else .2 for pid in arrays})

    monkeypatch.setattr(continuation, 'decision_diagnostics', diagnostic)
    monkeypatch.setattr(continuation.Model0923Policy, 'boundary_report',
                        lambda *args: dict(review_required=False, hits=[]))
    config = dict(base_seed=2)
    outcome = continuation.execute(Scorer(), config, policy, result, rows, support, tmp_path)
    assert calls == [continuation.SCREEN_FAMILY, continuation.AUDIT_FAMILY]
    assert outcome['selected'] == screen_winner
    assert outcome['status'] == 'unresolved'
    assert outcome['issues'] == ['independent_audit_unresolved']
    assert outcome['continuation']['further_automatic_continuations'] == 0


def test_old_seed_collision_is_rejected(tmp_path):
    _, _, policy = fixture()
    config = dict(base_seed=policy['base_seed'], seed_namespace='model0923_adaptive_full_v1')
    seed = continuation.phase_seeds(config['base_seed'], 335, continuation.AUDIT_FAMILY,
                                   policy['audit']['filter_seed_count'], namespace=config['seed_namespace'])[0]
    (tmp_path / 'cache' / '335' / f'R128_seed{seed}').mkdir(parents=True)
    with pytest.raises(ValueError, match='overlaps'):
        continuation.seed_overlap_check(tmp_path, config, 335, policy)


def test_report_resume_checks_separate_figure_and_report_roots(tmp_path):
    report, figures = tmp_path / 'report', tmp_path / 'png'
    report.mkdir()
    figures.mkdir()
    (report / 'report.html').write_text('report')
    (figures / 'figure.png').write_bytes(b'PNG')
    hashes = {'report/report.html': pipeline.digest(report / 'report.html'),
              'figures/figure.png': pipeline.digest(figures / 'figure.png')}
    pipeline.verify_report_artifacts(report, figures, hashes)
    (figures / 'figure.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        pipeline.verify_report_artifacts(report, figures, hashes)
    with pytest.raises(ValueError, match='receipt path'):
        pipeline.verify_report_artifacts(report, figures, {'report/../secret': 'wrong'})


def test_smoke_budgets_remain_small():
    policy = continuation.load_policy(ROOT / 'configs/shared/model0923_continuation.yaml', smoke=True)
    assert policy['parallel_budget'] == 1
    assert policy['screen'] == {'particle_count': 2, 'filter_seed_count': 2}
    assert policy['audit'] == {'particle_count': 4, 'filter_seed_count': 2}


def test_partial_report_retry_retains_old_output_and_has_a_limit(tmp_path):
    report, figures = tmp_path / 'evaluation', tmp_path / 'figures'
    report.mkdir()
    (report / 'unfinished.txt').write_text('keep')
    config = tmp_path / 'evaluation.yaml'
    config.write_text(yaml.safe_dump({'output_dir': str(report), 'figure_output_dir': str(figures)}))
    job = dict(subject=335, evaluation=str(report), figures=str(figures), evaluation_config=str(config))
    retry_report, retry_figures, retry_config = pipeline.evaluation_location(job, tmp_path, 2)
    assert retry_report.name == 'evaluation_retry_2'
    assert retry_figures.name == 'figures_retry_2'
    assert (report / 'unfinished.txt').read_text() == 'keep'
    assert yaml.safe_load(retry_config.read_text())['output_dir'] == str(retry_report)
    retry_report.mkdir()
    with pytest.raises(RuntimeError, match='retry budget exhausted'):
        pipeline.evaluation_location(job, tmp_path, 2)


def test_completed_report_is_reused_instead_of_retried(tmp_path):
    import json
    report = tmp_path / 'evaluation'
    report.mkdir()
    (report / 'manifest.json').write_text(json.dumps({'status': 'completed'}))
    job = dict(subject=335, evaluation=str(report), figures=str(tmp_path / 'figures'),
               evaluation_config=str(tmp_path / 'evaluation.yaml'))
    paths = pipeline.evaluation_location(job, tmp_path, 2)
    assert paths == (report, Path(job['figures']), Path(job['evaluation_config']))


def test_unresolved_fit_automatically_enters_evaluation_and_incomplete_bundle_fails(tmp_path, monkeypatch):
    import json
    source = tmp_path / 'source'
    source.mkdir()
    followup, report, figures = [tmp_path / name for name in ('continuation', 'evaluation', 'figures')]
    evaluation_config = tmp_path / 'evaluation.yaml'
    evaluation_config.write_text('{}')
    job = dict(subject=335, needs_continuation=True, source_fit=str(tmp_path / 'original'),
               continuation=str(followup), evaluation=str(report), figures=str(figures),
               evaluation_config=str(evaluation_config))
    plan = dict(source=str(source), snapshot_sha256={}, input_sha256={}, jobs=[job],
                policy=str(tmp_path / 'policy.yaml'), smoke=True, evaluation_attempts=2)
    (tmp_path / 'pipeline_plan.json').write_text(json.dumps(plan))
    commands = []

    class Process:
        pid = 7

        def __init__(self, command, **kwargs):
            commands.append(command[2])
            if 'model_0923_continuation' in command[2]:
                followup.mkdir()
                (followup / 'fit_results.json').write_text(json.dumps({'subjects': {'335': {
                    'status': 'unresolved', 'issues': ['boundary_review_required'],
                    'selected': 'nominee', 'selected_mean_nll': .5}}}))
            else:
                report.mkdir()
                (report / 'manifest.json').write_text(json.dumps({'status': 'assembling_subject_protocol'}))

        def wait(self):
            return 0

    monkeypatch.setattr(pipeline.subprocess, 'Popen', Process)
    with pytest.raises(RuntimeError, match='completed report bundle'):
        pipeline.run_prepared(tmp_path)
    assert commands == ['src.Bayesian_state.optimization.model_0923_continuation',
                        'CategoryLearning_codes.figures.fig3.model0923_evaluation']
    status = json.loads((tmp_path / 'pipeline_status.json').read_text())
    assert status['status'] == 'failed'
    assert status['jobs'][0]['fit_status'] == 'unresolved'


@pytest.mark.parametrize('fit_only', [False, True])
def test_queue_default_chains_complete_pipeline_and_fit_only_is_explicit(tmp_path, monkeypatch, fit_only):
    from src.Bayesian_state.workflows.runs import run_model_0923_plus_queue as queue
    calls = []
    arguments = ['queue', '--output-dir', str(tmp_path / 'queue')]
    if fit_only:
        arguments.append('--fit-only')
    monkeypatch.setattr(queue.sys, 'argv', arguments)
    monkeypatch.setattr(queue, 'prepare', lambda *args: calls.append('prepare_fit'))
    monkeypatch.setattr(queue, 'run_prepared', lambda *args: calls.append('fit') or {'status': 'completed'})
    monkeypatch.setattr(pipeline, 'prepare', lambda *args: calls.append('prepare_complete'))
    monkeypatch.setattr(pipeline, 'run_prepared', lambda *args: calls.append('complete') or {'status': 'completed'})
    queue.main()
    assert calls == (['prepare_fit', 'fit'] if fit_only else ['prepare_fit', 'fit', 'prepare_complete', 'complete'])
