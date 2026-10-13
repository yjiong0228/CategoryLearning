"""The v1 assessment keeps subject identity, task semantics and complete bundles."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.Bayesian_state.evaluation import model_0923_protocol as protocol
from src.Bayesian_state.metrics.task import category_learning_feedback
from src.Bayesian_state.workflows.analysis import evaluate_model_0923 as workflow


def _settings() -> dict:
    return {
        'protocol': 'model0923_subject_evaluation_v1',
        'fit_run': 'fit', 'output_dir': 'evaluation', 'figure_output_dir': 'figures',
        'subjects': [335, 307], 'analysis_seed': 2026101301,
        'particle_count': 16, 'pf_repeats': 4, 'parallel_budget': 1,
        'window_size': 4, 'autonomous_rollouts': 2,
        'calibration_particle_count': 32, 'calibration_repeats': 2,
        'segment_size': 4, 'late_window': 4, 'target_accuracy': .8,
        'task_criterion': {
            'mode': 'not_configured', 'reason': 'MEG task criterion requires its own specification.',
        },
    }


def _config(tmp_path: Path, settings: dict | None = None) -> Path:
    path = tmp_path / 'evaluation.yaml'
    path.write_text(yaml.safe_dump(_settings() if settings is None else settings))
    return path


def _source() -> dict:
    return {
        'subjects': [335, 307], 'source_kind': 'adaptive',
        'subject_acceptance': {'335': 'unresolved', '307': 'accepted'},
        'input_sha256': {},
    }


@pytest.mark.parametrize('mutation', ['unknown', 'missing'])
def test_configuration_requires_exact_v1_keys(tmp_path, mutation):
    settings = _settings()
    if mutation == 'unknown':
        settings['calibration_subjects'] = [335]
    else:
        del settings['calibration_repeats']
    with pytest.raises(ValueError, match='configuration keys'):
        protocol.load_protocol(_config(tmp_path, settings))
    assert not (tmp_path / 'evaluation').exists()
    assert not (tmp_path / 'figures').exists()


@pytest.mark.parametrize('patch', [
    {'pf_repeats': 1}, {'calibration_repeats': 1}, {'calibration_repeats': 5},
    {'calibration_particle_count': 16}, {'calibration_particle_count': 8},
    {'pf_repeats': True}, {'parallel_budget': 0}, {'subjects': [335, 335]},
])
def test_configuration_rejects_invalid_paired_budgets_and_subjects(tmp_path, patch):
    settings = _settings()
    settings.update(patch)
    with pytest.raises(ValueError):
        protocol.load_protocol(_config(tmp_path, settings))


def test_smoke_keeps_first_requested_subject_and_valid_paired_budgets(tmp_path):
    path = _config(tmp_path)
    settings = protocol.load_protocol(path, smoke=True)
    assert settings['subjects'] == [335]
    assert settings['analysis_seed'] == 2026101301
    assert settings['particle_count'] == 4
    assert settings['calibration_particle_count'] == 8
    assert settings['pf_repeats'] == settings['calibration_repeats'] == 2
    assert settings['parallel_budget'] == 1
    assert settings['fit_run'] == str(tmp_path / 'fit')
    assert yaml.safe_load(path.read_text())['subjects'] == [335, 307]


@pytest.mark.parametrize('value', [True, None, '0.8', float('nan'), float('inf'), -.1, 1.1])
def test_descriptive_accuracy_target_is_a_finite_probability(tmp_path, value):
    settings = _settings()
    settings['target_accuracy'] = value
    with pytest.raises(ValueError, match='accuracy target'):
        protocol.load_protocol(_config(tmp_path, settings))


@pytest.mark.parametrize('value', [True, None, '0.9', float('nan'), float('inf'), -.1, 1.1])
def test_task_accuracy_threshold_is_a_finite_probability(tmp_path, value):
    settings = _settings()
    settings['task_criterion'] = {
        'mode': 'exp123', 'source_program': 'task.m',
        'block_size': 64, 'accuracy_threshold': value,
    }
    with pytest.raises(ValueError, match='accuracy threshold'):
        protocol.load_protocol(_config(tmp_path, settings))


def _criterion_spec(block_size: int) -> dict:
    return {'mode': 'exp123', 'block_size': block_size, 'accuracy_threshold': .75}


def test_criterion_requires_every_category_exact_and_uses_first_task_trial():
    # First category-1 trial is an error. Ambiguous category-2 trials are errors
    # too, but only the first error enters the experiment's accuracy criterion.
    frame = pd.DataFrame({
        'iSession': [1] * 10, 'iBlock': [7] * 10,
        'category': [1, 1, 2, 2, 3, 3, 4, 4, 2, 2],
        'ambiguous': [0] * 8 + [1, 1],
        'feedback': [0, 1, 1, 1, 1, 1, 1, 1, 0, .5],
    })
    feedback = np.ones((2, 10))
    feedback[:, 8:] = [0, .5]
    feedback[1, 6] = .5  # Family-level partial reward is an exact species error.
    before = frame.copy(deep=True)
    summary, blocks, trajectories = protocol.criterion_tables(
        frame, feedback, np.array([12, 13]), 4, _criterion_spec(10),
    )
    assert summary['status'] == 'completed'
    assert summary['autonomous_any_pass_fraction'] == .5
    assert summary['observed_first_pass_block'] is None
    assert not blocks.iloc[0]['observed_reached']
    assert blocks.iloc[0]['observed_category_1_accuracy'] == .5
    assert blocks.iloc[0]['observed_category_2_accuracy'] == 1
    assert trajectories['reached'].tolist() == [True, False]
    assert trajectories.iloc[0]['first_pass_block'] == 1
    assert pd.isna(trajectories.iloc[1]['first_pass_block'])
    pd.testing.assert_frame_equal(frame, before)


def test_incomplete_block_remains_explicit_and_preserves_recorded_block_order():
    frame = pd.DataFrame({
        'iSession': [2] * 6, 'iBlock': [9] * 4 + [3] * 2,
        'category': [1, 2, 3, 4, 1, 2], 'ambiguous': [0] * 6,
        'feedback': [1] * 6,
    })
    summary, blocks, trajectories = protocol.criterion_tables(
        frame, np.ones((2, 6)), np.array([12, 13]), 4, _criterion_spec(4),
    )
    assert blocks['iBlock'].tolist() == [9, 3]
    assert blocks['complete'].tolist() == [True, False]
    assert pd.isna(blocks.iloc[1]['observed_reached'])
    assert summary['status'] == 'incomplete_blocks'
    assert summary['evaluated_blocks'] == 1
    assert summary['autonomous_any_pass_fraction'] is None
    assert trajectories['first_pass_block'].tolist() == [1, 1]


def test_missing_category_cannot_pass_despite_correct_present_categories():
    frame = pd.DataFrame({
        'iSession': [1] * 4, 'iBlock': [1] * 4, 'category': [1, 1, 2, 3],
        'ambiguous': [0] * 4, 'feedback': [1] * 4,
    })
    summary, blocks, trajectories = protocol.criterion_tables(
        frame, np.ones((2, 4)), np.array([12, 13]), 4, _criterion_spec(4),
    )
    assert summary['autonomous_any_pass_fraction'] == 0
    assert not blocks.iloc[0]['observed_reached']
    assert pd.isna(blocks.iloc[0]['observed_category_4_accuracy'])
    assert not trajectories['reached'].any()


@pytest.mark.parametrize('smoke, expected', [(False, 'not_configured'), (True, 'not_performed_smoke')])
def test_meg_criterion_and_smoke_have_explicit_unperformed_status(smoke, expected):
    spec = _settings()['task_criterion']
    summary, blocks, trajectories = protocol.criterion_tables(
        pd.DataFrame(), np.empty((0, 0)), np.array([], dtype=int), 4, spec, smoke=smoke,
    )
    assert summary['status'] == expected
    if not smoke:
        assert summary['reason'] == spec['reason']
    assert list(blocks.columns) == [
        'block_index', 'iSession', 'iBlock', 'complete',
        'observed_reached', 'autonomous_pass_fraction',
    ]
    assert list(trajectories.columns) == ['rollout', 'seed', 'reached', 'first_pass_block']
    assert blocks.empty and trajectories.empty


def test_missing_task_annotations_are_unavailable_not_successful():
    summary, blocks, trajectories = protocol.criterion_tables(
        pd.DataFrame({'category': [1, 2]}), np.ones((2, 2)), np.array([1, 2]),
        2, _criterion_spec(2),
    )
    assert summary['status'] == 'unavailable'
    assert blocks.empty and trajectories.empty


COHORT_ARTIFACTS = (
    'subjects.csv', 'parameters.csv', 'restarts.csv', 'finalists.csv',
    'search_summary.csv', 'original_replay_validation.json',
    'effective_config.yaml', 'report.html',
)
SUBJECT_ARTIFACTS = (
    'summary.csv', 'trials.csv', 'rule_states.csv', 'rule_catalog.csv',
    'pooled_states.npz', 'seed_nll.csv', 'budget_comparison.csv',
    'learning.csv', 'learning_segments.csv', 'task_metrics.json', 'task_criterion.json',
    'criterion_blocks.csv', 'criterion_trajectories.csv', 'evaluation.json',
    'autonomous_rollouts.npz', 'original_seed_replay.npz', 'fixed_simulation.yaml',
)


def _complete_bundle(output: Path, figures: Path, subject: int = 335) -> None:
    output.mkdir(parents=True)
    figures.mkdir(parents=True)
    directory = output / f'subject_{subject}'
    directory.mkdir()
    figure_directory = figures / f'subject_{subject}'
    figure_directory.mkdir()
    for name in COHORT_ARTIFACTS:
        (output / name).write_bytes(name.encode())
    for name in SUBJECT_ARTIFACTS:
        (directory / name).write_bytes(name.encode())
    for name in ('cohort_behavior.png', 'parameter_audit.png'):
        (figures / name).write_bytes(name.encode())
    for name in ('behavior.png', 'representations.png', 'diagnostics.png',
                 'behavior_plot_source.csv', 'diagnostics_plot_source.csv'):
        (figure_directory / name).write_bytes(name.encode())
    (directory / 'seeds').mkdir()
    (directory / 'seeds' / 'seed_0.npz').write_bytes(b'seed_0')


def test_artifact_inventory_is_portable_and_hashes_saved_content(tmp_path):
    output, figures = tmp_path / 'report', tmp_path / 'figures'
    _complete_bundle(output, figures)
    manifest = protocol.artifact_manifest(output, figures, [335])
    assert manifest['report/report.html'] == hashlib.sha256(b'report.html').hexdigest()
    assert manifest['figures/subject_335/diagnostics.png'] == hashlib.sha256(b'diagnostics.png').hexdigest()
    assert 'report/subject_335/seeds/seed_0.npz' in manifest
    assert all(not Path(name).is_absolute() for name in manifest)


@pytest.mark.parametrize('relative', [
    'report/report.html', 'figures/subject_335/diagnostics.png',
    'figures/subject_335/representations.png', 'figures/cohort_behavior.png',
])
def test_artifact_inventory_rejects_missing_required_report_or_figure(tmp_path, relative):
    output, figures = tmp_path / 'report', tmp_path / 'figures'
    _complete_bundle(output, figures)
    (tmp_path / relative).unlink()
    with pytest.raises(FileNotFoundError, match='Required evaluation artifact'):
        protocol.artifact_manifest(output, figures, [335])


def _mock_source(monkeypatch) -> None:
    monkeypatch.setattr(workflow, 'validate_fit_source', lambda _path: deepcopy(_source()))


def test_dry_run_validates_plan_without_computing_or_writing(tmp_path, monkeypatch):
    _mock_source(monkeypatch)
    monkeypatch.setattr(workflow, 'run_postfit', lambda *_args, **_kwargs: pytest.fail('Dry run computed evaluation'))
    path = _config(tmp_path)
    output = tmp_path / 'new_report'
    plan = workflow.run(path, output, smoke=True, dry_run=True)
    assert plan['status'] == 'planned'
    assert plan['protocol'] == 'model0923_subject_evaluation_v1'
    assert plan['settings']['subjects'] == [335]
    assert plan['fit_source']['subject_acceptance']['335'] == 'unresolved'
    assert plan['output_dir'] == str(output)
    assert not output.exists()
    assert list(tmp_path.iterdir()) == [path]


def test_smoke_fit_cannot_be_promoted_to_full_subject_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, 'validate_fit_source', lambda _path: {
        **deepcopy(_source()), 'smoke_only': True,
    })
    path = _config(tmp_path)
    with pytest.raises(ValueError, match='smoke fit supports smoke evaluation only'):
        workflow.preflight(path)
    assert workflow.preflight(path, smoke=True)['smoke']
    assert list(tmp_path.iterdir()) == [path]


def _renderer(*_args, **_kwargs) -> None:
    pass


@pytest.mark.parametrize('missing', ['plot_subject', 'plot_cohort', 'plot_diagnostics'])
def test_each_standard_renderer_is_required_before_computation(tmp_path, monkeypatch, missing):
    _mock_source(monkeypatch)
    monkeypatch.setattr(workflow, 'run_postfit', lambda *_args, **_kwargs: pytest.fail('Missing renderer reached computation'))
    renderers = dict(plot_subject=_renderer, plot_cohort=_renderer, plot_diagnostics=_renderer)
    renderers[missing] = None
    output = tmp_path / 'new_report'
    with pytest.raises(ValueError, match='all three report renderers'):
        workflow.run(_config(tmp_path), output, smoke=True, **renderers)
    assert not output.exists()


@pytest.mark.parametrize('existing', ['report', 'figures'])
def test_existing_output_is_retained_and_never_computed(tmp_path, monkeypatch, existing):
    _mock_source(monkeypatch)
    monkeypatch.setattr(workflow, 'run_postfit', lambda *_args, **_kwargs: pytest.fail('Existing output reached computation'))
    output, figures = tmp_path / 'report', tmp_path / 'plots'
    saved = output if existing == 'report' else figures
    saved.mkdir()
    sentinel = saved / 'saved_research_result.txt'
    sentinel.write_text('retain this result')
    with pytest.raises(FileExistsError, match='new evaluation and figure output'):
        workflow.run(_config(tmp_path), output, smoke=True, figure_output=figures)
    assert sentinel.read_text() == 'retain this result'
    assert sorted(saved.iterdir()) == [sentinel]


def test_declared_task_threshold_mismatch_fails_before_any_output(tmp_path, monkeypatch):
    _mock_source(monkeypatch)
    program = tmp_path / 'task.m'
    program.write_text('accstop = 0.9;\nnTpB = [32, 64];\n')
    settings = _settings()
    settings['task_criterion'] = {
        'mode': 'exp123', 'source_program': 'task.m',
        'block_size': 32, 'accuracy_threshold': .9,
    }
    monkeypatch.setattr(workflow, 'run_postfit', lambda *_args, **_kwargs: pytest.fail('Invalid task reached computation'))
    output = tmp_path / 'report'
    with pytest.raises(ValueError, match='declared experiment program'):
        workflow.run(_config(tmp_path, settings), output, smoke=True, dry_run=True)
    assert not output.exists()


@pytest.mark.parametrize('source_path', [
    Path('data/meg/behavior.csv'), Path('exp123_analysis/data/meg/behavior.csv'),
])
def test_exp123_criterion_rejects_meg_subject_before_any_output(tmp_path, monkeypatch, source_path):
    _mock_source(monkeypatch)
    program = tmp_path / 'task.m'
    program.write_text('accstop = 0.9;\nnTpB = [32, 64];\n')
    settings = _settings()
    settings['task_criterion'] = {
        'mode': 'exp123', 'source_program': 'task.m',
        'block_size': 64, 'accuracy_threshold': .9,
    }
    monkeypatch.setattr(workflow, 'resolve_subject', lambda *_args: {
        'paths': {'learning_data': source_path},
    })
    monkeypatch.setattr(workflow, 'run_postfit', lambda *_args, **_kwargs: pytest.fail('Task/data mismatch reached computation'))
    output = tmp_path / 'report'
    with pytest.raises(ValueError, match='exp123.*source data'):
        workflow.run(_config(tmp_path, settings), output, smoke=True, dry_run=True)
    assert not output.exists()


def _controlled_postfit(monkeypatch, *, omit_diagnostics: bool = False):
    """Supply tiny saved scientific arrays instead of running PF or optimization."""
    categories = np.array([4, 1, 3, 2, 4, 1, 3, 2])
    choices = np.array([3, 1, 4, 2, 4, 2, 1, 2])
    feedback = np.array([category_learning_feedback(int(y), int(c), condition=3)
                         for y, c in zip(choices, categories)])
    arrays = SimpleNamespace(choices=choices, categories=categories, feedback=feedback)
    frame = pd.DataFrame({'iTrial': np.arange(21, 29), 'category': categories,
                          'choice': choices, 'feedback': feedback})
    traces = []

    def resolve_subject(_fit, sid, _cache, _max_trials=None):
        traces.append(('resolve', sid))
        return dict(arrays=arrays, frame=frame, condition=3,
                    paths={'learning_data': Path('data/meg/behavior.csv')})

    def run_postfit(_config_path, output, smoke, plot_subject, plot_cohort, *,
                    settings_override, figure_output):
        traces.append(('postfit', list(settings_override['subjects'])))
        assert settings_override['calibration_subjects'] == [335]
        assert settings_override['particle_count'] == (4 if smoke else 16)
        assert settings_override['calibration_particle_count'] == (8 if smoke else 32)
        _complete_bundle(output, figure_output)
        directory = output / 'subject_335'
        for name in ('learning.csv', 'learning_segments.csv', 'task_metrics.json', 'task_criterion.json',
                     'criterion_blocks.csv', 'criterion_trajectories.csv', 'evaluation.json'):
            (directory / name).unlink()
        figure_directory = figure_output / 'subject_335'
        (figure_directory / 'diagnostics.png').unlink()
        (figure_directory / 'diagnostics_plot_source.csv').unlink()
        pd.DataFrame({
            'subject': [335], 'condition': [3], 'full_minus_half_repeats_nll': [.03],
            'calibration_high_minus_base_nll': [-.01],
            'calibration_mean_abs_probability_difference': [.02],
            'modeled_trials': [8], 'scored_trials': [7], 'mean_choice_nll': [.85],
            'choice_brier': [.42], 'nll_gain_over_uniform': [np.log(4) - .85],
        }).to_csv(output / 'subjects.csv', index=False)
        trials = frame.assign(predicted_correct_probability=np.linspace(.25, .75, 8),
                              executed_rule_pre_choice=np.arange(10, 18),
                              scored=[False] + [True] * 7)
        probabilities = np.repeat(((1 - trials.predicted_correct_probability.to_numpy()) / 3)[:, None], 4, axis=1)
        probabilities[np.arange(8), categories - 1] = trials.predicted_correct_probability
        for category in range(1, 5):
            trials[f'probability_choice_{category}'] = probabilities[:, category - 1]
        trials.to_csv(directory / 'trials.csv', index=False)
        pd.DataFrame({'seed': [11, 12], 'high_minus_base_nll': [.02, -.04]}).to_csv(
            directory / 'budget_comparison.csv', index=False)
        generated_choices = np.stack([categories, np.array([3, 2, 4, 1, 3, 2, 4, 1])])
        generated_feedback = np.array([
            [category_learning_feedback(int(y), int(c), condition=3)
             for y, c in zip(row, categories)] for row in generated_choices
        ])
        np.savez(directory / 'autonomous_rollouts.npz', choices=generated_choices,
                 feedback=generated_feedback,
                 expected_correct_probability=np.full((2, 8), .6), seeds=np.array([41, 42]))
        (output / 'report.html').write_text('<html><body>Base assessment</body></html>')
        return {'status': 'completed', 'source_fit': 'controlled test fixture'}

    def diagnostics(sid, condition, trials, figure_directory, *, budget_comparison):
        traces.append(('diagnostics', sid))
        assert condition == 3
        np.testing.assert_array_equal(trials['iTrial'], np.arange(21, 29))
        np.testing.assert_array_equal(trials['executed_rule_pre_choice'], np.arange(10, 18))
        np.testing.assert_array_equal(budget_comparison['seed'], [11, 12])
        if not omit_diagnostics:
            (figure_directory / 'diagnostics.png').write_bytes(b'controlled diagnostic')
            budget_comparison.to_csv(figure_directory / 'diagnostics_plot_source.csv', index=False)

    monkeypatch.setattr(workflow, 'resolve_subject', resolve_subject)
    monkeypatch.setattr(workflow, 'run_postfit', run_postfit)
    return arrays, frame, traces, diagnostics


@pytest.mark.parametrize('smoke', [False, True])
def test_fixed_workflow_assembles_same_sections_and_keeps_saved_state_and_task_values(tmp_path, monkeypatch, smoke):
    _mock_source(monkeypatch)
    arrays, frame, traces, diagnostics = _controlled_postfit(monkeypatch)
    before = frame.copy(deep=True)
    output = tmp_path / 'report'
    figures = tmp_path / 'CategoryLearning_codes/figures/outputs/test_protocol'
    monkeypatch.setattr(workflow, '__file__', str(tmp_path / 'src/Bayesian_state/workflows/analysis/evaluate_model_0923.py'))
    settings = _settings()
    settings['subjects'] = [335]
    manifest = workflow.run(_config(tmp_path, settings), output, smoke=smoke, figure_output=figures,
                            plot_subject=_renderer, plot_cohort=_renderer,
                            plot_diagnostics=diagnostics)
    assert manifest['status'] == 'completed'
    assert manifest['settings']['subjects'] == [335]
    evaluation = json.loads((output / 'subject_335/evaluation.json').read_text())
    assert evaluation['subject'] == 335
    assert evaluation['condition'] == 3
    assert evaluation['fit_acceptance'] == 'unresolved'
    assert list(evaluation['steps']) == [
        'input_and_search', 'choice_prediction', 'numerical_stability',
        'internal_states', 'autonomous_learning', 'task_criterion',
    ]
    assert evaluation['steps']['choice_prediction'] == ('smoke_only' if smoke else 'completed_in_sample')
    assert evaluation['task_criterion']['status'] == ('not_performed_smoke' if smoke else 'not_configured')
    assert set(evaluation['extensions'].values()) == {'not_performed'}
    assert evaluation['learning']['overall_scored_trials'] == 7
    assert evaluation['learning']['overall_observed_accuracy'] == pytest.approx(4 / 7)
    task_metrics = evaluation['task_metrics']
    assert task_metrics['status'] == 'completed'
    assert task_metrics['task_metrics_n_trials'] == 7
    assert task_metrics['observed_task_metrics']['species_accuracy'] == pytest.approx(4 / 7)
    assert task_metrics['observed_task_metrics']['family_accuracy'] == pytest.approx(6 / 7)
    assert task_metrics['observed_task_metrics']['mean_reward'] == pytest.approx(5 / 7)
    assert task_metrics['autonomous_task_metrics']['species_accuracy']['q50'] == .5
    assert task_metrics['autonomous_task_metrics']['family_accuracy']['q50'] == 1
    assert task_metrics['autonomous_task_metrics']['mean_reward']['q50'] == .75
    assert json.loads((output / 'subject_335/task_metrics.json').read_text()) == task_metrics
    assert evaluation['numerical']['paired_high_minus_base_nll'] == -.01
    trials = pd.read_csv(output / 'subject_335/trials.csv')
    np.testing.assert_array_equal(trials['feedback'], arrays.feedback)
    np.testing.assert_array_equal(trials['executed_rule_pre_choice'], np.arange(10, 18))
    pd.testing.assert_frame_equal(frame, before)
    assert [row for row in traces if row[0] != 'resolve'] == [('postfit', [335]), ('diagnostics', 335)]
    assert 'figures/subject_335/diagnostics.png' in manifest['artifacts_sha256']
    report = (output / 'report.html').read_text()
    assert '<h1>Model 0923：逐被试评估 v1</h1>' in report
    assert re.findall(r'<h3>([1-5]\. [^<]+)</h3>', report) == [
        '1. 输入与搜索审查', '2. 选择预测', '3. 数值稳定性',
        '4. 内部状态', '5. 自主学习与实验达标',
    ]
    criterion_section = report.split('<h3>5. 自主学习与实验达标</h3>', 1)[1]
    assert '90%' not in criterion_section
    assert '排除模糊刺激' not in report
    assert settings['task_criterion']['reason'] in report
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'completed'


def test_missing_diagnostics_cannot_publish_completed_manifest(tmp_path, monkeypatch):
    _mock_source(monkeypatch)
    _, _, _, diagnostics = _controlled_postfit(monkeypatch, omit_diagnostics=True)
    output = tmp_path / 'report'
    with pytest.raises(FileNotFoundError, match='diagnostics.png'):
        workflow.run(_config(tmp_path), output, smoke=True, figure_output=tmp_path / 'plots',
                     plot_subject=_renderer, plot_cohort=_renderer,
                     plot_diagnostics=diagnostics)
    manifest = json.loads((output / 'manifest.json').read_text())
    assert manifest['status'] == 'assembling_subject_protocol'
    assert (output / 'subject_335/trials.csv').exists()
