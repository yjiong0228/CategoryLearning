"""The v2 contract requires distinct behavior and oral evidence, preserving v1."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.Bayesian_state.evaluation import model_0923_protocol as protocol
from src.Bayesian_state.workflows.analysis import evaluate_model_0923 as workflow

READOUT = {'method': 'rule_content_v1', 'kwargs': {
    'mention_probability': .8, 'extra_probability': .2, 'other_probability': .02,
    'body_naming_probability': .5, 'body_reference_center': .75,
    'body_reference_width': .1, 'extra_selection': 'salience',
}}


def _settings(version=2):
    settings = dict(protocol=f'model0923_subject_evaluation_v{version}', fit_run='fit',
                    output_dir='evaluation', figure_output_dir='figures', subjects=[307],
                    analysis_seed=2026101301, particle_count=16, pf_repeats=4,
                    parallel_budget=1, window_size=4, autonomous_rollouts=2,
                    calibration_particle_count=32, calibration_repeats=2,
                    segment_size=4, late_window=4, target_accuracy=.8,
                    task_criterion={'mode': 'not_configured', 'reason': 'Independent task standard not configured.'})
    if version == 2:
        settings['oral_alignment'] = {'mode': 'not_available', 'reason': 'No oral observations in this dataset.'}
    return settings


def _config(tmp_path, settings):
    path = tmp_path / 'protocol.yaml'
    path.write_text(yaml.safe_dump(settings, sort_keys=False))
    return path


def test_v1_configuration_remains_compatible_and_keeps_its_three_panel_contract(tmp_path):
    settings = _settings(1)
    loaded = protocol.load_protocol(_config(tmp_path, settings))
    assert loaded['protocol'] == protocol.PROTOCOL
    assert 'oral_alignment' not in loaded
    assert protocol.SUBJECT_FIGURES == ('behavior.png', 'representations.png', 'diagnostics.png')
    settings['oral_alignment'] = {'mode': 'not_available', 'reason': 'Must be declared in v2.'}
    with pytest.raises(ValueError, match='configuration keys'):
        protocol.load_protocol(_config(tmp_path, settings))


def test_v2_requires_explicit_oral_availability_and_preserves_reason_in_smoke(tmp_path):
    settings = _settings()
    path = _config(tmp_path, settings)
    loaded = protocol.load_protocol(path, smoke=True)
    assert loaded['oral_alignment'] == settings['oral_alignment']
    assert loaded['particle_count'] == 4 and loaded['pf_repeats'] == 2
    assert loaded['calibration_particle_count'] == 8 and loaded['parallel_budget'] == 1
    del settings['oral_alignment']
    with pytest.raises(ValueError, match='configuration keys'):
        protocol.load_protocol(_config(tmp_path, settings))
    assert not (tmp_path / 'evaluation').exists()


def _reports(tmp_path):
    frame = pd.DataFrame(dict(iSub=[307] * 3, condition=[3] * 3,
        iSession=[1] * 3, iBlock=[1] * 3, iTrial=[1, 2, 3],
        category=[1, 2, 3], choice=[1, 2, 4], feedback=[1., 1., .5], text=['脖子长', '', '尾巴短']))
    for i, name in enumerate(('neck', 'head', 'leg', 'tail'), 1):
        frame[f'feature{i}_name'] = name
        frame[f'feature{i}'] = .1 * i
    path = tmp_path / 'oral.csv'
    frame.to_csv(path, index=False)
    inputs = dict(subject=307, condition=3, frame=frame.copy(),
        arrays=SimpleNamespace(choices=frame.choice.to_numpy(), categories=frame.category.to_numpy(),
            feedback=frame.feedback.to_numpy(), stimulus=frame[[f'feature{i}' for i in range(1, 5)]].to_numpy()))
    return path, inputs


def test_v2_predict_resolves_source_and_freezes_every_expression_setting(tmp_path):
    _reports(tmp_path)
    settings = _settings()
    settings['oral_alignment'] = {'mode': 'predict', 'trials_path': 'oral.csv', 'readout': deepcopy(READOUT)}
    loaded = protocol.load_protocol(_config(tmp_path, settings))
    assert loaded['oral_alignment']['trials_path'] == str(tmp_path / 'oral.csv')
    assert loaded['oral_alignment']['readout'] == READOUT
    assert settings['oral_alignment']['trials_path'] == 'oral.csv'


@pytest.mark.parametrize('corruption', ['duplicate', 'missing', 'choice', 'feedback', 'feature_order'])
def test_v2_preflight_rejects_misaligned_oral_before_output_or_pf(tmp_path, monkeypatch, corruption):
    source, inputs = _reports(tmp_path)
    raw = pd.read_csv(source, keep_default_na=False)
    if corruption == 'duplicate':
        raw = pd.concat([raw, raw.iloc[:1]])
    elif corruption == 'missing':
        raw = raw.iloc[1:]
    elif corruption == 'feature_order':
        raw.loc[0, ['feature1_name', 'feature2_name']] = ['head', 'neck']
    else:
        raw.loc[0, corruption] += 1
    raw.to_csv(source, index=False)
    monkeypatch.setattr(workflow, 'validate_fit_source', lambda *_args: dict(
        subjects=[307], input_sha256={}, source_kind='initial_cd', subject_acceptance={'307': 'unresolved'}))
    monkeypatch.setattr(workflow, 'resolve_subject', lambda *_args: deepcopy(inputs))
    monkeypatch.setattr(workflow, 'run_postfit', lambda *_args, **_kwargs: pytest.fail('Invalid oral data launched PF'))
    settings = _settings()
    settings['oral_alignment'] = {'mode': 'predict', 'trials_path': 'oral.csv', 'readout': READOUT}
    with pytest.raises(ValueError):
        workflow.run(_config(tmp_path, settings), smoke=True)
    assert not (tmp_path / 'evaluation').exists()
    assert not (tmp_path / 'figures').exists()


def _base_bundle(output: Path, figures: Path):
    # These small placeholders exercise artifact retention and inventory only;
    # the CLI smoke validation exercises the actual numerical and figure files.
    cohort = ('subjects.csv', 'parameters.csv', 'restarts.csv', 'finalists.csv',
              'search_summary.csv', 'original_replay_validation.json', 'effective_config.yaml', 'report.html')
    subject = ('summary.csv', 'trials.csv', 'rule_states.csv', 'rule_catalog.csv', 'pooled_states.npz',
               'seed_nll.csv', 'budget_comparison.csv', 'learning.csv', 'learning_segments.csv',
               'task_criterion.json', 'task_metrics.json', 'criterion_blocks.csv', 'criterion_trajectories.csv',
               'evaluation.json', 'autonomous_rollouts.npz', 'original_seed_replay.npz', 'fixed_simulation.yaml')
    paths = [output / name for name in cohort]
    paths += [output / 'subject_307' / name for name in subject]
    paths += [figures / name for name in ('cohort_behavior.png', 'parameter_audit.png')]
    paths += [figures / 'subject_307' / name for name in (
        'behavior.png', 'representations.png', 'diagnostics.png',
        'behavior_plot_source.csv', 'diagnostics_plot_source.csv')]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('controlled artifact inventory fixture')


def test_v2_completion_requires_both_alignment_panels_and_their_independent_source_tables(tmp_path):
    output, figures = tmp_path / 'report', tmp_path / 'figures'
    _base_bundle(output, figures)
    v1 = protocol.artifact_manifest(output, figures, [307], protocol=protocol.PROTOCOL)
    assert not any('oral_alignment' in name or 'behavior_alignment' in name for name in v1)
    with pytest.raises(FileNotFoundError, match='behavior_alignment'):
        protocol.artifact_manifest(output, figures, [307], protocol=protocol.PROTOCOL_V2)
    filenames = ('behavior_alignment.json', 'behavior_alignment.csv', 'behavior_alignment_trials.csv',
                 'behavior_alignment_curves.csv', 'oral_alignment.json', 'oral_trials.csv',
                 'oral_summary.csv', 'oral_predictions.npz')
    for name in filenames:
        (output / 'subject_307' / name).write_text('controlled v2 inventory artifact')
    for name in ('behavior_alignment.png', 'oral_alignment.png',
                 'behavior_alignment_plot_source.csv', 'oral_alignment_plot_source.csv'):
        (figures / 'subject_307' / name).write_text('controlled v2 figure/source artifact')
    v2 = protocol.artifact_manifest(output, figures, [307], protocol=protocol.PROTOCOL_V2)
    for name in filenames:
        assert f'report/subject_307/{name}' in v2
    assert 'figures/subject_307/behavior_alignment.png' in v2
    assert 'figures/subject_307/oral_alignment.png' in v2
    assert set(v1).issubset(v2)
    (output / 'subject_307/oral_predictions.npz').unlink()
    with pytest.raises(FileNotFoundError, match='oral_predictions'):
        protocol.artifact_manifest(output, figures, [307], protocol=protocol.PROTOCOL_V2)
