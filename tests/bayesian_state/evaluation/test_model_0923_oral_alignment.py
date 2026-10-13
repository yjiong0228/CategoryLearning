"""Complete-report evaluation retains original identities and all denominators."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.Bayesian_state.evaluation import model_0923_postfit
from src.Bayesian_state.evaluation.model_0923_oral_alignment import (
    load_observed_reports, summarize_oral_alignment, validate_oral_spec,
)
from src.Bayesian_state.evaluation.oral.structure_0923 import encode_report
from src.Bayesian_state.model.modules.readout.report_space import report_code

ROOT = Path(__file__).resolve().parents[3]
FEATURES = ('neck', 'head', 'leg', 'tail')
READOUT = {'method': 'rule_content_v1', 'kwargs': {
    'mention_probability': .8, 'extra_probability': .2, 'other_probability': .02,
    'body_naming_probability': .5, 'body_reference_center': .75,
    'body_reference_width': .1, 'extra_selection': 'salience',
}}


def _observations(tmp_path: Path):
    frame = pd.DataFrame({'iSub': [101] * 6, 'condition': [1] * 6,
        'iSession': [1] * 6, 'iBlock': [7] * 6, 'iTrial': list(range(1, 7)),
        'choice': [1, 2, 1, 2, 1, 2], 'category': [1, 1, 1, 2, 2, 2],
        'feedback': [1., 0., 1., 1., 0., 1.],
        'text': ['脖子长', '脖子长', '头短', '尾巴长', '', '它们看起来像狗']})
    for index, name in enumerate(FEATURES, 1):
        frame[f'feature{index}_name'] = name
        frame[f'feature{index}'] = .1 * index + np.arange(len(frame)) / 100
    source = tmp_path / 'reports.csv'
    frame.iloc[[4, 1, 3, 2, 5, 0]].to_csv(source, index=False)
    inputs = {'subject': 101, 'condition': 1, 'frame': frame.copy(),
              'arrays': SimpleNamespace(choices=frame.choice.to_numpy(),
                  categories=frame.category.to_numpy(), feedback=frame.feedback.to_numpy(),
                  stimulus=frame[[f'feature{i}' for i in range(1, 5)]].to_numpy())}
    spec = validate_oral_spec({'mode': 'predict', 'trials_path': 'reports.csv', 'readout': READOUT}, tmp_path)
    return inputs, spec, source


def _code(text: str) -> str:
    parsed = encode_report(text, FEATURES)
    assert parsed.status == 'coded'
    return report_code(parsed.tokens)


def _seeds(tmp_path: Path, n_trials=6) -> list[Path]:
    codes = [_code('脖子长'), _code('尾巴长'), 'OTHER']
    paths = []
    for seed, probability in [(31, .1), (72, .3)]:
        path = tmp_path / f'seed_{seed}.npz'
        np.savez_compressed(path, seed=seed, oral_report_codes=np.asarray(codes),
                            oral_report_probabilities=np.tile([probability, 0., 1. - probability], (n_trials, 1)),
                            oral_report_valid_mask=np.ones(n_trials, bool),
                            oral_readout_metadata_json=json.dumps({
                                'method': READOUT['method'], 'parameters': READOUT['kwargs'],
                                'timing': 'after_observed_choice_before_current_feedback'}))
        paths.append(path)
    return paths


def test_source_join_uses_original_trial_keys_and_ignores_legacy_scores(tmp_path):
    inputs, spec, source = _observations(tmp_path)
    raw = pd.read_csv(source, keep_default_na=False)
    raw['report_probability'] = .99  # Prior analysis values are not observations.
    raw['scored'] = True
    raw.to_csv(source, index=False)
    reports = load_observed_reports(inputs, spec)
    assert reports.iTrial.tolist() == [1, 2, 3, 4, 5, 6]
    assert reports.text.tolist() == inputs['frame'].text.tolist()
    assert 'report_probability' not in reports and 'scored' not in reports
    assert reports.attrs['oral_spec']['readout'] == READOUT
    assert len(reports.attrs['source_sha256']) == 64


@pytest.mark.parametrize('corruption', ['duplicate', 'missing_trial', 'choice', 'category', 'feedback',
                                        'condition', 'stimulus', 'feature_order'])
def test_source_requires_one_matching_observation_per_trial(tmp_path, corruption):
    inputs, spec, source = _observations(tmp_path)
    raw = pd.read_csv(source, keep_default_na=False)
    if corruption == 'duplicate':
        raw = pd.concat([raw, raw.iloc[:1]])
    elif corruption == 'missing_trial':
        raw = raw.iloc[1:]
    elif corruption == 'stimulus':
        raw.loc[0, 'feature1'] += .1
    elif corruption == 'feature_order':
        raw.loc[0, ['feature1_name', 'feature2_name']] = ['head', 'neck']
    else:
        raw.loc[0, corruption] += 1
    raw.to_csv(source, index=False)
    with pytest.raises(ValueError):
        load_observed_reports(inputs, spec)


def test_complete_scores_keep_outside_vocabulary_zero_and_review_denominators(tmp_path):
    inputs, spec, _ = _observations(tmp_path)
    reports = load_observed_reports(inputs, spec)
    paths = _seeds(tmp_path)
    before = [path.read_bytes() for path in paths]
    summary = summarize_oral_alignment(inputs, paths, reports, tmp_path / 'evaluation')
    json.dumps(summary, allow_nan=False)
    metrics = summary['metrics']
    assert metrics['n_scored_trials'] == 5
    assert metrics['n_scored_coded_reports'] == 3
    assert metrics['n_scored_missing_reports'] == metrics['n_scored_needs_review_reports'] == 1
    assert metrics['n_vocabulary_reports'] == 2
    assert metrics['n_positive_probability_reports'] == 1
    assert metrics['n_zero_probability_reports'] == 2
    assert metrics['n_out_of_vocabulary_reports'] == metrics['n_in_vocabulary_zero_reports'] == 1
    assert metrics['parser_coverage'] == .6
    assert metrics['vocabulary_coverage'] == 2 / 3
    assert metrics['mean_report_probability'] == pytest.approx(.2 / 3)
    assert metrics['mean_report_nll'] == 'infinity'
    assert metrics['mean_positive_report_nll'] == pytest.approx(-np.log(.2))
    saved = pd.read_csv(tmp_path / 'evaluation/oral_trials.csv')
    assert saved.report_probability.iloc[1] == pytest.approx(.2)
    assert saved.report_probability.iloc[2] == saved.report_probability.iloc[3] == 0
    assert saved.report_probability.iloc[4:].isna().all()
    assert saved.rolling_top3_hit.isna().all()
    assert pd.read_csv(tmp_path / 'evaluation/oral_summary.csv').mean_report_nll.iloc[0] == np.inf
    with np.load(tmp_path / 'evaluation/oral_predictions.npz', allow_pickle=False) as run:
        np.testing.assert_allclose(run['oral_report_probabilities'][:, 0], .2)
        np.testing.assert_array_equal(run['seeds'], [31, 72])
    assert [path.read_bytes() for path in paths] == before
    with pytest.raises(FileExistsError):
        summarize_oral_alignment(inputs, paths, reports, tmp_path / 'evaluation')


def test_trial_windows_require_16_reports_and_missing_current_reports_leave_gaps(tmp_path):
    inputs, spec, _ = _observations(tmp_path)
    reports = load_observed_reports(inputs, spec)
    reports = pd.concat([reports.iloc[[1]]] * 18, ignore_index=True)
    reports['trial'] = np.arange(1, 19)
    reports.loc[16, 'text'] = ''
    reports.attrs['oral_spec'] = spec
    inputs['arrays'].choices = reports.choice.to_numpy()
    summarize_oral_alignment(inputs, _seeds(tmp_path, 18), reports, tmp_path / 'evaluation', smoke=True)
    saved = pd.read_csv(tmp_path / 'evaluation/oral_trials.csv')
    assert saved.rolling_top3_hit.iloc[:17].isna().all()
    assert saved.rolling_top3_hit.iloc[17] == pytest.approx(1.)
    assert saved.rolling_n_coded_reports.iloc[17] == 16


def test_no_coded_reports_has_defined_counts_and_null_prediction_scores(tmp_path):
    inputs, spec, _ = _observations(tmp_path)
    reports = load_observed_reports(inputs, spec)
    reports['text'] = ''
    summary = summarize_oral_alignment(inputs, _seeds(tmp_path), reports, tmp_path / 'evaluation')
    assert summary['metrics']['n_scored_missing_reports'] == 5
    assert summary['metrics']['n_scored_coded_reports'] == 0
    assert summary['metrics']['parser_coverage'] == 0
    assert summary['metrics']['mean_report_nll'] is None
    assert summary['metrics']['top1_hit_rate'] is None
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize('corruption', ['settings', 'vocabulary', 'seed', 'normalization', 'validity'])
def test_seed_pooling_checks_provenance_support_independence_and_probability_invariants(tmp_path, corruption):
    inputs, spec, _ = _observations(tmp_path)
    reports = load_observed_reports(inputs, spec)
    paths = _seeds(tmp_path)
    with np.load(paths[1], allow_pickle=False) as stored:
        arrays = {name: np.array(stored[name]) for name in stored.files}
    if corruption == 'settings':
        receipt = json.loads(str(arrays['oral_readout_metadata_json']))
        receipt['parameters']['extra_probability'] = .3
        arrays['oral_readout_metadata_json'] = np.asarray(json.dumps(receipt))
    elif corruption == 'vocabulary':
        arrays['oral_report_codes'][0] = 'unrelated_report'
    elif corruption == 'seed':
        arrays['seed'] = np.asarray(31)
    elif corruption == 'normalization':
        arrays['oral_report_probabilities'][2] = 0.
    else:
        arrays['oral_report_valid_mask'][2] = False
        arrays['oral_report_probabilities'][2] = np.nan
    np.savez_compressed(paths[1], **arrays)
    with pytest.raises(ValueError):
        summarize_oral_alignment(inputs, paths, reports, tmp_path / 'evaluation')
    assert not (tmp_path / 'evaluation').exists()


def test_unavailable_observations_produce_explicit_fixed_artifacts(tmp_path):
    inputs, _, _ = _observations(tmp_path)
    spec = validate_oral_spec({'mode': 'not_available', 'reason': 'MEG records contain no oral reports.'}, tmp_path)
    assert load_observed_reports(inputs, spec) is None
    summary = summarize_oral_alignment(inputs, [], None, tmp_path / 'evaluation', spec=spec)
    assert summary['status'] == 'not_available' and summary['metrics'] == {}
    assert summary['reason'] == spec['reason']
    assert len(list((tmp_path / 'evaluation').iterdir())) == 3


@pytest.mark.parametrize('corruption', ['missing_parameter', 'unknown_parameter', 'uniform', 'boolean'])
def test_fixed_readout_requires_exact_complete_numerical_settings(tmp_path, corruption):
    _, spec, _ = _observations(tmp_path)
    spec = deepcopy(spec)
    if corruption == 'missing_parameter':
        del spec['readout']['kwargs']['extra_probability']
    elif corruption == 'unknown_parameter':
        spec['readout']['kwargs']['free_parameter'] = .1
    elif corruption == 'uniform':
        spec['readout']['kwargs']['extra_selection'] = 'uniform'
    else:
        spec['readout']['kwargs']['mention_probability'] = True
    with pytest.raises(ValueError):
        validate_oral_spec(spec, tmp_path)


@pytest.mark.parametrize('condition,subject', [(1, 104), (2, 215), (3, 328)])
def test_full_v3_real_trial_prefix_oral_output_preserves_behavior_and_rng(tmp_path, monkeypatch, condition, subject):
    path = ROOT / 'data/exp123/processed/Task2_processed.csv'
    if not path.is_file():
        pytest.skip('Original exp123 processed trial data are unavailable')
    columns = ['iSub', 'condition', 'choice', 'category', 'feedback', *[f'feature{i}' for i in range(1, 5)]]
    frame = pd.read_csv(path, usecols=columns)
    frame = frame.loc[frame.iSub.eq(subject)].iloc[:4].copy()
    assert len(frame) == 4 and frame.condition.eq(condition).all()
    engine = yaml.safe_load((ROOT / f'configs/exp123/model_struct/model_0923_cond{condition}_full_v3.yaml').read_text())
    engine['oral_readout'] = None
    # A short software check uses a smaller geometry sampling budget and a
    # disposable cache. The paired runs share every behavioral setting.
    engine['modules']['hypo_transitions_mod']['kwargs']['relation_samples'] = 512
    engine['provenance']['hypothesis_similarity']['sample_count'] = 512
    engine['partition']['kwargs']['similarity_cache_dir'] = str(tmp_path / 'similarity')
    inputs = dict(subject=subject, condition=condition, engine=engine, frame=frame,
                  arrays=SimpleNamespace(stimulus=frame[[f'feature{i}' for i in range(1, 5)]].to_numpy(),
                      choices=frame.choice.to_numpy(), feedback=frame.feedback.to_numpy()),
                  paths={'processed_dir': path.parent})
    monkeypatch.setattr(model_0923_postfit, 'resolve_subject', lambda *args, **kwargs: deepcopy(inputs))
    task = dict(fit_run=tmp_path / 'unused_fit', subject=subject, cache=tmp_path / 'similarity',
                seed=2026101302, particles=2, record_states=False, output=tmp_path / 'behavior.npz')
    model_0923_postfit.run_particle_task(task)
    reported = dict(task, oral_readout=READOUT, output=tmp_path / 'reported.npz')
    model_0923_postfit.run_particle_task(reported)
    with np.load(task['output'], allow_pickle=False) as before, np.load(reported['output'], allow_pickle=False) as after:
        for key in before.files:
            np.testing.assert_array_equal(before[key], after[key], err_msg=key)
        assert after['oral_report_valid_mask'].all()
        np.testing.assert_allclose(after['oral_report_probabilities'].sum(axis=1), 1., atol=1e-12)
        assert json.loads(str(after['oral_readout_metadata_json']))['parameters'] == READOUT['kwargs']
    assert inputs['engine']['oral_readout'] is None
