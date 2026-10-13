"""Behavior alignment preserves task truth, score masks and session boundaries."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.evaluation.model_0923_behavior_alignment import (
    ARTIFACTS, summarize_behavior_alignment,
)
from src.Bayesian_state.metrics.task import category_learning_feedback


def _example(choices, categories, probabilities, *, condition=1, sessions=None, scored=None):
    choices, categories = np.asarray(choices), np.asarray(categories)
    probabilities = np.asarray(probabilities, dtype=float)
    feedback = np.array([category_learning_feedback(y, c, condition=condition)
                         for y, c in zip(choices, categories)])
    n = len(choices)
    frame = pd.DataFrame(dict(iSub=[307] * n, iTrial=np.arange(41, 41 + n),
        choice=choices, category=categories, feedback=feedback,
        scored=[False] + [True] * (n - 1) if scored is None else scored))
    if sessions is not None:
        frame['iSession'] = sessions
    for column in range(probabilities.shape[1]):
        frame[f'probability_choice_{column + 1}'] = probabilities[:, column]
    inputs = dict(subject=307, condition=condition,
                  arrays=SimpleNamespace(choices=choices, categories=categories, feedback=feedback))
    return inputs, frame


def test_correctness_uses_true_category_probability_and_writes_strict_json(tmp_path):
    inputs, frame = _example([1, 2, 1, 2], [1, 1, 1, 1],
                            [[.8, .2], [.8, .2], [.6, .4], [.6, .4]])
    before = frame.copy(deep=True)
    summary = summarize_behavior_alignment(inputs, frame, 2, tmp_path / 'new')
    assert summary['observed_accuracy'] == pytest.approx(1 / 3)
    assert summary['conditional_correct_probability'] == pytest.approx(2 / 3)
    assert summary['trial_discrepancy']['signed_bias_predicted_minus_observed'] == pytest.approx(1 / 3)
    assert summary['trial_discrepancy']['acc_mae'] == pytest.approx(.6)
    assert summary['rolling']['acc_mae'] == pytest.approx(.15)
    assert summary['sequential']['status'] == 'unavailable'
    assert summary['behavioral_predictive_interval']['status'] == 'not_performed'
    assert summary['formal_sequential_residual_tests']['status'] == 'not_performed'
    assert sorted(path.name for path in (tmp_path / 'new').iterdir()) == sorted(ARTIFACTS)
    saved = json.loads((tmp_path / 'new/behavior_alignment.json').read_text(),
                       parse_constant=lambda value: pytest.fail(f'Non-JSON constant {value}'))
    assert saved == summary
    source = pd.read_csv(tmp_path / 'new/behavior_alignment_trials.csv')
    assert source['conditional_correct_probability'].tolist() == [.8, .8, .6, .6]
    curves = pd.read_csv(tmp_path / 'new/behavior_alignment_curves.csv')
    assert curves['valid_trial'].tolist() == [3, 4]
    assert curves['iTrial'].tolist() == [43, 44]
    assert curves['window_start_valid_trial'].tolist() == [2, 3]
    pd.testing.assert_frame_equal(frame, before)


def test_unscored_trial_invalidates_full_rolling_windows_without_compressing_time(tmp_path):
    inputs, frame = _example([1, 1, 2, 1, 2, 1], [1] * 6, [[.7, .3]] * 6,
                            scored=[False, True, False, True, True, True])
    summary = summarize_behavior_alignment(inputs, frame, 2, tmp_path)
    curves = pd.read_csv(tmp_path / 'behavior_alignment_curves.csv')
    assert curves['valid_trial'].tolist() == [3, 4, 5, 6]
    assert curves['scored_window'].tolist() == [False, False, True, True]
    assert curves['observed_rolling_exact_accuracy'].iloc[:2].isna().all()
    assert curves['observed_rolling_exact_accuracy'].iloc[2:].tolist() == [.5, .5]
    assert summary['scored_trials'] == 4
    assert summary['rolling']['n_observations'] == 2


def test_constant_choice_reference_estimates_frequencies_only_on_scored_trials(tmp_path):
    inputs, frame = _example([1, 2, 2, 2], [1] * 4, [[.7, .3]] * 4)
    summary = summarize_behavior_alignment(inputs, frame, 2, tmp_path)
    assert summary['constant_choice_frequency'] == [0., 1.]
    assert summary['constant_choice_frequency_nll'] == 0.
    assert summary['uniform_nll'] == pytest.approx(np.log(2))
    assert summary['mean_choice_nll'] == pytest.approx(-np.log(.3))
    assert summary['choice_brier'] == pytest.approx(.98)
    assert 'same scored trials' in summary['constant_reference_scope']


def test_switch_checks_exclude_initialization_unscored_pairs_and_session_boundaries(tmp_path):
    inputs, frame = _example([1, 2, 2, 1, 1, 2], [1] * 6, [[.8, .2]] * 6,
                            sessions=[1, 1, 1, 2, 2, 2])
    summary = summarize_behavior_alignment(inputs, frame, 2, tmp_path)
    sequential = summary['sequential']
    assert sequential['scored_pairs'] == 3
    assert sequential['switch']['switch_human'] == pytest.approx(1 / 3)
    assert sequential['switch']['switch_model'] == pytest.approx(.4)
    assert sequential['by_previous_feedback']['failure']['n_pairs'] == 1
    assert sequential['by_previous_feedback']['full_success']['n_pairs'] == 2
    source = pd.read_csv(tmp_path / 'behavior_alignment_trials.csv')
    assert source['scored_within_session_pair'].tolist() == [False, False, True, False, True, True]
    assert sequential['history']['status'] == 'insufficient_trials'


def test_condition3_keeps_exact_family_reward_and_three_feedback_strata_separate(tmp_path):
    inputs, frame = _example([1, 2, 3, 1, 2, 3, 1], [1] * 7,
                            [[.4, .3, .2, .1]] * 7, condition=3, sessions=[1] * 7)
    summary = summarize_behavior_alignment(inputs, frame, 2, tmp_path)
    task = summary['condition3']
    assert summary['observed_accuracy'] == pytest.approx(1 / 3)
    assert task['observed_task_metrics']['species_accuracy'] == pytest.approx(1 / 3)
    assert task['observed_task_metrics']['family_accuracy'] == pytest.approx(2 / 3)
    assert task['observed_task_metrics']['mean_reward'] == .5
    assert task['predicted_task_metrics']['species_accuracy'] == pytest.approx(.4)
    assert task['predicted_task_metrics']['family_accuracy'] == pytest.approx(.7)
    assert task['predicted_task_metrics']['mean_reward'] == pytest.approx(.55)
    assert summary['formal_sequential_residual_tests']['status'] == 'unsupported_for_multicategory'
    strata = summary['sequential']['by_previous_feedback']
    assert [strata[key]['n_pairs'] for key in ('full_success', 'partial_success', 'failure')] == [1, 2, 2]
    assert summary['sequential']['switch']['exact_error_includes_partial_success']
    curves = pd.read_csv(tmp_path / 'behavior_alignment_curves.csv')
    assert 'observed_rolling_family_accuracy' in curves
    np.testing.assert_allclose(curves['conditional_rolling_expected_reward'], .55)


def test_history_lags_use_only_scored_trials_within_contiguous_sessions(tmp_path):
    # Twelve usable lag rows are possible in a single 21-trial session, but two
    # sessions of length ten cannot support the shared eight-lag regression.
    choices = [1, 2] * 10
    inputs, frame = _example(choices, [1] * 20, [[.65, .35]] * 20,
                            sessions=[1] * 10 + [2] * 10)
    summary = summarize_behavior_alignment(inputs, frame, 2, tmp_path)
    assert summary['sequential']['history']['status'] == 'insufficient_trials'
    assert summary['sequential']['history']['n_rows'] == 0
    inputs2, frame2 = _example(choices, [1] * 20, [[.65, .35]] * 20,
                              sessions=[1] * 20)
    summary2 = summarize_behavior_alignment(inputs2, frame2, 2, tmp_path / 'one_session')
    assert summary2['sequential']['history']['status'] == 'completed_descriptive'
    assert summary2['sequential']['history']['n_rows'] == 11


def test_short_records_and_undefined_correlations_have_explicit_statuses(tmp_path):
    inputs, frame = _example([1, 1], [1, 1], [[.5, .5]] * 2, condition=1)
    summary = summarize_behavior_alignment(inputs, frame, 4, tmp_path)
    assert summary['rolling']['status'] == 'insufficient_scored_windows'
    assert summary['rolling']['acc_mae'] is None
    assert summary['trial_discrepancy']['acc_corr'] is None
    curves = pd.read_csv(tmp_path / 'behavior_alignment_curves.csv')
    assert curves.empty
    assert 'valid_trial' in curves


@pytest.mark.parametrize('mutation', ['probability', 'choice', 'feedback', 'mask', 'target_probability', 'iSub'])
def test_invalid_or_misaligned_inputs_fail_before_creating_output(tmp_path, mutation):
    inputs, frame = _example([1, 2, 1], [1] * 3, [[.6, .4]] * 3)
    if mutation == 'probability':
        frame.loc[1, 'probability_choice_1'] = .7
    elif mutation == 'mask':
        frame['scored'] = frame['scored'].astype(object)
        frame.loc[1, 'scored'] = 'false'
    elif mutation == 'target_probability':
        frame['predicted_correct_probability'] = [.6, .4, .6]
    else:
        frame.loc[1, mutation] = 1
    with pytest.raises(ValueError):
        summarize_behavior_alignment(inputs, frame, 2, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()


def test_existing_behavior_artifact_is_preserved(tmp_path):
    inputs, frame = _example([1, 2, 1], [1] * 3, [[.6, .4]] * 3)
    saved = tmp_path / 'behavior_alignment_curves.csv'
    saved.write_text('existing research output')
    with pytest.raises(FileExistsError):
        summarize_behavior_alignment(inputs, frame, 2, tmp_path)
    assert saved.read_text() == 'existing research output'
    assert list(tmp_path.iterdir()) == [saved]
