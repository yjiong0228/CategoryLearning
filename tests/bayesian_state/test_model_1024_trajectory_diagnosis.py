"""Scientific boundaries for the autonomous distribution diagnosis."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.workflows.analysis.model_1024_trajectory_diagnosis import (
    assignments, complete_followup, fit_centers, probability_record,
    process_features, select_candidates)
from src.Bayesian_state.workflows.analysis.model_1023_execution_summary import trajectory_diagnostics


def test_training_selection_preserves_leader_and_diversity_without_suffix():
    rows = [dict(id=str(i), mean_nll=score, hyperparams=dict(x=x, constant=1),
                 suffix_accuracy=99-i) for i, (x, score) in enumerate([(0, 1.), (1, 1.001), (4, 1.015), (3, 1.03)])]
    space = dict(x=[0, 1, 2, 3, 4], constant=[1])
    selected = select_candidates(rows, space, .02, 1)
    assert [r['id'] for r in selected] == ['0', '2']
    changed = deepcopy(rows)
    for r in changed:
        r['suffix_accuracy'] = -r['suffix_accuracy']
    assert [r['id'] for r in select_candidates(changed[::-1], space, .02, 1)] == ['0', '2']


def test_complete_maintenance_uses_only_positions_with_followup():
    categories = np.ones(128, int)
    choices = np.concatenate([np.full(96, 2), np.ones(32, int)])[None, :]
    previous = trajectory_diagnostics(choices, categories)
    complete = complete_followup(choices, categories, 32)
    assert previous['max_adjacent_gain'][0] == 1
    assert not previous['maintenance_observed'][0]
    assert np.isnan(previous['maintenance_accuracy'][0])
    assert complete['complete_center'][0] == 32  # earliest tie, all eligible gains zero
    assert complete['complete_gain'][0] == 0
    assert complete['complete_maintenance'][0] == 0


def test_complete_joint_gain_and_maintenance_remain_on_same_path():
    labels = np.ones(96, int)
    choices = np.array([[2]*32+[1]*64, [2]*32+[1]*32+[2]*32])
    result = complete_followup(choices, labels, 32)
    np.testing.assert_array_equal(result['complete_gain'], [1, 1])
    np.testing.assert_array_equal(result['complete_maintenance'], [1, 0])


@pytest.mark.parametrize('count,total', [(0, 128), (128, 128), (32, 128)])
def test_event_probability_and_monte_carlo_interval(count, total):
    result = probability_record(np.arange(total) < count)
    assert result['probability'] == count/total
    assert result['low'] <= count/total <= result['high']
    assert result['high'] > result['low']


def test_clusters_preserve_all_probability_mass_and_frozen_centers():
    x = np.array([[0, 0], [.01, 0], [1, 1], [.99, 1], [2, 2], [2.01, 2]])
    centers = fit_centers(x, np.array([0, 0, .5, .5, 1, 1]), 3, 42, 5)
    saved = centers.copy()
    labels, _ = assignments(x, centers)
    assert np.bincount(labels, minlength=3).tolist() == [2, 2, 2]
    far, distance = assignments(np.array([[100, 100]]), centers)
    assert far[0] == 2 and distance[0] > 100
    np.testing.assert_array_equal(centers, saved)


def states():
    return dict(rule=np.array([[[2, -1], [1, 2], [1, 2]]]),
        belief=np.array([[[1., 0], [.8, .2], [.8, .2]]]),
        contribution=np.array([[[1., 0], [0, 1.], [1., 0]]]),
        precision=np.full((1, 3, 2), 2.), executed=np.array([[2, 2, 1]]),
        search=np.array([[0, 1, 0]], bool), switch=np.array([[0, 0, 1]], bool),
        forced=np.zeros((1, 3), bool), feedback=np.zeros((1, 3)),
        correct_probability=np.array([[.1, .2, .3]]))


def test_discrete_error_sources_partition_errors_and_keep_timing():
    result = process_features(states(), [1], 'STICKY', 1)
    for name in ('error_absent', 'error_present_other', 'error_correct_executed'):
        assert result[name][0] == 1
    assert result['total_errors'][0] == 3
    assert result['first_target_trial'][0] == 2
    assert result['late_target_present'][0] == 1
    assert result['correct_probability_correct_executed'][0] == .3


def test_absent_target_and_mixture_have_explicit_missing_discrete_state():
    raw = states()
    raw['executed'][:] = -1
    raw['contribution'] = raw['belief'].copy()
    result = process_features(raw, [9], 'MIX', 1)
    assert not result['ever_target'][0]
    assert np.isnan(result['first_target_trial'][0])
    assert np.isnan(result['error_present_other'][0])
    assert result['error_absent'][0] == 3


def test_executed_rule_must_be_active():
    raw = states()
    raw['executed'][0, 0] = 1
    with pytest.raises(ValueError, match='outside the active set'):
        process_features(raw, [1], 'STICKY', 1)
