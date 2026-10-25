"""Training-only alternative selection and rare-state/content probability scores."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
from src.Bayesian_state.workflows.analysis.model_1024_retention_capacity import DELTA, CAPACITY, profile
from src.Bayesian_state.workflows.analysis.model_1024_state_oral_validation import far_alternative
from src.Bayesian_state.workflows.analysis.model_1024_state_oral_summary import (
    choice_catalogue_mass, error_scores, window_states)
from src.Bayesian_state.evaluation.oral.decision_basis_scoring import label_masses


def test_training_alternative_uses_qualified_rank_distance_and_stable_ties():
    p = dict(retention_values=[0., .3, .6, .8, 1.], capacity_values=[2, 3, 6])
    a = candidate({DELTA: .8, CAPACITY: 6, 'nuisance': 2.}, 'anchor')
    points = profile(a, p['retention_values'], p['capacity_values'])
    rows = [dict(candidate=x['id'], eligible=True, training_nll=1.) for x in points]
    result = far_alternative(a, points, rows, p)
    assert result['hyperparams'][DELTA] == 0 and result['hyperparams'][CAPACITY] == 2
    old = deepcopy(rows)
    next(r for r in rows if r['candidate'] == result['id'])['eligible'] = False
    result = far_alternative(a, points[::-1], rows[::-1], p)
    assert result['hyperparams'][DELTA] == .3 and result['hyperparams'][CAPACITY] == 2
    assert old != rows and a['hyperparams']['nuisance'] == result['hyperparams']['nuisance']
    for r in rows:
        r['eligible'] = r['candidate'] == a['id']
    with pytest.raises(ValueError):
        far_alternative(a, points, rows, p)


def test_choice_only_baseline_equal_rules_unequal_components_and_no_history():
    counts = np.array([[1, 2], [2, 1]])
    choices = np.array([1, 2, 1])
    mass = choice_catalogue_mass(choices, counts, 3)
    np.testing.assert_allclose(mass.sum(axis=2), .5)
    np.testing.assert_array_equal(mass[0], mass[2])
    np.testing.assert_array_equal(mass[0], [[.5, 0., 0.], [.25, .25, 0.]])
    np.testing.assert_array_equal(mass[1], [[.25, .25, 0.], [.5, 0., 0.]])
    labels = np.full((*mass.shape, 2), -1, dtype=int)
    for t in range(3):
        for h in range(2):
            labels[t, h, :counts[h, choices[t]-1]] = h
    scored = label_masses(mass, labels)
    np.testing.assert_allclose(scored, np.tile([[[.5, .5, 0.], [.5, .5, 0.]]], (3, 1, 1)))


@pytest.mark.parametrize('choices,counts,m', [([0], [[1, 1]], 2), ([3], [[1, 1]], 2),
                                           ([1], [[0, 1]], 2), ([1], [[3, 1]], 2)])
def test_invalid_choice_catalogue_coordinates_fail(choices, counts, m):
    with pytest.raises(ValueError):
        choice_catalogue_mass(np.array(choices), np.array(counts), m)


def test_error_source_score_keeps_the_rare_available_unexecuted_class():
    raw = dict(rule=np.array([[1, -1], [0, 1], [0, 1]]),
               belief=np.array([[1., 0.], [.6, .4], [.8, .2]]),
               contribution=np.array([[1., 0.], [0., 1.], [1., 0.]]), executed=np.array([1, 1, 0]))
    state = dict(predictive_belief=np.zeros((3, 2)), target_rules=np.array([0]),
                 choices=np.full(3, 2), categories=np.ones(3), conditional_decision_sources=np.eye(3))
    rows, confusion = error_scores(state, raw)
    rare = next(r for r in rows if r['state'] == 'error_available_other')
    assert rare['positives'] == 1 and rare['hard_true_positive'] == 1 and rare['hard_false_positive'] == 0
    assert rare['brier'] == 0 and rare['hard_recall'] == 1 and rare['true_event_probability'] == 1
    assert sum(r['count'] for r in confusion if r['truth'] == r['inferred']) == 3
    state['conditional_decision_sources'] = np.tile([1., 0., 0.], (3, 1))
    rows, _ = error_scores(state, raw)
    rare = next(r for r in rows if r['state'] == 'error_available_other')
    assert rare['hard_recall'] == 0 and rare['log_loss'] > 9


def test_windows_preserve_seed_and_catalogue_coordinates():
    state = dict(seeds=np.array([9]), target_rules=np.array([42]), choices=np.arange(10))
    selected = window_states(state, 5, 10)
    np.testing.assert_array_equal(selected['seeds'], [9])
    np.testing.assert_array_equal(selected['target_rules'], [42])
    np.testing.assert_array_equal(selected['choices'], np.arange(5, 10))
