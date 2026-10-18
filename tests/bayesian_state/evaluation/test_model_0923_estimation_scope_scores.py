"""Scope scores preserve common random units and deterministic point identity."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_estimation_scope import paired_difference, validate_generation


def point():
    return dict(metrics={name: 1. for name in ('choice_nll', 'energy', 'crps')},
                jackknife={name: np.array([1., 2., 3.]) for name in ('choice_nll', 'energy', 'crps')},
                filter_seeds=np.array([1, 2, 3]), path_seeds=np.array([4, 5, 6]))


def test_two_scope_labels_for_one_point_have_zero_paired_difference():
    row = point()
    assert all(value == 0 for value in paired_difference(row, deepcopy(row)).values())


def test_paired_uncertainty_uses_differences_not_independent_errors():
    reference, target = point(), point()
    for name in target['metrics']:
        target['metrics'][name] += .5
        target['jackknife'][name] += .5
    result = paired_difference(target, reference)
    assert result['delta_energy'] == .5
    assert result['delta_energy_mc_se'] == 0
    target['path_seeds'] = target['path_seeds'][::-1]
    with pytest.raises(AssertionError):paired_difference(target, reference)


def test_generation_checks_task_feedback_and_absent_rule_contribution():
    arrays = dict(seeds=np.array([4, 5]), categories=np.array([1, 2]), probabilities=np.full((2, 2, 4), .25),
                  choices=np.array([[1, 2], [2, 1]]), feedback=np.array([[1., 1.], [0., 0.]]),
                  correct_probability=np.full((2, 2), .25), present=np.zeros((2, 2)),
                  contribution=np.zeros((2, 2)), beta=np.zeros((2, 2)), active_count=np.ones((2, 2)),
                  search=np.zeros((2, 2)), target_exited=np.zeros((2, 2)), target_capacity_dropped=np.zeros((2, 2)))
    record = dict(condition=2, candidate=dict(hyperparams={'engine.modules.hypo_transitions_mod.kwargs.max_capacity': 2}))
    validate_generation(arrays, record)
    arrays['contribution'][0, 0] = .2
    with pytest.raises(ValueError, match='presence'):validate_generation(arrays, record)
    arrays['contribution'][0, 0] = 0
    arrays['feedback'][0, 0] = 0
    with pytest.raises(AssertionError):validate_generation(arrays, record)
