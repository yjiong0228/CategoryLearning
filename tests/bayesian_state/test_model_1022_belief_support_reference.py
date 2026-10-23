"""Historical baselines must preserve labels, complete draws and probabilities."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_1022_belief_support_reference import validate_arrays


def fixture():
    meta = dict(subject=108, condition=1, trials=4, training_trials=2, test_trials=2)
    observed = dict(choices=np.array([1, 2, 1, 2]), categories=np.array([1, 2, 2, 1]))
    probability = np.tile(np.array([[.6, .4], [.3, .7], [.5, .5], [.8, .2]]), (2, 1, 1))
    prediction = dict(probabilities=probability, seeds=np.array([101, 102]),
        choices=observed['choices'].copy(), categories=observed['categories'].copy(),
        observed_probability=probability[:, np.arange(4), observed['choices']-1],
        correct_probability=probability[:, np.arange(4), observed['categories']-1])
    choices = np.array([[1, 2, 1, 2], [2, 1, 1, 2], [1, 2, 2, 1]])
    autonomous = dict(probabilities=np.full((3, 4, 2), .5), seeds=np.array([201, 202, 203]),
        choices=choices, categories=observed['categories'].copy(),
        feedback=(choices == observed['categories'][None, :]).astype(float))
    return meta, observed, prediction, autonomous


def test_complete_reference_arrays_are_accepted_without_modification():
    data = fixture(); saved = deepcopy(data)
    validate_arrays(*data, filters=2, paths=3)
    for before, after in zip(saved, data):
        for key in before:
            np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize('change', [
    'missing_path', 'duplicate_seed', 'wrong_choices', 'wrong_categories',
    'unnormalized', 'nonfinite', 'wrong_readout', 'wrong_feedback', 'invalid_choice'])
def test_invalid_historical_reference_is_rejected(change):
    meta, observed, prediction, autonomous = fixture()
    if change == 'missing_path':
        autonomous['choices'] = autonomous['choices'][:-1]
    elif change == 'duplicate_seed':
        prediction['seeds'][1] = prediction['seeds'][0]
    elif change == 'wrong_choices':
        prediction['choices'] = prediction['choices'][::-1]
    elif change == 'wrong_categories':
        autonomous['categories'][0] = 2
    elif change == 'unnormalized':
        autonomous['probabilities'][0, 0] = [.6, .6]
    elif change == 'nonfinite':
        prediction['probabilities'][0, 0, 0] = np.nan
    elif change == 'wrong_readout':
        prediction['observed_probability'][0, 0] = .9
    elif change == 'wrong_feedback':
        autonomous['feedback'][0, 0] = 0.
    else:
        autonomous['choices'][0, 0] = 3
    with pytest.raises((ValueError, AssertionError)):
        validate_arrays(meta, observed, prediction, autonomous, filters=2, paths=3)
