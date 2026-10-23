"""Cross-study uncertainty must retain both independent simulation errors."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_1022_belief_fixed_references import (
    independent_difference, validate_comparison)


def pair():
    case = dict(meta=dict(subject=306, condition=3, trials=4, training_trials=2, test_trials=2),
        observed={'choices': np.array([1, 2, 3, 4]), 'categories': np.array([1, 2, 3, 4])},
        prediction={'seeds': [1, 2]}, autonomous={'seeds': [3, 4, 5]})
    current = deepcopy(case)
    current['prediction']['seeds'] = [6, 7]
    current['autonomous']['seeds'] = [8, 9, 10]
    return case, current


def test_independent_errors_add_in_variance_not_by_artificial_pairing():
    difference, error = independent_difference(.2, .03, .1, .04)
    assert difference == pytest.approx(-.1)
    assert error == pytest.approx(.05)


def test_matching_trials_with_disjoint_draws_are_comparable():
    validate_comparison(*pair())


@pytest.mark.parametrize('change', ['subject', 'boundary', 'order', 'seeds'])
def test_cross_study_identity_drift_is_rejected(change):
    reference, current = pair()
    if change == 'subject':
        current['meta']['subject'] = 305
    elif change == 'boundary':
        current['meta']['training_trials'] = 3
    elif change == 'order':
        current['observed']['choices'] = current['observed']['choices'][::-1]
    else:
        current['autonomous']['seeds'] = reference['autonomous']['seeds']
    with pytest.raises((ValueError, AssertionError)):
        validate_comparison(reference, current)
