"""Check audit definitions where path aggregation can change interpretation."""
import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig3.model0923_readout_review import group_rules, validate_states


def saved_example():
    # Two certain paths using different rules: E[N_eff]=1, N_eff(E[w])=2.
    matrix = np.full((2, 2), .5)
    states = {"predictive_" + field: matrix.copy()
              for field in ("belief", "choice_weights", "dominant", "active")}
    states.update(probabilities=matrix.copy(), predictive_candidate_count=np.ones((2, 4)),
                  predictive_effective_participation=np.ones((2, 4)))
    trials = pd.DataFrame(dict(valid_trial=[1, 2], scored=[False, True], choice=[1, 2],
                               category=[2, 1], probability_choice_1=[.5, .5],
                               probability_choice_2=[.5, .5], observed_choice_probability=[.5, .5],
                               predicted_correct_probability=[.5, .5]))
    return states, trials


def test_grouping_preserves_undisplayed_mass_and_column_order():
    matrix = np.array([[.1, .2, .3, .4], [.4, .3, .2, .1]])
    actual = group_rules(matrix, [2, 0])
    np.testing.assert_allclose(actual, [[.3, .1, .6], [.2, .4, .4]])
    np.testing.assert_allclose(actual.sum(axis=1), matrix.sum(axis=1))
    with pytest.raises(ValueError, match="unique"):
        group_rules(matrix, [0, 0])


def test_pathwise_participation_is_preserved_despite_marginal_dispersion():
    states, trials = saved_example()
    validate_states(states, trials, inertia=0, capacity=2, rule_count=2)
    assert np.all(states["predictive_effective_participation"][:, 0] == 1)
    assert np.all(1 / (states["predictive_choice_weights"]**2).sum(axis=1) == 2)


def test_zero_inertia_identity_is_checked():
    states, trials = saved_example()
    states["predictive_active"][:] = 1
    states["predictive_candidate_count"][:] = 2
    states["predictive_choice_weights"][:] = [.4, .6]
    with pytest.raises(AssertionError):
        validate_states(states, trials, inertia=0, capacity=2, rule_count=2)


@pytest.mark.parametrize("corruption", ["trial_order", "score_mask", "choice_probability", "nonfinite"])
def test_rejects_misaligned_or_invalid_saved_values(corruption):
    states, trials = saved_example()
    if corruption == "trial_order":
        trials.valid_trial = [2, 1]
    elif corruption == "score_mask":
        trials.scored = [True, True]
    elif corruption == "choice_probability":
        trials.loc[1, "observed_choice_probability"] = .8
    else:
        states["predictive_belief"][0, 0] = np.nan
    with pytest.raises((ValueError, AssertionError)):
        validate_states(states, trials, inertia=0, capacity=2, rule_count=2)
