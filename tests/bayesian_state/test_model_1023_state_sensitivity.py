"""Protect joint-before-conditional pooling and absent-rule precision semantics."""
import numpy as np

from src.Bayesian_state.evaluation.model_1023_state_sensitivity import (
    joint_state_bank, pooled_states, compare_states,
)


def captured(probability, active_rule, precision):
    active = np.zeros((1, 2))
    active[0, active_rule] = 1
    return dict(prior_component_mass=active[:, :, None],
        choice_component_joint_mass=active[:, :, None]*probability,
        choice_probability=np.array([probability]), filtered_active=active,
        filtered_belief=active, filtered_active_beta=active*precision,
        predictive_search=np.array([.3]), predictive_admission=np.array([.1]),
        choices=np.array([1]), component_counts=np.ones((2, 2), int))


def test_pooling_weights_filter_states_by_current_choice_probability():
    bank = joint_state_bank([captured(.9, 0, 2), captured(.1, 1, 8)])
    pooled = pooled_states(bank, 0)
    np.testing.assert_allclose(pooled['basis'], [[.9, .1]])
    np.testing.assert_allclose(pooled['active'], [[.9, .1]])
    assert pooled['trials']['target_precision_given_presence'][0] == 2
    assert pooled['trials']['target_predictive_contribution'][0] == .5


def test_absent_rule_precision_stays_undefined_and_self_difference_is_zero():
    pooled = pooled_states(joint_state_bank([captured(.8, 1, 8), captured(.2, 1, 4)]), 0)
    assert np.isnan(pooled['trials']['target_precision_given_presence'][0])
    comparison = compare_states(pooled, pooled)
    assert np.isnan(comparison['target_precision_delta'][0])
    for key in ('basis_total_variation', 'belief_total_variation', 'target_presence_delta', 'target_contribution_delta'):
        assert comparison[key][0] == 0


def test_total_variation_measures_rule_use_distribution_change():
    a = pooled_states(joint_state_bank([captured(.9, 0, 2), captured(.1, 1, 8)]), 0)
    b = pooled_states(joint_state_bank([captured(.1, 0, 2), captured(.9, 1, 8)]), 0)
    np.testing.assert_allclose(compare_states(a, b)['basis_total_variation'], [.8])
