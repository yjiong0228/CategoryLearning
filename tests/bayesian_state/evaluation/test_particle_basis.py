"""Preserve within-particle decision information and established marginals."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.decision_basis import DecisionBasisCollector
from src.Bayesian_state.evaluation.oral.particle_basis import with_particle_basis
from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
from src.Bayesian_state.workflows.analysis.model_0923_oral_reason_method_audit import contribution_example


def collect(snapshot, choices=(1,)):
    collector = with_particle_basis(DecisionBasisCollector)(build_continuous_hypothesis_space(4, 2), np.array(choices))
    for trial in range(len(choices)):
        collector(trial, 'predictive', deepcopy(snapshot))
    return collector.arrays()


def test_retains_information_lost_by_marginal_collector():
    old_a, a, _ = contribution_example('uncertain_single_rule')
    old_b, b, _ = contribution_example('simultaneous_mixture')
    new_a, new_b = collect(a), collect(b)
    for key in old_a:
        np.testing.assert_array_equal(old_a[key], old_b[key])
        np.testing.assert_array_equal(new_a[key], old_a[key])
        np.testing.assert_array_equal(new_b[key], old_b[key])
    np.testing.assert_array_equal(new_a['particle_basis_rule_ids'], [[[0, 2], [0, 2]]])
    np.testing.assert_array_equal(new_a['particle_basis_choice_weights'], [[[1., 0.], [0., 1.]]])
    np.testing.assert_array_equal(new_b['particle_basis_choice_weights'], np.full((1, 2, 2), .5))
    assert new_a['particle_basis_pairing_marginal'].shape == (1, 2, 0)


def test_snapshot_buffers_are_read_only_inputs_and_are_copied():
    _, snapshot, _ = contribution_example('simultaneous_mixture')
    snapshot['pairing'] = np.array([[.2, .3, .5], [.1, .4, .5]])
    before = deepcopy(snapshot)
    collector = with_particle_basis(DecisionBasisCollector)(build_continuous_hypothesis_space(4, 2), np.array([1]))
    collector(0, 'predictive', snapshot)
    for key in before:
        np.testing.assert_array_equal(snapshot[key], before[key])
    snapshot['choice_weights'][:] = 0.
    snapshot['perceived_stimulus'][:] = .9
    snapshot['pairing'][:] = 0.
    arrays = collector.arrays()
    np.testing.assert_array_equal(arrays['particle_basis_choice_weights'], np.full((1, 2, 2), .5))
    np.testing.assert_array_equal(arrays['particle_basis_perceived_stimulus'], before['perceived_stimulus'][None])
    np.testing.assert_array_equal(arrays['particle_basis_pairing_marginal'], before['pairing'][None])


def test_changing_active_set_uses_unambiguous_zero_padding():
    _, second, _ = contribution_example('simultaneous_mixture')
    first = deepcopy(second)
    first['active'][:, 2] = 0.
    first['choice_weights'][:, 0] = first['belief'][:, 0] = 1.
    first['choice_weights'][:, 2] = first['belief'][:, 2] = 0.
    space = build_continuous_hypothesis_space(4, 2)
    original = DecisionBasisCollector(space, np.array([1, 1]))
    extended = with_particle_basis(DecisionBasisCollector)(space, np.array([1, 1]))
    for trial, snapshot in enumerate((first, second)):
        original(trial, 'predictive', snapshot)
        extended(trial, 'predictive', snapshot)
        extended(trial, 'filtered', {})
    arrays = extended.arrays()
    for key, value in original.arrays().items():
        np.testing.assert_array_equal(arrays[key], value)
    np.testing.assert_array_equal(arrays['particle_basis_rule_ids'][0], [[0, -1], [0, -1]])
    for field in ('choice_weights', 'category_probabilities', 'active_beta', 'active_belief', 'component_weights'):
        assert np.all(arrays['particle_basis_' + field][0, :, 1] == 0)


def test_disconnected_components_and_ties_preserve_original_geometry():
    space = build_continuous_hypothesis_space(4, 2)
    rule, category = next((rule.index, y) for rule in space for y, region in enumerate(rule.categories)
                          if len(region.components) > 1)
    _, snapshot, _ = contribution_example('simultaneous_mixture')
    n, h = 3, len(space)
    contribution = np.zeros((n, h)); contribution[:, rule] = 1.
    likelihood = np.zeros((n, h, 2)); likelihood[:, rule, category] = 1.
    snapshot = dict(weights=np.array([.2, .3, .5]), choice_weights=contribution,
        rule_choice_probabilities=likelihood, perceived_stimulus=np.array([[.1, .2, .8, .9], [.9, .8, .2, .1], [.5, .5, .5, .5]]),
        active=contribution, prediction=likelihood[:, rule], belief=contribution, beta=np.ones((n, h)),
        search=np.zeros(n), admission_event=np.zeros(n))
    collector = with_particle_basis(DecisionBasisCollector)(space, np.array([category + 1]))
    collector(0, 'predictive', snapshot)
    arrays = collector.arrays()
    rho = arrays['particle_basis_component_weights'][0, :, 0]
    np.testing.assert_allclose(rho.sum(axis=1), 1., rtol=0, atol=1e-14)
    np.testing.assert_allclose(snapshot['weights'] @ rho, arrays['prior_component_mass'][0, rule], rtol=0, atol=1e-14)
    # The central stimulus is equally distant from both disconnected regions.
    np.testing.assert_allclose(rho[-1, :2], [.5, .5], rtol=0, atol=1e-14)


@pytest.mark.parametrize('field', ['beta', 'belief'])
@pytest.mark.parametrize('value', [np.nan, -1.])
def test_invalid_active_state_is_rejected(field, value):
    _, snapshot, _ = contribution_example('simultaneous_mixture')
    snapshot[field][0, 0] = value
    with pytest.raises(ValueError):
        collect(snapshot)


def test_zero_weight_particle_keeps_its_own_valid_decision_snapshot():
    _, snapshot, _ = contribution_example('uncertain_single_rule')
    snapshot['weights'] = np.array([1., 0.])
    arrays = collect(snapshot)
    np.testing.assert_array_equal(arrays['particle_basis_choice_weights'][0, 1], [0., 1.])
    np.testing.assert_array_equal(arrays['particle_basis_prior_weights'], [[1., 0.]])
    np.testing.assert_allclose(arrays['prior_component_mass'][0, [0, 2], 0], [1., 0.])


def test_install_once_and_complete_trajectory_required():
    cls = with_particle_basis(DecisionBasisCollector)
    with pytest.raises(ValueError):
        with_particle_basis(cls)
    instance = cls(build_continuous_hypothesis_space(4, 2), np.array([1]))
    with pytest.raises(ValueError):
        instance.arrays()
