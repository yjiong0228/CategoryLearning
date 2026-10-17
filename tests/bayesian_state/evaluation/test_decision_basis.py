"""Validate report-independent state aggregation and joint availability."""
import numpy as np

from src.Bayesian_state.evaluation.oral.decision_basis import DecisionBasisCollector, joint_availability
from src.Bayesian_state.hypothesis_space.spaces.continuous import build_continuous_hypothesis_space
from src.Bayesian_state.model.modules.readout.oral import OralReadout, OralReadoutParameters
from src.Bayesian_state.model.modules.readout.report_space import component_tokens, report_code


def test_joint_rule_mass_does_not_factorize_particle_means():
    space = build_continuous_hypothesis_space(4, 2)
    h = len(space)
    contribution = np.zeros((2, h)); contribution[0, 0] = contribution[1, 1] = 1.
    likelihood = np.zeros((2, h, 2))
    likelihood[0, 0] = [.9, .1]; likelihood[1, 1] = [.1, .9]
    snapshot = dict(weights=np.array([.5, .5]), choice_weights=contribution,
                    rule_choice_probabilities=likelihood, perceived_stimulus=np.full((2, 4), .3),
                    active=contribution, prediction=np.array([[.9, .1], [.1, .9]]),
                    belief=contribution, beta=np.ones((2, h)), search=np.zeros(2), admission_event=np.zeros(2))
    collector = DecisionBasisCollector(space, np.array([1]))
    collector(0, 'predictive', snapshot)
    values = collector.arrays()
    masses = values['choice_component_joint_mass'][0].sum(axis=1) / values['choice_probability'][0]
    np.testing.assert_allclose(masses[:2], [.9, .1], atol=1e-14)
    np.testing.assert_allclose(values['filtered_active'][0, :2], [.9, .1])
    assert values['prior_component_mass'].sum() == 1.


def test_availability_uses_union_within_each_particle():
    # The first two rules co-occur. Summing their marginal availability would
    # count the same particle twice and incorrectly give probability one.
    active = np.array([[[1, 1, 0], [0, 0, 1]]], dtype=np.uint8)
    packed = np.packbits(active, axis=2, bitorder='little')
    weights = np.array([[.5, .5]])
    np.testing.assert_allclose(joint_availability(packed, weights, [[True, True, False]]), [.5])
    np.testing.assert_allclose(joint_availability(packed, weights, [[True, False, True]]), [1.])


def test_disconnected_basis_uses_existing_oral_nearest_component_policy():
    space = build_continuous_hypothesis_space(4, 2)
    rule, category = next((h.index, y) for h in space for y, c in enumerate(h.categories) if len(c.components) > 1)
    n, h = 3, len(space)
    p = np.array([.2, .3, .5])
    contribution = np.zeros((n, h)); contribution[:, rule] = 1.
    likelihood = np.zeros((n, h, 2)); likelihood[:, rule, category] = 1.
    perceived = np.array([[.1, .2, .8, .9], [.9, .8, .2, .1], [.5, .5, .5, .5]])
    snapshot = dict(weights=p, choice_weights=contribution, rule_choice_probabilities=likelihood,
                    perceived_stimulus=perceived, active=contribution, prediction=likelihood[:, rule],
                    belief=contribution, beta=np.ones((n, h)), search=np.zeros(n), admission_event=np.zeros(n))
    collector = DecisionBasisCollector(space, np.array([category + 1]))
    collector(0, 'predictive', snapshot)
    mass = collector.arrays()['choice_component_joint_mass'][0, rule]
    readout = OralReadout(space, OralReadoutParameters(mention_probability=1., other_probability=0.,
                          body_naming_probability=0., body_reference_center=.75, body_reference_width=.1,
                          extra_probability=0.))
    expected = sum(weight * readout.distribution(rule, category + 1, x)[0] for weight, x in zip(p, perceived))
    actual = np.zeros(len(readout.codes))
    for i, component in enumerate(space[rule].categories[category].components):
        code = report_code(component_tokens(component.A, component.b))
        actual[readout.codes.index(code)] += mass[i]
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)
