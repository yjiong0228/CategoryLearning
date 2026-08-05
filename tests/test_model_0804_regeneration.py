from __future__ import annotations

import numpy as np

import src.Bayesian_state.utils.model_0804 as model0804
from src.Bayesian_state.utils.model_0804 import (
    Model0804Parameters,
    enumerate_initial_states,
    enumerate_transition_outcomes,
    run_model0804_exact,
    run_model0804_particle_filter,
)
from tests.test_model_0804 import _fixture


def test_fa2r_rho_zero_is_exactly_fa2_under_common_randomness() -> None:
    q_values, choices, feedback, prior, kernels = _fixture()
    parameters = Model0804Parameters(
        0.70, 0.40, 2.0, 0.15, 0.35, lapse=0.02, rho=0.0
    )
    common = dict(
        q_values=q_values,
        choices=choices,
        feedback=feedback,
        prior=prior,
        kernels=kernels,
        parameters=parameters,
        capacity=2,
        particle_count=128,
        filter_seed=804,
    )
    fa2 = run_model0804_particle_filter(model_id="FA2", **common)
    fa2r = run_model0804_particle_filter(model_id="FA2R", **common)
    np.testing.assert_array_equal(fa2.probabilities, fa2r.probabilities)
    np.testing.assert_array_equal(
        fa2.marginal_active_probability, fa2r.marginal_active_probability
    )
    np.testing.assert_array_equal(
        fa2.predictive_replacement_count, fa2r.predictive_replacement_count
    )
    assert fa2.nll == fa2r.nll


def test_forced_regeneration_coalesces_different_states_exactly() -> None:
    _, _, _, prior, kernels = _fixture()
    left = model0804._initial_state(np.asarray([0, 1]), prior)
    right = model0804._initial_state(np.asarray([3, 4]), prior)
    parameters = Model0804Parameters(
        0.70, 0.40, 2.0, 0.15, 0.35, rho=1.0
    )
    dimension = model0804._transition_uniform_dimension(parameters, 2)
    unit = np.random.default_rng(805).random(dimension)
    left_next, left_summary, left_sync = model0804._sample_transition(
        left, prior, kernels, parameters, 2, unit
    )
    right_next, right_summary, right_sync = model0804._sample_transition(
        right, prior, kernels, parameters, 2, unit
    )
    np.testing.assert_array_equal(left_next.active, right_next.active)
    np.testing.assert_array_equal(left_next.omega, right_next.omega)
    np.testing.assert_array_equal(left_next.fade, right_next.fade)
    np.testing.assert_array_equal(left_next.static, right_next.static)
    assert left_summary.regenerated and right_summary.regenerated
    assert left_summary.replacement_count == right_summary.replacement_count == 2
    assert left_summary.removed_mass == right_summary.removed_mass == 1.0
    assert left_sync == right_sync == 0.0


def test_regeneration_transition_enumeration_has_exact_rho_mass() -> None:
    _, _, _, prior, kernels = _fixture(n_hypotheses=4, n_trials=3)
    state = enumerate_initial_states(prior, capacity=1)[0][0]
    parameters = Model0804Parameters(
        0.70, 0.40, 2.0, 0.25, 0.35, rho=0.20
    )
    outcomes = enumerate_transition_outcomes(
        state, prior, kernels, parameters, capacity=1
    )
    assert np.isclose(sum(item[3] for item in outcomes), 1.0, atol=1e-13)
    reset_mass = sum(
        probability
        for _, summary, _, probability in outcomes
        if summary.regenerated
    )
    assert np.isclose(reset_mass, 0.20, atol=1e-13)
    for regenerated, summary, sync_error, _ in outcomes:
        if not summary.regenerated:
            continue
        active = regenerated.active
        np.testing.assert_allclose(
            regenerated.static[active] - regenerated.fade[active], 0.0
        )
        assert sync_error == 0.0


def test_fa2r_particle_filter_matches_exact_small_space() -> None:
    q_values, choices, feedback, prior, kernels = _fixture(
        n_hypotheses=3, n_trials=3
    )
    parameters = Model0804Parameters(
        0.72, 0.33, 1.5, 0.46, 0.40, lapse=0.02, rho=0.20
    )
    exact = run_model0804_exact(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2R",
        parameters=parameters,
        capacity=1,
    )
    particles = run_model0804_particle_filter(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2R",
        parameters=parameters,
        capacity=1,
        particle_count=8_192,
        filter_seed=83,
        resample_threshold_fraction=0.01,
    )
    np.testing.assert_allclose(
        particles.probabilities, exact.probabilities, atol=0.025, rtol=0.0
    )
    assert abs(particles.nll - exact.nll) < 0.05


def test_fa2r_dense_transition_matches_scalar_random_map() -> None:
    _, _, _, prior, kernels = _fixture()
    capacity = 2
    particle_count = 8
    parameters = Model0804Parameters(
        0.63, 0.37, 1.4, 0.30, 0.45, rho=0.35
    )
    states = model0804._sample_initial_states_qmc(
        prior, capacity, particle_count, 119
    )
    n_hypotheses = len(prior)
    active = np.zeros((particle_count, n_hypotheses), dtype=bool)
    omega = np.zeros((particle_count, n_hypotheses), dtype=float)
    fade = np.full_like(omega, -np.inf)
    static = np.full_like(omega, -np.inf)
    for index, state in enumerate(states):
        active[index, state.active] = True
        omega[index] = state.omega
        fade[index] = state.fade
        static[index] = state.static
    dimension = model0804._transition_uniform_dimension(parameters, capacity)
    unit = np.random.default_rng(209).random((particle_count, dimension))
    ordinary_dimension = 1 + 2 * capacity
    unit[:, ordinary_dimension] = np.asarray(
        [0.10, 0.90, 0.20, 0.80, 0.30, 0.70, 0.40, 0.60]
    )
    dense = model0804._sample_transition_candidates_dense(
        active,
        omega,
        fade,
        static,
        prior,
        kernels,
        parameters,
        capacity,
        1,
        unit,
    )
    for index, state in enumerate(states):
        scalar_state, scalar_summary, scalar_sync = model0804._sample_transition(
            state, prior, kernels, parameters, capacity, unit[index]
        )
        scalar_active = np.zeros(n_hypotheses, dtype=bool)
        scalar_active[scalar_state.active] = True
        np.testing.assert_array_equal(dense[0][index], scalar_active)
        np.testing.assert_allclose(dense[1][index], scalar_state.omega, atol=1e-14)
        np.testing.assert_allclose(dense[2][index], scalar_state.fade, atol=1e-14)
        np.testing.assert_allclose(dense[3][index], scalar_state.static, atol=1e-14)
        assert dense[4][index] == scalar_summary.replacement_count
        assert np.isclose(dense[5][index], scalar_summary.removed_mass)
        assert np.isclose(dense[6][index], scalar_summary.newcomer_distance)
        assert np.isclose(dense[7][index], scalar_sync)


def test_fa2r_current_feedback_does_not_change_own_prediction() -> None:
    q_values, choices, feedback, prior, kernels = _fixture()
    changed = feedback.copy()
    changed[2] = 1.0 - changed[2]
    parameters = Model0804Parameters(
        0.57, 0.24, 1.6, 0.15, 0.35, lapse=0.02, rho=0.05
    )
    common = dict(
        q_values=q_values,
        choices=choices,
        prior=prior,
        kernels=kernels,
        model_id="FA2R",
        parameters=parameters,
        capacity=2,
        particle_count=128,
        filter_seed=47,
    )
    first = run_model0804_particle_filter(feedback=feedback, **common)
    second = run_model0804_particle_filter(feedback=changed, **common)
    np.testing.assert_array_equal(first.probabilities[:3], second.probabilities[:3])
    assert not np.allclose(first.probabilities[3:], second.probabilities[3:])
