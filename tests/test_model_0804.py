from __future__ import annotations

import numpy as np

import src.Bayesian_state.utils.model_0804 as model0804

from src.Bayesian_state.utils.model_0803 import (
    FeatureScaling,
    build_transition_kernels,
    run_model0803,
)
from src.Bayesian_state.utils.model_0804 import (
    Model0804Fit,
    Model0804Parameters,
    combine_model0804_alive_islands,
    decode_parameters,
    enumerate_initial_states,
    enumerate_transition_outcomes,
    enumerate_weighted_wor,
    nested_child_start,
    parameter_definition,
    run_model0804_alive_particle_filter,
    run_model0804_exact,
    run_model0804_particle_filter,
    run_model0804_resample_move_particle_filter,
)


def _fixture(n_hypotheses: int = 5, n_trials: int = 7):
    rng = np.random.default_rng(804)
    raw = rng.uniform(0.05, 1.0, size=(n_trials, n_hypotheses, 2))
    q_values = raw / raw.sum(axis=2, keepdims=True)
    choices = np.asarray([0, 1, 0, 0, 1, 1, 0][:n_trials], dtype=int)
    feedback = np.asarray([1, 0, 1, 0, 0, 1, 1][:n_trials], dtype=float)
    prior = np.arange(1, n_hypotheses + 1, dtype=float)
    prior /= prior.sum()
    coordinates = np.linspace(0.0, 1.0, n_hypotheses)
    distance = np.abs(coordinates[:, None] - coordinates[None, :])
    similarity = 1.0 - distance
    kernels = build_transition_kernels(similarity, prior, tau_local=0.24)
    return q_values, choices, feedback, prior, kernels


def test_weighted_wor_enumeration_is_normalized() -> None:
    outcomes = enumerate_weighted_wor(
        np.asarray([2, 4, 7]),
        np.asarray([0.2, 0.3, 0.5]),
        2,
    )
    assert len(outcomes) == 6
    assert np.isclose(sum(probability for _, probability in outcomes), 1.0)
    assert all(len(order) == 2 and len(set(order)) == 2 for order, _ in outcomes)


def test_mass_conserving_transition_keeps_capacity_and_resets_newcomer_delta() -> None:
    _, _, _, prior, kernels = _fixture()
    state, _ = enumerate_initial_states(prior, capacity=2)[0]
    parameters = Model0804Parameters(
        gamma=0.63,
        w0=0.37,
        kappa=1.4,
        m=1.0,
        g=0.45,
    )
    outcomes = enumerate_transition_outcomes(
        state,
        prior,
        kernels,
        parameters,
        capacity=2,
    )
    assert np.isclose(sum(item[3] for item in outcomes), 1.0)
    for new_state, summary, sync_error, _ in outcomes:
        assert len(new_state.active) == 2
        assert summary.replacement_count == 2
        assert set(new_state.active).isdisjoint(state.active)
        assert np.isclose(new_state.omega.sum(), 1.0)
        np.testing.assert_allclose(
            new_state.static[new_state.active] - new_state.fade[new_state.active],
            0.0,
            atol=1e-12,
        )
        assert sync_error < 1e-12


def test_fa0_fullset_matches_model0803_h0_endpoint() -> None:
    q_values, choices, feedback, prior, kernels = _fixture(
        n_hypotheses=4, n_trials=7
    )
    parameters = Model0804Parameters(
        gamma=0.61,
        w0=0.28,
        kappa=1.7,
        m=0.0,
        g=0.0,
    )
    finite = run_model0804_particle_filter(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA0",
        parameters=parameters,
        capacity=4,
        particle_count=8,
        filter_seed=11,
    )
    full_parameters = np.zeros(11, dtype=float)
    full_parameters[:3] = [parameters.gamma, parameters.w0, parameters.kappa]
    full = run_model0803(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="H0",
        full_parameters=full_parameters,
        feature_scaling=FeatureScaling(np.zeros(2), np.ones(2), "test"),
    )
    np.testing.assert_allclose(
        finite.probabilities, full.probabilities, atol=2e-12, rtol=2e-12
    )
    assert np.isclose(finite.nll, full.nll, atol=2e-12)
    np.testing.assert_allclose(finite.marginal_active_probability, 1.0)


def test_fa0_exact_set_integration_matches_enumeration() -> None:
    q_values, choices, feedback, prior, kernels = _fixture(
        n_hypotheses=5, n_trials=5
    )
    parameters = Model0804Parameters(0.61, 0.28, 1.7, 0.0, 0.0)
    exact = run_model0804_exact(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA0",
        parameters=parameters,
        capacity=2,
    )
    integrated = run_model0804_particle_filter(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA0",
        parameters=parameters,
        capacity=2,
        particle_count=2,
        filter_seed=11,
        fa0_exact_initial_sets=True,
    )
    np.testing.assert_allclose(integrated.probabilities, exact.probabilities, atol=1e-13)
    assert np.isclose(integrated.nll, exact.nll, atol=1e-13)
    assert integrated.particle_count == 10
    assert integrated.integration_mode == "exact_successive_wor_initial_sets"


def test_fa1_is_exactly_fa2_at_global_share_zero_under_crn() -> None:
    q_values, choices, feedback, prior, kernels = _fixture()
    parameters = Model0804Parameters(0.68, 0.31, 1.8, 0.42, 0.0)
    common = dict(
        q_values=q_values,
        choices=choices,
        feedback=feedback,
        prior=prior,
        kernels=kernels,
        parameters=parameters,
        capacity=2,
        particle_count=96,
        filter_seed=31,
        resample_threshold_fraction=0.5,
    )
    fa1 = run_model0804_particle_filter(model_id="FA1", **common)
    fa2 = run_model0804_particle_filter(model_id="FA2", **common)
    np.testing.assert_array_equal(fa1.probabilities, fa2.probabilities)
    np.testing.assert_array_equal(
        fa1.marginal_active_probability, fa2.marginal_active_probability
    )
    np.testing.assert_array_equal(fa1.resampled, fa2.resampled)


def test_current_feedback_does_not_change_its_own_prediction() -> None:
    q_values, choices, feedback, prior, kernels = _fixture()
    changed = feedback.copy()
    changed[2] = 1.0 - changed[2]
    parameters = Model0804Parameters(0.57, 0.24, 1.6, 0.55, 0.35)
    common = dict(
        q_values=q_values,
        choices=choices,
        prior=prior,
        kernels=kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=2,
        particle_count=160,
        filter_seed=47,
    )
    first = run_model0804_particle_filter(feedback=feedback, **common)
    second = run_model0804_particle_filter(feedback=changed, **common)
    np.testing.assert_array_equal(first.probabilities[:3], second.probabilities[:3])
    assert not np.allclose(first.probabilities[3:], second.probabilities[3:])


def test_fixed_choice_lapse_is_uniform_observation_contamination() -> None:
    q_values, choices, feedback, prior, kernels = _fixture()
    base_parameters = Model0804Parameters(0.57, 0.24, 1.6, 0.55, 0.35)
    lapse_parameters = Model0804Parameters(
        0.57, 0.24, 1.6, 0.55, 0.35, lapse=0.02
    )
    common = dict(
        q_values=q_values,
        choices=choices,
        feedback=feedback,
        prior=prior,
        kernels=kernels,
        model_id="FA2",
        capacity=2,
        particle_count=256,
        filter_seed=47,
    )
    base = run_model0804_particle_filter(
        parameters=base_parameters, **common
    )
    contaminated = run_model0804_particle_filter(
        parameters=lapse_parameters, **common
    )
    np.testing.assert_allclose(
        contaminated.probabilities[0],
        0.98 * base.probabilities[0] + 0.01,
        atol=1e-14,
    )
    assert np.min(contaminated.probabilities) >= 0.01


def test_particle_filter_converges_to_exact_small_space_nll() -> None:
    q_values, choices, feedback, prior, kernels = _fixture(
        n_hypotheses=3, n_trials=3
    )
    parameters = Model0804Parameters(0.72, 0.33, 1.5, 0.46, 0.40)
    exact = run_model0804_exact(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
    )
    particles = run_model0804_particle_filter(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
        particle_count=16_000,
        filter_seed=83,
        resample_threshold_fraction=0.01,
    )
    np.testing.assert_allclose(
        particles.probabilities, exact.probabilities, atol=0.018, rtol=0.0
    )
    assert abs(particles.nll - exact.nll) < 0.035
    assert exact.branch_counts[-1] > exact.branch_counts[0]
    np.testing.assert_allclose(
        particles.marginal_active_probability.sum(axis=1), 1.0, atol=1e-12
    )
    expected_nll = -np.log(
        particles.probabilities[np.arange(len(choices)), choices]
    ).sum()
    assert np.isclose(particles.nll, expected_nll)


def test_multiple_transition_proposals_match_exact_small_space_filter() -> None:
    q_values, choices, feedback, prior, kernels = _fixture(
        n_hypotheses=3, n_trials=3
    )
    parameters = Model0804Parameters(0.72, 0.33, 1.5, 0.46, 0.40)
    exact = run_model0804_exact(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
    )
    particles = run_model0804_particle_filter(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
        particle_count=2_048,
        filter_seed=83,
        resample_threshold_fraction=0.01,
        transition_proposals_per_particle=8,
    )
    np.testing.assert_allclose(
        particles.probabilities, exact.probabilities, atol=0.018, rtol=0.0
    )
    assert abs(particles.nll - exact.nll) < 0.035
    assert particles.transition_proposals_per_particle == 8


def test_stratified_replacement_count_integrates_binomial_count_exactly() -> None:
    q_values, choices, feedback, prior, kernels = _fixture()
    parameters = Model0804Parameters(0.72, 0.33, 1.5, 0.40, 0.35)
    particles = run_model0804_particle_filter(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=2,
        particle_count=256,
        filter_seed=83,
        stratify_replacement_count=True,
    )
    np.testing.assert_allclose(
        particles.predictive_replacement_count[1:], 0.8, atol=2e-14
    )
    assert particles.transition_proposals_per_particle == 3
    assert particles.replacement_count_stratified
    assert particles.integration_mode == "particle_qmc_stratified_replacement_count"


def test_stratified_count_filter_matches_exact_small_space_filter() -> None:
    q_values, choices, feedback, prior, kernels = _fixture(
        n_hypotheses=3, n_trials=3
    )
    parameters = Model0804Parameters(0.72, 0.33, 1.5, 0.46, 0.40)
    exact = run_model0804_exact(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
    )
    particles = run_model0804_particle_filter(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
        particle_count=4_096,
        filter_seed=83,
        resample_threshold_fraction=0.01,
        stratify_replacement_count=True,
    )
    np.testing.assert_allclose(
        particles.probabilities, exact.probabilities, atol=0.018, rtol=0.0
    )
    assert abs(particles.nll - exact.nll) < 0.035


def test_dense_transition_batch_matches_scalar_transition_exactly() -> None:
    _, _, _, prior, kernels = _fixture()
    capacity = 2
    particle_count = 8
    proposal_count = 3
    parameters = Model0804Parameters(0.63, 0.37, 1.4, 0.58, 0.45)
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
    unit = np.random.default_rng(209).random(
        (particle_count * proposal_count, 1 + 2 * capacity)
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
        proposal_count,
        unit,
    )
    for parent_index, state in enumerate(states):
        for proposal_index in range(proposal_count):
            row = parent_index * proposal_count + proposal_index
            scalar_state, scalar_summary, scalar_sync = (
                model0804._sample_transition(
                    state,
                    prior,
                    kernels,
                    parameters,
                    capacity,
                    unit[row],
                )
            )
            scalar_active = np.zeros(n_hypotheses, dtype=bool)
            scalar_active[scalar_state.active] = True
            np.testing.assert_array_equal(dense[0][row], scalar_active)
            np.testing.assert_allclose(dense[1][row], scalar_state.omega, atol=1e-14)
            np.testing.assert_allclose(dense[2][row], scalar_state.fade, atol=1e-14)
            np.testing.assert_allclose(dense[3][row], scalar_state.static, atol=1e-14)
            assert dense[4][row] == scalar_summary.replacement_count
            assert np.isclose(dense[5][row], scalar_summary.removed_mass)
            assert np.isclose(dense[6][row], scalar_summary.newcomer_distance)
            assert np.isclose(dense[7][row], scalar_sync)


def test_multiple_proposals_do_not_condition_on_masked_choice() -> None:
    q_values, choices, feedback, prior, kernels = _fixture()
    changed = choices.copy()
    changed[2] = 1 - changed[2]
    changed_feedback = feedback.copy()
    changed_feedback[2] = 1.0 - changed_feedback[2]
    condition = np.ones(len(choices), dtype=bool)
    condition[2] = False
    parameters = Model0804Parameters(0.57, 0.24, 1.6, 0.55, 0.35)
    common = dict(
        q_values=q_values,
        prior=prior,
        kernels=kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=2,
        particle_count=128,
        filter_seed=47,
        transition_proposals_per_particle=4,
        score_mask=np.zeros(len(choices), dtype=bool),
        condition_on_choice_mask=condition,
    )
    first = run_model0804_particle_filter(
        choices=choices, feedback=feedback, **common
    )
    second = run_model0804_particle_filter(
        choices=changed, feedback=changed_feedback, **common
    )
    np.testing.assert_array_equal(first.probabilities, second.probabilities)
    np.testing.assert_array_equal(
        first.marginal_active_probability,
        second.marginal_active_probability,
    )


def test_alive_filter_matches_exact_small_space_and_stopping_identity() -> None:
    q_values, choices, feedback, prior, kernels = _fixture(
        n_hypotheses=3, n_trials=3
    )
    parameters = Model0804Parameters(
        0.72, 0.33, 1.5, 0.46, 0.40, lapse=0.02
    )
    exact = run_model0804_exact(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
    )
    alive = run_model0804_alive_particle_filter(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
        particle_count=8_192,
        filter_seed=83,
        alive_batch_size=1_024,
    )
    np.testing.assert_allclose(
        alive.probabilities, exact.probabilities, atol=0.025, rtol=0.0
    )
    assert abs(alive.nll - exact.nll) < 0.05
    expected_increment = alive.particle_count / (
        alive.alive_attempt_count.astype(float) - 1.0
    )
    np.testing.assert_allclose(
        alive.alive_incremental_likelihood, expected_increment, atol=0.0
    )
    np.testing.assert_allclose(
        alive.probabilities[np.arange(len(choices)), choices],
        expected_increment,
        atol=0.0,
    )
    np.testing.assert_allclose(alive.post_choice_ess, alive.particle_count)
    np.testing.assert_allclose(
        alive.final_weights, 1.0 / float(alive.particle_count)
    )
    assert alive.inference_method == "alive_categorical"


def test_alive_filter_normalizing_constant_is_unbiased_in_small_space() -> None:
    q_values, choices, feedback, prior, kernels = _fixture(
        n_hypotheses=3, n_trials=3
    )
    parameters = Model0804Parameters(
        0.72, 0.33, 1.5, 0.46, 0.40, lapse=0.02
    )
    exact = run_model0804_exact(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
    )
    estimates = []
    for seed in range(200):
        alive = run_model0804_alive_particle_filter(
            q_values,
            choices,
            feedback,
            prior,
            kernels,
            model_id="FA2",
            parameters=parameters,
            capacity=1,
            particle_count=32,
            filter_seed=seed,
            alive_batch_size=32,
        )
        estimates.append(np.exp(-alive.nll))
    estimates = np.asarray(estimates)
    target = float(np.exp(-exact.nll))
    monte_carlo_se = float(estimates.std(ddof=1) / np.sqrt(estimates.size))
    assert abs(float(estimates.mean()) - target) < 3.0 * monte_carlo_se


def test_alive_island_ensemble_is_mean_evidence_not_mean_nll() -> None:
    q_values, choices, feedback, prior, kernels = _fixture(
        n_hypotheses=3, n_trials=3
    )
    parameters = Model0804Parameters(
        0.72, 0.33, 1.5, 0.46, 0.40, lapse=0.02
    )
    exact = run_model0804_exact(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
    )
    islands = [
        run_model0804_alive_particle_filter(
            q_values,
            choices,
            feedback,
            prior,
            kernels,
            model_id="FA2",
            parameters=parameters,
            capacity=1,
            particle_count=64,
            filter_seed=seed,
            alive_batch_size=64,
        )
        for seed in range(32)
    ]
    ensemble = combine_model0804_alive_islands(islands, choices)
    mean_evidence = float(np.mean([np.exp(-trace.nll) for trace in islands]))
    assert np.isclose(np.exp(-ensemble.nll), mean_evidence, atol=1e-14)
    np.testing.assert_allclose(
        ensemble.probabilities, exact.probabilities, atol=0.035, rtol=0.0
    )
    assert abs(ensemble.nll - exact.nll) < 0.06
    assert np.all(ensemble.effective_island_count >= 1.0)
    assert np.all(ensemble.effective_island_count <= len(islands))


def test_alive_filter_fa1_equals_fa2_at_zero_global_share_under_crn() -> None:
    q_values, choices, feedback, prior, kernels = _fixture()
    parameters = Model0804Parameters(
        0.68, 0.31, 1.8, 0.42, 0.0, lapse=0.02
    )
    common = dict(
        q_values=q_values,
        choices=choices,
        feedback=feedback,
        prior=prior,
        kernels=kernels,
        parameters=parameters,
        capacity=2,
        particle_count=1_024,
        filter_seed=31,
        alive_batch_size=256,
    )
    fa1 = run_model0804_alive_particle_filter(model_id="FA1", **common)
    fa2 = run_model0804_alive_particle_filter(model_id="FA2", **common)
    np.testing.assert_array_equal(fa1.probabilities, fa2.probabilities)
    np.testing.assert_array_equal(
        fa1.marginal_active_probability, fa2.marginal_active_probability
    )
    np.testing.assert_array_equal(
        fa1.alive_attempt_count, fa2.alive_attempt_count
    )


def test_alive_filter_does_not_condition_on_masked_choice() -> None:
    q_values, choices, feedback, prior, kernels = _fixture()
    changed = choices.copy()
    changed[2] = 1 - changed[2]
    changed_feedback = feedback.copy()
    changed_feedback[2] = 1.0 - changed_feedback[2]
    condition = np.ones(len(choices), dtype=bool)
    condition[2] = False
    parameters = Model0804Parameters(
        0.57, 0.24, 1.6, 0.55, 0.35, lapse=0.02
    )
    common = dict(
        q_values=q_values,
        prior=prior,
        kernels=kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=2,
        particle_count=1_024,
        filter_seed=47,
        alive_batch_size=256,
        score_mask=np.zeros(len(choices), dtype=bool),
        condition_on_choice_mask=condition,
    )
    first = run_model0804_alive_particle_filter(
        choices=choices, feedback=feedback, **common
    )
    second = run_model0804_alive_particle_filter(
        choices=changed, feedback=changed_feedback, **common
    )
    np.testing.assert_array_equal(first.probabilities, second.probabilities)
    np.testing.assert_array_equal(
        first.marginal_active_probability,
        second.marginal_active_probability,
    )


def test_resample_move_filter_matches_exact_small_space() -> None:
    q_values, choices, feedback, prior, kernels = _fixture(
        n_hypotheses=3, n_trials=3
    )
    parameters = Model0804Parameters(
        0.72, 0.33, 1.5, 0.46, 0.40, lapse=0.02
    )
    exact = run_model0804_exact(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
    )
    moved = run_model0804_resample_move_particle_filter(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
        particle_count=8_192,
        filter_seed=83,
        rejuvenation_window=3,
        rejuvenation_sweeps=2,
    )
    np.testing.assert_allclose(
        moved.probabilities, exact.probabilities, atol=0.025, rtol=0.0
    )
    assert abs(moved.nll - exact.nll) < 0.05
    assert np.all((moved.rejuvenation_acceptance_rate > 0.0))
    assert np.all((moved.rejuvenation_acceptance_rate <= 1.0))
    assert moved.inference_method == "resample_move"
    assert moved.rejuvenation_window == 3
    assert moved.rejuvenation_sweeps == 2


def test_resample_move_normalizing_constant_is_unbiased_in_small_space() -> None:
    q_values, choices, feedback, prior, kernels = _fixture(
        n_hypotheses=3, n_trials=3
    )
    parameters = Model0804Parameters(
        0.72, 0.33, 1.5, 0.46, 0.40, lapse=0.02
    )
    exact = run_model0804_exact(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
    )
    estimates = []
    for seed in range(200):
        moved = run_model0804_resample_move_particle_filter(
            q_values,
            choices,
            feedback,
            prior,
            kernels,
            model_id="FA2",
            parameters=parameters,
            capacity=1,
            particle_count=32,
            filter_seed=seed,
            rejuvenation_window=3,
            rejuvenation_sweeps=1,
        )
        estimates.append(np.exp(-moved.nll))
    estimates = np.asarray(estimates)
    target = float(np.exp(-exact.nll))
    monte_carlo_se = float(estimates.std(ddof=1) / np.sqrt(estimates.size))
    assert abs(float(estimates.mean()) - target) < 3.0 * monte_carlo_se


def test_resample_move_fa1_equals_fa2_at_zero_global_share_under_crn() -> None:
    q_values, choices, feedback, prior, kernels = _fixture()
    parameters = Model0804Parameters(
        0.68, 0.31, 1.8, 0.42, 0.0, lapse=0.02
    )
    common = dict(
        q_values=q_values,
        choices=choices,
        feedback=feedback,
        prior=prior,
        kernels=kernels,
        parameters=parameters,
        capacity=2,
        particle_count=1_024,
        filter_seed=31,
        rejuvenation_window=4,
        rejuvenation_sweeps=1,
    )
    fa1 = run_model0804_resample_move_particle_filter(model_id="FA1", **common)
    fa2 = run_model0804_resample_move_particle_filter(model_id="FA2", **common)
    np.testing.assert_array_equal(fa1.probabilities, fa2.probabilities)
    np.testing.assert_array_equal(
        fa1.marginal_active_probability, fa2.marginal_active_probability
    )
    np.testing.assert_array_equal(
        fa1.rejuvenation_acceptance_rate,
        fa2.rejuvenation_acceptance_rate,
    )


def test_resample_move_has_no_current_feedback_or_masked_choice_leakage() -> None:
    q_values, choices, feedback, prior, kernels = _fixture()
    parameters = Model0804Parameters(
        0.57, 0.24, 1.6, 0.55, 0.35, lapse=0.02
    )
    common = dict(
        q_values=q_values,
        prior=prior,
        kernels=kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=2,
        particle_count=1_024,
        filter_seed=47,
        rejuvenation_window=4,
        rejuvenation_sweeps=1,
    )
    changed_feedback = feedback.copy()
    changed_feedback[2] = 1.0 - changed_feedback[2]
    first = run_model0804_resample_move_particle_filter(
        choices=choices, feedback=feedback, **common
    )
    second = run_model0804_resample_move_particle_filter(
        choices=choices, feedback=changed_feedback, **common
    )
    np.testing.assert_array_equal(first.probabilities[:3], second.probabilities[:3])

    changed_choice = choices.copy()
    changed_choice[2] = 1 - changed_choice[2]
    condition = np.ones(len(choices), dtype=bool)
    condition[2] = False
    masked_common = dict(
        common,
        score_mask=np.zeros(len(choices), dtype=bool),
        condition_on_choice_mask=condition,
    )
    masked_first = run_model0804_resample_move_particle_filter(
        choices=choices, feedback=feedback, **masked_common
    )
    masked_second = run_model0804_resample_move_particle_filter(
        choices=changed_choice,
        feedback=changed_feedback,
        **masked_common,
    )
    np.testing.assert_array_equal(
        masked_first.probabilities, masked_second.probabilities
    )


def test_fa_parameter_schemas_and_nested_starts_preserve_boundaries() -> None:
    fa0_definition = parameter_definition("FA0", "dual")
    fa0_parameters, fa0_reported = decode_parameters(
        fa0_definition.center, "FA0", "dual"
    )
    fa0 = Model0804Fit(
        model_id="FA0",
        memory_id="dual",
        raw_vector=fa0_definition.center.copy(),
        parameters=fa0_parameters,
        reported_parameters=fa0_reported,
        train_nll=0.0,
        diagnostics={},
    )
    fa1_raw = nested_child_start(fa0, "FA1")
    fa1_parameters, fa1_reported = decode_parameters(fa1_raw, "FA1", "dual")
    assert fa1_parameters.m == 0.0
    fa1 = Model0804Fit(
        model_id="FA1",
        memory_id="dual",
        raw_vector=fa1_raw,
        parameters=fa1_parameters,
        reported_parameters=fa1_reported,
        train_nll=0.0,
        diagnostics={},
    )
    fa2_raw = nested_child_start(fa1, "FA2")
    fa2_parameters, _ = decode_parameters(fa2_raw, "FA2", "dual")
    assert fa2_parameters.m == 0.0
    assert fa2_parameters.g == 0.0
    assert parameter_definition("FA1", "dual").bounds[-1] == (0.0, 1.0)
    assert parameter_definition("FA2", "dual").bounds[-1] == (0.0, 1.0)
