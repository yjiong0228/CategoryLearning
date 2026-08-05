from __future__ import annotations

import numpy as np

from src.Bayesian_state.utils.model_0804 import (
    Model0804Parameters,
    run_model0804_exact,
)
from src.Bayesian_state.utils.model_0804_pgas import (
    draw_model0804_innovation_path,
    replay_model0804_innovation_path,
    run_model0804_exact_smoothing,
    run_model0804_pgas,
)
from tests.test_model_0804 import _fixture


def _tiny_fixture():
    q_values, choices, feedback, prior, kernels = _fixture(
        n_hypotheses=3, n_trials=3
    )
    parameters = Model0804Parameters(
        0.72, 0.33, 1.5, 0.46, 0.40, lapse=0.02
    )
    return q_values, choices, feedback, prior, kernels, parameters


def test_exact_smoothing_evidence_matches_exact_forward_filter() -> None:
    q_values, choices, feedback, prior, kernels, parameters = _tiny_fixture()
    forward = run_model0804_exact(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
    )
    smoothing = run_model0804_exact_smoothing(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
    )
    assert np.isclose(smoothing.nll, forward.nll, atol=2e-15)
    np.testing.assert_allclose(smoothing.active_probability.sum(axis=1), 1.0)
    assert smoothing.path_count == 27


def test_full_suffix_pgas_matches_exact_tiny_space_smoothing() -> None:
    q_values, choices, feedback, prior, kernels, parameters = _tiny_fixture()
    exact = run_model0804_exact_smoothing(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
    )
    trace = run_model0804_pgas(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
        particle_count=16,
        iterations=500,
        burn_in=100,
        chain_seed=7,
    )
    np.testing.assert_allclose(
        trace.active_probability, exact.active_probability, atol=0.065
    )
    np.testing.assert_allclose(
        trace.expected_replacement_count,
        exact.expected_replacement_count,
        atol=0.035,
    )
    assert trace.ancestor_lookahead is None
    assert trace.normalizing_constant_estimated is False
    assert trace.retained_samples == 400


def test_pgas_fa1_equals_fa2_at_zero_global_share_under_crn() -> None:
    q_values, choices, feedback, prior, kernels, parameters = _tiny_fixture()
    parameters = Model0804Parameters(
        parameters.gamma,
        parameters.w0,
        parameters.kappa,
        parameters.m,
        0.0,
        lapse=parameters.lapse,
    )
    common = dict(
        q_values=q_values,
        choices=choices,
        feedback=feedback,
        prior=prior,
        kernels=kernels,
        parameters=parameters,
        capacity=1,
        particle_count=8,
        iterations=25,
        burn_in=5,
        chain_seed=31,
    )
    fa1 = run_model0804_pgas(model_id="FA1", **common)
    fa2 = run_model0804_pgas(model_id="FA2", **common)
    np.testing.assert_array_equal(
        fa1.retained_active_samples, fa2.retained_active_samples
    )
    np.testing.assert_array_equal(
        fa1.retained_replacement_samples, fa2.retained_replacement_samples
    )
    np.testing.assert_array_equal(
        fa1.iteration_log_choice_likelihood,
        fa2.iteration_log_choice_likelihood,
    )


def test_last_feedback_cannot_change_pre_feedback_path_posterior() -> None:
    q_values, choices, feedback, prior, kernels, parameters = _tiny_fixture()
    changed = feedback.copy()
    changed[-1] = 1.0 - changed[-1]
    first = run_model0804_exact_smoothing(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
    )
    second = run_model0804_exact_smoothing(
        q_values,
        choices,
        changed,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
    )
    np.testing.assert_array_equal(
        first.active_probability, second.active_probability
    )
    np.testing.assert_array_equal(
        first.expected_replacement_count,
        second.expected_replacement_count,
    )

    path = draw_model0804_innovation_path(3, 1, np.random.default_rng(19))
    replay_first = replay_model0804_innovation_path(
        path,
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
    )
    replay_second = replay_model0804_innovation_path(
        path,
        q_values,
        choices,
        changed,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
    )
    np.testing.assert_array_equal(replay_first.active, replay_second.active)
    assert replay_first.log_choice_likelihood == replay_second.log_choice_likelihood
