from __future__ import annotations

import numpy as np

from src.Bayesian_state.utils.model_0804 import (
    HFWState,
    Model0804Parameters,
    _initial_state,
)
from src.Bayesian_state.utils.model_0804_forgetting import (
    couple_model0804_histories,
    sample_model0804_filtered_anchor_states,
)
from tests.test_model_0804 import _fixture


def test_identical_anchor_states_remain_identical_under_common_randomness() -> None:
    q_values, choices, feedback, prior, kernels = _fixture(
        n_hypotheses=5, n_trials=7
    )
    parameters = Model0804Parameters(
        0.70, 0.40, 2.0, 0.15, 0.35, lapse=0.02
    )
    state = _initial_state(np.asarray([0, 3]), prior)
    trace = couple_model0804_histories(
        [state, state],
        [state.copy(), state.copy()],
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=2,
        anchor_trial=0,
        horizon=6,
        coupling_seed=804,
    )
    np.testing.assert_array_equal(trace.active_distance, 0.0)
    np.testing.assert_array_equal(trace.active_equal, True)
    np.testing.assert_array_equal(trace.state_exact_equal, True)
    np.testing.assert_array_equal(trace.omega_total_variation, 0.0)
    np.testing.assert_array_equal(trace.choice_probability_difference, 0.0)
    np.testing.assert_array_equal(
        trace.signed_choice_probability_difference, 0.0
    )
    np.testing.assert_array_equal(trace.common_memory_delta_difference, 0.0)


def test_fade_only_ignores_legacy_static_delta_but_dual_memory_transmits_it() -> None:
    q_values, choices, feedback, prior, kernels = _fixture(
        n_hypotheses=5, n_trials=7
    )
    base = _initial_state(np.asarray([0, 3]), prior)
    delta = np.asarray([4.0, -4.0])

    fade_only_changed = base.copy()
    fade_only_changed.static[base.active] += delta
    fade_parameters = Model0804Parameters(0.70, 0.0, 2.0, 0.0, 0.0)
    fade_trace = couple_model0804_histories(
        [base],
        [fade_only_changed],
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=fade_parameters,
        capacity=2,
        anchor_trial=0,
        horizon=6,
        coupling_seed=805,
    )
    np.testing.assert_array_equal(fade_trace.omega_total_variation, 0.0)
    np.testing.assert_array_equal(
        fade_trace.choice_probability_difference, 0.0
    )

    dual_left = base.copy()
    dual_right = base.copy()
    w0 = 0.40
    dual_right.fade[base.active] -= w0 * delta
    dual_right.static[base.active] += (1.0 - w0) * delta
    dual_parameters = Model0804Parameters(0.70, w0, 2.0, 0.0, 0.0)
    dual_trace = couple_model0804_histories(
        [dual_left],
        [dual_right],
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=dual_parameters,
        capacity=2,
        anchor_trial=0,
        horizon=6,
        coupling_seed=805,
    )
    assert np.max(dual_trace.choice_probability_difference[0, 1:]) > 1e-4
    assert np.max(dual_trace.omega_total_variation[0, 1:]) > 1e-4


def test_filtered_anchor_panel_is_seed_reproducible() -> None:
    q_values, choices, feedback, prior, kernels = _fixture(
        n_hypotheses=5, n_trials=7
    )
    parameters = Model0804Parameters(
        0.70, 0.40, 2.0, 0.15, 0.35, lapse=0.02
    )
    common = dict(
        q_values=q_values,
        choices=choices,
        feedback=feedback,
        prior=prior,
        kernels=kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=2,
        anchors=[1, 4],
        particle_count=32,
        sample_count=5,
        filter_seed=47,
    )
    first = sample_model0804_filtered_anchor_states(**common)
    second = sample_model0804_filtered_anchor_states(**common)
    np.testing.assert_array_equal(
        first.pre_resampling_ess, second.pre_resampling_ess
    )
    np.testing.assert_array_equal(first.resampled, second.resampled)
    for anchor in (1, 4):
        for left, right in zip(
            first.states_by_anchor[anchor], second.states_by_anchor[anchor]
        ):
            np.testing.assert_array_equal(left.active, right.active)
            np.testing.assert_array_equal(left.omega, right.omega)
            np.testing.assert_array_equal(left.fade, right.fade)
            np.testing.assert_array_equal(left.static, right.static)
