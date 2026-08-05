from __future__ import annotations

import numpy as np

from src.Bayesian_state.utils.model_0803 import (
    FeatureScaling,
    Model0803Fit,
    build_transition_kernels,
    decode_parameters,
    parameter_definition,
    reference_feature_scaling,
    run_model0803,
)
from src.Bayesian_state.utils.unified_newplan import rule_predictions
from scripts.run_model_0803_cond1 import nested_child_start


def _fixture():
    rng = np.random.default_rng(12)
    raw = rng.uniform(0.05, 1.0, size=(8, 4, 2))
    q_values = raw / raw.sum(axis=2, keepdims=True)
    choices = np.asarray([0, 1, 0, 0, 1, 1, 0, 1], dtype=np.int64)
    feedback = np.asarray([1, 0, 1, 0, 0, 1, 1, 0], dtype=float)
    prior = np.asarray([0.10, 0.20, 0.30, 0.40], dtype=float)
    similarity = np.asarray(
        [
            [1.0, 0.80, 0.45, 0.15],
            [0.80, 1.0, 0.55, 0.25],
            [0.45, 0.55, 1.0, 0.75],
            [0.15, 0.25, 0.75, 1.0],
        ],
        dtype=float,
    )
    kernels = build_transition_kernels(similarity, prior)
    scaling = FeatureScaling(np.zeros(2), np.ones(2), "test")
    return q_values, choices, feedback, prior, kernels, scaling


def _full(gamma=1.0, w0=1.0, kappa=1.0):
    values = np.zeros(11, dtype=float)
    values[:3] = [gamma, w0, kappa]
    return values


def test_transition_kernels_normalize_and_local_is_closer() -> None:
    _, _, _, _, kernels, _ = _fixture()
    np.testing.assert_allclose(kernels.local.sum(axis=1), 1.0)
    np.testing.assert_allclose(kernels.global_.sum(axis=1), 1.0)
    np.testing.assert_allclose(np.diag(kernels.local), 0.0)
    np.testing.assert_allclose(np.diag(kernels.global_), 0.0)
    assert np.all(
        kernels.expected_local_distance < kernels.expected_global_distance
    )


def test_h0_standard_bayes_matches_existing_direct_likelihood_recursion() -> None:
    q_values, choices, feedback, prior, kernels, scaling = _fixture()
    actual = run_model0803(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="H0",
        full_parameters=_full(1.0, 1.0, 1.7),
        feature_scaling=scaling,
    )
    expected = rule_predictions(
        q_values,
        choices,
        feedback,
        1,
        retention=1.0,
        sensitivity=1.7,
        prior=prior,
    )
    np.testing.assert_allclose(
        actual.probabilities, expected.probabilities, atol=1e-11, rtol=1e-11
    )


def test_gamma_one_and_w0_one_are_the_same_bayes_endpoint() -> None:
    q_values, choices, feedback, prior, kernels, scaling = _fixture()
    gamma_endpoint = run_model0803(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="H2",
        full_parameters=np.asarray([1.0, 0.23, 1.4, 0.18, 0.42, 0, 0, 0, 0, 0, 0]),
        feature_scaling=scaling,
    )
    w0_endpoint = run_model0803(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="H2",
        full_parameters=np.asarray([0.51, 1.0, 1.4, 0.18, 0.42, 0, 0, 0, 0, 0, 0]),
        feature_scaling=scaling,
    )
    np.testing.assert_allclose(
        gamma_endpoint.probabilities,
        w0_endpoint.probabilities,
        atol=2e-12,
        rtol=2e-12,
    )
    np.testing.assert_allclose(gamma_endpoint.pi_plus, w0_endpoint.pi_plus, atol=2e-12)


def test_fade_only_first_update_matches_closed_form_endpoint() -> None:
    q_values, choices, feedback, prior, kernels, scaling = _fixture()
    gamma = 0.61
    trace = run_model0803(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="H0",
        full_parameters=_full(gamma, 0.0, 1.0),
        feature_scaling=scaling,
    )
    compatible = choices[0] if feedback[0] == 1 else 1 - choices[0]
    likelihood = q_values[0, :, compatible]
    expected = prior**gamma * likelihood
    expected /= expected.sum()
    np.testing.assert_allclose(trace.pi_plus[0], expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(trace.pi_minus[1], expected, atol=1e-12, rtol=1e-12)


def test_current_feedback_never_changes_current_choice_prediction() -> None:
    q_values, choices, feedback, prior, kernels, scaling = _fixture()
    changed = feedback.copy()
    changed[3] = 1.0 - changed[3]
    parameters = np.asarray(
        [0.65, 0.35, 1.8, -1.2, -0.5, 0.6, 0.8, 0.5, 0.4, 0.3, -0.2]
    )
    first = run_model0803(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="H3_MG",
        full_parameters=parameters,
        feature_scaling=scaling,
    )
    second = run_model0803(
        q_values,
        choices,
        changed,
        prior,
        kernels,
        model_id="H3_MG",
        full_parameters=parameters,
        feature_scaling=scaling,
    )
    np.testing.assert_allclose(first.probabilities[:4], second.probabilities[:4])
    assert not np.allclose(first.probabilities[4], second.probabilities[4])


def test_condition1_state_depends_on_revealed_category_not_choice_encoding() -> None:
    q_values, _, _, prior, kernels, scaling = _fixture()
    categories = np.asarray([0, 1, 1, 0, 0, 1, 0, 1], dtype=np.int64)
    choices_a = categories.copy()
    feedback_a = np.ones(len(categories), dtype=float)
    choices_b = 1 - categories
    feedback_b = np.zeros(len(categories), dtype=float)
    parameters = np.asarray(
        [0.58, 0.31, 1.6, -1.1, -0.2, 0.5, 0.7, -0.3, 0.4, 0.2, 0.6]
    )
    first = run_model0803(
        q_values,
        choices_a,
        feedback_a,
        prior,
        kernels,
        model_id="H3_MG",
        full_parameters=parameters,
        feature_scaling=scaling,
    )
    second = run_model0803(
        q_values,
        choices_b,
        feedback_b,
        prior,
        kernels,
        model_id="H3_MG",
        full_parameters=parameters,
        feature_scaling=scaling,
    )
    np.testing.assert_allclose(first.probabilities, second.probabilities, atol=1e-12)
    np.testing.assert_allclose(first.pi_plus, second.pi_plus, atol=1e-12)
    np.testing.assert_allclose(first.m, second.m, atol=1e-12)
    np.testing.assert_allclose(first.g, second.g, atol=1e-12)


def test_h0_h1_h2_are_exactly_nested_at_closed_boundaries() -> None:
    q_values, choices, feedback, prior, kernels, scaling = _fixture()
    h0 = run_model0803(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="H0",
        full_parameters=_full(0.7, 0.4, 1.5),
        feature_scaling=scaling,
    )
    h1_parameters = _full(0.7, 0.4, 1.5)
    h1_parameters[3] = 0.0
    h1 = run_model0803(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="H1",
        full_parameters=h1_parameters,
        feature_scaling=scaling,
    )
    h2_parameters = h1_parameters.copy()
    h2_parameters[3] = 0.27
    h2_parameters[4] = 0.0
    h1_nonzero = h2_parameters.copy()
    h1_nonzero[4] = 0.73  # ignored by H1
    h1_nonzero_trace = run_model0803(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="H1",
        full_parameters=h1_nonzero,
        feature_scaling=scaling,
    )
    h2 = run_model0803(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="H2",
        full_parameters=h2_parameters,
        feature_scaling=scaling,
    )
    np.testing.assert_allclose(h0.probabilities, h1.probabilities, atol=1e-12)
    np.testing.assert_allclose(h1_nonzero_trace.probabilities, h2.probabilities, atol=1e-12)


def test_memory_sync_and_probability_invariants_hold_for_h3() -> None:
    q_values, choices, feedback, prior, kernels, scaling = _fixture()
    parameters = np.asarray(
        [0.43, 0.27, 2.1, -1.4, -0.4, 0.72, 1.1, -0.6, 0.55, 0.4, 0.3]
    )
    trace = run_model0803(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="H3_MG",
        full_parameters=parameters,
        feature_scaling=scaling,
    )
    np.testing.assert_allclose(trace.probabilities.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(trace.pi_minus.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(trace.pi_plus.sum(axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(trace.operation_weights.sum(axis=1), 1.0, atol=1e-12)
    assert np.max(trace.memory_sync_error) < 1e-12
    assert np.all((trace.m > 0.0) & (trace.m < 1.0))
    assert np.all((trace.g > 0.0) & (trace.g < 1.0))


def test_reference_scaling_uses_training_trials_and_is_finite() -> None:
    q_values, choices, feedback, prior, kernels, _ = _fixture()
    train = np.asarray([True, True, True, True, True, False, False, False])
    scaling = reference_feature_scaling(
        q_values, choices, feedback, prior, kernels, train
    )
    assert scaling.reference.startswith("training_H0_bayes")
    assert np.all(np.isfinite(scaling.center))
    assert np.all(np.isfinite(scaling.scale))
    assert np.all(scaling.scale > 0.0)


def test_parameter_schemas_keep_constant_models_on_closed_intervals() -> None:
    h1 = parameter_definition("H1", "dual")
    h2 = parameter_definition("H2", "dual")
    assert h1.bounds[h1.names.index("m")] == (0.0, 1.0)
    assert h2.bounds[h2.names.index("g")] == (0.0, 1.0)
    h3 = parameter_definition("H3_M", "dual")
    assert h3.bounds[h3.names.index("mu_m")] == (-30.0, 30.0)
    assert h3.bounds[h3.names.index("mu_g")] == (-30.0, 30.0)
    full, reported = decode_parameters(h2.center, "H2", "dual")
    assert full[3] == reported["m"]
    assert full[4] == reported["g"]


def test_runner_nested_warm_starts_preserve_parent_predictions() -> None:
    q_values, choices, feedback, prior, kernels, scaling = _fixture()

    def make_fit(model_id: str, raw: np.ndarray) -> Model0803Fit:
        full, reported = decode_parameters(raw, model_id, "dual")
        return Model0803Fit(
            model_id=model_id,
            memory_id="dual",
            raw_vector=raw,
            full_parameters=full,
            parameters=reported,
            train_nll=0.0,
            diagnostics={},
        )

    h0 = make_fit("H0", parameter_definition("H0", "dual").center.copy())
    h1_boundary = make_fit("H1", nested_child_start(h0, "H1"))
    h1_interior_raw = h1_boundary.raw_vector.copy()
    h1_interior_raw[parameter_definition("H1", "dual").names.index("m")] = 0.23
    h1_interior = make_fit("H1", h1_interior_raw)
    h2_boundary = make_fit("H2", nested_child_start(h1_interior, "H2"))
    h2_interior_raw = h2_boundary.raw_vector.copy()
    h2_interior_raw[parameter_definition("H2", "dual").names.index("g")] = 0.41
    h2_interior = make_fit("H2", h2_interior_raw)
    h3_m = make_fit("H3_M", nested_child_start(h2_interior, "H3_M"))
    h3_mg = make_fit("H3_MG", nested_child_start(h3_m, "H3_MG"))
    pairs = [
        (h0, h1_boundary),
        (h1_interior, h2_boundary),
        (h2_interior, h3_m),
        (h3_m, h3_mg),
    ]
    for parent, child in pairs:
        parent_trace = run_model0803(
            q_values,
            choices,
            feedback,
            prior,
            kernels,
            model_id=parent.model_id,
            full_parameters=parent.full_parameters,
            feature_scaling=scaling,
        )
        child_trace = run_model0803(
            q_values,
            choices,
            feedback,
            prior,
            kernels,
            model_id=child.model_id,
            full_parameters=child.full_parameters,
            feature_scaling=scaling,
        )
        np.testing.assert_allclose(
            parent_trace.probabilities, child_trace.probabilities, atol=2e-10
        )


def test_h3_closure_reproduces_h2_closed_boundaries_to_numerical_tolerance() -> None:
    q_values, choices, feedback, prior, kernels, scaling = _fixture()
    definition = parameter_definition("H2", "dual")
    for m_value, g_value in ((0.0, 0.73), (1.0, 0.0), (1.0, 1.0)):
        raw = definition.center.copy()
        raw[definition.names.index("m")] = m_value
        raw[definition.names.index("g")] = g_value
        full, reported = decode_parameters(raw, "H2", "dual")
        parent = Model0803Fit(
            model_id="H2",
            memory_id="dual",
            raw_vector=raw,
            full_parameters=full,
            parameters=reported,
            train_nll=0.0,
            diagnostics={},
        )
        child_raw = nested_child_start(parent, "H3_M")
        child_full, _ = decode_parameters(child_raw, "H3_M", "dual")
        parent_trace = run_model0803(
            q_values,
            choices,
            feedback,
            prior,
            kernels,
            model_id="H2",
            full_parameters=full,
            feature_scaling=scaling,
        )
        child_trace = run_model0803(
            q_values,
            choices,
            feedback,
            prior,
            kernels,
            model_id="H3_M",
            full_parameters=child_full,
            feature_scaling=scaling,
        )
        np.testing.assert_allclose(
            parent_trace.probabilities,
            child_trace.probabilities,
            atol=2e-10,
            rtol=2e-10,
        )
