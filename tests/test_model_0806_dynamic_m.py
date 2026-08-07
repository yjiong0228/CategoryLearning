from __future__ import annotations

import numpy as np

from scripts.run_model_0806_real_rolling import shared_choice_metrics
from src.Bayesian_state.utils.model_0803 import build_transition_kernels
from src.Bayesian_state.utils.model_0804 import (
    Model0804Parameters,
)
from src.Bayesian_state.utils.model_0804 import (
    run_model0804_particle_filter,
)
from src.Bayesian_state.utils.model_0806 import (
    Model0804RTParameters,
    simulate_model0806_log_rt,
)
from src.Bayesian_state.utils.model_0806 import simulate_model0806_choices


def _inputs(trials: int = 6) -> tuple[np.ndarray, np.ndarray, object]:
    rng = np.random.default_rng(806)
    q = rng.uniform(0.15, 1.0, size=(trials, 6, 2))
    q /= q.sum(axis=2, keepdims=True)
    prior = np.full(6, 1.0 / 6.0)
    similarity = np.eye(6) * 0.8 + 0.2
    kernels = build_transition_kernels(similarity, prior, tau_local=0.25)
    return q, prior, kernels


def test_rolling_uses_canonical_multiclass_choice_brier() -> None:
    probabilities = np.asarray([[0.8, 0.2], [0.1, 0.9]], dtype=float)
    choices = np.asarray([0, 1], dtype=int)
    metrics = shared_choice_metrics(
        probabilities,
        choices,
        observed_correct=np.asarray([1.0, 1.0]),
        predicted_correct=np.asarray([0.8, 0.9]),
        window_size=1,
    )
    # Canonical choice Brier sums squared error over both categories.
    assert np.isclose(metrics["choice_brier"], 0.05)


def test_zero_dynamic_effect_is_exact_fa2_endpoint() -> None:
    q, prior, kernels = _inputs(8)
    choices = np.asarray([0, 1, 0, 1, 1, 0, 1, 0])
    feedback = np.asarray([1, 0, 1, 1, 0, 0, 1, 1], dtype=float)
    common = dict(
        gamma=0.70,
        w0=0.40,
        kappa=2.0,
        m=0.15,
        g=0.35,
        lapse=0.02,
        rho=0.0,
    )
    static = run_model0804_particle_filter(
        q,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=Model0804Parameters(**common),
        capacity=3,
        particle_count=512,
        filter_seed=806,
    )
    endpoint = run_model0804_particle_filter(
        q,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=Model0804Parameters(
            **common,
            dynamic_m=True,
            m_phi=0.60,
            m_beta_surprise=0.0,
        ),
        capacity=3,
        particle_count=512,
        filter_seed=806,
    )
    assert static.nll == endpoint.nll
    assert np.array_equal(static.probabilities, endpoint.probabilities)
    assert np.array_equal(
        static.predictive_replacement_count,
        endpoint.predictive_replacement_count,
    )


def test_feedback_changes_only_the_next_dynamic_control() -> None:
    q, prior, kernels = _inputs(4)
    q[1, :, 0] = 0.90
    q[1, :, 1] = 0.10
    choices = np.zeros(4, dtype=int)
    feedback_low_surprise = np.ones(4, dtype=float)
    feedback_high_surprise = feedback_low_surprise.copy()
    feedback_high_surprise[1] = 0.0
    parameters = Model0804Parameters(
        gamma=0.70,
        w0=0.40,
        kappa=2.0,
        m=0.15,
        g=0.35,
        lapse=0.02,
        rho=0.0,
        dynamic_m=True,
        m_phi=0.0,
        m_beta_surprise=0.75,
        surprise_center=0.0,
        surprise_scale=1.0,
    )
    low = run_model0804_particle_filter(
        q,
        choices,
        feedback_low_surprise,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=3,
        particle_count=1024,
        filter_seed=807,
    )
    high = run_model0804_particle_filter(
        q,
        choices,
        feedback_high_surprise,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=3,
        particle_count=1024,
        filter_seed=807,
    )
    assert np.array_equal(low.probabilities[:2], high.probabilities[:2])
    assert np.array_equal(low.predictive_m[:2], high.predictive_m[:2])
    assert high.feedback_surprise[1] > low.feedback_surprise[1]
    assert high.predictive_m[2] > low.predictive_m[2]
    assert (
        high.predictive_replacement_fraction[2]
        > low.predictive_replacement_fraction[2]
    )


def test_uncertainty_drives_only_the_next_autonomous_control() -> None:
    q, prior, kernels = _inputs(12)
    categories = np.asarray([0, 1] * 6)
    baseline_m = 0.15
    beta_uncertainty = 0.80
    center = 0.20
    scale = 0.10
    simulation = simulate_model0806_choices(
        q,
        categories,
        prior,
        kernels,
        parameters=Model0804Parameters(
            gamma=0.70,
            w0=0.40,
            kappa=2.0,
            m=baseline_m,
            g=0.35,
            lapse=0.02,
            rho=0.0,
            dynamic_m=True,
            m_phi=0.0,
            m_beta_uncertainty=beta_uncertainty,
            uncertainty_center=center,
            uncertainty_scale=scale,
        ),
        capacity=3,
        seed=8071,
    )
    baseline_logit = np.log(baseline_m / (1.0 - baseline_m))
    expected_logit = baseline_logit + beta_uncertainty * (
        simulation.feedback_uncertainty[:-1] - center
    ) / scale
    expected = 1.0 / (1.0 + np.exp(-expected_logit))
    assert np.allclose(simulation.predictive_m[1:], expected)
    assert simulation.predictive_m[0] == baseline_m


def test_dynamic_trace_is_finite_and_bounded() -> None:
    q, prior, kernels = _inputs(12)
    choices = np.asarray([0, 1] * 6)
    feedback = np.asarray([1, 0, 0, 1] * 3, dtype=float)
    trace = run_model0804_particle_filter(
        q,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=Model0804Parameters(
            gamma=0.70,
            w0=0.40,
            kappa=2.0,
            m=0.15,
            g=0.35,
            lapse=0.02,
            rho=0.0,
            dynamic_m=True,
            m_phi=0.50,
            m_beta_surprise=0.40,
            surprise_center=0.7,
            surprise_scale=0.5,
        ),
        capacity=3,
        particle_count=512,
        filter_seed=808,
    )
    assert np.isfinite(trace.nll)
    assert np.all(np.isfinite(trace.probabilities))
    assert np.all(np.isfinite(trace.predictive_m))
    assert np.all((trace.predictive_m > 0.0) & (trace.predictive_m < 1.0))
    assert np.all(np.isfinite(trace.feedback_surprise))
    assert np.all(np.isfinite(trace.feedback_uncertainty))
    assert np.all(
        (trace.feedback_uncertainty >= 0.0)
        & (trace.feedback_uncertainty <= 1.0 + 1e-12)
    )


def test_autonomous_simulator_preserves_static_endpoint() -> None:
    q, prior, kernels = _inputs(10)
    categories = np.asarray([0, 1] * 5)
    common = dict(
        gamma=0.70,
        w0=0.40,
        kappa=2.0,
        m=0.15,
        g=0.35,
        lapse=0.02,
        rho=0.0,
    )
    static = simulate_model0806_choices(
        q,
        categories,
        prior,
        kernels,
        parameters=Model0804Parameters(**common),
        capacity=3,
        seed=809,
    )
    endpoint = simulate_model0806_choices(
        q,
        categories,
        prior,
        kernels,
        parameters=Model0804Parameters(
            **common,
            dynamic_m=True,
            m_phi=0.50,
            m_beta_surprise=0.0,
        ),
        capacity=3,
        seed=809,
    )
    assert np.array_equal(static.choices, endpoint.choices)
    assert np.array_equal(static.feedback, endpoint.feedback)
    assert np.array_equal(static.probabilities, endpoint.probabilities)
    assert np.array_equal(static.active_path, endpoint.active_path)


def test_joint_choice_rt_likelihood_uses_same_latent_paths() -> None:
    q, prior, kernels = _inputs(20)
    categories = np.asarray([0, 1] * 10)
    parameters = Model0804Parameters(
        gamma=0.70,
        w0=0.40,
        kappa=2.0,
        m=0.15,
        g=0.35,
        lapse=0.02,
        rho=0.0,
        dynamic_m=True,
        m_phi=0.50,
        m_beta_surprise=0.50,
        surprise_center=2.5,
        surprise_scale=4.5,
    )
    rt_parameters = Model0804RTParameters(
        intercept=-0.2,
        choice_entropy=0.25,
        replacement_fraction=0.40,
        sigma=0.15,
        degrees_of_freedom=5.0,
    )
    simulation = simulate_model0806_choices(
        q,
        categories,
        prior,
        kernels,
        parameters=parameters,
        capacity=3,
        seed=810,
    )
    log_rt = simulate_model0806_log_rt(
        simulation, rt_parameters, seed=811
    )
    trace = run_model0804_particle_filter(
        q,
        simulation.choices,
        simulation.feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=3,
        particle_count=1024,
        filter_seed=812,
        log_rt_values=log_rt,
        rt_parameters=rt_parameters,
    )
    assert np.all(np.isfinite(log_rt))
    assert trace.joint_nll is not None
    assert trace.rt_conditional_nll is not None
    assert trace.rt_predictive_log_density is not None
    assert np.isclose(trace.joint_nll, trace.nll + trace.rt_conditional_nll)
    assert np.all(np.isfinite(trace.rt_predictive_log_density))
