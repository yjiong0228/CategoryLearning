from __future__ import annotations

import numpy as np

from scripts.run_model_0805_real_predictive import (
    observed_log_probabilities,
    sequential_mixture,
    split_masks,
)
from src.Bayesian_state.utils.model_0803 import build_transition_kernels
from src.Bayesian_state.utils.model_0804 import (
    Model0804Parameters,
    run_model0804_particle_filter,
)


def test_split_masks_nests_inner_validation_inside_outer_training() -> None:
    outer = np.asarray([False] * 8 + [True] * 2)
    masks = split_masks(outer, 0.25)
    assert np.array_equal(masks["inner_fit"], [True] * 6 + [False] * 4)
    assert np.array_equal(
        masks["inner_validation"], [False] * 6 + [True] * 2 + [False] * 2
    )
    assert np.array_equal(masks["outer_train"], ~outer)
    assert not np.any(masks["inner_fit"] & masks["inner_validation"])


def test_observed_log_probabilities_selects_realized_choice() -> None:
    probabilities = np.asarray(
        [
            [[0.8, 0.2], [0.3, 0.7]],
            [[0.6, 0.4], [0.9, 0.1]],
        ]
    )
    observed = observed_log_probabilities(probabilities, np.asarray([0, 1]))
    assert np.allclose(observed, np.log([[0.8, 0.7], [0.6, 0.1]]))


def test_sequential_mixture_averages_before_log_and_updates_weights() -> None:
    probabilities = np.asarray(
        [
            [[0.9, 0.1], [0.9, 0.1]],
            [[0.1, 0.9], [0.1, 0.9]],
        ]
    )
    choices = np.asarray([0, 0])
    result = sequential_mixture(
        probabilities,
        choices,
        conditioning_mask=np.asarray([True, True]),
        score_mask=np.asarray([True, True]),
    )
    assert np.allclose(result["probabilities"][0], [0.5, 0.5])
    assert np.allclose(result["probabilities"][1], [0.82, 0.18])
    assert np.isclose(result["nll"], -np.log(0.5) - np.log(0.82))
    assert np.allclose(result["final_weights"], [0.9878048780487805, 0.0121951219512195])


def test_unscored_prefix_conditions_component_weights_without_leakage() -> None:
    probabilities = np.asarray(
        [
            [[0.8, 0.2], [0.75, 0.25]],
            [[0.2, 0.8], [0.25, 0.75]],
        ]
    )
    result = sequential_mixture(
        probabilities,
        np.asarray([0, 1]),
        conditioning_mask=np.asarray([True, True]),
        score_mask=np.asarray([False, True]),
    )
    assert np.allclose(result["probabilities"][1], [0.65, 0.35])
    assert np.isclose(result["nll"], -np.log(0.35))


def test_dense_feedback_update_is_finite_at_exact_memory_endpoints() -> None:
    rng = np.random.default_rng(805)
    q = rng.uniform(0.1, 1.0, size=(8, 6, 2))
    q /= q.sum(axis=2, keepdims=True)
    choices = np.asarray([0, 1, 0, 1, 1, 0, 1, 0])
    feedback = np.asarray([1, 0, 1, 1, 0, 0, 1, 1], dtype=float)
    prior = np.full(6, 1.0 / 6.0)
    similarity = np.eye(6) * 0.8 + 0.2
    kernels = build_transition_kernels(similarity, prior, tau_local=0.25)
    for w0 in (0.0, 1.0):
        with np.errstate(invalid="raise", divide="raise", over="raise"):
            trace = run_model0804_particle_filter(
                q,
                choices,
                feedback,
                prior,
                kernels,
                model_id="FA2R",
                parameters=Model0804Parameters(
                    gamma=0.70,
                    w0=w0,
                    kappa=2.0,
                    m=0.15,
                    g=0.35,
                    lapse=0.02,
                    rho=0.02,
                ),
                capacity=3,
                particle_count=128,
                filter_seed=805,
            )
        assert np.isfinite(trace.nll)
        assert np.all(np.isfinite(trace.probabilities))
