from __future__ import annotations

import numpy as np

from scripts.run_model_0806_targeted_diagnostics import (
    block_frozen_family_predictors,
    fit_offset_logistic,
    fit_student_regression,
)


def test_block_frozen_weights_update_only_at_block_boundary() -> None:
    probabilities = np.asarray([
        [[0.8, 0.2]] * 4,
        [[0.2, 0.8]] * 4,
    ])
    arrays = {
        "probabilities": probabilities,
        "correct_probability": np.asarray([[0.8] * 4, [0.2] * 4]),
        "replacement_fraction": np.asarray([[0.1] * 4, [0.4] * 4]),
        "feedback_surprise": np.asarray([[1.0] * 4, [3.0] * 4]),
        "feedback_uncertainty": np.asarray([[0.2] * 4, [0.6] * 4]),
    }
    result = block_frozen_family_predictors(
        arrays, np.asarray([0, 0, 1, 1]), block_size=2
    )
    assert np.allclose(result["choice_probability"][:2, 0], 0.5)
    posterior_first = 0.8**2 / (0.8**2 + 0.2**2)
    expected_second = posterior_first * 0.8 + (1.0 - posterior_first) * 0.2
    assert np.allclose(result["choice_probability"][2:, 0], expected_second)
    assert np.allclose(
        result["replacement_fraction"][2:],
        posterior_first * 0.1 + (1.0 - posterior_first) * 0.4,
    )
    assert np.allclose(
        result["feedback_uncertainty"][2:],
        posterior_first * 0.2 + (1.0 - posterior_first) * 0.6,
    )


def test_offset_logistic_recovers_increment_direction() -> None:
    rng = np.random.default_rng(8061)
    predictor = rng.normal(size=5000)
    design = np.column_stack([np.ones(predictor.size), predictor])
    probability = 1.0 / (1.0 + np.exp(-(-0.3 + 0.7 * predictor)))
    outcome = rng.binomial(1, probability)
    beta, diagnostics = fit_offset_logistic(
        design,
        outcome,
        np.zeros(predictor.size),
        ridge_penalty=0.01,
    )
    assert diagnostics["success"]
    assert 0.60 < beta[1] < 0.80


def test_student_regression_recovers_replacement_cost_direction() -> None:
    rng = np.random.default_rng(8062)
    predictor = rng.normal(size=3000)
    design = np.column_stack([np.ones(predictor.size), predictor])
    response = -0.2 + 0.4 * predictor + 0.15 * rng.standard_t(5.0, predictor.size)
    beta, scale, diagnostics = fit_student_regression(
        design,
        response,
        degrees_of_freedom=5.0,
        ridge_penalty=0.0001,
        minimum_scale=0.02,
    )
    assert diagnostics["success"]
    assert 0.35 < beta[1] < 0.45
    assert 0.12 < scale < 0.18
