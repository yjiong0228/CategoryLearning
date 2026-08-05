from __future__ import annotations

import numpy as np

from scripts.run_model_0804_regeneration_recovery import (
    _ensemble_nll,
    _monotonic_axes,
    _rate_summary,
)
from src.Bayesian_state.utils.model_0804 import Model0804Parameters
from src.Bayesian_state.utils.model_0804_recovery import (
    infer_correct_choices,
    simulate_model0804_choices,
)
from tests.test_model_0804 import _fixture


def _correct_choices(choices: np.ndarray, feedback: np.ndarray) -> np.ndarray:
    return np.where(feedback >= 0.5, choices, 1 - choices).astype(int)


def test_infer_correct_choices_recovers_binary_task_answer() -> None:
    _, choices, feedback, _, _ = _fixture()
    expected = _correct_choices(choices, feedback)
    np.testing.assert_array_equal(
        infer_correct_choices(choices, feedback), expected
    )


def test_simulation_is_reproducible_and_feedback_is_deterministic() -> None:
    q_values, choices, feedback, prior, kernels = _fixture()
    parameters = Model0804Parameters(
        0.70, 0.40, 2.0, 0.15, 0.35, lapse=0.02, rho=0.05
    )
    common = dict(
        q_values=q_values,
        correct_choices=_correct_choices(choices, feedback),
        prior=prior,
        kernels=kernels,
        model_id="FA2R",
        parameters=parameters,
        capacity=2,
        simulation_seed=804,
    )
    left = simulate_model0804_choices(**common)
    right = simulate_model0804_choices(**common)
    np.testing.assert_array_equal(left.choices, right.choices)
    np.testing.assert_array_equal(left.feedback, right.feedback)
    np.testing.assert_array_equal(left.active, right.active)
    np.testing.assert_array_equal(left.regenerated, right.regenerated)
    np.testing.assert_array_equal(
        left.feedback,
        (left.choices == left.correct_choices).astype(float),
    )
    np.testing.assert_allclose(left.choice_probabilities.sum(axis=1), 1.0)
    np.testing.assert_array_equal(left.active.sum(axis=1), 2)


def test_rho_one_regenerates_every_transition() -> None:
    q_values, choices, feedback, prior, kernels = _fixture()
    simulation = simulate_model0804_choices(
        q_values,
        _correct_choices(choices, feedback),
        prior,
        kernels,
        model_id="FA2R",
        parameters=Model0804Parameters(
            0.70, 0.40, 2.0, 0.15, 0.35, lapse=0.02, rho=1.0
        ),
        capacity=2,
        simulation_seed=805,
    )
    assert not simulation.regenerated[0]
    assert np.all(simulation.regenerated[1:])
    np.testing.assert_array_equal(simulation.replacement_count[1:], 2)


def test_fa2r_rho_zero_matches_fa2_simulation_under_common_randomness() -> None:
    q_values, choices, feedback, prior, kernels = _fixture()
    parameters = Model0804Parameters(
        0.70, 0.40, 2.0, 0.15, 0.35, lapse=0.02, rho=0.0
    )
    common = dict(
        q_values=q_values,
        correct_choices=_correct_choices(choices, feedback),
        prior=prior,
        kernels=kernels,
        parameters=parameters,
        capacity=2,
        simulation_seed=806,
    )
    fa2 = simulate_model0804_choices(model_id="FA2", **common)
    fa2r = simulate_model0804_choices(model_id="FA2R", **common)
    np.testing.assert_array_equal(fa2.choices, fa2r.choices)
    np.testing.assert_array_equal(fa2.feedback, fa2r.feedback)
    np.testing.assert_array_equal(fa2.active, fa2r.active)
    np.testing.assert_array_equal(fa2.replacement_count, fa2r.replacement_count)
    np.testing.assert_array_equal(
        fa2.choice_probabilities, fa2r.choice_probabilities
    )


def test_likelihood_ensemble_averages_likelihood_not_nll() -> None:
    values = np.asarray([2.0, 4.0])
    expected = -np.log(np.mean(np.exp(-values)))
    assert np.isclose(_ensemble_nll(values), expected)
    assert not np.isclose(_ensemble_nll(values), np.mean(values))


def test_replicated_monotonic_axis_uses_all_rows() -> None:
    rows = []
    estimates = {
        "rho_zero": [0.00, 0.01],
        "center": [0.02, 0.03],
        "rho_high": [0.04, 0.05],
    }
    for scenario_id, values in estimates.items():
        for replicate, value in enumerate(values):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "subject_id": 101,
                    "replicate": replicate,
                    "soft_parameter_mean_within_confirmed_set": {
                        "rho": value,
                        "m": 0.15,
                        "g": 0.35,
                        "lapse": 0.02,
                    },
                }
            )
    rho = _monotonic_axes(rows)["rho"]
    assert rho["passed"]
    assert rho["paired_comparison_count"] == 2
    assert rho["paired_order_fraction"] == 1.0
    np.testing.assert_allclose(
        rho["soft_estimate_means"], [0.005, 0.025, 0.045]
    )


def test_rate_summary_reports_wilson_interval() -> None:
    summary = _rate_summary([True, True, False, True])
    assert summary["value"] == 0.75
    assert summary["successes"] == 3
    assert summary["total"] == 4
    lower, upper = summary["wilson_95_interval"]
    assert 0.0 < lower < summary["value"] < upper < 1.0
