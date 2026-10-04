"""Binary-feedback precision learning is neutral at task chance."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.model.modules.beta import BetaModule


class _FixedPredictionPartition:
    """Supply controlled rule predictions without updating the rule geometry."""

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.n_cats = self.probabilities.shape[1]

    def get_category_probabilities(self, *, hypo, data, beta, distance_mode):
        return self.probabilities[hypo, :, None].copy()


def _beta(choice_probabilities, n_categories=4, *, choice=1, **kwargs):
    probabilities = np.asarray([
        [p] + [(1.0 - p) / (n_categories - 1)] * (n_categories - 1)
        for p in choice_probabilities
    ])
    probabilities = np.roll(probabilities, choice - 1, axis=1)
    engine = SimpleNamespace(
        set_size=len(probabilities),
        partition=_FixedPredictionPartition(probabilities),
        distance_mode="boundary",
        hypotheses_mask=np.ones(len(probabilities)),
        get_module=lambda role: None,
    )
    module = BetaModule(
        engine,
        **{
            "beta_init": 5.0,
            "beta_min": 0.1,
            "beta_max": 25.0,
            "increase_rate": 0.04,
            "decrease_rate": 0.15,
            "beta_update_mode": "probabilistic_feedback",
            "use_prior_scaling": False,
            **kwargs,
        },
    )
    return engine, module


@pytest.mark.parametrize("n_categories", [2, 3, 4])
@pytest.mark.parametrize("lapse", [0.0, 0.2])
def test_uniform_predictions_do_not_drift_under_correct_or_error_feedback(n_categories, lapse):
    _, module = _beta(
        [1.0 / n_categories] * 2,
        n_categories,
        probabilistic_feedback_lapse=lapse,
    )
    for feedback in [1.0, 0.0, 0.0, 1.0] * 3:
        module.update_beta(np.zeros(4), 1, feedback)
        np.testing.assert_allclose(module.beta, [5.0, 5.0], rtol=0, atol=1e-14)


@pytest.mark.parametrize("choice", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "feedback,expected",
    [(1.0, [4.55, 5.0, 5.16]), (0.0, [5.48, 5.0, 4.85])],
)
def test_four_category_support_is_centered_on_the_observed_feedback_event(choice, feedback, expected):
    # A .4 prediction beats four-choice chance although it is below .5.
    # Predicting "not the chosen answer" with .6 is below error chance (.75).
    _, module = _beta([0.1, 0.25, 0.4], choice=choice)

    module.update_beta(np.zeros(4), choice, feedback)

    np.testing.assert_allclose(module.beta, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize(
    "feedback,expected",
    [(1.0, [4.7, 5.0, 5.32]), (0.0, [5.32, 5.0, 4.7])],
)
def test_binary_feedback_retains_its_original_updates(feedback, expected):
    _, module = _beta([0.3, 0.5, 0.7], n_categories=2)

    module.update_beta(np.zeros(4), 1, feedback)

    np.testing.assert_allclose(module.beta, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("feedback", [0.0, 1.0])
def test_nonzero_lapse_attenuates_support_without_moving_the_neutral_point(feedback):
    _, sharp = _beta([0.1, 0.25, 0.4])
    _, softened = _beta([0.1, 0.25, 0.4], probabilistic_feedback_lapse=0.5)

    sharp.update_beta(np.zeros(4), 1, feedback)
    softened.update_beta(np.zeros(4), 1, feedback)

    np.testing.assert_allclose(softened.beta - 5.0, 0.5 * (sharp.beta - 5.0), atol=1e-14)


@pytest.mark.parametrize("feedback", [0.0, 1.0])
def test_chance_centering_preserves_active_scope_and_bounds(feedback):
    engine, module = _beta(
        [0.0, 1.0, 0.25, 0.9],
        increase_rate=1.0,
        decrease_rate=1.0,
    )
    engine.hypotheses_mask[-1] = 0.0

    module.update_beta(np.zeros(4), 1, feedback)

    assert np.isfinite(module.beta).all()
    assert (module.beta[:3] >= 0.1).all()
    assert (module.beta[:3] <= 25.0).all()
    assert module.beta[2] == 5.0
    assert module.beta[3] == 0.0
    extremes = [25.0, 0.1] if feedback == 0.0 else [0.1, 25.0]
    np.testing.assert_allclose(module.beta[:2], extremes, rtol=0, atol=1e-9)
