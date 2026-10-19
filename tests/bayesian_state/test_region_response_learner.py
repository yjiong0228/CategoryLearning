import numpy as np
import pytest

from src.Bayesian_state.hypothesis_space.observation_model.region_response import (
    region_response_feedback_gradient, region_response_probabilities,
)
from src.Bayesian_state.model.region_response_learner import RegionResponseLearner


@pytest.mark.parametrize('feedback', [0., 1.])
def test_feedback_gradient_matches_independent_finite_difference(feedback):
    geometry = np.array([.2, .5, .3])
    logits = np.array([[.1, -.4, .6, .2], [-.7, .4, .1, .3], [.8, -.1, .2, -.5]])
    log_event, gradient = region_response_feedback_gradient(geometry, logits, 3, feedback)

    def objective(values):
        a = np.exp(values); a /= a.sum(1, keepdims=True)
        prediction = geometry @ a
        return np.log(prediction[2] if feedback else prediction[[0, 1, 3]].sum())

    np.testing.assert_allclose(log_event, objective(logits), atol=1e-12, rtol=0)
    numeric = np.empty_like(logits); step = 1e-5
    for index in np.ndindex(logits.shape):
        positive = logits.copy(); negative = logits.copy()
        positive[index] += step; negative[index] -= step
        numeric[index] = (objective(positive) - objective(negative)) / (2*step)
    np.testing.assert_allclose(gradient, numeric, atol=3e-10, rtol=0)
    np.testing.assert_allclose(gradient.sum(-1), 0., atol=1e-12, rtol=0)


def test_multiple_regions_can_learn_same_response_and_unseen_row_stays_uniform():
    learner = RegionResponseLearner(4, 4, learning_rate=.5)
    for _ in range(80):
        for region in range(3):
            learner.update(np.eye(4)[region], 3, 1.)
    assert np.all(learner.associations[:3, 2] > .95)
    np.testing.assert_array_equal(learner.associations[3], np.full(4, .25))
    np.testing.assert_allclose(learner.associations.sum(1), 1., atol=1e-12, rtol=0)


def test_negative_feedback_does_not_reveal_the_true_alternative():
    learner = RegionResponseLearner(4, 4, learning_rate=1.)
    before = learner.predict(np.eye(4)[1]); result = learner.update(np.eye(4)[1], 2, 0.)
    np.testing.assert_array_equal(before, np.full(4, .25))
    np.testing.assert_allclose(result, np.log(.75))
    after = learner.predict(np.eye(4)[1])
    assert after[1] < .25
    np.testing.assert_allclose(after[[0, 2, 3]], np.repeat(after[0], 3))
    np.testing.assert_array_equal(learner.associations[[0, 2, 3]], np.full((3, 4), .25))


def test_prediction_precedes_feedback_and_returned_rows_do_not_mutate_state():
    learner = RegionResponseLearner(2, 2, learning_rate=.7)
    x = np.array([.8, .2]); before = learner.predict(x)
    copy = learner.associations; copy[:] = 0
    np.testing.assert_array_equal(learner.predict(x), before)
    log_event = learner.update(x, 2, 1.)
    np.testing.assert_allclose(log_event, np.log(before[1]))
    assert learner.predict(x)[1] > before[1]


def test_extremely_unlikely_feedback_keeps_finite_log_evidence_and_gradient():
    logits = np.array([[1000., -1000., -1000., -1000.]])
    evidence, gradient = region_response_feedback_gradient(np.array([1.]), logits, 1, 0.)
    np.testing.assert_allclose(evidence, -2000+np.log(3), atol=1e-10)
    np.testing.assert_allclose(gradient, [[-1., 1/3, 1/3, 1/3]], atol=1e-10)


def test_batched_rules_match_independent_rule_updates():
    logits = np.arange(24, dtype=float).reshape(2, 3, 4) / 10
    geometry = np.array([[.1, .2, .7], [.6, .3, .1]])
    evidence, gradient = region_response_feedback_gradient(geometry, logits, 2, 1.)
    for i in range(2):
        single = region_response_feedback_gradient(geometry[i], logits[i], 2, 1.)
        np.testing.assert_allclose(evidence[i], single[0]); np.testing.assert_allclose(gradient[i], single[1])
    np.testing.assert_allclose(region_response_probabilities(geometry, logits).sum(1), 1.)


@pytest.mark.parametrize('geometry,choice,feedback', [
    ([.2, .2], 1, 1.), ([-.1, 1.1], 1, 0.), ([np.nan, 1.], 1, 1.),
    ([.5, .5], 0, 1.), ([.5, .5], True, 1.), ([.5, .5], 1, .5),
])
def test_invalid_geometry_or_unsupported_feedback_is_rejected(geometry, choice, feedback):
    with pytest.raises(ValueError):
        region_response_feedback_gradient(np.array(geometry), np.zeros((2, 4)), choice, feedback)
