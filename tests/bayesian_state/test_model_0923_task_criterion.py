import numpy as np

from src.Bayesian_state.evaluation.model_0923_task_criterion import block_criterion


def test_all_categories_must_pass_even_if_aggregate_accuracy_is_high():
    categories = np.array([1] * 19 + [2])
    feedback = np.array([[1] * 19 + [0]])
    accuracy, reached = block_criterion(feedback, categories, np.zeros(20), 2, .9)
    assert feedback.mean() > .9
    np.testing.assert_array_equal(accuracy, [[1, 0]])
    assert not reached[0]


def test_ambiguity_is_excluded_and_half_credit_is_not_exact_correct():
    categories = np.array([1, 1, 2, 2])
    feedback = np.array([[1, 0, 1, 0], [1, 0, .5, 0]])
    accuracy, reached = block_criterion(feedback, categories, np.array([0, 1, 0, 1]), 2, .9)
    np.testing.assert_array_equal(accuracy, [[1, 1], [1, 0]])
    np.testing.assert_array_equal(reached, [True, False])


def test_absent_category_does_not_pass():
    _, reached = block_criterion(np.ones((2, 3)), np.ones(3, dtype=int), np.zeros(3), 2, .9)
    assert not reached.any()


def test_first_task_trial_enters_block_criterion():
    accuracy, reached = block_criterion(np.array([[0, 1, 1]]), np.array([1, 1, 2]), np.zeros(3), 2, .9)
    assert accuracy[0, 0] == .5
    assert not reached[0]
