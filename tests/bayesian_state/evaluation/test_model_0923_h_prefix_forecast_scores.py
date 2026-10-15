"""Score uncertainty keeps branches from one filter in one MC cluster."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_h_prefix_forecast import filter_cluster_jackknife
from src.Bayesian_state.evaluation.model_0923_h_validation import jackknife_se


def test_cluster_deletion_matches_known_three_filter_distribution():
    samples = np.array([[0.], [0.], [1.], [1.], [1.], [1.]])
    labels, scores = filter_cluster_jackknife(samples, np.array([0.]), np.array([11, 11, 22, 22, 33, 33]))
    np.testing.assert_array_equal(labels, [11, 22, 33])
    np.testing.assert_allclose(scores['energy'], [1., .25, .25])
    np.testing.assert_array_equal(scores['energy'], scores['crps'])
    assert jackknife_se(scores['energy']) == pytest.approx(.5)


@pytest.mark.parametrize('labels', [[1, 1, 1], [1, 1, 2], [1, 2]])
def test_invalid_filter_groups_are_rejected(labels):
    with pytest.raises(ValueError):
        filter_cluster_jackknife(np.zeros((3, 1)), np.zeros(1), np.asarray(labels))
