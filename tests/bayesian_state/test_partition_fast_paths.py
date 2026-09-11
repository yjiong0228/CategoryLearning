"""Regression checks for avoiding work that does not affect probabilities."""

import numpy as np
import pytest

from src.Bayesian_state.hypothesis_space import ContinuousPartition


@pytest.mark.parametrize("n_cats", [2, 4])
def test_full_feedback_does_not_require_family_geometry(n_cats, monkeypatch):
    partition = ContinuousPartition(4, n_cats, similarity_n_samples=8)

    def unexpected_neighbors(*args, **kwargs):
        raise AssertionError("full feedback should not evaluate family geometry")

    monkeypatch.setattr(partition, "get_category_neighbors", unexpected_neighbors)
    prob = np.tile(np.array([0.3, 0.8]) / (n_cats - 1), (n_cats, 1))
    prob[0] = [0.7, 0.2]
    actual = partition._category_feedback_likelihood(
        0, prob, np.array([0, 0]), np.array([1., 0.])
    )
    np.testing.assert_array_equal(actual, [0.7, 0.8])


def test_mixed_partial_feedback_keeps_neighbor_mass():
    partition = ContinuousPartition(4, 4, similarity_n_samples=8)
    # Rule 0 has category-0 neighbors 1 and 2, carrying mass 0.2 + 0.3.
    prob = np.tile(np.array([0.1, 0.2, 0.3, 0.4])[:, None], (1, 3))
    actual = partition._category_feedback_likelihood(
        0, prob, np.array([0, 0, 0]), np.array([1., 0.5, 0.])
    )
    np.testing.assert_array_equal(actual, [0.1, 0.5, 0.9])


def test_single_component_distance_needs_no_stack(monkeypatch):
    partition = ContinuousPartition(4, 2, similarity_n_samples=8)
    geometry = partition.boundary_geometry
    category = next(c for h in partition.hypothesis_space for c in h.categories
                    if len(c.components) == 1)
    stimuli = np.array([[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]])
    expected = geometry.distances_to_polytope(stimuli, category.components[0])

    def unexpected_stack(*args, **kwargs):
        raise AssertionError("one region does not require a stacked reduction")

    monkeypatch.setattr(np, "stack", unexpected_stack)
    np.testing.assert_array_equal(geometry.distances_to_category(stimuli, category), expected)
