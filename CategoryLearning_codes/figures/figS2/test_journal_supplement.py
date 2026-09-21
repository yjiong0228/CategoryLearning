"""Checks for the probability-distance definitions used by the supplement."""
import numpy as np
import pytest

from CategoryLearning_codes.figures.figS2.build_journal_supplement import candidate_distance, pairwise_distance


def test_distinct_replay_pairs_are_weighted_equally():
    # Three replay pairs have mean-trial TV distances .25, .5 and .25.
    values = np.array([[[1., 0.], [0., 1.]],
                       [[.5, .5], [0., 1.]],
                       [[0., 1.], [0., 1.]]])
    assert pairwise_distance(values) == pytest.approx(1 / 3)
    assert pairwise_distance(values[[2, 0, 1], ::-1]) == pytest.approx(1 / 3)


def test_candidate_distance_compares_means_without_pairing_replays():
    selected = np.array([[[1., 0.], [0., 1.]], [[.5, .5], [0., 1.]]])
    alternative = np.array([[[0., 1.], [0., 1.]]])
    assert candidate_distance(selected, alternative) == pytest.approx(.375)
    assert candidate_distance(alternative, selected) == pytest.approx(.375)
    assert candidate_distance(selected, selected) == 0


@pytest.mark.parametrize('invalid', [
    np.array([[[np.nan, 0.]], [[1., 0.]]]),
    np.array([[[-.2, 1.2]], [[1., 0.]]]),
    np.array([[[.2, .2]], [[1., 0.]]]),
])
def test_invalid_probabilities_are_not_dropped_or_normalized(invalid):
    with pytest.raises((ValueError, AssertionError)):
        pairwise_distance(invalid)


def test_distances_reject_missing_replicates_and_unaligned_state_spaces():
    with pytest.raises(ValueError):
        pairwise_distance(np.array([[[.5, .5]]]))
    with pytest.raises(ValueError):
        candidate_distance(np.array([[[.5, .5]]]), np.array([[[1 / 3] * 3]]))
