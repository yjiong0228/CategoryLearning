"""Reject partial or misaligned banks before comparing autonomous trajectories."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_execution_autonomous_review import combine_paths


def bank(seeds):
    seeds = np.asarray(seeds)
    return dict(seeds=seeds, categories=np.array([1, 2, 3]),
                choices=np.column_stack([seeds, seeds + 1, seeds + 2]))


def test_complete_bank_aligns_every_path_by_seed():
    result = combine_paths([bank([4, 1]), bank([3, 2])], [1, 2, 3, 4])
    np.testing.assert_array_equal(result['seeds'], [1, 2, 3, 4])
    np.testing.assert_array_equal(result['categories'], [1, 2, 3])
    np.testing.assert_array_equal(result['choices'], [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6]])


@pytest.mark.parametrize('chunks,expected', [
    ([bank([1, 2])], [1, 2, 3]),
    ([bank([1, 2]), bank([2, 3])], [1, 2, 3]),
    ([bank([1, 2])], [1, 1, 2]),
    ([bank([1, 2, 3])], [1, 2]),
    ([], [1, 2]),
])
def test_incomplete_or_duplicate_bank_is_rejected(chunks, expected):
    with pytest.raises(ValueError):
        combine_paths(chunks, expected)


@pytest.mark.parametrize('change', ['categories', 'shape', 'schema'])
def test_misaligned_arrays_are_rejected(change):
    first, second = bank([1]), bank([2])
    if change == 'categories':
        second['categories'] = np.array([2, 1, 3])
    elif change == 'shape':
        second['choices'] = np.ones((2, 3))
    else:
        second['unexpected'] = np.ones((1, 3))
    with pytest.raises((ValueError, AssertionError)):
        combine_paths([first, second], [1, 2])
