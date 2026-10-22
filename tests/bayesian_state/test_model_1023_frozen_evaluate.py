"""Protect matched filter/path seeds and disjoint chunk boundaries."""
from copy import deepcopy

import pytest
import numpy as np

from src.Bayesian_state.workflows.analysis.model_1023_frozen_evaluate import paired_seed_jobs
from src.Bayesian_state.evaluation.model_1023_frozen_evaluate import score_arrays


def test_paired_seed_remapping_preserves_every_path_and_filter_once():
    groups = {'conditional': [({'seeds': [1, 2, 3]}, [{'seeds': [1]}, {'seeds': [2]}, {'seeds': [3]}])],
              'autonomous': [({'seeds': [4, 5, 6, 7, 8]}, [{'seeds': [4, 5]}, {'seeds': [6, 7]}, {'seeds': [8]}])]}
    reference = {'conditional': [31, 12, 19], 'autonomous': [91, 23, 73, 42, 51]}
    frozen = deepcopy(reference)
    paired_seed_jobs(groups, reference)
    for role, records in groups.items():
        job, chunks = records[0]
        assert job['seeds'] == reference[role]
        assert [s for part in chunks for s in part['seeds']] == reference[role]
    assert reference == frozen


@pytest.mark.parametrize('seeds', [[1, 1], [1, 2, 3]])
def test_repeated_or_different_size_banks_are_rejected(seeds):
    groups = {'conditional': [({'seeds': [9, 10]}, [{'seeds': [9]}, {'seeds': [10]}])]}
    with pytest.raises(ValueError):
        paired_seed_jobs(groups, {'conditional': seeds})


def example_arrays():
    length = 7
    c = dict(choices=np.array([0, 1, 0, 1, 1, 0, 1]), categories=np.zeros(length, int),
             seeds=np.array([1, 2]), probabilities=np.full((2, length, 2), .5),
             observed_probability=np.full((2, length), .5), correct_probability=np.full((2, length), .5))
    a = dict(choices=np.zeros((2, length), int), categories=c['categories'], seeds=np.array([3, 4]),
             probabilities=c['probabilities'].copy())
    for key in ('present', 'contribution', 'beta', 'search', 'proposed', 'rejected',
                'target_capacity_dropped', 'active_count'):
        a[key] = np.zeros((2, length))
    a['present'][0, 0] = 1
    a['beta'][0, 0] = 8
    return c, a


def test_realized_paths_cut_and_conditional_precision_are_distinct():
    c, a = example_arrays()
    scores, curves, paths, states, _ = score_arrays(c, a, 3, 3)
    assert [r['trials'] for r in scores] == [3, 4, 7]
    assert [r['trial'] for r in curves] == [3, 6, 7]
    assert len(paths) == 6
    assert all(r['autonomous'] == 1 for r in curves)
    assert all(r['conditional'] == .5 for r in curves)
    assert states[0]['target_precision_given_presence'] == 8
    assert states[0]['target_precision_observations'] == 1
    assert np.isnan(states[1]['target_precision_given_presence'])


def test_inactive_precision_is_rejected():
    c, a = example_arrays()
    a['beta'][1, 3] = 2
    with pytest.raises(ValueError, match='Inactive target precision'):
        score_arrays(c, a, 3, 3)
