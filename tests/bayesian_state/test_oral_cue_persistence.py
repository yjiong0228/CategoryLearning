import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_oral_cue_persistence import (
    extremum_gap, next_opportunity, summarize,
)


def test_opportunity_is_first_future_stimulus_and_never_crosses_boundaries():
    eligible = np.array([True, False, True, True, True])
    sessions = np.array([1, 1, 1, 1, 2]); phases = np.array(['train', 'train', 'train', 'later', 'later'])
    assert next_opportunity(eligible, sessions, phases, 0) == (2, 'observed')
    assert next_opportunity(eligible, sessions, phases, 2) == (None, 'phase_boundary')
    assert next_opportunity(eligible, sessions, phases, 3) == (None, 'session_boundary')
    assert next_opportunity(eligible, sessions, phases, 4) == (None, 'sequence_end')
    assert next_opportunity(np.array([True, False, False, True, True]), sessions, phases, 0) == (None, 'phase_boundary')


def test_extrema_keep_ties_and_feature_permutations():
    x = np.array([[.8, .8, .2, .1], [.7, .8, .2, .1], [.9, .8, .2, .1]])
    gap = extremum_gap(x, 0, 'max')
    np.testing.assert_allclose(gap, [0, -.1, .1])
    np.testing.assert_array_equal(extremum_gap(x[:, [1, 2, 0, 3]], 2, 'max'), gap)
    np.testing.assert_array_equal(extremum_gap(-x, 0, 'min'), gap)
    for dimension, direction in [(4, 'max'), (0, 'bad')]:
        with pytest.raises(ValueError): extremum_gap(x, dimension, direction)


def test_censoring_is_excluded_from_recurrence_denominator():
    frame = pd.DataFrame(dict(subject=[207]*3, status=['observed', 'observed', 'phase_boundary'],
        same_wording=[True, False, np.nan], mentions_literal_phrase=[True, True, np.nan],
        same_choice=[False, True, np.nan], same_wording_and_choice=[False, False, np.nan], gap_trials=[2, 4, np.nan]))
    row = summarize(frame, ['subject']).iloc[0]
    assert row.origins == 3 and row.opportunities == 2 and row.censored == 1
    assert row.same_wording_fraction == .5 and row.mentions_literal_phrase_fraction == 1.
    assert row.median_gap_trials == 3
