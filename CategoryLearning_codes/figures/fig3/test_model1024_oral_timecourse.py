import numpy as np

from CategoryLearning_codes.figures.fig3.model1024_oral_timecourse import rolling


def test_missing_reports_keep_their_original_trial_positions():
    values = np.array([1., np.nan, 0., np.nan, np.nan, 1.])
    mean, count = rolling(values, np.isfinite(values), 3, 2)
    np.testing.assert_allclose(mean, [np.nan, np.nan, .5, np.nan, np.nan, np.nan], equal_nan=True)
    np.testing.assert_array_equal(count, [1, 1, 2, 1, 1, 1])


def test_display_at_a_trial_does_not_average_future_scores():
    original = np.array([0., .2, .4, .6, 1.])
    changed = original.copy()
    changed[3:] = 0
    a, _ = rolling(original, np.ones(5, bool), 3, 1)
    b, _ = rolling(changed, np.ones(5, bool), 3, 1)
    np.testing.assert_array_equal(a[:3], b[:3])
    assert a[-1] != b[-1]


def test_zero_support_is_scored_and_low_count_windows_are_missing():
    mean, count = rolling(np.array([0., 0., 1., 1.]), np.ones(4, bool), 3, 2)
    np.testing.assert_allclose(mean, [np.nan, 0, 1/3, 2/3], equal_nan=True)
    np.testing.assert_array_equal(count, [1, 2, 3, 3])
