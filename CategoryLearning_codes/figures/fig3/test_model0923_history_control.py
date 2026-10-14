"""Check trial boundaries and weighting used by the history-control plots."""
import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig3.model0923_history_control import window_table


def test_windows_exclude_initialization_and_preserve_short_tail():
    frame = pd.DataFrame(dict(valid_trial=np.arange(1, 9), scored=[False]+[True]*7,
                              A_correct_probability=[.99, .1, .2, .3, .4, .5, .6, .7]))
    result = window_table(frame, 3)
    np.testing.assert_array_equal(result.trials, [3, 3, 1])
    np.testing.assert_array_equal(result.first_valid_trial, [2, 5, 8])
    np.testing.assert_array_equal(result.last_valid_trial, [4, 7, 8])
    np.testing.assert_allclose(result.A_correct_probability, [.2, .5, .7])
    assert np.average(result.A_correct_probability, weights=result.trials) == pytest.approx(.4)


def test_windows_reject_additional_trial_exclusion():
    frame = pd.DataFrame(dict(valid_trial=[1, 2, 3], scored=[False, False, True]))
    with pytest.raises(ValueError, match="initialization"):
        window_table(frame, 2)
