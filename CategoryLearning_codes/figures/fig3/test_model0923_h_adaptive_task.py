import numpy as np
import pandas as pd

from CategoryLearning_codes.figures.fig3.model0923_h_adaptive_task import stopping_curve


def test_first_pass_curve_keeps_censored_paths_in_denominator():
    paths = pd.DataFrame(dict(seed=[1, 2, 3, 4], blocks=[2, 4, 5, 5], passed=[True, True, False, False]))
    np.testing.assert_array_equal(stopping_curve(paths, 5), [0, 0, .25, .25, .5, .5])


def test_all_censored_paths_produce_zero_empirical_pass_fraction():
    paths = pd.DataFrame(dict(seed=[1, 2], blocks=[5, 5], passed=[False, False]))
    np.testing.assert_array_equal(stopping_curve(paths, 5), np.zeros(6))
