import numpy as np

from CategoryLearning_codes.figures.fig3.model1024_oral_development import numeric_pair, rolling


def test_rolling_keeps_trial_gaps_and_minimum_count():
    raw = np.array([.1, .3, np.nan, .9, np.nan, np.nan, .4])
    got = rolling(raw, width=3, minimum=2)
    expected = np.array([np.nan, .2, .2, .6, np.nan, np.nan, np.nan])
    np.testing.assert_allclose(got, expected, equal_nan=True)


def test_pair_relation_excludes_other_pairs_bands_and_sums():
    assert numeric_pair('facet:[[0.0,1.0,-1.0,0.0,0.0]]', 1, 2)
    assert not numeric_pair('facet:[[0.0,1.0,1.0,0.0,1.0]]', 1, 2)
    assert not numeric_pair('facet:[[0.0,1.0,0.0,-1.0,0.0]]', 1, 2)
    assert not numeric_pair('facet:[[0.0,1.0,-1.0,0.0,0.1]]', 1, 2)
