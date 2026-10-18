"""Cumulative discovery plots retain paths that never find the target rule."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from CategoryLearning_codes.figures.fig3.model0923_estimation_scope import ecdf


def test_undiscovered_paths_remain_in_discovery_denominator():
    fig, ax = plt.subplots()
    ecdf(ax, np.array([2., 4., np.inf]), upper=8)
    x, y = ax.lines[0].get_data()
    assert x[-1] == 8 and y[-1] == 2/3
    assert y[np.searchsorted(x, 2, side='right')-1] == 1/3
    assert y[np.searchsorted(x, 1, side='right')-1] == 0
    plt.close(fig)


def test_absent_discoveries_plot_zero_through_the_task_end():
    fig, ax = plt.subplots()
    ecdf(ax, np.array([np.inf, np.inf]), upper=8)
    x, y = ax.lines[0].get_data()
    assert x[0] == 0 and x[-1] == 8
    assert np.all(y == 0)
    plt.close(fig)
