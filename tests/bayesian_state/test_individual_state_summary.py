"""Endpoint evidence and oral comparison definitions have explicit references."""
import numpy as np

from src.Bayesian_state.workflows.analysis.model_1024_individual_state_summary import (
    FIELDS, pool_arrays, discrepancy)
from src.Bayesian_state.workflows.analysis.model_1024_individual_oral_alignment import (
    uniform_basis, static_basis, constraint_text, tokens_text, subject_features)
from src.Bayesian_state.workflows.analysis.model_1024_individual_state_distributions import (
    weighted_histogram, weighted_quantiles)


def test_full_history_pool_uses_future_evidence_filtering_uses_prefix():
    banks = []
    for r in (0, 1):
        item = {}
        for mode in ('filtered', 'full'):
            for key in FIELDS:
                shape = (2, 2, 1) if key=='component' else (2, 3) if key=='source' else (2, 2) if key in (
                    'active', 'belief', 'belief_second', 'execution', 'active_precision', 'active_precision_second') else (2,)
                item[mode+'_'+key] = np.full(shape, float(r))
        item.update(component_counts=np.ones((2, 2), int),
                    full_unique_ancestors=np.array([1, 2]), full_ancestral_ess=np.array([1., 1.5]))
        banks.append(item)
    logz = np.log([[.8, .1], [.2, .9]]).cumsum(axis=1)
    result, _ = pool_arrays(banks, logz)
    np.testing.assert_allclose(result['filtered_target_present'], [.2, .18/.26])
    np.testing.assert_allclose(result['full_target_present'], [.18/.26, .18/.26])
    np.testing.assert_allclose(result['terminal_system_weights'], [.08/.26, .18/.26])


def test_distribution_discrepancy_is_not_diluted_by_unused_catalogue_rules():
    a, b = {}, {}
    for key in ('target_present', 'target_belief', 'target_high_belief', 'target_execution'):
        a['full_'+key] = b['full_'+key] = np.zeros(3)
    for key in ('belief', 'execution'):
        x = np.zeros((3, 152)); y = x.copy(); x[:, 0] = 1; y[:, 151] = 1
        a['full_'+key] = x; b['full_'+key] = y
    result = discrepancy(a, b, 'full')
    assert result['q95'] == 1.
    assert result['mean'] == 1./3
    assert result['target_present_q95'] == 0.
    assert result['belief_mean'] == result['execution_q95'] == 1.
    assert result['belief_whole'] == 1.


def test_oral_comparison_respects_current_category_and_preserves_total_mass():
    choices = np.array([1, 2, 1]); counts = np.array([[1, 2], [2, 1]])
    basis = uniform_basis(choices, counts, 2)
    np.testing.assert_allclose(basis.sum(axis=(1, 2)), 1.)
    np.testing.assert_array_equal(basis[0], [[.5, 0.], [.25, .25]])
    np.testing.assert_array_equal(basis[1], [[.25, .25], [.5, 0.]])
    varied = basis.copy(); varied[2] = [[1., 0.], [0., 0.]]
    static = static_basis(varied, choices)
    np.testing.assert_allclose(static[0], (varied[0]+varied[2])/2)
    np.testing.assert_array_equal(static[0], static[2])
    np.testing.assert_array_equal(static[1], varied[1])


def test_review_text_preserves_inequality_direction_and_symbolic_references():
    assert constraint_text(np.array([[-1., 0, 0, 0], [0, 1., -1., 0]]),
                           np.array([-.5, 0.])) == '脖子≥0.5 且 头≤腿'
    assert tokens_text('["body:2:lt"]') == '腿短于躯干（参照未定）'


def test_histograms_keep_endpoint_atoms_and_joint_presence_mass():
    values = np.array([[0., .5, 1.], [1., .5, 0.]])
    weights = np.array([[.2, .3, .5], [.6, .3, .1]])
    edges = np.array([0., .5, 1.])
    np.testing.assert_allclose(weighted_histogram(values, weights, edges), [[.2, .8], [.1, .9]])
    mask = values > 0
    np.testing.assert_allclose(weighted_histogram(values, weights, edges, mask), [[0., .8], [0., .9]])
    np.testing.assert_array_equal(weighted_quantiles(np.array([1., 0., .5]), np.array([.1, .8, .1])), [0., 0., 1.])


def test_oral_review_uses_subject_specific_counterbalanced_dimensions():
    import pandas as pd
    frame = pd.DataFrame(dict(feature1_name=['head'], feature2_name=['neck'],
                              feature3_name=['tail'], feature4_name=['leg']))
    features = subject_features(frame)
    assert tokens_text('["facet:[[0,-1,0,0,-0.5],[1,0,0,0,0.5]]"]', features) == '脖子≥0.5 且 头≤0.5'
