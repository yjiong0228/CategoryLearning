"""Scope selection uses the same candidates and pre-choice probability arrays."""
import numpy as np
import pytest

from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
from src.Bayesian_state.workflows.analysis.model_0923_estimation_scope import (
    common_bank, promoted_banks, score_scopes,
)


def test_scope_scores_pool_before_logs_and_keep_the_original_boundary():
    p = np.array([[.2, .6, .9, .9], [.8, .4, .7, .7]])
    scores = score_scopes(p, 2)
    assert scores['prefix_nll'] == pytest.approx(-np.log(.5))
    assert scores['later_nll'] == pytest.approx(-np.log(.8))
    assert scores['full_nll'] == pytest.approx((-np.log(.5)-np.log(.8))/2)
    assert scores['prefix_nll'] != pytest.approx(-np.log(p[:, :2]).mean())
    assert scores['full_nll_mc_se'] > 0


@pytest.mark.parametrize('p,cut', [(np.ones((1, 4)), 2), (np.ones((2, 4)), 4),
                                  (np.ones((2, 4))*1.01, 2), (np.full((2, 4), np.nan), 2)])
def test_invalid_filter_arrays_or_boundaries_rejected(p, cut):
    with pytest.raises(ValueError):score_scopes(p, cut)


def test_candidate_union_is_shared_and_does_not_select_by_old_scores():
    points = [candidate(dict(value=value), 'fixture') for value in range(3)]
    groups = [dict(subject=308, model='POOL', rows=points[:2]),
              dict(subject=308, model='SEARCH', rows=points[1:])]
    banks = common_bank(groups, [308], ['POOL', 'SEARCH'])
    assert banks[308, 'POOL'] == banks[308, 'SEARCH']
    assert {r['id'] for r in banks[308, 'POOL']} == {r['id'] for r in points}
    banks[308, 'POOL'][0]['hyperparams']['value'] = -1
    assert all(r['hyperparams']['value'] >= 0 for r in banks[308, 'SEARCH'])


def test_shortlist_keeps_each_objective_and_old_nominees_in_both_arms():
    points = [dict(candidate(dict(value=i), 'fixture'), prefix_nll=p, full_nll=f)
              for i, (p, f) in enumerate([(1, 4), (4, 1), (3, 3), (5, 5)])]
    scored = {(308, arm): points for arm in ['POOL', 'SEARCH']}
    result = promoted_banks(scored, {308: [points[-1]]}, 1)
    assert result[308, 'POOL'] == result[308, 'SEARCH']
    assert {r['id'] for r in result[308, 'POOL']} == {points[i]['id'] for i in (0, 1, 3)}
