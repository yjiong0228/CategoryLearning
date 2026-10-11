"""Complete-report scores preserve coverage, zero mass and numerical ties."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.evaluation.oral.generative_scoring import (
    score_report_predictions,
    summarize_report_scores,
)
from src.Bayesian_state.evaluation.oral.structure_0923 import encode_report
from src.Bayesian_state.model.modules.readout.report_space import report_code


def _example():
    codes = ['A', 'B', 'C', 'D', 'E']
    probability = np.array([
        [.4, .25, .25, .1, 0.],
        [.25, .25, .25, .25, 0.],
        [.4, .25, .25, .1, 0.],
        [.4, .25, .25, .1, 0.],
        [.4, .25, .25, .1, 0.],
        [.4, .25, .25, .1, 0.],
        [.4, .25, .25, .1, 0.],
    ])
    reports = ['B', 'B', 'outside', '', '', 'A', 'E']
    statuses = ['coded', 'coded', 'coded', 'missing', 'needs_review', 'coded', 'coded']
    mask = np.array([True, True, True, True, True, False, True])
    return codes, probability, reports, statuses, mask


def test_probability_denominator_retains_coded_reports_outside_vocabulary():
    codes, probability, reports, statuses, mask = _example()
    before = probability.copy()
    result = score_report_predictions(codes, probability, reports, statuses, mask)
    np.testing.assert_array_equal(probability, before)
    np.testing.assert_allclose(result['report_probability'][[0, 1, 2, 6]], [.25, .25, 0., 0.])
    assert np.isnan(result['report_probability'][[3, 4, 5]]).all()
    assert not result['report_in_vocabulary'][2]
    assert result['report_in_vocabulary'][6]
    np.testing.assert_array_equal(result['report_zero_probability'], [False, False, True, False, False, False, True])
    assert np.isinf(result['report_nll'][[2, 6]]).all()
    assert np.isnan(result['report_finite_nll'][[2, 3, 4, 5, 6]]).all()
    np.testing.assert_allclose(result['report_top1_hit'][[0, 1, 2, 6]], [0., .25, 0., 0.])
    np.testing.assert_allclose(result['report_top3_hit'][[0, 1, 2, 6]], [1., .75, 0., 0.])


def test_fractional_top_k_is_invariant_to_code_order_and_has_zero_mass_guard():
    codes, probability, reports, statuses, mask = _example()
    result = score_report_predictions(codes, probability, reports, statuses, mask, top_k=(1, 3, 20))
    order = np.array([4, 2, 0, 3, 1])
    permuted = score_report_predictions([codes[i] for i in order], probability[:, order],
                                        reports, statuses, mask, top_k=(1, 3, 20))
    for key in result:
        np.testing.assert_allclose(result[key], permuted[key], equal_nan=True)
    assert result['report_top20_hit'][0] == 1.
    assert result['report_top20_hit'][2] == 0.
    assert result['report_top20_hit'][6] == 0.


def test_relative_tie_tolerance_preserves_tiny_probability_distinctions():
    values = np.array([[1 - 3e-15, 1e-15, 2e-15]])
    score = score_report_predictions(['A', 'B', 'C'], values, ['B'], ['coded'], [True], top_k=(2,))
    assert score['report_top2_hit'][0] == 0.
    # Numerical perturbations at the same magnitude join one boundary tie.
    values = np.array([[.5, .25 + 1e-12, .25 - 1e-12]])
    score = score_report_predictions(['A', 'B', 'C'], values, ['B'], ['coded'], [True], top_k=(2,))
    assert score['report_top2_hit'][0] == .5


def test_subject_summary_uses_explicit_coverage_and_nll_denominators():
    codes, probability, reports, statuses, mask = _example()
    result = score_report_predictions(codes, probability, reports, statuses, mask)
    frame = pd.DataFrame({'iSub': [102] * 7, 'condition': [1] * 7,
                          'scored': mask, 'parse_status': statuses, **result})
    summary = summarize_report_scores(frame).iloc[0]
    assert summary.n_trials == 7
    assert summary.n_scored_trials == 6
    assert summary.n_scored_coded_reports == 4
    assert summary.n_scored_missing_reports == 1
    assert summary.n_scored_needs_review_reports == 1
    assert summary.n_vocabulary_reports == 3
    assert summary.n_positive_probability_reports == 2
    assert summary.parser_coverage == pytest.approx(4 / 6)
    assert summary.vocabulary_coverage == .75
    assert summary.positive_support_coverage == .5
    assert summary.zero_probability_fraction == .5
    assert summary.mean_report_probability == .125
    assert summary.top1_hit_rate == .0625
    assert summary.top3_hit_rate == .4375
    assert np.isinf(summary.mean_report_nll)
    assert summary.mean_positive_report_nll == pytest.approx(-np.log(.25))


def test_subject_with_no_coded_reports_has_missing_scores_and_coverage():
    score = score_report_predictions(['A'], np.ones((1, 1)), [''], ['missing'], [True])
    frame = pd.DataFrame({'iSub': [222], 'condition': [2], 'scored': [True],
                          'parse_status': ['missing'], **score})
    summary = summarize_report_scores(frame).iloc[0]
    assert summary.parser_coverage == 0.
    assert summary.n_scored_coded_reports == 0
    for name in ('vocabulary_coverage', 'positive_support_coverage', 'mean_report_probability',
                 'mean_report_nll', 'mean_positive_report_nll', 'top1_hit_rate', 'top3_hit_rate'):
        assert np.isnan(summary[name])


def test_current_conservative_encoder_preserves_feature_order_and_review_mask():
    features = ('neck', 'leg', 'tail', 'head')
    parsed = [encode_report(text, features) for text in ['脖子长。', '躯干长于脖子。', '三个部位很长。', '']]
    codes = [report_code(row.tokens) if row.status == 'coded' else '' for row in parsed]
    swapped = encode_report('脖子长。', ('leg', 'neck', 'tail', 'head'))
    assert codes[0] != report_code(swapped.tokens)
    result = score_report_predictions(codes[:2], np.full((4, 2), .5), codes,
                                       [row.status for row in parsed], [True] * 4)
    np.testing.assert_array_equal(result['oral_score_mask'], [True, True, False, False])
    assert 'body:0:lt' in parsed[1].tokens
    assert parsed[2].unparsed


@pytest.mark.parametrize('replacement', [
    {'codes': ['A', 'A']},
    {'probabilities': np.array([[.8, .3]])},
    {'probabilities': np.array([[np.nan, .5]])},
    {'probabilities': np.array([[1.1, -.1]])},
    {'score_mask': [1]},
    {'report_codes': ['']},
    {'statuses': ['unreviewed']},
    {'top_k': (0,)},
    {'top_k': (True,)},
    {'top_k': (1, 1)},
    {'tie_rtol': -1.},
])
def test_invalid_prediction_or_encoding_contract_is_rejected(replacement):
    values = dict(codes=['A', 'B'], probabilities=np.array([[.8, .2]]),
                  report_codes=['A'], statuses=['coded'], score_mask=[True])
    values.update(replacement)
    with pytest.raises(ValueError):
        score_report_predictions(**values)
