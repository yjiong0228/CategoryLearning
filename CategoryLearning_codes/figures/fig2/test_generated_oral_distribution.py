"""Scientific guards for report-space heatmap labels and display compression."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig2 import build_generated_oral_distribution as figure
from src.Bayesian_state.metrics.oral_distribution_alignment import (
    distribution_overlap,
    js_similarity,
)


def _code(token: str) -> str:
    return json.dumps([token], separators=(",", ":"))


def test_display_groups_preserve_all_mass_and_keep_observed_only_codes_separate(monkeypatch):
    monkeypatch.setattr(figure, "DISPLAY_REPORTS", 1)
    codes = ("EMPTY", "OTHER", _code("body:0:gt"), _code("body:1:gt"),
             _code("body:2:gt"), _code("body:3:gt"))
    model = np.array([.1, .1, .7, .1, 0., 0.])
    observed = np.array([0., 0., .1, .2, .3, .4])
    groups, labels, vocabulary, rows = figure._display_rows(
        codes, 4, model, observed, ("tail", "neck", "head", "leg"), 101, 1,
    )
    # Named reports, model remainder and observed-only reports partition every
    # original code exactly once, even though only one report is named.
    np.testing.assert_array_equal(np.sort(np.concatenate(groups)), np.arange(len(codes)))
    displayed = figure.display_distribution(np.stack([model, observed]), groups)
    np.testing.assert_allclose(displayed.sum(axis=1), [1., 1.])
    np.testing.assert_allclose(rows.model_mean.sum(), 1.)
    np.testing.assert_allclose(rows.observed_frequency.sum(), 1.)

    other_row = int(rows.loc[rows.row_kind.eq("OTHER"), "display_row"].iloc[0])
    outside_row = int(rows.loc[rows.row_kind.eq("outside_vocabulary"), "display_row"].iloc[0])
    assert other_row != outside_row
    np.testing.assert_array_equal(groups[other_row], [1])
    np.testing.assert_array_equal(groups[outside_row], [4, 5])
    np.testing.assert_allclose(displayed[:, other_row], [.1, 0.])
    np.testing.assert_allclose(displayed[:, outside_row], [0., .7])
    np.testing.assert_array_equal(vocabulary.out_of_vocabulary, [False] * 4 + [True] * 2)
    assert len(labels) == len(groups)


def test_frequent_observed_only_report_gets_a_named_zero_prediction_row(monkeypatch):
    monkeypatch.setattr(figure, "DISPLAY_REPORTS", 1)
    codes = ("EMPTY", "OTHER", _code("body:0:gt"), _code("body:1:gt"),
             _code("body:2:gt"), _code("body:3:gt"))
    model = np.array([.1, .1, .4, .4, 0., 0.])
    observed = np.array([0., 0., .1, 0., .8, .1])
    groups, labels, vocabulary, rows = figure._display_rows(
        codes, 4, model, observed, ("tail", "neck", "head", "leg"), 101, 1,
    )
    named_row = int(vocabulary.loc[vocabulary.code_index.eq(4), "display_row"].iloc[0])
    remainder_row = int(vocabulary.loc[vocabulary.code_index.eq(5), "display_row"].iloc[0])
    assert named_row != remainder_row
    np.testing.assert_array_equal(groups[named_row], [4])
    np.testing.assert_array_equal(groups[remainder_row], [5])
    assert "head > body ref" in labels[named_row]
    displayed = figure.display_distribution(np.stack([model, observed]), groups)
    np.testing.assert_allclose(displayed[:, named_row], [0., .8])
    np.testing.assert_allclose(displayed[:, remainder_row], [0., .1])
    np.testing.assert_allclose(displayed.sum(axis=1), [1., 1.])
    assert rows.loc[named_row, "n_codes"] == 1


def test_compression_can_increase_visible_overlap_while_full_space_scores_are_preserved():
    full_model = np.array([[.6, .2, .2]])
    full_observed = np.array([[.2, .6, .2]])
    before_model, before_observed = full_model.copy(), full_observed.copy()
    groups = (np.array([0, 1]), np.array([2]))
    full_overlap = distribution_overlap(full_model, full_observed)
    full_js = js_similarity(full_model, full_observed)
    display_model = figure.display_distribution(full_model, groups)
    display_observed = figure.display_distribution(full_observed, groups)
    assert full_overlap[0] == pytest.approx(.6)
    assert distribution_overlap(display_model, display_observed)[0] == 1.
    assert full_js[0] < 1.
    assert js_similarity(display_model, display_observed)[0] == 1.
    np.testing.assert_array_equal(full_model, before_model)
    np.testing.assert_array_equal(full_observed, before_observed)
    np.testing.assert_allclose(distribution_overlap(full_model, full_observed), full_overlap)
    np.testing.assert_allclose(js_similarity(full_model, full_observed), full_js)


def test_report_labels_follow_subject_feature_order_and_preserve_reference_type():
    tail_first = ("tail", "neck", "head", "leg")
    neck_first = ("neck", "tail", "head", "leg")
    body_comparison = _code("body:0:gt")
    absolute_long = _code("facet:[[-1,0,0,0,-0.5]]")
    absolute_short = _code("facet:[[1,0,0,0,0.5]]")
    pair_comparison = _code("facet:[[1,-1,0,0,0]]")
    assert figure.report_label(body_comparison, tail_first) == "tail > body ref"
    assert figure.report_label(body_comparison, neck_first) == "neck > body ref"
    assert figure.report_label(absolute_long, tail_first) == "tail long"
    assert figure.report_label(absolute_short, tail_first) == "tail short"
    assert figure.report_label(body_comparison, tail_first) != figure.report_label(absolute_long, tail_first)
    assert figure.report_label(pair_comparison, tail_first) == "tail <= neck"
    assert figure.report_label(pair_comparison, neck_first) == "neck <= tail"


def test_raw_heatmaps_share_missing_mask_and_keep_an_observed_only_report():
    model = np.array([[.2, .1, .7, 0.], [.1, .1, .8, 0.],
                      [.4, .2, .4, 0.], [.2, .1, .7, 0.]])
    before = model.copy()
    mask = np.array([False, True, False, True])
    case = figure.SubjectDistribution(
        subject=101, condition=1, trials=pd.DataFrame({"trial": np.arange(4)}),
        codes=("EMPTY", "OTHER", _code("body:0:gt"), _code("body:1:gt")),
        vocabulary_size=3, model=model, observed_indices=np.array([0, 3, -1, 2]),
        score_mask=mask, rolling={}, row_groups=tuple(np.array([i]) for i in range(4)),
        row_labels=("Empty", "Other", "Tail", "Outside"),
        vocabulary_table=pd.DataFrame(), row_table=pd.DataFrame(),
        mean_model=np.zeros(4), observed_frequency=np.zeros(4),
    )
    displayed_model, displayed_observed = figure.raw_display(case)
    assert np.isnan(displayed_model[~mask]).all()
    assert np.isnan(displayed_observed[~mask]).all()
    np.testing.assert_allclose(displayed_model[mask], model[mask])
    np.testing.assert_array_equal(displayed_observed[mask], [[0., 0., 0., 1.], [0., 0., 1., 0.]])
    np.testing.assert_array_equal(model, before)


def test_display_compression_retains_missing_rows_including_empty_groups():
    groups = (np.array([0]), np.array([1]), np.array([], dtype=int))
    displayed = figure.display_distribution(np.array([[.4, .6], [np.nan, np.nan]]), groups)
    np.testing.assert_allclose(displayed[0], [.4, .6, 0.])
    assert np.isnan(displayed[1]).all()
