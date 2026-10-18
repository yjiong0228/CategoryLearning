"""Check exact oral scope and training-only response-map diagnostics."""
import json

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import build_relational_0923_space
from src.Bayesian_state.workflows.analysis.model_0923_oral_extreme_mapping_audit import (
    classify_extreme, compare_mappings, extreme_signatures, mapping_strata,
)


def encoded(signature):
    return json.dumps(['facet:' + json.dumps([list(plane)]) for plane in signature])


def test_all_actual_extreme_regions_and_fixed_labels():
    space = build_relational_0923_space(); signatures = extreme_signatures(space)
    assert len(signatures) == 8
    for signature, expected in signatures.items():
        assert classify_extreme(encoded(signature), signatures) == expected
        kind, dimension, index = expected
        part = space[index].categories[dimension - 1].components[0]
        x = np.full(4, .5); x[dimension - 1] = .1 if kind == 'min' else .9
        assert (part.A @ x <= part.b).all()


def test_whole_report_requires_all_and_only_extreme_constraints():
    signatures = extreme_signatures(build_relational_0923_space()); signature = next(iter(signatures))
    tokens = json.loads(encoded(signature))
    assert classify_extreme(json.dumps(tokens[:-1]), signatures) is None
    assert classify_extreme(json.dumps(tokens + ['facet:[[1,0,0,0,0.5]]']), signatures) is None
    assert classify_extreme(json.dumps(tokens + ['symbolic:body']), signatures) is None
    assert classify_extreme('[]', signatures) is None
    assert classify_extreme(json.dumps(tokens + tokens[:1]), signatures) == signatures[signature]


def from_counts(train, test):
    return pd.DataFrame([dict(phase=phase, dimension=i + 1, choice=j + 1)
        for phase, matrix in [('training', train), ('test', test)]
        for i, row in enumerate(matrix) for j, count in enumerate(row) for _ in range(count)])


def test_known_training_map_transfers_without_later_selection():
    train = [[0, 0, 40, 5], [1, 82, 0, 1], [12, 1, 0, 5], [22, 1, 0, 7]]
    later = [[0, 0, 5, 0], [0, 9, 0, 0], [0, 0, 0, 4], [2, 1, 0, 0]]
    frame = from_counts(train, later); scores, summary = compare_mappings(frame)
    assert len(scores) == 24 and scores.mapping.nunique() == 24
    assert scores.loc[scores.training_best, 'mapping'].tolist() == ['[3, 2, 4, 1]']
    assert (summary['training_reports'], summary['canonical_training'], summary['best_training']) == (177, 89, 149)
    assert (summary['test_reports'], summary['canonical_test'], summary['training_best_test_min']) == (21, 9, 20)
    changed = frame.copy(); changed.loc[changed.phase.eq('test'), 'choice'] = 1
    scores2, _ = compare_mappings(changed)
    pd.testing.assert_series_equal(scores.training_best, scores2.training_best)


def test_tied_training_maps_keep_test_range():
    frame = pd.DataFrame([dict(phase='training', dimension=1, choice=2)] * 2
                         + [dict(phase='test', dimension=2, choice=3)])
    scores, summary = compare_mappings(frame)
    assert scores.training_best.sum() == summary['training_best_count'] == 6
    assert summary['training_best_test_min'] == 0 and summary['training_best_test_max'] == 1
    assert summary['training_observed_dimensions'] == 1


def test_no_training_reports_does_not_choose_map_from_test():
    scores, summary = compare_mappings(pd.DataFrame([dict(phase='test', dimension=1, choice=3)]))
    assert not scores.training_best.any() and summary['best_training'] is None
    assert summary['training_best_test_min'] is None


@pytest.mark.parametrize('bad', [dict(phase='other', dimension=1, choice=1),
                               dict(phase='training', dimension=5, choice=1),
                               dict(phase='training', dimension=1, choice=0)])
def test_invalid_coordinate_or_boundary_rejected(bad):
    with pytest.raises(ValueError): compare_mappings(pd.DataFrame([bad]))


def test_feedback_and_literal_strata_retain_partial_feedback_and_all_maps():
    frame = pd.DataFrame([
        dict(phase='training', dimension=1, choice=2, feedback=1., literal_holds=True),
        dict(phase='training', dimension=2, choice=3, feedback=.5, literal_holds=False),
        dict(phase='test', dimension=1, choice=2, feedback=0., literal_holds=True),
        dict(phase='test', dimension=2, choice=1, feedback=.5, literal_holds=True),
        dict(phase='test', dimension=2, choice=3, feedback=1., literal_holds=False),
    ])
    scores, _ = compare_mappings(frame); before = scores.copy(deep=True)
    split = mapping_strata(frame, scores)
    pd.testing.assert_frame_equal(scores, before)
    assert len(split) == 24 * 2 * 3 * 2
    assert split.groupby('mapping').reports.sum().eq(len(frame)).all()
    assert split.loc[split.feedback.eq(.5)].groupby('mapping').reports.sum().eq(2).all()
    expected = scores.set_index('mapping')
    for phase in ('training', 'test'):
        actual = split.loc[split.phase.eq(phase)].groupby('mapping').matches.sum()
        pd.testing.assert_series_equal(actual, expected[phase + '_matches'].sort_index(), check_names=False)
    assert split.groupby('mapping').training_best.nunique().eq(1).all()
    assert set(split.loc[split.training_best, 'mapping']) == set(scores.loc[scores.training_best, 'mapping'])
    best_errors = split.loc[split.training_best & split.phase.eq('test') & split.feedback.eq(0)]
    assert best_errors.matches.sum() == 2  # Two training-tied maps retain the same error match.
    best_partial = split.loc[split.training_best & split.phase.eq('test') & split.feedback.eq(.5)]
    assert best_partial.matches.sum() == 0  # An unfavorable later stratum stays unfavorable.


@pytest.mark.parametrize('feedback,holds', [(0.25, True), (0., None)])
def test_unknown_strata_not_silently_assigned(feedback, holds):
    frame = pd.DataFrame([dict(phase='training', dimension=1, choice=1,
                              feedback=feedback, literal_holds=holds)])
    scores, _ = compare_mappings(frame)
    with pytest.raises(ValueError):
        mapping_strata(frame, scores)
