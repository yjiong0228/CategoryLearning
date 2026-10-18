"""Check exact oral scope and training-only response-map diagnostics."""
import json

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.hypothesis_space.spaces.structural_0923 import build_relational_0923_space
from src.Bayesian_state.workflows.analysis.model_0923_oral_extreme_mapping_audit import (
    classify_extreme, compare_mappings, extreme_signatures,
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
