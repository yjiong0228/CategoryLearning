"""Subject-boundary and differing-reference guards for precision extension."""
from copy import deepcopy

import pytest

from src.Bayesian_state.workflows.analysis.model_0923_fixed_h_precision import precision_points
from src.Bayesian_state.workflows.analysis.model_0923_fixed_h_precision_cohort import (
    INIT, RATE, resolved_family, validate_boundary,
)


def test_differing_anchors_preserve_the_common_grid_and_h_parameters():
    family = dict(upper=25., rates=[.01, .06, .1, .3, 1.], anchor='reference')
    before = deepcopy(family)
    for initial, rate, inertia in [(2., .3, 1.), (.5, .06, 4.)]:
        reference = {INIT: initial, RATE: rate, 'inertia': inertia, 'gamma': .8}
        resolved = resolved_family(reference, family)
        assert resolved['anchor'] == [initial, rate]
        points = precision_points(reference, [.1, .5, 2., 5.], resolved)
        assert len(points) == 20
        assert any(p['hyperparams'] == reference for p in points)
        assert all(p['hyperparams']['inertia'] == inertia and p['hyperparams']['gamma'] == .8 for p in points)
    assert family == before


def test_explicit_s107_grid_is_identical_to_original():
    reference = {INIT: .1, RATE: .1, 'capacity': 8}
    family = dict(upper=25., rates=[.01, .03, .1, .3, 1.], anchor=[.1, .1])
    assert precision_points(reference, [.1, .5, 1., 5.], family) == precision_points(
        reference, [.1, .5, 1., 5.], resolved_family(reference, family))


@pytest.mark.parametrize('boundary', [(1,128,64), (2,384,192), (2,672,288), (3,512,256)])
def test_original_condition_and_boundary_are_preserved(boundary):
    condition, train, later = boundary
    fields = dict(condition=condition, training_trials=train, test_trials=later)
    assert validate_boundary(dict(subject=208, expected_boundary=fields), dict(subject=208, **fields)) == (train, train+later)


@pytest.mark.parametrize('changed', [dict(subject=207), dict(condition=3), dict(training_trials=383), dict(test_trials=191)])
def test_changed_identity_or_boundary_rejected(changed):
    fields = dict(condition=2, training_trials=384, test_trials=192)
    with pytest.raises(ValueError):
        validate_boundary(dict(subject=208, expected_boundary=fields), dict(dict(subject=208, **fields), **changed))


def test_reference_anchor_missing_from_grid_is_rejected():
    family = resolved_family({INIT: 2., RATE: .3}, dict(upper=25., rates=[.06], anchor='reference'))
    with pytest.raises(ValueError):
        precision_points({INIT: 2., RATE: .3}, [.1, .5], family)
