"""Parameter scope labels share paths only for the same scientific point."""
from copy import deepcopy

import pytest

from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
from src.Bayesian_state.workflows.analysis.model_0923_scope_generation import point_bindings


def inputs():
    point = candidate(dict(value=1), 'fixture')
    original = [dict(subject=308, model=model, candidate=point) for model in ('POOL', 'SEARCH')]
    selected = [dict(row, objective=objective) for row in original for objective in ('prefix', 'full')]
    return selected, original


def test_scope_aliases_share_points_but_mechanisms_remain_distinct():
    selected, original = inputs()
    bindings, points = point_bindings(selected, original)
    assert len(bindings) == 6
    assert len(points) == 2
    assert {key[1] for key in points} == {'POOL', 'SEARCH'}


def test_distinct_full_sequence_point_adds_one_bank():
    selected, original = inputs()
    selected = deepcopy(selected)
    selected[1]['candidate'] = candidate(dict(value=2), 'fixture')
    bindings, points = point_bindings(selected, original)
    assert len(points) == 3
    assert len(bindings) == 6


def test_missing_scope_and_mislabeled_parameter_are_rejected():
    selected, original = inputs()
    with pytest.raises(ValueError, match='Incomplete'):
        point_bindings(selected[:-1], original)
    selected = deepcopy(selected)
    selected[0]['candidate']['hyperparams']['value'] = 3
    with pytest.raises(ValueError, match='identity'):
        point_bindings(selected, original)
