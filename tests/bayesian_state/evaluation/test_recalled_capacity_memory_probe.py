"""Only the declared intervention coordinates may change at fixed parameters."""
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_recalled_capacity_memory_probe import (
    CAPACITY, MEMORY, points,
)


@pytest.mark.parametrize('memory,capacity', [(0., 8), (.3, 4)])
def test_crossed_intervention_includes_anchor_and_keeps_other_parameters(memory, capacity):
    anchor = {MEMORY: memory, CAPACITY: capacity, 'precision': 15., 'inertia': 1.}
    grid = points(anchor, [memory, .8], [2, 3, capacity])
    assert len(grid) == len({row['id'] for row in grid}) == 6
    assert sum(row['hyperparams'] == anchor for row in grid) == 1
    assert {(row['hyperparams'][MEMORY],row['hyperparams'][CAPACITY]) for row in grid} == {
        (g,c) for g in [memory,.8] for c in [2,3,capacity]}
    assert all(row['hyperparams']['precision'] == 15. and row['hyperparams']['inertia'] == 1.
               for row in grid)
    assert anchor[MEMORY] == memory and anchor[CAPACITY] == capacity


@pytest.mark.parametrize('memory,capacity', [([0.,0.],[2,8]), ([0.,float('nan')],[2,8]),
    ([0.,1.1],[2,8]), ([0.,.8],[2,2,8]), ([0.,.8],[2,8.]), ([.3,.8],[2,8]),
    ([0.,.8],[2,3]), ([0.,True],[2,8]), ([0.,.8],[1,8])])
def test_invalid_or_unanchored_grids_fail(memory,capacity):
    with pytest.raises(ValueError,match='original coordinates'):
        points({MEMORY:0.,CAPACITY:8},memory,capacity)
