import pytest

from src.Bayesian_state.workflows.analysis.model_0923_recalled_memory_recovery import grid_points, MEMORY, DECAY


def test_recovery_grid_changes_only_two_coordinates_and_contains_every_truth():
    anchor = {MEMORY: .3, DECAY: .95, 'other_coordinate': 7}
    rows = grid_points(anchor, [0., .8], [.8, 1.])
    assert len(rows) == len({r['id'] for r in rows}) == 4
    assert {(r['hyperparams'][MEMORY], r['hyperparams'][DECAY]) for r in rows} == {
        (0., .8), (0., 1.), (.8, .8), (.8, 1.)}
    assert all(r['hyperparams']['other_coordinate'] == 7 for r in rows)
    assert anchor[MEMORY] == .3 and anchor[DECAY] == .95


@pytest.mark.parametrize('memory,decay', [([0., 0.], [.8, 1.]), ([0.], [.8, 1.]), ([0., .8], [.8, float('nan')])])
def test_recovery_rejects_degenerate_or_invalid_levels(memory, decay):
    with pytest.raises(ValueError):
        grid_points({MEMORY: .3, DECAY: .95}, memory, decay)
