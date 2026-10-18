"""A factorial intervention must isolate its two named coordinates."""
from copy import deepcopy
import pytest

from src.Bayesian_state.workflows.analysis import model_0923_reentry_capacity_search as workflow


def fixture():
    levels = {workflow.CAPACITY: [4, 6], workflow.GLOBAL: [.2, .8]}
    bank = [workflow.candidate({workflow.CAPACITY: capacity, workflow.GLOBAL: global_probability,
                                'memory': .3}, 'behavior_confirmation')
            for capacity, global_probability in [(6, .2), (4, .8)]]
    for row in bank:
        row['mean_nll'] = .9
    return bank, levels


def test_four_cells_reuse_existing_records_and_only_new_cells_lack_fitted_scores():
    bank, levels = fixture()
    before = deepcopy(bank)
    rows = workflow.crossed_points(bank, levels)
    assert len(rows) == 4 and bank == before
    known = {row['id']: row for row in bank}
    for row in rows:
        assert row['hyperparams']['memory'] == .3
        if row['id'] in known:
            assert row == known[row['id']]
        else:
            assert 'mean_nll' not in row


@pytest.mark.parametrize('change', ['third_coordinate', 'duplicate_candidate', 'off_grid',
                                  'same_axis', 'duplicate_level', 'invalid_capacity', 'invalid_probability'])
def test_uncontrolled_or_degenerate_grid_is_rejected(change):
    bank, levels = fixture()
    if change == 'third_coordinate':
        bank[1]['hyperparams']['memory'] = .8
    elif change == 'duplicate_candidate':
        bank[1] = deepcopy(bank[0])
    elif change == 'off_grid':
        levels[workflow.CAPACITY] = [2, 6]
    elif change == 'same_axis':
        point = dict(bank[1]['hyperparams'], **{workflow.GLOBAL: .2})
        bank[1] = workflow.candidate(point, 'changed')
    elif change == 'duplicate_level':
        levels[workflow.GLOBAL] = [.2, .2]
    elif change == 'invalid_capacity':
        levels[workflow.CAPACITY] = [4, True]
    else:
        levels[workflow.GLOBAL] = [.2, 1.1]
    with pytest.raises(ValueError):
        workflow.crossed_points(bank, levels)
