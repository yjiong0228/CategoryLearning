from copy import deepcopy
from itertools import product

import pytest

from src.Bayesian_state.workflows.analysis.model_0923_execution_coordinate_cross import MEMORY, SEARCH, hybrid_candidate
from src.Bayesian_state.optimization.search.adaptive_proposals import candidate


@pytest.mark.parametrize('memory,search', list(product(('MIX', 'SELECT'), repeat=2)))
def test_hybrid_changes_exactly_requested_coordinates(memory, search):
    a = candidate({MEMORY: .95, SEARCH: .02, 'other': 7}, 'original')
    b = candidate({MEMORY: .3, SEARCH: .05, 'other': 7}, 'revised')
    frozen = deepcopy((a, b)); result = hybrid_candidate(a, b, memory, search)
    assert (a, b) == frozen
    assert result['hyperparams'] == {MEMORY: .95 if memory == 'MIX' else .3,
                                   SEARCH: .02 if search == 'MIX' else .05, 'other': 7}
    if memory == search:
        assert result['id'] == (a if memory == 'MIX' else b)['id']


@pytest.mark.parametrize('other', [8, None])
def test_unplanned_parameter_change_is_rejected(other):
    a = candidate({MEMORY: .95, SEARCH: .02, 'other': 7}, 'original')
    values = {MEMORY: .3, SEARCH: .05}
    if other is not None:
        values['other'] = other
    with pytest.raises(ValueError):
        hybrid_candidate(a, candidate(values, 'revised'), 'MIX', 'SELECT')
