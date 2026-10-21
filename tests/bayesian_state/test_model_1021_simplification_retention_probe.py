"""Guard the two-coordinate intervention and its conditional retention statistic."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.optimization.search.adaptive_proposals import candidate
from src.Bayesian_state.workflows.analysis.model_1021_simplification_retention_probe import (
    DECAY, SEARCH, check_points, retained_after_discovery,
)


def points():
    return [dict(candidate({SEARCH: p, DECAY: d, 'fixed': 8}, 'test'),
                 baseline_probability=p, evidence_decay=d)
            for p in (.02, .05) for d in (.95, .99)]


def test_intervention_rejects_third_coordinate_confound_and_missing_cell():
    bank = points()
    check_points(bank, bank[-1])
    changed = deepcopy(bank)
    changed[0]['hyperparams']['fixed'] = 4
    changed[0]['id'] = candidate(changed[0]['hyperparams'], 'test')['id']
    with pytest.raises(ValueError, match='undeclared coordinate'):
        check_points(changed, bank[-1])
    with pytest.raises(ValueError, match='four registered'):
        check_points(bank[:-1], bank[-1])


def test_retention_starts_at_discovery_and_keeps_non_discovery_undefined():
    present = np.array([[0, 0, 1, 0, 1], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]])
    result = retained_after_discovery(present)
    np.testing.assert_allclose(result[[0, 2]], [2/3, 1])
    assert np.isnan(result[1])
