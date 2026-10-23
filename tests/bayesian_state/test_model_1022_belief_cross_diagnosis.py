"""Crossed cells preserve parameter identities while routing the mechanism."""
from copy import deepcopy
from pathlib import Path

import pytest

from src.Bayesian_state.workflows.analysis import model_1022_belief_cross_diagnosis as cross


@pytest.mark.parametrize('policy', [{}, {'mechanisms': ['SUPPORT', 'SLOT'],
                                        'parameter_sources': ['SUPPORT', 'SLOT']}])
def test_crossed_jobs_keep_parameters_paths_and_cell_identity(monkeypatch, policy):
    cells = cross.cell_spec(policy)
    jobs = [dict(model=cell, name=cell, candidate={'id': origin, 'hyperparams': {'p': origin}},
                 target='/tmp/' + cell) for cell, (_, origin) in cells.items()]
    before = deepcopy(jobs)
    captured = []
    monkeypatch.setattr(cross.pilot, 'execute', lambda jobs, output, budget: captured.extend(jobs))
    cross.execute(jobs, Path('/tmp'), 1, cells if policy else None)
    assert jobs == before
    for actual, original in zip(captured, before):
        assert actual['model'] == cells[original['model']][0]
        assert actual['diagnosis_cell'] == original['model']
        assert actual['candidate'] == original['candidate']
        assert actual['target'] == original['target']
        assert actual['name'] == original['name']


@pytest.mark.parametrize('policy', [
    {'mechanisms': ['SUPPORT', 'SUPPORT']},
    {'parameter_sources': ['SUPPORT', 'UNKNOWN']},
    {'mechanisms': ['SUPPORT', 'BELIEF', 'SLOT']},
])
def test_cross_requires_two_distinct_known_arms_per_factor(policy):
    with pytest.raises(ValueError, match='two distinct supported arms'):
        cross.cell_spec(policy)
