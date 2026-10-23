"""Crossed cells preserve parameter identities while routing the mechanism."""
from copy import deepcopy
from pathlib import Path

from src.Bayesian_state.workflows.analysis import model_1022_belief_cross_diagnosis as cross


def test_crossed_jobs_keep_parameters_paths_and_cell_identity(monkeypatch):
    jobs = [dict(model=cell, name=cell, candidate={'id': origin, 'hyperparams': {'p': origin}},
                 target='/tmp/' + cell) for cell, (_, origin) in cross.CELLS.items()]
    before = deepcopy(jobs)
    captured = []
    monkeypatch.setattr(cross.pilot, 'execute', lambda jobs, output, budget: captured.extend(jobs))
    cross.execute(jobs, Path('/tmp'), 1)
    assert jobs == before
    for actual, original in zip(captured, before):
        assert actual['model'] == cross.CELLS[original['model']][0]
        assert actual['diagnosis_cell'] == original['model']
        assert actual['candidate'] == original['candidate']
        assert actual['target'] == original['target']
        assert actual['name'] == original['name']
