"""Probe routing preserves explicit mechanism, parameter and cell identities."""
from copy import deepcopy
from pathlib import Path

from src.Bayesian_state.workflows.analysis import model_1022_belief_outcome_probe as probe


def test_routing_preserves_cells_and_restores_launcher(monkeypatch):
    jobs = [dict(model=cell, candidate={'id': cell}, target='/tmp/' + cell)
            for cell in ('SUPPORT', 'SLOT_AT_REFERENCE', 'OUTCOME_AT_REFERENCE', 'OUTCOME_AT_JOINT')]
    before, captured = deepcopy(jobs), []
    entry = probe.pilot.ENTRY
    def execute(mapped, output, budget):
        assert probe.pilot.ENTRY == probe.ENTRY
        captured.extend(mapped)
    monkeypatch.setattr(probe.pilot, 'execute', execute)
    probe.execute(jobs, Path('/tmp'), 1)
    assert jobs == before
    assert probe.pilot.ENTRY == entry
    for original, routed in zip(before, captured):
        mechanism = original['model'].split('_AT_')[0]
        assert routed['belief_variant'] == mechanism
        assert routed['model'] == ('SLOT' if mechanism == 'OUTCOME' else mechanism)
        assert routed['diagnosis_cell'] == original['model']
        assert routed['candidate'] == original['candidate']
        assert routed['target'] == original['target']
