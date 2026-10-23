"""Passive rule diagnostics for autonomous continuation of frozen models.

All trial values describe the prediction before current choice and feedback.
Absent-target precision, prediction and conditional pairing are encoded as zero;
summaries must condition on target presence. Endpoint values describe the full
post-feedback cloud, before the first future pre-choice transition.
"""
from __future__ import annotations

from typing import Any
import numpy as np


def endpoint_state(model: Any, target: int) -> dict[str, Any]:
    from ..model.modules.base_module import ModuleRole
    h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
    memory = model.engine.get_module(ModuleRole.MEMORY)
    present = target in h.active
    pairing = callable(getattr(memory, 'conditional_pairing', None))
    return dict(present=present, beta=float(model.engine.beta[target]) if present else 0.,
        belief=float(model.engine.prior[target]), active_count=len(h.active),
        pairing_available=pairing,
        target_pairing=memory.conditional_pairing()[target].copy() if pairing and present else np.zeros(3),
        pairing_marginal=memory.pairing_marginal().copy() if pairing else np.zeros(3))


class ContinuationObserver:
    """Read rule components with the shared deterministic choice readout."""
    def __init__(self, target: int):
        self.target = target

    def __call__(self, model: Any, prediction: Any) -> dict[str, Any]:
        from ..model.modules.base_module import ModuleRole
        row = endpoint_state(model, self.target)
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        event = h.transition_log[-1]
        snapshot = prediction.rule_snapshot
        if snapshot is None:
            from ..model.modules.readout.choice import read_choice_probabilities_from_model
            probability, snapshot = read_choice_probabilities_from_model(
                model, model._pending_trial.perceived_stimulus, return_snapshot=True)
            np.testing.assert_array_equal(probability, prediction.observed_probabilities)
        ids = np.asarray(snapshot.hypothesis_indices)
        match = np.flatnonzero(ids == self.target)
        np.testing.assert_allclose(snapshot.rule_weights @ snapshot.category_probabilities,
                                   prediction.observed_probabilities, atol=1e-12, rtol=0)
        if bool(len(match)) != row['present']:
            raise ValueError('Choice snapshot and active rule set differ')
        row.update(contribution=float(snapshot.rule_weights[match[0]]) if len(match) else 0.,
            target_probabilities=snapshot.category_probabilities[match[0]].copy() if len(match) else np.zeros(model.n_cats),
            search=int(event['search_event']),
            proposed=-1 if event['proposed_hypothesis'] is None else int(event['proposed_hypothesis']),
            rejected=int(event.get('newcomer_rejected', False)),
            target_capacity_dropped=int(event['capacity_dropped'] == self.target))
        return row
