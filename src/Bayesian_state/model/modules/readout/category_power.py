"""Process-local research intervention on final category response probabilities.

The exponent acts after mixing rules. Rule weights, rule-specific predictions,
feedback likelihoods and precision updates retain their original definitions.
Snapshots therefore describe the untransformed cognitive mixture. Consumers of
joint choice/report probabilities must supply an explicit observation bridge.
Use this context in an isolated numerical worker, never across worker threads.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from typing import Iterator

import numpy as np


def category_power(probabilities: np.ndarray, exponent: float) -> np.ndarray:
    """Normalize positive powers along the category axis, retaining exact zeros."""
    p = np.asarray(probabilities, dtype=float)
    if (p.ndim < 1 or not p.shape[-1] or not np.isfinite(p).all()
            or np.any(p < 0) or not np.isfinite(exponent) or exponent <= 0):
        raise ValueError('Need nonnegative finite probabilities and a positive exponent')
    if not np.allclose(p.sum(axis=-1), 1., atol=1e-12, rtol=0):
        raise ValueError('Category probabilities must sum to one')
    if exponent == 1.:
        return p.copy()
    powered = (p / p.max(axis=-1, keepdims=True)) ** exponent
    return powered / powered.sum(axis=-1, keepdims=True)


@contextmanager
def category_power_readout(exponent: float) -> Iterator[dict]:
    """Apply the same final response transformation in filtering and simulation.

    Power one is an exact no-op. Other powers currently require deterministic
    expectation over rule contributions, with output lapse and confidence
    interventions disabled. Raw snapshots are retained for state diagnostics.
    """
    from . import choice
    if not np.isfinite(exponent) or exponent <= 0:
        raise ValueError('Response exponent must be finite and positive')
    audit = dict(exponent=float(exponent), filter_reads=0, trajectory_reads=0,
                 snapshot_semantics='untransformed_rule_mixture')
    if exponent == 1.:
        yield audit
        return
    read, predict = choice.read_choice_probabilities_from_model, choice.predict_choice_from_model

    def transformed_read(*args, **kwargs):
        if kwargs.get('power', 1.) != 1. or kwargs.get('lapse', 0.) != 0.:
            raise ValueError('Final category power requires original rule weights and zero lapse')
        result = read(*args, **kwargs)
        audit['filter_reads'] += 1
        if kwargs.get('return_snapshot', False):
            p, snapshot = result
            return category_power(p, exponent), snapshot
        return category_power(result, exponent)

    def transformed_predict(*args, **kwargs):
        config, noise = kwargs['choice_readout_config'], kwargs['output_noise_config']
        if (config.get('method', 'expectation') != 'expectation' or config.get('power', 1.) != 1.
                or noise.get('enabled', False) or config.get('strategy_confidence_gain', 0.) != 0.
                or config.get('rule_commitment_confidence_gain', 0.) != 0.):
            raise ValueError('Final category power requires pure expectation and disabled output noise')
        original = predict(*args, **kwargs)
        if original.output_lapse != 0. or not np.allclose(
                original.observed_probabilities, original.cognitive_probabilities, atol=1e-12, rtol=0):
            raise ValueError('Unexpected response transformation before category power')
        audit['trajectory_reads'] += 1
        return replace(original, observed_probabilities=category_power(original.observed_probabilities, exponent),
                       readout_details=dict(original.readout_details, category_response_exponent=float(exponent)))

    try:
        choice.read_choice_probabilities_from_model = transformed_read
        choice.predict_choice_from_model = transformed_predict
        yield audit
    finally:
        choice.read_choice_probabilities_from_model = read
        choice.predict_choice_from_model = predict
