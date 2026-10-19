"""Bound active-candidate content and retain exact saved decision-basis mass.

Candidate bounds range over category components whose local weights were not
recorded for inactive output channels. They concern the observed response label,
not the candidate's preferred response or a psychological discourse role.
"""
from __future__ import annotations

import numpy as np

from .candidate_cues import candidate_feature_profiles

LABELS = ('support', 'conflict', 'undetermined')


def content_support_profile(rule_ids: np.ndarray, labels: np.ndarray,
        component_counts: np.ndarray, probabilities: np.ndarray, belief: np.ndarray,
        contribution: np.ndarray, component_weights: np.ndarray,
        prior_weights: np.ndarray, choice: int) -> dict:
    """Return candidate bounds and response-use probabilities for one query.

    Labels are rule × component integers 0/1/2, padded with -1. Active rules may
    have zero output weight. Saved local component weights sum to one exactly
    for rules with positive output weight. Condition the joint particle/rule/
    component distribution once on the observed response, including each rule's
    response probability. Candidate belief bounds use only particle conditioning.
    """
    ids = np.asarray(rule_ids)
    lab = np.asarray(labels)
    count = np.asarray(component_counts)
    p, b, u, rho, prior = map(lambda x: np.asarray(x, float),
        (probabilities, belief, contribution, component_weights, prior_weights))
    if (lab.ndim != 2 or not len(lab) or not lab.shape[1]
            or not np.issubdtype(lab.dtype, np.integer)
            or count.shape != (len(lab),) or not np.issubdtype(count.dtype, np.integer)
            or np.any((count < 1) | (count > lab.shape[1]))):
        raise ValueError('Require component labels and valid per-rule counts')
    valid = np.arange(lab.shape[1])[None, :] < count[:, None]
    if np.any(~np.isin(lab[valid], (0, 1, 2))) or np.any(lab[~valid] != -1):
        raise ValueError('Invalid component verdict or padding')
    # Reuse the established sparse-state checks, including rule uniqueness and
    # probability normalization, before adding component-level calculations.
    candidate_feature_profiles(ids, np.ones((len(lab), 1), bool), p, b, u, prior, choice)
    if (rho.shape != (*ids.shape, lab.shape[1]) or not np.isfinite(rho).all()
            or np.any(rho < 0)):
        raise ValueError('Invalid saved component weights')
    active = ids >= 0
    safe_ids = np.maximum(ids, 0)
    if np.any(rho[~(active[..., None] & valid[safe_ids])] != 0):
        raise ValueError('Unused rules and components must have zero weight')
    np.testing.assert_allclose(rho.sum(-1), (u > 0).astype(float), atol=1e-12, rtol=0)
    q = p[..., choice-1]
    predictive = np.sum(u*q, axis=1)
    probability = float(prior @ predictive)
    if probability <= 0:
        raise ValueError('Cannot condition on an impossible response')
    posterior = prior*predictive/probability
    use_prior = prior[:, None, None]*u[..., None]*rho
    use_conditional = use_prior*q[..., None]/probability
    result = {}
    for weighting, weights, use in [('prior', prior, use_prior),
                                    ('choice_conditioned', posterior, use_conditional)]:
        values = {}
        for number, label in enumerate(LABELS):
            match = lab == number
            lower = np.all(match | ~valid, axis=1)[safe_ids] & active
            upper = np.any(match & valid, axis=1)[safe_ids] & active
            values[label+'_availability_lower'] = float(weights @ lower.any(axis=1))
            values[label+'_availability_upper'] = float(weights @ upper.any(axis=1))
            values[label+'_belief_lower'] = float(weights @ np.sum(b*lower, axis=1))
            values[label+'_belief_upper'] = float(weights @ np.sum(b*upper, axis=1))
            values[label+'_decision_mass'] = float(np.sum(use*match[safe_ids]))
        np.testing.assert_allclose(sum(values[x+'_decision_mass'] for x in LABELS), 1., atol=1e-12, rtol=0)
        result[weighting] = values
    return dict(choice_probability=probability, scores=result)
