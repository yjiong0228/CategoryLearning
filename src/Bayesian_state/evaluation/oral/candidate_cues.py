"""Describe active rule coverage separately from belief and response use.

A feature belongs to a rule when its declared boundary depends on that feature.
These memberships overlap across features. Availability records a rule's place
in the active set, including rules with zero response weight. Preference margins
compare each rule's probability for the observed choice with its largest other
response probability. Maxima describe available alternatives, not report scores.
"""
from __future__ import annotations

import numpy as np


def candidate_feature_profiles(rule_ids: np.ndarray, rule_features: np.ndarray,
        probabilities: np.ndarray, belief: np.ndarray, contribution: np.ndarray,
        weights: np.ndarray, choice: int) -> dict:
    """Return particle feature descriptions and their original weighted means.

    ``rule_ids`` is particle × active-slot, padded with -1. Rule probabilities,
    belief and contribution use the same slots; choice is a one-based response.
    Preserve each particle until availability, counts and preference maxima have
    been calculated. No new distribution over rules or particles is introduced.
    """
    ids=np.asarray(rule_ids);features=np.asarray(rule_features);p=np.asarray(probabilities,float)
    b=np.asarray(belief,float);u=np.asarray(contribution,float);w=np.asarray(weights,float)
    if (ids.ndim!=2 or not np.issubdtype(ids.dtype,np.integer) or not len(ids) or not ids.shape[1]
            or features.ndim!=2 or features.dtype!=bool or not len(features) or not features.shape[1]
            or np.any((ids < -1)|(ids >= len(features))) or p.ndim!=3 or p.shape[:2]!=ids.shape
            or p.shape[-1]<2 or not isinstance(choice,(int,np.integer)) or not 1<=choice<=p.shape[-1]
            or b.shape!=ids.shape or u.shape!=ids.shape or w.shape!=(len(ids),)
            or any(not np.isfinite(v).all() or np.any(v<0) for v in (p,b,u,w)) or np.any(p>1)):
        raise ValueError('Require valid aligned sparse rule states and normalized weights')
    active=ids>=0
    if not active.any(axis=1).all() or any(len(set(row[row>=0]))!=int((row>=0).sum()) for row in ids):
        raise ValueError('Each particle requires distinct active rule identities')
    if np.any(p[~active]!=0) or np.any(b[~active]!=0) or np.any(u[~active]!=0):
        raise ValueError('Padded slots must have zero probability and weight')
    np.testing.assert_allclose(p[active].sum(-1),1.,atol=1e-12,rtol=0)
    for values in (b.sum(1),u.sum(1),w.sum()):
        np.testing.assert_allclose(values,1.,atol=1e-12,rtol=0)
    membership=features[np.maximum(ids,0)] & active[...,None]
    chosen=p[...,choice-1]
    other=np.max(np.delete(p,choice-1,axis=-1),axis=-1)
    opposition=np.maximum(other-chosen,0.)
    support=np.maximum(chosen-other,0.)
    particle=dict(
        availability=membership.any(1),rule_count=membership.sum(1),
        belief_mass=np.sum(membership*b[...,None],axis=1),
        choice_weight=np.sum(membership*u[...,None],axis=1),
        max_opposing_margin=np.max(membership*opposition[...,None],axis=1),
        max_supporting_margin=np.max(membership*support[...,None],axis=1),
        belief_opposing_margin=np.sum(membership*(b*opposition)[...,None],axis=1),
        contribution_opposing_margin=np.sum(membership*(u*opposition)[...,None],axis=1))
    return dict(particle=particle,means={name:w@value for name,value in particle.items()})
