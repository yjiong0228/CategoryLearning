"""Preserve signed feature effects and joint relevance for declared text groups.

Groups identify features in text spans. Their position in a sentence does not
assign causal direction. Positive, negative and absolute context effects remain
separate; each within-state minimum is computed before averaging particles.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .joint_relevance import context_effects


def group_effect_profiles(
    probability: np.ndarray,
    masks: np.ndarray,
    groups: Mapping[str, np.ndarray],
    weights: np.ndarray,
) -> dict[str, dict]:
    """Evaluate named feature groups under common all-feature reflection contexts.

    ``probability[p, s]`` gives the probability of the observed choice in particle
    p when the coordinates selected by masks[s] are replaced with 1-x. Groups and
    weights leave that state unchanged. Results retain particle arrays as well as
    their weighted means, including the effect of reflecting a whole group.
    """
    q = np.asarray(probability, dtype=float)
    m = np.asarray(masks)
    w = np.asarray(weights, dtype=float)
    if m.ndim != 2:
        raise ValueError('Require a complete two-dimensional reflection mask bank')
    effects = context_effects(q, m, np.ones(m.shape[1], dtype=bool))
    if (w.shape != (q.shape[0],) or not np.isfinite(w).all() or (w < 0).any()
            or not np.isclose(w.sum(), 1., rtol=0, atol=1e-12)):
        raise ValueError('Require normalized nonnegative weights for the same particles')
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError('Require at least one named feature group')
    keys = m.astype(int) @ (1 << np.arange(m.shape[1]))
    columns = {int(key): index for index, key in enumerate(keys)}
    result = {}
    for name, value in groups.items():
        chosen = np.asarray(value)
        if (not isinstance(name, str) or not name or chosen.dtype != np.bool_
                or chosen.shape != (m.shape[1],) or not chosen.any()):
            raise ValueError('Each named group needs a nonempty boolean feature mask')
        features = {key: effects[key][:, chosen] for key in (
            'actual_effect', 'positive_context_effect',
            'negative_context_effect', 'absolute_context_effect')}
        group_key = int(chosen.astype(int) @ (1 << np.arange(len(chosen))))
        particle = dict(
            minimum_positive_effect=features['positive_context_effect'].min(1),
            minimum_absolute_effect=features['absolute_context_effect'].min(1),
            minimum_actual_positive_effect=np.maximum(features['actual_effect'], 0.).min(1),
            mean_positive_effect=features['positive_context_effect'].mean(1),
            mean_negative_effect=features['negative_context_effect'].mean(1),
            joint_reflection_effect=q[:, columns[0]] - q[:, columns[group_key]],
        )
        result[name] = dict(
            feature_indices=np.flatnonzero(chosen),
            feature_effects=features,
            feature_means={key: w @ values for key, values in features.items()},
            particle=particle,
            means={key: float(w @ values) for key, values in particle.items()},
        )
    return result
