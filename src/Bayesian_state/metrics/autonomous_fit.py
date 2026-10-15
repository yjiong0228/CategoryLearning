"""Training-only behavior summaries and unbiased ensemble energy scores.

The two feature families are block accuracy and category-conditioned response
frequencies. Their scales depend only on the declared weights and dimensions.
The U-statistic excludes identical-path pairs, removing the finite-ensemble
penalty on dispersion present in an empirical-distribution V-statistic.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


def behavior_features(choices: np.ndarray, categories: np.ndarray, *,
                      n_categories: int, block_width: int = 32,
                      temporal_weight: float = .5) -> np.ndarray:
    """Return path-by-feature summaries on one fixed stimulus schedule."""
    x, labels = np.asarray(choices), np.asarray(categories)
    if x.ndim != 2 or labels.ndim != 1 or x.shape[1] != len(labels) or not len(labels):
        raise ValueError('Need nonempty path-by-trial choices and aligned categories')
    if n_categories < 2 or block_width < 1 or not 0 < temporal_weight < 1:
        raise ValueError('Invalid feature dimensions or weights')
    for values in (x, labels):
        if (not np.isfinite(values).all() or np.any(values != values.astype(int))
                or np.any(values < 1) or np.any(values > n_categories)):
            raise ValueError('Choice/category labels must be integers in task support')
    temporal = np.stack([(x[:, a:a+block_width] == labels[a:a+block_width]).mean(axis=1)
                         for a in range(0, len(labels), block_width)], axis=1)
    # Missing categories are omitted using the observed stimulus schedule only.
    confusion = np.stack([(x[:, labels == category] == response).mean(axis=1)
                          for category in range(1, n_categories+1) if np.any(labels == category)
                          for response in range(1, n_categories+1)], axis=1)
    return np.concatenate((temporal*np.sqrt(temporal_weight/temporal.shape[1]),
                           confusion*np.sqrt((1-temporal_weight)/confusion.shape[1])), axis=1)


def energy_u(samples: np.ndarray, observed: np.ndarray) -> dict:
    """Score every independent path; jackknife units are complete paths.

    The unbiased finite-sample estimate can be negative. At least three paths
    are required to estimate its simulation error by delete-one jackknife.
    """
    x, y = np.asarray(samples, float), np.asarray(observed, float)
    if (x.ndim != 2 or y.shape != (x.shape[1],) or len(x) < 3
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Need at least three finite paths and one aligned target')
    n = len(x)
    distances = cdist(x, x)
    np.fill_diagonal(distances, 0.)
    target = np.linalg.norm(x-y, axis=1)
    pairs = distances.sum()/2
    value = target.mean()-pairs/(n*(n-1))
    deleted = ((target.sum()-target)/(n-1)
               - (pairs-distances.sum(axis=1))/((n-1)*(n-2)))
    se = np.sqrt((n-1)/n*np.square(deleted-deleted.mean()).sum())
    return dict(value=float(value), mc_se=float(se), paths=n)
