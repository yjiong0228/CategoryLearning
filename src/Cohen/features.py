"""Features for the Cohen–Schneidman reimplementation."""
from __future__ import annotations

from typing import Optional, Sequence, Tuple, List
import numpy as np
from itertools import combinations, product
import math

Array = np.ndarray


def all_binary_patterns(n_bits: int) -> Array:
    positive_integer(n_bits, "n_bits")
    return np.asarray(list(product([-1.0, 1.0], repeat=n_bits)), dtype=float)


def feature_subsets(n_bits: int, max_order: Optional[int] = None) -> List[Tuple[int, ...]]:
    positive_integer(n_bits, "n_bits")
    if max_order is None:
        max_order = n_bits
    positive_integer(max_order, "max_order")
    if not (1 <= max_order <= n_bits):
        raise ValueError("max_order must be in [1, n_bits]")
    out: List[Tuple[int, ...]] = []
    for k in range(1, max_order + 1):
        out.extend(combinations(range(n_bits), k))
    return out


def transform_features(X: Array, subsets: Sequence[Tuple[int, ...]]) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    return np.column_stack([np.prod(X[:, s], axis=1) for s in subsets])


def check_pm1(x: Array, name: str) -> Array:
    x = np.asarray(x, dtype=float)
    if not np.all(np.isin(x, [-1.0, 1.0])):
        raise ValueError(f"{name} must contain only -1 and +1")
    return x


def normalize(v: Array, eps: float = 1e-12) -> Array:
    v = np.asarray(v, dtype=float).copy()
    if v.ndim != 1 or not v.size or not np.all(np.isfinite(v)):
        raise ValueError("weights must be a nonempty finite vector")
    n = float(np.linalg.norm(v))
    if n < eps:
        v[:] = 0.0
        v[0] = 1.0
        return v
    return v / n


def logsumexp(a: Array) -> float:
    a = np.asarray(a, dtype=float)
    m = float(np.max(a))
    return m + math.log(float(np.sum(np.exp(a - m))))


def sigmoid_neg_logodds(z: Array) -> Array:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = np.exp(-z[pos]) / (1.0 + np.exp(-z[pos]))
    out[~pos] = 1.0 / (1.0 + np.exp(z[~pos]))
    return out


def feature_name(subset: Tuple[int, ...]) -> str:
    return "x" + "".join(str(i + 1) for i in subset)


UPDATE_RULES = ("si_ratio", "paper_eq3", "exact_gradient")


def positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def update_weights(alpha: Array, gradient_without_beta: Array, beta: float,
                   eta: float, rule: str) -> Array:
    """Apply an explicit learning convention, then restore unit vector length.

    The SI rotation formula uses eta / beta. Eq. 3 uses eta; the exact
    derivative of Eq. 2 with respect to alpha includes an additional beta.
    """
    if rule == "si_ratio":
        return normalize(beta * alpha + eta * gradient_without_beta)
    if rule == "paper_eq3":
        return normalize(alpha + eta * gradient_without_beta)
    if rule == "exact_gradient":
        return normalize(alpha + eta * beta * gradient_without_beta)
    raise ValueError(f"update_rule must be one of {UPDATE_RULES}")


def validate_trial_lengths(X: Array, y: Array, r: Optional[Array] = None) -> None:
    if len(X) == 0 or len(X) != len(y) or (r is not None and len(r) != len(X)):
        raise ValueError("stimuli, labels and choices must have equal, nonzero trial counts")
