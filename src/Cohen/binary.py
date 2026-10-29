"""Binary for the Cohen–Schneidman reimplementation."""
from __future__ import annotations

from typing import Optional, List, Dict
import numpy as np
from dataclasses import dataclass
from copy import deepcopy
from .features import UPDATE_RULES, positive_integer, update_weights, validate_trial_lengths
from .features import all_binary_patterns, feature_subsets, transform_features, check_pm1, normalize, logsumexp, sigmoid_neg_logodds

Array = np.ndarray


@dataclass
class FeatureMixtureStrict:
    n_bits: int
    max_order: Optional[int] = None
    beta: float = 5.0
    eta: float = 0.05
    alpha_pos: Optional[Array] = None
    alpha_neg: Optional[Array] = None
    gamma: float = 0.0
    memory_window: int = 1
    seed: Optional[int] = None
    update_rule: str = "si_ratio"

    def __post_init__(self):
        positive_integer(self.memory_window, "memory_window")
        if self.update_rule not in UPDATE_RULES:
            raise ValueError(f"update_rule must be one of {UPDATE_RULES}")
        if not np.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be positive")
        if not np.isfinite(self.eta) or self.eta < 0:
            raise ValueError("eta must be non-negative")
        self.subsets = feature_subsets(self.n_bits, self.max_order)
        self.n_features = len(self.subsets)
        self.patterns = all_binary_patterns(self.n_bits)
        self.F_all = transform_features(self.patterns, self.subsets)
        rng = np.random.default_rng(self.seed)
        if self.alpha_pos is None:
            self.alpha_pos = rng.normal(size=self.n_features)
        if self.alpha_neg is None:
            self.alpha_neg = rng.normal(size=self.n_features)
        self.alpha_pos = normalize(self.alpha_pos)
        self.alpha_neg = normalize(self.alpha_neg)
        if self.alpha_pos.shape != (self.n_features,) or self.alpha_neg.shape != (self.n_features,):
            raise ValueError("alpha vectors must match the feature count")
        if not np.isfinite(self.gamma):
            raise ValueError("gamma must be finite")
        self.gamma = float(self.gamma)
        self._hist_x: List[Array] = []
        self._hist_f: List[Array] = []
        self._hist_y: List[float] = []

    def copy(self):
        """Copy parameters and the feedback history needed to continue learning."""
        return deepcopy(self)

    def reset_history(self):
        self._hist_x.clear()
        self._hist_f.clear()
        self._hist_y.clear()

    def features(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.n_bits:
            raise ValueError(f"Expected {self.n_bits} stimulus bits")
        check_pm1(X, "X")
        return transform_features(X, self.subsets)

    def _logZ(self, c: int) -> float:
        if c not in (-1, 1):
            raise ValueError("category must be -1 or +1")
        a = self.alpha_pos if c == 1 else self.alpha_neg
        return logsumexp(self.beta * (self.F_all @ a))

    def category_distribution(self, c: int) -> Array:
        if c not in (-1, 1):
            raise ValueError("category must be -1 or +1")
        a = self.alpha_pos if c == 1 else self.alpha_neg
        logits = self.beta * (self.F_all @ a)
        return np.exp(logits - logsumexp(logits))

    def expected_features(self, c: int) -> Array:
        return self.category_distribution(c) @ self.F_all

    def prob_pos(self, X: Array) -> Array:
        return self.prob_pos_features(self.features(X))

    def prob_pos_features(self, F: Array) -> Array:
        F = np.asarray(F, dtype=float)
        if F.ndim == 1:
            F = F[None, :]
        if F.ndim != 2 or F.shape[1] != self.n_features or not np.all(np.isfinite(F)):
            raise ValueError("F must be a finite trial-by-feature matrix")
        # Eq. 2 with alpha = alpha_-1 - alpha_1
        z = (self._logZ(1) - self._logZ(-1)) + self.beta * (F @ (self.alpha_neg - self.alpha_pos)) + self.gamma
        return sigmoid_neg_logodds(z)

    def prob_label(self, X: Array, y: Array) -> Array:
        y = check_pm1(y, "y").reshape(-1)
        p1 = self.prob_pos(X).reshape(-1)
        if len(y) == 1 and len(p1) > 1:
            y = np.repeat(y, len(p1))
        if len(y) != len(p1):
            raise ValueError("labels and stimuli must have equal trial counts")
        return np.where(y == 1.0, p1, 1.0 - p1)

    def predict(self, X: Array) -> Array:
        return np.where(self.prob_pos(X) >= 0.5, 1.0, -1.0)

    def update_batch(self, Xw: Array, yw: Array):
        Xw = np.asarray(Xw, dtype=float)
        if Xw.ndim == 1:
            Xw = Xw[None, :]
        self.update_batch_features(self.features(Xw), yw)

    def update_batch_features(self, Fw: Array, yw: Array):
        Fw = np.asarray(Fw, dtype=float)
        if Fw.ndim == 1:
            Fw = Fw[None, :]
        yw = check_pm1(yw, "yw").reshape(-1)
        validate_trial_lengths(Fw, yw)
        Epos = self.expected_features(1)
        Eneg = self.expected_features(-1)
        p1 = self.prob_pos_features(Fw).reshape(-1)
        py = np.where(yw == 1.0, p1, 1.0 - p1)
        err = 1.0 - py

        gpos = np.zeros(self.n_features)
        gneg = np.zeros(self.n_features)
        ggam = 0.0

        for f, y, e in zip(Fw, yw, err):
            # Eq. 3: d alpha_c = - (1-P(y|x)) * (E_c[f]-f(x)) * y * c
            gpos += -e * (Epos - f) * y * 1.0
            gneg += -e * (Eneg - f) * y * (-1.0)
            # SI S3
            ggam += -e * y

        scale = 1.0 / len(yw)
        self.alpha_pos = update_weights(self.alpha_pos, scale * gpos, self.beta, self.eta, self.update_rule)
        self.alpha_neg = update_weights(self.alpha_neg, scale * gneg, self.beta, self.eta, self.update_rule)
        self.gamma = float(self.gamma + self.eta * scale * ggam)

    def learn_from_feedback(self, x: Array, y: float):
        x = np.asarray(x, dtype=float).reshape(self.n_bits)
        f = self.features(x)[0]
        self.learn_from_feedback_features(x, f, y)

    def learn_from_feedback_features(self, x: Array, f: Array, y: float):
        x = np.asarray(x, dtype=float).reshape(self.n_bits)
        f = np.asarray(f, dtype=float).reshape(self.n_features)
        y = float(check_pm1(np.asarray([y]), "true_y")[0])
        self._hist_x.append(x.copy())
        self._hist_f.append(f.copy())
        self._hist_y.append(y)
        start = max(0, len(self._hist_y) - self.memory_window)
        self.update_batch_features(np.asarray(self._hist_f[start:]), np.asarray(self._hist_y[start:]))

    def run_session(self, stimuli: Array, true_labels: Array, choices: Optional[Array] = None, update: bool = True) -> Dict[str, Array]:
        from .trace import run_session_with_formula_trace

        result, _ = run_session_with_formula_trace(
            self, stimuli, true_labels, choices, update, make_formulas=False
        )
        return result
