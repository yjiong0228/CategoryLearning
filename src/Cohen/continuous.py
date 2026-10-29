"""Continuous exponential-family extension with feedback-conditioned learning.

The stimulus domain is the fixed unit cube. Gauss–Legendre quadrature integrates
category densities; raw stimuli retain their continuous values. This extension
uses the exact conditional-likelihood derivative by default (see README.md).
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np
from scipy.special import logsumexp

from .features import (UPDATE_RULES, feature_subsets, normalize, positive_integer,
                       transform_features, update_weights)
from .feedback import category_count, category_indices, feedback_mask, task_feedback


@lru_cache(maxsize=16)
def integration_grid(n_dimensions: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Tensor quadrature nodes and log volume weights on [0, 1]^d."""
    positive_integer(n_dimensions, "n_dimensions")
    positive_integer(order, "quadrature_order")
    if order < 2 or order ** n_dimensions > 1_000_000:
        raise ValueError("quadrature requires order >= 2 and at most 1,000,000 nodes")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    indices = np.asarray(list(product(range(order), repeat=n_dimensions)))
    points = (nodes[indices] + 1.0) / 2.0
    log_weights = np.log(weights[indices] / 2.0).sum(axis=1)
    points.setflags(write=False)
    log_weights.setflags(write=False)
    return points, log_weights


@dataclass
class ContinuousFeatureMixture:
    n_dimensions: int = 4
    n_categories: int = 2
    max_order: int | None = None
    beta: float = 5.0
    eta: float = 0.05
    quadrature_order: int = 8
    alpha: np.ndarray | None = None
    bias: np.ndarray | None = None
    seed: int | None = None
    update_rule: str = "exact_gradient"

    def __post_init__(self) -> None:
        positive_integer(self.n_categories, "n_categories")
        if self.n_categories < 2:
            raise ValueError("at least two categories are required")
        if not np.isfinite(self.beta) or self.beta <= 0 or not np.isfinite(self.eta) or self.eta < 0:
            raise ValueError("beta must be finite and positive; eta finite and nonnegative")
        if self.update_rule not in UPDATE_RULES:
            raise ValueError(f"update_rule must be one of {UPDATE_RULES}")
        self.subsets = feature_subsets(self.n_dimensions, self.max_order)
        self.n_features = len(self.subsets)
        self.nodes, self.log_weights = integration_grid(self.n_dimensions, self.quadrature_order)
        self.F_grid = self.features(self.nodes)
        rng = np.random.default_rng(self.seed)
        if self.alpha is None:
            self.alpha = rng.normal(size=(self.n_categories, self.n_features))
        self.alpha = np.asarray(self.alpha, dtype=float)
        if self.alpha.shape != (self.n_categories, self.n_features):
            raise ValueError("alpha must have shape (n_categories, n_features)")
        self.alpha = np.stack([normalize(row) for row in self.alpha])
        self.bias = np.zeros(self.n_categories) if self.bias is None else np.array(self.bias, dtype=float)
        if self.bias.shape != (self.n_categories,) or not np.all(np.isfinite(self.bias)):
            raise ValueError("bias must be a finite category vector")
        # Fix the last category as reference; only K - 1 bias parameters learn.
        self.bias -= self.bias[-1]

    def copy(self) -> ContinuousFeatureMixture:
        return deepcopy(self)

    def features(self, stimuli: np.ndarray) -> np.ndarray:
        X = np.asarray(stimuli, dtype=float)
        if X.ndim == 1:
            X = X[None, :]
        if X.ndim != 2 or X.shape[1] != self.n_dimensions or not np.all(np.isfinite(X)):
            raise ValueError("stimuli must be a finite trial-by-dimension matrix")
        if np.any((X < 0) | (X > 1)):
            raise ValueError("continuous stimuli must lie in the fixed [0, 1] domain")
        return transform_features(2.0 * X - 1.0, self.subsets)

    def _statistics(self) -> tuple[np.ndarray, np.ndarray]:
        weighted_logits = self.beta * (self.F_grid @ self.alpha.T) + self.log_weights[:, None]
        log_z = logsumexp(weighted_logits, axis=0)
        masses = np.exp(weighted_logits - log_z)
        return log_z, masses.T @ self.F_grid

    def _log_probabilities(self, F: np.ndarray, log_z: np.ndarray) -> np.ndarray:
        logits = self.beta * (F @ self.alpha.T) - log_z + self.bias
        return logits - logsumexp(logits, axis=-1, keepdims=True)

    def log_probabilities(self, stimuli: np.ndarray) -> np.ndarray:
        return self._log_probabilities(self.features(stimuli), self._statistics()[0])

    def predict_proba(self, stimuli: np.ndarray) -> np.ndarray:
        return np.exp(self.log_probabilities(stimuli))

    def category_density(self, stimuli: np.ndarray) -> np.ndarray:
        """Return density (not point probability), with columns in category order."""
        return np.exp(self.beta * (self.features(stimuli) @ self.alpha.T) - self._statistics()[0])

    def evidence_gradients(self, stimulus: np.ndarray, compatible: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Exact derivatives of log P(compatible feedback | stimulus, choice).

        The conditional target q renormalizes current category probabilities
        over the compatible set; the gradient uses q - p. Derivatives are taken
        before the unit-norm projection. The reference bias remains fixed.
        """
        f = self.features(stimulus)
        if len(f) != 1:
            raise ValueError("one stimulus is required for a feedback update")
        log_z, expected = self._statistics()
        logp = self._log_probabilities(f[0], log_z)
        delta = self._evidence_delta(logp, compatible)
        alpha_gradient = self.beta * delta[:, None] * (f[0] - expected)
        bias_gradient = delta.copy()
        bias_gradient[-1] = 0.0
        return alpha_gradient, bias_gradient

    def _evidence_delta(self, logp: np.ndarray, compatible: np.ndarray) -> np.ndarray:
        mask = np.asarray(compatible)
        if mask.shape != (self.n_categories,) or mask.dtype != np.bool_ or not mask.any():
            raise ValueError("compatible must be a nonempty boolean category mask")
        q = np.zeros(self.n_categories)
        q[mask] = np.exp(logp[mask] - logsumexp(logp[mask]))
        return q - np.exp(logp)

    def _update(self, f: np.ndarray, logp: np.ndarray, expected: np.ndarray,
                compatible: np.ndarray) -> None:
        delta = self._evidence_delta(logp, compatible)
        gradient = delta[:, None] * (f - expected)
        self.alpha = np.stack([update_weights(a, g, self.beta, self.eta, self.update_rule)
                               for a, g in zip(self.alpha, gradient)])
        self.bias[:-1] += self.eta * delta[:-1]

    def learn_from_feedback(self, stimulus: np.ndarray, choice: int,
                            feedback: float, condition: int) -> None:
        if category_count(condition) != self.n_categories:
            raise ValueError("condition does not match model category count")
        f = self.features(stimulus)
        if len(f) != 1:
            raise ValueError("one stimulus is required for a feedback update")
        log_z, expected = self._statistics()
        self._update(f[0], self._log_probabilities(f[0], log_z), expected,
                     feedback_mask(choice, feedback, condition))

    def run_session(self, stimuli: np.ndarray, choices: np.ndarray | None,
                    feedback: np.ndarray | None, condition: int,
                    true_labels: np.ndarray | None = None, *, update: bool = True,
                    simulation_seed: int | None = None) -> dict[str, np.ndarray]:
        """Predict before feedback; mutate state after each observed/simulated trial.

        Passing choices=None generates choices and their own task feedback.
        In replay, true_labels are used only for scoring, never for learning.
        """
        F = self.features(stimuli)
        n = len(F)
        if not n or category_count(condition) != self.n_categories:
            raise ValueError("nonempty session and matching condition are required")
        truth = None if true_labels is None else category_indices(true_labels, self.n_categories)
        if truth is not None and truth.shape != (n,):
            raise ValueError("true_labels must match the trial count")
        simulate = choices is None
        if simulate:
            if truth is None or feedback is not None:
                raise ValueError("simulation needs true_labels and generates its own feedback")
        else:
            observed = category_indices(choices, self.n_categories, "choices")
            fb = np.asarray(feedback, dtype=float)
            if observed.shape != (n,) or fb.shape != (n,):
                raise ValueError("choices and feedback must match the trial count")
            masks = [feedback_mask(int(c + 1), float(v), condition) for c, v in zip(observed, fb)]
        rng = np.random.default_rng(simulation_seed)
        probabilities, log_likelihoods, alphas, biases, responses, scores = [], [], [], [], [], []
        for t, f in enumerate(F):
            log_z, expected = self._statistics()
            logp = self._log_probabilities(f, log_z)
            p = np.exp(logp)
            alphas.append(self.alpha.copy())
            biases.append(self.bias.copy())
            probabilities.append(p)
            if simulate:
                chosen = int(rng.choice(self.n_categories, p=p))
                score = task_feedback(chosen + 1, int(truth[t] + 1), condition)
                mask = feedback_mask(chosen + 1, score, condition)
            else:
                chosen, score, mask = observed[t], fb[t], masks[t]
            responses.append(chosen + 1)
            scores.append(score)
            log_likelihoods.append(logp[chosen])
            if update:
                self._update(f, logp, expected, mask)
        P = np.asarray(probabilities)
        out = {"probabilities": P, "log_likelihood": np.asarray(log_likelihoods),
               "alpha_before_feedback": np.asarray(alphas), "bias_before_feedback": np.asarray(biases),
               "choices": np.asarray(responses), "feedback": np.asarray(scores),
               "predicted_category": P.argmax(axis=1) + 1}
        if truth is not None:
            out["p_correct"] = P[np.arange(n), truth]
            reward = np.asarray([[task_feedback(c, int(y + 1), condition)
                                  for c in range(1, self.n_categories + 1)] for y in truth])
            out["expected_feedback"] = (P * reward).sum(axis=1)
        return out
