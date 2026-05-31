"""Discrete rule hypothesis spaces for Bayesian_state.

The continuous ``Partition`` class models hypotheses as geometric partitions in
a continuous feature space.  Exp5 stimuli are instead binary feature vectors, so
this module represents each hypothesis as a symbolic rule
``{-1, 1}^n -> {1, 2}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DiscreteRule:
    """One monomial/parity rule over binary features."""

    dims: tuple[int, ...]
    polarity: int

    @property
    def label(self) -> str:
        if not self.dims:
            base = "Intercept"
        else:
            base = "*".join(f"X{i + 1}" for i in self.dims)
        return base if self.polarity > 0 else f"-{base}"


class DiscreteRulePartition:
    """Parity/monomial hypothesis space for binary discrete stimuli.

    Each hypothesis predicts category 1 when ``polarity * prod(x[dims])`` is
    positive and category 2 otherwise.  Stimuli are expected to use ``-1`` and
    ``1`` values; positive values are coerced to ``1`` and non-positive values
    to ``-1`` for robustness.
    """

    EPS = 1e-12
    DISTANCE_MODE_RULE = "rule"
    VALID_DISTANCE_MODES = (DISTANCE_MODE_RULE,)

    def __init__(
        self,
        n_dims: int = 5,
        n_cats: int = 2,
        include_intercept: bool = False,
        **kwargs,
    ) -> None:
        if int(n_cats) != 2:
            raise ValueError("DiscreteRulePartition currently supports binary categories only.")
        self.n_dims = int(n_dims)
        self.n_cats = int(n_cats)
        self.include_intercept = bool(include_intercept)
        self.rules = self.get_all_rules()
        self._stimulus_space = self._build_stimulus_space()
        self._prediction_table = self._build_prediction_table()
        self._similarity_matrix = None

    @property
    def length(self) -> int:
        return len(self.rules)

    @property
    def similarity_matrix(self) -> np.ndarray:
        if self._similarity_matrix is None:
            pred = self._prediction_table
            self._similarity_matrix = (pred[:, None, :] == pred[None, :, :]).mean(axis=2)
        return self._similarity_matrix

    def get_all_rules(self) -> list[DiscreteRule]:
        rules: list[DiscreteRule] = []
        start_size = 0 if self.include_intercept else 1
        for size in range(start_size, self.n_dims + 1):
            for dims in combinations(range(self.n_dims), size):
                for polarity in (1, -1):
                    rules.append(DiscreteRule(tuple(dims), polarity))
        return rules

    def describe_rule(self, hypo: int) -> str:
        return self.rules[int(hypo)].label

    def get_category_assignment(
        self,
        hypo: int,
        stimulus: np.ndarray,
        distance_mode: str = DISTANCE_MODE_RULE,
        beta: float = 1.0,
        **kwargs,
    ) -> int:
        """Return the 0-based predicted category for one stimulus."""
        return int(self._predict_categories_for_rule(int(hypo), np.asarray([stimulus]))[0] - 1)

    def get_category_probabilities(
        self,
        hypo: int,
        data: list | tuple,
        beta: float,
        distance_mode: str = DISTANCE_MODE_RULE,
        **kwargs,
    ) -> np.ndarray:
        """Return category probabilities with shape ``[n_cats, n_trials]``."""
        self._resolve_distance_mode(distance_mode)
        stimuli = self._coerce_stimuli(np.asarray(data[0], dtype=float))
        predicted = self._predict_categories_for_rule(int(hypo), stimuli) - 1
        scores = np.zeros((stimuli.shape[0], self.n_cats), dtype=float)
        scores[np.arange(stimuli.shape[0]), predicted] = float(beta)
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return prob.T

    def calc_likelihood(
        self,
        hypos: List[int] | Tuple[int] | Sequence[int],
        data: list | tuple,
        beta: list | tuple | float | np.ndarray = 1.0,
        distance_mode: str = DISTANCE_MODE_RULE,
        normalized: bool = True,
        **kwargs,
    ) -> np.ndarray:
        beta_values = self._resolve_beta_vector(beta, len(hypos))
        ret = np.zeros((len(data[2]), len(hypos)), dtype=float)
        for col, hypo in enumerate(hypos):
            ret[:, col] = self.calc_likelihood_entry(
                int(hypo),
                data,
                beta_values[col],
                distance_mode=distance_mode,
                **kwargs,
            )
        if normalized:
            denom = np.sum(ret, axis=1, keepdims=True)
            return ret / np.clip(denom, self.EPS, None)
        return ret

    def calc_likelihood_entry(
        self,
        hypo: int,
        data: list | tuple,
        beta: float,
        distance_mode: str = DISTANCE_MODE_RULE,
        **kwargs,
    ) -> np.ndarray:
        prob = self.get_category_probabilities(
            hypo=hypo,
            data=data,
            beta=beta,
            distance_mode=distance_mode,
            **kwargs,
        )
        return self._feedback_likelihood_from_category_probabilities(prob, data)

    def calc_trueprob_entry(
        self,
        hypo: int,
        data: list | tuple,
        beta: float | list | tuple | np.ndarray,
        distance_mode: str = DISTANCE_MODE_RULE,
        **kwargs,
    ) -> np.ndarray:
        beta_value = self._resolve_beta_vector(beta, 1)[0]
        prob = self.get_category_probabilities(
            hypo=hypo,
            data=data,
            beta=beta_value,
            distance_mode=distance_mode,
            **kwargs,
        )
        category = np.asarray(data[3], dtype=int) - 1
        return prob[category.flatten(), np.arange(prob.shape[1])]

    def _build_stimulus_space(self) -> np.ndarray:
        return np.asarray(list(product([-1, 1], repeat=self.n_dims)), dtype=int)

    def _build_prediction_table(self) -> np.ndarray:
        table = np.zeros((self.length, self._stimulus_space.shape[0]), dtype=int)
        for hypo in range(self.length):
            table[hypo] = self._predict_categories_for_rule(hypo, self._stimulus_space)
        return table

    def _predict_categories_for_rule(self, hypo: int, stimuli: np.ndarray) -> np.ndarray:
        stimuli = self._coerce_stimuli(stimuli)
        rule = self.rules[hypo]
        if not rule.dims:
            value = np.full(stimuli.shape[0], rule.polarity, dtype=int)
        else:
            value = rule.polarity * np.prod(stimuli[:, list(rule.dims)], axis=1)
        return np.where(value > 0, 1, 2)

    def _coerce_stimuli(self, stimuli: np.ndarray) -> np.ndarray:
        arr = np.asarray(stimuli, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.n_dims:
            raise ValueError(
                f"Expected stimuli with {self.n_dims} features, got shape {arr.shape}."
            )
        return np.where(arr > 0, 1, -1).astype(int)

    def _feedback_likelihood_from_category_probabilities(
        self,
        prob: np.ndarray,
        data: list | tuple,
    ) -> np.ndarray:
        choices = np.asarray(data[1], dtype=int).copy() - 1
        responses = np.asarray(data[2], dtype=float)
        n_trials = len(choices)
        p_choice = prob[choices, np.arange(n_trials)]
        likelihood = np.where(responses == 1, p_choice, 1.0 - p_choice)
        return np.clip(likelihood, self.EPS, 1.0 - self.EPS)

    @classmethod
    def _resolve_distance_mode(cls, distance_mode: str) -> str:
        if distance_mode not in cls.VALID_DISTANCE_MODES:
            raise ValueError(
                f"Unsupported distance_mode '{distance_mode}'. "
                f"Expected one of: {cls.VALID_DISTANCE_MODES}."
            )
        return distance_mode

    @staticmethod
    def _resolve_beta_vector(beta, n_hypos: int) -> list[float]:
        if isinstance(beta, np.ndarray):
            values = beta.flatten().tolist()
        elif isinstance(beta, (int, float)):
            values = [float(beta)] * n_hypos
        elif isinstance(beta, (list, tuple)):
            values = list(beta)
        else:
            values = [float(beta)] * n_hypos

        if len(values) != n_hypos:
            default = values[0] if values else 1.0
            values = [default] * n_hypos
        return [float(x) for x in values]
