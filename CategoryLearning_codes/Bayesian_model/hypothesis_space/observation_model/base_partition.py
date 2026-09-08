"""Likelihood flow shared by model-facing partition implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class BasePartition(ABC):
    """Convert category probabilities from a concrete geometry into likelihoods.

    Subclasses own their hypothesis space and implement
    :meth:`get_category_probabilities`. This base class knows nothing about
    polytopes, prototypes, boundaries, or parity rules.
    """

    EPS = 1e-12
    DEFAULT_DISTANCE_MODE: str | None = None
    VALID_DISTANCE_MODES: tuple[str, ...] = ()
    FEEDBACK_MODE_CATEGORY = "category_feedback"
    FEEDBACK_MODE_BERNOULLI_CHOICE = "bernoulli_choice"
    VALID_FEEDBACK_MODES = (
        FEEDBACK_MODE_CATEGORY,
        FEEDBACK_MODE_BERNOULLI_CHOICE,
    )

    def __init__(self, n_dims: int, n_cats: int) -> None:
        self.n_dims = int(n_dims)
        self.n_cats = int(n_cats)

    @property
    @abstractmethod
    def length(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_category_probabilities(
        self,
        hypo: int,
        data: list | tuple,
        beta: float,
        distance_mode: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError

    def calc_likelihood(
        self,
        hypos: Sequence[int],
        data: list | tuple,
        beta: list | tuple | float | np.ndarray = 1.0,
        distance_mode: str | None = None,
        normalized: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """Return a ``[n_trials, n_hypotheses]`` feedback likelihood matrix."""
        beta_values = self._resolve_beta_vector(beta, len(hypos))
        resolved_mode = self._resolve_distance_mode(distance_mode)
        result = np.zeros((len(data[2]), len(hypos)), dtype=float)
        for column, hypothesis in enumerate(hypos):
            result[:, column] = self.calc_likelihood_entry(
                hypothesis,
                data,
                beta_values[column],
                distance_mode=resolved_mode,
                **kwargs,
            )
        if not normalized:
            return result
        return result / np.sum(result, axis=1, keepdims=True)

    def calc_likelihood_entry(
        self,
        hypo: int,
        data: list | tuple,
        beta: float,
        distance_mode: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        probability = self.get_category_probabilities(
            hypo=hypo,
            data=data,
            beta=beta,
            distance_mode=self._resolve_distance_mode(distance_mode),
            **kwargs,
        )
        return self._feedback_likelihood_from_category_probabilities(
            hypo=hypo,
            prob=probability,
            data=data,
            feedback_likelihood_mode=kwargs.get(
                "feedback_likelihood_mode",
                self.FEEDBACK_MODE_CATEGORY,
            ),
            feedback_lapse=kwargs.get("feedback_lapse", 0.0),
        )

    def calc_trueprob_entry(
        self,
        hypo: int,
        data: list | tuple,
        beta: float | list | tuple | np.ndarray,
        distance_mode: str | None = None,
        **kwargs,
    ) -> np.ndarray:
        beta_value = self._resolve_beta_vector(beta, 1)[0]
        probability = self.get_category_probabilities(
            hypo=hypo,
            data=data,
            beta=beta_value,
            distance_mode=self._resolve_distance_mode(distance_mode),
            **kwargs,
        )
        category = np.asarray(data[3], dtype=int) - 1
        if probability.ndim == 1:
            probability = probability.reshape(-1, 1)
        return probability[category.flatten(), np.arange(probability.shape[1])]

    def get_category_assignment(
        self,
        hypo: int,
        stimulus: np.ndarray,
        distance_mode: str | None = None,
        beta: float = 1.0,
        **kwargs,
    ) -> int:
        trial_data = ([np.asarray(stimulus, dtype=float)], [1], [1.0])
        probability = self.get_category_probabilities(
            hypo=hypo,
            data=trial_data,
            beta=beta,
            distance_mode=self._resolve_distance_mode(distance_mode),
            **kwargs,
        )
        return int(np.argmax(probability[:, 0]))

    @classmethod
    def _resolve_distance_mode(cls, distance_mode: str | None) -> str:
        resolved = (
            cls.DEFAULT_DISTANCE_MODE
            if distance_mode is None
            else str(distance_mode)
        )
        if resolved not in cls.VALID_DISTANCE_MODES:
            raise ValueError(
                f"Unsupported distance_mode '{resolved}'. "
                f"Expected one of: {cls.VALID_DISTANCE_MODES}."
            )
        return resolved

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
        return [float(value) for value in values]

    def _feedback_likelihood_from_category_probabilities(
        self,
        hypo: int,
        prob: np.ndarray,
        data: list | tuple,
        feedback_likelihood_mode: str = FEEDBACK_MODE_CATEGORY,
        feedback_lapse: float = 0.0,
    ) -> np.ndarray:
        choices = np.asarray(data[1], dtype=int).copy() - 1
        responses = np.asarray(data[2])
        n_trials = len(choices)
        mode = self._resolve_feedback_likelihood_mode(feedback_likelihood_mode)
        lapse = self._resolve_feedback_lapse(feedback_lapse)

        p_category = prob[choices, np.arange(n_trials)]
        if mode == self.FEEDBACK_MODE_BERNOULLI_CHOICE:
            chance = 1.0 / float(max(1, prob.shape[0]))
            p_choice = (1.0 - lapse) * p_category + lapse * chance
            response_values = np.clip(np.asarray(responses, dtype=float), 0.0, 1.0)
            likelihood = np.power(p_choice, response_values) * np.power(
                1.0 - p_choice,
                1.0 - response_values,
            )
            return np.clip(likelihood, self.EPS, 1.0 - self.EPS)

        likelihood = self._category_feedback_likelihood(
            hypo=hypo,
            prob=prob,
            choices=choices,
            responses=responses,
        )
        return np.clip(likelihood, self.EPS, 1.0 - self.EPS)

    @abstractmethod
    def _category_feedback_likelihood(
        self,
        hypo: int,
        prob: np.ndarray,
        choices: np.ndarray,
        responses: np.ndarray,
    ) -> np.ndarray:
        """Implement task-specific categorical feedback in each subclass."""
        raise NotImplementedError

    @classmethod
    def _resolve_feedback_likelihood_mode(cls, mode: str) -> str:
        mode = str(mode).strip().lower()
        aliases = {
            "category": cls.FEEDBACK_MODE_CATEGORY,
            "categorical": cls.FEEDBACK_MODE_CATEGORY,
            "legacy": cls.FEEDBACK_MODE_CATEGORY,
            "deterministic": cls.FEEDBACK_MODE_CATEGORY,
            "deterministic_feedback": cls.FEEDBACK_MODE_CATEGORY,
            "probabilistic": cls.FEEDBACK_MODE_BERNOULLI_CHOICE,
            "probabilistic_feedback": cls.FEEDBACK_MODE_BERNOULLI_CHOICE,
            "bernoulli": cls.FEEDBACK_MODE_BERNOULLI_CHOICE,
        }
        resolved = aliases.get(mode, mode)
        if resolved not in cls.VALID_FEEDBACK_MODES:
            raise ValueError(
                f"Unsupported feedback_likelihood_mode '{mode}'. "
                f"Expected one of: {cls.VALID_FEEDBACK_MODES}."
            )
        return resolved

    @staticmethod
    def _resolve_feedback_lapse(value: float) -> float:
        lapse = float(value)
        if not np.isfinite(lapse) or lapse < 0.0 or lapse >= 1.0:
            raise ValueError(f"feedback_lapse must be in [0, 1), got {value!r}.")
        return lapse


__all__ = ["BasePartition"]
