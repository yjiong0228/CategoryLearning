"""Model-facing partition for the discrete parity-rule space.

The rule inventory lives in :mod:`hypothesis_space.spaces`; rule evaluation
lives in :mod:`hypothesis_space.geometry.discrete_rule`. This module only joins those
layers to the likelihood API shared with continuous partitions.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..spaces import (
    DiscreteHypothesisSpace,
    build_discrete_hypothesis_space,
)
from ..geometry import DiscreteRuleGeometry
from .base_partition import BasePartition


class DiscreteRulePartition(BasePartition):
    """Facade for fixed-label parity rules over binary stimuli.

    Stimuli use ``-1`` and ``1`` values. For compatibility, positive inputs
    are coerced to ``1`` and non-positive inputs to ``-1``.
    """

    DISTANCE_MODE_RULE = "rule"
    DEFAULT_DISTANCE_MODE = DISTANCE_MODE_RULE
    VALID_DISTANCE_MODES = (DISTANCE_MODE_RULE,)

    def __init__(
        self,
        n_dims: int = 5,
        n_cats: int = 2,
        include_intercept: bool = False,
    ) -> None:
        super().__init__(n_dims, n_cats)
        self.include_intercept = bool(include_intercept)
        self.hypothesis_space: DiscreteHypothesisSpace = (
            build_discrete_hypothesis_space(
                n_dims=self.n_dims,
                n_cats=self.n_cats,
                include_intercept=self.include_intercept,
            )
        )
        self.rule_geometry = DiscreteRuleGeometry(self.hypothesis_space)

    @property
    def length(self) -> int:
        return len(self.hypothesis_space)

    @property
    def similarity_matrix(self) -> np.ndarray:
        warnings.warn(
            "DiscreteRulePartition.similarity_matrix is deprecated; call "
            "get_similarity_matrix() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.rule_geometry.similarity_matrix

    def get_similarity_matrix(
        self,
        *,
        kind: str = "assignment_agreement",
        distance_mode: str = DISTANCE_MODE_RULE,
        **kwargs,
    ) -> np.ndarray:
        del kwargs
        if kind != "assignment_agreement":
            raise ValueError(f"Unsupported discrete similarity kind '{kind}'.")
        self._resolve_distance_mode(distance_mode)
        return self.rule_geometry.similarity_matrix

    def describe_rule(self, hypo: int) -> str:
        return self.hypothesis_space[int(hypo)].label

    def get_category_assignment(
        self,
        hypo: int,
        stimulus: np.ndarray,
        distance_mode: str = DISTANCE_MODE_RULE,
        beta: float = 1.0,
        **kwargs,
    ) -> int:
        """Return the zero-based predicted category for one stimulus."""
        self._resolve_distance_mode(distance_mode)
        return int(
            self.rule_geometry.category_assignments(
                int(hypo),
                np.asarray([stimulus]),
            )[0]
        )

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
        return self.rule_geometry.category_probabilities(hypo, data[0], beta)

    def _category_feedback_likelihood(
        self,
        hypo: int,
        prob: np.ndarray,
        choices: np.ndarray,
        responses: np.ndarray,
    ) -> np.ndarray:
        """Preserve Exp5's binary correct/incorrect feedback semantics.

        Unlike Task2's continuous partition, this task has no related-family
        feedback code. Historically every value other than ``1`` meant an
        incorrect response; retaining that rule avoids a behavioral change.
        """
        del hypo
        p_choice = prob[choices, np.arange(len(choices))]
        return np.where(responses == 1, p_choice, 1.0 - p_choice)

__all__ = ["DiscreteRulePartition"]
