"""Exact five-square parity rules for Model 1024's Exp5 interface baseline.

The legacy discrete partition retains its permissive input coercion. This
opt-in partition requires black/white values, supplies unit mismatch distances
for the shared precision derivative, and enumerates all 32 task stimuli for
functional relations. Local search uses explicit one-edit rule neighbors.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .discrete_rule_partition import DiscreteRulePartition

GRAMMAR_VERSION = "exp5_parity_position_polarity_edits_v1"


def _readonly(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


class Exp5ParityPartition(DiscreteRulePartition):
    """62 fixed-label rules with exact binary inputs and task-specific relations."""

    def __init__(self, n_dims: int = 5, n_cats: int = 2, include_intercept: bool = False) -> None:
        if n_dims != 5 or n_cats != 2 or include_intercept:
            raise ValueError("Exp5 baseline requires five features, two categories and 62 rules")
        super().__init__(n_dims, n_cats, include_intercept)
        self._search_resources = {}

    def _stimuli(self, stimuli: Sequence) -> np.ndarray:
        values = np.asarray(stimuli, dtype=float)
        if (values.ndim != 2 or values.shape[1] != 5
                or not np.isin(values, [-1., 1.]).all()):
            raise ValueError("Exp5 stimuli must be finite (n, 5) black/white values -1 or 1")
        return values

    def get_category_assignment(self, hypo: int, stimulus: Sequence, distance_mode: str = "rule",
                                beta: float = 1., **kwargs) -> int:
        self._stimuli([stimulus])
        return super().get_category_assignment(hypo, stimulus, distance_mode, beta, **kwargs)

    def get_category_probabilities(self, hypo: int, data: Sequence, beta: float,
                                   distance_mode: str = "rule", **kwargs) -> np.ndarray:
        self._stimuli(data[0])
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("Exp5 precision must be finite and nonnegative")
        return super().get_category_probabilities(hypo, data, beta, distance_mode, **kwargs)

    def precision_category_distances(self, stimulus: Sequence, hypotheses: Sequence[int]) -> np.ndarray:
        """Return d(h,c)=0 for the rule's label, 1 for the other label.

        softmax(-beta*d) is exactly the existing discrete rule emission. The
        generic feedback derivative therefore also updates all active rules.
        """
        values = self._stimuli([stimulus])
        predicted = np.array([self.rule_geometry.category_assignments(h, values)[0]
                              for h in hypotheses])
        return (predicted[:, None] != np.arange(2)).astype(float)

    def rule_search_resources(self, edit_costs: Mapping[str, float], *, distance_mode: str) -> tuple:
        """Return grammar, programs, one-edit costs and exact label confusion.

        A position edit adds, removes or substitutes one position; a polarity
        edit reverses the two labels with the same positions. Other pairs have
        infinite direct cost. Functional relations use the uniform full design,
        independently of the subject's observed stimulus frequencies.
        """
        self._resolve_distance_mode(distance_mode)
        if not isinstance(edit_costs, Mapping) or set(edit_costs) != {"position", "polarity"}:
            raise ValueError("Exp5 edit_costs requires position and polarity")
        costs = {key: float(value) for key, value in edit_costs.items()}
        if any(not np.isfinite(value) or value <= 0 for value in costs.values()):
            raise ValueError("Exp5 edit costs must be finite and positive")
        key = tuple(sorted(costs.items()))
        if key not in self._search_resources:
            programs = tuple(self.hypothesis_space)
            distance = np.full((self.length, self.length), np.inf)
            np.fill_diagonal(distance, 0.)
            for i, first in enumerate(programs):
                for j in range(i + 1, len(programs)):
                    second = programs[j]
                    left, right = set(first.dims), set(second.dims)
                    if left == right and first.polarity != second.polarity:
                        distance[i, j] = distance[j, i] = costs["polarity"]
                    elif first.polarity == second.polarity and max(len(left-right), len(right-left)) == 1:
                        distance[i, j] = distance[j, i] = costs["position"]
            predictions = self.rule_geometry.prediction_table
            indicators = np.eye(2)[predictions]
            confusion = np.einsum("hsc,ksd->hkcd", indicators, indicators) / len(predictions[0])
            self._search_resources[key] = (programs, _readonly(distance), _readonly(confusion))
        programs, distance, confusion = self._search_resources[key]
        return GRAMMAR_VERSION, programs, distance.view(), confusion.view()
