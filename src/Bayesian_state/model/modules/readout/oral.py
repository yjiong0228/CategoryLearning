"""Causal report readout: local rule content, omissions, and one extra feature.

This is an observation component, not an engine learning module. Its grammar
is frozen from the catalogue before seeing reports, stimuli or parameter values.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ....hypothesis_space.geometry.boundary import BoundaryGeometry
from ....hypothesis_space.spaces.continuous import ContinuousHypothesisSpace
from .observation import predict_choice_oral
from .report_space import (
    EMPTY_REPORT, OTHER_REPORT, ReportKernelParameters,
    _component_distribution, component_tokens, facet_token, report_code,
)
from .shared import normalize_probability_vector


@dataclass(frozen=True)
class OralReadoutParameters(ReportKernelParameters):
    """Shared expression parameters; no participant-specific learning state."""

    extra_probability: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not np.isfinite(self.extra_probability) or not 0 <= self.extra_probability <= 1:
            raise ValueError('extra_probability must be finite and in [0,1]')

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> OralReadoutParameters:
        if not isinstance(config, Mapping) or set(config) != {'method', 'kwargs'}:
            raise ValueError('oral_readout requires exactly method and kwargs')
        if config['method'] != 'rule_content_v1' or not isinstance(config['kwargs'], Mapping):
            raise ValueError('oral_readout requires rule_content_v1 and parameter kwargs')
        return cls(**dict(config['kwargs']))


@dataclass(frozen=True)
class OralPrediction:
    """One selected-category distribution; choices are one-based task responses."""

    codes: tuple[str, ...]
    probabilities: np.ndarray
    extra_probabilities: np.ndarray
    choice: int
    choice_probability: float
    hypothesis_indices: tuple[int, ...]
    rule_weights_given_choice: np.ndarray

    def probability(self, code: str) -> float:
        """Unknown codes have zero mass; they are not automatically OTHER."""
        return float(self.probabilities[self.codes.index(code)]) if code in self.codes else 0.0

    def extra_given_report(self, code: str) -> float:
        """Model attribution of an extra phrase, not a human annotation."""
        mass = self.probability(code)
        if mass <= 0:
            raise ValueError('cannot condition on a zero-probability report')
        return float(self.extra_probabilities[self.codes.index(code)] / mass)

    def sample(self, rng: np.random.Generator) -> str:
        """Use a separate caller-owned oral RNG to preserve behavioral seeds."""
        return self.codes[int(rng.choice(len(self.codes), p=self.probabilities))]


class OralReadout:
    """Read local rule content after choice and before external feedback.

    A disconnected category uses its nearest component(s), with uniform ties.
    Every core predicate is independently omitted/worded using the existing R1
    kernel. At most one ordinary .5-threshold description is added, uniformly
    among dimensions unused by that complete component, even if core clauses
    were omitted. These restrictions prevent duplicate/contradictory extras.
    No eligible extra (including an exact .5 value) means no extra event.
    """

    def __init__(self, space: ContinuousHypothesisSpace, parameters: OralReadoutParameters,
                 *, boundary_geometry: BoundaryGeometry | None = None) -> None:
        if not isinstance(space, ContinuousHypothesisSpace):
            raise TypeError('oral readout requires a continuous hypothesis space')
        self.space, self.parameters = space, parameters
        self.geometry = boundary_geometry if boundary_geometry is not None else BoundaryGeometry(space)
        if self.geometry.space.signature != space.signature:
            raise ValueError('oral geometry and rule catalogue must agree')
        axes = np.eye(space.n_dims)
        self._short = tuple(facet_token([row], [.5]) for row in axes)
        self._long = tuple(facet_token([-row], [-.5]) for row in axes)
        self._components: dict[tuple[int, int], tuple[tuple[dict, tuple[int, ...]], ...]] = {}
        outcomes = {EMPTY_REPORT, OTHER_REPORT}
        for h in space:
            for y, category in enumerate(h.categories):
                components = []
                for part in category.components:
                    description = component_tokens(part.A, part.b)
                    unused = tuple(np.flatnonzero(~np.any(part.A != 0, axis=0)).tolist())
                    support = _component_distribution(description, None)
                    for report in support:
                        outcomes.add(report_code(report))
                        for dim in unused:
                            for extra in (self._short[dim], self._long[dim]):
                                outcomes.add(report_code(report + (extra,)))
                    components.append((_component_distribution(description, parameters), unused))
                self._components[h.index, y] = tuple(components)
        self.codes = (EMPTY_REPORT, OTHER_REPORT, *sorted(outcomes - {EMPTY_REPORT, OTHER_REPORT}))
        self._lookup = {code: i for i, code in enumerate(self.codes)}

    def _stimulus(self, stimulus: np.ndarray) -> np.ndarray:
        values = np.asarray(stimulus, dtype=float)
        if values.shape != (self.space.n_dims,) or not np.isfinite(values).all():
            raise ValueError('perceived stimulus must be one finite feature vector')
        return values

    def _distribution(self, hypothesis: int, category_index: int, stimulus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        category = self.space[hypothesis].categories[category_index]
        components = self._components[hypothesis, category_index]
        if len(components) == 1:
            selected = np.array([0])
        else:
            distances = np.array([self.geometry.distances_to_polytope(stimulus[None, :], part)[0]
                                  for part in category.components])
            selected = np.flatnonzero(np.isclose(distances, distances.min(), atol=self.geometry.tolerance, rtol=0))
        probabilities = np.zeros(len(self.codes))
        extras = np.zeros(len(self.codes))
        probabilities[self._lookup[OTHER_REPORT]] = self.parameters.other_probability
        scale = (1 - self.parameters.other_probability) / len(selected)
        for index in selected:
            core, unused = components[index]
            additions = [self._long[dim] if stimulus[dim] > .5 else self._short[dim]
                         for dim in unused if stimulus[dim] != .5]
            chance = self.parameters.extra_probability if additions else 0.0
            for report, mass in core.items():
                probabilities[self._lookup[report_code(report)]] += scale * mass * (1 - chance)
                for extra in additions:
                    code = self._lookup[report_code(report + (extra,))]
                    contribution = scale * mass * chance / len(additions)
                    probabilities[code] += contribution
                    extras[code] += contribution
        return probabilities, extras

    def distribution(self, hypothesis: int, choice: int, perceived_stimulus: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return P(report|h,choice,x) and its extra-event probability mass."""
        self._validate_choice(choice)
        if isinstance(hypothesis, (bool, np.bool_)) or not isinstance(hypothesis, (int, np.integer)) or not 0 <= hypothesis < len(self.space):
            raise ValueError('hypothesis must index the rule catalogue')
        return self._distribution(int(hypothesis), int(choice) - 1, self._stimulus(perceived_stimulus))

    def _validate_choice(self, choice: int) -> None:
        if isinstance(choice, (bool, np.bool_)) or not isinstance(choice, (int, np.integer)) or not 1 <= choice <= self.space.n_cats:
            raise ValueError('choice must be a one-based integer task response')

    def predict(self, *, hypothesis_indices: tuple[int, ...], rule_weights: np.ndarray,
                category_probabilities: np.ndarray, choice: int,
                perceived_stimulus: np.ndarray) -> OralPrediction:
        """Condition on a choice using the SAME rule source as decision readout.

        Work within one model/particle. PF aggregation must use the joint
        choice-report mass before conditioning; it is not implemented here.
        """
        self._validate_choice(choice)
        stimulus = self._stimulus(perceived_stimulus)
        indices = tuple(hypothesis_indices)
        if (not indices or any(isinstance(h, (bool, np.bool_)) or not isinstance(h, (int, np.integer))
                               or not 0 <= h < len(self.space) for h in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError('hypothesis_indices must be unique catalogue indices')
        weights = normalize_probability_vector(rule_weights, strict=True)
        choices = np.asarray(category_probabilities, dtype=float)
        if weights.shape != (len(indices),) or choices.shape != (len(indices), self.space.n_cats):
            raise ValueError('rule weights, indices and category probabilities must align')
        # Zero-weight rows need no geometry/readout. Live decision readout stores
        # zeros in those rows; active rows still require normalized probabilities.
        active = np.flatnonzero(weights > 0)
        selected_weights = weights[active]
        selected_choices = choices[active]
        kernel = np.zeros((len(active), self.space.n_cats, len(self.codes)))
        extra_kernel = np.zeros_like(kernel)
        for i, source in enumerate(active):
            for y in range(self.space.n_cats):
                kernel[i, y], extra_kernel[i, y] = self._distribution(indices[source], y, stimulus)
        prediction = predict_choice_oral(selected_weights, selected_choices, kernel)
        category = int(choice) - 1
        probabilities = prediction.report_given_choice(category)
        posterior = selected_weights * selected_choices[:, category]
        posterior /= posterior.sum()
        extra_mass = posterior @ extra_kernel[:, category]
        all_weights = np.zeros(len(indices)); all_weights[active] = posterior
        return OralPrediction(self.codes, probabilities, extra_mass, int(choice),
                              float(prediction.choice_probabilities[category]), indices, all_weights)
