"""Canonical judgment programs and explicit edit costs for Model 0923 full model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .spaces.continuous import ContinuousHypothesisSpace, ContinuousHypothesisSpec


GRAMMAR_VERSION = "judgment_program_edits_v1"
EDIT_OPERATIONS = frozenset({"feature", "relation", "reference", "branch"})


@dataclass(frozen=True)
class Judgment:
    kind: str
    positive: tuple[int, ...]
    negative: tuple[int, ...] = ()
    reference: tuple[float, ...] = ()

    def margin(self, stimuli: np.ndarray) -> np.ndarray:
        if self.kind == "center_band":
            center, tolerance = self.reference
            return np.abs(stimuli[:, self.positive[0]] - center) - tolerance
        if self.kind == "difference_band":
            first, second = self.positive
            return np.abs(stimuli[:, first] - stimuli[:, second]) - self.reference[0]
        value = stimuli[:, self.positive].sum(axis=1)
        if self.negative:
            value -= stimuli[:, self.negative].sum(axis=1)
        return value - self.reference[0]

    def evaluate(self, stimuli: np.ndarray) -> np.ndarray:
        return self.margin(stimuli) <= 0


@dataclass(frozen=True)
class RuleProgram:
    topology: str
    judgments: tuple[Judgment, ...]
    labels: tuple

    def evaluate(self, stimuli: np.ndarray) -> np.ndarray:
        """Evaluate for representation audits; live emissions use shared geometry."""
        values = np.asarray(stimuli, dtype=float)
        if self.topology == "extreme":
            judgment = self.judgments[0]
            local = values[:, judgment.positive]
            method = np.argmax if judgment.kind == "maximum" else np.argmin
            return np.asarray(self.labels)[method(local, axis=1)]
        margins = np.asarray([test.margin(values) for test in self.judgments]).T
        if self.topology == "binary":
            return np.where(margins[:, 0] <= 0, self.labels[0], self.labels[1])
        matches = []
        for components in self.labels:
            component_matches = []
            for pattern in components:
                selected = np.asarray(pattern) != 0
                signs = np.asarray(pattern)[selected]
                component_matches.append(np.all(margins[:, selected] * signs <= 0, axis=1))
            matches.append(np.any(component_matches, axis=0))
        matches = np.asarray(matches)
        if not np.all(matches.any(axis=0)):
            raise ValueError("judgment program does not cover every supplied stimulus")
        return np.argmax(matches, axis=0)


def _plane_judgment(plane: tuple) -> Judgment:
    coefficients, intercept = plane
    coefficients = np.asarray(coefficients, dtype=float)
    nonzero = coefficients[coefficients != 0]
    scale = float(np.abs(nonzero[0]))
    if not np.allclose(np.abs(nonzero), scale) or not np.any(coefficients > 0):
        raise ValueError("0923 full model judgment grammar requires canonical equal-weight planes")
    positive = tuple(np.flatnonzero(coefficients > 0).tolist())
    negative = tuple(np.flatnonzero(coefficients < 0).tolist())
    kinds = {(1, 0): "axis", (1, 1): "difference", (2, 0): "sum",
             (2, 2): "paired_sum"}
    kind = kinds.get((len(positive), len(negative)))
    if kind is None:
        raise ValueError("unsupported judgment arity")
    return Judgment(kind, positive, negative, (float(intercept) / scale,))


def _category_patterns(spec: ContinuousHypothesisSpec, order: list[int]) -> tuple:
    categories = []
    for category in spec.categories:
        components = []
        for component in category.components:
            pattern = [0] * len(spec.hyperplanes)
            for row, bound in zip(component.A, component.b):
                found = False
                for position, (coefficients, intercept) in enumerate(spec.hyperplanes):
                    for sign in (1, -1):
                        if (np.allclose(row, sign * np.asarray(coefficients), atol=1e-12)
                                and np.isclose(bound, sign * intercept, atol=1e-12)):
                            pattern[position] = sign
                            found = True
                            break
                    if found:
                        break
                if not found:
                    raise ValueError("category constraint is absent from the judgment program")
            components.append(tuple(pattern[index] for index in order))
        categories.append(tuple(sorted(set(components))))
    return tuple(categories)


def compile_rule_program(spec: ContinuousHypothesisSpec) -> RuleProgram:
    """Represent all adopted 29/128 rules without changing their labels."""
    params = spec.parameters
    if spec.family == "univariate_center_band":
        test = Judgment("center_band", (int(params["dimension"]),), (),
                        (float(params["center"]), float(params["tolerance"])))
        return RuleProgram("binary", (test,), tuple(spec.label_permutation))
    if spec.family == "pairwise_similarity_band":
        test = Judgment("difference_band", tuple(params["dimensions"]), (),
                        (float(params["tolerance"]),))
        return RuleProgram("binary", (test,), tuple(spec.label_permutation))
    if spec.family in {"dimension_max", "dimension_min"}:
        kind = "maximum" if spec.family == "dimension_max" else "minimum"
        dims = tuple(range(len(spec.categories)))
        return RuleProgram("extreme", (Judgment(kind, dims),),
                           tuple(spec.label_permutation))
    tests = tuple(_plane_judgment(plane) for plane in spec.hyperplanes)
    if len(spec.categories) == 2:
        if len(tests) != 1:
            raise ValueError("unsupported binary judgment topology")
        return RuleProgram("binary", tests, tuple(spec.label_permutation))
    topology = "cross" if len(tests) == 2 else "tree"
    if topology == "tree" and len(tests) != 3:
        raise ValueError("unsupported four-category judgment topology")
    # Parallel tests are commutative only together with their labeled outputs.
    # Branch tests retain their positions; swapping branches is not an alias.
    order = sorted(range(len(tests)), key=lambda i: repr(tests[i])) if topology == "cross" else list(range(len(tests)))
    return RuleProgram(topology, tuple(tests[index] for index in order),
                       _category_patterns(spec, order))


def validate_edit_costs(costs: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(costs, Mapping) or set(costs) != EDIT_OPERATIONS:
        raise ValueError(f"edit_costs requires exactly {sorted(EDIT_OPERATIONS)}")
    resolved = {key: float(value) for key, value in costs.items()}
    if any(not np.isfinite(value) or value <= 0 for value in resolved.values()):
        raise ValueError("edit costs must be finite and strictly positive")
    return resolved


def _judgment_edit(first: Judgment, second: Judgment) -> str | None:
    if first == second:
        return None
    if first.kind == second.kind and first.reference == second.reference:
        if (len(first.positive), len(first.negative)) == (len(second.positive), len(second.negative)):
            changes = (len(set(first.positive) ^ set(second.positive))
                       + len(set(first.negative) ^ set(second.negative))) // 2
            if changes == 1:
                return "feature"
    first_features = sorted(first.positive + first.negative)
    second_features = sorted(second.positive + second.negative)
    if first.kind != second.kind and first_features == second_features:
        # Substituting the comparison primitive includes its standard reference,
        # e.g. x_i < x_j versus x_i + x_j < 1, not an arbitrary fitted threshold.
        return "relation"
    if (first.kind, first.positive, first.negative) == (second.kind, second.positive, second.negative):
        if len(first.reference) == len(second.reference) and sum(a != b for a, b in zip(first.reference, second.reference)) == 1:
            return "reference"
    return None


def build_rule_edit_costs(
    space: ContinuousHypothesisSpace, costs: Mapping[str, float],
) -> tuple[tuple[RuleProgram, ...], np.ndarray]:
    """Return legal one-edit costs, using infinity for unavailable edits."""
    costs = validate_edit_costs(costs)
    programs = tuple(compile_rule_program(spec) for spec in space)
    if len(set(programs)) != len(programs):
        raise ValueError("the rule catalogue contains duplicate canonical programs")
    distance = np.full((len(space), len(space)), np.inf)
    np.fill_diagonal(distance, 0)
    for i, first in enumerate(programs):
        for j in range(i + 1, len(programs)):
            second = programs[j]
            if first.topology != second.topology or first.labels != second.labels:
                continue
            changed = [k for k, (a, b) in enumerate(zip(first.judgments, second.judgments)) if a != b]
            if len(changed) != 1:
                continue
            position = changed[0]
            operation = _judgment_edit(first.judgments[position], second.judgments[position])
            options = [] if operation is None else [costs[operation]]
            if first.topology == "tree" and position > 0:
                options.append(costs["branch"])
            if options:
                distance[i, j] = distance[j, i] = min(options)
    distance.setflags(write=False)
    return programs, distance
