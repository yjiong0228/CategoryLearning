"""Experimental 0923 report grammar, generated without observing report text.

The normalized kernel describes category components with independent omissions
and optional fuzzy body-reference wording. It is not a fitted language model.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json

import numpy as np

from ..hypothesis_space.spaces.continuous import ContinuousHypothesisSpace

Report = tuple[str, ...]
EMPTY_REPORT = "EMPTY"
OTHER_REPORT = "OTHER"


def facet_token(A: np.ndarray, b: np.ndarray) -> str:
    """Canonical conjunction of halfspaces; positive scaling is immaterial."""
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    if A.ndim != 2 or b.shape != (len(A),) or not len(A):
        raise ValueError("expected nonempty A[m,d], b[m]")
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("constraints must be finite")
    scale = np.max(np.abs(A), axis=1)
    if np.any(scale == 0):
        raise ValueError("zero normal is not a report predicate")
    rows = np.round(np.column_stack((A, b)) / scale[:, None], 12)
    rows[rows == 0] = 0.0
    return "facet:" + json.dumps(sorted(set(map(tuple, rows.tolist()))), separators=(",", ":"))


def body_token(dimension: int, direction: str) -> str:
    """Symbolic comparison to a psychological reference; no hard threshold."""
    if not isinstance(dimension, (int, np.integer)) or isinstance(dimension, (bool, np.bool_)) or dimension < 0:
        raise ValueError("dimension must be a nonnegative integer")
    if direction not in {"gt", "lt"}:
        raise ValueError("body direction must be gt or lt")
    return f"body:{dimension}:{direction}"


def report_code(report: Report) -> str:
    return json.dumps(sorted(set(report)), separators=(",", ":")) if report else EMPTY_REPORT


def component_tokens(A: np.ndarray, b: np.ndarray) -> Report:
    """Opposing bounds form one band predicate, not two separate comparisons."""
    rows = np.array(json.loads(facet_token(A, b)[6:]))
    used: set[int] = set()
    tokens = []
    for i, row in enumerate(rows):
        if i in used:
            continue
        indices = [i]
        for j in range(i + 1, len(rows)):
            if j not in used and np.allclose(row[:-1], -rows[j, :-1], atol=1e-12, rtol=0):
                indices.append(j)
        used.update(indices)
        selected = rows[indices]
        tokens.append(facet_token(selected[:, :-1], selected[:, -1]))
    return tuple(sorted(set(tokens)))


def _body_variant(token: str) -> tuple[str, float] | None:
    rows = np.asarray(json.loads(token[6:]), dtype=float)
    if len(rows) != 1:
        return None
    dimensions = np.flatnonzero(rows[0, :-1])
    if len(dimensions) != 1:
        return None
    dim = int(dimensions[0])
    coefficient = rows[0, dim]
    return body_token(dim, "lt" if coefficient > 0 else "gt"), float(rows[0, -1] / coefficient)


@dataclass(frozen=True)
class ReportKernelParameters:
    """Shared measurement parameters. No subject-specific shift is introduced."""

    mention_probability: float
    other_probability: float
    body_naming_probability: float
    body_reference_center: float
    body_reference_width: float

    def __post_init__(self) -> None:
        for name in ("mention_probability", "other_probability", "body_naming_probability", "body_reference_center"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if not np.isfinite(self.body_reference_width) or self.body_reference_width <= 0:
            raise ValueError("body_reference_width must be positive and finite")


def _component_distribution(description: Report, parameters: ReportKernelParameters | None) -> dict[Report, float]:
    distribution: dict[Report, float] = {(): 1.0}
    for token in description:
        variant = _body_variant(token)
        if parameters is None:
            # Enumerate support independently of calibration parameter values.
            choices = [(None, 1.0), (token, 1.0)]
            if variant is not None:
                choices.append((variant[0], 1.0))
        else:
            q = parameters.mention_probability
            body_mass = 0.0
            if variant is not None:
                distance = (variant[1] - parameters.body_reference_center) / parameters.body_reference_width
                body_mass = parameters.body_naming_probability * np.exp(-0.5 * distance**2)
            choices = [(None, 1 - q), (token, q * (1 - body_mass))]
            if variant is not None:
                choices.append((variant[0], q * body_mass))
        updated: dict[Report, float] = defaultdict(float)
        for report, mass in distribution.items():
            for wording, probability in choices:
                outcome = tuple(sorted(set(report + ((wording,) if wording else ()))))
                updated[outcome] += mass * probability
        distribution = dict(updated)
    return distribution


@dataclass(frozen=True)
class CatalogueReportSpace:
    """Fixed report outcomes for a fixed catalogue, including EMPTY and OTHER.

    A union category first selects one component uniformly. Wording and omission
    are then sampled per predicate. This first candidate ignores x conditional
    on (h, choice); stimulus-dependent extra descriptions are not yet modeled.
    """

    codes: tuple[str, ...]
    components: tuple[tuple[tuple[Report, ...], ...], ...]

    @classmethod
    def from_catalogue(cls, space: ContinuousHypothesisSpace) -> CatalogueReportSpace:
        components = []
        outcomes: set[str] = set()
        for hypothesis in space:
            categories = []
            for category in hypothesis.categories:
                descriptions = tuple(component_tokens(part.A, part.b) for part in category.components)
                categories.append(descriptions)
                for description in descriptions:
                    outcomes.update(report_code(outcome) for outcome in _component_distribution(description, None))
            components.append(tuple(categories))
        outcomes.discard(EMPTY_REPORT)
        return cls((EMPTY_REPORT, OTHER_REPORT, *sorted(outcomes)), tuple(components))

    def matrix(self, parameters: ReportKernelParameters) -> np.ndarray:
        """Return normalized R[H,C,O], without compatibility-score normalization."""
        lookup = {code: index for index, code in enumerate(self.codes)}
        matrix = np.zeros((len(self.components), len(self.components[0]), len(self.codes)))
        matrix[:, :, lookup[OTHER_REPORT]] = parameters.other_probability
        for h, categories in enumerate(self.components):
            for y, descriptions in enumerate(categories):
                for description in descriptions:
                    for outcome, mass in _component_distribution(description, parameters).items():
                        matrix[h, y, lookup[report_code(outcome)]] += (
                            (1 - parameters.other_probability) * mass / len(descriptions)
                        )
        if not np.isfinite(matrix).all() or not np.allclose(matrix.sum(-1), 1, atol=1e-12, rtol=0):
            raise ValueError("report probability conservation failed")
        return matrix

    def source_index(self) -> dict[tuple[str, int], dict[str, list[int]]]:
        """Symbolic support at each label; support does not imply a good fit."""
        index: dict[tuple[str, int], dict[str, list[int]]] = {}
        for h, categories in enumerate(self.components):
            for y, descriptions in enumerate(categories):
                for description in descriptions:
                    for outcome in _component_distribution(description, None):
                        if not outcome:
                            continue
                        status = "fuzzy_reference" if any(t.startswith("body:") for t in outcome) else (
                            "full_component" if outcome == description else "omission"
                        )
                        values = index.setdefault((report_code(outcome), y), {
                            "full_component": [], "omission": [], "fuzzy_reference": [],
                        })[status]
                        if h not in values:
                            values.append(h)
        return index
