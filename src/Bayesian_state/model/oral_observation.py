"""Choice-before-report observation kernel; no text encoder is assumed."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .readout import normalize_probability_vector, read_oral_report


@dataclass(frozen=True)
class ChoiceOralPrediction:
    """Joint probabilities on [category, report code], before feedback."""

    choice_probabilities: np.ndarray
    joint_probabilities: np.ndarray

    def report_given_choice(self, category_index: int) -> np.ndarray:
        """Return P(report | observed choice), with zero-based category index."""
        if isinstance(category_index, (bool, np.bool_)) or not isinstance(
            category_index, (int, np.integer)
        ):
            raise ValueError("category_index must be a zero-based integer")
        if not 0 <= category_index < self.choice_probabilities.size:
            raise ValueError("category_index is out of range")
        mass = float(self.choice_probabilities[category_index])
        if mass <= 0.0:
            raise ValueError("cannot condition a report on an impossible choice")
        return self.joint_probabilities[category_index] / mass


def predict_choice_oral(
    hypothesis_weights: np.ndarray,
    category_probabilities: np.ndarray,
    report_mapping: np.ndarray,
    *,
    reliability: float = 1.0,
    baseline: np.ndarray | None = None,
) -> ChoiceOralPrediction:
    """Marginalize one rule shared by choice and the subsequent report.

    ``category_probabilities`` is [H, C]. ``report_mapping`` is [H, C, R]
    and describes reports about the selected category, including omissions.
    Rows must be probability distributions, not posterior-over-rule oral
    compatibility scores. A calibrated report vocabulary/kernel is a required
    external input; this function neither supplies one nor fits real reports.
    No current feedback or target category is an input.
    """
    weights = normalize_probability_vector(hypothesis_weights, strict=True)
    choices = np.asarray(category_probabilities, dtype=float)
    mapping = np.asarray(report_mapping, dtype=float)
    if choices.ndim != 2 or choices.shape[0] != weights.size or choices.shape[1] < 2:
        raise ValueError("category_probabilities must have shape [H, C] with C >= 2")
    if mapping.ndim != 3 or mapping.shape[:2] != choices.shape or mapping.shape[2] < 1:
        raise ValueError("report_mapping must have shape [H, C, R]")
    for name, values in (("category_probabilities", choices), ("report_mapping", mapping)):
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
        if not np.allclose(values.sum(axis=-1), 1.0, rtol=0.0, atol=1e-10):
            raise ValueError(f"{name} rows must sum to 1")
    choice_mass = weights @ choices
    joint = np.zeros((choices.shape[1], mapping.shape[2]), dtype=float)
    for category_index in range(choices.shape[1]):
        weighted_choice = weights * choices[:, category_index]
        # Calling the shared readout also validates report noise on zero-mass
        # branches. Their arbitrary conditional distributions are never exposed.
        report = read_oral_report(
            weighted_choice if choice_mass[category_index] > 0.0 else weights,
            mapping[:, category_index, :],
            reliability=reliability,
            baseline=baseline,
        )
        joint[category_index] = choice_mass[category_index] * report.probabilities
    return ChoiceOralPrediction(choice_mass, joint)
