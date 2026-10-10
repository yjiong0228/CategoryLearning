"""Reaction-time observation channel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math

import numpy as np

from .shared import _float_from_mapping, normalize_probability_vector


@dataclass(frozen=True)
class ReactionTimeReadoutResult:
    """Student-t parameters for log reaction time."""

    log_location: float
    scale: float
    degrees_of_freedom: float
    choice_uncertainty: float


def read_reaction_time(
    choice_probabilities: Sequence[float] | np.ndarray,
    *,
    trial_index: int,
    replacement_fraction: float = 0.0,
    newcomer_distance: float = 0.0,
    config: Mapping[str, Any] | None = None,
) -> ReactionTimeReadoutResult:
    """Return a normalized-entropy RT readout without sampling an RT value."""

    cfg = dict(config or {})
    probabilities = normalize_probability_vector(choice_probabilities, strict=True)
    if probabilities.size <= 1:
        uncertainty = 0.0
    else:
        uncertainty = float(
            -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)))
            / math.log(probabilities.size)
        )
    context = "reaction_time_readout"
    location = (
        _float_from_mapping(cfg, "intercept", 0.0, context=context)
        + _float_from_mapping(cfg, "choice_uncertainty", 0.0, context=context) * uncertainty
        + _float_from_mapping(cfg, "replacement_fraction", 0.0, context=context)
        * float(replacement_fraction)
        + _float_from_mapping(cfg, "newcomer_distance", 0.0, context=context)
        * float(newcomer_distance)
        + _float_from_mapping(cfg, "practice", 0.0, context=context)
        * math.log1p(max(0, int(trial_index)))
    )
    return ReactionTimeReadoutResult(
        log_location=float(location),
        scale=_float_from_mapping(cfg, "scale", 1.0, min_value=1e-12, context=context),
        degrees_of_freedom=_float_from_mapping(
            cfg, "degrees_of_freedom", 5.0, min_value=2.0, context=context
        ),
        choice_uncertainty=uncertainty,
    )


__all__ = ["ReactionTimeReadoutResult", "read_reaction_time"]
