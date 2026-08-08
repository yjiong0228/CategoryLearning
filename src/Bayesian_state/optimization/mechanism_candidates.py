"""One-mechanism-at-a-time active-set candidate definitions for condition 1.

The registry keeps cognitive interpretation separate from run orchestration.
Every candidate starts from the same frozen C1 engine and changes only the
parameter paths declared here.  This makes substitution and incremental
comparisons auditable and prevents accidental carry-over from the V14
controller configuration.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


BASE_GAMMA = 0.55
BASE_W0 = 0.10
BASE_FEEDBACK_GAIN = 1.0
BASE_CAPACITY = 38
BASE_THETA = 0.0
BASE_BETA_ADDITIVE = 0.50
BASE_BETA_DECREASE = 0.15


@dataclass(frozen=True)
class MechanismCandidate:
    family: str
    candidate_id: str
    value: float
    parameters: tuple[tuple[str, Any], ...]
    is_reference: bool = False

    def parameter_dict(self) -> dict[str, Any]:
        return {key: deepcopy(value) for key, value in self.parameters}


def _token(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _finite_grid(values: Sequence[float], name: str) -> list[float]:
    grid = sorted({float(value) for value in values})
    if not grid or not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must contain finite values.")
    return grid


def feedback_candidates(
    values: Sequence[float] = (0.4, 0.7, 1.0, 1.4, 2.0),
) -> list[MechanismCandidate]:
    grid = _finite_grid(values, "feedback_gain grid")
    if any(value < 0.0 for value in grid):
        raise ValueError("feedback_gain candidates must be non-negative.")
    return [
        MechanismCandidate(
            family="F",
            candidate_id=f"F_kappa_{_token(value)}",
            value=value,
            parameters=((
                "engine.modules.memory_mod.kwargs.feedback_gain",
                value,
            ),),
            is_reference=np.isclose(value, BASE_FEEDBACK_GAIN),
        )
        for value in grid
    ]


def memory_candidates(
    values: Sequence[float] = (0.2, 0.4, 0.55, 0.7, 0.9),
) -> list[MechanismCandidate]:
    grid = _finite_grid(values, "gamma grid")
    if any(value < 0.0 or value > 1.0 for value in grid):
        raise ValueError("gamma candidates must lie in [0, 1].")
    return [
        MechanismCandidate(
            family="M",
            candidate_id=f"M_gamma_{_token(value)}",
            value=value,
            parameters=(("engine.modules.memory_mod.kwargs.gamma", value),),
            is_reference=np.isclose(value, BASE_GAMMA),
        )
        for value in grid
    ]


def capacity_candidates(
    values: Sequence[int] = (3, 5, 10, 38),
    *,
    shared_theta: float = 0.75,
) -> list[MechanismCandidate]:
    theta = float(shared_theta)
    if not np.isfinite(theta) or not 0.0 <= theta <= 1.0:
        raise ValueError("shared_theta must lie in [0, 1].")
    capacities = sorted({int(value) for value in values})
    if not capacities or any(value < 1 or value > BASE_CAPACITY for value in capacities):
        raise ValueError("capacity candidates must lie in [1, 38].")
    return [
        MechanismCandidate(
            family="H",
            candidate_id=f"H_capacity_{capacity}",
            value=float(capacity),
            parameters=(
                ("engine.modules.hypo_transitions_mod.kwargs.capacity", capacity),
                (
                    "engine.modules.hypo_transitions_mod.kwargs.theta",
                    BASE_THETA if capacity == BASE_CAPACITY else theta,
                ),
            ),
            is_reference=capacity == BASE_CAPACITY,
        )
        for capacity in capacities
    ]


def plasticity_candidates(
    values: Sequence[float] = (0.0, 0.5, 1.0, 1.5, 2.0),
) -> list[MechanismCandidate]:
    grid = _finite_grid(values, "plasticity grid")
    if any(value < 0.0 for value in grid):
        raise ValueError("plasticity candidates must be non-negative.")
    return [
        MechanismCandidate(
            family="P",
            candidate_id=f"P_zeta_{_token(value)}",
            value=value,
            parameters=(
                (
                    "engine.modules.beta_mod.kwargs.correct_additive",
                    BASE_BETA_ADDITIVE * value,
                ),
                (
                    "engine.modules.beta_mod.kwargs.decrease_rate",
                    BASE_BETA_DECREASE * value,
                ),
            ),
            is_reference=np.isclose(value, 1.0),
        )
        for value in grid
    ]


def strategy_candidates(
    values: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    *,
    capacity: int = 5,
    reference_theta: float = 0.75,
) -> list[MechanismCandidate]:
    grid = _finite_grid(values, "theta grid")
    if any(value < 0.0 or value > 1.0 for value in grid):
        raise ValueError("theta candidates must lie in [0, 1].")
    capacity_value = int(capacity)
    if not 1 <= capacity_value < BASE_CAPACITY:
        raise ValueError("strategy screening requires capacity in [1, 37].")
    reference_value = float(reference_theta)
    if not np.isfinite(reference_value) or reference_value not in grid:
        raise ValueError("reference_theta must be a member of the theta grid.")
    return [
        MechanismCandidate(
            family="S",
            candidate_id=f"S_theta_{_token(value)}",
            value=value,
            parameters=(
                (
                    "engine.modules.hypo_transitions_mod.kwargs.capacity",
                    capacity_value,
                ),
                ("engine.modules.hypo_transitions_mod.kwargs.theta", value),
            ),
            is_reference=np.isclose(value, reference_value),
        )
        for value in grid
    ]


def candidates_for_family(
    family: str,
    *,
    shared_theta: float = 0.75,
    strategy_capacity: int = 5,
) -> list[MechanismCandidate]:
    key = str(family).strip().upper()
    if key == "F":
        return feedback_candidates()
    if key == "M":
        return memory_candidates()
    if key == "H":
        return capacity_candidates(shared_theta=shared_theta)
    if key == "P":
        return plasticity_candidates()
    if key == "S":
        return strategy_candidates(capacity=strategy_capacity)
    raise ValueError(f"Unknown mechanism family {family!r}.")


def _set_by_path(root: dict[str, Any], path: str, value: Any) -> None:
    parts = str(path).split(".")
    if parts and parts[0] == "engine":
        parts = parts[1:]
    current: dict[str, Any] = root
    for part in parts[:-1]:
        nested = current.get(part)
        if not isinstance(nested, dict):
            nested = {}
            current[part] = nested
        current = nested
    current[parts[-1]] = deepcopy(value)


def apply_candidate(
    base_engine: Mapping[str, Any],
    candidate: MechanismCandidate,
) -> dict[str, Any]:
    config = deepcopy(dict(base_engine))
    for path, value in candidate.parameters:
        _set_by_path(config, path, value)
    return config


__all__ = [
    "MechanismCandidate",
    "apply_candidate",
    "candidates_for_family",
    "capacity_candidates",
    "feedback_candidates",
    "memory_candidates",
    "plasticity_candidates",
    "strategy_candidates",
]
