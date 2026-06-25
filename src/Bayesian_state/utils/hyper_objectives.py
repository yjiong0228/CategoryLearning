"""Objective ordering helpers for hyperparameter optimizers."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cmp_to_key
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, TypeVar

import numpy as np

from src.Bayesian_state.utils.simulation_statistics import get_stat_value


LEGACY_HYPER_CONFIG_KEYS = (
    "selection_metric",
    "secondary_selection",
    "simulation_statistics",
    "tie_break_metric",
    "acceptance_selection",
)
OBJECTIVE_ROOTS = {"simulation", "statistics"}
ObjectiveValues = Dict[str, float]
T = TypeVar("T")


@dataclass(frozen=True)
class ObjectiveSpec:
    path: str
    rel_tolerance: float
    abs_tolerance: float
    scale_floor: float
    anchor_guard: bool = False

    def tolerance(self, left: float, right: float) -> float:
        scale = max(abs(float(left)), abs(float(right)), self.scale_floor)
        return max(self.abs_tolerance, self.rel_tolerance * scale)

    def anchor_tolerance(self, value: float) -> float:
        scale = max(abs(float(value)), self.scale_floor)
        return max(self.abs_tolerance, self.rel_tolerance * scale)


def safe_objective_float(value: Any, default: float = float("inf")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def validate_no_legacy_hyper_config(config: Mapping[str, Any]) -> None:
    found = [key for key in LEGACY_HYPER_CONFIG_KEYS if key in config]
    if found:
        raise ValueError(
            "Legacy hyper config keys are not supported: "
            + ", ".join(found)
            + ". Use objective_order and statistics_config instead."
        )


def resolve_objective_order(config: Mapping[str, Any]) -> list[ObjectiveSpec]:
    validate_no_legacy_hyper_config(config)
    raw = config.get("objective_order")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("objective_order must be a non-empty list of objective definitions.")

    specs: list[ObjectiveSpec] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"objective_order[{idx}] must be a mapping.")
        direction = str(item.get("direction", "minimize")).strip().lower()
        if direction != "minimize":
            raise ValueError(f"objective_order[{idx}].direction must be 'minimize'.")
        path = str(item.get("path") or "").strip()
        if "." not in path:
            raise ValueError(f"objective_order[{idx}].path must be a dotted path.")
        root = path.split(".", 1)[0]
        if root not in OBJECTIVE_ROOTS:
            raise ValueError(
                f"objective_order[{idx}].path must start with 'simulation.' or 'statistics.'."
            )
        rel_tolerance = _non_negative_float(
            item.get("rel_tolerance", 0.0),
            f"objective_order[{idx}].rel_tolerance",
        )
        abs_tolerance = _non_negative_float(
            item.get("abs_tolerance", 0.0),
            f"objective_order[{idx}].abs_tolerance",
        )
        scale_floor = _non_negative_float(
            item.get("scale_floor", 0.0),
            f"objective_order[{idx}].scale_floor",
        )
        specs.append(
            ObjectiveSpec(
                path=path,
                rel_tolerance=rel_tolerance,
                abs_tolerance=abs_tolerance,
                scale_floor=scale_floor,
                anchor_guard=bool(item.get("anchor_guard", False)),
            )
        )
    return specs


def _non_negative_float(value: Any, name: str) -> float:
    out = safe_objective_float(value, default=float("nan"))
    if not np.isfinite(out) or out < 0:
        raise ValueError(f"{name} must be a non-negative finite number.")
    return float(out)


def objective_order_payload(specs: Sequence[ObjectiveSpec]) -> list[dict[str, Any]]:
    return [
        {
            "path": spec.path,
            "rel_tolerance": spec.rel_tolerance,
            "abs_tolerance": spec.abs_tolerance,
            "scale_floor": spec.scale_floor,
            "anchor_guard": spec.anchor_guard,
        }
        for spec in specs
    ]


def extract_subject_objective_values(
    subject_record: Mapping[str, Any],
    specs: Sequence[ObjectiveSpec],
) -> ObjectiveValues:
    return {
        spec.path: safe_objective_float(get_stat_value(subject_record, spec.path, float("inf")))
        for spec in specs
    }


def aggregate_objective_values(
    values_by_subject: Iterable[Mapping[str, Any]],
    specs: Sequence[ObjectiveSpec],
) -> ObjectiveValues:
    aggregated: ObjectiveValues = {}
    rows = list(values_by_subject)
    for spec in specs:
        values = [safe_objective_float(row.get(spec.path)) for row in rows]
        if not values or any(not np.isfinite(value) for value in values):
            aggregated[spec.path] = float("inf")
        else:
            aggregated[spec.path] = float(np.mean(values))
    return aggregated


def first_objective_value(values: Mapping[str, Any], specs: Sequence[ObjectiveSpec]) -> float:
    if not specs:
        return float("inf")
    return safe_objective_float(values.get(specs[0].path))


def compare_objective_values(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    specs: Sequence[ObjectiveSpec],
) -> int:
    """Return -1 when left is better, 1 when right is better, and 0 on a tie."""
    for spec in specs:
        lv = safe_objective_float(left.get(spec.path))
        rv = safe_objective_float(right.get(spec.path))
        finite_left = np.isfinite(lv)
        finite_right = np.isfinite(rv)
        if finite_left and not finite_right:
            return -1
        if finite_right and not finite_left:
            return 1
        if not finite_left and not finite_right:
            continue
        if abs(lv - rv) > spec.tolerance(lv, rv):
            return -1 if lv < rv else 1

    for spec in specs:
        lv = safe_objective_float(left.get(spec.path))
        rv = safe_objective_float(right.get(spec.path))
        finite_left = np.isfinite(lv)
        finite_right = np.isfinite(rv)
        if finite_left and not finite_right:
            return -1
        if finite_right and not finite_left:
            return 1
        if not finite_left and not finite_right:
            continue
        if lv < rv:
            return -1
        if lv > rv:
            return 1
    return 0


def rank_by_objectives(
    items: Sequence[T],
    value_getter: Callable[[T], Mapping[str, Any]],
    specs: Sequence[ObjectiveSpec],
    tie_breaker: Callable[[T], Any] | None = None,
) -> list[T]:
    def _compare(left: T, right: T) -> int:
        cmp = compare_objective_values(value_getter(left), value_getter(right), specs)
        if cmp != 0 or tie_breaker is None:
            return cmp
        left_key = tie_breaker(left)
        right_key = tie_breaker(right)
        if left_key < right_key:
            return -1
        if left_key > right_key:
            return 1
        return 0

    return sorted(items, key=cmp_to_key(_compare))


def select_best_by_objectives(
    items: Sequence[T],
    value_getter: Callable[[T], Mapping[str, Any]],
    specs: Sequence[ObjectiveSpec],
    tie_breaker: Callable[[T], Any] | None = None,
) -> tuple[T, dict[str, Any]]:
    if not items:
        raise ValueError("Cannot select from an empty candidate list.")

    eligible = list(items)
    filters: list[dict[str, Any]] = []
    for spec in specs:
        finite_values = [
            safe_objective_float(value_getter(item).get(spec.path))
            for item in eligible
            if np.isfinite(safe_objective_float(value_getter(item).get(spec.path)))
        ]
        if not finite_values:
            filters.append(
                {
                    "path": spec.path,
                    "best_value": None,
                    "tolerance": None,
                    "eligible_count": len(eligible),
                }
            )
            continue
        best_value = min(finite_values)
        tolerance = spec.anchor_tolerance(best_value)
        next_eligible = [
            item
            for item in eligible
            if safe_objective_float(value_getter(item).get(spec.path)) <= best_value + tolerance
        ]
        eligible = next_eligible or eligible
        filters.append(
            {
                "path": spec.path,
                "best_value": best_value,
                "tolerance": tolerance,
                "eligible_count": len(eligible),
            }
        )

    ranked = rank_by_objectives(eligible, value_getter, specs, tie_breaker=tie_breaker)
    selected = ranked[0]
    return selected, {
        "selected_by": "objective_order",
        "filters": filters,
        "eligible_count": len(eligible),
        "selected_objective_values": dict(value_getter(selected)),
    }


def passes_anchor_guard(
    candidate_values: Mapping[str, Any],
    anchor_values: Mapping[str, Any],
    specs: Sequence[ObjectiveSpec],
) -> tuple[bool, dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for spec in specs:
        if not spec.anchor_guard:
            continue
        anchor = safe_objective_float(anchor_values.get(spec.path))
        candidate = safe_objective_float(candidate_values.get(spec.path))
        if not np.isfinite(anchor):
            checks.append(
                {
                    "path": spec.path,
                    "anchor": None,
                    "candidate": candidate if np.isfinite(candidate) else None,
                    "passed": True,
                }
            )
            continue
        tolerance = spec.anchor_tolerance(anchor)
        passed = np.isfinite(candidate) and candidate <= anchor + tolerance
        checks.append(
            {
                "path": spec.path,
                "anchor": anchor,
                "candidate": candidate if np.isfinite(candidate) else None,
                "tolerance": tolerance,
                "passed": bool(passed),
            }
        )
        if not passed:
            return False, {"checks": checks}
    return True, {"checks": checks}


def update_anchor_values(
    anchor_values: Mapping[str, Any],
    accepted_values: Mapping[str, Any],
    specs: Sequence[ObjectiveSpec],
) -> ObjectiveValues:
    out: ObjectiveValues = {
        spec.path: safe_objective_float(anchor_values.get(spec.path))
        for spec in specs
    }
    for spec in specs:
        current = out[spec.path]
        accepted = safe_objective_float(accepted_values.get(spec.path))
        if not np.isfinite(current) or (np.isfinite(accepted) and accepted < current):
            out[spec.path] = accepted
    return out


__all__ = [
    "LEGACY_HYPER_CONFIG_KEYS",
    "ObjectiveSpec",
    "ObjectiveValues",
    "aggregate_objective_values",
    "compare_objective_values",
    "extract_subject_objective_values",
    "first_objective_value",
    "objective_order_payload",
    "passes_anchor_guard",
    "rank_by_objectives",
    "resolve_objective_order",
    "safe_objective_float",
    "select_best_by_objectives",
    "update_anchor_values",
    "validate_no_legacy_hyper_config",
]
