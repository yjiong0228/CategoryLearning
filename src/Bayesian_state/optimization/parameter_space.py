"""Validation helpers for frozen-model parameter-space specifications."""

from __future__ import annotations

from copy import deepcopy
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


MODEL_0818_SPIKE_PARAMETERS = ("delta_E", "c_A", "c_G")
MODEL_0818_SUBJECT_PARAMETERS = {
    "workspace_execution",
    "gamma",
    "E_C",
    "delta_E",
    "g_0",
    "c_A",
    "c_G",
    "beta_0",
    "eta_plus",
    "eta_minus",
}


def reactive_error_probability(event_after_correct: float, delta_e: float) -> float:
    """Return ``E_E = sigmoid(logit(E_C) + delta_E)`` stably."""

    event_correct = float(event_after_correct)
    delta = float(delta_e)
    if not np.isfinite(event_correct) or not 0.0 < event_correct < 1.0:
        raise ValueError("event_after_correct must lie strictly between 0 and 1.")
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError("delta_e must be finite and non-negative.")
    # Preserve the nested-model boundary bit-for-bit instead of relying on a
    # numerically approximate sigmoid(logit(p)) round trip.
    if delta == 0.0:
        return event_correct
    shifted_logit = float(
        np.log(event_correct) - np.log1p(-event_correct) + delta
    )
    if shifted_logit >= 0.0:
        return float(1.0 / (1.0 + np.exp(-shifted_logit)))
    exponential = float(np.exp(shifted_logit))
    return float(exponential / (1.0 + exponential))


def spike_and_positive_values(
    config: Mapping[str, Any], parameter: str
) -> tuple[float, ...]:
    """Return the exact spike followed by the declared positive support."""

    subject_parameters = _mapping(config.get("subject_parameters"), "subject_parameters")
    specification = _mapping(
        subject_parameters.get(parameter), f"subject_parameters.{parameter}"
    )
    zero = float(specification.get("zero_value", np.nan))
    positives = _finite_values(
        specification.get("positive_values"),
        f"subject_parameters.{parameter}.positive_values",
    )
    return (zero, *positives)


def load_parameter_space(path: str | Path) -> dict[str, Any]:
    """Load and validate a Model 0818 parameter-space YAML file."""

    source = Path(path)
    try:
        parsed = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read parameter-space config: {source}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid parameter-space YAML: {source}") from exc
    config = deepcopy(_mapping(parsed, "parameter-space root"))
    validate_model_0818_parameter_space(config)
    return config


def validate_model_0818_parameter_space(config: Mapping[str, Any]) -> None:
    """Validate scientific boundaries and recovery-design invariants."""

    provenance = _mapping(config.get("provenance"), "provenance")
    if provenance.get("model_id") != "model_0818":
        raise ValueError("provenance.model_id must be 'model_0818'.")
    if provenance.get("support_status") != "provisional_until_recovery_passes":
        raise ValueError(
            "The Model 0818 candidate support must remain provisional until recovery passes."
        )

    scope = _mapping(config.get("scope"), "scope")
    if int(scope.get("condition", -1)) != 1:
        raise ValueError("Model 0818 first-stage parameter recovery is restricted to cond1.")
    subjects = [int(value) for value in _sequence(scope.get("subjects"), "scope.subjects")]
    if subjects != list(range(101, 133)):
        raise ValueError("scope.subjects must be the ordered cond1 subjects 101--132.")
    if scope.get("perception_parameters") != "subject_specific_fixed_external":
        raise ValueError("Perception parameters must remain subject-specific and externally fixed.")
    if scope.get("observed_data_fit_authorized") is not False:
        raise ValueError("The pre-recovery parameter space cannot authorize observed-data fitting.")

    parameters = _mapping(config.get("subject_parameters"), "subject_parameters")
    missing = MODEL_0818_SUBJECT_PARAMETERS - set(parameters)
    extra = set(parameters) - MODEL_0818_SUBJECT_PARAMETERS
    if missing or extra:
        raise ValueError(
            "subject_parameters must match Model 0818 exactly; "
            f"missing={sorted(missing)}, extra={sorted(extra)}."
        )

    _validate_workspace_execution(parameters["workspace_execution"])
    for name in ("gamma", "E_C", "g_0", "beta_0", "eta_plus", "eta_minus"):
        _validate_bounded_grid(name, parameters[name])
    for name in MODEL_0818_SPIKE_PARAMETERS:
        _validate_spike_parameter(name, parameters[name])

    event_correct_values = _finite_values(
        _mapping(parameters["E_C"], "subject_parameters.E_C").get("coarse_values"),
        "subject_parameters.E_C.coarse_values",
    )
    delta_values = spike_and_positive_values(config, "delta_E")
    for event_correct in event_correct_values:
        for delta in delta_values:
            event_error = reactive_error_probability(event_correct, delta)
            if not event_correct <= event_error < 1.0:
                raise ValueError("delta_E transformation violated E_C <= E_E < 1.")
            if delta == 0.0 and event_error != event_correct:
                raise ValueError("delta_E=0 must give the exact E_E=E_C boundary.")
            if delta > 0.0 and not event_error > event_correct:
                raise ValueError("positive delta_E must give E_E > E_C.")

    _validate_architecture_cells(config)
    _validate_fixed_parameters(config)


def _validate_workspace_execution(raw: Any) -> None:
    specification = _mapping(raw, "subject_parameters.workspace_execution")
    candidates = _sequence(
        specification.get("candidates"),
        "subject_parameters.workspace_execution.candidates",
    )
    pairs: list[tuple[int, int]] = []
    for index, raw_candidate in enumerate(candidates):
        candidate = _mapping(
            raw_candidate,
            f"subject_parameters.workspace_execution.candidates[{index}]",
        )
        if set(candidate) != {"M", "chi"}:
            raise ValueError("Each workspace candidate must contain exactly M and chi.")
        capacity_raw = candidate["M"]
        if isinstance(capacity_raw, bool) or not isinstance(capacity_raw, Integral):
            raise ValueError("Workspace capacity M must be an integer in [1, 14].")
        capacity = int(capacity_raw)
        chi_raw = candidate["chi"]
        if isinstance(chi_raw, bool) or not isinstance(chi_raw, Integral):
            raise ValueError("Persistent-execution indicator chi must be 0 or 1.")
        chi = int(chi_raw)
        if not 1 <= capacity <= 14:
            raise ValueError("Workspace capacity M must be an integer in [1, 14].")
        if chi not in {0, 1}:
            raise ValueError("Persistent-execution indicator chi must be 0 or 1.")
        if chi == 1 and capacity < 2:
            raise ValueError("chi=1 requires M>=2 so at least one slot remains searchable.")
        pairs.append((capacity, chi))
    if len(pairs) != len(set(pairs)):
        raise ValueError("Workspace/execution candidates must be unique.")
    starts = {
        (int(candidate["M"]), int(candidate["chi"]))
        for candidate in _sequence(
            specification.get("start_candidates"),
            "subject_parameters.workspace_execution.start_candidates",
        )
    }
    if not starts or not starts.issubset(set(pairs)):
        raise ValueError("All workspace start candidates must occur in candidates.")
    if starts != {(3, 0), (3, 1)}:
        raise ValueError("Model 0818 recovery must start from both chi states at M=3.")


def _validate_bounded_grid(name: str, raw: Any) -> None:
    specification = _mapping(raw, f"subject_parameters.{name}")
    domain = _mapping(
        specification.get("theoretical_domain"),
        f"subject_parameters.{name}.theoretical_domain",
    )
    values = _finite_values(
        specification.get("coarse_values"),
        f"subject_parameters.{name}.coarse_values",
    )
    if len(values) != len(set(values)):
        raise ValueError(f"subject_parameters.{name}.coarse_values must be unique.")
    if values != sorted(values):
        raise ValueError(f"subject_parameters.{name}.coarse_values must be increasing.")
    for value in values:
        _validate_domain_value(value, domain, name)
    anchor = float(specification.get("anchor", np.nan))
    if not np.isfinite(anchor) or anchor not in values:
        raise ValueError(f"subject_parameters.{name}.anchor must occur in coarse_values.")
    if name in {"eta_plus", "eta_minus"} and any(value <= 0.0 for value in values):
        raise ValueError(f"{name}=0 is reserved for the separate static-beta ablation.")


def _validate_spike_parameter(name: str, raw: Any) -> None:
    specification = _mapping(raw, f"subject_parameters.{name}")
    if specification.get("kind") != "spike_and_positive_grid":
        raise ValueError(f"{name} must use kind='spike_and_positive_grid'.")
    zero = float(specification.get("zero_value", np.nan))
    if zero != 0.0:
        raise ValueError(f"{name}.zero_value must be the exact numeric value 0.0.")
    positives = _finite_values(
        specification.get("positive_values"),
        f"subject_parameters.{name}.positive_values",
    )
    if any(value <= 0.0 for value in positives):
        raise ValueError(f"{name}.positive_values must contain strictly positive values only.")
    if len(positives) != len(set(positives)):
        raise ValueError(f"{name}.positive_values must be unique.")
    if positives != sorted(positives):
        raise ValueError(f"{name}.positive_values must be increasing.")
    domain = _mapping(
        specification.get("theoretical_domain"),
        f"subject_parameters.{name}.theoretical_domain",
    )
    _validate_domain_value(zero, domain, name)
    for value in positives:
        _validate_domain_value(value, domain, name)
    if name == "delta_E":
        positive_anchor = float(specification.get("positive_anchor", np.nan))
        if not np.isfinite(positive_anchor) or positive_anchor not in positives:
            raise ValueError("delta_E.positive_anchor must occur in positive_values.")


def _validate_architecture_cells(config: Mapping[str, Any]) -> None:
    cells = _mapping(config.get("architecture_cells"), "architecture_cells")
    expected = {
        "P": {"beta_0", "eta_plus", "eta_minus"},
        "PM": {"gamma", "beta_0", "eta_plus", "eta_minus"},
        "PH": {
            "workspace_execution", "E_C", "delta_E", "g_0", "c_A", "c_G",
            "beta_0", "eta_plus", "eta_minus",
        },
        "PMH": MODEL_0818_SUBJECT_PARAMETERS,
    }
    if set(cells) != set(expected):
        raise ValueError("architecture_cells must contain exactly P, PM, PH, and PMH.")
    for cell, parameters in expected.items():
        specification = _mapping(cells[cell], f"architecture_cells.{cell}")
        free_parameters = set(
            str(value)
            for value in _sequence(
                specification.get("free_parameters"),
                f"architecture_cells.{cell}.free_parameters",
            )
        )
        if free_parameters != parameters:
            raise ValueError(f"architecture_cells.{cell}.free_parameters is inconsistent.")


def _validate_fixed_parameters(config: Mapping[str, Any]) -> None:
    fixed = _mapping(config.get("shared_fixed_parameters"), "shared_fixed_parameters")
    expected = {
        "delta_F": 0.60,
        "tau_L": 0.10,
        "s_exec": 0.20,
        "beta_min": 0.10,
        "beta_max": 25.00,
        "alpha": 1.00,
        "w_0": 0.00,
    }
    if set(fixed) != set(expected):
        raise ValueError("shared_fixed_parameters does not match Model 0818.")
    for name, expected_value in expected.items():
        if float(fixed[name]) != expected_value:
            raise ValueError(f"shared_fixed_parameters.{name} must equal {expected_value}.")


def _validate_domain_value(value: float, domain: Mapping[str, Any], name: str) -> None:
    lower = domain.get("lower")
    upper = domain.get("upper")
    lower_closed = bool(domain.get("lower_closed", False))
    upper_closed = bool(domain.get("upper_closed", False))
    if lower is not None:
        lower_value = float(lower)
        if value < lower_value or (value == lower_value and not lower_closed):
            raise ValueError(f"{name} candidate {value} is below its theoretical domain.")
    if upper is not None:
        upper_value = float(upper)
        if value > upper_value or (value == upper_value and not upper_closed):
            raise ValueError(f"{name} candidate {value} is above its theoretical domain.")


def _finite_values(raw: Any, name: str) -> list[float]:
    values = [float(value) for value in _sequence(raw, name)]
    if not values or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain at least one finite value.")
    return values


def _mapping(raw: Any, name: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    return dict(raw)


def _sequence(raw: Any, name: str) -> Sequence[Any]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError(f"{name} must be a sequence.")
    if not raw:
        raise ValueError(f"{name} cannot be empty.")
    return raw


__all__ = [
    "MODEL_0818_SPIKE_PARAMETERS",
    "load_parameter_space",
    "reactive_error_probability",
    "spike_and_positive_values",
    "validate_model_0818_parameter_space",
]
