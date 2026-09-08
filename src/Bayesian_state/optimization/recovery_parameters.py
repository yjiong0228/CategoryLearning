"""Translate recovery truth parameters into shared model settings."""
from __future__ import annotations
from typing import Any, Mapping
from src.Bayesian_state.optimization.model_0826 import (
    ACCUMULATOR_GAIN_PATH,
    BETA_PATH,
    CAPACITY_PATH,
    ETA_MINUS_PATH,
    ETA_PLUS_PATH,
    EVENT_CORRECT_PATH,
    EVENT_ERROR_PATH,
    EXECUTION_PATH,
    GAMMA_PATH,
    GLOBAL_GAIN_PATH,
    GLOBAL_SEARCH_PATH,
    INITIAL_EVENT_PATH,
)
from src.Bayesian_state.optimization.parameter_space import reactive_error_probability


def _declared_support(parameter_space: Mapping[str, Any], name: str) -> list[Any]:
    specification = dict(parameter_space["subject_parameters"][name])
    if name == "workspace_execution":
        return [dict(value) for value in specification["fine_candidates"]]
    if specification["kind"] == "spike_and_positive_grid":
        return [
            float(specification["zero_value"]),
            *map(float, specification["fine_positive_values"]),
        ]
    return [float(value) for value in specification["fine_values"]]


def model_0826_truth_hyperparams(
    truth: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert named recovery truth values to executable engine paths."""

    hyperparams: dict[str, Any] = {
        BETA_PATH: float(truth["beta_0"]),
        ETA_PLUS_PATH: float(truth["eta_plus"]),
        ETA_MINUS_PATH: float(truth["eta_minus"]),
    }
    if "gamma" in truth:
        hyperparams[GAMMA_PATH] = float(truth["gamma"])
    if "M" in truth:
        event_correct = float(truth["E_C"])
        hyperparams.update(
            {
                CAPACITY_PATH: int(truth["M"]),
                EXECUTION_PATH: bool(int(truth["chi"])),
                EVENT_CORRECT_PATH: event_correct,
                EVENT_ERROR_PATH: reactive_error_probability(
                    event_correct, float(truth["delta_E"])
                ),
                INITIAL_EVENT_PATH: event_correct,
                GLOBAL_SEARCH_PATH: float(truth["g_0"]),
                ACCUMULATOR_GAIN_PATH: float(truth["c_A"]),
                GLOBAL_GAIN_PATH: float(truth["c_G"]),
            }
        )
    return hyperparams
