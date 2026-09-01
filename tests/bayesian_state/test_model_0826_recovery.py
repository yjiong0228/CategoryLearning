from __future__ import annotations

from pathlib import Path

import pytest

from src.Bayesian_state.optimization.parameter_space import (
    load_model_parameter_space,
    load_parameter_space,
)


ROOT = Path(__file__).resolve().parents[2]
PARAMETER_SPACE_0818 = (
    ROOT / "configs/specific_models/model_0818_cond1_parameter_space.yaml"
)
PARAMETER_SPACE_0826 = (
    ROOT / "configs/specific_models/model_0826_cond1_parameter_space.yaml"
)


def test_parameter_loader_accepts_0818_and_0826_without_cross_version_aliasing() -> None:
    old = load_model_parameter_space(
        PARAMETER_SPACE_0818,
        expected_model_id="model_0818",
    )
    new = load_model_parameter_space(
        PARAMETER_SPACE_0826,
        expected_model_id="model_0826",
    )

    assert old["provenance"]["model_id"] == "model_0818"
    assert new["provenance"]["model_id"] == "model_0826"
    assert new["provenance"]["event_history_excludes_latest_error"] is True
    assert load_parameter_space(PARAMETER_SPACE_0818) == old
    with pytest.raises(ValueError, match="model_0818"):
        load_parameter_space(PARAMETER_SPACE_0826)
    with pytest.raises(ValueError, match="model_0826"):
        load_model_parameter_space(
            PARAMETER_SPACE_0818,
            expected_model_id="model_0826",
        )


def test_model_0826_fine_supports_are_explicit_and_preserve_exact_spikes() -> None:
    config = load_model_parameter_space(
        PARAMETER_SPACE_0826,
        expected_model_id="model_0826",
    )
    parameters = config["subject_parameters"]

    assert parameters["workspace_execution"]["fine_candidates"] == parameters[
        "workspace_execution"
    ]["candidates"]
    assert parameters["gamma"]["fine_values"] == [
        0.0, 0.125, 0.25, 0.375, 0.50, 0.60, 0.70,
        0.75, 0.80, 0.85, 0.90, 0.935, 0.97,
    ]
    assert parameters["delta_E"]["zero_value"] == 0.0
    assert 0.0 not in parameters["delta_E"]["fine_positive_values"]
    assert parameters["c_A"]["zero_value"] == 0.0
    assert 0.0 not in parameters["c_A"]["fine_positive_values"]
    assert parameters["c_G"]["zero_value"] == 0.0
    assert 0.0 not in parameters["c_G"]["fine_positive_values"]
