from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from src.Bayesian_state.optimization.parameter_space import (
    MODEL_0818_SPIKE_PARAMETERS,
    load_parameter_space,
    reactive_error_probability,
    spike_and_positive_values,
    validate_model_0818_parameter_space,
)


ROOT = Path(__file__).resolve().parents[2]
PARAMETER_SPACE = (
    ROOT / "configs/exp123/specific_models/model_0818_cond1_parameter_space.yaml"
)
MANUSCRIPT = ROOT / "src/Bayesian_state/docs/model_architecture/model_0818.tex"


def test_parameter_space_matches_frozen_manuscript_and_cond1_scope() -> None:
    config = load_parameter_space(PARAMETER_SPACE)

    digest = sha256(MANUSCRIPT.read_bytes()).hexdigest()
    assert config["provenance"]["manuscript_sha256"] == digest
    assert config["scope"]["subjects"] == list(range(101, 133))
    assert config["scope"]["perception_parameters"] == (
        "subject_specific_fixed_external"
    )
    assert config["scope"]["observed_data_fit_authorized"] is False


@pytest.mark.parametrize("parameter", MODEL_0818_SPIKE_PARAMETERS)
def test_special_mechanisms_have_exact_zero_and_separate_positive_support(
    parameter: str,
) -> None:
    config = load_parameter_space(PARAMETER_SPACE)
    specification = config["subject_parameters"][parameter]
    values = spike_and_positive_values(config, parameter)

    assert specification["kind"] == "spike_and_positive_grid"
    assert values[0] == 0.0
    assert all(value > 0.0 for value in values[1:])
    assert 0.0 not in specification["positive_values"]


def test_delta_e_parameterization_has_exact_boundary_and_ordering() -> None:
    config = load_parameter_space(PARAMETER_SPACE)
    event_correct_values = config["subject_parameters"]["E_C"]["coarse_values"]
    delta_values = spike_and_positive_values(config, "delta_E")

    for event_correct in event_correct_values:
        assert reactive_error_probability(event_correct, 0.0) == event_correct
        previous = event_correct
        for delta in delta_values[1:]:
            event_error = reactive_error_probability(event_correct, delta)
            assert event_correct < event_error < 1.0
            assert event_error > previous
            previous = event_error


def test_workspace_execution_candidates_respect_searchable_slot_constraint() -> None:
    config = load_parameter_space(PARAMETER_SPACE)
    candidates = config["subject_parameters"]["workspace_execution"]["candidates"]

    assert {(candidate["M"], candidate["chi"]) for candidate in candidates} == {
        (1, 0),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (4, 0),
        (4, 1),
        (5, 0),
        (5, 1),
    }
    assert all(candidate["M"] >= 2 for candidate in candidates if candidate["chi"])


@pytest.mark.parametrize("parameter", MODEL_0818_SPIKE_PARAMETERS)
def test_validator_rejects_zero_inside_positive_support(parameter: str) -> None:
    config = load_parameter_space(PARAMETER_SPACE)
    invalid = deepcopy(config)
    invalid["subject_parameters"][parameter]["positive_values"].insert(0, 0.0)

    with pytest.raises(ValueError, match="strictly positive"):
        validate_model_0818_parameter_space(invalid)


def test_validator_rejects_missing_exact_zero_boundary() -> None:
    config = load_parameter_space(PARAMETER_SPACE)
    invalid = deepcopy(config)
    invalid["subject_parameters"]["c_G"]["zero_value"] = 1e-9

    with pytest.raises(ValueError, match="exact numeric value 0.0"):
        validate_model_0818_parameter_space(invalid)


def test_validator_rejects_out_of_range_positive_c_g() -> None:
    config = load_parameter_space(PARAMETER_SPACE)
    invalid = deepcopy(config)
    invalid["subject_parameters"]["c_G"]["positive_values"].append(1.01)

    with pytest.raises(ValueError, match="above its theoretical domain"):
        validate_model_0818_parameter_space(invalid)
