from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.Bayesian_state.optimization.model_0826 import (
    build_model_0826_cell_engine,
    build_model_0826_hyper_config,
    extract_model_0826_parameters,
)
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
MODEL_0826_ENGINE = ROOT / "configs/model_struct/pmh_model_cond1_0826.yaml"


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


@pytest.mark.parametrize(
    "cell,has_m,has_h",
    [
        ("P", False, False),
        ("PM", True, False),
        ("PH", False, True),
        ("PMH", True, True),
    ],
)
def test_cell_builder_changes_only_m_and_h(
    cell: str,
    has_m: bool,
    has_h: bool,
) -> None:
    base_engine = yaml.safe_load(MODEL_0826_ENGINE.read_text(encoding="utf-8"))

    engine = build_model_0826_cell_engine(base_engine, cell)

    assert ("hypo_transitions_mod" in engine["modules"]) is has_h
    assert ("hypo_transitions_mod" in engine["agenda"]) is has_h
    is_dual = engine["modules"]["memory_mod"]["class"].endswith(
        "DualMemoryModule"
    )
    assert is_dual is has_m
    assert engine["modules"]["beta_mod"]["kwargs"]["update_scope"] == (
        "active_hypotheses"
    )
    assert engine["modules"]["beta_mod"]["kwargs"]["increase_rate"] > 0.0
    assert engine["modules"]["beta_mod"]["kwargs"]["decrease_rate"] > 0.0
    assert engine["choice_readout"]["kwargs"] == {
        "method": "expectation",
        "power": 1.0,
        "strategy_confidence_gain": 0.0,
    }
    assert engine["output_noise"]["kwargs"]["base_lapse"] == 0.0


@pytest.mark.parametrize(
    "cell,coordinate_count",
    [("P", 3), ("PM", 4), ("PH", 7), ("PMH", 8)],
)
def test_hyper_config_contains_only_cell_free_coordinates(
    tmp_path: Path,
    cell: str,
    coordinate_count: int,
) -> None:
    parameter_space = load_model_parameter_space(
        PARAMETER_SPACE_0826,
        expected_model_id="model_0826",
    )
    analysis = {
        "analysis_id": "model0826_recovery_test",
        "subjects": [101],
        "hyper_base_seed": 9,
        "max_trials": None,
        "evaluation_protocol": {
            "mode": "sequential_holdout",
            "train_fraction": 0.70,
            "optimization_partition": "train",
            "simulation_partition": "evaluation",
        },
        "shortlist_size": 2,
        "cd": {"parallel_budget": 4},
    }
    budgets = {
        "coarse": {"particle_count": 16, "filter_seed_count": 2},
        "fine": {"particle_count": 32, "filter_seed_count": 2},
        "final_rescore": {
            "particle_count": 64,
            "filter_seed_count": 4,
            "seed_family": "independent_test_v1",
        },
    }

    config = build_model_0826_hyper_config(
        analysis,
        parameter_space,
        cell,
        tmp_path / "base.yaml",
        tmp_path / cell,
        budgets,
    )

    assert config["search_schema_version"] == 2
    assert len(config["stages"]["coarse"]["hyperparam_space"]) == coordinate_count
    assert len(config["stages"]["fine"]["hyperparam_space"]) == coordinate_count
    assert config["recovery"]["free_parameters"] == parameter_space[
        "architecture_cells"
    ][cell]["free_parameters"]
    assert config["final_rescore"]["simulation_overrides"][
        "repeat_aggregation"
    ] == "mean_probability"
    assert config["cd"]["resume_mode"] == "explicit"
    for initial_point in config["cd"]["initial_points"]:
        assert set(initial_point) == set(config["hyperparam_space"])


def test_named_parameter_extraction_round_trips_pmh_anchor(tmp_path: Path) -> None:
    parameter_space = load_model_parameter_space(
        PARAMETER_SPACE_0826,
        expected_model_id="model_0826",
    )
    config = build_model_0826_hyper_config(
        {
            "analysis_id": "model0826_recovery_test",
            "subjects": [101],
            "hyper_base_seed": 9,
            "max_trials": None,
        },
        parameter_space,
        "PMH",
        tmp_path / "base.yaml",
        tmp_path / "PMH",
        {
            "coarse": {"particle_count": 16, "filter_seed_count": 2},
            "fine": {"particle_count": 32, "filter_seed_count": 2},
            "final_rescore": {
                "particle_count": 64,
                "filter_seed_count": 4,
                "seed_family": "independent_test_v1",
            },
        },
    )

    named = extract_model_0826_parameters(config["cd"]["initial_points"][0])

    assert named == {
        "M": 3,
        "chi": 0,
        "gamma": 0.8,
        "E_C": 0.25,
        "delta_E": pytest.approx(0.4795730802618863),
        "E_E": pytest.approx(0.35),
        "g_0": 0.2,
        "c_A": 0.0,
        "c_G": 0.0,
        "beta_0": 5.0,
        "eta_plus": 0.04,
        "eta_minus": 0.15,
    }
