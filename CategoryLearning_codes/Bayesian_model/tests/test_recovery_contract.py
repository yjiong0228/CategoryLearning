"""Focused 0826 recovery-contract tests reused from the existing suite."""
from pathlib import Path
from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
import yaml
from CategoryLearning_codes.Bayesian_model.optimization.model_0826 import build_model_0826_cell_engine, build_model_0826_hyper_config, extract_model_0826_parameters
from CategoryLearning_codes.Bayesian_model.optimization.parameter_space import load_model_parameter_space
from CategoryLearning_codes.Bayesian_model.evaluation.model_recovery import mean_probability_nll, summarize_module_recovery, summarize_parameter_recovery, freeze_smallest_passing_budget
PACKAGE=Path(__file__).resolve().parents[1]
PARAMETER_SPACE_0826=PACKAGE/'configs/parameter_space_0826.yaml'
MODEL_0826_ENGINE=PACKAGE/'configs/model_0826.yaml'

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


def test_calibration_freezes_smallest_budget_passing_every_gate() -> None:
    summary = {
        "budget_decisions": [
            {"particle_count": 64, "filter_seed_count": 8, "passes_all_gates": True},
            {"particle_count": 128, "filter_seed_count": 16, "passes_all_gates": True},
        ]
    }

    assert freeze_smallest_passing_budget(summary) == {
        "particle_count": 64,
        "filter_seed_count": 8,
    }
    assert freeze_smallest_passing_budget(
        {
            "budget_decisions": [
                {
                    "particle_count": 128,
                    "filter_seed_count": 16,
                    "passes_all_gates": False,
                }
            ]
        }
    ) is None


def test_nll_is_computed_after_probability_averaging_on_requested_mask() -> None:
    probability_runs = np.asarray(
        [
            [[0.9, 0.1], [0.9, 0.1], [0.8, 0.2]],
            [[0.1, 0.9], [0.1, 0.9], [0.4, 0.6]],
        ],
        dtype=float,
    )
    choices = np.asarray([1, 2, 1])
    mask = np.asarray([False, True, True])

    total_nll = mean_probability_nll(probability_runs, choices, mask)

    assert total_nll == pytest.approx(-np.log(0.5) - np.log(0.6))


def test_module_summary_uses_total_nll_and_delta_two_near_best() -> None:
    winners = {"P": "P", "PM": "PM", "PH": "PH", "PMH": "PH"}
    rows = []
    for dataset_index, true_cell in enumerate(("P", "PM", "PH", "PMH")):
        winner = winners[true_cell]
        for candidate_cell in ("P", "PM", "PH", "PMH"):
            nll = 100.0 + float(candidate_cell != winner) * 5.0
            if candidate_cell == true_cell and true_cell == "PMH":
                nll = 101.5
            rows.append(
                {
                    "dataset_id": f"dataset_{dataset_index}",
                    "true_cell": true_cell,
                    "candidate_cell": candidate_cell,
                    "total_nll": nll,
                    "generated_accuracy": 0.7 + dataset_index * 0.02,
                }
            )

    summary = summarize_module_recovery(
        pd.DataFrame(rows),
        near_best_delta_nll=2.0,
    )

    assert summary["overall_exact_recovery"] == pytest.approx(0.75)
    assert summary["true_cell_near_best_coverage"] == pytest.approx(1.0)
    assert sum(row["count"] for row in summary["confusion_rows"]) == 4


def test_parameter_summary_separates_zero_positive_classification() -> None:
    rows = []
    for index in range(8):
        truth_zero = index < 4
        rows.append(
            {
                "dataset_id": f"parameter_{index}",
                "true_M": 2 + (index % 4),
                "estimated_M": 2 + (index % 4),
                "true_chi": index % 2,
                "estimated_chi": index % 2,
                "true_gamma": 0.1 * index,
                "estimated_gamma": 0.1 * index + 0.01,
                "true_E_C": 0.1 + 0.05 * index,
                "estimated_E_C": 0.1 + 0.05 * index,
                "true_delta_E": 0.0 if truth_zero else 0.25 * (index - 3),
                "estimated_delta_E": 0.0 if truth_zero else 0.25 * (index - 3),
                "true_g_0": 0.05 * index,
                "estimated_g_0": 0.05 * index,
                "true_c_A": 0.0 if truth_zero else float(index - 3),
                "estimated_c_A": 0.0 if truth_zero else float(index - 3),
                "true_c_G": 0.0 if truth_zero else 0.1 * (index - 3),
                "estimated_c_G": 0.0 if truth_zero else 0.1 * (index - 3),
                "true_beta_0": 1.0 + index,
                "estimated_beta_0": 1.0 + index,
                "true_eta_plus": 0.01 + 0.01 * index,
                "estimated_eta_plus": 0.01 + 0.01 * index,
                "true_eta_minus": 0.03 + 0.03 * index,
                "estimated_eta_minus": 0.03 + 0.03 * index,
                "true_within_near_best": True,
            }
        )
    parameter_space = load_model_parameter_space(
        PARAMETER_SPACE_0826,
        expected_model_id="model_0826",
    )

    summary = summarize_parameter_recovery(
        pd.DataFrame(rows),
        parameter_space=parameter_space,
    )

    by_parameter = {
        row["parameter"]: row for row in summary["parameter_rows"]
    }
    assert "zero_positive_balanced_accuracy" in by_parameter["c_A"]
    assert by_parameter["c_A"]["zero_positive_balanced_accuracy"] == 1.0
    assert summary["chi_exact_recovery"] == 1.0
    assert sum(row["count"] for row in summary["M_confusion_rows"]) == 8
    assert {
        (row["true_M"], row["estimated_M"])
        for row in summary["M_confusion_rows"]
    } == {(truth, estimate) for truth in range(1, 6) for estimate in range(1, 6)}

