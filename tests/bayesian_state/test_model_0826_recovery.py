from __future__ import annotations

from pathlib import Path
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.Bayesian_state.evaluation.model_recovery import (
    build_calibration_bank,
    freeze_smallest_passing_budget,
    fit_recovery_dataset,
    generate_synthetic_dataset,
    load_recovery_design,
    resolve_calibration_filter_seeds,
    resolve_recovery_stage_budgets,
    mean_probability_nll,
    plot_module_recovery,
    plot_parameter_recovery,
    schedule_fingerprint,
    score_frozen_candidate,
    score_pf_bank,
    score_pf_bank_parallel,
    summarize_module_recovery,
    summarize_parameter_recovery,
    summarize_pf_calibration,
    summarize_search_budget_retention,
    synthetic_dataset_frame,
)
from src.Bayesian_state.optimization.model_0826 import (
    build_model_0826_cell_engine,
    build_model_0826_hyper_config,
    extract_model_0826_parameters,
)
from src.Bayesian_state.optimization.parameter_space import (
    load_model_parameter_space,
    load_parameter_space,
)
from scripts.run_model_0826_recovery import (
    PHASES,
    build_calibration_specs,
    build_parser,
    load_subject_schedules,
    prepare_output,
    require_frozen_budget,
    resolve_final_score_seeds,
    resolve_subject_order,
)


ROOT = Path(__file__).resolve().parents[2]
PARAMETER_SPACE_0818 = (
    ROOT / "configs/specific_models/model_0818_cond1_parameter_space.yaml"
)
PARAMETER_SPACE_0826 = (
    ROOT / "configs/specific_models/model_0826_cond1_parameter_space.yaml"
)
MODEL_0826_ENGINE = ROOT / "configs/model_struct/pmh_model_cond1_0826.yaml"
RECOVERY_CONFIG = (
    ROOT / "configs/specific_models/model_0826_recovery_v1.yaml"
)
RECOVERY_CONFIG_V2 = (
    ROOT / "configs/specific_models/model_0826_recovery_v2.yaml"
)


def test_priority_all_orders_requested_subject_before_remaining_subjects() -> None:
    assert resolve_subject_order((101, 111, 118), 101) == (101, 111, 118)
    assert resolve_subject_order((101, 111, 118), 118) == (118, 101, 111)
    with pytest.raises(ValueError, match="priority subject"):
        resolve_subject_order((101, 111, 118), 999)


def test_parser_accepts_subject_first_continuous_recovery() -> None:
    args = build_parser().parse_args(
        ["--phase", "priority-all", "--priority-subject", "101"]
    )

    assert args.phase == "priority-all"
    assert args.priority_subject == 101


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


def test_recovery_design_has_exact_pre_registered_counts_and_assignments() -> None:
    design = load_recovery_design(RECOVERY_CONFIG)

    assert design.subject_trial_counts == {101: 320, 111: 320, 118: 256}
    assert len(design.module_datasets) == 36
    assert len(design.parameter_datasets) == 40
    assert Counter(row.truth["chi"] for row in design.parameter_datasets) == {
        0: 20,
        1: 20,
    }
    assert Counter(row.subject_id for row in design.parameter_datasets) == {
        101: 14,
        111: 13,
        118: 13,
    }
    assert {
        row.subject_id: (row.train_trial_count, row.evaluation_trial_count)
        for row in design.module_datasets
    } == {
        101: (224, 96),
        111: (224, 96),
        118: (179, 77),
    }
    assert len({row.generation_seed for row in design.all_datasets}) == 76


def test_schedule_fingerprint_and_synthetic_frame_ignore_observed_choice() -> None:
    schedule = pd.DataFrame(
        {
            "iSub": [101, 101, 101],
            "condition": [1, 1, 1],
            "iSession": [1, 1, 1],
            "iBlock": [1, 1, 1],
            "iTrial": [1, 2, 3],
            "feature1": [1.0, 2.0, 3.0],
            "feature2": [2.0, 3.0, 4.0],
            "feature3": [3.0, 4.0, 5.0],
            "feature4": [4.0, 5.0, 6.0],
            "category": [1, 2, 1],
            "choice": [1, 1, 2],
            "feedback": [1.0, 0.0, 0.0],
        }
    )
    altered_observed_behavior = schedule.copy()
    altered_observed_behavior["choice"] = [2, 2, 1]
    altered_observed_behavior["feedback"] = [0.0, 1.0, 1.0]

    assert schedule_fingerprint(schedule) == schedule_fingerprint(
        altered_observed_behavior
    )
    generated = synthetic_dataset_frame(
        altered_observed_behavior,
        choices=np.asarray([2, 1, 1]),
        feedback=np.asarray([0.0, 0.0, 1.0]),
    )
    assert generated["choice"].tolist() == [2, 1, 1]
    assert generated["feedback"].tolist() == [0.0, 0.0, 1.0]
    assert generated.attrs["observed_choices_used"] is False


def test_synthetic_generation_uses_schedule_not_observed_choice_and_is_resumable(
    tmp_path: Path,
) -> None:
    design = load_recovery_design(RECOVERY_CONFIG)
    specification = design.module_datasets[0]
    trial_count = specification.trial_count
    schedule = pd.DataFrame(
        {
            "iSub": np.full(trial_count, specification.subject_id),
            "condition": np.ones(trial_count, dtype=int),
            "iSession": np.ones(trial_count, dtype=int),
            "iBlock": np.repeat(np.arange(1, 6), 64)[:trial_count],
            "iTrial": np.arange(1, trial_count + 1),
            "feature1": np.linspace(0.0, 1.0, trial_count),
            "feature2": np.linspace(1.0, 2.0, trial_count),
            "feature3": np.linspace(2.0, 3.0, trial_count),
            "feature4": np.linspace(3.0, 4.0, trial_count),
            "category": 1 + (np.arange(trial_count) % 2),
            "choice": np.ones(trial_count, dtype=int),
            "feedback": np.zeros(trial_count, dtype=float),
        }
    )
    captured = []

    def fake_generator(**kwargs):
        captured.append(kwargs)
        choices = 1 + (np.arange(trial_count) % 2)
        feedback = (choices == kwargs["categories"]).astype(float)
        probabilities = np.eye(2, dtype=float)[choices - 1]
        return SimpleNamespace(
            trajectory=SimpleNamespace(
                choices=choices,
                feedback=feedback,
                observed_probabilities=probabilities,
            )
        )

    base_engine = yaml.safe_load(MODEL_0826_ENGINE.read_text(encoding="utf-8"))
    manifest = generate_synthetic_dataset(
        specification,
        schedule_frame=schedule,
        base_engine_config=base_engine,
        output_dir=tmp_path / "synthetic",
        generator=fake_generator,
    )

    assert manifest["trial_count"] == 320
    assert manifest["observed_choices_used"] is False
    assert captured[0]["trajectory_seed"] == specification.generation_seed
    assert "hypo_transitions_mod" not in captured[0]["engine_config"]["modules"]
    assert Path(manifest["csv_path"]).is_file()
    assert Path(manifest["npz_path"]).is_file()

    altered = schedule.copy()
    altered["choice"] = 2
    altered["feedback"] = 1.0
    resumed = generate_synthetic_dataset(
        specification,
        schedule_frame=altered,
        base_engine_config=base_engine,
        output_dir=tmp_path / "synthetic",
        resume=True,
        generator=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("completed synthetic dataset was regenerated")
        ),
    )
    assert resumed["fingerprint"] == manifest["fingerprint"]
    with pytest.raises(FileExistsError):
        generate_synthetic_dataset(
            specification,
            schedule_frame=schedule,
            base_engine_config=base_engine,
            output_dir=tmp_path / "synthetic",
            generator=fake_generator,
        )


def test_calibration_bank_is_eight_fixed_candidates() -> None:
    anchor = {
        "M": 3,
        "gamma": 0.8,
        "E_C": 0.25,
        "delta_E": 0.8,
        "g_0": 0.2,
        "c_A": 2.0,
        "c_G": 0.5,
        "beta_0": 5.0,
        "eta_plus": 0.04,
        "eta_minus": 0.15,
    }

    bank = build_calibration_bank(anchor)

    assert len(bank) == 8
    assert {
        (row["truth"]["chi"], row["variant"]) for row in bank
    } == {
        (chi, variant)
        for chi in (0, 1)
        for variant in ("anchor", "gamma_050", "gains_zero", "beta_slow")
    }
    assert len({row["candidate_id"] for row in bank}) == 8


def test_calibration_seed_families_are_nested_within_and_disjoint_between_ensembles() -> None:
    a4 = resolve_calibration_filter_seeds(
        dataset_id="calibration_subject_101_chi_0",
        base_seed=9,
        ensemble="A",
        count=4,
    )
    a8 = resolve_calibration_filter_seeds(
        dataset_id="calibration_subject_101_chi_0",
        base_seed=9,
        ensemble="A",
        count=8,
    )
    b8 = resolve_calibration_filter_seeds(
        dataset_id="calibration_subject_101_chi_0",
        base_seed=9,
        ensemble="B",
        count=8,
    )

    assert a8[:4] == a4
    assert set(a8).isdisjoint(b8)


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


def test_optimized_recovery_uses_multifidelity_search_budgets() -> None:
    design = load_recovery_design(RECOVERY_CONFIG_V2)

    budgets = resolve_recovery_stage_budgets(
        design.config["search"],
        {"particle_count": 128, "filter_seed_count": 16},
    )

    assert design.analysis_id == "model_0826_recovery_v2"
    assert design.output_root.name == "recovery_v2"
    assert budgets == {
        "coarse": {"particle_count": 16, "filter_seed_count": 4},
        "fine": {"particle_count": 32, "filter_seed_count": 4},
        "final_rescore": {"particle_count": 128, "filter_seed_count": 16},
    }


def test_legacy_recovery_defaults_every_stage_to_frozen_budget() -> None:
    design = load_recovery_design(RECOVERY_CONFIG)

    budgets = resolve_recovery_stage_budgets(
        design.config["search"],
        {"particle_count": 64, "filter_seed_count": 8},
    )

    assert budgets == {
        stage: {"particle_count": 64, "filter_seed_count": 8}
        for stage in ("coarse", "fine", "final_rescore")
    }


def test_search_budget_retention_uses_high_budget_winner_rank() -> None:
    rows = []
    settings = (
        (16, 4, "A"),
        (32, 4, "A"),
        (128, 16, "A"),
    )
    for dataset_index in range(6):
        for particle_count, seed_count, ensemble in settings:
            for candidate_index in range(8):
                nll = float(candidate_index)
                if particle_count == 128:
                    nll = 0.0 if candidate_index == 1 else (
                        0.1 if candidate_index == 0 else float(candidate_index)
                    )
                rows.append(
                    {
                        "dataset_id": f"dataset_{dataset_index}",
                        "candidate_id": f"candidate_{candidate_index}",
                        "particle_count": particle_count,
                        "filter_seed_count": seed_count,
                        "ensemble": ensemble,
                        "total_nll": nll,
                        "mean_probability": np.full((4, 2), 0.5),
                        "trial_probability_mcse": np.zeros(4),
                    }
                )

    summary = summarize_search_budget_retention(
        rows,
        {
            "reference": {
                "particle_count": 128,
                "filter_seed_count": 16,
                "ensemble": "A",
            },
            "stages": {
                "coarse": {
                    "particle_count": 16,
                    "filter_seed_count": 4,
                    "ensemble": "A",
                    "winner_top_k": 4,
                    "minimum_dataset_count": 6,
                },
                "fine": {
                    "particle_count": 32,
                    "filter_seed_count": 4,
                    "ensemble": "A",
                    "winner_top_k": 2,
                    "minimum_dataset_count": 6,
                },
            },
        },
    )

    assert summary["status"] == "passed"
    assert summary["stages"]["coarse"]["retained_dataset_count"] == 6
    assert summary["stages"]["fine"]["retained_dataset_count"] == 6
    assert summary["stages"]["fine"]["right_winner_ranks"] == [2] * 6


def test_pf_bank_scores_nll_after_seed_probability_averaging() -> None:
    anchor = {
        "M": 3,
        "gamma": 0.8,
        "E_C": 0.25,
        "delta_E": 0.8,
        "g_0": 0.2,
        "c_A": 2.0,
        "c_G": 0.5,
        "beta_0": 5.0,
        "eta_plus": 0.04,
        "eta_minus": 0.15,
    }
    probabilities = {
        11: np.asarray([[0.9, 0.1], [0.9, 0.1]]),
        12: np.asarray([[0.1, 0.9], [0.1, 0.9]]),
    }

    rows = score_pf_bank(
        dataset_id="calibration_subject_101_chi_0",
        subject_id=101,
        stimulus=np.ones((2, 4), dtype=float),
        choices=np.asarray([1, 2]),
        feedback=np.asarray([1.0, 1.0]),
        base_engine_config=yaml.safe_load(
            MODEL_0826_ENGINE.read_text(encoding="utf-8")
        ),
        candidates=build_calibration_bank(anchor)[:1],
        particle_count=16,
        filter_seeds=[11, 12],
        ensemble="A",
        pf_runner=lambda **kwargs: SimpleNamespace(
            marginal_probabilities=probabilities[kwargs["filter_seed"]]
        ),
    )

    assert rows[0]["total_nll"] == pytest.approx(-2.0 * np.log(0.5))
    assert np.allclose(rows[0]["mean_probability"], 0.5)
    assert rows[0]["probability_aggregation"] == "mean_probability_then_nll"


def test_parallel_pf_bank_matches_serial_candidate_seed_aggregation() -> None:
    stimulus = np.ones((5, 4), dtype=float)
    choices = np.asarray([1, 2, 1, 2, 1], dtype=int)
    feedback = np.ones(5, dtype=float)
    anchor = {
        "M": 3,
        "gamma": 0.8,
        "E_C": 0.25,
        "delta_E": 0.8,
        "g_0": 0.2,
        "c_A": 2.0,
        "c_G": 0.5,
        "beta_0": 5.0,
        "eta_plus": 0.04,
        "eta_minus": 0.15,
    }

    def fake_pf(**kwargs):
        probability_two = 0.2 + 0.1 * (int(kwargs["filter_seed"]) % 3)
        probabilities = np.column_stack(
            [
                np.full(5, 1.0 - probability_two),
                np.full(5, probability_two),
            ]
        )
        return SimpleNamespace(marginal_probabilities=probabilities)

    common = {
        "dataset_id": "parallel_check",
        "subject_id": 101,
        "stimulus": stimulus,
        "choices": choices,
        "feedback": feedback,
        "base_engine_config": yaml.safe_load(
            MODEL_0826_ENGINE.read_text(encoding="utf-8")
        ),
        "candidates": build_calibration_bank(anchor)[:2],
        "particle_count": 8,
        "filter_seeds": [101, 102, 103],
        "ensemble": "A",
        "pf_runner": fake_pf,
    }
    serial = score_pf_bank(**common)
    parallel = score_pf_bank_parallel(**common, n_jobs=2)

    assert [row["candidate_id"] for row in parallel] == [
        row["candidate_id"] for row in serial
    ]
    for serial_row, parallel_row in zip(serial, parallel):
        assert parallel_row["total_nll"] == pytest.approx(
            serial_row["total_nll"]
        )
        assert np.allclose(
            parallel_row["mean_probability"],
            serial_row["mean_probability"],
        )
        assert np.allclose(
            parallel_row["probability_runs"],
            serial_row["probability_runs"],
        )


def test_pf_calibration_summary_applies_all_rank_winner_rmse_and_mcse_gates() -> None:
    score_rows = []
    settings = (
        (16, 4, "A"),
        (32, 4, "A"),
        (64, 4, "A"),
        (64, 8, "A"),
        (64, 8, "B"),
    )
    for dataset_index in range(6):
        for particle_count, seed_count, ensemble in settings:
            for candidate_index in range(8):
                probability = np.full((4, 2), 0.5, dtype=float)
                score_rows.append(
                    {
                        "dataset_id": f"dataset_{dataset_index}",
                        "candidate_id": f"candidate_{candidate_index}",
                        "particle_count": particle_count,
                        "filter_seed_count": seed_count,
                        "ensemble": ensemble,
                        "total_nll": float(candidate_index),
                        "mean_probability": probability,
                        "trial_probability_mcse": np.full(4, 0.005),
                    }
                )
    gates = {
        "dataset_count": 6,
        "median_adjacent_rank_spearman_min": 0.90,
        "minimum_adjacent_rank_spearman_min": 0.70,
        "adjacent_winner_agreement_min_count": 5,
        "independent_winner_agreement_min_count": 5,
        "median_probability_rmse_max": 0.015,
        "trial_probability_mcse_q95_max": 0.020,
    }

    summary = summarize_pf_calibration(score_rows, gates)

    assert summary["status"] == "passed"
    assert summary["budget_decisions"][0]["passes_all_gates"] is True
    assert freeze_smallest_passing_budget(summary) == {
        "particle_count": 64,
        "filter_seed_count": 8,
    }


def test_pf_calibration_can_gate_on_high_budget_winner_top_k_retention() -> None:
    score_rows = []
    settings = (
        (16, 4, "A"),
        (32, 4, "A"),
        (64, 4, "A"),
        (64, 8, "A"),
        (64, 8, "B"),
        (128, 16, "A"),
        (128, 16, "B"),
    )
    for dataset_index in range(6):
        for particle_count, seed_count, ensemble in settings:
            for candidate_index in range(8):
                nll = float(candidate_index)
                if particle_count == 128:
                    nll = 0.0 if candidate_index == 1 else (
                        0.1 if candidate_index == 0 else float(candidate_index)
                    )
                score_rows.append(
                    {
                        "dataset_id": f"dataset_{dataset_index}",
                        "candidate_id": f"candidate_{candidate_index}",
                        "particle_count": particle_count,
                        "filter_seed_count": seed_count,
                        "ensemble": ensemble,
                        "total_nll": nll,
                        "mean_probability": np.full((4, 2), 0.5),
                        "trial_probability_mcse": np.full(4, 0.005),
                    }
                )

    summary = summarize_pf_calibration(
        score_rows,
        {
            "dataset_count": 6,
            "median_adjacent_rank_spearman_min": 0.90,
            "minimum_adjacent_rank_spearman_min": 0.70,
            "adjacent_winner_top_k": 2,
            "adjacent_winner_top_k_min_count": 6,
            "independent_winner_agreement_min_count": 5,
            "median_probability_rmse_max": 0.015,
            "trial_probability_mcse_q95_max": 0.020,
        },
    )

    high_budget = summary["budget_decisions"][1]
    assert high_budget["adjacent_high_budget_winner_top_k"] == 2
    assert high_budget["adjacent_high_budget_winner_top_k_counts"] == [6]
    assert high_budget["passes_all_gates"] is True


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


def test_module_fit_configures_prefix_only_parameter_selection(
    tmp_path: Path,
) -> None:
    design = load_recovery_design(RECOVERY_CONFIG)
    specification = design.module_datasets[0]
    synthetic_csv = tmp_path / "synthetic.csv"
    pd.DataFrame(
        {
            "iSub": np.full(specification.trial_count, specification.subject_id),
            "condition": np.ones(specification.trial_count, dtype=int),
            "iSession": np.ones(specification.trial_count, dtype=int),
            "iBlock": np.ones(specification.trial_count, dtype=int),
            "iTrial": np.arange(1, specification.trial_count + 1),
            "feature1": np.zeros(specification.trial_count),
            "feature2": np.zeros(specification.trial_count),
            "feature3": np.zeros(specification.trial_count),
            "feature4": np.zeros(specification.trial_count),
            "category": np.ones(specification.trial_count, dtype=int),
            "choice": np.ones(specification.trial_count, dtype=int),
            "feedback": np.ones(specification.trial_count),
        }
    ).to_csv(synthetic_csv, index=False)
    captured = {}

    class FakeOptimizer:
        def __init__(self, config, config_path):
            captured["config"] = config
            captured["config_path"] = config_path

        def run(self, subjects, stage, resume):
            captured["run"] = {
                "subjects": subjects,
                "stage": stage,
                "resume": resume,
            }
            return {"best": {"selected": {"best_hyperparams": {}}}}

    result = fit_recovery_dataset(
        design,
        specification,
        candidate_cell="P",
        synthetic_csv=synthetic_csv,
        frozen_budget={"particle_count": 64, "filter_seed_count": 8},
        output_dir=tmp_path / "fit",
        optimizer_factory=FakeOptimizer,
    )

    resolved_simulation = yaml.safe_load(
        Path(result["simulation_config_path"]).read_text(encoding="utf-8")
    )
    protocol = resolved_simulation["evaluation_protocol"]
    assert protocol["train_fraction"] == 0.70
    assert protocol["optimization_partition"] == "train"
    assert protocol["simulation_partition"] == "evaluation"
    assert resolved_simulation["max_trials"] is None
    assert captured["run"] == {
        "subjects": [specification.subject_id],
        "stage": "all",
        "resume": False,
    }
    assert captured["config"]["final_rescore"]["simulation_overrides"][
        "evaluation_protocol"
    ]["optimization_partition"] == "train"


def test_fit_resume_reuses_configs_but_starts_fresh_before_checkpoint(
    tmp_path: Path,
) -> None:
    design = load_recovery_design(RECOVERY_CONFIG)
    specification = design.module_datasets[0]
    synthetic_csv = tmp_path / "synthetic.csv"
    pd.DataFrame(
        {
            "choice": np.ones(specification.trial_count, dtype=int),
            "feedback": np.ones(specification.trial_count),
        }
    ).to_csv(synthetic_csv, index=False)
    resume_values = []

    class FakeOptimizer:
        def __init__(self, config, config_path):
            del config, config_path

        def run(self, subjects, stage, resume):
            del subjects, stage
            resume_values.append(bool(resume))
            return {"best": {"selected": {"best_hyperparams": {}}}}

    kwargs = {
        "candidate_cell": "P",
        "synthetic_csv": synthetic_csv,
        "frozen_budget": {"particle_count": 64, "filter_seed_count": 8},
        "output_dir": tmp_path / "fit",
        "optimizer_factory": FakeOptimizer,
    }
    fit_recovery_dataset(design, specification, **kwargs)
    fit_recovery_dataset(design, specification, resume=True, **kwargs)

    assert resume_values == [False, False]


def test_frozen_module_candidate_runs_full_history_but_scores_only_suffix() -> None:
    trial_count = 320
    observed_trial_counts = []

    def fake_pf(**kwargs):
        observed_trial_counts.append(len(kwargs["choices"]))
        return SimpleNamespace(
            marginal_probabilities=np.full((trial_count, 2), 0.5, dtype=float)
        )

    score = score_frozen_candidate(
        subject_id=101,
        stimulus=np.ones((trial_count, 4), dtype=float),
        choices=np.ones(trial_count, dtype=int),
        feedback=np.ones(trial_count, dtype=float),
        base_engine_config=yaml.safe_load(
            MODEL_0826_ENGINE.read_text(encoding="utf-8")
        ),
        candidate_cell="P",
        fixed_hyperparams={},
        particle_count=64,
        filter_seeds=[1011, 1012],
        evaluation_protocol={
            "mode": "sequential_holdout",
            "train_fraction": 0.70,
            "optimization_partition": "train",
            "simulation_partition": "evaluation",
        },
        pf_runner=fake_pf,
    )

    assert observed_trial_counts == [320, 320]
    assert score["score_context"]["train_trial_count"] == 224
    assert score["score_context"]["evaluation_trial_count"] == 96
    assert score["score_context"]["score_trial_count"] == 96
    assert score["total_nll"] == pytest.approx(96 * np.log(2.0))


def test_frozen_candidate_parallel_seed_scoring_matches_serial() -> None:
    trial_count = 12

    def fake_pf(**kwargs):
        probability_one = 0.25 + 0.1 * (int(kwargs["filter_seed"]) % 3)
        probabilities = np.column_stack(
            (
                np.full(trial_count, probability_one),
                np.full(trial_count, 1.0 - probability_one),
            )
        )
        return SimpleNamespace(marginal_probabilities=probabilities)

    common = {
        "subject_id": 101,
        "stimulus": np.ones((trial_count, 4), dtype=float),
        "choices": np.asarray([1, 2] * (trial_count // 2), dtype=int),
        "feedback": np.ones(trial_count, dtype=float),
        "base_engine_config": yaml.safe_load(
            MODEL_0826_ENGINE.read_text(encoding="utf-8")
        ),
        "candidate_cell": "P",
        "fixed_hyperparams": {},
        "particle_count": 8,
        "filter_seeds": [1011, 1012, 1013],
        "evaluation_protocol": {"mode": "all"},
        "pf_runner": fake_pf,
    }

    serial = score_frozen_candidate(**common, n_jobs=1)
    parallel = score_frozen_candidate(**common, n_jobs=3)

    assert parallel["parallel_n_jobs"] == 3
    assert parallel["total_nll"] == pytest.approx(serial["total_nll"])
    assert np.array_equal(
        parallel["mean_probability"], serial["mean_probability"]
    )


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


def test_recovery_plots_emit_png_and_csv_source_data(tmp_path: Path) -> None:
    module_summary = {
        "confusion_rows": [
            {"true_cell": truth, "predicted_cell": prediction, "count": int(truth == prediction)}
            for truth in ("P", "PM", "PH", "PMH")
            for prediction in ("P", "PM", "PH", "PMH")
        ],
        "cell_rows": [
            {
                "true_cell": cell,
                "dataset_n": 3,
                "exact_recovery": 0.75,
                "wilson_low": 0.30,
                "wilson_high": 0.95,
            }
            for cell in ("P", "PM", "PH", "PMH")
        ],
        "dataset_rows": [
            {
                "dataset_id": f"d{index}",
                "true_cell": cell,
                "predicted_cell": cell,
                "true_delta_nll": float(index),
                "generated_accuracy": 0.7,
            }
            for index, cell in enumerate(("P", "PM", "PH", "PMH"))
        ],
    }
    parameter_summary = {
        "M_confusion_rows": [
            {"true_M": truth, "estimated_M": estimate, "count": int(truth == estimate) * 2}
            for truth in range(1, 6)
            for estimate in range(1, 6)
        ],
        "chi_confusion_rows": [
            {"true_chi": truth, "estimated_chi": estimate, "count": int(truth == estimate) * 4}
            for truth in (0, 1)
            for estimate in (0, 1)
        ],
        "parameter_rows": [
            {
                "parameter": name,
                "normalized_mae": 0.1,
                "spearman": 0.8,
                "near_best_coverage": 0.9,
                "zero_positive_balanced_accuracy": 0.8 if name in {"delta_E", "c_A", "c_G"} else None,
                "supported": True,
            }
            for name in ("gamma", "E_C", "delta_E", "g_0", "c_A", "c_G", "beta_0", "eta_plus", "eta_minus")
        ],
        "dataset_rows": [
            {"dataset_id": "d1", "true_chi": 0, "estimated_chi": 0}
        ],
    }

    module_png = tmp_path / "module_recovery_overview.png"
    parameter_png = tmp_path / "parameter_recovery_overview.png"
    plot_module_recovery(module_summary, module_png)
    plot_parameter_recovery(parameter_summary, parameter_png)

    assert module_png.is_file()
    assert parameter_png.is_file()
    assert list(tmp_path.glob("*.csv"))
    assert (tmp_path / "parameter_recovery_M_source.csv").is_file()
    assert not list(tmp_path.glob("*.pdf"))
    assert not list(tmp_path.glob("*.svg"))
    assert not list(tmp_path.glob("*.tiff"))


def test_recovery_cli_has_all_pre_registered_phases() -> None:
    parser = build_parser()
    phase_action = next(
        action for action in parser._actions if action.dest == "phase"
    )

    assert tuple(phase_action.choices) == PHASES
    assert PHASES == (
        "smoke",
        "generate",
        "calibrate",
        "module-fit",
        "parameter-fit",
        "summarize",
        "priority-all",
        "all",
    )


def test_existing_output_requires_resume_and_matching_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "recovery_v1"
    created = prepare_output(
        output,
        resume=False,
        analysis_id="model_0826_recovery_v1",
        config_fingerprint="abc",
    )
    assert created["status"] == "initialized"

    with pytest.raises(FileExistsError):
        prepare_output(
            output,
            resume=False,
            analysis_id="model_0826_recovery_v1",
            config_fingerprint="abc",
        )
    resumed = prepare_output(
        output,
        resume=True,
        analysis_id="model_0826_recovery_v1",
        config_fingerprint="abc",
    )
    assert resumed["config_fingerprint"] == "abc"
    with pytest.raises(ValueError, match="fingerprint"):
        prepare_output(
            output,
            resume=True,
            analysis_id="model_0826_recovery_v1",
            config_fingerprint="changed",
        )


def test_formal_fit_requires_successfully_frozen_pf_budget(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="calibration"):
        require_frozen_budget(tmp_path / "missing.json")

    failed = tmp_path / "failed.json"
    failed.write_text('{"status": "failed"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not frozen"):
        require_frozen_budget(failed)


def test_cli_uses_complete_registered_subject_schedules() -> None:
    design = load_recovery_design(RECOVERY_CONFIG)
    schedules, _ = load_subject_schedules(design)

    assert {subject_id: len(frame) for subject_id, frame in schedules.items()} == {
        101: 320,
        111: 320,
        118: 256,
    }


def test_calibration_specs_are_six_independent_full_trajectories() -> None:
    design = load_recovery_design(RECOVERY_CONFIG)
    specifications = build_calibration_specs(design)

    assert len(specifications) == 6
    assert len({row.dataset_id for row in specifications}) == 6
    assert len({row.generation_seed for row in specifications}) == 6
    assert Counter(row.subject_id for row in specifications) == {
        101: 2,
        111: 2,
        118: 2,
    }
    assert {row.trial_count for row in specifications} == {256, 320}


def test_final_score_seeds_are_paired_across_candidates_but_role_disjoint() -> None:
    first = resolve_final_score_seeds(
        analysis_id="model_0826_recovery_v1",
        dataset_id="dataset_1",
        role="module_suffix",
        count=4,
    )
    repeated = resolve_final_score_seeds(
        analysis_id="model_0826_recovery_v1",
        dataset_id="dataset_1",
        role="module_suffix",
        count=4,
    )
    other_role = resolve_final_score_seeds(
        analysis_id="model_0826_recovery_v1",
        dataset_id="dataset_1",
        role="parameter_near_best",
        count=4,
    )

    assert first == repeated
    assert len(set(first)) == 4
    assert set(first).isdisjoint(other_role)
