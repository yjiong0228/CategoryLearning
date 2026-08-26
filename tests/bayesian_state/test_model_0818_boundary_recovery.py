from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.run_model_0818_boundary_recovery import (
    build_boundary_profiles,
    resolve_filter_seeds,
    summarize_boundary_recovery,
    summarize_high_budget_calibration,
    summarize_numerical_stability,
    validate_high_budget_seed_design,
)
from src.Bayesian_state.optimization.parameter_space import load_parameter_space
from src.Bayesian_state.simulation.parameters import (
    apply_fixed_hyperparams_to_engine_config,
)


ROOT = Path(__file__).resolve().parents[2]
PARAMETER_SPACE = (
    ROOT / "configs/specific_models/model_0818_cond1_parameter_space.yaml"
)
RECOVERY_CONFIG = (
    ROOT / "configs/specific_models/model_0818_boundary_recovery.yaml"
)
MODEL_CONFIG = ROOT / "configs/model_struct/pmh_model_cond1_0818.yaml"
MANUSCRIPT = ROOT / "manuscript/model_0818.tex"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _profiles() -> list[dict]:
    parameter_space = load_parameter_space(PARAMETER_SPACE)
    recovery = _yaml(RECOVERY_CONFIG)
    return build_boundary_profiles(parameter_space, recovery["positive_probes"])


def test_executable_model_matches_frozen_manuscript_and_0818_defaults() -> None:
    model = _yaml(MODEL_CONFIG)
    assert model["provenance"]["model_id"] == "model_0818"
    assert model["provenance"]["manuscript_sha256"] == sha256(
        MANUSCRIPT.read_bytes()
    ).hexdigest()
    controller = model["modules"]["hypo_transitions_mod"]["kwargs"]
    assert controller["prior_assignment"]["method"] == "similarity_transport"
    assert controller["persistent_execution"] == {
        "enabled": False,
        "switch_scale": 0.20,
    }
    assert controller["nested_feedback_accumulator_controller"][
        "accumulator_decay"
    ] == pytest.approx(0.60)
    beta = model["modules"]["beta_mod"]["kwargs"]
    assert beta["increase_rate"] == pytest.approx(0.04)
    assert "correct_additive" not in beta


def test_boundary_profiles_cover_all_zero_and_each_positive_mechanism() -> None:
    profiles = _profiles()
    values = {profile["profile_id"]: profile["values"] for profile in profiles}

    assert len(profiles) == 4
    assert values["B000_all_zero"] == {"delta_E": 0.0, "c_A": 0.0, "c_G": 0.0}
    assert values["B100_delta_E"]["delta_E"] > 0.0
    assert values["B100_delta_E"]["c_A"] == 0.0
    assert values["B100_delta_E"]["c_G"] == 0.0
    assert values["B010_c_A"]["c_A"] > 0.0
    assert values["B001_c_G"]["c_G"] > 0.0


def test_pilot_increases_schedule_replicate_and_pf_budget_over_smoke() -> None:
    recovery = _yaml(RECOVERY_CONFIG)
    smoke = recovery["smoke"]
    pilot = recovery["pilot"]

    assert len(pilot["template_subjects"]) > 1
    assert pilot["datasets_per_profile_per_subject"] > 1
    assert pilot["trials_per_dataset"] > smoke["trials_per_dataset"]
    assert pilot["particle_count"] > smoke["particle_count"]
    assert recovery["numerical_stability"]["enabled"] is True
    high_budget = [
        setting
        for setting in recovery["numerical_stability"]["settings"]
        if setting.get("high_budget_calibration", False)
    ]
    assert {
        (
            setting["particle_count"],
            setting["filter_seed_count"],
            setting["seed_ensemble"],
        )
        for setting in high_budget
    } == {(32, 4, "A"), (64, 4, "A"), (32, 4, "B"), (64, 4, "B")}


def test_profile_application_preserves_exact_delta_e_zero_boundary() -> None:
    model = _yaml(MODEL_CONFIG)
    profiles = {profile["profile_id"]: profile for profile in _profiles()}

    zero = apply_fixed_hyperparams_to_engine_config(
        model, profiles["B000_all_zero"]["hyperparams"]
    )
    positive = apply_fixed_hyperparams_to_engine_config(
        model, profiles["B100_delta_E"]["hyperparams"]
    )
    zero_controller = zero["modules"]["hypo_transitions_mod"]["kwargs"][
        "nested_feedback_accumulator_controller"
    ]
    positive_controller = positive["modules"]["hypo_transitions_mod"]["kwargs"][
        "nested_feedback_accumulator_controller"
    ]
    assert zero_controller["event_after_error"] == zero_controller["event_after_correct"]
    assert positive_controller["event_after_error"] > positive_controller[
        "event_after_correct"
    ]


def test_positive_probe_must_come_from_declared_positive_support() -> None:
    parameter_space = load_parameter_space(PARAMETER_SPACE)
    recovery = _yaml(RECOVERY_CONFIG)
    probes = dict(recovery["positive_probes"])
    probes["c_G"] = 0.333

    with pytest.raises(ValueError, match="declared positive candidate"):
        build_boundary_profiles(parameter_space, probes)


def test_stability_seed_family_pairs_particle_counts_and_nests_seed_counts() -> None:
    common = {
        "dataset_id": "synthetic_001",
        "base_seed": 20260818,
        "seed_family": "paired_budget_test",
    }
    r8_b2 = resolve_filter_seeds(
        **common, particle_count=8, filter_seed_count=2
    )
    r32_b2 = resolve_filter_seeds(
        **common, particle_count=32, filter_seed_count=2
    )
    r16_b4 = resolve_filter_seeds(
        **common, particle_count=16, filter_seed_count=4
    )

    assert r8_b2 == r32_b2
    assert r16_b4[:2] == r8_b2
    assert len(set(r16_b4)) == 4


def test_boundary_summary_recovers_mock_known_winners() -> None:
    profiles = _profiles()
    rows = []
    for true_profile in profiles:
        for fit_index, fit_profile in enumerate(profiles):
            exact = fit_profile["profile_id"] == true_profile["profile_id"]
            rows.append(
                {
                    "dataset_id": f"data_{true_profile['profile_id']}",
                    "subject_id": 103,
                    "trial_count": 32,
                    "true_profile_id": true_profile["profile_id"],
                    "fit_profile_id": fit_profile["profile_id"],
                    "total_nll": float(fit_index + (0 if exact else 10)),
                    "generated_accuracy": 0.75,
                }
            )
        true_rows = [
            row
            for row in rows
            if row["dataset_id"] == f"data_{true_profile['profile_id']}"
            and row["fit_profile_id"] == true_profile["profile_id"]
        ]
        true_rows[0]["total_nll"] = 0.0

    recovered, boundaries, summary = summarize_boundary_recovery(
        pd.DataFrame(rows), profiles, near_best_delta_nll=2.0
    )
    assert recovered["exact_profile_recovered"].all()
    assert boundaries["boundary_recovery_rate"].eq(1.0).all()
    assert summary["all_zero_positive_boundaries_exercised"] is True
    assert summary["exact_profile_recovery_rate"] == 1.0


def test_numerical_stability_summary_detects_consistent_rankings() -> None:
    profiles = _profiles()
    settings = ((8, 2), (16, 2), (32, 2), (16, 4))
    rows = []
    for true_profile in profiles[:2]:
        true_index = next(
            index
            for index, profile in enumerate(profiles)
            if profile["profile_id"] == true_profile["profile_id"]
        )
        for particle_count, seed_count in settings:
            for fit_index, fit_profile in enumerate(profiles):
                rows.append(
                    {
                        "dataset_id": f"data_{true_profile['profile_id']}",
                        "true_profile_id": true_profile["profile_id"],
                        "fit_profile_id": fit_profile["profile_id"],
                        "particle_count": particle_count,
                        "filter_seed_count": seed_count,
                        "total_nll": float(abs(fit_index - true_index)),
                    }
                )

    winners, correlations, summary = summarize_numerical_stability(
        pd.DataFrame(rows)
    )
    assert len(winners) == 8
    assert correlations["candidate_nll_spearman"].eq(1.0).all()
    assert summary["setting_n"] == 4
    assert summary["mean_within_dataset_modal_winner_agreement"] == 1.0
    assert summary["true_profile_recovery_rate_across_settings"] == 1.0


def test_high_budget_calibration_freezes_r32_when_all_gates_pass() -> None:
    profiles = _profiles()
    rows = []
    for dataset_id in ("data_zero", "data_positive"):
        for particle_count in (32, 64):
            for ensemble in ("A", "B"):
                for fit_index, profile in enumerate(profiles):
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "fit_profile_id": profile["profile_id"],
                            "particle_count": particle_count,
                            "filter_seed_count": 4,
                            "seed_ensemble": ensemble,
                            "total_nll": float(fit_index),
                        }
                    )
    gates = _yaml(RECOVERY_CONFIG)["numerical_stability"]["calibration_gates"]

    comparisons, summary = summarize_high_budget_calibration(
        pd.DataFrame(rows), gates
    )

    assert len(comparisons) == 8
    assert comparisons["candidate_nll_rank_spearman"].eq(1.0).all()
    assert comparisons["same_winner"].all()
    assert summary["budget_status"] == "provisional_minimum_R32_B4"
    assert summary["provisional_minimum_budget"] == {
        "particle_count": 32,
        "filter_seed_count": 4,
    }
    assert summary["formal_recovery_authorized"] is True
    assert summary["observed_data_fit_authorized"] is False
    assert summary["score_separation_diagnostics"] == {
        "setting_dataset_n": 8,
        "median_winner_to_runner_up_delta_nll": 1.0,
        "maximum_winner_to_runner_up_delta_nll": 1.0,
        "median_cross_setting_maximum_absolute_delta_nll_difference": 0.0,
        "maximum_cross_setting_absolute_delta_nll_difference": 0.0,
    }


def test_high_budget_seed_design_pairs_r_and_separates_ensembles() -> None:
    rows = []
    ensemble_seeds = {"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]}
    for dataset_id in ("data_zero", "data_positive"):
        for particle_count in (32, 64):
            for ensemble in ("A", "B"):
                for fit_profile_id in ("zero", "positive"):
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "fit_profile_id": fit_profile_id,
                            "particle_count": particle_count,
                            "filter_seed_count": 4,
                            "seed_ensemble": ensemble,
                            "filter_seeds": ensemble_seeds[ensemble],
                        }
                    )

    audit = validate_high_budget_seed_design(pd.DataFrame(rows))

    assert audit["dataset_n"] == 2
    assert audit["candidates_share_seeds_within_setting"] is True
    assert audit["paired_R32_R64_within_ensemble"] is True
    assert audit["A_B_seed_sets_disjoint"] is True
