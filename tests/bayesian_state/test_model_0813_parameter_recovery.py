from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from scripts.run_model_0813_pf_parameter_recovery import (
    _wilson_interval,
    build_profile_grid,
    summarize_primary,
    validate_profile_balance,
)


def _factors() -> dict:
    return {
        "memory_gamma": {
            "label": "Memory gamma",
            "path": "engine.modules.memory_mod.kwargs.gamma",
            "levels": [0.60, 0.80, 0.95],
        },
        "exploration_failure_threshold": {
            "label": "Failure threshold",
            "path": (
                "engine.modules.hypo_transitions_mod.kwargs."
                "continuous_controller.exploration.failure_threshold"
            ),
            "levels": [0.40, 0.55, 0.70],
        },
        "execution_switch_scale": {
            "label": "Switch scale",
            "path": (
                "engine.modules.hypo_transitions_mod.kwargs."
                "continuous_controller.execution.switch_scale"
            ),
            "levels": [0.10, 0.20, 0.40],
        },
    }


def test_l9_profiles_are_pairwise_balanced_and_contain_baseline() -> None:
    profiles = build_profile_grid(_factors())
    validate_profile_balance(profiles)
    assert len(profiles) == 9
    baseline = profiles[4]
    assert baseline["profile_id"] == "P05"
    assert baseline["is_baseline"] is True
    assert baseline["values"] == {
        "memory_gamma": 0.80,
        "exploration_failure_threshold": 0.55,
        "execution_switch_scale": 0.20,
    }


def test_recovery_config_uses_authoritative_0813_pf_model() -> None:
    with open(
        "configs/specific_models/model_0813_pf_parameter_recovery.yaml",
        "r",
        encoding="utf-8",
    ) as stream:
        config = yaml.safe_load(stream)
    assert config["base_simulation_config"].endswith(
        "model0813_v2f_cond1_all_subjects.yaml"
    )
    profiles = build_profile_grid(config["factors"])
    validate_profile_balance(profiles)


def test_summary_recovers_known_best_profiles_and_parameters() -> None:
    factors = _factors()
    profiles = build_profile_grid(factors)
    rows = []
    for dataset_index, true_profile in enumerate(profiles):
        for fit_index, fit_profile in enumerate(profiles):
            row = {
                "dataset_id": f"dataset_{dataset_index}",
                "subject_id": 103,
                "replicate": 0,
                "trial_count": 128,
                "true_profile_id": true_profile["profile_id"],
                "fit_profile_id": fit_profile["profile_id"],
                "log_likelihood": -float(abs(fit_index - dataset_index)),
                "nll": float(abs(fit_index - dataset_index)) / 128.0,
                "generated_accuracy": 0.75,
                "mean_pre_choice_ess": 24.0,
                "resampling_fraction": 0.20,
            }
            for factor_name in factors:
                row[f"true_{factor_name}"] = true_profile["values"][factor_name]
                row[f"fit_{factor_name}"] = fit_profile["values"][factor_name]
            rows.append(row)
    recovered, parameters, confusion, by_subject, summary = summarize_primary(
        pd.DataFrame(rows),
        factors=factors,
        near_best_delta_nll=0.1,
        confidence_level=0.95,
    )
    assert recovered["exact_profile_recovered"].all()
    assert np.allclose(parameters["exact_recovery_rate"], 1.0)
    assert summary["exact_profile_recovery_rate"] == 1.0
    assert summary["true_profile_within_delta_nll_rate"] == 1.0
    assert confusion["dataset_n"].sum() == 9
    assert by_subject.loc[0, "exact_profile_recovery_rate"] == 1.0


def test_wilson_interval_is_bounded_and_not_degenerate() -> None:
    low, high = _wilson_interval(6, 10)
    assert 0.0 < low < 0.6 < high < 1.0
