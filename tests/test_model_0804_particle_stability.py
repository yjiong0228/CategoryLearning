from __future__ import annotations

import numpy as np

from scripts.run_model_0804_particle_stability_audit import (
    _analyse_level,
    _kendall_tau,
    _requires_escalation,
    _select_source_rows,
)


THRESHOLDS = {
    "maximum_plausible_candidate_paired_difference_se_nll": 0.35,
    "maximum_true_candidate_absolute_nll_se": 0.50,
    "minimum_seed_modal_exact_winner_fraction": 0.75,
    "minimum_seed_modal_rho_class_fraction": 1.00,
    "plausible_candidate_delta_nll": 2.00,
}


def _candidate(identifier: str, rho: float, g: float) -> dict[str, float | str]:
    return {"id": identifier, "rho": rho, "m": 0.15, "g": g, "lapse": 0.02}


def test_selection_checks_frozen_original_seed_stratum() -> None:
    report = {
        "datasets": [
            {
                "dataset_id": "d1",
                "subject_id": 101,
                "scenario_id": "center",
                "confirmed_ranking": [
                    {"id": "a", "seed_nll": [1.0, 2.0]},
                    {"id": "b", "seed_nll": [2.0, 1.0]},
                ],
            }
        ]
    }
    slots = [
        {
            "dataset_id": "d1",
            "subject_id": 101,
            "scenario_id": "center",
            "expected_original_seed_agreement": False,
        }
    ]
    selected = _select_source_rows(report, slots)
    assert selected[0]["original_seed_winners"] == ["a", "b"]
    assert not selected[0]["original_seed_winner_agreement"]


def test_level_analysis_uses_likelihood_ensemble_and_paired_noise() -> None:
    candidates = [_candidate("a", 0.0, 0.0), _candidate("b", 0.02, 0.7)]
    raw = {
        "tier_id": "n8192",
        "particle_count": 8192,
        "seeds": [1, 2, 3, 4],
        "nll": [[10.0, 10.1, 9.9, 10.0], [10.8, 10.9, 10.7, 10.8]],
        "new_compute_runtime_seconds": 1.0,
    }
    result = _analyse_level(raw, candidates, "a", THRESHOLDS)
    assert result["ensemble_winner"]["id"] == "a"
    assert result["seed_modal_exact_winner_fraction"] == 1.0
    assert result["seed_modal_rho_class"] == "zero"
    assert result["numerically_resolved"]
    assert result["true_candidate_delta_nll"] == 0.0


def test_escalation_is_triggered_by_changed_winner() -> None:
    config = {
        "diagnostic_thresholds": THRESHOLDS,
        "escalation": {
            "escalate_if_n8192_numerically_unresolved": True,
            "escalate_if_exact_ensemble_winner_changed_from_n2048": True,
            "escalate_if_seed_modal_exact_winner_below_threshold": True,
            "escalate_if_seed_rho_class_below_threshold": True,
        },
    }
    baseline = {"ensemble_winner": {"id": "a"}}
    high = {
        "ensemble_winner": {"id": "b"},
        "numerically_resolved": True,
        "seed_modal_exact_winner_fraction": 1.0,
        "seed_modal_rho_class_fraction": 1.0,
    }
    escalate, reasons = _requires_escalation(baseline, high, config)
    assert escalate
    assert reasons == ["ensemble_winner_changed_from_n2048"]


def test_kendall_tau_reports_reversal_and_identity() -> None:
    assert _kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0
    assert _kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0
