from __future__ import annotations

import numpy as np

from scripts.run_model_0804_fixed_g_recovery import (
    _confirmation_candidates,
    _project_to_fixed_g,
    _rename_reference_fields,
    _restricted_grid,
)


CONFIG = {
    "architecture": {"fixed_g": 0.35},
    "candidate_grid": {
        "rho": [0.0, 0.02, 0.05],
        "m": [0.05, 0.15, 0.30],
        "lapse": [0.005, 0.02, 0.05],
    },
    "screening": {
        "top_k_restricted_stage1": 5,
        "always_include_reference_projection": True,
        "always_include_rho_zero_parent": True,
        "always_include_profile_winner_per_level": True,
    },
}


def test_restricted_grid_has_only_fixed_g_and_27_points() -> None:
    candidates = _restricted_grid(CONFIG)
    assert len(candidates) == 27
    assert {candidate["g"] for candidate in candidates} == {0.35}
    assert len({candidate["id"] for candidate in candidates}) == 27


def test_projection_changes_only_g_and_builds_matching_id() -> None:
    target = {"rho": 0.02, "m": 0.15, "g": 0.7, "lapse": 0.02}
    projected = _project_to_fixed_g(target, 0.35)
    assert projected["rho"] == target["rho"]
    assert projected["m"] == target["m"]
    assert projected["lapse"] == target["lapse"]
    assert projected["g"] == 0.35
    assert "g0p350" in projected["id"]


def test_confirmation_covers_reference_parent_and_every_parameter_level() -> None:
    candidates = _restricted_grid(CONFIG)
    stage = np.linspace(10.0, 20.0, len(candidates))
    reference = _project_to_fixed_g(
        {"rho": 0.05, "m": 0.30, "g": 0.35, "lapse": 0.05}, 0.35
    )
    confirmed = _confirmation_candidates(candidates, stage, reference, CONFIG)
    identifiers = {row["id"] for row in confirmed}
    assert reference["id"] in identifiers
    parent = _project_to_fixed_g(
        {"rho": 0.0, "m": 0.30, "g": 0.35, "lapse": 0.05}, 0.35
    )
    assert parent["id"] in identifiers
    for parameter in ("rho", "m", "lapse"):
        assert {row[parameter] for row in confirmed} == {
            row[parameter] for row in candidates
        }


def test_reference_fields_are_not_mislabeled_as_true_under_misspecification() -> None:
    level = {
        "true_candidate_absolute_nll_se": 0.2,
        "true_candidate_delta_nll": 1.1,
        "true_candidate_within_2_nll": True,
        "other": 3,
    }
    renamed = _rename_reference_fields(level)
    assert "true_candidate_delta_nll" not in renamed
    assert renamed["reference_candidate_delta_nll"] == 1.1
    assert renamed["other"] == 3
