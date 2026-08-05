from __future__ import annotations

import numpy as np

from scripts.run_model_0804_marginalized_lapse_recovery import (
    _analyse_marginal_level,
    _component_grid,
    _components_for_marginals,
    _confirmation_marginal_candidates,
    _marginal_grid,
    _marginalize_component_nll,
    _weighted_mixture_nll,
)


CONFIG = {
    "architecture": {"fixed_g": 0.35},
    "candidate_grid": {
        "rho": [0.0, 0.02, 0.05],
        "m": [0.05, 0.15, 0.30],
    },
    "nuisance_lapse": {
        "levels": [0.005, 0.02, 0.05],
        "prior_weights": [1.0 / 3.0] * 3,
        "marginalization_scope": "whole_sequence",
    },
    "screening": {
        "top_k_marginal_stage1": 4,
        "always_include_reference_rho_m": True,
        "always_include_rho_zero_parent": True,
        "always_include_profile_winner_per_rho_m_level": True,
    },
}


def test_grids_fix_g_and_expand_each_rho_m_point_over_three_lapses() -> None:
    marginal = _marginal_grid(CONFIG)
    components = _component_grid(CONFIG)
    assert len(marginal) == 9
    assert len(components) == 27
    assert {row["g"] for row in marginal + components} == {0.35}
    for candidate in marginal:
        matched = [
            row
            for row in components
            if row["rho"] == candidate["rho"] and row["m"] == candidate["m"]
        ]
        assert {row["lapse"] for row in matched} == {0.005, 0.02, 0.05}


def test_weighted_mixture_operates_in_likelihood_not_nll_space() -> None:
    likelihoods = np.asarray([0.2, 0.4, 0.8])
    nll = -np.log(likelihoods)
    actual = _weighted_mixture_nll(nll, np.asarray([1.0 / 3.0] * 3))
    assert np.isclose(actual, -np.log(np.mean(likelihoods)))
    assert not np.isclose(actual, np.mean(nll))


def test_marginalization_keeps_filter_seeds_separate() -> None:
    marginal = _marginal_grid(CONFIG)[:2]
    components = _components_for_marginals(
        _component_grid(CONFIG), marginal, CONFIG["nuisance_lapse"]["levels"]
    )
    component_nll = -np.log(
        np.asarray(
            [
                [0.2, 0.3],
                [0.4, 0.5],
                [0.8, 0.7],
                [0.1, 0.2],
                [0.2, 0.3],
                [0.3, 0.4],
            ]
        )
    )
    actual = _marginalize_component_nll(
        component_nll,
        components,
        marginal,
        CONFIG["nuisance_lapse"]["levels"],
        np.asarray([1.0 / 3.0] * 3),
    )
    expected = -np.log(
        np.asarray(
            [
                [np.mean([0.2, 0.4, 0.8]), np.mean([0.3, 0.5, 0.7])],
                [np.mean([0.1, 0.2, 0.3]), np.mean([0.2, 0.3, 0.4])],
            ]
        )
    )
    assert actual.shape == (2, 2)
    assert np.allclose(actual, expected)


def test_confirmation_covers_reference_parent_and_every_rho_m_level() -> None:
    candidates = _marginal_grid(CONFIG)
    stage = np.linspace(10.0, 20.0, len(candidates))
    reference = next(
        row for row in candidates if row["rho"] == 0.05 and row["m"] == 0.30
    )
    confirmed = _confirmation_marginal_candidates(
        candidates, stage, reference, CONFIG
    )
    identifiers = {row["id"] for row in confirmed}
    assert reference["id"] in identifiers
    parent = next(
        row for row in candidates if row["rho"] == 0.0 and row["m"] == 0.30
    )
    assert parent["id"] in identifiers
    for parameter in ("rho", "m"):
        assert {row[parameter] for row in confirmed} == {
            row[parameter] for row in candidates
        }
    components = _components_for_marginals(
        _component_grid(CONFIG), confirmed, CONFIG["nuisance_lapse"]["levels"]
    )
    assert len(components) == 3 * len(confirmed)


def test_analysis_selects_rho_m_after_sequence_level_lapse_mixture() -> None:
    marginal = _marginal_grid(CONFIG)[:2]
    components = _components_for_marginals(
        _component_grid(CONFIG), marginal, CONFIG["nuisance_lapse"]["levels"]
    )
    # Candidate 0 has the better mixture on every seed even though its first
    # lapse component is deliberately worse than candidate 1's first component.
    likelihoods = np.asarray(
        [
            [0.10, 0.10, 0.10, 0.10],
            [0.90, 0.90, 0.90, 0.90],
            [0.90, 0.90, 0.90, 0.90],
            [0.50, 0.50, 0.50, 0.50],
            [0.50, 0.50, 0.50, 0.50],
            [0.50, 0.50, 0.50, 0.50],
        ]
    )
    result = _analyse_marginal_level(
        {
            "tier_id": "test",
            "particle_count": 1,
            "seeds": [1, 2, 3, 4],
            "nll": -np.log(likelihoods),
            "new_compute_runtime_seconds": 0.0,
        },
        components,
        marginal,
        marginal[0]["id"],
        CONFIG["nuisance_lapse"]["levels"],
        np.asarray([1.0 / 3.0] * 3),
        {
            "plausible_candidate_delta_nll": 2.0,
            "maximum_plausible_candidate_paired_difference_se_nll": 0.35,
            "maximum_reference_candidate_absolute_nll_se": 0.50,
        },
    )
    assert result["ensemble_winner"]["id"] == marginal[0]["id"]
    assert result["seed_winner_ids"] == [marginal[0]["id"]] * 4
    posterior = result["candidate_ranking"][0]["lapse_components"]
    assert np.isclose(sum(row["posterior_weight"] for row in posterior), 1.0)
