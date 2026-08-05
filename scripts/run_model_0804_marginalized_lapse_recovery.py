#!/usr/bin/env python3
"""Recover FA2R rho and m after sequence-level lapse marginalization."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0803_cond1 import build_frozen_geometry, load_config  # noqa: E402
from scripts.run_model_0804_fixed_g_recovery import (  # noqa: E402
    _load_json,
    _load_yaml,
    _requires_escalation,
    _reuse_values,
    _sha256,
    _source_stage_nll,
)
from scripts.run_model_0804_particle_stability_audit import (  # noqa: E402
    _modal_fraction,
    _run_level_payload,
    _standard_error,
)
from scripts.run_model_0804_regeneration_recovery import (  # noqa: E402
    _atomic_json,
    _candidate_id,
    _ensemble_nll,
    _rate_summary,
)


DEFAULT_CONFIG = ROOT / "configs/model_0804_marginalized_lapse_recovery.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "results/zhuran/model_0804_cond1/marginalized_lapse_recovery_20260805_v1"
)
RECOVERED_PARAMETERS = ("rho", "m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--dataset-ids",
        type=str,
        default=None,
        help="comma-separated subset for smoke/resume diagnostics",
    )
    parser.add_argument(
        "--stop-after",
        choices=("n8192", "n32768"),
        default=None,
        help="stop after a particle tier; makes the report non-gating",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _value_token(value: float) -> str:
    return f"{float(value):.3f}".replace("-", "m").replace(".", "p")


def _marginal_candidate_id(rho: float, m: float, fixed_g: float) -> str:
    return (
        f"rho{_value_token(rho)}_m{_value_token(m)}_g{_value_token(fixed_g)}"
        "_lapseMarginal"
    )


def _marginal_grid(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    fixed_g = float(config["architecture"]["fixed_g"])
    candidates = [
        {
            "id": _marginal_candidate_id(float(rho), float(m), fixed_g),
            "rho": float(rho),
            "m": float(m),
            "g": fixed_g,
        }
        for rho in config["candidate_grid"]["rho"]
        for m in config["candidate_grid"]["m"]
    ]
    if len(candidates) != 9 or len({row["id"] for row in candidates}) != 9:
        raise ValueError("marginal grid must contain 9 unique rho-m points")
    return candidates


def _component_grid(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    fixed_g = float(config["architecture"]["fixed_g"])
    candidates = []
    for rho in config["candidate_grid"]["rho"]:
        for m in config["candidate_grid"]["m"]:
            for lapse in config["nuisance_lapse"]["levels"]:
                values = {
                    "rho": float(rho),
                    "m": float(m),
                    "g": fixed_g,
                    "lapse": float(lapse),
                }
                candidates.append({"id": _candidate_id(values), **values})
    if len(candidates) != 27 or len({row["id"] for row in candidates}) != 27:
        raise ValueError("component grid must contain 27 unique rho-m-lapse points")
    return candidates


def _matching_marginal_index(
    candidates: Sequence[Mapping[str, Any]], target: Mapping[str, Any]
) -> int:
    matches = [
        index
        for index, candidate in enumerate(candidates)
        if all(
            np.isclose(float(candidate[name]), float(target[name]), atol=1e-12)
            for name in RECOVERED_PARAMETERS
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"target is not a unique marginal point: {target}")
    return int(matches[0])


def _project_reference_rho_m(
    target: Mapping[str, Any], fixed_g: float
) -> dict[str, Any]:
    rho = float(target["rho"])
    m = float(target["m"])
    return {
        "id": _marginal_candidate_id(rho, m, fixed_g),
        "rho": rho,
        "m": m,
        "g": float(fixed_g),
    }


def _component_groups(
    component_candidates: Sequence[Mapping[str, Any]],
    marginal_candidates: Sequence[Mapping[str, Any]],
    lapse_levels: Sequence[float],
) -> list[list[int]]:
    groups: list[list[int]] = []
    for marginal in marginal_candidates:
        indices = []
        for lapse in lapse_levels:
            matches = [
                index
                for index, component in enumerate(component_candidates)
                if np.isclose(float(component["rho"]), float(marginal["rho"]), atol=1e-12)
                and np.isclose(float(component["m"]), float(marginal["m"]), atol=1e-12)
                and np.isclose(float(component["lapse"]), float(lapse), atol=1e-12)
            ]
            if len(matches) != 1:
                raise ValueError("each marginal point must have one component per lapse level")
            indices.append(int(matches[0]))
        groups.append(indices)
    return groups


def _validate_nuisance(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    levels = np.asarray(config["nuisance_lapse"]["levels"], dtype=float)
    weights = np.asarray(config["nuisance_lapse"]["prior_weights"], dtype=float)
    if levels.shape != (3,) or len(set(levels.tolist())) != 3:
        raise ValueError("exactly three unique lapse levels are required")
    if weights.shape != levels.shape or np.any(weights <= 0.0):
        raise ValueError("positive nuisance weights must match lapse levels")
    if not np.isclose(float(weights.sum()), 1.0, atol=1e-12):
        raise ValueError("nuisance prior weights must sum to one")
    if str(config["nuisance_lapse"]["marginalization_scope"]) != "whole_sequence":
        raise ValueError("only whole-sequence lapse marginalization is allowed")
    return levels, weights


def _weighted_mixture_nll(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return -log(sum_lapse weight_lapse * exp(-NLL_lapse)) stably."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.shape[0] != weights.size or np.any(~np.isfinite(values)):
        raise ValueError("finite component NLL values with lapse on axis 0 required")
    log_terms = np.log(weights).reshape((-1,) + (1,) * (values.ndim - 1)) - values
    maximum = np.max(log_terms, axis=0)
    return -(maximum + np.log(np.sum(np.exp(log_terms - maximum), axis=0)))


def _marginalize_component_nll(
    component_nll: np.ndarray,
    component_candidates: Sequence[Mapping[str, Any]],
    marginal_candidates: Sequence[Mapping[str, Any]],
    lapse_levels: Sequence[float],
    weights: np.ndarray,
) -> np.ndarray:
    values = np.asarray(component_nll, dtype=float)
    if values.shape[0] != len(component_candidates):
        raise ValueError("component NLL first axis does not match component candidates")
    groups = _component_groups(component_candidates, marginal_candidates, lapse_levels)
    rows = [_weighted_mixture_nll(values[group], weights) for group in groups]
    return np.asarray(rows, dtype=float)


def _confirmation_marginal_candidates(
    candidates: Sequence[Mapping[str, Any]],
    stage_nll: np.ndarray,
    reference: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    values = np.asarray(stage_nll, dtype=float)
    if values.shape != (len(candidates),) or np.any(~np.isfinite(values)):
        raise ValueError("complete finite marginal stage-1 NLL vector required")
    screening = config["screening"]
    order = np.argsort(values, kind="stable")
    selected = set(int(index) for index in order[: int(screening["top_k_marginal_stage1"])])
    reference_index = _matching_marginal_index(candidates, reference)
    if bool(screening["always_include_reference_rho_m"]):
        selected.add(reference_index)
    if bool(screening["always_include_rho_zero_parent"]):
        parent = dict(reference)
        parent["rho"] = 0.0
        parent["id"] = _marginal_candidate_id(0.0, float(parent["m"]), float(parent["g"]))
        selected.add(_matching_marginal_index(candidates, parent))
    if bool(screening["always_include_profile_winner_per_rho_m_level"]):
        for parameter in RECOVERED_PARAMETERS:
            for level in sorted({float(row[parameter]) for row in candidates}):
                eligible = np.asarray(
                    [
                        index
                        for index, row in enumerate(candidates)
                        if np.isclose(float(row[parameter]), level, atol=1e-12)
                    ],
                    dtype=int,
                )
                selected.add(int(eligible[np.argmin(values[eligible])]))
    return [dict(candidates[index]) for index in sorted(selected)]


def _components_for_marginals(
    component_candidates: Sequence[Mapping[str, Any]],
    marginal_candidates: Sequence[Mapping[str, Any]],
    lapse_levels: Sequence[float],
) -> list[dict[str, Any]]:
    groups = _component_groups(component_candidates, marginal_candidates, lapse_levels)
    indices = [index for group in groups for index in group]
    if len(indices) != len(set(indices)):
        raise ValueError("selected marginal candidates share component rows")
    return [dict(component_candidates[index]) for index in indices]


def _posterior_weights_from_nll(values: np.ndarray, prior_weights: np.ndarray) -> np.ndarray:
    log_values = np.log(prior_weights) - np.asarray(values, dtype=float)
    log_values -= np.max(log_values)
    posterior = np.exp(log_values)
    return posterior / posterior.sum()


def _analyse_marginal_level(
    raw: Mapping[str, Any],
    component_candidates: Sequence[Mapping[str, Any]],
    marginal_candidates: Sequence[Mapping[str, Any]],
    reference_candidate_id: str,
    lapse_levels: Sequence[float],
    weights: np.ndarray,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    component_nll = np.asarray(raw["nll"], dtype=float)
    if component_nll.shape != (len(component_candidates), len(raw["seeds"])):
        raise ValueError("complete component NLL matrix required")
    marginal_nll = _marginalize_component_nll(
        component_nll,
        component_candidates,
        marginal_candidates,
        lapse_levels,
        weights,
    )
    combined = np.asarray([_ensemble_nll(row) for row in marginal_nll])
    order = np.argsort(combined, kind="stable")
    ids = [str(row["id"]) for row in marginal_candidates]
    best_index = int(order[0])
    seed_winner_indices = np.argmin(marginal_nll, axis=0)
    seed_winner_ids = [ids[int(index)] for index in seed_winner_indices]
    modal_winner, modal_winner_fraction = _modal_fraction(seed_winner_ids)
    seed_rho_classes = [
        "positive" if float(marginal_candidates[int(index)]["rho"]) > 0.0 else "zero"
        for index in seed_winner_indices
    ]
    modal_rho_class, modal_rho_fraction = _modal_fraction(seed_rho_classes)
    delta = combined - combined[best_index]
    plausible = np.flatnonzero(delta <= float(thresholds["plausible_candidate_delta_nll"]))
    difference_se = [
        _standard_error(marginal_nll[index] - marginal_nll[best_index])
        for index in plausible
        if int(index) != best_index
    ]
    maximum_difference_se = float(max(difference_se, default=0.0))
    reference_positions = [index for index, value in enumerate(ids) if value == reference_candidate_id]
    if len(reference_positions) != 1:
        raise ValueError("reference rho-m point must occur once in confirmation set")
    reference_index = int(reference_positions[0])
    reference_nll_se = _standard_error(marginal_nll[reference_index])
    numerical_resolved = bool(
        maximum_difference_se
        <= float(thresholds["maximum_plausible_candidate_paired_difference_se_nll"])
        and reference_nll_se
        <= float(thresholds["maximum_reference_candidate_absolute_nll_se"])
    )
    groups = _component_groups(component_candidates, marginal_candidates, lapse_levels)
    ranking = []
    for index in order:
        component_indices = groups[int(index)]
        component_combined = np.asarray(
            [_ensemble_nll(component_nll[component_index]) for component_index in component_indices]
        )
        nuisance_posterior = _posterior_weights_from_nll(component_combined, weights)
        ranking.append(
            {
                **dict(marginal_candidates[int(index)]),
                "combined_nll": float(combined[index]),
                "delta_nll": float(delta[index]),
                "seed_nll": marginal_nll[index].tolist(),
                "seed_nll_se": _standard_error(marginal_nll[index]),
                "lapse_components": [
                    {
                        "lapse": float(lapse),
                        "component_id": str(component_candidates[component_index]["id"]),
                        "combined_nll": float(component_value),
                        "posterior_weight": float(posterior),
                    }
                    for lapse, component_index, component_value, posterior in zip(
                        lapse_levels,
                        component_indices,
                        component_combined,
                        nuisance_posterior,
                    )
                ],
            }
        )
    soft_weights = np.exp(-delta)
    soft_weights /= soft_weights.sum()
    soft_parameter_mean = {
        name: float(
            np.sum(
                soft_weights
                * np.asarray([float(candidate[name]) for candidate in marginal_candidates])
            )
        )
        for name in RECOVERED_PARAMETERS
    }
    profiles = {}
    for parameter in RECOVERED_PARAMETERS:
        profiles[parameter] = {}
        for value in sorted({float(row[parameter]) for row in marginal_candidates}):
            eligible = [
                index
                for index, candidate in enumerate(marginal_candidates)
                if np.isclose(float(candidate[parameter]), value, atol=1e-12)
            ]
            winner = min(eligible, key=lambda index: combined[index])
            profiles[parameter][f"{value:.3f}"] = {
                "candidate_id": ids[winner],
                "combined_nll": float(combined[winner]),
                "delta_nll": float(delta[winner]),
                "paired_difference_se_vs_best": _standard_error(
                    marginal_nll[winner] - marginal_nll[best_index]
                ),
            }
    return {
        "tier_id": str(raw["tier_id"]),
        "particle_count": int(raw["particle_count"]),
        "seeds": [int(value) for value in raw["seeds"]],
        "ensemble_winner": dict(marginal_candidates[best_index]),
        "ensemble_order": [ids[int(index)] for index in order],
        "seed_winner_ids": seed_winner_ids,
        "seed_modal_exact_winner": modal_winner,
        "seed_modal_exact_winner_fraction": modal_winner_fraction,
        "seed_rho_classes": seed_rho_classes,
        "seed_modal_rho_class": modal_rho_class,
        "seed_modal_rho_class_fraction": modal_rho_fraction,
        "plausible_candidate_count": int(len(plausible)),
        "maximum_plausible_candidate_paired_difference_se_nll": maximum_difference_se,
        "reference_candidate_absolute_nll_se": reference_nll_se,
        "reference_candidate_delta_nll": float(delta[reference_index]),
        "reference_candidate_within_2_nll": bool(
            delta[reference_index] <= float(thresholds["plausible_candidate_delta_nll"])
        ),
        "numerically_resolved": numerical_resolved,
        "soft_parameter_mean": soft_parameter_mean,
        "parameter_profiles": profiles,
        "candidate_ranking": ranking,
        "component_candidate_count": len(component_candidates),
        "marginal_candidate_count": len(marginal_candidates),
        "new_compute_runtime_seconds": float(raw["new_compute_runtime_seconds"]),
    }


def _run_payloads(
    payloads: Sequence[Mapping[str, Any]], workers: int
) -> list[dict[str, Any]]:
    if not payloads:
        return []
    if workers == 1:
        return [_run_level_payload(payload) for payload in payloads]
    results: list[dict[str, Any] | None] = [None] * len(payloads)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_run_level_payload, payload): index
            for index, payload in enumerate(payloads)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            completed += 1
            print(f"[marginal-lapse] completed={completed}/{len(payloads)}", flush=True)
    if any(row is None for row in results):
        raise AssertionError("incomplete marginalized-lapse worker results")
    return [row for row in results if row is not None]


def _monotonic_axes(primary: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    definitions = {
        "rho": ("rho_zero", "center", "rho_high"),
        "m": ("m_low", "center", "m_high"),
    }
    lookup = {
        (int(row["subject_id"]), int(row["replicate"]), str(row["scenario_id"])): row
        for row in primary
    }
    axes = {}
    for parameter, scenarios in definitions.items():
        estimates = {scenario: [] for scenario in scenarios}
        ordered = []
        for subject_id in sorted({int(row["subject_id"]) for row in primary}):
            for replicate in sorted({int(row["replicate"]) for row in primary}):
                rows = [lookup[(subject_id, replicate, scenario)] for scenario in scenarios]
                values = [
                    float(row["final_level"]["soft_parameter_mean"][parameter])
                    for row in rows
                ]
                for scenario, value in zip(scenarios, values):
                    estimates[scenario].append(value)
                ordered.append(values[0] < values[1] < values[2])
        means = [float(np.mean(estimates[scenario])) for scenario in scenarios]
        axes[parameter] = {
            "scenario_ids": list(scenarios),
            "soft_estimate_means": means,
            "group_mean_monotonic": bool(means[0] < means[1] < means[2]),
            "paired_order": _rate_summary(ordered),
        }
    return axes


def _parameter_summary(
    primary: Sequence[Mapping[str, Any]], parameter: str
) -> dict[str, Any]:
    exact = []
    errors = []
    for row in primary:
        true_value = float(row["true_candidate"][parameter])
        selected = float(row["final_level"]["ensemble_winner"][parameter])
        exact.append(np.isclose(true_value, selected, atol=1e-12))
        errors.append(abs(true_value - selected))
    return {
        "exact_recovery": _rate_summary(exact),
        "mean_absolute_error": float(np.mean(errors)),
        "median_absolute_error": float(np.median(errors)),
    }


def _aggregate(
    datasets: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    primary = [row for row in datasets if row["analysis_role"] == "primary_recovery"]
    stress = [row for row in datasets if row["analysis_role"] == "misspecification_stress"]
    gates = config["recovery_gates"]
    thresholds = config["diagnostic_thresholds"]
    final_levels = [row["final_level"] for row in datasets]
    numerical = _rate_summary([bool(level["numerically_resolved"]) for level in final_levels])
    exact_stable = _rate_summary(
        [
            float(level["seed_modal_exact_winner_fraction"])
            >= float(thresholds["minimum_seed_modal_exact_winner_fraction"])
            for level in final_levels
        ]
    )
    rho_seed_stable = _rate_summary(
        [
            float(level["seed_modal_rho_class_fraction"])
            >= float(thresholds["minimum_seed_modal_rho_class_fraction"])
            for level in final_levels
        ]
    )
    true_within = _rate_summary(
        [bool(row["final_level"]["reference_candidate_within_2_nll"]) for row in primary]
    )
    rho_correct = [
        (float(row["true_candidate"]["rho"]) > 0.0)
        == (float(row["final_level"]["ensemble_winner"]["rho"]) > 0.0)
        for row in primary
    ]
    rho_zero = [
        correct
        for correct, row in zip(rho_correct, primary)
        if np.isclose(float(row["true_candidate"]["rho"]), 0.0, atol=1e-12)
    ]
    rho_positive = [
        correct
        for correct, row in zip(rho_correct, primary)
        if float(row["true_candidate"]["rho"]) > 0.0
    ]
    rho_accuracy = _rate_summary(rho_correct)
    rho_specificity = _rate_summary(rho_zero)
    rho_sensitivity = _rate_summary(rho_positive)
    parameter = {
        name: _parameter_summary(primary, name) for name in RECOVERED_PARAMETERS
    }
    scenario_summaries = {}
    for scenario_id in config["data_scope"]["primary_scenarios"]:
        rows = [row for row in primary if row["scenario_id"] == scenario_id]
        scenario_summaries[str(scenario_id)] = {
            "dataset_count": len(rows),
            "true_rho_m_point_within_2_nll": _rate_summary(
                [bool(row["final_level"]["reference_candidate_within_2_nll"]) for row in rows]
            ),
            "median_reference_delta_nll": float(
                np.median(
                    [float(row["final_level"]["reference_candidate_delta_nll"]) for row in rows]
                )
            ),
        }
    minimum_scenario = min(
        float(row["true_rho_m_point_within_2_nll"]["value"])
        for row in scenario_summaries.values()
    )
    monotonic = _monotonic_axes(primary)
    monotonic_count = int(sum(bool(row["group_mean_monotonic"]) for row in monotonic.values()))
    minimum_paired_monotonic = float(
        min(row["paired_order"]["value"] for row in monotonic.values())
    )
    lapse_strata = {}
    for lapse in config["nuisance_lapse"]["levels"]:
        rows = [
            row
            for row in primary
            if np.isclose(float(row["true_candidate"]["lapse"]), float(lapse), atol=1e-12)
        ]
        stratum_rho_correct = [
            (float(row["true_candidate"]["rho"]) > 0.0)
            == (float(row["final_level"]["ensemble_winner"]["rho"]) > 0.0)
            for row in rows
        ]
        stratum_m_exact = [
            np.isclose(
                float(row["true_candidate"]["m"]),
                float(row["final_level"]["ensemble_winner"]["m"]),
                atol=1e-12,
            )
            for row in rows
        ]
        lapse_strata[f"{float(lapse):.3f}"] = {
            "dataset_count": len(rows),
            "true_rho_m_point_within_2_nll": _rate_summary(
                [bool(row["final_level"]["reference_candidate_within_2_nll"]) for row in rows]
            ),
            "rho_classification": _rate_summary(stratum_rho_correct),
            "exact_m_recovery": _rate_summary(stratum_m_exact),
        }
    minimum_lapse_within = min(
        float(row["true_rho_m_point_within_2_nll"]["value"])
        for row in lapse_strata.values()
    )
    minimum_lapse_rho = min(
        float(row["rho_classification"]["value"]) for row in lapse_strata.values()
    )
    minimum_lapse_m = min(
        float(row["exact_m_recovery"]["value"]) for row in lapse_strata.values()
    )
    stress_summary = {}
    for scenario_id in config["data_scope"]["misspecification_scenarios"]:
        rows = [row for row in stress if row["scenario_id"] == scenario_id]
        stress_summary[str(scenario_id)] = {
            "dataset_count": len(rows),
            "projected_rho_m_point_within_2_nll": _rate_summary(
                [bool(row["final_level"]["reference_candidate_within_2_nll"]) for row in rows]
            ),
            "median_projected_reference_delta_nll": float(
                np.median(
                    [float(row["final_level"]["reference_candidate_delta_nll"]) for row in rows]
                )
            ),
        }

    checks = {
        "numerically_resolved_dataset_fraction": {
            **numerical,
            "threshold": float(gates["minimum_numerically_resolved_dataset_fraction"]),
            "passed": numerical["value"] >= float(gates["minimum_numerically_resolved_dataset_fraction"]),
        },
        "seed_exact_winner_stable_fraction": {
            **exact_stable,
            "threshold": float(gates["minimum_seed_exact_winner_stable_fraction"]),
            "passed": exact_stable["value"] >= float(gates["minimum_seed_exact_winner_stable_fraction"]),
        },
        "seed_rho_class_stable_fraction": {
            **rho_seed_stable,
            "threshold": float(gates["minimum_seed_rho_class_stable_fraction"]),
            "passed": rho_seed_stable["value"] >= float(gates["minimum_seed_rho_class_stable_fraction"]),
        },
        "primary_true_rho_m_point_within_2_nll_fraction": {
            **true_within,
            "threshold": float(gates["minimum_primary_true_grid_point_within_2_nll_fraction"]),
            "passed": true_within["value"] >= float(gates["minimum_primary_true_grid_point_within_2_nll_fraction"]),
        },
        "minimum_per_scenario_true_rho_m_point_within_2_nll_fraction": {
            "value": minimum_scenario,
            "threshold": float(gates["minimum_per_scenario_true_grid_point_within_2_nll_fraction"]),
            "passed": minimum_scenario >= float(gates["minimum_per_scenario_true_grid_point_within_2_nll_fraction"]),
        },
        "rho_zero_vs_positive_classification_accuracy": {
            **rho_accuracy,
            "threshold": float(gates["minimum_rho_zero_vs_positive_classification_accuracy"]),
            "passed": rho_accuracy["value"] >= float(gates["minimum_rho_zero_vs_positive_classification_accuracy"]),
        },
        "rho_zero_specificity": {
            **rho_specificity,
            "threshold": float(gates["minimum_rho_zero_specificity"]),
            "passed": rho_specificity["value"] >= float(gates["minimum_rho_zero_specificity"]),
        },
        "rho_positive_sensitivity": {
            **rho_sensitivity,
            "threshold": float(gates["minimum_rho_positive_sensitivity"]),
            "passed": rho_sensitivity["value"] >= float(gates["minimum_rho_positive_sensitivity"]),
        },
        "exact_m_recovery_fraction": {
            **parameter["m"]["exact_recovery"],
            "threshold": float(gates["minimum_exact_m_recovery_fraction"]),
            "passed": parameter["m"]["exact_recovery"]["value"] >= float(gates["minimum_exact_m_recovery_fraction"]),
        },
        "median_absolute_rho_error": {
            "value": parameter["rho"]["median_absolute_error"],
            "threshold": float(gates["maximum_median_absolute_rho_error"]),
            "passed": parameter["rho"]["median_absolute_error"] <= float(gates["maximum_median_absolute_rho_error"]),
        },
        "median_absolute_m_error": {
            "value": parameter["m"]["median_absolute_error"],
            "threshold": float(gates["maximum_median_absolute_m_error"]),
            "passed": parameter["m"]["median_absolute_error"] <= float(gates["maximum_median_absolute_m_error"]),
        },
        "monotonic_parameter_axes": {
            "value": monotonic_count,
            "threshold": int(gates["minimum_monotonic_parameter_axes"]),
            "passed": monotonic_count >= int(gates["minimum_monotonic_parameter_axes"]),
        },
        "minimum_paired_monotonic_fraction_per_axis": {
            "value": minimum_paired_monotonic,
            "threshold": float(gates["minimum_paired_monotonic_fraction_per_axis"]),
            "passed": minimum_paired_monotonic >= float(gates["minimum_paired_monotonic_fraction_per_axis"]),
        },
        "minimum_per_generating_lapse_stratum_true_within_2_nll_fraction": {
            "value": minimum_lapse_within,
            "threshold": float(gates["minimum_per_generating_lapse_stratum_true_within_2_nll_fraction"]),
            "passed": minimum_lapse_within >= float(gates["minimum_per_generating_lapse_stratum_true_within_2_nll_fraction"]),
        },
        "minimum_per_generating_lapse_stratum_rho_classification_accuracy": {
            "value": minimum_lapse_rho,
            "threshold": float(gates["minimum_per_generating_lapse_stratum_rho_classification_accuracy"]),
            "passed": minimum_lapse_rho >= float(gates["minimum_per_generating_lapse_stratum_rho_classification_accuracy"]),
        },
        "minimum_per_generating_lapse_stratum_exact_m_recovery_fraction": {
            "value": minimum_lapse_m,
            "threshold": float(gates["minimum_per_generating_lapse_stratum_exact_m_recovery_fraction"]),
            "passed": minimum_lapse_m >= float(gates["minimum_per_generating_lapse_stratum_exact_m_recovery_fraction"]),
        },
    }
    passed = bool(all(bool(row["passed"]) for row in checks.values()))
    route = str(gates["route_on_pass"] if passed else gates["route_on_fail"])
    return {
        "primary_dataset_count": len(primary),
        "misspecification_dataset_count": len(stress),
        "escalated_dataset_count": int(sum(bool(row["escalated"]) for row in datasets)),
        "numerical_stability": {
            "numerically_resolved": numerical,
            "seed_exact_winner_stable": exact_stable,
            "seed_rho_class_stable": rho_seed_stable,
        },
        "primary_true_rho_m_point_within_2_nll": true_within,
        "rho_zero_vs_positive_classification": rho_accuracy,
        "rho_zero_specificity": rho_specificity,
        "rho_positive_sensitivity": rho_sensitivity,
        "parameter_recovery": parameter,
        "scenario_summaries": scenario_summaries,
        "monotonic_axes": monotonic,
        "generating_lapse_strata": lapse_strata,
        "misspecification_stress": stress_summary,
        "gate_checks": checks,
        "all_recovery_gates_passed": passed,
        "interpretation_boundary": (
            "rho and m only are recovered; lapse is a whole-sequence nuisance "
            "marginalized with frozen prior weights; g-extreme rows are stress only "
            "and no real choices are fitted"
        ),
    }, route


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = args.config.resolve()
    output = args.output.resolve()
    previous_report_path = output / "marginalized_lapse_recovery_report.json"
    previous_compute_wall_runtime = None
    if previous_report_path.exists() and not args.force:
        try:
            previous_report = _load_json(previous_report_path)
            previous_compute_wall_runtime = float(
                previous_report.get("compute_wall_runtime_seconds", previous_report["runtime_seconds"])
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            previous_compute_wall_runtime = None
    config = _load_yaml(config_path)
    lapse_levels, lapse_weights = _validate_nuisance(config)
    fixed_report_path = (ROOT / str(config["source"]["fixed_g_report"])).resolve()
    if _sha256(fixed_report_path) != str(config["source"]["expected_fixed_g_report_sha256"]):
        raise ValueError("fixed-g source report hash mismatch")
    fixed_report = _load_json(fixed_report_path)
    if fixed_report["status"] != "fixed_g_recovery_complete":
        raise ValueError("complete fixed-g recovery report required")
    recovery_path = Path(str(fixed_report["source_recovery_report_path"])).resolve()
    if _sha256(recovery_path) != str(fixed_report["source_recovery_report_sha256"]):
        raise ValueError("upstream recovery report hash mismatch")
    source = _load_json(recovery_path)
    if source["model_id"] != config["data_scope"]["model_id"]:
        raise ValueError("source model id mismatch")
    if int(source["stage1_particle_count"]) != int(config["screening"]["source_particle_count"]):
        raise ValueError("source stage-1 particle count mismatch")
    source_rows = {str(row["dataset_id"]): row for row in source["datasets"]}
    fixed_rows = {str(row["dataset_id"]): row for row in fixed_report["datasets"]}
    if len(source_rows) != int(source["dataset_count"]):
        raise ValueError("source report contains duplicate datasets")
    primary_scenarios = set(config["data_scope"]["primary_scenarios"])
    stress_scenarios = set(config["data_scope"]["misspecification_scenarios"])
    selected_rows = [
        row
        for row in source["datasets"]
        if str(row["scenario_id"]) in primary_scenarios | stress_scenarios
    ]
    requested_ids = None
    if args.dataset_ids:
        requested_ids = {value.strip() for value in args.dataset_ids.split(",") if value.strip()}
        unknown = requested_ids - set(source_rows)
        if unknown:
            raise ValueError(f"unknown dataset ids: {sorted(unknown)}")
        selected_rows = [row for row in selected_rows if row["dataset_id"] in requested_ids]
    if not selected_rows:
        raise ValueError("no datasets selected")

    base_path = Path(source["base_config_path"]).resolve()
    if _sha256(base_path) != str(source["base_config_sha256"]):
        raise ValueError("base config changed since source recovery")
    base = load_config(base_path)
    priors, kernels_by_prior, _ = build_frozen_geometry(base)
    prior_id = str(base["rule_space"]["primary_prior"])
    prior = priors[prior_id]
    kernels = kernels_by_prior[prior_id]
    model_path = ROOT / "src/Bayesian_state/utils/model_0804.py"
    recovery_impl_path = ROOT / "src/Bayesian_state/utils/model_0804_recovery.py"
    model_sha256 = _sha256(model_path)
    recovery_sha256 = _sha256(recovery_impl_path)
    if model_sha256 != source["implementation_sha256"]["model_0804.py"]:
        raise ValueError("model implementation changed since source recovery")
    if recovery_sha256 != source["implementation_sha256"]["model_0804_recovery.py"]:
        raise ValueError("recovery implementation changed since source recovery")

    marginal_grid = _marginal_grid(config)
    component_grid = _component_grid(config)
    source_candidates = list(source["candidate_grid"])
    fixed_g = float(config["architecture"]["fixed_g"])
    source_output = recovery_path.parent
    config_sha256 = _sha256(config_path)
    seeds = [int(value) for value in config["filter"]["seeds"]]
    tier_by_id = {str(row["id"]): row for row in config["filter"]["tiers"]}
    if set(tier_by_id) != {"n8192", "n32768"}:
        raise ValueError("marginalized-lapse recovery requires n8192 and n32768 tiers")
    workers = int(args.workers or config["execution"]["workers"])
    worker_threads = int(config["execution"]["worker_blas_threads"])
    if workers < 1:
        raise ValueError("workers must be positive")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(worker_threads)

    datasets: dict[str, dict[str, Any]] = {}
    common: dict[str, dict[str, Any]] = {}
    for source_row in selected_rows:
        dataset_id = str(source_row["dataset_id"])
        true_candidate = dict(source_row["true_candidate"])
        reference = _project_reference_rho_m(true_candidate, fixed_g)
        component_stage_nll = _source_stage_nll(
            source_output,
            dataset_id,
            source_candidates,
            component_grid,
        )
        marginal_stage_nll = _marginalize_component_nll(
            component_stage_nll,
            component_grid,
            marginal_grid,
            lapse_levels,
            lapse_weights,
        )
        confirmed_marginal = _confirmation_marginal_candidates(
            marginal_grid,
            marginal_stage_nll,
            reference,
            config,
        )
        confirmed_components = _components_for_marginals(
            component_grid,
            confirmed_marginal,
            lapse_levels,
        )
        data_path = source_output / "datasets" / f"{dataset_id}.npz"
        with np.load(data_path, allow_pickle=False) as stored:
            arrays = {name: stored[name].copy() for name in ("q_values", "choices", "feedback")}
        role = (
            "primary_recovery"
            if str(source_row["scenario_id"]) in primary_scenarios
            else "misspecification_stress"
        )
        datasets[dataset_id] = {
            "dataset_id": dataset_id,
            "subject_id": int(source_row["subject_id"]),
            "scenario_id": str(source_row["scenario_id"]),
            "replicate": int(source_row["replicate"]),
            "n_trials": int(source_row["n_trials"]),
            "analysis_role": role,
            "true_candidate": true_candidate,
            "reference_rho_m_projection": reference,
            "marginal_grid_count": len(marginal_grid),
            "component_grid_count": len(component_grid),
            "confirmed_marginal_candidate_count": len(confirmed_marginal),
            "confirmed_component_candidate_count": len(confirmed_components),
            "stage1_best_marginal_candidate": dict(
                marginal_grid[int(np.argmin(marginal_stage_nll))]
            ),
            "levels": {},
        }
        common[dataset_id] = {
            "analysis_id": config["analysis_id"],
            "config_sha256": config_sha256,
            "dataset_id": dataset_id,
            "q_values": arrays["q_values"],
            "choices": arrays["choices"],
            "feedback": arrays["feedback"],
            "prior": prior,
            "kernels": kernels,
            "model_id": source["model_id"],
            "architecture": {**dict(config["architecture"]), "g": fixed_g},
            "candidates": confirmed_components,
            "marginal_candidates": confirmed_marginal,
            "seeds": seeds,
            "resample_threshold_fraction": config["filter"]["resample_threshold_fraction"],
            "worker_blas_threads": worker_threads,
            "model_sha256": model_sha256,
            "recovery_sha256": recovery_sha256,
            "force": bool(args.force),
        }

    def payload_for(dataset_id: str, tier_id: str) -> dict[str, Any]:
        candidate_ids = {str(row["id"]) for row in common[dataset_id]["candidates"]}
        reused = {}
        if bool(config["source"]["reuse_compatible_fixed_g_high_particle_values"]):
            reused = _reuse_values(fixed_rows.get(dataset_id), tier_id, candidate_ids)
        return {
            **common[dataset_id],
            "tier_id": tier_id,
            "particle_count": int(tier_by_id[tier_id]["particle_count"]),
            "checkpoint_path": output / "checkpoints" / f"{dataset_id}_{tier_id}.npz",
            "reused_nll": reused,
        }

    raw8192 = _run_payloads(
        [payload_for(dataset_id, "n8192") for dataset_id in datasets],
        workers,
    )
    for raw in raw8192:
        dataset_id = str(raw["dataset_id"])
        datasets[dataset_id]["levels"]["n8192"] = _analyse_marginal_level(
            raw,
            common[dataset_id]["candidates"],
            common[dataset_id]["marginal_candidates"],
            str(datasets[dataset_id]["reference_rho_m_projection"]["id"]),
            lapse_levels,
            lapse_weights,
            config["diagnostic_thresholds"],
        )
    if args.stop_after != "n8192":
        escalated_ids = []
        for dataset_id, dataset in datasets.items():
            escalated, reasons = _requires_escalation(dataset["levels"]["n8192"], config)
            dataset["escalated"] = escalated
            dataset["escalation_reasons"] = reasons
            if escalated:
                escalated_ids.append(dataset_id)
        raw32768 = _run_payloads(
            [payload_for(dataset_id, "n32768") for dataset_id in escalated_ids],
            workers,
        )
        for raw in raw32768:
            dataset_id = str(raw["dataset_id"])
            datasets[dataset_id]["levels"]["n32768"] = _analyse_marginal_level(
                raw,
                common[dataset_id]["candidates"],
                common[dataset_id]["marginal_candidates"],
                str(datasets[dataset_id]["reference_rho_m_projection"]["id"]),
                lapse_levels,
                lapse_weights,
                config["diagnostic_thresholds"],
            )

    rows = list(datasets.values())
    for row in rows:
        final_tier = "n32768" if "n32768" in row["levels"] else "n8192"
        row["final_tier_id"] = final_tier
        row["final_level"] = row["levels"][final_tier]
    expected_primary = int(config["data_scope"]["expected_primary_dataset_count"])
    expected_stress = int(config["data_scope"]["expected_misspecification_dataset_count"])
    full_selection = requested_ids is None
    complete = bool(
        args.stop_after is None
        and full_selection
        and sum(row["analysis_role"] == "primary_recovery" for row in rows) == expected_primary
        and sum(row["analysis_role"] == "misspecification_stress" for row in rows) == expected_stress
        and all((not row.get("escalated", False)) or "n32768" in row["levels"] for row in rows)
    )
    if complete:
        aggregate, route = _aggregate(rows, config)
        status = "marginalized_lapse_recovery_complete"
    else:
        aggregate = None
        route = "partial_run_no_route_decision"
        status = "marginalized_lapse_recovery_partial"
    refresh_runtime = float(time.time() - started)
    compute_wall_runtime = (
        refresh_runtime if previous_compute_wall_runtime is None else previous_compute_wall_runtime
    )
    scored_runtime = float(
        sum(
            float(level["new_compute_runtime_seconds"])
            for row in rows
            for level in row["levels"].values()
        )
    )
    report = {
        "analysis_id": config["analysis_id"],
        "status": status,
        "scope": config["scope"],
        "route_decision": route,
        "runtime_seconds": compute_wall_runtime,
        "compute_wall_runtime_seconds": compute_wall_runtime,
        "report_refresh_runtime_seconds": refresh_runtime,
        "scored_candidate_runtime_sum_seconds": scored_runtime,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "source_fixed_g_report_path": str(fixed_report_path),
        "source_fixed_g_report_sha256": _sha256(fixed_report_path),
        "source_recovery_report_path": str(recovery_path),
        "source_recovery_report_sha256": _sha256(recovery_path),
        "implementation_sha256": {
            "model_0804.py": model_sha256,
            "model_0804_recovery.py": recovery_sha256,
            "marginalized_lapse_runner": _sha256(Path(__file__).resolve()),
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "fixed_g": fixed_g,
        "nuisance_lapse": {
            "levels": lapse_levels.tolist(),
            "prior_weights": lapse_weights.tolist(),
            "marginalization_scope": "whole_sequence",
        },
        "marginal_candidate_count": len(marginal_grid),
        "component_candidate_count": len(component_grid),
        "selected_dataset_count": len(rows),
        "workers": workers,
        "filter_seeds": seeds,
        "particle_tiers": {
            key: int(value["particle_count"]) for key, value in tier_by_id.items()
        },
        "recovery_gates": dict(config["recovery_gates"]),
        "diagnostic_thresholds": dict(config["diagnostic_thresholds"]),
        "guardrails": list(config["guardrails"]),
        "aggregate": aggregate,
        "datasets": rows,
    }
    report_path = output / "marginalized_lapse_recovery_report.json"
    _atomic_json(report_path, report)
    print(
        f"MARGINAL_LAPSE status={status} route={route} output={report_path} "
        f"runtime={compute_wall_runtime:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
