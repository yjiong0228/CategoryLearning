#!/usr/bin/env python3
"""Run replicated FA2R recovery with g fixed at the frozen grid centre."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0803_cond1 import build_frozen_geometry, load_config  # noqa: E402
from scripts.run_model_0804_particle_stability_audit import (  # noqa: E402
    _analyse_level,
    _run_level_payload,
)
from scripts.run_model_0804_regeneration_recovery import (  # noqa: E402
    _atomic_json,
    _candidate_id,
    _rate_summary,
)


DEFAULT_CONFIG = ROOT / "configs/model_0804_fixed_g_recovery.yaml"
DEFAULT_OUTPUT = (
    ROOT / "results/zhuran/model_0804_cond1/fixed_g_recovery_20260805_v1"
)
PARAMETERS = ("rho", "m", "lapse")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("fixed-g config must contain a mapping")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _restricted_grid(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    fixed_g = float(config["architecture"]["fixed_g"])
    grid = config["candidate_grid"]
    candidates = []
    for rho in grid["rho"]:
        for m in grid["m"]:
            for lapse in grid["lapse"]:
                values = {
                    "rho": float(rho),
                    "m": float(m),
                    "g": fixed_g,
                    "lapse": float(lapse),
                }
                candidates.append({"id": _candidate_id(values), **values})
    if len(candidates) != 27 or len({row["id"] for row in candidates}) != 27:
        raise ValueError("restricted candidate grid must contain 27 unique points")
    return candidates


def _matching_index(
    candidates: Sequence[Mapping[str, Any]], target: Mapping[str, Any]
) -> int:
    matches = [
        index
        for index, candidate in enumerate(candidates)
        if all(
            np.isclose(float(candidate[name]), float(target[name]), atol=1e-12)
            for name in ("rho", "m", "g", "lapse")
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"target is not a unique restricted point: {target}")
    return int(matches[0])


def _project_to_fixed_g(
    target: Mapping[str, Any], fixed_g: float
) -> dict[str, Any]:
    projected = {
        name: float(target[name]) for name in ("rho", "m", "lapse")
    }
    projected["g"] = float(fixed_g)
    projected["id"] = _candidate_id(projected)
    return projected


def _confirmation_candidates(
    candidates: Sequence[Mapping[str, Any]],
    stage_nll: np.ndarray,
    reference: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    values = np.asarray(stage_nll, dtype=float)
    if values.shape != (len(candidates),) or np.any(~np.isfinite(values)):
        raise ValueError("complete restricted stage-1 NLL vector required")
    screening = config["screening"]
    order = np.argsort(values, kind="stable")
    selected = set(int(index) for index in order[: int(screening["top_k_restricted_stage1"])])
    reference_index = _matching_index(candidates, reference)
    if bool(screening["always_include_reference_projection"]):
        selected.add(reference_index)
    if bool(screening["always_include_rho_zero_parent"]):
        parent = dict(reference)
        parent["rho"] = 0.0
        parent["id"] = _candidate_id(parent)
        selected.add(_matching_index(candidates, parent))
    if bool(screening["always_include_profile_winner_per_level"]):
        for parameter in PARAMETERS:
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


def _source_stage_nll(
    source_output: Path,
    dataset_id: str,
    source_candidates: Sequence[Mapping[str, Any]],
    restricted_candidates: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    checkpoint_path = source_output / "checkpoints" / f"{dataset_id}.npz"
    with np.load(checkpoint_path, allow_pickle=False) as stored:
        source_values = stored["stage_nll"].astype(float)
    if source_values.shape != (len(source_candidates),) or np.any(~np.isfinite(source_values)):
        raise ValueError(f"invalid source stage-1 checkpoint: {dataset_id}")
    by_id = {
        str(candidate["id"]): float(source_values[index])
        for index, candidate in enumerate(source_candidates)
    }
    missing = {str(row["id"]) for row in restricted_candidates} - set(by_id)
    if missing:
        raise ValueError(f"restricted candidates missing from source grid: {sorted(missing)}")
    return np.asarray([by_id[str(row["id"])] for row in restricted_candidates])


def _reuse_values(
    stability_row: Mapping[str, Any] | None,
    tier_id: str,
    candidate_ids: set[str],
) -> dict[str, dict[str, float]]:
    if stability_row is None or tier_id not in stability_row["levels"]:
        return {}
    level = stability_row["levels"][tier_id]
    seeds = [int(value) for value in level["seeds"]]
    reused = {}
    for row in level["candidate_ranking"]:
        identifier = str(row["id"])
        if identifier not in candidate_ids:
            continue
        reused[identifier] = {
            str(seed): float(value)
            for seed, value in zip(seeds, row["seed_nll"])
        }
    return reused


def _rename_reference_fields(level: dict[str, Any]) -> dict[str, Any]:
    renamed = dict(level)
    mapping = {
        "true_candidate_absolute_nll_se": "reference_candidate_absolute_nll_se",
        "true_candidate_delta_nll": "reference_candidate_delta_nll",
        "true_candidate_within_2_nll": "reference_candidate_within_2_nll",
    }
    for old, new in mapping.items():
        renamed[new] = renamed.pop(old)
    return renamed


def _soft_parameter_mean(level: Mapping[str, Any]) -> dict[str, float]:
    ranking = list(level["candidate_ranking"])
    delta = np.asarray([float(row["delta_nll"]) for row in ranking])
    weights = np.exp(-delta)
    weights /= weights.sum()
    return {
        name: float(
            np.sum(weights * np.asarray([float(row[name]) for row in ranking]))
        )
        for name in PARAMETERS
    }


def _analyse_restricted_level(
    raw: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    reference_candidate_id: str,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    compatible = dict(thresholds)
    compatible["maximum_true_candidate_absolute_nll_se"] = float(
        thresholds["maximum_reference_candidate_absolute_nll_se"]
    )
    level = _analyse_level(raw, candidates, reference_candidate_id, compatible)
    level = _rename_reference_fields(level)
    level["soft_parameter_mean"] = _soft_parameter_mean(level)
    return level


def _requires_escalation(
    level: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    rules = config["escalation"]
    thresholds = config["diagnostic_thresholds"]
    reasons = []
    if bool(rules["escalate_if_n8192_numerically_unresolved"]) and not bool(
        level["numerically_resolved"]
    ):
        reasons.append("n8192_numerically_unresolved")
    if bool(rules["escalate_if_seed_modal_exact_winner_below_threshold"]) and float(
        level["seed_modal_exact_winner_fraction"]
    ) < float(thresholds["minimum_seed_modal_exact_winner_fraction"]):
        reasons.append("seed_modal_exact_winner_below_threshold")
    if bool(rules["escalate_if_seed_rho_class_below_threshold"]) and float(
        level["seed_modal_rho_class_fraction"]
    ) < float(thresholds["minimum_seed_modal_rho_class_fraction"]):
        reasons.append("seed_rho_class_below_threshold")
    return bool(reasons), reasons


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
            print(f"[fixed-g] completed={completed}/{len(payloads)}", flush=True)
    if any(row is None for row in results):
        raise AssertionError("incomplete fixed-g worker results")
    return [row for row in results if row is not None]


def _monotonic_axes(primary: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    definitions = {
        "rho": ("rho_zero", "center", "rho_high"),
        "m": ("m_low", "center", "m_high"),
        "lapse": ("lapse_low", "center", "lapse_high"),
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
        name: _parameter_summary(primary, name) for name in PARAMETERS
    }
    scenario_summaries = {}
    for scenario_id in config["data_scope"]["primary_scenarios"]:
        rows = [row for row in primary if row["scenario_id"] == scenario_id]
        scenario_summaries[str(scenario_id)] = {
            "dataset_count": len(rows),
            "true_grid_point_within_2_nll": _rate_summary(
                [bool(row["final_level"]["reference_candidate_within_2_nll"]) for row in rows]
            ),
            "median_true_candidate_delta_nll": float(
                np.median(
                    [float(row["final_level"]["reference_candidate_delta_nll"]) for row in rows]
                )
            ),
        }
    minimum_scenario = min(
        float(row["true_grid_point_within_2_nll"]["value"])
        for row in scenario_summaries.values()
    )
    monotonic = _monotonic_axes(primary)
    monotonic_count = int(
        sum(bool(row["group_mean_monotonic"]) for row in monotonic.values())
    )
    minimum_paired_monotonic = float(
        min(row["paired_order"]["value"] for row in monotonic.values())
    )
    stress_summary = {}
    for scenario_id in config["data_scope"]["misspecification_scenarios"]:
        rows = [row for row in stress if row["scenario_id"] == scenario_id]
        stress_summary[str(scenario_id)] = {
            "dataset_count": len(rows),
            "projected_reference_within_2_nll": _rate_summary(
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
        "primary_true_grid_point_within_2_nll_fraction": {
            **true_within,
            "threshold": float(gates["minimum_primary_true_grid_point_within_2_nll_fraction"]),
            "passed": true_within["value"] >= float(gates["minimum_primary_true_grid_point_within_2_nll_fraction"]),
        },
        "minimum_per_scenario_true_grid_point_within_2_nll_fraction": {
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
        "exact_lapse_recovery_fraction": {
            **parameter["lapse"]["exact_recovery"],
            "threshold": float(gates["minimum_exact_lapse_recovery_fraction"]),
            "passed": parameter["lapse"]["exact_recovery"]["value"] >= float(gates["minimum_exact_lapse_recovery_fraction"]),
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
        "median_absolute_lapse_error": {
            "value": parameter["lapse"]["median_absolute_error"],
            "threshold": float(gates["maximum_median_absolute_lapse_error"]),
            "passed": parameter["lapse"]["median_absolute_error"] <= float(gates["maximum_median_absolute_lapse_error"]),
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
        "primary_true_grid_point_within_2_nll": true_within,
        "rho_zero_vs_positive_classification": rho_accuracy,
        "rho_zero_specificity": rho_specificity,
        "rho_positive_sensitivity": rho_sensitivity,
        "parameter_recovery": parameter,
        "scenario_summaries": scenario_summaries,
        "monotonic_axes": monotonic,
        "misspecification_stress": stress_summary,
        "gate_checks": checks,
        "all_recovery_gates_passed": passed,
        "interpretation_boundary": (
            "primary gates apply only to correctly specified g=0.35 synthetic data; "
            "g-extreme rows are misspecification stress and no real choices are fitted"
        ),
    }, route


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = args.config.resolve()
    output = args.output.resolve()
    previous_report_path = output / "fixed_g_recovery_report.json"
    previous_compute_wall_runtime = None
    if previous_report_path.exists() and not args.force:
        try:
            previous_report = _load_json(previous_report_path)
            previous_compute_wall_runtime = float(
                previous_report.get(
                    "compute_wall_runtime_seconds",
                    previous_report["runtime_seconds"],
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            previous_compute_wall_runtime = None
    config = _load_yaml(config_path)
    recovery_path = (ROOT / str(config["source"]["recovery_report"])).resolve()
    stability_path = (ROOT / str(config["source"]["stability_report"])).resolve()
    if _sha256(recovery_path) != str(config["source"]["expected_recovery_report_sha256"]):
        raise ValueError("source recovery report hash mismatch")
    if _sha256(stability_path) != str(config["source"]["expected_stability_report_sha256"]):
        raise ValueError("source stability report hash mismatch")
    source = _load_json(recovery_path)
    stability = _load_json(stability_path)
    if source["model_id"] != config["data_scope"]["model_id"]:
        raise ValueError("source model id mismatch")
    if int(source["stage1_particle_count"]) != int(config["screening"]["source_particle_count"]):
        raise ValueError("source stage-1 particle count mismatch")
    source_rows = {str(row["dataset_id"]): row for row in source["datasets"]}
    stability_rows = {str(row["dataset_id"]): row for row in stability["datasets"]}
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

    candidates = _restricted_grid(config)
    source_candidates = list(source["candidate_grid"])
    fixed_g = float(config["architecture"]["fixed_g"])
    source_output = recovery_path.parent
    config_sha256 = _sha256(config_path)
    seeds = [int(value) for value in config["filter"]["seeds"]]
    tier_by_id = {str(row["id"]): row for row in config["filter"]["tiers"]}
    if set(tier_by_id) != {"n8192", "n32768"}:
        raise ValueError("fixed-g recovery requires n8192 and n32768 tiers")
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
        reference = _project_to_fixed_g(true_candidate, fixed_g)
        stage_nll = _source_stage_nll(
            source_output, dataset_id, source_candidates, candidates
        )
        confirmed = _confirmation_candidates(candidates, stage_nll, reference, config)
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
            "reference_projection": reference,
            "restricted_grid_count": len(candidates),
            "confirmed_candidate_count": len(confirmed),
            "stage1_best_restricted_candidate": dict(candidates[int(np.argmin(stage_nll))]),
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
            "candidates": confirmed,
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
        if bool(config["source"]["reuse_compatible_high_particle_values"]):
            reused = _reuse_values(stability_rows.get(dataset_id), tier_id, candidate_ids)
        return {
            **common[dataset_id],
            "tier_id": tier_id,
            "particle_count": int(tier_by_id[tier_id]["particle_count"]),
            "checkpoint_path": output / "checkpoints" / f"{dataset_id}_{tier_id}.npz",
            "reused_nll": reused,
        }

    raw8192 = _run_payloads(
        [payload_for(dataset_id, "n8192") for dataset_id in datasets], workers
    )
    for raw in raw8192:
        dataset_id = str(raw["dataset_id"])
        datasets[dataset_id]["levels"]["n8192"] = _analyse_restricted_level(
            raw,
            common[dataset_id]["candidates"],
            str(datasets[dataset_id]["reference_projection"]["id"]),
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
            [payload_for(dataset_id, "n32768") for dataset_id in escalated_ids], workers
        )
        for raw in raw32768:
            dataset_id = str(raw["dataset_id"])
            datasets[dataset_id]["levels"]["n32768"] = _analyse_restricted_level(
                raw,
                common[dataset_id]["candidates"],
                str(datasets[dataset_id]["reference_projection"]["id"]),
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
        status = "fixed_g_recovery_complete"
    else:
        aggregate = None
        route = "partial_run_no_route_decision"
        status = "fixed_g_recovery_partial"
    refresh_runtime = float(time.time() - started)
    compute_wall_runtime = (
        refresh_runtime
        if previous_compute_wall_runtime is None
        else previous_compute_wall_runtime
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
        "source_recovery_report_path": str(recovery_path),
        "source_recovery_report_sha256": _sha256(recovery_path),
        "source_stability_report_path": str(stability_path),
        "source_stability_report_sha256": _sha256(stability_path),
        "implementation_sha256": {
            "model_0804.py": model_sha256,
            "model_0804_recovery.py": recovery_sha256,
            "fixed_g_runner": _sha256(Path(__file__).resolve()),
        },
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "fixed_g": fixed_g,
        "restricted_candidate_count": len(candidates),
        "selected_dataset_count": len(rows),
        "workers": workers,
        "filter_seeds": seeds,
        "particle_tiers": {key: int(value["particle_count"]) for key, value in tier_by_id.items()},
        "recovery_gates": dict(config["recovery_gates"]),
        "diagnostic_thresholds": dict(config["diagnostic_thresholds"]),
        "guardrails": list(config["guardrails"]),
        "aggregate": aggregate,
        "datasets": rows,
    }
    report_path = output / "fixed_g_recovery_report.json"
    _atomic_json(report_path, report)
    print(
        f"FIXED_G status={status} route={route} output={report_path} "
        f"runtime={compute_wall_runtime:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
