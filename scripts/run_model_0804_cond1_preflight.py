#!/usr/bin/env python3
"""Run the frozen condition-1 HFW implementation preflight.

This entry point validates inputs, exact/particle agreement, the full-set H0
endpoint, FA1/FA2 nesting at g=0, and particle-count/seed stability on pilot
subjects.  It does not perform the formal 32-subject model comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping

import numpy as np
import scipy
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0803_cond1 import (  # noqa: E402
    build_frozen_geometry,
    load_config,
    validate_and_load_inputs,
    validate_subject_cache,
)
from src.Bayesian_state.utils.model_0803 import (  # noqa: E402
    FeatureScaling,
    run_model0803,
    score_choice_predictions,
)
from src.Bayesian_state.utils.model_0804 import (  # noqa: E402
    FA_MODEL_IDS,
    Model0804Parameters,
    fit_model0804,
    nested_child_start,
    run_model0804_exact,
    run_model0804_particle_filter,
)


DEFAULT_CONFIG = ROOT / "configs/model_0804_cond1_preflight.yaml"
DEFAULT_OUTPUT = ROOT / "results/zhuran/model_0804_cond1/preflight_20260804_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--fit-smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _load_subject_arrays(audit: Mapping[str, Any], max_trials: int | None):
    with np.load(Path(audit["q_path"]), allow_pickle=False) as q_payload:
        q_values = q_payload["q"].astype(float)
    with np.load(Path(audit["prediction_path"]), allow_pickle=False) as prediction:
        arrays = {
            "q": q_values,
            "choice": prediction["choice"].astype(int),
            "feedback": prediction["feedback"].astype(float),
            "holdout": prediction["holdout_mask"].astype(bool),
        }
    if max_trials is not None:
        stop = min(int(max_trials), len(arrays["choice"]))
        if stop < 2:
            raise ValueError("--max-trials must retain at least two trials")
        arrays = {key: value[:stop] for key, value in arrays.items()}
    return arrays


def _parameters(config: Mapping[str, Any], model_id: str) -> Model0804Parameters:
    probe = config["preflight"]["fixed_parameter_probe"]
    control = probe[model_id]
    return Model0804Parameters(
        gamma=float(probe["gamma"]),
        w0=float(probe["w0"]),
        kappa=float(probe["kappa"]),
        m=float(control["m"]),
        g=float(control["g"]),
    )


def _resample_threshold(config: Mapping[str, Any], model_id: str) -> float:
    values = config["preflight"]["resample_threshold_fraction"]
    if isinstance(values, Mapping):
        return float(values[model_id])
    return float(values)


def _particle_counts(config: Mapping[str, Any], model_id: str) -> list[int]:
    values = config["preflight"]["particle_counts"]
    selected = values[model_id] if isinstance(values, Mapping) else values
    counts = sorted({int(value) for value in selected})
    minimum_count_entries = 1 if _fa0_exact(config, model_id) else 2
    if len(counts) < minimum_count_entries or counts[0] < 2:
        raise ValueError(
            f"preflight particle_counts for {model_id} must contain at least "
            f"{minimum_count_entries} distinct values >= 2"
        )
    return counts


def _transition_proposals(config: Mapping[str, Any], model_id: str) -> int:
    values = config["preflight"].get("transition_proposals_per_particle", 1)
    selected = values[model_id] if isinstance(values, Mapping) else values
    count = int(selected)
    if count < 1:
        raise ValueError("transition proposals per particle must be positive")
    return count


def _fa0_exact(config: Mapping[str, Any], model_id: str) -> bool:
    if model_id != "FA0":
        return False
    values = config["preflight"].get("fa0_integration", {})
    return str(values.get("mode", "qmc_static_panel")) == "exact_successive_wor_initial_sets"


def _fa0_maximum_exact_sets(config: Mapping[str, Any]) -> int:
    values = config["preflight"].get("fa0_integration", {})
    return int(values.get("maximum_exact_initial_sets", 1_000_000))


def _prefix_arrays(
    arrays: Mapping[str, np.ndarray],
    maximum_trials: int,
) -> dict[str, np.ndarray]:
    stop = min(int(maximum_trials), len(arrays["choice"]))
    return {key: np.asarray(value)[:stop] for key, value in arrays.items()}


def _segment_scores(
    probabilities: np.ndarray,
    choices: np.ndarray,
    holdout: np.ndarray,
) -> dict[str, dict[str, float]]:
    out = {}
    for name, mask in (("train", ~holdout), ("holdout", holdout), ("all", np.ones_like(holdout))):
        if not np.any(mask):
            continue
        score = score_choice_predictions(probabilities, choices, mask)
        out[name] = {key: float(value) for key, value in score.items()}
    return out


def _synthetic_exact_check(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from src.Bayesian_state.utils.model_0803 import build_transition_kernels

    rng = np.random.default_rng(804)
    raw = rng.uniform(0.05, 1.0, size=(3, 3, 2))
    q_values = raw / raw.sum(axis=2, keepdims=True)
    choices = np.asarray([0, 1, 0], dtype=int)
    feedback = np.asarray([1.0, 0.0, 1.0])
    prior = np.asarray([0.2, 0.3, 0.5])
    similarity = np.asarray(
        [[1.0, 0.7, 0.2], [0.7, 1.0, 0.55], [0.2, 0.55, 1.0]]
    )
    kernels = build_transition_kernels(similarity, prior, tau_local=0.25)
    parameters = Model0804Parameters(0.70, 0.40, 2.0, 0.46, 0.35)
    exact = run_model0804_exact(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
        max_branches=int(config["preflight"]["exact_enumeration"]["maximum_branches"]),
    )
    particle_count = max(4096, max(_particle_counts(config, "FA2")))
    proposal_count = _transition_proposals(config, "FA2")
    particle = run_model0804_particle_filter(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=1,
        particle_count=particle_count,
        filter_seed=int(config["preflight"]["independent_filter_seeds"][0]),
        fa0_exact_initial_sets=_fa0_exact(config, "FA0"),
        fa0_maximum_exact_initial_sets=_fa0_maximum_exact_sets(config),
        resample_threshold_fraction=0.01,
        transition_proposals_per_particle=proposal_count,
    )
    max_probability_error = float(
        np.max(np.abs(particle.probabilities - exact.probabilities))
    )
    nll_error = float(abs(particle.nll - exact.nll))
    probability_tolerance = float(
        config["numerics"]["exact_particle_probability_tolerance"]
    )
    nll_tolerance = float(config["numerics"]["exact_particle_nll_tolerance"])
    return {
        "particle_count": particle_count,
        "transition_proposals_per_particle": proposal_count,
        "exact_nll": float(exact.nll),
        "particle_nll": float(particle.nll),
        "nll_error": nll_error,
        "maximum_probability_error": max_probability_error,
        "branch_counts": exact.branch_counts.tolist(),
        "probability_tolerance": probability_tolerance,
        "nll_tolerance": nll_tolerance,
        "passed": bool(
            max_probability_error <= probability_tolerance
            and nll_error <= nll_tolerance
        ),
    }


def _fullset_endpoint_check(
    arrays: Mapping[str, np.ndarray],
    prior: np.ndarray,
    kernels,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    parameters = _parameters(config, "FA0")
    finite = run_model0804_particle_filter(
        arrays["q"],
        arrays["choice"],
        arrays["feedback"],
        prior,
        kernels,
        model_id="FA0",
        parameters=parameters,
        capacity=len(prior),
        particle_count=2,
        filter_seed=int(config["preflight"]["independent_filter_seeds"][0]),
    )
    full_parameters = np.zeros(11, dtype=float)
    full_parameters[:3] = [parameters.gamma, parameters.w0, parameters.kappa]
    full = run_model0803(
        arrays["q"],
        arrays["choice"],
        arrays["feedback"],
        prior,
        kernels,
        model_id="H0",
        full_parameters=full_parameters,
        feature_scaling=FeatureScaling(np.zeros(2), np.ones(2), "preflight"),
    )
    maximum_error = float(np.max(np.abs(finite.probabilities - full.probabilities)))
    tolerance = float(config["numerics"]["probability_tolerance"])
    return {
        "maximum_probability_error": maximum_error,
        "nll_error": float(abs(finite.nll - full.nll)),
        "tolerance": tolerance,
        "passed": bool(maximum_error <= tolerance),
    }


def _g_zero_check(
    arrays: Mapping[str, np.ndarray],
    prior: np.ndarray,
    kernels,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    base = _parameters(config, "FA1")
    fa2_parameters = Model0804Parameters(
        base.gamma, base.w0, base.kappa, base.m, 0.0
    )
    particle_count = max(
        max(_particle_counts(config, "FA1")),
        max(_particle_counts(config, "FA2")),
    )
    proposal_count = _transition_proposals(config, "FA1")
    if proposal_count != _transition_proposals(config, "FA2"):
        raise ValueError(
            "FA1 and FA2 must use the same transition-proposal count for "
            "the exact g=0 nesting check"
        )
    common = dict(
        q_values=arrays["q"],
        choices=arrays["choice"],
        feedback=arrays["feedback"],
        prior=prior,
        kernels=kernels,
        capacity=int(config["architecture"]["capacity"]),
        particle_count=particle_count,
        filter_seed=int(config["preflight"]["independent_filter_seeds"][0]),
        transition_proposals_per_particle=proposal_count,
    )
    fa1 = run_model0804_particle_filter(
        model_id="FA1", parameters=base, **common
    )
    fa2 = run_model0804_particle_filter(
        model_id="FA2", parameters=fa2_parameters, **common
    )
    maximum_error = float(np.max(np.abs(fa1.probabilities - fa2.probabilities)))
    tolerance = float(config["numerics"]["probability_tolerance"])
    return {
        "particle_count": particle_count,
        "transition_proposals_per_particle": proposal_count,
        "maximum_probability_error": maximum_error,
        "tolerance": tolerance,
        "passed": bool(maximum_error <= tolerance),
    }


def _current_feedback_order_check(
    arrays: Mapping[str, np.ndarray],
    prior: np.ndarray,
    kernels,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    changed_feedback = arrays["feedback"].copy()
    changed_trial = min(2, len(changed_feedback) - 2)
    changed_feedback[changed_trial] = 1.0 - changed_feedback[changed_trial]
    parameters = _parameters(config, "FA2")
    common = dict(
        q_values=arrays["q"],
        choices=arrays["choice"],
        prior=prior,
        kernels=kernels,
        model_id="FA2",
        parameters=parameters,
        capacity=int(config["architecture"]["capacity"]),
        particle_count=min(2048, max(_particle_counts(config, "FA2"))),
        filter_seed=int(config["preflight"]["independent_filter_seeds"][0]),
        transition_proposals_per_particle=_transition_proposals(config, "FA2"),
    )
    original = run_model0804_particle_filter(
        feedback=arrays["feedback"], **common
    )
    changed = run_model0804_particle_filter(
        feedback=changed_feedback, **common
    )
    current_error = float(
        np.max(
            np.abs(
                original.probabilities[: changed_trial + 1]
                - changed.probabilities[: changed_trial + 1]
            )
        )
    )
    later_difference = float(
        np.max(
            np.abs(
                original.probabilities[changed_trial + 1 :]
                - changed.probabilities[changed_trial + 1 :]
            )
        )
    )
    tolerance = float(config["numerics"]["probability_tolerance"])
    return {
        "changed_feedback_trial_zero_based": int(changed_trial),
        "maximum_prediction_error_through_changed_trial": current_error,
        "maximum_later_prediction_difference": later_difference,
        "tolerance": tolerance,
        "passed": bool(current_error <= tolerance),
    }


def _particle_stability(
    subject_id: int,
    arrays: Mapping[str, np.ndarray],
    prior: np.ndarray,
    kernels,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    seeds = [int(value) for value in config["preflight"]["independent_filter_seeds"]]
    rows: list[dict[str, Any]] = []
    for model_id in FA_MODEL_IDS:
        counts = _particle_counts(config, model_id)
        proposal_count = _transition_proposals(config, model_id)
        parameters = _parameters(config, model_id)
        model_seeds = seeds[:1] if _fa0_exact(config, model_id) else seeds
        for seed in model_seeds:
            traces = {}
            for particle_count in counts:
                started = time.time()
                trace = run_model0804_particle_filter(
                    arrays["q"],
                    arrays["choice"],
                    arrays["feedback"],
                    prior,
                    kernels,
                    model_id=model_id,
                    parameters=parameters,
                    capacity=int(config["architecture"]["capacity"]),
                    particle_count=particle_count,
                    filter_seed=seed,
                    resample_threshold_fraction=_resample_threshold(
                        config, model_id
                    ),
                    transition_proposals_per_particle=proposal_count,
                    fa0_exact_initial_sets=_fa0_exact(config, model_id),
                    fa0_maximum_exact_initial_sets=_fa0_maximum_exact_sets(config),
                )
                if _fa0_exact(config, model_id):
                    expected = int(
                        config["preflight"]["fa0_integration"][
                            "expected_initial_sets"
                        ]
                    )
                    if int(trace.particle_count) != expected:
                        raise AssertionError(
                            f"FA0 enumerated {trace.particle_count} sets; "
                            f"expected {expected}"
                        )
                traces[particle_count] = (trace, float(time.time() - started))
            reference = traces[counts[-1]][0]
            for particle_count in counts:
                trace, runtime = traces[particle_count]
                difference = np.abs(trace.probabilities - reference.probabilities)
                rows.append(
                    {
                        "subject_id": int(subject_id),
                        "model_id": model_id,
                        "filter_seed": seed,
                        "particle_count": int(trace.particle_count),
                        "requested_particle_count": particle_count,
                        "transition_proposals_per_particle": proposal_count,
                        "integration_mode": trace.integration_mode,
                        "n_trials": int(len(arrays["choice"])),
                        "runtime_seconds": runtime,
                        "nll": float(trace.nll),
                        "maximum_probability_difference_from_largest_particle_count": float(np.max(difference)),
                        "mean_probability_difference_from_largest_particle_count": float(np.mean(difference)),
                        "minimum_pre_choice_ess": float(np.min(trace.pre_choice_ess)),
                        "minimum_post_choice_ess": float(np.min(trace.post_choice_ess)),
                        "resampling_count": int(np.sum(trace.resampled)),
                        "maximum_memory_sync_error": float(np.max(trace.memory_sync_error)),
                        "segments": _segment_scores(
                            trace.probabilities,
                            arrays["choice"],
                            arrays["holdout"],
                        ),
                    }
                )
    return rows


def _evaluate_particle_stability(
    rows: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply an a-priori numerical gate to the largest two particle counts."""

    limits = config["preflight"]["quantitative_stability_gate"]
    maximum_seed_range = float(limits["maximum_largest_count_seed_nll_range"])
    maximum_count_change = float(limits["maximum_top_two_count_nll_change"])
    maximum_probability_change = float(
        limits["maximum_top_two_probability_difference"]
    )
    maximum_mean_probability_change = float(
        limits["maximum_top_two_mean_probability_difference"]
    )
    details: list[dict[str, Any]] = []
    for subject_id in sorted({int(row["subject_id"]) for row in rows}):
        for model_id in FA_MODEL_IDS:
            group = [
                row
                for row in rows
                if int(row["subject_id"]) == subject_id
                and str(row["model_id"]) == model_id
            ]
            counts = sorted({int(row["particle_count"]) for row in group})
            exact_fa0 = bool(
                model_id == "FA0"
                and group
                and all(
                    str(row.get("integration_mode"))
                    == "exact_successive_wor_initial_sets"
                    for row in group
                )
            )
            if len(counts) < 2 and not exact_fa0:
                raise ValueError(
                    f"stability gate needs two particle counts for {subject_id=} "
                    f"and {model_id=}"
                )
            if exact_fa0:
                second_largest = largest = counts[-1]
            else:
                second_largest, largest = counts[-2:]
            largest_rows = [
                row for row in group if int(row["particle_count"]) == largest
            ]
            nll_values = [float(row["nll"]) for row in largest_rows]
            seed_range = float(max(nll_values) - min(nll_values))
            count_changes = []
            probability_changes = []
            mean_probability_changes = []
            if exact_fa0:
                count_changes.append(0.0)
                probability_changes.append(0.0)
                mean_probability_changes.append(0.0)
            else:
                for row in group:
                    if int(row["particle_count"]) != second_largest:
                        continue
                    matching = [
                        reference
                        for reference in largest_rows
                        if int(reference["filter_seed"])
                        == int(row["filter_seed"])
                    ]
                    if len(matching) != 1:
                        raise ValueError(
                            "missing largest-count seed-matched reference"
                        )
                    count_changes.append(
                        abs(float(row["nll"]) - float(matching[0]["nll"]))
                    )
                    probability_changes.append(
                        float(
                            row[
                                "maximum_probability_difference_from_largest_particle_count"
                            ]
                        )
                    )
                    mean_probability_changes.append(
                        float(
                            row[
                                "mean_probability_difference_from_largest_particle_count"
                            ]
                        )
                    )
            maximum_observed_count_change = float(max(count_changes))
            maximum_observed_probability_change = float(max(probability_changes))
            maximum_observed_mean_probability_change = float(
                max(mean_probability_changes)
            )
            passed = bool(
                seed_range <= maximum_seed_range
                and maximum_observed_count_change <= maximum_count_change
                and maximum_observed_probability_change
                <= maximum_probability_change
                and maximum_observed_mean_probability_change
                <= maximum_mean_probability_change
            )
            details.append(
                {
                    "subject_id": subject_id,
                    "model_id": model_id,
                    "second_largest_particle_count": second_largest,
                    "largest_particle_count": largest,
                    "integration_mode": str(group[0].get("integration_mode")),
                    "largest_count_seed_nll_range": seed_range,
                    "maximum_top_two_count_nll_change": maximum_observed_count_change,
                    "maximum_top_two_probability_difference": maximum_observed_probability_change,
                    "maximum_top_two_mean_probability_difference": maximum_observed_mean_probability_change,
                    "passed": passed,
                }
            )
    return {
        "thresholds": {
            "maximum_largest_count_seed_nll_range": maximum_seed_range,
            "maximum_top_two_count_nll_change": maximum_count_change,
            "maximum_top_two_probability_difference": maximum_probability_change,
            "maximum_top_two_mean_probability_difference": maximum_mean_probability_change,
        },
        "details": details,
        "passed": bool(details and all(item["passed"] for item in details)),
    }


def _optimization_smoke(
    arrays: Mapping[str, np.ndarray],
    prior: np.ndarray,
    kernels,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    smoke = config["optimization_smoke"]
    fits = []
    parent = None
    for model_id in FA_MODEL_IDS:
        extra = None if parent is None else [nested_child_start(parent, model_id)]
        fit = fit_model0804(
            arrays["q"],
            arrays["choice"],
            arrays["feedback"],
            prior,
            kernels,
            arrays["holdout"] == 0,
            model_id=model_id,
            capacity=int(config["architecture"]["capacity"]),
            memory_id=str(config["models"]["memory"]),
            particle_count=int(smoke["particle_count"]),
            filter_seed=int(config["preflight"]["independent_filter_seeds"][0]),
            transition_proposals_per_particle=int(
                smoke.get(
                    "transition_proposals_per_particle",
                    _transition_proposals(config, model_id),
                )
            ),
            fa0_exact_initial_sets=_fa0_exact(config, model_id),
            fa0_maximum_exact_initial_sets=_fa0_maximum_exact_sets(config),
            n_starts=int(smoke["n_starts"]),
            maxiter=int(smoke["max_iterations"]),
            extra_starts=extra,
        )
        fits.append(
            {
                "model_id": model_id,
                "train_nll": fit.train_nll,
                "parameters": fit.reported_parameters,
                "diagnostics": fit.diagnostics,
            }
        )
        parent = fit
    return fits


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = args.config.resolve()
    output = args.output.resolve()
    report_path = output / "preflight_report.json"
    if report_path.exists() and not args.force:
        raise FileExistsError(f"preflight output already exists: {report_path}; use --force")
    config = load_config(config_path)
    configured_subjects = [int(value) for value in config["preflight"]["pilot_subjects"]]
    if args.subjects:
        requested = [int(value.strip()) for value in args.subjects.split(",") if value.strip()]
    else:
        requested = configured_subjects
    frame, subjects, input_audit = validate_and_load_inputs(config, set(requested))
    priors, kernels_by_prior, geometry_audit = build_frozen_geometry(config)
    prior_id = str(config["rule_space"]["primary_prior"])
    prior = priors[prior_id]
    kernels = kernels_by_prior[prior_id]
    cache_audit = {
        subject_id: validate_subject_cache(config, frame, subject_id)
        for subject_id in subjects
    }
    arrays_by_subject = {
        subject_id: _load_subject_arrays(cache_audit[subject_id], args.max_trials)
        for subject_id in subjects
    }

    exact_check = _synthetic_exact_check(config)
    first_arrays = arrays_by_subject[subjects[0]]
    structural_arrays = _prefix_arrays(
        first_arrays,
        int(config["preflight"].get("structural_check_trials", 16)),
    )
    endpoint_check = _fullset_endpoint_check(
        structural_arrays, prior, kernels, config
    )
    g_zero_check = _g_zero_check(structural_arrays, prior, kernels, config)
    feedback_order_check = _current_feedback_order_check(
        structural_arrays, prior, kernels, config
    )
    stability = []
    for subject_id in subjects:
        print(f"[preflight] particle stability subject={subject_id}", flush=True)
        stability.extend(
            _particle_stability(
                subject_id,
                arrays_by_subject[subject_id],
                prior,
                kernels,
                config,
            )
        )
    optimization = None
    if args.fit_smoke:
        print(f"[preflight] optimization smoke subject={subjects[0]}", flush=True)
        optimization = _optimization_smoke(
            first_arrays, prior, kernels, config
        )

    sync_tolerance = float(config["numerics"]["synchronization_tolerance"])
    stability_sync_passed = all(
        float(row["maximum_memory_sync_error"]) <= sync_tolerance
        for row in stability
    )
    quantitative_stability = _evaluate_particle_stability(stability, config)
    gates = {
        "exact_small_space_particle_agreement": bool(exact_check["passed"]),
        "fullset_FA0_matches_FS_H0": bool(endpoint_check["passed"]),
        "FA1_matches_FA2_at_g_zero": bool(g_zero_check["passed"]),
        "no_current_feedback_leakage": bool(feedback_order_check["passed"]),
        "memory_synchronization": bool(stability_sync_passed),
        "particle_stability_report_complete": bool(len(stability) > 0),
        "particle_stability_quantitative": bool(
            quantitative_stability["passed"]
        ),
    }
    gates["passed"] = bool(all(gates.values()))
    full_pilot_trajectories = args.max_trials is None
    formal_run_ready = bool(gates["passed"] and full_pilot_trajectories)
    if args.max_trials is not None:
        status = (
            "truncated_smoke_passed_formal_run_blocked"
            if gates["passed"]
            else "truncated_smoke_failed_formal_run_blocked"
        )
    else:
        status = "preflight_passed" if formal_run_ready else "preflight_failed"
    payload = {
        "analysis_id": str(config["analysis_id"]),
        "status": status,
        "scope": "implementation_preflight_not_formal_cognitive_inference",
        "formal_run_ready": formal_run_ready,
        "full_pilot_trajectories": full_pilot_trajectories,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "implementation_sha256": {
            "model_0804.py": _sha256(
                ROOT / "src/Bayesian_state/utils/model_0804.py"
            ),
            "preflight_runner": _sha256(Path(__file__).resolve()),
            "model_0804_tests": _sha256(ROOT / "tests/test_model_0804.py"),
        },
        "runtime_seconds": float(time.time() - started),
        "max_trials": None if args.max_trials is None else int(args.max_trials),
        "subjects": subjects,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "input_audit": input_audit,
        "cache_audit": cache_audit,
        "geometry_audit": geometry_audit,
        "exact_check": exact_check,
        "fullset_endpoint_check": endpoint_check,
        "g_zero_nesting_check": g_zero_check,
        "current_feedback_order_check": feedback_order_check,
        "particle_stability": stability,
        "quantitative_particle_stability": quantitative_stability,
        "optimization_smoke": optimization,
        "gates": gates,
    }
    _atomic_json(report_path, payload)
    print(
        f"PREFLIGHT status={payload['status']} output={report_path} "
        f"runtime={payload['runtime_seconds']:.1f}s",
        flush=True,
    )
    if not gates["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
