#!/usr/bin/env python3
"""Diagnose full-history particle collapse and choice-adapted proposals.

Each subject/model/seed/RxB run is checkpointed independently.  The script
saves complete trial-wise ESS, resampling, marginal predictions, and latent
summaries, then compares every setting with a configured high-budget
reference.  It performs no parameter optimization and no model selection.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0803_cond1 import (  # noqa: E402
    build_frozen_geometry,
    load_config,
    validate_and_load_inputs,
    validate_subject_cache,
)
from scripts.run_model_0804_cond1_preflight import (  # noqa: E402
    _load_subject_arrays,
    _parameters,
    _resample_threshold,
    _segment_scores,
)
from src.Bayesian_state.manuscript_models.model_0804 import (  # noqa: E402
    run_model0804_alive_particle_filter,
    run_model0804_particle_filter,
    run_model0804_resample_move_particle_filter,
)


DEFAULT_CONFIG = ROOT / "configs/model_0804_cond1_preflight.yaml"
DEFAULT_OUTPUT = ROOT / "results/zhuran/model_0804_cond1/adaptation_20260804_v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--settings", type=str, default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--lapse", type=float, default=None)
    parser.add_argument(
        "--filter-method",
        choices=("bootstrap", "alive", "resample_move"),
        default=None,
    )
    parser.add_argument("--alive-batch-size", type=int, default=8192)
    parser.add_argument(
        "--maximum-alive-attempts", type=int, default=100_000_000
    )
    parser.add_argument("--reference-setting", type=str, default=None)
    parser.add_argument("--rejuvenation-window", type=int, default=None)
    parser.add_argument("--rejuvenation-sweeps", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _csv_values(value: str | None, default: list[Any], cast) -> list[Any]:
    if value is None:
        return list(default)
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


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


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _run_paths(
    output: Path,
    subject_id: int,
    model_id: str,
    setting_id: str,
    seed: int,
) -> tuple[Path, Path]:
    directory = (
        output
        / f"subject_{int(subject_id)}"
        / str(model_id)
        / str(setting_id)
        / f"seed_{int(seed)}"
    )
    return directory / "summary.json", directory / "trial_trace.npz"


def _collapse_trials(
    trace,
    choices: np.ndarray,
    holdout: np.ndarray,
    count: int,
) -> list[dict[str, Any]]:
    if trace.alive_incremental_likelihood is not None:
        order = np.argsort(trace.alive_incremental_likelihood)[: int(count)]
    else:
        order = np.argsort(trace.post_choice_ess)[: int(count)]
    records = []
    for trial_index in order:
        observed_probability = float(
            trace.probabilities[int(trial_index), choices[int(trial_index)]]
        )
        records.append(
            {
                "trial_zero_based": int(trial_index),
                "post_choice_ess": float(trace.post_choice_ess[trial_index]),
                "post_choice_ess_fraction": float(
                    trace.post_choice_ess[trial_index] / trace.particle_count
                ),
                "pre_choice_ess": float(trace.pre_choice_ess[trial_index]),
                "observed_choice_probability": observed_probability,
                "choice": int(choices[trial_index]),
                "holdout": bool(holdout[trial_index]),
                "resampled": bool(trace.resampled[trial_index]),
                "unique_ancestors_after_resampling": int(
                    trace.resampling_unique_ancestors[trial_index]
                ),
            }
        )
    return records


def _trace_summary(
    trace,
    arrays: Mapping[str, np.ndarray],
    *,
    subject_id: int,
    model_id: str,
    setting: Mapping[str, Any],
    seed: int,
    runtime_seconds: float,
    config: Mapping[str, Any],
    parameters,
) -> dict[str, Any]:
    diagnostic = config["adaptation_diagnostic"]
    fractions = [float(value) for value in diagnostic["ess_fraction_diagnostics"]]
    ess_fraction = trace.post_choice_ess / float(trace.particle_count)
    return {
        "subject_id": int(subject_id),
        "model_id": str(model_id),
        "setting_id": str(setting["id"]),
        "filter_seed": int(seed),
        "particle_count": int(trace.particle_count),
        "proposals_per_parent": int(trace.transition_proposals_per_particle),
        "replacement_count_stratified": bool(
            trace.replacement_count_stratified
        ),
        "inference_method": str(trace.inference_method),
        "total_transition_candidates_per_trial": int(
            trace.particle_count * trace.transition_proposals_per_particle
        ),
        "n_trials": int(len(arrays["choice"])),
        "runtime_seconds": float(runtime_seconds),
        "nll": float(trace.nll),
        "fixed_parameters": {
            "gamma": float(parameters.gamma),
            "w0": float(parameters.w0),
            "kappa": float(parameters.kappa),
            "m": float(parameters.m),
            "g": float(parameters.g),
            "lapse": float(parameters.lapse),
        },
        "segments": _segment_scores(
            trace.probabilities, arrays["choice"], arrays["holdout"]
        ),
        "minimum_pre_choice_ess": float(np.min(trace.pre_choice_ess)),
        "minimum_post_choice_ess": float(np.min(trace.post_choice_ess)),
        "minimum_post_choice_ess_fraction": float(np.min(ess_fraction)),
        "post_choice_ess_fraction_quantiles": {
            str(quantile): float(np.quantile(ess_fraction, quantile))
            for quantile in (0.0, 0.01, 0.05, 0.10, 0.50)
        },
        "trials_below_ess_fraction": {
            str(fraction): int(np.sum(ess_fraction < fraction))
            for fraction in fractions
        },
        "resampling_count": int(np.sum(trace.resampled)),
        "minimum_unique_ancestors_on_resampling": int(
            np.min(trace.resampling_unique_ancestors[trace.resampled])
            if np.any(trace.resampled)
            else trace.particle_count
        ),
        "maximum_memory_sync_error": float(np.max(trace.memory_sync_error)),
        "collapse_trials": _collapse_trials(
            trace,
            arrays["choice"],
            arrays["holdout"],
            int(diagnostic["collapse_report_trials"]),
        ),
        "alive_diagnostics": (
            None
            if trace.alive_attempt_count is None
            else {
                "maximum_attempt_count": int(
                    np.max(trace.alive_attempt_count)
                ),
                "total_attempt_count": int(
                    np.sum(trace.alive_attempt_count)
                ),
                "minimum_incremental_likelihood": float(
                    np.min(trace.alive_incremental_likelihood)
                ),
                "maximum_attempts_per_retained_particle": float(
                    np.max(trace.alive_attempt_count) / trace.particle_count
                ),
                "minimum_unique_parent_count": int(
                    np.min(trace.resampling_unique_ancestors)
                ),
            }
        ),
        "rejuvenation_diagnostics": (
            None
            if trace.rejuvenation_acceptance_rate is None
            else {
                "window": int(trace.rejuvenation_window),
                "sweeps": int(trace.rejuvenation_sweeps),
                "mean_acceptance_rate": float(
                    np.mean(trace.rejuvenation_acceptance_rate)
                ),
                "minimum_acceptance_rate": float(
                    np.min(trace.rejuvenation_acceptance_rate)
                ),
                "acceptance_rate_quantiles": {
                    str(value): float(
                        np.quantile(trace.rejuvenation_acceptance_rate, value)
                    )
                    for value in (0.0, 0.05, 0.50, 0.95, 1.0)
                },
                "minimum_unique_active_sets": int(
                    np.min(trace.rejuvenation_unique_active_sets)
                ),
                "median_unique_active_sets": float(
                    np.median(trace.rejuvenation_unique_active_sets)
                ),
            }
        ),
    }


def _save_trace(path: Path, trace, arrays: Mapping[str, np.ndarray]) -> None:
    payload = dict(
        choices=np.asarray(arrays["choice"], dtype=np.int8),
        feedback=np.asarray(arrays["feedback"], dtype=np.int8),
        holdout=np.asarray(arrays["holdout"], dtype=bool),
        probabilities=trace.probabilities,
        marginal_hypothesis_prior=trace.marginal_hypothesis_prior,
        marginal_active_probability=trace.marginal_active_probability,
        predictive_replacement_count=trace.predictive_replacement_count,
        predictive_replacement_fraction=trace.predictive_replacement_fraction,
        predictive_removed_mass=trace.predictive_removed_mass,
        predictive_newcomer_distance=trace.predictive_newcomer_distance,
        pre_choice_ess=trace.pre_choice_ess,
        post_choice_ess=trace.post_choice_ess,
        resampled=trace.resampled,
        resampling_unique_ancestors=trace.resampling_unique_ancestors,
        memory_sync_error=trace.memory_sync_error,
    )
    if trace.alive_attempt_count is not None:
        payload["alive_attempt_count"] = trace.alive_attempt_count
        payload["alive_incremental_likelihood"] = (
            trace.alive_incremental_likelihood
        )
    if trace.rejuvenation_acceptance_rate is not None:
        payload["rejuvenation_acceptance_rate"] = (
            trace.rejuvenation_acceptance_rate
        )
        payload["rejuvenation_unique_active_sets"] = (
            trace.rejuvenation_unique_active_sets
        )
    _atomic_npz(path, **payload)


def _comparison_rows(
    rows: list[dict[str, Any]],
    trace_paths: Mapping[tuple[int, str, str, int], Path],
    reference_setting: str,
) -> list[dict[str, Any]]:
    comparisons = []
    for row in rows:
        key = (
            int(row["subject_id"]),
            str(row["model_id"]),
            str(row["setting_id"]),
            int(row["filter_seed"]),
        )
        reference_key = (key[0], key[1], str(reference_setting), key[3])
        if reference_key not in trace_paths:
            continue
        with np.load(trace_paths[key], allow_pickle=False) as payload:
            probabilities = payload["probabilities"]
        with np.load(trace_paths[reference_key], allow_pickle=False) as payload:
            reference_probabilities = payload["probabilities"]
        reference_rows = [
            item
            for item in rows
            if int(item["subject_id"]) == key[0]
            and str(item["model_id"]) == key[1]
            and str(item["setting_id"]) == str(reference_setting)
            and int(item["filter_seed"]) == key[3]
        ]
        if len(reference_rows) != 1:
            raise ValueError("missing unique adaptation reference row")
        difference = np.abs(probabilities - reference_probabilities)
        comparisons.append(
            {
                "subject_id": key[0],
                "model_id": key[1],
                "filter_seed": key[3],
                "setting_id": key[2],
                "reference_setting_id": str(reference_setting),
                "nll_difference_from_reference": float(
                    row["nll"] - reference_rows[0]["nll"]
                ),
                "maximum_probability_difference_from_reference": float(
                    np.max(difference)
                ),
                "mean_probability_difference_from_reference": float(
                    np.mean(difference)
                ),
            }
        )
    return comparisons


def _seed_stability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    keys = sorted(
        {
            (int(row["subject_id"]), str(row["model_id"]), str(row["setting_id"]))
            for row in rows
        }
    )
    for subject_id, model_id, setting_id in keys:
        selected = [
            row
            for row in rows
            if int(row["subject_id"]) == subject_id
            and str(row["model_id"]) == model_id
            and str(row["setting_id"]) == setting_id
        ]
        if len(selected) < 2:
            continue
        nll_values = [float(row["nll"]) for row in selected]
        output.append(
            {
                "subject_id": subject_id,
                "model_id": model_id,
                "setting_id": setting_id,
                "seeds": [int(row["filter_seed"]) for row in selected],
                "nll_range": float(max(nll_values) - min(nll_values)),
                "minimum_post_choice_ess": float(
                    min(float(row["minimum_post_choice_ess"]) for row in selected)
                ),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = args.config.resolve()
    output = args.output.resolve()
    config = load_config(config_path)
    diagnostic = config["adaptation_diagnostic"]
    subjects_requested = _csv_values(
        args.subjects, [int(value) for value in diagnostic["pilot_subjects"]], int
    )
    models = _csv_values(args.models, list(diagnostic["models"]), str)
    seeds = _csv_values(
        args.seeds, [int(value) for value in diagnostic["filter_seeds"]], int
    )
    all_settings = {str(item["id"]): item for item in diagnostic["settings"]}
    setting_ids = _csv_values(args.settings, list(all_settings), str)
    missing = sorted(set(setting_ids) - set(all_settings))
    if missing:
        raise ValueError(f"unknown adaptation settings: {missing}")
    settings = [all_settings[setting_id] for setting_id in setting_ids]
    invalid_models = sorted(set(models) - {"FA1", "FA2"})
    if invalid_models:
        raise ValueError(f"adaptation diagnostic only supports FA1/FA2: {invalid_models}")

    frame, subjects, input_audit = validate_and_load_inputs(
        config, set(subjects_requested)
    )
    priors, kernels_by_prior, geometry_audit = build_frozen_geometry(config)
    prior_id = str(config["rule_space"]["primary_prior"])
    prior = priors[prior_id]
    kernels = kernels_by_prior[prior_id]
    cache_audit = {
        subject_id: validate_subject_cache(config, frame, subject_id)
        for subject_id in subjects
    }
    arrays_by_subject = {
        subject_id: _load_subject_arrays(
            cache_audit[subject_id], args.max_trials
        )
        for subject_id in subjects
    }

    rows: list[dict[str, Any]] = []
    trace_paths: dict[tuple[int, str, str, int], Path] = {}
    for subject_id in subjects:
        arrays = arrays_by_subject[subject_id]
        for model_id in models:
            parameters = _parameters(config, model_id)
            if args.kappa is not None:
                if not np.isfinite(args.kappa) or float(args.kappa) <= 0.0:
                    raise ValueError("--kappa must be finite and positive")
                parameters = replace(parameters, kappa=float(args.kappa))
            if args.lapse is not None:
                if (
                    not np.isfinite(args.lapse)
                    or float(args.lapse) < 0.0
                    or float(args.lapse) >= 1.0
                ):
                    raise ValueError("--lapse must lie in [0, 1)")
                parameters = replace(parameters, lapse=float(args.lapse))
            for setting in settings:
                setting_id = str(setting["id"])
                filter_method = str(
                    args.filter_method
                    or setting.get("filter_method", "bootstrap")
                )
                for seed in seeds:
                    summary_path, trace_path = _run_paths(
                        output, subject_id, model_id, setting_id, seed
                    )
                    key = (int(subject_id), model_id, setting_id, int(seed))
                    trace_paths[key] = trace_path
                    if summary_path.exists() and trace_path.exists() and not args.force:
                        with summary_path.open("r", encoding="utf-8") as stream:
                            summary = json.load(stream)
                        print(
                            f"[adaptation] skip completed subject={subject_id} "
                            f"model={model_id} setting={setting_id} seed={seed}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[adaptation] run subject={subject_id} model={model_id} "
                            f"setting={setting_id} seed={seed}",
                            flush=True,
                        )
                        run_started = time.time()
                        common = dict(
                            q_values=arrays["q"],
                            choices=arrays["choice"],
                            feedback=arrays["feedback"],
                            prior=prior,
                            kernels=kernels,
                            model_id=model_id,
                            parameters=parameters,
                            capacity=int(config["architecture"]["capacity"]),
                            particle_count=int(setting["particle_count"]),
                            filter_seed=int(seed),
                        )
                        if filter_method == "alive":
                            trace = run_model0804_alive_particle_filter(
                                **common,
                                alive_batch_size=int(args.alive_batch_size),
                                maximum_attempts_per_trial=int(
                                    args.maximum_alive_attempts
                                ),
                            )
                        elif filter_method == "resample_move":
                            trace = run_model0804_resample_move_particle_filter(
                                **common,
                                rejuvenation_window=int(
                                    args.rejuvenation_window
                                    if args.rejuvenation_window is not None
                                    else setting.get("rejuvenation_window", 4)
                                ),
                                rejuvenation_sweeps=int(
                                    args.rejuvenation_sweeps
                                    if args.rejuvenation_sweeps is not None
                                    else setting.get("rejuvenation_sweeps", 1)
                                ),
                            )
                        else:
                            trace = run_model0804_particle_filter(
                                **common,
                                resample_threshold_fraction=_resample_threshold(
                                    config, model_id
                                ),
                                transition_proposals_per_particle=int(
                                    setting["proposals_per_parent"]
                                ),
                                stratify_replacement_count=bool(
                                    setting.get(
                                        "stratify_replacement_count", False
                                    )
                                ),
                            )
                        runtime = float(time.time() - run_started)
                        summary = _trace_summary(
                            trace,
                            arrays,
                            subject_id=subject_id,
                            model_id=model_id,
                            setting=setting,
                            seed=seed,
                            runtime_seconds=runtime,
                            config=config,
                            parameters=parameters,
                        )
                        _save_trace(trace_path, trace, arrays)
                        _atomic_json(summary_path, summary)
                        print(
                            f"[adaptation] done setting={setting_id} "
                            f"runtime={runtime:.1f}s nll={trace.nll:.4f} "
                            f"min_ess={np.min(trace.post_choice_ess):.1f}",
                            flush=True,
                        )
                    rows.append(summary)
                    progress = {
                        "analysis_id": str(config["analysis_id"]),
                        "scope": "adaptation_diagnostic_not_model_comparison",
                        "rows": rows,
                        "completed_runs": len(rows),
                    }
                    _atomic_json(output / "diagnostic_progress.json", progress)

    reference_setting = str(
        args.reference_setting or diagnostic["comparison_reference_setting"]
    )
    comparisons = _comparison_rows(
        rows, trace_paths, reference_setting=reference_setting
    )
    payload = {
        "analysis_id": str(config["analysis_id"]),
        "status": "adaptation_diagnostic_complete",
        "scope": "adaptation_diagnostic_not_model_comparison",
        "runtime_seconds": float(time.time() - started),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "implementation_sha256": {
            "model_0804.py": _sha256(
                ROOT / "src/Bayesian_state/manuscript_models/model_0804.py"
            ),
            "diagnostic_runner": _sha256(Path(__file__).resolve()),
            "model_0804_tests": _sha256(ROOT / "tests/test_model_0804.py"),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "max_trials": None if args.max_trials is None else int(args.max_trials),
        "kappa_override": None if args.kappa is None else float(args.kappa),
        "lapse_override": None if args.lapse is None else float(args.lapse),
        "subjects": subjects,
        "models": models,
        "seeds": seeds,
        "settings": settings,
        "input_audit": input_audit,
        "cache_audit": cache_audit,
        "geometry_audit": geometry_audit,
        "runs": rows,
        "comparisons_with_reference": comparisons,
        "seed_stability": _seed_stability(rows),
    }
    report_path = output / "adaptation_report.json"
    _atomic_json(report_path, payload)
    print(
        f"ADAPTATION status={payload['status']} output={report_path} "
        f"runtime={payload['runtime_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
