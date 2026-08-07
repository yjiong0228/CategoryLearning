#!/usr/bin/env python3
"""Run the condition-1 PGAS latent-history diagnostic.

This runner samples active-hypothesis-set histories at frozen parameters.  It
does not optimize parameters, compare models, or estimate a marginal
likelihood.  Full-suffix ancestor sampling is exact; a finite lookahead is an
explicit non-Markovian approximation used only for the long-sequence pilot.
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
)
from src.Bayesian_state.manuscript_models.model_0804_pgas import (  # noqa: E402
    run_model0804_pgas,
)


DEFAULT_CONFIG = ROOT / "configs/model_0804_cond1_preflight.yaml"
DEFAULT_OUTPUT = ROOT / "results/zhuran/model_0804_cond1/pgas_20260804_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--particles", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--burn-in", type=int, default=None)
    parser.add_argument("--thin", type=int, default=None)
    parser.add_argument(
        "--lookahead",
        type=str,
        default=None,
        help="'full' for exact full suffix, otherwise a positive integer",
    )
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--kappa", type=float, default=None)
    parser.add_argument("--lapse", type=float, default=None)
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


def _lookahead(value: str | None, default: Any) -> int | None:
    selected = default if value is None else value
    if selected is None or str(selected).lower() == "full":
        return None
    lookahead = int(selected)
    if lookahead < 1:
        raise ValueError("PGAS lookahead must be positive or 'full'")
    return lookahead


def _run_paths(
    output: Path, subject_id: int, model_id: str, seed: int
) -> tuple[Path, Path]:
    directory = (
        output
        / f"subject_{int(subject_id)}"
        / str(model_id)
        / f"seed_{int(seed)}"
    )
    return directory / "summary.json", directory / "posterior_trace.npz"


def _summary(
    trace,
    *,
    subject_id: int,
    model_id: str,
    seed: int,
    runtime_seconds: float,
    parameters,
) -> dict[str, Any]:
    burn = int(trace.burn_in)
    active_paths = trace.retained_active_samples.reshape(
        trace.retained_samples, -1
    )
    unique_paths = np.unique(
        np.packbits(active_paths, axis=1, bitorder="little"), axis=0
    ).shape[0]
    mean_trial_change = np.mean(
        trace.iteration_trial_active_change_fraction[burn:], axis=0
    )
    mean_trial_ancestor_switch = np.mean(
        trace.iteration_trial_ancestor_switched[burn:], axis=0
    )
    median_trial_ancestor_ess = np.median(
        trace.iteration_trial_ancestor_ess[burn:], axis=0
    )
    split = max(1, int(trace.active_probability.shape[0]) // 4)
    return {
        "subject_id": int(subject_id),
        "model_id": str(model_id),
        "chain_seed": int(seed),
        "n_trials": int(trace.active_probability.shape[0]),
        "n_hypotheses": int(trace.active_probability.shape[1]),
        "particle_count": int(trace.particle_count),
        "iterations": int(trace.iterations),
        "burn_in": burn,
        "thin": int(trace.thin),
        "retained_samples": int(trace.retained_samples),
        "ancestor_lookahead": (
            "full" if trace.ancestor_lookahead is None else int(trace.ancestor_lookahead)
        ),
        "ancestor_sampling_exact": trace.ancestor_lookahead is None,
        "runtime_seconds": float(runtime_seconds),
        "normalizing_constant_estimated": bool(
            trace.normalizing_constant_estimated
        ),
        "fixed_parameters": {
            "gamma": float(parameters.gamma),
            "w0": float(parameters.w0),
            "kappa": float(parameters.kappa),
            "m": float(parameters.m),
            "g": float(parameters.g),
            "lapse": float(parameters.lapse),
        },
        "posterior_path_log_choice_likelihood": {
            "mean": float(np.mean(trace.iteration_log_choice_likelihood[burn:])),
            "sd": float(np.std(trace.iteration_log_choice_likelihood[burn:], ddof=1)),
            "minimum": float(np.min(trace.iteration_log_choice_likelihood[burn:])),
            "maximum": float(np.max(trace.iteration_log_choice_likelihood[burn:])),
        },
        "mixing": {
            "mean_path_change_fraction": float(
                np.mean(trace.iteration_path_change_fraction[burn:])
            ),
            "zero_path_change_fraction": float(
                np.mean(trace.iteration_path_change_fraction[burn:] == 0.0)
            ),
            "mean_reference_ancestor_switch_fraction": float(
                np.mean(trace.iteration_ancestor_switch_fraction[burn:])
            ),
            "minimum_ancestor_ess": float(
                np.min(trace.iteration_minimum_ancestor_ess[burn:])
            ),
            "median_minimum_ancestor_ess": float(
                np.median(trace.iteration_minimum_ancestor_ess[burn:])
            ),
            "unique_retained_active_paths": int(unique_paths),
            "mean_active_change_first_quarter": float(
                np.mean(mean_trial_change[:split])
            ),
            "mean_active_change_last_quarter": float(
                np.mean(mean_trial_change[-split:])
            ),
            "trials_never_changed_after_burn_in": int(
                np.sum(mean_trial_change == 0.0)
            ),
            "mean_reference_ancestor_switch_first_quarter": float(
                np.mean(mean_trial_ancestor_switch[:split])
            ),
            "mean_reference_ancestor_switch_last_quarter": float(
                np.mean(mean_trial_ancestor_switch[-split:])
            ),
            "median_ancestor_ess_first_quarter": float(
                np.median(median_trial_ancestor_ess[:split])
            ),
            "median_ancestor_ess_last_quarter": float(
                np.median(median_trial_ancestor_ess[-split:])
            ),
        },
        "posterior_checks": {
            "minimum_active_probability": float(
                np.min(trace.active_probability)
            ),
            "maximum_active_probability": float(
                np.max(trace.active_probability)
            ),
            "maximum_active_mass_error": float(
                np.max(
                    np.abs(
                        trace.active_probability.sum(axis=1)
                        - trace.retained_active_samples.sum(axis=2)[0]
                    )
                )
            ),
            "mean_expected_replacement_count": float(
                np.mean(trace.expected_replacement_count[1:])
            ),
        },
    }


def _save_trace(path: Path, trace, arrays: Mapping[str, np.ndarray]) -> None:
    _atomic_npz(
        path,
        choices=np.asarray(arrays["choice"], dtype=np.int8),
        feedback=np.asarray(arrays["feedback"], dtype=np.int8),
        holdout=np.asarray(arrays["holdout"], dtype=bool),
        active_probability=trace.active_probability,
        expected_replacement_count=trace.expected_replacement_count,
        retained_active_samples=trace.retained_active_samples,
        retained_replacement_samples=trace.retained_replacement_samples,
        iteration_log_choice_likelihood=trace.iteration_log_choice_likelihood,
        iteration_path_change_fraction=trace.iteration_path_change_fraction,
        iteration_ancestor_switch_fraction=(
            trace.iteration_ancestor_switch_fraction
        ),
        iteration_minimum_ancestor_ess=(
            trace.iteration_minimum_ancestor_ess
        ),
        iteration_trial_active_change_fraction=(
            trace.iteration_trial_active_change_fraction
        ),
        iteration_trial_ancestor_switched=(
            trace.iteration_trial_ancestor_switched
        ),
        iteration_trial_ancestor_ess=trace.iteration_trial_ancestor_ess,
    )


def _chain_comparisons(
    rows: list[dict[str, Any]],
    trace_paths: Mapping[tuple[int, str, int], Path],
) -> list[dict[str, Any]]:
    output = []
    keys = sorted({(int(row["subject_id"]), str(row["model_id"])) for row in rows})
    for subject_id, model_id in keys:
        selected = sorted(
            [
                row
                for row in rows
                if int(row["subject_id"]) == subject_id
                and str(row["model_id"]) == model_id
            ],
            key=lambda row: int(row["chain_seed"]),
        )
        for left_index in range(len(selected)):
            for right_index in range(left_index + 1, len(selected)):
                left = selected[left_index]
                right = selected[right_index]
                left_seed = int(left["chain_seed"])
                right_seed = int(right["chain_seed"])
                with np.load(
                    trace_paths[(subject_id, model_id, left_seed)],
                    allow_pickle=False,
                ) as payload:
                    left_active = payload["active_probability"]
                    left_replacement = payload["expected_replacement_count"]
                with np.load(
                    trace_paths[(subject_id, model_id, right_seed)],
                    allow_pickle=False,
                ) as payload:
                    right_active = payload["active_probability"]
                    right_replacement = payload["expected_replacement_count"]
                active_difference = np.abs(left_active - right_active)
                replacement_difference = np.abs(
                    left_replacement - right_replacement
                )
                output.append(
                    {
                        "subject_id": subject_id,
                        "model_id": model_id,
                        "left_seed": left_seed,
                        "right_seed": right_seed,
                        "maximum_active_probability_difference": float(
                            np.max(active_difference)
                        ),
                        "mean_active_probability_difference": float(
                            np.mean(active_difference)
                        ),
                        "maximum_expected_replacement_difference": float(
                            np.max(replacement_difference)
                        ),
                        "mean_expected_replacement_difference": float(
                            np.mean(replacement_difference)
                        ),
                        "posterior_path_loglikelihood_mean_difference": float(
                            abs(
                                left["posterior_path_log_choice_likelihood"]["mean"]
                                - right["posterior_path_log_choice_likelihood"]["mean"]
                            )
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
    diagnostic = config["pgas_diagnostic"]
    subjects_requested = _csv_values(
        args.subjects,
        [int(value) for value in diagnostic["pilot_subjects"]],
        int,
    )
    models = _csv_values(args.models, list(diagnostic["models"]), str)
    seeds = _csv_values(
        args.seeds,
        [int(value) for value in diagnostic["chain_seeds"]],
        int,
    )
    if sorted(set(models) - {"FA1", "FA2"}):
        raise ValueError("PGAS diagnostic only supports dynamic FA1/FA2")
    particle_count = int(args.particles or diagnostic["particle_count"])
    iterations = int(args.iterations or diagnostic["iterations"])
    burn_in = int(
        diagnostic["burn_in"] if args.burn_in is None else args.burn_in
    )
    thin = int(diagnostic["thin"] if args.thin is None else args.thin)
    lookahead = _lookahead(args.lookahead, diagnostic["ancestor_lookahead"])

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
    trace_paths: dict[tuple[int, str, int], Path] = {}
    for subject_id in subjects:
        arrays = arrays_by_subject[subject_id]
        for model_id in models:
            parameters = _parameters(config, model_id)
            if args.kappa is not None:
                if not np.isfinite(args.kappa) or float(args.kappa) <= 0.0:
                    raise ValueError("--kappa must be finite and positive")
                parameters = replace(parameters, kappa=float(args.kappa))
            if args.lapse is not None:
                if not 0.0 <= float(args.lapse) < 1.0:
                    raise ValueError("--lapse must lie in [0, 1)")
                parameters = replace(parameters, lapse=float(args.lapse))
            for seed in seeds:
                summary_path, trace_path = _run_paths(
                    output, subject_id, model_id, seed
                )
                trace_paths[(int(subject_id), model_id, int(seed))] = trace_path
                if summary_path.exists() and trace_path.exists() and not args.force:
                    with summary_path.open("r", encoding="utf-8") as stream:
                        summary = json.load(stream)
                    print(
                        f"[pgas] skip subject={subject_id} model={model_id} seed={seed}",
                        flush=True,
                    )
                else:
                    exact_label = "full-exact" if lookahead is None else f"L{lookahead}-approx"
                    print(
                        f"[pgas] run subject={subject_id} model={model_id} "
                        f"seed={seed} N={particle_count} iterations={iterations} "
                        f"ancestor={exact_label}",
                        flush=True,
                    )
                    run_started = time.time()
                    trace = run_model0804_pgas(
                        arrays["q"],
                        arrays["choice"],
                        arrays["feedback"],
                        prior,
                        kernels,
                        model_id=model_id,
                        parameters=parameters,
                        capacity=int(config["architecture"]["capacity"]),
                        particle_count=particle_count,
                        iterations=iterations,
                        burn_in=burn_in,
                        thin=thin,
                        ancestor_lookahead=lookahead,
                        chain_seed=int(seed),
                    )
                    runtime = float(time.time() - run_started)
                    summary = _summary(
                        trace,
                        subject_id=subject_id,
                        model_id=model_id,
                        seed=seed,
                        runtime_seconds=runtime,
                        parameters=parameters,
                    )
                    _save_trace(trace_path, trace, arrays)
                    _atomic_json(summary_path, summary)
                    print(
                        f"[pgas] done runtime={runtime:.1f}s "
                        f"change={summary['mixing']['mean_path_change_fraction']:.4f} "
                        f"zero={summary['mixing']['zero_path_change_fraction']:.3f}",
                        flush=True,
                    )
                rows.append(summary)
                _atomic_json(
                    output / "diagnostic_progress.json",
                    {
                        "analysis_id": str(config["analysis_id"]),
                        "scope": "posterior_path_diagnostic_not_likelihood_estimator",
                        "completed_runs": len(rows),
                        "runs": rows,
                    },
                )

    payload = {
        "analysis_id": str(config["analysis_id"]),
        "status": "pgas_diagnostic_complete",
        "scope": "posterior_path_diagnostic_not_likelihood_estimator",
        "normalizing_constant_estimated": False,
        "runtime_seconds": float(time.time() - started),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "implementation_sha256": {
            "model_0804.py": _sha256(
                ROOT / "src/Bayesian_state/manuscript_models/model_0804.py"
            ),
            "model_0804_pgas.py": _sha256(
                ROOT / "src/Bayesian_state/manuscript_models/model_0804_pgas.py"
            ),
            "diagnostic_runner": _sha256(Path(__file__).resolve()),
            "pgas_tests": _sha256(ROOT / "tests/test_model_0804_pgas.py"),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "max_trials": None if args.max_trials is None else int(args.max_trials),
        "subjects": subjects,
        "models": models,
        "chain_seeds": seeds,
        "particle_count": particle_count,
        "iterations": iterations,
        "burn_in": burn_in,
        "thin": thin,
        "ancestor_lookahead": "full" if lookahead is None else lookahead,
        "ancestor_sampling_exact": lookahead is None,
        "input_audit": input_audit,
        "cache_audit": cache_audit,
        "geometry_audit": geometry_audit,
        "runs": rows,
        "chain_comparisons": _chain_comparisons(rows, trace_paths),
        "interpretation_guardrail": (
            "PGAS diagnoses posterior path mixing at frozen parameters; it does "
            "not supply NLL or reopen the formal model-comparison gate."
        ),
    }
    report_path = output / "pgas_report.json"
    _atomic_json(report_path, payload)
    print(
        f"PGAS status={payload['status']} output={report_path} "
        f"runtime={payload['runtime_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
