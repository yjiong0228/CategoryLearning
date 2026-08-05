#!/usr/bin/env python3
"""Run the FA2R behaviour-level mechanism-recovery pilot.

Known FA2R parameters generate new choices on one frozen real q/category
sequence.  A finite candidate grid is then scored by choice-marginal particle
likelihood.  Stage-1 winners, the true grid point, and its rho=0 parent receive
an independent-seed, higher-particle confirmation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
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

from scripts.run_model_0803_cond1 import (  # noqa: E402
    build_frozen_geometry,
    load_config,
    validate_and_load_inputs,
    validate_subject_cache,
)
from scripts.run_model_0804_cond1_preflight import (  # noqa: E402
    _load_subject_arrays,
)
from src.Bayesian_state.utils.model_0804 import (  # noqa: E402
    Model0804Parameters,
    run_model0804_particle_filter,
)
from src.Bayesian_state.utils.model_0804_recovery import (  # noqa: E402
    infer_correct_choices,
    simulate_model0804_choices,
)


DEFAULT_CONFIG = ROOT / "configs/model_0804_regeneration_recovery.yaml"
DEFAULT_OUTPUT = (
    ROOT / "results/zhuran/model_0804_cond1/regeneration_recovery_20260804_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="comma-separated scenario ids; default is the frozen full list",
    )
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument(
        "--subjects",
        type=str,
        default=None,
        help="comma-separated subject ids; default is the frozen config list",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=None,
        help="override generated paths per subject (smoke tests only)",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--stage-particles", type=int, default=None)
    parser.add_argument("--confirmation-particles", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--reconfirm",
        action="store_true",
        help="reuse compatible stage-1 checkpoints and rebuild confirmation",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_seed(base_seed: int, *parts: object) -> int:
    text = ":".join([str(int(base_seed)), *(str(value) for value in parts)])
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32 - 1)


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


def _load_recovery_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("recovery config must contain a mapping")
    return payload


def _candidate_id(candidate: Mapping[str, float]) -> str:
    def encode(value: float) -> str:
        return f"{float(value):.3f}".replace(".", "p")

    return "_".join(
        f"{name}{encode(float(candidate[name]))}"
        for name in ("rho", "m", "g", "lapse")
    )


def _candidate_grid(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    grid = config["candidate_grid"]
    values = {
        name: [float(value) for value in grid[name]]
        for name in ("rho", "m", "g", "lapse")
    }
    candidates = []
    for combination in itertools.product(
        values["rho"], values["m"], values["g"], values["lapse"]
    ):
        candidate = dict(zip(("rho", "m", "g", "lapse"), combination))
        candidates.append({"id": _candidate_id(candidate), **candidate})
    identifiers = [candidate["id"] for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate grid contains duplicate points")
    return candidates


def _matching_candidate_index(
    candidates: Sequence[Mapping[str, Any]],
    target: Mapping[str, Any],
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
        raise ValueError(f"target is not a unique candidate-grid point: {target}")
    return int(matches[0])


def _model_parameters(
    architecture: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Model0804Parameters:
    return Model0804Parameters(
        gamma=float(architecture["gamma"]),
        w0=float(architecture["w0"]),
        kappa=float(architecture["kappa"]),
        m=float(candidate["m"]),
        g=float(candidate["g"]),
        lapse=float(candidate["lapse"]),
        rho=float(candidate["rho"]),
    )


def _ensemble_nll(nll_values: np.ndarray) -> float:
    values = np.asarray(nll_values, dtype=float).reshape(-1)
    if values.size < 1 or np.any(~np.isfinite(values)):
        raise ValueError("ensemble NLL requires finite seed-specific NLLs")
    minimum = float(np.min(values))
    return float(
        minimum - np.log(np.mean(np.exp(-(values - minimum))))
    )


def _score_candidate(
    q_values: np.ndarray,
    choices: np.ndarray,
    feedback: np.ndarray,
    prior: np.ndarray,
    kernels,
    *,
    model_id: str,
    architecture: Mapping[str, Any],
    candidate: Mapping[str, Any],
    particle_count: int,
    filter_seed: int,
    resample_threshold_fraction: float,
) -> tuple[float, dict[str, float]]:
    started = time.time()
    trace = run_model0804_particle_filter(
        q_values,
        choices,
        feedback,
        prior,
        kernels,
        model_id=model_id,
        parameters=_model_parameters(architecture, candidate),
        capacity=int(architecture["capacity"]),
        particle_count=int(particle_count),
        filter_seed=int(filter_seed),
        resample_threshold_fraction=float(resample_threshold_fraction),
    )
    diagnostics = {
        "runtime_seconds": float(time.time() - started),
        "minimum_pre_choice_ess": float(np.min(trace.pre_choice_ess)),
        "resampling_count": int(np.sum(trace.resampled)),
        "maximum_memory_sync_error": float(np.max(trace.memory_sync_error)),
    }
    return float(trace.nll), diagnostics


def _checkpoint_metadata(
    config_sha256: str,
    model_sha256: str,
    recovery_sha256: str,
    n_trials: int,
    stage_particles: int,
    stage_seed: int,
) -> str:
    return json.dumps(
        {
            "config_sha256": config_sha256,
            "model_sha256": model_sha256,
            "recovery_sha256": recovery_sha256,
            "n_trials": int(n_trials),
            "stage_particles": int(stage_particles),
            "stage_seed": int(stage_seed),
        },
        sort_keys=True,
    )


def _soft_summary(
    candidates: Sequence[Mapping[str, Any]],
    confirmed_indices: np.ndarray,
    confirmed_nll: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    relative = np.asarray(confirmed_nll, dtype=float) - float(np.min(confirmed_nll))
    weights = np.exp(-relative)
    weights /= weights.sum()
    means = {
        name: float(
            np.sum(
                weights
                * np.asarray(
                    [float(candidates[index][name]) for index in confirmed_indices]
                )
            )
        )
        for name in ("rho", "m", "g", "lapse")
    }
    return means, weights


def _analyse_dataset(
    candidates: Sequence[Mapping[str, Any]],
    true_index: int,
    confirmed_indices: np.ndarray,
    confirmation_nll: np.ndarray,
) -> dict[str, Any]:
    combined = np.asarray(
        [_ensemble_nll(confirmation_nll[index]) for index in confirmed_indices]
    )
    order = np.argsort(combined, kind="stable")
    best_position = int(order[0])
    best_index = int(confirmed_indices[best_position])
    true_positions = np.flatnonzero(confirmed_indices == int(true_index))
    if true_positions.size != 1:
        raise AssertionError("true candidate was not confirmed exactly once")
    true_position = int(true_positions[0])
    true_rank = int(np.flatnonzero(order == true_position)[0]) + 1
    soft_means, soft_weights = _soft_summary(
        candidates, confirmed_indices, combined
    )
    zero_positions = np.asarray(
        [
            position
            for position, index in enumerate(confirmed_indices)
            if np.isclose(float(candidates[index]["rho"]), 0.0)
        ],
        dtype=int,
    )
    positive_positions = np.asarray(
        [
            position
            for position, index in enumerate(confirmed_indices)
            if float(candidates[index]["rho"]) > 0.0
        ],
        dtype=int,
    )
    best_zero = (
        float(np.min(combined[zero_positions])) if zero_positions.size else None
    )
    best_positive = (
        float(np.min(combined[positive_positions]))
        if positive_positions.size
        else None
    )
    rows = []
    for position in order:
        index = int(confirmed_indices[position])
        rows.append(
            {
                **dict(candidates[index]),
                "combined_nll": float(combined[position]),
                "delta_nll": float(combined[position] - combined[best_position]),
                "likelihood_weight_within_confirmed_set": float(
                    soft_weights[position]
                ),
                "seed_nll": confirmation_nll[index].tolist(),
            }
        )
    return {
        "best_candidate": dict(candidates[best_index]),
        "true_candidate": dict(candidates[true_index]),
        "true_candidate_rank_within_confirmed_set": true_rank,
        "true_candidate_delta_nll": float(
            combined[true_position] - combined[best_position]
        ),
        "true_candidate_within_2_nll": bool(
            combined[true_position] - combined[best_position] <= 2.0
        ),
        "soft_parameter_mean_within_confirmed_set": soft_means,
        "best_rho_zero_nll": best_zero,
        "best_rho_positive_nll": best_positive,
        "positive_rho_nll_advantage": (
            None
            if best_zero is None or best_positive is None
            else float(best_zero - best_positive)
        ),
        "confirmed_ranking": rows,
    }


def _run_dataset(
    *,
    output: Path,
    scenario: Mapping[str, Any],
    replicate: int,
    subject_id: int,
    include_subject_in_dataset_id: bool,
    q_values: np.ndarray,
    correct_choices: np.ndarray,
    prior: np.ndarray,
    kernels,
    model_id: str,
    architecture: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    config_sha256: str,
    model_sha256: str,
    recovery_sha256: str,
    stage_particles: int,
    confirmation_particles: int,
    top_k: int,
    reconfirm: bool,
    force: bool,
) -> dict[str, Any]:
    scenario_id = str(scenario["id"])
    subject_prefix = (
        f"subject_{int(subject_id)}_" if include_subject_in_dataset_id else ""
    )
    dataset_id = (
        f"{subject_prefix}{scenario_id}_replicate_{int(replicate):02d}"
    )
    summary_path = output / "recovery" / f"{dataset_id}.json"
    trace_path = output / "recovery" / f"{dataset_id}.npz"
    dataset_path = output / "datasets" / f"{dataset_id}.npz"
    checkpoint_path = output / "checkpoints" / f"{dataset_id}.npz"
    if summary_path.exists() and trace_path.exists() and not force and not reconfirm:
        return json.loads(summary_path.read_text(encoding="utf-8"))

    generation_parts: tuple[object, ...]
    if include_subject_in_dataset_id:
        generation_parts = (int(subject_id), scenario_id, int(replicate))
    else:
        generation_parts = (scenario_id, int(replicate))
    generation_seed = _stable_seed(
        int(config["generation"]["base_seed"]), *generation_parts
    )
    true_index = _matching_candidate_index(candidates, scenario)
    simulation = simulate_model0804_choices(
        q_values,
        correct_choices,
        prior,
        kernels,
        model_id=model_id,
        parameters=_model_parameters(architecture, scenario),
        capacity=int(architecture["capacity"]),
        simulation_seed=generation_seed,
    )
    _atomic_npz(
        dataset_path,
        q_values=q_values,
        correct_choices=correct_choices,
        choices=simulation.choices,
        feedback=simulation.feedback,
        choice_probabilities=simulation.choice_probabilities,
        active=simulation.active,
        replacement_count=simulation.replacement_count,
        regenerated=simulation.regenerated,
    )

    filter_config = config["filter"]
    stage_seed = int(filter_config["stage1"]["filter_seed"])
    threshold = float(filter_config["resample_threshold_fraction"])
    metadata = _checkpoint_metadata(
        config_sha256,
        model_sha256,
        recovery_sha256,
        len(q_values),
        stage_particles,
        stage_seed,
    )
    stage_nll = np.full(len(candidates), np.nan)
    stage_runtime = np.full(len(candidates), np.nan)
    if checkpoint_path.exists() and not force:
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            observed_metadata = str(checkpoint["metadata_json"].item())
            metadata_matches = observed_metadata == metadata
            if reconfirm and not metadata_matches:
                observed = json.loads(observed_metadata)
                expected = json.loads(metadata)
                metadata_matches = all(
                    observed.get(key) == expected.get(key)
                    for key in (
                        "model_sha256",
                        "recovery_sha256",
                        "n_trials",
                        "stage_particles",
                        "stage_seed",
                    )
                )
            if metadata_matches:
                stage_nll = checkpoint["stage_nll"].astype(float)
                stage_runtime = checkpoint["stage_runtime"].astype(float)

    missing = np.flatnonzero(~np.isfinite(stage_nll))
    print(
        f"[recovery] dataset={dataset_id} trials={len(q_values)} "
        f"stage_missing={len(missing)}/{len(candidates)}",
        flush=True,
    )
    dataset_started = time.time()
    for completed, candidate_index in enumerate(missing, start=1):
        nll, diagnostics = _score_candidate(
            q_values,
            simulation.choices,
            simulation.feedback,
            prior,
            kernels,
            model_id=model_id,
            architecture=architecture,
            candidate=candidates[int(candidate_index)],
            particle_count=stage_particles,
            filter_seed=stage_seed,
            resample_threshold_fraction=threshold,
        )
        stage_nll[candidate_index] = nll
        stage_runtime[candidate_index] = diagnostics["runtime_seconds"]
        if completed % 5 == 0 or completed == len(missing):
            _atomic_npz(
                checkpoint_path,
                metadata_json=np.asarray(metadata),
                stage_nll=stage_nll,
                stage_runtime=stage_runtime,
            )
        if completed % 10 == 0 or completed == len(missing):
            print(
                f"[recovery] dataset={dataset_id} stage={completed}/{len(missing)} "
                f"elapsed={time.time() - dataset_started:.1f}s",
                flush=True,
            )

    _atomic_npz(
        checkpoint_path,
        metadata_json=np.asarray(metadata),
        stage_nll=stage_nll,
        stage_runtime=stage_runtime,
    )

    stage_order = np.argsort(stage_nll, kind="stable")
    confirmation_set = set(int(value) for value in stage_order[: int(top_k)])
    confirmation_set.add(int(true_index))
    parent_target = dict(scenario)
    parent_target["rho"] = 0.0
    confirmation_set.add(_matching_candidate_index(candidates, parent_target))
    if bool(
        filter_config["confirmation"].get(
            "always_include_stage1_profile_winner_per_parameter_level", False
        )
    ):
        for parameter_name in ("rho", "m", "g", "lapse"):
            levels = sorted(
                {float(candidate[parameter_name]) for candidate in candidates}
            )
            for level in levels:
                eligible = np.asarray(
                    [
                        index
                        for index, candidate in enumerate(candidates)
                        if np.isclose(
                            float(candidate[parameter_name]), level, atol=1e-12
                        )
                    ],
                    dtype=int,
                )
                profile_winner = int(eligible[np.argmin(stage_nll[eligible])])
                confirmation_set.add(profile_winner)
    confirmed_indices = np.asarray(sorted(confirmation_set), dtype=int)
    confirmation_seeds = [
        int(value)
        for value in filter_config["confirmation"]["independent_filter_seeds"]
    ]
    confirmation_nll = np.full(
        (len(candidates), len(confirmation_seeds)), np.nan
    )
    confirmation_runtime = np.full_like(confirmation_nll, np.nan)
    for candidate_index in confirmed_indices:
        for seed_index, filter_seed in enumerate(confirmation_seeds):
            nll, diagnostics = _score_candidate(
                q_values,
                simulation.choices,
                simulation.feedback,
                prior,
                kernels,
                model_id=model_id,
                architecture=architecture,
                candidate=candidates[int(candidate_index)],
                particle_count=confirmation_particles,
                filter_seed=filter_seed,
                resample_threshold_fraction=threshold,
            )
            confirmation_nll[candidate_index, seed_index] = nll
            confirmation_runtime[candidate_index, seed_index] = diagnostics[
                "runtime_seconds"
            ]

    analysis = _analyse_dataset(
        candidates, true_index, confirmed_indices, confirmation_nll
    )
    result = {
        "dataset_id": dataset_id,
        "subject_id": int(subject_id),
        "scenario_id": scenario_id,
        "replicate": int(replicate),
        "n_trials": int(len(q_values)),
        "generation_seed": int(generation_seed),
        "realized_regeneration_count": int(np.sum(simulation.regenerated)),
        "realized_regeneration_fraction_over_transitions": float(
            np.mean(simulation.regenerated[1:])
            if len(simulation.regenerated) > 1
            else 0.0
        ),
        "realized_mean_replacement_count": float(
            np.mean(simulation.replacement_count[1:])
            if len(simulation.replacement_count) > 1
            else 0.0
        ),
        "simulated_accuracy": float(np.mean(simulation.feedback)),
        "stage1_best_candidate": dict(candidates[int(stage_order[0])]),
        "stage1_true_candidate_rank": int(
            np.flatnonzero(stage_order == true_index)[0]
        )
        + 1,
        "stage1_true_candidate_delta_nll": float(
            stage_nll[true_index] - stage_nll[stage_order[0]]
        ),
        "stage1_nll_range": float(np.max(stage_nll) - np.min(stage_nll)),
        "stage1_total_runtime_seconds": float(np.sum(stage_runtime)),
        "confirmed_candidate_count": int(len(confirmed_indices)),
        "confirmation_total_runtime_seconds": float(
            np.nansum(confirmation_runtime)
        ),
        "runtime_seconds": float(time.time() - dataset_started),
        **analysis,
    }
    _atomic_npz(
        trace_path,
        stage_nll=stage_nll,
        stage_runtime=stage_runtime,
        stage_order=stage_order,
        confirmed_indices=confirmed_indices,
        confirmation_nll=confirmation_nll,
        confirmation_runtime=confirmation_runtime,
    )
    _atomic_json(summary_path, result)
    print(
        f"[recovery] done dataset={dataset_id} "
        f"best={result['best_candidate']['id']} "
        f"true_delta={result['true_candidate_delta_nll']:.3f} "
        f"runtime={result['runtime_seconds']:.1f}s",
        flush=True,
    )
    return result


def _run_dataset_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Process-pool entry point with one dataset as the isolation boundary."""

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    return _run_dataset(**dict(payload))


def _monotonic_axes(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for row in results:
        by_scenario.setdefault(str(row["scenario_id"]), []).append(row)
    definitions = {
        "rho": ("rho_zero", "center", "rho_high"),
        "m": ("m_low", "center", "m_high"),
        "g": ("g_low", "center", "g_high"),
        "lapse": ("lapse_low", "center", "lapse_high"),
    }
    axes = {}
    for parameter, scenario_ids in definitions.items():
        if not all(scenario_id in by_scenario for scenario_id in scenario_ids):
            axes[parameter] = {
                "available": False,
                "passed": None,
                "scenario_ids": list(scenario_ids),
            }
            continue
        estimates = []
        medians = []
        for scenario_id in scenario_ids:
            values = np.asarray(
                [
                    float(
                        row["soft_parameter_mean_within_confirmed_set"][
                            parameter
                        ]
                    )
                    for row in by_scenario[scenario_id]
                ],
                dtype=float,
            )
            estimates.append(float(np.mean(values)))
            medians.append(float(np.median(values)))
        passed = bool(
            estimates[0] <= estimates[1] <= estimates[2]
            and estimates[2] - estimates[0] > 1e-12
        )
        indexed = {
            scenario_id: {
                (int(row.get("subject_id", -1)), int(row["replicate"])): row
                for row in by_scenario[scenario_id]
            }
            for scenario_id in scenario_ids
        }
        common_keys = set(indexed[scenario_ids[0]])
        for scenario_id in scenario_ids[1:]:
            common_keys &= set(indexed[scenario_id])
        paired_order = []
        for key in sorted(common_keys):
            values = [
                float(
                    indexed[scenario_id][key][
                        "soft_parameter_mean_within_confirmed_set"
                    ][parameter]
                )
                for scenario_id in scenario_ids
            ]
            paired_order.append(
                bool(values[0] <= values[1] <= values[2] and values[2] > values[0])
            )
        axes[parameter] = {
            "available": True,
            "passed": passed,
            "scenario_ids": list(scenario_ids),
            "soft_estimate_means": estimates,
            "soft_estimate_medians": medians,
            "paired_order_count": int(sum(paired_order)),
            "paired_comparison_count": int(len(paired_order)),
            "paired_order_fraction": (
                float(np.mean(paired_order)) if paired_order else None
            ),
        }
    return axes


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total < 1:
        return None
    z = 1.959963984540054
    probability = float(successes) / float(total)
    denominator = 1.0 + z * z / float(total)
    center = (probability + z * z / (2.0 * float(total))) / denominator
    radius = (
        z
        * np.sqrt(
            probability * (1.0 - probability) / float(total)
            + z * z / (4.0 * float(total) ** 2)
        )
        / denominator
    )
    return [float(center - radius), float(center + radius)]


def _rate_summary(flags: Sequence[bool]) -> dict[str, Any]:
    values = [bool(value) for value in flags]
    successes = int(sum(values))
    total = int(len(values))
    return {
        "value": float(successes / total) if total else None,
        "successes": successes,
        "total": total,
        "wilson_95_interval": _wilson_interval(successes, total),
    }


def _parameter_confusion(
    results: Sequence[Mapping[str, Any]], parameter: str
) -> dict[str, Any]:
    levels = sorted(
        {
            float(row["true_candidate"][parameter])
            for row in results
        }
        | {
            float(row["best_candidate"][parameter])
            for row in results
        }
    )
    level_to_index = {value: index for index, value in enumerate(levels)}
    counts = np.zeros((len(levels), len(levels)), dtype=int)
    for row in results:
        true_value = float(row["true_candidate"][parameter])
        selected_value = float(row["best_candidate"][parameter])
        counts[level_to_index[true_value], level_to_index[selected_value]] += 1
    row_totals = counts.sum(axis=1, keepdims=True)
    proportions = np.divide(
        counts,
        row_totals,
        out=np.zeros_like(counts, dtype=float),
        where=row_totals > 0,
    )
    return {
        "levels": levels,
        "rows_true_columns_selected": counts.tolist(),
        "row_proportions": proportions.tolist(),
    }


def _aggregate_results(
    results: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
    *,
    full_pilot: bool,
) -> tuple[dict[str, Any], str]:
    if not results:
        raise ValueError("recovery aggregation requires at least one dataset")
    true_within_flags = [
        bool(row["true_candidate_within_2_nll"]) for row in results
    ]
    rho_class_correct = []
    rho_zero_specific = []
    rho_positive_sensitive = []
    selected_errors = {name: [] for name in ("rho", "m", "g", "lapse")}
    soft_errors = {name: [] for name in ("rho", "m", "g", "lapse")}
    exact_flags = {name: [] for name in selected_errors}
    for row in results:
        true = row["true_candidate"]
        selected = row["best_candidate"]
        soft = row["soft_parameter_mean_within_confirmed_set"]
        true_positive = float(true["rho"]) > 0.0
        selected_positive = float(selected["rho"]) > 0.0
        rho_class_correct.append(true_positive == selected_positive)
        if true_positive:
            rho_positive_sensitive.append(selected_positive)
        else:
            rho_zero_specific.append(not selected_positive)
        for name in selected_errors:
            selected_errors[name].append(
                abs(float(selected[name]) - float(true[name]))
            )
            soft_errors[name].append(abs(float(soft[name]) - float(true[name])))
            exact_flags[name].append(
                bool(np.isclose(float(selected[name]), float(true[name]), atol=1e-12))
            )

    by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for row in results:
        by_scenario.setdefault(str(row["scenario_id"]), []).append(row)
    scenario_summaries = {}
    for scenario_id, rows in sorted(by_scenario.items()):
        scenario_summaries[scenario_id] = {
            "dataset_count": int(len(rows)),
            "true_grid_point_within_2_nll": _rate_summary(
                [bool(row["true_candidate_within_2_nll"]) for row in rows]
            ),
            "median_true_candidate_delta_nll": float(
                np.median([float(row["true_candidate_delta_nll"]) for row in rows])
            ),
            "p90_true_candidate_delta_nll": float(
                np.quantile(
                    [float(row["true_candidate_delta_nll"]) for row in rows], 0.90
                )
            ),
            "exact_selected_parameter_rate": {
                name: _rate_summary(
                    [
                        bool(
                            np.isclose(
                                float(row["best_candidate"][name]),
                                float(row["true_candidate"][name]),
                                atol=1e-12,
                            )
                        )
                        for row in rows
                    ]
                )
                for name in selected_errors
            },
        }

    subject_summaries = {}
    subject_ids = sorted({int(row.get("subject_id", -1)) for row in results})
    for subject_id in subject_ids:
        rows = [
            row for row in results if int(row.get("subject_id", -1)) == subject_id
        ]
        subject_g_extremes = [
            row
            for row in rows
            if str(row["scenario_id"]) in {"g_low", "g_high"}
        ]
        subject_summaries[str(subject_id)] = {
            "dataset_count": int(len(rows)),
            "n_trials": int(rows[0]["n_trials"]),
            "true_grid_point_within_2_nll": _rate_summary(
                [bool(row["true_candidate_within_2_nll"]) for row in rows]
            ),
            "rho_zero_vs_positive_classification": _rate_summary(
                [
                    (float(row["true_candidate"]["rho"]) > 0.0)
                    == (float(row["best_candidate"]["rho"]) > 0.0)
                    for row in rows
                ]
            ),
            "g_extreme_exact_level_accuracy": _rate_summary(
                [
                    bool(
                        np.isclose(
                            float(row["best_candidate"]["g"]),
                            float(row["true_candidate"]["g"]),
                            atol=1e-12,
                        )
                    )
                    for row in subject_g_extremes
                ]
            ),
            "exact_selected_parameter_rate": {
                name: _rate_summary(
                    [
                        bool(
                            np.isclose(
                                float(row["best_candidate"][name]),
                                float(row["true_candidate"][name]),
                                atol=1e-12,
                            )
                        )
                        for row in rows
                    ]
                )
                for name in selected_errors
            },
        }

    minimum_scenario_within = min(
        float(summary["true_grid_point_within_2_nll"]["value"])
        for summary in scenario_summaries.values()
    )
    g_extreme_rows = [
        row for row in results if str(row["scenario_id"]) in {"g_low", "g_high"}
    ]
    g_extreme_exact = [
        bool(
            np.isclose(
                float(row["best_candidate"]["g"]),
                float(row["true_candidate"]["g"]),
                atol=1e-12,
            )
        )
        for row in g_extreme_rows
    ]
    g_direction = [
        bool(
            float(row["best_candidate"]["g"]) < 0.35
            if str(row["scenario_id"]) == "g_low"
            else float(row["best_candidate"]["g"]) > 0.35
        )
        for row in g_extreme_rows
    ]
    seed_winner_agreement = []
    true_candidate_seed_nll_spread = []
    for row in results:
        ranking = row["confirmed_ranking"]
        seed_count = len(ranking[0]["seed_nll"])
        seed_winners = [
            min(ranking, key=lambda candidate: candidate["seed_nll"][seed_index])[
                "id"
            ]
            for seed_index in range(seed_count)
        ]
        seed_winner_agreement.append(len(set(seed_winners)) == 1)
        true_id = str(row["true_candidate"]["id"])
        true_row = next(candidate for candidate in ranking if candidate["id"] == true_id)
        true_seed_nll = np.asarray(true_row["seed_nll"], dtype=float)
        true_candidate_seed_nll_spread.append(
            float(np.max(true_seed_nll) - np.min(true_seed_nll))
        )

    overall_within = _rate_summary(true_within_flags)
    rho_accuracy = _rate_summary(rho_class_correct)
    rho_specificity = _rate_summary(rho_zero_specific)
    rho_sensitivity = _rate_summary(rho_positive_sensitive)
    g_exact = _rate_summary(g_extreme_exact)
    g_direction_rate = _rate_summary(g_direction)
    monotonic = _monotonic_axes(results)
    monotonic_count = int(
        sum(value.get("passed") is True for value in monotonic.values())
    )
    available_axes = int(
        sum(value.get("available") is True for value in monotonic.values())
    )
    checks = {
        "true_grid_point_within_2_nll_fraction": {
            **overall_within,
            "threshold": float(
                gates["minimum_true_grid_point_within_2_nll_fraction"]
            ),
            "passed": bool(
                float(overall_within["value"])
                >= float(gates["minimum_true_grid_point_within_2_nll_fraction"])
            ),
        },
        "rho_zero_vs_positive_classification_accuracy": {
            **rho_accuracy,
            "threshold": float(
                gates["minimum_rho_zero_vs_positive_classification_accuracy"]
            ),
            "passed": bool(
                float(rho_accuracy["value"])
                >= float(
                    gates["minimum_rho_zero_vs_positive_classification_accuracy"]
                )
            ),
        },
        "median_absolute_rho_error": {
            "value": float(np.median(selected_errors["rho"])),
            "threshold": float(gates["maximum_median_absolute_rho_error"]),
            "passed": bool(
                np.median(selected_errors["rho"])
                <= float(gates["maximum_median_absolute_rho_error"])
            ),
        },
        "monotonic_parameter_axes": {
            "value": monotonic_count,
            "available_axes": available_axes,
            "threshold": int(gates["minimum_monotonic_parameter_axes"]),
            "passed": bool(
                full_pilot
                and monotonic_count >= int(gates["minimum_monotonic_parameter_axes"])
            ),
        },
    }
    optional_checks = (
        (
            "minimum_per_scenario_true_grid_point_within_2_nll_fraction",
            "minimum_per_scenario_true_grid_point_within_2_nll_fraction",
            minimum_scenario_within,
            ">=",
        ),
        (
            "minimum_rho_zero_specificity",
            "rho_zero_specificity",
            rho_specificity["value"],
            ">=",
        ),
        (
            "minimum_rho_positive_sensitivity",
            "rho_positive_sensitivity",
            rho_sensitivity["value"],
            ">=",
        ),
        (
            "minimum_g_extreme_exact_level_accuracy",
            "g_extreme_exact_level_accuracy",
            g_exact["value"],
            ">=",
        ),
        (
            "minimum_g_low_vs_high_direction_accuracy",
            "g_low_vs_high_direction_accuracy",
            g_direction_rate["value"],
            ">=",
        ),
    )
    rate_details = {
        "rho_zero_specificity": rho_specificity,
        "rho_positive_sensitivity": rho_sensitivity,
        "g_extreme_exact_level_accuracy": g_exact,
        "g_low_vs_high_direction_accuracy": g_direction_rate,
    }
    for gate_key, check_name, value, _ in optional_checks:
        if gate_key not in gates:
            continue
        threshold = float(gates[gate_key])
        check = {
            "value": None if value is None else float(value),
            "threshold": threshold,
            "passed": bool(value is not None and float(value) >= threshold),
        }
        if check_name in rate_details:
            check.update(rate_details[check_name])
            check["threshold"] = threshold
            check["passed"] = bool(value is not None and float(value) >= threshold)
        checks[check_name] = check

    all_passed = bool(
        full_pilot and all(bool(check["passed"]) for check in checks.values())
    )
    if not full_pilot:
        route = "smoke_test_only_no_gate_decision"
    elif all_passed:
        route = str(
            gates.get("route_on_pass", "advance_to_replicated_mechanism_recovery")
        )
    else:
        route = str(
            gates.get("route_on_fail", "mechanism_separation_not_yet_established")
        )
    summary = {
        "dataset_count": int(len(results)),
        "true_grid_point_within_2_nll": overall_within,
        "rho_zero_vs_positive_classification": rho_accuracy,
        "rho_zero_specificity": rho_specificity,
        "rho_positive_sensitivity": rho_sensitivity,
        "g_extreme_exact_level_accuracy": g_exact,
        "g_low_vs_high_direction_accuracy": g_direction_rate,
        "true_grid_point_within_2_nll_fraction": float(overall_within["value"]),
        "rho_zero_vs_positive_classification_accuracy": float(
            rho_accuracy["value"]
        ),
        "selected_parameter_exact_accuracy": {
            name: _rate_summary(values) for name, values in exact_flags.items()
        },
        "selected_parameter_mean_absolute_error": {
            name: float(np.mean(values)) for name, values in selected_errors.items()
        },
        "selected_parameter_median_absolute_error": {
            name: float(np.median(values)) for name, values in selected_errors.items()
        },
        "soft_parameter_mean_absolute_error": {
            name: float(np.mean(values)) for name, values in soft_errors.items()
        },
        "parameter_confusion": {
            name: _parameter_confusion(results, name) for name in selected_errors
        },
        "scenario_summaries": scenario_summaries,
        "subject_summaries": subject_summaries,
        "posthoc_numerical_stability_diagnostic_not_a_frozen_gate": {
            "independent_seed_winner_agreement": _rate_summary(
                seed_winner_agreement
            ),
            "true_candidate_seed_nll_range_median": float(
                np.median(true_candidate_seed_nll_spread)
            ),
            "true_candidate_seed_nll_range_p90": float(
                np.quantile(true_candidate_seed_nll_spread, 0.90)
            ),
            "true_candidate_seed_nll_range_maximum": float(
                np.max(true_candidate_seed_nll_spread)
            ),
            "interpretation": (
                "descriptive_only_because_no_seed_stability_threshold_was_frozen"
            ),
        },
        "minimum_per_scenario_true_grid_point_within_2_nll_fraction": float(
            minimum_scenario_within
        ),
        "monotonic_axes": monotonic,
        "gate_checks": checks,
        "all_recovery_gates_passed": all_passed,
        "all_pilot_gates_passed": all_passed,
    }
    return summary, route


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = args.config.resolve()
    output = args.output.resolve()
    previous_report_path = output / "recovery_report.json"
    previous_compute_wall_runtime = None
    if previous_report_path.exists() and not args.force and not args.reconfirm:
        try:
            previous_report = json.loads(
                previous_report_path.read_text(encoding="utf-8")
            )
            previous_compute_wall_runtime = float(
                previous_report.get(
                    "compute_wall_runtime_seconds",
                    previous_report["runtime_seconds"],
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            previous_compute_wall_runtime = None
    config = _load_recovery_config(config_path)
    base_path = (ROOT / str(config["base_config"])).resolve()
    base = load_config(base_path)
    data_scope = config["data_scope"]
    if "subject_ids" in data_scope:
        configured_subjects = [int(value) for value in data_scope["subject_ids"]]
    else:
        configured_subjects = [int(data_scope["subject_id"])]
    if len(configured_subjects) != len(set(configured_subjects)):
        raise ValueError("recovery config subject ids must be unique")
    selected_subjects = None
    if args.subjects:
        selected_subjects = {
            int(value.strip())
            for value in args.subjects.split(",")
            if value.strip()
        }
        unknown_subjects = selected_subjects - set(configured_subjects)
        if unknown_subjects:
            raise ValueError(
                f"subjects are outside the frozen recovery set: {sorted(unknown_subjects)}"
            )
        requested_subjects = [
            value for value in configured_subjects if value in selected_subjects
        ]
    else:
        requested_subjects = configured_subjects.copy()
    frame, subjects, input_audit = validate_and_load_inputs(
        base, set(requested_subjects)
    )
    if subjects != sorted(requested_subjects):
        raise ValueError("recovery run did not resolve the frozen subjects")
    priors, kernels_by_prior, geometry_audit = build_frozen_geometry(base)
    prior_id = str(base["rule_space"]["primary_prior"])
    prior = priors[prior_id]
    kernels = kernels_by_prior[prior_id]
    cache_audits: dict[int, dict[str, Any]] = {}
    arrays_by_subject: dict[int, dict[str, np.ndarray]] = {}
    correct_by_subject: dict[int, np.ndarray] = {}
    for subject_id in subjects:
        cache_audit = validate_subject_cache(base, frame, subject_id)
        arrays = _load_subject_arrays(cache_audit, args.max_trials)
        with np.load(
            Path(cache_audit["prediction_path"]), allow_pickle=False
        ) as payload:
            cached_category = payload["category"].astype(int)[: len(arrays["choice"])]
        correct_choices = infer_correct_choices(
            arrays["choice"], arrays["feedback"]
        )
        if not np.array_equal(correct_choices, cached_category):
            raise ValueError(
                f"subject {subject_id} feedback-derived correct choices do not "
                "match cached category"
            )
        cache_audits[int(subject_id)] = cache_audit
        arrays_by_subject[int(subject_id)] = arrays
        correct_by_subject[int(subject_id)] = correct_choices

    scenarios = list(config["generation"]["scenarios"])
    selected_scenarios = None
    if args.scenarios:
        selected_scenarios = {
            value.strip() for value in args.scenarios.split(",") if value.strip()
        }
        known = {str(row["id"]) for row in scenarios}
        unknown = selected_scenarios - known
        if unknown:
            raise ValueError(f"unknown scenarios: {sorted(unknown)}")
        scenarios = [
            row for row in scenarios if str(row["id"]) in selected_scenarios
        ]
    if not scenarios:
        raise ValueError("no generation scenarios selected")

    candidates = _candidate_grid(config)
    filter_config = config["filter"]
    stage_particles = int(
        args.stage_particles
        if args.stage_particles is not None
        else filter_config["stage1"]["particle_count"]
    )
    confirmation_particles = int(
        args.confirmation_particles
        if args.confirmation_particles is not None
        else filter_config["confirmation"]["particle_count"]
    )
    top_k = int(
        args.top_k
        if args.top_k is not None
        else filter_config["confirmation"]["top_k_stage1"]
    )
    if stage_particles < 2 or confirmation_particles < 2:
        raise ValueError("particle counts must be at least two")
    if not 1 <= top_k <= len(candidates):
        raise ValueError("top-k confirmation count is outside the candidate grid")

    model_path = ROOT / "src/Bayesian_state/utils/model_0804.py"
    recovery_path = ROOT / "src/Bayesian_state/utils/model_0804_recovery.py"
    config_sha256 = _sha256(config_path)
    model_sha256 = _sha256(model_path)
    recovery_sha256 = _sha256(recovery_path)
    configured_repeats = int(
        config["generation"].get(
            "replicates_per_subject", config["generation"].get("replicates", 1)
        )
    )
    repeats = int(
        args.replicates if args.replicates is not None else configured_repeats
    )
    if repeats < 1:
        raise ValueError("generation replicates must be positive")
    configured_workers = int(config.get("execution", {}).get("workers", 1))
    workers = int(args.workers if args.workers is not None else configured_workers)
    if workers < 1:
        raise ValueError("workers must be positive")
    worker_threads = int(
        config.get("execution", {}).get("worker_blas_threads", 1)
    )
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(worker_threads)

    include_subject_in_dataset_id = len(configured_subjects) > 1
    payloads: list[dict[str, Any]] = []
    for scenario in scenarios:
        _matching_candidate_index(candidates, scenario)
        for subject_id in subjects:
            for replicate in range(repeats):
                payloads.append(
                    dict(
                        output=output,
                        scenario=scenario,
                        replicate=replicate,
                        subject_id=int(subject_id),
                        include_subject_in_dataset_id=include_subject_in_dataset_id,
                        q_values=arrays_by_subject[subject_id]["q"],
                        correct_choices=correct_by_subject[subject_id],
                        prior=prior,
                        kernels=kernels,
                        model_id=str(data_scope["model_id"]),
                        architecture=config["architecture"],
                        candidates=candidates,
                        config=config,
                        config_sha256=config_sha256,
                        model_sha256=model_sha256,
                        recovery_sha256=recovery_sha256,
                        stage_particles=stage_particles,
                        confirmation_particles=confirmation_particles,
                        top_k=top_k,
                        reconfirm=bool(args.reconfirm),
                        force=bool(args.force),
                    )
                )
    print(
        f"[recovery] scheduled_datasets={len(payloads)} subjects={subjects} "
        f"scenarios={len(scenarios)} replicates={repeats} workers={workers}",
        flush=True,
    )
    results: list[dict[str, Any] | None] = [None] * len(payloads)
    if workers == 1:
        for index, payload in enumerate(payloads):
            results[index] = _run_dataset_payload(payload)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(_run_dataset_payload, payload): index
                for index, payload in enumerate(payloads)
            }
            completed_count = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
                completed_count += 1
                print(
                    f"[recovery] completed_datasets={completed_count}/{len(payloads)}",
                    flush=True,
                )
    completed_results = [row for row in results if row is not None]
    if len(completed_results) != len(payloads):
        raise AssertionError("recovery worker result count is incomplete")

    is_full_pilot = bool(
        selected_subjects is None
        and selected_scenarios is None
        and repeats == configured_repeats
        and args.max_trials is None
        and stage_particles == int(filter_config["stage1"]["particle_count"])
        and confirmation_particles
        == int(filter_config["confirmation"]["particle_count"])
        and top_k == int(filter_config["confirmation"]["top_k_stage1"])
        and len(scenarios) == len(config["generation"]["scenarios"])
        and subjects == sorted(configured_subjects)
    )
    gates = config.get("recovery_gates", config.get("pilot_gates"))
    if not isinstance(gates, Mapping):
        raise ValueError("recovery config is missing recovery/pilot gates")
    aggregate, route = _aggregate_results(
        completed_results, gates, full_pilot=is_full_pilot
    )
    replicated = bool(len(configured_subjects) > 1 or configured_repeats > 1)
    refresh_runtime = float(time.time() - started)
    compute_wall_runtime = (
        refresh_runtime
        if previous_compute_wall_runtime is None
        else previous_compute_wall_runtime
    )
    report = {
        "analysis_id": str(config["analysis_id"]),
        "status": (
            "replicated_mechanism_recovery_complete"
            if replicated
            else "mechanism_recovery_pilot_complete"
        ),
        "scope": str(config["scope"]),
        "route_decision": route,
        "is_frozen_full_pilot": is_full_pilot,
        "is_frozen_full_run": is_full_pilot,
        "runtime_seconds": compute_wall_runtime,
        "compute_wall_runtime_seconds": compute_wall_runtime,
        "report_refresh_runtime_seconds": refresh_runtime,
        "dataset_runtime_sum_seconds": float(
            np.sum([float(row["runtime_seconds"]) for row in completed_results])
        ),
        "subject_id": int(subjects[0]) if len(subjects) == 1 else None,
        "subject_ids": [int(value) for value in subjects],
        "model_id": str(data_scope["model_id"]),
        "n_trials": (
            int(len(arrays_by_subject[subjects[0]]["choice"]))
            if len(subjects) == 1
            else None
        ),
        "n_trials_by_subject": {
            str(subject_id): int(len(arrays_by_subject[subject_id]["choice"]))
            for subject_id in subjects
        },
        "candidate_count": int(len(candidates)),
        "scenario_count": int(len(scenarios)),
        "replicates_per_scenario": repeats * len(subjects),
        "replicates_per_subject": repeats,
        "dataset_count": int(len(completed_results)),
        "workers": workers,
        "stage1_particle_count": stage_particles,
        "confirmation_particle_count": confirmation_particles,
        "confirmation_top_k_stage1": top_k,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "base_config_path": str(base_path),
        "base_config_sha256": _sha256(base_path),
        "implementation_sha256": {
            "model_0804.py": model_sha256,
            "model_0804_recovery.py": recovery_sha256,
            "recovery_runner": _sha256(Path(__file__).resolve()),
            "recovery_tests": _sha256(
                ROOT / "tests/test_model_0804_recovery.py"
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "input_audit": input_audit,
        "cache_audit": (
            cache_audits[subjects[0]] if len(subjects) == 1 else None
        ),
        "cache_audits": {
            str(subject_id): cache_audits[subject_id] for subject_id in subjects
        },
        "geometry_audit": geometry_audit,
        "category_reconstruction_match": True,
        "architecture": dict(config["architecture"]),
        "candidate_grid": [dict(candidate) for candidate in candidates],
        "aggregate": aggregate,
        "datasets": completed_results,
        "pilot_gates": dict(gates),
        "recovery_gates": dict(gates),
        "guardrails": list(config["guardrails"]),
    }
    report_path = output / "recovery_report.json"
    _atomic_json(report_path, report)
    print(
        f"RECOVERY status={report['status']} route={route} "
        f"output={report_path} runtime={report['runtime_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
