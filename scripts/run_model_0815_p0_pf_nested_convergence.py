#!/usr/bin/env python3
"""Pilot nested PF particle-count and independent-seed convergence.

Unlike the first convergence runner, this analysis retains every seed's
choice probabilities, filtered executed-state posterior, and ESS trajectory.
It can therefore separate particle-count error from seed-ensemble error and
can be independently recomputed without rerunning the particle filter.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Bayesian_state.simulation.config import (  # noqa: E402
    load_yaml,
    resolve_engine_config,
    resolve_loss_delta,
    resolve_loss_metric,
    resolve_prediction_modes,
    resolve_window_size,
)
from src.Bayesian_state.simulation.parameters import (  # noqa: E402
    apply_fixed_hyperparams_to_engine_config,
    infer_fixed_hyperparams_from_engine_config,
)
from src.Bayesian_state.simulation.runner import (  # noqa: E402
    StateModelSimulationRunner,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.utils.seeding import stable_seed  # noqa: E402
from src.Bayesian_state.utils.subjects import resolve_subject_config  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "configs/specific_models/"
    "model_0815_p0_calibrated_pf_nested_convergence_pilot.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("run", "summarize", "all"), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _choice_nll(
    probabilities: np.ndarray,
    choice_index: np.ndarray,
    valid: np.ndarray,
) -> float:
    keep = (
        np.asarray(valid, dtype=bool)
        & (choice_index >= 0)
        & (choice_index < probabilities.shape[1])
        & np.all(np.isfinite(probabilities), axis=1)
    )
    rows = np.flatnonzero(keep)
    selected = probabilities[rows, choice_index[rows]]
    return float(np.mean(-np.log(np.clip(selected, 1e-12, 1.0))))


def _mean_js(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=float)
    right = np.asarray(second, dtype=float)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("executed-state matrices must have equal 2-D shapes")
    left = np.clip(left, 0.0, None)
    right = np.clip(right, 0.0, None)
    left /= np.sum(left, axis=1, keepdims=True)
    right /= np.sum(right, axis=1, keepdims=True)
    midpoint = 0.5 * (left + right)

    def kl(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        mask = values > 0.0
        terms = np.zeros_like(values)
        terms[mask] = values[mask] * np.log(
            values[mask] / np.clip(reference[mask], 1e-12, None)
        )
        return np.sum(terms, axis=1)

    return float(np.mean(0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)))


def _normalize_probability_stack(values: Any, *, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{label} must have shape (seeds, trials, states)")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{label} must be finite and non-negative")
    row_sum = np.sum(array, axis=2, keepdims=True)
    if np.any(row_sum <= 0.0):
        raise ValueError(f"{label} contains zero-mass rows")
    return array / row_sum


def validate_panel(panel: Mapping[str, Any]) -> dict[str, np.ndarray]:
    probability = _normalize_probability_stack(
        panel["choice_probability"], label="choice_probability"
    )
    executed = _normalize_probability_stack(
        panel["filtered_executed_probability"],
        label="filtered_executed_probability",
    )
    if probability.shape[:2] != executed.shape[:2]:
        raise ValueError("choice and executed stacks disagree on seeds/trials")
    seed_n, trial_n = probability.shape[:2]
    ess = np.asarray(panel["post_choice_ess"], dtype=float)
    if ess.shape != (seed_n, trial_n) or not np.all(np.isfinite(ess)):
        raise ValueError("post_choice_ess has an invalid shape or value")
    seeds = np.asarray(panel["filter_seed"], dtype=np.uint64).reshape(-1)
    repeat_index = np.asarray(panel["repeat_index"], dtype=int).reshape(-1)
    if seeds.size != seed_n or np.unique(seeds).size != seed_n:
        raise ValueError("filter seeds must be unique and match the stack")
    if not np.array_equal(repeat_index, np.arange(seed_n, dtype=int)):
        raise ValueError("repeat indices must be contiguous and ordered")
    choices = np.asarray(panel["observed_choice_index"], dtype=int).reshape(-1)
    valid = np.asarray(panel["valid_trial_mask"], dtype=bool).reshape(-1)
    if choices.size != trial_n or valid.size != trial_n:
        raise ValueError("observed arrays do not match the trial count")
    return {
        "choice_probability": probability,
        "filtered_executed_probability": executed,
        "post_choice_ess": ess,
        "filter_seed": seeds,
        "repeat_index": repeat_index,
        "observed_choice_index": choices,
        "valid_trial_mask": valid,
    }


def summarize_ensemble(
    panel: Mapping[str, Any],
    *,
    subject_id: int,
    particle_count: int,
    seed_count: int,
    gates: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    normalized = validate_panel(panel)
    k = int(seed_count)
    if k < 2 or k % 2 or k > normalized["choice_probability"].shape[0]:
        raise ValueError("seed_count must be even and available in the panel")
    probability_stack = normalized["choice_probability"][:k]
    executed_stack = normalized["filtered_executed_probability"][:k]
    ess = normalized["post_choice_ess"][:k]
    choices = normalized["observed_choice_index"]
    valid = normalized["valid_trial_mask"]
    half = k // 2
    mean_probability = np.mean(probability_stack, axis=0)
    mean_executed = np.mean(executed_stack, axis=0)
    first_probability = np.mean(probability_stack[:half], axis=0)
    second_probability = np.mean(probability_stack[half:k], axis=0)
    first_executed = np.mean(executed_stack[:half], axis=0)
    second_executed = np.mean(executed_stack[half:k], axis=0)
    split_choice_rmse = float(
        np.sqrt(np.mean(np.square(first_probability[valid] - second_probability[valid])))
    )
    split_executed_js = _mean_js(first_executed[valid], second_executed[valid])
    run_nll = np.asarray(
        [_choice_nll(values, choices, valid) for values in probability_stack],
        dtype=float,
    )
    probability_mcse_rms = float(
        np.sqrt(
            np.mean(
                np.var(probability_stack[:, valid], axis=0, ddof=1) / float(k)
            )
        )
    )
    row = {
        "subject_id": int(subject_id),
        "particle_count": int(particle_count),
        "seed_count": k,
        "total_particle_trajectories": int(particle_count) * k,
        "choice_nll": _choice_nll(mean_probability, choices, valid),
        "run_choice_nll_mean": float(np.mean(run_nll)),
        "run_choice_nll_sd": float(np.std(run_nll, ddof=1)),
        "choice_probability_mcse_rms": probability_mcse_rms,
        "split_half_choice_probability_rmse": split_choice_rmse,
        "split_half_executed_posterior_js": split_executed_js,
        "median_post_choice_ess_fraction": float(
            np.median(ess / float(particle_count))
        ),
        "gate_split_choice_passed": bool(
            split_choice_rmse
            <= float(gates["maximum_split_half_choice_probability_rmse"])
        ),
        "gate_split_executed_passed": bool(
            split_executed_js
            <= float(gates["maximum_split_half_executed_posterior_js"])
        ),
        "gate_ess_passed": bool(
            np.median(ess / float(particle_count))
            >= float(gates["minimum_median_post_choice_ess_fraction"])
        ),
    }
    row["all_internal_gates_passed"] = bool(
        row["gate_split_choice_passed"]
        and row["gate_split_executed_passed"]
        and row["gate_ess_passed"]
    )
    arrays = {
        "mean_choice_probability": mean_probability,
        "mean_filtered_executed_probability": mean_executed,
        "observed_choice_index": choices,
        "valid_trial_mask": valid,
    }
    return row, arrays


def compare_ensembles(
    left_row: Mapping[str, Any],
    left: Mapping[str, np.ndarray],
    right_row: Mapping[str, Any],
    right: Mapping[str, np.ndarray],
    *,
    comparison_id: str,
    comparison_role: str,
    gates: Mapping[str, float],
) -> dict[str, Any]:
    if not np.array_equal(left["observed_choice_index"], right["observed_choice_index"]):
        raise ValueError("ensemble comparison has mismatched observed choices")
    valid = np.asarray(left["valid_trial_mask"], dtype=bool) & np.asarray(
        right["valid_trial_mask"], dtype=bool
    )
    nll_change = abs(float(left_row["choice_nll"]) - float(right_row["choice_nll"]))
    choice_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    left["mean_choice_probability"][valid]
                    - right["mean_choice_probability"][valid]
                )
            )
        )
    )
    executed_js = _mean_js(
        left["mean_filtered_executed_probability"][valid],
        right["mean_filtered_executed_probability"][valid],
    )
    row = {
        "comparison_id": str(comparison_id),
        "comparison_role": str(comparison_role),
        "subject_id": int(left_row["subject_id"]),
        "left_particle_count": int(left_row["particle_count"]),
        "left_seed_count": int(left_row["seed_count"]),
        "left_total_particle_trajectories": int(
            left_row["total_particle_trajectories"]
        ),
        "right_particle_count": int(right_row["particle_count"]),
        "right_seed_count": int(right_row["seed_count"]),
        "right_total_particle_trajectories": int(
            right_row["total_particle_trajectories"]
        ),
        "absolute_choice_nll_change": nll_change,
        "choice_probability_rmse": choice_rmse,
        "executed_posterior_js": executed_js,
        "gate_choice_nll_passed": bool(
            nll_change <= float(gates["maximum_choice_nll_change"])
        ),
        "gate_choice_probability_passed": bool(
            choice_rmse <= float(gates["maximum_choice_probability_rmse"])
        ),
        "gate_executed_posterior_passed": bool(
            executed_js <= float(gates["maximum_executed_posterior_js"])
        ),
    }
    row["all_comparison_gates_passed"] = bool(
        row["gate_choice_nll_passed"]
        and row["gate_choice_probability_passed"]
        and row["gate_executed_posterior_passed"]
    )
    return row


def _filter_seeds(base_seed: int, subject_id: int, repeat_count: int) -> np.ndarray:
    return np.asarray(
        [
            stable_seed(
                {
                    "seed_role": "model0815_p0_pf_convergence",
                    "base_seed": int(base_seed),
                    "subject_id": int(subject_id),
                    "repeat_index": int(index),
                }
            )
            for index in range(int(repeat_count))
        ],
        dtype=np.uint64,
    )


def _panel_paths(output: Path, subject_id: int, particle_count: int) -> tuple[Path, Path]:
    stem = f"subject_{int(subject_id)}_R{int(particle_count)}"
    return output / "cache" / f"{stem}.npz", output / "cache" / f"{stem}.json"


def _load_panel(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as bundle:
        panel = {key: np.asarray(bundle[key]) for key in bundle.files}
    return validate_panel(panel)


def _run_panel(
    *,
    simulation_config_path: Path,
    simulation_config: Mapping[str, Any],
    output: Path,
    subject_id: int,
    particle_count: int,
    repeat_count: int,
    max_trials: int | None,
    base_seed: int,
    n_jobs: int,
    force: bool,
) -> dict[str, Any]:
    npz_path, json_path = _panel_paths(output, subject_id, particle_count)
    expected_seeds = _filter_seeds(base_seed, subject_id, repeat_count)
    if npz_path.exists() and json_path.exists() and not force:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        panel = _load_panel(npz_path)
        if _sha256(npz_path) != metadata["npz_sha256"]:
            raise ValueError(f"cache hash mismatch: {npz_path}")
        if not np.array_equal(panel["filter_seed"], expected_seeds):
            raise ValueError(f"cache seeds do not match the design: {npz_path}")
        return metadata

    subject_cfg = resolve_subject_config(simulation_config, subject_id)
    engine = resolve_engine_config(
        subject_cfg,
        simulation_config_path.parent,
        subject_id=subject_id,
    )
    fixed = {
        **infer_fixed_hyperparams_from_engine_config(engine),
        **dict(subject_cfg.get("fixed_hyperparams") or {}),
    }
    engine = apply_fixed_hyperparams_to_engine_config(engine, fixed)
    engine.setdefault("inference", {})["particle_count"] = int(particle_count)
    dataset_paths = resolve_dataset_paths(subject_cfg, simulation_config_path.parent)
    runner = StateModelSimulationRunner(
        engine_config=engine,
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
        n_jobs=int(n_jobs),
    )
    runner.prepare_data(dataset_paths["learning_data"])
    prediction_mode, selection_mode = resolve_prediction_modes(subject_cfg)
    loss_metric = resolve_loss_metric(subject_cfg)
    result = runner.simulate_subject(
        subject_id=int(subject_id),
        simulation_repeats=int(repeat_count),
        fixed_hyperparams=fixed,
        window_size=resolve_window_size(subject_cfg, subject_id, [subject_id]),
        stop_at=float(subject_cfg.get("stop_at", 1.0)),
        max_trials=max_trials,
        keep_logs=True,
        prediction_mode=prediction_mode,
        selection_prediction_mode=selection_mode,
        loss_metric=loss_metric,
        loss_delta=resolve_loss_delta(subject_cfg, loss_metric),
        hyper_candidate_seed=int(base_seed),
        trajectory_seeds=[int(value) for value in expected_seeds],
        compute_statistics=False,
        repeat_aggregation="mean_probability",
        evaluation_protocol=subject_cfg.get("evaluation_protocol"),
    )
    raw_runs = list(result["best"].raw_runs or [])
    if len(raw_runs) != int(repeat_count):
        raise RuntimeError("nested convergence did not return every requested seed")
    probabilities: list[np.ndarray] = []
    executed: list[np.ndarray] = []
    ess: list[np.ndarray] = []
    observed_choice_index = None
    valid_trial_mask = None
    observed_seeds: list[int] = []
    for run in raw_runs:
        metrics = run["metrics_by_mode"][selection_mode]
        probability = np.asarray(metrics["pred_category_probs"], dtype=float)
        current_choices = np.asarray(metrics["observed_choice_index"], dtype=int)
        current_valid = np.asarray(metrics["valid_trial_mask"], dtype=bool)
        if observed_choice_index is None:
            observed_choice_index = current_choices
            valid_trial_mask = current_valid
        elif not np.array_equal(observed_choice_index, current_choices) or not np.array_equal(
            valid_trial_mask, current_valid
        ):
            raise ValueError("observed trials differ across PF seeds")
        state_log = run.get("state_log") or {}
        probabilities.append(probability)
        executed.append(
            np.asarray(state_log["filtered_executed_probability"], dtype=float)
        )
        ess.append(np.asarray(state_log["post_choice_ess"], dtype=float))
        observed_seeds.append(int(run["trajectory_seed"]))
    if observed_choice_index is None or valid_trial_mask is None:
        raise RuntimeError("nested convergence returned no observed trials")
    panel = validate_panel(
        {
            "choice_probability": np.stack(probabilities),
            "filtered_executed_probability": np.stack(executed),
            "post_choice_ess": np.stack(ess),
            "filter_seed": np.asarray(observed_seeds, dtype=np.uint64),
            "repeat_index": np.arange(repeat_count, dtype=int),
            "observed_choice_index": observed_choice_index,
            "valid_trial_mask": valid_trial_mask,
        }
    )
    if not np.array_equal(panel["filter_seed"], expected_seeds):
        raise ValueError("returned PF seeds are not in the requested order")
    _atomic_npz(npz_path, panel)
    metadata = {
        "subject_id": int(subject_id),
        "particle_count": int(particle_count),
        "repeat_count": int(repeat_count),
        "trial_count": int(panel["choice_probability"].shape[1]),
        "state_count": int(panel["filtered_executed_probability"].shape[2]),
        "filter_seeds": [int(value) for value in panel["filter_seed"]],
        "npz_path": _relative(npz_path),
        "npz_sha256": _sha256(npz_path),
    }
    _atomic_json(json_path, metadata)
    return metadata


def _resolved_design(config: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    design = deepcopy(dict(config["pilot"]))
    if smoke:
        subject = int(design["subjects"][0])
        design.update(
            {
                "subjects": [subject],
                "particle_seed_budget": {4: 4, 8: 4},
                "nested_seed_counts": {4: [2, 4], 8: [2, 4]},
                "comparisons": [
                    {
                        "comparison_id": "smoke_R4K4_vs_R8K4",
                        "comparison_role": "same_seed_count",
                        "left": {"particle_count": 4, "seed_count": 4},
                        "right": {"particle_count": 8, "seed_count": 4},
                    }
                ],
                "max_trials": 24,
                "n_jobs": 1,
            }
        )
    design["particle_seed_budget"] = {
        int(key): int(value)
        for key, value in design["particle_seed_budget"].items()
    }
    design["nested_seed_counts"] = {
        int(key): [int(value) for value in values]
        for key, values in design["nested_seed_counts"].items()
    }
    return design


def _write_readme(
    output: Path,
    ensemble: pd.DataFrame,
    comparisons: pd.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    lines = [
        "# 0815 calibrated P0 nested PF convergence pilot",
        "",
        "## Result",
        "",
        (
            f"Pilot subject `{summary['subjects'][0]}` was evaluated with retained "
            "per-seed probability and executed-state arrays."
        ),
        "",
        "| R | K | total R x K | NLL | split choice RMSE | split executed JS | internal pass |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in ensemble.to_dict(orient="records"):
        lines.append(
            "| {particle_count} | {seed_count} | {total_particle_trajectories} | "
            "{choice_nll:.5f} | {split_half_choice_probability_rmse:.5f} | "
            "{split_half_executed_posterior_js:.5f} | {passed} |".format(
                **row,
                passed="yes" if row["all_internal_gates_passed"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "| comparison | role | choice NLL change | choice RMSE | executed JS | pass |",
            "|---|---|---:|---:|---:|:---:|",
        ]
    )
    for row in comparisons.to_dict(orient="records"):
        lines.append(
            "| {comparison_id} | {comparison_role} | {absolute_choice_nll_change:.5f} | "
            "{choice_probability_rmse:.5f} | {executed_posterior_js:.5f} | {passed} |".format(
                **row,
                passed="yes" if row["all_comparison_gates_passed"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "This is an engineering pilot on one frozen subject, not a formal particle-budget selection.",
            "Every cached NPZ retains the individual PF seeds, choice probabilities, filtered executed-state posteriors, ESS trajectories, observed choices, and valid-trial mask.",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    config = load_yaml(config_path)
    design = _resolved_design(config, smoke=bool(args.smoke))
    if args.n_jobs is not None:
        design["n_jobs"] = int(args.n_jobs)
    output = (
        _repo_path(args.output_dir)
        if args.output_dir is not None
        else _repo_path(config["output_dir"])
    )
    if args.smoke:
        output = output / "smoke"
    if args.phase in {"run", "all"} and (output / "summary.json").exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite completed output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    simulation_config_path = _repo_path(config["base_simulation_config"])
    simulation_config = load_yaml(simulation_config_path)
    max_trials_raw = design.get("max_trials")
    max_trials = None if max_trials_raw is None else int(max_trials_raw)

    if args.phase in {"run", "all"}:
        for subject_id in [int(value) for value in design["subjects"]]:
            for particle_count, repeat_count in sorted(
                design["particle_seed_budget"].items()
            ):
                _run_panel(
                    simulation_config_path=simulation_config_path,
                    simulation_config=simulation_config,
                    output=output,
                    subject_id=subject_id,
                    particle_count=particle_count,
                    repeat_count=repeat_count,
                    max_trials=max_trials,
                    base_seed=int(design["base_seed"]),
                    n_jobs=int(design["n_jobs"]),
                    force=bool(args.force),
                )

    if args.phase in {"summarize", "all"}:
        gates = dict(design["stability_gates"])
        ensemble_rows: list[dict[str, Any]] = []
        arrays_by_key: dict[tuple[int, int, int], dict[str, np.ndarray]] = {}
        row_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
        cache_metadata: list[dict[str, Any]] = []
        per_seed_rows: list[dict[str, Any]] = []
        for subject_id in [int(value) for value in design["subjects"]]:
            for particle_count, repeat_count in sorted(
                design["particle_seed_budget"].items()
            ):
                npz_path, json_path = _panel_paths(output, subject_id, particle_count)
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
                if metadata["npz_sha256"] != _sha256(npz_path):
                    raise ValueError(f"cache hash mismatch: {npz_path}")
                panel = _load_panel(npz_path)
                cache_metadata.append(metadata)
                for index, seed in enumerate(panel["filter_seed"]):
                    per_seed_rows.append(
                        {
                            "subject_id": subject_id,
                            "particle_count": particle_count,
                            "repeat_index": int(index),
                            "filter_seed": int(seed),
                            "choice_nll": _choice_nll(
                                panel["choice_probability"][index],
                                panel["observed_choice_index"],
                                panel["valid_trial_mask"],
                            ),
                            "median_post_choice_ess_fraction": float(
                                np.median(panel["post_choice_ess"][index])
                                / float(particle_count)
                            ),
                            "cache_npz": _relative(npz_path),
                            "cache_npz_sha256": metadata["npz_sha256"],
                        }
                    )
                for seed_count in design["nested_seed_counts"][particle_count]:
                    row, arrays = summarize_ensemble(
                        panel,
                        subject_id=subject_id,
                        particle_count=particle_count,
                        seed_count=seed_count,
                        gates=gates,
                    )
                    key = (subject_id, particle_count, seed_count)
                    row_by_key[key] = row
                    arrays_by_key[key] = arrays
                    ensemble_rows.append(row)

        comparison_rows: list[dict[str, Any]] = []
        for subject_id in [int(value) for value in design["subjects"]]:
            for particle_count, seed_counts in sorted(
                design["nested_seed_counts"].items()
            ):
                for lower, upper in zip(seed_counts[:-1], seed_counts[1:]):
                    left_key = (subject_id, particle_count, lower)
                    right_key = (subject_id, particle_count, upper)
                    comparison_rows.append(
                        compare_ensembles(
                            row_by_key[left_key],
                            arrays_by_key[left_key],
                            row_by_key[right_key],
                            arrays_by_key[right_key],
                            comparison_id=f"R{particle_count}_K{lower}_vs_K{upper}",
                            comparison_role="nested_seed_count",
                            gates=gates,
                        )
                    )
            for comparison in design["comparisons"]:
                left = comparison["left"]
                right = comparison["right"]
                left_key = (
                    subject_id,
                    int(left["particle_count"]),
                    int(left["seed_count"]),
                )
                right_key = (
                    subject_id,
                    int(right["particle_count"]),
                    int(right["seed_count"]),
                )
                comparison_rows.append(
                    compare_ensembles(
                        row_by_key[left_key],
                        arrays_by_key[left_key],
                        row_by_key[right_key],
                        arrays_by_key[right_key],
                        comparison_id=str(comparison["comparison_id"]),
                        comparison_role=str(comparison["comparison_role"]),
                        gates=gates,
                    )
                )

        ensemble_frame = pd.DataFrame(ensemble_rows).sort_values(
            ["subject_id", "particle_count", "seed_count"]
        )
        comparison_frame = pd.DataFrame(comparison_rows).sort_values(
            ["subject_id", "comparison_role", "comparison_id"]
        )
        per_seed_frame = pd.DataFrame(per_seed_rows).sort_values(
            ["subject_id", "particle_count", "repeat_index"]
        )
        _atomic_csv(output / "per_seed_summary.csv", per_seed_frame)
        _atomic_csv(output / "ensemble_summary.csv", ensemble_frame)
        _atomic_csv(output / "ensemble_comparisons.csv", comparison_frame)

        reference = config.get("reference_reproduction")
        reference_check = None
        if reference and not args.smoke:
            reference_path = _repo_path(reference["particle_count_summary"])
            reference_frame = pd.read_csv(reference_path)
            subject_id = int(reference["subject_id"])
            particle_count = int(reference["particle_count"])
            seed_count = int(reference["seed_count"])
            old = reference_frame.loc[
                reference_frame["subject_id"].astype(int).eq(subject_id)
                & reference_frame["particle_count"].astype(int).eq(particle_count)
            ].iloc[0]
            new = ensemble_frame.loc[
                ensemble_frame["subject_id"].astype(int).eq(subject_id)
                & ensemble_frame["particle_count"].astype(int).eq(particle_count)
                & ensemble_frame["seed_count"].astype(int).eq(seed_count)
            ].iloc[0]
            fields = {
                "choice_nll": "choice_nll",
                "run_choice_nll_mean": "run_choice_nll_mean",
                "run_choice_nll_sd": "run_choice_nll_sd",
                "split_half_choice_probability_rmse": (
                    "split_half_choice_probability_rmse"
                ),
                "median_post_choice_ess_fraction": (
                    "median_post_choice_ess_fraction"
                ),
            }
            differences = {
                key: abs(float(new[new_key]) - float(old[old_key]))
                for key, (new_key, old_key) in {
                    key: (value, value) for key, value in fields.items()
                }.items()
            }
            tolerance = float(reference.get("absolute_tolerance", 1e-12))
            reference_check = {
                "reference": _relative(reference_path),
                "maximum_absolute_difference": float(max(differences.values())),
                "absolute_tolerance": tolerance,
                "passed": bool(max(differences.values()) <= tolerance),
                "field_differences": differences,
            }
            if not reference_check["passed"]:
                raise ValueError("nested pilot did not reproduce the prior R256 K8 panel")

        summary = {
            "analysis_id": str(config["analysis_id"]),
            "status": "complete_engineering_pilot",
            "subjects": [int(value) for value in design["subjects"]],
            "particle_seed_budget": design["particle_seed_budget"],
            "nested_seed_counts": design["nested_seed_counts"],
            "max_trials": max_trials,
            "stability_gates": gates,
            "internal_gate_pass_count": int(
                ensemble_frame["all_internal_gates_passed"].sum()
            ),
            "internal_gate_row_count": int(len(ensemble_frame)),
            "comparison_gate_pass_count": int(
                comparison_frame["all_comparison_gates_passed"].sum()
            ),
            "comparison_gate_row_count": int(len(comparison_frame)),
            "reference_reproduction": reference_check,
            "smoke": bool(args.smoke),
            "interpretation_boundary": (
                "single-subject engineering pilot; not a formal particle-budget selection"
            ),
        }
        _atomic_json(output / "summary.json", summary)
        _atomic_json(output / "analysis_config_snapshot.json", config)
        manifest = {
            "analysis_id": str(config["analysis_id"]),
            "config": _relative(config_path),
            "config_sha256": _sha256(config_path),
            "base_simulation_config": _relative(simulation_config_path),
            "base_simulation_config_sha256": _sha256(simulation_config_path),
            "runner": _relative(Path(__file__)),
            "runner_sha256": _sha256(Path(__file__)),
            "cache": cache_metadata,
            "summary_sha256": _sha256(output / "summary.json"),
            "ensemble_summary_sha256": _sha256(output / "ensemble_summary.csv"),
            "ensemble_comparisons_sha256": _sha256(
                output / "ensemble_comparisons.csv"
            ),
        }
        _atomic_json(output / "analysis_manifest.json", manifest)
        _write_readme(output, ensemble_frame, comparison_frame, summary)
        print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
