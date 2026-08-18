#!/usr/bin/env python3
"""Run the calibrated 0815 P0 persistent-execution counterfactual pilot.

The bank contains three models so that persistent execution is not silently
confounded with the beta update scope. Every subject/variant uses an identical
ordered PF-seed panel. The primary numerical target is the paired mechanism
delta, not absolute convergence of either provisional architecture.
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
from src.Bayesian_state.simulation.runner import StateModelSimulationRunner  # noqa: E402
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.utils.seeding import stable_seed  # noqa: E402
from src.Bayesian_state.utils.subjects import resolve_subject_config  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "configs/specific_models/"
    "model_0815_p0_persistent_execution_counterfactual_pilot.yaml"
)
EXECUTION_PATH = (
    "modules.hypo_transitions_mod.kwargs."
    "continuous_controller.execution.enabled"
)
BETA_SCOPE_PATH = "modules.beta_mod.kwargs.update_scope"
ALLOWED_VARIANT_PATHS = {EXECUTION_PATH, BETA_SCOPE_PATH}


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


def _set_path(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    current = root
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot traverse non-mapping at {part!r} in {path!r}")
        current = child
    current[parts[-1]] = deepcopy(value)


def _flatten(root: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in root.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten(value, path))
        else:
            flattened[path] = value
    return flattened


def build_variant_engine(
    base_engine: Mapping[str, Any], variant: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the only two permitted structural changes for this audit."""
    execution_enabled = bool(variant["persistent_execution_enabled"])
    beta_scope = str(variant["beta_update_scope"])
    if beta_scope not in {"active_hypotheses", "executed_hypothesis"}:
        raise ValueError(f"unsupported beta update scope: {beta_scope!r}")
    if not execution_enabled and beta_scope == "executed_hypothesis":
        raise ValueError("executed beta scope is undefined when execution is off")
    engine = deepcopy(dict(base_engine))
    _set_path(engine, EXECUTION_PATH, execution_enabled)
    _set_path(engine, BETA_SCOPE_PATH, beta_scope)
    changed = {
        path
        for path in set(_flatten(base_engine)) | set(_flatten(engine))
        if _flatten(base_engine).get(path) != _flatten(engine).get(path)
    }
    if not changed.issubset(ALLOWED_VARIANT_PATHS):
        raise RuntimeError(f"counterfactual changed undeclared paths: {sorted(changed)}")
    return engine


def validate_variant_bank(
    base_engine: Mapping[str, Any], variants: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    variant_by_id: dict[str, dict[str, Any]] = {}
    signatures: set[tuple[bool, str]] = set()
    for raw in variants:
        variant_id = str(raw["variant_id"])
        if variant_id in variant_by_id:
            raise ValueError(f"duplicate variant_id: {variant_id}")
        engine = build_variant_engine(base_engine, raw)
        signature = (
            bool(raw["persistent_execution_enabled"]),
            str(raw["beta_update_scope"]),
        )
        if signature in signatures:
            raise ValueError(f"duplicate counterfactual signature: {signature}")
        signatures.add(signature)
        variant_by_id[variant_id] = engine
    required = {
        (False, "active_hypotheses"),
        (True, "active_hypotheses"),
        (True, "executed_hypothesis"),
    }
    if signatures != required:
        raise ValueError(f"variant bank must contain exactly {sorted(required)}")
    return variant_by_id


def _filter_seeds(base_seed: int, subject_id: int, seed_count: int) -> np.ndarray:
    return np.asarray(
        [
            stable_seed(
                {
                    "seed_role": "model0815_p0_persistent_execution_counterfactual",
                    "base_seed": int(base_seed),
                    "subject_id": int(subject_id),
                    "repeat_index": int(index),
                }
            )
            for index in range(int(seed_count))
        ],
        dtype=np.uint64,
    )


def _choice_nll(probability: np.ndarray, choices: np.ndarray, valid: np.ndarray) -> float:
    keep = (
        np.asarray(valid, dtype=bool)
        & (choices >= 0)
        & (choices < probability.shape[1])
        & np.all(np.isfinite(probability), axis=1)
    )
    rows = np.flatnonzero(keep)
    selected = probability[rows, choices[rows]]
    return float(np.mean(-np.log(np.clip(selected, 1e-12, 1.0))))


def _choice_brier(probability: np.ndarray, choices: np.ndarray, valid: np.ndarray) -> float:
    rows = np.flatnonzero(valid)
    target = np.zeros_like(probability[rows])
    target[np.arange(rows.size), choices[rows]] = 1.0
    return float(np.mean(np.sum(np.square(probability[rows] - target), axis=1)))


def _mean_js(first: np.ndarray, second: np.ndarray) -> float:
    left = np.clip(np.asarray(first, dtype=float), 0.0, None)
    right = np.clip(np.asarray(second, dtype=float), 0.0, None)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("JS inputs must have equal two-dimensional shapes")
    left /= np.sum(left, axis=1, keepdims=True)
    right /= np.sum(right, axis=1, keepdims=True)
    midpoint = 0.5 * (left + right)

    def kl(values: np.ndarray) -> np.ndarray:
        output = np.zeros_like(values)
        mask = values > 0.0
        output[mask] = values[mask] * np.log(
            values[mask] / np.clip(midpoint[mask], 1e-12, None)
        )
        return np.sum(output, axis=1)

    return float(np.mean(0.5 * kl(left) + 0.5 * kl(right)))


def validate_panel(panel: Mapping[str, Any]) -> dict[str, np.ndarray]:
    probability = np.asarray(panel["choice_probability"], dtype=float)
    prior = np.asarray(panel["marginal_prior"], dtype=float)
    if probability.ndim != 3 or probability.shape[0] < 2 or probability.shape[2] != 2:
        raise ValueError("choice_probability must have shape (seeds, trials, 2)")
    if prior.ndim != 3 or prior.shape[:2] != probability.shape[:2]:
        raise ValueError("marginal_prior must share seed and trial dimensions")
    for name, values in (("choice_probability", probability), ("marginal_prior", prior)):
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"{name} must be finite and non-negative")
        sums = np.sum(values, axis=2, keepdims=True)
        if np.any(sums <= 0.0):
            raise ValueError(f"{name} contains zero-mass rows")
        if name == "choice_probability":
            probability = values / sums
        else:
            prior = values / sums
    seed_n, trial_n = probability.shape[:2]
    output = {
        "choice_probability": probability,
        "marginal_prior": prior,
        "pre_choice_ess": np.asarray(panel["pre_choice_ess"], dtype=float),
        "post_choice_ess": np.asarray(panel["post_choice_ess"], dtype=float),
        "resampled": np.asarray(panel["resampled"], dtype=bool),
        "predictive_strategy_exploit": np.asarray(
            panel["predictive_strategy_exploit"], dtype=float
        ),
        "predictive_strategy_local_explore": np.asarray(
            panel["predictive_strategy_local_explore"], dtype=float
        ),
        "predictive_strategy_global_explore": np.asarray(
            panel["predictive_strategy_global_explore"], dtype=float
        ),
        "predictive_execution_switch_event_probability": np.asarray(
            panel["predictive_execution_switch_event_probability"], dtype=float
        ),
        "predictive_execution_dwell_trials": np.asarray(
            panel["predictive_execution_dwell_trials"], dtype=float
        ),
        "filter_seed": np.asarray(panel["filter_seed"], dtype=np.uint64).reshape(-1),
        "repeat_index": np.asarray(panel["repeat_index"], dtype=int).reshape(-1),
        "observed_choice_index": np.asarray(
            panel["observed_choice_index"], dtype=int
        ).reshape(-1),
        "valid_trial_mask": np.asarray(panel["valid_trial_mask"], dtype=bool).reshape(-1),
    }
    for name in (
        "pre_choice_ess",
        "post_choice_ess",
        "resampled",
        "predictive_strategy_exploit",
        "predictive_strategy_local_explore",
        "predictive_strategy_global_explore",
        "predictive_execution_switch_event_probability",
        "predictive_execution_dwell_trials",
    ):
        if output[name].shape != (seed_n, trial_n):
            raise ValueError(f"{name} has invalid shape {output[name].shape}")
    if not np.all(np.isfinite(output["pre_choice_ess"])) or not np.all(
        np.isfinite(output["post_choice_ess"])
    ):
        raise ValueError("ESS arrays must be finite")
    if output["filter_seed"].size != seed_n or np.unique(output["filter_seed"]).size != seed_n:
        raise ValueError("filter seeds must be unique and match the panel")
    if not np.array_equal(output["repeat_index"], np.arange(seed_n)):
        raise ValueError("repeat indices must be contiguous")
    if output["observed_choice_index"].size != trial_n or output["valid_trial_mask"].size != trial_n:
        raise ValueError("observed arrays must match the trial dimension")
    return output


def _cache_paths(output: Path, subject_id: int, variant_id: str) -> tuple[Path, Path]:
    stem = f"subject_{int(subject_id)}_{variant_id}"
    return output / "cache" / f"{stem}.npz", output / "cache" / f"{stem}.json"


def _load_panel(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return validate_panel({key: bundle[key] for key in bundle.files})


def _run_panel(
    *,
    simulation_config_path: Path,
    simulation_config: Mapping[str, Any],
    output: Path,
    subject_id: int,
    variant: Mapping[str, Any],
    particle_count: int,
    seed_count: int,
    trials_per_subject: int,
    base_seed: int,
    n_jobs: int,
    force: bool,
) -> dict[str, Any]:
    variant_id = str(variant["variant_id"])
    npz_path, json_path = _cache_paths(output, subject_id, variant_id)
    expected_seeds = _filter_seeds(base_seed, subject_id, seed_count)
    if npz_path.exists() and json_path.exists() and not force:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        panel = _load_panel(npz_path)
        if _sha256(npz_path) != metadata["npz_sha256"]:
            raise ValueError(f"cache hash mismatch: {npz_path}")
        if not np.array_equal(panel["filter_seed"], expected_seeds):
            raise ValueError(f"cache seed panel differs from the design: {npz_path}")
        return metadata

    subject_cfg = resolve_subject_config(simulation_config, subject_id)
    base_engine = resolve_engine_config(
        subject_cfg, simulation_config_path.parent, subject_id=subject_id
    )
    fixed = {
        **infer_fixed_hyperparams_from_engine_config(base_engine),
        **dict(subject_cfg.get("fixed_hyperparams") or {}),
    }
    base_engine = apply_fixed_hyperparams_to_engine_config(base_engine, fixed)
    validate_variant_bank(base_engine, [
        {
            "variant_id": "off",
            "persistent_execution_enabled": False,
            "beta_update_scope": "active_hypotheses",
        },
        {
            "variant_id": "on_active",
            "persistent_execution_enabled": True,
            "beta_update_scope": "active_hypotheses",
        },
        {
            "variant_id": "on_executed",
            "persistent_execution_enabled": True,
            "beta_update_scope": "executed_hypothesis",
        },
    ])
    engine = build_variant_engine(base_engine, variant)
    engine.setdefault("inference", {})["particle_count"] = int(particle_count)
    resolved_engine_path = output / "resolved_engines" / f"subject_{subject_id}_{variant_id}.json"
    _atomic_json(resolved_engine_path, engine)

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
        simulation_repeats=int(seed_count),
        fixed_hyperparams=fixed,
        window_size=resolve_window_size(subject_cfg, subject_id, [subject_id]),
        stop_at=float(subject_cfg.get("stop_at", 1.0)),
        max_trials=int(trials_per_subject),
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
    if len(raw_runs) != seed_count:
        raise RuntimeError("counterfactual did not return every requested PF seed")

    stack_keys = (
        "marginal_prior",
        "pre_choice_ess",
        "post_choice_ess",
        "resampled",
        "predictive_strategy_exploit",
        "predictive_strategy_local_explore",
        "predictive_strategy_global_explore",
        "predictive_execution_switch_event_probability",
        "predictive_execution_dwell_trials",
    )
    stacked: dict[str, list[np.ndarray]] = {key: [] for key in stack_keys}
    probabilities: list[np.ndarray] = []
    observed_choices = None
    valid_mask = None
    observed_seeds: list[int] = []
    for run in raw_runs:
        metrics = run["metrics_by_mode"][selection_mode]
        current_choices = np.asarray(metrics["observed_choice_index"], dtype=int)
        current_valid = np.asarray(metrics["valid_trial_mask"], dtype=bool)
        if observed_choices is None:
            observed_choices = current_choices
            valid_mask = current_valid
        elif not np.array_equal(observed_choices, current_choices) or not np.array_equal(
            valid_mask, current_valid
        ):
            raise ValueError("observed data changed across paired PF seeds")
        probabilities.append(np.asarray(metrics["pred_category_probs"], dtype=float))
        state_log = run.get("state_log") or {}
        for key in stack_keys:
            stacked[key].append(np.asarray(state_log[key]))
        observed_seeds.append(int(run["trajectory_seed"]))
    if observed_choices is None or valid_mask is None:
        raise RuntimeError("counterfactual panel returned no observed trials")
    panel = validate_panel(
        {
            "choice_probability": np.stack(probabilities),
            **{key: np.stack(values) for key, values in stacked.items()},
            "filter_seed": np.asarray(observed_seeds, dtype=np.uint64),
            "repeat_index": np.arange(seed_count, dtype=int),
            "observed_choice_index": observed_choices,
            "valid_trial_mask": valid_mask,
        }
    )
    if not np.array_equal(panel["filter_seed"], expected_seeds):
        raise ValueError("PF seeds were returned in an unexpected order")
    _atomic_npz(npz_path, panel)
    metadata = {
        "subject_id": int(subject_id),
        "variant_id": variant_id,
        "variant_role": str(variant.get("role", "")),
        "persistent_execution_enabled": bool(variant["persistent_execution_enabled"]),
        "beta_update_scope": str(variant["beta_update_scope"]),
        "particle_count": int(particle_count),
        "seed_count": int(seed_count),
        "trial_count": int(panel["choice_probability"].shape[1]),
        "valid_choice_trial_count": int(np.sum(panel["valid_trial_mask"])),
        "filter_seeds": [int(value) for value in panel["filter_seed"]],
        "resolved_engine": _relative(resolved_engine_path),
        "resolved_engine_sha256": _sha256(resolved_engine_path),
        "npz_path": _relative(npz_path),
        "npz_sha256": _sha256(npz_path),
    }
    _atomic_json(json_path, metadata)
    return metadata


def summarize_variant(
    panel: Mapping[str, Any], *, subject_id: int, variant_id: str, particle_count: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    values = validate_panel(panel)
    probability = values["choice_probability"]
    choices = values["observed_choice_index"]
    valid = values["valid_trial_mask"]
    run_nll = np.asarray([_choice_nll(row, choices, valid) for row in probability])
    run_brier = np.asarray([_choice_brier(row, choices, valid) for row in probability])
    mean_probability = np.mean(probability, axis=0)
    mean_prior = np.mean(values["marginal_prior"], axis=0)
    row = {
        "subject_id": int(subject_id),
        "variant_id": str(variant_id),
        "particle_count": int(particle_count),
        "seed_count": int(probability.shape[0]),
        "ensemble_choice_nll": _choice_nll(mean_probability, choices, valid),
        "run_choice_nll_mean": float(np.mean(run_nll)),
        "run_choice_nll_sd": float(np.std(run_nll, ddof=1)),
        "ensemble_choice_brier": _choice_brier(mean_probability, choices, valid),
        "run_choice_brier_mean": float(np.mean(run_brier)),
        "median_post_choice_ess_fraction": float(
            np.median(values["post_choice_ess"] / float(particle_count))
        ),
        "mean_resampling_fraction": float(np.mean(values["resampled"])),
        "mean_predictive_exploit": float(np.mean(values["predictive_strategy_exploit"][:, valid])),
        "mean_predictive_local_explore": float(np.mean(values["predictive_strategy_local_explore"][:, valid])),
        "mean_predictive_global_explore": float(np.mean(values["predictive_strategy_global_explore"][:, valid])),
        "mean_execution_switch_event_probability": float(
            np.mean(values["predictive_execution_switch_event_probability"][:, valid])
        ),
        "mean_execution_dwell_trials": float(
            np.mean(values["predictive_execution_dwell_trials"][:, valid])
        ),
    }
    arrays = {
        "mean_choice_probability": mean_probability,
        "mean_marginal_prior": mean_prior,
        "run_choice_nll": run_nll,
        "run_choice_brier": run_brier,
        "observed_choice_index": choices,
        "valid_trial_mask": valid,
        "filter_seed": values["filter_seed"],
    }
    return row, arrays


def summarize_contrast(
    comparator_row: Mapping[str, Any],
    comparator: Mapping[str, np.ndarray],
    mechanism_row: Mapping[str, Any],
    mechanism: Mapping[str, np.ndarray],
    *,
    contrast: Mapping[str, Any],
    numerical_gates: Mapping[str, float],
    practical_rule: Mapping[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not np.array_equal(comparator["filter_seed"], mechanism["filter_seed"]):
        raise ValueError("counterfactual variants do not share ordered PF seeds")
    if not np.array_equal(
        comparator["observed_choice_index"], mechanism["observed_choice_index"]
    ) or not np.array_equal(comparator["valid_trial_mask"], mechanism["valid_trial_mask"]):
        raise ValueError("counterfactual variants do not share observed trials")
    delta = np.asarray(comparator["run_choice_nll"], dtype=float) - np.asarray(
        mechanism["run_choice_nll"], dtype=float
    )
    k = int(delta.size)
    if k < 2 or k % 2:
        raise ValueError("paired counterfactual requires an even seed count >= 2")
    half = k // 2
    delta_mean = float(np.mean(delta))
    delta_sd = float(np.std(delta, ddof=1))
    delta_mcse = float(delta_sd / np.sqrt(k))
    first_half = float(np.mean(delta[:half]))
    second_half = float(np.mean(delta[half:]))
    half_difference = abs(first_half - second_half)
    valid = np.asarray(comparator["valid_trial_mask"], dtype=bool)
    probability_rmse = float(
        np.sqrt(
            np.mean(
                np.square(
                    comparator["mean_choice_probability"][valid]
                    - mechanism["mean_choice_probability"][valid]
                )
            )
        )
    )
    geometry_js = _mean_js(
        comparator["mean_marginal_prior"][valid],
        mechanism["mean_marginal_prior"][valid],
    )
    practical_threshold = max(
        float(practical_rule["baseline_mean_nll_fraction"])
        * float(comparator_row["ensemble_choice_nll"]),
        float(practical_rule["paired_seed_sd_multiplier"]) * delta_sd,
    )
    stable = bool(
        delta_mcse <= float(numerical_gates["maximum_paired_mean_nll_mcse"])
        and half_difference
        <= float(numerical_gates["maximum_disjoint_half_delta_difference"])
    )
    row = {
        "subject_id": int(comparator_row["subject_id"]),
        "contrast_id": str(contrast["contrast_id"]),
        "interpretation": str(contrast["interpretation"]),
        "comparator_variant": str(comparator_row["variant_id"]),
        "mechanism_variant": str(mechanism_row["variant_id"]),
        "particle_count": int(comparator_row["particle_count"]),
        "seed_count": k,
        "paired_delta_mean_nll": delta_mean,
        "paired_delta_mean_nll_sd": delta_sd,
        "paired_delta_mean_nll_mcse": delta_mcse,
        "paired_delta_first_half": first_half,
        "paired_delta_second_half": second_half,
        "absolute_half_delta_difference": half_difference,
        "ensemble_delta_mean_nll": float(comparator_row["ensemble_choice_nll"])
        - float(mechanism_row["ensemble_choice_nll"]),
        "ensemble_choice_probability_rmse": probability_rmse,
        "predictive_geometry_prior_js": geometry_js,
        "practical_effect_threshold": practical_threshold,
        "positive_seed_fraction": float(np.mean(delta > 0.0)),
        "numerically_stable": stable,
        "exceeds_practical_threshold": bool(delta_mean > practical_threshold),
    }
    seed_rows = [
        {
            "subject_id": int(comparator_row["subject_id"]),
            "contrast_id": str(contrast["contrast_id"]),
            "repeat_index": int(index),
            "filter_seed": int(seed),
            "comparator_mean_nll": float(comparator["run_choice_nll"][index]),
            "mechanism_mean_nll": float(mechanism["run_choice_nll"][index]),
            "paired_delta_mean_nll": float(delta[index]),
        }
        for index, seed in enumerate(comparator["filter_seed"])
    ]
    return row, seed_rows


def _resolved_design(config: Mapping[str, Any], smoke: bool) -> dict[str, Any]:
    design = deepcopy(dict(config["design"]))
    design["subjects"] = [int(value) for value in design["subjects"]]
    design["seed_count_by_subject"] = {
        int(key): int(value) for key, value in design["seed_count_by_subject"].items()
    }
    if set(design["subjects"]) != set(design["seed_count_by_subject"]):
        raise ValueError("seed_count_by_subject must cover exactly the selected subjects")
    if smoke:
        subject = design["subjects"][0]
        design.update(
            {
                "subjects": [subject],
                "seed_count_by_subject": {subject: 2},
                "particle_count": 8,
                "trials_per_subject": 24,
                "n_jobs": 1,
            }
        )
    return design


def _write_readme(
    output: Path,
    variant_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    lines = [
        "# 0815 P0 persistent-execution full counterfactual pilot",
        "",
        "## Design",
        "",
        (
            "This calibrated-boundary pilot separates persistent execution from "
            "executed-rule-only beta updating with a three-model bank. Positive "
            "delta NLL favors the mechanism-side variant."
        ),
        "",
        "| contrast | mean subject delta NLL | subjects numerically stable | subjects practically positive |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate.to_dict(orient="records"):
        lines.append(
            "| {contrast_id} | {subject_mean_paired_delta_mean_nll:.5f} | "
            "{numerically_stable_subject_count}/{subject_count} | "
            "{practically_positive_subject_count}/{subject_count} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Subject-level paired effects",
            "",
            "| subject | contrast | delta NLL | MCSE | half difference | choice RMSE | geometry JS | stable | practical |",
            "|---:|---|---:|---:|---:|---:|---:|:---:|:---:|",
        ]
    )
    for row in contrast_summary.to_dict(orient="records"):
        lines.append(
            "| {subject_id} | {contrast_id} | {paired_delta_mean_nll:.5f} | "
            "{paired_delta_mean_nll_mcse:.5f} | {absolute_half_delta_difference:.5f} | "
            "{ensemble_choice_probability_rmse:.5f} | {predictive_geometry_prior_js:.5f} | "
            "{stable} | {practical} |".format(
                **row,
                stable="yes" if row["numerically_stable"] else "no",
                practical="yes" if row["exceeds_practical_threshold"] else "no",
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            (
                f"This is a {len(summary['subjects'])}-subject architecture pilot with frozen "
                "calibration, not a final mechanism decision or PF-budget selection. "
                "Absolute PF convergence is deliberately not required here; the numerical "
                "gate applies to the paired mechanism delta."
            ),
            "",
            "The model bank, raw per-seed arrays, paired deltas, resolved engine configurations, and hashes are retained for independent recomputation.",
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
    output = _repo_path(args.output_dir) if args.output_dir else _repo_path(config["output_dir"])
    if args.smoke:
        output = output / "smoke"
    if args.phase in {"run", "all"} and (output / "summary.json").exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite completed output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    simulation_config_path = _repo_path(config["base_simulation_config"])
    simulation_config = load_yaml(simulation_config_path)
    variants = [dict(value) for value in design["variants"]]
    variant_ids = {str(value["variant_id"]) for value in variants}
    for contrast in design["contrasts"]:
        if not {
            str(contrast["comparator_variant"]),
            str(contrast["mechanism_variant"]),
        }.issubset(variant_ids):
            raise ValueError(f"contrast refers to an unknown variant: {contrast}")

    if args.phase in {"run", "all"}:
        for subject_id in design["subjects"]:
            for variant in variants:
                _run_panel(
                    simulation_config_path=simulation_config_path,
                    simulation_config=simulation_config,
                    output=output,
                    subject_id=subject_id,
                    variant=variant,
                    particle_count=int(design["particle_count"]),
                    seed_count=int(design["seed_count_by_subject"][subject_id]),
                    trials_per_subject=int(design["trials_per_subject"]),
                    base_seed=int(design["base_seed"]),
                    n_jobs=int(design["n_jobs"]),
                    force=bool(args.force),
                )

    if args.phase in {"summarize", "all"}:
        cache_metadata: list[dict[str, Any]] = []
        variant_rows: list[dict[str, Any]] = []
        per_seed_variant_rows: list[dict[str, Any]] = []
        arrays: dict[tuple[int, str], dict[str, np.ndarray]] = {}
        rows: dict[tuple[int, str], dict[str, Any]] = {}
        for subject_id in design["subjects"]:
            expected_seeds = _filter_seeds(
                int(design["base_seed"]),
                subject_id,
                int(design["seed_count_by_subject"][subject_id]),
            )
            for variant in variants:
                variant_id = str(variant["variant_id"])
                npz_path, json_path = _cache_paths(output, subject_id, variant_id)
                metadata = json.loads(json_path.read_text(encoding="utf-8"))
                if metadata["npz_sha256"] != _sha256(npz_path):
                    raise ValueError(f"cache hash mismatch: {npz_path}")
                panel = _load_panel(npz_path)
                if not np.array_equal(panel["filter_seed"], expected_seeds):
                    raise ValueError("variant cache violates paired-seed design")
                cache_metadata.append(metadata)
                row, retained = summarize_variant(
                    panel,
                    subject_id=subject_id,
                    variant_id=variant_id,
                    particle_count=int(design["particle_count"]),
                )
                key = (subject_id, variant_id)
                rows[key] = row
                arrays[key] = retained
                variant_rows.append(row)
                for index, seed in enumerate(retained["filter_seed"]):
                    per_seed_variant_rows.append(
                        {
                            "subject_id": subject_id,
                            "variant_id": variant_id,
                            "repeat_index": index,
                            "filter_seed": int(seed),
                            "choice_nll": float(retained["run_choice_nll"][index]),
                            "choice_brier": float(retained["run_choice_brier"][index]),
                            "cache_npz": _relative(npz_path),
                            "cache_npz_sha256": metadata["npz_sha256"],
                        }
                    )

        contrast_rows: list[dict[str, Any]] = []
        seed_contrast_rows: list[dict[str, Any]] = []
        for subject_id in design["subjects"]:
            for contrast in design["contrasts"]:
                comparator_key = (subject_id, str(contrast["comparator_variant"]))
                mechanism_key = (subject_id, str(contrast["mechanism_variant"]))
                row, seed_rows = summarize_contrast(
                    rows[comparator_key],
                    arrays[comparator_key],
                    rows[mechanism_key],
                    arrays[mechanism_key],
                    contrast=contrast,
                    numerical_gates=design["numerical_gates"],
                    practical_rule=design["practical_effect_rule"],
                )
                contrast_rows.append(row)
                seed_contrast_rows.extend(seed_rows)

        variant_frame = pd.DataFrame(variant_rows).sort_values(["subject_id", "variant_id"])
        per_seed_variant_frame = pd.DataFrame(per_seed_variant_rows).sort_values(
            ["subject_id", "variant_id", "repeat_index"]
        )
        contrast_frame = pd.DataFrame(contrast_rows).sort_values(
            ["contrast_id", "subject_id"]
        )
        seed_contrast_frame = pd.DataFrame(seed_contrast_rows).sort_values(
            ["contrast_id", "subject_id", "repeat_index"]
        )
        aggregate_rows: list[dict[str, Any]] = []
        for contrast_id, group in contrast_frame.groupby("contrast_id", sort=True):
            aggregate_rows.append(
                {
                    "contrast_id": contrast_id,
                    "subject_count": int(len(group)),
                    "subject_mean_paired_delta_mean_nll": float(
                        group["paired_delta_mean_nll"].mean()
                    ),
                    "subject_median_paired_delta_mean_nll": float(
                        group["paired_delta_mean_nll"].median()
                    ),
                    "subject_min_paired_delta_mean_nll": float(
                        group["paired_delta_mean_nll"].min()
                    ),
                    "subject_max_paired_delta_mean_nll": float(
                        group["paired_delta_mean_nll"].max()
                    ),
                    "median_paired_delta_mean_nll_mcse": float(
                        group["paired_delta_mean_nll_mcse"].median()
                    ),
                    "numerically_stable_subject_count": int(
                        group["numerically_stable"].sum()
                    ),
                    "practically_positive_subject_count": int(
                        group["exceeds_practical_threshold"].sum()
                    ),
                    "positive_subject_count": int(
                        (group["paired_delta_mean_nll"] > 0.0).sum()
                    ),
                }
            )
        aggregate_frame = pd.DataFrame(aggregate_rows)
        _atomic_csv(output / "variant_summary.csv", variant_frame)
        _atomic_csv(output / "per_seed_variant_scores.csv", per_seed_variant_frame)
        _atomic_csv(output / "subject_contrast_summary.csv", contrast_frame)
        _atomic_csv(output / "paired_seed_contrasts.csv", seed_contrast_frame)
        _atomic_csv(output / "contrast_summary.csv", aggregate_frame)

        summary = {
            "analysis_id": str(config["analysis_id"]),
            "status": "complete_engineering_pilot",
            "subjects": design["subjects"],
            "trials_per_subject": int(design["trials_per_subject"]),
            "particle_count": int(design["particle_count"]),
            "seed_count_by_subject": design["seed_count_by_subject"],
            "variant_count": len(variants),
            "contrast_count": len(design["contrasts"]),
            "numerical_gates": design["numerical_gates"],
            "practical_effect_rule": design["practical_effect_rule"],
            "mapping_enabled": False,
            "smoke": bool(args.smoke),
            "interpretation_boundary": (
                "provisional fixed-parameter architecture pilot; paired contrast "
                "convergence only; not final retention or PF budget"
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
            "output_hashes": {
                name: _sha256(output / name)
                for name in (
                    "summary.json",
                    "variant_summary.csv",
                    "per_seed_variant_scores.csv",
                    "subject_contrast_summary.csv",
                    "paired_seed_contrasts.csv",
                    "contrast_summary.csv",
                )
            },
        }
        _atomic_json(output / "analysis_manifest.json", manifest)
        _write_readme(output, variant_frame, contrast_frame, aggregate_frame, summary)
        print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
