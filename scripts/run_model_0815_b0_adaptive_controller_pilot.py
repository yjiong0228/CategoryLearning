#!/usr/bin/env python3
"""Screen adaptive search against a training-matched fixed-search baseline.

The adaptive branch is the deliberately minimal Model 0815 B0 architecture.
For each subject, its fixed comparator receives the same mean probability of
any workspace replacement and the same mean global-search mixture measured on
the adaptive branch's training trials.  Held-out trials therefore test the
value of history-dependent control, rather than a difference in mean search
intensity.
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
    / "configs/specific_models/model_0815_b0_adaptive_controller_pilot.yaml"
)
ADAPTIVE_CLASS = (
    "src.Bayesian_state.model.modules.hypothesis_transition."
    "dynamic_adaptive_control.DynamicAdaptiveControlHypothesisTransitionModule"
)
FIXED_CLASS = (
    "src.Bayesian_state.model.modules.hypothesis_transition."
    "fixed_strategy.FixedWorkspaceHypothesisTransitionModule"
)
SINGLE_MEMORY_CLASS = "src.Bayesian_state.model.modules.memory.BayesianMemoryModule"
DUAL_MEMORY_CLASS = "src.Bayesian_state.model.modules.memory.DualMemoryModule"
LEGACY_B0_PROFILE = "legacy_b0_fixed_beta_m_off"
H1_PROFILE = "h1_leaky_m_unified_dynamic_beta"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("run", "summarize", "all"), default="all")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-jobs", type=int)
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _choice_nll(probability: np.ndarray, choices: np.ndarray, mask: np.ndarray) -> float:
    keep = (
        np.asarray(mask, dtype=bool)
        & (choices >= 0)
        & (choices < probability.shape[1])
        & np.all(np.isfinite(probability), axis=1)
    )
    rows = np.flatnonzero(keep)
    if rows.size == 0:
        raise ValueError("choice NLL mask contains no valid trials")
    selected = probability[rows, choices[rows]]
    return float(np.mean(-np.log(np.clip(selected, 1e-12, 1.0))))


def _choice_brier(
    probability: np.ndarray, choices: np.ndarray, mask: np.ndarray
) -> float:
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        raise ValueError("choice Brier mask contains no valid trials")
    target = np.zeros_like(probability[rows])
    target[np.arange(rows.size), choices[rows]] = 1.0
    return float(np.mean(np.sum(np.square(probability[rows] - target), axis=1)))


def _mean_js(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    left = np.clip(np.asarray(first, dtype=float)[mask], 0.0, None)
    right = np.clip(np.asarray(second, dtype=float)[mask], 0.0, None)
    if left.shape != right.shape or left.ndim != 2 or left.shape[0] == 0:
        raise ValueError("JS inputs must have equal non-empty two-dimensional shapes")
    left /= np.sum(left, axis=1, keepdims=True)
    right /= np.sum(right, axis=1, keepdims=True)
    midpoint = 0.5 * (left + right)

    def kl(values: np.ndarray) -> np.ndarray:
        terms = np.zeros_like(values)
        positive = values > 0.0
        terms[positive] = values[positive] * np.log(
            values[positive] / np.clip(midpoint[positive], 1e-12, None)
        )
        return np.sum(terms, axis=1)

    return float(np.mean(0.5 * kl(left) + 0.5 * kl(right)))


def _flatten(root: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in root.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten(value, path))
        else:
            flattened[path] = value
    return flattened


def _architecture_profile(engine: Mapping[str, Any]) -> str:
    provenance = engine.get("provenance", {})
    return str(provenance.get("architecture_profile", LEGACY_B0_PROFILE))


def validate_minimal_adaptive_engine(engine: Mapping[str, Any]) -> None:
    """Validate either frozen controller-screen architecture.

    The legacy B0 profile is retained so its completed result remains exactly
    reproducible.  H1 changes only the common cognitive baseline: it uses one
    fading memory channel and restores the original unified, dynamic rule
    confidence beta.  Controller comparators are still allowed to differ only
    inside the hypothesis-transition module.
    """
    modules = engine["modules"]
    if any("mapping" in str(name).lower() for name in modules):
        raise ValueError("controller screen must use fixed mapping with no mapping module")

    profile = _architecture_profile(engine)
    memory = modules["memory_mod"]
    beta = modules["beta_mod"]["kwargs"]
    likelihood = engine.get("likelihood", {})
    if profile == LEGACY_B0_PROFILE:
        if str(memory["class"]) != SINGLE_MEMORY_CLASS:
            raise ValueError("legacy B0 controller screen requires BayesianMemoryModule")
        if float(beta["decrease_rate"]) != 0.0 or float(beta["correct_additive"]) != 0.0:
            raise ValueError("legacy B0 controller screen requires fixed action beta")
        if str(likelihood.get("beta_source", "action")) != "fixed":
            raise ValueError("legacy B0 requires an independent fixed evidence beta")
    elif profile == H1_PROFILE:
        if str(memory["class"]) != DUAL_MEMORY_CLASS:
            raise ValueError("H1 requires the fading-memory implementation")
        memory_kwargs = memory.get("kwargs", {})
        if float(memory_kwargs.get("w0", np.nan)) != 0.0:
            raise ValueError("H1 must use one fading channel (w0=0)")
        gamma = float(memory_kwargs.get("gamma", np.nan))
        if not np.isfinite(gamma) or not 0.0 <= gamma < 1.0:
            raise ValueError("H1 fading memory requires gamma in [0, 1)")
        if float(beta["decrease_rate"]) <= 0.0 or float(beta["correct_additive"]) <= 0.0:
            raise ValueError("H1 requires feedback-responsive dynamic beta")
        if str(likelihood.get("beta_source")) != "action":
            raise ValueError("H1 requires one beta shared by evidence and choice")
    else:
        raise ValueError(f"unsupported controller-screen architecture profile: {profile}")

    if bool(beta.get("use_prior_scaling", False)):
        raise ValueError("controller screen beta cannot use prior scaling")
    if str(beta.get("update_scope")) != "active_hypotheses":
        raise ValueError("execution-off controller screen requires active_hypotheses beta scope")

    transition = modules["hypo_transitions_mod"]
    if str(transition["class"]) != ADAPTIVE_CLASS:
        raise ValueError("adaptive controller screen must use DynamicAdaptiveControl transition")
    controller = transition["kwargs"]["continuous_controller"]
    if str(controller.get("mode")) != "failure_accumulator_v2":
        raise ValueError("adaptive controller screen requires failure_accumulator_v2")
    if bool(controller.get("execution", {}).get("enabled", False)):
        raise ValueError("controller screen requires persistent execution off")

    readout = engine.get("choice_readout", {}).get("kwargs", {})
    if str(readout.get("method")) != "expectation":
        raise ValueError("controller screen requires expectation readout")
    if float(readout.get("power", 1.0)) != 1.0:
        raise ValueError("controller screen requires readout power one")
    if float(readout.get("strategy_confidence_gain", 0.0)) != 0.0:
        raise ValueError("controller screen requires strategy confidence off")
    noise = engine.get("output_noise", {}).get("kwargs", {})
    if any(
        float(noise.get(key, 0.0)) != 0.0
        for key in ("base_lapse", "post_error_lapse", "low_accuracy_lapse")
    ):
        raise ValueError("controller screen requires output lapse off")


def event_probability_to_slot_rate(probability: float, capacity: int) -> float:
    probability = float(probability)
    capacity = int(capacity)
    if not 0.0 <= probability <= 1.0 or capacity <= 0:
        raise ValueError("invalid event probability or workspace capacity")
    return float(1.0 - (1.0 - probability) ** (1.0 / capacity))


def build_training_matched_fixed_engine(
    adaptive_engine: Mapping[str, Any],
    *,
    event_probability: float,
    global_search: float,
) -> dict[str, Any]:
    """Replace only the adaptive controller block with fixed ``m`` and ``g``."""
    validate_minimal_adaptive_engine(adaptive_engine)
    fixed = deepcopy(dict(adaptive_engine))
    adaptive_transition = adaptive_engine["modules"]["hypo_transitions_mod"]
    adaptive_kwargs = adaptive_transition["kwargs"]
    capacity = int(adaptive_kwargs["capacity"])
    common_keys = {
        "capacity",
        "init_hypotheses",
        "tau_local",
        "epsilon",
        "selection_strategy",
        "prior_assignment",
    }
    fixed_kwargs = {
        key: deepcopy(value)
        for key, value in adaptive_kwargs.items()
        if key in common_keys
    }
    fixed_kwargs["m"] = event_probability_to_slot_rate(
        event_probability, capacity
    )
    fixed_kwargs["g"] = float(global_search)
    fixed["modules"]["hypo_transitions_mod"] = {
        "class": FIXED_CLASS,
        "kwargs": fixed_kwargs,
    }

    changed = {
        path
        for path in set(_flatten(adaptive_engine)) | set(_flatten(fixed))
        if _flatten(adaptive_engine).get(path) != _flatten(fixed).get(path)
    }
    illegal = [
        path
        for path in changed
        if not path.startswith("modules.hypo_transitions_mod.")
    ]
    if illegal:
        raise RuntimeError(f"fixed comparator changed non-controller paths: {illegal}")
    return fixed


def _filter_seeds(
    base_seed: int,
    subject_id: int,
    seed_count: int,
    *,
    seed_role: str = "model0815_b0_adaptive_controller_pilot",
) -> np.ndarray:
    return np.asarray(
        [
            stable_seed(
                {
                    "seed_role": str(seed_role),
                    "base_seed": int(base_seed),
                    "subject_id": int(subject_id),
                    "repeat_index": int(index),
                }
            )
            for index in range(int(seed_count))
        ],
        dtype=np.uint64,
    )


PANEL_TRACE_KEYS = (
    "marginal_prior",
    "pre_choice_ess",
    "post_choice_ess",
    "resampled",
    "predictive_transition_rate",
    "predictive_search_range",
    "predictive_swap_probability",
    "predictive_swap_event_probability",
    "predictive_strategy_exploit",
    "predictive_strategy_local_explore",
    "predictive_strategy_global_explore",
)


def validate_panel(panel: Mapping[str, Any]) -> dict[str, np.ndarray]:
    probability = np.asarray(panel["choice_probability"], dtype=float)
    prior = np.asarray(panel["marginal_prior"], dtype=float)
    if probability.ndim != 3 or probability.shape[0] < 2 or probability.shape[2] != 2:
        raise ValueError("choice_probability must have shape (seeds, trials, 2)")
    if prior.ndim != 3 or prior.shape[:2] != probability.shape[:2]:
        raise ValueError("marginal_prior must share seed/trial dimensions")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("choice probabilities must be finite and non-negative")
    if not np.all(np.isfinite(prior)) or np.any(prior < 0.0):
        raise ValueError("marginal priors must be finite and non-negative")
    probability /= np.sum(probability, axis=2, keepdims=True)
    prior /= np.sum(prior, axis=2, keepdims=True)
    seed_n, trial_n = probability.shape[:2]
    output = {
        "choice_probability": probability,
        "marginal_prior": prior,
        "filter_seed": np.asarray(panel["filter_seed"], dtype=np.uint64).reshape(-1),
        "repeat_index": np.asarray(panel["repeat_index"], dtype=int).reshape(-1),
        "observed_choice_index": np.asarray(
            panel["observed_choice_index"], dtype=int
        ).reshape(-1),
        "valid_trial_mask": np.asarray(panel["valid_trial_mask"], dtype=bool).reshape(-1),
    }
    for name in PANEL_TRACE_KEYS[1:]:
        values = np.asarray(panel[name])
        if values.shape != (seed_n, trial_n):
            raise ValueError(f"{name} has invalid shape {values.shape}")
        if name != "resampled" and not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
        output[name] = values.astype(bool) if name == "resampled" else values.astype(float)
    if output["filter_seed"].size != seed_n:
        raise ValueError("filter seed count does not match panel")
    if np.unique(output["filter_seed"]).size != seed_n:
        raise ValueError("filter seeds must be unique")
    if not np.array_equal(output["repeat_index"], np.arange(seed_n)):
        raise ValueError("repeat indices must be contiguous")
    if output["observed_choice_index"].size != trial_n:
        raise ValueError("observed choices do not match trial count")
    if output["valid_trial_mask"].size != trial_n:
        raise ValueError("valid mask does not match trial count")
    return output


def derive_training_match(
    adaptive_panel: Mapping[str, Any],
    *,
    train_trials: int,
    exclude_initialization_trial: bool,
    capacity: int,
) -> dict[str, Any]:
    values = validate_panel(adaptive_panel)
    trial_n = values["choice_probability"].shape[1]
    if not 1 < int(train_trials) < trial_n:
        raise ValueError("train_trials must split the observed sequence")
    mask = values["valid_trial_mask"].copy()
    mask[np.arange(trial_n) >= int(train_trials)] = False
    if exclude_initialization_trial:
        mask[0] = False
    if not np.any(mask):
        raise ValueError("training controller match contains no valid trials")
    event_probability = float(
        np.mean(values["predictive_swap_probability"][:, mask])
    )
    global_search = float(
        np.mean(values["predictive_search_range"][:, mask])
    )
    return {
        "event_probability": event_probability,
        "slot_rate": event_probability_to_slot_rate(event_probability, capacity),
        "global_search": global_search,
        "capacity": int(capacity),
        "matched_trial_count": int(np.sum(mask)),
        "seed_count": int(values["choice_probability"].shape[0]),
        "exclude_initialization_trial": bool(exclude_initialization_trial),
    }


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
    variant_id: str,
    engine: Mapping[str, Any],
    fixed_hyperparams: Mapping[str, Any],
    particle_count: int,
    seed_count: int,
    trials_per_subject: int,
    base_seed: int,
    n_jobs: int,
    seed_role: str = "model0815_b0_adaptive_controller_pilot",
) -> dict[str, Any]:
    npz_path, json_path = _cache_paths(output, subject_id, variant_id)
    expected_seeds = _filter_seeds(
        base_seed,
        subject_id,
        seed_count,
        seed_role=seed_role,
    )
    if npz_path.exists() and json_path.exists():
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        panel = _load_panel(npz_path)
        if _sha256(npz_path) != metadata["npz_sha256"]:
            raise ValueError(f"cache hash mismatch: {npz_path}")
        if not np.array_equal(panel["filter_seed"], expected_seeds):
            raise ValueError(f"cached PF seed panel differs from design: {npz_path}")
        return metadata
    if npz_path.exists() != json_path.exists():
        raise FileExistsError(f"incomplete cache pair requires manual review: {npz_path}")

    resolved_engine = deepcopy(dict(engine))
    resolved_engine.setdefault("inference", {})["particle_count"] = int(particle_count)
    engine_path = (
        output / "resolved_engines" / f"subject_{subject_id}_{variant_id}.json"
    )
    if engine_path.exists():
        raise FileExistsError(f"refusing to overwrite resolved engine: {engine_path}")
    _atomic_json(engine_path, resolved_engine)

    subject_cfg = resolve_subject_config(simulation_config, subject_id)
    dataset_paths = resolve_dataset_paths(subject_cfg, simulation_config_path.parent)
    runner = StateModelSimulationRunner(
        engine_config=resolved_engine,
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
        fixed_hyperparams=dict(fixed_hyperparams),
        window_size=resolve_window_size(subject_cfg, subject_id, [subject_id]),
        stop_at=float(subject_cfg.get("stop_at", 1.0)),
        max_trials=int(trials_per_subject),
        keep_logs=True,
        prediction_mode=prediction_mode,
        selection_prediction_mode=selection_mode,
        loss_metric=loss_metric,
        loss_delta=resolve_loss_delta(subject_cfg, loss_metric),
        hyper_candidate_seed=int(base_seed),
        trajectory_seeds=[int(seed) for seed in expected_seeds],
        compute_statistics=False,
        repeat_aggregation="mean_probability",
        evaluation_protocol=subject_cfg.get("evaluation_protocol"),
    )
    raw_runs = list(result["best"].raw_runs or [])
    if len(raw_runs) != int(seed_count):
        raise RuntimeError("controller panel did not return every requested PF seed")

    probabilities: list[np.ndarray] = []
    stacked: dict[str, list[np.ndarray]] = {key: [] for key in PANEL_TRACE_KEYS}
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
        for key in PANEL_TRACE_KEYS:
            stacked[key].append(np.asarray(state_log[key]))
        observed_seeds.append(int(run["trajectory_seed"]))
    if observed_choices is None or valid_mask is None:
        raise RuntimeError("controller panel returned no observed trials")
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
        "variant_id": str(variant_id),
        "particle_count": int(particle_count),
        "seed_count": int(seed_count),
        "trial_count": int(panel["choice_probability"].shape[1]),
        "valid_choice_trial_count": int(np.sum(panel["valid_trial_mask"])),
        "filter_seeds": [int(value) for value in panel["filter_seed"]],
        "resolved_engine": _relative(engine_path),
        "resolved_engine_sha256": _sha256(engine_path),
        "npz_path": _relative(npz_path),
        "npz_sha256": _sha256(npz_path),
    }
    _atomic_json(json_path, metadata)
    return metadata


def _masks(panel: Mapping[str, Any], train_trials: int) -> dict[str, np.ndarray]:
    values = validate_panel(panel)
    trial_n = values["choice_probability"].shape[1]
    index = np.arange(trial_n)
    valid = values["valid_trial_mask"]
    train = valid & (index < int(train_trials))
    heldout = valid & (index >= int(train_trials))
    heldout_indices = np.flatnonzero(heldout)
    early_n = min(16, max(1, heldout_indices.size // 2))
    early = np.zeros(trial_n, dtype=bool)
    early[heldout_indices[:early_n]] = True
    late = heldout & ~early
    if not np.any(train) or not np.any(heldout):
        raise ValueError("train/heldout split contains an empty scoring segment")
    return {"train": train, "heldout": heldout, "early_heldout": early, "late_heldout": late}


def summarize_variant(
    panel: Mapping[str, Any],
    *,
    subject_id: int,
    variant_id: str,
    particle_count: int,
    train_trials: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    values = validate_panel(panel)
    masks = _masks(values, train_trials)
    probability = values["choice_probability"]
    choices = values["observed_choice_index"]
    mean_probability = np.mean(probability, axis=0)
    mean_prior = np.mean(values["marginal_prior"], axis=0)
    arrays: dict[str, np.ndarray] = {
        "mean_choice_probability": mean_probability,
        "mean_marginal_prior": mean_prior,
        "observed_choice_index": choices,
        "valid_trial_mask": values["valid_trial_mask"],
        "filter_seed": values["filter_seed"],
    }
    row: dict[str, Any] = {
        "subject_id": int(subject_id),
        "variant_id": str(variant_id),
        "particle_count": int(particle_count),
        "seed_count": int(probability.shape[0]),
        "valid_train_trials": int(np.sum(masks["train"])),
        "valid_heldout_trials": int(np.sum(masks["heldout"])),
        "median_post_choice_ess_fraction": float(
            np.median(values["post_choice_ess"] / float(particle_count))
        ),
        "mean_resampling_fraction": float(np.mean(values["resampled"])),
    }
    for segment, mask in masks.items():
        run_nll = np.asarray(
            [_choice_nll(run, choices, mask) for run in probability], dtype=float
        )
        arrays[f"run_nll_{segment}"] = run_nll
        row[f"ensemble_nll_{segment}"] = _choice_nll(mean_probability, choices, mask)
        row[f"run_nll_mean_{segment}"] = float(np.mean(run_nll))
        row[f"run_nll_sd_{segment}"] = float(np.std(run_nll, ddof=1))
        row[f"ensemble_brier_{segment}"] = _choice_brier(
            mean_probability, choices, mask
        )
        row[f"mean_event_probability_{segment}"] = float(
            np.mean(values["predictive_swap_probability"][:, mask])
        )
        row[f"mean_global_search_{segment}"] = float(
            np.mean(values["predictive_search_range"][:, mask])
        )
        mean_event_trajectory = np.mean(
            values["predictive_swap_probability"][:, mask], axis=0
        )
        mean_global_trajectory = np.mean(
            values["predictive_search_range"][:, mask], axis=0
        )
        row[f"temporal_sd_event_probability_{segment}"] = float(
            np.std(mean_event_trajectory, ddof=0)
        )
        row[f"temporal_sd_global_search_{segment}"] = float(
            np.std(mean_global_trajectory, ddof=0)
        )
    return row, arrays


def summarize_contrast(
    fixed_row: Mapping[str, Any],
    fixed: Mapping[str, np.ndarray],
    adaptive_row: Mapping[str, Any],
    adaptive: Mapping[str, np.ndarray],
    *,
    train_trials: int,
    practical_fraction: float,
    seed_noise_multiplier: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not np.array_equal(fixed["filter_seed"], adaptive["filter_seed"]):
        raise ValueError("controller variants do not share ordered PF seeds")
    if not np.array_equal(
        fixed["observed_choice_index"], adaptive["observed_choice_index"]
    ) or not np.array_equal(fixed["valid_trial_mask"], adaptive["valid_trial_mask"]):
        raise ValueError("controller variants do not share observed trials")

    masks = _masks(
        {
            "choice_probability": fixed["mean_choice_probability"][None, ...].repeat(2, axis=0),
            "marginal_prior": fixed["mean_marginal_prior"][None, ...].repeat(2, axis=0),
            "filter_seed": np.asarray([0, 1]),
            "repeat_index": np.asarray([0, 1]),
            "observed_choice_index": fixed["observed_choice_index"],
            "valid_trial_mask": fixed["valid_trial_mask"],
            **{
                key: np.zeros((2, fixed["valid_trial_mask"].size), dtype=bool if key == "resampled" else float)
                for key in PANEL_TRACE_KEYS[1:]
            },
        },
        train_trials,
    )
    row: dict[str, Any] = {
        "subject_id": int(fixed_row["subject_id"]),
        "fixed_variant": str(fixed_row["variant_id"]),
        "adaptive_variant": str(adaptive_row["variant_id"]),
        "particle_count": int(fixed_row["particle_count"]),
        "seed_count": int(fixed_row["seed_count"]),
    }
    seed_rows: list[dict[str, Any]] = []
    for segment in ("train", "heldout", "early_heldout", "late_heldout"):
        delta = np.asarray(fixed[f"run_nll_{segment}"], dtype=float) - np.asarray(
            adaptive[f"run_nll_{segment}"], dtype=float
        )
        delta_mean = float(np.mean(delta))
        delta_sd = float(np.std(delta, ddof=1))
        delta_mcse = float(delta_sd / np.sqrt(delta.size))
        row[f"paired_delta_nll_{segment}"] = delta_mean
        row[f"paired_delta_nll_sd_{segment}"] = delta_sd
        row[f"paired_delta_nll_mcse_{segment}"] = delta_mcse
        row[f"ensemble_delta_nll_{segment}"] = float(
            fixed_row[f"ensemble_nll_{segment}"]
        ) - float(adaptive_row[f"ensemble_nll_{segment}"])
        row[f"positive_seed_fraction_{segment}"] = float(np.mean(delta > 0.0))
        for index, seed in enumerate(fixed["filter_seed"]):
            seed_rows.append(
                {
                    "subject_id": int(fixed_row["subject_id"]),
                    "segment": segment,
                    "repeat_index": int(index),
                    "filter_seed": int(seed),
                    "fixed_nll": float(fixed[f"run_nll_{segment}"][index]),
                    "adaptive_nll": float(adaptive[f"run_nll_{segment}"][index]),
                    "paired_delta_nll": float(delta[index]),
                }
            )

    heldout = masks["heldout"]
    probability_difference = (
        fixed["mean_choice_probability"][heldout]
        - adaptive["mean_choice_probability"][heldout]
    )
    row["heldout_choice_probability_rmse"] = float(
        np.sqrt(np.mean(np.square(probability_difference)))
    )
    row["heldout_predictive_geometry_prior_js"] = _mean_js(
        fixed["mean_marginal_prior"],
        adaptive["mean_marginal_prior"],
        heldout,
    )
    practical_threshold = float(
        practical_fraction * float(fixed_row["ensemble_nll_heldout"])
    )
    heldout_delta = float(row["paired_delta_nll_heldout"])
    heldout_mcse = float(row["paired_delta_nll_mcse_heldout"])
    row["practical_delta_threshold"] = practical_threshold
    row["exceeds_positive_practical_threshold"] = bool(
        heldout_delta > practical_threshold
    )
    row["heldout_effect_exceeds_seed_noise"] = bool(
        abs(heldout_delta) > seed_noise_multiplier * heldout_mcse
    )
    return row, seed_rows


def _resolved_design(config: Mapping[str, Any], smoke: bool) -> dict[str, Any]:
    design = deepcopy(dict(config["design"]))
    design["subjects"] = [int(value) for value in design["subjects"]]
    if smoke:
        design.update(
            {
                "subjects": [design["subjects"][0]],
                "trials_per_subject": 24,
                "train_trials": 12,
                "particle_count": 8,
                "seed_count": 2,
                "n_jobs": 1,
            }
        )
    if not 1 < int(design["train_trials"]) < int(design["trials_per_subject"]):
        raise ValueError("train_trials must divide the selected trial window")
    if int(design["seed_count"]) < 2:
        raise ValueError("paired controller screen requires at least two PF seeds")
    return design


def _load_subject_base_engine(
    simulation_config_path: Path,
    simulation_config: Mapping[str, Any],
    subject_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    subject_cfg = resolve_subject_config(simulation_config, subject_id)
    engine = resolve_engine_config(
        subject_cfg, simulation_config_path.parent, subject_id=subject_id
    )
    fixed = {
        **infer_fixed_hyperparams_from_engine_config(engine),
        **dict(subject_cfg.get("fixed_hyperparams") or {}),
    }
    engine = apply_fixed_hyperparams_to_engine_config(engine, fixed)
    validate_minimal_adaptive_engine(engine)
    return engine, fixed


def _write_readme(
    output: Path,
    variant_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    matching_summary: pd.DataFrame,
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    report = dict(config.get("report") or {})
    title = str(report.get("title", "Model 0815 B0 adaptive-controller pilot"))
    common_architecture = str(
        report.get(
            "common_architecture",
            "Both variants use the frozen common controller-screen architecture.",
        )
    )
    lines = [
        f"# {title}",
        "",
        "## Question and design",
        "",
        (
            "This low-cost architecture screen asks whether failure/mastery-dependent "
            "search predicts held-out choices better than fixed search with the same "
            "subject-specific mean search-event probability and global-search mixture. "
            "Matching uses only trials before the train/held-out boundary."
        ),
        "",
        common_architecture,
        "Positive delta NLL favors the adaptive controller.",
        "",
        "## Subject-level held-out results",
        "",
        "| subject | fixed NLL | adaptive NLL | paired delta | MCSE | positive seeds | practical | above seed noise | geometry JS |",
        "|---:|---:|---:|---:|---:|---:|:---:|:---:|---:|",
    ]
    fixed_id = str(summary["fixed_variant_id"])
    adaptive_id = str(summary["adaptive_variant_id"])
    by_variant = variant_summary.set_index(["subject_id", "variant_id"])
    for row in contrast_summary.to_dict(orient="records"):
        subject = int(row["subject_id"])
        fixed_nll = by_variant.loc[(subject, fixed_id), "ensemble_nll_heldout"]
        adaptive_nll = by_variant.loc[(subject, adaptive_id), "ensemble_nll_heldout"]
        lines.append(
            "| {subject} | {fixed:.5f} | {adaptive:.5f} | {delta:+.5f} | "
            "{mcse:.5f} | {positive:.2f} | {practical} | {noise} | {js:.5f} |".format(
                subject=subject,
                fixed=float(fixed_nll),
                adaptive=float(adaptive_nll),
                delta=float(row["paired_delta_nll_heldout"]),
                mcse=float(row["paired_delta_nll_mcse_heldout"]),
                positive=float(row["positive_seed_fraction_heldout"]),
                practical="yes" if row["exceeds_positive_practical_threshold"] else "no",
                noise="yes" if row["heldout_effect_exceeds_seed_noise"] else "no",
                js=float(row["heldout_predictive_geometry_prior_js"]),
            )
        )
    lines.extend(
        [
            "",
            "## Training-only controller matching",
            "",
            "| subject | matched event probability | matched slot rate | matched global mixture | fixed event error | fixed global error |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in matching_summary.to_dict(orient="records"):
        lines.append(
            "| {subject_id} | {event_probability:.5f} | {slot_rate:.5f} | "
            "{global_search:.5f} | {fixed_train_event_match_error:+.2e} | "
            "{fixed_train_global_match_error:+.2e} |".format(**row)
        )
    cohort = summary["cohort"]
    lines.extend(
        [
            "",
            "## Cohort screen",
            "",
            f"- Mean held-out paired delta NLL: {cohort['mean_paired_delta_nll_heldout']:+.5f}.",
            f"- Positive subjects: {cohort['positive_subject_count']}/{cohort['subject_count']}.",
            f"- Practically positive subjects: {cohort['practically_positive_subject_count']}/{cohort['subject_count']}.",
            f"- Provisional decision: **{summary['provisional_decision']}**.",
            "",
            "## Interpretation boundary",
            "",
            (
                "This is a small architecture screen with a provisional shared PF budget, "
                "not a final mechanism decision or a final parameter/PF calibration. A "
                "promising result triggers a larger paired-seed replication before the "
                "controller is retained; a null result does not justify decomposing the "
                "controller bundle."
            ),
            "",
            "Raw per-seed arrays, resolved engines, hashes, matches, and paired deltas are retained for recomputation.",
            "",
        ]
    )
    readme = output / "README.md"
    if readme.exists():
        raise FileExistsError(f"refusing to overwrite report: {readme}")
    readme.write_text("\n".join(lines), encoding="utf-8")


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
    if (output / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    simulation_config_path = _repo_path(config["base_simulation_config"])
    simulation_config = load_yaml(simulation_config_path)
    adaptive_id = str(design["adaptive_variant_id"])
    fixed_id = str(design["fixed_variant_id"])
    metadata_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []

    if args.phase in {"run", "all"}:
        for subject_id in design["subjects"]:
            adaptive_engine, fixed_hyperparams = _load_subject_base_engine(
                simulation_config_path, simulation_config, subject_id
            )
            adaptive_metadata = _run_panel(
                simulation_config_path=simulation_config_path,
                simulation_config=simulation_config,
                output=output,
                subject_id=subject_id,
                variant_id=adaptive_id,
                engine=adaptive_engine,
                fixed_hyperparams=fixed_hyperparams,
                particle_count=int(design["particle_count"]),
                seed_count=int(design["seed_count"]),
                trials_per_subject=int(design["trials_per_subject"]),
                base_seed=int(design["base_seed"]),
                n_jobs=int(design["n_jobs"]),
            )
            metadata_rows.append(adaptive_metadata)
            adaptive_panel = _load_panel(_repo_path(adaptive_metadata["npz_path"]))
            capacity = int(
                adaptive_engine["modules"]["hypo_transitions_mod"]["kwargs"]["capacity"]
            )
            match = derive_training_match(
                adaptive_panel,
                train_trials=int(design["train_trials"]),
                exclude_initialization_trial=bool(
                    config["matching"]["exclude_initialization_trial"]
                ),
                capacity=capacity,
            )
            fixed_engine = build_training_matched_fixed_engine(
                adaptive_engine,
                event_probability=float(match["event_probability"]),
                global_search=float(match["global_search"]),
            )
            fixed_metadata = _run_panel(
                simulation_config_path=simulation_config_path,
                simulation_config=simulation_config,
                output=output,
                subject_id=subject_id,
                variant_id=fixed_id,
                engine=fixed_engine,
                fixed_hyperparams=fixed_hyperparams,
                particle_count=int(design["particle_count"]),
                seed_count=int(design["seed_count"]),
                trials_per_subject=int(design["trials_per_subject"]),
                base_seed=int(design["base_seed"]),
                n_jobs=int(design["n_jobs"]),
            )
            metadata_rows.append(fixed_metadata)
            fixed_panel = _load_panel(_repo_path(fixed_metadata["npz_path"]))
            train_mask = _masks(fixed_panel, int(design["train_trials"]))["train"].copy()
            if bool(config["matching"]["exclude_initialization_trial"]):
                train_mask[0] = False
            fixed_event = float(
                np.mean(fixed_panel["predictive_swap_probability"][:, train_mask])
            )
            fixed_global = float(
                np.mean(fixed_panel["predictive_search_range"][:, train_mask])
            )
            match_rows.append(
                {
                    "subject_id": int(subject_id),
                    **match,
                    "fixed_train_event_probability": fixed_event,
                    "fixed_train_global_search": fixed_global,
                    "fixed_train_event_match_error": fixed_event
                    - float(match["event_probability"]),
                    "fixed_train_global_match_error": fixed_global
                    - float(match["global_search"]),
                }
            )
        _atomic_json(
            output / "run_manifest.json",
            {
                "analysis_id": config["analysis_id"],
                "config": _relative(config_path),
                "config_sha256": _sha256(config_path),
                "base_simulation_config": _relative(simulation_config_path),
                "base_simulation_config_sha256": _sha256(simulation_config_path),
                "design": design,
                "runs": metadata_rows,
                "matches": match_rows,
            },
        )
        _atomic_csv(output / "training_match_summary.csv", pd.DataFrame(match_rows))

    if args.phase in {"summarize", "all"}:
        manifest_path = output / "run_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing run manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        match_df = pd.DataFrame(manifest["matches"])
        metadata = {
            (int(row["subject_id"]), str(row["variant_id"])): row
            for row in manifest["runs"]
        }
        variant_rows: list[dict[str, Any]] = []
        contrast_rows: list[dict[str, Any]] = []
        seed_rows: list[dict[str, Any]] = []
        summarized: dict[tuple[int, str], dict[str, np.ndarray]] = {}
        for subject_id in design["subjects"]:
            for variant_id in (adaptive_id, fixed_id):
                row = metadata[(int(subject_id), variant_id)]
                panel = _load_panel(_repo_path(row["npz_path"]))
                variant_row, arrays = summarize_variant(
                    panel,
                    subject_id=subject_id,
                    variant_id=variant_id,
                    particle_count=int(design["particle_count"]),
                    train_trials=int(design["train_trials"]),
                )
                variant_rows.append(variant_row)
                summarized[(int(subject_id), variant_id)] = arrays
            fixed_row = variant_rows[-1]
            adaptive_row = variant_rows[-2]
            contrast_row, current_seed_rows = summarize_contrast(
                fixed_row,
                summarized[(int(subject_id), fixed_id)],
                adaptive_row,
                summarized[(int(subject_id), adaptive_id)],
                train_trials=int(design["train_trials"]),
                practical_fraction=float(
                    config["screening_rule"]["practical_fraction_of_fixed_heldout_nll"]
                ),
                seed_noise_multiplier=float(
                    config["screening_rule"]["seed_noise_multiplier"]
                ),
            )
            contrast_rows.append(contrast_row)
            seed_rows.extend(current_seed_rows)

        variant_df = pd.DataFrame(variant_rows)
        contrast_df = pd.DataFrame(contrast_rows)
        seed_df = pd.DataFrame(seed_rows)
        cohort_delta = float(np.mean(contrast_df["paired_delta_nll_heldout"]))
        fixed_heldout = variant_df.loc[
            variant_df["variant_id"] == fixed_id, "ensemble_nll_heldout"
        ]
        cohort_practical_threshold = float(
            config["screening_rule"]["practical_fraction_of_fixed_heldout_nll"]
            * np.mean(fixed_heldout)
        )
        positive_count = int(np.sum(contrast_df["paired_delta_nll_heldout"] > 0.0))
        practical_count = int(
            np.sum(contrast_df["exceeds_positive_practical_threshold"])
        )
        majority = int(np.ceil(len(contrast_df) / 2.0))
        promote = bool(
            cohort_delta > cohort_practical_threshold and positive_count >= majority
        )
        provisional_decision = (
            "promising: replicate with more paired PF seeds"
            if promote
            else "not promoted: retain fixed search in the current minimal baseline"
        )
        summary = {
            "analysis_id": config["analysis_id"],
            "subjects": [int(value) for value in design["subjects"]],
            "adaptive_variant_id": adaptive_id,
            "fixed_variant_id": fixed_id,
            "train_trials": int(design["train_trials"]),
            "trials_per_subject": int(design["trials_per_subject"]),
            "particle_count": int(design["particle_count"]),
            "seed_count": int(design["seed_count"]),
            "cohort": {
                "subject_count": int(len(contrast_df)),
                "mean_paired_delta_nll_train": float(
                    np.mean(contrast_df["paired_delta_nll_train"])
                ),
                "mean_paired_delta_nll_heldout": cohort_delta,
                "mean_ensemble_delta_nll_heldout": float(
                    np.mean(contrast_df["ensemble_delta_nll_heldout"])
                ),
                "cohort_practical_delta_threshold": cohort_practical_threshold,
                "positive_subject_count": positive_count,
                "practically_positive_subject_count": practical_count,
                "seed_noise_resolved_subject_count": int(
                    np.sum(contrast_df["heldout_effect_exceeds_seed_noise"])
                ),
            },
            "provisional_decision": provisional_decision,
            "decision_is_final": False,
        }
        _atomic_csv(output / "variant_summary.csv", variant_df)
        _atomic_csv(output / "contrast_summary.csv", contrast_df)
        _atomic_csv(output / "paired_seed_effects.csv", seed_df)
        _atomic_json(output / "summary.json", summary)
        _write_readme(output, variant_df, contrast_df, match_df, summary, config)


if __name__ == "__main__":
    main()
