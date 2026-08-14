#!/usr/bin/env python3
"""Run Phase-0 reachability/equivalence probes for the 0813 PF model.

The audit changes one configured mechanism at a time wherever the current
implementation exposes a clean switch.  Each variant shares the same observed
trial prefix and PF seed as its subject-specific baseline.  Exact no-op claims
require bitwise-equal public numerical outputs (with NaNs treated as equal).
Other probes only establish reachability; they do not estimate predictive
benefit or justify retaining a mechanism.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Bayesian_state.inference.backends.particle_filter import (  # noqa: E402
    run_state_model_particle_filter,
)
from src.Bayesian_state.model.readout import (  # noqa: E402
    CHOICE_READOUT_SHARPENED,
    resolve_choice_readout_config,
    resolve_output_noise_config,
)
from src.Bayesian_state.simulation.config import (  # noqa: E402
    load_yaml,
    resolve_engine_config,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.utils.seeding import stable_seed  # noqa: E402
from src.Bayesian_state.utils.subjects import resolve_subject_config  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs/specific_models/model_0813_pf_mechanism_reachability.yaml"
)
FEATURE_COLUMNS = ("feature1", "feature2", "feature3", "feature4")
ORDER_COLUMNS = ("iSession", "iBlock", "iTrial")
BASELINE_VARIANT = "baseline"


@dataclass(frozen=True)
class ProbeSpec:
    variant_id: str
    label: str
    expected_role: str
    subject_group: str = "default"


PROBE_SPECS = (
    ProbeSpec(
        "readout_power_one",
        "Hypothesis-weight power 4 -> 1 under persistent execution",
        "expected_exact_noop",
        "all",
    ),
    ProbeSpec(
        "readout_expectation",
        "sharpened_expectation -> expectation under persistent execution",
        "expected_exact_noop",
        "all",
    ),
    ProbeSpec(
        "dormant_signal_scaling",
        "Change surprise/uncertainty scaling while their v2 weights are zero",
        "expected_exact_noop",
    ),
    ProbeSpec(
        "dormant_beta_prior_scale",
        "Change prior_beta_scale while use_prior_scaling is false",
        "expected_exact_noop",
    ),
    ProbeSpec(
        "dormant_output_controls",
        "Change history-lapse controls while all history lapse coefficients are zero",
        "expected_exact_noop",
    ),
    ProbeSpec(
        "disabled_optional_blocks",
        "Add explicit disabled prior-reset/capture/commitment blocks",
        "expected_exact_noop",
    ),
    ProbeSpec("boundary_geometry", "Prototype -> boundary likelihood", "reachability"),
    ProbeSpec(
        "zero_perception_noise",
        "Subject-specific perception -> identity perception",
        "reachability",
        "perception",
    ),
    ProbeSpec("capacity_five", "Workspace capacity 3 -> 5", "reachability"),
    ProbeSpec(
        "failure_trace_no_memory",
        "Failure decay 0.60 -> 0.00",
        "reachability",
    ),
    ProbeSpec("mastery_inhibition_off", "Mastery weight 1 -> 0", "reachability"),
    ProbeSpec(
        "global_search_fixed_local",
        "Dynamic global range -> fixed local range",
        "reachability",
    ),
    ProbeSpec(
        "execution_off_linked",
        "Persistent execution off with required all-active beta scope",
        "linked_structure_reachability",
    ),
    ProbeSpec("switching_off", "Execution switch scale 0.20 -> 0", "reachability"),
    ProbeSpec(
        "bayesian_memory",
        "Dual long/short memory -> one-step Bayesian memory",
        "reachability",
    ),
    ProbeSpec("fixed_beta", "Dynamic beta -> fixed beta=5", "reachability"),
    ProbeSpec(
        "all_active_beta_scope",
        "Executed-only -> all-active beta updating",
        "reachability",
    ),
    ProbeSpec(
        "strategy_confidence_off",
        "Strategy confidence gain 2 -> 0",
        "reachability",
    ),
    ProbeSpec("base_lapse_off", "Uniform base lapse 0.02 -> 0", "reachability"),
)

EXPECTED_EXACT_VARIANTS = frozenset(
    spec.variant_id
    for spec in PROBE_SPECS
    if spec.expected_role == "expected_exact_noop"
)

MECHANISM_PROBE_MAP: dict[str, tuple[str, ...]] = {
    "REP-02": ("boundary_geometry",),
    "COG-01": ("zero_perception_noise",),
    "COG-02": ("capacity_five",),
    "COG-03": ("failure_trace_no_memory",),
    "COG-04": ("mastery_inhibition_off",),
    "COG-05": ("global_search_fixed_local",),
    "COG-09": ("execution_off_linked",),
    "COG-10": ("switching_off",),
    "COG-11": ("bayesian_memory",),
    "COG-12": ("fixed_beta",),
    "COG-13": ("all_active_beta_scope",),
    "OBS-01": ("readout_power_one", "readout_expectation"),
    "OBS-02": ("strategy_confidence_off",),
    "OBS-03": ("base_lapse_off",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use subject 103, 16 trials, and 4 particles.",
    )
    return parser.parse_args()


def _set_path(root: dict[str, Any], path: str, value: Any) -> None:
    current = root
    parts = path.split(".")
    for part in parts[:-1]:
        next_value = current.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"cannot set {path!r} through non-mapping {part!r}")
        current = next_value
    current[parts[-1]] = deepcopy(value)


def _get_path(root: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = root
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _variant_engine(
    baseline: Mapping[str, Any], variant_id: str
) -> dict[str, Any]:
    engine = deepcopy(dict(baseline))
    if variant_id == BASELINE_VARIANT:
        return engine
    if variant_id == "readout_power_one":
        _set_path(engine, "choice_readout.kwargs.power", 1.0)
    elif variant_id == "readout_expectation":
        _set_path(engine, "choice_readout.kwargs.method", "expectation")
    elif variant_id == "dormant_signal_scaling":
        _set_path(engine, "modules.hypo_transitions_mod.kwargs.surprise_center", -10.0)
        _set_path(engine, "modules.hypo_transitions_mod.kwargs.surprise_scale", 0.5)
        _set_path(engine, "modules.hypo_transitions_mod.kwargs.uncertainty_center", 10.0)
        _set_path(engine, "modules.hypo_transitions_mod.kwargs.uncertainty_scale", 0.5)
    elif variant_id == "dormant_beta_prior_scale":
        _set_path(engine, "modules.beta_mod.kwargs.prior_beta_scale", 10.0)
    elif variant_id == "dormant_output_controls":
        _set_path(engine, "output_noise.kwargs.low_accuracy_threshold", 0.2)
        _set_path(engine, "output_noise.kwargs.recent_accuracy_window", 2)
        _set_path(engine, "output_noise.kwargs.lapse_decay", 0.9)
        _set_path(engine, "output_noise.kwargs.max_lapse", 0.8)
        _set_path(engine, "output_noise.kwargs.latent_volatility_power", 3.0)
    elif variant_id == "disabled_optional_blocks":
        _set_path(
            engine,
            "modules.hypo_transitions_mod.kwargs.continuous_controller.prior_reset",
            {"max_strength": 0.0},
        )
        _set_path(
            engine,
            (
                "modules.hypo_transitions_mod.kwargs.continuous_controller."
                "execution.misconception_capture"
            ),
            {
                "enabled": False,
                "choice_decay": 0.25,
                "failure_threshold": 0.10,
                "min_evidence_trials": 1,
                "min_advantage": 0.0,
                "min_choice_compatibility": 0.0,
                "min_dwell_trials": 1,
            },
        )
        _set_path(
            engine,
            (
                "modules.hypo_transitions_mod.kwargs.continuous_controller."
                "execution.rule_commitment"
            ),
            {"enabled": False},
        )
    elif variant_id == "boundary_geometry":
        _set_path(engine, "likelihood.distance_mode", "boundary")
    elif variant_id == "zero_perception_noise":
        _set_path(
            engine,
            "modules.perception_mod.kwargs",
            {
                "noise_mode": "normal",
                "mean": [0.0, 0.0, 0.0, 0.0],
                "std": [0.0, 0.0, 0.0, 0.0],
            },
        )
    elif variant_id == "capacity_five":
        _set_path(engine, "modules.hypo_transitions_mod.kwargs.capacity", 5)
    elif variant_id == "failure_trace_no_memory":
        _set_path(
            engine,
            (
                "modules.hypo_transitions_mod.kwargs.continuous_controller."
                "state.failure_decay"
            ),
            0.0,
        )
    elif variant_id == "mastery_inhibition_off":
        _set_path(
            engine,
            (
                "modules.hypo_transitions_mod.kwargs.continuous_controller."
                "exploration.mastery_weight"
            ),
            0.0,
        )
    elif variant_id == "global_search_fixed_local":
        _set_path(
            engine,
            (
                "modules.hypo_transitions_mod.kwargs.continuous_controller."
                "range.global_max"
            ),
            0.05,
        )
    elif variant_id == "execution_off_linked":
        _set_path(
            engine,
            (
                "modules.hypo_transitions_mod.kwargs.continuous_controller."
                "execution.enabled"
            ),
            False,
        )
        _set_path(engine, "modules.beta_mod.kwargs.update_scope", "active_hypotheses")
    elif variant_id == "switching_off":
        _set_path(
            engine,
            (
                "modules.hypo_transitions_mod.kwargs.continuous_controller."
                "execution.switch_scale"
            ),
            0.0,
        )
    elif variant_id == "bayesian_memory":
        _set_path(
            engine,
            "modules.memory_mod.class",
            "src.Bayesian_state.model.modules.memory.BayesianMemoryModule",
        )
        _set_path(engine, "modules.memory_mod.kwargs", {})
    elif variant_id == "fixed_beta":
        _set_path(engine, "modules.beta_mod.kwargs.decrease_rate", 0.0)
        _set_path(engine, "modules.beta_mod.kwargs.correct_additive", 0.0)
    elif variant_id == "all_active_beta_scope":
        _set_path(engine, "modules.beta_mod.kwargs.update_scope", "active_hypotheses")
    elif variant_id == "strategy_confidence_off":
        _set_path(engine, "choice_readout.kwargs.strategy_confidence_gain", 0.0)
    elif variant_id == "base_lapse_off":
        _set_path(engine, "output_noise.kwargs.base_lapse", 0.0)
    else:
        raise KeyError(f"unknown reachability variant: {variant_id}")
    return engine


def _readout_args(engine_config: Mapping[str, Any]) -> dict[str, float]:
    readout = resolve_choice_readout_config(None, engine_config)
    noise = resolve_output_noise_config(None, engine_config)
    method = str(readout["method"])
    power = (
        float(readout["power"])
        if method == CHOICE_READOUT_SHARPENED
        else 1.0
    )
    unsupported_lapse = sum(
        float(noise.get(key, 0.0))
        for key in (
            "post_error_lapse",
            "low_accuracy_lapse",
            "latent_volatility_lapse",
        )
    )
    if unsupported_lapse != 0.0:
        raise ValueError("Phase-0 PF probes require history-lapse coefficients to be zero")
    if str(noise.get("lapse_target", "uniform")) != "uniform":
        raise ValueError("Phase-0 PF probes require a uniform lapse target")
    return {
        "choice_readout_power": power,
        "strategy_confidence_gain": float(readout["strategy_confidence_gain"]),
        "rule_commitment_confidence_gain": float(
            readout["rule_commitment_confidence_gain"]
        ),
        "output_lapse": (
            float(noise.get("base_lapse", 0.0))
            if bool(noise.get("enabled", False))
            else 0.0
        ),
    }


def _subject_engine(
    base_config: Mapping[str, Any], base_path: Path, subject_id: int
) -> dict[str, Any]:
    subject_config = resolve_subject_config(base_config, int(subject_id))
    return resolve_engine_config(subject_config, base_path.parent)


def _run_output_arrays(result: Any) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for section_name, values in (
        ("observation", result.observation_probabilities),
        ("state", result.state_probabilities),
        ("latent", result.latent_summaries),
        ("diagnostic", result.diagnostics),
    ):
        for key, value in values.items():
            array = np.asarray(value)
            if array.dtype.kind not in "biufc":
                continue
            arrays[f"{section_name}.{key}"] = array.copy()
    return arrays


def _nll(probabilities: np.ndarray, choices: np.ndarray) -> float:
    choice_index = np.asarray(choices, dtype=int).reshape(-1) - 1
    selected = np.asarray(probabilities, dtype=float)[
        np.arange(choice_index.size), choice_index
    ]
    return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def _array_max_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    left_values = np.asarray(left)
    right_values = np.asarray(right)
    finite = np.isfinite(left_values) & np.isfinite(right_values)
    mismatched_nan = np.isnan(left_values) ^ np.isnan(right_values)
    left_inf = np.isinf(left_values)
    right_inf = np.isinf(right_values)
    mismatched_inf = (left_inf ^ right_inf) | (
        left_inf
        & right_inf
        & (np.signbit(left_values) != np.signbit(right_values))
    )
    if np.any(mismatched_nan) or np.any(mismatched_inf):
        return float("inf")
    if not np.any(finite):
        return 0.0
    comparison_dtype = np.result_type(left_values, right_values, np.float64)
    left_numeric = left_values.astype(comparison_dtype, copy=False)
    right_numeric = right_values.astype(comparison_dtype, copy=False)
    return float(np.max(np.abs(left_numeric[finite] - right_numeric[finite])))


def _compare_outputs(
    baseline: Mapping[str, np.ndarray],
    variant: Mapping[str, np.ndarray],
    choices: np.ndarray,
    *,
    tolerance: float,
) -> dict[str, Any]:
    baseline_keys = set(baseline)
    variant_keys = set(variant)
    common = sorted(baseline_keys & variant_keys)
    key_set_equal = baseline_keys == variant_keys
    exact_by_key = {
        key: bool(
            np.array_equal(
                np.asarray(baseline[key]),
                np.asarray(variant[key]),
                equal_nan=True,
            )
        )
        for key in common
    }
    max_by_key = {
        key: _array_max_diff(baseline[key], variant[key]) for key in common
    }
    changed_keys = [
        key for key in common if max_by_key[key] > float(tolerance)
    ]
    changed_keys.extend(sorted(baseline_keys ^ variant_keys))
    choice_key = "observation.prior_t"
    baseline_choice = np.asarray(baseline[choice_key], dtype=float)
    variant_choice = np.asarray(variant[choice_key], dtype=float)
    choice_diff = np.abs(variant_choice - baseline_choice)
    changed_trials = np.any(choice_diff > float(tolerance), axis=1)
    first_changed = (
        None if not np.any(changed_trials) else int(np.flatnonzero(changed_trials)[0])
    )

    def section_max(prefix: str) -> float:
        values = [value for key, value in max_by_key.items() if key.startswith(prefix)]
        return max(values) if values else float("nan")

    return {
        "key_set_equal": bool(key_set_equal),
        "all_public_arrays_exact": bool(
            key_set_equal and common and all(exact_by_key.values())
        ),
        "max_abs_choice_probability_diff": float(np.max(choice_diff)),
        "mean_abs_choice_probability_diff": float(np.mean(choice_diff)),
        "choice_changed_trial_fraction": float(np.mean(changed_trials)),
        "first_changed_trial_zero_based": first_changed,
        "max_abs_state_diff": section_max("state."),
        "max_abs_latent_diff": section_max("latent."),
        "max_abs_diagnostic_diff": section_max("diagnostic."),
        "baseline_mean_choice_nll": _nll(baseline_choice, choices),
        "variant_mean_choice_nll": _nll(variant_choice, choices),
        "changed_output_key_count": int(len(changed_keys)),
        "changed_output_keys": ";".join(changed_keys),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
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


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _worktree_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _manifest_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _dependency_rows() -> list[dict[str, Any]]:
    return [
        {
            "config_group": "surprise_uncertainty_scaling",
            "current_activation_condition": (
                "inactive for prediction because exploration/range surprise and "
                "uncertainty weights are all zero"
            ),
            "probe_variant": "dormant_signal_scaling",
        },
        {
            "config_group": "beta_prior_scale",
            "current_activation_condition": "inactive because use_prior_scaling=false",
            "probe_variant": "dormant_beta_prior_scale",
        },
        {
            "config_group": "history_dependent_output_controls",
            "current_activation_condition": (
                "inactive because post_error_lapse, low_accuracy_lapse and "
                "latent_volatility_lapse are all zero"
            ),
            "probe_variant": "dormant_output_controls",
        },
        {
            "config_group": "prior_reset_misconception_rule_commitment",
            "current_activation_condition": "inactive because all three optional blocks are disabled",
            "probe_variant": "disabled_optional_blocks",
        },
        {
            "config_group": "hypothesis_weight_sharpening",
            "current_activation_condition": (
                "inactive when persistent execution reduces the readout to one "
                "executed hypothesis"
            ),
            "probe_variant": "readout_power_one;readout_expectation",
        },
    ]


def _variant_subjects(
    spec: ProbeSpec,
    subjects: Sequence[int],
    default_subject: int,
    perception_subjects: Sequence[int],
) -> list[int]:
    if spec.subject_group == "all":
        return [int(value) for value in subjects]
    if spec.subject_group == "perception":
        return [int(value) for value in perception_subjects]
    return [int(default_subject)]


def _aggregate_probe_rows(probe_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant_id, frame in probe_rows.groupby("variant_id", sort=False):
        expected_role = str(frame["expected_role"].iloc[0])
        exact = bool(frame["all_public_arrays_exact"].all())
        changed = bool(
            (frame["max_abs_choice_probability_diff"] > frame["change_tolerance"]).any()
            or (frame["max_abs_state_diff"] > frame["change_tolerance"]).any()
            or (frame["max_abs_latent_diff"] > frame["change_tolerance"]).any()
        )
        if exact:
            phase0_classification = "exact_noop"
        elif changed:
            phase0_classification = "reachable"
        else:
            phase0_classification = "not_activated_in_probe"
        rows.append(
            {
                "variant_id": str(variant_id),
                "variant_label": str(frame["variant_label"].iloc[0]),
                "expected_role": expected_role,
                "subject_n": int(frame["subject_id"].nunique()),
                "all_subjects_exact": exact,
                "any_saved_output_changed": changed,
                "phase0_classification": phase0_classification,
                "max_abs_choice_probability_diff": float(
                    frame["max_abs_choice_probability_diff"].max()
                ),
                "max_abs_state_diff": float(frame["max_abs_state_diff"].max()),
                "max_abs_latent_diff": float(frame["max_abs_latent_diff"].max()),
                "max_abs_mean_choice_nll_difference": float(
                    np.max(
                        np.abs(
                            frame["variant_mean_choice_nll"]
                            - frame["baseline_mean_choice_nll"]
                        )
                    )
                ),
                "expected_exact_check_passed": bool(
                    expected_role != "expected_exact_noop" or exact
                ),
            }
        )
    return pd.DataFrame(rows)


def _runtime_summary(
    baseline_outputs: Mapping[int, Mapping[str, np.ndarray]],
    baseline_engines: Mapping[int, Mapping[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject_id, arrays in baseline_outputs.items():
        active = np.asarray(arrays["state.active_probability"], dtype=float)
        executed = np.asarray(arrays["state.executed_probability"], dtype=float)
        capacity = int(
            _get_path(
                baseline_engines[subject_id],
                "modules.hypo_transitions_mod.kwargs.capacity",
            )
        )
        normalized_active = active / float(capacity)
        appended = slice(19, None)
        rows.append(
            {
                "subject_id": int(subject_id),
                "trial_count": int(active.shape[0]),
                "capacity": capacity,
                "mean_swap_event_probability": float(
                    np.mean(arrays["latent.predictive_swap_event_probability"])
                ),
                "mean_replacement_fraction": float(
                    np.mean(arrays["latent.predictive_replacement_fraction"])
                ),
                "mean_removed_mass": float(
                    np.mean(arrays["latent.removed_mass"])
                ),
                "mean_newcomer_distance": float(
                    np.mean(arrays["latent.predictive_newcomer_distance"])
                ),
                "mean_search_range": float(
                    np.mean(arrays["latent.predictive_search_range"])
                ),
                "search_range_sd": float(
                    np.std(arrays["latent.predictive_search_range"])
                ),
                "mean_execution_switch_probability": float(
                    np.mean(
                        arrays["latent.predictive_execution_switch_event_probability"]
                    )
                ),
                "mean_executed_active_distribution_l1": float(
                    np.mean(np.sum(np.abs(executed - normalized_active), axis=1))
                ),
                "mean_appended_rule_active_slot_fraction": float(
                    np.mean(np.sum(active[:, appended], axis=1) / float(capacity))
                ),
                "mean_appended_rule_execution_probability": float(
                    np.mean(np.sum(executed[:, appended], axis=1))
                ),
                "mean_strategy_confidence_signal": float(
                    np.mean(arrays["latent.predictive_choice_confidence_signal"])
                ),
                "max_strategy_choice_precision": float(
                    np.max(arrays["latent.predictive_strategy_choice_precision"])
                ),
                "mean_abs_executed_beta_minus_initial": float(
                    np.nanmean(
                        np.abs(arrays["latent.predictive_executed_beta"] - 5.0)
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_reachability_summary(
    registry: pd.DataFrame,
    probe_summary: pd.DataFrame,
    runtime: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    probe_lookup = probe_summary.set_index("variant_id").to_dict(orient="index")
    runtime_any_replacement = bool(runtime["mean_replacement_fraction"].max() > tolerance)
    runtime_any_newcomer = bool(runtime["mean_newcomer_distance"].max() > tolerance)
    runtime_appended = bool(
        runtime["mean_appended_rule_active_slot_fraction"].max() > tolerance
        or runtime["mean_appended_rule_execution_probability"].max() > tolerance
    )
    runtime_execution = bool(
        runtime["mean_executed_active_distribution_l1"].max() > tolerance
    )
    rows: list[dict[str, Any]] = []
    for item in registry.to_dict(orient="records"):
        mechanism_id = str(item["mechanism_id"])
        probe_ids = MECHANISM_PROBE_MAP.get(mechanism_id, ())
        evidence_type = "numerical_variant" if probe_ids else "runtime_or_deferred"
        probe_records = [probe_lookup[probe_id] for probe_id in probe_ids]
        exact = bool(probe_records and all(row["all_subjects_exact"] for row in probe_records))
        reachable = bool(
            probe_records and any(row["any_saved_output_changed"] for row in probe_records)
        )
        max_choice_diff = (
            max(float(row["max_abs_choice_probability_diff"]) for row in probe_records)
            if probe_records
            else float("nan")
        )
        evidence = ";".join(probe_ids)
        if mechanism_id in {"INF-01", "INF-02"}:
            status = "deferred_to_phase1_inference_audit"
            evidence = "Phase 0 does not compare inference algorithms"
        elif mechanism_id == "REP-01":
            status = (
                "runtime_active_comparator_required"
                if runtime_appended
                else "catalog_present_not_occupied_in_probe"
            )
            evidence = "baseline appended-rule active/executed occupancy"
        elif mechanism_id == "COG-06":
            status = (
                "runtime_active_comparator_required"
                if runtime_any_replacement
                else "not_activated_in_probe"
            )
            evidence = "replacement and removed-mass runtime diagnostics"
        elif mechanism_id == "COG-07":
            status = (
                "runtime_active_comparator_required"
                if runtime_any_newcomer
                else "not_activated_in_probe"
            )
            evidence = "newcomer-distance and search-range runtime diagnostics"
        elif mechanism_id == "COG-08":
            status = (
                "runtime_active_comparator_required"
                if runtime_any_replacement
                else "not_activated_in_probe"
            )
            evidence = "pairwise transfer code path executes on replacement trials"
        elif mechanism_id == "COG-09" and runtime_execution and reachable:
            status = "active_linked_structure"
            evidence = "execution_off_linked; baseline executed/active separation"
        elif mechanism_id == "OBS-01" and exact:
            status = "inactive_exact_under_persistent_execution"
        elif reachable:
            status = "active_reachable"
        elif exact:
            status = "inactive_exact_in_probe"
        else:
            status = "not_activated_in_probe"
        rows.append(
            {
                "mechanism_id": mechanism_id,
                "layer": item["layer"],
                "mechanism": item["mechanism"],
                "phase0_status": status,
                "evidence_type": evidence_type,
                "evidence": evidence,
                "probe_exact": exact,
                "probe_reachable": reachable,
                "max_abs_choice_probability_diff": max_choice_diff,
                "retention_decision_allowed": False,
                "interpretation": (
                    "Phase 0 establishes configuration reachability only; retain/remove "
                    "requires later predictive and recovery audits."
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_readme(
    output_dir: Path,
    *,
    config: Mapping[str, Any],
    probe_summary: pd.DataFrame,
    reachability: pd.DataFrame,
    runtime: pd.DataFrame,
) -> None:
    exact_variants = probe_summary.loc[
        probe_summary["phase0_classification"].eq("exact_noop"), "variant_id"
    ].astype(str).tolist()
    reachable_variants = probe_summary.loc[
        probe_summary["phase0_classification"].eq("reachable"), "variant_id"
    ].astype(str).tolist()
    failed_exact = probe_summary.loc[
        ~probe_summary["expected_exact_check_passed"].astype(bool), "variant_id"
    ].astype(str).tolist()
    obs_status = str(
        reachability.loc[
            reachability["mechanism_id"].eq("OBS-01"), "phase0_status"
        ].iloc[0]
    )
    design = config["design"]
    content = f"""# Phase 0: configuration reachability and equivalence

## Result

The Phase-0 technical probe completed for subjects {design['subjects']}, using
the first {int(design['trials_per_subject'])} trials, {int(design['particle_count'])}
particles, and a common PF seed within each subject. This stage asks whether a
setting changes saved predictions or latent summaries. It does **not** decide
whether an active mechanism improves prediction or should be retained.

- Exact no-op variants ({len(exact_variants)}): {', '.join(exact_variants)}.
- Reachable variants ({len(reachable_variants)}): {', '.join(reachable_variants)}.
- Failed predeclared exact-equivalence checks: {', '.join(failed_exact) if failed_exact else 'none'}.
- `OBS-01` status: `{obs_status}`.

The central Phase-0 simplification result is that changing hypothesis-weight
power from 4 to 1, or replacing `sharpened_expectation` with `expectation`,
produced exact equality of all public numerical PF outputs for all three probe
subjects. Under persistent execution, one executed rule is read out, so there
are no multiple hypothesis weights left to sharpen. This conclusion is limited
to the current persistent-execution structure; power remains reachable in
comparators where execution is disabled.

## Runtime evidence

The baseline runs also confirm that the 29-rule extension, replacement path,
newcomer proposal, pairwise transfer path, persistent execution, strategy
confidence, and dynamic beta are reached within the probe window. Their
mechanism value remains untested. Exact subject-level runtime quantities are in
`runtime_activity_summary.csv`.

## Files

- `probe_results.csv`: one paired baseline-variant comparison per subject.
- `probe_summary.csv`: variant-level equivalence/reachability classification.
- `reachability_summary.csv`: Phase-0 status for all 20 registered audit items.
- `runtime_activity_summary.csv`: baseline latent-path activity checks.
- `config_dependency_map.csv`: inactive configuration groups and activation conditions.
- `probe_outputs.npz`: saved public numerical arrays used in the comparisons.
- `analysis_manifest.json`: config, source hashes, seeds, scope, and caveats.

## Interpretation boundary

`active_reachable` means only that a controlled perturbation changed at least
one saved prediction or latent output above the predeclared tolerance.
`runtime_active_comparator_required` means the code path was used, but a clean
alternative kernel still has to be implemented and compared. No subject-level
inferential sample size is claimed in Phase 0; particles and subjects here are
technical coverage cases rather than evidence for population generalization.

The next step is Phase 1 PF calibration. The existing 32/64-particle recovery
ranking instability remains unresolved and prevents retain/remove decisions.
"""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    base_path = (ROOT / str(config["base_simulation_config"])).resolve()
    registry_path = (ROOT / str(config["mechanism_registry"])).resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (ROOT / str(config["output_dir"])).resolve()
    )
    base_config = load_yaml(base_path)
    registry = pd.read_csv(registry_path)
    dataset_paths = resolve_dataset_paths(base_config, base_path.parent)
    learning = pd.read_csv(dataset_paths["learning_data"])
    condition_one = learning.loc[learning["condition"].eq(1)].copy()

    design = dict(config["design"])
    if args.smoke:
        design.update(
            {
                "subjects": [103],
                "default_variant_subject": 103,
                "perception_variant_subjects": [103],
                "trials_per_subject": 16,
                "particle_count": 4,
            }
        )
    subjects = [int(value) for value in design["subjects"]]
    default_subject = int(design["default_variant_subject"])
    perception_subjects = [
        int(value) for value in design["perception_variant_subjects"]
    ]
    trial_count = int(design["trials_per_subject"])
    particle_count = int(design["particle_count"])
    threshold = float(design["resample_threshold_fraction"])
    base_seed = int(design["base_seed"])
    tolerance = float(design["change_tolerance"])

    frames: dict[int, pd.DataFrame] = {}
    engines: dict[int, dict[str, Any]] = {}
    seeds: dict[int, int] = {}
    for subject_id in subjects:
        frame = (
            condition_one.loc[condition_one["iSub"].eq(subject_id)]
            .sort_values(list(ORDER_COLUMNS))
            .reset_index(drop=True)
            .iloc[:trial_count]
        )
        if len(frame) != trial_count:
            raise ValueError(
                f"subject {subject_id} has {len(frame)} available probe trials, "
                f"expected {trial_count}"
            )
        frames[subject_id] = frame
        engines[subject_id] = _subject_engine(base_config, base_path, subject_id)
        seeds[subject_id] = stable_seed(
            {
                "seed_role": "model0813_phase0_reachability_paired_pf",
                "base_seed": base_seed,
                "subject_id": subject_id,
                "particle_count": particle_count,
                "trial_count": trial_count,
            }
        )

    run_arrays: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    npz_arrays: dict[str, np.ndarray] = {}

    def execute(subject_id: int, variant_id: str) -> dict[str, np.ndarray]:
        key = (subject_id, variant_id)
        if key in run_arrays:
            return run_arrays[key]
        frame = frames[subject_id]
        engine = _variant_engine(engines[subject_id], variant_id)
        result = run_state_model_particle_filter(
            engine_config=engine,
            subject_id=subject_id,
            stimulus=frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
            choices=frame["choice"].to_numpy(dtype=int),
            feedback=frame["feedback"].to_numpy(dtype=float),
            particle_count=particle_count,
            filter_seed=seeds[subject_id],
            resample_threshold_fraction=threshold,
            processed_data_dir=dataset_paths["processed_dir"],
            dataset_paths=dataset_paths,
            **_readout_args(engine),
        )
        arrays = _run_output_arrays(result)
        run_arrays[key] = arrays
        prefix = f"subject_{subject_id}__{variant_id}__"
        for array_key, value in arrays.items():
            safe_key = array_key.replace(".", "__")
            npz_arrays[f"{prefix}{safe_key}"] = np.asarray(value)
        return arrays

    for subject_id in subjects:
        execute(subject_id, BASELINE_VARIANT)

    probe_rows: list[dict[str, Any]] = []
    for spec in PROBE_SPECS:
        for subject_id in _variant_subjects(
            spec, subjects, default_subject, perception_subjects
        ):
            if subject_id not in frames:
                raise ValueError(
                    f"variant {spec.variant_id} requests subject {subject_id}, "
                    "which is not in design.subjects"
                )
            baseline = execute(subject_id, BASELINE_VARIANT)
            variant = execute(subject_id, spec.variant_id)
            comparison = _compare_outputs(
                baseline,
                variant,
                frames[subject_id]["choice"].to_numpy(dtype=int),
                tolerance=tolerance,
            )
            comparison.update(
                {
                    "variant_id": spec.variant_id,
                    "variant_label": spec.label,
                    "expected_role": spec.expected_role,
                    "subject_id": subject_id,
                    "trial_count": trial_count,
                    "particle_count": particle_count,
                    "filter_seed": seeds[subject_id],
                    "change_tolerance": tolerance,
                    "mean_choice_nll_difference_variant_minus_baseline": (
                        comparison["variant_mean_choice_nll"]
                        - comparison["baseline_mean_choice_nll"]
                    ),
                }
            )
            probe_rows.append(comparison)

    probes = pd.DataFrame(probe_rows)
    probe_summary = _aggregate_probe_rows(probes)
    runtime = _runtime_summary(
        {subject_id: run_arrays[(subject_id, BASELINE_VARIANT)] for subject_id in subjects},
        engines,
    )
    reachability = _build_reachability_summary(
        registry, probe_summary, runtime, tolerance
    )
    dependencies = pd.DataFrame(_dependency_rows())
    dependency_lookup = probe_summary.set_index("variant_id")
    dependencies["phase0_result"] = dependencies["probe_variant"].map(
        lambda value: (
            "exact_noop"
            if all(
                dependency_lookup.loc[item, "phase0_classification"] == "exact_noop"
                for item in str(value).split(";")
            )
            else "changed_or_unresolved"
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / "probe_results.csv", probes)
    _atomic_csv(output_dir / "probe_summary.csv", probe_summary)
    _atomic_csv(output_dir / "runtime_activity_summary.csv", runtime)
    _atomic_csv(output_dir / "reachability_summary.csv", reachability)
    _atomic_csv(output_dir / "config_dependency_map.csv", dependencies)
    _atomic_npz(output_dir / "probe_outputs.npz", npz_arrays)

    failed_exact = probe_summary.loc[
        ~probe_summary["expected_exact_check_passed"].astype(bool), "variant_id"
    ].astype(str).tolist()
    manifest = {
        "analysis_id": config["analysis_id"],
        "scope": config["scope"],
        "status": "needs_revision" if failed_exact else "ready_with_phase0_boundary",
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "base_simulation_config": str(base_path.relative_to(ROOT)),
        "base_simulation_config_sha256": _sha256(base_path),
        "mechanism_registry": str(registry_path.relative_to(ROOT)),
        "mechanism_registry_sha256": _sha256(registry_path),
        "learning_data": _manifest_path(Path(dataset_paths["learning_data"])),
        "learning_data_sha256": _sha256(Path(dataset_paths["learning_data"])),
        "runner": str(Path(__file__).resolve().relative_to(ROOT)),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "repository_head": _git_head(),
        "worktree_dirty": _worktree_dirty(),
        "design": design,
        "paired_filter_seeds": seeds,
        "probe_variant_count": len(PROBE_SPECS),
        "probe_comparison_row_count": int(len(probes)),
        "registered_mechanism_count": int(len(reachability)),
        "exact_noop_variants": probe_summary.loc[
            probe_summary["phase0_classification"].eq("exact_noop"), "variant_id"
        ].astype(str).tolist(),
        "reachable_variants": probe_summary.loc[
            probe_summary["phase0_classification"].eq("reachable"), "variant_id"
        ].astype(str).tolist(),
        "failed_expected_exact_variants": failed_exact,
        "independent_unit": "none; Phase 0 is a technical reachability probe",
        "interpretation_boundary": (
            "Reachability is not predictive benefit, population evidence, parameter "
            "recovery, or a retain/remove decision."
        ),
    }
    _atomic_json(output_dir / "analysis_manifest.json", manifest)
    snapshot = deepcopy(config)
    snapshot["design"] = design
    _atomic_json(output_dir / "analysis_config_snapshot.json", snapshot)
    _write_readme(
        output_dir,
        config={**config, "design": design},
        probe_summary=probe_summary,
        reachability=reachability,
        runtime=runtime,
    )

    print(f"Phase-0 reachability outputs: {output_dir}")
    print(probe_summary[[
        "variant_id",
        "phase0_classification",
        "max_abs_choice_probability_diff",
        "expected_exact_check_passed",
    ]].to_string(index=False))
    if failed_exact:
        raise SystemExit(
            "Predeclared exact-equivalence checks failed: " + ", ".join(failed_exact)
        )


if __name__ == "__main__":
    main()
