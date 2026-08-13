#!/usr/bin/env python3
"""Audit whether model_0804 forgets different early HFW histories."""

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
from typing import Any

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
    _parameters,
)
from src.Bayesian_state.reference_models.model_0804.forgetting import (  # noqa: E402
    couple_model0804_histories,
    sample_model0804_filtered_anchor_states,
)


DEFAULT_CONFIG = ROOT / "configs/model_0804_forgetting_audit.yaml"
DEFAULT_OUTPUT = ROOT / "results/zhuran/model_0804_cond1/forgetting_20260804_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def _load_audit_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("forgetting audit config must contain a mapping")
    return payload


def _quantile_curves(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": np.nanmean(values, axis=(0, 1)),
        "median": np.nanquantile(values, 0.50, axis=(0, 1)),
        "p95": np.nanquantile(values, 0.95, axis=(0, 1)),
    }


def _first_terminal_stable_lag(
    conditions: list[np.ndarray],
) -> int | None:
    combined = np.logical_and.reduce(conditions)
    for index in range(combined.size):
        if np.all(combined[index:]):
            return index + 1
    return None


def _probe_summary(
    probe_id: str,
    parameters,
    anchors: np.ndarray,
    active_distance: np.ndarray,
    active_equal: np.ndarray,
    state_exact_equal: np.ndarray,
    regenerated: np.ndarray,
    ever_regenerated: np.ndarray,
    omega_tv: np.ndarray,
    choice_difference: np.ndarray,
    signed_choice_difference: np.ndarray,
    memory_difference: np.ndarray,
    initial_distinct: np.ndarray,
    audit: dict[str, Any],
    runtime_seconds: float,
    left_panel,
    right_panel,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    active_curve = _quantile_curves(active_distance)
    omega_curve = _quantile_curves(omega_tv)
    choice_curve = _quantile_curves(choice_difference)
    memory_curve = _quantile_curves(memory_difference)
    equal_fraction = np.mean(active_equal, axis=(0, 1))
    exact_equal_fraction = np.mean(state_exact_equal, axis=(0, 1))
    ever_regenerated_fraction = np.mean(ever_regenerated, axis=(0, 1))
    theoretical_no_reset = np.power(
        1.0 - float(parameters.rho),
        np.arange(1, choice_difference.shape[2] + 1, dtype=float),
    )
    gate = audit["forgetting_gate"]
    predictive = gate["predictive"]
    state = gate["state"]
    terminal_window = int(gate["terminal_window"])
    if terminal_window < 1 or terminal_window > choice_difference.shape[2]:
        raise ValueError("terminal_window must lie inside the audit horizon")
    predictive_conditions = [
        choice_curve["mean"]
        <= float(predictive["maximum_mean_choice_probability_difference"]),
        choice_curve["p95"]
        <= float(predictive["maximum_p95_choice_probability_difference"]),
    ]
    state_conditions = [
        omega_curve["median"]
        <= float(state["maximum_median_omega_total_variation"]),
        active_curve["median"]
        <= float(state["maximum_median_active_distance"]),
    ]
    predictive_pass = bool(
        all(np.all(condition[-terminal_window:]) for condition in predictive_conditions)
    )
    state_pass = bool(
        all(np.all(condition[-terminal_window:]) for condition in state_conditions)
    )
    choice_tail_threshold = float(
        predictive["maximum_p95_choice_probability_difference"]
    )
    terminal_choice = choice_difference[:, :, -terminal_window:]
    terminal_pair_maximum = np.max(terminal_choice, axis=2)
    final_choice = choice_difference[:, :, -1]
    anchor_marginal_choice_difference = np.abs(
        np.mean(signed_choice_difference, axis=1)
    )
    per_anchor = []
    for anchor_index, anchor in enumerate(anchors):
        per_anchor.append(
            {
                "anchor_zero_based": int(anchor),
                "initial_distinct_pair_fraction": float(
                    np.mean(initial_distinct[anchor_index])
                ),
                "terminal_mean_choice_probability_difference": float(
                    np.mean(choice_difference[anchor_index, :, -terminal_window:])
                ),
                "terminal_p95_choice_probability_difference": float(
                    np.quantile(
                        choice_difference[anchor_index, :, -terminal_window:], 0.95
                    )
                ),
                "terminal_p99_choice_probability_difference": float(
                    np.quantile(
                        choice_difference[anchor_index, :, -terminal_window:], 0.99
                    )
                ),
                "terminal_fraction_choice_difference_above_gate": float(
                    np.mean(
                        choice_difference[anchor_index, :, -terminal_window:]
                        > choice_tail_threshold
                    )
                ),
                "terminal_median_omega_total_variation": float(
                    np.median(omega_tv[anchor_index, :, -terminal_window:])
                ),
                "terminal_median_active_distance": float(
                    np.median(active_distance[anchor_index, :, -terminal_window:])
                ),
                "terminal_active_equal_fraction": float(
                    np.mean(active_equal[anchor_index, :, -terminal_window:])
                ),
            }
        )
    summary = {
        "probe_id": str(probe_id),
        "runtime_seconds": float(runtime_seconds),
        "parameters": {
            "gamma": float(parameters.gamma),
            "w0": float(parameters.w0),
            "kappa": float(parameters.kappa),
            "m": float(parameters.m),
            "g": float(parameters.g),
            "lapse": float(parameters.lapse),
            "rho": float(parameters.rho),
        },
        "anchor_filter": {
            "left_seed": int(left_panel.filter_seed),
            "right_seed": int(right_panel.filter_seed),
            "left_resampling_count_to_last_anchor": int(
                np.sum(left_panel.resampled)
            ),
            "right_resampling_count_to_last_anchor": int(
                np.sum(right_panel.resampled)
            ),
            "left_minimum_ess": float(np.nanmin(left_panel.pre_resampling_ess)),
            "right_minimum_ess": float(np.nanmin(right_panel.pre_resampling_ess)),
            "overall_initial_distinct_pair_fraction": float(
                np.mean(initial_distinct)
            ),
        },
        "terminal_window": terminal_window,
        "terminal_metrics": {
            "maximum_mean_choice_probability_difference": float(
                np.max(choice_curve["mean"][-terminal_window:])
            ),
            "maximum_p95_choice_probability_difference": float(
                np.max(choice_curve["p95"][-terminal_window:])
            ),
            "maximum_median_omega_total_variation": float(
                np.max(omega_curve["median"][-terminal_window:])
            ),
            "maximum_median_active_distance": float(
                np.max(active_curve["median"][-terminal_window:])
            ),
            "minimum_active_equal_fraction": float(
                np.min(equal_fraction[-terminal_window:])
            ),
            "pooled_mean_choice_probability_difference": float(
                np.mean(terminal_choice)
            ),
            "pooled_p95_choice_probability_difference": float(
                np.quantile(terminal_choice, 0.95)
            ),
            "pooled_p99_choice_probability_difference": float(
                np.quantile(terminal_choice, 0.99)
            ),
            "pooled_fraction_choice_difference_above_gate": float(
                np.mean(terminal_choice > choice_tail_threshold)
            ),
            "pooled_fraction_choice_difference_above_0p1": float(
                np.mean(terminal_choice > 0.1)
            ),
            "fraction_pairs_with_any_terminal_difference_above_gate": float(
                np.mean(terminal_pair_maximum > choice_tail_threshold)
            ),
            "final_lag_mean_choice_probability_difference": float(
                np.mean(final_choice)
            ),
            "final_lag_p95_choice_probability_difference": float(
                np.quantile(final_choice, 0.95)
            ),
            "final_lag_p99_choice_probability_difference": float(
                np.quantile(final_choice, 0.99)
            ),
            "final_lag_fraction_choice_difference_above_gate": float(
                np.mean(final_choice > choice_tail_threshold)
            ),
            "final_lag_active_equal_fraction": float(
                np.mean(active_equal[:, :, -1])
            ),
            "final_lag_state_exact_equal_fraction": float(
                np.mean(state_exact_equal[:, :, -1])
            ),
            "final_lag_ever_regenerated_fraction": float(
                np.mean(ever_regenerated[:, :, -1])
            ),
            "theoretical_no_reset_probability": float(
                theoretical_no_reset[-1]
            ),
            "post_reset_exact_coupling_violations": int(
                np.sum(ever_regenerated & ~state_exact_equal)
            ),
            "maximum_anchor_marginal_choice_difference": float(
                np.max(
                    anchor_marginal_choice_difference[:, -terminal_window:]
                )
            ),
            "mean_anchor_marginal_choice_difference": float(
                np.mean(
                    anchor_marginal_choice_difference[:, -terminal_window:]
                )
            ),
            "final_lag_maximum_anchor_marginal_choice_difference": float(
                np.max(anchor_marginal_choice_difference[:, -1])
            ),
        },
        "predictive_forgetting_pass": predictive_pass,
        "state_forgetting_pass": state_pass,
        "first_terminal_predictive_stable_lag": _first_terminal_stable_lag(
            predictive_conditions
        ),
        "first_terminal_state_stable_lag": _first_terminal_stable_lag(
            state_conditions
        ),
        "per_anchor": per_anchor,
    }
    curves = {
        "active_distance_mean": active_curve["mean"],
        "active_distance_median": active_curve["median"],
        "active_distance_p95": active_curve["p95"],
        "active_equal_fraction": equal_fraction,
        "state_exact_equal_fraction": exact_equal_fraction,
        "ever_regenerated_fraction": ever_regenerated_fraction,
        "theoretical_no_reset_probability": theoretical_no_reset,
        "omega_tv_mean": omega_curve["mean"],
        "omega_tv_median": omega_curve["median"],
        "omega_tv_p95": omega_curve["p95"],
        "choice_difference_mean": choice_curve["mean"],
        "choice_difference_median": choice_curve["median"],
        "choice_difference_p95": choice_curve["p95"],
        "anchor_marginal_choice_difference": anchor_marginal_choice_difference,
        "memory_difference_mean": memory_curve["mean"],
        "memory_difference_median": memory_curve["median"],
        "memory_difference_p95": memory_curve["p95"],
    }
    return summary, curves


def _route_decision(rows: list[dict[str, Any]]) -> str:
    by_id = {str(row["probe_id"]): row for row in rows}
    if "frozen_dual" not in by_id or "fade_only" not in by_id:
        return "parameter_sensitivity_only_no_primary_route_decision"
    dual = by_id["frozen_dual"]
    fade = by_id["fade_only"]
    dual_predictive = bool(dual["predictive_forgetting_pass"])
    dual_state = bool(dual["state_forgetting_pass"])
    fade_both = bool(
        fade["predictive_forgetting_pass"] and fade["state_forgetting_pass"]
    )
    if dual_predictive and dual_state:
        return "bridge_filter_eligible"
    if dual_predictive and not dual_state:
        return "behavioral_forgetting_without_state_coupling"
    if fade_both:
        return "static_memory_is_structural_obstacle"
    return "active_set_dynamics_is_structural_obstacle"


def main() -> None:
    args = parse_args()
    started = time.time()
    config_path = args.config.resolve()
    output = args.output.resolve()
    report_path = output / "forgetting_report.json"
    trace_path = output / "forgetting_trace.npz"
    if report_path.exists() and trace_path.exists() and not args.force:
        print(f"[forgetting] skip completed output={report_path}", flush=True)
        return

    audit = _load_audit_config(config_path)
    base_path = ROOT / str(audit["base_config"])
    base = load_config(base_path)
    scope = audit["data_scope"]
    subject_id = int(scope["subject_id"])
    model_id = str(scope["model_id"])
    anchors = np.asarray(scope["anchors_zero_based"], dtype=int)
    horizon = int(scope["horizon"])
    frame, subjects, input_audit = validate_and_load_inputs(base, {subject_id})
    if subjects != [subject_id]:
        raise ValueError("forgetting audit did not resolve the frozen subject")
    priors, kernels_by_prior, geometry_audit = build_frozen_geometry(base)
    prior_id = str(base["rule_space"]["primary_prior"])
    prior = priors[prior_id]
    kernels = kernels_by_prior[prior_id]
    cache_audit = validate_subject_cache(base, frame, subject_id)
    arrays = _load_subject_arrays(cache_audit, None)
    if int(np.max(anchors)) + horizon >= len(arrays["choice"]):
        raise ValueError("frozen anchors and horizon exceed the subject sequence")

    sampling = audit["filter_anchor_sampling"]
    pair_count = int(sampling["posterior_state_pairs_per_anchor"])
    filter_seeds = [int(value) for value in sampling["independent_filter_seeds"]]
    if len(filter_seeds) != 2:
        raise ValueError("forgetting audit requires exactly two filter seeds")
    base_parameters = _parameters(
        base, "FA2" if model_id == "FA2R" else model_id
    )
    base_parameters = replace(
        base_parameters, lapse=float(audit["fixed_parameters"]["lapse"])
    )
    capacity = int(base["architecture"]["capacity"])
    payload: dict[str, np.ndarray] = {
        "anchors": anchors,
        "lags": np.arange(1, horizon + 1, dtype=int),
    }
    rows = []
    for probe in audit["fixed_parameters"]["probes"]:
        probe_id = str(probe["id"])
        parameters = replace(
            base_parameters,
            w0=float(probe.get("w0", base_parameters.w0)),
            gamma=float(probe.get("gamma", base_parameters.gamma)),
            m=float(probe.get("m", base_parameters.m)),
            g=float(probe.get("g", base_parameters.g)),
            rho=float(probe.get("rho", base_parameters.rho)),
        )
        print(
            f"[forgetting] probe={probe_id} subject={subject_id} "
            f"anchors={anchors.tolist()} pairs={pair_count} horizon={horizon}",
            flush=True,
        )
        probe_started = time.time()
        panels = [
            sample_model0804_filtered_anchor_states(
                arrays["q"],
                arrays["choice"],
                arrays["feedback"],
                prior,
                kernels,
                model_id=model_id,
                parameters=parameters,
                capacity=capacity,
                anchors=anchors,
                particle_count=int(sampling["particle_count"]),
                sample_count=pair_count,
                filter_seed=seed,
                resample_threshold_fraction=float(
                    sampling["resample_threshold_fraction"]
                ),
            )
            for seed in filter_seeds
        ]
        active_distance = np.zeros((len(anchors), pair_count, horizon))
        active_equal = np.zeros((len(anchors), pair_count, horizon), dtype=bool)
        state_exact_equal = np.zeros(
            (len(anchors), pair_count, horizon), dtype=bool
        )
        regenerated = np.zeros(
            (len(anchors), pair_count, horizon), dtype=bool
        )
        ever_regenerated = np.zeros(
            (len(anchors), pair_count, horizon), dtype=bool
        )
        omega_tv = np.zeros((len(anchors), pair_count, horizon))
        choice_difference = np.zeros((len(anchors), pair_count, horizon))
        signed_choice_difference = np.zeros(
            (len(anchors), pair_count, horizon)
        )
        memory_difference = np.full(
            (len(anchors), pair_count, horizon), np.nan
        )
        initial_distinct = np.zeros((len(anchors), pair_count), dtype=bool)
        for anchor_index, anchor in enumerate(anchors):
            left_states = panels[0].states_by_anchor[int(anchor)]
            right_states = panels[1].states_by_anchor[int(anchor)]
            for pair_index, (left, right) in enumerate(
                zip(left_states, right_states)
            ):
                initial_distinct[anchor_index, pair_index] = not (
                    np.array_equal(left.active, right.active)
                    and np.allclose(left.omega, right.omega, atol=1e-14, rtol=0.0)
                )
            trace = couple_model0804_histories(
                left_states,
                right_states,
                arrays["q"],
                arrays["choice"],
                arrays["feedback"],
                prior,
                kernels,
                model_id=model_id,
                parameters=parameters,
                capacity=capacity,
                anchor_trial=int(anchor),
                horizon=horizon,
                coupling_seed=_stable_seed(
                    int(audit["common_future"]["coupling_seed"]),
                    probe_id,
                    int(anchor),
                ),
            )
            active_distance[anchor_index] = trace.active_distance
            active_equal[anchor_index] = trace.active_equal
            state_exact_equal[anchor_index] = trace.state_exact_equal
            regenerated[anchor_index] = trace.regenerated
            ever_regenerated[anchor_index] = trace.ever_regenerated
            omega_tv[anchor_index] = trace.omega_total_variation
            choice_difference[anchor_index] = trace.choice_probability_difference
            signed_choice_difference[anchor_index] = (
                trace.signed_choice_probability_difference
            )
            memory_difference[anchor_index] = trace.common_memory_delta_difference
        runtime = float(time.time() - probe_started)
        summary, curves = _probe_summary(
            probe_id,
            parameters,
            anchors,
            active_distance,
            active_equal,
            state_exact_equal,
            regenerated,
            ever_regenerated,
            omega_tv,
            choice_difference,
            signed_choice_difference,
            memory_difference,
            initial_distinct,
            audit,
            runtime,
            panels[0],
            panels[1],
        )
        rows.append(summary)
        prefix = f"{probe_id}_"
        payload.update(
            {
                prefix + "active_distance": active_distance,
                prefix + "active_equal": active_equal,
                prefix + "state_exact_equal": state_exact_equal,
                prefix + "regenerated": regenerated,
                prefix + "ever_regenerated": ever_regenerated,
                prefix + "omega_total_variation": omega_tv,
                prefix + "choice_probability_difference": choice_difference,
                prefix + "signed_choice_probability_difference": (
                    signed_choice_difference
                ),
                prefix + "common_memory_delta_difference": memory_difference,
                prefix + "initial_distinct": initial_distinct,
                **{prefix + key: value for key, value in curves.items()},
            }
        )
        print(
            f"[forgetting] done probe={probe_id} runtime={runtime:.1f}s "
            f"predictive_pass={summary['predictive_forgetting_pass']} "
            f"state_pass={summary['state_forgetting_pass']}",
            flush=True,
        )

    _atomic_npz(trace_path, **payload)
    report = {
        "analysis_id": str(audit["analysis_id"]),
        "status": "forgetting_audit_complete",
        "scope": str(audit["scope"]),
        "runtime_seconds": float(time.time() - started),
        "route_decision": _route_decision(rows),
        "subject_id": subject_id,
        "model_id": model_id,
        "n_trials": int(len(arrays["choice"])),
        "anchors_zero_based": anchors.tolist(),
        "horizon": horizon,
        "pair_count_per_anchor": pair_count,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "base_config_path": str(base_path.resolve()),
        "base_config_sha256": _sha256(base_path),
        "implementation_sha256": {
            "model_0804.py": _sha256(
                ROOT / "src/Bayesian_state/reference_models/model_0804/core.py"
            ),
            "model_0804/forgetting.py": _sha256(
                ROOT / "src/Bayesian_state/reference_models/model_0804/forgetting.py"
            ),
            "diagnostic_runner": _sha256(Path(__file__).resolve()),
            "forgetting_tests": _sha256(
                ROOT / "tests/test_model_0804_forgetting.py"
            ),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "input_audit": input_audit,
        "cache_audit": cache_audit,
        "geometry_audit": geometry_audit,
        "forgetting_gate": audit["forgetting_gate"],
        "probes": rows,
        "guardrails": list(audit["guardrails"]),
    }
    _atomic_json(report_path, report)
    print(
        f"FORGETTING status={report['status']} route={report['route_decision']} "
        f"output={report_path} runtime={report['runtime_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
