#!/usr/bin/env python3
"""Rolling-origin development comparison of static FA2 and one-signal FA3-M."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0803_cond1 import (  # noqa: E402
    PRIMARY_PRIOR,
    build_frozen_geometry,
    validate_and_load_inputs,
    validate_subject_cache,
)
from scripts.run_model_0805_real_predictive import load_config  # noqa: E402
from scripts.run_model_0806_dynamic_m_recovery import (  # noqa: E402
    atomic_csv,
    atomic_json,
    atomic_savez,
    dynamic_signal,
    load_geometry,
    save_geometry,
    softmax,
)
from src.Bayesian_state.reference_models.model_0804.core import (  # noqa: E402
    Model0804Parameters,
)
from src.Bayesian_state.reference_models.model_0804.core import (  # noqa: E402
    run_model0804_particle_filter,
)
from src.Bayesian_state.evaluation.evaluator import (  # noqa: E402
    ModelEvaluator as ModelEvaluationReport,
)


DEFAULT_CONFIG = ROOT / "configs/model_0806_dynamic_m_real_rolling.yaml"
DEFAULT_OUTPUT = (
    ROOT / "results/zhuran/model_0806_cond1/dynamic_m_real_rolling_20260806_v1"
)


SHARED_EVALUATOR = ModelEvaluationReport()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=32)
    parser.add_argument(
        "--phase", choices=("all", "fit", "report"), default="all"
    )
    parser.add_argument("--subjects", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("real rolling config must be a mapping")
    return payload


def parse_subjects(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def candidate_grid(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    signal = dynamic_signal(config)
    beta_key = f"beta_{signal}"
    candidate_prefix = "FA3MS" if signal == "surprise" else "FA3MU"
    fixed = config["fixed_parameters"]
    support = config["candidate_support"]
    rows: list[dict[str, Any]] = []
    for memory in config["nuisance_support"]["memory_states"]:
        for kappa in config["nuisance_support"]["kappa"]:
            for lapse in config["nuisance_support"]["lapse"]:
                nuisance = {
                    "memory_id": str(memory["id"]),
                    "gamma": float(memory["gamma"]),
                    "w0": float(memory["w0"]),
                    "kappa": float(kappa),
                    "lapse": float(lapse),
                    "g": float(fixed["g"]),
                    "rho": float(fixed["rho"]),
                }
                for m_value in support["m"]:
                    base_id = (
                        f"{memory['id']}_k{float(kappa):.2f}_l{float(lapse):.2f}"
                        f"_m{float(m_value):.2f}"
                    )
                    rows.append({
                        "candidate_id": f"FA2_{base_id}",
                        "family": "static",
                        **nuisance,
                        "m": float(m_value),
                        "phi": 0.0,
                        beta_key: 0.0,
                    })
                    for phi in support["phi"]:
                        for beta in support[beta_key]:
                            rows.append({
                                "candidate_id": (
                                    f"{candidate_prefix}_{base_id}_p{float(phi):.2f}"
                                    f"_b{float(beta):.2f}"
                                ),
                                "family": "dynamic",
                                **nuisance,
                                "m": float(m_value),
                                "phi": float(phi),
                                beta_key: float(beta),
                            })
    identifiers = [row["candidate_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("candidate identifiers are not unique")
    return rows


def parameters(
    config: Mapping[str, Any], candidate: Mapping[str, Any]
) -> Model0804Parameters:
    signal = dynamic_signal(config)
    standard = config[f"{signal}_standardization"]
    beta = float(candidate[f"beta_{signal}"])
    signal_parameters = (
        {
            "m_beta_surprise": beta,
            "surprise_center": float(standard["center"]),
            "surprise_scale": float(standard["scale"]),
        }
        if signal == "surprise"
        else {
            "m_beta_uncertainty": beta,
            "uncertainty_center": float(standard["center"]),
            "uncertainty_scale": float(standard["scale"]),
        }
    )
    return Model0804Parameters(
        gamma=float(candidate["gamma"]),
        w0=float(candidate["w0"]),
        kappa=float(candidate["kappa"]),
        m=float(candidate["m"]),
        g=float(candidate["g"]),
        lapse=float(candidate["lapse"]),
        rho=float(candidate["rho"]),
        dynamic_m=beta > 0.0,
        m_phi=float(candidate["phi"]),
        **signal_parameters,
    )


def component_path(
    output: Path,
    subject_id: int,
    candidate_id: str,
    seed: int,
    primary_seed: int,
) -> Path:
    suffix = "" if int(seed) == int(primary_seed) else f"_seed{int(seed)}"
    return (
        output / "components" / f"subject_{subject_id}"
        / f"{candidate_id}{suffix}.npz"
    )


def fit_task(task: Mapping[str, Any]) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    path = Path(task["output_path"])
    if path.exists() and not bool(task["force"]):
        require_diagnostic_traces = str(task["candidate"]["family"]) == "static"
        if not require_diagnostic_traces:
            return {"path": str(path), "skipped": True}
        with np.load(path, allow_pickle=False) as existing:
            required = {"feedback_surprise", "feedback_uncertainty"}
            if required.issubset(existing.files):
                return {"path": str(path), "skipped": True}
    config = read_yaml(Path(task["config_path"]))
    prior, kernels = load_geometry(Path(task["geometry_path"]))
    with np.load(Path(task["q_path"]), allow_pickle=False) as payload:
        q = payload["q"].astype(float)
    with np.load(Path(task["prediction_path"]), allow_pickle=False) as payload:
        choice = payload["choice"].astype(int)
        feedback = payload["feedback"].astype(float)
        category = payload["category"].astype(int)
    candidate = task["candidate"]
    trace = run_model0804_particle_filter(
        q,
        choice,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=parameters(config, candidate),
        capacity=int(config["design"]["capacity"]),
        particle_count=int(task["particle_count"]),
        filter_seed=int(task["filter_seed"]),
    )
    observed = trace.probabilities[np.arange(choice.size), choice]
    correct = trace.probabilities[np.arange(choice.size), category]
    metadata = {
        "subject_id": int(task["subject_id"]),
        "candidate": candidate,
        "particle_count": int(task["particle_count"]),
        "filter_seed": int(task["filter_seed"]),
        "nll": float(trace.nll),
    }
    atomic_savez(
        path,
        probabilities=trace.probabilities.astype(np.float32),
        observed_log_probability=np.log(np.clip(observed, 1e-300, 1.0)),
        correct_probability=correct.astype(np.float32),
        predictive_m=np.asarray(trace.predictive_m, dtype=np.float32),
        replacement_fraction=trace.predictive_replacement_fraction.astype(np.float32),
        feedback_surprise=np.asarray(trace.feedback_surprise, dtype=np.float32),
        feedback_uncertainty=np.asarray(
            trace.feedback_uncertainty, dtype=np.float32
        ),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {"path": str(path), "skipped": False}


def prepare(
    config: Mapping[str, Any], output: Path, selected: set[int] | None
) -> tuple[list[dict[str, Any]], Path]:
    base = load_config(ROOT / str(config["base_config"]))
    frame, subjects, data_audit = validate_and_load_inputs(base, selected)
    priors, kernels, geometry_audit = build_frozen_geometry(base)
    geometry_path = output / "geometry.npz"
    if not geometry_path.exists():
        save_geometry(geometry_path, priors[PRIMARY_PRIOR], kernels[PRIMARY_PRIOR])
    subject_audits = [validate_subject_cache(base, frame, subject) for subject in subjects]
    atomic_json(output / "input_audit.json", {
        "data": data_audit,
        "geometry": geometry_audit,
        "subjects": subject_audits,
    })
    return subject_audits, geometry_path


def load_component(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "probabilities": payload["probabilities"].astype(float),
            "observed_log_probability": payload["observed_log_probability"].astype(float),
            "correct_probability": payload["correct_probability"].astype(float),
            "predictive_m": payload["predictive_m"].astype(float),
            "replacement_fraction": payload["replacement_fraction"].astype(float),
        }


def stable_logmeanexp(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    maximum = float(np.max(array))
    return maximum + math.log(float(np.mean(np.exp(array - maximum))))


def phase_label(config: Mapping[str, Any], evaluation_block: int) -> str:
    phases = config["phases"]
    if evaluation_block in {int(value) for value in phases["early_evaluation_blocks"]}:
        return "early"
    if evaluation_block in {int(value) for value in phases["middle_evaluation_blocks"]}:
        return "middle"
    if evaluation_block >= int(phases["late_evaluation_block_minimum"]):
        return "late"
    return "unclassified"


def curve_correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    actual_curve = pd.Series(observed).rolling(16, min_periods=4).mean().to_numpy()
    predicted_curve = pd.Series(predicted).rolling(16, min_periods=4).mean().to_numpy()
    keep = np.isfinite(actual_curve) & np.isfinite(predicted_curve)
    if np.sum(keep) < 4 or np.std(actual_curve[keep]) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(actual_curve[keep], predicted_curve[keep])[0, 1])


def shared_choice_metrics(
    probabilities: np.ndarray,
    choices: np.ndarray,
    observed_correct: np.ndarray,
    predicted_correct: np.ndarray,
    window_size: int = 16,
) -> dict[str, float]:
    """Use the repository's canonical Brier and accuracy-curve definitions."""
    info = {
        "pred_category_probs": np.asarray(probabilities, dtype=float),
        "observed_choice_index": np.asarray(choices, dtype=int),
        "valid_trial_mask": np.ones(len(choices), dtype=bool),
        "true_acc": np.asarray(observed_correct, dtype=float),
        "pred_acc": np.asarray(predicted_correct, dtype=float),
        "window_size": int(window_size),
    }
    brier = SHARED_EVALUATOR.compute_choice_brier_metrics(
        info, window_size=window_size
    )
    accuracy = SHARED_EVALUATOR.compute_accuracy_metrics(
        info, window_size=window_size
    )
    curve = SHARED_EVALUATOR._accuracy_curve_summary(accuracy)
    trial_brier = np.asarray(brier.get("choice_brier", []), dtype=float)
    true_curve = np.asarray(accuracy.get("sliding_true_acc", []), dtype=float)
    predicted_curve = np.asarray(
        accuracy.get("sliding_pred_acc", []), dtype=float
    )
    keep = np.isfinite(true_curve) & np.isfinite(predicted_curve)
    if (
        np.sum(keep) >= 4
        and np.std(true_curve[keep]) > 1e-12
        and np.std(predicted_curve[keep]) > 1e-12
    ):
        correlation = float(
            np.corrcoef(true_curve[keep], predicted_curve[keep])[0, 1]
        )
    else:
        correlation = float("nan")
    return {
        "choice_brier": (
            float(np.nanmean(trial_brier))
            if np.any(np.isfinite(trial_brier))
            else float("nan")
        ),
        "accuracy_curve_mae": float(curve["acc_mae"]),
        "accuracy_curve_rmse": float(curve["acc_rmse"]),
        "accuracy_curve_correlation": correlation,
    }


def finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array[np.isfinite(array)])) if np.any(np.isfinite(array)) else float("nan")


def component_integrity(
    config: Mapping[str, Any],
    output: Path,
    subject_audits: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Reject mixed or incomplete component directories before reporting."""
    candidates = candidate_grid(config)
    seeds = [int(value) for value in config["design"]["filter_seeds"]]
    primary_seed = seeds[0]
    expected = {
        component_path(
            output,
            int(audit["subject_id"]),
            str(candidate["candidate_id"]),
            seed,
            primary_seed,
        ).resolve()
        for audit in subject_audits
        for candidate in candidates
        for seed in seeds
    }
    actual = {
        path.resolve() for path in (output / "components").rglob("*.npz")
    }
    return {
        "expected_files": len(expected),
        "actual_files": len(actual),
        "missing_files": len(expected - actual),
        "unexpected_files": len(actual - expected),
        "passed": expected == actual,
    }


def reproduce_0805(config: Mapping[str, Any]) -> dict[str, Any]:
    root = ROOT / str(config["model_0805_output"])
    subjects = pd.read_csv(root / "outer_holdout_subjects.csv")
    with (root / "outer_holdout_report.json").open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    reported = {row["model_key"]: row for row in report["group"]}
    result: dict[str, Any] = {}
    for model_key in ("FS_H0", "FA2_M3"):
        selected = subjects[subjects.model_key == model_key]
        recomputed = float(selected["nll_per_trial"].mean())
        expected = float(reported[model_key]["mean_nll_per_trial"])
        if not np.isclose(recomputed, expected, atol=1e-12, rtol=0.0):
            raise AssertionError(f"0805 {model_key} group NLL did not reproduce")
        result[model_key] = {
            "subjects": int(len(selected)),
            "recomputed_mean_nll_per_trial": recomputed,
            "reported_mean_nll_per_trial": expected,
            "exact_match": True,
        }
    result["FA2_M3_minus_FS_H0"] = float(
        result["FA2_M3"]["recomputed_mean_nll_per_trial"]
        - result["FS_H0"]["recomputed_mean_nll_per_trial"]
    )
    return result


def summarize(
    config: Mapping[str, Any], output: Path, subject_audits: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    integrity = component_integrity(config, output, subject_audits)
    if not integrity["passed"]:
        raise RuntimeError(
            "component directory is mixed or incomplete: "
            f"expected={integrity['expected_files']}, "
            f"actual={integrity['actual_files']}, "
            f"missing={integrity['missing_files']}, "
            f"unexpected={integrity['unexpected_files']}"
        )
    candidates = candidate_grid(config)
    filter_seeds = [int(value) for value in config["design"]["filter_seeds"]]
    primary_seed = filter_seeds[0]
    component_specs = [
        (candidate, seed) for candidate in candidates for seed in filter_seeds
    ]
    family_indices = {
        family: np.asarray(
            [
                index
                for index, (value, _) in enumerate(component_specs)
                if value["family"] == family
            ],
            dtype=int,
        )
        for family in ("static", "dynamic")
    }
    block_size = int(config["design"]["block_size"])
    minimum_train = int(config["design"]["minimum_training_trials"])
    fold_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    for audit in subject_audits:
        subject_id = int(audit["subject_id"])
        components = [
            load_component(
                component_path(
                    output,
                    subject_id,
                    candidate["candidate_id"],
                    seed,
                    primary_seed,
                )
            )
            for candidate, seed in component_specs
        ]
        probabilities = np.stack([value["probabilities"] for value in components])
        log_probability = np.stack(
            [value["observed_log_probability"] for value in components]
        )
        correct_probability = np.stack(
            [value["correct_probability"] for value in components]
        )
        predictive_m = np.stack([value["predictive_m"] for value in components])
        with np.load(Path(audit["prediction_path"]), allow_pickle=False) as payload:
            choice = payload["choice"].astype(int)
            feedback = payload["feedback"].astype(float)
        n_trials = choice.size
        subject_folds: list[dict[str, Any]] = []
        for origin in range(minimum_train, n_trials, block_size):
            stop = min(origin + block_size, n_trials)
            if stop <= origin:
                continue
            evaluation = slice(origin, stop)
            family_payload: dict[str, Any] = {}
            family_log_evidence: dict[str, float] = {}
            for family, indices in family_indices.items():
                train_log_likelihood = np.sum(log_probability[indices, :origin], axis=1)
                weights = softmax(train_log_likelihood)
                mixture = np.tensordot(weights, probabilities[indices, evaluation, :], axes=(0, 0))
                mixture /= mixture.sum(axis=1, keepdims=True)
                observed_probability = mixture[
                    np.arange(stop - origin), choice[evaluation]
                ]
                predicted_correct = np.tensordot(
                    weights, correct_probability[indices, evaluation], axes=(0, 0)
                )
                shared = shared_choice_metrics(
                    mixture,
                    choice[evaluation],
                    feedback[evaluation],
                    predicted_correct,
                    window_size=16,
                )
                family_payload[family] = {
                    "nll": float(-np.sum(np.log(np.clip(observed_probability, 1e-300, 1.0)))),
                    **shared,
                    "predicted_correct_mean": float(np.mean(predicted_correct)),
                    "accuracy_bias": float(np.mean(predicted_correct) - np.mean(feedback[evaluation])),
                    "mean_predictive_m": float(
                        np.mean(np.tensordot(weights, predictive_m[indices, evaluation], axes=(0, 0)))
                    ),
                }
                family_log_evidence[family] = stable_logmeanexp(train_log_likelihood)
            dynamic_probability = float(
                softmax(np.asarray([
                    family_log_evidence["static"], family_log_evidence["dynamic"]
                ]))[1]
            )
            evaluation_block = origin // block_size + 1
            row: dict[str, Any] = {
                "subject_id": subject_id,
                "origin": int(origin),
                "stop": int(stop),
                "n_trials": int(stop - origin),
                "evaluation_block": int(evaluation_block),
                "phase": phase_label(config, evaluation_block),
                "observed_accuracy": float(np.mean(feedback[evaluation])),
                "prefix_dynamic_family_posterior": dynamic_probability,
            }
            for family in ("static", "dynamic"):
                for key, value in family_payload[family].items():
                    row[f"{family}_{key}"] = value
            row["delta_nll_static_minus_dynamic"] = (
                row["static_nll"] - row["dynamic_nll"]
            )
            row["delta_nll_per_trial"] = (
                row["delta_nll_static_minus_dynamic"] / row["n_trials"]
            )
            row["delta_absolute_accuracy_bias_static_minus_dynamic"] = (
                abs(row["static_accuracy_bias"]) - abs(row["dynamic_accuracy_bias"])
            )
            subject_folds.append(row)
            fold_rows.append(row)
        if not subject_folds:
            continue
        total_trials = sum(row["n_trials"] for row in subject_folds)
        subject_row: dict[str, Any] = {
            "subject_id": subject_id,
            "folds": len(subject_folds),
            "n_trials": total_trials,
            "delta_nll_per_trial": float(
                sum(row["delta_nll_static_minus_dynamic"] for row in subject_folds)
                / total_trials
            ),
            "improved": int(
                sum(row["delta_nll_static_minus_dynamic"] for row in subject_folds) > 0.0
            ),
        }
        for family in ("static", "dynamic"):
            subject_row[f"{family}_nll_per_trial"] = float(
                sum(row[f"{family}_nll"] for row in subject_folds) / total_trials
            )
            subject_row[f"{family}_choice_brier"] = float(
                sum(
                    row[f"{family}_choice_brier"] * row["n_trials"]
                    for row in subject_folds
                )
                / total_trials
            )
            subject_row[f"{family}_accuracy_bias"] = float(
                sum(row[f"{family}_accuracy_bias"] * row["n_trials"] for row in subject_folds)
                / total_trials
            )
            correlations = np.asarray(
                [
                    row[f"{family}_accuracy_curve_correlation"]
                    for row in subject_folds
                ],
                dtype=float,
            )
            subject_row[f"{family}_curve_correlation"] = finite_mean(correlations)
            subject_row[f"{family}_curve_mae"] = finite_mean(
                [row[f"{family}_accuracy_curve_mae"] for row in subject_folds]
            )
        late = [row for row in subject_folds if row["phase"] == "late"]
        for family in ("static", "dynamic"):
            subject_row[f"{family}_late_accuracy_bias"] = (
                float(np.mean([row[f"{family}_accuracy_bias"] for row in late]))
                if late else float("nan")
            )
        subject_rows.append(subject_row)

    atomic_csv(output / "rolling_folds.csv", fold_rows)
    atomic_csv(output / "rolling_subjects.csv", subject_rows)
    deltas = np.asarray([row["delta_nll_per_trial"] for row in subject_rows])
    rng = np.random.default_rng(int(config["design"]["bootstrap_seed"]))
    bootstrap = np.mean(
        rng.choice(
            deltas,
            size=(int(config["design"]["bootstrap_replicates"]), deltas.size),
            replace=True,
        ),
        axis=1,
    )
    phase_summary: dict[str, Any] = {}
    for phase in ("early", "middle", "late"):
        selected = [row for row in fold_rows if row["phase"] == phase]
        if not selected:
            continue
        phase_subject_rows: list[dict[str, float]] = []
        for subject_id in sorted({row["subject_id"] for row in selected}):
            subject_selected = [
                row for row in selected if row["subject_id"] == subject_id
            ]
            phase_trials = sum(row["n_trials"] for row in subject_selected)
            phase_subject_rows.append({
                "delta": sum(
                    row["delta_nll_static_minus_dynamic"]
                    for row in subject_selected
                )
                / phase_trials,
                "static_bias": sum(
                    row["static_accuracy_bias"] * row["n_trials"]
                    for row in subject_selected
                )
                / phase_trials,
                "dynamic_bias": sum(
                    row["dynamic_accuracy_bias"] * row["n_trials"]
                    for row in subject_selected
                )
                / phase_trials,
                "static_curve": finite_mean([
                    row["static_accuracy_curve_correlation"]
                    for row in subject_selected
                ]),
                "dynamic_curve": finite_mean([
                    row["dynamic_accuracy_curve_correlation"]
                    for row in subject_selected
                ]),
                "static_m": sum(
                    row["static_mean_predictive_m"] * row["n_trials"]
                    for row in subject_selected
                )
                / phase_trials,
                "dynamic_m": sum(
                    row["dynamic_mean_predictive_m"] * row["n_trials"]
                    for row in subject_selected
                )
                / phase_trials,
                "dynamic_family_posterior": float(np.mean([
                    row["prefix_dynamic_family_posterior"]
                    for row in subject_selected
                ])),
            })
        phase_summary[phase] = {
            "folds": len(selected),
            "subjects": len(phase_subject_rows),
            "subject_equal_mean_delta_nll_per_trial": float(
                np.mean([row["delta"] for row in phase_subject_rows])
            ),
            "fold_equal_mean_delta_nll_per_trial": float(
                np.mean([row["delta_nll_per_trial"] for row in selected])
            ),
            "static_accuracy_bias_subject_equal_mean": float(
                np.mean([row["static_bias"] for row in phase_subject_rows])
            ),
            "dynamic_accuracy_bias_subject_equal_mean": float(
                np.mean([row["dynamic_bias"] for row in phase_subject_rows])
            ),
            "static_curve_correlation_subject_equal_mean": float(
                np.nanmean([row["static_curve"] for row in phase_subject_rows])
            ),
            "dynamic_curve_correlation_subject_equal_mean": float(
                np.nanmean([row["dynamic_curve"] for row in phase_subject_rows])
            ),
            "static_predictive_m_subject_equal_mean": float(
                np.mean([row["static_m"] for row in phase_subject_rows])
            ),
            "dynamic_predictive_m_subject_equal_mean": float(
                np.mean([row["dynamic_m"] for row in phase_subject_rows])
            ),
            "dynamic_family_posterior_subject_equal_mean": float(
                np.mean([
                    row["dynamic_family_posterior"] for row in phase_subject_rows
                ])
            ),
        }
    short_subjects = [row for row in subject_rows if row["folds"] <= 2]
    long_subjects = [row for row in subject_rows if row["folds"] >= 3]
    summary = {
        "analysis_id": config["analysis_id"],
        "dynamic_signal": dynamic_signal(config),
        "development_only": True,
        "candidate_counts": {
            family: int(sum(value["family"] == family for value in candidates))
            for family in family_indices
        },
        "filter_seeds": filter_seeds,
        "mixture_component_counts": {
            family: int(family_indices[family].size) for family in family_indices
        },
        "component_integrity": integrity,
        "metric_implementation": {
            "choice_brier": "src.Bayesian_state.evaluation.evaluator.ModelEvaluator",
            "accuracy_curve": "src.Bayesian_state.evaluation.evaluator.ModelEvaluator",
        },
        "model_0805_reproduction": reproduce_0805(config),
        "rolling": {
            "subjects": len(subject_rows),
            "folds": len(fold_rows),
            "mean_delta_nll_per_trial": float(np.mean(deltas)),
            "pooled_trial_delta_nll_per_trial": float(
                sum(row["delta_nll_static_minus_dynamic"] for row in fold_rows)
                / sum(row["n_trials"] for row in fold_rows)
            ),
            "bootstrap_mean_95_interval": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "median_delta_nll_per_trial": float(np.median(deltas)),
            "improved_subjects": int(np.sum(deltas > 0.0)),
            "correlation_folds_with_delta": float(
                np.corrcoef(
                    [row["folds"] for row in subject_rows], deltas
                )[0, 1]
            ),
            "short_duration_subjects": len(short_subjects),
            "short_duration_mean_delta_nll_per_trial": float(
                np.mean([row["delta_nll_per_trial"] for row in short_subjects])
            ),
            "long_duration_subjects": len(long_subjects),
            "long_duration_mean_delta_nll_per_trial": float(
                np.mean([row["delta_nll_per_trial"] for row in long_subjects])
            ),
            "static_mean_nll_per_trial": float(
                np.mean([row["static_nll_per_trial"] for row in subject_rows])
            ),
            "dynamic_mean_nll_per_trial": float(
                np.mean([row["dynamic_nll_per_trial"] for row in subject_rows])
            ),
            "static_mean_choice_brier": float(
                np.mean([row["static_choice_brier"] for row in subject_rows])
            ),
            "dynamic_mean_choice_brier": float(
                np.mean([row["dynamic_choice_brier"] for row in subject_rows])
            ),
            "static_accuracy_bias_mean": float(
                np.mean([row["static_accuracy_bias"] for row in subject_rows])
            ),
            "dynamic_accuracy_bias_mean": float(
                np.mean([row["dynamic_accuracy_bias"] for row in subject_rows])
            ),
            "static_curve_correlation_mean": float(
                np.nanmean([row["static_curve_correlation"] for row in subject_rows])
            ),
            "dynamic_curve_correlation_mean": float(
                np.nanmean([row["dynamic_curve_correlation"] for row in subject_rows])
            ),
            "static_curve_mae_mean": float(
                np.nanmean([row["static_curve_mae"] for row in subject_rows])
            ),
            "dynamic_curve_mae_mean": float(
                np.nanmean([row["dynamic_curve_mae"] for row in subject_rows])
            ),
        },
        "by_phase": phase_summary,
    }
    atomic_json(output / "rolling_summary.json", summary)
    write_report(output / "rolling_report.md", summary)
    return summary


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    rolling = summary["rolling"]
    interval = rolling["bootstrap_mean_95_interval"]
    signal = str(summary.get("dynamic_signal", "surprise"))
    model_label = "FA3-M-S" if signal == "surprise" else "FA3-M-U"
    lines = [
        "# 0806 condition 1 滚动开发分析",
        "",
        "本分析只作开发证据。0805 的旧 outer holdout 已参与机制设计，不再称为独立确认集。",
        "",
        "## 0805 基线复核",
        "",
        f"- FS-H0 外部留出 NLL/试次："
        f"{summary['model_0805_reproduction']['FS_H0']['recomputed_mean_nll_per_trial']:.4f}。",
        f"- FA2-M3 外部留出 NLL/试次："
        f"{summary['model_0805_reproduction']['FA2_M3']['recomputed_mean_nll_per_trial']:.4f}。",
        "- 两项均由被试级结果重新汇总，并与 0805 报告精确一致。",
        "",
        "## 静态与动态模型",
        "",
        f"- 可评价被试 {rolling['subjects']} 名，共 {rolling['folds']} 个自然块后缀。",
        f"- 静态 FA2 NLL/试次={rolling['static_mean_nll_per_trial']:.4f}；"
        f"动态 {model_label}={rolling['dynamic_mean_nll_per_trial']:.4f}。",
        f"- 静态减动态的被试等权平均差={rolling['mean_delta_nll_per_trial']:.4f}，"
        f"bootstrap 95% 区间 [{interval[0]:.4f}, {interval[1]:.4f}]；"
        f"{rolling['improved_subjects']}/{rolling['subjects']} 名被试改善。",
        f"- 把所有预测试次合并后的差为 "
        f"{rolling['pooled_trial_delta_nll_per_trial']:.4f}；"
        f"改善与可评价块数的相关为 {rolling['correlation_folds_with_delta']:.3f}。",
        f"- 只有 1--2 个预测块的 {rolling['short_duration_subjects']} 人平均改善 "
        f"{rolling['short_duration_mean_delta_nll_per_trial']:.4f}；"
        f"至少 3 个预测块的 {rolling['long_duration_subjects']} 人平均改善 "
        f"{rolling['long_duration_mean_delta_nll_per_trial']:.4f}。",
        f"- Choice Brier（沿用仓库公共定义）：静态="
        f"{rolling['static_mean_choice_brier']:.4f}，动态="
        f"{rolling['dynamic_mean_choice_brier']:.4f}。",
        f"- 正确率偏差（预测正确率减实际正确率）：静态="
        f"{rolling['static_accuracy_bias_mean']:.4f}，动态="
        f"{rolling['dynamic_accuracy_bias_mean']:.4f}。",
        f"- 16-trial 局部曲线相关：静态="
        f"{rolling['static_curve_correlation_mean']:.3f}，动态="
        f"{rolling['dynamic_curve_correlation_mean']:.3f}。",
        f"- 16-trial accuracy-curve MAE：静态="
        f"{rolling['static_curve_mae_mean']:.4f}，动态="
        f"{rolling['dynamic_curve_mae_mean']:.4f}。",
        "",
        "## 分阶段",
        "",
    ]
    for phase, values in summary["by_phase"].items():
        lines.append(
            f"- {phase}: 被试等权 ΔNLL/试次="
            f"{values['subject_equal_mean_delta_nll_per_trial']:.4f}；"
            f"正确率偏差 static="
            f"{values['static_accuracy_bias_subject_equal_mean']:.4f}, "
            f"dynamic={values['dynamic_accuracy_bias_subject_equal_mean']:.4f}；"
            f"曲线相关 static="
            f"{values['static_curve_correlation_subject_equal_mean']:.3f}, "
            f"dynamic={values['dynamic_curve_correlation_subject_equal_mean']:.3f}；"
            f"预测替换率 static="
            f"{values['static_predictive_m_subject_equal_mean']:.3f}, "
            f"dynamic={values['dynamic_predictive_m_subject_equal_mean']:.3f}；"
            f"动态家族前缀后验="
            f"{values['dynamic_family_posterior_subject_equal_mean']:.3f}。"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    selected = parse_subjects(args.subjects)
    if args.smoke and selected is None:
        selected = {101}
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "analysis_config_snapshot.json", config)
    subject_audits, geometry_path = prepare(config, args.output, selected)
    candidates = candidate_grid(config)
    filter_seeds = [int(value) for value in config["design"]["filter_seeds"]]
    particle_count = int(config["design"]["particle_count"])
    if args.smoke:
        particle_count = min(particle_count, 128)
        candidates = [
            value for value in candidates if value["family"] == "static"
        ][:2] + [
            next(value for value in candidates if value["family"] == "dynamic")
        ]
    tasks: list[dict[str, Any]] = []
    primary_seed = filter_seeds[0]
    for audit in subject_audits:
        for candidate in candidates:
            for filter_seed in filter_seeds:
                tasks.append({
                    "config_path": str(args.config.resolve()),
                    "geometry_path": str(geometry_path),
                    "q_path": audit["q_path"],
                    "prediction_path": audit["prediction_path"],
                    "subject_id": int(audit["subject_id"]),
                    "candidate": candidate,
                    "particle_count": particle_count,
                    "filter_seed": int(filter_seed),
                    "output_path": str(component_path(
                        args.output,
                        int(audit["subject_id"]),
                        str(candidate["candidate_id"]),
                        int(filter_seed),
                        primary_seed,
                    )),
                    "force": bool(args.force),
                })
    if args.phase in ("all", "fit"):
        completed = 0
        with ProcessPoolExecutor(max_workers=max(1, int(args.jobs))) as executor:
            futures = [executor.submit(fit_task, task) for task in tasks]
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 250 == 0 or completed == len(tasks):
                    print(f"completed {completed}/{len(tasks)}", flush=True)
    if args.phase in ("all", "report"):
        if args.smoke:
            print("smoke run completed; full-grid reporting is intentionally skipped")
        else:
            summary = summarize(config, args.output, subject_audits)
            print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
