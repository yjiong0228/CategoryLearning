#!/usr/bin/env python3
"""Recover static FA2 versus one-signal FA3-M on autonomous data."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
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
from src.Bayesian_state.reference_models.model_0803 import TransitionKernels  # noqa: E402
from src.Bayesian_state.reference_models.model_0804.core import (  # noqa: E402
    Model0804Parameters,
)
from src.Bayesian_state.reference_models.model_0804.core import (  # noqa: E402
    run_model0804_particle_filter,
)
from src.Bayesian_state.reference_models.model_0806 import (  # noqa: E402
    Model0804RTParameters,
    simulate_model0806_log_rt,
)
from src.Bayesian_state.reference_models.model_0806 import (  # noqa: E402
    simulate_model0806_choices,
)


DEFAULT_CONFIG = ROOT / "configs/model_0806_dynamic_m_recovery.yaml"
DEFAULT_OUTPUT = (
    ROOT / "results/zhuran/model_0806_cond1/dynamic_m_recovery_20260806_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument(
        "--phase", choices=("all", "simulate", "fit", "report"), default="all"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("recovery config must be a mapping")
    return payload


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        if fields:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def dynamic_signal(config: Mapping[str, Any]) -> str:
    signal = str(config.get("dynamic_signal", "surprise"))
    if signal not in {"surprise", "uncertainty"}:
        raise ValueError("dynamic_signal must be surprise or uncertainty")
    return signal


def choice_parameters(
    config: Mapping[str, Any], specification: Mapping[str, Any]
) -> Model0804Parameters:
    fixed = config["fixed_choice_parameters"]
    signal = dynamic_signal(config)
    standard = config[f"{signal}_standardization"]
    beta = float(specification[f"beta_{signal}"])
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
        gamma=float(fixed["gamma"]),
        w0=float(fixed["w0"]),
        kappa=float(fixed["kappa"]),
        m=float(specification["m"]),
        g=float(fixed["g"]),
        lapse=float(fixed["lapse"]),
        rho=float(fixed["rho"]),
        dynamic_m=beta > 0.0,
        m_phi=float(specification["phi"]),
        **signal_parameters,
    )


def rt_parameters(config: Mapping[str, Any]) -> Model0804RTParameters:
    return Model0804RTParameters(**{
        key: float(value) for key, value in config["rt_emission"].items()
    })


def candidate_grid(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    signal = dynamic_signal(config)
    beta_key = f"beta_{signal}"
    candidate_prefix = "FA3MS" if signal == "surprise" else "FA3MU"
    support = config["candidate_support"]
    rows: list[dict[str, Any]] = []
    for m_value in support["m"]:
        rows.append({
            "candidate_id": f"FA2_m{float(m_value):.2f}",
            "family": "static",
            "m": float(m_value),
            "phi": 0.0,
            beta_key: 0.0,
        })
    for m_value in support["m"]:
        for phi in support["phi"]:
            for beta in support[beta_key]:
                rows.append({
                    "candidate_id": (
                        f"{candidate_prefix}_m{float(m_value):.2f}_p{float(phi):.2f}"
                        f"_b{float(beta):.2f}"
                    ),
                    "family": "dynamic",
                    "m": float(m_value),
                    "phi": float(phi),
                    beta_key: float(beta),
                })
    return rows


def save_geometry(
    path: Path, prior: np.ndarray, kernels: TransitionKernels
) -> None:
    atomic_savez(
        path,
        prior=np.asarray(prior, dtype=float),
        local=np.asarray(kernels.local, dtype=float),
        global_kernel=np.asarray(kernels.global_, dtype=float),
        distance=np.asarray(kernels.distance, dtype=float),
        tau_local=np.asarray(kernels.tau_local),
        expected_local=np.asarray(kernels.expected_local_distance, dtype=float),
        expected_global=np.asarray(kernels.expected_global_distance, dtype=float),
    )


def load_geometry(path: Path) -> tuple[np.ndarray, TransitionKernels]:
    with np.load(path, allow_pickle=False) as payload:
        prior = payload["prior"].astype(float)
        kernels = TransitionKernels(
            local=payload["local"].astype(float),
            global_=payload["global_kernel"].astype(float),
            distance=payload["distance"].astype(float),
            tau_local=float(payload["tau_local"]),
            expected_local_distance=payload["expected_local"].astype(float),
            expected_global_distance=payload["expected_global"].astype(float),
        )
    return prior, kernels


def prepare_inputs(
    config: Mapping[str, Any], output: Path, *, smoke: bool, force: bool
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    base_path = ROOT / str(config["base_config"])
    base = load_config(base_path)
    requested = [int(value) for value in config["design"]["template_subjects"]]
    if smoke:
        requested = requested[:2]
    frame, subjects, data_audit = validate_and_load_inputs(base, set(requested))
    priors, kernels, geometry_audit = build_frozen_geometry(base)
    geometry_path = output / "geometry.npz"
    if force or not geometry_path.exists():
        save_geometry(geometry_path, priors[PRIMARY_PRIOR], kernels[PRIMARY_PRIOR])

    n_trials = int(config["design"]["trials_per_dataset"])
    if smoke:
        n_trials = min(n_trials, 96)
    datasets: list[dict[str, Any]] = []
    for subject_id in subjects:
        audit = validate_subject_cache(base, frame, subject_id)
        with np.load(Path(audit["q_path"]), allow_pickle=False) as payload:
            q = payload["q"].astype(float)[:n_trials]
        with np.load(Path(audit["prediction_path"]), allow_pickle=False) as payload:
            category = payload["category"].astype(int)[:n_trials]
        if q.shape[0] != n_trials:
            raise ValueError(f"subject {subject_id} has fewer than {n_trials} trials")
        for family_index, family in enumerate(("static", "dynamic")):
            generator = config["generators"][family]
            dataset_id = f"{family}_subject_{subject_id}"
            dataset_path = output / "synthetic" / f"{dataset_id}.npz"
            if force or not dataset_path.exists():
                simulation = simulate_model0806_choices(
                    q,
                    category,
                    priors[PRIMARY_PRIOR],
                    kernels[PRIMARY_PRIOR],
                    parameters=choice_parameters(config, generator),
                    capacity=int(config["design"]["capacity"]),
                    seed=2026080600 + 10 * int(subject_id) + family_index,
                )
                log_rt = simulate_model0806_log_rt(
                    simulation,
                    rt_parameters(config),
                    seed=2026081600 + 10 * int(subject_id) + family_index,
                )
                metadata = {
                    "dataset_id": dataset_id,
                    "true_family": family,
                    "template_subject": int(subject_id),
                    "generator": dict(generator),
                }
                signal = dynamic_signal(config)
                true_signal = (
                    simulation.feedback_surprise
                    if signal == "surprise"
                    else simulation.feedback_uncertainty
                )
                atomic_savez(
                    dataset_path,
                    q=q.astype(np.float32),
                    category=category.astype(np.int8),
                    choice=simulation.choices.astype(np.int8),
                    feedback=simulation.feedback.astype(np.float32),
                    log_rt=log_rt.astype(np.float32),
                    true_m=simulation.predictive_m.astype(np.float32),
                    true_signal=true_signal.astype(np.float32),
                    true_surprise=simulation.feedback_surprise.astype(np.float32),
                    true_uncertainty=simulation.feedback_uncertainty.astype(np.float32),
                    true_replacement=simulation.replacement_fraction.astype(np.float32),
                    metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                )
            datasets.append({
                "dataset_id": dataset_id,
                "path": str(dataset_path),
                "true_family": family,
                "template_subject": int(subject_id),
            })
    audit = {
        "data": data_audit,
        "geometry": geometry_audit,
        "base_config": str(base_path),
        "datasets": datasets,
        "smoke": bool(smoke),
    }
    atomic_json(output / "input_audit.json", audit)
    return datasets, geometry_path, audit


def component_path(
    output: Path, dataset_id: str, candidate_id: str, mode: str
) -> Path:
    return output / "components" / dataset_id / candidate_id / f"{mode}.npz"


def fit_task(task: Mapping[str, Any]) -> dict[str, Any]:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    path = Path(task["output_path"])
    if path.exists() and not bool(task["force"]):
        return {"path": str(path), "skipped": True}
    config = read_yaml(Path(task["config_path"]))
    prior, kernels = load_geometry(Path(task["geometry_path"]))
    with np.load(Path(task["dataset_path"]), allow_pickle=False) as payload:
        q = payload["q"].astype(float)
        choice = payload["choice"].astype(int)
        feedback = payload["feedback"].astype(float)
        log_rt = payload["log_rt"].astype(float)
    specification = task["candidate"]
    mode = str(task["mode"])
    kwargs: dict[str, Any] = {}
    if mode == "joint":
        kwargs = {
            "log_rt_values": log_rt,
            "rt_parameters": rt_parameters(config),
        }
    trace = run_model0804_particle_filter(
        q,
        choice,
        feedback,
        prior,
        kernels,
        model_id="FA2",
        parameters=choice_parameters(config, specification),
        capacity=int(config["design"]["capacity"]),
        particle_count=int(task["particle_count"]),
        filter_seed=int(config["design"]["filter_seed"]),
        **kwargs,
    )
    observed = trace.probabilities[np.arange(choice.size), choice]
    log_predictive = np.log(np.clip(observed, 1e-300, 1.0))
    if mode == "joint":
        log_predictive += np.asarray(trace.rt_predictive_log_density, dtype=float)
        total_nll = float(trace.joint_nll)
    else:
        total_nll = float(trace.nll)
    metadata = {
        "dataset_id": str(task["dataset_id"]),
        "candidate": specification,
        "mode": mode,
        "particle_count": int(task["particle_count"]),
        "total_nll": total_nll,
    }
    atomic_savez(
        path,
        probabilities=trace.probabilities.astype(np.float32),
        log_predictive=log_predictive.astype(np.float64),
        predictive_m=np.asarray(trace.predictive_m, dtype=np.float32),
        replacement_fraction=trace.predictive_replacement_fraction.astype(np.float32),
        feedback_surprise=np.asarray(trace.feedback_surprise, dtype=np.float32),
        feedback_uncertainty=np.asarray(trace.feedback_uncertainty, dtype=np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {"path": str(path), "skipped": False}


def softmax(log_values: np.ndarray) -> np.ndarray:
    values = np.asarray(log_values, dtype=float)
    maximum = float(np.max(values))
    weights = np.exp(values - maximum)
    return weights / float(np.sum(weights))


def log_mixture(log_predictive: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.log(np.clip(weights, 1e-300, 1.0))[:, None] + log_predictive
    maximum = np.max(values, axis=0)
    return maximum + np.log(np.sum(np.exp(values - maximum[None, :]), axis=0))


def load_component(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            "log_predictive": payload["log_predictive"].astype(float),
            "predictive_m": payload["predictive_m"].astype(float),
            "replacement_fraction": payload["replacement_fraction"].astype(float),
            "metadata": json.loads(str(payload["metadata_json"].item())),
        }


def summarize(
    config: Mapping[str, Any], output: Path, datasets: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    signal = dynamic_signal(config)
    candidates = candidate_grid(config)
    families = ("static", "dynamic")
    rows: list[dict[str, Any]] = []
    delta_limit = float(config["equivalence"]["maximum_delta_nll"])
    train_fraction = float(config["design"]["train_fraction"])
    for dataset in datasets:
        with np.load(Path(dataset["path"]), allow_pickle=False) as payload:
            true_m = payload["true_m"].astype(float)
            if "true_signal" in payload.files:
                true_signal = payload["true_signal"].astype(float)
            elif signal == "surprise":
                true_signal = payload["true_surprise"].astype(float)
            else:
                true_signal = payload["true_uncertainty"].astype(float)
        n_trials = true_m.size
        train_end = max(2, min(n_trials - 1, int(round(train_fraction * n_trials))))
        loaded: dict[str, list[dict[str, Any]]] = {}
        for mode in ("choice", "joint"):
            loaded[mode] = [
                load_component(
                    component_path(
                        output,
                        str(dataset["dataset_id"]),
                        str(candidate["candidate_id"]),
                        mode,
                    )
                )
                for candidate in candidates
            ]
        row: dict[str, Any] = {
            "dataset_id": dataset["dataset_id"],
            "true_family": dataset["true_family"],
            "template_subject": int(dataset["template_subject"]),
            "n_trials": int(n_trials),
            "n_train": int(train_end),
            "n_validation": int(n_trials - train_end),
        }
        for mode in ("choice", "joint"):
            log_matrix = np.stack(
                [component["log_predictive"] for component in loaded[mode]], axis=0
            )
            full_nll = -np.sum(log_matrix, axis=1)
            family_validation: dict[str, float] = {}
            family_evidence: dict[str, float] = {}
            for family in families:
                indices = np.asarray(
                    [i for i, value in enumerate(candidates) if value["family"] == family],
                    dtype=int,
                )
                train_log_likelihood = np.sum(log_matrix[indices, :train_end], axis=1)
                frozen_weights = softmax(train_log_likelihood)
                validation_log = log_mixture(
                    log_matrix[indices, train_end:], frozen_weights
                )
                family_validation[family] = float(-np.sum(validation_log))
                evidence_values = -full_nll[indices] - math.log(indices.size)
                maximum = float(np.max(evidence_values))
                family_evidence[family] = maximum + math.log(
                    float(np.sum(np.exp(evidence_values - maximum)))
                )
                row[f"{mode}_{family}_validation_nll"] = family_validation[family]
            row[f"{mode}_validation_delta_static_minus_dynamic"] = (
                family_validation["static"] - family_validation["dynamic"]
            )
            row[f"{mode}_recovered_family"] = min(
                family_validation, key=family_validation.get
            )
            row[f"{mode}_correct"] = int(
                row[f"{mode}_recovered_family"] == dataset["true_family"]
            )

            log_prior = np.asarray([
                math.log(0.5)
                - math.log(sum(value["family"] == candidate["family"] for value in candidates))
                for candidate in candidates
            ])
            posterior = softmax(-full_nll + log_prior)
            row[f"{mode}_dynamic_posterior"] = float(
                sum(
                    posterior[i]
                    for i, candidate in enumerate(candidates)
                    if candidate["family"] == "dynamic"
                )
            )
            row[f"{mode}_evidence_recovered_family"] = (
                "dynamic" if row[f"{mode}_dynamic_posterior"] >= 0.5 else "static"
            )
            row[f"{mode}_evidence_correct"] = int(
                row[f"{mode}_evidence_recovered_family"]
                == dataset["true_family"]
            )
            row[f"{mode}_effective_candidate_count"] = float(
                math.exp(-np.sum(posterior * np.log(np.clip(posterior, 1e-300, 1.0))))
            )
            row[f"{mode}_near_best_count"] = int(
                np.sum(full_nll <= float(np.min(full_nll)) + delta_limit)
            )
            predicted_m = np.sum(
                posterior[:, None]
                * np.stack(
                    [component["predictive_m"] for component in loaded[mode]], axis=0
                ),
                axis=0,
            )
            if np.std(true_m) > 1e-10 and np.std(predicted_m) > 1e-10:
                row[f"{mode}_m_trajectory_correlation"] = float(
                    np.corrcoef(true_m, predicted_m)[0, 1]
                )
            else:
                row[f"{mode}_m_trajectory_correlation"] = 1.0 if np.allclose(
                    true_m, predicted_m, atol=0.02
                ) else 0.0
            previous_signal = true_signal[:-1]
            next_m = predicted_m[1:]
            low, high = np.quantile(previous_signal, [0.25, 0.75])
            effect = float(
                np.mean(next_m[previous_signal >= high])
                - np.mean(next_m[previous_signal <= low])
            )
            row[f"{mode}_recovered_signal_effect"] = effect
            row[f"{mode}_positive_effect"] = int(effect > 0.0)
        rows.append(row)

    atomic_csv(output / "recovery_rows.csv", rows)
    truth_groups: dict[str, Any] = {}
    for truth in families:
        selected = [row for row in rows if row["true_family"] == truth]
        truth_groups[truth] = {
            "n": len(selected),
            "choice_evidence_accuracy": float(
                np.mean([row["choice_evidence_correct"] for row in selected])
            ),
            "joint_evidence_accuracy": float(
                np.mean([row["joint_evidence_correct"] for row in selected])
            ),
            "choice_frozen_validation_accuracy": float(
                np.mean([row["choice_correct"] for row in selected])
            ),
            "joint_frozen_validation_accuracy": float(
                np.mean([row["joint_correct"] for row in selected])
            ),
            "choice_dynamic_posterior_mean": float(
                np.mean([row["choice_dynamic_posterior"] for row in selected])
            ),
            "joint_dynamic_posterior_mean": float(
                np.mean([row["joint_dynamic_posterior"] for row in selected])
            ),
            "choice_effective_count_mean": float(
                np.mean([row["choice_effective_candidate_count"] for row in selected])
            ),
            "joint_effective_count_mean": float(
                np.mean([row["joint_effective_candidate_count"] for row in selected])
            ),
            "choice_near_best_count_mean": float(
                np.mean([row["choice_near_best_count"] for row in selected])
            ),
            "joint_near_best_count_mean": float(
                np.mean([row["joint_near_best_count"] for row in selected])
            ),
            "choice_signal_direction_rate": float(
                np.mean([row["choice_positive_effect"] for row in selected])
            ),
            "joint_signal_direction_rate": float(
                np.mean([row["joint_positive_effect"] for row in selected])
            ),
            "choice_m_correlation_mean": float(
                np.mean([row["choice_m_trajectory_correlation"] for row in selected])
            ),
            "joint_m_correlation_mean": float(
                np.mean([row["joint_m_trajectory_correlation"] for row in selected])
            ),
        }
    overall = {
        "n": len(rows),
        "choice_evidence_accuracy": float(
            np.mean([row["choice_evidence_correct"] for row in rows])
        ),
        "joint_evidence_accuracy": float(
            np.mean([row["joint_evidence_correct"] for row in rows])
        ),
        "choice_frozen_validation_accuracy": float(
            np.mean([row["choice_correct"] for row in rows])
        ),
        "joint_frozen_validation_accuracy": float(
            np.mean([row["joint_correct"] for row in rows])
        ),
        "choice_effective_count_mean": float(
            np.mean([row["choice_effective_candidate_count"] for row in rows])
        ),
        "joint_effective_count_mean": float(
            np.mean([row["joint_effective_candidate_count"] for row in rows])
        ),
        "choice_near_best_count_mean": float(
            np.mean([row["choice_near_best_count"] for row in rows])
        ),
        "joint_near_best_count_mean": float(
            np.mean([row["joint_near_best_count"] for row in rows])
        ),
    }
    overall["effective_count_reduction_fraction"] = float(
        1.0
        - overall["joint_effective_count_mean"]
        / overall["choice_effective_count_mean"]
    )
    overall["near_best_count_reduction_fraction"] = float(
        1.0
        - overall["joint_near_best_count_mean"]
        / overall["choice_near_best_count_mean"]
    )
    summary = {
        "analysis_id": config["analysis_id"],
        "dynamic_signal": signal,
        "candidate_count": len(candidates),
        "static_candidate_count": sum(row["family"] == "static" for row in candidates),
        "dynamic_candidate_count": sum(row["family"] == "dynamic" for row in candidates),
        "rt_interpretation": (
            "idealized upper-bound recovery with the generating RT emission fixed; "
            "real-data nuisance estimation can only be harder"
        ),
        "overall": overall,
        "by_true_family": truth_groups,
    }
    atomic_json(output / "recovery_summary.json", summary)
    write_report(output / "recovery_report.md", summary)
    return summary


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    overall = summary["overall"]
    static = summary["by_true_family"]["static"]
    dynamic = summary["by_true_family"]["dynamic"]
    signal = str(summary["dynamic_signal"])
    model_label = "FA3-M-S" if signal == "surprise" else "FA3-M-U"
    signal_label = "surprise" if signal == "surprise" else "规则不确定性"
    lines = [
        "# 0806 动态替换率恢复实验",
        "",
        f"共比较 {summary['candidate_count']} 个候选："
        f"{summary['static_candidate_count']} 个静态 FA2 和 "
        f"{summary['dynamic_candidate_count']} 个 {model_label}。",
        "",
        "## 主要结果",
        "",
        f"- 完整序列家族证据的恢复率：choice="
        f"{overall['choice_evidence_accuracy']:.3f}，choice+RT="
        f"{overall['joint_evidence_accuracy']:.3f}。",
        f"- 静态生成数据的家族证据恢复率：choice="
        f"{static['choice_evidence_accuracy']:.3f}，choice+RT="
        f"{static['joint_evidence_accuracy']:.3f}。",
        f"- 动态生成数据的家族证据恢复率：choice="
        f"{dynamic['choice_evidence_accuracy']:.3f}，choice+RT="
        f"{dynamic['joint_evidence_accuracy']:.3f}。",
        f"- 单个冻结后缀的预测胜负正确率：choice="
        f"{overall['choice_frozen_validation_accuracy']:.3f}，choice+RT="
        f"{overall['joint_frozen_validation_accuracy']:.3f}；它与完整序列的模型恢复是不同问题。",
        f"- 有效候选数从 {overall['choice_effective_count_mean']:.2f} "
        f"降到 {overall['joint_effective_count_mean']:.2f}，"
        f"收缩 {100.0 * overall['effective_count_reduction_fraction']:.1f}%。",
        f"- 距最佳 NLL 不超过 2 的候选数从 "
        f"{overall['choice_near_best_count_mean']:.2f} 降到 "
        f"{overall['joint_near_best_count_mean']:.2f}，"
        f"收缩 {100.0 * overall['near_best_count_reduction_fraction']:.1f}%。",
        f"- 动态数据中{signal_label}→下一试次替换率的正方向恢复率："
        f"choice={dynamic['choice_signal_direction_rate']:.3f}，"
        f"choice+RT={dynamic['joint_signal_direction_rate']:.3f}。",
        "",
        "## 边界",
        "",
        "这里把 RT 生成参数固定为真值，检验的是理想条件下 RT 是否有可能缩小等价集合。"
        "真实数据中还要估计 RT 基线、尺度和协变量，所以实际识别只会更困难。",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    config = read_yaml(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / "analysis_config_snapshot.json", config)
    datasets, geometry_path, _ = prepare_inputs(
        config, args.output, smoke=args.smoke, force=args.force
    )
    if args.phase == "simulate":
        return
    candidates = candidate_grid(config)
    particle_count = int(config["design"]["particle_count"])
    if args.smoke:
        particle_count = min(particle_count, 128)
        candidates = candidates[:2] + [
            next(value for value in candidates if value["family"] == "dynamic")
        ]
    tasks: list[dict[str, Any]] = []
    for dataset in datasets:
        for candidate in candidates:
            for mode in ("choice", "joint"):
                tasks.append({
                    "config_path": str(args.config.resolve()),
                    "geometry_path": str(geometry_path),
                    "dataset_path": str(dataset["path"]),
                    "dataset_id": dataset["dataset_id"],
                    "candidate": candidate,
                    "mode": mode,
                    "particle_count": particle_count,
                    "output_path": str(component_path(
                        args.output,
                        str(dataset["dataset_id"]),
                        str(candidate["candidate_id"]),
                        mode,
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
                if completed % 50 == 0 or completed == len(tasks):
                    print(f"completed {completed}/{len(tasks)}", flush=True)
    if args.phase in ("all", "report"):
        if args.smoke:
            print("smoke run completed; full-grid reporting is intentionally skipped")
        else:
            summary = summarize(config, args.output, datasets)
            print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
