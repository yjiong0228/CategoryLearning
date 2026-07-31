#!/usr/bin/env python3
"""Closed-loop parameter and model-family recovery for the new-plan model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Bayesian_state.run_simulation import apply_fixed_hyperparams_to_engine_config
from src.Bayesian_state.utils.datasets import resolve_dataset_paths
from src.Bayesian_state.utils.newplan_generation import generate_condition1_trajectory
from src.Bayesian_state.utils.optimization_config import DEFAULT_DATA_PATH, load_yaml
from src.Bayesian_state.utils.optimizer_common import (
    TrialArrays,
    derive_trajectory_seed,
    evaluate_state_model_run,
    sequential_importance_marginal,
    stable_seed,
)
from src.Bayesian_state.utils.optimizer_simulation import StateModelSimulationRunner


SCENARIOS = {
    "B0": {"theta": 0.0, "epsilon": 0.0},
    "B1": {"theta": 0.0, "epsilon": 0.10},
    "D0": {"theta": 0.75, "epsilon": 0.0},
    "D1": {"theta": 0.75, "epsilon": 0.10},
}
MODEL_IDS = ("B0", "B1", "D0", "D1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=[103, 117, 131])
    parser.add_argument("--scenarios", choices=tuple(SCENARIOS), nargs="+", default=list(SCENARIOS))
    parser.add_argument("--datasets-per-scenario", type=int, default=3)
    parser.add_argument("--theta-grid", type=float, nargs="+", default=[0.0, 0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--epsilon-grid", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.10, 0.20, 0.40])
    parser.add_argument("--gamma", type=float, default=0.55)
    parser.add_argument("--w0", type=float, default=0.10)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--fit-repeats", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--max-trials", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=20260729)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/zhuran/cond1_newplan/recovery",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )


def validate_grid(values: Sequence[float], name: str) -> list[float]:
    grid = sorted({float(value) for value in values})
    if not grid or 0.0 not in grid:
        raise ValueError(f"{name} must be non-empty and contain 0.")
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in grid):
        raise ValueError(f"{name} values must lie in [0,1].")
    return grid


def params_for(args: argparse.Namespace, theta: float) -> dict[str, Any]:
    return {
        "engine.modules.hypo_transitions_mod.kwargs.theta": float(theta),
        "engine.modules.memory_mod.kwargs.gamma": float(args.gamma),
        "engine.modules.memory_mod.kwargs.w0": float(args.w0),
        "engine.choice_readout.kwargs": {
            "method": "sharpened_expectation",
            "power": float(args.rho),
        },
        "engine.output_noise.kwargs.base_lapse": 0.0,
    }


def one_fit_repeat(
    *,
    subject_id: int,
    arrays: TrialArrays,
    params: Mapping[str, Any],
    engine_config: Mapping[str, Any],
    processed_data_dir: Path,
    dataset_paths: Mapping[str, Path],
    window_size: int,
    point_seed: int,
    repeat_index: int,
) -> Any:
    trajectory_seed = derive_trajectory_seed(point_seed, "newplan_recovery_fit", repeat_index)
    return evaluate_state_model_run(
        subject_id=subject_id,
        condition=1,
        arrays=arrays,
        params=dict(params),
        engine_config_template=deepcopy(dict(engine_config)),
        processed_data_dir=processed_data_dir,
        window_size=window_size,
        dataset_paths=dataset_paths,
        keep_logs=False,
        include_step_log=False,
        prediction_mode="prior_t",
        selection_prediction_mode="prior_t",
        loss_metric="choice_brier",
        simulation_point_seed=point_seed,
        trajectory_seed=trajectory_seed,
    )


def cognitive_probability_stack(
    *,
    args: argparse.Namespace,
    subject_id: int,
    arrays: TrialArrays,
    theta: float,
    base_engine: Mapping[str, Any],
    processed_data_dir: Path,
    dataset_paths: Mapping[str, Path],
    paired_point_seed: int,
) -> np.ndarray:
    params = params_for(args, theta)
    engine_config = apply_fixed_hyperparams_to_engine_config(base_engine, params)
    runs = Parallel(n_jobs=max(1, args.n_jobs))(
        delayed(one_fit_repeat)(
            subject_id=subject_id,
            arrays=arrays,
            params=params,
            engine_config=engine_config,
            processed_data_dir=processed_data_dir,
            dataset_paths=dataset_paths,
            window_size=args.window_size,
            point_seed=paired_point_seed,
            repeat_index=repeat_index,
        )
        for repeat_index in range(args.fit_repeats)
    )
    stack = np.stack(
        [
            np.asarray(run.metrics_by_mode["prior_t"]["pred_category_probs"], dtype=float)
            for run in runs
        ],
        axis=0,
    )
    return stack


def score(
    probabilities: np.ndarray,
    choices: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    probs = np.asarray(probabilities, dtype=float)
    observed = np.asarray(choices, dtype=int) - 1
    keep = np.asarray(mask, dtype=bool).copy()
    keep &= (
        (observed >= 0)
        & (observed < probs.shape[1])
        & np.all(np.isfinite(probs), axis=1)
    )
    if not np.any(keep):
        return {"n": 0, "brier": float("nan"), "nll": float("nan")}
    selected = observed[keep]
    selected_probs = probs[keep]
    one_hot = np.zeros_like(selected_probs)
    one_hot[np.arange(selected_probs.shape[0]), selected] = 1.0
    return {
        "n": int(np.sum(keep)),
        "brier": float(np.mean(np.sum(np.square(selected_probs - one_hot), axis=1))),
        "nll": float(
            np.mean(
                -np.log(
                    np.clip(
                        selected_probs[np.arange(selected_probs.shape[0]), selected],
                        1e-12,
                        1.0,
                    )
                )
            )
        ),
    }


def time_split_masks(subject_frame) -> tuple[np.ndarray, np.ndarray]:
    n_trials = len(subject_frame)
    valid = np.ones(n_trials, dtype=bool)
    valid[0] = False
    if "iSession" in subject_frame.columns and "iBlock" in subject_frame.columns:
        pairs = list(
            dict.fromkeys(
                zip(
                    subject_frame["iSession"].astype(int),
                    subject_frame["iBlock"].astype(int),
                )
            )
        )
        if len(pairs) >= 2:
            last_session, last_block = pairs[-1]
            test = (
                (subject_frame["iSession"].to_numpy(dtype=int) == last_session)
                & (subject_frame["iBlock"].to_numpy(dtype=int) == last_block)
            )
            test &= valid
            return valid & (~test), test
    split = max(2, int(np.floor(0.75 * n_trials)))
    train = valid.copy()
    train[split:] = False
    test = valid.copy()
    test[:split] = False
    return train, test


def candidate_rows(
    *,
    args: argparse.Namespace,
    theta_probability_stacks: Mapping[float, np.ndarray],
    choices: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    valid_mask = np.asarray(train_mask, dtype=bool) | np.asarray(test_mask, dtype=bool)
    observed_zero_index = np.asarray(choices, dtype=int) - 1
    for theta, cognitive_stack in sorted(theta_probability_stacks.items()):
        for epsilon in args.epsilon_grid:
            observed_stack = (1.0 - epsilon) * cognitive_stack + epsilon / 2.0
            observed, ess = sequential_importance_marginal(
                observed_stack,
                observed_zero_index,
                valid_mask,
            )
            train_score = score(observed, choices, train_mask)
            test_score = score(observed, choices, test_mask)
            rows.append(
                {
                    "theta": float(theta),
                    "epsilon": float(epsilon),
                    "train_n": train_score["n"],
                    "train_brier": train_score["brier"],
                    "train_nll": train_score["nll"],
                    "test_n": test_score["n"],
                    "test_brier": test_score["brier"],
                    "test_nll": test_score["nll"],
                    "mean_ess": float(np.nanmean(ess)),
                    "final_ess": float(ess[np.flatnonzero(np.isfinite(ess))[-1]]),
                }
            )
    return rows


def select_family(rows: Sequence[Mapping[str, Any]], model_id: str) -> dict[str, Any]:
    allowed = []
    for row in rows:
        theta = float(row["theta"])
        epsilon = float(row["epsilon"])
        include = {
            "B0": theta == 0.0 and epsilon == 0.0,
            "B1": theta == 0.0,
            "D0": epsilon == 0.0,
            "D1": True,
        }[model_id]
        if include:
            allowed.append(dict(row))
    selected = min(
        allowed,
        key=lambda item: (
            float(item["train_brier"]),
            int(float(item["theta"]) > 0.0) + int(float(item["epsilon"]) > 0.0),
            float(item["theta"]),
            float(item["epsilon"]),
        ),
    )
    selected["model_id"] = model_id
    return selected


def select_winner(families: Sequence[Mapping[str, Any]]) -> str:
    complexity = {"B0": 0, "B1": 1, "D0": 1, "D1": 2}
    return str(
        min(
            families,
            key=lambda item: (
                float(item["test_brier"]),
                complexity[str(item["model_id"])],
                str(item["model_id"]),
            ),
        )["model_id"]
    )


def recover_dataset(
    *,
    args: argparse.Namespace,
    runner: StateModelSimulationRunner,
    base_engine: Mapping[str, Any],
    dataset_paths: Mapping[str, Path],
    subject_id: int,
    scenario_id: str,
    replicate: int,
) -> dict[str, Any]:
    subject_frame = runner._get_subject_frame(subject_id, 1.0).iloc[: args.max_trials].copy()
    arrays = runner._extract_arrays(subject_frame, None)
    if arrays.categories is None:
        raise ValueError("Closed-loop recovery requires hard condition-1 categories.")
    true = SCENARIOS[scenario_id]
    true_params = params_for(args, float(true["theta"]))
    true_engine = apply_fixed_hyperparams_to_engine_config(base_engine, true_params)
    generation_seed = stable_seed(
        {
            "seed_role": "newplan_recovery_generation",
            "base_seed": args.base_seed,
            "subject_id": subject_id,
            "scenario_id": scenario_id,
            "replicate": replicate,
        }
    )
    generated = generate_condition1_trajectory(
        engine_config=true_engine,
        subject_id=subject_id,
        stimulus=arrays.stimulus,
        categories=arrays.categories,
        epsilon=float(true["epsilon"]),
        rho=args.rho,
        trajectory_seed=generation_seed,
        processed_data_dir=runner._processed_data_dir,
        dataset_paths=dataset_paths,
    )
    synthetic_arrays = TrialArrays(
        stimulus=arrays.stimulus,
        choices=generated.choices,
        feedback=generated.feedback,
        categories=arrays.categories,
        target_probs=None,
    )
    paired_point_seed = stable_seed(
        {
            "seed_role": "newplan_recovery_paired_fit",
            "base_seed": args.base_seed,
            "subject_id": subject_id,
            "scenario_id": scenario_id,
            "replicate": replicate,
        }
    )
    theta_probability_stacks = {
        float(theta): cognitive_probability_stack(
            args=args,
            subject_id=subject_id,
            arrays=synthetic_arrays,
            theta=float(theta),
            base_engine=base_engine,
            processed_data_dir=runner._processed_data_dir,
            dataset_paths=dataset_paths,
            paired_point_seed=paired_point_seed,
        )
        for theta in args.theta_grid
    }
    train_mask, test_mask = time_split_masks(subject_frame)
    grid_rows = candidate_rows(
        args=args,
        theta_probability_stacks=theta_probability_stacks,
        choices=generated.choices,
        train_mask=train_mask,
        test_mask=test_mask,
    )
    families = [select_family(grid_rows, model_id) for model_id in MODEL_IDS]
    d1 = next(item for item in families if item["model_id"] == "D1")
    prefix_repeats = int(args.fit_repeats) // 2
    prefix_families = None
    prefix_d1 = None
    prefix_selected_model = None
    if prefix_repeats >= 2:
        prefix_grid_rows = candidate_rows(
            args=args,
            theta_probability_stacks={
                theta: np.asarray(stack, dtype=float)[:prefix_repeats]
                for theta, stack in theta_probability_stacks.items()
            },
            choices=generated.choices,
            train_mask=train_mask,
            test_mask=test_mask,
        )
        prefix_families = [
            select_family(prefix_grid_rows, model_id) for model_id in MODEL_IDS
        ]
        prefix_d1 = next(
            item for item in prefix_families if item["model_id"] == "D1"
        )
        prefix_selected_model = select_winner(prefix_families)
    selected_model = select_winner(families)
    selected_family = next(
        item for item in families if item["model_id"] == selected_model
    )
    return {
        "subject_id": int(subject_id),
        "true_model": scenario_id,
        "true_theta": float(true["theta"]),
        "true_epsilon": float(true["epsilon"]),
        "replicate": int(replicate),
        "generation_seed": int(generation_seed),
        "fit_point_seed": int(paired_point_seed),
        "generated_accuracy": float(np.mean(generated.feedback)),
        "generated_swap_count": int(
            sum(bool(item["swap_event"]) for item in generated.transition_log)
        ),
        "estimated_d1_theta": float(d1["theta"]),
        "estimated_d1_epsilon": float(d1["epsilon"]),
        "selected_model": selected_model,
        "selected_family_mean_ess": float(selected_family["mean_ess"]),
        "selected_family_final_ess": float(selected_family["final_ess"]),
        "prefix_repeats": prefix_repeats if prefix_families is not None else None,
        "prefix_selected_model": prefix_selected_model,
        "prefix_estimated_d1_theta": (
            float(prefix_d1["theta"]) if prefix_d1 is not None else None
        ),
        "prefix_estimated_d1_epsilon": (
            float(prefix_d1["epsilon"]) if prefix_d1 is not None else None
        ),
        "selected_model_agrees_prefix": (
            bool(prefix_selected_model == selected_model)
            if prefix_selected_model is not None
            else None
        ),
        "families": families,
        "grid_rows": grid_rows,
        "generated_choices": generated.choices,
        "generated_feedback": generated.feedback,
    }


def flatten_result(result: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in result.items()
        if key not in {"families", "grid_rows", "generated_choices", "generated_feedback"}
    }
    for family in result["families"]:
        model_id = str(family["model_id"])
        for key in ("theta", "epsilon", "train_brier", "test_brier", "test_nll"):
            row[f"{model_id}_{key}"] = family[key]
    return row


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    confusion = {
        truth: {
            estimate: int(
                sum(
                    row["true_model"] == truth and row["selected_model"] == estimate
                    for row in rows
                )
            )
            for estimate in MODEL_IDS
        }
        for truth in MODEL_IDS
    }
    theta_error = np.asarray(
        [float(row["estimated_d1_theta"]) - float(row["true_theta"]) for row in rows],
        dtype=float,
    )
    epsilon_error = np.asarray(
        [float(row["estimated_d1_epsilon"]) - float(row["true_epsilon"]) for row in rows],
        dtype=float,
    )
    true_swap = np.asarray(
        [str(row["true_model"]) in {"D0", "D1"} for row in rows], dtype=bool
    )
    selected_swap = np.asarray(
        [str(row["selected_model"]) in {"D0", "D1"} for row in rows], dtype=bool
    )
    true_lapse = np.asarray(
        [str(row["true_model"]) in {"B1", "D1"} for row in rows], dtype=bool
    )
    selected_lapse = np.asarray(
        [str(row["selected_model"]) in {"B1", "D1"} for row in rows], dtype=bool
    )

    def binary_recovery(
        truth: np.ndarray,
        estimate: np.ndarray,
    ) -> dict[str, int | float]:
        positive_n = int(np.sum(truth))
        negative_n = int(np.sum(~truth))
        true_positive = int(np.sum(truth & estimate))
        true_negative = int(np.sum((~truth) & (~estimate)))
        return {
            "accuracy_count": int(np.sum(truth == estimate)),
            "n": int(truth.size),
            "true_positive": true_positive,
            "positive_n": positive_n,
            "true_negative": true_negative,
            "negative_n": negative_n,
            "sensitivity": (
                float(true_positive / positive_n)
                if positive_n
                else float("nan")
            ),
            "specificity": (
                float(true_negative / negative_n)
                if negative_n
                else float("nan")
            ),
        }
    return {
        "n_datasets": len(rows),
        "correct_family_count": int(
            sum(row["true_model"] == row["selected_model"] for row in rows)
        ),
        "confusion": confusion,
        "factor_recovery": {
            "swap_present": binary_recovery(true_swap, selected_swap),
            "lapse_present": binary_recovery(true_lapse, selected_lapse),
        },
        "nested_monte_carlo": {
            "prefix_repeats": next(
                (
                    int(row["prefix_repeats"])
                    for row in rows
                    if row.get("prefix_repeats") is not None
                ),
                None,
            ),
            "selected_family_agreement_count": int(
                sum(row.get("selected_model_agrees_prefix") is True for row in rows)
            ),
            "selected_family_agreement_n": int(
                sum(row.get("selected_model_agrees_prefix") is not None for row in rows)
            ),
            "full_selected_final_ess_median": float(
                np.median(
                    [float(row["selected_family_final_ess"]) for row in rows]
                )
            ),
            "full_selected_final_ess_min": float(
                np.min(
                    [float(row["selected_family_final_ess"]) for row in rows]
                )
            ),
        },
        "d1_parameter_recovery": {
            "theta_bias": float(np.mean(theta_error)),
            "theta_rmse": float(np.sqrt(np.mean(np.square(theta_error)))),
            "epsilon_bias": float(np.mean(epsilon_error)),
            "epsilon_rmse": float(np.sqrt(np.mean(np.square(epsilon_error)))),
            "estimate_correlation": (
                float(
                    np.corrcoef(
                        [float(row["estimated_d1_theta"]) for row in rows],
                        [float(row["estimated_d1_epsilon"]) for row in rows],
                    )[0, 1]
                )
                if len(rows) >= 3
                else float("nan")
            ),
        },
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: Mapping[str, Any]) -> None:
    recovery = summary["d1_parameter_recovery"]
    convergence = summary["nested_monte_carlo"]
    factors = summary["factor_recovery"]
    lines = [
        "# Condition-1 new-plan recovery",
        "",
        f"- datasets: {summary['n_datasets']}",
        f"- correct family: {summary['correct_family_count']}/{summary['n_datasets']}",
        f"- theta bias: {recovery['theta_bias']:.4f}",
        f"- theta RMSE: {recovery['theta_rmse']:.4f}",
        f"- epsilon bias: {recovery['epsilon_bias']:.4f}",
        f"- epsilon RMSE: {recovery['epsilon_rmse']:.4f}",
        f"- theta/epsilon estimate correlation: {recovery['estimate_correlation']:.4f}",
        f"- nested {convergence['prefix_repeats']}→"
        f"{convergence['full_repeats']} family agreement: "
        f"{convergence['selected_family_agreement_count']}/"
        f"{convergence['selected_family_agreement_n']}",
        f"- selected-family final ESS: median="
        f"{convergence['full_selected_final_ess_median']:.2f}, "
        f"min={convergence['full_selected_final_ess_min']:.2f}",
        f"- swap-present recovery: "
        f"{factors['swap_present']['accuracy_count']}/"
        f"{factors['swap_present']['n']} "
        f"(sensitivity={factors['swap_present']['sensitivity']:.3f}, "
        f"specificity={factors['swap_present']['specificity']:.3f})",
        f"- lapse-present recovery: "
        f"{factors['lapse_present']['accuracy_count']}/"
        f"{factors['lapse_present']['n']} "
        f"(sensitivity={factors['lapse_present']['sensitivity']:.3f}, "
        f"specificity={factors['lapse_present']['specificity']:.3f})",
        "",
        "## Family confusion",
        "",
        "| true \\ selected | B0 | B1 | D0 | D1 |",
        "|:---:|---:|---:|---:|---:|",
    ]
    for truth in MODEL_IDS:
        values = summary["confusion"][truth]
        lines.append(
            f"| {truth} | {values['B0']} | {values['B1']} | {values['D0']} | {values['D1']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.theta_grid = validate_grid(args.theta_grid, "theta_grid")
    args.epsilon_grid = validate_grid(args.epsilon_grid, "epsilon_grid")
    if args.datasets_per_scenario <= 0 or args.fit_repeats <= 0 or args.n_jobs <= 0:
        raise ValueError("datasets_per_scenario, fit_repeats, and n_jobs must be positive.")
    if args.max_trials < 32:
        raise ValueError("max_trials must be at least 32 for recovery.")
    for scenario_id in args.scenarios:
        true = SCENARIOS[scenario_id]
        if float(true["theta"]) not in args.theta_grid:
            raise ValueError(f"theta_grid does not contain true theta for {scenario_id}.")
        if float(true["epsilon"]) not in args.epsilon_grid:
            raise ValueError(f"epsilon_grid does not contain true epsilon for {scenario_id}.")

    model_path = ROOT / "configs/model_struct/pmh_model_cond1_newplan.yaml"
    sim_path = ROOT / "configs/simulation_cfg/pmh_cond1_simulation_v14.yaml"
    base_engine = load_yaml(model_path)
    sim_cfg = load_yaml(sim_path)
    dataset_paths = resolve_dataset_paths(sim_cfg, sim_path.parent, DEFAULT_DATA_PATH)
    runner = StateModelSimulationRunner(
        engine_config=base_engine,
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
        n_jobs=args.n_jobs,
    )
    runner.prepare_data(dataset_paths["learning_data"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "manifest.json",
        {
            "result_type": "cond1_newplan_closed_loop_recovery",
            "subjects": args.subjects,
            "scenarios": {key: SCENARIOS[key] for key in args.scenarios},
            "datasets_per_scenario": args.datasets_per_scenario,
            "theta_grid": args.theta_grid,
            "epsilon_grid": args.epsilon_grid,
            "gamma": args.gamma,
            "w0": args.w0,
            "rho": args.rho,
            "fit_repeats": args.fit_repeats,
            "max_trials": args.max_trials,
            "base_seed": args.base_seed,
        },
    )

    detailed = []
    for subject_id in args.subjects:
        for scenario_id in args.scenarios:
            for replicate in range(args.datasets_per_scenario):
                result_path = (
                    args.output_dir
                    / "datasets"
                    / f"subject_{subject_id}"
                    / f"{scenario_id}_rep{replicate}.json"
                )
                if result_path.exists() and not args.force:
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    print(
                        f"LOAD subject={subject_id} true={scenario_id} rep={replicate}",
                        flush=True,
                    )
                else:
                    print(
                        f"RUN subject={subject_id} true={scenario_id} rep={replicate}",
                        flush=True,
                    )
                    result = recover_dataset(
                        args=args,
                        runner=runner,
                        base_engine=base_engine,
                        dataset_paths=dataset_paths,
                        subject_id=subject_id,
                        scenario_id=scenario_id,
                        replicate=replicate,
                    )
                    write_json(result_path, result)
                detailed.append(result)
                flat_rows = [flatten_result(item) for item in detailed]
                summary = summarize(flat_rows)
                summary["nested_monte_carlo"]["full_repeats"] = int(
                    args.fit_repeats
                )
                write_csv(args.output_dir / "recovery_rows.csv", flat_rows)
                write_json(args.output_dir / "aggregate_summary.json", summary)
                write_report(args.output_dir / "RESULTS.md", summary)

    print(f"COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
