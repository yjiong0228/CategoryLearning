#!/usr/bin/env python3
"""Paired theta-by-epsilon comparison for the condition-1 new-plan model.

The expensive latent-state simulation is run once per theta value.  Constant
lapse is then applied analytically to the marginal cognitive probabilities, so
B0/B1 and D0/D1 share exactly the same latent trajectories.  This script is
intended first for engineering smoke tests and then for the frozen development
comparison with common memory/readout parameters.
"""

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
from src.Bayesian_state.utils.optimization_config import (
    DEFAULT_DATA_PATH,
    load_yaml,
)
from src.Bayesian_state.utils.optimizer_common import (
    derive_trajectory_seed,
    evaluate_state_model_run,
    sequential_importance_marginal,
    stable_seed,
)
from src.Bayesian_state.utils.optimizer_simulation import StateModelSimulationRunner


DEFAULT_SUBJECTS = (103, 105, 111, 112, 117, 118, 127, 131)
DEFAULT_THETA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
DEFAULT_EPSILON_GRID = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)
MODEL_IDS = ("B0", "B1", "D0", "D1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--theta-grid", type=float, nargs="+", default=list(DEFAULT_THETA_GRID))
    parser.add_argument("--epsilon-grid", type=float, nargs="+", default=list(DEFAULT_EPSILON_GRID))
    parser.add_argument("--gamma", type=float, default=0.55)
    parser.add_argument("--w0", type=float, default=0.10)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--base-seed", type=int, default=20260728)
    parser.add_argument(
        "--marginal-method",
        choices=("importance", "equal"),
        default="importance",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/zhuran/cond1_newplan/factorial_dev",
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


def theta_token(theta: float) -> str:
    return f"{float(theta):.4f}".rstrip("0").rstrip(".").replace(".", "p")


def validate_probability_grid(values: Sequence[float], name: str) -> list[float]:
    out = sorted({float(value) for value in values})
    if not out or any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in out):
        raise ValueError(f"{name} must contain finite values in [0, 1].")
    if 0.0 not in out:
        raise ValueError(f"{name} must contain the exact zero boundary.")
    return out


def common_params(args: argparse.Namespace, theta: float) -> dict[str, Any]:
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


def score_probabilities(
    probabilities: np.ndarray,
    observed_choice: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | int]:
    probs = np.asarray(probabilities, dtype=float)
    choice = np.asarray(observed_choice, dtype=int).reshape(-1)
    keep = np.asarray(mask, dtype=bool).reshape(-1)
    keep &= (
        (choice >= 0)
        & (choice < probs.shape[1])
        & np.all(np.isfinite(probs), axis=1)
    )
    if not np.any(keep):
        return {"n": 0, "choice_brier": float("nan"), "choice_nll": float("nan")}
    selected = choice[keep]
    chosen_probs = probs[keep]
    one_hot = np.zeros_like(chosen_probs)
    one_hot[np.arange(chosen_probs.shape[0]), selected] = 1.0
    brier = float(np.mean(np.sum(np.square(chosen_probs - one_hot), axis=1)))
    nll = float(
        np.mean(
            -np.log(
                np.clip(
                    chosen_probs[np.arange(chosen_probs.shape[0]), selected],
                    1e-12,
                    1.0,
                )
            )
        )
    )
    return {"n": int(np.sum(keep)), "choice_brier": brier, "choice_nll": nll}


def split_masks(subject_frame, valid_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    valid = np.asarray(valid_mask, dtype=bool).copy()
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
            train = valid & (~test)
            return train, test, "last_block"
    return valid.copy(), np.zeros_like(valid), "development_full_no_heldout"


def one_repeat(
    *,
    subject_id: int,
    condition: int,
    arrays,
    params: Mapping[str, Any],
    engine_config: Mapping[str, Any],
    processed_data_dir: Path,
    dataset_paths: Mapping[str, Path],
    window_size: int,
    simulation_point_seed: int,
    repeat_index: int,
) -> Any:
    trajectory_seed = derive_trajectory_seed(
        int(simulation_point_seed),
        "newplan_factorial",
        int(repeat_index),
    )
    return evaluate_state_model_run(
        subject_id=int(subject_id),
        condition=int(condition),
        arrays=arrays,
        params=dict(params),
        engine_config_template=deepcopy(dict(engine_config)),
        processed_data_dir=processed_data_dir,
        window_size=int(window_size),
        dataset_paths=dataset_paths,
        keep_logs=repeat_index == 0,
        include_step_log=False,
        prediction_mode="prior_t",
        selection_prediction_mode="prior_t",
        loss_metric="choice_brier",
        simulation_point_seed=int(simulation_point_seed),
        trajectory_seed=int(trajectory_seed),
        seed_context={
            "phase": "newplan_factorial",
            "subject_id": int(subject_id),
            "repeat_index": int(repeat_index),
        },
    )


def simulate_theta(
    *,
    args: argparse.Namespace,
    runner: StateModelSimulationRunner,
    subject_id: int,
    theta: float,
    base_engine: Mapping[str, Any],
    dataset_paths: Mapping[str, Path],
) -> dict[str, Any]:
    subject_frame = runner._get_subject_frame(subject_id, 1.0)
    if args.max_trials is not None:
        subject_frame = subject_frame.iloc[: int(args.max_trials)].copy()
    arrays = runner._extract_arrays(subject_frame, None)
    condition = runner._get_condition_value(subject_frame)
    params = common_params(args, theta)
    engine_config = apply_fixed_hyperparams_to_engine_config(base_engine, params)
    simulation_point_seed = stable_seed(
        {
            "seed_role": "newplan_factorial_paired_point",
            "crn_scope_version": 2,
            "base_seed": int(args.base_seed),
            "subject_id": int(subject_id),
            "max_trials": args.max_trials,
        }
    )
    runs = Parallel(n_jobs=max(1, int(args.n_jobs)))(
        delayed(one_repeat)(
            subject_id=subject_id,
            condition=condition,
            arrays=arrays,
            params=params,
            engine_config=engine_config,
            processed_data_dir=runner._processed_data_dir,
            dataset_paths=dataset_paths,
            window_size=args.window_size,
            simulation_point_seed=simulation_point_seed,
            repeat_index=repeat_index,
        )
        for repeat_index in range(int(args.repeats))
    )
    metrics = [run.metrics_by_mode["prior_t"] for run in runs]
    probability_stack = np.stack(
        [np.asarray(item["pred_category_probs"], dtype=float) for item in metrics],
        axis=0,
    )
    observed_choice = np.asarray(metrics[0]["observed_choice_index"], dtype=int)
    valid_mask = np.asarray(metrics[0]["valid_trial_mask"], dtype=bool)
    train_mask, test_mask, split_status = split_masks(subject_frame, valid_mask)
    transition_log = runs[0].transition_counts or []
    return {
        "subject_id": int(subject_id),
        "theta": float(theta),
        "params": params,
        "simulation_point_seed": int(simulation_point_seed),
        "simulation_repeats": int(args.repeats),
        "probability_stack": probability_stack,
        "observed_choice_index": observed_choice,
        "valid_mask": valid_mask,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "split_status": split_status,
        "representative_transition": {
            "swap_count": int(sum(bool(item.get("swap_event")) for item in transition_log)),
            "mean_swap_probability": (
                float(np.mean([float(item.get("swap_probability", 0.0)) for item in transition_log]))
                if transition_log
                else float("nan")
            ),
            "fallback_count": int(sum(bool(item.get("fallback")) for item in transition_log)),
        },
    }


def save_theta_cache(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        probability_stack=np.asarray(result["probability_stack"], dtype=float),
        observed_choice_index=np.asarray(result["observed_choice_index"], dtype=int),
        valid_mask=np.asarray(result["valid_mask"], dtype=bool),
        train_mask=np.asarray(result["train_mask"], dtype=bool),
        test_mask=np.asarray(result["test_mask"], dtype=bool),
        metadata=np.asarray(
            json.dumps(
                _jsonable(
                    {
                        key: value
                        for key, value in result.items()
                        if key
                        not in {
                            "probability_stack",
                            "observed_choice_index",
                            "valid_mask",
                            "train_mask",
                            "test_mask",
                        }
                    }
                ),
                ensure_ascii=False,
            )
        ),
    )


def load_theta_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        return {
            **metadata,
            "probability_stack": payload["probability_stack"],
            "observed_choice_index": payload["observed_choice_index"],
            "valid_mask": payload["valid_mask"],
            "train_mask": payload["train_mask"],
            "test_mask": payload["test_mask"],
        }


def parameter_rows(
    args: argparse.Namespace,
    theta_results: Sequence[Mapping[str, Any]],
    epsilon_grid: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in theta_results:
        cognitive_stack = np.asarray(result["probability_stack"], dtype=float)
        n_cats = int(cognitive_stack.shape[2])
        for epsilon in epsilon_grid:
            observed_stack = (
                (1.0 - float(epsilon)) * cognitive_stack
                + float(epsilon) / float(n_cats)
            )
            if args.marginal_method == "importance":
                observed, ess = sequential_importance_marginal(
                    observed_stack,
                    np.asarray(result["observed_choice_index"], dtype=int),
                    np.asarray(result["valid_mask"], dtype=bool),
                )
            else:
                finite = np.isfinite(observed_stack)
                finite_count = np.sum(finite, axis=0)
                observed = np.divide(
                    np.nansum(observed_stack, axis=0),
                    finite_count,
                    out=np.full(observed_stack.shape[1:], np.nan, dtype=float),
                    where=finite_count > 0,
                )
                ess = np.full(
                    observed.shape[0],
                    float(observed_stack.shape[0]),
                    dtype=float,
                )
            split_scores = {
                name: score_probabilities(
                    observed,
                    np.asarray(result["observed_choice_index"], dtype=int),
                    np.asarray(result[f"{name}_mask"], dtype=bool)
                    if name in {"train", "test"}
                    else np.asarray(result["valid_mask"], dtype=bool),
                )
                for name in ("train", "test", "full")
            }
            row: dict[str, Any] = {
                "subject_id": int(result["subject_id"]),
                "theta": float(result["theta"]),
                "epsilon": float(epsilon),
                "split_status": str(result["split_status"]),
                "marginal_method": str(args.marginal_method),
                "mean_ess": float(np.nanmean(ess)),
                "final_ess": float(ess[np.flatnonzero(np.isfinite(ess))[-1]]),
            }
            for split, scores in split_scores.items():
                for key, value in scores.items():
                    row[f"{split}_{key}"] = value
            rows.append(row)
    return rows


def _best_row(rows: Sequence[Mapping[str, Any]], model_id: str) -> dict[str, Any]:
    candidates = []
    for row in rows:
        theta = float(row["theta"])
        epsilon = float(row["epsilon"])
        allowed = {
            "B0": theta == 0.0 and epsilon == 0.0,
            "B1": theta == 0.0,
            "D0": epsilon == 0.0,
            "D1": True,
        }[model_id]
        if allowed and np.isfinite(float(row["train_choice_brier"])):
            candidates.append(dict(row))
    if not candidates:
        raise ValueError(f"No finite candidates for {model_id}.")
    selected = min(
        candidates,
        key=lambda item: (
            float(item["train_choice_brier"]),
            float(item["theta"] > 0.0) + float(item["epsilon"] > 0.0),
            float(item["theta"]),
            float(item["epsilon"]),
        ),
    )
    selected["model_id"] = model_id
    return selected


def select_models(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_subject: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_subject.setdefault(int(row["subject_id"]), []).append(row)
    selected: list[dict[str, Any]] = []
    for subject_id, subject_rows in sorted(by_subject.items()):
        for model_id in MODEL_IDS:
            item = _best_row(subject_rows, model_id)
            item["subject_id"] = subject_id
            selected.append(item)
    return selected


def paired_summary(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lookup = {
        (int(row["subject_id"]), str(row["model_id"])): row
        for row in selected
    }
    comparisons = {
        "lapse_without_swap_B1_minus_B0": ("B1", "B0"),
        "swap_without_lapse_D0_minus_B0": ("D0", "B0"),
        "swap_after_lapse_D1_minus_B1": ("D1", "B1"),
        "lapse_after_swap_D1_minus_D0": ("D1", "D0"),
    }
    out: dict[str, Any] = {}
    subjects = sorted({int(row["subject_id"]) for row in selected})
    for label, (left, right) in comparisons.items():
        values = []
        for subject_id in subjects:
            left_row = lookup[(subject_id, left)]
            right_row = lookup[(subject_id, right)]
            if int(left_row["test_n"]) <= 0 or int(right_row["test_n"]) <= 0:
                continue
            delta = float(left_row["test_choice_brier"]) - float(right_row["test_choice_brier"])
            if np.isfinite(delta):
                values.append(delta)
        out[label] = {
            "n": len(values),
            "mean_test_brier_delta": float(np.mean(values)) if values else float("nan"),
            "median_test_brier_delta": float(np.median(values)) if values else float("nan"),
            "improved_count": int(sum(value < 0.0 for value in values)),
            "values": values,
        }
    return out


def monte_carlo_convergence(
    args: argparse.Namespace,
    theta_results: Sequence[Mapping[str, Any]],
    full_rows: Sequence[Mapping[str, Any]],
    full_selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Compare the full particle set with its nested first-half prefix."""

    prefix_repeats = int(args.repeats) // 2
    if prefix_repeats < 2:
        return None
    prefix_results = []
    for result in theta_results:
        item = dict(result)
        item["probability_stack"] = np.asarray(
            result["probability_stack"], dtype=float
        )[:prefix_repeats]
        item["simulation_repeats"] = prefix_repeats
        prefix_results.append(item)
    prefix_rows = parameter_rows(args, prefix_results, args.epsilon_grid)
    prefix_selected = select_models(prefix_rows)

    def grid_key(row: Mapping[str, Any]) -> tuple[int, float, float]:
        return (
            int(row["subject_id"]),
            float(row["theta"]),
            float(row["epsilon"]),
        )

    prefix_grid = {grid_key(row): row for row in prefix_rows}
    fixed_cell_deltas: dict[str, list[float]] = {
        "train_choice_brier": [],
        "test_choice_brier": [],
    }
    for row in full_rows:
        other = prefix_grid[grid_key(row)]
        for metric in fixed_cell_deltas:
            left = float(row[metric])
            right = float(other[metric])
            if np.isfinite(left) and np.isfinite(right):
                fixed_cell_deltas[metric].append(abs(left - right))

    def selected_key(row: Mapping[str, Any]) -> tuple[int, str]:
        return int(row["subject_id"]), str(row["model_id"])

    prefix_lookup = {selected_key(row): row for row in prefix_selected}
    selection_agreement = []
    selected_test_deltas = []
    for row in full_selected:
        other = prefix_lookup[selected_key(row)]
        selection_agreement.append(
            float(row["theta"]) == float(other["theta"])
            and float(row["epsilon"]) == float(other["epsilon"])
        )
        left = float(row["test_choice_brier"])
        right = float(other["test_choice_brier"])
        if np.isfinite(left) and np.isfinite(right):
            selected_test_deltas.append(abs(left - right))

    prefix_paired = paired_summary(prefix_selected)
    full_paired = paired_summary(full_selected)
    paired_mean_delta_change = {
        label: abs(
            float(full_paired[label]["mean_test_brier_delta"])
            - float(prefix_paired[label]["mean_test_brier_delta"])
        )
        for label in full_paired
        if np.isfinite(float(full_paired[label]["mean_test_brier_delta"]))
        and np.isfinite(float(prefix_paired[label]["mean_test_brier_delta"]))
    }
    final_ess = np.asarray(
        [float(row["final_ess"]) for row in full_selected], dtype=float
    )
    mean_ess = np.asarray(
        [float(row["mean_ess"]) for row in full_selected], dtype=float
    )
    return {
        "prefix_repeats": prefix_repeats,
        "full_repeats": int(args.repeats),
        "common_random_number_prefix": True,
        "fixed_cell_absolute_difference": {
            metric: {
                "mean": float(np.mean(values)) if values else float("nan"),
                "max": float(np.max(values)) if values else float("nan"),
                "n": len(values),
            }
            for metric, values in fixed_cell_deltas.items()
        },
        "selected_parameter_agreement": {
            "count": int(sum(selection_agreement)),
            "n": len(selection_agreement),
        },
        "selected_test_brier_absolute_difference": {
            "mean": (
                float(np.mean(selected_test_deltas))
                if selected_test_deltas
                else float("nan")
            ),
            "max": (
                float(np.max(selected_test_deltas))
                if selected_test_deltas
                else float("nan")
            ),
            "n": len(selected_test_deltas),
        },
        "paired_mean_delta_absolute_change": paired_mean_delta_change,
        "full_selected_ess": {
            "mean_over_time_mean": float(np.nanmean(mean_ess)),
            "final_median": float(np.nanmedian(final_ess)),
            "final_min": float(np.nanmin(final_ess)),
            "final_median_fraction": float(
                np.nanmedian(final_ess) / float(args.repeats)
            ),
        },
        "prefix_paired_heldout": prefix_paired,
        "prefix_selected_models": prefix_selected,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    selected: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    convergence: Mapping[str, Any] | None,
) -> None:
    lines = [
        "# Condition-1 new-plan factorial result",
        "",
        "This is a fixed-common-core theta × epsilon comparison. It is an engineering/development result, not the final jointly re-optimized confirmation.",
        "",
        f"- subjects: {', '.join(str(value) for value in args.subjects)}",
        f"- repeats per theta: {args.repeats}",
        f"- gamma={args.gamma}, w0={args.w0}, rho={args.rho}",
        f"- marginal method: {args.marginal_method}",
        f"- max_trials={args.max_trials}",
        f"- theta grid: {', '.join(str(value) for value in args.theta_grid)}",
        f"- epsilon grid: {', '.join(str(value) for value in args.epsilon_grid)}",
        "",
        "## Paired held-out effects",
        "",
        "Negative Brier delta favors the first (more complex) model.",
        "",
    ]
    for label, item in summary.items():
        lines.append(
            f"- {label}: mean={item['mean_test_brier_delta']:.6f}, "
            f"median={item['median_test_brier_delta']:.6f}, "
            f"improved={item['improved_count']}/{item['n']}"
        )
    if convergence is not None:
        fixed = convergence["fixed_cell_absolute_difference"]
        agreement = convergence["selected_parameter_agreement"]
        selected_delta = convergence["selected_test_brier_absolute_difference"]
        ess = convergence["full_selected_ess"]
        lines.extend(
            [
                "",
                "## Nested Monte Carlo convergence",
                "",
                f"The first {convergence['prefix_repeats']} paths are exactly shared "
                f"with the full {convergence['full_repeats']}-path estimate.",
                "",
                f"- fixed-cell mean absolute train-Brier change: "
                f"{fixed['train_choice_brier']['mean']:.6f}",
                f"- fixed-cell mean absolute test-Brier change: "
                f"{fixed['test_choice_brier']['mean']:.6f}",
                f"- selected theta/epsilon agreement: "
                f"{agreement['count']}/{agreement['n']}",
                f"- selected-model mean absolute test-Brier change: "
                f"{selected_delta['mean']:.6f}",
                f"- full-run final ESS: median={ess['final_median']:.2f}, "
                f"min={ess['final_min']:.2f}, "
                f"median fraction={ess['final_median_fraction']:.3f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Selected parameters",
            "",
            "| subject | model | theta | epsilon | train Brier | test Brier | test n |",
            "|---:|:---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selected:
        lines.append(
            f"| {row['subject_id']} | {row['model_id']} | {row['theta']:.3f} | "
            f"{row['epsilon']:.3f} | {row['train_choice_brier']:.6f} | "
            f"{row['test_choice_brier']:.6f} | {row['test_n']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.theta_grid = validate_probability_grid(args.theta_grid, "theta_grid")
    args.epsilon_grid = validate_probability_grid(args.epsilon_grid, "epsilon_grid")
    if args.repeats <= 0 or args.n_jobs <= 0:
        raise ValueError("repeats and n_jobs must be positive.")
    if not 0.0 <= args.w0 <= 1.0 or not 0.0 <= args.gamma <= 1.0 or args.rho < 1.0:
        raise ValueError("Require gamma,w0 in [0,1] and rho >= 1.")

    model_path = ROOT / "configs/model_struct/pmh_model_cond1_newplan.yaml"
    sim_path = ROOT / "configs/simulation_cfg/pmh_cond1_simulation_v14.yaml"
    base_engine = load_yaml(model_path)
    sim_cfg = load_yaml(sim_path)
    dataset_paths = resolve_dataset_paths(sim_cfg, sim_path.parent, DEFAULT_DATA_PATH)
    runner = StateModelSimulationRunner(
        engine_config=base_engine,
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
        n_jobs=max(1, args.n_jobs),
    )
    runner.prepare_data(dataset_paths["learning_data"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "manifest.json",
        {
            "result_type": "cond1_newplan_factorial",
            "subjects": args.subjects,
            "theta_grid": args.theta_grid,
            "epsilon_grid": args.epsilon_grid,
            "gamma": args.gamma,
            "w0": args.w0,
            "rho": args.rho,
            "repeats": args.repeats,
            "n_jobs": args.n_jobs,
            "max_trials": args.max_trials,
            "base_seed": args.base_seed,
            "common_random_numbers": True,
            "crn_scope_version": 2,
            "common_random_numbers_across_common_core": True,
            "marginal_method": args.marginal_method,
            "lapse_applied_to_each_path_before_importance_weighting": (
                args.marginal_method == "importance"
            ),
            "model_config": str(model_path),
        },
    )

    all_theta_results: list[dict[str, Any]] = []
    for subject_id in args.subjects:
        for theta in args.theta_grid:
            cache_path = (
                args.output_dir
                / "cache"
                / f"subject_{int(subject_id)}"
                / f"theta_{theta_token(theta)}.npz"
            )
            if cache_path.exists() and not args.force:
                result = load_theta_cache(cache_path)
                cached_repeats = int(
                    np.asarray(result["probability_stack"]).shape[0]
                )
                if cached_repeats != int(args.repeats):
                    print(
                        f"RERUN subject={subject_id} theta={theta:.3f} "
                        f"(cache repeats={cached_repeats}, requested={args.repeats})",
                        flush=True,
                    )
                    result = simulate_theta(
                        args=args,
                        runner=runner,
                        subject_id=int(subject_id),
                        theta=float(theta),
                        base_engine=base_engine,
                        dataset_paths=dataset_paths,
                    )
                    save_theta_cache(cache_path, result)
                else:
                    print(f"LOAD subject={subject_id} theta={theta:.3f}", flush=True)
            else:
                print(f"RUN subject={subject_id} theta={theta:.3f}", flush=True)
                result = simulate_theta(
                    args=args,
                    runner=runner,
                    subject_id=int(subject_id),
                    theta=float(theta),
                    base_engine=base_engine,
                    dataset_paths=dataset_paths,
                )
                save_theta_cache(cache_path, result)
            all_theta_results.append(result)

    rows = parameter_rows(args, all_theta_results, args.epsilon_grid)
    selected = select_models(rows)
    summary = paired_summary(selected)
    convergence = monte_carlo_convergence(
        args,
        all_theta_results,
        rows,
        selected,
    )
    write_csv(args.output_dir / "parameter_grid.csv", rows)
    write_csv(args.output_dir / "selected_models.csv", selected)
    write_json(
        args.output_dir / "aggregate_summary.json",
        {
            "paired_heldout": summary,
            "selected_models": selected,
            "monte_carlo_convergence": convergence,
        },
    )
    write_report(
        args.output_dir / "RESULTS.md",
        args=args,
        selected=selected,
        summary=summary,
        convergence=convergence,
    )
    print(f"COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
