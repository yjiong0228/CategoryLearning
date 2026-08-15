#!/usr/bin/env python3
"""Evaluate PF repeat and particle-count convergence for the 0815 P0 model.

The diagnostic compares probability-averaged predictions, not the best PF
seed. It also checks the inferred executed-rule distribution so numerical
stability of choice NLL cannot hide unstable latent-state conclusions.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
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


DEFAULT_CONFIG = ROOT / "configs/specific_models/model_0815_p0_pf_convergence.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one subject, two small particle counts, two repeats, and 24 trials.",
    )
    return parser.parse_args()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


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
        raise ValueError("Executed-posterior matrices must have equal 2-D shapes.")
    keep = np.all(np.isfinite(left), axis=1) & np.all(np.isfinite(right), axis=1)
    if not np.any(keep):
        return float("nan")
    left = np.clip(left[keep], 0.0, None)
    right = np.clip(right[keep], 0.0, None)
    left_sum = np.sum(left, axis=1, keepdims=True)
    right_sum = np.sum(right, axis=1, keepdims=True)
    valid = (left_sum[:, 0] > 0.0) & (right_sum[:, 0] > 0.0)
    left = left[valid] / left_sum[valid]
    right = right[valid] / right_sum[valid]
    if left.size == 0:
        return float("nan")
    midpoint = 0.5 * (left + right)

    def kl(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
        mask = values > 0.0
        terms = np.zeros_like(values)
        terms[mask] = values[mask] * np.log(
            values[mask] / np.clip(reference[mask], 1e-12, None)
        )
        return np.sum(terms, axis=1)

    return float(np.mean(0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)))


def summarize_repeat_panel(
    raw_runs: Sequence[Mapping[str, Any]],
    *,
    prediction_mode: str,
    particle_count: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Summarize one subject/count panel and return arrays for count comparison."""

    if len(raw_runs) < 2:
        raise ValueError("PF convergence requires at least two filter repeats.")
    probability_rows: list[np.ndarray] = []
    executed_rows: list[np.ndarray] = []
    ess_rows: list[np.ndarray] = []
    choice_index = None
    valid = None
    run_nll: list[float] = []
    for run in raw_runs:
        metrics = run["metrics_by_mode"][prediction_mode]
        probability = np.asarray(metrics["pred_category_probs"], dtype=float)
        current_choice = np.asarray(metrics["observed_choice_index"], dtype=int)
        current_valid = np.asarray(metrics["valid_trial_mask"], dtype=bool)
        if choice_index is None:
            choice_index = current_choice
            valid = current_valid
        elif not np.array_equal(choice_index, current_choice) or not np.array_equal(
            valid,
            current_valid,
        ):
            raise ValueError("Observed trials differ across PF repeats.")
        probability_rows.append(probability)
        run_nll.append(_choice_nll(probability, current_choice, current_valid))
        state_log = run.get("state_log") or {}
        executed = state_log.get("filtered_executed_probability")
        if executed is None:
            raise ValueError(
                "PF convergence requires filtered_executed_probability logs."
            )
        executed_rows.append(np.asarray(executed, dtype=float))
        ess_rows.append(np.asarray(state_log["post_choice_ess"], dtype=float))

    probabilities = np.stack(probability_rows, axis=0)
    executed = np.stack(executed_rows, axis=0)
    ess = np.stack(ess_rows, axis=0)
    mean_probability = np.mean(probabilities, axis=0)
    mean_executed = np.mean(executed, axis=0)
    split = len(raw_runs) // 2
    first_half = np.mean(probabilities[:split], axis=0)
    second_half = np.mean(probabilities[split:], axis=0)
    score_mask = np.asarray(valid, dtype=bool)
    split_rmse = float(
        np.sqrt(np.mean(np.square(first_half[score_mask] - second_half[score_mask])))
    )
    run_nll_values = np.asarray(run_nll, dtype=float)
    row = {
        "particle_count": int(particle_count),
        "repeat_count": int(len(raw_runs)),
        "choice_nll": _choice_nll(
            mean_probability,
            np.asarray(choice_index, dtype=int),
            score_mask,
        ),
        "run_choice_nll_mean": float(np.mean(run_nll_values)),
        "run_choice_nll_sd": float(np.std(run_nll_values, ddof=1)),
        "split_half_choice_probability_rmse": split_rmse,
        "median_post_choice_ess_fraction": float(
            np.median(ess / float(particle_count))
        ),
    }
    arrays = {
        "choice_probability": mean_probability,
        "filtered_executed_probability": mean_executed,
        "valid_trial_mask": score_mask,
    }
    return row, arrays


def compare_successive_counts(
    count_rows: Sequence[Mapping[str, Any]],
    arrays_by_count: Mapping[int, Mapping[str, np.ndarray]],
    *,
    gates: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Apply frozen probability/state convergence gates to adjacent counts."""

    by_count = {int(row["particle_count"]): dict(row) for row in count_rows}
    counts = sorted(by_count)
    comparisons: list[dict[str, Any]] = []
    for lower, upper in zip(counts[:-1], counts[1:]):
        left = arrays_by_count[lower]
        right = arrays_by_count[upper]
        valid = np.asarray(left["valid_trial_mask"], dtype=bool) & np.asarray(
            right["valid_trial_mask"], dtype=bool
        )
        probability_rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        left["choice_probability"][valid]
                        - right["choice_probability"][valid]
                    )
                )
            )
        )
        nll_change = abs(
            float(by_count[lower]["choice_nll"])
            - float(by_count[upper]["choice_nll"])
        )
        executed_js = _mean_js(
            left["filtered_executed_probability"],
            right["filtered_executed_probability"],
        )
        checks = {
            "choice_nll": nll_change
            <= float(gates["maximum_successive_choice_nll_change"]),
            "choice_probability": probability_rmse
            <= float(gates["maximum_successive_choice_probability_rmse"]),
            "executed_posterior": executed_js
            <= float(gates["maximum_successive_executed_posterior_js"]),
            "split_half": float(
                by_count[lower]["split_half_choice_probability_rmse"]
            )
            <= float(gates["maximum_split_half_choice_probability_rmse"]),
            "ess": float(by_count[lower]["median_post_choice_ess_fraction"])
            >= float(gates["minimum_median_post_choice_ess_fraction"]),
        }
        comparisons.append(
            {
                "lower_particle_count": lower,
                "upper_particle_count": upper,
                "successive_choice_nll_change": nll_change,
                "successive_choice_probability_rmse": probability_rmse,
                "successive_executed_posterior_js": executed_js,
                **{f"gate_{name}_passed": bool(value) for name, value in checks.items()},
                "all_gates_passed": bool(all(checks.values())),
            }
        )
    return comparisons


def _run_subject_count(
    *,
    simulation_config_path: Path,
    simulation_config: Mapping[str, Any],
    subject_id: int,
    particle_count: int,
    repeat_count: int,
    max_trials: int | None,
    base_seed: int,
    n_jobs: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    subject_cfg = resolve_subject_config(simulation_config, subject_id)
    engine_config = resolve_engine_config(
        subject_cfg,
        simulation_config_path.parent,
        subject_id=subject_id,
    )
    fixed_hyperparams = {
        **infer_fixed_hyperparams_from_engine_config(engine_config),
        **dict(subject_cfg.get("fixed_hyperparams") or {}),
    }
    engine_config = apply_fixed_hyperparams_to_engine_config(
        engine_config,
        fixed_hyperparams,
    )
    engine_config.setdefault("inference", {})["particle_count"] = int(
        particle_count
    )
    dataset_paths = resolve_dataset_paths(
        subject_cfg,
        simulation_config_path.parent,
    )
    runner = StateModelSimulationRunner(
        engine_config=engine_config,
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
        n_jobs=int(n_jobs),
    )
    runner.prepare_data(dataset_paths["learning_data"])
    prediction_mode, selection_mode = resolve_prediction_modes(subject_cfg)
    loss_metric = resolve_loss_metric(subject_cfg)
    loss_delta = resolve_loss_delta(subject_cfg, loss_metric)
    seeds = [
        stable_seed(
            {
                "seed_role": "model0815_p0_pf_convergence",
                "base_seed": int(base_seed),
                "subject_id": int(subject_id),
                "repeat_index": int(index),
            }
        )
        for index in range(int(repeat_count))
    ]
    result = runner.simulate_subject(
        subject_id=int(subject_id),
        simulation_repeats=int(repeat_count),
        fixed_hyperparams=fixed_hyperparams,
        window_size=resolve_window_size(subject_cfg, subject_id, [subject_id]),
        stop_at=float(subject_cfg.get("stop_at", 1.0)),
        max_trials=max_trials,
        keep_logs=True,
        prediction_mode=prediction_mode,
        selection_prediction_mode=selection_mode,
        loss_metric=loss_metric,
        loss_delta=loss_delta,
        hyper_candidate_seed=int(base_seed),
        trajectory_seeds=seeds,
        compute_statistics=False,
        repeat_aggregation="mean_probability",
        evaluation_protocol=subject_cfg.get("evaluation_protocol"),
    )
    row, arrays = summarize_repeat_panel(
        result["best"].raw_runs or [],
        prediction_mode=selection_mode,
        particle_count=particle_count,
    )
    row["subject_id"] = int(subject_id)
    return row, arrays


def main() -> None:
    args = parse_args()
    config_path = _repo_path(args.config)
    config = load_yaml(config_path)
    calibration = deepcopy(dict(config["calibration"]))
    if args.smoke:
        calibration["subjects"] = [int(calibration["subjects"][0])]
        calibration["particle_counts"] = [4, 8]
        calibration["filter_seed_repeats"] = 2
        calibration["max_trials"] = 24
        calibration["n_jobs"] = 1
    if args.n_jobs is not None:
        calibration["n_jobs"] = int(args.n_jobs)

    output = (
        _repo_path(args.output_dir)
        if args.output_dir is not None
        else _repo_path(config["output_dir"])
    )
    if args.smoke:
        output = output / "smoke"
    if (output / "summary.json").exists():
        raise FileExistsError(
            f"Refusing to overwrite completed convergence output: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)

    simulation_config_path = _repo_path(config["base_simulation_config"])
    simulation_config = load_yaml(simulation_config_path)
    subjects = [int(value) for value in calibration["subjects"]]
    counts = [int(value) for value in calibration["particle_counts"]]
    repeat_count = int(calibration["filter_seed_repeats"])
    max_trials_raw = calibration.get("max_trials")
    max_trials = None if max_trials_raw is None else int(max_trials_raw)
    gates = dict(calibration["stability_gates"])

    rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    for subject_id in subjects:
        arrays_by_count: dict[int, dict[str, np.ndarray]] = {}
        subject_rows: list[dict[str, Any]] = []
        for particle_count in counts:
            row, arrays = _run_subject_count(
                simulation_config_path=simulation_config_path,
                simulation_config=simulation_config,
                subject_id=subject_id,
                particle_count=particle_count,
                repeat_count=repeat_count,
                max_trials=max_trials,
                base_seed=int(calibration["base_seed"]),
                n_jobs=int(calibration["n_jobs"]),
            )
            rows.append(row)
            subject_rows.append(row)
            arrays_by_count[particle_count] = arrays
        subject_comparisons = compare_successive_counts(
            subject_rows,
            arrays_by_count,
            gates=gates,
        )
        for comparison in subject_comparisons:
            comparison["subject_id"] = subject_id
        comparisons.extend(subject_comparisons)

    count_frame = pd.DataFrame(rows).sort_values(["subject_id", "particle_count"])
    comparison_frame = pd.DataFrame(comparisons).sort_values(
        ["subject_id", "lower_particle_count"]
    )
    count_frame.to_csv(output / "particle_count_summary.csv", index=False)
    comparison_frame.to_csv(output / "successive_count_comparisons.csv", index=False)

    count_pass = (
        comparison_frame.groupby("lower_particle_count")["all_gates_passed"]
        .all()
        .to_dict()
    )
    stable_counts = sorted(int(key) for key, value in count_pass.items() if bool(value))
    summary = {
        "analysis_id": config["analysis_id"],
        "base_simulation_config": str(simulation_config_path.relative_to(ROOT)),
        "subjects": subjects,
        "particle_counts": counts,
        "filter_seed_repeats": repeat_count,
        "stability_gates": gates,
        "particle_count_passes_all_subjects": {
            str(key): bool(value) for key, value in count_pass.items()
        },
        "minimum_stable_particle_count": stable_counts[0] if stable_counts else None,
        "all_gates_pass": bool(stable_counts),
        "smoke": bool(args.smoke),
    }
    (output / "summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "analysis_config_snapshot.json").write_text(
        json.dumps(_json_safe(config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
