#!/usr/bin/env python3
"""Conditioned posterior-predictive trajectory check for frozen condition-1 B0.

For each subject, the filter reads only the prefix before the prediction
boundary.  It then generates many autonomous suffix trajectories on the exact
physical stimulus/category schedule.  Adequacy is evaluated against the
model's own repeated-sampling distribution, using both a simultaneous rolling
accuracy envelope and a predeclared vector of trajectory summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Bayesian_state.run_simulation import (  # noqa: E402
    apply_fixed_hyperparams_to_engine_config,
)
from src.Bayesian_state.utils.datasets import (  # noqa: E402
    resolve_dataset_paths,
)
from src.Bayesian_state.inference_engine.posterior_predictive import (  # noqa: E402
    DynamicRhoConfig,
    run_conditioned_condition1_rollouts,
)
from src.Bayesian_state.inference_engine.backends.particle_filter import (  # noqa: E402
    run_state_model_particle_filter,
)
from src.Bayesian_state.optimization.optimization_config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    load_yaml,
)
from src.Bayesian_state.utils.seeding import stable_seed  # noqa: E402


DEVELOPMENT_SUBJECTS = (103, 105, 111, 112, 117, 118, 127, 131)
KEY_COLUMNS = ("iSub", "iSession", "iBlock", "iTrial")
FEATURE_COLUMNS = ("feature1", "feature2", "feature3", "feature4")


@dataclass(frozen=True)
class MetricSpec:
    metric: str
    label: str
    resolution_kind: str


METRIC_SPECS = (
    MetricSpec("accuracy", "测试段总体正确率", "rate"),
    MetricSpec("early_accuracy", "测试段前窗口正确率", "window_rate"),
    MetricSpec("late_accuracy", "测试段末窗口正确率", "window_rate"),
    MetricSpec("accuracy_slope", "测试段正确率线性斜率", "rate"),
    MetricSpec("max_adjacent_rise", "相邻窗口最大正确率上升", "window_rate"),
    MetricSpec("max_adjacent_drop", "相邻窗口最大正确率下降", "window_rate"),
    MetricSpec("trend_reversal_count", "窗口趋势方向反转数", "count"),
    MetricSpec("event_count", "突降事件数", "count"),
    MetricSpec("max_event_duration", "最长突降事件长度", "count"),
    MetricSpec("longest_error_streak", "最长连续错误", "count"),
    MetricSpec(
        "longest_low_accuracy_run",
        "低正确率滚动窗口最长连续段",
        "count",
    ),
    MetricSpec("dominant_choice_rate", "优势反应比例", "rate"),
    MetricSpec("choice_entropy", "选择熵", "rate"),
    MetricSpec("switch_rate", "反应切换率", "rate"),
    MetricSpec("lose_stay_rate", "错误后保持率", "rate"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/processed/Task2_processed.csv",
    )
    parser.add_argument("--subjects", type=int, nargs="+")
    parser.add_argument("--particle-count", type=int, default=128)
    parser.add_argument("--rollout-count", type=int, default=512)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--gamma", type=float, default=0.55)
    parser.add_argument("--w0", type=float, default=0.10)
    parser.add_argument("--rho", type=float, default=2.0)
    parser.add_argument(
        "--dynamic-rho-start",
        type=float,
        help=(
            "Population-median rho at the first trial. Supplying this "
            "activates the continuous dynamic-rho model."
        ),
    )
    parser.add_argument(
        "--dynamic-rho-end",
        type=float,
        help=(
            "Population-median rho at the shared absolute reference trial, "
            "not at each subject's observed final trial."
        ),
    )
    parser.add_argument(
        "--dynamic-rho-volatility",
        type=float,
        default=0.0,
        help="Innovation scale of the log-rho AR(1) deviation.",
    )
    parser.add_argument(
        "--dynamic-rho-persistence",
        type=float,
        default=0.95,
        help="AR(1) persistence of the log-rho deviation.",
    )
    parser.add_argument(
        "--dynamic-rho-start-log-sd",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--dynamic-rho-gain-log-sd",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--dynamic-rho-volatility-log-sd",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--dynamic-rho-reference-trials",
        type=int,
        default=128,
        help=(
            "Shared absolute trial at which dynamic-rho-end is defined."
        ),
    )
    parser.add_argument("--beta-init", type=float, default=5.0)
    parser.add_argument("--beta-correct-additive", type=float, default=0.5)
    parser.add_argument("--beta-decrease-rate", type=float, default=0.15)
    parser.add_argument(
        "--beta-additive-grid",
        type=float,
        nargs="+",
        help=(
            "Optional candidate grid selected separately for each subject "
            "using only the observed prediction prefix."
        ),
    )
    parser.add_argument("--lapse-start", type=float, default=0.0)
    parser.add_argument(
        "--lapse-start-grid",
        type=float,
        nargs="+",
        help=(
            "Optional initial lapse/exploration grid selected separately "
            "for each subject using only the prediction prefix."
        ),
    )
    parser.add_argument(
        "--lapse-half-life",
        type=float,
        default=128.0,
        help="Trials required for the lapse mixture weight to halve.",
    )
    parser.add_argument(
        "--learning-update-probability",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--acquisition-half-life",
        type=float,
        help=(
            "Optional median trial of one irreversible novice-to-acquired "
            "readout change-point. Omit to use the ordinary static readout."
        ),
    )
    parser.add_argument(
        "--pre-acquisition-lapse",
        type=float,
        default=1.0,
        help=(
            "Uninformed-response mixture before the irreversible acquisition "
            "boundary; 1 is pure guessing and 0 is ordinary readout."
        ),
    )
    parser.add_argument(
        "--learning-update-grid",
        type=float,
        nargs="+",
        help=(
            "Optional per-trial evidence-update probability grid selected "
            "per subject from the observed prediction prefix."
        ),
    )
    parser.add_argument("--selection-particle-count", type=int, default=8)
    parser.add_argument("--capacity", type=int, default=5)
    parser.add_argument("--resample-threshold", type=float, default=0.5)
    parser.add_argument("--base-seed", type=int, default=20260801)
    parser.add_argument(
        "--split-mode",
        choices=("last_block", "early_anchor"),
        default="last_block",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_active_set/b0_trajectory_ppc"
        ),
    )
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=True,
        )
        + "\n",
        encoding="utf-8",
    )


def benjamini_hochberg(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not finite_indices.size:
        return adjusted
    finite = values[finite_indices]
    order = np.argsort(finite)
    ranked = finite[order]
    n_tests = ranked.size
    raw = ranked * n_tests / np.arange(1, n_tests + 1)
    monotone = np.minimum.accumulate(raw[::-1])[::-1]
    local = np.empty(n_tests, dtype=float)
    local[order] = np.clip(monotone, 0.0, 1.0)
    adjusted[finite_indices] = local
    return adjusted


def longest_true_run(values: Sequence[bool] | np.ndarray) -> int:
    array = np.asarray(values, dtype=bool).reshape(-1)
    best = 0
    current = 0
    for value in array:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def rolling_accuracy(feedback: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(feedback, dtype=float)
    if values.shape[-1] < int(window):
        return np.empty((*values.shape[:-1], 0), dtype=float)
    kernel = np.ones(int(window), dtype=float) / float(window)
    if values.ndim == 1:
        return np.convolve(values, kernel, mode="valid")
    return np.vstack(
        [np.convolve(row, kernel, mode="valid") for row in values]
    )


def decaying_lapse_schedule(
    n_trials: int,
    *,
    initial_lapse: float,
    half_life: float,
) -> np.ndarray:
    start = float(initial_lapse)
    duration = float(half_life)
    if not 0.0 <= start <= 1.0:
        raise ValueError("initial_lapse must lie in [0, 1].")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("half_life must be positive and finite.")
    positions = np.arange(int(n_trials), dtype=float)
    return start * np.power(0.5, positions / duration)


def acquisition_hazard_from_half_life(half_life: float) -> float:
    """Convert a geometric acquisition-time median into a trial hazard."""

    duration = float(half_life)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("acquisition half-life must be positive and finite.")
    return float(1.0 - np.power(0.5, 1.0 / duration))


def block_bounded_rolling_accuracy(
    feedback: np.ndarray,
    block_ids: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(feedback, dtype=float)
    blocks = np.asarray(block_ids).reshape(-1)
    if values.shape[-1] != blocks.size:
        raise ValueError("block_ids must align with the trajectory.")
    unique_blocks = list(dict.fromkeys(blocks.tolist()))
    pieces: list[np.ndarray] = []
    end_indices: list[int] = []
    for block in unique_blocks:
        indices = np.flatnonzero(blocks == block)
        if indices.size < int(window):
            continue
        local = values[..., indices]
        pieces.append(rolling_accuracy(local, int(window)))
        end_indices.extend(indices[int(window) - 1 :].tolist())
    if not pieces:
        empty_shape = (*values.shape[:-1], 0)
        return np.empty(empty_shape, dtype=float), np.asarray([], dtype=int)
    return np.concatenate(pieces, axis=-1), np.asarray(end_indices, dtype=int)


def detect_events(feedback: np.ndarray, window: int) -> list[dict[str, int]]:
    values = np.asarray(feedback, dtype=float)
    candidates: list[int] = []
    for onset in range(int(window), len(values) - int(window) + 1):
        pre = float(np.mean(values[onset - window : onset]))
        post = float(np.mean(values[onset : onset + window]))
        if pre >= 2.0 / 3.0 and post <= 0.5 and pre - post >= 0.25:
            candidates.append(onset)
    events: list[dict[str, int]] = []
    blocked_until = -1
    for onset in candidates:
        if onset <= blocked_until:
            continue
        recovery = None
        for position in range(
            onset + int(window),
            len(values) - int(window) + 1,
        ):
            if float(np.mean(values[position : position + window])) >= 2.0 / 3.0:
                recovery = position
                break
        end = len(values) - 1 if recovery is None else recovery - 1
        events.append(
            {
                "onset": int(onset),
                "end": int(end),
                "duration": int(end - onset + 1),
            }
        )
        blocked_until = end
    return events


def trajectory_metrics(
    feedback: np.ndarray,
    choices: np.ndarray,
    *,
    window: int,
    block_ids: np.ndarray | None = None,
) -> dict[str, float]:
    outcomes = np.asarray(feedback, dtype=float).reshape(-1)
    response = np.asarray(choices, dtype=int).reshape(-1)
    n_trials = int(outcomes.size)
    blocks = (
        np.zeros(n_trials, dtype=int)
        if block_ids is None
        else np.asarray(block_ids).reshape(-1)
    )
    if blocks.size != n_trials:
        raise ValueError("block_ids must align with feedback.")
    local_window = min(int(window), n_trials)
    positions = np.linspace(-0.5, 0.5, n_trials)
    slope = (
        float(np.polyfit(positions, outcomes, 1)[0])
        if n_trials >= 2
        else 0.0
    )
    drops: list[float] = []
    rises: list[float] = []
    trend_reversal_count = 0
    events: list[dict[str, int]] = []
    longest_error = 0
    longest_low = 0
    for block in dict.fromkeys(blocks.tolist()):
        indices = np.flatnonzero(blocks == block)
        local_outcomes = outcomes[indices]
        block_window = min(int(window), len(local_outcomes))
        adjacent_changes = [
            float(
                np.mean(local_outcomes[cut : cut + block_window])
                - np.mean(local_outcomes[cut - block_window : cut])
            )
            for cut in range(
                block_window,
                len(local_outcomes) - block_window + 1,
            )
        ]
        rises.extend(adjacent_changes)
        drops.extend(-change for change in adjacent_changes)
        nonoverlapping_means = np.asarray(
            [
                np.mean(local_outcomes[start : start + block_window])
                for start in range(
                    0,
                    len(local_outcomes) - block_window + 1,
                    block_window,
                )
            ],
            dtype=float,
        )
        if nonoverlapping_means.size >= 3:
            changes = np.diff(nonoverlapping_means)
            signs = np.sign(
                np.where(
                    np.abs(changes) >= 1.0 / block_window - 1e-12,
                    changes,
                    0.0,
                )
            )
            nonzero = signs[signs != 0.0]
            if nonzero.size >= 2:
                trend_reversal_count += int(
                    np.sum(nonzero[1:] != nonzero[:-1])
                )
        events.extend(detect_events(local_outcomes, block_window))
        local_rolling = rolling_accuracy(local_outcomes, block_window)
        longest_error = max(
            longest_error,
            longest_true_run(local_outcomes < 1.0),
        )
        longest_low = max(
            longest_low,
            longest_true_run(local_rolling <= 0.5),
        )
    proportions = np.asarray(
        [
            np.mean(response == category)
            for category in (1, 2)
        ],
        dtype=float,
    )
    positive = proportions[proportions > 0.0]
    entropy = float(-np.sum(positive * np.log(positive)))
    if n_trials >= 2:
        stay = response[1:] == response[:-1]
        prior_loss = outcomes[:-1] < 1.0
        within_block = blocks[1:] == blocks[:-1]
        loss_transitions = prior_loss & within_block
        lose_stay = (
            float(np.mean(stay[loss_transitions]))
            if np.any(loss_transitions)
            else float("nan")
        )
        switch_rate = (
            float(np.mean((~stay)[within_block]))
            if np.any(within_block)
            else float("nan")
        )
    else:
        lose_stay = float("nan")
        switch_rate = float("nan")
    return {
        "accuracy": float(np.mean(outcomes)),
        "early_accuracy": float(np.mean(outcomes[:local_window])),
        "late_accuracy": float(np.mean(outcomes[-local_window:])),
        "accuracy_slope": slope,
        "max_adjacent_rise": max([0.0, *rises]),
        "max_adjacent_drop": max([0.0, *drops]),
        "trend_reversal_count": float(trend_reversal_count),
        "event_count": float(len(events)),
        "max_event_duration": float(
            max((event["duration"] for event in events), default=0)
        ),
        "longest_error_streak": float(longest_error),
        "longest_low_accuracy_run": float(longest_low),
        "dominant_choice_rate": float(np.max(proportions)),
        "choice_entropy": entropy,
        "switch_rate": switch_rate,
        "lose_stay_rate": lose_stay,
    }


def metric_resolution(
    spec: MetricSpec,
    *,
    n_trials: int,
    window: int,
) -> float:
    if spec.resolution_kind == "count":
        return 1.0
    if spec.resolution_kind == "window_rate":
        return 1.0 / max(1, min(int(window), int(n_trials)))
    return 1.0 / max(1, int(n_trials))


def robust_scales(
    simulations: np.ndarray,
    resolutions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    median = np.nanmedian(simulations, axis=0)
    q25 = np.nanquantile(simulations, 0.25, axis=0)
    q75 = np.nanquantile(simulations, 0.75, axis=0)
    scale = (q75 - q25) / 1.349
    std = np.nanstd(simulations, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, std)
    scale = np.where(
        np.isfinite(scale) & (scale > 1e-12),
        scale,
        resolutions,
    )
    return median, np.maximum(scale, resolutions)


def empirical_crps(
    observed: np.ndarray,
    simulations: np.ndarray,
) -> float:
    y = np.asarray(observed, dtype=float).reshape(-1)
    samples = np.asarray(simulations, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != y.size:
        raise ValueError("CRPS simulations must be rollout-by-position.")
    n = samples.shape[0]
    sorted_samples = np.sort(samples, axis=0)
    coefficients = (
        2.0 * np.arange(1, n + 1, dtype=float) - n - 1.0
    )[:, None]
    pair_penalty = np.sum(
        coefficients * sorted_samples, axis=0
    ) / float(n * n)
    point_crps = np.mean(np.abs(samples - y[None, :]), axis=0) - pair_penalty
    return float(np.mean(point_crps))


def split_for_subject(
    subject_frame: pd.DataFrame,
    *,
    mode: str = "last_block",
    window: int = 12,
) -> tuple[int, str]:
    pairs = list(
        dict.fromkeys(
            zip(
                subject_frame["iSession"].astype(int),
                subject_frame["iBlock"].astype(int),
            )
        )
    )
    if mode == "early_anchor" and len(pairs) >= 2:
        first_session, first_block = pairs[0]
        first_block_mask = (
            subject_frame["iSession"].to_numpy(dtype=int) == first_session
        ) & (
            subject_frame["iBlock"].to_numpy(dtype=int) == first_block
        )
        split_index = int(np.flatnonzero(~first_block_mask)[0])
        return split_index, "after_first_block"
    if mode == "last_block" and len(pairs) >= 2:
        last_session, last_block = pairs[-1]
        test = (
            subject_frame["iSession"].to_numpy(dtype=int) == last_session
        ) & (
            subject_frame["iBlock"].to_numpy(dtype=int) == last_block
        )
        split_index = int(np.flatnonzero(test)[0])
        return split_index, "last_block"
    if mode == "early_anchor":
        split_index = min(
            len(subject_frame) - 1,
            max(int(window), len(subject_frame) // 4),
        )
        return split_index, "after_first_quarter_single_block"
    split_index = max(1, len(subject_frame) // 2)
    return split_index, "last_half_single_block"


def cache_path(
    output_dir: Path,
    subject_id: int,
    particle_count: int,
    rollout_count: int,
) -> Path:
    return (
        output_dir
        / "cache"
        / f"subject_{int(subject_id)}"
        / f"particles_{int(particle_count)}"
        / f"rollouts_{int(rollout_count)}.npz"
    )


def simulate_subject(
    *,
    args: argparse.Namespace,
    subject_frame: pd.DataFrame,
    engine_config: Mapping[str, Any],
    dataset_paths: Mapping[str, Path],
) -> Path:
    subject_id = int(subject_frame["iSub"].iloc[0])
    output = cache_path(
        args.output_dir,
        subject_id,
        args.particle_count,
        args.rollout_count,
    )
    if output.exists() and not args.force:
        return output
    split_index, split_status = split_for_subject(
        subject_frame,
        mode=str(args.split_mode),
        window=int(args.window),
    )
    stimulus = subject_frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
    categories = subject_frame["category"].to_numpy(dtype=int)
    choices = subject_frame["choice"].to_numpy(dtype=int)
    feedback = subject_frame["feedback"].to_numpy(dtype=float)
    filter_seed = stable_seed(
        {
            "seed_role": "b0_trajectory_ppc_filter",
            "base_seed": int(args.base_seed),
            "subject_id": subject_id,
            "split_index": split_index,
            "split_mode": str(args.split_mode),
            "particle_count": int(args.particle_count),
        }
    )
    rollout_seed = stable_seed(
        {
            "seed_role": "b0_trajectory_ppc_rollout",
            "base_seed": int(args.base_seed),
            "subject_id": subject_id,
            "split_index": split_index,
            "rollout_count": int(args.rollout_count),
        }
    )
    selected_engine = deepcopy(dict(engine_config))
    selected_beta_additive = float(args.beta_correct_additive)
    selected_lapse_start = float(args.lapse_start)
    selected_update_probability = float(
        args.learning_update_probability
    )
    parameter_selection_scores: list[dict[str, float]] = []
    if (
        args.beta_additive_grid
        or args.lapse_start_grid
        or args.learning_update_grid
    ):
        selection_seed = stable_seed(
            {
                "seed_role": "b0_trajectory_ppc_parameter_selection",
                "base_seed": int(args.base_seed),
                "subject_id": subject_id,
                "split_index": split_index,
                "split_mode": str(args.split_mode),
                "selection_particle_count": int(
                    args.selection_particle_count
                ),
            }
        )
        observed_indices = choices[:split_index] - 1
        score_start = 1 if split_index > 1 else 0
        beta_candidates = (
            sorted({float(value) for value in args.beta_additive_grid})
            if args.beta_additive_grid
            else [float(args.beta_correct_additive)]
        )
        lapse_candidates = (
            sorted({float(value) for value in args.lapse_start_grid})
            if args.lapse_start_grid
            else [float(args.lapse_start)]
        )
        update_candidates = (
            sorted(
                {float(value) for value in args.learning_update_grid}
            )
            if args.learning_update_grid
            else [float(args.learning_update_probability)]
        )
        for beta_additive, lapse_start, update_probability in product(
            beta_candidates,
            lapse_candidates,
            update_candidates,
        ):
            candidate_engine = apply_fixed_hyperparams_to_engine_config(
                deepcopy(dict(engine_config)),
                {
                    (
                        "engine.modules.beta_mod.kwargs."
                        "correct_additive"
                    ): float(beta_additive)
                },
            )
            candidate = run_state_model_particle_filter(
                engine_config=candidate_engine,
                subject_id=subject_id,
                stimulus=stimulus[:split_index],
                choices=choices[:split_index],
                feedback=feedback[:split_index],
                particle_count=int(args.selection_particle_count),
                choice_readout_power=float(args.rho),
                output_lapse_schedule=decaying_lapse_schedule(
                    split_index,
                    initial_lapse=float(lapse_start),
                    half_life=float(args.lapse_half_life),
                ),
                learning_update_probability=float(update_probability),
                filter_seed=int(selection_seed),
                resample_threshold_fraction=float(
                    args.resample_threshold
                ),
                processed_data_dir=dataset_paths["processed_dir"],
                dataset_paths=dataset_paths,
            )
            probabilities = candidate.marginal_probabilities[
                np.arange(split_index),
                observed_indices,
            ]
            scored = probabilities[score_start:]
            parameter_selection_scores.append(
                {
                    "beta_correct_additive": float(beta_additive),
                    "initial_lapse": float(lapse_start),
                    "learning_update_probability": float(
                        update_probability
                    ),
                    "prefix_choice_brier": float(
                        np.mean(2.0 * np.square(1.0 - scored))
                    ),
                    "prefix_choice_nll": float(
                        -np.mean(np.log(np.clip(scored, 1e-12, 1.0)))
                    ),
                }
            )
        selected = min(
            parameter_selection_scores,
            key=lambda row: (
                row["prefix_choice_brier"],
                row["beta_correct_additive"],
                row["initial_lapse"],
                row["learning_update_probability"],
            ),
        )
        selected_beta_additive = float(
            selected["beta_correct_additive"]
        )
        selected_lapse_start = float(selected["initial_lapse"])
        selected_update_probability = float(
            selected["learning_update_probability"]
        )
        selected_engine = apply_fixed_hyperparams_to_engine_config(
            deepcopy(dict(engine_config)),
            {
                (
                    "engine.modules.beta_mod.kwargs.correct_additive"
                ): selected_beta_additive
            },
        )
    lapse_schedule = decaying_lapse_schedule(
        len(subject_frame),
        initial_lapse=selected_lapse_start,
        half_life=float(args.lapse_half_life),
    )
    dynamic_rho_start = getattr(args, "dynamic_rho_start", None)
    dynamic_rho_config = (
        None
        if dynamic_rho_start is None
        else DynamicRhoConfig(
            start=float(dynamic_rho_start),
            end=float(args.dynamic_rho_end),
            volatility=float(args.dynamic_rho_volatility),
            persistence=float(args.dynamic_rho_persistence),
            start_log_sd=float(args.dynamic_rho_start_log_sd),
            gain_log_sd=float(args.dynamic_rho_gain_log_sd),
            volatility_log_sd=float(
                args.dynamic_rho_volatility_log_sd
            ),
            trend_reference_trials=int(
                args.dynamic_rho_reference_trials
            ),
        )
    )

    result = run_conditioned_condition1_rollouts(
        engine_config=selected_engine,
        subject_id=subject_id,
        stimulus=stimulus,
        categories=categories,
        observed_prefix_choices=choices[:split_index],
        observed_prefix_feedback=feedback[:split_index],
        particle_count=int(args.particle_count),
        rollout_count=int(args.rollout_count),
        rho=float(args.rho),
        epsilon_schedule=lapse_schedule,
        learning_update_probability=selected_update_probability,
        acquisition_hazard=(
            None
            if args.acquisition_half_life is None
            else acquisition_hazard_from_half_life(
                float(args.acquisition_half_life)
            )
        ),
        pre_acquisition_lapse=float(args.pre_acquisition_lapse),
        dynamic_rho=dynamic_rho_config,
        filter_seed=int(filter_seed),
        rollout_seed=int(rollout_seed),
        resample_threshold_fraction=float(args.resample_threshold),
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        choices=result.choices,
        feedback=result.feedback,
        probabilities=result.probabilities,
        ancestor_indices=result.ancestor_indices,
        boundary_weights=result.boundary_weights,
        prefix_pre_choice_ess=result.prefix_pre_choice_ess,
        prefix_post_choice_ess=result.prefix_post_choice_ess,
        prefix_resampled=result.prefix_resampled,
        prefix_choice_probabilities=result.prefix_choice_probabilities,
        prefix_observed_choice_probability=(
            result.prefix_observed_choice_probability
        ),
        prefix_log_predictive_density=np.asarray(
            result.prefix_log_predictive_density,
            dtype=float,
        ),
        prefix_acquired_probability=result.prefix_acquired_probability,
        boundary_acquired=result.boundary_acquired,
        generated_acquired=result.generated_acquired,
        prefix_rho_posterior_mean=(
            result.prefix_rho_posterior_mean
        ),
        boundary_rho=result.boundary_rho,
        boundary_rho_start=result.boundary_rho_start,
        boundary_rho_gain=result.boundary_rho_gain,
        boundary_rho_volatility=(
            result.boundary_rho_volatility
        ),
        generated_rho=result.generated_rho,
        observed_prefix_choices=choices[:split_index],
        observed_prefix_feedback=feedback[:split_index],
        observed_test_choices=choices[split_index:],
        observed_test_feedback=feedback[split_index:],
        test_iTrial=subject_frame["iTrial"].to_numpy(dtype=int)[split_index:],
        test_iSession=subject_frame["iSession"].to_numpy(dtype=int)[
            split_index:
        ],
        test_iBlock=subject_frame["iBlock"].to_numpy(dtype=int)[split_index:],
        metadata=np.asarray(
            json.dumps(
                {
                    "subject_id": subject_id,
                    "cohort": (
                        "development"
                        if subject_id in DEVELOPMENT_SUBJECTS
                        else "reserved_application"
                    ),
                    "split_index": split_index,
                    "split_status": split_status,
                    "split_mode": str(args.split_mode),
                    "train_n": split_index,
                    "test_n": len(subject_frame) - split_index,
                    "particle_count": int(args.particle_count),
                    "rollout_count": int(args.rollout_count),
                    "filter_seed": int(filter_seed),
                    "rollout_seed": int(rollout_seed),
                    "selected_beta_correct_additive": (
                        selected_beta_additive
                    ),
                    "selected_initial_lapse": selected_lapse_start,
                    "selected_learning_update_probability": (
                        selected_update_probability
                    ),
                    "parameter_selection_scores": (
                        parameter_selection_scores
                    ),
                    "acquisition_half_life": (
                        None
                        if args.acquisition_half_life is None
                        else float(args.acquisition_half_life)
                    ),
                    "acquisition_hazard": (
                        None
                        if args.acquisition_half_life is None
                        else acquisition_hazard_from_half_life(
                            float(args.acquisition_half_life)
                        )
                    ),
                    "pre_acquisition_lapse": float(
                        args.pre_acquisition_lapse
                    ),
                    "dynamic_rho": (
                        None
                        if dynamic_rho_config is None
                        else asdict(dynamic_rho_config)
                    ),
                },
                ensure_ascii=False,
            )
        ),
    )
    return output


def load_subject_cache(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        return {
            **json.loads(str(payload["metadata"].item())),
            **{
                key: np.asarray(payload[key])
                for key in payload.files
                if key != "metadata"
            },
        }


def evaluate_subject(
    cache: Mapping[str, Any],
    *,
    window: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    np.ndarray,
]:
    subject_id = int(cache["subject_id"])
    actual_feedback = np.asarray(
        cache["observed_test_feedback"], dtype=float
    )
    actual_choices = np.asarray(cache["observed_test_choices"], dtype=int)
    simulated_feedback = np.asarray(cache["feedback"], dtype=float)
    simulated_choices = np.asarray(cache["choices"], dtype=int)
    n_rollouts, test_n = simulated_feedback.shape
    if "test_iSession" in cache and "test_iBlock" in cache:
        block_ids = np.asarray(
            [
                f"{int(session)}:{int(block)}"
                for session, block in zip(
                    cache["test_iSession"],
                    cache["test_iBlock"],
                )
            ]
        )
    else:
        block_ids = np.zeros(test_n, dtype=int)

    actual_metrics = trajectory_metrics(
        actual_feedback,
        actual_choices,
        window=int(window),
        block_ids=block_ids,
    )
    simulation_rows = [
        trajectory_metrics(
            simulated_feedback[index],
            simulated_choices[index],
            window=int(window),
            block_ids=block_ids,
        )
        for index in range(n_rollouts)
    ]
    metric_names = [
        spec.metric
        for spec in METRIC_SPECS
        if np.isfinite(actual_metrics[spec.metric])
        and np.mean(
            [
                np.isfinite(row[spec.metric])
                for row in simulation_rows
            ]
        )
        >= 0.95
    ]
    specs = {
        spec.metric: spec for spec in METRIC_SPECS
    }
    simulation_matrix = np.asarray(
        [
            [row[metric] for metric in metric_names]
            for row in simulation_rows
        ],
        dtype=float,
    )
    for column in range(simulation_matrix.shape[1]):
        values = simulation_matrix[:, column]
        finite = np.isfinite(values)
        if not np.all(finite):
            values[~finite] = float(np.nanmedian(values))
            simulation_matrix[:, column] = values
    actual_vector = np.asarray(
        [actual_metrics[metric] for metric in metric_names],
        dtype=float,
    )
    resolutions = np.asarray(
        [
            metric_resolution(
                specs[metric],
                n_trials=test_n,
                window=int(window),
            )
            for metric in metric_names
        ],
        dtype=float,
    )
    median, scale = robust_scales(simulation_matrix, resolutions)
    z_actual = (actual_vector - median) / scale
    z_simulation = (simulation_matrix - median[None, :]) / scale[None, :]
    summary_discrepancy = float(np.max(np.abs(z_actual)))
    summary_sim_discrepancy = np.max(np.abs(z_simulation), axis=1)
    summary_p = float(
        (1.0 + np.sum(summary_sim_discrepancy >= summary_discrepancy - 1e-12))
        / (1.0 + n_rollouts)
    )
    summary_threshold = float(
        np.quantile(summary_sim_discrepancy, 0.95)
    )

    metric_rows: list[dict[str, Any]] = []
    for index, metric in enumerate(metric_names):
        simulations = simulation_matrix[:, index]
        observed = actual_vector[index]
        percentile = float(
            (
                1.0
                + np.sum(simulations < observed)
                + 0.5 * np.sum(np.isclose(simulations, observed))
            )
            / (1.0 + n_rollouts)
        )
        q025, q05, q50, q95, q975 = np.quantile(
            simulations,
            [0.025, 0.05, 0.5, 0.95, 0.975],
        )
        metric_rows.append(
            {
                "iSub": subject_id,
                "cohort": cache["cohort"],
                "metric": metric,
                "metric_label": specs[metric].label,
                "observed": float(observed),
                "sim_mean": float(np.mean(simulations)),
                "sim_median": float(q50),
                "sim_q025": float(q025),
                "sim_q05": float(q05),
                "sim_q95": float(q95),
                "sim_q975": float(q975),
                "observed_percentile": percentile,
                "inside_marginal_95": bool(q025 <= observed <= q975),
                "robust_z": float(z_actual[index]),
                "absolute_robust_z": float(abs(z_actual[index])),
            }
        )

    actual_curve, curve_end_indices = block_bounded_rolling_accuracy(
        actual_feedback,
        block_ids,
        int(window),
    )
    simulated_curve, simulated_end_indices = (
        block_bounded_rolling_accuracy(
            simulated_feedback,
            block_ids,
            int(window),
        )
    )
    if not np.array_equal(curve_end_indices, simulated_end_indices):
        raise RuntimeError("Observed and simulated curve positions do not align.")
    curve_resolution = np.full(actual_curve.size, 1.0 / int(window))
    curve_median, curve_scale = robust_scales(
        simulated_curve,
        curve_resolution,
    )
    curve_z_actual = (actual_curve - curve_median) / curve_scale
    curve_z_sim = (
        simulated_curve - curve_median[None, :]
    ) / curve_scale[None, :]
    curve_discrepancy = float(np.max(np.abs(curve_z_actual)))
    curve_sim_discrepancy = np.max(np.abs(curve_z_sim), axis=1)
    curve_p = float(
        (1.0 + np.sum(curve_sim_discrepancy >= curve_discrepancy - 1e-12))
        / (1.0 + n_rollouts)
    )
    curve_threshold = float(np.quantile(curve_sim_discrepancy, 0.95))
    simultaneous_lower = np.clip(
        curve_median - curve_threshold * curve_scale,
        0.0,
        1.0,
    )
    simultaneous_upper = np.clip(
        curve_median + curve_threshold * curve_scale,
        0.0,
        1.0,
    )
    pointwise_q025 = np.quantile(simulated_curve, 0.025, axis=0)
    pointwise_q05 = np.quantile(simulated_curve, 0.05, axis=0)
    pointwise_q95 = np.quantile(simulated_curve, 0.95, axis=0)
    pointwise_q975 = np.quantile(simulated_curve, 0.975, axis=0)
    curve_rows = [
        {
            "iSub": subject_id,
            "cohort": cache["cohort"],
            "test_position": int(curve_end_indices[index] + 1),
            "iTrial": int(cache["test_iTrial"][curve_end_indices[index]]),
            "observed_rolling_accuracy": float(actual_curve[index]),
            "sim_median": float(curve_median[index]),
            "pointwise_q025": float(pointwise_q025[index]),
            "pointwise_q05": float(pointwise_q05[index]),
            "pointwise_q95": float(pointwise_q95[index]),
            "pointwise_q975": float(pointwise_q975[index]),
            "simultaneous_lower_95": float(simultaneous_lower[index]),
            "simultaneous_upper_95": float(simultaneous_upper[index]),
        }
        for index in range(actual_curve.size)
    ]

    normalized_summary = summary_discrepancy / max(
        summary_threshold, 1e-12
    )
    normalized_curve = curve_discrepancy / max(
        curve_threshold, 1e-12
    )
    combined_discrepancy = max(normalized_summary, normalized_curve)
    combined_sim_discrepancy = np.maximum(
        summary_sim_discrepancy / max(summary_threshold, 1e-12),
        curve_sim_discrepancy / max(curve_threshold, 1e-12),
    )
    combined_p = float(
        (
            1.0
            + np.sum(
                combined_sim_discrepancy
                >= combined_discrepancy - 1e-12
            )
        )
        / (1.0 + n_rollouts)
    )
    combined_threshold = float(
        np.quantile(combined_sim_discrepancy, 0.95)
    )
    combined_sim_pass = (
        combined_sim_discrepancy <= combined_threshold + 1e-12
    )
    prefix_post_ess = np.asarray(
        cache["prefix_post_choice_ess"], dtype=float
    )
    boundary_weights = np.asarray(
        cache["boundary_weights"], dtype=float
    )
    boundary_weights = boundary_weights / np.sum(boundary_weights)
    boundary_rho = np.asarray(
        cache.get(
            "boundary_rho",
            np.full(int(cache["particle_count"]), 2.0),
        ),
        dtype=float,
    )
    boundary_rho_volatility = np.asarray(
        cache.get(
            "boundary_rho_volatility",
            np.zeros(int(cache["particle_count"])),
        ),
        dtype=float,
    )
    generated_rho = np.asarray(
        cache.get(
            "generated_rho",
            np.full_like(cache["feedback"], 2.0, dtype=float),
        ),
        dtype=float,
    )
    row = {
        "iSub": subject_id,
        "cohort": cache["cohort"],
        "split_status": cache["split_status"],
        "train_n": int(cache["train_n"]),
        "test_n": int(cache["test_n"]),
        "particle_count": int(cache["particle_count"]),
        "rollout_count": int(cache["rollout_count"]),
        "metric_n": len(metric_names),
        "summary_discrepancy": summary_discrepancy,
        "summary_threshold_95": summary_threshold,
        "summary_calibration_p": summary_p,
        "summary_pass_95": bool(
            summary_discrepancy <= summary_threshold + 1e-12
        ),
        "curve_discrepancy": curve_discrepancy,
        "curve_threshold_95": curve_threshold,
        "curve_calibration_p": curve_p,
        "curve_pass_95": bool(
            curve_discrepancy <= curve_threshold + 1e-12
        ),
        "combined_discrepancy": float(combined_discrepancy),
        "combined_threshold_95": combined_threshold,
        "combined_calibration_p": combined_p,
        "combined_pass_95": bool(
            combined_discrepancy <= combined_threshold + 1e-12
        ),
        "marginal_metric_coverage_95": float(
            np.mean([row["inside_marginal_95"] for row in metric_rows])
        ),
        "curve_pointwise_coverage_95": float(
            np.mean(
                (actual_curve >= pointwise_q025)
                & (actual_curve <= pointwise_q975)
            )
        ),
        "curve_simultaneous_coverage_95": bool(
            np.all(
                (actual_curve >= simultaneous_lower - 1e-12)
                & (actual_curve <= simultaneous_upper + 1e-12)
            )
        ),
        "curve_crps": empirical_crps(actual_curve, simulated_curve),
        "curve_pointwise_interval_width_90": float(
            np.median(pointwise_q95 - pointwise_q05)
        ),
        "curve_pointwise_interval_width_95": float(
            np.median(pointwise_q975 - pointwise_q025)
        ),
        "observed_test_accuracy": float(np.mean(actual_feedback)),
        "simulated_test_accuracy_mean": float(
            np.mean(simulated_feedback)
        ),
        "simulated_test_accuracy_q025": float(
            np.quantile(np.mean(simulated_feedback, axis=1), 0.025)
        ),
        "simulated_test_accuracy_q975": float(
            np.quantile(np.mean(simulated_feedback, axis=1), 0.975)
        ),
        "boundary_ess": float(effective_ess(boundary_weights)),
        "min_prefix_post_ess_fraction": float(
            np.min(prefix_post_ess) / int(cache["particle_count"])
        ),
        "prefix_resampling_fraction": float(
            np.mean(np.asarray(cache["prefix_resampled"], dtype=bool))
        ),
        "selected_beta_correct_additive": float(
            cache.get("selected_beta_correct_additive", np.nan)
        ),
        "selected_initial_lapse": float(
            cache.get("selected_initial_lapse", 0.0)
        ),
        "selected_learning_update_probability": float(
            cache.get("selected_learning_update_probability", 1.0)
        ),
        "acquisition_half_life": float(
            cache.get("acquisition_half_life")
            if cache.get("acquisition_half_life") is not None
            else np.nan
        ),
        "boundary_acquired_probability": float(
            np.sum(
                np.asarray(cache.get("boundary_weights"), dtype=float)
                * np.asarray(
                    cache.get(
                        "boundary_acquired",
                        np.ones(int(cache["particle_count"]), dtype=bool),
                    ),
                    dtype=float,
                )
            )
        ),
        "suffix_acquired_fraction_mean": float(
            np.mean(
                np.asarray(
                    cache.get(
                        "generated_acquired",
                        np.ones_like(cache["feedback"], dtype=bool),
                    ),
                    dtype=float,
                )
            )
        ),
        "boundary_rho_posterior_mean": float(
            np.sum(boundary_weights * boundary_rho)
        ),
        "boundary_rho_volatility_posterior_mean": float(
            np.sum(boundary_weights * boundary_rho_volatility)
        ),
        "suffix_rho_mean": float(np.mean(generated_rho)),
        "suffix_rho_within_trajectory_sd_mean": float(
            np.mean(np.std(generated_rho, axis=1))
        ),
    }
    return row, metric_rows, curve_rows, combined_sim_pass


def effective_ess(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    values = values / np.sum(values)
    return float(1.0 / np.sum(np.square(values)))


def cohort_calibration(
    subjects: pd.DataFrame,
    simulated_pass: Mapping[int, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [
        ("development", subjects["cohort"].eq("development")),
        (
            "reserved_application",
            subjects["cohort"].eq("reserved_application"),
        ),
        ("all_subjects", np.ones(len(subjects), dtype=bool)),
    ]
    for label, mask in groups:
        selected = subjects.loc[mask]
        if selected.empty:
            continue
        matrix = np.vstack(
            [
                simulated_pass[int(subject)]
                for subject in selected["iSub"]
            ]
        )
        null_counts = np.sum(matrix, axis=0)
        observed = int(selected["combined_pass_95"].sum())
        rows.append(
            {
                "cohort": label,
                "subject_n": int(len(selected)),
                "observed_pass_n": observed,
                "observed_pass_fraction": observed / max(1, len(selected)),
                "b0_self_expected_pass_mean": float(
                    np.mean(null_counts)
                ),
                "b0_self_expected_pass_q025": float(
                    np.quantile(null_counts, 0.025)
                ),
                "b0_self_expected_pass_q975": float(
                    np.quantile(null_counts, 0.975)
                ),
                "lower_tail_calibration_p": float(
                    (1.0 + np.sum(null_counts <= observed))
                    / (1.0 + null_counts.size)
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.particle_count < 2 or args.rollout_count < 20:
        raise ValueError("Require at least 2 particles and 20 rollouts.")
    if args.n_jobs <= 0 or args.window < 4:
        raise ValueError("n_jobs must be positive and window at least 4.")
    if not 1 <= int(args.capacity) <= 38:
        raise ValueError("capacity must lie between 1 and 38.")
    if args.beta_init <= 0.0 or args.beta_correct_additive < 0.0:
        raise ValueError("beta_init must be positive and beta increment non-negative.")
    if not 0.0 <= args.beta_decrease_rate <= 1.0:
        raise ValueError("beta_decrease_rate must lie in [0, 1].")
    if args.selection_particle_count < 2:
        raise ValueError("selection_particle_count must be at least 2.")
    if args.beta_additive_grid and any(
        value < 0.0 for value in args.beta_additive_grid
    ):
        raise ValueError("All beta additive candidates must be non-negative.")
    if not 0.0 <= args.lapse_start <= 1.0:
        raise ValueError("lapse_start must lie in [0, 1].")
    if args.lapse_start_grid and any(
        value < 0.0 or value > 1.0
        for value in args.lapse_start_grid
    ):
        raise ValueError("All lapse start candidates must lie in [0, 1].")
    if not np.isfinite(args.lapse_half_life) or args.lapse_half_life <= 0.0:
        raise ValueError("lapse_half_life must be positive and finite.")
    if not 0.0 <= args.learning_update_probability <= 1.0:
        raise ValueError(
            "learning_update_probability must lie in [0, 1]."
        )
    if args.learning_update_grid and any(
        value < 0.0 or value > 1.0
        for value in args.learning_update_grid
    ):
        raise ValueError(
            "All learning update candidates must lie in [0, 1]."
        )
    if args.acquisition_half_life is not None:
        acquisition_hazard_from_half_life(args.acquisition_half_life)
    if not 0.0 <= args.pre_acquisition_lapse <= 1.0:
        raise ValueError("pre-acquisition-lapse must lie in [0, 1].")
    if args.dynamic_rho_start is None:
        if args.dynamic_rho_end is not None:
            raise ValueError(
                "dynamic-rho-end requires dynamic-rho-start."
            )
    else:
        if args.dynamic_rho_end is None:
            raise ValueError(
                "dynamic-rho-start requires dynamic-rho-end."
            )
        if args.acquisition_half_life is not None:
            raise ValueError(
                "dynamic rho and acquisition change-point are mutually "
                "exclusive."
            )
        DynamicRhoConfig(
            start=float(args.dynamic_rho_start),
            end=float(args.dynamic_rho_end),
            volatility=float(args.dynamic_rho_volatility),
            persistence=float(args.dynamic_rho_persistence),
            start_log_sd=float(args.dynamic_rho_start_log_sd),
            gain_log_sd=float(args.dynamic_rho_gain_log_sd),
            volatility_log_sd=float(
                args.dynamic_rho_volatility_log_sd
            ),
            trend_reference_trials=int(
                args.dynamic_rho_reference_trials
            ),
        )
        if (
            args.dynamic_rho_start <= 0.0
            or args.dynamic_rho_end < args.dynamic_rho_start
            or args.dynamic_rho_volatility < 0.0
            or not 0.0 <= args.dynamic_rho_persistence < 1.0
            or min(
                args.dynamic_rho_start_log_sd,
                args.dynamic_rho_gain_log_sd,
                args.dynamic_rho_volatility_log_sd,
            )
            < 0.0
            or args.dynamic_rho_reference_trials < 2
        ):
            raise ValueError("Invalid dynamic-rho specification.")

    data = pd.read_csv(args.data)
    data = (
        data.loc[data["condition"].eq(1)]
        .sort_values(list(KEY_COLUMNS))
        .reset_index(drop=True)
    )
    subjects = (
        sorted(int(value) for value in data["iSub"].unique())
        if args.subjects is None
        else sorted({int(value) for value in args.subjects})
    )
    data = data.loc[data["iSub"].isin(subjects)].copy()
    missing = sorted(set(subjects) - set(data["iSub"].astype(int)))
    if missing:
        raise ValueError(f"Subjects absent from condition 1: {missing}")

    model_path = ROOT / "configs/model_struct/pmh_model_cond1_active_set.yaml"
    simulation_path = (
        ROOT / "configs/simulation_cfg/pmh_cond1_simulation_v14.yaml"
    )
    base_engine = load_yaml(model_path)
    simulation_config = load_yaml(simulation_path)
    dataset_paths = resolve_dataset_paths(
        simulation_config,
        simulation_path.parent,
        DEFAULT_DATA_PATH,
    )
    parameters = {
        "engine.modules.hypo_transitions_mod.kwargs.theta": 0.0,
        "engine.modules.hypo_transitions_mod.kwargs.capacity": int(
            args.capacity
        ),
        "engine.modules.memory_mod.kwargs.gamma": float(args.gamma),
        "engine.modules.memory_mod.kwargs.w0": float(args.w0),
        "engine.modules.beta_mod.kwargs.beta_init": float(args.beta_init),
        "engine.modules.beta_mod.kwargs.correct_additive": float(
            args.beta_correct_additive
        ),
        "engine.modules.beta_mod.kwargs.decrease_rate": float(
            args.beta_decrease_rate
        ),
        "engine.choice_readout.kwargs": {
            "method": "sharpened_expectation",
            "power": float(args.rho),
        },
        "engine.output_noise.kwargs.base_lapse": 0.0,
    }
    engine = apply_fixed_hyperparams_to_engine_config(
        deepcopy(base_engine),
        parameters,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    subject_frames = [
        group.copy()
        for _, group in data.groupby("iSub", sort=True)
    ]
    cache_paths = Parallel(
        n_jobs=min(int(args.n_jobs), len(subject_frames)),
        verbose=10,
    )(
        delayed(simulate_subject)(
            args=args,
            subject_frame=subject_frame,
            engine_config=engine,
            dataset_paths=dataset_paths,
        )
        for subject_frame in subject_frames
    )

    subject_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    simulated_pass: dict[int, np.ndarray] = {}
    for path in cache_paths:
        cache = load_subject_cache(path)
        subject_row, local_metrics, local_curve, local_sim_pass = (
            evaluate_subject(cache, window=int(args.window))
        )
        subject_rows.append(subject_row)
        metric_rows.extend(local_metrics)
        curve_rows.extend(local_curve)
        simulated_pass[int(subject_row["iSub"])] = local_sim_pass

    subject_summary = pd.DataFrame(subject_rows).sort_values("iSub")
    subject_summary["combined_calibration_fdr_q"] = benjamini_hochberg(
        subject_summary["combined_calibration_p"].to_numpy(dtype=float)
    )
    metric_summary = pd.DataFrame(metric_rows)
    curve_summary = pd.DataFrame(curve_rows)
    cohort_summary = cohort_calibration(subject_summary, simulated_pass)
    parameter_selection_rows: list[dict[str, Any]] = []
    for path in cache_paths:
        cache = load_subject_cache(path)
        scores = cache.get(
            "parameter_selection_scores",
            cache.get("beta_selection_scores", []),
        )
        for score in scores:
            parameter_selection_rows.append(
                {
                    "iSub": int(cache["subject_id"]),
                    "cohort": cache["cohort"],
                    "selected": bool(
                        np.isclose(
                            float(score["beta_correct_additive"]),
                            float(
                                cache[
                                    "selected_beta_correct_additive"
                                ]
                            ),
                        )
                        and np.isclose(
                            float(score.get("initial_lapse", 0.0)),
                            float(cache.get("selected_initial_lapse", 0.0)),
                        )
                        and np.isclose(
                            float(
                                score.get(
                                    "learning_update_probability", 1.0
                                )
                            ),
                            float(
                                cache.get(
                                    "selected_learning_update_probability",
                                    1.0,
                                )
                            ),
                        )
                    ),
                    **score,
                }
            )
    parameter_selection = pd.DataFrame(parameter_selection_rows)

    metric_failure = (
        metric_summary.loc[~metric_summary["inside_marginal_95"]]
        .groupby(["metric", "metric_label"], as_index=False)
        .agg(
            failed_subject_n=("iSub", "nunique"),
            failed_observation_n=("iSub", "size"),
        )
        .sort_values(
            ["failed_subject_n", "metric"],
            ascending=[False, True],
        )
    )
    if metric_failure.empty:
        metric_failure = pd.DataFrame(
            columns=[
                "metric",
                "metric_label",
                "failed_subject_n",
                "failed_observation_n",
            ]
        )

    all_cohort = cohort_summary.loc[
        cohort_summary["cohort"].eq("all_subjects")
    ].iloc[0]
    reserved_rows = cohort_summary.loc[
        cohort_summary["cohort"].eq("reserved_application")
    ]
    cohort_gate = bool(
        all_cohort["lower_tail_calibration_p"] >= 0.05
        and (
            reserved_rows.empty
            or reserved_rows.iloc[0]["lower_tail_calibration_p"] >= 0.05
        )
    )
    fdr_failure_n = int(
        np.sum(subject_summary["combined_calibration_fdr_q"] <= 0.05)
    )
    if cohort_gate:
        if args.dynamic_rho_start is not None:
            dynamic_label = (
                "C0"
                if np.isclose(args.dynamic_rho_volatility, 0.0)
                else "C1"
            )
            recommended_action = (
                f"retain_dynamic_continuous_rho_{dynamic_label}"
            )
            reason = (
                "The frozen continuous readout-concentration process "
                "produces a joint trajectory pass count consistent with "
                "its own repeated-sampling calibration."
            )
        elif args.acquisition_half_life is not None:
            recommended_action = "retain_single_acquisition_changepoint"
            reason = (
                "The frozen one-way acquisition candidate produces a joint "
                "trajectory pass count consistent with its own repeated-"
                "sampling calibration in the available cohort."
            )
        elif int(args.capacity) == 5:
            recommended_action = "retain_finite_capacity_B0"
            reason = (
                "The number of subjects inside the joint 95% trajectory "
                "region is consistent with B0's own repeated-sampling "
                "calibration in both the reserved and full cohorts."
            )
        else:
            recommended_action = "retain_static_fullset_boundary"
            reason = (
                "The number of subjects inside the joint 95% trajectory "
                "region is consistent with B0's own repeated-sampling "
                "calibration in both the reserved and full cohorts."
            )
        recommendation = {
            "generative_adequacy": "adequate_at_cohort_level",
            "recommended_model_action": recommended_action,
            "reason": reason,
            "subject_level_fdr_failures": fdr_failure_n,
        }
    else:
        leading = (
            metric_failure.iloc[0]["metric"]
            if not metric_failure.empty
            else "joint_trajectory_shape"
        )
        recommendation = {
            "generative_adequacy": "systematic_coverage_failure",
            "recommended_model_action": (
                "diagnose_one_minimal_extension"
            ),
            "reason": (
                "Observed joint trajectory coverage falls below B0's own "
                "95% repeated-sampling calibration."
            ),
            "leading_failed_metric": str(leading),
            "subject_level_fdr_failures": fdr_failure_n,
        }

    manifest = {
        "analysis": "condition1_frozen_b0_conditioned_trajectory_ppc",
        "primary_target": (
            "Generative adequacy of one stochastic realization, not recovery "
            "of a unique latent path."
        ),
        "data": str(args.data),
        "subjects": subjects,
        "development_subjects": [
            subject for subject in subjects if subject in DEVELOPMENT_SUBJECTS
        ],
        "reserved_application_subjects": [
            subject for subject in subjects if subject not in DEVELOPMENT_SUBJECTS
        ],
        "fixed_physical_stimulus_and_category_schedule": True,
        "future_observed_choices_read_by_rollout": False,
        "future_feedback_generated_from_simulated_choice": True,
        "split_rule": (
            (
                "last block when at least two blocks; otherwise last half of "
                "the single block"
            )
            if args.split_mode == "last_block"
            else (
                "all trials after the first block; otherwise all trials after "
                "the first quarter of the single block"
            )
        ),
        "split_mode": str(args.split_mode),
        "model": (
            (
                "B0_fullset_dynamic_continuous_rho_C0"
                if np.isclose(args.dynamic_rho_volatility, 0.0)
                else "B0_fullset_dynamic_continuous_rho_C1"
            )
            if args.dynamic_rho_start is not None
            else (
                "B0_fullset_single_acquisition_changepoint"
                if args.acquisition_half_life is not None
                else ("B0" if int(args.capacity) == 5 else "B0_fullset")
            )
        ),
        "capacity": int(args.capacity),
        "theta": 0.0,
        "epsilon": 0.0,
        "lapse_schedule": (
            "epsilon_s(t) = epsilon_s0 * 0.5 ** (trial_index / half_life)"
        ),
        "lapse_start": float(args.lapse_start),
        "lapse_start_grid": (
            None
            if not args.lapse_start_grid
            else sorted({float(value) for value in args.lapse_start_grid})
        ),
        "lapse_half_life": float(args.lapse_half_life),
        "learning_update_probability": float(
            args.learning_update_probability
        ),
        "learning_update_grid": (
            None
            if not args.learning_update_grid
            else sorted(
                {float(value) for value in args.learning_update_grid}
            )
        ),
        "learning_update_mechanism": (
            "At each trial, likelihood/memory/beta feedback updating occurs "
            "with the selected Bernoulli probability; a skipped update leaves "
            "the latent belief unchanged."
        ),
        "acquisition_half_life": (
            None
            if args.acquisition_half_life is None
            else float(args.acquisition_half_life)
        ),
        "acquisition_hazard": (
            None
            if args.acquisition_half_life is None
            else acquisition_hazard_from_half_life(
                float(args.acquisition_half_life)
            )
        ),
        "pre_acquisition_lapse": float(args.pre_acquisition_lapse),
        "acquisition_mechanism": (
            "Optional one-way novice-to-acquired readout boundary. Before "
            "the boundary the ordinary full-set readout is mixed with a "
            "fixed uninformed-response probability; after it that extra "
            "lapse is zero. Latent evidence updating continues before and "
            "after the boundary."
        ),
        "dynamic_rho": (
            None
            if args.dynamic_rho_start is None
            else asdict(
                DynamicRhoConfig(
                    start=float(args.dynamic_rho_start),
                    end=float(args.dynamic_rho_end),
                    volatility=float(args.dynamic_rho_volatility),
                    persistence=float(args.dynamic_rho_persistence),
                    start_log_sd=float(
                        args.dynamic_rho_start_log_sd
                    ),
                    gain_log_sd=float(
                        args.dynamic_rho_gain_log_sd
                    ),
                    volatility_log_sd=float(
                        args.dynamic_rho_volatility_log_sd
                    ),
                    trend_reference_trials=int(
                        args.dynamic_rho_reference_trials
                    ),
                )
            )
        ),
        "dynamic_rho_mechanism": (
            "rho_t = exp(a_s + d_s * trial/reference_trial + u_t), "
            "u_t = phi * u_(t-1) + sigma_s * innovation. "
            "Particle-level start, gain, and volatility are drawn from "
            "shared population distributions and conditioned only on the "
            "observed prefix."
        ),
        "gamma": float(args.gamma),
        "w0": float(args.w0),
        "rho": float(args.rho),
        "beta_init": float(args.beta_init),
        "beta_correct_additive": float(args.beta_correct_additive),
        "beta_decrease_rate": float(args.beta_decrease_rate),
        "beta_additive_grid": (
            None
            if not args.beta_additive_grid
            else sorted({float(value) for value in args.beta_additive_grid})
        ),
        "beta_selection": (
            "none"
            if (
                not args.beta_additive_grid
                and not args.lapse_start_grid
                and not args.learning_update_grid
            )
            else "minimum causal choice Brier on observed prediction prefix"
        ),
        "selection_particle_count": int(args.selection_particle_count),
        "particle_count": int(args.particle_count),
        "rollout_count": int(args.rollout_count),
        "window": int(args.window),
        "base_seed": int(args.base_seed),
        "metric_specs": [asdict(spec) for spec in METRIC_SPECS],
        "joint_test": (
            "max robust standardized discrepancy across predeclared summaries "
            "and rolling-accuracy curve, calibrated against B0 rollouts"
        ),
        "multiplicity": (
            "subject-level joint p values additionally controlled across "
            "subjects by Benjamini-Hochberg FDR"
        ),
    }
    data_quality = {
        "condition1_row_n": int(len(data)),
        "subject_n": int(data["iSub"].nunique()),
        "duplicate_trial_key_n": int(
            data.duplicated(list(KEY_COLUMNS)).sum()
        ),
        "missing_values": {
            column: int(data[column].isna().sum())
            for column in (
                *KEY_COLUMNS,
                *FEATURE_COLUMNS,
                "category",
                "choice",
                "feedback",
            )
        },
        "feedback_mismatch_n": int(
            np.sum(
                data["feedback"].to_numpy(dtype=int)
                != (
                    data["choice"].to_numpy(dtype=int)
                    == data["category"].to_numpy(dtype=int)
                )
            )
        ),
        "train_n_total": int(subject_summary["train_n"].sum()),
        "test_n_total": int(subject_summary["test_n"].sum()),
        "split_status_counts": subject_summary[
            "split_status"
        ].value_counts().to_dict(),
        "boundary_ess_fraction_mean": float(
            np.mean(
                subject_summary["boundary_ess"]
                / subject_summary["particle_count"]
            )
        ),
        "minimum_prefix_post_ess_fraction": float(
            subject_summary["min_prefix_post_ess_fraction"].min()
        ),
    }
    decision = {
        **recommendation,
        "interpretation": (
            "Passing means the observed suffix is not an extreme draw from "
            "the conditioned B0 generative distribution. It does not establish "
            "B0 as the unique cognitive mechanism."
        ),
        "sharpness": {
            "median_90pct_rolling_interval_width": float(
                subject_summary[
                    "curve_pointwise_interval_width_90"
                ].median()
            ),
            "median_95pct_rolling_interval_width": float(
                subject_summary[
                    "curve_pointwise_interval_width_95"
                ].median()
            ),
            "mean_curve_crps": float(
                subject_summary["curve_crps"].mean()
            ),
        },
        "cohort_calibration": cohort_summary.to_dict(orient="records"),
    }

    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "data_quality.json", data_quality)
    write_json(args.output_dir / "decision.json", decision)
    subject_summary.to_csv(
        args.output_dir / "subject_summary.csv", index=False
    )
    metric_summary.to_csv(
        args.output_dir / "metric_summary.csv", index=False
    )
    curve_summary.to_csv(
        args.output_dir / "rolling_curve_summary.csv", index=False
    )
    cohort_summary.to_csv(
        args.output_dir / "cohort_calibration.csv", index=False
    )
    metric_failure.to_csv(
        args.output_dir / "metric_failures.csv", index=False
    )
    parameter_selection.to_csv(
        args.output_dir / "subject_parameter_selection.csv",
        index=False,
    )
    print(
        json.dumps(
            {
                "subjects": int(len(subject_summary)),
                "combined_pass_n": int(
                    subject_summary["combined_pass_95"].sum()
                ),
                "combined_fdr_failure_n": fdr_failure_n,
                "generative_adequacy": recommendation[
                    "generative_adequacy"
                ],
                "recommended_model_action": recommendation[
                    "recommended_model_action"
                ],
                "output_dir": str(args.output_dir),
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
