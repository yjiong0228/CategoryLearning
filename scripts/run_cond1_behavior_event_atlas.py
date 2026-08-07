#!/usr/bin/env python3
"""Build a block-bounded atlas of abrupt performance drops in condition 1.

The analysis is deliberately descriptive.  It does not fit a new latent-state
model and it does not infer a cognitive mechanism from a single episode.
Instead it:

1. detects abrupt drops using a frozen, model-independent rule;
2. quantifies pre-event, event, and recovery behavior;
3. asks whether choices are consistent with a wrong rule, perseveration,
   engagement/RT changes, or only transient noise;
4. applies the same rule to the reserved subjects; and
5. emits an explicit gate for whether any one minimal extension is justified.

All rolling windows and recovery searches are reset at block boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Bayesian_state.problems.partitions import Partition  # noqa: E402


DEFAULT_DEVELOPMENT_SUBJECTS = (103, 105, 111, 112, 117, 118, 127, 131)
KEY_COLUMNS = ("iSub", "condition", "iSession", "iBlock", "iTrial")
REQUIRED_COLUMNS = (
    "iSub",
    "condition",
    "iSession",
    "iBlock",
    "iTrial",
    "feature1",
    "feature2",
    "feature3",
    "feature4",
    "category",
    "choice",
    "feedback",
    "ambiguous",
    "choRT",
)


@dataclass(frozen=True)
class DetectorSpec:
    window: int
    baseline_threshold: float = 2.0 / 3.0
    low_threshold: float = 0.5
    minimum_drop: float = 0.25


PRIMARY_SPEC = DetectorSpec(window=12)
SENSITIVITY_SPECS = (DetectorSpec(window=8), DetectorSpec(window=16))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data/processed/Task2_processed.csv",
    )
    parser.add_argument("--condition", type=int, default=1)
    parser.add_argument(
        "--development-subjects",
        type=int,
        nargs="+",
        default=list(DEFAULT_DEVELOPMENT_SUBJECTS),
    )
    parser.add_argument(
        "--b0-cache-root",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_active_set/event_atlas_b0_r128/cache"
        ),
    )
    parser.add_argument("--particle-count", type=int, default=128)
    parser.add_argument("--filter-replicate", type=int, default=0)
    parser.add_argument("--rt-qc-threshold", type=float, default=4.0)
    parser.add_argument("--rule-permutations", type=int, default=1000)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--sequence-null-repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "results/zhuran/cond1_active_set/behavior_event_atlas"
        ),
    )
    return parser.parse_args()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def safe_rate(values: Sequence[bool] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def benjamini_hochberg(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted q values, retaining NaN positions."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not finite_indices.size:
        return adjusted
    finite_values = values[finite_indices]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    n_tests = ranked.size
    raw_adjusted = ranked * n_tests / np.arange(1, n_tests + 1)
    monotone = np.minimum.accumulate(raw_adjusted[::-1])[::-1]
    monotone = np.clip(monotone, 0.0, 1.0)
    local = np.empty(n_tests, dtype=float)
    local[order] = monotone
    adjusted[finite_indices] = local
    return adjusted


def robust_scale(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return float("nan"), float("nan")
    center = float(np.median(finite))
    mad_scale = float(1.4826 * np.median(np.abs(finite - center)))
    if not np.isfinite(mad_scale) or mad_scale <= 1e-12:
        mad_scale = float(np.std(finite, ddof=1)) if finite.size > 1 else 1.0
    if not np.isfinite(mad_scale) or mad_scale <= 1e-12:
        mad_scale = 1.0
    return center, mad_scale


def prepare_data(
    path: Path,
    *,
    condition: int,
    development_subjects: set[int],
    rt_qc_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Task2 data is missing required columns: {missing}")
    frame = frame[frame["condition"].eq(int(condition))].copy()
    frame = frame.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No condition-1 rows remain.")

    duplicate_rows = int(frame.duplicated(list(KEY_COLUMNS)).sum())
    feedback_expected = (
        frame["choice"].to_numpy(dtype=int)
        == frame["category"].to_numpy(dtype=int)
    ).astype(float)
    feedback_mismatch = int(
        np.sum(
            ~np.isclose(
                frame["feedback"].to_numpy(dtype=float),
                feedback_expected,
                atol=1e-12,
            )
        )
    )
    feature_columns = [f"feature{idx}" for idx in range(1, 5)]
    feature_out_of_range = int(
        np.sum(
            (frame[feature_columns].to_numpy(dtype=float) < 0.0)
            | (frame[feature_columns].to_numpy(dtype=float) > 1.0)
        )
    )

    frame["cohort"] = np.where(
        frame["iSub"].isin(development_subjects),
        "development",
        "reserved_application",
    )
    frame["subject_trial"] = frame.groupby("iSub", sort=False).cumcount() + 1
    frame["block_position"] = (
        frame.groupby(["iSub", "iSession", "iBlock"], sort=False).cumcount() + 1
    )
    frame["block_length"] = frame.groupby(
        ["iSub", "iSession", "iBlock"], sort=False
    )["choice"].transform("size")
    frame["log_rt"] = np.log(frame["choRT"].astype(float))
    frame["rt_robust_z"] = np.nan
    frame["rt_qc_keep"] = False
    rt_qc_rows: list[dict[str, Any]] = []
    for subject, subject_frame in frame.groupby("iSub", sort=True):
        indices = subject_frame.index
        center, scale = robust_scale(subject_frame["log_rt"].to_numpy(dtype=float))
        z = (subject_frame["log_rt"].to_numpy(dtype=float) - center) / scale
        keep = np.isfinite(z) & (np.abs(z) <= float(rt_qc_threshold))
        frame.loc[indices, "rt_robust_z"] = z
        frame.loc[indices, "rt_qc_keep"] = keep
        rt_qc_rows.append(
            {
                "iSub": int(subject),
                "cohort": str(subject_frame["cohort"].iloc[0]),
                "n_trials": int(len(subject_frame)),
                "log_rt_median": center,
                "log_rt_mad_scale": scale,
                "rt_qc_excluded": int(np.sum(~keep)),
                "rt_qc_excluded_fraction": float(np.mean(~keep)),
            }
        )

    n_subjects = int(frame["iSub"].nunique())
    n_development = int(
        frame.loc[frame["cohort"].eq("development"), "iSub"].nunique()
    )
    n_reserved = int(
        frame.loc[frame["cohort"].eq("reserved_application"), "iSub"].nunique()
    )
    quality = {
        "source": str(path),
        "condition": int(condition),
        "row_count": int(len(frame)),
        "subject_count": n_subjects,
        "development_subject_count": n_development,
        "reserved_application_subject_count": n_reserved,
        "duplicate_key_rows": duplicate_rows,
        "feedback_choice_category_mismatch_rows": feedback_mismatch,
        "feature_values_outside_unit_interval": feature_out_of_range,
        "missing_required_values": {
            column: int(frame[column].isna().sum()) for column in REQUIRED_COLUMNS
        },
        "trials_per_subject": {
            key: float(value)
            for key, value in frame.groupby("iSub").size().describe().to_dict().items()
        },
        "choRT_quantiles_seconds": {
            str(key): float(value)
            for key, value in frame["choRT"]
            .quantile([0.0, 0.01, 0.5, 0.95, 0.99, 1.0])
            .to_dict()
            .items()
        },
        "rt_qc": rt_qc_rows,
    }
    return frame, quality


def load_b0_predictions(
    frame: pd.DataFrame,
    *,
    cache_root: Path,
    particle_count: int,
    replicate: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    enriched = frame.copy()
    for column in (
        "b0_prob_observed_choice",
        "b0_choice_brier_trial",
        "b0_surprise",
        "b0_entropy",
        "b0_valid_mask",
        "b0_train_mask",
        "b0_test_mask",
    ):
        enriched[column] = np.nan

    loaded_subjects: list[int] = []
    missing_subjects: list[int] = []
    mismatched_subjects: list[int] = []
    for subject, subject_frame in enriched.groupby("iSub", sort=True):
        cache_path = (
            cache_root
            / f"subject_{int(subject)}"
            / f"particles_{int(particle_count)}"
            / f"replicate_{int(replicate)}"
            / "theta_0.npz"
        )
        if not cache_path.exists():
            missing_subjects.append(int(subject))
            continue
        with np.load(cache_path, allow_pickle=False) as payload:
            probabilities = np.asarray(
                payload["marginal_probabilities"], dtype=float
            )
            observed_index = np.asarray(
                payload["observed_choice_index"], dtype=int
            )
            valid_mask = np.asarray(payload["valid_mask"], dtype=bool)
            train_mask = np.asarray(payload["train_mask"], dtype=bool)
            test_mask = np.asarray(payload["test_mask"], dtype=bool)
        local_choices = subject_frame["choice"].to_numpy(dtype=int) - 1
        if (
            probabilities.shape != (len(subject_frame), 2)
            or observed_index.shape[0] != len(subject_frame)
            or not np.array_equal(observed_index, local_choices)
        ):
            mismatched_subjects.append(int(subject))
            continue
        probability_observed = probabilities[
            np.arange(len(subject_frame)), local_choices
        ]
        one_hot = np.eye(2, dtype=float)[local_choices]
        brier = np.sum((probabilities - one_hot) ** 2, axis=1)
        entropy = -np.sum(
            probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)),
            axis=1,
        )
        indices = subject_frame.index
        enriched.loc[indices, "b0_prob_observed_choice"] = probability_observed
        enriched.loc[indices, "b0_choice_brier_trial"] = brier
        enriched.loc[indices, "b0_surprise"] = -np.log(
            np.clip(probability_observed, 1e-12, 1.0)
        )
        enriched.loc[indices, "b0_entropy"] = entropy
        enriched.loc[indices, "b0_valid_mask"] = valid_mask.astype(float)
        enriched.loc[indices, "b0_train_mask"] = train_mask.astype(float)
        enriched.loc[indices, "b0_test_mask"] = test_mask.astype(float)
        loaded_subjects.append(int(subject))

    audit = {
        "cache_root": str(cache_root),
        "particle_count": int(particle_count),
        "replicate": int(replicate),
        "loaded_subjects": loaded_subjects,
        "missing_subjects": missing_subjects,
        "mismatched_subjects": mismatched_subjects,
        "coverage_fraction": len(loaded_subjects)
        / max(1, enriched["iSub"].nunique()),
    }
    return enriched, audit


def plane_text(coefficients: Sequence[float], boundary: float) -> str:
    terms: list[str] = []
    for idx, coefficient in enumerate(coefficients, start=1):
        value = float(coefficient)
        if abs(value) <= 1e-12:
            continue
        sign = "+" if value > 0 and terms else ""
        if value == 1:
            terms.append(f"{sign}x{idx}")
        elif value == -1:
            terms.append(f"-x{idx}")
        else:
            terms.append(f"{sign}{value:g}x{idx}")
    return "".join(terms) + f"={float(boundary):g}"


def rule_names(partition: Partition) -> list[str]:
    names: list[str] = []
    for hypothesis, (split, metadata) in enumerate(
        zip(partition.splits, partition.hypothesis_metadata)
    ):
        planes = "&".join(
            plane_text(coefficients, boundary)
            for coefficients, boundary in split.hyperplanes
        )
        permutation = "".join(
            str(int(value) + 1) for value in metadata["label_permutation"]
        )
        names.append(
            f"h{hypothesis}:{split.type}:{planes}:labels{permutation}"
        )
    return names


def build_rule_predictions(
    frame: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], list[str], dict[int, dict[str, Any]]]:
    partition = Partition(
        n_dims=4,
        n_cats=2,
        include_label_reversals=True,
    )
    names = rule_names(partition)
    predictions: dict[int, np.ndarray] = {}
    subject_info: dict[int, dict[str, Any]] = {}
    feature_columns = [f"feature{idx}" for idx in range(1, 5)]
    for subject, subject_frame in frame.groupby("iSub", sort=True):
        stimuli = subject_frame[feature_columns].to_numpy(dtype=float)
        matrix = np.vstack(
            [
                partition._get_category_assignments_region(  # noqa: SLF001
                    hypothesis, stimuli
                )
                + 1
                for hypothesis in range(partition.length)
            ]
        )
        categories = subject_frame["category"].to_numpy(dtype=int)
        target_agreement = np.mean(matrix == categories[None, :], axis=1)
        best_target = int(np.argmax(target_agreement))
        target_equivalent = np.flatnonzero(
            np.isclose(
                target_agreement,
                float(np.max(target_agreement)),
                atol=1e-12,
            )
        )
        predictions[int(subject)] = matrix
        subject_info[int(subject)] = {
            "target_hypothesis": best_target,
            "target_rule": names[best_target],
            "target_rule_accuracy": float(target_agreement[best_target]),
            "target_equivalent_hypotheses": [
                int(value) for value in target_equivalent
            ],
            "hypothesis_count": int(partition.length),
        }
    return predictions, names, subject_info


def detect_abrupt_events(
    feedback: np.ndarray,
    spec: DetectorSpec,
) -> list[dict[str, Any]]:
    values = np.asarray(feedback, dtype=float)
    n_trials = values.size
    window = int(spec.window)
    candidates: list[dict[str, Any]] = []
    for onset in range(window, n_trials - window + 1):
        pre = values[onset - window : onset]
        post = values[onset : onset + window]
        pre_accuracy = float(np.mean(pre))
        post_accuracy = float(np.mean(post))
        drop = pre_accuracy - post_accuracy
        if (
            pre_accuracy >= float(spec.baseline_threshold)
            and post_accuracy <= float(spec.low_threshold)
            and drop >= float(spec.minimum_drop)
        ):
            candidates.append(
                {
                    "onset": int(onset),
                    "detection_pre_accuracy": pre_accuracy,
                    "detection_post_accuracy": post_accuracy,
                    "detection_drop": drop,
                }
            )

    events: list[dict[str, Any]] = []
    blocked_until = -1
    for candidate in candidates:
        onset = int(candidate["onset"])
        if onset <= blocked_until:
            continue
        recovery_onset: int | None = None
        # The post-onset window is the evidence that establishes the drop.
        # Searching for recovery inside that same window would allow an event
        # to end before the evidence used to detect it has finished.
        for position in range(onset + window, n_trials - window + 1):
            recovery_accuracy = float(np.mean(values[position : position + window]))
            if recovery_accuracy >= float(spec.baseline_threshold):
                recovery_onset = int(position)
                break
        end = (
            int(recovery_onset - 1)
            if recovery_onset is not None
            else int(n_trials - 1)
        )
        event = {
            **candidate,
            "end": end,
            "recovered": recovery_onset is not None,
            "recovery_onset": recovery_onset,
            "recovery_latency": (
                int(recovery_onset - onset)
                if recovery_onset is not None
                else float("nan")
            ),
            "duration": int(end - onset + 1),
        }
        events.append(event)
        blocked_until = end
    return events


def phase_slice(
    block_frame: pd.DataFrame,
    *,
    start: int,
    end: int,
) -> pd.DataFrame:
    left = max(0, int(start))
    right = min(len(block_frame), int(end))
    return block_frame.iloc[left:right]


def stay_metrics(phase: pd.DataFrame) -> dict[str, float]:
    if len(phase) < 2:
        return {
            "stay_rate": float("nan"),
            "win_stay_rate": float("nan"),
            "lose_stay_rate": float("nan"),
            "switch_after_win_rate": float("nan"),
            "switch_after_loss_rate": float("nan"),
        }
    choice = phase["choice"].to_numpy(dtype=int)
    feedback = phase["feedback"].to_numpy(dtype=float)
    stay = choice[1:] == choice[:-1]
    prior_win = feedback[:-1] >= 1.0
    prior_loss = feedback[:-1] < 1.0
    win_stay = safe_rate(stay[prior_win])
    lose_stay = safe_rate(stay[prior_loss])
    return {
        "stay_rate": safe_rate(stay),
        "win_stay_rate": win_stay,
        "lose_stay_rate": lose_stay,
        "switch_after_win_rate": (
            1.0 - win_stay if np.isfinite(win_stay) else float("nan")
        ),
        "switch_after_loss_rate": (
            1.0 - lose_stay if np.isfinite(lose_stay) else float("nan")
        ),
    }


def phase_metrics(phase: pd.DataFrame) -> dict[str, float]:
    if phase.empty:
        return {
            key: float("nan")
            for key in (
                "n",
                "accuracy",
                "dominant_choice_rate",
                "choice_entropy",
                "ambiguous_rate",
                "log_rt_mean",
                "rt_robust_z_mean",
                "b0_brier_mean",
                "b0_surprise_mean",
                "b0_entropy_mean",
                "stay_rate",
                "win_stay_rate",
                "lose_stay_rate",
                "switch_after_win_rate",
                "switch_after_loss_rate",
            )
        }
    choice_counts = (
        phase["choice"].value_counts(normalize=True).sort_index().to_numpy(dtype=float)
    )
    choice_entropy = float(
        -np.sum(choice_counts * np.log(np.clip(choice_counts, 1e-12, 1.0)))
    )
    rt_phase = phase.loc[phase["rt_qc_keep"].astype(bool)]
    return {
        "n": int(len(phase)),
        "accuracy": float(phase["feedback"].mean()),
        "dominant_choice_rate": float(np.max(choice_counts)),
        "choice_entropy": choice_entropy,
        "ambiguous_rate": float(phase["ambiguous"].astype(float).mean()),
        "log_rt_mean": finite_mean(rt_phase["log_rt"].to_numpy(dtype=float)),
        "rt_robust_z_mean": finite_mean(
            rt_phase["rt_robust_z"].to_numpy(dtype=float)
        ),
        "b0_brier_mean": finite_mean(
            phase["b0_choice_brier_trial"].to_numpy(dtype=float)
        ),
        "b0_surprise_mean": finite_mean(
            phase["b0_surprise"].to_numpy(dtype=float)
        ),
        "b0_entropy_mean": finite_mean(
            phase["b0_entropy"].to_numpy(dtype=float)
        ),
        **stay_metrics(phase),
    }


def event_rule_diagnostics(
    *,
    choices: np.ndarray,
    rule_matrix: np.ndarray,
    target_equivalent: Sequence[int],
    rule_name_values: Sequence[str],
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    choices = np.asarray(choices, dtype=int)
    n_trials = int(choices.size)
    candidate_indices = np.asarray(
        [
            index
            for index in range(rule_matrix.shape[0])
            if index not in set(int(value) for value in target_equivalent)
        ],
        dtype=int,
    )
    if n_trials < 4 or not candidate_indices.size:
        return {
            "best_wrong_hypothesis": -1,
            "best_wrong_rule": "",
            "best_wrong_rule_accuracy": float("nan"),
            "wrong_rule_gain_over_target": float("nan"),
            "wrong_rule_permutation_p": float("nan"),
            "wrong_rule_crossfit_accuracy": float("nan"),
            "wrong_rule_half_agreement": False,
        }

    local_matrix = rule_matrix[candidate_indices]
    accuracies = np.mean(local_matrix == choices[None, :], axis=1)
    best_local = int(np.argmax(accuracies))
    best_hypothesis = int(candidate_indices[best_local])
    best_accuracy = float(accuracies[best_local])
    target_accuracy = float(
        np.mean(
            rule_matrix[int(target_equivalent[0])] == choices
        )
    )

    midpoint = n_trials // 2
    first = np.arange(0, midpoint)
    second = np.arange(midpoint, n_trials)
    first_accuracy = np.mean(
        local_matrix[:, first] == choices[first][None, :], axis=1
    )
    second_accuracy = np.mean(
        local_matrix[:, second] == choices[second][None, :], axis=1
    )
    first_best = int(np.argmax(first_accuracy))
    second_best = int(np.argmax(second_accuracy))
    crossfit = 0.5 * (
        float(
            np.mean(
                local_matrix[first_best, second] == choices[second]
            )
        )
        + float(
            np.mean(
                local_matrix[second_best, first] == choices[first]
            )
        )
    )

    rng = np.random.default_rng(int(seed))
    null_max = np.empty(int(permutations), dtype=float)
    for permutation in range(int(permutations)):
        shuffled = rng.permutation(choices)
        null_max[permutation] = float(
            np.max(np.mean(local_matrix == shuffled[None, :], axis=1))
        )
    permutation_p = float(
        (1.0 + np.sum(null_max >= best_accuracy - 1e-12))
        / (1.0 + int(permutations))
    )
    return {
        "best_wrong_hypothesis": best_hypothesis,
        "best_wrong_rule": str(rule_name_values[best_hypothesis]),
        "best_wrong_rule_accuracy": best_accuracy,
        "wrong_rule_gain_over_target": best_accuracy - target_accuracy,
        "wrong_rule_permutation_p": permutation_p,
        "wrong_rule_crossfit_accuracy": crossfit,
        "wrong_rule_half_agreement": bool(first_best == second_best),
    }


def classify_event(row: Mapping[str, Any]) -> str:
    rule_significance = float(row["wrong_rule_permutation_p"])
    if (
        "wrong_rule_fdr_q" in row
        and np.isfinite(float(row["wrong_rule_fdr_q"]))
    ):
        rule_significance = float(row["wrong_rule_fdr_q"])
    wrong_rule = (
        float(row["best_wrong_rule_accuracy"]) >= 0.75
        and float(row["wrong_rule_gain_over_target"]) >= 0.20
        and rule_significance <= 0.05
        and float(row["wrong_rule_crossfit_accuracy"]) >= 0.65
    )
    if wrong_rule:
        return "candidate_wrong_rule"

    dominant_delta = float(row["event_dominant_choice_rate"]) - float(
        row["pre_dominant_choice_rate"]
    )
    lose_stay_delta = float(row["event_lose_stay_rate"]) - float(
        row["pre_lose_stay_rate"]
    )
    perseveration = (
        (
            float(row["event_dominant_choice_rate"]) >= 0.80
            and dominant_delta >= 0.15
        )
        or (
            np.isfinite(float(row["event_lose_stay_rate"]))
            and np.isfinite(lose_stay_delta)
            and float(row["event_lose_stay_rate"]) >= 0.70
            and lose_stay_delta >= 0.15
        )
    )
    if perseveration:
        return "choice_bias_or_perseveration"

    rt_delta = float(row["delta_rt_robust_z_mean"])
    low_engagement = (
        float(row["event_accuracy"]) <= 0.50
        and np.isfinite(rt_delta)
        and abs(rt_delta) >= 0.35
        and float(row["best_wrong_rule_accuracy"]) < 0.75
        and float(row["event_dominant_choice_rate"]) < 0.80
    )
    if low_engagement:
        return (
            "candidate_engagement_speedup"
            if rt_delta < 0.0
            else "candidate_engagement_slowdown"
        )

    if (
        int(row["duration"]) <= int(row["detector_window"])
        and bool(row["recovered"])
        and float(row["best_wrong_rule_accuracy"]) < 0.75
    ):
        return "transient_noise_or_lapse"
    return "mixed_unresolved"


def make_event_row(
    *,
    subject_frame: pd.DataFrame,
    block_frame: pd.DataFrame,
    event: Mapping[str, Any],
    spec: DetectorSpec,
    detector_id: str,
    rule_matrix_subject: np.ndarray,
    rule_name_values: Sequence[str],
    rule_info: Mapping[str, Any],
    rule_permutations: int,
    seed: int,
) -> dict[str, Any]:
    onset = int(event["onset"])
    end = int(event["end"])
    window = int(spec.window)
    pre = phase_slice(block_frame, start=onset - window, end=onset)
    during = phase_slice(block_frame, start=onset, end=end + 1)
    recovery = phase_slice(
        block_frame,
        start=end + 1,
        end=end + 1 + window,
    )
    pre_metrics = phase_metrics(pre)
    event_metrics = phase_metrics(during)
    recovery_metrics = phase_metrics(recovery)

    subject_positions = subject_frame.index.to_numpy()
    subject_index_lookup = {
        int(index): local for local, index in enumerate(subject_positions)
    }
    during_subject_indices = np.asarray(
        [subject_index_lookup[int(index)] for index in during.index],
        dtype=int,
    )
    rule_diagnostics = event_rule_diagnostics(
        choices=during["choice"].to_numpy(dtype=int),
        rule_matrix=rule_matrix_subject[:, during_subject_indices],
        target_equivalent=rule_info["target_equivalent_hypotheses"],
        rule_name_values=rule_name_values,
        permutations=int(rule_permutations),
        seed=int(seed),
    )
    row: dict[str, Any] = {
        "iSub": int(block_frame["iSub"].iloc[0]),
        "cohort": str(block_frame["cohort"].iloc[0]),
        "iSession": int(block_frame["iSession"].iloc[0]),
        "iBlock": int(block_frame["iBlock"].iloc[0]),
        "detector_id": detector_id,
        "detector_window": window,
        "baseline_threshold": float(spec.baseline_threshold),
        "low_threshold": float(spec.low_threshold),
        "minimum_drop": float(spec.minimum_drop),
        "onset_block_position": onset + 1,
        "end_block_position": end + 1,
        "onset_iTrial": int(block_frame.iloc[onset]["iTrial"]),
        "end_iTrial": int(block_frame.iloc[end]["iTrial"]),
        "block_length": int(len(block_frame)),
        **event,
        "target_hypothesis": int(rule_info["target_hypothesis"]),
        "target_rule": str(rule_info["target_rule"]),
        "target_rule_accuracy_subject": float(
            rule_info["target_rule_accuracy"]
        ),
        **rule_diagnostics,
    }
    for prefix, metrics in (
        ("pre", pre_metrics),
        ("event", event_metrics),
        ("recovery", recovery_metrics),
    ):
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value

    for metric in (
        "accuracy",
        "dominant_choice_rate",
        "choice_entropy",
        "ambiguous_rate",
        "log_rt_mean",
        "rt_robust_z_mean",
        "b0_brier_mean",
        "b0_surprise_mean",
        "b0_entropy_mean",
        "stay_rate",
        "win_stay_rate",
        "lose_stay_rate",
        "switch_after_loss_rate",
    ):
        row[f"delta_{metric}"] = float(row[f"event_{metric}"]) - float(
            row[f"pre_{metric}"]
        )
    row["event_type"] = classify_event(row)
    return row


def intervals_overlap(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    if int(left["iSession"]) != int(right["iSession"]):
        return False
    if int(left["iBlock"]) != int(right["iBlock"]):
        return False
    return not (
        int(left["end_block_position"]) < int(right["onset_block_position"])
        or int(right["end_block_position"]) < int(left["onset_block_position"])
    )


def detect_all_events(
    frame: pd.DataFrame,
    *,
    rule_predictions: Mapping[int, np.ndarray],
    rule_name_values: Sequence[str],
    rule_subject_info: Mapping[int, Mapping[str, Any]],
    specs: Sequence[tuple[str, DetectorSpec]],
    rule_permutations: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, subject_frame in frame.groupby("iSub", sort=True):
        for (session, block), block_frame in subject_frame.groupby(
            ["iSession", "iBlock"], sort=True
        ):
            block_frame = block_frame.copy()
            events_by_spec = {
                detector_id: detect_abrupt_events(
                    block_frame["feedback"].to_numpy(dtype=float),
                    spec,
                )
                for detector_id, spec in specs
            }
            for detector_id, spec in specs:
                for event_index, event in enumerate(
                    events_by_spec[detector_id], start=1
                ):
                    event_seed = (
                        int(seed)
                        + 1_000_003 * int(subject)
                        + 10_007 * int(session)
                        + 101 * int(block)
                        + 17 * int(spec.window)
                        + event_index
                    )
                    row = make_event_row(
                        subject_frame=subject_frame,
                        block_frame=block_frame,
                        event=event,
                        spec=spec,
                        detector_id=detector_id,
                        rule_matrix_subject=rule_predictions[int(subject)],
                        rule_name_values=rule_name_values,
                        rule_info=rule_subject_info[int(subject)],
                        rule_permutations=int(rule_permutations),
                        seed=event_seed,
                    )
                    row["event_index_within_block"] = int(event_index)
                    rows.append(row)
    return pd.DataFrame(rows)


def add_sensitivity_support(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    primary = events.loc[events["detector_id"].eq("primary_w12")].copy()
    for detector_id in ("sensitivity_w8", "sensitivity_w16"):
        detector_events = events.loc[events["detector_id"].eq(detector_id)]
        support = []
        for _, row in primary.iterrows():
            candidates = detector_events.loc[
                detector_events["iSub"].eq(row["iSub"])
            ]
            support.append(
                any(
                    intervals_overlap(row, candidate)
                    for _, candidate in candidates.iterrows()
                )
            )
        primary[f"supported_by_{detector_id}"] = support
    primary["sensitivity_support_count"] = (
        1
        + primary["supported_by_sensitivity_w8"].astype(int)
        + primary["supported_by_sensitivity_w16"].astype(int)
    )
    primary["robust_across_windows"] = (
        primary["sensitivity_support_count"] >= 2
    )
    return primary


def add_event_level_rule_fdr(events: pd.DataFrame) -> pd.DataFrame:
    """Correct wrong-rule permutation tests across primary events by cohort."""
    if events.empty:
        return events.copy()
    result = events.copy()
    result["event_type_uncorrected"] = result["event_type"]
    result["wrong_rule_fdr_q"] = np.nan
    for _, group in result.groupby("cohort", sort=True):
        result.loc[group.index, "wrong_rule_fdr_q"] = benjamini_hochberg(
            group["wrong_rule_permutation_p"].to_numpy(dtype=float)
        )
    result["event_type"] = [
        classify_event(row) for row in result.to_dict(orient="records")
    ]
    return result


def bootstrap_subject_mean(
    frame: pd.DataFrame,
    column: str,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    subject_values = (
        frame.groupby("iSub", sort=True)[column]
        .mean()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if subject_values.empty:
        return {
            "column": column,
            "subject_n": 0,
            "mean": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
        }
    values = subject_values.to_numpy(dtype=float)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(
        0,
        values.size,
        size=(int(repetitions), values.size),
    )
    estimates = np.mean(values[indices], axis=1)
    return {
        "column": column,
        "subject_n": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci_lower": float(np.quantile(estimates, 0.025)),
        "ci_upper": float(np.quantile(estimates, 0.975)),
        "positive_subject_fraction": float(np.mean(values > 0.0)),
        "negative_subject_fraction": float(np.mean(values < 0.0)),
    }


def sequence_null_benchmark(
    frame: pd.DataFrame,
    primary_events: pd.DataFrame,
    *,
    spec: DetectorSpec,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark event counts after shuffling feedback within each block.

    The randomization preserves each block's accuracy and number of trials but
    destroys serial clustering.  It therefore asks whether the detector finds
    more sequential episodes than expected from the same marginal performance.
    """
    subjects = sorted(int(value) for value in frame["iSub"].unique())
    subject_index = {subject: index for index, subject in enumerate(subjects)}
    subject_cohort = {
        int(subject): str(group["cohort"].iloc[0])
        for subject, group in frame.groupby("iSub", sort=True)
    }
    blocks = [
        (
            subject_index[int(subject)],
            block_frame["feedback"].to_numpy(dtype=float),
        )
        for (subject, _, _), block_frame in frame.groupby(
            ["iSub", "iSession", "iBlock"], sort=True
        )
    ]
    rng = np.random.default_rng(int(seed))
    null_counts = np.zeros(
        (int(repetitions), len(subjects)),
        dtype=np.int16,
    )
    for repetition in range(int(repetitions)):
        for local_subject, feedback in blocks:
            shuffled = rng.permutation(feedback)
            null_counts[repetition, local_subject] += len(
                detect_abrupt_events(shuffled, spec)
            )

    observed_by_subject = (
        primary_events.groupby("iSub").size().reindex(subjects, fill_value=0)
    )
    summary_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []

    def add_summary(
        *,
        level: str,
        label: str,
        metric: str,
        observed: int,
        null_values: np.ndarray,
        subject_n: int,
    ) -> None:
        values = np.asarray(null_values, dtype=float)
        summary_rows.append(
            {
                "level": level,
                "label": label,
                "metric": metric,
                "subject_n": int(subject_n),
                "observed": int(observed),
                "null_mean": float(np.mean(values)),
                "null_ci_lower": float(np.quantile(values, 0.025)),
                "null_ci_upper": float(np.quantile(values, 0.975)),
                "observed_minus_null_mean": float(
                    observed - np.mean(values)
                ),
                "randomization_p_greater_equal": float(
                    (1.0 + np.sum(values >= observed))
                    / (1.0 + values.size)
                ),
            }
        )

    for local_subject, subject in enumerate(subjects):
        add_summary(
            level="subject",
            label=str(subject),
            metric="event_count",
            observed=int(observed_by_subject.loc[subject]),
            null_values=null_counts[:, local_subject],
            subject_n=1,
        )

    cohorts = sorted(set(subject_cohort.values()))
    for cohort in cohorts:
        indices = np.asarray(
            [
                subject_index[subject]
                for subject in subjects
                if subject_cohort[subject] == cohort
            ],
            dtype=int,
        )
        observed_counts = observed_by_subject.iloc[indices].to_numpy(dtype=int)
        null_event_count = np.sum(null_counts[:, indices], axis=1)
        null_affected = np.sum(null_counts[:, indices] > 0, axis=1)
        add_summary(
            level="cohort",
            label=cohort,
            metric="event_count",
            observed=int(np.sum(observed_counts)),
            null_values=null_event_count,
            subject_n=int(indices.size),
        )
        add_summary(
            level="cohort",
            label=cohort,
            metric="affected_subject_count",
            observed=int(np.sum(observed_counts > 0)),
            null_values=null_affected,
            subject_n=int(indices.size),
        )
        for repetition in range(int(repetitions)):
            distribution_rows.append(
                {
                    "repetition": repetition,
                    "level": "cohort",
                    "label": cohort,
                    "event_count": int(null_event_count[repetition]),
                    "affected_subject_count": int(
                        null_affected[repetition]
                    ),
                }
            )

    observed_all = observed_by_subject.to_numpy(dtype=int)
    null_all_events = np.sum(null_counts, axis=1)
    null_all_affected = np.sum(null_counts > 0, axis=1)
    add_summary(
        level="overall",
        label="all_subjects",
        metric="event_count",
        observed=int(np.sum(observed_all)),
        null_values=null_all_events,
        subject_n=len(subjects),
    )
    add_summary(
        level="overall",
        label="all_subjects",
        metric="affected_subject_count",
        observed=int(np.sum(observed_all > 0)),
        null_values=null_all_affected,
        subject_n=len(subjects),
    )
    for repetition in range(int(repetitions)):
        distribution_rows.append(
            {
                "repetition": repetition,
                "level": "overall",
                "label": "all_subjects",
                "event_count": int(null_all_events[repetition]),
                "affected_subject_count": int(
                    null_all_affected[repetition]
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["randomization_fdr_q"] = np.nan
    subject_rows = summary["level"].eq("subject") & summary["metric"].eq(
        "event_count"
    )
    summary.loc[subject_rows, "randomization_fdr_q"] = benjamini_hochberg(
        summary.loc[
            subject_rows, "randomization_p_greater_equal"
        ].to_numpy(dtype=float)
    )
    return summary, pd.DataFrame(distribution_rows)


def summarize_subjects(
    frame: pd.DataFrame,
    primary_events: pd.DataFrame,
    *,
    primary_spec: DetectorSpec,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, subject_frame in frame.groupby("iSub", sort=True):
        events = primary_events.loc[primary_events["iSub"].eq(subject)]
        first_windows = []
        for _, block_frame in subject_frame.groupby(
            ["iSession", "iBlock"], sort=True
        ):
            first_windows.append(
                float(
                    block_frame["feedback"]
                    .iloc[: int(primary_spec.window)]
                    .mean()
                )
            )
        event_types = (
            events["event_type"].value_counts().to_dict()
            if not events.empty
            else {}
        )
        rows.append(
            {
                "iSub": int(subject),
                "cohort": str(subject_frame["cohort"].iloc[0]),
                "n_trials": int(len(subject_frame)),
                "n_blocks": int(
                    subject_frame.groupby(["iSession", "iBlock"]).ngroups
                ),
                "mean_accuracy": float(subject_frame["feedback"].mean()),
                "median_log_rt": float(subject_frame["log_rt"].median()),
                "first_window_accuracy_mean": float(np.mean(first_windows)),
                "primary_event_count": int(len(events)),
                "robust_event_count": int(
                    events["robust_across_windows"].sum()
                )
                if not events.empty
                else 0,
                "event_trials_total": int(events["duration"].sum())
                if not events.empty
                else 0,
                "candidate_wrong_rule_count": int(
                    event_types.get("candidate_wrong_rule", 0)
                ),
                "choice_bias_or_perseveration_count": int(
                    event_types.get("choice_bias_or_perseveration", 0)
                ),
                "candidate_engagement_speedup_count": int(
                    event_types.get("candidate_engagement_speedup", 0)
                ),
                "candidate_engagement_slowdown_count": int(
                    event_types.get("candidate_engagement_slowdown", 0)
                ),
                "transient_noise_or_lapse_count": int(
                    event_types.get("transient_noise_or_lapse", 0)
                ),
                "mixed_unresolved_count": int(
                    event_types.get("mixed_unresolved", 0)
                ),
            }
        )
    return pd.DataFrame(rows)


def event_aligned_rows(
    frame: pd.DataFrame,
    primary_events: pd.DataFrame,
    *,
    pre_trials: int = 12,
    post_trials: int = 24,
) -> pd.DataFrame:
    if primary_events.empty:
        return pd.DataFrame()
    lookup = {
        (
            int(subject),
            int(session),
            int(block),
        ): block_frame.reset_index(drop=True)
        for (subject, session, block), block_frame in frame.groupby(
            ["iSub", "iSession", "iBlock"], sort=True
        )
    }
    rows: list[dict[str, Any]] = []
    for event_id, event in primary_events.reset_index(drop=True).iterrows():
        block = lookup[
            (
                int(event["iSub"]),
                int(event["iSession"]),
                int(event["iBlock"]),
            )
        ]
        onset = int(event["onset_block_position"]) - 1
        for relative_trial in range(-int(pre_trials), int(post_trials) + 1):
            position = onset + relative_trial
            if position < 0 or position >= len(block):
                continue
            trial = block.iloc[position]
            rows.append(
                {
                    "event_id": int(event_id),
                    "iSub": int(event["iSub"]),
                    "cohort": str(event["cohort"]),
                    "relative_trial": int(relative_trial),
                    "feedback": float(trial["feedback"]),
                    "rt_robust_z": (
                        float(trial["rt_robust_z"])
                        if bool(trial["rt_qc_keep"])
                        else float("nan")
                    ),
                    "b0_choice_brier_trial": float(
                        trial["b0_choice_brier_trial"]
                    ),
                    "ambiguous": float(trial["ambiguous"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_event_alignment(aligned: pd.DataFrame) -> pd.DataFrame:
    if aligned.empty:
        return pd.DataFrame()
    subject_means = (
        aligned.groupby(["cohort", "relative_trial", "iSub"], as_index=False)
        .agg(
            feedback=("feedback", "mean"),
            rt_robust_z=("rt_robust_z", "mean"),
            b0_choice_brier_trial=("b0_choice_brier_trial", "mean"),
            ambiguous=("ambiguous", "mean"),
        )
    )
    rows: list[dict[str, Any]] = []
    for (cohort, relative_trial), group in subject_means.groupby(
        ["cohort", "relative_trial"], sort=True
    ):
        row: dict[str, Any] = {
            "cohort": str(cohort),
            "relative_trial": int(relative_trial),
            "subject_n": int(group["iSub"].nunique()),
        }
        for metric in (
            "feedback",
            "rt_robust_z",
            "b0_choice_brier_trial",
            "ambiguous",
        ):
            values = group[metric].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{metric}_mean"] = (
                float(np.mean(values)) if values.size else float("nan")
            )
            row[f"{metric}_sem"] = (
                float(np.std(values, ddof=1) / math.sqrt(values.size))
                if values.size > 1
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def cohort_summary(
    subject_summary: pd.DataFrame,
    primary_events: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, subjects in subject_summary.groupby("cohort", sort=True):
        events = primary_events.loc[primary_events["cohort"].eq(cohort)]
        event_subjects = int(events["iSub"].nunique()) if not events.empty else 0
        rows.append(
            {
                "cohort": str(cohort),
                "subject_n": int(len(subjects)),
                "trial_n": int(subjects["n_trials"].sum()),
                "event_n": int(len(events)),
                "event_subject_n": event_subjects,
                "event_subject_fraction": event_subjects / max(1, len(subjects)),
                "robust_event_n": int(events["robust_across_windows"].sum())
                if not events.empty
                else 0,
                "robust_event_fraction": float(
                    events["robust_across_windows"].mean()
                )
                if not events.empty
                else float("nan"),
                "median_event_duration": float(events["duration"].median())
                if not events.empty
                else float("nan"),
                "mean_event_accuracy": float(events["event_accuracy"].mean())
                if not events.empty
                else float("nan"),
                "mean_pre_accuracy": float(events["pre_accuracy"].mean())
                if not events.empty
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def mechanism_gate_summary(
    primary_events: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    reserved = primary_events.loc[
        primary_events["cohort"].eq("reserved_application")
    ].copy()
    definitions = [
        (
            "wrong_rule_switch",
            ["candidate_wrong_rule"],
            "wrong_rule_gain_over_target",
            0.10,
        ),
        (
            "perseveration",
            ["choice_bias_or_perseveration"],
            "delta_lose_stay_rate",
            0.10,
        ),
        (
            "engagement_speedup",
            ["candidate_engagement_speedup"],
            "delta_rt_robust_z_mean",
            None,
        ),
        (
            "engagement_slowdown",
            ["candidate_engagement_slowdown"],
            "delta_rt_robust_z_mean",
            None,
        ),
    ]
    rows: list[dict[str, Any]] = []
    for index, (mechanism, event_types, effect_column, effect_floor) in enumerate(
        definitions
    ):
        events = reserved.loc[reserved["event_type"].isin(event_types)].copy()
        event_share = len(events) / max(1, len(reserved))
        subject_n = int(events["iSub"].nunique()) if not events.empty else 0
        robust_fraction = (
            float(events["robust_across_windows"].mean())
            if not events.empty
            else float("nan")
        )
        effect = bootstrap_subject_mean(
            events,
            effect_column,
            repetitions=int(repetitions),
            seed=int(seed) + index,
        )
        b0_effect = bootstrap_subject_mean(
            events,
            "delta_b0_brier_mean",
            repetitions=int(repetitions),
            seed=int(seed) + 100 + index,
        )
        common_gate = (
            subject_n >= 4
            and event_share >= 0.25
            and np.isfinite(robust_fraction)
            and robust_fraction >= 0.60
        )
        if mechanism == "wrong_rule_switch":
            effect_gate = (
                np.isfinite(effect["ci_lower"])
                and float(effect["ci_lower"]) > float(effect_floor)
            )
        elif mechanism == "perseveration":
            effect_gate = (
                np.isfinite(effect["ci_lower"])
                and float(effect["ci_lower"]) > float(effect_floor)
            )
        elif mechanism == "engagement_speedup":
            effect_gate = (
                np.isfinite(effect["ci_upper"])
                and float(effect["ci_upper"]) < 0.0
            )
        else:
            effect_gate = (
                np.isfinite(effect["ci_lower"])
                and float(effect["ci_lower"]) > 0.0
            )
        b0_gate = (
            np.isfinite(b0_effect["ci_lower"])
            and float(b0_effect["ci_lower"]) > 0.0
        )
        passed = bool(common_gate and effect_gate and b0_gate)
        rows.append(
            {
                "mechanism": mechanism,
                "event_n": int(len(events)),
                "subject_n": subject_n,
                "reserved_event_share": float(event_share),
                "robust_event_fraction": robust_fraction,
                "effect_column": effect_column,
                "effect_mean": effect["mean"],
                "effect_ci_lower": effect["ci_lower"],
                "effect_ci_upper": effect["ci_upper"],
                "b0_brier_delta_mean": b0_effect["mean"],
                "b0_brier_delta_ci_lower": b0_effect["ci_lower"],
                "b0_brier_delta_ci_upper": b0_effect["ci_upper"],
                "common_prevalence_gate": bool(common_gate),
                "defining_effect_gate": bool(effect_gate),
                "b0_residual_gate": bool(b0_gate),
                "passed": passed,
            }
        )
    gate_frame = pd.DataFrame(rows)
    passed = gate_frame.loc[gate_frame["passed"]]
    if passed.empty:
        recommendation = {
            "recommended_module": "none_hold_B0",
            "reason": (
                "No mechanism-specific event family met the frozen prevalence, "
                "cross-window robustness, defining-effect, and B0-residual gates."
            ),
            "next_action": (
                "Keep B0 confirmatory; retain the event atlas as descriptive "
                "heterogeneity and do not add a hidden state."
            ),
        }
    else:
        selected = passed.sort_values(
            ["subject_n", "reserved_event_share", "robust_event_fraction"],
            ascending=False,
        ).iloc[0]
        recommendation = {
            "recommended_module": str(selected["mechanism"]),
            "reason": (
                f"{selected['mechanism']} was the strongest mechanism family "
                "to pass all frozen event-level gates."
            ),
            "next_action": (
                "Implement only this single minimal extension and require "
                "actual-design model recovery before fitting real subjects."
            ),
        }
    return gate_frame, recommendation


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    development_subjects = set(int(value) for value in args.development_subjects)

    frame, quality = prepare_data(
        args.data,
        condition=int(args.condition),
        development_subjects=development_subjects,
        rt_qc_threshold=float(args.rt_qc_threshold),
    )
    frame, b0_audit = load_b0_predictions(
        frame,
        cache_root=args.b0_cache_root,
        particle_count=int(args.particle_count),
        replicate=int(args.filter_replicate),
    )
    quality["b0_prediction_audit"] = b0_audit
    if b0_audit["coverage_fraction"] < 1.0:
        raise RuntimeError(
            "Frozen B0 predictions are incomplete or misaligned: "
            f"{b0_audit}"
        )

    rule_predictions, names, rule_subject_info = build_rule_predictions(frame)
    specs = [
        ("primary_w12", PRIMARY_SPEC),
        ("sensitivity_w8", SENSITIVITY_SPECS[0]),
        ("sensitivity_w16", SENSITIVITY_SPECS[1]),
    ]
    all_events = detect_all_events(
        frame,
        rule_predictions=rule_predictions,
        rule_name_values=names,
        rule_subject_info=rule_subject_info,
        specs=specs,
        rule_permutations=int(args.rule_permutations),
        seed=int(args.seed),
    )
    primary_events = add_event_level_rule_fdr(
        add_sensitivity_support(all_events)
    )
    subject_summary = summarize_subjects(
        frame,
        primary_events,
        primary_spec=PRIMARY_SPEC,
    )
    alignment = event_aligned_rows(frame, primary_events)
    alignment_summary = summarize_event_alignment(alignment)
    cohort = cohort_summary(subject_summary, primary_events)
    mechanism_gates, recommendation = mechanism_gate_summary(
        primary_events,
        repetitions=int(args.bootstrap_repetitions),
        seed=int(args.seed) + 500,
    )
    sequence_null_summary, sequence_null_distribution = (
        sequence_null_benchmark(
            frame,
            primary_events,
            spec=PRIMARY_SPEC,
            repetitions=int(args.sequence_null_repetitions),
            seed=int(args.seed) + 750,
        )
    )

    overall_effects = [
        bootstrap_subject_mean(
            primary_events.loc[
                primary_events["cohort"].eq("reserved_application")
            ],
            column,
            repetitions=int(args.bootstrap_repetitions),
            seed=int(args.seed) + 1000 + index,
        )
        for index, column in enumerate(
            (
                "delta_accuracy",
                "delta_rt_robust_z_mean",
                "delta_b0_brier_mean",
                "delta_dominant_choice_rate",
                "delta_lose_stay_rate",
                "wrong_rule_gain_over_target",
            )
        )
    ]
    event_type_counts = (
        primary_events.groupby(["cohort", "event_type"], as_index=False)
        .agg(
            event_n=("event_type", "size"),
            subject_n=("iSub", "nunique"),
            robust_event_n=("robust_across_windows", "sum"),
        )
        if not primary_events.empty
        else pd.DataFrame(
            columns=[
                "cohort",
                "event_type",
                "event_n",
                "subject_n",
                "robust_event_n",
            ]
        )
    )

    metadata = {
        "analysis": "condition_1_behavior_event_atlas",
        "status": "descriptive_mechanism_diagnostic",
        "data": str(args.data),
        "condition": int(args.condition),
        "development_subjects": sorted(development_subjects),
        "reserved_application_subjects": sorted(
            set(frame["iSub"].astype(int)) - development_subjects
        ),
        "reserved_set_note": (
            "These subjects were not used in B0/D0 development, but an earlier "
            "all-subject exploratory behavior scan was viewed before this "
            "detector was frozen; results are quasi-confirmatory, not blinded."
        ),
        "primary_detector": asdict(PRIMARY_SPEC),
        "sensitivity_detectors": [
            asdict(spec) for spec in SENSITIVITY_SPECS
        ],
        "rt_qc_threshold_mad": float(args.rt_qc_threshold),
        "rule_hypothesis_count": int(len(names)),
        "rule_permutations": int(args.rule_permutations),
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "sequence_null_repetitions": int(args.sequence_null_repetitions),
        "seed": int(args.seed),
        "mechanism_gate": {
            "minimum_reserved_subjects": 4,
            "minimum_reserved_event_share": 0.25,
            "minimum_cross_window_robust_fraction": 0.60,
            "requires_defining_effect_ci": True,
            "requires_positive_b0_residual_ci": True,
        },
    }
    decision = {
        **recommendation,
        "confirmatory_model": "B0",
        "event_atlas_role": "descriptive_not_subject_diagnosis",
        "overall_reserved_effects": overall_effects,
        "sequence_null_reserved": sequence_null_summary.loc[
            sequence_null_summary["level"].eq("cohort")
            & sequence_null_summary["label"].eq("reserved_application")
        ].to_dict(orient="records"),
        "quasi_confirmatory_caveat": metadata["reserved_set_note"],
    }

    write_json(output_dir / "metadata.json", metadata)
    write_json(output_dir / "data_quality.json", quality)
    write_json(output_dir / "rule_subject_info.json", rule_subject_info)
    write_json(output_dir / "decision.json", decision)
    frame.to_csv(output_dir / "trial_features.csv", index=False)
    all_events.to_csv(output_dir / "events_all_detectors.csv", index=False)
    primary_events.to_csv(output_dir / "events_primary.csv", index=False)
    subject_summary.to_csv(output_dir / "subject_summary.csv", index=False)
    alignment.to_csv(output_dir / "event_aligned_trials.csv", index=False)
    alignment_summary.to_csv(
        output_dir / "event_aligned_summary.csv", index=False
    )
    cohort.to_csv(output_dir / "cohort_summary.csv", index=False)
    event_type_counts.to_csv(
        output_dir / "event_type_counts.csv", index=False
    )
    mechanism_gates.to_csv(
        output_dir / "mechanism_gate_summary.csv", index=False
    )
    sequence_null_summary.to_csv(
        output_dir / "sequence_null_summary.csv", index=False
    )
    sequence_null_distribution.to_csv(
        output_dir / "sequence_null_distribution.csv", index=False
    )
    print(
        json.dumps(
            {
                "subjects": int(frame["iSub"].nunique()),
                "primary_events": int(len(primary_events)),
                "primary_event_subjects": int(
                    primary_events["iSub"].nunique()
                )
                if not primary_events.empty
                else 0,
                "reserved_primary_events": int(
                    primary_events["cohort"]
                    .eq("reserved_application")
                    .sum()
                )
                if not primary_events.empty
                else 0,
                "recommended_module": recommendation["recommended_module"],
                "output_dir": str(output_dir),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
