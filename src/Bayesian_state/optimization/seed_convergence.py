"""Numerical convergence summaries for seed-averaged particle likelihoods."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _validate_probability_bank(
    probabilities_by_candidate: Mapping[str, np.ndarray],
    choices: Sequence[int] | np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    observed_choices = np.asarray(choices, dtype=int).reshape(-1)
    if not np.all(np.isin(observed_choices, [1, 2])):
        raise ValueError("choices must be encoded as 1 or 2")
    if len(probabilities_by_candidate) < 2:
        raise ValueError("at least two candidate probability banks are required")
    validated: dict[str, np.ndarray] = {}
    expected_shape: tuple[int, int, int] | None = None
    for candidate, raw in sorted(probabilities_by_candidate.items()):
        probabilities = np.asarray(raw, dtype=float)
        if probabilities.ndim != 3:
            raise ValueError("candidate probabilities must have shape (B, T, C)")
        if probabilities.shape[1] != observed_choices.size:
            raise ValueError("probability trial count does not match choices")
        if probabilities.shape[2] != 2:
            raise ValueError("condition-1 probabilities must have two choices")
        if expected_shape is None:
            expected_shape = probabilities.shape
        elif probabilities.shape != expected_shape:
            raise ValueError("candidate probability banks must share one shape")
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
            raise ValueError("candidate probabilities must be finite and nonnegative")
        if not np.allclose(probabilities.sum(axis=2), 1.0, atol=1e-8):
            raise ValueError("candidate choice probabilities must sum to one")
        validated[str(candidate)] = probabilities
    return validated, observed_choices


def _nll_from_mean_probabilities(
    probabilities: np.ndarray,
    choices: np.ndarray,
) -> float:
    mean_probability = np.mean(probabilities, axis=0)
    selected = mean_probability[np.arange(choices.size), choices - 1]
    if not np.all(np.isfinite(selected)) or np.any(selected <= 0.0):
        raise ValueError("mean observed-choice probabilities are invalid")
    return float(-np.log(np.clip(selected, 1e-12, 1.0)).sum())


def summarize_nested_seed_budgets(
    probabilities_by_candidate: Mapping[str, np.ndarray],
    choices: Sequence[int] | np.ndarray,
    checkpoints: Sequence[int],
) -> pd.DataFrame:
    """Score nested B prefixes with mean-probability-then-NLL aggregation."""

    banks, observed_choices = _validate_probability_bank(
        probabilities_by_candidate, choices
    )
    seed_count = next(iter(banks.values())).shape[0]
    resolved_checkpoints = [int(value) for value in checkpoints]
    if (
        not resolved_checkpoints
        or resolved_checkpoints != sorted(set(resolved_checkpoints))
        or resolved_checkpoints[0] < 1
        or resolved_checkpoints[-1] > seed_count
    ):
        raise ValueError("checkpoints must be unique increasing prefixes within B")
    rows: list[dict[str, Any]] = []
    for checkpoint in resolved_checkpoints:
        for candidate, probabilities in banks.items():
            total_nll = _nll_from_mean_probabilities(
                probabilities[:checkpoint], observed_choices
            )
            rows.append(
                {
                    "fit_profile_id": candidate,
                    "filter_seed_count": checkpoint,
                    "total_nll": total_nll,
                    "mean_trial_nll": total_nll / float(observed_choices.size),
                    "seed_subset": f"prefix_0_{checkpoint - 1}",
                }
            )
    scores = pd.DataFrame(rows)
    scores["best_total_nll"] = scores.groupby("filter_seed_count")[
        "total_nll"
    ].transform("min")
    scores["delta_nll"] = scores["total_nll"] - scores["best_total_nll"]
    return scores


def summarize_independent_seed_halves(
    probabilities_by_candidate: Mapping[str, np.ndarray],
    choices: Sequence[int] | np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare independent first/second halves of an even seed bank."""

    banks, observed_choices = _validate_probability_bank(
        probabilities_by_candidate, choices
    )
    seed_count = next(iter(banks.values())).shape[0]
    if seed_count < 4 or seed_count % 2:
        raise ValueError("independent-half comparison requires an even B >= 4")
    half = seed_count // 2
    rows: list[dict[str, Any]] = []
    for half_id, start, stop in (("A", 0, half), ("B", half, seed_count)):
        for candidate, probabilities in banks.items():
            total_nll = _nll_from_mean_probabilities(
                probabilities[start:stop], observed_choices
            )
            rows.append(
                {
                    "seed_half": half_id,
                    "fit_profile_id": candidate,
                    "filter_seed_count": half,
                    "total_nll": total_nll,
                    "seed_subset": f"indices_{start}_{stop - 1}",
                }
            )
    scores = pd.DataFrame(rows)
    scores["best_total_nll"] = scores.groupby("seed_half")["total_nll"].transform(
        "min"
    )
    scores["delta_nll"] = scores["total_nll"] - scores["best_total_nll"]
    pivot_nll = scores.pivot(
        index="fit_profile_id", columns="seed_half", values="total_nll"
    )
    pivot_delta = scores.pivot(
        index="fit_profile_id", columns="seed_half", values="delta_nll"
    )
    winner_a = str(
        scores.loc[scores["seed_half"].eq("A")]
        .sort_values(["total_nll", "fit_profile_id"])
        .iloc[0]["fit_profile_id"]
    )
    winner_b = str(
        scores.loc[scores["seed_half"].eq("B")]
        .sort_values(["total_nll", "fit_profile_id"])
        .iloc[0]["fit_profile_id"]
    )
    summary = {
        "seed_count_per_half": half,
        "winner_A": winner_a,
        "winner_B": winner_b,
        "same_winner": winner_a == winner_b,
        "maximum_absolute_candidate_nll_difference": float(
            np.max(np.abs(pivot_nll["A"] - pivot_nll["B"]))
        ),
        "maximum_absolute_candidate_delta_nll_difference": float(
            np.max(np.abs(pivot_delta["A"] - pivot_delta["B"]))
        ),
    }
    return scores, summary


def bootstrap_pairwise_delta_nll(
    probabilities_by_candidate: Mapping[str, np.ndarray],
    choices: Sequence[int] | np.ndarray,
    *,
    replicates: int,
    confidence_level: float,
    bootstrap_seed: int,
) -> pd.DataFrame:
    """Paired seed bootstrap for all candidate delta-NLL comparisons."""

    banks, observed_choices = _validate_probability_bank(
        probabilities_by_candidate, choices
    )
    bootstrap_replicates = int(replicates)
    confidence = float(confidence_level)
    if bootstrap_replicates < 100:
        raise ValueError("bootstrap requires at least 100 replicates")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence_level must lie in (0.5, 1)")
    seed_count = next(iter(banks.values())).shape[0]
    rng = np.random.default_rng(int(bootstrap_seed))
    sample_indices = rng.integers(
        0, seed_count, size=(bootstrap_replicates, seed_count)
    )
    observed_index = observed_choices - 1
    bootstrap_nll: dict[str, np.ndarray] = {}
    point_nll: dict[str, float] = {}
    for candidate, probabilities in banks.items():
        selected = probabilities[
            :, np.arange(observed_choices.size), observed_index
        ]
        bootstrap_selected = selected[sample_indices].mean(axis=1)
        if np.any(bootstrap_selected <= 0.0):
            raise ValueError("bootstrap produced invalid observed-choice probability")
        bootstrap_nll[candidate] = -np.log(
            np.clip(bootstrap_selected, 1e-12, 1.0)
        ).sum(axis=1)
        point_nll[candidate] = _nll_from_mean_probabilities(
            probabilities, observed_choices
        )
    alpha = (1.0 - confidence) / 2.0
    rows: list[dict[str, Any]] = []
    for left, right in combinations(sorted(banks), 2):
        samples = bootstrap_nll[left] - bootstrap_nll[right]
        lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
        rows.append(
            {
                "left_profile_id": left,
                "right_profile_id": right,
                "point_delta_nll_left_minus_right": (
                    point_nll[left] - point_nll[right]
                ),
                "ci_lower": float(lower),
                "ci_upper": float(upper),
                "ci_half_width": float((upper - lower) / 2.0),
                "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
                "bootstrap_replicates": bootstrap_replicates,
                "confidence_level": confidence,
                "bootstrap_seed": int(bootstrap_seed),
            }
        )
    return pd.DataFrame(rows)


def evaluate_seed_convergence(
    checkpoint_scores: pd.DataFrame,
    pairwise_intervals: pd.DataFrame,
    gates: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply predeclared B-prefix and Monte Carlo interval gates."""

    required_scores = {
        "dataset_id",
        "fit_profile_id",
        "filter_seed_count",
        "total_nll",
    }
    required_intervals = {"dataset_id", "ci_half_width", "ci_excludes_zero"}
    if not required_scores.issubset(checkpoint_scores):
        raise ValueError("checkpoint scores are missing required columns")
    if not required_intervals.issubset(pairwise_intervals):
        raise ValueError("pairwise intervals are missing required columns")
    from_count = int(gates["comparison_from_seed_count"])
    to_count = int(gates["comparison_to_seed_count"])
    nll_change_gate = float(gates["maximum_absolute_candidate_nll_change"])
    interval_gate = float(gates["maximum_pairwise_delta_nll_ci_half_width"])
    selected = checkpoint_scores[
        checkpoint_scores["filter_seed_count"].isin([from_count, to_count])
    ]
    pivot = selected.pivot(
        index=["dataset_id", "fit_profile_id"],
        columns="filter_seed_count",
        values="total_nll",
    )
    if list(pivot.columns) != [from_count, to_count] or pivot.isna().any().any():
        raise ValueError("B-prefix comparison is incomplete")
    changes = pivot.reset_index().rename(
        columns={from_count: "from_total_nll", to_count: "to_total_nll"}
    )
    changes["absolute_nll_change"] = np.abs(
        changes["to_total_nll"] - changes["from_total_nll"]
    )
    maximum_change = float(changes["absolute_nll_change"].max())
    maximum_half_width = float(pairwise_intervals["ci_half_width"].max())
    prefix_passed = maximum_change <= nll_change_gate
    interval_passed = maximum_half_width <= interval_gate
    summary = {
        "interpretation": "nested_seed_convergence_not_recovery_evidence",
        "comparison": {"from_B": from_count, "to_B": to_count},
        "gates": {
            "maximum_absolute_candidate_nll_change": nll_change_gate,
            "maximum_pairwise_delta_nll_ci_half_width": interval_gate,
        },
        "observed": {
            "maximum_absolute_candidate_nll_change": maximum_change,
            "median_absolute_candidate_nll_change": float(
                changes["absolute_nll_change"].median()
            ),
            "maximum_pairwise_delta_nll_ci_half_width": maximum_half_width,
            "median_pairwise_delta_nll_ci_half_width": float(
                pairwise_intervals["ci_half_width"].median()
            ),
            "resolved_pair_count": int(
                pairwise_intervals["ci_excludes_zero"].astype(bool).sum()
            ),
            "pair_count": int(len(pairwise_intervals)),
        },
        "prefix_change_gate_passed": bool(prefix_passed),
        "pairwise_interval_gate_passed": bool(interval_passed),
        "budget_status": (
            "seed_convergence_gate_passed"
            if prefix_passed and interval_passed
            else "not_seed_stable"
        ),
        "paired_particle_count_comparison_authorized": bool(
            prefix_passed and interval_passed
        ),
        "formal_recovery_authorized": False,
        "observed_data_fit_authorized": False,
    }
    return changes, summary
