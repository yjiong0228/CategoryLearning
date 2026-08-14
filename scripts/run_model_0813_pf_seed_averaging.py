#!/usr/bin/env python3
"""Calibrate a replicated particle-filter likelihood for model 0813.

The analysis reuses the first three 128-particle filter repeats, adds fixed
repeats through 16, and evaluates nested aggregation plus a predeclared 8-seed
training / 8-seed validation split.  Candidate selection uses log of the mean
of independent likelihood estimates; mean log likelihood is retained only as
a secondary numerical diagnostic.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0813_pf_calibration import (  # noqa: E402
    _atomic_csv,
    _atomic_json,
    _git_head,
    _python_tree_sha256,
    _rank_filter_seed,
    _ranking_cache_path,
    _relative,
    _repo_path,
    _score_ranking_candidate,
    _sha256,
    _worktree_dirty,
)
from scripts.run_model_0813_pf_parameter_recovery import (  # noqa: E402
    _load_subject_frames,
)
from src.Bayesian_state.simulation.config import load_yaml  # noqa: E402
from src.Bayesian_state.utils.seeding import stable_seed  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/specific_models/model_0813_pf_seed_averaging.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase", choices=("run", "summarize", "all"), default="all"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one dataset, three candidates, four particles and four seeds.",
    )
    return parser.parse_args()


def _require_complete_keys(
    scores: pd.DataFrame,
    dataset_ids: Sequence[str],
    profile_ids: Sequence[str],
    seed_indices: Sequence[int],
) -> None:
    expected = {
        (str(dataset_id), str(profile_id), int(seed_index))
        for dataset_id in dataset_ids
        for profile_id in profile_ids
        for seed_index in seed_indices
    }
    observed = set(
        scores[["dataset_id", "fit_profile_id", "filter_repeat"]]
        .assign(
            dataset_id=lambda frame: frame["dataset_id"].astype(str),
            fit_profile_id=lambda frame: frame["fit_profile_id"].astype(str),
            filter_repeat=lambda frame: frame["filter_repeat"].astype(int),
        )
        .itertuples(index=False, name=None)
    )
    missing = expected - observed
    extra = observed - expected
    if missing or extra:
        raise ValueError(
            "score keys do not match the frozen design: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    if len(scores) != len(expected):
        raise ValueError("score table contains duplicate design keys")


def _reuse_scores(
    path: Path,
    dataset_ids: Sequence[str],
    profile_ids: Sequence[str],
    particle_count: int,
    available_seed_indices: Sequence[int],
) -> pd.DataFrame:
    scores = pd.read_csv(path)
    selected = scores.loc[
        scores["dataset_id"].astype(str).isin({str(value) for value in dataset_ids})
        & scores["fit_profile_id"].astype(str).isin(
            {str(value) for value in profile_ids}
        )
        & scores["particle_count"].astype(int).eq(int(particle_count))
        & scores["filter_repeat"].astype(int).isin(
            {int(value) for value in available_seed_indices}
        )
    ].copy()
    if len(selected):
        _require_complete_keys(
            selected, dataset_ids, profile_ids, available_seed_indices
        )
    selected["source"] = "phase1_128_particle_reuse"
    return selected


def _score_new_repeat(
    *,
    output: Path,
    dataset_path: Path,
    base_config: Mapping[str, Any],
    base_path: Path,
    dataset_paths: Mapping[str, Path],
    profile: Mapping[str, Any],
    particle_count: int,
    resample_threshold_fraction: float,
    filter_repeat: int,
    base_seed: int,
    force: bool,
) -> dict[str, Any]:
    row = _score_ranking_candidate(
        output=output,
        dataset_path=dataset_path,
        base_config=base_config,
        base_path=base_path,
        dataset_paths=dataset_paths,
        profile=profile,
        particle_count=particle_count,
        resample_threshold_fraction=resample_threshold_fraction,
        filter_repeat=filter_repeat,
        base_seed=base_seed,
        force=force,
    )
    row["source"] = "phase1b_seed_averaging_new"
    row["analysis_role"] = "phase1b_seed_averaging"
    cache_path = _ranking_cache_path(
        output,
        dataset_path.stem,
        int(particle_count),
        int(filter_repeat),
        str(profile["profile_id"]),
    )
    _atomic_json(cache_path, row)
    return row


def replicated_log_likelihood(log_likelihoods: Sequence[float]) -> dict[str, float]:
    """Aggregate independent likelihood estimates on a stable log scale."""

    values = np.asarray(log_likelihoods, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("log_likelihoods must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("log_likelihoods must all be finite")
    count = int(values.size)
    aggregate = float(logsumexp(values) - math.log(count))
    normalized = np.exp(values - logsumexp(values))
    effective_count = float(1.0 / np.sum(np.square(normalized)))
    maximum = float(np.max(values))
    scaled = np.exp(values - maximum)
    scaled_mean = float(np.mean(scaled))
    if count > 1 and scaled_mean > 0.0:
        mcse = float(
            np.std(scaled, ddof=1) / math.sqrt(count) / scaled_mean
        )
        log_sd = float(np.std(values, ddof=1))
    else:
        mcse = float("nan")
        log_sd = float("nan")
    return {
        "aggregate_log_likelihood": aggregate,
        "aggregate_total_nll": -aggregate,
        "mean_log_likelihood": float(np.mean(values)),
        "mean_total_nll": float(-np.mean(values)),
        "median_log_likelihood": float(np.median(values)),
        "log_likelihood_sd": log_sd,
        "aggregate_log_likelihood_mcse": mcse,
        "effective_seed_count": effective_count,
        "effective_seed_fraction": effective_count / count,
        "best_seed_weight": float(np.max(normalized)),
    }


def aggregate_candidate_scores(
    scores: pd.DataFrame,
    seed_indices: Sequence[int],
    *,
    panel_id: str,
    seed_count_label: int | None = None,
) -> pd.DataFrame:
    seeds = tuple(int(value) for value in seed_indices)
    if not seeds:
        raise ValueError("seed_indices cannot be empty")
    selected = scores.loc[
        scores["filter_repeat"].astype(int).isin(set(seeds))
    ].copy()
    rows: list[dict[str, Any]] = []
    for (dataset_id, profile_id), frame in selected.groupby(
        ["dataset_id", "fit_profile_id"], sort=True
    ):
        frame = frame.sort_values("filter_repeat")
        observed = tuple(frame["filter_repeat"].astype(int))
        if observed != tuple(sorted(seeds)):
            raise ValueError(
                f"incomplete seed panel for {dataset_id}/{profile_id}: {observed}"
            )
        metrics = replicated_log_likelihood(frame["log_likelihood"].to_numpy())
        rows.append(
            {
                "panel_id": str(panel_id),
                "seed_count_label": int(seed_count_label or len(seeds)),
                "seed_count": int(len(seeds)),
                "seed_indices": ",".join(str(value) for value in seeds),
                "dataset_id": str(dataset_id),
                "fit_profile_id": str(profile_id),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def _top_k(frame: pd.DataFrame, k: int) -> set[str]:
    return set(
        frame.nsmallest(int(k), "aggregate_total_nll")[
            "fit_profile_id"
        ].astype(str)
    )


def compare_aggregate_panels(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    comparison_id: str,
    comparison_seed_count: int,
    top_k: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    left_datasets = set(left["dataset_id"].astype(str))
    right_datasets = set(right["dataset_id"].astype(str))
    if left_datasets != right_datasets:
        raise ValueError("aggregate panels contain different datasets")
    for dataset_id in sorted(left_datasets):
        a = left.loc[left["dataset_id"].astype(str).eq(dataset_id)].sort_values(
            "fit_profile_id"
        )
        b = right.loc[right["dataset_id"].astype(str).eq(dataset_id)].sort_values(
            "fit_profile_id"
        )
        if not a["fit_profile_id"].reset_index(drop=True).equals(
            b["fit_profile_id"].reset_index(drop=True)
        ):
            raise ValueError("aggregate candidate panels do not align")
        rho = float(
            spearmanr(
                a["aggregate_total_nll"].to_numpy(dtype=float),
                b["aggregate_total_nll"].to_numpy(dtype=float),
            ).statistic
        )
        left_winner = str(
            a.loc[a["aggregate_total_nll"].idxmin(), "fit_profile_id"]
        )
        right_winner = str(
            b.loc[b["aggregate_total_nll"].idxmin(), "fit_profile_id"]
        )
        left_top = _top_k(a, top_k)
        right_top = _top_k(b, top_k)
        union = left_top | right_top
        rows.append(
            {
                "comparison_id": str(comparison_id),
                "comparison_seed_count": int(comparison_seed_count),
                "left_panel_id": str(a["panel_id"].iloc[0]),
                "right_panel_id": str(b["panel_id"].iloc[0]),
                "left_seed_count": int(a["seed_count"].iloc[0]),
                "right_seed_count": int(b["seed_count"].iloc[0]),
                "dataset_id": dataset_id,
                "candidate_rank_spearman": rho,
                "left_winner": left_winner,
                "right_winner": right_winner,
                "winner_agreement": bool(left_winner == right_winner),
                "left_top_k": ",".join(sorted(left_top)),
                "right_top_k": ",".join(sorted(right_top)),
                "top_k_jaccard": float(
                    len(left_top & right_top) / len(union)
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_equivalence_sets(
    scores: pd.DataFrame,
    seed_indices: Sequence[int],
    *,
    bootstrap_repeats: int,
    confidence: float,
    base_seed: int,
) -> pd.DataFrame:
    seeds = tuple(sorted(int(value) for value in seed_indices))
    if bootstrap_repeats < 100:
        raise ValueError("bootstrap_repeats must be at least 100")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    alpha = (1.0 - float(confidence)) / 2.0
    aggregate = aggregate_candidate_scores(
        scores, seeds, panel_id="all_seeds", seed_count_label=len(seeds)
    )
    rows: list[dict[str, Any]] = []
    for dataset_id, dataset_aggregate in aggregate.groupby(
        "dataset_id", sort=True
    ):
        best_row = dataset_aggregate.loc[
            dataset_aggregate["aggregate_total_nll"].idxmin()
        ]
        best_profile = str(best_row["fit_profile_id"])
        dataset_raw = scores.loc[
            scores["dataset_id"].astype(str).eq(str(dataset_id))
            & scores["filter_repeat"].astype(int).isin(set(seeds))
        ]
        arrays = {
            str(profile_id): frame.sort_values("filter_repeat")[
                "log_likelihood"
            ].to_numpy(dtype=float)
            for profile_id, frame in dataset_raw.groupby(
                "fit_profile_id", sort=True
            )
        }
        if any(len(values) != len(seeds) for values in arrays.values()):
            raise ValueError("bootstrap panel is incomplete")
        rng = np.random.default_rng(
            stable_seed(
                {
                    "seed_role": "model0813_phase1b_equivalence_bootstrap",
                    "base_seed": int(base_seed),
                    "dataset_id": str(dataset_id),
                }
            )
        )
        indices = rng.integers(
            0, len(seeds), size=(int(bootstrap_repeats), len(seeds))
        )
        best_values = arrays[best_profile]
        best_boot = logsumexp(best_values[indices], axis=1) - math.log(len(seeds))
        for _, item in dataset_aggregate.sort_values(
            "aggregate_total_nll"
        ).iterrows():
            profile_id = str(item["fit_profile_id"])
            values = arrays[profile_id]
            candidate_boot = (
                logsumexp(values[indices], axis=1) - math.log(len(seeds))
            )
            delta_boot = best_boot - candidate_boot
            delta = float(
                best_row["aggregate_log_likelihood"]
                - item["aggregate_log_likelihood"]
            )
            paired_mean_delta = best_values - values
            rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "best_profile_id": best_profile,
                    "fit_profile_id": profile_id,
                    "seed_count": int(len(seeds)),
                    "aggregate_total_nll": float(item["aggregate_total_nll"]),
                    "delta_nll_from_selected_best": delta,
                    "bootstrap_ci_low": float(np.quantile(delta_boot, alpha)),
                    "bootstrap_ci_high": float(
                        np.quantile(delta_boot, 1.0 - alpha)
                    ),
                    "equivalent_to_selected_best": bool(
                        np.quantile(delta_boot, alpha) <= 0.0
                    ),
                    "paired_mean_log_nll_delta": float(
                        np.mean(paired_mean_delta)
                    ),
                    "paired_mean_log_nll_delta_se": float(
                        np.std(paired_mean_delta, ddof=1)
                        / math.sqrt(len(seeds))
                    ),
                    "effective_seed_fraction": float(
                        item["effective_seed_fraction"]
                    ),
                    "aggregate_log_likelihood_mcse": float(
                        item["aggregate_log_likelihood_mcse"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_seed_averaging(
    scores: pd.DataFrame,
    *,
    aggregation_seed_counts: Sequence[int],
    training_seed_indices: Sequence[int],
    validation_seed_indices: Sequence[int],
    top_k: int,
    bootstrap_repeats: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
    gates: Mapping[str, float],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    counts = tuple(sorted(int(value) for value in aggregation_seed_counts))
    total = counts[-1]
    if counts != tuple(sorted(set(counts))) or counts[0] < 2:
        raise ValueError("aggregation_seed_counts must be unique and start at >=2")
    if any(value % 2 for value in counts):
        raise ValueError("every aggregation seed count must be even")
    if tuple(sorted(training_seed_indices + validation_seed_indices)) != tuple(
        range(total)
    ):
        raise ValueError("training and validation seeds must partition all repeats")
    if set(training_seed_indices) & set(validation_seed_indices):
        raise ValueError("training and validation seed panels must be disjoint")

    aggregate_frames: list[pd.DataFrame] = []
    split_frames: list[pd.DataFrame] = []
    running_frames: list[pd.DataFrame] = []
    prefix_panels: dict[int, pd.DataFrame] = {}
    for count in counts:
        prefix = tuple(range(count))
        prefix_panel = aggregate_candidate_scores(
            scores,
            prefix,
            panel_id=f"prefix_{count}",
            seed_count_label=count,
        )
        prefix_panels[count] = prefix_panel
        aggregate_frames.append(prefix_panel)
        half = count // 2
        left = aggregate_candidate_scores(
            scores,
            tuple(range(half)),
            panel_id=f"split_{count}_left",
            seed_count_label=count,
        )
        right = aggregate_candidate_scores(
            scores,
            tuple(range(half, count)),
            panel_id=f"split_{count}_right",
            seed_count_label=count,
        )
        aggregate_frames.extend((left, right))
        split_frames.append(
            compare_aggregate_panels(
                left,
                right,
                comparison_id=f"split_half_{count}",
                comparison_seed_count=count,
                top_k=top_k,
            )
        )
        if count != counts[0]:
            previous = counts[counts.index(count) - 1]
            running_frames.append(
                compare_aggregate_panels(
                    prefix_panels[previous],
                    prefix_panel,
                    comparison_id=f"prefix_{previous}_vs_{count}",
                    comparison_seed_count=count,
                    top_k=top_k,
                )
            )

    training = aggregate_candidate_scores(
        scores,
        training_seed_indices,
        panel_id="training_8",
        seed_count_label=total,
    )
    validation = aggregate_candidate_scores(
        scores,
        validation_seed_indices,
        panel_id="validation_8",
        seed_count_label=total,
    )
    aggregate_frames.extend((training, validation))
    training_validation = compare_aggregate_panels(
        training,
        validation,
        comparison_id="training_8_vs_validation_8",
        comparison_seed_count=total,
        top_k=top_k,
    )
    final_split = pd.concat(split_frames, ignore_index=True)
    final_running = pd.concat(running_frames, ignore_index=True)
    aggregate_table = pd.concat(aggregate_frames, ignore_index=True)
    equivalence = bootstrap_equivalence_sets(
        scores,
        tuple(range(total)),
        bootstrap_repeats=bootstrap_repeats,
        confidence=bootstrap_confidence,
        base_seed=bootstrap_seed,
    )

    full = prefix_panels[total]
    dataset_rows: list[dict[str, Any]] = []
    for dataset_id, frame in full.groupby("dataset_id", sort=True):
        nll_range = float(
            frame["aggregate_total_nll"].max()
            - frame["aggregate_total_nll"].min()
        )
        median_mcse = float(frame["aggregate_log_likelihood_mcse"].median())
        eq_frame = equivalence.loc[
            equivalence["dataset_id"].astype(str).eq(str(dataset_id))
        ]
        split_row = training_validation.loc[
            training_validation["dataset_id"].astype(str).eq(str(dataset_id))
        ].iloc[0]
        final_running_row = final_running.loc[
            final_running["comparison_seed_count"].astype(int).eq(total)
            & final_running["dataset_id"].astype(str).eq(str(dataset_id))
        ].iloc[0]
        method_rho = float(
            spearmanr(
                frame["aggregate_total_nll"].to_numpy(dtype=float),
                frame["mean_total_nll"].to_numpy(dtype=float),
            ).statistic
        )
        logmeanexp_winner = str(
            frame.loc[frame["aggregate_total_nll"].idxmin(), "fit_profile_id"]
        )
        mean_log_winner = str(
            frame.loc[frame["mean_total_nll"].idxmin(), "fit_profile_id"]
        )
        dataset_rows.append(
            {
                "dataset_id": str(dataset_id),
                "selected_profile_id": str(eq_frame["best_profile_id"].iloc[0]),
                "equivalence_set_size": int(
                    eq_frame["equivalent_to_selected_best"].astype(bool).sum()
                ),
                "candidate_nll_range": nll_range,
                "median_aggregate_log_likelihood_mcse": median_mcse,
                "noise_to_signal_ratio": (
                    median_mcse / nll_range if nll_range > 0.0 else float("inf")
                ),
                "median_effective_seed_fraction": float(
                    frame["effective_seed_fraction"].median()
                ),
                "minimum_effective_seed_fraction": float(
                    frame["effective_seed_fraction"].min()
                ),
                "training_validation_rank_spearman": float(
                    split_row["candidate_rank_spearman"]
                ),
                "training_validation_winner_agreement": bool(
                    split_row["winner_agreement"]
                ),
                "training_validation_top_k_jaccard": float(
                    split_row["top_k_jaccard"]
                ),
                "prefix_8_to_16_rank_spearman": float(
                    final_running_row["candidate_rank_spearman"]
                ),
                "prefix_8_to_16_winner_agreement": bool(
                    final_running_row["winner_agreement"]
                ),
                "logmeanexp_vs_meanlog_rank_spearman": method_rho,
                "logmeanexp_winner": logmeanexp_winner,
                "mean_log_winner": mean_log_winner,
                "aggregation_method_winner_agreement": bool(
                    logmeanexp_winner == mean_log_winner
                ),
            }
        )
    dataset_summary = pd.DataFrame(dataset_rows)

    split_rhos = training_validation["candidate_rank_spearman"].to_numpy(
        dtype=float
    )
    running_final = final_running.loc[
        final_running["comparison_seed_count"].astype(int).eq(total)
    ]
    all_gates = {
        "final_split_median_rank_pass": bool(
            np.median(split_rhos)
            >= float(gates["final_split_median_rank_spearman"])
        ),
        "final_split_minimum_rank_pass": bool(
            np.min(split_rhos)
            >= float(gates["final_split_minimum_rank_spearman"])
        ),
        "final_split_winner_pass": bool(
            training_validation["winner_agreement"].astype(float).mean()
            >= float(gates["final_split_winner_agreement"])
        ),
        "final_split_top_k_pass": bool(
            training_validation["top_k_jaccard"].mean()
            >= float(gates["final_split_mean_top_k_jaccard"])
        ),
        "running_8_to_16_rank_pass": bool(
            running_final["candidate_rank_spearman"].median()
            >= float(gates["running_8_to_16_median_rank_spearman"])
        ),
        "running_8_to_16_winner_pass": bool(
            running_final["winner_agreement"].astype(float).mean()
            >= float(gates["running_8_to_16_winner_agreement"])
        ),
        "effective_seed_fraction_pass": bool(
            full["effective_seed_fraction"].median()
            >= float(gates["median_effective_seed_fraction"])
        ),
        "aggregate_mcse_pass": bool(
            full["aggregate_log_likelihood_mcse"].median()
            <= float(gates["median_aggregate_log_likelihood_mcse"])
        ),
        "noise_to_signal_pass": bool(
            dataset_summary["noise_to_signal_ratio"].median()
            <= float(gates["maximum_noise_to_signal_ratio"])
        ),
    }
    summary = {
        "status": (
            "replicated_likelihood_gate_passed"
            if all(all_gates.values())
            else "replicated_likelihood_gate_failed"
        ),
        "all_stability_gates_pass": bool(all(all_gates.values())),
        "particle_count": int(scores["particle_count"].iloc[0]),
        "total_filter_seed_repeats": int(total),
        "training_seed_indices": [int(value) for value in training_seed_indices],
        "validation_seed_indices": [
            int(value) for value in validation_seed_indices
        ],
        "dataset_n": int(scores["dataset_id"].nunique()),
        "candidate_n": int(scores["fit_profile_id"].nunique()),
        "final_split_median_rank_spearman": float(np.median(split_rhos)),
        "final_split_minimum_rank_spearman": float(np.min(split_rhos)),
        "final_split_winner_agreement": float(
            training_validation["winner_agreement"].astype(float).mean()
        ),
        "final_split_mean_top_k_jaccard": float(
            training_validation["top_k_jaccard"].mean()
        ),
        "running_8_to_16_median_rank_spearman": float(
            running_final["candidate_rank_spearman"].median()
        ),
        "running_8_to_16_winner_agreement": float(
            running_final["winner_agreement"].astype(float).mean()
        ),
        "median_effective_seed_fraction": float(
            full["effective_seed_fraction"].median()
        ),
        "minimum_effective_seed_fraction": float(
            full["effective_seed_fraction"].min()
        ),
        "median_aggregate_log_likelihood_mcse": float(
            full["aggregate_log_likelihood_mcse"].median()
        ),
        "median_noise_to_signal_ratio": float(
            dataset_summary["noise_to_signal_ratio"].median()
        ),
        "median_logmeanexp_vs_meanlog_rank_spearman": float(
            dataset_summary["logmeanexp_vs_meanlog_rank_spearman"].median()
        ),
        "aggregation_method_winner_agreement": float(
            dataset_summary["aggregation_method_winner_agreement"]
            .astype(float)
            .mean()
        ),
        "equivalence_set_sizes": {
            str(row["dataset_id"]): int(row["equivalence_set_size"])
            for row in dataset_rows
        },
        "selected_profiles": {
            str(row["dataset_id"]): str(row["selected_profile_id"])
            for row in dataset_rows
        },
        "aggregation_definition": (
            "log(mean(exp(independent PF log-likelihood estimates)))"
        ),
        "equivalence_definition": (
            "candidate is numerically indistinguishable from the selected best "
            "when the paired seed-bootstrap delta-NLL interval includes zero"
        ),
        "independent_unit": (
            "none; datasets are fixed technical calibration cases and PF seeds "
            "are Monte Carlo repeats"
        ),
        "stability_gates": dict(gates),
        "gate_checks": all_gates,
    }
    return (
        aggregate_table,
        final_split,
        final_running,
        equivalence,
        dataset_summary,
        summary,
    )


def _write_chart_map(output: Path, summary: Mapping[str, Any]) -> None:
    conclusion = (
        "Sixteen-seed replicated PF likelihood passes every frozen gate."
        if summary["all_stability_gates_pass"]
        else "Sixteen-seed replicated PF likelihood does not pass every frozen gate."
    )
    content = f"""# Figure contract and chart map

Core conclusion: {conclusion}

- Figure archetype: quantitative grid.
- Target: technical audit report; Python/matplotlib; 7.2 x 6.6 inches; PNG at 300 dpi.
- Hero evidence: independent 8-seed training versus 8-seed validation candidate-rank agreement.
- Validation evidence: nested prefix convergence, final candidate delta NLLs, and effective-seed/MCSE diagnostics.
- Statistics: three fixed synthetic calibration datasets; nine candidates; PF seeds are technical repeats, not independent subjects.
- Image integrity: every declared dataset, candidate and seed-count stage is retained; display jitter is deterministic and does not alter values.

| Panel | Analytical question | Form | Non-color encoding |
|---|---|---|---|
| a | Do independent likelihood panels rank candidates similarly? | median line plus all dataset split-half rho values | circles and threshold line |
| b | Does the nested aggregate stop changing as seeds accumulate? | median line plus all dataset prefix-comparison rho values | squares and threshold line |
| c | Which candidates remain close to the full 16-seed winner? | dataset lines of delta NLL by candidate | distinct markers and line styles |
| d | Is aggregation supported by multiple effective seeds with tolerable MCSE? | candidate scatter with frozen gates | dataset-specific markers and open fills |

Reviewer risk: low rank correlation may arise from genuinely near-equivalent candidates as well as Monte Carlo noise. Log-mean-exp is appropriate only insofar as each saved PF normalizing-constant estimate is a valid likelihood estimate; mean-log results are retained as a sensitivity diagnostic. The selected-best bootstrap interval is descriptive because the reference candidate is selected on the same 16 seeds.
"""
    (output / "chart_map.md").write_text(content, encoding="utf-8")


def _write_figure(
    output: Path,
    split: pd.DataFrame,
    running: pd.DataFrame,
    equivalence: pd.DataFrame,
    aggregate: pd.DataFrame,
    gates: Mapping[str, float],
    filename: str,
) -> Path:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    blue = "#3B6FB6"
    orange = "#D9822B"
    olive = "#7A8F3A"
    charcoal = "#252A34"
    grey = "#A8ADB5"
    grid = "#E6E9ED"
    colors = (blue, orange, olive)
    markers = ("o", "s", "^")
    lines = ("-", "--", "-.")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    split_counts = sorted(split["comparison_seed_count"].astype(int).unique())
    for index, count in enumerate(split_counts):
        values = split.loc[
            split["comparison_seed_count"].astype(int).eq(count),
            "candidate_rank_spearman",
        ].to_numpy(dtype=float)
        jitter = np.linspace(-0.05, 0.05, len(values))
        ax_a.scatter(
            np.full(len(values), index) + jitter,
            values,
            marker="o",
            facecolors="white",
            edgecolors=blue,
            s=24,
            linewidths=0.8,
            zorder=3,
        )
        ax_a.plot(index, np.median(values), marker="o", color=blue, markersize=5)
    ax_a.plot(
        np.arange(len(split_counts)),
        [
            split.loc[
                split["comparison_seed_count"].astype(int).eq(count),
                "candidate_rank_spearman",
            ].median()
            for count in split_counts
        ],
        color=blue,
        linewidth=1.3,
    )
    ax_a.axhline(
        float(gates["final_split_median_rank_spearman"]),
        color=charcoal,
        linestyle="--",
        linewidth=1.0,
        label="Final median-rank gate",
    )
    ax_a.axhline(0.0, color=grey, linewidth=0.7)
    ax_a.set_xticks(np.arange(len(split_counts)), [str(value) for value in split_counts])
    ax_a.set_ylim(-1.05, 1.05)
    ax_a.set_xlabel("Total seeds available to the split")
    ax_a.set_ylabel("Half-panel rank Spearman rho")
    ax_a.set_title("Independent split-half ranking", loc="left", fontsize=9)
    ax_a.grid(axis="y", color=grid, linewidth=0.7)
    ax_a.legend(fontsize=6.2, loc="lower right")

    running_counts = sorted(running["comparison_seed_count"].astype(int).unique())
    for index, count in enumerate(running_counts):
        values = running.loc[
            running["comparison_seed_count"].astype(int).eq(count),
            "candidate_rank_spearman",
        ].to_numpy(dtype=float)
        jitter = np.linspace(-0.05, 0.05, len(values))
        ax_b.scatter(
            np.full(len(values), index) + jitter,
            values,
            marker="s",
            facecolors="white",
            edgecolors=orange,
            s=23,
            linewidths=0.8,
            zorder=3,
        )
        ax_b.plot(index, np.median(values), marker="s", color=orange, markersize=5)
    ax_b.plot(
        np.arange(len(running_counts)),
        [
            running.loc[
                running["comparison_seed_count"].astype(int).eq(count),
                "candidate_rank_spearman",
            ].median()
            for count in running_counts
        ],
        color=orange,
        linewidth=1.3,
    )
    ax_b.axhline(
        float(gates["running_8_to_16_median_rank_spearman"]),
        color=charcoal,
        linestyle="--",
        linewidth=1.0,
        label="8-to-16 median-rank gate",
    )
    ax_b.axhline(0.0, color=grey, linewidth=0.7)
    ax_b.set_xticks(
        np.arange(len(running_counts)),
        [f"{value // 2}->{value}" for value in running_counts],
    )
    ax_b.set_ylim(-1.05, 1.05)
    ax_b.set_xlabel("Nested prefix comparison")
    ax_b.set_ylabel("Candidate-rank Spearman rho")
    ax_b.set_title("Running aggregate convergence", loc="left", fontsize=9)
    ax_b.grid(axis="y", color=grid, linewidth=0.7)
    ax_b.legend(fontsize=6.2, loc="lower right")

    profile_order = sorted(equivalence["fit_profile_id"].astype(str).unique())
    for index, (dataset_id, frame) in enumerate(
        equivalence.groupby("dataset_id", sort=True)
    ):
        frame = frame.set_index("fit_profile_id").reindex(profile_order)
        ax_c.plot(
            np.arange(len(profile_order)),
            frame["delta_nll_from_selected_best"].to_numpy(dtype=float),
            color=colors[index % len(colors)],
            marker=markers[index % len(markers)],
            linestyle=lines[index % len(lines)],
            linewidth=1.1,
            markersize=4,
            label=str(dataset_id).split("_subject")[0],
        )
    ax_c.axhline(0.0, color=grey, linewidth=0.7)
    ax_c.set_xticks(np.arange(len(profile_order)), profile_order, rotation=45)
    ax_c.set_ylabel("16-seed aggregate Delta NLL from winner")
    ax_c.set_title("Final candidate separation", loc="left", fontsize=9)
    ax_c.grid(axis="y", color=grid, linewidth=0.7)
    ax_c.legend(fontsize=6.2, loc="upper right")

    full = aggregate.loc[aggregate["panel_id"].eq("prefix_16")]
    if full.empty:
        maximum = int(aggregate["seed_count"].max())
        full = aggregate.loc[
            aggregate["panel_id"].eq(f"prefix_{maximum}")
        ]
    for index, (dataset_id, frame) in enumerate(full.groupby("dataset_id", sort=True)):
        ax_d.scatter(
            frame["effective_seed_fraction"],
            frame["aggregate_log_likelihood_mcse"],
            marker=markers[index % len(markers)],
            facecolors="none",
            edgecolors=colors[index % len(colors)],
            s=30,
            linewidths=0.9,
            label=str(dataset_id).split("_subject")[0],
        )
    ax_d.axvline(
        float(gates["median_effective_seed_fraction"]),
        color=charcoal,
        linestyle="--",
        linewidth=1.0,
        label="Median gates",
    )
    ax_d.axhline(
        float(gates["median_aggregate_log_likelihood_mcse"]),
        color=charcoal,
        linestyle="--",
        linewidth=1.0,
    )
    ax_d.set_xlim(0.0, 1.03)
    ax_d.set_xlabel("Effective seed count / nominal seeds")
    ax_d.set_ylabel("Aggregate log-likelihood MCSE")
    ax_d.set_title("Replicate support and Monte Carlo error", loc="left", fontsize=9)
    ax_d.grid(color=grid, linewidth=0.7)
    ax_d.legend(fontsize=5.8, loc="upper right", ncol=2)

    for label, axis in zip("abcd", axes.flat):
        axis.text(
            -0.16,
            1.08,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
        )
    fig.suptitle(
        "0813 replicated particle-filter likelihood calibration",
        x=0.01,
        ha="left",
        fontsize=11,
        fontweight="bold",
        color=charcoal,
    )
    destination = output / filename
    fig.savefig(destination, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return destination


def _format(value: Any, digits: int = 3) -> str:
    number = float(value)
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def _write_readme(
    output: Path,
    summary: Mapping[str, Any],
    dataset_summary: pd.DataFrame,
) -> None:
    passed = bool(summary["all_stability_gates_pass"])
    decision = (
        "16-seed replicated likelihood passed every frozen gate and can be used "
        "as the numerical baseline for the next bounded candidate comparison."
        if passed
        else "16-seed replicated likelihood did not pass every frozen gate; it "
        "must not yet be used for unconstrained hyperparameter selection."
    )
    rows = []
    for item in dataset_summary.to_dict(orient="records"):
        rows.append(
            "| {dataset} | {winner} | {eq} | {split} | {same} | {running} | "
            "{effective} | {mcse} | {noise} |".format(
                dataset=str(item["dataset_id"]),
                winner=str(item["selected_profile_id"]),
                eq=int(item["equivalence_set_size"]),
                split=_format(item["training_validation_rank_spearman"]),
                same="yes" if item["training_validation_winner_agreement"] else "no",
                running=_format(item["prefix_8_to_16_rank_spearman"]),
                effective=_format(item["median_effective_seed_fraction"]),
                mcse=_format(item["median_aggregate_log_likelihood_mcse"]),
                noise=_format(item["noise_to_signal_ratio"]),
            )
        )
    failed = [
        key for key, value in summary["gate_checks"].items() if not bool(value)
    ]
    content = f"""# Phase 1b: 16-seed replicated PF likelihood

## Technical conclusion

**{decision}**

The primary score is `log(mean(exp(log_likelihood_seed)))` over independent PF
repeats. Seeds 0-7 form a fixed training panel and seeds 8-15 form a disjoint
validation panel. Mean log likelihood is retained as a sensitivity diagnostic
and is not substituted after seeing the result.

- Training-versus-validation candidate-rank rho: median
  `{_format(summary['final_split_median_rank_spearman'])}`, minimum
  `{_format(summary['final_split_minimum_rank_spearman'])}`.
- Training-versus-validation winner agreement:
  `{_format(summary['final_split_winner_agreement'])}`; mean top-3 Jaccard
  `{_format(summary['final_split_mean_top_k_jaccard'])}`.
- Prefix 8-versus-16 rank rho median:
  `{_format(summary['running_8_to_16_median_rank_spearman'])}`; winner agreement
  `{_format(summary['running_8_to_16_winner_agreement'])}`.
- Median effective-seed fraction:
  `{_format(summary['median_effective_seed_fraction'])}`; median aggregate
  log-likelihood MCSE `{_format(summary['median_aggregate_log_likelihood_mcse'])}`.
- Primary log-mean-exp versus secondary mean-log candidate-rank rho median:
  `{_format(summary['median_logmeanexp_vs_meanlog_rank_spearman'])}`; winner
  agreement `{_format(summary['aggregation_method_winner_agreement'])}`.
- Failed gates: `{', '.join(failed) if failed else 'none'}`.

| dataset | selected profile | equivalence-set size | train/validation rho | same winner | prefix 8/16 rho | median effective-seed fraction | median MCSE | noise/signal |
|---|---|---:|---:|:---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## What an equivalence set means

For each fixed dataset, the full 16-seed aggregate selects a reference winner.
Candidate and winner likelihoods are then resampled with the same seed indices.
A candidate remains in the numerical equivalence set when the paired bootstrap
interval for `candidate NLL - selected-winner NLL` includes zero. This is a
descriptive numerical diagnostic, not a post-selection confidence interval for
psychological truth.

## Files

- `seed_scores.csv`: all three reused and thirteen new repeats per dataset-candidate.
- `aggregate_candidate_scores.csv`: prefix, split, training and validation likelihood aggregates.
- `split_half_stability.csv`: independent half-panel ranking and top-k comparisons.
- `running_prefix_stability.csv`: nested 2-to-4, 4-to-8 and 8-to-16 convergence.
- `candidate_equivalence_sets.csv`: paired-bootstrap delta-NLL intervals.
- `aggregation_method_sensitivity.csv`: log-mean-exp versus mean-log ranking diagnostic.
- `dataset_summary.csv`, `summary.json`: decision-level metrics and frozen gates.
- `seed_averaging_overview.png`, `chart_map.md`: quantitative overview and figure contract.
- `analysis_manifest.json`, `analysis_config_snapshot.json`: complete provenance.
- `VALIDATION.md`: independent recomputation, data completeness and figure QA.
- `artifact.json`: canonical technical-report input; HTML build status is recorded separately.

## Interpretation boundary

These are three fixed technical calibration trajectories, not an experimental
sample. PF seeds estimate numerical uncertainty and do not increase behavioral
`n`. Passing would authorize only a bounded, seed-averaged candidate objective;
it would not by itself establish parameter recovery or psychological uniqueness.
Failing would motivate more repeats, a lower-variance likelihood estimator, or a
smaller and more widely separated candidate bank rather than a forced winner.
"""
    (output / "README.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")

    config_path = args.config.resolve()
    config = load_yaml(config_path)
    calibration = deepcopy(dict(config["calibration"]))
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _repo_path(config["output_dir"])
    )
    if args.smoke:
        output = output / "smoke"
        calibration.update(
            {
                "dataset_ids": calibration["dataset_ids"][:1],
                "candidate_profile_ids": calibration["candidate_profile_ids"][:3],
                "particle_count": 4,
                "total_filter_seed_repeats": 4,
                "aggregation_seed_counts": [2, 4],
                "training_seed_indices": [0, 1],
                "validation_seed_indices": [2, 3],
                "bootstrap_repeats": 100,
                "n_jobs": 2,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    base_path = _repo_path(config["base_simulation_config"])
    profiles_path = _repo_path(config["candidate_profiles"])
    synthetic_dir = _repo_path(config["synthetic_dir"])
    existing_path = _repo_path(config["existing_ranking_scores"])
    base_config = load_yaml(base_path)
    profiles = list(json.loads(profiles_path.read_text(encoding="utf-8")))
    dataset_ids = [str(value) for value in calibration["dataset_ids"]]
    profile_ids = [str(value) for value in calibration["candidate_profile_ids"]]
    selected_profiles = [
        profile for profile in profiles if str(profile["profile_id"]) in profile_ids
    ]
    if [str(item["profile_id"]) for item in selected_profiles] != profile_ids:
        raise ValueError("candidate profiles are missing or out of frozen order")
    for dataset_id in dataset_ids:
        if not (synthetic_dir / f"{dataset_id}.npz").exists():
            raise FileNotFoundError(f"missing calibration dataset: {dataset_id}")
    _, dataset_paths = _load_subject_frames(base_config, base_path, [103])

    total_repeats = int(calibration["total_filter_seed_repeats"])
    particle_count = int(calibration["particle_count"])
    snapshot = deepcopy(config)
    snapshot["calibration"] = calibration
    _atomic_json(output / "analysis_config_snapshot.json", snapshot)
    manifest = {
        "analysis_id": str(config["analysis_id"]),
        "scope": str(config["scope"]),
        "status": "configured",
        "config": _relative(config_path),
        "config_sha256": _sha256(config_path),
        "base_simulation_config": _relative(base_path),
        "base_simulation_config_sha256": _sha256(base_path),
        "candidate_profiles": _relative(profiles_path),
        "candidate_profiles_sha256": _sha256(profiles_path),
        "existing_ranking_scores": _relative(existing_path),
        "existing_ranking_scores_sha256": _sha256(existing_path),
        "runner": _relative(Path(__file__).resolve()),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "calibration_runner_sha256": _sha256(
            ROOT / "scripts/run_model_0813_pf_calibration.py"
        ),
        "bayesian_state_python_tree_sha256": _python_tree_sha256(
            ROOT / "src/Bayesian_state"
        ),
        "repository_head": _git_head(),
        "worktree_dirty": _worktree_dirty(),
        "design": calibration,
        "smoke": bool(args.smoke),
        "primary_aggregation": (
            "log mean of independent particle-filter likelihood estimates"
        ),
        "interpretation_boundary": (
            "technical likelihood calibration; no population, recovery, or "
            "psychological uniqueness inference"
        ),
    }
    _atomic_json(output / "analysis_manifest.json", manifest)

    if args.phase in {"run", "all"}:
        reusable = pd.DataFrame()
        reused_seeds: list[int] = []
        if not args.smoke:
            source = pd.read_csv(existing_path)
            available = sorted(
                source.loc[
                    source["particle_count"].astype(int).eq(particle_count),
                    "filter_repeat",
                ]
                .astype(int)
                .unique()
            )
            reused_seeds = [value for value in available if value < total_repeats]
            reusable = _reuse_scores(
                existing_path,
                dataset_ids,
                profile_ids,
                particle_count,
                reused_seeds,
            )
        missing_seeds = [
            value for value in range(total_repeats) if value not in reused_seeds
        ]
        jobs = [
            (dataset_id, profile, repeat)
            for dataset_id in dataset_ids
            for repeat in missing_seeds
            for profile in selected_profiles
        ]
        n_jobs = int(args.n_jobs or calibration["n_jobs"])
        started = time.perf_counter()
        results = Parallel(
            n_jobs=min(n_jobs, len(jobs)), backend="loky", verbose=10
        )(
            delayed(_score_new_repeat)(
                output=output,
                dataset_path=synthetic_dir / f"{dataset_id}.npz",
                base_config=base_config,
                base_path=base_path,
                dataset_paths=dataset_paths,
                profile=profile,
                particle_count=particle_count,
                resample_threshold_fraction=float(
                    calibration["resample_threshold_fraction"]
                ),
                filter_repeat=repeat,
                base_seed=int(calibration["base_seed"]),
                force=bool(args.force),
            )
            for dataset_id, profile, repeat in jobs
        )
        scores = pd.concat(
            [reusable, pd.DataFrame(results)], ignore_index=True
        ).sort_values(
            ["dataset_id", "filter_repeat", "fit_profile_id"]
        ).reset_index(drop=True)
        _require_complete_keys(
            scores, dataset_ids, profile_ids, tuple(range(total_repeats))
        )
        if not np.isfinite(scores["log_likelihood"].to_numpy(dtype=float)).all():
            raise ValueError("seed scores contain non-finite log likelihoods")
        _atomic_csv(output / "seed_scores.csv", scores)
        manifest.update(
            {
                "reused_seed_indices": reused_seeds,
                "new_seed_indices": missing_seeds,
                "run_runtime_seconds": float(time.perf_counter() - started),
                "score_row_count": int(len(scores)),
            }
        )
        _atomic_json(output / "analysis_manifest.json", manifest)
        if args.phase == "run":
            print(f"[seed-averaging] scores={output / 'seed_scores.csv'}")
            return
    elif not (output / "seed_scores.csv").exists():
        raise FileNotFoundError("seed_scores.csv is required for summarize")

    scores = pd.read_csv(output / "seed_scores.csv")
    (
        aggregate,
        split,
        running,
        equivalence,
        dataset_summary,
        summary,
    ) = summarize_seed_averaging(
        scores,
        aggregation_seed_counts=calibration["aggregation_seed_counts"],
        training_seed_indices=calibration["training_seed_indices"],
        validation_seed_indices=calibration["validation_seed_indices"],
        top_k=int(calibration["top_k"]),
        bootstrap_repeats=int(calibration["bootstrap_repeats"]),
        bootstrap_confidence=float(calibration["bootstrap_confidence"]),
        bootstrap_seed=int(calibration["base_seed"]),
        gates=calibration["stability_gates"],
    )
    _atomic_csv(output / "aggregate_candidate_scores.csv", aggregate)
    _atomic_csv(output / "split_half_stability.csv", split)
    _atomic_csv(output / "running_prefix_stability.csv", running)
    _atomic_csv(output / "candidate_equivalence_sets.csv", equivalence)
    _atomic_csv(output / "dataset_summary.csv", dataset_summary)
    _atomic_csv(
        output / "aggregation_method_sensitivity.csv",
        dataset_summary[
            [
                "dataset_id",
                "logmeanexp_vs_meanlog_rank_spearman",
                "logmeanexp_winner",
                "mean_log_winner",
                "aggregation_method_winner_agreement",
            ]
        ],
    )
    _atomic_json(output / "summary.json", summary)
    _write_chart_map(output, summary)
    _write_figure(
        output,
        split,
        running,
        equivalence,
        aggregate,
        calibration["stability_gates"],
        str(config["report"]["figure_png"]),
    )
    _write_readme(output, summary, dataset_summary)
    manifest.update(
        {
            "status": "complete_with_replicated_likelihood_gate_result",
            "all_stability_gates_pass": bool(
                summary["all_stability_gates_pass"]
            ),
            "summary_sha256": _sha256(output / "summary.json"),
            "seed_scores_sha256": _sha256(output / "seed_scores.csv"),
        }
    )
    _atomic_json(output / "analysis_manifest.json", manifest)
    print(f"[seed-averaging] outputs={output}", flush=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
