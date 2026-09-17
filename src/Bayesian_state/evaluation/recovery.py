"""Score frozen recovery candidates, evaluate numerical budgets, and report recovery."""
from __future__ import annotations
from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ..utils.parallel import parallel_job_count, single_threaded_processes
from scipy.stats import spearmanr
from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
from src.Bayesian_state.hypothesis_space.geometry import warmup_dykstra_numba
from src.Bayesian_state.model.readout import (
    resolve_choice_readout_config,
    resolve_output_noise_config,
)
from src.Bayesian_state.optimization.model_0826 import build_model_0826_cell_engine
from src.Bayesian_state.simulation.config import (
    EVALUATION_ROLE_SIMULATION,
    resolve_evaluation_score_mask,
)
from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
from src.Bayesian_state.utils.seeding import stable_seed
from src.Bayesian_state.optimization.recovery_parameters import (
    _declared_support,
    model_0826_truth_hyperparams,
)
from src.Bayesian_state.simulation.recovery import (
    MODEL_PARAMETER_NAMES,
)
from src.Bayesian_state.utils.recovery_artifacts import (
    _atomic_csv,
    _atomic_json,
)


def build_calibration_bank(anchor: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the frozen eight-candidate numerical calibration bank."""

    required = set(MODEL_PARAMETER_NAMES) - {"chi"}
    if set(anchor) != required:
        raise ValueError(
            "calibration anchor must define every PMH parameter except chi"
        )
    variants: list[tuple[str, dict[str, Any]]] = []
    variants.append(("anchor", deepcopy(dict(anchor))))
    gamma = deepcopy(dict(anchor))
    gamma["gamma"] = 0.50
    variants.append(("gamma_050", gamma))
    gains = deepcopy(dict(anchor))
    gains.update({"c_A": 0.0, "c_G": 0.0})
    variants.append(("gains_zero", gains))
    beta = deepcopy(dict(anchor))
    beta.update({"beta_0": 1.0, "eta_plus": 0.01, "eta_minus": 0.03})
    variants.append(("beta_slow", beta))

    bank: list[dict[str, Any]] = []
    for variant, values in variants:
        for chi in (0, 1):
            truth = deepcopy(values)
            truth["chi"] = chi
            bank.append(
                {
                    "candidate_id": f"{variant}_chi_{chi}",
                    "variant": variant,
                    "truth": truth,
                }
            )
    return bank


def resolve_calibration_filter_seeds(
    *,
    dataset_id: str,
    base_seed: int,
    ensemble: str,
    count: int,
) -> list[int]:
    """Resolve nested logical seeds within A/B and disjoint seeds between them."""

    ensemble_name = str(ensemble).strip().upper()
    if ensemble_name not in {"A", "B"}:
        raise ValueError("calibration ensemble must be A or B")
    seed_count = int(count)
    if seed_count <= 0:
        raise ValueError("calibration seed count must be positive")
    return [
        stable_seed(
            {
                "seed_role": "model0826_recovery_pf_calibration",
                "dataset_id": str(dataset_id),
                "base_seed": int(base_seed),
                "ensemble": ensemble_name,
                "logical_seed_index": int(index),
            }
        )
        for index in range(seed_count)
    ]


def _frozen_readout_args(engine_config: Mapping[str, Any]) -> dict[str, float]:
    readout = resolve_choice_readout_config(None, engine_config)
    noise = resolve_output_noise_config(None, engine_config)
    if (
        readout["method"] != "expectation"
        or float(readout["power"]) != 1.0
        or float(readout["strategy_confidence_gain"]) != 0.0
        or float(readout["rule_commitment_confidence_gain"]) != 0.0
    ):
        raise ValueError("Model0826 recovery requires the frozen expectation readout")
    noise_terms = (
        "base_lapse",
        "post_error_lapse",
        "low_accuracy_lapse",
        "latent_volatility_lapse",
    )
    if any(float(noise.get(name, 0.0)) != 0.0 for name in noise_terms):
        raise ValueError("Model0826 recovery requires zero output lapse")
    return {
        "choice_readout_power": 1.0,
        "strategy_confidence_gain": 0.0,
        "rule_commitment_confidence_gain": 0.0,
        "output_lapse": 0.0,
    }


def score_pf_bank(
    *,
    dataset_id: str,
    subject_id: int,
    stimulus: Sequence[Sequence[float]] | np.ndarray,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    base_engine_config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    particle_count: int,
    filter_seeds: Sequence[int],
    ensemble: str,
    resample_threshold_fraction: float = 0.5,
    processed_data_dir: str | Path | None = None,
    dataset_paths: Mapping[str, str | Path] | None = None,
    pf_runner: Callable[..., Any] = run_state_model_particle_filter,
) -> list[dict[str, Any]]:
    """Score one fixed candidate bank using paired PF seeds."""

    physical = np.asarray(stimulus, dtype=float)
    observed_choice = np.asarray(choices, dtype=int).reshape(-1)
    observed_feedback = np.asarray(feedback, dtype=float).reshape(-1)
    if physical.ndim != 2 or physical.shape[0] != observed_choice.size:
        raise ValueError("PF calibration stimulus and choices are misaligned")
    if observed_feedback.size != observed_choice.size:
        raise ValueError("PF calibration feedback and choices are misaligned")
    seeds = [int(value) for value in filter_seeds]
    if not seeds:
        raise ValueError("PF calibration requires at least one filter seed")
    if len(seeds) != len(set(seeds)):
        raise ValueError("PF calibration filter seeds must be unique")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        truth = dict(candidate["truth"])
        engine = build_model_0826_cell_engine(base_engine_config, "PMH")
        engine = apply_fixed_hyperparams_to_engine_config(
            engine,
            model_0826_truth_hyperparams(truth),
        )
        readout_args = _frozen_readout_args(engine)
        probability_runs: list[np.ndarray] = []
        for filter_seed in seeds:
            result = pf_runner(
                engine_config=engine,
                subject_id=int(subject_id),
                stimulus=physical,
                choices=observed_choice,
                feedback=observed_feedback,
                particle_count=int(particle_count),
                filter_seed=int(filter_seed),
                resample_threshold_fraction=float(resample_threshold_fraction),
                processed_data_dir=processed_data_dir,
                dataset_paths=dataset_paths,
                **readout_args,
            )
            probabilities = np.asarray(result.marginal_probabilities, dtype=float)
            if probabilities.shape != (observed_choice.size, 2):
                raise ValueError("PF calibration probabilities must have shape (T, 2)")
            if not np.all(np.isfinite(probabilities)) or not np.allclose(
                probabilities.sum(axis=1), 1.0, atol=1e-8
            ):
                raise ValueError("PF calibration returned invalid probabilities")
            probability_runs.append(probabilities)
        stack = np.stack(probability_runs, axis=0)
        mean_probability = np.mean(stack, axis=0)
        selected = mean_probability[
            np.arange(observed_choice.size), observed_choice - 1
        ]
        total_nll = float(-np.log(np.clip(selected, 1e-12, 1.0)).sum())
        if stack.shape[0] > 1:
            probability_mcse = np.std(stack[:, :, 1], axis=0, ddof=1) / np.sqrt(
                float(stack.shape[0])
            )
        else:
            probability_mcse = np.zeros(observed_choice.size, dtype=float)
        rows.append(
            {
                "dataset_id": str(dataset_id),
                "subject_id": int(subject_id),
                "candidate_id": candidate_id,
                "variant": str(candidate["variant"]),
                "candidate_chi": int(truth["chi"]),
                "particle_count": int(particle_count),
                "filter_seed_count": int(len(seeds)),
                "ensemble": str(ensemble).upper(),
                "filter_seeds": seeds,
                "total_nll": total_nll,
                "mean_trial_nll": total_nll / float(observed_choice.size),
                "mean_probability": mean_probability,
                "probability_runs": stack,
                "trial_probability_mcse": probability_mcse,
                "probability_aggregation": "mean_probability_then_nll",
            }
        )
    return rows


def _score_pf_candidate_seed(
    *,
    common_kwargs: Mapping[str, Any],
    candidate: Mapping[str, Any],
    filter_seed: int,
) -> dict[str, Any]:
    return score_pf_bank(
        **dict(common_kwargs),
        candidates=[dict(candidate)],
        filter_seeds=[int(filter_seed)],
    )[0]


@single_threaded_processes()
def score_pf_bank_parallel(
    *,
    dataset_id: str,
    subject_id: int,
    stimulus: Sequence[Sequence[float]] | np.ndarray,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    base_engine_config: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    particle_count: int,
    filter_seeds: Sequence[int],
    ensemble: str,
    n_jobs: int,
    resample_threshold_fraction: float = 0.5,
    processed_data_dir: str | Path | None = None,
    dataset_paths: Mapping[str, str | Path] | None = None,
    pf_runner: Callable[..., Any] = run_state_model_particle_filter,
) -> list[dict[str, Any]]:
    """Parallelize independent candidate×seed PF runs, then aggregate exactly."""

    candidate_rows = [deepcopy(dict(candidate)) for candidate in candidates]
    seeds = [int(value) for value in filter_seeds]
    if not candidate_rows:
        raise ValueError("parallel PF calibration requires at least one candidate")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("parallel PF calibration requires unique filter seeds")
    if int(n_jobs) < 1:
        raise ValueError("parallel PF calibration n_jobs must be positive")
    jobs = parallel_job_count(int(n_jobs), len(candidate_rows) * len(seeds))
    common_kwargs = {
        "dataset_id": str(dataset_id),
        "subject_id": int(subject_id),
        "stimulus": np.asarray(stimulus, dtype=float),
        "choices": np.asarray(choices, dtype=int),
        "feedback": np.asarray(feedback, dtype=float),
        "base_engine_config": deepcopy(dict(base_engine_config)),
        "particle_count": int(particle_count),
        "ensemble": str(ensemble),
        "resample_threshold_fraction": float(resample_threshold_fraction),
        "processed_data_dir": processed_data_dir,
        "dataset_paths": dataset_paths,
        "pf_runner": pf_runner,
    }
    if jobs == 1:
        return score_pf_bank(
            **common_kwargs,
            candidates=candidate_rows,
            filter_seeds=seeds,
        )
    warmup_dykstra_numba()
    single_rows = Parallel(n_jobs=jobs, verbose=10)(
        delayed(_score_pf_candidate_seed)(
            common_kwargs=common_kwargs,
            candidate=candidate,
            filter_seed=filter_seed,
        )
        for candidate in candidate_rows
        for filter_seed in seeds
    )
    observed = np.asarray(choices, dtype=int).reshape(-1)
    combined: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidate_rows):
        start = candidate_index * len(seeds)
        candidate_seed_rows = single_rows[start : start + len(seeds)]
        stack = np.stack(
            [
                np.asarray(row["mean_probability"], dtype=float)
                for row in candidate_seed_rows
            ],
            axis=0,
        )
        if stack.shape[0] > 1:
            probability_mcse = np.std(stack[:, :, 1], axis=0, ddof=1) / np.sqrt(
                float(stack.shape[0])
            )
        else:
            probability_mcse = np.zeros(observed.size, dtype=float)
        total_nll = mean_probability_nll(stack, observed)
        row = dict(candidate_seed_rows[0])
        row.update(
            {
                "candidate_id": str(candidate["candidate_id"]),
                "filter_seed_count": int(len(seeds)),
                "filter_seeds": seeds,
                "total_nll": total_nll,
                "mean_trial_nll": total_nll / float(observed.size),
                "mean_probability": np.mean(stack, axis=0),
                "probability_runs": stack,
                "trial_probability_mcse": probability_mcse,
                "parallel_n_jobs": int(jobs),
            }
        )
        combined.append(row)
    return combined


def _setting_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, int, str], dict[str, Mapping[str, Any]]]:
    settings: dict[
        tuple[str, int, int, str], dict[str, Mapping[str, Any]]
    ] = {}
    for row in rows:
        key = (
            str(row["dataset_id"]),
            int(row["particle_count"]),
            int(row["filter_seed_count"]),
            str(row["ensemble"]).upper(),
        )
        candidate_id = str(row["candidate_id"])
        candidate_rows = settings.setdefault(key, {})
        if candidate_id in candidate_rows:
            raise ValueError("duplicate PF calibration candidate row")
        candidate_rows[candidate_id] = row
    return settings


def _compare_pf_settings(
    settings: Mapping[
        tuple[str, int, int, str], Mapping[str, Mapping[str, Any]]
    ],
    left: tuple[int, int, str],
    right: tuple[int, int, str],
) -> list[dict[str, Any]]:
    dataset_ids = sorted({key[0] for key in settings})
    comparisons: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        left_rows = settings.get((dataset_id, *left))
        right_rows = settings.get((dataset_id, *right))
        if left_rows is None or right_rows is None:
            continue
        candidate_ids = sorted(left_rows)
        if candidate_ids != sorted(right_rows):
            raise ValueError("PF calibration candidate banks differ between settings")
        left_nll = np.asarray(
            [float(left_rows[name]["total_nll"]) for name in candidate_ids]
        )
        right_nll = np.asarray(
            [float(right_rows[name]["total_nll"]) for name in candidate_ids]
        )
        right_winner_index = int(np.argmin(right_nll))
        left_order = np.argsort(left_nll, kind="stable")
        right_winner_rank_in_left = int(
            np.flatnonzero(left_order == right_winner_index)[0] + 1
        )
        rho = float(spearmanr(left_nll, right_nll).statistic)
        if not np.isfinite(rho):
            rho = 1.0 if np.allclose(left_nll, right_nll) else 0.0
        probability_differences = []
        for name in candidate_ids:
            left_probability = np.asarray(
                left_rows[name]["mean_probability"], dtype=float
            )
            right_probability = np.asarray(
                right_rows[name]["mean_probability"], dtype=float
            )
            if left_probability.shape != right_probability.shape:
                raise ValueError("PF calibration probability shapes differ")
            probability_differences.append(
                np.square(left_probability - right_probability).reshape(-1)
            )
        comparisons.append(
            {
                "dataset_id": dataset_id,
                "left": {"particle_count": left[0], "filter_seed_count": left[1], "ensemble": left[2]},
                "right": {"particle_count": right[0], "filter_seed_count": right[1], "ensemble": right[2]},
                "candidate_nll_spearman": rho,
                "winner_agreement": bool(
                    candidate_ids[int(np.argmin(left_nll))]
                    == candidate_ids[right_winner_index]
                ),
                "right_winner_rank_in_left": right_winner_rank_in_left,
                "probability_rmse": float(
                    np.sqrt(np.mean(np.concatenate(probability_differences)))
                ),
            }
        )
    return comparisons


def summarize_search_budget_retention(
    score_rows: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Check whether low-budget stages retain the high-budget winner."""

    settings = _setting_rows(score_rows)
    reference_config = dict(policy.get("reference") or {})
    reference = (
        int(reference_config["particle_count"]),
        int(reference_config["filter_seed_count"]),
        str(reference_config.get("ensemble", "A")).upper(),
    )
    stage_configs = policy.get("stages") or {}
    if not isinstance(stage_configs, Mapping) or not stage_configs:
        raise ValueError("search budget retention requires at least one stage")

    stage_summaries: dict[str, Any] = {}
    comparison_rows: list[dict[str, Any]] = []
    for stage_name, raw_config in stage_configs.items():
        config = dict(raw_config)
        low = (
            int(config["particle_count"]),
            int(config["filter_seed_count"]),
            str(config.get("ensemble", "A")).upper(),
        )
        top_k = int(config["winner_top_k"])
        minimum_count = int(config["minimum_dataset_count"])
        if top_k < 1 or minimum_count < 1:
            raise ValueError(
                "winner_top_k and minimum_dataset_count must be positive"
            )
        rows = _compare_pf_settings(settings, low, reference)
        ranks = [int(row["right_winner_rank_in_left"]) for row in rows]
        retained_count = sum(rank <= top_k for rank in ranks)
        stage_summaries[str(stage_name)] = {
            "particle_count": low[0],
            "filter_seed_count": low[1],
            "ensemble": low[2],
            "reference": {
                "particle_count": reference[0],
                "filter_seed_count": reference[1],
                "ensemble": reference[2],
            },
            "winner_top_k": top_k,
            "minimum_dataset_count": minimum_count,
            "comparison_dataset_count": len(rows),
            "retained_dataset_count": int(retained_count),
            "right_winner_ranks": ranks,
            "median_candidate_nll_spearman": (
                float(np.median([row["candidate_nll_spearman"] for row in rows]))
                if rows
                else None
            ),
            "passes": bool(
                len(rows) >= minimum_count and retained_count >= minimum_count
            ),
        }
        comparison_rows.extend(
            {"stage": str(stage_name), **row} for row in rows
        )

    return {
        "status": (
            "passed"
            if all(row["passes"] for row in stage_summaries.values())
            else "failed"
        ),
        "stages": stage_summaries,
        "comparisons": comparison_rows,
    }


def _budget_mcse_q95(
    settings: Mapping[
        tuple[str, int, int, str], Mapping[str, Mapping[str, Any]]
    ],
    particle_count: int,
    filter_seed_count: int,
) -> float:
    values: list[np.ndarray] = []
    for (dataset_id, particles, seeds, ensemble), candidates in settings.items():
        del dataset_id
        if particles == particle_count and seeds == filter_seed_count and ensemble in {"A", "B"}:
            values.extend(
                np.asarray(row["trial_probability_mcse"], dtype=float).reshape(-1)
                for row in candidates.values()
            )
    if not values:
        return float("inf")
    return float(np.quantile(np.concatenate(values), 0.95))


def summarize_pf_calibration(
    score_rows: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the pre-registered rank, winner, RMSE, and MCSE gates."""

    settings = _setting_rows(score_rows)
    dataset_count = int(gates.get("dataset_count", 6))

    def decision(
        *,
        particle_count: int,
        filter_seed_count: int,
        scaling_pairs: Sequence[
            tuple[tuple[int, int, str], tuple[int, int, str]]
        ],
        independent_left: tuple[int, int, str],
        independent_right: tuple[int, int, str],
        extra_probability_comparison: tuple[
            tuple[int, int, str], tuple[int, int, str]
        ] | None = None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        scaling_by_pair = [
            _compare_pf_settings(settings, left, right)
            for left, right in scaling_pairs
        ]
        scaling = [row for pair_rows in scaling_by_pair for row in pair_rows]
        independent = _compare_pf_settings(
            settings, independent_left, independent_right
        )
        probability_rows = list(scaling) + list(independent)
        if extra_probability_comparison is not None:
            probability_rows.extend(
                _compare_pf_settings(settings, *extra_probability_comparison)
            )
        rank_values = [row["candidate_nll_spearman"] for row in scaling]
        scaling_winner_counts = [
            sum(bool(row["winner_agreement"]) for row in pair_rows)
            for pair_rows in scaling_by_pair
        ]
        winner_top_k = int(gates.get("adjacent_winner_top_k", 1))
        if winner_top_k < 1:
            raise ValueError("adjacent_winner_top_k must be positive")
        winner_top_k_min_count = int(
            gates.get(
                "adjacent_winner_top_k_min_count",
                gates.get("adjacent_winner_agreement_min_count", dataset_count),
            )
        )
        scaling_winner_top_k_counts = [
            sum(
                int(row["right_winner_rank_in_left"]) <= winner_top_k
                for row in pair_rows
            )
            for pair_rows in scaling_by_pair
        ]
        independent_winners = sum(
            bool(row["winner_agreement"]) for row in independent
        )
        probability_rmse = [row["probability_rmse"] for row in probability_rows]
        mcse_q95 = _budget_mcse_q95(
            settings, particle_count, filter_seed_count
        )
        complete = all(
            len(pair_rows) == dataset_count for pair_rows in scaling_by_pair
        ) and len(independent) == dataset_count
        metrics = {
            "particle_count": int(particle_count),
            "filter_seed_count": int(filter_seed_count),
            "complete": complete,
            "median_adjacent_rank_spearman": (
                float(np.median(rank_values)) if rank_values else None
            ),
            "minimum_adjacent_rank_spearman": (
                float(np.min(rank_values)) if rank_values else None
            ),
            "adjacent_winner_agreement_counts": [
                int(value) for value in scaling_winner_counts
            ],
            "minimum_adjacent_winner_agreement_count": (
                int(min(scaling_winner_counts)) if scaling_winner_counts else 0
            ),
            "adjacent_high_budget_winner_top_k": winner_top_k,
            "adjacent_high_budget_winner_top_k_counts": [
                int(value) for value in scaling_winner_top_k_counts
            ],
            "minimum_adjacent_high_budget_winner_top_k_count": (
                int(min(scaling_winner_top_k_counts))
                if scaling_winner_top_k_counts
                else 0
            ),
            "independent_winner_agreement_count": int(independent_winners),
            "median_probability_rmse": (
                float(np.median(probability_rmse)) if probability_rmse else None
            ),
            "trial_probability_mcse_q95": mcse_q95,
        }
        metrics["passes_all_gates"] = bool(
            complete
            and metrics["median_adjacent_rank_spearman"]
            >= float(gates["median_adjacent_rank_spearman_min"])
            and metrics["minimum_adjacent_rank_spearman"]
            >= float(gates["minimum_adjacent_rank_spearman_min"])
            and metrics["minimum_adjacent_high_budget_winner_top_k_count"]
            >= winner_top_k_min_count
            and independent_winners
            >= int(gates["independent_winner_agreement_min_count"])
            and metrics["median_probability_rmse"]
            <= float(gates["median_probability_rmse_max"])
            and mcse_q95 <= float(gates["trial_probability_mcse_q95_max"])
        )
        return metrics, probability_rows

    decisions: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    primary, primary_comparisons = decision(
        particle_count=64,
        filter_seed_count=8,
        scaling_pairs=(
            ((16, 4, "A"), (32, 4, "A")),
            ((32, 4, "A"), (64, 4, "A")),
        ),
        independent_left=(64, 8, "A"),
        independent_right=(64, 8, "B"),
        extra_probability_comparison=((64, 4, "A"), (64, 8, "A")),
    )
    decisions.append(primary)
    comparisons.extend(primary_comparisons)
    high_setting_keys = {
        (key[1], key[2], key[3]) for key in settings
    }
    if (128, 16, "A") in high_setting_keys and (128, 16, "B") in high_setting_keys:
        escalated, escalated_comparisons = decision(
            particle_count=128,
            filter_seed_count=16,
            scaling_pairs=(((64, 8, "A"), (128, 16, "A")),),
            independent_left=(128, 16, "A"),
            independent_right=(128, 16, "B"),
        )
        decisions.append(escalated)
        comparisons.extend(escalated_comparisons)
    return {
        "status": (
            "passed" if any(row["passes_all_gates"] for row in decisions) else "failed"
        ),
        "budget_decisions": decisions,
        "comparisons": comparisons,
    }


def freeze_smallest_passing_budget(
    summary: Mapping[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, int] | None:
    """Return and optionally persist the smallest budget passing every gate."""

    decisions = [
        dict(row)
        for row in summary.get("budget_decisions", [])
        if bool(row.get("passes_all_gates", False))
    ]
    if not decisions:
        return None
    selected = min(
        decisions,
        key=lambda row: (
            int(row["particle_count"]) * int(row["filter_seed_count"]),
            int(row["particle_count"]),
            int(row["filter_seed_count"]),
        ),
    )
    budget = {
        "particle_count": int(selected["particle_count"]),
        "filter_seed_count": int(selected["filter_seed_count"]),
    }
    if output_path is not None:
        _atomic_json(
            Path(output_path),
            {
                "status": "frozen",
                **budget,
                "source_decision": selected,
            },
        )
    return budget


def mean_probability_nll(
    probability_runs: Sequence[Any] | np.ndarray,
    choices: Sequence[int] | np.ndarray,
    mask: Sequence[bool] | np.ndarray | None = None,
) -> float:
    """Average PF probabilities first, then compute masked total choice NLL."""

    runs = np.asarray(probability_runs, dtype=float)
    observed = np.asarray(choices, dtype=int).reshape(-1)
    if runs.ndim != 3 or runs.shape[1:] != (observed.size, 2):
        raise ValueError("probability_runs must have shape (B, T, 2)")
    if runs.shape[0] < 1 or not np.all(np.isfinite(runs)) or np.any(runs < 0.0):
        raise ValueError("probability_runs must contain finite nonnegative values")
    if not np.allclose(runs.sum(axis=2), 1.0, atol=1e-8):
        raise ValueError("probability rows must sum to one")
    if not np.all(np.isin(observed, [1, 2])):
        raise ValueError("choices must be encoded as 1 or 2")
    score_mask = np.ones(observed.size, dtype=bool)
    if mask is not None:
        score_mask = np.asarray(mask, dtype=bool).reshape(-1)
        if score_mask.size != observed.size:
            raise ValueError("NLL mask must align with choices")
    if not np.any(score_mask):
        raise ValueError("NLL mask must select at least one trial")
    mean_probability = np.mean(runs, axis=0)
    selected = mean_probability[np.arange(observed.size), observed - 1]
    return float(-np.log(np.clip(selected[score_mask], 1e-12, 1.0)).sum())


@single_threaded_processes()
def score_frozen_candidate(
    *,
    subject_id: int,
    stimulus: Sequence[Sequence[float]] | np.ndarray,
    choices: Sequence[int] | np.ndarray,
    feedback: Sequence[float] | np.ndarray,
    base_engine_config: Mapping[str, Any],
    candidate_cell: str,
    fixed_hyperparams: Mapping[str, Any],
    particle_count: int,
    filter_seeds: Sequence[int],
    evaluation_protocol: Mapping[str, Any] | None,
    n_jobs: int = 1,
    resample_threshold_fraction: float = 0.5,
    processed_data_dir: str | Path | None = None,
    dataset_paths: Mapping[str, str | Path] | None = None,
    pf_runner: Callable[..., Any] = run_state_model_particle_filter,
) -> dict[str, Any]:
    """Run frozen parameters on the full sequence and score only the held-out mask."""

    physical = np.asarray(stimulus, dtype=float)
    observed = np.asarray(choices, dtype=int).reshape(-1)
    observed_feedback = np.asarray(feedback, dtype=float).reshape(-1)
    if physical.ndim != 2 or physical.shape[0] != observed.size:
        raise ValueError("frozen scoring arrays are misaligned")
    if observed_feedback.size != observed.size:
        raise ValueError("frozen scoring feedback is misaligned")
    score_mask, score_context = resolve_evaluation_score_mask(
        observed.size,
        evaluation_protocol,
        role=EVALUATION_ROLE_SIMULATION,
    )
    engine = build_model_0826_cell_engine(base_engine_config, candidate_cell)
    engine = apply_fixed_hyperparams_to_engine_config(engine, fixed_hyperparams)
    readout_args = _frozen_readout_args(engine)
    seeds = [int(value) for value in filter_seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("frozen scoring requires unique filter seeds")
    if int(n_jobs) < 1:
        raise ValueError("frozen scoring n_jobs must be positive")
    jobs = parallel_job_count(int(n_jobs), len(seeds))

    def run_seed(filter_seed: int) -> np.ndarray:
        result = pf_runner(
            engine_config=engine,
            subject_id=int(subject_id),
            stimulus=physical,
            choices=observed,
            feedback=observed_feedback,
            particle_count=int(particle_count),
            filter_seed=int(filter_seed),
            resample_threshold_fraction=float(resample_threshold_fraction),
            processed_data_dir=processed_data_dir,
            dataset_paths=dataset_paths,
            **readout_args,
        )
        probabilities = np.asarray(result.marginal_probabilities, dtype=float)
        if probabilities.shape != (observed.size, 2):
            raise ValueError("frozen scoring probabilities must have shape (T, 2)")
        return probabilities

    if jobs == 1:
        probability_runs = [run_seed(filter_seed) for filter_seed in seeds]
    else:
        warmup_dykstra_numba()
        probability_runs = list(
            Parallel(n_jobs=jobs, verbose=10)(
                delayed(run_seed)(filter_seed) for filter_seed in seeds
            )
        )
    stack = np.stack(probability_runs, axis=0)
    total_nll = mean_probability_nll(stack, observed, score_mask)
    return {
        "total_nll": total_nll,
        "mean_trial_nll": total_nll / float(score_context["score_trial_count"]),
        "score_context": score_context,
        "particle_count": int(particle_count),
        "filter_seed_count": int(len(seeds)),
        "filter_seeds": seeds,
        "parallel_n_jobs": int(jobs),
        "probability_aggregation": "mean_probability_then_nll",
        "mean_probability": np.mean(stack, axis=0),
    }


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    proportion = float(successes) / float(total)
    denominator = 1.0 + z * z / float(total)
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / float(total)
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return float(center - half_width), float(center + half_width)


def summarize_module_recovery(
    scores: pd.DataFrame,
    *,
    near_best_delta_nll: float = 2.0,
    gates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize held-out total-NLL architecture recovery."""

    required = {
        "dataset_id", "true_cell", "candidate_cell", "total_nll",
    }
    if not required.issubset(scores.columns):
        raise ValueError("module recovery scores are missing required columns")
    cells = ("P", "PM", "PH", "PMH")
    dataset_rows: list[dict[str, Any]] = []
    for dataset_id, frame in scores.groupby("dataset_id", sort=True):
        if set(frame["candidate_cell"].astype(str)) != set(cells) or len(frame) != 4:
            raise ValueError(f"module dataset {dataset_id} requires four candidate cells")
        if frame["true_cell"].nunique() != 1:
            raise ValueError(f"module dataset {dataset_id} has inconsistent truth")
        ranked = frame.assign(
            _cell_order=frame["candidate_cell"].map(
                {cell: index for index, cell in enumerate(cells)}
            )
        ).sort_values(["total_nll", "_cell_order"])
        if not np.all(np.isfinite(ranked["total_nll"].to_numpy(dtype=float))):
            raise ValueError(f"module dataset {dataset_id} has non-finite NLL")
        winner = ranked.iloc[0]
        true_cell = str(frame["true_cell"].iloc[0])
        true_score = frame.loc[
            frame["candidate_cell"].astype(str).eq(true_cell), "total_nll"
        ]
        if len(true_score) != 1:
            raise ValueError(f"module dataset {dataset_id} lacks one true-cell score")
        best_nll = float(winner["total_nll"])
        true_nll = float(true_score.iloc[0])
        row = {
            "dataset_id": str(dataset_id),
            "true_cell": true_cell,
            "predicted_cell": str(winner["candidate_cell"]),
            "best_total_nll": best_nll,
            "true_total_nll": true_nll,
            "true_delta_nll": true_nll - best_nll,
            "exact_recovery": str(winner["candidate_cell"]) == true_cell,
            "true_within_near_best": (
                true_nll <= best_nll + float(near_best_delta_nll)
            ),
        }
        if "generated_accuracy" in frame:
            row["generated_accuracy"] = float(frame["generated_accuracy"].iloc[0])
        dataset_rows.append(row)
    dataset_frame = pd.DataFrame(dataset_rows)
    confusion_rows = []
    for true_cell in cells:
        for predicted_cell in cells:
            confusion_rows.append(
                {
                    "true_cell": true_cell,
                    "predicted_cell": predicted_cell,
                    "count": int(
                        np.sum(
                            dataset_frame["true_cell"].eq(true_cell)
                            & dataset_frame["predicted_cell"].eq(predicted_cell)
                        )
                    ),
                }
            )
    cell_rows = []
    for cell in cells:
        selected = dataset_frame.loc[dataset_frame["true_cell"].eq(cell)]
        successes = int(selected["exact_recovery"].sum())
        low, high = _wilson_interval(successes, len(selected))
        cell_rows.append(
            {
                "true_cell": cell,
                "dataset_n": int(len(selected)),
                "exact_recovery_count": successes,
                "exact_recovery": float(selected["exact_recovery"].mean()),
                "wilson_low": low,
                "wilson_high": high,
                "near_best_coverage": float(
                    selected["true_within_near_best"].mean()
                ),
            }
        )
    total = len(dataset_frame)
    wrong_absorption = {
        cell: float(
            np.mean(
                dataset_frame["predicted_cell"].eq(cell)
                & ~dataset_frame["true_cell"].eq(cell)
            )
        )
        for cell in cells
    }
    gate_config = {
        "overall_exact_recovery_min": 0.70,
        "per_cell_exact_recovery_min": 0.50,
        "true_cell_near_best_coverage_min": 0.85,
        "maximum_single_wrong_cell_absorption": 0.30,
        **dict(gates or {}),
    }
    overall_exact = float(dataset_frame["exact_recovery"].mean())
    near_best_coverage = float(dataset_frame["true_within_near_best"].mean())
    passes = bool(
        total > 0
        and overall_exact >= float(gate_config["overall_exact_recovery_min"])
        and min(row["exact_recovery"] for row in cell_rows)
        >= float(gate_config["per_cell_exact_recovery_min"])
        and near_best_coverage
        >= float(gate_config["true_cell_near_best_coverage_min"])
        and max(wrong_absorption.values())
        <= float(gate_config["maximum_single_wrong_cell_absorption"])
    )
    return {
        "dataset_n": int(total),
        "near_best_delta_nll": float(near_best_delta_nll),
        "overall_exact_recovery": overall_exact,
        "true_cell_near_best_coverage": near_best_coverage,
        "wrong_cell_absorption": wrong_absorption,
        "passes_pre_registered_gates": passes,
        "gates": gate_config,
        "confusion_rows": confusion_rows,
        "cell_rows": cell_rows,
        "dataset_rows": dataset_frame.to_dict(orient="records"),
    }


def _parameter_support_values(
    parameter_space: Mapping[str, Any],
    parameter: str,
) -> list[float]:
    return [float(value) for value in _declared_support(parameter_space, parameter)]


def _workspace_support_values(
    parameter_space: Mapping[str, Any],
    parameter: str,
) -> list[int]:
    workspace = parameter_space["subject_parameters"]["workspace_execution"]
    values = {
        int(candidate[parameter])
        for candidate in workspace["candidates"]
    }
    if not values:
        raise ValueError(f"workspace parameter {parameter} has empty support")
    return sorted(values)


def _safe_spearman(truth: np.ndarray, estimate: np.ndarray) -> float:
    if truth.size < 2 or np.allclose(truth, truth[0]) or np.allclose(
        estimate, estimate[0]
    ):
        return 1.0 if np.allclose(truth, estimate) else 0.0
    value = float(spearmanr(truth, estimate).statistic)
    return value if np.isfinite(value) else 0.0


def _balanced_accuracy_binary(truth: np.ndarray, estimate: np.ndarray) -> float:
    truth_positive = truth > 0.0
    estimate_positive = estimate > 0.0
    if not np.any(truth_positive) or not np.any(~truth_positive):
        return float("nan")
    sensitivity = float(np.mean(estimate_positive[truth_positive]))
    specificity = float(np.mean(~estimate_positive[~truth_positive]))
    return 0.5 * (sensitivity + specificity)


def summarize_parameter_recovery(
    estimates: pd.DataFrame,
    *,
    parameter_space: Mapping[str, Any],
    gates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize discrete, continuous, and zero-boundary recovery."""

    required = {"dataset_id", "true_M", "estimated_M", "true_chi", "estimated_chi"}
    continuous = (
        "gamma", "E_C", "delta_E", "g_0", "c_A", "c_G",
        "beta_0", "eta_plus", "eta_minus",
    )
    for parameter in continuous:
        required.update({f"true_{parameter}", f"estimated_{parameter}"})
    if not required.issubset(estimates.columns):
        missing = sorted(required - set(estimates.columns))
        raise ValueError(f"parameter recovery estimates are missing: {missing}")
    frame = estimates.copy()
    if "true_within_near_best" not in frame:
        frame["true_within_near_best"] = False
    if frame["dataset_id"].duplicated().any():
        raise ValueError("parameter recovery estimates require one row per dataset")
    gate_config = {
        "chi_exact_recovery_min": 0.70,
        "chi_near_best_coverage_min": 0.85,
        "continuous_spearman_min": 0.60,
        "continuous_normalized_mae_max": 0.20,
        "continuous_near_best_coverage_min": 0.80,
        "zero_positive_balanced_accuracy_min": 0.70,
        **dict(gates or {}),
    }
    near_best_coverage = float(frame["true_within_near_best"].mean())
    parameter_rows = []
    error_columns: dict[str, np.ndarray] = {}
    for parameter in continuous:
        truth = frame[f"true_{parameter}"].to_numpy(dtype=float)
        estimate = frame[f"estimated_{parameter}"].to_numpy(dtype=float)
        if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(estimate)):
            raise ValueError(f"parameter {parameter} contains non-finite values")
        error = estimate - truth
        support = _parameter_support_values(parameter_space, parameter)
        support_span = float(max(support) - min(support))
        if support_span <= 0.0:
            raise ValueError(f"parameter {parameter} has zero support span")
        balanced_accuracy = None
        positive_mae = None
        if parameter in {"delta_E", "c_A", "c_G"}:
            balanced_accuracy = _balanced_accuracy_binary(truth, estimate)
            positive = truth > 0.0
            positive_mae = (
                float(np.mean(np.abs(error[positive])))
                if np.any(positive)
                else None
            )
        spearman = _safe_spearman(truth, estimate)
        normalized_mae = float(np.mean(np.abs(error)) / support_span)
        supported = bool(
            spearman >= float(gate_config["continuous_spearman_min"])
            and normalized_mae
            <= float(gate_config["continuous_normalized_mae_max"])
            and near_best_coverage
            >= float(gate_config["continuous_near_best_coverage_min"])
            and (
                balanced_accuracy is None
                or (
                    np.isfinite(balanced_accuracy)
                    and balanced_accuracy
                    >= float(gate_config["zero_positive_balanced_accuracy_min"])
                )
            )
        )
        parameter_rows.append(
            {
                "parameter": parameter,
                "dataset_n": int(len(frame)),
                "bias": float(np.mean(error)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "spearman": spearman,
                "support_span": support_span,
                "normalized_mae": normalized_mae,
                "near_best_coverage": near_best_coverage,
                "zero_positive_balanced_accuracy": balanced_accuracy,
                "positive_truth_mae": positive_mae,
                "supported": supported,
            }
        )
        error_columns[parameter] = error

    chi_truth = frame["true_chi"].to_numpy(dtype=int)
    chi_estimate = frame["estimated_chi"].to_numpy(dtype=int)
    chi_exact = float(np.mean(chi_truth == chi_estimate))
    chi_successes = int(np.sum(chi_truth == chi_estimate))
    chi_low, chi_high = _wilson_interval(chi_successes, len(frame))
    chi_confusion_rows = [
        {
            "true_chi": truth,
            "estimated_chi": estimate,
            "count": int(np.sum((chi_truth == truth) & (chi_estimate == estimate))),
        }
        for truth in (0, 1)
        for estimate in (0, 1)
    ]
    m_truth = frame["true_M"].to_numpy(dtype=int)
    m_estimate = frame["estimated_M"].to_numpy(dtype=int)
    m_exact = float(np.mean(m_truth == m_estimate))
    m_successes = int(np.sum(m_truth == m_estimate))
    m_low, m_high = _wilson_interval(m_successes, len(frame))
    m_support = _workspace_support_values(parameter_space, "M")
    if not set(m_truth).issubset(m_support) or not set(m_estimate).issubset(
        m_support
    ):
        raise ValueError("M truth or estimate falls outside declared support")
    m_confusion_rows = [
        {
            "true_M": truth,
            "estimated_M": estimate,
            "count": int(np.sum((m_truth == truth) & (m_estimate == estimate))),
        }
        for truth in m_support
        for estimate in m_support
    ]

    correlation_rows: list[dict[str, Any]] = []
    names = list(continuous)
    for left in names:
        for right in names:
            left_error = error_columns[left]
            right_error = error_columns[right]
            if np.std(left_error) == 0.0 or np.std(right_error) == 0.0:
                correlation = 1.0 if left == right else 0.0
            else:
                correlation = float(np.corrcoef(left_error, right_error)[0, 1])
            correlation_rows.append(
                {
                    "left_parameter": left,
                    "right_parameter": right,
                    "error_correlation": correlation,
                }
            )
    chi_supported = bool(
        chi_exact >= float(gate_config["chi_exact_recovery_min"])
        and near_best_coverage
        >= float(gate_config["chi_near_best_coverage_min"])
    )
    return {
        "dataset_n": int(len(frame)),
        "M_exact_recovery": m_exact,
        "M_wilson_low": m_low,
        "M_wilson_high": m_high,
        "chi_exact_recovery": chi_exact,
        "chi_wilson_low": chi_low,
        "chi_wilson_high": chi_high,
        "chi_near_best_coverage": near_best_coverage,
        "chi_supported": chi_supported,
        "gates": gate_config,
        "M_confusion_rows": m_confusion_rows,
        "chi_confusion_rows": chi_confusion_rows,
        "parameter_rows": parameter_rows,
        "error_correlation_rows": correlation_rows,
        "dataset_rows": frame.to_dict(orient="records"),
    }


def _save_png_atomic(figure: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.png"
    )
    try:
        figure.savefig(
            temporary,
            dpi=600,
            bbox_inches="tight",
            facecolor="white",
        )
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _configure_recovery_figure_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.dpi": 600,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def plot_module_recovery(
    summary: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, str]:
    """Plot the architecture confusion hero panel and supporting diagnostics."""

    import matplotlib.pyplot as plt

    _configure_recovery_figure_style()
    output = Path(output_path)
    confusion = pd.DataFrame(summary["confusion_rows"])
    cells = ["P", "PM", "PH", "PMH"]
    matrix = (
        confusion.pivot(index="true_cell", columns="predicted_cell", values="count")
        .reindex(index=cells, columns=cells)
        .fillna(0)
        .to_numpy(dtype=float)
    )
    cell_frame = pd.DataFrame(summary["cell_rows"])
    dataset_frame = pd.DataFrame(summary["dataset_rows"])
    source_paths = {
        "confusion": output.with_name("module_recovery_confusion_source.csv"),
        "cells": output.with_name("module_recovery_cell_source.csv"),
        "datasets": output.with_name("module_recovery_dataset_source.csv"),
    }
    _atomic_csv(source_paths["confusion"], confusion)
    _atomic_csv(source_paths["cells"], cell_frame)
    _atomic_csv(source_paths["datasets"], dataset_frame)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    ax = axes[0, 0]
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0)
    for row in range(4):
        for column in range(4):
            ax.text(column, row, f"{int(matrix[row, column])}", ha="center", va="center")
    ax.set_xticks(range(4), cells)
    ax.set_yticks(range(4), cells)
    ax.set_xlabel("Recovered architecture")
    ax.set_ylabel("Generating architecture")
    ax.set_title("a  Held-out architecture recovery", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Datasets", fraction=0.046)

    ax = axes[0, 1]
    ordered = cell_frame.set_index("true_cell").reindex(cells)
    values = ordered["exact_recovery"].to_numpy(dtype=float)
    lower = values - ordered["wilson_low"].to_numpy(dtype=float)
    upper = ordered["wilson_high"].to_numpy(dtype=float) - values
    ax.bar(cells, values, color="#5B8DB8", width=0.68)
    ax.errorbar(cells, values, yerr=np.vstack([lower, upper]), fmt="none", color="#263746", capsize=2)
    ax.axhline(0.5, color="#A65E4E", linestyle="--", linewidth=1)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Exact recovery rate")
    ax.set_title("b  Recovery by true cell", loc="left", fontweight="bold")

    ax = axes[1, 0]
    for index, cell in enumerate(cells):
        values = dataset_frame.loc[
            dataset_frame["true_cell"].eq(cell), "true_delta_nll"
        ].to_numpy(dtype=float)
        ax.scatter(
            np.full(values.size, index),
            values,
            color="#5B8DB8",
            edgecolor="white",
            linewidth=0.4,
            s=22,
            zorder=3,
        )
    ax.axhline(2.0, color="#A65E4E", linestyle="--", linewidth=1)
    ax.set_xticks(range(4), cells)
    ax.set_ylabel(r"True-cell $\Delta$ total NLL")
    ax.set_title("c  Near-best coverage", loc="left", fontweight="bold")

    ax = axes[1, 1]
    if "generated_accuracy" in dataset_frame:
        for index, cell in enumerate(cells):
            values = dataset_frame.loc[
                dataset_frame["true_cell"].eq(cell), "generated_accuracy"
            ].to_numpy(dtype=float)
            ax.scatter(
                np.full(values.size, index),
                values,
                color="#8AAE92",
                edgecolor="white",
                linewidth=0.4,
                s=22,
            )
        ax.set_xticks(range(4), cells)
        ax.set_ylabel("Generated choice accuracy")
        ax.set_ylim(0.0, 1.0)
    else:
        ax.text(0.5, 0.5, "Accuracy unavailable", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("d  Synthetic behavior", loc="left", fontweight="bold")
    _save_png_atomic(fig, output)
    plt.close(fig)
    return {name: str(path) for name, path in source_paths.items()}


def plot_parameter_recovery(
    summary: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, str]:
    """Plot readout confusion and parameter-level identifiability diagnostics."""

    import matplotlib.pyplot as plt

    _configure_recovery_figure_style()
    output = Path(output_path)
    m_confusion = pd.DataFrame(summary["M_confusion_rows"])
    confusion = pd.DataFrame(summary["chi_confusion_rows"])
    parameter_frame = pd.DataFrame(summary["parameter_rows"])
    dataset_frame = pd.DataFrame(summary["dataset_rows"])
    source_paths = {
        "M": output.with_name("parameter_recovery_M_source.csv"),
        "chi": output.with_name("parameter_recovery_chi_source.csv"),
        "parameters": output.with_name("parameter_recovery_metric_source.csv"),
        "datasets": output.with_name("parameter_recovery_dataset_source.csv"),
    }
    _atomic_csv(source_paths["M"], m_confusion)
    _atomic_csv(source_paths["chi"], confusion)
    _atomic_csv(source_paths["parameters"], parameter_frame)
    _atomic_csv(source_paths["datasets"], dataset_frame)
    m_levels = sorted(
        set(m_confusion["true_M"].astype(int))
        | set(m_confusion["estimated_M"].astype(int))
    )
    m_matrix = (
        m_confusion.pivot(index="true_M", columns="estimated_M", values="count")
        .reindex(index=m_levels, columns=m_levels)
        .fillna(0)
        .to_numpy(dtype=float)
    )
    chi_matrix = (
        confusion.pivot(index="true_chi", columns="estimated_chi", values="count")
        .reindex(index=[0, 1], columns=[0, 1])
        .fillna(0)
        .to_numpy(dtype=float)
    )
    parameters = parameter_frame["parameter"].astype(str).tolist()
    colors = [
        "#5B8DB8" if bool(value) else "#B7BEC5"
        for value in parameter_frame["supported"]
    ]
    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    axes = fig.subplot_mosaic(
        [["M", "chi", "zero"], ["mae", "mae", "rank"]],
        width_ratios=[1.0, 1.0, 1.0],
    )
    ax = axes["M"]
    image = ax.imshow(m_matrix, cmap="Blues", vmin=0.0)
    for row in range(len(m_levels)):
        for column in range(len(m_levels)):
            ax.text(
                column,
                row,
                f"{int(m_matrix[row, column])}",
                ha="center",
                va="center",
            )
    ax.set_xticks(range(len(m_levels)), m_levels)
    ax.set_yticks(range(len(m_levels)), m_levels)
    ax.set_xlabel("Recovered M")
    ax.set_ylabel("Generating M")
    ax.set_title("a  Capacity recovery", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Datasets", fraction=0.046)

    ax = axes["chi"]
    image = ax.imshow(chi_matrix, cmap="Blues", vmin=0.0)
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{int(chi_matrix[row, column])}", ha="center", va="center")
    ax.set_xticks([0, 1], ["Mixture", "Single rule"])
    ax.set_yticks([0, 1], ["Mixture", "Single rule"])
    ax.set_xlabel("Recovered readout")
    ax.set_ylabel("Generating readout")
    ax.set_title("b  Readout recovery", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, label="Datasets", fraction=0.046)

    ax = axes["mae"]
    ax.barh(parameters, parameter_frame["normalized_mae"], color=colors)
    ax.axvline(0.20, color="#A65E4E", linestyle="--", linewidth=1)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized MAE")
    ax.set_title("d  Parameter error", loc="left", fontweight="bold")

    ax = axes["rank"]
    ax.barh(parameters, parameter_frame["spearman"], color=colors)
    ax.axvline(0.60, color="#A65E4E", linestyle="--", linewidth=1)
    ax.set_xlim(-1.0, 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("Truth–estimate Spearman")
    ax.set_title("e  Rank recovery", loc="left", fontweight="bold")

    ax = axes["zero"]
    zero_frame = parameter_frame.loc[
        parameter_frame["zero_positive_balanced_accuracy"].notna()
    ]
    ax.bar(
        zero_frame["parameter"],
        zero_frame["zero_positive_balanced_accuracy"],
        color="#8AAE92",
        width=0.68,
    )
    ax.axhline(0.70, color="#A65E4E", linestyle="--", linewidth=1)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("c  Exact-zero detection", loc="left", fontweight="bold")
    _save_png_atomic(fig, output)
    plt.close(fig)
    return {name: str(path) for name, path in source_paths.items()}
