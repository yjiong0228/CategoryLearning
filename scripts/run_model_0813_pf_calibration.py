#!/usr/bin/env python3
"""Run Phase-1 numerical calibration for the 0813 particle-filter model.

The calibration reuses the existing 32/64-particle candidate-bank scores,
adds progressively larger particle counts only while the preregistered stop
gate fails, and separately decomposes uniform trajectory mixing, choice
importance weighting, and resampling at an equal particle budget.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from itertools import combinations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0813_pf_parameter_recovery import (  # noqa: E402
    _fit_candidate_bank,
    _load_subject_frames,
    _readout_args,
    _subject_engine,
)
from src.Bayesian_state.inference.backends.particle_filter import (  # noqa: E402
    effective_sample_size,
    run_state_model_particle_filter,
)
from src.Bayesian_state.simulation.config import load_yaml  # noqa: E402
from src.Bayesian_state.utils.seeding import stable_seed  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/specific_models/model_0813_pf_calibration.yaml"
FEATURE_COLUMNS = ("feature1", "feature2", "feature3", "feature4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase",
        choices=("ranking", "decomposition", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use one dataset, three candidates, short trials, and small particles.",
    )
    return parser.parse_args()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _worktree_dirty() -> bool | None:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _ranking_cache_path(
    output: Path,
    dataset_id: str,
    particle_count: int,
    filter_repeat: int,
    profile_id: str,
) -> Path:
    return (
        output
        / "cache"
        / "ranking"
        / dataset_id
        / f"particles_{particle_count}_seed_{filter_repeat:02d}"
        / f"{profile_id}.json"
    )


def _decomposition_cache_path(
    output: Path, subject_id: int, mode_id: str, filter_repeat: int
) -> Path:
    return (
        output
        / "cache"
        / "filter_decomposition"
        / f"subject_{subject_id}"
        / f"{mode_id}_seed_{filter_repeat:02d}.json"
    )


def _rank_filter_seed(
    base_seed: int, dataset_id: str, particle_count: int, filter_repeat: int
) -> int:
    return stable_seed(
        {
            "seed_role": "model0813_phase1_candidate_ranking",
            "base_seed": int(base_seed),
            "dataset_id": str(dataset_id),
            "particle_count": int(particle_count),
            "filter_repeat": int(filter_repeat),
        }
    )


def _decomposition_filter_seed(
    base_seed: int, subject_id: int, particle_count: int, filter_repeat: int
) -> int:
    return stable_seed(
        {
            "seed_role": "model0813_phase1_filter_decomposition",
            "base_seed": int(base_seed),
            "subject_id": int(subject_id),
            "particle_count": int(particle_count),
            "filter_repeat": int(filter_repeat),
        }
    )


def _score_ranking_candidate(
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
    dataset_id = dataset_path.stem
    profile_id = str(profile["profile_id"])
    cache_path = _ranking_cache_path(
        output, dataset_id, particle_count, filter_repeat, profile_id
    )
    if cache_path.exists() and not force:
        return dict(json.loads(cache_path.read_text(encoding="utf-8")))
    filter_seed = _rank_filter_seed(
        base_seed, dataset_id, particle_count, filter_repeat
    )
    started = time.perf_counter()
    rows = _fit_candidate_bank(
        dataset_path=dataset_path,
        base_config=base_config,
        base_path=base_path,
        dataset_paths=dataset_paths,
        profiles=[profile],
        particle_count=int(particle_count),
        resample_threshold_fraction=float(resample_threshold_fraction),
        filter_seed=int(filter_seed),
        analysis_role="phase1_ranking_calibration",
        filter_repeat=int(filter_repeat),
    )
    if len(rows) != 1:
        raise RuntimeError("one ranking task must produce exactly one score row")
    row = dict(rows[0])
    row.update(
        {
            "source": "phase1_new",
            "runtime_seconds": float(time.perf_counter() - started),
        }
    )
    _atomic_json(cache_path, row)
    return row


def _legacy_scores(
    path: Path,
    dataset_ids: Sequence[str],
    profile_ids: Sequence[str],
    particle_counts: Sequence[int],
) -> pd.DataFrame:
    scores = pd.read_csv(path)
    selected = scores.loc[
        scores["dataset_id"].astype(str).isin({str(value) for value in dataset_ids})
        & scores["fit_profile_id"].astype(str).isin(
            {str(value) for value in profile_ids}
        )
        & scores["particle_count"].astype(int).isin(
            {int(value) for value in particle_counts}
        )
    ].copy()
    expected = len(dataset_ids) * len(profile_ids) * len(particle_counts) * int(
        selected["filter_repeat"].nunique()
    )
    if len(selected) != expected:
        raise ValueError(
            f"legacy stability scores are incomplete: {len(selected)} vs {expected}"
        )
    selected["source"] = "parameter_recovery_reuse"
    selected["runtime_seconds"] = np.nan
    return selected


def summarize_ranking(
    scores: pd.DataFrame,
    thresholds: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {
        "dataset_id",
        "fit_profile_id",
        "particle_count",
        "filter_repeat",
        "log_likelihood",
    }
    missing = required - set(scores)
    if missing:
        raise ValueError(f"ranking scores missing columns: {sorted(missing)}")
    working = scores.copy()
    working["total_nll"] = -pd.to_numeric(
        working["log_likelihood"], errors="raise"
    )
    counts = sorted(working["particle_count"].astype(int).unique())
    correlation_rows: list[dict[str, Any]] = []
    winner_rows: list[dict[str, Any]] = []
    sd_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []

    for count_index, particle_count in enumerate(counts):
        count_frame = working.loc[
            working["particle_count"].astype(int).eq(particle_count)
        ]
        repeats = sorted(count_frame["filter_repeat"].astype(int).unique())
        seed_rhos: list[float] = []
        modal_agreements: list[float] = []
        for dataset_id, dataset_frame in count_frame.groupby(
            "dataset_id", sort=True
        ):
            by_repeat = {
                int(repeat): frame.sort_values("fit_profile_id")
                for repeat, frame in dataset_frame.groupby(
                    "filter_repeat", sort=True
                )
            }
            candidate_orders = [
                tuple(frame["fit_profile_id"].astype(str))
                for frame in by_repeat.values()
            ]
            if len(set(candidate_orders)) != 1:
                raise ValueError(
                    f"candidate bank differs across seeds for {dataset_id}"
                )
            winners: list[str] = []
            for repeat, frame in by_repeat.items():
                best = frame.loc[frame["total_nll"].idxmin()]
                winners.append(str(best["fit_profile_id"]))
                winner_rows.append(
                    {
                        "dataset_id": str(dataset_id),
                        "particle_count": int(particle_count),
                        "filter_repeat": int(repeat),
                        "winner_profile_id": str(best["fit_profile_id"]),
                        "winner_total_nll": float(best["total_nll"]),
                    }
                )
            modal = pd.Series(winners).value_counts().iloc[0] / float(len(winners))
            modal_agreements.append(float(modal))
            for left, right in combinations(repeats, 2):
                left_values = by_repeat[left]["total_nll"].to_numpy(dtype=float)
                right_values = by_repeat[right]["total_nll"].to_numpy(dtype=float)
                rho = float(spearmanr(left_values, right_values).statistic)
                seed_rhos.append(rho)
                correlation_rows.append(
                    {
                        "dataset_id": str(dataset_id),
                        "particle_count": int(particle_count),
                        "left_repeat": int(left),
                        "right_repeat": int(right),
                        "candidate_nll_spearman": rho,
                    }
                )
        candidate_sd = (
            count_frame.groupby(["dataset_id", "fit_profile_id"])["total_nll"]
            .std(ddof=1)
            .dropna()
        )
        for (dataset_id, profile_id), value in candidate_sd.items():
            sd_rows.append(
                {
                    "dataset_id": str(dataset_id),
                    "fit_profile_id": str(profile_id),
                    "particle_count": int(particle_count),
                    "candidate_total_nll_sd": float(value),
                }
            )
        within_seed_ranges = count_frame.groupby(
            ["dataset_id", "filter_repeat"]
        )["total_nll"].agg(lambda values: float(values.max() - values.min()))
        median_sd = float(candidate_sd.median()) if len(candidate_sd) else float("nan")
        median_range = float(within_seed_ranges.median())
        noise_to_signal = median_sd / median_range if median_range > 0.0 else float("inf")

        cross_median = float("nan")
        cross_minimum = float("nan")
        cross_winner_agreement = float("nan")
        previous_count = None
        if count_index > 0:
            previous_count = counts[count_index - 1]
            previous = working.loc[
                working["particle_count"].astype(int).eq(previous_count)
            ]
            cross_rhos: list[float] = []
            winner_matches: list[float] = []
            for dataset_id in sorted(count_frame["dataset_id"].astype(str).unique()):
                current_mean = (
                    count_frame.loc[count_frame["dataset_id"].astype(str).eq(dataset_id)]
                    .groupby("fit_profile_id")["total_nll"]
                    .mean()
                    .sort_index()
                )
                previous_mean = (
                    previous.loc[previous["dataset_id"].astype(str).eq(dataset_id)]
                    .groupby("fit_profile_id")["total_nll"]
                    .mean()
                    .sort_index()
                )
                if not current_mean.index.equals(previous_mean.index):
                    raise ValueError("cross-count candidate banks do not align")
                rho = float(
                    spearmanr(
                        previous_mean.to_numpy(dtype=float),
                        current_mean.to_numpy(dtype=float),
                    ).statistic
                )
                same_winner = float(previous_mean.idxmin() == current_mean.idxmin())
                cross_rhos.append(rho)
                winner_matches.append(same_winner)
                cross_rows.append(
                    {
                        "dataset_id": str(dataset_id),
                        "lower_particle_count": int(previous_count),
                        "higher_particle_count": int(particle_count),
                        "pooled_candidate_nll_spearman": rho,
                        "winner_agreement": bool(same_winner),
                        "lower_winner": str(previous_mean.idxmin()),
                        "higher_winner": str(current_mean.idxmin()),
                    }
                )
            cross_median = float(np.median(cross_rhos))
            cross_minimum = float(np.min(cross_rhos))
            cross_winner_agreement = float(np.mean(winner_matches))

        repeat_pass = len(repeats) >= int(thresholds["required_seed_repeats"])
        median_rho = float(np.median(seed_rhos)) if seed_rhos else float("nan")
        minimum_rho = float(np.min(seed_rhos)) if seed_rhos else float("nan")
        modal_mean = float(np.mean(modal_agreements))
        gate_checks = {
            "seed_repeat_pass": repeat_pass,
            "median_seed_rank_pass": bool(
                seed_rhos
                and median_rho >= float(thresholds["median_seed_rank_spearman"])
            ),
            "minimum_seed_rank_pass": bool(
                seed_rhos
                and minimum_rho >= float(thresholds["minimum_seed_rank_spearman"])
            ),
            "modal_winner_pass": bool(
                modal_mean
                >= float(thresholds["mean_modal_winner_agreement"])
            ),
            "nll_sd_pass": bool(
                np.isfinite(median_sd)
                and median_sd
                <= float(thresholds["median_candidate_total_nll_sd"])
            ),
            "noise_signal_pass": bool(
                np.isfinite(noise_to_signal)
                and noise_to_signal
                <= float(thresholds["maximum_noise_to_signal_ratio"])
            ),
            "cross_count_rank_pass": bool(
                np.isfinite(cross_median)
                and cross_median
                >= float(thresholds["median_cross_count_rank_spearman"])
            ),
            "cross_count_winner_pass": bool(
                np.isfinite(cross_winner_agreement)
                and cross_winner_agreement
                >= float(thresholds["cross_count_winner_agreement"])
            ),
        }
        count_rows.append(
            {
                "particle_count": int(particle_count),
                "dataset_n": int(count_frame["dataset_id"].nunique()),
                "candidate_n": int(count_frame["fit_profile_id"].nunique()),
                "seed_repeat_n": int(len(repeats)),
                "median_seed_candidate_rank_spearman": median_rho,
                "minimum_seed_candidate_rank_spearman": minimum_rho,
                "mean_modal_winner_agreement": modal_mean,
                "median_candidate_total_nll_sd": median_sd,
                "median_within_seed_candidate_nll_range": median_range,
                "noise_to_signal_ratio": float(noise_to_signal),
                "previous_particle_count": previous_count,
                "median_cross_count_rank_spearman": cross_median,
                "minimum_cross_count_rank_spearman": cross_minimum,
                "cross_count_winner_agreement": cross_winner_agreement,
                **gate_checks,
                "all_stability_gates_pass": bool(all(gate_checks.values())),
            }
        )

    particle_summary = pd.DataFrame(count_rows)
    stable = particle_summary.loc[
        particle_summary["all_stability_gates_pass"].astype(bool),
        "particle_count",
    ]
    summary = {
        "dataset_n": int(working["dataset_id"].nunique()),
        "candidate_n": int(working["fit_profile_id"].nunique()),
        "particle_counts_evaluated": [int(value) for value in counts],
        "minimum_stable_particle_count": (
            None if stable.empty else int(stable.min())
        ),
        "status": (
            "stable_setting_found" if not stable.empty else "unstable_through_maximum_evaluated"
        ),
        "stop_thresholds": dict(thresholds),
        "independent_unit": (
            "none; three fixed synthetic trajectories are technical calibration cases"
        ),
    }
    return (
        particle_summary,
        pd.DataFrame(correlation_rows),
        pd.DataFrame(cross_rows),
        pd.DataFrame(winner_rows),
        summary,
    )


def _score_filter_mode(
    *,
    output: Path,
    subject_id: int,
    frame: pd.DataFrame,
    base_config: Mapping[str, Any],
    base_path: Path,
    dataset_paths: Mapping[str, Path],
    mode: Mapping[str, Any],
    particle_count: int,
    filter_repeat: int,
    base_seed: int,
    force: bool,
) -> dict[str, Any]:
    mode_id = str(mode["id"])
    cache_path = _decomposition_cache_path(
        output, subject_id, mode_id, filter_repeat
    )
    if cache_path.exists() and not force:
        return dict(json.loads(cache_path.read_text(encoding="utf-8")))
    engine = _subject_engine(base_config, base_path, subject_id)
    filter_seed = _decomposition_filter_seed(
        base_seed, subject_id, particle_count, filter_repeat
    )
    choices = frame["choice"].to_numpy(dtype=int)
    started = time.perf_counter()
    result = run_state_model_particle_filter(
        engine_config=engine,
        subject_id=subject_id,
        stimulus=frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
        choices=choices,
        feedback=frame["feedback"].to_numpy(dtype=float),
        particle_count=int(particle_count),
        filter_seed=int(filter_seed),
        resample_threshold_fraction=float(mode["resample_threshold_fraction"]),
        condition_on_observed_choice=bool(mode["condition_on_observed_choice"]),
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
        **_readout_args(engine),
    )
    probabilities = np.asarray(result.marginal_probabilities, dtype=float)
    selected = probabilities[np.arange(len(choices)), choices - 1]
    log_likelihood = float(np.log(np.clip(selected, 1e-12, 1.0)).sum())
    pre_ess = np.asarray(result.pre_choice_ess, dtype=float)
    post_ess = np.asarray(result.post_choice_ess, dtype=float)
    resampled = np.asarray(result.resampled, dtype=bool)
    final_weights = np.asarray(result.final_weights, dtype=float)
    unique = np.asarray(result.resampling_unique_ancestors, dtype=float)
    row = {
        "subject_id": int(subject_id),
        "mode_id": mode_id,
        "mode_label": str(mode["label"]),
        "condition_on_observed_choice": bool(
            mode["condition_on_observed_choice"]
        ),
        "resample_threshold_fraction": float(
            mode["resample_threshold_fraction"]
        ),
        "filter_repeat": int(filter_repeat),
        "filter_seed": int(filter_seed),
        "particle_count": int(particle_count),
        "trial_count": int(len(frame)),
        "log_likelihood": log_likelihood,
        "total_nll": float(-log_likelihood),
        "mean_nll": float(-log_likelihood / len(frame)),
        "mean_pre_choice_ess_fraction": float(np.mean(pre_ess) / particle_count),
        "mean_post_choice_ess_fraction": float(np.mean(post_ess) / particle_count),
        "terminal_post_choice_ess_fraction": float(post_ess[-1] / particle_count),
        "final_weight_ess_fraction": float(
            effective_sample_size(final_weights) / particle_count
        ),
        "resampling_fraction": float(np.mean(resampled)),
        "mean_unique_ancestor_fraction_on_resampled_trials": (
            float(np.mean(unique[resampled]) / particle_count)
            if np.any(resampled)
            else 1.0
        ),
        "runtime_seconds": float(time.perf_counter() - started),
        "selected_choice_probabilities": selected,
        "pre_choice_ess": pre_ess,
        "post_choice_ess": post_ess,
        "resampled": resampled,
        "final_weights": final_weights,
    }
    if not bool(mode["condition_on_observed_choice"]):
        np.testing.assert_allclose(post_ess, float(particle_count))
        np.testing.assert_allclose(final_weights, 1.0 / float(particle_count))
        if np.any(resampled):
            raise AssertionError("unweighted mixture must not resample")
    _atomic_json(cache_path, row)
    return row


def summarize_decomposition(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metric_columns = [
        "mean_nll",
        "mean_pre_choice_ess_fraction",
        "mean_post_choice_ess_fraction",
        "terminal_post_choice_ess_fraction",
        "final_weight_ess_fraction",
        "resampling_fraction",
        "mean_unique_ancestor_fraction_on_resampled_trials",
        "runtime_seconds",
    ]
    subject_mode = (
        scores.groupby(["subject_id", "mode_id", "mode_label"], as_index=False)[
            metric_columns
        ]
        .agg(["mean", "std"])
    )
    subject_mode.columns = [
        "_".join(value for value in column if value)
        if isinstance(column, tuple)
        else str(column)
        for column in subject_mode.columns
    ]
    subject_mode = subject_mode.rename(
        columns={
            "subject_id_": "subject_id",
            "mode_id_": "mode_id",
            "mode_label_": "mode_label",
        }
    )

    pivot = scores.pivot(
        index=["subject_id", "filter_repeat"],
        columns="mode_id",
        values="mean_nll",
    )
    required_modes = {
        "unweighted_mixture",
        "importance_no_resampling",
        "full_particle_filter",
    }
    if set(pivot.columns) != required_modes:
        raise ValueError("filter decomposition requires the three declared modes")
    contrasts = pivot.reset_index()
    contrasts["choice_weighting_gain"] = (
        contrasts["unweighted_mixture"]
        - contrasts["importance_no_resampling"]
    )
    contrasts["resampling_gain"] = (
        contrasts["importance_no_resampling"]
        - contrasts["full_particle_filter"]
    )
    contrasts["full_filter_gain"] = (
        contrasts["unweighted_mixture"]
        - contrasts["full_particle_filter"]
    )
    contrast_summary: dict[str, Any] = {}
    for column in (
        "choice_weighting_gain",
        "resampling_gain",
        "full_filter_gain",
    ):
        values = contrasts[column].to_numpy(dtype=float)
        by_subject = contrasts.groupby("subject_id")[column].mean()
        contrast_summary[column] = {
            "mean_across_subject_seed_pairs": float(np.mean(values)),
            "median_across_subject_seed_pairs": float(np.median(values)),
            "minimum_subject_seed_pair": float(np.min(values)),
            "maximum_subject_seed_pair": float(np.max(values)),
            "mean_of_subject_means": float(by_subject.mean()),
            "subject_means": {
                str(int(key)): float(value) for key, value in by_subject.items()
            },
            "positive_pair_fraction": float(np.mean(values > 0.0)),
        }
    summary = {
        "subject_n": int(scores["subject_id"].nunique()),
        "filter_seed_repeats": int(scores["filter_repeat"].nunique()),
        "technical_pair_n": int(len(contrasts)),
        "particle_count": int(scores["particle_count"].iloc[0]),
        "trial_count": int(scores["trial_count"].iloc[0]),
        "contrasts": contrast_summary,
        "sign_definition": (
            "positive gain means the later inference layer has lower prequential mean NLL"
        ),
        "independent_unit": (
            "subject for descriptive heterogeneity; PF seeds are technical repeats"
        ),
    }
    return subject_mode, contrasts, summary


def _write_figure_contract(output: Path, ranking_summary: Mapping[str, Any]) -> None:
    stable = ranking_summary.get("minimum_stable_particle_count")
    conclusion = (
        f"Candidate ordering first satisfies every numerical gate at {stable} particles."
        if stable is not None
        else "Candidate ordering does not satisfy every numerical gate through the largest evaluated particle count."
    )
    content = f"""# Figure contract and chart map

Core conclusion: {conclusion}

- Figure archetype: quantitative grid.
- Target output: technical Phase-1 report; Python/matplotlib; 7.2 × 6.6 inches; PNG at 300 dpi.
- Backend: Python only.
- Export policy: PNG only under the repository artifact rule.
- Hero evidence: within-count candidate-rank stability and its preregistered threshold.
- Validation evidence: candidate NLL Monte Carlo SD, inference-layer NLL contrasts, ESS and resampling diagnostics.
- Independent unit: the three fixed synthetic datasets and three observed subjects are coverage cases; PF seeds are technical repeats.
- Image integrity: all declared datasets, candidates, subjects, modes, and completed particle stages are plotted; no observations are removed.

| Panel | Question | Form | Evidence role | Palette/non-color encoding |
|---|---|---|---|---|
| a | Does candidate ordering stabilize as particles increase? | Median line with full seed-pair range and gate line | Hero numerical evidence | blue circles; dashed neutral gate |
| b | Does single-candidate likelihood noise shrink below the gate? | Boxplots plus individual dataset-candidate points | Monte Carlo uncertainty | blue open points; orange gate |
| c | How do weighting and resampling change choice NLL? | Paired subject lines over three ordered modes | Inference decomposition | subject-specific markers and line styles |
| d | How do the modes change ESS and resampling? | Grouped bars with subject points | Degeneracy diagnostics | blue/orange with marker overlay |

Reviewer risk: a low rank correlation can reflect genuinely near-equivalent candidates as well as PF noise. A positive in-sample/prequential NLL gain from choice conditioning is not parameter identification or held-out generalization. The three calibration datasets and subjects do not support population inference.
"""
    (output / "chart_map.md").write_text(content, encoding="utf-8")


def _write_figure(
    output: Path,
    particle_summary: pd.DataFrame,
    rank_correlations: pd.DataFrame,
    candidate_sd: pd.DataFrame,
    subject_mode: pd.DataFrame,
    decomposition_scores: pd.DataFrame,
    thresholds: Mapping[str, Any],
    filename: str,
) -> Path:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
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
    grid_color = "#E6E9ED"

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.flat

    counts = particle_summary["particle_count"].astype(int).to_numpy()
    medians = particle_summary["median_seed_candidate_rank_spearman"].to_numpy(
        dtype=float
    )
    grouped_rhos = {
        int(count): frame["candidate_nll_spearman"].to_numpy(dtype=float)
        for count, frame in rank_correlations.groupby("particle_count", sort=True)
    }
    minima = np.asarray([np.min(grouped_rhos[int(count)]) for count in counts])
    maxima = np.asarray([np.max(grouped_rhos[int(count)]) for count in counts])
    ax_a.errorbar(
        counts,
        medians,
        yerr=np.vstack([medians - minima, maxima - medians]),
        color=blue,
        marker="o",
        markersize=5,
        linewidth=1.5,
        capsize=3,
        label="Median and full seed-pair range",
    )
    ax_a.axhline(
        float(thresholds["median_seed_rank_spearman"]),
        color=charcoal,
        linestyle="--",
        linewidth=1.0,
        label="Median-rank gate",
    )
    ax_a.axhline(0.0, color="#A8ADB5", linewidth=0.7)
    ax_a.set_xscale("log", base=2)
    ax_a.set_xticks(counts, [str(value) for value in counts])
    ax_a.set_ylim(-1.05, 1.05)
    ax_a.set_xlabel("Particle count")
    ax_a.set_ylabel("Candidate-rank Spearman ρ")
    ax_a.set_title("Within-count candidate-rank stability", loc="left", fontsize=9)
    ax_a.grid(axis="y", color=grid_color, linewidth=0.7)
    ax_a.legend(fontsize=6.5, loc="lower right")

    sd_groups = [
        candidate_sd.loc[
            candidate_sd["particle_count"].astype(int).eq(count),
            "candidate_total_nll_sd",
        ].to_numpy(dtype=float)
        for count in counts
    ]
    ax_b.boxplot(
        sd_groups,
        positions=np.arange(len(counts)),
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "#DCE8F6", "edgecolor": blue},
        medianprops={"color": charcoal, "linewidth": 1.2},
        whiskerprops={"color": blue},
        capprops={"color": blue},
    )
    rng = np.random.default_rng(20260814)
    for index, values in enumerate(sd_groups):
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax_b.scatter(
            index + jitter,
            values,
            s=10,
            facecolors="none",
            edgecolors=blue,
            linewidths=0.55,
            alpha=0.55,
        )
    ax_b.axhline(
        float(thresholds["median_candidate_total_nll_sd"]),
        color=orange,
        linestyle="--",
        linewidth=1.0,
        label="Median-SD gate",
    )
    ax_b.set_xticks(np.arange(len(counts)), [str(value) for value in counts])
    ax_b.set_xlabel("Particle count")
    ax_b.set_ylabel("Across-seed total NLL SD")
    ax_b.set_title("Single-candidate Monte Carlo variation", loc="left", fontsize=9)
    ax_b.grid(axis="y", color=grid_color, linewidth=0.7)
    ax_b.legend(fontsize=6.5, loc="upper right")

    mode_order = [
        "unweighted_mixture",
        "importance_no_resampling",
        "full_particle_filter",
    ]
    mode_labels = ["Uniform\nmixture", "Choice\nweighting", "Full PF"]
    subject_mean = (
        decomposition_scores.groupby(["subject_id", "mode_id"])["mean_nll"]
        .mean()
        .unstack("mode_id")
        .reindex(columns=mode_order)
    )
    markers = ("o", "s", "^")
    linestyles = ("-", "--", "-.")
    colors = (blue, orange, olive)
    for index, (subject_id, row) in enumerate(subject_mean.iterrows()):
        ax_c.plot(
            np.arange(3),
            row.to_numpy(dtype=float),
            marker=markers[index % len(markers)],
            linestyle=linestyles[index % len(linestyles)],
            color=colors[index % len(colors)],
            linewidth=1.2,
            markersize=4,
            label=f"S{int(subject_id)}",
        )
    ax_c.set_xticks(np.arange(3), mode_labels)
    ax_c.set_ylabel("Prequential mean choice NLL")
    ax_c.set_title("Equal-budget filtering decomposition", loc="left", fontsize=9)
    ax_c.grid(axis="y", color=grid_color, linewidth=0.7)
    ax_c.legend(fontsize=6.5, loc="best", ncol=3)

    metrics = ["mean_post_choice_ess_fraction", "resampling_fraction"]
    metric_labels = ["Post-choice ESS / particles", "Resampled-trial fraction"]
    width = 0.34
    positions = np.arange(3)
    for metric_index, (metric, label, color) in enumerate(
        zip(metrics, metric_labels, (blue, orange))
    ):
        means = (
            decomposition_scores.groupby("mode_id")[metric]
            .mean()
            .reindex(mode_order)
            .to_numpy(dtype=float)
        )
        x = positions + (metric_index - 0.5) * width
        ax_d.bar(
            x,
            means,
            width=width,
            color=color,
            alpha=0.75,
            edgecolor=charcoal,
            linewidth=0.5,
            label=label,
        )
        subject_values = (
            decomposition_scores.groupby(["subject_id", "mode_id"])[metric]
            .mean()
            .unstack("mode_id")
            .reindex(columns=mode_order)
        )
        for subject_index, row in enumerate(subject_values.to_numpy(dtype=float)):
            ax_d.scatter(
                x,
                row,
                marker=markers[subject_index % len(markers)],
                s=15,
                facecolors="white",
                edgecolors=charcoal,
                linewidths=0.6,
                zorder=3,
            )
    ax_d.set_xticks(positions, mode_labels)
    ax_d.set_ylim(0.0, 1.05)
    ax_d.set_ylabel("Fraction")
    ax_d.set_title("Weight degeneracy and resampling", loc="left", fontsize=9)
    ax_d.grid(axis="y", color=grid_color, linewidth=0.7)
    ax_d.legend(fontsize=6.2, loc="upper right")

    for label, axis in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
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
        "0813 particle-filter numerical calibration",
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


def _format_number(value: Any, digits: int = 3) -> str:
    number = float(value)
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def _write_readme(
    output: Path,
    particle_summary: pd.DataFrame,
    ranking_summary: Mapping[str, Any],
    decomposition_summary: Mapping[str, Any],
) -> None:
    stable = ranking_summary["minimum_stable_particle_count"]
    maximum = max(ranking_summary["particle_counts_evaluated"])
    if stable is None:
        ranking_text = (
            f"候选排序在最高 {maximum} particles 内仍未同时通过全部预注册门槛；"
            "因此暂时没有可用于正式机制去留比较的最小稳定 PF 设置。"
        )
    else:
        ranking_text = (
            f"{int(stable)} particles 是本轮第一个同时通过全部预注册数值门槛的设置。"
        )
    rows = []
    for item in particle_summary.to_dict(orient="records"):
        rows.append(
            "| {particle_count} | {seed_repeat_n} | {rho} | {minimum} | "
            "{agreement} | {sd} | {cross} | {passed} |".format(
                particle_count=int(item["particle_count"]),
                seed_repeat_n=int(item["seed_repeat_n"]),
                rho=_format_number(item["median_seed_candidate_rank_spearman"]),
                minimum=_format_number(item["minimum_seed_candidate_rank_spearman"]),
                agreement=_format_number(item["mean_modal_winner_agreement"]),
                sd=_format_number(item["median_candidate_total_nll_sd"]),
                cross=_format_number(item["median_cross_count_rank_spearman"]),
                passed="yes" if item["all_stability_gates_pass"] else "no",
            )
        )
    contrasts = decomposition_summary["contrasts"]
    weighting = contrasts["choice_weighting_gain"]
    resampling = contrasts["resampling_gain"]
    full = contrasts["full_filter_gain"]
    content = f"""# Phase 1：PF 数值校准与过滤贡献分解

## 技术结论

{ranking_text}

本阶段复用了已有 32/64-particle 结果，并只在前一级未达门槛时增加粒子数。排名校准固定使用
3 条预注册自主合成轨迹、9 个联合参数候选和候选内配对 PF seeds。它们是技术覆盖案例，不是
总体被试样本。

| particles | seeds | seed-rank ρ 中位数 | seed-rank ρ 最小值 | 模态胜者一致率 | 候选 NLL SD 中位数 | 与前一级 pooled-rank ρ | 全部门槛 |
|---:|---:|---:|---:|---:|---:|---:|:---:|
{chr(10).join(rows)}

## Choice weighting 与 resampling 分别做了什么

等粒子预算分解使用被试 103、120、127 的前 {decomposition_summary['trial_count']} trials、
{decomposition_summary['particle_count']} particles 和 {decomposition_summary['filter_seed_repeats']}
个配对 seeds。定义 `gain = 前一层 mean NLL - 后一层 mean NLL`，正值表示后一层的逐试次
历史条件化预测更好。

- choice importance weighting 的 subject-mean gain：
  `{_format_number(weighting['mean_of_subject_means'], 4)}`；正向技术配对比例
  `{_format_number(weighting['positive_pair_fraction'], 3)}`。
- 在 weighting 之上增加 resampling 的 subject-mean gain：
  `{_format_number(resampling['mean_of_subject_means'], 4)}`；正向技术配对比例
  `{_format_number(resampling['positive_pair_fraction'], 3)}`。
- 完整 PF 相对均匀轨迹混合的 subject-mean gain：
  `{_format_number(full['mean_of_subject_means'], 4)}`；正向技术配对比例
  `{_format_number(full['positive_pair_fraction'], 3)}`。

这些是同一 observed history 上的 prequential/in-sample-history 分解，不是 held-out 泛化检验。
Choice weighting 会让后续 latent-path 分布利用过去选择；resampling 的作用是缓解长期 importance
weights 退化，它不是额外的认知机制。

## 预注册门槛

单个粒子数必须同时满足：至少 3 seeds；seed-pair 排名相关中位数 ≥0.80、最小值 ≥0.50；
平均模态胜者一致率 ≥0.75；单候选跨 seed 总 NLL SD 中位数 ≤0.50；noise/signal ≤0.20；
相对前一级的 pooled-candidate rank ρ 中位数 ≥0.90，且 pooled 胜者完全一致。门槛在查看
128/256 结果前写入配置，没有按结果调整。

## 文件

- `ranking_scores.csv`：复用及新增的所有候选得分。
- `particle_count_summary.csv`：逐粒子数门槛结果。
- `seed_rank_correlations.csv`、`cross_count_correlations.csv`：seed 内和跨粒子数排名复核。
- `candidate_nll_sd.csv`、`winner_stability.csv`：似然方差与胜者稳定性。
- `filter_decomposition_scores.csv`、`filter_decomposition_contrasts.csv`：三种推断模式的配对结果。
- `ranking_summary.json`、`filter_decomposition_summary.json`：机器可读结论。
- `pf_calibration_overview.png`、`chart_map.md`：技术图与预声明图形契约。
- `analysis_manifest.json`、`analysis_config_snapshot.json`：配置、代码和输入哈希。
- `VALIDATION.md`：独立重算、缓存完整性和图形 QA。
- `report.html`：可移植技术报告；若构建环境不兼容，具体 blocker 会记录在本目录。

## 解释边界与下一步

本阶段只能决定 PF 数值设置是否足以支持后续的**相对模型比较**。它不证明九个参数候选在
心理学上可区分，也不把 choice-conditioned filtering 的拟合收益解释为认知机制收益。只有找到
稳定设置后才进入 Phase 2/3；若最大粒子数仍失败，应先缩小候选差异、增加独立 seeds 或采用
更低方差的似然估计，不能继续在不稳定排名上做机制去留判断。
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
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _repo_path(config["output_dir"])
    )
    if args.smoke:
        output = output / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    base_path = _repo_path(config["base_simulation_config"])
    recovery_config_path = _repo_path(config["parameter_recovery_config"])
    profiles_path = _repo_path(config["candidate_profiles"])
    synthetic_dir = _repo_path(config["synthetic_dir"])
    legacy_path = _repo_path(config["legacy_stability_scores"])
    base_config = load_yaml(base_path)
    profiles = list(json.loads(profiles_path.read_text(encoding="utf-8")))
    ranking_config = deepcopy(dict(config["ranking_calibration"]))
    decomposition_config = deepcopy(dict(config["filter_decomposition"]))
    if args.smoke:
        ranking_config.update(
            {
                "dataset_ids": ranking_config["dataset_ids"][:1],
                "candidate_profile_ids": ranking_config["candidate_profile_ids"][:3],
                "legacy_particle_counts": [],
                "progressive_particle_counts": [4, 8],
                "filter_seed_repeats": 2,
                "n_jobs": 2,
            }
        )
        ranking_config["stop_thresholds"] = {
            **ranking_config["stop_thresholds"],
            "required_seed_repeats": 2,
            "median_cross_count_rank_spearman": -1.0,
            "cross_count_winner_agreement": 0.0,
        }
        decomposition_config.update(
            {
                "subjects": decomposition_config["subjects"][:1],
                "trials_per_subject": 16,
                "particle_count": 4,
                "filter_seed_repeats": 2,
                "n_jobs": 2,
            }
        )
    profile_ids = [str(value) for value in ranking_config["candidate_profile_ids"]]
    selected_profiles = [
        profile for profile in profiles if str(profile["profile_id"]) in profile_ids
    ]
    if [str(profile["profile_id"]) for profile in selected_profiles] != profile_ids:
        raise ValueError("candidate profile selection is incomplete or out of order")
    dataset_ids = [str(value) for value in ranking_config["dataset_ids"]]
    for dataset_id in dataset_ids:
        if not (synthetic_dir / f"{dataset_id}.npz").exists():
            raise FileNotFoundError(f"missing calibration dataset: {dataset_id}")

    subjects = [int(value) for value in decomposition_config["subjects"]]
    frames, dataset_paths = _load_subject_frames(
        base_config, base_path, subjects
    )
    trial_count = int(decomposition_config["trials_per_subject"])
    for subject_id, frame in frames.items():
        if len(frame) < trial_count:
            raise ValueError(f"subject {subject_id} has fewer than {trial_count} trials")
        frames[subject_id] = frame.iloc[:trial_count].reset_index(drop=True)

    snapshot = deepcopy(config)
    snapshot["ranking_calibration"] = ranking_config
    snapshot["filter_decomposition"] = decomposition_config
    _atomic_json(output / "analysis_config_snapshot.json", snapshot)
    manifest = {
        "analysis_id": str(config["analysis_id"]),
        "scope": str(config["scope"]),
        "status": "configured",
        "config": _relative(config_path),
        "config_sha256": _sha256(config_path),
        "base_simulation_config": _relative(base_path),
        "base_simulation_config_sha256": _sha256(base_path),
        "parameter_recovery_config": _relative(recovery_config_path),
        "parameter_recovery_config_sha256": _sha256(recovery_config_path),
        "candidate_profiles": _relative(profiles_path),
        "candidate_profiles_sha256": _sha256(profiles_path),
        "legacy_stability_scores": _relative(legacy_path),
        "legacy_stability_scores_sha256": _sha256(legacy_path),
        "runner": _relative(Path(__file__).resolve()),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "parameter_recovery_runner_sha256": _sha256(
            ROOT / "scripts/run_model_0813_pf_parameter_recovery.py"
        ),
        "bayesian_state_python_tree_sha256": _python_tree_sha256(
            ROOT / "src/Bayesian_state"
        ),
        "repository_head": _git_head(),
        "worktree_dirty": _worktree_dirty(),
        "ranking_design": ranking_config,
        "filter_decomposition_design": decomposition_config,
        "smoke": bool(args.smoke),
        "interpretation_boundary": (
            "technical numerical calibration; no population or mechanism-retention inference"
        ),
    }
    _atomic_json(output / "analysis_manifest.json", manifest)

    n_jobs = int(
        args.n_jobs
        if args.n_jobs is not None
        else max(
            int(ranking_config["n_jobs"]),
            int(decomposition_config["n_jobs"]),
        )
    )
    ranking_scores: pd.DataFrame | None = None
    if args.phase in {"ranking", "all"}:
        legacy_counts = [int(value) for value in ranking_config["legacy_particle_counts"]]
        if legacy_counts:
            ranking_scores = _legacy_scores(
                legacy_path, dataset_ids, profile_ids, legacy_counts
            )
        else:
            ranking_scores = pd.DataFrame()
        progressive_counts = [
            int(value) for value in ranking_config["progressive_particle_counts"]
        ]
        for particle_count in progressive_counts:
            jobs = [
                (dataset_id, profile, repeat)
                for dataset_id in dataset_ids
                for repeat in range(int(ranking_config["filter_seed_repeats"]))
                for profile in selected_profiles
            ]
            results = Parallel(
                n_jobs=min(n_jobs, len(jobs)), backend="loky", verbose=10
            )(
                delayed(_score_ranking_candidate)(
                    output=output,
                    dataset_path=synthetic_dir / f"{dataset_id}.npz",
                    base_config=base_config,
                    base_path=base_path,
                    dataset_paths=dataset_paths,
                    profile=profile,
                    particle_count=particle_count,
                    resample_threshold_fraction=float(
                        ranking_config["resample_threshold_fraction"]
                    ),
                    filter_repeat=repeat,
                    base_seed=int(ranking_config["base_seed"]),
                    force=bool(args.force),
                )
                for dataset_id, profile, repeat in jobs
            )
            ranking_scores = pd.concat(
                [ranking_scores, pd.DataFrame(results)], ignore_index=True
            )
            interim = summarize_ranking(
                ranking_scores, ranking_config["stop_thresholds"]
            )
            interim_particle, correlations, cross, winners, interim_summary = interim
            _atomic_csv(output / "ranking_scores.csv", ranking_scores)
            _atomic_csv(output / "particle_count_summary.csv", interim_particle)
            _atomic_csv(output / "seed_rank_correlations.csv", correlations)
            _atomic_csv(output / "cross_count_correlations.csv", cross)
            _atomic_csv(output / "winner_stability.csv", winners)
            _atomic_json(output / "ranking_summary.json", interim_summary)
            current = interim_particle.loc[
                interim_particle["particle_count"].astype(int).eq(particle_count)
            ]
            passed = bool(current["all_stability_gates_pass"].iloc[0])
            print(
                f"[phase1] particles={particle_count} gates_pass={passed}",
                flush=True,
            )
            if passed:
                break
        if args.phase == "ranking":
            return
    elif not (output / "ranking_scores.csv").exists():
        raise FileNotFoundError("ranking_scores.csv is required")

    decomposition_scores: pd.DataFrame | None = None
    if args.phase in {"decomposition", "all"}:
        modes = [dict(value) for value in decomposition_config["modes"]]
        jobs = [
            (subject_id, mode, repeat)
            for subject_id in subjects
            for repeat in range(int(decomposition_config["filter_seed_repeats"]))
            for mode in modes
        ]
        results = Parallel(
            n_jobs=min(n_jobs, len(jobs)), backend="loky", verbose=10
        )(
            delayed(_score_filter_mode)(
                output=output,
                subject_id=subject_id,
                frame=frames[subject_id],
                base_config=base_config,
                base_path=base_path,
                dataset_paths=dataset_paths,
                mode=mode,
                particle_count=int(decomposition_config["particle_count"]),
                filter_repeat=repeat,
                base_seed=int(decomposition_config["base_seed"]),
                force=bool(args.force),
            )
            for subject_id, mode, repeat in jobs
        )
        decomposition_scores = pd.DataFrame(results)
        scalar_columns = [
            column
            for column in decomposition_scores.columns
            if column
            not in {
                "selected_choice_probabilities",
                "pre_choice_ess",
                "post_choice_ess",
                "resampled",
                "final_weights",
            }
        ]
        _atomic_csv(
            output / "filter_decomposition_scores.csv",
            decomposition_scores[scalar_columns],
        )
        subject_mode, contrasts, decomposition_summary = summarize_decomposition(
            decomposition_scores
        )
        _atomic_csv(output / "filter_decomposition_by_subject.csv", subject_mode)
        _atomic_csv(output / "filter_decomposition_contrasts.csv", contrasts)
        _atomic_json(
            output / "filter_decomposition_summary.json", decomposition_summary
        )
        if args.phase == "decomposition":
            return
    elif not (output / "filter_decomposition_scores.csv").exists():
        raise FileNotFoundError("filter_decomposition_scores.csv is required")

    if args.phase in {"summarize", "all"}:
        ranking_scores = pd.read_csv(output / "ranking_scores.csv")
        (
            particle_summary,
            correlations,
            cross,
            winners,
            ranking_summary,
        ) = summarize_ranking(ranking_scores, ranking_config["stop_thresholds"])
        candidate_sd = (
            ranking_scores.assign(
                total_nll=-pd.to_numeric(
                    ranking_scores["log_likelihood"], errors="raise"
                )
            )
            .groupby(["dataset_id", "fit_profile_id", "particle_count"])[
                "total_nll"
            ]
            .std(ddof=1)
            .dropna()
            .rename("candidate_total_nll_sd")
            .reset_index()
        )
        _atomic_csv(output / "particle_count_summary.csv", particle_summary)
        _atomic_csv(output / "seed_rank_correlations.csv", correlations)
        _atomic_csv(output / "cross_count_correlations.csv", cross)
        _atomic_csv(output / "winner_stability.csv", winners)
        _atomic_csv(output / "candidate_nll_sd.csv", candidate_sd)
        _atomic_json(output / "ranking_summary.json", ranking_summary)

        decomposition_scalar = pd.read_csv(
            output / "filter_decomposition_scores.csv"
        )
        subject_mode, contrasts, decomposition_summary = summarize_decomposition(
            decomposition_scalar
        )
        _atomic_csv(output / "filter_decomposition_by_subject.csv", subject_mode)
        _atomic_csv(output / "filter_decomposition_contrasts.csv", contrasts)
        _atomic_json(
            output / "filter_decomposition_summary.json", decomposition_summary
        )
        _write_figure_contract(output, ranking_summary)
        _write_figure(
            output,
            particle_summary,
            correlations,
            candidate_sd,
            subject_mode,
            decomposition_scalar,
            ranking_config["stop_thresholds"],
            str(config["report"]["figure_png"]),
        )
        _write_readme(
            output, particle_summary, ranking_summary, decomposition_summary
        )
        manifest.update(
            {
                "status": "complete_with_numerical_gate_result",
                "particle_counts_evaluated": ranking_summary[
                    "particle_counts_evaluated"
                ],
                "minimum_stable_particle_count": ranking_summary[
                    "minimum_stable_particle_count"
                ],
                "ranking_score_row_count": int(len(ranking_scores)),
                "decomposition_score_row_count": int(len(decomposition_scalar)),
                "ranking_summary_sha256": _sha256(
                    output / "ranking_summary.json"
                ),
                "filter_decomposition_summary_sha256": _sha256(
                    output / "filter_decomposition_summary.json"
                ),
            }
        )
        _atomic_json(output / "analysis_manifest.json", manifest)
        print(f"[phase1] outputs={output}", flush=True)
        print(particle_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
