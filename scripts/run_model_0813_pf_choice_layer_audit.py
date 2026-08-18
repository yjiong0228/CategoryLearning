#!/usr/bin/env python3
"""Audit model-0813 choice-layer mechanisms on shared PF states.

Each subject/seed pair runs the fitted particle filter once with the existing
choice-transmission audit enabled.  Alternative readouts are then scored on
the identical pre-choice particles and weights.  This estimates an
instantaneous conditional readout contribution; it is deliberately not a
full counterfactual fit in which the alternative readout changes future
particle weights and cognitive states.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_model_0813_pf_calibration import (  # noqa: E402
    _atomic_csv,
    _atomic_json,
    _git_head,
    _python_tree_sha256,
    _relative,
    _repo_path,
    _sha256,
    _worktree_dirty,
)
from scripts.run_model_0813_pf_parameter_recovery import (  # noqa: E402
    _load_subject_frames,
    _readout_args,
    _subject_engine,
)
from src.Bayesian_state.inference.backends.particle_filter import (  # noqa: E402
    run_state_model_particle_filter,
)
from src.Bayesian_state.simulation.config import load_yaml  # noqa: E402
from src.Bayesian_state.utils.seeding import stable_seed  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/specific_models/model_0813_pf_choice_layer_audit.yaml"
FEATURE_COLUMNS = ("feature1", "feature2", "feature3", "feature4")
LAYER_KEYS = {
    "active_unsharpened": "audit_unsharpened_expectation",
    "active_sharpened": "audit_sharpened_no_lapse",
    "active_strategy_confidence": "audit_strategy_confidence_no_lapse",
    "executed_no_strategy": (
        "audit_persistent_execution_no_strategy_no_lapse"
    ),
    "executed_strategy": "audit_persistent_execution_no_lapse",
    "executed_counterfactual_strategy": (
        "audit_persistent_execution_counterfactual_strategy_no_lapse"
    ),
    "fitted_with_lapse": "prior_t",
}


@dataclass(frozen=True)
class RuntimeDesign:
    subjects: tuple[int, ...]
    trials_per_subject: int
    particle_count: int
    seed_indices: tuple[int, ...]
    training_seed_indices: tuple[int, ...]
    validation_seed_indices: tuple[int, ...]
    n_jobs: int
    bootstrap_repeats: int


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
        help="Use one subject, 16 trials, four particles and four seeds.",
    )
    return parser.parse_args()


def _resolved_paths(
    config_path: Path,
    config: Mapping[str, Any],
    output_override: Path | None,
) -> tuple[Path, Path, Path]:
    del config_path
    base_path = _repo_path(str(config["base_simulation_config"]))
    phase0_path = _repo_path(str(config["phase0_probe_summary"]))
    output = (
        output_override.resolve()
        if output_override is not None
        else _repo_path(str(config["output_dir"]))
    )
    return base_path, phase0_path, output


def _runtime_design(
    design: Mapping[str, Any], *, smoke: bool, n_jobs: int | None
) -> RuntimeDesign:
    subjects = tuple(int(value) for value in design["subjects"])
    repeat_n = int(design["total_filter_seed_repeats"])
    seeds = tuple(range(repeat_n))
    training = tuple(int(value) for value in design["training_seed_indices"])
    validation = tuple(int(value) for value in design["validation_seed_indices"])
    if smoke:
        return RuntimeDesign(
            subjects=subjects[:1],
            trials_per_subject=min(16, int(design["trials_per_subject"])),
            particle_count=4,
            seed_indices=(0, 1, 2, 3),
            training_seed_indices=(0, 1),
            validation_seed_indices=(2, 3),
            n_jobs=1,
            bootstrap_repeats=min(200, int(design["bootstrap_repeats"])),
        )
    if set(training) & set(validation):
        raise ValueError("training and validation PF seed panels must be disjoint")
    if set(training) | set(validation) != set(seeds):
        raise ValueError("training and validation panels must cover all PF repeats")
    return RuntimeDesign(
        subjects=subjects,
        trials_per_subject=int(design["trials_per_subject"]),
        particle_count=int(design["particle_count"]),
        seed_indices=seeds,
        training_seed_indices=training,
        validation_seed_indices=validation,
        n_jobs=int(n_jobs if n_jobs is not None else design["n_jobs"]),
        bootstrap_repeats=int(design["bootstrap_repeats"]),
    )


def _cache_paths(output: Path, subject_id: int, repeat: int) -> tuple[Path, Path]:
    stem = f"subject_{int(subject_id)}_seed_{int(repeat):02d}"
    folder = output / "cache" / f"subject_{int(subject_id)}"
    return folder / f"{stem}.json", folder / f"{stem}.npz"


def _filter_seed(
    base_seed: int, subject_id: int, repeat: int, *, seed_role: str
) -> int:
    return stable_seed(
        {
            "seed_role": str(seed_role),
            "base_seed": int(base_seed),
            "subject_id": int(subject_id),
            "filter_repeat": int(repeat),
        }
    )


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _validate_probabilities(
    values: Any, *, layer_id: str, trial_count: int
) -> np.ndarray:
    probability = np.asarray(values, dtype=float)
    if probability.shape != (int(trial_count), 2):
        raise ValueError(
            f"{layer_id} has shape {probability.shape}, expected {(trial_count, 2)}"
        )
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError(f"{layer_id} contains invalid probabilities")
    if not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError(f"{layer_id} probabilities are not normalized")
    return probability


def _layer_metrics(probability: np.ndarray, choices: np.ndarray) -> dict[str, float]:
    indices = np.asarray(choices, dtype=int) - 1
    selected = probability[np.arange(indices.size), indices]
    clipped = np.clip(selected, 1e-12, 1.0)
    one_hot = np.zeros_like(probability)
    one_hot[np.arange(indices.size), indices] = 1.0
    total_nll = float(-np.log(clipped).sum())
    return {
        "total_nll": total_nll,
        "mean_nll": float(total_nll / indices.size),
        "mean_brier": float(np.mean(np.sum((probability - one_hot) ** 2, axis=1))),
        "minimum_selected_probability": float(np.min(selected)),
        "maximum_normalization_error": float(
            np.max(np.abs(probability.sum(axis=1) - 1.0))
        ),
    }


def _score_subject_seed(
    *,
    output: Path,
    subject_id: int,
    frame: pd.DataFrame,
    base_config: Mapping[str, Any],
    base_path: Path,
    dataset_paths: Mapping[str, Path],
    comparisons: Sequence[Mapping[str, Any]],
    particle_count: int,
    resample_threshold_fraction: float,
    filter_repeat: int,
    base_seed: int,
    filter_seed_role: str,
    counterfactual_strategy_confidence_gain: float | None,
    force: bool,
) -> dict[str, Any]:
    json_path, npz_path = _cache_paths(output, subject_id, filter_repeat)
    if json_path.exists() and npz_path.exists() and not force:
        return dict(json.loads(json_path.read_text(encoding="utf-8")))

    engine = _subject_engine(base_config, base_path, int(subject_id))
    readout_args = _readout_args(engine)
    choices = frame["choice"].to_numpy(dtype=int)
    filter_seed = _filter_seed(
        base_seed,
        subject_id,
        filter_repeat,
        seed_role=filter_seed_role,
    )
    started = time.perf_counter()
    result = run_state_model_particle_filter(
        engine_config=engine,
        subject_id=int(subject_id),
        stimulus=frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
        choices=choices,
        feedback=frame["feedback"].to_numpy(dtype=float),
        particle_count=int(particle_count),
        filter_seed=int(filter_seed),
        resample_threshold_fraction=float(resample_threshold_fraction),
        choice_transmission_audit=True,
        choice_transmission_counterfactual_gain=(
            counterfactual_strategy_confidence_gain
        ),
        processed_data_dir=dataset_paths["processed_dir"],
        dataset_paths=dataset_paths,
        **readout_args,
    )
    requested_layer_ids = sorted(
        {
            str(comparison[layer_key])
            for comparison in comparisons
            for layer_key in ("comparator_layer", "mechanism_layer")
        }
    )
    unknown_layers = set(requested_layer_ids) - set(LAYER_KEYS)
    if unknown_layers:
        raise ValueError(f"unknown audit layers: {sorted(unknown_layers)}")
    layer_probabilities = {
        layer_id: _validate_probabilities(
            result.observation_probabilities[source_key],
            layer_id=layer_id,
            trial_count=len(frame),
        )
        for layer_id, source_key in LAYER_KEYS.items()
        if layer_id in requested_layer_ids
    }
    layer_scores = {
        layer_id: _layer_metrics(probability, choices)
        for layer_id, probability in layer_probabilities.items()
    }
    contrast_scores: dict[str, dict[str, Any]] = {}
    for comparison in comparisons:
        contrast_id = str(comparison["contrast_id"])
        comparator_layer = str(comparison["comparator_layer"])
        mechanism_layer = str(comparison["mechanism_layer"])
        comparator = layer_scores[comparator_layer]
        mechanism = layer_scores[mechanism_layer]
        contrast_scores[contrast_id] = {
            "mechanism_id": str(comparison["mechanism_id"]),
            "interpretation_role": str(comparison["interpretation_role"]),
            "comparator_layer": comparator_layer,
            "mechanism_layer": mechanism_layer,
            "comparator_total_nll": float(comparator["total_nll"]),
            "mechanism_total_nll": float(mechanism["total_nll"]),
            "delta_total_nll": float(
                comparator["total_nll"] - mechanism["total_nll"]
            ),
            "delta_mean_nll": float(
                comparator["mean_nll"] - mechanism["mean_nll"]
            ),
            "delta_mean_brier": float(
                comparator["mean_brier"] - mechanism["mean_brier"]
            ),
        }
    payload = {
        "analysis_role": "phase1c_common_state_choice_layer_audit",
        "subject_id": int(subject_id),
        "filter_repeat": int(filter_repeat),
        "filter_seed": int(filter_seed),
        "particle_count": int(particle_count),
        "trial_count": int(len(frame)),
        "runtime_seconds": float(time.perf_counter() - started),
        "mean_pre_choice_ess_fraction": float(
            np.mean(np.asarray(result.pre_choice_ess, dtype=float)) / particle_count
        ),
        "mean_post_choice_ess_fraction": float(
            np.mean(np.asarray(result.post_choice_ess, dtype=float)) / particle_count
        ),
        "resampling_fraction": float(
            np.mean(np.asarray(result.resampled, dtype=bool))
        ),
        "layer_scores": layer_scores,
        "contrast_scores": contrast_scores,
        "interpretation_boundary": (
            "shared pre-choice baseline PF states; alternative readouts do not "
            "change subsequent filtering"
        ),
    }
    _atomic_npz(
        npz_path,
        {
            "choices": choices,
            "feedback": frame["feedback"].to_numpy(dtype=float),
            **{
                f"probability__{layer_id}": probability
                for layer_id, probability in layer_probabilities.items()
            },
        },
    )
    payload["probability_npz"] = _relative(npz_path)
    payload["probability_npz_sha256"] = _sha256(npz_path)
    _atomic_json(json_path, payload)
    return payload


def _flatten_payloads(
    payloads: Sequence[Mapping[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    layer_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    for payload in payloads:
        shared = {
            key: payload[key]
            for key in (
                "subject_id",
                "filter_repeat",
                "filter_seed",
                "particle_count",
                "trial_count",
                "runtime_seconds",
                "mean_pre_choice_ess_fraction",
                "mean_post_choice_ess_fraction",
                "resampling_fraction",
                "probability_npz",
                "probability_npz_sha256",
            )
        }
        run_rows.append(shared)
        for layer_id, metrics in payload["layer_scores"].items():
            layer_rows.append({**shared, "layer_id": layer_id, **metrics})
        for contrast_id, metrics in payload["contrast_scores"].items():
            contrast_rows.append({**shared, "contrast_id": contrast_id, **metrics})
    return pd.DataFrame(run_rows), pd.DataFrame(layer_rows), pd.DataFrame(contrast_rows)


def _require_complete_design(
    runs: pd.DataFrame, subjects: Sequence[int], seeds: Sequence[int]
) -> None:
    expected = {
        (int(subject), int(seed)) for subject in subjects for seed in seeds
    }
    observed = set(
        runs[["subject_id", "filter_repeat"]]
        .astype(int)
        .itertuples(index=False, name=None)
    )
    if observed != expected or len(runs) != len(expected):
        raise ValueError(
            "run table does not match the frozen subject-seed design: "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}"
        )


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    repeats: int,
    confidence: float,
    seed: int,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be non-empty and finite")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, array.size, size=(int(repeats), array.size))
    means = np.mean(array[indices], axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.size < 2 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(spearmanr(a, b).statistic)


def summarize_effects(
    contrast_scores: pd.DataFrame,
    comparisons: Sequence[Mapping[str, Any]],
    runtime: RuntimeDesign,
    design: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    confidence = float(design["bootstrap_confidence"])
    base_seed = int(design["base_seed"])
    gates = design["stability_gates"]
    practical_rule = design["practical_effect_rule"]
    training = set(runtime.training_seed_indices)
    validation = set(runtime.validation_seed_indices)
    subject_rows: list[dict[str, Any]] = []

    for comparison in comparisons:
        contrast_id = str(comparison["contrast_id"])
        frame = contrast_scores.loc[
            contrast_scores["contrast_id"].astype(str).eq(contrast_id)
        ].copy()
        for subject_id, subject_frame in frame.groupby("subject_id", sort=True):
            subject_frame = subject_frame.sort_values("filter_repeat")
            effects = subject_frame["delta_mean_nll"].to_numpy(dtype=float)
            train_effects = subject_frame.loc[
                subject_frame["filter_repeat"].astype(int).isin(training),
                "delta_mean_nll",
            ].to_numpy(dtype=float)
            validation_effects = subject_frame.loc[
                subject_frame["filter_repeat"].astype(int).isin(validation),
                "delta_mean_nll",
            ].to_numpy(dtype=float)
            low, high = _bootstrap_mean_interval(
                effects,
                repeats=runtime.bootstrap_repeats,
                confidence=confidence,
                seed=stable_seed(
                    {
                        "seed_role": "phase1c_subject_seed_bootstrap",
                        "base_seed": base_seed,
                        "contrast_id": contrast_id,
                        "subject_id": int(subject_id),
                    }
                ),
            )
            sd = float(np.std(effects, ddof=1))
            train_mean = float(np.mean(train_effects))
            validation_mean = float(np.mean(validation_effects))
            subject_rows.append(
                {
                    "contrast_id": contrast_id,
                    "mechanism_id": str(comparison["mechanism_id"]),
                    "interpretation_role": str(comparison["interpretation_role"]),
                    "comparator_layer": str(comparison["comparator_layer"]),
                    "mechanism_layer": str(comparison["mechanism_layer"]),
                    "subject_id": int(subject_id),
                    "seed_n": int(effects.size),
                    "training_mean_delta_nll": train_mean,
                    "validation_mean_delta_nll": validation_mean,
                    "split_sign_agreement": bool(
                        np.sign(train_mean) == np.sign(validation_mean)
                    ),
                    "full_mean_delta_nll": float(np.mean(effects)),
                    "full_median_delta_nll": float(np.median(effects)),
                    "paired_seed_delta_nll_sd": sd,
                    "paired_mean_delta_nll_mcse": float(sd / math.sqrt(effects.size)),
                    "seed_positive_fraction": float(np.mean(effects > 0.0)),
                    "seed_bootstrap_ci_low": low,
                    "seed_bootstrap_ci_high": high,
                    "mean_comparator_nll": float(
                        subject_frame["comparator_total_nll"].mean()
                        / subject_frame["trial_count"].iloc[0]
                    ),
                    "mean_mechanism_nll": float(
                        subject_frame["mechanism_total_nll"].mean()
                        / subject_frame["trial_count"].iloc[0]
                    ),
                }
            )

    subject_summary = pd.DataFrame(subject_rows)
    contrast_rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        contrast_id = str(comparison["contrast_id"])
        frame = subject_summary.loc[
            subject_summary["contrast_id"].astype(str).eq(contrast_id)
        ].copy()
        train_values = frame["training_mean_delta_nll"].to_numpy(dtype=float)
        validation_values = frame["validation_mean_delta_nll"].to_numpy(dtype=float)
        effects = frame["full_mean_delta_nll"].to_numpy(dtype=float)
        low, high = _bootstrap_mean_interval(
            effects,
            repeats=runtime.bootstrap_repeats,
            confidence=confidence,
            seed=stable_seed(
                {
                    "seed_role": "phase1c_subject_bootstrap",
                    "base_seed": base_seed,
                    "contrast_id": contrast_id,
                }
            ),
        )
        rho = _safe_spearman(train_values, validation_values)
        sign_agreement = float(frame["split_sign_agreement"].mean())
        training_mean = float(np.mean(train_values))
        validation_mean = float(np.mean(validation_values))
        aggregate_sign_agreement = bool(
            np.sign(training_mean) == np.sign(validation_mean)
        )
        median_mcse = float(frame["paired_mean_delta_nll_mcse"].median())
        median_seed_sd = float(frame["paired_seed_delta_nll_sd"].median())
        comparator_nll = float(frame["mean_comparator_nll"].mean())
        practical_threshold = float(
            max(
                float(practical_rule["baseline_mean_nll_fraction"]) * comparator_nll,
                float(practical_rule["paired_seed_sd_multiplier"]) * median_seed_sd,
            )
        )
        rank_pass = bool(
            np.isfinite(rho)
            and rho
            >= float(gates["minimum_train_validation_subject_spearman"])
        )
        sign_pass = bool(
            sign_agreement >= float(gates["minimum_subject_sign_agreement"])
        )
        aggregate_pass = bool(
            aggregate_sign_agreement
            if bool(gates["require_aggregate_sign_agreement"])
            else True
        )
        mcse_pass = bool(
            median_mcse
            <= float(gates["maximum_median_paired_mean_nll_mcse"])
        )
        numerically_stable = bool(
            rank_pass and sign_pass and aggregate_pass and mcse_pass
        )
        if not numerically_stable:
            triage = "unresolved_numerical"
        elif low > practical_threshold:
            triage = "advance_conditional_benefit"
        elif high < practical_threshold:
            triage = "deprioritize_no_practical_benefit"
        else:
            triage = "unresolved_effect"
        contrast_rows.append(
            {
                "contrast_id": contrast_id,
                "mechanism_id": str(comparison["mechanism_id"]),
                "interpretation_role": str(comparison["interpretation_role"]),
                "comparator_layer": str(comparison["comparator_layer"]),
                "mechanism_layer": str(comparison["mechanism_layer"]),
                "subject_n": int(len(frame)),
                "mean_subject_delta_nll": float(np.mean(effects)),
                "median_subject_delta_nll": float(np.median(effects)),
                "subject_positive_fraction": float(np.mean(effects > 0.0)),
                "subject_bootstrap_ci_low": low,
                "subject_bootstrap_ci_high": high,
                "training_mean_delta_nll": training_mean,
                "validation_mean_delta_nll": validation_mean,
                "train_validation_subject_spearman": rho,
                "subject_split_sign_agreement": sign_agreement,
                "aggregate_sign_agreement": aggregate_sign_agreement,
                "median_paired_seed_delta_nll_sd": median_seed_sd,
                "median_paired_mean_delta_nll_mcse": median_mcse,
                "mean_comparator_nll": comparator_nll,
                "practical_effect_threshold": practical_threshold,
                "rank_stability_pass": rank_pass,
                "sign_stability_pass": sign_pass,
                "aggregate_sign_pass": aggregate_pass,
                "mcse_pass": mcse_pass,
                "all_numerical_stability_gates_pass": numerically_stable,
                "conditional_triage": triage,
            }
        )
    contrast_summary = pd.DataFrame(contrast_rows)
    summary = {
        "status": (
            "all_choice_layer_contrasts_numerically_stable"
            if bool(contrast_summary["all_numerical_stability_gates_pass"].all())
            else "one_or_more_choice_layer_contrasts_numerically_unresolved"
        ),
        "subject_n": int(subject_summary["subject_id"].nunique()),
        "subjects": sorted(subject_summary["subject_id"].astype(int).unique().tolist()),
        "particle_count": int(runtime.particle_count),
        "trials_per_subject": int(runtime.trials_per_subject),
        "filter_seed_repeats": int(len(runtime.seed_indices)),
        "training_seed_indices": list(runtime.training_seed_indices),
        "validation_seed_indices": list(runtime.validation_seed_indices),
        "contrast_n": int(len(contrast_summary)),
        "numerically_stable_contrast_n": int(
            contrast_summary["all_numerical_stability_gates_pass"].sum()
        ),
        "conditional_triage_counts": {
            str(key): int(value)
            for key, value in contrast_summary["conditional_triage"]
            .value_counts()
            .items()
        },
        "current_obs01_exact_noop_under_persistent_execution": True,
        "positive_effect_definition": (
            "comparator mean NLL minus mechanism mean NLL on the same "
            "pre-choice baseline PF states"
        ),
        "interpretation_boundary": (
            "conditional instantaneous readout audit; not a full counterfactual "
            "fit and not a population mechanism-retention decision"
        ),
        "stability_gates": dict(gates),
        "practical_effect_rule": dict(practical_rule),
    }
    return subject_summary, contrast_summary, summary


def _write_figure_contract(output: Path) -> None:
    content = """# Figure contract and chart map

Core conclusion: Common-state paired rereading identifies which choice-layer effects are numerically repeatable enough to justify a full counterfactual mechanism comparison.

- Figure archetype: quantitative grid with a dominant paired-effect summary.
- Target output: technical mechanism-audit report; Python/matplotlib; 7.2 x 7.0 inches; PNG at 300 dpi.
- Backend: Python only.
- Export policy: PNG only under the repository artifact rule.
- Hero evidence: subject-mean paired delta NLL and subject-bootstrap interval for every frozen contrast.
- Validation evidence: disjoint 4-vs-4 seed panel agreement, subject-by-contrast heterogeneity, and paired Monte Carlo error.
- Statistics: eight fixed pilot subjects; eight PF seeds are technical repeats; intervals across subjects are descriptive pilot intervals.
- Source data: `contrast_seed_scores.csv`, `subject_contrast_summary.csv`, and `contrast_summary.csv`.
- Image integrity: all declared subjects, contrasts and completed seeds are retained; no display subsampling is used.

| Panel | Analytical question | Form | Evidence role | Non-color encoding |
|---|---|---|---|---|
| a | Which readout transformations improve or worsen mean choice NLL? | Horizontal dot-and-interval plot with zero and practical thresholds | Hero paired effect | filled circles and interval lines |
| b | Do disjoint seed panels recover the same subject-level effects? | Training-versus-validation scatter with identity line | Independent numerical validation | contrast-specific markers |
| c | Is the aggregate effect hiding strong subject heterogeneity? | Subject-by-contrast diverging heatmap | Heterogeneity | signed labels and shared zero center |
| d | Which contrasts pass the frozen stability gates? | Grouped stability bars plus gate lines | Decision audit | bars, hatch and threshold lines |

Reviewer risk: these alternative readouts are evaluated on baseline-filtered states. They isolate instantaneous readout consequences but do not allow the counterfactual readout to alter future particle weights, latent paths, or learning. A stable conditional effect therefore authorizes a full comparator; it does not by itself retain or remove a psychological mechanism.
"""
    (output / "chart_map.md").write_text(content, encoding="utf-8")


def _write_figure(
    output: Path,
    subject_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    comparisons: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
    filename: str,
) -> Path:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.2,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )
    order = [str(item["contrast_id"]) for item in comparisons]
    labels = {
        "active_weight_sharpening": "Active weighting power",
        "active_strategy_confidence": "Active strategy confidence",
        "persistent_execution": "Persistent execution",
        "strategy_confidence_under_execution": "Strategy confidence | execution",
        "uniform_output_lapse": "Uniform output lapse",
    }
    labels = {
        contrast_id: labels.get(contrast_id, contrast_id.replace("_", " "))
        for contrast_id in order
    }
    palette = {
        contrast_id: color
        for contrast_id, color in zip(
            order,
            ("#356DB5", "#7C90C7", "#D77B1F", "#A65E20", "#7A8733"),
        )
    }
    markers = dict(zip(order, ("o", "s", "^", "D", "P")))
    summary = contrast_summary.set_index("contrast_id").loc[order].reset_index()
    subjects = sorted(subject_summary["subject_id"].astype(int).unique())

    fig = plt.figure(figsize=(7.2, 7.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    ax_a, ax_b, ax_c, ax_d = axes

    y = np.arange(len(order))
    effects = summary["mean_subject_delta_nll"].to_numpy(dtype=float)
    low = summary["subject_bootstrap_ci_low"].to_numpy(dtype=float)
    high = summary["subject_bootstrap_ci_high"].to_numpy(dtype=float)
    for index, contrast_id in enumerate(order):
        ax_a.errorbar(
            effects[index],
            y[index],
            xerr=np.asarray([[effects[index] - low[index]], [high[index] - effects[index]]]),
            fmt=markers[contrast_id],
            ms=5.0,
            mfc=palette[contrast_id],
            mec=palette[contrast_id],
            ecolor=palette[contrast_id],
            capsize=2.0,
            lw=1.4,
        )
        threshold = float(summary.loc[index, "practical_effect_threshold"])
        ax_a.plot(threshold, y[index], marker="|", color="#252A34", ms=9, mew=1.2)
    ax_a.axvline(0.0, color="#8E96A3", lw=0.9)
    ax_a.set_yticks(y, [labels[value] for value in order])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Paired mean NLL gain (comparator - mechanism)")
    ax_a.set_title("Conditional readout effects")
    ax_a.grid(axis="x", color="#E3E6EA", lw=0.7)
    ax_a.text(
        0.99,
        0.02,
        "black ticks: practical thresholds",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
        color="#4E5663",
    )

    for contrast_id in order:
        frame = subject_summary.loc[
            subject_summary["contrast_id"].astype(str).eq(contrast_id)
        ]
        ax_b.scatter(
            frame["training_mean_delta_nll"],
            frame["validation_mean_delta_nll"],
            s=28,
            facecolors="none",
            edgecolors=palette[contrast_id],
            marker=markers[contrast_id],
            linewidths=1.1,
            label=labels[contrast_id],
        )
    limits = np.asarray(
        [
            subject_summary["training_mean_delta_nll"].min(),
            subject_summary["training_mean_delta_nll"].max(),
            subject_summary["validation_mean_delta_nll"].min(),
            subject_summary["validation_mean_delta_nll"].max(),
        ],
        dtype=float,
    )
    span = max(float(limits.max() - limits.min()), 1e-4)
    lower = float(limits.min() - 0.08 * span)
    upper = float(limits.max() + 0.08 * span)
    ax_b.plot([lower, upper], [lower, upper], ls="--", color="#252A34", lw=1.0)
    ax_b.axhline(0.0, color="#B1B7C0", lw=0.7)
    ax_b.axvline(0.0, color="#B1B7C0", lw=0.7)
    ax_b.set_xlim(lower, upper)
    ax_b.set_ylim(lower, upper)
    ax_b.set_xlabel("Training seeds 0-3: paired delta NLL")
    ax_b.set_ylabel("Validation seeds 4-7: paired delta NLL")
    ax_b.set_title("Independent seed-panel agreement")
    ax_b.legend(frameon=False, loc="best", handletextpad=0.4)

    heat = (
        subject_summary.pivot(
            index="subject_id", columns="contrast_id", values="full_mean_delta_nll"
        )
        .reindex(index=subjects, columns=order)
        .to_numpy(dtype=float)
    )
    bound = max(float(np.max(np.abs(heat))), 1e-6)
    cmap = LinearSegmentedColormap.from_list(
        "blue_white_orange", ("#356DB5", "#F7F7F5", "#D77B1F")
    )
    image = ax_c.imshow(
        heat,
        aspect="auto",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound),
    )
    ax_c.set_xticks(np.arange(len(order)), [labels[value] for value in order], rotation=35, ha="right")
    ax_c.set_yticks(np.arange(len(subjects)), [str(value) for value in subjects])
    ax_c.set_xlabel("Frozen choice-layer contrast")
    ax_c.set_ylabel("Pilot subject")
    ax_c.set_title("Subject-level paired effects")
    colorbar = fig.colorbar(image, ax=ax_c, fraction=0.046, pad=0.03)
    colorbar.set_label("Mean NLL gain")

    x = np.arange(len(order))
    width = 0.36
    rank_values = summary["train_validation_subject_spearman"].fillna(0.0).to_numpy(dtype=float)
    sign_values = summary["subject_split_sign_agreement"].to_numpy(dtype=float)
    ax_d.bar(
        x - width / 2,
        rank_values,
        width,
        color="#356DB5",
        edgecolor="#234B80",
        label="Subject-rank rho",
    )
    ax_d.bar(
        x + width / 2,
        sign_values,
        width,
        facecolor="white",
        edgecolor="#D77B1F",
        hatch="///",
        label="Subject sign agreement",
    )
    ax_d.axhline(
        float(gates["minimum_train_validation_subject_spearman"]),
        color="#234B80",
        ls="--",
        lw=0.9,
    )
    ax_d.axhline(
        float(gates["minimum_subject_sign_agreement"]),
        color="#A65E20",
        ls=":",
        lw=1.0,
    )
    ax_d.set_xticks(x, [labels[value] for value in order], rotation=35, ha="right")
    ax_d.set_ylim(-0.05, 1.05)
    ax_d.set_ylabel("Stability metric")
    ax_d.set_title("Frozen numerical-stability gates", pad=30)
    ax_d.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    ax_d.grid(axis="y", color="#E3E6EA", lw=0.7)

    for label, axis in zip("abcd", axes):
        axis.text(
            -0.16,
            1.05,
            label,
            transform=axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
    fig.suptitle(
        "Common-state choice-layer mechanism audit",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    path = output / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _phase0_obs01_exact_noop(phase0_path: Path) -> bool:
    frame = pd.read_csv(phase0_path)
    selected = frame.loc[
        frame["variant_id"].astype(str).isin(
            {"readout_power_one", "readout_expectation"}
        )
    ]
    if len(selected) != 2:
        raise ValueError("Phase-0 OBS-01 exact-noop evidence is incomplete")
    return bool(selected["all_subjects_exact"].astype(bool).all())


def _write_readme(
    output: Path,
    summary: Mapping[str, Any],
    contrast_summary: pd.DataFrame,
    *,
    title: str = "Phase 1c / 2：common-state choice-layer 配对审计",
) -> None:
    lines = [
        f"# {title}",
        "",
        "## 技术结论",
        "",
        (
            "本分析在同一批 pre-choice PF particles 和 weights 上重读多个 choice layers，"
            "用来判断纯读出效应是否值得进入成本更高的完整反事实机制比较。"
        ),
        "",
        f"- 固定 pilot subjects：`{summary['subject_n']}`；每人 trials：`{summary['trials_per_subject']}`。",
        f"- PF：`{summary['particle_count']}` particles × `{summary['filter_seed_repeats']}` technical seeds。",
        f"- 通过全部数值稳定门槛的 contrasts：`{summary['numerically_stable_contrast_n']}/{summary['contrast_n']}`。",
        "- OBS-01 在当前 persistent execution 路径下仍沿用 Phase 0 的 exact no-op 结论。",
        "",
        "| contrast | mechanism | mean delta NLL | 95% pilot interval | split rho | sign agreement | practical threshold | stable | triage |",
        "|---|---|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for row in contrast_summary.to_dict(orient="records"):
        rho = row["train_validation_subject_spearman"]
        rho_text = "NA" if not np.isfinite(rho) else f"{rho:.3f}"
        lines.append(
            "| {contrast_id} | {mechanism_id} | {mean_subject_delta_nll:.5f} | "
            "[{subject_bootstrap_ci_low:.5f}, {subject_bootstrap_ci_high:.5f}] | "
            "{rho} | {subject_split_sign_agreement:.3f} | {practical_effect_threshold:.5f} | "
            "{stable} | {conditional_triage} |".format(
                **row,
                rho=rho_text,
                stable=(
                    "yes" if row["all_numerical_stability_gates_pass"] else "no"
                ),
            )
        )
    lines.extend(
        [
            "",
            "## 指标解释",
            "",
            "`delta mean NLL = comparator NLL - mechanism NLL`，正值表示 mechanism-side readout 对观察选择的即时预测更好。训练 seeds 0–3 与验证 seeds 4–7 完全不重叠。practical threshold 沿用冻结规则：`max(1% × comparator mean NLL, 2 × paired-seed effect SD)`。",
            "",
            "## 解释边界",
            "",
            "这些替代读出没有反过来改变后续 particle weights、resampling 或 cognitive states，因此它们是 baseline filtering 条件下的即时分解，不是完整反事实模型 likelihood。稳定且有实际量级的效应只表示该机制值得进入下一轮 full comparator；不能在本阶段直接作最终保留或删除决定。",
            "",
            "## 文件",
            "",
            "- `run_summary.csv`：每个 subject–seed baseline PF 运行与 ESS 诊断。",
            "- `layer_scores.csv`：本次共享状态 readout 的逐 subject–seed NLL/Brier。",
            "- `contrast_seed_scores.csv`：预声明配对的逐 seed delta NLL。",
            "- `subject_contrast_summary.csv`、`contrast_summary.csv`：独立 seed 面板、MCSE、区间和条件 triage。",
            "- `choice_layer_paired_audit.png`、`chart_map.md`：结果图和图形契约。",
            "- `summary.json`、`analysis_manifest.json`、`analysis_config_snapshot.json`：机器可读结论与 provenance。",
            "- `VALIDATION.md`：独立复算、缓存完整性、测试和图形 QA。",
            "",
        ]
    )
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config = load_yaml(config_path)
    base_path, phase0_path, output = _resolved_paths(
        config_path, config, args.output_dir
    )
    if args.smoke and args.output_dir is None:
        output = output / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    design = config["design"]
    runtime = _runtime_design(design, smoke=bool(args.smoke), n_jobs=args.n_jobs)
    comparisons = tuple(dict(value) for value in design["comparisons"])
    if len({item["contrast_id"] for item in comparisons}) != len(comparisons):
        raise ValueError("contrast ids must be unique")
    if not _phase0_obs01_exact_noop(phase0_path):
        raise ValueError("Phase-0 OBS-01 exact-noop prerequisite did not pass")
    counterfactual_gain = design.get(
        "counterfactual_strategy_confidence_gain"
    )
    if counterfactual_gain is not None:
        counterfactual_gain = float(counterfactual_gain)
    filter_seed_role = str(
        design.get(
            "filter_seed_role",
            "model0813_phase1c_choice_layer_audit",
        )
    )
    requested_layer_ids = {
        str(comparison[layer_key])
        for comparison in comparisons
        for layer_key in ("comparator_layer", "mechanism_layer")
    }

    base_config = load_yaml(base_path)
    subject_frames, dataset_paths = _load_subject_frames(
        base_config, base_path, runtime.subjects
    )
    for subject_id in runtime.subjects:
        subject_frames[subject_id] = subject_frames[subject_id].iloc[
            : runtime.trials_per_subject
        ].copy()
        if len(subject_frames[subject_id]) != runtime.trials_per_subject:
            raise ValueError(f"subject {subject_id} has too few trials")

    payloads: list[dict[str, Any]] = []
    if args.phase in {"run", "all"}:
        tasks = [
            delayed(_score_subject_seed)(
                output=output,
                subject_id=subject_id,
                frame=subject_frames[subject_id],
                base_config=base_config,
                base_path=base_path,
                dataset_paths=dataset_paths,
                comparisons=comparisons,
                particle_count=runtime.particle_count,
                resample_threshold_fraction=float(
                    design["resample_threshold_fraction"]
                ),
                filter_repeat=repeat,
                base_seed=int(design["base_seed"]),
                filter_seed_role=filter_seed_role,
                counterfactual_strategy_confidence_gain=counterfactual_gain,
                force=bool(args.force),
            )
            for subject_id in runtime.subjects
            for repeat in runtime.seed_indices
        ]
        payloads = Parallel(n_jobs=runtime.n_jobs, verbose=10)(tasks)
        runs, layers, contrasts = _flatten_payloads(payloads)
        _require_complete_design(runs, runtime.subjects, runtime.seed_indices)
        _atomic_csv(output / "run_summary.csv", runs)
        _atomic_csv(output / "layer_scores.csv", layers)
        _atomic_csv(output / "contrast_seed_scores.csv", contrasts)

    if args.phase in {"summarize", "all"}:
        runs = pd.read_csv(output / "run_summary.csv")
        layers = pd.read_csv(output / "layer_scores.csv")
        contrasts = pd.read_csv(output / "contrast_seed_scores.csv")
        _require_complete_design(runs, runtime.subjects, runtime.seed_indices)
        expected_layer_rows = len(runs) * len(requested_layer_ids)
        expected_contrast_rows = len(runs) * len(comparisons)
        if len(layers) != expected_layer_rows or len(contrasts) != expected_contrast_rows:
            raise ValueError("layer or contrast score table is incomplete")
        subject_summary, contrast_summary, summary = summarize_effects(
            contrasts, comparisons, runtime, design
        )
        summary["counterfactual_strategy_confidence_gain"] = (
            counterfactual_gain
        )
        _atomic_csv(output / "subject_contrast_summary.csv", subject_summary)
        _atomic_csv(output / "contrast_summary.csv", contrast_summary)
        _atomic_json(output / "summary.json", summary)
        _atomic_json(output / "analysis_config_snapshot.json", config)
        _write_figure_contract(output)
        figure_path = _write_figure(
            output,
            subject_summary,
            contrast_summary,
            comparisons,
            design["stability_gates"],
            str(config["report"]["figure_png"]),
        )
        _write_readme(
            output,
            summary,
            contrast_summary,
            title=str(
                config.get("report", {}).get(
                    "readme_title",
                    "Phase 1c / 2：common-state choice-layer 配对审计",
                )
            ),
        )
        manifest = {
            "analysis_id": str(config["analysis_id"]),
            "scope": str(config["scope"]),
            "status": "complete_with_conditional_choice_layer_triage",
            "config": _relative(config_path),
            "config_sha256": _sha256(config_path),
            "base_simulation_config": _relative(base_path),
            "base_simulation_config_sha256": _sha256(base_path),
            "phase0_probe_summary": _relative(phase0_path),
            "phase0_probe_summary_sha256": _sha256(phase0_path),
            "runner": _relative(Path(__file__)),
            "runner_sha256": _sha256(Path(__file__)),
            "bayesian_state_python_tree_sha256": _python_tree_sha256(
                ROOT / "src/Bayesian_state"
            ),
            "repository_head": _git_head(),
            "worktree_dirty": _worktree_dirty(),
            "smoke": bool(args.smoke),
            "design": {
                "subjects": list(runtime.subjects),
                "trials_per_subject": runtime.trials_per_subject,
                "particle_count": runtime.particle_count,
                "seed_indices": list(runtime.seed_indices),
                "training_seed_indices": list(runtime.training_seed_indices),
                "validation_seed_indices": list(runtime.validation_seed_indices),
                "comparisons": list(comparisons),
                "stability_gates": dict(design["stability_gates"]),
                "practical_effect_rule": dict(design["practical_effect_rule"]),
                "counterfactual_strategy_confidence_gain": (
                    counterfactual_gain
                ),
            },
            "run_row_count": int(len(runs)),
            "layer_score_row_count": int(len(layers)),
            "contrast_score_row_count": int(len(contrasts)),
            "summary_sha256": _sha256(output / "summary.json"),
            "contrast_summary_sha256": _sha256(output / "contrast_summary.csv"),
            "figure": _relative(figure_path),
            "figure_sha256": _sha256(figure_path),
            "interpretation_boundary": summary["interpretation_boundary"],
        }
        _atomic_json(output / "analysis_manifest.json", manifest)
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_analysis(parse_args())
