#!/usr/bin/env python3
"""Frozen rule-belief validation against structured oral report centers.

An oral center is mapped to the set of hypotheses that assign the participant's
just-made choice to that point.  The R0KT pre-feedback belief is then
choice-conditioned exactly as specified in model_newplan.tex.  No oral outcome
changes the cognitive state or its parameters.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Bayesian_state.utils.unified_newplan import (  # noqa: E402
    FEATURE_COLUMNS,
    ORDER_COLUMNS,
    build_partition,
    partition_prior,
    rule_predictions,
)


DEFAULT_DATA = ROOT / "data/processed/Task2_processed.csv"
DEFAULT_CORE = ROOT / "results/zhuran/unified_newplan/core_sobol512_20260802"
DEFAULT_OUTPUT = ROOT / "results/zhuran/unified_newplan/oral_external_validation_20260802"
MASS_FLOOR = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260802)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def parse_center(value: Any, n_dimensions: int = 4) -> np.ndarray | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        parsed = ast.literal_eval(value) if isinstance(value, str) else value
        center = np.asarray(parsed, dtype=float).reshape(-1)
    except (ValueError, SyntaxError, TypeError):
        return None
    if center.size != n_dimensions or not np.all(np.isfinite(center)):
        return None
    return center


def hypothesis_assignments(partition: Any, points: np.ndarray) -> np.ndarray:
    assignments = np.empty((len(points), int(partition.length)), dtype=np.int16)
    for hypothesis in range(int(partition.length)):
        assignments[:, hypothesis] = partition._get_category_assignments_region(
            hypothesis, points
        )
    return assignments


def fit_previous_smoothing(
    compatible: np.ndarray, encoded: np.ndarray, train: np.ndarray
) -> tuple[float, np.ndarray]:
    n_trials, n_hypotheses = compatible.shape
    overlap = np.zeros(n_trials, dtype=float)
    uniform_mass = compatible.mean(axis=1)
    for trial in range(1, n_trials):
        if encoded[trial - 1]:
            previous = compatible[trial - 1]
            overlap[trial] = (
                np.sum(previous & compatible[trial]) / max(1, int(previous.sum()))
            )
        else:
            overlap[trial] = uniform_mass[trial]
    overlap[0] = uniform_mass[0]
    valid = train & encoded

    def objective(epsilon: float) -> float:
        mass = (1.0 - epsilon) * overlap[valid] + epsilon * uniform_mass[valid]
        return float(-np.log(np.clip(mass, MASS_FLOOR, 1.0)).sum())

    fit = minimize_scalar(objective, bounds=(0.0, 1.0), method="bounded")
    epsilon = float(fit.x)
    mass = (1.0 - epsilon) * overlap + epsilon * uniform_mass
    return epsilon, mass


def fit_frequency_smoothing(
    compatible: np.ndarray, encoded: np.ndarray, train: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    n_hypotheses = compatible.shape[1]
    train_rows = np.flatnonzero(train & encoded)
    contributions = np.zeros_like(compatible, dtype=float)
    sizes = compatible.sum(axis=1)
    valid_rows = np.flatnonzero(encoded)
    contributions[valid_rows] = compatible[valid_rows] / sizes[valid_rows, None]
    counts = contributions[train_rows].sum(axis=0)
    train_compatible = compatible[train_rows]
    count_mass = train_compatible @ counts
    train_set_sizes = train_compatible.sum(axis=1)
    leave_one_out_denominator = max(len(train_rows) - 1, 0)

    def objective(log_alpha: float) -> float:
        alpha = float(np.exp(log_alpha))
        mass = (
            count_mass - 1.0 + alpha * train_set_sizes / n_hypotheses
        ) / (leave_one_out_denominator + alpha)
        return float(-np.log(np.clip(mass, MASS_FLOOR, 1.0)).sum())

    fit = minimize_scalar(
        objective, bounds=(math.log(1e-3), math.log(100.0)), method="bounded"
    )
    alpha = float(np.exp(fit.x))
    distribution = (counts + alpha / n_hypotheses) / (counts.sum() + alpha)
    mass = compatible @ distribution
    # Use leave-one-out frequency mass on training rows for honest baseline
    # selection; held-out rows use the distribution frozen from all training
    # reports.
    mass[train_rows] = (
        count_mass - 1.0 + alpha * train_set_sizes / n_hypotheses
    ) / (leave_one_out_denominator + alpha)
    return alpha, distribution, mass


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def summarize(metrics: pd.DataFrame, seed: int) -> pd.DataFrame:
    specifications = [
        ("RULE_ORAL", "BASE_SELECT", "rule_vs_selected_baseline"),
        ("RULE_ORAL", "PREVIOUS_REPORT", "rule_vs_previous_report"),
        ("RULE_ORAL", "STIMULUS_CHOICE", "rule_vs_stimulus_choice"),
        ("RULE_ORAL", "INDIVIDUAL_FREQUENCY", "rule_vs_individual_frequency"),
    ]
    holdout = metrics[metrics["segment"] == "holdout"]
    rows = []
    for candidate, reference, label in specifications:
        for condition_label, group in [
            (str(condition), holdout[holdout["condition"] == condition])
            for condition in (1, 2, 3)
        ] + [("all", holdout)]:
            paired = group[group["model"] == candidate].merge(
                group[group["model"] == reference],
                on=["subject_id", "condition", "segment"],
                suffixes=("_candidate", "_reference"),
                validate="one_to_one",
            )
            paired = paired.dropna(
                subset=["mean_log_score_candidate", "mean_log_score_reference"]
            )
            # Positive values mean lower compatible-set log score for the rule model.
            delta = (
                paired["mean_log_score_reference"].to_numpy()
                - paired["mean_log_score_candidate"].to_numpy()
            )
            low, high = bootstrap_interval(
                delta, seed + sum(ord(character) for character in label + condition_label)
            )
            rows.append(
                {
                    "comparison": label,
                    "candidate": candidate,
                    "reference": reference,
                    "condition": condition_label,
                    "n_subjects": int(len(delta)),
                    "mean_delta_log_score": float(delta.mean()),
                    "median_delta_log_score": float(np.median(delta)),
                    "bootstrap_mean_ci_low": low,
                    "bootstrap_mean_ci_high": high,
                    "n_improved": int((delta > 0).sum()),
                    "proportion_improved": float((delta > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def floor_sensitivity(trials: pd.DataFrame, seed: int) -> pd.DataFrame:
    holdout = trials[trials["segment"] == "holdout"]
    rows = []
    for floor in (1e-6, 1e-9, 1e-12):
        subject = (
            holdout.groupby(["subject_id", "condition"], as_index=False)
            .agg(
                rule_mean_log_score=(
                    "rule_mass",
                    lambda values: float(-np.log(np.clip(values, floor, 1.0)).mean()),
                ),
                baseline_mean_log_score=(
                    "selected_baseline_mass",
                    lambda values: float(-np.log(np.clip(values, floor, 1.0)).mean()),
                ),
            )
        )
        subject["delta"] = (
            subject["baseline_mean_log_score"] - subject["rule_mean_log_score"]
        )
        for condition_label, group in [
            (str(condition), subject[subject["condition"] == condition])
            for condition in (1, 2, 3)
        ] + [("all", subject)]:
            values = group["delta"].to_numpy(dtype=float)
            low, high = bootstrap_interval(
                values,
                seed
                + int(round(-math.log10(floor))) * 1000
                + sum(ord(character) for character in condition_label),
            )
            rows.append(
                {
                    "mass_floor": floor,
                    "condition": condition_label,
                    "n_subjects": int(len(values)),
                    "mean_delta_log_score": float(values.mean()),
                    "median_delta_log_score": float(np.median(values)),
                    "bootstrap_mean_ci_low": low,
                    "bootstrap_mean_ci_high": high,
                    "n_improved": int((values > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def render_report(
    output: Path,
    comparisons: pd.DataFrame,
    metrics: pd.DataFrame,
    parameters: pd.DataFrame,
    coverage: pd.DataFrame,
    floor_results: pd.DataFrame,
) -> None:
    primary = comparisons[comparisons["comparison"] == "rule_vs_selected_baseline"]
    all_row = primary[primary["condition"] == "all"].iloc[0]
    selections = parameters["selected_baseline"].value_counts().to_dict()
    lines = [
        "# Frozen rule-belief oral-report validation",
        "",
        "> R0KT choice parameters and pre-feedback beliefs were frozen. Oral reports never update or re-fit the cognitive model.",
        "",
        "## Primary result",
        "",
        f"Against each subject's training-selected oral baseline, R0KT improved held-out compatible-set log score by {all_row.mean_delta_log_score:.6f} (95% subject-bootstrap CI [{all_row.bootstrap_mean_ci_low:.6f}, {all_row.bootstrap_mean_ci_high:.6f}]; improved {int(all_row.n_improved)}/{int(all_row.n_subjects)} subjects).",
        "",
        "Positive Δ means the rule belief assigned more probability mass to the observed report-compatible hypothesis set.",
        "",
        "| Condition | Mean Δ log score | 95% CI | Improved |",
        "|:--|--:|:--|:--|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.condition} | {row.mean_delta_log_score:.6f} | "
            f"[{row.bootstrap_mean_ci_low:.6f}, {row.bootstrap_mean_ci_high:.6f}] | "
            f"{int(row.n_improved)}/{int(row.n_subjects)} |"
        )
    lines.extend(
        [
            "",
            "## Encoding and baselines",
            "",
            "- Frozen mapping: an oral center is compatible with hypothesis h when h assigns the participant's just-made category choice to that center. Unspecified features remain at the encoded center value 0.5.",
            f"- Encodable reports: {int(coverage.encoded_n.sum())}/{int(coverage.total_n.sum())}; held-out {int(coverage.holdout_encoded_n.sum())}/{int(coverage.holdout_total_n.sum())}.",
            f"- Mean compatible-set size across encoded trials: {coverage.mean_set_size.mean():.2f} hypotheses; set size is reported because broad reports mechanically receive more mass.",
            "- Baselines are normalized on the identical rule library: previous compatible set, current physical stimulus plus choice, and training-segment individual report frequency.",
            f"- Training-selected baselines across subjects: {json.dumps(selections, ensure_ascii=False, sort_keys=True)}.",
            "- Previous-report uniform-mixture ε and frequency Dirichlet mass α were estimated only on training reports; the latter used leave-one-out training scores for baseline selection.",
            "",
            "## Probability-floor sensitivity",
            "",
            "The primary floor is 1e-12. These rows show whether a small number of nearly incompatible reports determines the subject-level mean.",
            "",
            "| Floor | Condition | N | Mean Δ | 95% CI | Median Δ | Improved |",
            "|--:|:--|--:|--:|:--|--:|:--|",
        ]
    )
    for row in floor_results.itertuples(index=False):
        lines.append(
            f"| {row.mass_floor:.0e} | {row.condition} | {int(row.n_subjects)} | "
            f"{row.mean_delta_log_score:.6f} | [{row.bootstrap_mean_ci_low:.6f}, "
            f"{row.bootstrap_mean_ci_high:.6f}] | {row.median_delta_log_score:.6f} | "
            f"{int(row.n_improved)}/{int(row.n_subjects)} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "This primary mapping uses the structured center encoding because it is available for virtually all trials and yields an explicit compatible set. Region-overlap and free-text similarity are diagnostics, not silently substituted likelihoods. Support here would validate an external consequence of the rule state, but would not prove that every report is generated by one uniquely true latent rule.",
            "",
            "## Artifacts",
            "",
            "- `subject_oral_metrics.csv`, `model_comparisons.csv`, `baseline_parameters.csv`.",
            "- `trial_oral_scores.csv`, `encoding_coverage.csv`, `manifest.json`.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    data_path = args.data.resolve()
    core = args.core.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(data_path, low_memory=False).sort_values(
        ["condition", "iSub", *ORDER_COLUMNS], kind="stable"
    )
    metric_rows = []
    trial_rows = []
    parameter_rows = []
    coverage_rows = []
    for subject_id, frame in data.groupby("iSub", sort=True):
        subject_id = int(subject_id)
        condition = int(frame["condition"].iloc[0])
        frame = frame.reset_index(drop=True)
        choices = frame["choice"].to_numpy(dtype=np.int64) - 1
        feedback = frame["feedback"].to_numpy(dtype=float)
        partition = build_partition(condition)
        n_hypotheses = int(partition.length)
        with np.load(core / "q_cache" / f"subject_{subject_id}.npz", allow_pickle=False) as archive:
            q_values = archive["q"].astype(np.float64)
        with np.load(core / "subject_predictions" / f"subject_{subject_id}.npz", allow_pickle=False) as archive:
            holdout = archive["holdout_mask"].astype(bool)
        train = ~holdout
        result = rule_predictions(
            q_values,
            choices,
            feedback,
            condition,
            retention=1.0,
            sensitivity=1.0,
            prior=partition_prior(partition, "uniform_rule"),
            return_beliefs=True,
        )
        beliefs = result.beliefs.astype(float)
        choice_q = np.take_along_axis(
            q_values, choices[:, None, None], axis=2
        )[:, :, 0]
        oral_weight = beliefs * choice_q
        oral_weight /= np.maximum(oral_weight.sum(axis=1, keepdims=True), MASS_FLOOR)

        centers: list[np.ndarray | None] = [parse_center(value) for value in frame["oral_center"]]
        encoded = np.asarray([center is not None for center in centers], dtype=bool)
        compatible = np.zeros((len(frame), n_hypotheses), dtype=bool)
        if encoded.any():
            encoded_centers = np.vstack([centers[index] for index in np.flatnonzero(encoded)])
            unique_centers, inverse = np.unique(encoded_centers, axis=0, return_inverse=True)
            assignments = hypothesis_assignments(partition, unique_centers)
            encoded_rows = np.flatnonzero(encoded)
            compatible[encoded_rows] = assignments[inverse] == choices[encoded_rows, None]
        set_size = compatible.sum(axis=1)
        nonempty = set_size > 0
        encoded &= nonempty
        rule_mass = np.sum(oral_weight * compatible, axis=1)

        physical_assignments = hypothesis_assignments(
            partition, frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        )
        stimulus_support = physical_assignments == choices[:, None]
        stimulus_mass = np.sum(stimulus_support & compatible, axis=1) / np.maximum(
            stimulus_support.sum(axis=1), 1
        )
        previous_epsilon, previous_mass = fit_previous_smoothing(
            compatible, encoded, train
        )
        frequency_alpha, frequency_distribution, frequency_mass = fit_frequency_smoothing(
            compatible, encoded, train
        )
        masses = {
            "RULE_ORAL": rule_mass,
            "PREVIOUS_REPORT": previous_mass,
            "STIMULUS_CHOICE": stimulus_mass,
            "INDIVIDUAL_FREQUENCY": frequency_mass,
        }
        training_scores = {
            model: float(
                -np.log(np.clip(mass[train & encoded], MASS_FLOOR, 1.0)).mean()
            )
            for model, mass in masses.items()
            if model != "RULE_ORAL"
        }
        selected_baseline = min(training_scores, key=lambda model: (training_scores[model], model))
        masses["BASE_SELECT"] = masses[selected_baseline]
        parameter_rows.append(
            {
                "subject_id": subject_id,
                "condition": condition,
                "previous_uniform_epsilon": previous_epsilon,
                "frequency_dirichlet_alpha": frequency_alpha,
                "selected_baseline": selected_baseline,
                **{f"train_mean_log_score_{model}": value for model, value in training_scores.items()},
            }
        )
        coverage_rows.append(
            {
                "subject_id": subject_id,
                "condition": condition,
                "total_n": int(len(frame)),
                "encoded_n": int(encoded.sum()),
                "failed_or_empty_n": int((~encoded).sum()),
                "holdout_total_n": int(holdout.sum()),
                "holdout_encoded_n": int((holdout & encoded).sum()),
                "mean_set_size": float(set_size[encoded].mean()),
                "median_set_size": float(np.median(set_size[encoded])),
                "min_set_size": int(set_size[encoded].min()),
                "max_set_size": int(set_size[encoded].max()),
            }
        )
        for model, mass in masses.items():
            for segment, mask in (("train", train & encoded), ("holdout", holdout & encoded)):
                log_score = -np.log(np.clip(mass[mask], MASS_FLOOR, 1.0))
                mean_log_score = float(log_score.mean()) if len(log_score) else np.nan
                mean_mass = float(mass[mask].mean()) if mask.any() else np.nan
                metric_rows.append(
                    {
                        "subject_id": subject_id,
                        "condition": condition,
                        "model": model,
                        "segment": segment,
                        "n_reports": int(mask.sum()),
                        "total_log_score": float(log_score.sum()),
                        "mean_log_score": mean_log_score,
                        "mean_probability_mass": mean_mass,
                        "selected_baseline": selected_baseline,
                    }
                )
        for trial in np.flatnonzero(encoded):
            trial_rows.append(
                {
                    "subject_id": subject_id,
                    "condition": condition,
                    "trial_index": int(trial),
                    "segment": "holdout" if holdout[trial] else "train",
                    "compatible_set_size": int(set_size[trial]),
                    "rule_mass": float(rule_mass[trial]),
                    "previous_mass": float(previous_mass[trial]),
                    "stimulus_choice_mass": float(stimulus_mass[trial]),
                    "individual_frequency_mass": float(frequency_mass[trial]),
                    "selected_baseline": selected_baseline,
                    "selected_baseline_mass": float(masses[selected_baseline][trial]),
                }
            )

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["condition", "subject_id", "segment", "model"]
    )
    trials = pd.DataFrame(trial_rows).sort_values(
        ["condition", "subject_id", "trial_index"]
    )
    parameters = pd.DataFrame(parameter_rows).sort_values(["condition", "subject_id"])
    coverage = pd.DataFrame(coverage_rows).sort_values(["condition", "subject_id"])
    comparisons = summarize(metrics, args.seed)
    floor_results = floor_sensitivity(trials, args.seed)
    atomic_csv(output / "subject_oral_metrics.csv", metrics)
    atomic_csv(output / "trial_oral_scores.csv", trials)
    atomic_csv(output / "baseline_parameters.csv", parameters)
    atomic_csv(output / "encoding_coverage.csv", coverage)
    atomic_csv(output / "model_comparisons.csv", comparisons)
    atomic_csv(output / "floor_sensitivity.csv", floor_results)
    render_report(
        output, comparisons, metrics, parameters, coverage, floor_results
    )
    manifest = {
        "result_type": "unified_newplan_oral_external_validation",
        "status": "complete",
        "mapping": (
            "oral-center compatible set: hypothesis assigns the just-made choice to the encoded center"
        ),
        "choice_conditioning": "pi_oral proportional to prefeedback belief times q(h,current choice)",
        "mass_floor": MASS_FLOOR,
        "baselines": ["previous report", "physical stimulus plus choice", "training individual frequency"],
        "baseline_selection": "minimum mean training compatible-set log score; frequency score leave-one-out",
        "cognitive_parameters": "frozen; oral data never update beliefs or choice parameters",
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "core_run": str(core),
        "core_manifest_sha256": sha256_file(core / "manifest.json"),
        "n_subjects": int(metrics.subject_id.nunique()),
        "runtime_seconds": float(time.time() - started),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy_version,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(output / "manifest.json", manifest)
    print(f"[done] wrote {output} in {manifest['runtime_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
