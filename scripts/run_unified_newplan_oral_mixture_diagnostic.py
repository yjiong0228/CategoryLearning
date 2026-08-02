#!/usr/bin/env python3
"""Adaptive one-parameter oral measurement mixture diagnostic.

The pure rule oral readout is the primary test and remains unchanged.  This
follow-up asks whether a single training-fitted cross-condition weight on the
frozen rule readout adds held-out information beyond each subject's already
selected oral baseline:

    M_mix = w * M_rule + (1 - w) * M_baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORAL = ROOT / "results/zhuran/unified_newplan/oral_external_validation_20260802"
DEFAULT_OUTPUT = ROOT / "results/zhuran/unified_newplan/oral_mixture_diagnostic_20260802"
MASS_FLOOR = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oral", type=Path, default=DEFAULT_ORAL)
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


def fit_weight(frame: pd.DataFrame) -> tuple[float, float]:
    rule = frame["rule_mass"].to_numpy(dtype=float)
    baseline = frame["selected_baseline_mass"].to_numpy(dtype=float)

    def objective(weight: float) -> float:
        mass = weight * rule + (1.0 - weight) * baseline
        return float(-np.log(np.clip(mass, MASS_FLOOR, 1.0)).sum())

    fit = minimize_scalar(
        objective, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-12}
    )
    return float(fit.x), float(fit.fun)


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def summarize(subjects: pd.DataFrame, seed: int) -> pd.DataFrame:
    specifications = [
        ("global_mixture_score", "baseline_score", "global_mixture_vs_baseline"),
        ("global_mixture_score", "rule_score", "global_mixture_vs_pure_rule"),
        ("condition_mixture_score", "global_mixture_score", "condition_weight_increment"),
        ("individual_mixture_score", "global_mixture_score", "individual_weight_increment"),
    ]
    rows = []
    for candidate, reference, label in specifications:
        for condition_label, group in [
            (str(condition), subjects[subjects["condition"] == condition])
            for condition in (1, 2, 3)
        ] + [("all", subjects)]:
            # Lower log score is better, so reference minus candidate is positive.
            delta = group[reference].to_numpy() - group[candidate].to_numpy()
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


def floor_sensitivity(holdout: pd.DataFrame, global_weight: float, seed: int) -> pd.DataFrame:
    rows = []
    for floor in (1e-6, 1e-9, 1e-12):
        payload = []
        for (subject_id, condition), group in holdout.groupby(["subject_id", "condition"]):
            rule = group["rule_mass"].to_numpy(dtype=float)
            baseline = group["selected_baseline_mass"].to_numpy(dtype=float)
            mixture = global_weight * rule + (1.0 - global_weight) * baseline
            delta = float(
                np.log(np.clip(mixture, floor, 1.0)).mean()
                - np.log(np.clip(baseline, floor, 1.0)).mean()
            )
            payload.append({"subject_id": subject_id, "condition": condition, "delta": delta})
        subject = pd.DataFrame(payload)
        for condition_label, group in [
            (str(condition), subject[subject["condition"] == condition])
            for condition in (1, 2, 3)
        ] + [("all", subject)]:
            values = group["delta"].to_numpy(dtype=float)
            low, high = bootstrap_interval(
                values,
                seed + int(-math.log10(floor)) * 1000 + sum(ord(c) for c in condition_label),
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
    weights: pd.DataFrame,
    comparisons: pd.DataFrame,
    floor_results: pd.DataFrame,
) -> None:
    global_weight = float(weights[weights["scope"] == "global"]["weight"].iloc[0])
    primary = comparisons[comparisons["comparison"] == "global_mixture_vs_baseline"]
    all_row = primary[primary["condition"] == "all"].iloc[0]
    lines = [
        "# Adaptive oral measurement-mixture diagnostic",
        "",
        "> Status: exploratory follow-up after the pure rule oral readout failed its primary mean-score gate. It does not replace or retroactively change that primary result.",
        "",
        "## Result",
        "",
        f"The single cross-condition training weight was w={global_weight:.6f} on the frozen rule readout and {1-global_weight:.6f} on the subject's training-selected oral baseline.",
        f"On held-out reports, the mixture improved compatible-set log score by {all_row.mean_delta_log_score:.6f} (95% subject-bootstrap CI [{all_row.bootstrap_mean_ci_low:.6f}, {all_row.bootstrap_mean_ci_high:.6f}]; {int(all_row.n_improved)}/{int(all_row.n_subjects)} subjects).",
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
            "## Complexity checks",
            "",
            "Condition-specific and individual weights were fitted on the same training reports only. Their direct held-out increments over the global weight determine whether added heterogeneity is retained.",
            "",
            "| Comparison | Condition | Mean Δ | 95% CI | Improved |",
            "|:--|:--|--:|:--|:--|",
        ]
    )
    for row in comparisons[
        comparisons["comparison"].isin(
            ["condition_weight_increment", "individual_weight_increment"]
        )
    ].itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {row.condition} | {row.mean_delta_log_score:.6f} | "
            f"[{row.bootstrap_mean_ci_low:.6f}, {row.bootstrap_mean_ci_high:.6f}] | "
            f"{int(row.n_improved)}/{int(row.n_subjects)} |"
        )
    lines.extend(
        [
            "",
            "## Floor sensitivity",
            "",
            "| Floor | Condition | Mean Δ | 95% CI | Improved |",
            "|--:|:--|--:|:--|:--|",
        ]
    )
    for row in floor_results.itertuples(index=False):
        lines.append(
            f"| {row.mass_floor:.0e} | {row.condition} | {row.mean_delta_log_score:.6f} | "
            f"[{row.bootstrap_mean_ci_low:.6f}, {row.bootstrap_mean_ci_high:.6f}] | "
            f"{int(row.n_improved)}/{int(row.n_subjects)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This result supports incremental rule-belief information in oral reports under a minimal contamination/measurement model. Because the measurement mixture was introduced after seeing the pure-readout failure, it is adaptive evidence and needs independent confirmation. It does not rescue the failed RT prediction or establish that every verbal report is a direct sample from the latent rule belief.",
            "",
            "## Artifacts",
            "",
            "- `weights.csv`, `subject_scores.csv`, `model_comparisons.csv`.",
            "- `floor_sensitivity.csv`, `manifest.json`.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    oral = args.oral.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trials = pd.read_csv(oral / "trial_oral_scores.csv")
    train = trials[trials["segment"] == "train"]
    holdout = trials[trials["segment"] == "holdout"]
    global_weight, global_train_nll = fit_weight(train)
    condition_weights = {
        int(condition): fit_weight(group)[0]
        for condition, group in train.groupby("condition")
    }
    individual_weights = {
        int(subject_id): fit_weight(group)[0]
        for subject_id, group in train.groupby("subject_id")
    }

    subject_rows = []
    for (subject_id, condition), group in holdout.groupby(["subject_id", "condition"]):
        rule = group["rule_mass"].to_numpy(dtype=float)
        baseline = group["selected_baseline_mass"].to_numpy(dtype=float)
        weights_for_models = {
            "global_mixture_score": global_weight,
            "condition_mixture_score": condition_weights[int(condition)],
            "individual_mixture_score": individual_weights[int(subject_id)],
        }
        scores = {
            "rule_score": float(-np.log(np.clip(rule, MASS_FLOOR, 1.0)).mean()),
            "baseline_score": float(-np.log(np.clip(baseline, MASS_FLOOR, 1.0)).mean()),
        }
        for name, weight in weights_for_models.items():
            mixture = weight * rule + (1.0 - weight) * baseline
            scores[name] = float(
                -np.log(np.clip(mixture, MASS_FLOOR, 1.0)).mean()
            )
        subject_rows.append(
            {
                "subject_id": int(subject_id),
                "condition": int(condition),
                "n_reports": int(len(group)),
                "global_weight": global_weight,
                "condition_weight": condition_weights[int(condition)],
                "individual_weight": individual_weights[int(subject_id)],
                **scores,
            }
        )
    subjects = pd.DataFrame(subject_rows).sort_values(["condition", "subject_id"])
    weights = pd.DataFrame(
        [
            {
                "scope": "global",
                "identifier": "all",
                "weight": global_weight,
                "train_nll": global_train_nll,
                "n_train_reports": int(len(train)),
            }
        ]
        + [
            {
                "scope": "condition",
                "identifier": str(condition),
                "weight": weight,
                "train_nll": fit_weight(train[train["condition"] == condition])[1],
                "n_train_reports": int((train["condition"] == condition).sum()),
            }
            for condition, weight in condition_weights.items()
        ]
        + [
            {
                "scope": "individual",
                "identifier": str(subject_id),
                "weight": weight,
                "train_nll": fit_weight(train[train["subject_id"] == subject_id])[1],
                "n_train_reports": int((train["subject_id"] == subject_id).sum()),
            }
            for subject_id, weight in individual_weights.items()
        ]
    )
    comparisons = summarize(subjects, args.seed)
    floor_results = floor_sensitivity(holdout, global_weight, args.seed)
    atomic_csv(output / "weights.csv", weights)
    atomic_csv(output / "subject_scores.csv", subjects)
    atomic_csv(output / "model_comparisons.csv", comparisons)
    atomic_csv(output / "floor_sensitivity.csv", floor_results)
    render_report(output, weights, comparisons, floor_results)
    oral_manifest = oral / "manifest.json"
    manifest = {
        "result_type": "unified_newplan_adaptive_oral_measurement_mixture",
        "status": "complete",
        "evidence_status": "adaptive_exploratory",
        "equation": "M_mix = w * M_rule + (1-w) * M_selected_baseline",
        "primary_model_unchanged": "pure rule oral primary test remains failed",
        "mass_floor": MASS_FLOOR,
        "oral_input": str(oral),
        "oral_manifest_sha256": sha256_file(oral_manifest),
        "n_subjects_with_heldout_reports": int(len(subjects)),
        "global_weight": global_weight,
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
