#!/usr/bin/env python3
"""Validate choice-only strategy states against feedback-before oral reports.

The tested predictor is the target-rule state probability available before the
current choice.  It was inferred from earlier choices only: neither the current
choice, current oral report, nor current feedback contributes to it.  A pooled
condition slope with subject intercepts is fitted on training reports and
scored on the final-block oral reports.  The baseline is each subject's
smoothed training frequency of target-compatible reports.
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
from scipy.optimize import minimize
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = (
    ROOT / "results/zhuran/unified_newplan/behavior_anchored_state_20260802"
)
DEFAULT_OUTPUT = (
    ROOT / "results/zhuran/unified_newplan/behavior_state_oral_validation_20260802"
)
PROBABILITY_EPS = 1e-7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
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


def binary_log_score(outcome: np.ndarray, probability: np.ndarray) -> np.ndarray:
    outcome = np.asarray(outcome, dtype=float)
    probability = np.clip(
        np.asarray(probability, dtype=float), PROBABILITY_EPS, 1.0 - PROBABILITY_EPS
    )
    return -(outcome * np.log(probability) + (1.0 - outcome) * np.log1p(-probability))


def fit_condition(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    subject_ids = sorted(frame["subject_id"].unique().astype(int).tolist())
    subject_lookup = {subject_id: index for index, subject_id in enumerate(subject_ids)}
    subject_code = frame["subject_id"].map(subject_lookup).to_numpy(dtype=int)
    train = frame["segment"].eq("train").to_numpy()
    outcome = frame["oral_target_compatible"].to_numpy(dtype=float)
    raw_probability = np.clip(
        frame["choice_only_prior_target_rule_probability"].to_numpy(dtype=float),
        PROBABILITY_EPS,
        1.0 - PROBABILITY_EPS,
    )
    raw_logit = np.log(raw_probability) - np.log1p(-raw_probability)
    x_mean = float(raw_logit[train].mean())
    x_sd = float(raw_logit[train].std(ddof=0))
    if not np.isfinite(x_sd) or x_sd < 1e-8:
        x_sd = 1.0
    predictor = (raw_logit - x_mean) / x_sd

    baseline_probability = np.empty(len(frame), dtype=float)
    intercept_start = np.empty(len(subject_ids), dtype=float)
    for subject_id, code in subject_lookup.items():
        mask = (subject_code == code) & train
        successes = float(outcome[mask].sum())
        probability = (successes + 0.5) / (int(mask.sum()) + 1.0)
        baseline_probability[subject_code == code] = probability
        intercept_start[code] = math.log(probability / (1.0 - probability))

    def objective(values: np.ndarray) -> float:
        intercept = values[:-1]
        slope = float(values[-1])
        probability = expit(intercept[subject_code[train]] + slope * predictor[train])
        loss = float(binary_log_score(outcome[train], probability).sum())
        # Weak regularization only prevents separation in subjects whose
        # training reports are uniformly compatible or incompatible.
        return loss + 0.5 * float(np.sum((intercept / 10.0) ** 2)) + 0.5 * (slope / 5.0) ** 2

    starts = [
        np.concatenate([intercept_start, [slope]])
        for slope in (-0.5, 0.0, 0.5, 1.0)
    ]
    bounds = [(-12.0, 12.0)] * len(subject_ids) + [(-5.0, 5.0)]
    fits = [
        minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        for start in starts
    ]
    converged = [fit for fit in fits if bool(fit.success)]
    best = min(converged if converged else fits, key=lambda result: float(result.fun))
    fitted_probability = expit(
        best.x[:-1][subject_code] + float(best.x[-1]) * predictor
    )
    prediction = frame[
        [
            "subject_id",
            "condition",
            "trial",
            "segment",
            "oral_target_compatible",
            "choice_only_prior_target_rule_probability",
        ]
    ].copy()
    prediction["baseline_probability"] = baseline_probability
    prediction["state_probability"] = fitted_probability
    prediction["baseline_log_score"] = binary_log_score(
        outcome, baseline_probability
    )
    prediction["state_log_score"] = binary_log_score(outcome, fitted_probability)
    diagnostics = {
        "condition": int(frame["condition"].iloc[0]),
        "n_subjects": int(len(subject_ids)),
        "n_training_reports": int(train.sum()),
        "n_holdout_reports": int((~train).sum()),
        "state_logit_train_mean": x_mean,
        "state_logit_train_sd": x_sd,
        "state_slope": float(best.x[-1]),
        "optimizer_success": bool(converged),
        "optimizer_message": str(best.message),
        "n_starts": int(len(fits)),
        "n_same_optimal_region": int(
            sum(abs(float(result.fun) - float(best.fun)) <= 1e-4 for result in fits)
        ),
        "penalized_train_objective": float(best.fun),
    }
    return prediction, diagnostics


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def summarize(subject_metrics: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for condition_label, group in [
        (str(condition), subject_metrics[subject_metrics["condition"].eq(condition)])
        for condition in (1, 2, 3)
    ] + [("all", subject_metrics)]:
        delta = (
            group["holdout_mean_baseline_log_score"].to_numpy(dtype=float)
            - group["holdout_mean_state_log_score"].to_numpy(dtype=float)
        )
        low, high = bootstrap_interval(
            delta, seed + sum(ord(character) for character in condition_label)
        )
        rows.append(
            {
                "condition": condition_label,
                "n_subjects": int(len(delta)),
                "mean_delta_log_score": float(delta.mean()),
                "median_delta_log_score": float(np.median(delta)),
                "bootstrap_mean_ci_low": low,
                "bootstrap_mean_ci_high": high,
                "n_improved": int((delta > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def render_report(
    output: Path,
    comparisons: pd.DataFrame,
    parameters: pd.DataFrame,
    coverage: dict[str, int],
) -> None:
    primary = comparisons[comparisons["condition"].eq("all")].iloc[0]
    lines = [
        "# Choice-only state validation against feedback-before oral reports",
        "",
        "> The tested target-rule probability was available before the current choice and was inferred from earlier choices only. Current choice, oral report, and feedback were unavailable.",
        "",
        "## Result",
        "",
        f"Adding the pre-choice target-rule state to subject report-frequency baselines changed held-out compatible/not-compatible log score by {primary.mean_delta_log_score:.6f} (95% subject-bootstrap CI [{primary.bootstrap_mean_ci_low:.6f}, {primary.bootstrap_mean_ci_high:.6f}]; improved {int(primary.n_improved)}/{int(primary.n_subjects)} subjects).",
        "",
        "Positive values mean the choice-only state better predicted whether the next feedback-before report was compatible with the target rule.",
        "",
        "| Condition | Mean Δ log score | 95% CI | Improved |",
        "|:--|--:|:--|:--|",
    ]
    for row in comparisons.itertuples(index=False):
        lines.append(
            f"| {row.condition} | {row.mean_delta_log_score:.6f} | "
            f"[{row.bootstrap_mean_ci_low:.6f}, {row.bootstrap_mean_ci_high:.6f}] | "
            f"{int(row.n_improved)}/{int(row.n_subjects)} |"
        )
    lines.extend(
        [
            "",
            "## Direction and coverage",
            "",
            f"Encoded reports: {coverage['encoded']}/{coverage['total']}; encoded holdout reports: {coverage['holdout_encoded']}/{coverage['holdout_total']}.",
            "",
            "| Condition | Standardized state slope | Same optimum |",
            "|:--|--:|:--|",
        ]
    )
    for row in parameters.itertuples(index=False):
        lines.append(
            f"| {int(row.condition)} | {row.state_slope:.6f} | "
            f"{int(row.n_same_optimal_region)}/{int(row.n_starts)} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Target compatibility is a broad structured-center indicator, not proof that a participant verbally stated one unique rule. The test is valuable only as an external temporal consequence of the choice-only state, and does not by itself establish a verbal-rule representation.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    state_path = args.state.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trials = pd.read_csv(state_path / "trial_states.csv.gz")
    required = [
        "subject_id",
        "condition",
        "trial",
        "segment",
        "oral_encoded",
        "oral_target_compatible",
        "choice_only_prior_target_rule_probability",
    ]
    missing = set(required) - set(trials.columns)
    if missing:
        raise ValueError(f"Missing state columns: {sorted(missing)}")
    total = len(trials)
    holdout_total = int(trials["segment"].eq("holdout").sum())
    encoded = trials[trials["oral_encoded"].astype(bool)].copy()
    predictions = []
    parameter_rows = []
    for condition in (1, 2, 3):
        prediction, diagnostics = fit_condition(
            encoded[encoded["condition"].eq(condition)].reset_index(drop=True)
        )
        predictions.append(prediction)
        parameter_rows.append(diagnostics)
    prediction = pd.concat(predictions, ignore_index=True).sort_values(
        ["condition", "subject_id", "trial"]
    )
    parameters = pd.DataFrame(parameter_rows).sort_values("condition")
    holdout = prediction[prediction["segment"].eq("holdout")]
    subject_metrics = (
        holdout.groupby(["subject_id", "condition"], as_index=False)
        .agg(
            n_holdout_reports=("trial", "size"),
            holdout_mean_baseline_log_score=("baseline_log_score", "mean"),
            holdout_mean_state_log_score=("state_log_score", "mean"),
        )
        .sort_values(["condition", "subject_id"])
    )
    comparisons = summarize(subject_metrics, args.seed)
    coverage = {
        "total": int(total),
        "encoded": int(len(encoded)),
        "holdout_total": holdout_total,
        "holdout_encoded": int(len(holdout)),
    }
    atomic_csv(output / "parameters.csv", parameters)
    atomic_csv(output / "trial_predictions.csv", prediction)
    atomic_csv(output / "subject_metrics.csv", subject_metrics)
    atomic_csv(output / "model_comparisons.csv", comparisons)
    render_report(output, comparisons, parameters, coverage)
    manifest = {
        "result_type": "behavior_state_external_oral_validation",
        "status": "complete",
        "predictor": "choice-only target-rule state prior before current choice",
        "outcome": "structured oral center compatible with target rule and current choice",
        "baseline": "Jeffreys-smoothed subject training frequency",
        "candidate": "subject intercept plus condition-shared standardized state slope",
        "temporal_order": (
            "predictor uses history through t-1; outcome is report after choice t and "
            "before feedback t"
        ),
        "coverage": coverage,
        "n_subjects": int(subject_metrics.subject_id.nunique()),
        "state_manifest_sha256": sha256_file(state_path / "manifest.json"),
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
