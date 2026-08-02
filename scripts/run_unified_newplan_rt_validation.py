#!/usr/bin/env python3
"""Frozen-core RT validation for dynamic rule and dynamic feature models.

Per subject, Student-t log-RT regressions are fitted on the temporal training
prefix and scored on the final block.  The shared covariates are physical
ambiguity, log practice, log block position, and session start.  Cognitive
parameters and entropy trajectories are never re-fitted from RT.
"""

from __future__ import annotations

import argparse
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
from scipy.optimize import minimize
from scipy.special import gammaln


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/processed/Task2_processed.csv"
DEFAULT_CORE = ROOT / "results/zhuran/unified_newplan/core_sobol512_20260802"
DEFAULT_DYNAMIC = ROOT / "results/zhuran/unified_newplan/dynamic_readout_20260802"
DEFAULT_JOINT_NR2 = ROOT / "results/zhuran/unified_newplan/joint_dynamic_nr2_20260802"
DEFAULT_OUTPUT = ROOT / "results/zhuran/unified_newplan/rt_external_validation_20260802"
STUDENT_DF = 4.0
QC_MAD_Z = 4.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--dynamic", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--joint-nr2", type=Path, default=DEFAULT_JOINT_NR2)
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


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    location = float(np.median(values))
    scale = float(1.4826 * np.median(np.abs(values - location)))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(values, ddof=0))
    return location, max(scale, 1e-6)


def student_logpdf(y: np.ndarray, mean: np.ndarray, scale: float) -> np.ndarray:
    residual = (np.asarray(y) - np.asarray(mean)) / float(scale)
    constant = (
        gammaln((STUDENT_DF + 1.0) / 2.0)
        - gammaln(STUDENT_DF / 2.0)
        - 0.5 * math.log(STUDENT_DF * math.pi)
        - math.log(float(scale))
    )
    return constant - 0.5 * (STUDENT_DF + 1.0) * np.log1p(residual**2 / STUDENT_DF)


def fit_student_regression(
    design: np.ndarray, response: np.ndarray
) -> tuple[np.ndarray, float, dict[str, Any]]:
    design = np.asarray(design, dtype=float)
    response = np.asarray(response, dtype=float)
    beta_start = np.linalg.lstsq(design, response, rcond=None)[0]
    residual = response - design @ beta_start
    scale_start = max(float(np.sqrt(np.mean(residual**2))), 0.05)

    def objective(values: np.ndarray) -> float:
        scale = float(np.exp(values[-1]))
        return float(-student_logpdf(response, design @ values[:-1], scale).sum())

    starts = [
        np.concatenate([beta_start, [math.log(scale_start * multiplier)]])
        for multiplier in (0.5, 1.0, 2.0)
    ]
    bounds = [(None, None)] * design.shape[1] + [(math.log(0.02), math.log(5.0))]
    fits = [minimize(objective, start, method="L-BFGS-B", bounds=bounds) for start in starts]
    converged = [fit for fit in fits if bool(fit.success)]
    best = min(converged if converged else fits, key=lambda fit: float(fit.fun))
    return np.asarray(best.x[:-1]), float(np.exp(best.x[-1])), {
        "optimizer_success": bool(converged),
        "optimizer_message": str(best.message),
        "train_nll": float(best.fun),
        "n_converged_starts": int(len(converged)),
    }


def standardize_from_train(values: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(values[train_mask]))
    scale = float(np.std(values[train_mask], ddof=0))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return (values - mean) / scale, mean, scale


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def summarize_comparisons(subject_metrics: pd.DataFrame, seed: int) -> pd.DataFrame:
    specifications = [
        ("RULE_ENTROPY", "BASE", "rule_entropy_increment"),
        ("NR_ENTROPY", "BASE", "nr_entropy_increment"),
        ("RULE_ENTROPY", "NR_ENTROPY", "rule_vs_nr_entropy"),
        ("BOTH_ENTROPY", "NR_ENTROPY", "unique_rule_entropy"),
        ("BOTH_ENTROPY", "RULE_ENTROPY", "unique_nr_entropy"),
    ]
    rows = []
    for qc_specification in ("main_qc", "all_positive"):
        subset = subject_metrics[subject_metrics["qc_specification"] == qc_specification]
        for candidate, reference, label in specifications:
            for condition_label, group in [
                (str(condition), subset[subset["condition"] == condition])
                for condition in (1, 2, 3)
            ] + [("all", subset)]:
                paired = group[group["model"] == candidate].merge(
                    group[group["model"] == reference],
                    on=["subject_id", "condition", "qc_specification"],
                    suffixes=("_candidate", "_reference"),
                    validate="one_to_one",
                )
                delta = (
                    paired["holdout_mean_log_predictive_density_candidate"].to_numpy()
                    - paired["holdout_mean_log_predictive_density_reference"].to_numpy()
                )
                low, high = bootstrap_interval(
                    delta,
                    seed + sum(ord(character) for character in label + condition_label + qc_specification),
                )
                rows.append(
                    {
                        "qc_specification": qc_specification,
                        "comparison": label,
                        "candidate": candidate,
                        "reference": reference,
                        "condition": condition_label,
                        "n_subjects": int(len(delta)),
                        "mean_delta_log_predictive_density": float(delta.mean()),
                        "median_delta_log_predictive_density": float(np.median(delta)),
                        "bootstrap_mean_ci_low": low,
                        "bootstrap_mean_ci_high": high,
                        "n_improved": int((delta > 0).sum()),
                        "proportion_improved": float((delta > 0).mean()),
                    }
                )
    return pd.DataFrame(rows)


def coefficient_summary(subject_metrics: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    subset = subject_metrics[subject_metrics["qc_specification"] == "main_qc"]
    for model, column in (
        ("RULE_ENTROPY", "rule_entropy_coefficient"),
        ("NR_ENTROPY", "nr_entropy_coefficient"),
        ("BOTH_ENTROPY", "rule_entropy_coefficient"),
        ("BOTH_ENTROPY", "nr_entropy_coefficient"),
    ):
        model_rows = subset[subset["model"] == model]
        for condition_label, group in [
            (str(condition), model_rows[model_rows["condition"] == condition])
            for condition in (1, 2, 3)
        ] + [("all", model_rows)]:
            values = group[column].dropna().to_numpy(dtype=float)
            low, high = bootstrap_interval(
                values, seed + sum(ord(character) for character in model + column + condition_label)
            )
            rows.append(
                {
                    "model": model,
                    "coefficient": column,
                    "condition": condition_label,
                    "n_subjects": int(len(values)),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "bootstrap_mean_ci_low": low,
                    "bootstrap_mean_ci_high": high,
                    "n_positive": int((values > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def render_report(output: Path, comparisons: pd.DataFrame, coefficients: pd.DataFrame, qc: pd.DataFrame) -> None:
    main = comparisons[comparisons["qc_specification"] == "main_qc"]
    target = main[
        main["comparison"].isin(
            ["rule_entropy_increment", "nr_entropy_increment", "rule_vs_nr_entropy", "unique_rule_entropy"]
        )
    ]
    all_rule = target[
        (target["comparison"] == "rule_entropy_increment") & (target["condition"] == "all")
    ].iloc[0]
    direction = coefficients[
        (coefficients["model"] == "RULE_ENTROPY")
        & (coefficients["coefficient"] == "rule_entropy_coefficient")
        & (coefficients["condition"] == "all")
    ].iloc[0]
    lines = [
        "# Frozen-core RT external validation",
        "",
        "> Cognitive parameters were frozen from choice. Per-subject Student-t measurement models used training RT only and were scored on the final-block holdout.",
        "",
        "## Primary result",
        "",
        f"Adding R0KT choice entropy changed held-out mean log predictive density by {all_rule.mean_delta_log_predictive_density:.6f} (95% subject-bootstrap CI [{all_rule.bootstrap_mean_ci_low:.6f}, {all_rule.bootstrap_mean_ci_high:.6f}]; improved {int(all_rule.n_improved)}/{int(all_rule.n_subjects)} subjects).",
        f"The training coefficient for standardized rule entropy averaged {direction['mean']:.6f} (95% CI [{direction.bootstrap_mean_ci_low:.6f}, {direction.bootstrap_mean_ci_high:.6f}]); positive in {int(direction.n_positive)}/{int(direction.n_subjects)} subjects. Positive means uncertain choices are slower.",
        "",
        "## Paired held-out comparisons",
        "",
        "Positive Δ log predictive density favors the candidate.",
        "",
        "| Comparison | Condition | Mean ΔLPD/trial | 95% CI | Improved |",
        "|:--|:--|--:|:--|:--|",
    ]
    for row in target.sort_values(["comparison", "condition"]).itertuples(index=False):
        lines.append(
            f"| {row.comparison} | {row.condition} | {row.mean_delta_log_predictive_density:.6f} | "
            f"[{row.bootstrap_mean_ci_low:.6f}, {row.bootstrap_mean_ci_high:.6f}] | "
            f"{int(row.n_improved)}/{int(row.n_subjects)} |"
        )
    lines.extend(
        [
            "",
            "## Frozen measurement specification",
            "",
            f"- Student-t residuals with ν={STUDENT_DF:g}; outcome log(choRT).",
            "- Baseline: subject intercept, physical ambiguous flag, log practice, log block position, and session-start indicator. No free final-block intercept.",
            f"- Main QC: absolute log-RT deviation ≤ {QC_MAD_Z:g} Gaussian-consistent MADs from each subject's training median; threshold then applied unchanged to holdout. All-positive RT is a sensitivity analysis.",
            f"- Main QC retained {int(qc.main_holdout_kept.sum())}/{int(qc.holdout_total.sum())} holdout trials and excluded {int(qc.main_holdout_excluded.sum())}.",
            "",
            "## Boundary",
            "",
            "RT is an external channel relative to the choice loss, not an independent sample. Entropy support requires both the pre-specified positive direction and held-out predictive gain. A failure here does not erase choice prediction, but it prevents claiming that the rule state explains RT through uncertainty.",
            "",
            "## Artifacts",
            "",
            "- `subject_rt_metrics.csv`, `model_comparisons.csv`, `coefficient_summary.csv`.",
            "- `rt_qc.csv`, `trial_predictions.csv`, `manifest.json`.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    paths = {
        "data": args.data.resolve(),
        "core": args.core.resolve(),
        "dynamic": args.dynamic.resolve(),
        "joint_nr2": args.joint_nr2.resolve(),
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(paths["data"], low_memory=False).sort_values(
        ["condition", "iSub", "iSession", "iBlock", "iTrial"], kind="stable"
    )
    if not np.all(np.isfinite(data["choRT"])) or np.any(data["choRT"] <= 0):
        raise ValueError("RT must be finite and positive")

    subject_rows = []
    trial_rows = []
    qc_rows = []
    for subject_id, frame in data.groupby("iSub", sort=True):
        subject_id = int(subject_id)
        condition = int(frame["condition"].iloc[0])
        frame = frame.reset_index(drop=True)
        with np.load(paths["core"] / "subject_predictions" / f"subject_{subject_id}.npz", allow_pickle=False) as archive:
            holdout = archive["holdout_mask"].astype(bool)
        with np.load(paths["dynamic"] / "subject_predictions" / f"subject_{subject_id}.npz", allow_pickle=False) as archive:
            rule_entropy = archive["choice_entropy_R0KT_GLOBAL"].astype(float)
        with np.load(paths["joint_nr2"] / "subject_predictions" / f"subject_{subject_id}.npz", allow_pickle=False) as archive:
            nr_entropy = archive["choice_entropy_NR2T_JOINT_INDIVIDUAL"].astype(float)
        if len(frame) != len(rule_entropy) or len(frame) != len(nr_entropy):
            raise ValueError(f"prediction/data length mismatch for subject {subject_id}")

        log_rt = np.log(frame["choRT"].to_numpy(dtype=float))
        block_position = frame.groupby(["iSession", "iBlock"], sort=False).cumcount().to_numpy() + 1
        subject_trial = np.arange(1, len(frame) + 1)
        session_start = (
            frame.groupby("iSession", sort=False).cumcount().to_numpy() == 0
        ).astype(float)
        baseline_raw = np.column_stack(
            [
                frame["ambiguous"].to_numpy(dtype=float),
                np.log1p(subject_trial),
                np.log1p(block_position),
                session_start,
            ]
        )
        train = ~holdout
        standardized_baseline = []
        for column in range(baseline_raw.shape[1]):
            values, _, _ = standardize_from_train(baseline_raw[:, column], train)
            standardized_baseline.append(values)
        baseline = np.column_stack([np.ones(len(frame)), *standardized_baseline])
        rule_z, rule_mean, rule_sd = standardize_from_train(rule_entropy, train)
        nr_z, nr_mean, nr_sd = standardize_from_train(nr_entropy, train)

        location, mad_scale = robust_location_scale(log_rt[train])
        robust_z = np.abs((log_rt - location) / mad_scale)
        main_keep = robust_z <= QC_MAD_Z
        qc_rows.append(
            {
                "subject_id": subject_id,
                "condition": condition,
                "train_log_rt_median": location,
                "train_log_rt_mad_scale": mad_scale,
                "qc_threshold": QC_MAD_Z,
                "train_total": int(train.sum()),
                "holdout_total": int(holdout.sum()),
                "main_train_kept": int((train & main_keep).sum()),
                "main_holdout_kept": int((holdout & main_keep).sum()),
                "main_train_excluded": int((train & ~main_keep).sum()),
                "main_holdout_excluded": int((holdout & ~main_keep).sum()),
            }
        )
        designs = {
            "BASE": baseline,
            "RULE_ENTROPY": np.column_stack([baseline, rule_z]),
            "NR_ENTROPY": np.column_stack([baseline, nr_z]),
            "BOTH_ENTROPY": np.column_stack([baseline, rule_z, nr_z]),
        }
        for qc_specification, keep in (("main_qc", main_keep), ("all_positive", np.ones(len(frame), dtype=bool))):
            train_mask = train & keep
            test_mask = holdout & keep
            for model, design in designs.items():
                beta, scale, diagnostics = fit_student_regression(
                    design[train_mask], log_rt[train_mask]
                )
                prediction = design[test_mask] @ beta
                lpd = student_logpdf(log_rt[test_mask], prediction, scale)
                rule_coefficient = np.nan
                nr_coefficient = np.nan
                if model == "RULE_ENTROPY":
                    rule_coefficient = float(beta[-1])
                elif model == "NR_ENTROPY":
                    nr_coefficient = float(beta[-1])
                elif model == "BOTH_ENTROPY":
                    rule_coefficient = float(beta[-2])
                    nr_coefficient = float(beta[-1])
                subject_rows.append(
                    {
                        "subject_id": subject_id,
                        "condition": condition,
                        "qc_specification": qc_specification,
                        "model": model,
                        "train_n": int(train_mask.sum()),
                        "holdout_n": int(test_mask.sum()),
                        "holdout_total_log_predictive_density": float(lpd.sum()),
                        "holdout_mean_log_predictive_density": float(lpd.mean()),
                        "holdout_rmse_log_rt": float(np.sqrt(np.mean((log_rt[test_mask] - prediction) ** 2))),
                        "rule_entropy_coefficient": rule_coefficient,
                        "nr_entropy_coefficient": nr_coefficient,
                        "student_scale": scale,
                        **diagnostics,
                    }
                )
                for trial, predicted, log_density in zip(
                    np.flatnonzero(test_mask), prediction, lpd
                ):
                    trial_rows.append(
                        {
                            "subject_id": subject_id,
                            "condition": condition,
                            "qc_specification": qc_specification,
                            "model": model,
                            "trial_index": int(trial),
                            "observed_log_rt": float(log_rt[trial]),
                            "predicted_log_rt": float(predicted),
                            "log_predictive_density": float(log_density),
                            "rule_entropy": float(rule_entropy[trial]),
                            "nr_entropy": float(nr_entropy[trial]),
                        }
                    )

    subject_metrics = pd.DataFrame(subject_rows).sort_values(
        ["condition", "subject_id", "qc_specification", "model"]
    )
    trial_predictions = pd.DataFrame(trial_rows).sort_values(
        ["condition", "subject_id", "qc_specification", "model", "trial_index"]
    )
    qc = pd.DataFrame(qc_rows).sort_values(["condition", "subject_id"])
    comparisons = summarize_comparisons(subject_metrics, args.seed)
    coefficients = coefficient_summary(subject_metrics, args.seed)
    atomic_csv(output / "subject_rt_metrics.csv", subject_metrics)
    atomic_csv(output / "trial_predictions.csv", trial_predictions)
    atomic_csv(output / "rt_qc.csv", qc)
    atomic_csv(output / "model_comparisons.csv", comparisons)
    atomic_csv(output / "coefficient_summary.csv", coefficients)
    render_report(output, comparisons, coefficients, qc)
    manifest = {
        "result_type": "unified_newplan_rt_external_validation",
        "status": "complete",
        "student_df": STUDENT_DF,
        "main_qc_mad_z": QC_MAD_Z,
        "outcome": "log(choRT)",
        "baseline_covariates": [
            "subject intercept",
            "physical ambiguous flag",
            "log practice",
            "log block position",
            "session start",
        ],
        "parameter_scope": "per-subject RT measurement parameters fit on training prefix only",
        "cognitive_parameter_scope": "frozen from choice; RT never changes model probabilities or entropy",
        "paths": {key: str(value) for key, value in paths.items()},
        "input_hashes": {
            "data": sha256_file(paths["data"]),
            "core_manifest": sha256_file(paths["core"] / "manifest.json"),
            "dynamic_manifest": sha256_file(paths["dynamic"] / "manifest.json"),
            "joint_nr2_manifest": sha256_file(paths["joint_nr2"] / "manifest.json"),
        },
        "n_subjects": int(subject_metrics.subject_id.nunique()),
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
