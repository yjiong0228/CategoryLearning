#!/usr/bin/env python3
"""Jointly fit a time-varying readout inside the NR2 learning recursion.

This is the conservative non-rule competitor to R0KT_GLOBAL.  Each subject's
learning rate, initial sensitivity, and practice slope are estimated from the
training prefix only.  Giving NR2 an individual slope makes it more flexible
than the rule model's single cross-condition shared slope.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any

import numpy as np
import pandas as pd
from scipy import __version__ as scipy_version
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_unified_newplan_dynamic_readout import entropy, score  # noqa: E402
from src.Bayesian_state.utils.unified_newplan import (  # noqa: E402
    FEATURE_COLUMNS,
    ORDER_COLUMNS,
    feedback_target_matrix,
    nr2_dynamic_readout_predictions,
)


DEFAULT_DATA = ROOT / "data/processed/Task2_processed.csv"
DEFAULT_CORE = ROOT / "results/zhuran/unified_newplan/core_sobol512_20260802"
DEFAULT_DYNAMIC = ROOT / "results/zhuran/unified_newplan/dynamic_readout_20260802"
DEFAULT_OUTPUT = ROOT / "results/zhuran/unified_newplan/joint_dynamic_nr2_20260802"
LOG_KAPPA_BOUNDS = (math.log(0.01), math.log(20.0))
SLOPE_BOUNDS = (-4.0, 4.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--dynamic", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=96)
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


def atomic_savez(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def fit_joint_dynamic_nr2(
    stimuli: np.ndarray,
    choices: np.ndarray,
    targets: np.ndarray,
    train_mask: np.ndarray,
    practice: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the individual dynamic NR2 candidate on a temporal prefix."""

    stimuli = np.asarray(stimuli, dtype=np.float64)
    choices = np.asarray(choices, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.float64)
    train_rows = np.flatnonzero(np.asarray(train_mask, dtype=bool))
    practice = np.asarray(practice, dtype=np.float64)

    def objective(raw: np.ndarray) -> float:
        learning_rate = float(expit(raw[0]))
        probabilities = nr2_dynamic_readout_predictions(
            stimuli,
            targets,
            learning_rate,
            float(raw[1]),
            float(raw[2]),
            practice,
        )
        observed = np.clip(
            probabilities[train_rows, choices[train_rows]], 1e-7, 1.0
        )
        return float(-np.log(observed).sum())

    starts = [
        np.asarray(
            [math.log(alpha / (1.0 - alpha)), math.log(kappa), slope],
            dtype=float,
        )
        for alpha in (0.03, 0.15, 0.5)
        for kappa in (0.1, 0.5, 1.0)
        for slope in (0.0, 0.8, 1.6)
    ]
    bounds = [
        (math.log(1e-4 / (1 - 1e-4)), math.log(0.999 / 0.001)),
        LOG_KAPPA_BOUNDS,
        SLOPE_BOUNDS,
    ]
    fits = [
        minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        for start in starts
    ]
    converged = [result for result in fits if bool(result.success)]
    best = min(converged if converged else fits, key=lambda result: float(result.fun))
    learning_rate = float(expit(best.x[0]))
    intercept = float(best.x[1])
    slope = float(best.x[2])
    probability = nr2_dynamic_readout_predictions(
        stimuli, targets, learning_rate, intercept, slope, practice
    )
    return probability, {
        "learning_rate": learning_rate,
        "intercept": intercept,
        "practice_slope": slope,
        "optimizer_success": bool(converged),
        "optimizer_message": str(best.message),
        "n_converged_starts": int(len(converged)),
        "n_starts": len(starts),
        "n_same_optimal_region": int(
            sum(abs(float(result.fun) - float(best.fun)) <= 1e-5 for result in fits)
        ),
        "train_nll": float(best.fun),
    }


def _fit_worker(task: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    started = time.time()
    frame = pd.DataFrame(task["frame"]).sort_values(
        list(ORDER_COLUMNS), kind="stable"
    ).reset_index(drop=True)
    subject_id = int(frame["iSub"].iloc[0])
    condition = int(frame["condition"].iloc[0])
    stimuli = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    choices = frame["choice"].to_numpy(dtype=np.int64) - 1
    feedback = frame["feedback"].to_numpy(dtype=float)
    targets = feedback_target_matrix(condition, choices, feedback)
    with np.load(Path(task["core_prediction"]), allow_pickle=False) as archive:
        holdout = archive["holdout_mask"].astype(bool)
        p_nr2 = archive["p_NR2"].astype(float)
    with np.load(Path(task["dynamic_prediction"]), allow_pickle=False) as archive:
        practice = archive["practice"].astype(np.float64)
        p_rule = archive["p_R0KT_GLOBAL"].astype(float)
        p_nr2_posthoc = archive["p_NR2T_GLOBAL"].astype(float)
    train = ~holdout
    train_rows = np.flatnonzero(train)
    probability, fitted = fit_joint_dynamic_nr2(
        stimuli, choices, targets, train, practice
    )
    learning_rate = float(fitted["learning_rate"])
    intercept = float(fitted["intercept"])
    slope = float(fitted["practice_slope"])
    models = {
        "R0KT_GLOBAL": p_rule,
        "NR2": p_nr2,
        "NR2T_GLOBAL_POSTHOC": p_nr2_posthoc,
        "NR2T_JOINT_INDIVIDUAL": probability,
    }
    metric_rows = []
    for model, p in models.items():
        for segment, mask in (("train", train), ("holdout", holdout)):
            metric_rows.append(
                {
                    "subject_id": subject_id,
                    "condition": condition,
                    "model": model,
                    "segment": segment,
                    **score(p, choices, mask),
                }
            )
    atomic_savez(
        Path(task["output_prediction"]),
        subject_id=np.asarray(subject_id),
        condition=np.asarray(condition),
        choice=choices.astype(np.int8),
        holdout_mask=holdout,
        practice=practice.astype(np.float32),
        p_NR2T_JOINT_INDIVIDUAL=probability.astype(np.float32),
        choice_entropy_NR2T_JOINT_INDIVIDUAL=entropy(probability).astype(np.float32),
        parameters_json=np.asarray(
            json.dumps(
                {
                    "learning_rate": learning_rate,
                    "intercept": intercept,
                    "practice_slope": slope,
                },
                sort_keys=True,
            )
        ),
    )
    return {
        "subject_id": subject_id,
        "condition": condition,
        "metrics": metric_rows,
        "learning_rate": learning_rate,
        "intercept": intercept,
        "practice_slope": slope,
        "kappa_first": float(math.exp(np.clip(intercept, *LOG_KAPPA_BOUNDS))),
        "kappa_train_end": float(
            math.exp(
                np.clip(
                    intercept + slope * practice[train_rows[-1]],
                    *LOG_KAPPA_BOUNDS,
                )
            )
        ),
        "kappa_holdout_end": float(
            math.exp(
                np.clip(intercept + slope * practice[-1], *LOG_KAPPA_BOUNDS)
            )
        ),
        "learning_rate_at_boundary": bool(
            learning_rate <= 0.000101 or learning_rate >= 0.9989
        ),
        "intercept_at_boundary": bool(
            intercept <= LOG_KAPPA_BOUNDS[0] + 1e-5
            or intercept >= LOG_KAPPA_BOUNDS[1] - 1e-5
        ),
        "slope_at_boundary": bool(abs(slope) >= 3.9999),
        "optimizer_success": bool(fitted["optimizer_success"]),
        "optimizer_message": str(fitted["optimizer_message"]),
        "n_starts": int(fitted["n_starts"]),
        "n_same_optimal_region": int(fitted["n_same_optimal_region"]),
        "train_nll": float(fitted["train_nll"]),
        "runtime_seconds": float(time.time() - started),
    }


def bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def comparisons(metrics: pd.DataFrame, seed: int) -> pd.DataFrame:
    specifications = [
        ("R0KT_GLOBAL", "NR2T_JOINT_INDIVIDUAL", "rule_vs_joint_dynamic_NR2"),
        ("R0KT_GLOBAL", "NR2", "rule_vs_static_NR2"),
        (
            "NR2T_JOINT_INDIVIDUAL",
            "NR2",
            "joint_dynamic_increment_NR2",
        ),
        (
            "NR2T_JOINT_INDIVIDUAL",
            "NR2T_GLOBAL_POSTHOC",
            "joint_vs_posthoc_dynamic_NR2",
        ),
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
            delta = (
                paired["nll_per_trial_reference"].to_numpy()
                - paired["nll_per_trial_candidate"].to_numpy()
            )
            low, high = bootstrap_interval(
                delta,
                seed + sum(ord(character) for character in label + condition_label),
            )
            try:
                p_value = float(wilcoxon(delta[~np.isclose(delta, 0)]).pvalue)
            except ValueError:
                p_value = float("nan")
            rows.append(
                {
                    "comparison": label,
                    "candidate": candidate,
                    "reference": reference,
                    "condition": condition_label,
                    "n_subjects": int(len(delta)),
                    "mean_delta_nll_per_trial": float(delta.mean()),
                    "median_delta_nll_per_trial": float(np.median(delta)),
                    "bootstrap_mean_ci_low": low,
                    "bootstrap_mean_ci_high": high,
                    "n_improved": int((delta > 0).sum()),
                    "proportion_improved": float((delta > 0).mean()),
                    "wilcoxon_p_uncorrected": p_value,
                }
            )
    return pd.DataFrame(rows)


def render_report(output: Path, comparison: pd.DataFrame, parameters: pd.DataFrame) -> None:
    primary = comparison[comparison["comparison"] == "rule_vs_joint_dynamic_NR2"]
    all_row = primary[primary["condition"] == "all"].iloc[0]
    lines = [
        "# Joint dynamic-NR2 fairness screen",
        "",
        "> Status: conservative held-out fairness check. NR2 receives an individually fitted practice slope, whereas the rule candidate retains one cross-condition shared slope.",
        "",
        "## Result",
        "",
        f"R0KT_GLOBAL versus jointly fitted NR2T_JOINT_INDIVIDUAL: mean ΔNLL/trial {all_row.mean_delta_nll_per_trial:.6f}, 95% subject-bootstrap CI [{all_row.bootstrap_mean_ci_low:.6f}, {all_row.bootstrap_mean_ci_high:.6f}], improved {int(all_row.n_improved)}/{int(all_row.n_subjects)} subjects.",
        "",
        "Positive values favor the rule candidate. All NR2 learning, intercept, and slope parameters were estimated only on each subject's training prefix.",
        "",
        "| Condition | Mean ΔNLL/trial | 95% CI | Improved |",
        "|:--|--:|:--|:--|",
    ]
    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.condition} | {row.mean_delta_nll_per_trial:.6f} | "
            f"[{row.bootstrap_mean_ci_low:.6f}, {row.bootstrap_mean_ci_high:.6f}] | "
            f"{int(row.n_improved)}/{int(row.n_subjects)} |"
        )
    lines.extend(
        [
            "",
            "## Parameter and optimization audit",
            "",
            f"- Individual NR2 practice slopes: median {parameters.practice_slope.median():.6f}, positive in {(parameters.practice_slope > 0).sum()}/{len(parameters)} subjects, at bound in {parameters.slope_at_boundary.sum()}/{len(parameters)}.",
            f"- Learning-rate boundary hits: {parameters.learning_rate_at_boundary.sum()}/{len(parameters)}; intercept boundary hits: {parameters.intercept_at_boundary.sum()}/{len(parameters)}.",
            f"- Optimizer failures: {(~parameters.optimizer_success).sum()}/{len(parameters)}; minimum number of starts in the same optimum: {parameters.n_same_optimal_region.min()} of 27.",
            "",
            "## Interpretation boundary",
            "",
            "This closes the obvious post-hoc-calibration loophole in the real-data choice comparison, but it is still a MAP screen. A generative rule-versus-feature model-recovery analysis is required before treating the representation identity as established. Individual NR2 slopes are a deliberately generous nuisance alternative and are not interpreted psychologically.",
            "",
            "## Artifacts",
            "",
            "- `subject_metrics.csv`, `model_comparisons.csv`, `parameters.csv`.",
            "- `subject_predictions/`: jointly generated NR2 probabilities and entropy.",
            "- `worker_errors.json`, `manifest.json`: computation and provenance audit.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    data_path = args.data.resolve()
    core = args.core.resolve()
    dynamic = args.dynamic.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_output = output / "subject_predictions"
    prediction_output.mkdir(exist_ok=True)
    data = pd.read_csv(data_path, low_memory=False).sort_values(
        ["condition", "iSub", *ORDER_COLUMNS], kind="stable"
    )
    core_manifest = json.loads((core / "manifest.json").read_text(encoding="utf-8"))
    dynamic_manifest = json.loads(
        (dynamic / "manifest.json").read_text(encoding="utf-8")
    )
    if core_manifest.get("status") != "complete" or dynamic_manifest.get("status") != "complete":
        raise ValueError("core and dynamic inputs must be complete")

    # Compile the numba recursion before forking.
    nr2_dynamic_readout_predictions(
        np.zeros((2, 4)),
        np.full((2, 2), 0.5),
        0.1,
        math.log(0.5),
        1.0,
        np.asarray([0.0, 1.0]),
    )
    tasks = []
    for subject_id, frame in data.groupby("iSub", sort=True):
        subject_id = int(subject_id)
        tasks.append(
            {
                "frame": frame.to_dict(orient="list"),
                "core_prediction": str(
                    core / "subject_predictions" / f"subject_{subject_id}.npz"
                ),
                "dynamic_prediction": str(
                    dynamic / "subject_predictions" / f"subject_{subject_id}.npz"
                ),
                "output_prediction": str(
                    prediction_output / f"subject_{subject_id}.npz"
                ),
            }
        )
    payloads = []
    errors = []
    with ProcessPoolExecutor(
        max_workers=min(int(args.jobs), len(tasks)), mp_context=mp.get_context("fork")
    ) as executor:
        futures = {executor.submit(_fit_worker, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                payload = future.result()
                payloads.append(payload)
                print(
                    f"[fit] {completed}/{len(tasks)} s{payload['subject_id']} "
                    f"({payload['runtime_seconds']:.1f}s)",
                    flush=True,
                )
            except Exception as error:
                errors.append(
                    {
                        "output_prediction": task["output_prediction"],
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                )
    atomic_json(output / "worker_errors.json", errors)
    if errors:
        raise RuntimeError(f"{len(errors)} joint NR2 workers failed")

    metric_rows = [row for payload in payloads for row in payload.pop("metrics")]
    metrics = pd.DataFrame(metric_rows).sort_values(
        ["condition", "subject_id", "segment", "model"]
    )
    parameters = pd.DataFrame(payloads).sort_values(["condition", "subject_id"])
    comparison = comparisons(metrics, args.seed)
    atomic_csv(output / "subject_metrics.csv", metrics)
    atomic_csv(output / "parameters.csv", parameters)
    atomic_csv(output / "model_comparisons.csv", comparison)
    render_report(output, comparison, parameters)
    manifest = {
        "result_type": "unified_newplan_joint_dynamic_nr2_screen",
        "status": "complete",
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "core_run": str(core),
        "core_manifest_sha256": sha256_file(core / "manifest.json"),
        "dynamic_run": str(dynamic),
        "dynamic_manifest_sha256": sha256_file(dynamic / "manifest.json"),
        "n_subjects": len(payloads),
        "models": [
            "R0KT_GLOBAL",
            "NR2",
            "NR2T_GLOBAL_POSTHOC",
            "NR2T_JOINT_INDIVIDUAL",
        ],
        "nr2_parameter_scope": "individual learning rate, intercept, and practice slope fitted on training prefix",
        "rule_parameter_scope": "individual intercept plus one cross-condition shared practice slope",
        "jobs": int(args.jobs),
        "base_seed": int(args.seed),
        "runtime_seconds": float(time.time() - started),
        "evidence_scope": "subject-wise MAP temporal-holdout fairness screen; not model recovery",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy_version,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "module_sha256": sha256_file(
            ROOT / "src/Bayesian_state/utils/unified_newplan.py"
        ),
    }
    atomic_json(output / "manifest.json", manifest)
    print(f"[done] wrote {output} in {manifest['runtime_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
