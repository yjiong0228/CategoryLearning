#!/usr/bin/env python3
"""Twenty-cohort recovery screen for static versus practice-dynamic readout.

Each synthetic cohort preserves the 96 real stimulus sequences, trial counts,
Task-1b-integrated q values, and temporal holdout boundaries.  Choices and
feedback are generated autonomously from either R0K (static kappa) or
R0KT_GLOBAL (one shared practice slope plus subject intercepts).  Both models
are then re-fitted using synthetic training trials only and selected by the
synthetic final-block holdout.
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
from scipy.special import logsumexp
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_unified_newplan_dynamic_readout import (  # noqa: E402
    EPS,
    LOG_KAPPA_MAX,
    LOG_KAPPA_MIN,
    fit_shared_readout,
    probabilities_from_readout,
    score,
)
from src.Bayesian_state.utils.unified_newplan import (  # noqa: E402
    FEATURE_COLUMNS,
    build_partition,
    expected_feedback_from_category,
    feedback_compatible_categories,
    partition_prior,
    rule_predictions,
    stable_softmax,
)


DEFAULT_CORE = ROOT / "results/zhuran/unified_newplan/core_sobol512_20260802"
DEFAULT_DYNAMIC = ROOT / "results/zhuran/unified_newplan/dynamic_readout_20260802"
DEFAULT_DATA = ROOT / "data/processed/Task2_processed.csv"
DEFAULT_OUTPUT = ROOT / "results/zhuran/unified_newplan/readout_recovery_screen20_20260802"
N_SCREEN_REPLICATES = 20
SCREEN_MIN_IDENTIFICATION = 0.80
SCREEN_MAX_RELATIVE_SLOPE_BIAS = 0.10
FINAL_DECISION_THRESHOLD = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--dynamic", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=N_SCREEN_REPLICATES)
    parser.add_argument("--jobs", type=int, default=N_SCREEN_REPLICATES)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--stage", choices=("screen", "final"), default="screen")
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=FINAL_DECISION_THRESHOLD,
        help="Select the dynamic model only above this mean held-out Delta NLL/trial.",
    )
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


def binomial_interval(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def fit_static_readout(
    log_probability: np.ndarray,
    choices: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, Any]]:
    x = np.zeros(len(choices), dtype=float)
    rows = np.flatnonzero(train_mask)
    train_log_probability = log_probability[rows]
    train_choices = choices[rows]

    def objective_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = float(values[0])
        kappa = math.exp(intercept)
        logits = kappa * train_log_probability
        normalizer = logsumexp(logits, axis=1)
        observed = train_log_probability[np.arange(len(rows)), train_choices]
        loss = float(np.sum(-kappa * observed + normalizer))
        calibrated = np.exp(logits - normalizer[:, None])
        derivative = kappa * (
            -observed + np.sum(calibrated * train_log_probability, axis=1)
        )
        return loss, np.asarray([np.sum(derivative)])

    fits = [
        minimize(
            lambda values: objective_gradient(values)[0],
            np.asarray([start]),
            jac=lambda values: objective_gradient(values)[1],
            method="L-BFGS-B",
            bounds=[(LOG_KAPPA_MIN, LOG_KAPPA_MAX)],
        )
        for start in (math.log(0.03), math.log(0.1), math.log(0.5), math.log(2.0))
    ]
    best = min(fits, key=lambda result: float(result.fun))
    probabilities, _, _ = probabilities_from_readout(
        log_probability, x, float(best.x[0]), 0.0
    )
    return float(best.x[0]), probabilities, {
        "optimizer_success": bool(best.success),
        "n_same_optimal_region": int(
            sum(abs(float(result.fun) - float(best.fun)) <= 1e-5 for result in fits)
        ),
        "train_nll": float(best.fun),
    }


def generate_rule_trajectory(
    q_values: np.ndarray,
    true_categories: np.ndarray,
    condition: int,
    practice: np.ndarray,
    intercept: float,
    slope: float,
    prior: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_trials, n_hypotheses, n_categories = q_values.shape
    choices = np.empty(n_trials, dtype=np.int64)
    feedback = np.empty(n_trials, dtype=float)
    log_prior = np.log(prior)
    evidence_plus = np.zeros(n_hypotheses, dtype=float)
    for trial in range(n_trials):
        evidence_minus = np.zeros(n_hypotheses) if trial == 0 else evidence_plus
        belief = stable_softmax(log_prior + evidence_minus)
        core_probability = np.maximum(belief @ q_values[trial], EPS)
        core_probability /= core_probability.sum()
        log_kappa = np.clip(
            float(intercept) + float(slope) * float(practice[trial]),
            LOG_KAPPA_MIN,
            LOG_KAPPA_MAX,
        )
        probability = stable_softmax(math.exp(log_kappa) * np.log(core_probability))
        choices[trial] = int(rng.choice(n_categories, p=probability))
        feedback[trial] = expected_feedback_from_category(
            condition,
            np.asarray([choices[trial] + 1]),
            np.asarray([true_categories[trial] + 1]),
        )[0]
        compatible = feedback_compatible_categories(
            condition, int(choices[trial]), float(feedback[trial])
        )
        likelihood = q_values[trial][:, compatible].sum(axis=1)
        evidence_plus = evidence_minus + np.log(np.clip(likelihood, EPS, 1.0))
    return choices, feedback


def load_template(task: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    core = Path(task["core"])
    dynamic = Path(task["dynamic"])
    data = pd.read_csv(Path(task["data"]), low_memory=False).sort_values(
        ["condition", "iSub", "iSession", "iBlock", "iTrial"], kind="stable"
    )
    parameter_frame = pd.read_csv(dynamic / "parameters.csv", dtype={"condition": str})
    dynamic_subject = parameter_frame[
        (parameter_frame["model"] == "R0KT_GLOBAL")
        & (parameter_frame["subject_id"] != -1)
    ].set_index("subject_id")
    true_slope = float(
        parameter_frame[
            (parameter_frame["model"] == "R0KT_GLOBAL")
            & (parameter_frame["subject_id"] == -1)
        ]["slope"].iloc[0]
    )
    templates = []
    target_hypothesis = {1: 0, 2: 42, 3: 42}
    for subject_id, frame in data.groupby("iSub", sort=True):
        subject_id = int(subject_id)
        condition = int(frame["condition"].iloc[0])
        frame = frame.reset_index(drop=True)
        with np.load(core / "q_cache" / f"subject_{subject_id}.npz", allow_pickle=False) as archive:
            q_values = archive["q"].astype(np.float64)
        with np.load(
            core / "subject_predictions" / f"subject_{subject_id}.npz",
            allow_pickle=False,
        ) as archive:
            holdout = archive["holdout_mask"].astype(bool)
            core_parameters = json.loads(str(archive["parameters_json"].item()))
        train_rows = np.flatnonzero(~holdout)
        practice = np.arange(len(frame), dtype=float) / max(1, int(train_rows[-1]))
        partition = build_partition(condition)
        true_categories = partition._get_category_assignments_region(
            target_hypothesis[condition],
            frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float),
        ).astype(np.int64)
        implied_feedback = expected_feedback_from_category(
            condition,
            frame["choice"].to_numpy(dtype=int),
            true_categories + 1,
        )
        if not np.allclose(implied_feedback, frame["feedback"].to_numpy(dtype=float)):
            raise ValueError(
                f"target-rule categories do not reproduce delivered feedback for subject {subject_id}"
            )
        templates.append(
            {
                "subject_id": subject_id,
                "condition": condition,
                "q": q_values,
                "true_categories": true_categories,
                "holdout": holdout,
                "practice": practice,
                "prior": partition_prior(partition, "uniform_rule"),
                "static_intercept": math.log(core_parameters["R0K"]["sensitivity"]),
                "dynamic_intercept": float(dynamic_subject.loc[subject_id, "intercept"]),
            }
        )
    return templates, true_slope


def _recovery_worker(task: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    started = time.time()
    templates, true_dynamic_slope = load_template(task)
    generator = str(task["generator"])
    replicate = int(task["replicate"])
    seed_sequence = np.random.SeedSequence(
        [int(task["seed"]), replicate, 1 if generator == "R0KT_GLOBAL" else 0]
    )
    subject_seeds = seed_sequence.spawn(len(templates))
    synthetic_subjects = []
    truth_rows = []
    static_predictions = {}
    static_diagnostics = []
    for template, subject_seed in zip(templates, subject_seeds):
        true_intercept = (
            template["dynamic_intercept"]
            if generator == "R0KT_GLOBAL"
            else template["static_intercept"]
        )
        true_slope = true_dynamic_slope if generator == "R0KT_GLOBAL" else 0.0
        choices, feedback = generate_rule_trajectory(
            template["q"],
            template["true_categories"],
            template["condition"],
            template["practice"],
            true_intercept,
            true_slope,
            template["prior"],
            np.random.default_rng(subject_seed),
        )
        core_probability = rule_predictions(
            template["q"],
            choices,
            feedback,
            template["condition"],
            retention=1.0,
            sensitivity=1.0,
            prior=template["prior"],
        ).probabilities
        log_core = np.log(np.clip(core_probability, EPS, 1.0))
        static_intercept, static_probability, diagnostics = fit_static_readout(
            log_core, choices, ~template["holdout"]
        )
        subject_id = template["subject_id"]
        static_predictions[subject_id] = static_probability
        static_diagnostics.append(diagnostics)
        synthetic_subjects.append(
            {
                "subject_id": subject_id,
                "condition": template["condition"],
                "choices": choices,
                "feedback": feedback,
                "category": template["true_categories"],
                "holdout": template["holdout"],
                "practice": template["practice"],
                "log_R0": log_core,
                "parameters": {"R0K": {"sensitivity": math.exp(static_intercept)}},
            }
        )
        truth_rows.append(
            {
                "generator": generator,
                "replicate": replicate,
                "subject_id": subject_id,
                "condition": template["condition"],
                "true_intercept": float(true_intercept),
                "static_fitted_intercept": float(static_intercept),
            }
        )

    dynamic_fit = fit_shared_readout(synthetic_subjects, "R0", "global")
    fitted_slope = float(dynamic_fit["slopes"][0])
    metric_rows = []
    for index, subject in enumerate(synthetic_subjects):
        subject_id = subject["subject_id"]
        for model, probability in (
            ("R0K", static_predictions[subject_id]),
            ("R0KT_GLOBAL", dynamic_fit["predictions"][subject_id]),
        ):
            metric_rows.append(
                {
                    "generator": generator,
                    "replicate": replicate,
                    "subject_id": subject_id,
                    "condition": subject["condition"],
                    "model": model,
                    **score(probability, subject["choices"], subject["holdout"]),
                }
            )
        truth_rows[index]["dynamic_fitted_intercept"] = float(
            dynamic_fit["intercepts"][index]
        )
        truth_rows[index]["true_slope"] = float(
            true_dynamic_slope if generator == "R0KT_GLOBAL" else 0.0
        )
        truth_rows[index]["fitted_slope"] = fitted_slope
        truth_rows[index]["dynamic_fitted_intercept_se"] = float(
            dynamic_fit["standard_errors"][index]
        )
        truth_rows[index]["dynamic_intercept_wald95_covered"] = bool(
            abs(
                truth_rows[index]["dynamic_fitted_intercept"]
                - truth_rows[index]["true_intercept"]
            )
            <= 1.96 * truth_rows[index]["dynamic_fitted_intercept_se"]
        )

    fitted_slope_se = float(dynamic_fit["standard_errors"][len(synthetic_subjects)])
    generating_slope = float(
        true_dynamic_slope if generator == "R0KT_GLOBAL" else 0.0
    )

    metrics = pd.DataFrame(metric_rows)
    pivot = metrics.pivot(
        index=["subject_id", "condition"], columns="model", values="nll_per_trial"
    ).reset_index()
    pivot["delta_nll_per_trial"] = pivot["R0K"] - pivot["R0KT_GLOBAL"]
    summary_rows = []
    for condition_label, group in [
        (str(condition), pivot[pivot["condition"] == condition])
        for condition in (1, 2, 3)
    ] + [("all", pivot)]:
        summary_rows.append(
            {
                "generator": generator,
                "replicate": replicate,
                "condition": condition_label,
                "mean_delta_nll_per_trial": float(group["delta_nll_per_trial"].mean()),
                "median_delta_nll_per_trial": float(group["delta_nll_per_trial"].median()),
                "n_dynamic_improved": int((group["delta_nll_per_trial"] > 0).sum()),
                "n_subjects": int(len(group)),
                "selected_model": (
                    "R0KT_GLOBAL"
                    if float(group["delta_nll_per_trial"].mean())
                    > float(task["decision_threshold"])
                    else "R0K"
                ),
                "decision_threshold": float(task["decision_threshold"]),
                "true_slope": generating_slope,
                "fitted_slope": fitted_slope,
                "fitted_slope_se": fitted_slope_se,
                "slope_wald95_covered": bool(
                    abs(fitted_slope - generating_slope) <= 1.96 * fitted_slope_se
                ),
            }
        )
    return {
        "generator": generator,
        "replicate": replicate,
        "summaries": summary_rows,
        "parameters": truth_rows,
        "dynamic_optimizer_success": bool(dynamic_fit["optimizer_success"]),
        "dynamic_same_region": int(dynamic_fit["n_same_optimal_region"]),
        "n_static_optimizer_failures": int(
            sum(not diagnostics["optimizer_success"] for diagnostics in static_diagnostics)
        ),
        "runtime_seconds": float(time.time() - started),
    }


def render_report(
    output: Path,
    recovery: pd.DataFrame,
    parameters: pd.DataFrame,
    screen: pd.DataFrame,
) -> None:
    all_rows = recovery[recovery["condition"] == "all"]
    dynamic = all_rows[all_rows["generator"] == "R0KT_GLOBAL"]
    static = all_rows[all_rows["generator"] == "R0K"]
    dynamic_identified = int((dynamic["selected_model"] == "R0KT_GLOBAL").sum())
    static_identified = int((static["selected_model"] == "R0K").sum())
    n_dynamic = len(dynamic)
    n_static = len(static)
    sensitivity_interval = binomial_interval(dynamic_identified, n_dynamic)
    specificity_interval = binomial_interval(static_identified, n_static)
    slope_rows = screen[screen["generator"] == "R0KT_GLOBAL"].iloc[0]
    static_slope_rows = screen[screen["generator"] == "R0K"].iloc[0]
    individual_dynamic = parameters[parameters["generator"] == "R0KT_GLOBAL"]
    individual_static = parameters[parameters["generator"] == "R0K"]

    lines = [
        f"# Dynamic-readout {screen['stage'].iloc[0]} recovery ({n_dynamic} cohorts per branch)",
        "",
        "> Autonomous synthetic choices and feedback on all 96 real stimulus schedules. Intervals are observed-Hessian/Wald diagnostics, not a hierarchical posterior.",
        "",
        "## Frozen decision rule",
        "",
        f"- Select R0KT_GLOBAL only when its mean held-out improvement exceeds {float(dynamic.decision_threshold.iloc[0]):.3f} NLL/trial; smaller fluctuations are assigned to nested R0K.",
        f"- Point identification at least {SCREEN_MIN_IDENTIFICATION:.0%} for both R0KT_GLOBAL sensitivity and R0K specificity.",
        f"- Absolute dynamic-slope bias below {SCREEN_MAX_RELATIVE_SLOPE_BIAS:.0%} of the generating slope.",
        "- No optimizer failures. In the final 100-cohort run, the exact binomial interval must also lie above the identification threshold.",
        "",
        "## Model recovery",
        "",
        f"- R0KT_GLOBAL generated: recovered {dynamic_identified}/{n_dynamic} ({dynamic_identified/n_dynamic:.1%}; exact 95% CI [{sensitivity_interval[0]:.3f}, {sensitivity_interval[1]:.3f}]).",
        f"- R0K generated: recovered {static_identified}/{n_static} ({static_identified/n_static:.1%}; exact 95% CI [{specificity_interval[0]:.3f}, {specificity_interval[1]:.3f}]).",
        "",
        "## Parameter recovery",
        "",
        f"- Dynamic shared slope: true {slope_rows.true_slope:.6f}; mean recovered {slope_rows.mean_fitted_slope:.6f}; bias {slope_rows.slope_bias:.6f}; RMSE {slope_rows.slope_rmse:.6f}.",
        f"- Dynamic shared-slope 95% Wald coverage: {slope_rows.slope_wald95_coverage:.1%}; static-branch zero-slope coverage: {static_slope_rows.slope_wald95_coverage:.1%}.",
        f"- Static branch fitted slope (true zero): mean {static_slope_rows.mean_fitted_slope:.6f}; RMSE {static_slope_rows.slope_rmse:.6f}.",
        f"- Dynamic subject intercepts: pooled bias {(individual_dynamic.dynamic_fitted_intercept-individual_dynamic.true_intercept).mean():.6f}, RMSE {np.sqrt(np.mean((individual_dynamic.dynamic_fitted_intercept-individual_dynamic.true_intercept)**2)):.6f}, correlation {individual_dynamic[['true_intercept','dynamic_fitted_intercept']].corr().iloc[0,1]:.6f}.",
        f"- Static subject intercepts: pooled bias {(individual_static.static_fitted_intercept-individual_static.true_intercept).mean():.6f}, RMSE {np.sqrt(np.mean((individual_static.static_fitted_intercept-individual_static.true_intercept)**2)):.6f}, correlation {individual_static[['true_intercept','static_fitted_intercept']].corr().iloc[0,1]:.6f}.",
        "",
        "## Decision",
        "",
    ]
    passed = bool(screen["screen_passed"].all())
    if passed:
        lines.append(
            "The candidate passes this static-versus-dynamic recovery stage. This establishes readout identifiability within the rule family, but a joint feature-RL generative comparison and external validation are still required for a rule-representation claim."
        )
    else:
        failed = ", ".join(screen.loc[~screen["screen_passed"], "generator"])
        lines.append(
            f"The candidate does not pass the frozen screening rule ({failed}). It must be simplified or rejected rather than advanced by adding complexity."
        )
    lines.extend(
        [
            "",
            "## Data boundary",
            "",
            "Synthetic feedback uses category assignments generated directly by the known task rule (h0 in condition 1; h42 in conditions 2/3). These assignments reproduce 100% of recorded feedback, including condition-3 subject 319. This avoids propagating that subject's corrupted session-5 category column into autonomous generation.",
            "",
            "## Artifacts",
            "",
            "- `model_recovery.csv`: per-replicate and per-condition held-out identification.",
            "- `parameter_recovery.csv`, `recovery_summary.csv`: shared-slope and subject-intercept recovery.",
            "- `worker_manifest.csv`, `worker_errors.json`: reproducibility and failure audit.",
            "- `manifest.json`: generators, thresholds, seeds, hashes, and scope.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    core = args.core.resolve()
    dynamic = args.dynamic.resolve()
    data = args.data.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for path in (core / "manifest.json", dynamic / "manifest.json", data):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.replicates <= 0:
        raise ValueError("replicates must be positive")
    if args.decision_threshold < 0:
        raise ValueError("decision threshold must be non-negative")

    tasks = [
        {
            "core": str(core),
            "dynamic": str(dynamic),
            "data": str(data),
            "generator": generator,
            "replicate": replicate,
            "seed": int(args.seed),
            "decision_threshold": float(args.decision_threshold),
        }
        for generator in ("R0K", "R0KT_GLOBAL")
        for replicate in range(int(args.replicates))
    ]
    payloads = []
    errors = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=min(int(args.jobs), len(tasks)), mp_context=context
    ) as executor:
        futures = {executor.submit(_recovery_worker, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                payload = future.result()
                payloads.append(payload)
                print(
                    f"[recovery] {completed}/{len(tasks)} {payload['generator']} "
                    f"r{payload['replicate']:02d} ({payload['runtime_seconds']:.1f}s)",
                    flush=True,
                )
            except Exception as error:
                errors.append(
                    {
                        **task,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(
                    f"[error] {task['generator']} r{task['replicate']:02d}: {error}",
                    flush=True,
                )
    atomic_json(output / "worker_errors.json", errors)
    if errors:
        raise RuntimeError(f"{len(errors)} recovery workers failed")

    model_recovery = pd.DataFrame(
        [row for payload in payloads for row in payload["summaries"]]
    ).sort_values(["generator", "replicate", "condition"])
    parameter_recovery = pd.DataFrame(
        [row for payload in payloads for row in payload["parameters"]]
    ).sort_values(["generator", "replicate", "condition", "subject_id"])
    worker_manifest = pd.DataFrame(
        [
            {
                key: value
                for key, value in payload.items()
                if key not in {"summaries", "parameters"}
            }
            for payload in payloads
        ]
    ).sort_values(["generator", "replicate"])

    recovery_rows = []
    all_rows = model_recovery[model_recovery["condition"] == "all"]
    for generator, group in all_rows.groupby("generator"):
        expected = generator
        successes = int((group["selected_model"] == expected).sum())
        interval = binomial_interval(successes, len(group))
        slope_error = group["fitted_slope"] - group["true_slope"]
        true_slope = float(group["true_slope"].iloc[0])
        relative_bias = (
            abs(float(slope_error.mean())) / abs(true_slope)
            if not np.isclose(true_slope, 0.0)
            else float("nan")
        )
        point_pass = successes / len(group) >= SCREEN_MIN_IDENTIFICATION
        interval_pass = (
            interval[0] >= SCREEN_MIN_IDENTIFICATION
            if args.stage == "final"
            else True
        )
        model_pass = bool(point_pass and interval_pass)
        slope_pass = (
            relative_bias <= SCREEN_MAX_RELATIVE_SLOPE_BIAS
            if generator == "R0KT_GLOBAL"
            else True
        )
        optimizer_pass = bool(
            worker_manifest[worker_manifest["generator"] == generator][
                "dynamic_optimizer_success"
            ].all()
            and (
                worker_manifest[worker_manifest["generator"] == generator][
                    "n_static_optimizer_failures"
                ]
                == 0
            ).all()
        )
        recovery_rows.append(
            {
                "generator": generator,
                "n_replicates": int(len(group)),
                "n_correctly_identified": successes,
                "identification_rate": successes / len(group),
                "identification_ci_low": interval[0],
                "identification_ci_high": interval[1],
                "decision_threshold": float(args.decision_threshold),
                "true_slope": true_slope,
                "mean_fitted_slope": float(group["fitted_slope"].mean()),
                "slope_bias": float(slope_error.mean()),
                "slope_rmse": float(np.sqrt(np.mean(slope_error**2))),
                "slope_wald95_coverage": float(group["slope_wald95_covered"].mean()),
                "relative_abs_slope_bias": relative_bias,
                "model_recovery_pass": model_pass,
                "slope_recovery_pass": slope_pass,
                "optimizer_pass": optimizer_pass,
                "screen_passed": bool(model_pass and slope_pass and optimizer_pass),
                "stage": args.stage,
            }
        )
    recovery_summary = pd.DataFrame(recovery_rows)

    atomic_csv(output / "model_recovery.csv", model_recovery)
    atomic_csv(output / "parameter_recovery.csv", parameter_recovery)
    atomic_csv(output / "recovery_summary.csv", recovery_summary)
    atomic_csv(output / "worker_manifest.csv", worker_manifest)
    render_report(output, model_recovery, parameter_recovery, recovery_summary)
    manifest = {
        "result_type": "unified_newplan_dynamic_readout_recovery_screen",
        "status": "complete",
        "screen_or_final": args.stage,
        "replicates_per_generator": int(args.replicates),
        "generators": ["R0K", "R0KT_GLOBAL"],
        "selection_metric": "mean subject heldout delta NLL per trial",
        "screen_thresholds": {
            "minimum_point_identification": SCREEN_MIN_IDENTIFICATION,
            "maximum_relative_dynamic_slope_bias": SCREEN_MAX_RELATIVE_SLOPE_BIAS,
            "optimizer_failures_allowed": 0,
            "mean_delta_nll_per_trial_decision_threshold": float(args.decision_threshold),
            "final_requires_exact_ci_above_identification_threshold": True,
        },
        "core_run": str(core),
        "core_manifest_sha256": sha256_file(core / "manifest.json"),
        "dynamic_run": str(dynamic),
        "dynamic_manifest_sha256": sha256_file(dynamic / "manifest.json"),
        "data_path": str(data),
        "data_sha256": sha256_file(data),
        "base_seed": int(args.seed),
        "jobs": int(args.jobs),
        "runtime_seconds": float(time.time() - started),
        "synthetic_category_source": "known task rule h0/h42; verified against 100% of recorded feedback",
        "evidence_scope": (
            "MAP model recovery with observed-Hessian Wald coverage; "
            "no hierarchical posterior and no NR2 generative recovery"
        ),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy_version,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(output / "manifest.json", manifest)
    print(f"[done] wrote {output} in {manifest['runtime_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
