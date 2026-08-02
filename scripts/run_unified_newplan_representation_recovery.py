#!/usr/bin/env python3
"""Generative model recovery for dynamic rule versus dynamic feature RL.

Both generators autonomously choose, receive task-contingent feedback, and
update their own states on the 96 real stimulus schedules.  Each synthetic
cohort is then re-fitted with R0KT_GLOBAL and the deliberately flexible
NR2T_JOINT_INDIVIDUAL using training trials only; identity is decided on the
last-block holdout.
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
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_unified_newplan_dynamic_readout import (  # noqa: E402
    EPS,
    fit_shared_readout,
    score,
)
from scripts.run_unified_newplan_joint_nr2 import (  # noqa: E402
    fit_joint_dynamic_nr2,
)
from scripts.run_unified_newplan_readout_recovery import (  # noqa: E402
    fit_static_readout,
    generate_rule_trajectory,
)
from src.Bayesian_state.utils.unified_newplan import (  # noqa: E402
    FEATURE_COLUMNS,
    build_partition,
    expected_feedback_from_category,
    feedback_compatible_categories,
    feedback_target_matrix,
    nr2_dynamic_readout_predictions,
    partition_prior,
    rule_predictions,
    stable_softmax,
)


DEFAULT_DATA = ROOT / "data/processed/Task2_processed.csv"
DEFAULT_CORE = ROOT / "results/zhuran/unified_newplan/core_sobol512_20260802"
DEFAULT_DYNAMIC = ROOT / "results/zhuran/unified_newplan/dynamic_readout_20260802"
DEFAULT_JOINT_NR2 = ROOT / "results/zhuran/unified_newplan/joint_dynamic_nr2_20260802"
DEFAULT_OUTPUT = ROOT / "results/zhuran/unified_newplan/representation_recovery_screen20_20260802"
MIN_IDENTIFICATION = 0.80


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--dynamic", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--joint-nr2", type=Path, default=DEFAULT_JOINT_NR2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--stage", choices=("screen", "final"), default="screen")
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


def binomial_interval(successes: int, trials: int) -> tuple[float, float]:
    low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return low, high


def generate_nr2_trajectory(
    stimuli: np.ndarray,
    true_categories: np.ndarray,
    condition: int,
    practice: np.ndarray,
    learning_rate: float,
    intercept: float,
    slope: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.column_stack([np.ones(len(stimuli)), stimuli - 0.5])
    n_categories = 2 if condition == 1 else 4
    weights = np.zeros((n_categories, features.shape[1]), dtype=float)
    choices = np.empty(len(stimuli), dtype=np.int64)
    feedback = np.empty(len(stimuli), dtype=float)
    for trial, feature in enumerate(features):
        log_kappa = np.clip(intercept + slope * practice[trial], math.log(0.01), math.log(20.0))
        probability = stable_softmax(math.exp(log_kappa) * (weights @ feature))
        choices[trial] = int(rng.choice(n_categories, p=probability))
        feedback[trial] = expected_feedback_from_category(
            condition,
            np.asarray([choices[trial] + 1]),
            np.asarray([true_categories[trial] + 1]),
        )[0]
        compatible = feedback_compatible_categories(
            condition, int(choices[trial]), float(feedback[trial])
        )
        target = np.zeros(n_categories, dtype=float)
        target[compatible] = 1.0 / len(compatible)
        weights += (
            learning_rate
            * np.outer(target - probability, feature)
            / max(1.0, float(feature @ feature))
        )
    return choices, feedback


def load_templates(task: dict[str, Any]) -> tuple[list[dict[str, Any]], float]:
    data = pd.read_csv(Path(task["data"]), low_memory=False).sort_values(
        ["condition", "iSub", "iSession", "iBlock", "iTrial"], kind="stable"
    )
    core = Path(task["core"])
    dynamic = Path(task["dynamic"])
    joint_nr2 = Path(task["joint_nr2"])
    rule_parameters = pd.read_csv(dynamic / "parameters.csv", dtype={"condition": str})
    rule_subject = rule_parameters[
        (rule_parameters["model"] == "R0KT_GLOBAL")
        & (rule_parameters["subject_id"] != -1)
    ].set_index("subject_id")
    rule_slope = float(
        rule_parameters[
            (rule_parameters["model"] == "R0KT_GLOBAL")
            & (rule_parameters["subject_id"] == -1)
        ]["slope"].iloc[0]
    )
    nr_parameters = pd.read_csv(joint_nr2 / "parameters.csv").set_index("subject_id")
    target_hypothesis = {1: 0, 2: 42, 3: 42}
    templates = []
    for subject_id, frame in data.groupby("iSub", sort=True):
        subject_id = int(subject_id)
        condition = int(frame["condition"].iloc[0])
        frame = frame.reset_index(drop=True)
        stimuli = frame[list(FEATURE_COLUMNS)].to_numpy(dtype=float)
        with np.load(core / "q_cache" / f"subject_{subject_id}.npz", allow_pickle=False) as archive:
            q_values = archive["q"].astype(np.float64)
        with np.load(core / "subject_predictions" / f"subject_{subject_id}.npz", allow_pickle=False) as archive:
            holdout = archive["holdout_mask"].astype(bool)
        train_rows = np.flatnonzero(~holdout)
        practice = np.arange(len(frame), dtype=float) / max(1, int(train_rows[-1]))
        partition = build_partition(condition)
        true_categories = partition._get_category_assignments_region(
            target_hypothesis[condition], stimuli
        ).astype(np.int64)
        if not np.allclose(
            expected_feedback_from_category(
                condition, frame["choice"].to_numpy(dtype=int), true_categories + 1
            ),
            frame["feedback"].to_numpy(dtype=float),
        ):
            raise ValueError(f"known task rule does not reproduce feedback for {subject_id}")
        templates.append(
            {
                "subject_id": subject_id,
                "condition": condition,
                "stimuli": stimuli,
                "q": q_values,
                "holdout": holdout,
                "practice": practice,
                "true_categories": true_categories,
                "prior": partition_prior(partition, "uniform_rule"),
                "rule_intercept": float(rule_subject.loc[subject_id, "intercept"]),
                "nr_learning_rate": float(nr_parameters.loc[subject_id, "learning_rate"]),
                "nr_intercept": float(nr_parameters.loc[subject_id, "intercept"]),
                "nr_slope": float(nr_parameters.loc[subject_id, "practice_slope"]),
            }
        )
    return templates, rule_slope


def _worker(task: dict[str, Any]) -> dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    started = time.time()
    templates, true_rule_slope = load_templates(task)
    generator = str(task["generator"])
    replicate = int(task["replicate"])
    seed_sequence = np.random.SeedSequence(
        [int(task["seed"]), replicate, 0 if generator == "R0KT_GLOBAL" else 1]
    )
    subject_seeds = seed_sequence.spawn(len(templates))
    synthetic_subjects = []
    nr_predictions = {}
    parameter_rows = []
    nr_failures = 0
    for template, subject_seed in zip(templates, subject_seeds):
        rng = np.random.default_rng(subject_seed)
        if generator == "R0KT_GLOBAL":
            choices, feedback = generate_rule_trajectory(
                template["q"],
                template["true_categories"],
                template["condition"],
                template["practice"],
                template["rule_intercept"],
                true_rule_slope,
                template["prior"],
                rng,
            )
        else:
            choices, feedback = generate_nr2_trajectory(
                template["stimuli"],
                template["true_categories"],
                template["condition"],
                template["practice"],
                template["nr_learning_rate"],
                template["nr_intercept"],
                template["nr_slope"],
                rng,
            )
        targets = feedback_target_matrix(template["condition"], choices, feedback)
        nr_probability, nr_fit = fit_joint_dynamic_nr2(
            template["stimuli"],
            choices,
            targets,
            ~template["holdout"],
            template["practice"],
        )
        nr_predictions[template["subject_id"]] = nr_probability
        nr_failures += int(not nr_fit["optimizer_success"])

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
        static_intercept, _, static_diagnostics = fit_static_readout(
            log_core, choices, ~template["holdout"]
        )
        synthetic_subjects.append(
            {
                "subject_id": template["subject_id"],
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
        parameter_rows.append(
            {
                "generator": generator,
                "replicate": replicate,
                "subject_id": template["subject_id"],
                "condition": template["condition"],
                "true_nr_learning_rate": template["nr_learning_rate"]
                if generator == "NR2T_JOINT_INDIVIDUAL"
                else np.nan,
                "fitted_nr_learning_rate": nr_fit["learning_rate"],
                "true_nr_intercept": template["nr_intercept"]
                if generator == "NR2T_JOINT_INDIVIDUAL"
                else np.nan,
                "fitted_nr_intercept": nr_fit["intercept"],
                "true_nr_slope": template["nr_slope"]
                if generator == "NR2T_JOINT_INDIVIDUAL"
                else np.nan,
                "fitted_nr_slope": nr_fit["practice_slope"],
                "nr_same_optimal_region": nr_fit["n_same_optimal_region"],
                "static_rule_optimizer_success": static_diagnostics["optimizer_success"],
            }
        )

    rule_fit = fit_shared_readout(synthetic_subjects, "R0", "global")
    metric_rows = []
    for subject in synthetic_subjects:
        subject_id = subject["subject_id"]
        for model, probability in (
            ("R0KT_GLOBAL", rule_fit["predictions"][subject_id]),
            ("NR2T_JOINT_INDIVIDUAL", nr_predictions[subject_id]),
        ):
            metric_rows.append(
                {
                    "subject_id": subject_id,
                    "condition": subject["condition"],
                    "model": model,
                    **score(probability, subject["choices"], subject["holdout"]),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    pivot = metrics.pivot(
        index=["subject_id", "condition"], columns="model", values="nll_per_trial"
    ).reset_index()
    pivot["delta_rule_vs_nr"] = (
        pivot["NR2T_JOINT_INDIVIDUAL"] - pivot["R0KT_GLOBAL"]
    )
    summary_rows = []
    for condition_label, group in [
        (str(condition), pivot[pivot["condition"] == condition])
        for condition in (1, 2, 3)
    ] + [("all", pivot)]:
        mean_delta = float(group["delta_rule_vs_nr"].mean())
        summary_rows.append(
            {
                "generator": generator,
                "replicate": replicate,
                "condition": condition_label,
                "mean_delta_nll_per_trial_rule_vs_nr": mean_delta,
                "median_delta_nll_per_trial_rule_vs_nr": float(
                    group["delta_rule_vs_nr"].median()
                ),
                "n_rule_improved": int((group["delta_rule_vs_nr"] > 0).sum()),
                "n_subjects": int(len(group)),
                "selected_model": (
                    "R0KT_GLOBAL" if mean_delta > 0 else "NR2T_JOINT_INDIVIDUAL"
                ),
                "true_rule_slope": true_rule_slope
                if generator == "R0KT_GLOBAL"
                else np.nan,
                "fitted_rule_slope": float(rule_fit["slopes"][0]),
            }
        )
    return {
        "generator": generator,
        "replicate": replicate,
        "summaries": summary_rows,
        "parameters": parameter_rows,
        "rule_optimizer_success": bool(rule_fit["optimizer_success"]),
        "rule_same_optimal_region": int(rule_fit["n_same_optimal_region"]),
        "nr_optimizer_failures": nr_failures,
        "runtime_seconds": float(time.time() - started),
    }


def render_report(output: Path, recovery: pd.DataFrame, summary: pd.DataFrame, parameters: pd.DataFrame) -> None:
    all_rows = recovery[recovery["condition"] == "all"]
    lines = [
        f"# Rule-versus-feature representation recovery ({summary.stage.iloc[0]}, {int(summary.n_replicates.iloc[0])} cohorts per branch)",
        "",
        "> Both models generated autonomous choices and contingent feedback, then were re-fitted on training prefixes and selected on final-block holdouts.",
        "",
        "## Model recovery",
        "",
        "| Generator | Correct | Rate | Exact 95% CI | Mean held-out Δ rule-vs-NR |",
        "|:--|:--|--:|:--|--:|",
    ]
    for row in summary.itertuples(index=False):
        generator_rows = all_rows[all_rows["generator"] == row.generator]
        lines.append(
            f"| {row.generator} | {int(row.n_correct)}/{int(row.n_replicates)} | "
            f"{row.identification_rate:.1%} | [{row.ci_low:.3f}, {row.ci_high:.3f}] | "
            f"{generator_rows.mean_delta_nll_per_trial_rule_vs_nr.mean():.6f} |"
        )
    nr = parameters[parameters["generator"] == "NR2T_JOINT_INDIVIDUAL"]
    lines.extend(
        [
            "",
            "## Parameter diagnostics",
            "",
            f"- NR2-generated learning-rate recovery: bias {(nr.fitted_nr_learning_rate-nr.true_nr_learning_rate).mean():.6f}, RMSE {np.sqrt(np.mean((nr.fitted_nr_learning_rate-nr.true_nr_learning_rate)**2)):.6f}, correlation {nr[['true_nr_learning_rate','fitted_nr_learning_rate']].corr().iloc[0,1]:.6f}.",
            f"- NR2-generated slope recovery: bias {(nr.fitted_nr_slope-nr.true_nr_slope).mean():.6f}, RMSE {np.sqrt(np.mean((nr.fitted_nr_slope-nr.true_nr_slope)**2)):.6f}, correlation {nr[['true_nr_slope','fitted_nr_slope']].corr().iloc[0,1]:.6f}.",
            "",
            "## Decision",
            "",
        ]
    )
    if bool(summary.passed.all()):
        lines.append(
            "Both representation identities clear the recovery requirement. The real-data R0KT advantage is therefore not explained by an inability of the fitting pipeline to recognize dynamic feature RL."
        )
    else:
        lines.append(
            "At least one generator does not clear the recovery requirement. Real-data rule-versus-feature differences cannot support a unique representation identity."
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Recovery is conditional on the fitted parameter ranges and the enumerated rule library. It establishes distinguishability of these two implemented models, not uniqueness among all possible cognitive processes. RT/oral external validation remains separate.",
            "",
            "## Artifacts",
            "",
            "- `model_recovery.csv`, `recovery_summary.csv`, `parameter_recovery.csv`.",
            "- `worker_manifest.csv`, `worker_errors.json`, `manifest.json`.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "data": args.data.resolve(),
        "core": args.core.resolve(),
        "dynamic": args.dynamic.resolve(),
        "joint_nr2": args.joint_nr2.resolve(),
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)

    # Compile dynamic NR2 before forking.
    nr2_dynamic_readout_predictions(
        np.zeros((2, 4)), np.full((2, 2), 0.5), 0.1, -1.0, 1.0, np.asarray([0.0, 1.0])
    )
    tasks = [
        {
            **{key: str(value) for key, value in paths.items()},
            "generator": generator,
            "replicate": replicate,
            "seed": int(args.seed),
        }
        for generator in ("R0KT_GLOBAL", "NR2T_JOINT_INDIVIDUAL")
        for replicate in range(int(args.replicates))
    ]
    payloads = []
    errors = []
    with ProcessPoolExecutor(
        max_workers=min(int(args.jobs), len(tasks)), mp_context=mp.get_context("fork")
    ) as executor:
        futures = {executor.submit(_worker, task): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                payload = future.result()
                payloads.append(payload)
                print(
                    f"[recovery] {completed}/{len(tasks)} {payload['generator']} "
                    f"r{payload['replicate']:03d} ({payload['runtime_seconds']:.1f}s)",
                    flush=True,
                )
            except Exception as error:
                errors.append({**task, "error": repr(error), "traceback": traceback.format_exc()})
                print(f"[error] {task['generator']} r{task['replicate']}: {error}", flush=True)
    atomic_json(output / "worker_errors.json", errors)
    if errors:
        raise RuntimeError(f"{len(errors)} recovery workers failed")

    recovery = pd.DataFrame([row for payload in payloads for row in payload["summaries"]]).sort_values(
        ["generator", "replicate", "condition"]
    )
    parameters = pd.DataFrame([row for payload in payloads for row in payload["parameters"]]).sort_values(
        ["generator", "replicate", "condition", "subject_id"]
    )
    workers = pd.DataFrame(
        [
            {key: value for key, value in payload.items() if key not in {"summaries", "parameters"}}
            for payload in payloads
        ]
    ).sort_values(["generator", "replicate"])
    summary_rows = []
    all_rows = recovery[recovery["condition"] == "all"]
    for generator, group in all_rows.groupby("generator"):
        correct = int((group["selected_model"] == generator).sum())
        low, high = binomial_interval(correct, len(group))
        point_pass = correct / len(group) >= MIN_IDENTIFICATION
        interval_pass = high >= MIN_IDENTIFICATION if args.stage == "screen" else low >= MIN_IDENTIFICATION
        optimizer_pass = bool(
            workers[workers["generator"] == generator]["rule_optimizer_success"].all()
            and (workers[workers["generator"] == generator]["nr_optimizer_failures"] == 0).all()
        )
        summary_rows.append(
            {
                "generator": generator,
                "stage": args.stage,
                "n_replicates": int(len(group)),
                "n_correct": correct,
                "identification_rate": correct / len(group),
                "ci_low": low,
                "ci_high": high,
                "point_pass": point_pass,
                "interval_pass": interval_pass,
                "optimizer_pass": optimizer_pass,
                "passed": bool(point_pass and interval_pass and optimizer_pass),
            }
        )
    summary = pd.DataFrame(summary_rows)
    atomic_csv(output / "model_recovery.csv", recovery)
    atomic_csv(output / "parameter_recovery.csv", parameters)
    atomic_csv(output / "worker_manifest.csv", workers)
    atomic_csv(output / "recovery_summary.csv", summary)
    render_report(output, recovery, summary, parameters)
    manifest = {
        "result_type": "unified_newplan_representation_model_recovery",
        "status": "complete",
        "stage": args.stage,
        "replicates_per_generator": int(args.replicates),
        "generators": ["R0KT_GLOBAL", "NR2T_JOINT_INDIVIDUAL"],
        "minimum_identification": MIN_IDENTIFICATION,
        "selection_metric": "sign of mean subject heldout Delta NLL/trial (NR2 minus rule)",
        "base_seed": int(args.seed),
        "jobs": int(args.jobs),
        "runtime_seconds": float(time.time() - started),
        "paths": {key: str(value) for key, value in paths.items()},
        "input_hashes": {
            "data": sha256_file(paths["data"]),
            "core_manifest": sha256_file(paths["core"] / "manifest.json"),
            "dynamic_manifest": sha256_file(paths["dynamic"] / "manifest.json"),
            "joint_nr2_manifest": sha256_file(paths["joint_nr2"] / "manifest.json"),
        },
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
