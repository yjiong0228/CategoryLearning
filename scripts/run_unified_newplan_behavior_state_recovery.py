#!/usr/bin/env python3
"""Conditional recovery audit for the behavior-anchored choice-state filter.

Recovery conditions on each observed stimulus sequence and the frozen feature-RL
emission trajectory.  It tests whether the structured filter can recover known
simulated strategy states and whether held-out model selection distinguishes
filter-generated choices from feature-RL-generated choices.  It is not a full
autonomous recovery of the underlying feature learner.
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_unified_newplan_behavior_anchored_state import (  # noqa: E402
    concatenate_training,
    fit_condition,
    load_subjects,
    run_filter,
    score,
)


BASE = ROOT / "results/zhuran/unified_newplan"
DEFAULT_DATA = ROOT / "data/processed/Task2_processed.csv"
DEFAULT_CORE = BASE / "core_sobol512_20260802"
DEFAULT_DYNAMIC = BASE / "dynamic_readout_20260802"
DEFAULT_JOINT = BASE / "joint_dynamic_nr2_20260802"
DEFAULT_STATE = BASE / "behavior_anchored_state_20260802"
DEFAULT_OUTPUT = BASE / "behavior_state_recovery_20260802"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--dynamic", type=Path, default=DEFAULT_DYNAMIC)
    parser.add_argument("--joint", type=Path, default=DEFAULT_JOINT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--replicates", type=int, default=20)
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


def simulate_filter_subject(
    subject: dict[str, Any],
    parameters: pd.Series,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], np.ndarray]:
    n_trials, n_hypotheses, n_categories = subject["q"].shape
    base = np.concatenate(
        [
            [parameters.base_guess_mass, parameters.base_feature_mass],
            np.full(n_hypotheses, parameters.base_rule_mass / n_hypotheses),
        ]
    )
    persistence = float(parameters.state_persistence)
    state = np.empty(n_trials, dtype=np.int16)
    choices = np.empty(n_trials, dtype=np.int8)
    for trial in range(n_trials):
        if trial == 0 or rng.random() > persistence:
            state[trial] = int(rng.choice(n_hypotheses + 2, p=base))
        else:
            state[trial] = state[trial - 1]
        if state[trial] == 0:
            probability = np.full(n_categories, 1.0 / n_categories)
        elif state[trial] == 1:
            probability = subject["probability_feature"][trial]
        else:
            probability = subject["q"][trial, state[trial] - 2]
        probability = np.maximum(probability, 1e-12)
        probability /= probability.sum()
        choices[trial] = int(rng.choice(n_categories, p=probability))
    simulated = dict(subject)
    simulated["choices"] = choices.astype(np.int64)
    return simulated, state


def simulate_feature_subject(
    subject: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    probability = subject["probability_feature"]
    choices = np.asarray(
        [rng.choice(probability.shape[1], p=row / row.sum()) for row in probability],
        dtype=np.int64,
    )
    simulated = dict(subject)
    simulated["choices"] = choices
    return simulated


def fit_all_conditions(
    subjects: list[dict[str, Any]],
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    values: dict[int, np.ndarray] = {}
    rows = []
    for condition in (1, 2, 3):
        fitted, diagnostics = fit_condition(
            concatenate_training(subjects, condition), include_oral=False
        )
        values[condition] = fitted
        rows.append({"condition": condition, **diagnostics})
    return values, pd.DataFrame(rows)


def state_recovery_rows(
    subjects: list[dict[str, Any]],
    latent_states: dict[int, np.ndarray],
    fitted: dict[int, np.ndarray],
    replicate: int,
) -> list[dict[str, Any]]:
    rows = []
    for subject in subjects:
        subject_id = int(subject["subject_id"])
        condition = int(subject["condition"])
        latent = latent_states[subject_id]
        result = run_filter(subject, fitted[condition], include_oral=False)
        posterior = result["post_choice_state"]
        decoded = np.argmax(posterior, axis=1)
        true_mode = np.where(latent == 0, 0, np.where(latent == 1, 1, 2))
        rule_probability = posterior[:, 2:].sum(axis=1)
        decoded_mode = np.argmax(
            np.column_stack([posterior[:, 0], posterior[:, 1], rule_probability]),
            axis=1,
        )
        true_rule = true_mode == 2
        rows.append(
            {
                "replicate": replicate,
                "subject_id": subject_id,
                "condition": condition,
                "n_trials": int(len(latent)),
                "exact_state_accuracy": float(np.mean(decoded == latent)),
                "mode_accuracy": float(np.mean(decoded_mode == true_mode)),
                "mean_probability_true_state": float(
                    posterior[np.arange(len(latent)), latent].mean()
                ),
                "rule_probability_when_true_rule": float(
                    rule_probability[true_rule].mean()
                )
                if true_rule.any()
                else np.nan,
                "rule_probability_when_nonrule": float(
                    rule_probability[~true_rule].mean()
                )
                if (~true_rule).any()
                else np.nan,
                "n_true_rule_trials": int(true_rule.sum()),
            }
        )
    return rows


def model_recovery_rows(
    subjects: list[dict[str, Any]],
    fitted: dict[int, np.ndarray],
    generator: str,
    replicate: int,
) -> list[dict[str, Any]]:
    subject_delta = []
    for subject in subjects:
        condition = int(subject["condition"])
        result = run_filter(subject, fitted[condition], include_oral=False)
        holdout = subject["holdout"]
        choices = subject["choices"]
        filter_nll = score(
            result["predictive_probability"], choices, holdout
        )["nll_per_trial"]
        feature_nll = score(
            subject["probability_feature"], choices, holdout
        )["nll_per_trial"]
        subject_delta.append(
            {
                "condition": condition,
                "delta_nll_filter_advantage": feature_nll - filter_nll,
            }
        )
    subject_delta_frame = pd.DataFrame(subject_delta)
    rows = []
    for condition_label, group in [
        (str(condition), subject_delta_frame[subject_delta_frame["condition"].eq(condition)])
        for condition in (1, 2, 3)
    ] + [("all", subject_delta_frame)]:
        delta = group["delta_nll_filter_advantage"].to_numpy(dtype=float)
        rows.append(
            {
                "replicate": replicate,
                "generator": generator,
                "condition": condition_label,
                "mean_delta_nll_filter_advantage": float(delta.mean()),
                "median_delta_nll_filter_advantage": float(np.median(delta)),
                "n_filter_improved": int((delta > 0).sum()),
                "n_subjects": int(len(delta)),
            }
        )
    return rows


def render_report(
    output: Path,
    state_recovery: pd.DataFrame,
    model_recovery: pd.DataFrame,
    parameter_recovery: pd.DataFrame,
) -> None:
    state_all = state_recovery.groupby("replicate", as_index=False).agg(
        exact_state_accuracy=("exact_state_accuracy", "mean"),
        mode_accuracy=("mode_accuracy", "mean"),
        mean_probability_true_state=("mean_probability_true_state", "mean"),
        rule_probability_when_true_rule=("rule_probability_when_true_rule", "mean"),
        rule_probability_when_nonrule=("rule_probability_when_nonrule", "mean"),
    )
    model_all = model_recovery[model_recovery["condition"].eq("all")]
    filter_generated = model_all[model_all["generator"].eq("strategy_filter")]
    feature_generated = model_all[model_all["generator"].eq("feature_rl")]
    parameter_summary = parameter_recovery.groupby("parameter").agg(
        true_value=("true_value", "mean"),
        recovered_mean=("recovered_value", "mean"),
        recovered_sd=("recovered_value", "std"),
        mae=("absolute_error", "mean"),
    )
    lines = [
        "# Behavior-state conditional recovery audit",
        "",
        "> Recovery conditions on real stimulus sequences and frozen feature-RL emission trajectories. It is not a full autonomous recovery of the feature learner.",
        "",
        "## State recovery",
        "",
        f"Across replicates, exact latent-state decoding accuracy was {state_all.exact_state_accuracy.mean():.3f} and three-mode decoding accuracy (guess/feature/any rule) was {state_all.mode_accuracy.mean():.3f}. The posterior assigned the true state mean probability {state_all.mean_probability_true_state.mean():.3f}.",
        f"Mean posterior rule mass was {state_all.rule_probability_when_true_rule.mean():.3f} on simulated rule trials and {state_all.rule_probability_when_nonrule.mean():.3f} on simulated non-rule trials.",
        "",
        "## Held-out model recovery",
        "",
        f"When the strategy filter generated choices, its mean held-out NLL advantage over frozen feature-RL was {filter_generated.mean_delta_nll_filter_advantage.mean():.4f}/trial and was positive in {(filter_generated.mean_delta_nll_filter_advantage > 0).sum()}/{len(filter_generated)} replicate datasets.",
        f"When feature-RL generated choices, the filter's mean advantage was {feature_generated.mean_delta_nll_filter_advantage.mean():.4f}/trial and was positive in {(feature_generated.mean_delta_nll_filter_advantage > 0).sum()}/{len(feature_generated)} replicate datasets.",
        "",
        "## Parameter recovery",
        "",
        "| Parameter | Generating mean | Recovered mean | Recovered SD | MAE |",
        "|:--|--:|--:|--:|--:|",
    ]
    for name, row in parameter_summary.iterrows():
        lines.append(
            f"| {name} | {row.true_value:.4f} | {row.recovered_mean:.4f} | "
            f"{row.recovered_sd:.4f} | {row.mae:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Good conditional recovery shows that the implemented filter can distinguish its own states under the frozen emissions. It does not prove that the real participant generated data from those states, nor does it validate the oral compatible-set measurement model.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    started = time.time()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = args.state.resolve()
    subjects = load_subjects(
        args.data.resolve(),
        args.core.resolve(),
        args.dynamic.resolve(),
        args.joint.resolve(),
    )
    parameter_table = pd.read_csv(state_path / "parameters.csv")
    parameter_table = parameter_table[
        parameter_table["model"].eq("BEHAVIOR_CHOICE")
    ].set_index("condition")
    generating_values = {
        int(condition): np.asarray(ast.literal_eval(row.raw_parameters), dtype=float)
        for condition, row in parameter_table.iterrows()
    }

    state_rows = []
    model_rows = []
    parameter_rows = []
    for replicate in range(int(args.replicates)):
        rng = np.random.default_rng(int(args.seed) + replicate * 1009)
        filter_subjects = []
        latent_states: dict[int, np.ndarray] = {}
        for subject in subjects:
            simulated, latent = simulate_filter_subject(
                subject,
                parameter_table.loc[int(subject["condition"])],
                rng,
            )
            filter_subjects.append(simulated)
            latent_states[int(subject["subject_id"])] = latent
        fitted_filter, recovered = fit_all_conditions(filter_subjects)
        state_rows.extend(
            state_recovery_rows(
                filter_subjects, latent_states, fitted_filter, replicate
            )
        )
        model_rows.extend(
            model_recovery_rows(
                filter_subjects, fitted_filter, "strategy_filter", replicate
            )
        )
        for row in recovered.itertuples(index=False):
            condition = int(row.condition)
            generating = parameter_table.loc[condition]
            for parameter in (
                "base_guess_mass",
                "base_feature_mass",
                "base_rule_mass",
                "state_persistence",
            ):
                recovered_value = float(getattr(row, parameter))
                true_value = float(generating[parameter])
                parameter_rows.append(
                    {
                        "replicate": replicate,
                        "condition": condition,
                        "parameter": parameter,
                        "true_value": true_value,
                        "recovered_value": recovered_value,
                        "absolute_error": abs(recovered_value - true_value),
                    }
                )

        feature_subjects = [simulate_feature_subject(subject, rng) for subject in subjects]
        fitted_feature, _ = fit_all_conditions(feature_subjects)
        model_rows.extend(
            model_recovery_rows(
                feature_subjects, fitted_feature, "feature_rl", replicate
            )
        )
        print(f"[recovery] completed replicate {replicate + 1}/{args.replicates}", flush=True)

    state_recovery = pd.DataFrame(state_rows).sort_values(
        ["replicate", "condition", "subject_id"]
    )
    model_recovery = pd.DataFrame(model_rows).sort_values(
        ["replicate", "generator", "condition"]
    )
    parameter_recovery = pd.DataFrame(parameter_rows).sort_values(
        ["replicate", "condition", "parameter"]
    )
    atomic_csv(output / "state_recovery.csv", state_recovery)
    atomic_csv(output / "model_recovery.csv", model_recovery)
    atomic_csv(output / "parameter_recovery.csv", parameter_recovery)
    render_report(output, state_recovery, model_recovery, parameter_recovery)
    manifest = {
        "result_type": "behavior_state_conditional_recovery",
        "status": "complete",
        "replicates": int(args.replicates),
        "base_seed": int(args.seed),
        "recovery_scope": (
            "conditional on real q arrays and frozen feature-RL emission trajectories; "
            "choice-only state filter"
        ),
        "n_subjects": int(len(subjects)),
        "state_manifest_sha256": sha256_file(state_path / "manifest.json"),
        "runtime_seconds": float(time.time() - started),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(output / "manifest.json", manifest)
    print(f"[done] wrote {output} in {manifest['runtime_seconds']:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
