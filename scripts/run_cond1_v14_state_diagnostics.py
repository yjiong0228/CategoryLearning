#!/usr/bin/env python3
"""Verify that the V14 belief-instability state is active and persistent."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cond1_v14_pilot import (  # noqa: E402
    DEFAULT_SUBJECTS,
    load_inputs,
    selected_hyperparams,
    variants_for_subject,
    write_json,
)
from src.Bayesian_state.simulation.parameters import (  # noqa: E402
    apply_fixed_hyperparams_to_engine_config,
)
from src.Bayesian_state.utils.datasets import resolve_dataset_paths  # noqa: E402
from src.Bayesian_state.simulation.config import (  # noqa: E402
    DEFAULT_DATA_PATH,
    resolve_engine_config,
)
from src.Bayesian_state.simulation.runner import (  # noqa: E402
    StateModelSimulationRunner,
)


def lag1(values: np.ndarray) -> float:
    if values.size < 3 or np.std(values[:-1]) == 0 or np.std(values[1:]) == 0:
        return float("nan")
    return float(np.corrcoef(values[:-1], values[1:])[0, 1])


def mean_spell_length(mask: np.ndarray) -> float:
    lengths: list[int] = []
    current = 0
    for value in mask:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return float(mean(lengths)) if lengths else 0.0


def summarize_run(run: dict[str, Any], threshold: float) -> dict[str, Any]:
    latent_log = run["state_log"]["latent_volatility"]
    transitions = run["transition_counts"]
    state = np.asarray([float(item["state"]) for item in latent_log], dtype=float)
    raw_error = np.asarray(
        [float(item.get("raw_error_severity", 0.0)) for item in latent_log], dtype=float
    )
    selected = [str(item.get("selected_policy_method")) for item in transitions]
    aggressive_probability = np.asarray(
        [
            float((item.get("policy_probabilities") or {}).get("aggressive", np.nan))
            for item in transitions
        ],
        dtype=float,
    )
    high = state >= threshold
    low = ~high
    out: dict[str, Any] = {
        "state_mean": float(np.mean(state)),
        "state_max": float(np.max(state)),
        "state_nonzero_fraction": float(np.mean(state > 1e-12)),
        "state_above_threshold_fraction": float(np.mean(high)),
        "state_above_threshold_spell_mean": mean_spell_length(high),
        "state_lag1": lag1(state),
        "state_after_error": float(np.mean(state[raw_error > 0]))
        if np.any(raw_error > 0)
        else float("nan"),
        "state_after_correct": float(np.mean(state[raw_error == 0]))
        if np.any(raw_error == 0)
        else float("nan"),
        "aggressive_probability_high_state": float(np.nanmean(aggressive_probability[high]))
        if np.any(high)
        else float("nan"),
        "aggressive_probability_low_state": float(np.nanmean(aggressive_probability[low]))
        if np.any(low)
        else float("nan"),
    }
    counts = Counter(selected)
    for policy in ("conservative", "stable", "aggressive", "stubborn"):
        out[f"policy_{policy}_fraction"] = counts[policy] / max(1, len(selected))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subjects", type=int, nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--error-gain", type=float, default=0.35)
    parser.add_argument("--decay", type=float, default=0.80)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/cond1_v14/state_diagnostics"
    )
    args = parser.parse_args()

    best, candidates, sim_path, sim_cfg = load_inputs()
    base_engine = resolve_engine_config(sim_cfg, sim_path.parent)
    dataset_paths = resolve_dataset_paths(sim_cfg, sim_path.parent, DEFAULT_DATA_PATH)
    run_rows: list[dict[str, Any]] = []

    for subject_id in args.subjects:
        hyperparams = selected_hyperparams(best, subject_id)
        variant = next(
            item
            for item in variants_for_subject(
                hyperparams,
                candidates,
                models="core",
                error_gain=args.error_gain,
                decay=args.decay,
                threshold=args.threshold,
            )
            if item["variant_id"] == "m2_core6_expectation"
        )
        engine = apply_fixed_hyperparams_to_engine_config(
            base_engine, variant["hyperparams"]
        )
        runner = StateModelSimulationRunner(
            engine,
            processed_data_dir=dataset_paths["processed_dir"],
            dataset_paths=dataset_paths,
            n_jobs=args.n_jobs,
        )
        runner.prepare_data(dataset_paths["learning_data"])
        result = runner.simulate_subject(
            subject_id,
            simulation_repeats=args.repeats,
            fixed_hyperparams=variant["hyperparams"],
            window_size=16,
            keep_logs=True,
            prediction_mode="prior_t",
            selection_prediction_mode="prior_t",
            loss_metric="choice_brier",
            hyper_candidate_seed=140014,
            seed_hyperparams={"paired_seed_group": "cond1_v14_state_diagnostic"},
        )
        for run in result["best"].raw_runs:
            run_rows.append(
                {
                    "subject_id": int(subject_id),
                    "run_index": int(run["run_index"]),
                    **summarize_run(run, args.threshold),
                }
            )
        print(f"DONE subject={subject_id} runs={args.repeats}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "state_diagnostic_runs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(run_rows[0]))
        writer.writeheader()
        writer.writerows(run_rows)
    metric_names = [key for key in run_rows[0] if key not in {"subject_id", "run_index"}]
    subjects_summary = []
    for subject_id in args.subjects:
        rows = [row for row in run_rows if row["subject_id"] == subject_id]
        subjects_summary.append(
            {
                "subject_id": subject_id,
                **{
                    metric: float(np.nanmean([float(row[metric]) for row in rows]))
                    for metric in metric_names
                },
            }
        )
    aggregate = {
        metric: float(np.nanmean([float(row[metric]) for row in run_rows]))
        for metric in metric_names
    }
    write_json(
        args.output_dir / "state_diagnostic_summary.json",
        {
            "repeats_per_subject": args.repeats,
            "subject_count": len(args.subjects),
            "run_count": len(run_rows),
            "threshold": args.threshold,
            "aggregate": aggregate,
            "subjects": subjects_summary,
        },
    )
    print(f"COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
