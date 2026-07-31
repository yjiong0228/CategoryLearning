#!/usr/bin/env python3
"""Combine strictly paired common-core runs for the condition-1 new plan."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_cond1_newplan_factorial import (
    load_theta_cache,
    paired_summary,
    parameter_rows,
    select_models,
    theta_token,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Core label and run directory as LABEL=PATH; repeat for each core.",
    )
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results/zhuran/cond1_newplan/core_sensitivity",
    )
    return parser.parse_args()


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected LABEL=PATH, got {value!r}.")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError("Core labels cannot be empty.")
    return label, Path(path).resolve()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def signature(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: manifest[key]
        for key in (
            "subjects",
            "theta_grid",
            "epsilon_grid",
            "max_trials",
            "base_seed",
            "marginal_method",
        )
    }


def selected_core_counts(
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for model_id in ("B0", "B1", "D0", "D1"):
        out[model_id] = dict(
            Counter(
                str(row["common_core"])
                for row in selected
                if str(row["model_id"]) == model_id
            )
        )
    return out


def write_report(
    path: Path,
    *,
    repeats: int,
    core_summaries: Mapping[str, Mapping[str, Any]],
    joint_summary: Mapping[str, Any],
    selected: Sequence[Mapping[str, Any]],
    core_counts: Mapping[str, Mapping[str, int]],
) -> None:
    labels = (
        "lapse_without_swap_B1_minus_B0",
        "swap_without_lapse_D0_minus_B0",
        "swap_after_lapse_D1_minus_B1",
        "lapse_after_swap_D1_minus_D0",
    )
    lines = [
        "# Condition-1 common-core sensitivity",
        "",
        f"- particles per core: {repeats}",
        "- common random numbers across cores: verified from every cached point seed",
        "- selection: training Brier only; evaluation: held-out Brier",
        "",
        "Negative held-out Brier delta favors the first model.",
        "",
        "| core | B1−B0 | D0−B0 | D1−B1 | D1−D0 |",
        "|:---|---:|---:|---:|---:|",
    ]
    for core, summary in core_summaries.items():
        values = [float(summary[label]["mean_test_brier_delta"]) for label in labels]
        lines.append(
            f"| {core} | {values[0]:.6f} | {values[1]:.6f} | "
            f"{values[2]:.6f} | {values[3]:.6f} |"
        )
    joint_values = [
        float(joint_summary[label]["mean_test_brier_delta"]) for label in labels
    ]
    lines.extend(
        [
            f"| **joint training-selected core** | {joint_values[0]:.6f} | "
            f"{joint_values[1]:.6f} | {joint_values[2]:.6f} | "
            f"{joint_values[3]:.6f} |",
            "",
            "## Joint paired held-out effects",
            "",
        ]
    )
    for label in labels:
        item = joint_summary[label]
        lines.append(
            f"- {label}: mean={item['mean_test_brier_delta']:.6f}, "
            f"median={item['median_test_brier_delta']:.6f}, "
            f"improved={item['improved_count']}/{item['n']}"
        )
    lines.extend(["", "## Selected common cores", ""])
    for model_id, counts in core_counts.items():
        rendered = ", ".join(
            f"{core}={count}" for core, count in sorted(counts.items())
        )
        lines.append(f"- {model_id}: {rendered}")
    lines.extend(
        [
            "",
            "## Joint training-selected parameters",
            "",
            "| subject | model | core | theta | epsilon | train Brier | test Brier |",
            "|---:|:---:|:---|---:|---:|---:|---:|",
        ]
    )
    for row in selected:
        lines.append(
            f"| {row['subject_id']} | {row['model_id']} | "
            f"{row['common_core']} | {float(row['theta']):.3f} | "
            f"{float(row['epsilon']):.3f} | "
            f"{float(row['train_choice_brier']):.6f} | "
            f"{float(row['test_choice_brier']):.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = [parse_run(value) for value in args.run]
    if len({label for label, _ in runs}) != len(runs):
        raise ValueError("Core labels must be unique.")
    if args.repeats < 2:
        raise ValueError("repeats must be at least 2.")

    manifests = {label: load_json(path / "manifest.json") for label, path in runs}
    first_label = runs[0][0]
    reference = signature(manifests[first_label])
    for label, _ in runs[1:]:
        if signature(manifests[label]) != reference:
            raise ValueError(f"Run manifest mismatch for {label}.")

    all_rows: list[dict[str, Any]] = []
    core_summaries: dict[str, Mapping[str, Any]] = {}
    point_seeds: dict[tuple[int, float], int] = {}
    for label, run_dir in runs:
        manifest = manifests[label]
        theta_results = []
        for subject_id in manifest["subjects"]:
            for theta in manifest["theta_grid"]:
                cache_path = (
                    run_dir
                    / "cache"
                    / f"subject_{int(subject_id)}"
                    / f"theta_{theta_token(float(theta))}.npz"
                )
                result = load_theta_cache(cache_path)
                stack = np.asarray(result["probability_stack"], dtype=float)
                if stack.shape[0] < args.repeats:
                    raise ValueError(
                        f"{cache_path} has {stack.shape[0]} paths, "
                        f"fewer than requested {args.repeats}."
                    )
                seed_key = int(subject_id), float(theta)
                seed = int(result["simulation_point_seed"])
                if seed_key in point_seeds and point_seeds[seed_key] != seed:
                    raise ValueError(
                        f"CRN seed mismatch for subject={subject_id}, theta={theta}: "
                        f"{point_seeds[seed_key]} != {seed}."
                    )
                point_seeds[seed_key] = seed
                result["probability_stack"] = stack[: args.repeats]
                theta_results.append(result)
        row_args = SimpleNamespace(marginal_method=manifest["marginal_method"])
        rows = parameter_rows(row_args, theta_results, manifest["epsilon_grid"])
        for row in rows:
            row.update(
                {
                    "common_core": label,
                    "gamma": float(manifest["gamma"]),
                    "w0": float(manifest["w0"]),
                    "rho": float(manifest["rho"]),
                    "particles": int(args.repeats),
                }
            )
        core_selected = select_models(rows)
        core_summaries[label] = paired_summary(core_selected)
        all_rows.extend(rows)

    selected = select_models(all_rows)
    joint_summary = paired_summary(selected)
    core_counts = selected_core_counts(selected)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "combined_parameter_grid.csv", all_rows)
    write_csv(args.output_dir / "joint_selected_models.csv", selected)
    write_json(
        args.output_dir / "aggregate_summary.json",
        {
            "particles_per_core": args.repeats,
            "crn_seed_points_verified": len(point_seeds),
            "core_manifests": manifests,
            "per_core_paired_heldout": core_summaries,
            "joint_paired_heldout": joint_summary,
            "selected_core_counts": core_counts,
            "joint_selected_models": selected,
        },
    )
    write_report(
        args.output_dir / "RESULTS.md",
        repeats=args.repeats,
        core_summaries=core_summaries,
        joint_summary=joint_summary,
        selected=selected,
        core_counts=core_counts,
    )
    print(f"COMPLETE output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
