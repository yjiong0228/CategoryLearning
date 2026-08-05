#!/usr/bin/env python3
"""Combine checkpointed alive filters into disjoint likelihood-weighted islands."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.Bayesian_state.utils.model_0804 import (  # noqa: E402
    combine_model0804_alive_islands,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--model", type=str, default="FA2")
    parser.add_argument("--setting", type=str, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def _available_seeds(directory: Path) -> list[int]:
    seeds = []
    for path in directory.glob("seed_*/trial_trace.npz"):
        match = re.fullmatch(r"seed_(\d+)", path.parent.name)
        if match:
            seeds.append(int(match.group(1)))
    return sorted(seeds)


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    setting_directory = (
        input_root
        / f"subject_{int(args.subject)}"
        / str(args.model)
        / str(args.setting)
    )
    available = _available_seeds(setting_directory)
    if args.seeds:
        requested = [
            int(value.strip())
            for value in args.seeds.split(",")
            if value.strip()
        ]
    else:
        requested = available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise ValueError(f"missing alive traces for seeds {missing}")
    group_size = int(args.group_size)
    if group_size < 2 or len(requested) < 2 * group_size:
        raise ValueError("at least two complete groups of two or more islands are required")
    if len(requested) % group_size != 0:
        raise ValueError("the requested seed count must be divisible by group_size")

    traces = {}
    choices = None
    for seed in requested:
        trace_path = setting_directory / f"seed_{seed}" / "trial_trace.npz"
        with np.load(trace_path, allow_pickle=False) as payload:
            trace_choices = np.asarray(payload["choices"], dtype=int)
            probabilities = np.asarray(payload["probabilities"], dtype=float)
            increments = np.asarray(
                payload["alive_incremental_likelihood"], dtype=float
            )
        if choices is None:
            choices = trace_choices
        elif not np.array_equal(choices, trace_choices):
            raise ValueError("alive island traces do not share the same choices")
        traces[seed] = SimpleNamespace(
            inference_method="alive_categorical",
            probabilities=probabilities,
            alive_incremental_likelihood=increments,
        )
    if choices is None:
        raise ValueError("no alive traces were loaded")

    seed_groups = [
        requested[start : start + group_size]
        for start in range(0, len(requested), group_size)
    ]
    ensembles = []
    group_rows = []
    output = (
        args.output.resolve()
        if args.output is not None
        else input_root / "island_ensemble"
    )
    for group_index, seeds in enumerate(seed_groups):
        ensemble = combine_model0804_alive_islands(
            [traces[seed] for seed in seeds], choices
        )
        ensembles.append(ensemble)
        group_rows.append(
            {
                "group_index": int(group_index),
                "seeds": seeds,
                "island_count": int(ensemble.island_count),
                "nll": float(ensemble.nll),
                "minimum_effective_island_count": float(
                    np.min(ensemble.effective_island_count)
                ),
                "final_pretrial_effective_island_count": float(
                    ensemble.effective_island_count[-1]
                ),
                "final_island_log_evidence_range": float(
                    np.ptp(ensemble.final_island_log_evidence)
                ),
            }
        )
        _atomic_npz(
            output / f"group_{group_index}_ensemble_trace.npz",
            choices=choices,
            probabilities=ensemble.probabilities,
            incremental_likelihood=ensemble.incremental_likelihood,
            pretrial_island_weights=ensemble.pretrial_island_weights,
            effective_island_count=ensemble.effective_island_count,
            final_island_log_evidence=ensemble.final_island_log_evidence,
        )

    comparisons = []
    for first in range(len(ensembles)):
        for second in range(first + 1, len(ensembles)):
            difference = np.abs(
                ensembles[first].probabilities
                - ensembles[second].probabilities
            )
            comparisons.append(
                {
                    "first_group": first,
                    "second_group": second,
                    "absolute_nll_difference": float(
                        abs(ensembles[first].nll - ensembles[second].nll)
                    ),
                    "maximum_probability_difference": float(np.max(difference)),
                    "mean_probability_difference": float(np.mean(difference)),
                    "p95_probability_difference": float(
                        np.quantile(difference, 0.95)
                    ),
                }
            )

    implementation = ROOT / "src/Bayesian_state/utils/model_0804.py"
    payload = {
        "status": "alive_island_ensemble_complete",
        "scope": "numerical_diagnostic_not_model_comparison",
        "input": str(input_root),
        "subject": int(args.subject),
        "model": str(args.model),
        "setting": str(args.setting),
        "group_size": group_size,
        "seed_groups": seed_groups,
        "groups": group_rows,
        "comparisons": comparisons,
        "combination_rule": (
            "pretrial_cumulative_evidence_weighting; final likelihood equals "
            "arithmetic mean of independent island likelihood estimates"
        ),
        "implementation_sha256": _sha256(implementation),
    }
    report = output / "island_ensemble_report.json"
    _atomic_json(report, payload)
    print(f"ISLAND status=complete output={report}")


if __name__ == "__main__":
    main()
