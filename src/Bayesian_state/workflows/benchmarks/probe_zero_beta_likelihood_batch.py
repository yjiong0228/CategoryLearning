"""Historical process-local likelihood prototype; use round2 for new ablations.

For the exact ContinuousPartition class, finite unit-cube stimuli and binary
category feedback, all zero-beta columns are identical. Evaluate one such
column through the original implementation, reuse it, and retain the original
full-matrix normalization. All other inputs use the original implementation.
This is a performance probe, not a supported alternate inference backend.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter
from unittest.mock import patch

import numpy as np

from ...hypothesis_space.geometry.stimuli import as_stimuli
from ...hypothesis_space.observation_model.base_partition import BasePartition
from ...hypothesis_space.observation_model.continuous_partition import ContinuousPartition
from .benchmark_model_0826_acceleration import cases, compare_arrays, run_case


_ORIGINAL = BasePartition.calc_likelihood


def _probe_likelihood(self, hypos, data, beta=1.0, distance_mode=None,
                      normalized=True, **kwargs):
    def fallback():
        return _ORIGINAL(self, hypos, data, beta, distance_mode, normalized, **kwargs)

    if type(self) is not ContinuousPartition or not self.zero_beta_fast_path:
        return fallback()
    beta_values = self._resolve_beta_vector(beta, len(hypos))
    zero_columns = [i for i, value in enumerate(beta_values) if value == 0.0]
    if len(zero_columns) < 2:
        return fallback()
    responses = np.asarray(data[2])
    mode = self._resolve_feedback_likelihood_mode(
        kwargs.get("feedback_likelihood_mode", self.FEEDBACK_MODE_CATEGORY)
    )
    if mode != self.FEEDBACK_MODE_CATEGORY or not np.all((responses == 0) | (responses == 1)):
        return fallback()
    values = as_stimuli(data[0], self.n_dims)
    if not values.shape[0] or not np.all((values >= 0) & (values <= 1)):
        return fallback()
    resolved_mode = self._resolve_distance_mode(distance_mode)
    # Keep hypothesis-index checking for columns whose evaluation is skipped.
    for column in zero_columns:
        self.hypothesis_space[int(hypos[column])]
    result = np.zeros((len(data[2]), len(hypos)), dtype=float)
    first_zero = zero_columns[0]
    zero_likelihood = self.calc_likelihood_entry(
        hypos[first_zero], data, beta_values[first_zero],
        distance_mode=resolved_mode, **kwargs,
    )
    result[:, zero_columns] = zero_likelihood[:, None]
    for column, hypothesis in enumerate(hypos):
        if beta_values[column] != 0.0:
            result[:, column] = self.calc_likelihood_entry(
                hypothesis, data, beta_values[column],
                distance_mode=resolved_mode, **kwargs,
            )
    return result if not normalized else result / np.sum(result, axis=1, keepdims=True)


def _matrix_checks() -> int:
    """Check normalized/raw outputs and fallback paths independently of PF."""
    count = 0
    for n_cats in (2, 4):
        partition = ContinuousPartition(4, n_cats)
        hypos = list(range(partition.length))
        mixed = np.zeros(len(hypos))
        mixed[::9] = 3.0
        stimuli = np.array([[0.1, 0.4, 0.7, 0.8], [0.9, 0.2, 0.3, 0.6]])
        for beta in (0.0, mixed, 1e-12, 2.0):
            for feedback in ([1.0, 0.0], [0.5, 1.0]):
                for mode in ("boundary", "prototype"):
                    for normalized in (False, True):
                        data = (stimuli, [1, n_cats], feedback)
                        expected = _ORIGINAL(partition, hypos, data, beta, mode, normalized)
                        actual = _probe_likelihood(partition, hypos, data, beta, mode, normalized)
                        np.testing.assert_array_equal(actual, expected)
                        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cases", nargs="+", default=[
        "c1_M3_chi0", "c2_M3_chi0", "c3_M3_chi0", "c1_P", "c1_long",
    ])
    args = parser.parse_args()
    selected = [case for case in cases() if case["name"] in args.cases]
    if args.repeats < 1 or set(args.cases) != {case["name"] for case in selected}:
        parser.error("Require positive repeats and known benchmark case names")
    for case in selected:
        if not (args.reference / (case["name"] + ".npz")).is_file():
            parser.error(f"Missing reference for {case['name']}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    matrix_checks = _matrix_checks()
    rows = []
    for case in selected:
        records = {}
        for repeat in range(args.repeats):
            order = ("current", "probe") if repeat % 2 == 0 else ("probe", "current")
            for mode in order:
                context = patch.object(ContinuousPartition, "calc_likelihood", _probe_likelihood) if mode == "probe" else nullcontext()
                with context:
                    row, arrays = run_case(case, "default", 1)
                with np.load(args.reference / (case["name"] + ".npz"), allow_pickle=False) as ref:
                    compare_arrays(ref, arrays)
                if mode not in records:
                    row["mode"] = mode
                    row["exact_reference_match"] = True
                    records[mode] = row
                else:
                    records[mode]["warm_seconds"].extend(row["warm_seconds"])
        for row in records.values():
            row["warm_median_seconds"] = float(np.median(row["warm_seconds"]))
        speedup = records["current"]["warm_median_seconds"] / records["probe"]["warm_median_seconds"]
        rows.append({"name": case["name"], "additional_speedup": speedup, "measurements": records})
        print(f"{case['name']}: additional {speedup:.3f}x, exact match", flush=True)
    paths = [Path("data/exp123/processed/Task2_processed.csv"),
             *Path("src/Bayesian_state").rglob("*.py"),
             *Path("configs/exp123/model_struct").glob("*0826.yaml"),
             *(args.reference / (case["name"] + ".npz") for case in selected)]
    report = {"cases": rows, "matrix_exact_checks": matrix_checks,
              "wall_seconds": perf_counter() - started, "reference": str(args.reference),
              "python": platform.python_version(),
              "versions": {name: importlib.metadata.version(name)
                           for name in ("numpy", "scipy", "pandas", "numba", "joblib")},
              "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted(paths)}}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
