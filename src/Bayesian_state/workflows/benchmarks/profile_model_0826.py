"""Profile a small observed-data PF run without changing its numerical budget."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import json
import marshal
import platform
import pstats
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

from src.Bayesian_state.hypothesis_space import BoundaryGeometry, ContinuousPartition
from src.Bayesian_state.inference.backends.particle_filter import (
    run_state_model_particle_filter,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, default=129)
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--particles", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=8326)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, help="Earlier arrays.npz to compare exactly")
    args = parser.parse_args()
    if min(args.trials, args.repeats) < 1 or args.particles < 2:
        parser.error("positive trials/repeats and at least two particles are required")
    engine_path = Path("configs/exp123/model_struct/pmh_model_cond1_0826.yaml")
    data_path = Path("data/exp123/processed/Task2_processed.csv")
    data = pd.read_csv(data_path)
    subject = data.loc[data.iSub.eq(args.subject)].sort_values(
        ["iSession", "iBlock", "iTrial"], kind="stable"
    )
    if subject.empty or not subject.condition.eq(1).all():
        parser.error("select an existing condition-1 subject")
    selected = subject.iloc[:args.trials]
    kwargs = dict(
        engine_config=yaml.safe_load(engine_path.read_text()),
        subject_id=args.subject,
        stimulus=selected[[f"feature{i}" for i in range(1, 5)]].to_numpy(),
        choices=selected.choice.to_numpy(),
        feedback=selected.feedback.to_numpy(),
        particle_count=args.particles,
        choice_readout_power=1.0,
        filter_seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    timings = []
    # The first measurement includes initialization; subsequent runs reuse only
    # process-local immutable resources, with the same seed and fresh PF state.
    for _ in range(args.repeats + 1):
        started = perf_counter()
        result = run_state_model_particle_filter(**kwargs)
        timings.append(perf_counter() - started)
    arrays = {}
    for name in ("observation_probabilities", "state_probabilities", "latent_summaries"):
        for key, value in getattr(result, name).items():
            if value is not None:
                arrays[f"{name}/{key}"] = np.asarray(value)
    arrays["resampled"] = np.asarray(result.resampled)
    np.savez_compressed(args.output_dir / "arrays.npz", **arrays)
    if args.reference:
        with np.load(args.reference) as reference:
            if set(reference.files) != set(arrays):
                raise AssertionError("reference and current output keys differ")
            for key, value in arrays.items():
                np.testing.assert_array_equal(value, reference[key], err_msg=key)
    profiler = cProfile.Profile()
    profiler.runcall(run_state_model_particle_filter, **kwargs)
    profiler.dump_stats(args.output_dir / "warm.prof")
    with (args.output_dir / "profile.txt").open("w") as stream:
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(50)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    try:
        numba_version = importlib.metadata.version("numba")
    except importlib.metadata.PackageNotFoundError:
        numba_version = None
    summary = {
        "subject": args.subject, "trials": len(selected),
        "particles": args.particles, "seed": args.seed,
        "cold_seconds": timings[0], "warm_seconds": timings[1:],
        "warm_median_seconds": float(np.median(timings[1:])),
        "exact_reference_match": True if args.reference else None,
        "python": platform.python_version(), "numpy": np.__version__,
        "numba": numba_version, "git_revision": revision,
        "reference": None if args.reference is None else {
            "path": str(args.reference),
            "sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        },
        # Runtime code hashes also distinguish isolated benchmarks that inject
        # the previous revision's methods without changing the live checkout.
        "runtime_code_sha256": {
            method.__qualname__: hashlib.sha256(marshal.dumps(method.__code__)).hexdigest()
            for method in (BoundaryGeometry.distances_to_category,
                           BoundaryGeometry.category_distances,
                           ContinuousPartition.get_category_probabilities,
                           ContinuousPartition._category_feedback_likelihood)
        },
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (engine_path, data_path, *Path("src/Bayesian_state").rglob("*.py"))
        },
    }
    (args.output_dir / "timing.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
