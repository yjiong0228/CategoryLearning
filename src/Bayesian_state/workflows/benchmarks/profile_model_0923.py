"""Time a short Model 0923 PF and compare outputs with a saved reference."""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import json
import platform
import pstats
import subprocess
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import yaml

from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
from src.Bayesian_state.model import StateModel
from src.Bayesian_state.simulation.data import SubjectTrialDataLoader
from src.Bayesian_state.utils.parallel import single_threaded_processes


def _flatten(value, prefix: str, arrays: dict[str, np.ndarray]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(item, f"{prefix}/{key}", arrays)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _flatten(item, f"{prefix}/{index}", arrays)
    elif value is not None:
        array = np.asarray(value)
        # RNG states contain integers larger than int64. Keep their exact text.
        arrays[prefix] = np.asarray(str(value)) if array.dtype == object else array


def _result_arrays(result) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name in ("observation_probabilities", "state_probabilities", "latent_summaries", "diagnostics"):
        _flatten(getattr(result, name), name, arrays)
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--subject", type=int)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=923)
    parser.add_argument("--relation-samples", type=int, default=100_000)
    parser.add_argument("--without-oral", action="store_true", help="Measure choice-only fitting")
    parser.add_argument("--capture-states", action="store_true", help="Save pre/post-feedback state and RNGs")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--similarity-cache-dir", type=Path, help="Reuse a resource cache across processes")
    parser.add_argument("--reference", type=Path, help="Earlier arrays.npz; require exact equality")
    args = parser.parse_args()
    if min(args.trials, args.repeats, args.relation_samples) < 1 or args.particles < 2:
        parser.error("positive trials/repeats/samples and at least two particles are required")
    subject = args.subject or {1: 122, 2: 206, 3: 314}[args.condition]
    engine_path = Path(f"configs/exp123/model_struct/model_0923_cond{args.condition}_full_v3.yaml")
    config = yaml.safe_load(engine_path.read_text())
    config["inference"]["particle_count"] = args.particles
    config["modules"]["hypo_transitions_mod"]["kwargs"]["relation_samples"] = args.relation_samples
    config["provenance"]["hypothesis_similarity"]["sample_count"] = args.relation_samples
    if args.without_oral:
        config.pop("oral_readout", None)
    loader = SubjectTrialDataLoader(config)
    loader.prepare_data()
    frame = loader._get_subject_frame(subject, 1.)
    if loader._get_condition_value(frame) != args.condition:
        parser.error("subject does not belong to the requested condition")
    trials = loader._extract_arrays(frame, args.trials)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config["partition"]["kwargs"]["similarity_cache_dir"] = str(
        args.similarity_cache_dir or args.output_dir / "similarity"
    )
    kwargs = dict(
        engine_config=config, subject_id=subject, condition=args.condition,
        stimulus=trials.stimulus, choices=trials.choices, feedback=trials.feedback,
        particle_count=args.particles, choice_readout_power=1., filter_seed=args.seed,
    )
    with single_threaded_processes():
        timings = []
        for _ in range(args.repeats + 1):
            started = perf_counter()
            result = run_state_model_particle_filter(**kwargs)
            timings.append(perf_counter() - started)
        arrays = _result_arrays(result)
        if args.capture_states:
            with ExitStack() as stack:
                for method_name in ("begin_trial", "complete_trial"):
                    original = getattr(StateModel, method_name)
                    counter = [0]

                    def capture(model, *a, _original=original, _name=method_name, _counter=counter, **kw):
                        value = _original(model, *a, **kw)
                        _flatten(model.engine.state_dict(), f"states/{_name}/{_counter[0]}", arrays)
                        _counter[0] += 1
                        return value

                    stack.enter_context(patch.object(StateModel, method_name, capture))
                run_state_model_particle_filter(**kwargs)
        np.savez_compressed(args.output_dir / "arrays.npz", **arrays)
        if args.reference:
            with np.load(args.reference, allow_pickle=False) as reference:
                if set(reference.files) != set(arrays):
                    raise AssertionError("reference and current output keys differ")
                for key, value in arrays.items():
                    np.testing.assert_array_equal(value, reference[key], err_msg=key)
        profiler = cProfile.Profile()
        profiler.runcall(run_state_model_particle_filter, **kwargs)
        profiler.dump_stats(args.output_dir / "warm.prof")
    with (args.output_dir / "profile.txt").open("w") as stream:
        pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime").print_stats(60)
    source_paths = [engine_path, *Path("src/Bayesian_state").rglob("*.py"),
                    *Path("data/exp123/processed").glob("Task1b*csv"),
                    Path("data/exp123/processed/Task2_processed.csv")]
    summary = {
        "condition": args.condition, "subject": subject, "trials": len(trials.choices),
        "particles": args.particles, "seed": args.seed, "relation_samples": args.relation_samples,
        "oral_enabled": not args.without_oral, "capture_states": args.capture_states,
        "cold_seconds": timings[0], "warm_seconds": timings[1:],
        "warm_median_seconds": float(np.median(timings[1:])),
        "exact_reference_match": True if args.reference else None, "array_count": len(arrays),
        "python": platform.python_version(),
        "versions": {},
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "reference_sha256": None if args.reference is None else hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
    }
    for name in ("numpy", "scipy", "numba", "pandas", "PyYAML"):
        try:
            summary["versions"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            summary["versions"][name] = None
    (args.output_dir / "resolved_engine.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (args.output_dir / "timing.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
