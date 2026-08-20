"""Non-CI benchmark for continuous boundary projection backends."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from src.Bayesian_state.hypothesis_space import BoundaryGeometry, ContinuousPartition


def benchmark(n_cats: int, label_policy: str, n_stimuli: int) -> None:
    stimuli = np.random.default_rng(0).random((n_stimuli, 4))
    for method in BoundaryGeometry.VALID_METHODS:
        BoundaryGeometry.clear_compilation_cache()
        partition = ContinuousPartition(
            4,
            n_cats,
            label_permutation_policy=label_policy,
            boundary_distance_method=method,
            similarity_n_samples=8,
        )
        for pass_name in ("cold", "warm"):
            started = perf_counter()
            for hypothesis in range(partition.length):
                partition.boundary_geometry.category_distances(
                    hypothesis, stimuli
                )
            elapsed = perf_counter() - started
            print(
                f"categories={n_cats} hypotheses={partition.length} "
                f"stimuli={n_stimuli} method={method} pass={pass_name} "
                f"seconds={elapsed:.6f} "
                f"compiled={BoundaryGeometry.compilation_cache_info()['size']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-stimuli", type=int, default=512)
    args = parser.parse_args()
    benchmark(2, "identity_only", args.n_stimuli)
    benchmark(2, "binary_identity_and_reverse", args.n_stimuli)
    benchmark(4, "identity_only", args.n_stimuli)


if __name__ == "__main__":
    main()
