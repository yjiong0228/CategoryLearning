"""Audit repeated geometry and fixed-label coverage without changing the catalogue."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc

from ...hypothesis_space.observation_model.experience_workspace import ExperienceWorkspacePartition
from .model_1031_finalization import digest, write_new


def category_signature(category) -> tuple:
    components = []
    for polytope in category.components:
        rows = []
        for normal, offset in zip(polytope.A, polytope.b):
            scale = np.linalg.norm(normal)
            if scale == 0:
                continue
            rows.append(tuple(np.round(np.r_[normal, offset] / scale, 12)))
        components.append(tuple(sorted(set(rows))))
    return tuple(sorted(set(components)))


def audit() -> dict:
    started, cpu = time.time(), time.process_time()
    rows = []
    probes = qmc.Sobol(4, scramble=True, seed=10312026).random_base2(8)
    for categories in (2, 4):
        partition = ExperienceWorkspacePartition(n_cats=categories)
        catalogue = partition.catalogue
        space = catalogue.boundary_geometry.space
        signatures = [tuple(category_signature(c) for c in spec.categories) for spec in space]
        groups = defaultdict(list)
        for i, signature in enumerate(signatures):
            groups[signature].append(i)
        present = set(signatures)
        permutation_coverage = [sum(tuple(s[j] for j in order) in present for order in permutations(range(categories)))
                                for s in signatures]
        predictions = np.stack([catalogue.get_category_probabilities(i, (probes, [], []), 3., distance_mode="boundary")
                                for i in range(catalogue.length)])
        matches = []
        minimum = float("inf")
        for i in range(len(predictions)):
            for j in range(i):
                difference = float(np.max(np.abs(predictions[i] - predictions[j])))
                minimum = min(minimum, difference)
                if difference < 1e-12:
                    matches.append([j, i])
        rows.append(dict(categories=categories, catalogue_size=len(space), catalogue_signature=space.content_signature,
            family_counts=dict(Counter(s.family for s in space)),
            identical_normalized_constraint_groups=[g for g in groups.values() if len(g) > 1],
            matching_predictions_on_probes=matches, minimum_pairwise_max_prediction_difference=minimum,
            label_permutation_coverage_histogram=dict(Counter(permutation_coverage)),
            total_label_permutations=2 if categories == 2 else 24))
    return dict(rows=rows, probes=256, probe_seed=10312026, beta=3., constraint_rounding_digits=12,
        scope="Sufficient syntactic duplicate check plus finite prediction probes; not a proof of semantic equivalence or complete coverage.",
        label_policy="Existing fixed labels retained; arbitrary response relabeling is not a model symmetry.",
        source_sha256=digest(Path(__file__)), cpu_seconds=time.process_time() - cpu, seconds=time.time() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_new(args.output, audit())


if __name__ == "__main__":
    main()
