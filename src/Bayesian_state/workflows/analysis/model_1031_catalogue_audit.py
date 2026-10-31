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
from .model_1031_finalization import digest, validate_source, write_new


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


def task_audit(study: Path) -> dict:
    """Check catalogue coverage of physical task labels without reading choices."""
    started, cpu = time.time(), time.process_time()
    registration = validate_source(study)
    rows = []
    for condition in (1, 2, 3):
        subjects = [s for s in registration["subjects"] if s["condition"] == condition]
        inputs = []
        for subject in subjects:
            with np.load(study / "inputs" / f"s{subject['subject']}.npz") as data:
                inputs.append((data["stimulus"].copy(), data["categories"].astype(int)))
        unique, inverse = np.unique(np.concatenate([x for x, _ in inputs]), axis=0, return_inverse=True)
        catalogue = ExperienceWorkspacePartition(n_cats=2 if condition == 1 else 4).catalogue
        p = np.stack([catalogue.get_category_probabilities(h, (unique, [], []), 25., distance_mode="boundary")
                      for h in range(catalogue.length)])
        maxima = np.isclose(p, p.max(axis=1, keepdims=True), atol=1e-12, rtol=0.)
        offset = 0
        for subject, (x, y) in zip(subjects, inputs):
            index = inverse[offset:offset + len(x)]
            score = maxima[:, y - 1, index] / maxima.sum(axis=1)[:, index]
            cut = subject["train_trials"]
            training, future, full = [score[:, where].mean(axis=1) for where in (slice(None, cut), slice(cut, None), slice(None))]
            rows.append(dict(subject=subject["subject"], condition=condition, trials=len(x),
                training_perfect_rules=np.flatnonzero(training == 1.).tolist(),
                full_perfect_rules=np.flatnonzero(full == 1.).tolist(),
                maximum_full_accuracy=float(full.max()), training_accuracy=training.tolist(),
                future_accuracy=future.tolist(), full_accuracy=full.tolist(),
                catalogue_signature=catalogue.boundary_geometry.space.content_signature))
            offset += len(x)
    return dict(rows=rows, beta=25., tie_policy="equal mass among tied maximal categories",
        scope="Recorded physical stimuli and environment category labels; human choices and oral reports are not read",
        inference="Catalogue coverage of recorded schedules; not evidence that a learner discovers or executes the rule",
        source_sha256=digest(Path(__file__)), cpu_seconds=time.process_time() - cpu, seconds=time.time() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--study", type=Path, help="Audit recorded task-label coverage instead of geometry duplicates")
    args = parser.parse_args()
    write_new(args.output, audit() if args.study is None else task_audit(args.study.resolve()))


if __name__ == "__main__":
    main()
