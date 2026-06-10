from __future__ import annotations

import numpy as np

from src.Bayesian_state.problems.partitions import Partition
from src.Bayesian_state.utils.oral_model_alignment import (
    OralModelAlignmentMixin,
    Oral_region_mapping,
    RegionOverlapScorer,
)
from src.oral_coding import Recording_Processor_Region


def test_partition_boundary_regions_match_prototype_category_labels() -> None:
    for n_cats in (2, 4):
        partition = Partition(n_dims=4, n_cats=n_cats)
        for hypo_idx in range(partition.length):
            for cat_idx in range(partition.n_cats):
                center = partition.prototypes[hypo_idx, 0, cat_idx, :]
                region = partition.regions[hypo_idx][cat_idx]
                vals = np.asarray(region["A"], dtype=float) @ center - np.asarray(region["b"], dtype=float)
                assert np.all(vals <= 1e-9), (n_cats, hypo_idx, cat_idx, partition.splits[hypo_idx])


def test_model_boundary_region_uses_canonical_less_equal_form() -> None:
    for n_cats in (2, 4):
        partition = Partition(n_dims=4, n_cats=n_cats)
        for hypo_idx in range(partition.length):
            for cat_idx in range(partition.n_cats):
                center = partition.prototypes[hypo_idx, 0, cat_idx, :]
                region = partition.regions[hypo_idx][cat_idx]
                A, b = Oral_region_mapping._parse_region(region)
                assert A is not None
                assert b is not None
                assert Oral_region_mapping._points_in_region(center[None, :], A, b, dist_tol=1e-9)[0]


def test_region_overlap_scorer_uses_canonical_less_equal_regions() -> None:
    partition = Partition(n_dims=4, n_cats=2)
    scorer = RegionOverlapScorer(partition, n_samples=2000, random_state=0)
    oral_region = partition.regions[0][0]

    scores = scorer.score_all(oral_region, cat_idx=0, metric="iou")

    assert np.isclose(scores[0], 1.0)


def test_oral_region_encoder_outputs_model_form_constraints() -> None:
    processor = Recording_Processor_Region()

    A, b, _, _ = processor.extract_region("头比较长。")

    assert A == [[0.0, -1.0, 0.0, 0.0]]
    assert b == [-0.5]


def test_center_oral_distribution_rejects_invalid_choice() -> None:
    partition = Partition(n_dims=4, n_cats=2)
    center = partition.prototypes[0, 0, 0, :]

    dist = OralModelAlignmentMixin._center_oral_distribution(center, choice=0, partition=partition)

    assert np.isnan(dist).all()
