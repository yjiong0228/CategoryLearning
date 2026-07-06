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


def test_partition_label_reversals_are_opt_in_for_cond1() -> None:
    base = Partition(n_dims=4, n_cats=2)
    reversed_partition = Partition(n_dims=4, n_cats=2, include_label_reversals=True)

    assert base.length == 19
    assert reversed_partition.base_hypothesis_count == base.length
    assert reversed_partition.length == 38
    np.testing.assert_allclose(reversed_partition.prototypes[: base.length], base.prototypes)
    assert reversed_partition.splits[: base.length] == base.splits


def test_partition_label_reversal_swaps_ground_truth_categories() -> None:
    partition = Partition(n_dims=4, n_cats=2, include_label_reversals=True)
    reversed_hypo = partition.base_hypothesis_count

    np.testing.assert_allclose(
        partition.prototypes[reversed_hypo, 0, 0, :],
        partition.prototypes[0, 0, 1, :],
    )
    np.testing.assert_allclose(
        partition.prototypes[reversed_hypo, 0, 1, :],
        partition.prototypes[0, 0, 0, :],
    )
    assert partition.hypothesis_metadata[reversed_hypo] == {
        "base_hypo": 0,
        "label_permutation": (1, 0),
        "is_label_permuted": True,
    }


def test_partition_label_reversal_predicts_ground_truth_prototypes_below_chance() -> None:
    partition = Partition(n_dims=4, n_cats=2, include_label_reversals=True)
    reversed_hypo = partition.base_hypothesis_count
    beta = 15.0

    for true_cat in range(partition.n_cats):
        true_proto = partition.prototypes[0, 0, true_cat, :]
        prob = partition.get_category_probabilities(
            hypo=reversed_hypo,
            data=([true_proto], [true_cat + 1], [1.0]),
            beta=beta,
            distance_mode="prototype",
        )[:, 0]
        assert prob[true_cat] < 0.001


def test_partition_label_reversal_similarity_matrix_shape() -> None:
    partition = Partition(
        n_dims=4,
        n_cats=2,
        include_label_reversals=True,
        similarity_n_samples=256,
    )

    sim = partition._compute_hypothesis_similarity_matrix(n_samples=256, random_state=0)

    assert sim.shape == (partition.length, partition.length)
    assert np.all(np.isfinite(sim))
    assert "labels_01_10" in partition._similarity_matrix_path.name
