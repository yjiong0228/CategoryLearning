from __future__ import annotations

import numpy as np
import pandas as pd

from src.Bayesian_state.evaluation.evaluator import ModelEvaluator
from src.Bayesian_state.evaluation.oral.reporting import OralAlignmentReportingMixin
from src.Bayesian_state.evaluation.oral.scoring import OralAlignmentScoringMixin
from src.Bayesian_state.hypothesis_space import ContinuousPartition


def _manual_gaussian_component_mixture(partition, center, choice, sigma):
    cat_idx = int(choice) - 1
    log_likelihood = []
    for hypo_idx in range(partition.length):
        prototypes = partition.get_category_prototypes(hypo_idx, cat_idx)
        squared_distance = np.sum((prototypes - center) ** 2, axis=1)
        component = -squared_distance / (2.0 * sigma ** 2)
        maximum = float(np.max(component))
        log_likelihood.append(
            maximum
            + np.log(np.sum(np.exp(component - maximum)))
            - np.log(len(prototypes))
        )
    log_likelihood = np.asarray(log_likelihood, dtype=float)
    weights = np.exp(log_likelihood - np.max(log_likelihood))
    return weights / weights.sum()


def test_center_oral_encoder_is_normalized_fixed_gaussian_mixture():
    partition = ContinuousPartition(n_dims=4, n_cats=2)
    center = partition.get_category_prototypes(0, 0)[0]
    sigma = 0.10

    actual, diagnostics = OralAlignmentScoringMixin._center_oral_distribution(
        center,
        choice=1,
        partition=partition,
        center_sigma=sigma,
        return_diagnostics=True,
    )
    expected = _manual_gaussian_component_mixture(partition, center, choice=1, sigma=sigma)

    assert actual.shape == (29,)
    assert np.isclose(actual.sum(), 1.0)
    assert np.all(actual >= 0.0)
    assert np.allclose(actual, expected)
    assert int(np.argmax(actual)) == 0
    assert diagnostics["oral_encoder_version"] == "fixed_likelihood_v1"
    assert diagnostics["oral_distribution_method"] == "gaussian_component_mixture"
    assert diagnostics["oral_hypothesis_prior"] == "uniform_hypothesis"
    assert diagnostics["oral_center_sigma"] == sigma
    assert diagnostics["hypothesis_space_version"] == "continuous_fixed_labels_v1"
    assert diagnostics["oral_min_distance"] == 0.0
    assert diagnostics["oral_fit_score"] == 1.0


def test_center_oral_encoder_concentration_depends_on_fixed_sigma():
    partition = ContinuousPartition(n_dims=4, n_cats=2)
    center = partition.get_category_prototypes(0, 0)[0]

    narrow, narrow_diagnostics = OralAlignmentScoringMixin._center_oral_distribution(
        center,
        choice=1,
        partition=partition,
        center_sigma=0.05,
        return_diagnostics=True,
    )
    wide, wide_diagnostics = OralAlignmentScoringMixin._center_oral_distribution(
        center,
        choice=1,
        partition=partition,
        center_sigma=0.20,
        return_diagnostics=True,
    )

    assert narrow[0] > wide[0]
    assert narrow_diagnostics["oral_effective_hypotheses"] < wide_diagnostics[
        "oral_effective_hypotheses"
    ]
    assert not hasattr(OralAlignmentScoringMixin, "_adaptive_softmax_from_distances")


def test_invalid_center_remains_missing_instead_of_becoming_uniform():
    partition = ContinuousPartition(n_dims=4, n_cats=2)

    probability, diagnostics = OralAlignmentScoringMixin._center_oral_distribution(
        np.full(4, np.nan),
        choice=1,
        partition=partition,
        center_sigma=0.10,
        return_diagnostics=True,
    )

    assert np.isnan(probability).all()
    assert np.isnan(diagnostics["oral_min_distance"])
    assert np.isnan(diagnostics["oral_log_evidence"])


def test_region_oral_encoder_uses_fixed_mismatch_scale_and_normalizes():
    partition = ContinuousPartition(n_dims=4, n_cats=2)
    oral_region = partition.hypothesis_space[0].categories[0].components[0]

    narrow, narrow_diagnostics = OralAlignmentScoringMixin._region_oral_distribution(
        oral_region,
        choice=1,
        partition=partition,
        n_samples=2000,
        random_state=7,
        region_temperature=0.05,
        return_diagnostics=True,
    )
    wide, wide_diagnostics = OralAlignmentScoringMixin._region_oral_distribution(
        oral_region,
        choice=1,
        partition=partition,
        n_samples=2000,
        random_state=7,
        region_temperature=0.20,
        return_diagnostics=True,
    )

    assert np.isclose(narrow.sum(), 1.0)
    assert np.isclose(wide.sum(), 1.0)
    assert narrow_diagnostics["oral_distribution_method"] == "fixed_iou_energy"
    assert narrow_diagnostics["oral_fit_score"] == 1.0
    assert narrow_diagnostics["oral_effective_hypotheses"] < wide_diagnostics[
        "oral_effective_hypotheses"
    ]


def test_oral_mass_npz_round_trip_preserves_encoder_provenance(tmp_path):
    oral_df = pd.DataFrame(
        {
            "iSub": [101, 101],
            "condition": [1, 1],
            "choice": [1, 2],
            "oral_center": ["[0.25, 0.5, 0.5, 0.5]", "[0.75, 0.5, 0.5, 0.5]"],
        }
    )
    evaluator = OralAlignmentReportingMixin()
    results = evaluator.compute_oral_mass_probabilities(
        oral_df,
        oral_mode="center",
        oral_center_sigma=0.10,
    )
    path = tmp_path / "oral_mass_probabilities.npz"
    evaluator.save_oral_mass_probabilities(results, path)
    restored = evaluator.load_oral_mass_probabilities(path)

    assert restored[101]["oral_mass"].shape == (2, 29)
    assert np.allclose(restored[101]["oral_mass"].sum(axis=1), 1.0)
    assert restored[101]["oral_encoder_version"] == "fixed_likelihood_v1"
    assert restored[101]["oral_distribution_method"] == "gaussian_component_mixture"
    assert restored[101]["oral_center_sigma"] == 0.10
    assert restored[101]["hypothesis_space_version"] == "continuous_fixed_labels_v1"
    assert restored[101]["oral_min_distance"].shape == (2,)
    assert np.isfinite(restored[101]["oral_distribution_entropy"]).all()


def test_oral_mass_diagnostic_csv_has_one_row_per_trial(tmp_path):
    oral_df = pd.DataFrame(
        {
            "iSub": [101, 101],
            "condition": [1, 1],
            "choice": [1, 2],
            "oral_center": ["[0.25, 0.5, 0.5, 0.5]", np.nan],
        }
    )
    evaluator = OralAlignmentReportingMixin()
    results = evaluator.compute_oral_mass_probabilities(oral_df, oral_mode="center")
    path = tmp_path / "oral_mass_diagnostics.csv"
    evaluator.save_oral_mass_diagnostics(results, path)
    diagnostics = pd.read_csv(path)

    assert len(diagnostics) == 2
    assert diagnostics["valid_oral"].tolist() == [True, False]
    assert diagnostics.loc[0, "oral_distribution_method"] == "gaussian_component_mixture"
    assert np.isfinite(diagnostics.loc[0, "oral_log_evidence"])
    assert np.isnan(diagnostics.loc[1, "oral_log_evidence"])


def test_all_five_based_families_share_the_precomputed_oral_distribution():
    oral_df = pd.DataFrame(
        {
            "iSub": [101, 101],
            "condition": [1, 1],
            "choice": [1, 2],
            "oral_center": ["[0.25, 0.5, 0.5, 0.5]", "[0.75, 0.5, 0.5, 0.5]"],
            "feature1": [0.2, 0.8],
            "feature2": [0.5, 0.5],
            "feature3": [0.5, 0.5],
            "feature4": [0.5, 0.5],
        }
    )
    prior = np.full(29, 1.0 / 29.0)
    model_results = {
        101: {
            "condition": 1,
            "target_hypothesis": 0,
            "prior_log": [prior.copy(), prior.copy()],
        }
    }
    evaluator = ModelEvaluator()
    oral_mass = evaluator.compute_oral_mass_probabilities(
        oral_df,
        subjects=[101],
        oral_center_sigma=0.10,
    )

    distribution = evaluator.compute_distribution_based_alignment(
        model_results,
        oral_df,
        subjects=[101],
        oral_mass_results=oral_mass,
        alignment_spaces=("full",),
    )
    oral_based = evaluator.compute_oral_based_alignment(
        model_results,
        oral_df,
        subjects=[101],
        model_distribution="prior",
    )
    target = evaluator.compute_target_based_alignment(
        model_results,
        oral_df,
        subjects=[101],
        oral_mass_results=oral_mass,
        alignment_spaces=("full",),
    )
    hit = evaluator.compute_hit_based_alignment(
        model_results,
        oral_df,
        subjects=[101],
        oral_mass_results=oral_mass,
        rank_top_k=3,
    )
    coverage = evaluator.compute_coverage_based_alignment(
        model_results,
        oral_df,
        subjects=[101],
        oral_mass_results=oral_mass,
    )

    assert len(distribution) == 2
    assert len(oral_based) == 2
    assert len(target) == 2
    assert len(hit) == 2
    assert len(coverage) == 2
    expected_target_mass = oral_mass[101]["oral_mass"][:, 0]
    assert np.allclose(target["oral_target_mass"], expected_target_mass)
    target_summary = evaluator.summarize_target_based_alignment(target)
    assert set(target_summary["oral_distribution_method"]) == {
        "gaussian_component_mixture"
    }
    for frame in (distribution, target, hit, coverage):
        assert set(frame["oral_distribution_method"]) == {"gaussian_component_mixture"}
        assert set(frame["oral_center_sigma"]) == {0.10}
