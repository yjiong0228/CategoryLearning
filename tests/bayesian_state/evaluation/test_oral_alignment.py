from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    assert diagnostics["oral_encoder_version"] == "fixed_likelihood_category_state_v2"
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
    assert restored[101]["instantaneous_oral_mass"].shape == (2, 29)
    assert restored[101]["valid_oral_report"] == [True, True]
    assert restored[101]["oral_encoder_version"] == "fixed_likelihood_category_state_v2"
    assert restored[101]["oral_distribution_method"] == "gaussian_component_mixture"
    assert restored[101]["oral_state_mode"] == "latest_by_category"
    assert (
        restored[101]["oral_aggregation_method"]
        == "latest_by_category_likelihood_product"
    )
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
    assert diagnostics["valid_oral"].tolist() == [True, True]
    assert diagnostics["valid_oral_state"].tolist() == [True, True]
    assert diagnostics["valid_oral_report"].tolist() == [True, False]
    assert diagnostics.loc[0, "oral_distribution_method"] == "gaussian_component_mixture"
    assert np.isfinite(diagnostics.loc[0, "oral_log_evidence"])
    assert np.isfinite(diagnostics.loc[1, "oral_log_evidence"])
    assert np.isnan(diagnostics.loc[1, "instantaneous_oral_log_evidence"])
    assert diagnostics["oral_state_observed_categories"].tolist() == [1.0, 1.0]


def test_latest_category_state_multiplies_categories_and_replaces_old_report():
    oral_df = pd.DataFrame(
        {
            "iSub": [101, 101, 101],
            "condition": [1, 1, 1],
            "choice": [1, 2, 1],
            "oral_center": [
                "[0.25, 0.5, 0.5, 0.5]",
                "[0.75, 0.5, 0.5, 0.5]",
                "[0.25, 0.25, 0.5, 0.5]",
            ],
        }
    )
    evaluator = ModelEvaluator()
    results = evaluator.compute_oral_mass_probabilities(oral_df, oral_mode="center")
    info = results[101]
    instantaneous = info["instantaneous_oral_mass"]
    state = info["oral_mass"]

    expected_second = instantaneous[0] * instantaneous[1]
    expected_second /= expected_second.sum()
    expected_third = instantaneous[2] * instantaneous[1]
    expected_third /= expected_third.sum()
    incorrectly_accumulated = instantaneous[0] * instantaneous[1] * instantaneous[2]
    incorrectly_accumulated /= incorrectly_accumulated.sum()

    assert np.allclose(state[0], instantaneous[0])
    assert np.allclose(state[1], expected_second)
    assert np.allclose(state[2], expected_third)
    assert not np.allclose(state[2], incorrectly_accumulated)
    assert state[1, 0] > instantaneous[1, 0]
    assert info["oral_state_observed_categories"].tolist() == [1.0, 2.0, 2.0]
    assert info["oral_state_category_mask"].tolist() == [1.0, 3.0, 3.0]


def test_instantaneous_state_mode_reproduces_current_report_only_behavior():
    oral_df = pd.DataFrame(
        {
            "iSub": [101, 101],
            "condition": [1, 1],
            "choice": [1, 2],
            "oral_center": [
                "[0.25, 0.5, 0.5, 0.5]",
                "[0.75, 0.5, 0.5, 0.5]",
            ],
        }
    )
    evaluator = ModelEvaluator()
    results = evaluator.compute_oral_mass_probabilities(
        oral_df,
        oral_state_mode="instantaneous",
    )
    info = results[101]

    assert np.allclose(info["oral_mass"], info["instantaneous_oral_mass"])
    assert info["oral_state_mode"] == "instantaneous"
    assert info["oral_aggregation_method"] == "current_report_only"


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
        assert set(frame["oral_state_mode"]) == {"latest_by_category"}
        assert set(frame["oral_aggregation_method"]) == {
            "latest_by_category_likelihood_product"
        }
        assert set(frame["oral_center_sigma"]) == {0.10}


def test_target_alignment_averages_pf_repeat_marginal_priors(monkeypatch):
    oral_df = pd.DataFrame(
        {
            "iSub": [101, 101],
            "condition": [1, 1],
            "choice": [1, 2],
            "oral_center": [
                "[0.25, 0.5, 0.5, 0.5]",
                "[0.75, 0.5, 0.5, 0.5]",
            ],
        }
    )
    low = np.full((2, 29), 0.8 / 28.0)
    high = np.full((2, 29), 0.2 / 28.0)
    low[:, 0] = 0.2
    high[:, 0] = 0.8
    model_results = {
        101: {
            "condition": 1,
            "target_hypothesis": 0,
            "prior_log": low.tolist(),
            "state_distribution_kind": "particle_marginal",
        }
    }
    evaluator = ModelEvaluator()
    monkeypatch.setattr(
        evaluator,
        "_extract_prior_repeat_logs",
        lambda info: ([low, high], "pf_repeat_mean_marginal_prior"),
    )

    result = evaluator.compute_target_based_alignment(
        model_results,
        oral_df,
        subjects=[101],
        alignment_spaces=("full",),
    )

    np.testing.assert_allclose(result["model_target_prior"], 0.5)
    assert set(result["model_target_n_pf_runs"]) == {2}
    assert set(result["model_state_source"]) == {
        "pf_repeat_mean_marginal_prior"
    }
    np.testing.assert_allclose(
        result["model_target_repeat_sd"],
        np.std([0.2, 0.8], ddof=1),
    )


def test_target_alignment_precomputes_compact_trajectory_band(monkeypatch):
    oral_df = pd.DataFrame(
        {
            "iSub": [101, 101],
            "condition": [1, 1],
            "choice": [1, 2],
            "oral_center": [
                "[0.25, 0.5, 0.5, 0.5]",
                "[0.75, 0.5, 0.5, 0.5]",
            ],
        }
    )
    low = np.full((2, 29), 0.8 / 28.0)
    high = np.full((2, 29), 0.2 / 28.0)
    low[:, 0] = 0.2
    high[:, 0] = 0.8
    model_results = {
        101: {
            "condition": 1,
            "target_hypothesis": 0,
            "prior_log": low.tolist(),
            "state_distribution_kind": "trajectory",
        }
    }
    evaluator = ModelEvaluator()
    monkeypatch.setattr(
        evaluator,
        "_extract_prior_repeat_logs",
        lambda info: ([low, high], "trajectory_repeat_prior_ensemble"),
    )

    result = evaluator.compute_target_based_alignment(
        model_results,
        oral_df,
        subjects=[101],
        alignment_spaces=("full",),
        trajectory_band_window_size=2,
    )

    assert "model_target_repeat_probabilities" not in result
    assert set(result["model_inference_backend"]) == {"trajectory"}
    assert set(result["model_target_band_n_runs"]) == {2}
    assert set(result["model_target_band_type"]) == {
        OralAlignmentScoringMixin.TRAJECTORY_TARGET_BAND_TYPE
    }
    np.testing.assert_allclose(result["model_target_prior"], 0.5)
    assert np.isnan(result.loc[0, "model_target_expected_rolling"])
    assert np.isclose(result.loc[1, "model_target_expected_rolling"], 0.5)

    attached = evaluator._attach_target_sampling_bands(result, window_size=2)
    np.testing.assert_allclose(
        attached["model_target_q05_rolling"],
        result["model_target_q05_rolling"],
        equal_nan=True,
    )


def test_target_sampling_band_matches_conditional_bernoulli_protocol():
    probabilities = np.full(32, 0.5, dtype=float)

    first = OralAlignmentScoringMixin.compute_target_sampling_band(
        probabilities,
        window_size=4,
        n_draws=4000,
        seed=23,
    )
    second = OralAlignmentScoringMixin.compute_target_sampling_band(
        probabilities,
        window_size=4,
        n_draws=4000,
        seed=23,
    )

    assert first["band_type"] == (
        "observed_history_conditional_latent_target_occupancy"
    )
    assert np.isnan(first["expected"][:3]).all()
    np.testing.assert_allclose(first["expected"][3:], 0.5)
    for key in ("q05", "q25", "q50", "q75", "q95"):
        np.testing.assert_allclose(first[key], second[key], equal_nan=True)
    assert np.nanmean(first["q95"] - first["q05"]) > 0.0


def test_trajectory_target_band_uses_repeat_ensemble():
    probability_runs = np.asarray(
        [
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )

    band = OralAlignmentScoringMixin.compute_trajectory_target_band(
        probability_runs,
        window_size=2,
    )

    assert band["band_type"] == (
        "observed_history_conditional_trajectory_repeat_target_mass"
    )
    assert band["n_runs"] == 2
    np.testing.assert_allclose(
        band["expected"],
        [np.nan, 0.5, 0.5, 0.5],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        band["q05"],
        [np.nan, 0.05, 0.5, 0.05],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        band["q95"],
        [np.nan, 0.95, 0.5, 0.95],
        equal_nan=True,
    )


def test_trajectory_target_plot_uses_trajectory_labels_and_run_count():
    evaluator = ModelEvaluator()
    raw = pd.DataFrame(
        [
            {
                "subject": 101,
                "condition": 1,
                "trial": trial,
                "trial_pct": trial / 8.0,
                "oral_mode": "center",
                "alignment_space": "active",
                "model_inference_backend": "trajectory",
                "model_target_n_runs": 2,
                "model_target_repeat_probabilities": (
                    trial / 8.0,
                    1.0 - trial / 8.0,
                ),
                "model_target_prior": 0.5,
                "oral_target_mass": trial / 8.0,
            }
            for trial in range(1, 9)
        ]
    )

    plotted = evaluator._attach_target_sampling_bands(raw, window_size=2)
    assert set(plotted["model_target_band_n_runs"].dropna()) == {2.0}
    assert plotted["model_target_band_n_draws"].isna().all()
    fig = evaluator.plot_target_based_alignment_subjectwise(
        plotted,
        alignment_space="active",
        window_size=2,
    )

    first = fig.axes[0]
    assert first.get_title() == "Subject 101 | Trajectory runs=2"
    assert first.lines[0].get_label() == "Trajectory mean target mass"
    assert "2 trajectory runs" in fig._suptitle.get_text()
    plt.close(fig)


def test_target_subject_plot_matches_pf_accuracy_band_style():
    rows = []
    for subject in (101, 102):
        for trial in range(1, 21):
            expected = trial / 20.0
            rows.append(
                {
                    "subject": subject,
                    "condition": 1,
                    "trial": trial,
                    "trial_pct": trial / 20.0,
                    "oral_mode": "center",
                    "alignment_space": "active",
                    "model_target_prior": expected,
                    "oral_target_mass": 1.0 - expected,
                    "model_target_expected_rolling": expected,
                    "model_target_q05_rolling": max(0.0, expected - 0.20),
                    "model_target_q25_rolling": max(0.0, expected - 0.10),
                    "model_target_q50_rolling": expected,
                    "model_target_q75_rolling": min(1.0, expected + 0.10),
                    "model_target_q95_rolling": min(1.0, expected + 0.20),
                    "model_target_band_n_draws": 5000,
                    "model_target_n_pf_runs": 4,
                }
            )
    evaluator = ModelEvaluator()
    fig = evaluator.plot_target_based_alignment_subjectwise(
        pd.DataFrame(rows),
        alignment_space="active",
        window_size=4,
    )

    first, second = fig.axes[:2]
    assert fig.get_size_inches().tolist() == [8.4, 3.1]
    assert first.get_title() == "Subject 101 | PF runs=4"
    assert first.get_xlabel() == "Trial"
    assert first.get_ylabel() == "Rolling target probability"
    assert first.lines[0].get_color() == "#E69F00"
    assert first.lines[0].get_linewidth() == 2.0
    assert first.lines[1].get_color() == "#111111"
    assert first.lines[1].get_linewidth() == 2.1
    assert first.get_legend() is not None
    assert second.get_legend() is None
    assert "50% and 90% pointwise intervals | 5,000 draws" in (
        fig._suptitle.get_text()
    )
    assert not first.spines["top"].get_visible()
    assert not first.spines["right"].get_visible()
    plt.close(fig)
