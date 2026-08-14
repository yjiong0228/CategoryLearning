from __future__ import annotations

import pandas as pd
import pytest
import numpy as np

from scripts.run_model_0813_pf_calibration import (
    summarize_decomposition,
    summarize_ranking,
)


THRESHOLDS = {
    "required_seed_repeats": 3,
    "median_seed_rank_spearman": 0.80,
    "minimum_seed_rank_spearman": 0.50,
    "mean_modal_winner_agreement": 0.75,
    "median_candidate_total_nll_sd": 0.50,
    "maximum_noise_to_signal_ratio": 0.20,
    "median_cross_count_rank_spearman": 0.90,
    "cross_count_winner_agreement": 1.00,
}


def _ranking_frame() -> pd.DataFrame:
    rows = []
    candidate_nll = {"P01": 10.0, "P02": 12.0, "P03": 14.0}
    for particle_count, repeats in ((64, range(2)), (128, range(3))):
        for dataset_id in ("D01", "D02"):
            for repeat in repeats:
                for profile_id, total_nll in candidate_nll.items():
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "fit_profile_id": profile_id,
                            "particle_count": particle_count,
                            "filter_repeat": repeat,
                            "log_likelihood": -total_nll,
                        }
                    )
    return pd.DataFrame(rows)


def test_ranking_gate_requires_repeats_and_cross_count_stability():
    particle, correlations, cross, winners, summary = summarize_ranking(
        _ranking_frame(), THRESHOLDS
    )

    by_count = particle.set_index("particle_count")
    assert not bool(by_count.loc[64, "all_stability_gates_pass"])
    assert bool(by_count.loc[128, "all_stability_gates_pass"])
    assert summary["minimum_stable_particle_count"] == 128
    assert len(correlations) == 2 * (1 + 3)
    assert len(cross) == 2
    assert len(winners) == 2 * (2 + 3)
    assert by_count.loc[128, "median_cross_count_rank_spearman"] == pytest.approx(
        1.0
    )


def _decomposition_frame() -> pd.DataFrame:
    rows = []
    mode_nll = {
        "unweighted_mixture": 0.70,
        "importance_no_resampling": 0.60,
        "full_particle_filter": 0.55,
    }
    for subject_id in (103, 120):
        for repeat in range(3):
            for mode_id, mean_nll in mode_nll.items():
                rows.append(
                    {
                        "subject_id": subject_id,
                        "filter_repeat": repeat,
                        "mode_id": mode_id,
                        "mode_label": mode_id,
                        "mean_nll": mean_nll,
                        "mean_pre_choice_ess_fraction": 1.0,
                        "mean_post_choice_ess_fraction": 0.5,
                        "terminal_post_choice_ess_fraction": 0.4,
                        "final_weight_ess_fraction": 0.3,
                        "resampling_fraction": 0.2,
                        "mean_unique_ancestor_fraction_on_resampled_trials": 0.6,
                        "runtime_seconds": 1.0,
                        "particle_count": 128,
                        "trial_count": 128,
                    }
                )
    return pd.DataFrame(rows)


def test_filter_decomposition_uses_paired_seed_gain_direction():
    subject_mode, contrasts, summary = summarize_decomposition(
        _decomposition_frame()
    )

    assert len(subject_mode) == 6
    assert len(contrasts) == 6
    np.testing.assert_allclose(contrasts["choice_weighting_gain"], 0.10)
    np.testing.assert_allclose(contrasts["resampling_gain"], 0.05)
    np.testing.assert_allclose(contrasts["full_filter_gain"], 0.15)
    assert summary["contrasts"]["choice_weighting_gain"][
        "positive_pair_fraction"
    ] == pytest.approx(1.0)


def test_filter_decomposition_rejects_missing_mode():
    incomplete = _decomposition_frame().loc[
        lambda frame: frame["mode_id"].ne("full_particle_filter")
    ]
    with pytest.raises(ValueError, match="three declared modes"):
        summarize_decomposition(incomplete)
