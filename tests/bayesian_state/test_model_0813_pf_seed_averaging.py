from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_model_0813_pf_seed_averaging import (
    aggregate_candidate_scores,
    replicated_log_likelihood,
    summarize_seed_averaging,
)


GATES = {
    "final_split_median_rank_spearman": 0.80,
    "final_split_minimum_rank_spearman": 0.50,
    "final_split_winner_agreement": 0.75,
    "final_split_mean_top_k_jaccard": 0.75,
    "running_8_to_16_median_rank_spearman": 0.90,
    "running_8_to_16_winner_agreement": 1.00,
    "median_effective_seed_fraction": 0.50,
    "median_aggregate_log_likelihood_mcse": 0.50,
    "maximum_noise_to_signal_ratio": 0.20,
}


def _stable_scores() -> pd.DataFrame:
    rows = []
    profile_log_likelihood = {"P01": -10.0, "P02": -11.0, "P03": -13.0}
    for dataset_id in ("D01", "D02"):
        for repeat in range(16):
            for profile_id, log_likelihood in profile_log_likelihood.items():
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "fit_profile_id": profile_id,
                        "particle_count": 128,
                        "filter_repeat": repeat,
                        "log_likelihood": log_likelihood,
                    }
                )
    return pd.DataFrame(rows)


def test_replicated_log_likelihood_reports_effective_seed_support():
    result = replicated_log_likelihood([-10.0, -10.0, -10.0, -10.0])

    assert result["aggregate_log_likelihood"] == pytest.approx(-10.0)
    assert result["aggregate_total_nll"] == pytest.approx(10.0)
    assert result["effective_seed_count"] == pytest.approx(4.0)
    assert result["effective_seed_fraction"] == pytest.approx(1.0)
    assert result["aggregate_log_likelihood_mcse"] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="finite"):
        replicated_log_likelihood([-10.0, np.nan])


def test_candidate_aggregation_uses_requested_complete_seed_panel():
    aggregate = aggregate_candidate_scores(
        _stable_scores(), range(4), panel_id="prefix_4"
    )

    assert len(aggregate) == 6
    assert aggregate["seed_count"].eq(4).all()
    np.testing.assert_allclose(aggregate["effective_seed_fraction"], 1.0)
    assert (
        aggregate.loc[
            aggregate["fit_profile_id"].eq("P01"), "aggregate_total_nll"
        ]
        .eq(10.0)
        .all()
    )

    incomplete = _stable_scores().iloc[1:].copy()
    with pytest.raises(ValueError, match="incomplete seed panel"):
        aggregate_candidate_scores(incomplete, range(4), panel_id="bad")


def test_seed_averaging_gate_and_equivalence_sets_are_paired():
    aggregate, split, running, equivalence, dataset, summary = (
        summarize_seed_averaging(
            _stable_scores(),
            aggregation_seed_counts=[2, 4, 8, 16],
            training_seed_indices=list(range(8)),
            validation_seed_indices=list(range(8, 16)),
            top_k=2,
            bootstrap_repeats=200,
            bootstrap_confidence=0.95,
            bootstrap_seed=20260814,
            gates=GATES,
        )
    )

    assert summary["all_stability_gates_pass"] is True
    assert summary["final_split_median_rank_spearman"] == pytest.approx(1.0)
    assert summary["running_8_to_16_winner_agreement"] == pytest.approx(1.0)
    assert summary["median_logmeanexp_vs_meanlog_rank_spearman"] == pytest.approx(
        1.0
    )
    assert summary["aggregation_method_winner_agreement"] == pytest.approx(1.0)
    assert dataset["equivalence_set_size"].eq(1).all()
    assert equivalence.loc[
        equivalence["fit_profile_id"].eq("P01"),
        "equivalent_to_selected_best",
    ].all()
    assert not equivalence.loc[
        equivalence["fit_profile_id"].ne("P01"),
        "equivalent_to_selected_best",
    ].any()
    assert set(split["comparison_seed_count"]) == {2, 4, 8, 16}
    assert set(running["comparison_seed_count"]) == {4, 8, 16}
    assert "training_8" in set(aggregate["panel_id"])
    assert "validation_8" in set(aggregate["panel_id"])
