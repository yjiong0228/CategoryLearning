from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_cond1_mechanism_recovery import (
    candidate_grid,
    cross_candidate_bank,
    summarize_recovery,
)


def test_cross_bank_deduplicates_the_common_reference() -> None:
    grid = candidate_grid(("F", "M", "H", "P"), shared_theta=0.75)
    bank = cross_candidate_bank(grid)
    assert bank[0].candidate_id == "BASE"
    assert sum(candidate.is_reference for candidate in bank) == 1
    assert len(bank) == 16
    assert len({candidate.candidate_id for candidate in bank}) == len(bank)


def test_within_recovery_summary_selects_the_highest_evidence() -> None:
    rows = []
    for true_id, true_value, best_id in (("F_low", 0.5, "F_low"), ("F_high", 1.5, "F_high")):
        for fit_id, fit_value in (("F_low", 0.5), ("F_high", 1.5)):
            rows.append(
                {
                    "subject_id": 1,
                    "replicate": 0,
                    "true_family": "F",
                    "true_candidate_id": true_id,
                    "true_value": true_value,
                    "fit_family": "F",
                    "fit_candidate_id": fit_id,
                    "fit_value": fit_value,
                    "prefix_log_predictive_density": 0.0 if fit_id == best_id else -10.0,
                    "generated_accuracy": 0.75,
                }
            )
    recovered, summary, confusion = summarize_recovery(
        pd.DataFrame(rows), scope="within"
    )
    assert recovered["exact_candidate_recovered"].all()
    assert np.isclose(summary.loc[0, "exact_candidate_accuracy"], 1.0)
    assert confusion.loc[0, "predicted_family"] == "F"


def test_cross_recovery_uses_mean_evidence_within_family() -> None:
    frame = pd.DataFrame(
        [
            {
                "subject_id": 1,
                "replicate": 0,
                "true_family": "H",
                "true_candidate_id": "H3",
                "true_value": 3.0,
                "fit_family": "H",
                "fit_candidate_id": "H3",
                "fit_value": 3.0,
                "prefix_log_predictive_density": 0.0,
                "generated_accuracy": 0.7,
            },
            {
                "subject_id": 1,
                "replicate": 0,
                "true_family": "H",
                "true_candidate_id": "H3",
                "true_value": 3.0,
                "fit_family": "F",
                "fit_candidate_id": "F1",
                "fit_value": 1.0,
                "prefix_log_predictive_density": -3.0,
                "generated_accuracy": 0.7,
            },
            {
                "subject_id": 1,
                "replicate": 0,
                "true_family": "H",
                "true_candidate_id": "H3",
                "true_value": 3.0,
                "fit_family": "F",
                "fit_candidate_id": "F2",
                "fit_value": 2.0,
                "prefix_log_predictive_density": -3.0,
                "generated_accuracy": 0.7,
            },
        ]
    )
    recovered, summary, _ = summarize_recovery(frame, scope="cross")
    assert recovered.loc[0, "predicted_family"] == "H"
    assert bool(recovered.loc[0, "family_recovered"])
    assert np.isclose(summary.loc[0, "family_accuracy"], 1.0)
