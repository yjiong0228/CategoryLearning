from __future__ import annotations

import numpy as np

from scripts.run_model_0815_p0_pf_nested_convergence import (
    compare_ensembles,
    summarize_ensemble,
    validate_panel,
)


GATES = {
    "maximum_choice_nll_change": 0.002,
    "maximum_choice_probability_rmse": 0.010,
    "maximum_executed_posterior_js": 0.020,
    "maximum_split_half_choice_probability_rmse": 0.010,
    "maximum_split_half_executed_posterior_js": 0.020,
    "minimum_median_post_choice_ess_fraction": 0.20,
}


def _panel(offset: float = 0.0) -> dict[str, np.ndarray]:
    p = np.asarray(
        [
            [0.70, 0.30],
            [0.30, 0.70],
            [0.68, 0.32],
            [0.32, 0.68],
        ],
        dtype=float,
    )
    probabilities = np.stack(
        [p + [[value, -value]] * 4 for value in (-0.003, 0.003, -0.002, 0.002)]
    )
    probabilities[:, :, 0] += offset
    probabilities[:, :, 1] -= offset
    executed = np.tile(np.asarray([[[0.8, 0.2]] * 4]), (4, 1, 1))
    return {
        "choice_probability": probabilities,
        "filtered_executed_probability": executed,
        "post_choice_ess": np.full((4, 4), 48.0),
        "filter_seed": np.asarray([10, 11, 12, 13], dtype=np.uint64),
        "repeat_index": np.arange(4),
        "observed_choice_index": np.asarray([0, 1, 0, 1]),
        "valid_trial_mask": np.ones(4, dtype=bool),
    }


def test_nested_ensemble_summary_retains_seed_and_state_dimensions() -> None:
    panel = validate_panel(_panel())
    row, arrays = summarize_ensemble(
        panel,
        subject_id=103,
        particle_count=64,
        seed_count=4,
        gates=GATES,
    )
    assert row["total_particle_trajectories"] == 256
    assert row["all_internal_gates_passed"] is True
    assert arrays["mean_choice_probability"].shape == (4, 2)
    assert arrays["mean_filtered_executed_probability"].shape == (4, 2)


def test_nested_ensemble_comparison_applies_all_three_gates() -> None:
    left_row, left = summarize_ensemble(
        _panel(),
        subject_id=103,
        particle_count=64,
        seed_count=4,
        gates=GATES,
    )
    right_row, right = summarize_ensemble(
        _panel(offset=0.001),
        subject_id=103,
        particle_count=128,
        seed_count=4,
        gates=GATES,
    )
    comparison = compare_ensembles(
        left_row,
        left,
        right_row,
        right,
        comparison_id="test",
        comparison_role="same_seed_count",
        gates=GATES,
    )
    assert comparison["all_comparison_gates_passed"] is True
    assert np.isclose(comparison["choice_probability_rmse"], 0.001)
