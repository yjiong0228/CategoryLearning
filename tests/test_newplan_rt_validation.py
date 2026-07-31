from __future__ import annotations

import numpy as np

from src.Bayesian_state.utils.newplan_rt_validation import (
    cr1_standard_error,
    entropy_rows,
    jensen_shannon_rows,
    robust_location_scale,
    subject_bootstrap_interval,
    total_variation_rows,
)


def test_probability_distances_have_expected_boundaries() -> None:
    uniform = np.asarray([[0.5, 0.5], [0.5, 0.5]])
    certain = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    assert np.allclose(entropy_rows(uniform), np.log(2.0))
    assert np.allclose(jensen_shannon_rows(uniform, uniform), 0.0)
    assert np.allclose(
        jensen_shannon_rows(uniform, certain),
        jensen_shannon_rows(certain, uniform),
    )
    assert np.allclose(total_variation_rows(certain, certain), 0.0)
    assert np.allclose(
        total_variation_rows(certain[:1], certain[1:]),
        1.0,
    )


def test_robust_location_scale_resists_a_large_tail_value() -> None:
    location, scale = robust_location_scale([0.9, 1.0, 1.1, 1.2, 20.0])

    assert np.isclose(location, 1.1)
    assert 0.0 < scale < 1.0


def test_subject_bootstrap_interval_is_reproducible() -> None:
    first = subject_bootstrap_interval([1.0, 2.0, 3.0], seed=11)
    second = subject_bootstrap_interval([1.0, 2.0, 3.0], seed=11)

    assert first == second
    assert first[0] <= 2.0 <= first[1]


def test_cr1_standard_error_is_finite_for_multiple_clusters() -> None:
    x = np.column_stack([np.ones(8), np.arange(8, dtype=float)])
    y = 0.5 + 0.25 * x[:, 1] + np.asarray(
        [0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15, -0.15]
    )
    beta = np.linalg.pinv(x) @ y
    residuals = y - x @ beta
    groups = np.repeat(np.arange(4), 2)

    standard_error = cr1_standard_error(
        x,
        residuals,
        groups,
        coefficient_index=1,
    )

    assert np.isfinite(standard_error)
    assert standard_error > 0.0
