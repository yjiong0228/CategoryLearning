from __future__ import annotations

import numpy as np

from src.Bayesian_state.run_amr_optimization import resolve_loss_delta as resolve_loss_delta_amr
from src.Bayesian_state.run_grid_optimization import resolve_loss_delta as resolve_loss_delta_grid
from src.Bayesian_state.utils.optimizer_common import build_loss_strategy


def test_berhu_numeric_piecewise() -> None:
    strategy = build_loss_strategy("berhu", loss_delta=0.1)
    metrics = {
        "sliding_true_acc": np.asarray([0.0, 0.0], dtype=float),
        "sliding_pred_acc": np.asarray([0.05, 0.2], dtype=float),
    }
    got = strategy.compute(metrics)
    expected = (0.05 + ((0.2 ** 2 + 0.1 ** 2) / (2.0 * 0.1))) / 2.0
    assert np.isclose(got, expected)


def test_berhu_boundary_is_continuous() -> None:
    strategy = build_loss_strategy("berhu", loss_delta=0.1)
    metrics = {
        "sliding_true_acc": np.asarray([0.0], dtype=float),
        "sliding_pred_acc": np.asarray([0.1], dtype=float),
    }
    got = strategy.compute(metrics)
    assert np.isclose(got, 0.1)


def test_berhu_missing_delta_raises() -> None:
    try:
        _ = build_loss_strategy("berhu")
        assert False, "Expected ValueError for missing loss_delta with berhu"
    except ValueError as e:
        assert "loss_delta" in str(e)


def test_resolve_loss_delta_requires_positive_for_berhu() -> None:
    for resolver in (resolve_loss_delta_grid, resolve_loss_delta_amr):
        assert resolver({"loss_delta": 0.05}, "berhu") == 0.05
        try:
            _ = resolver({}, "berhu")
            assert False, "Expected ValueError for missing loss_delta"
        except ValueError as e:
            assert "loss_delta" in str(e)
        try:
            _ = resolver({"loss_delta": 0.0}, "berhu")
            assert False, "Expected ValueError for non-positive loss_delta"
        except ValueError as e:
            assert "loss_delta" in str(e)


def test_resolve_loss_delta_ignored_for_other_losses() -> None:
    for resolver in (resolve_loss_delta_grid, resolve_loss_delta_amr):
        assert resolver({}, "mse") is None
        assert resolver({"loss_delta": 0.5}, "nll") is None
