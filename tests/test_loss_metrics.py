from __future__ import annotations

import numpy as np

from src.Bayesian_state.utils.optimizer_common import build_loss_strategy
from src.Bayesian_state.utils.optimization_config import resolve_loss_delta


def test_berhu_numeric_piecewise() -> None:
    strategy = build_loss_strategy("accuracy_curve_berhu", loss_delta=0.1)
    metrics = {
        "sliding_true_acc": np.asarray([0.0, 0.0], dtype=float),
        "sliding_pred_acc": np.asarray([0.05, 0.2], dtype=float),
    }
    got = strategy.compute(metrics)
    expected = (0.05 + ((0.2 ** 2 + 0.1 ** 2) / (2.0 * 0.1))) / 2.0
    assert np.isclose(got, expected)


def test_berhu_boundary_is_continuous() -> None:
    strategy = build_loss_strategy("accuracy_curve_berhu", loss_delta=0.1)
    metrics = {
        "sliding_true_acc": np.asarray([0.0], dtype=float),
        "sliding_pred_acc": np.asarray([0.1], dtype=float),
    }
    got = strategy.compute(metrics)
    assert np.isclose(got, 0.1)


def test_berhu_missing_delta_raises() -> None:
    try:
        _ = build_loss_strategy("accuracy_curve_berhu")
        assert False, "Expected ValueError for missing loss_delta with berhu"
    except ValueError as e:
        assert "loss_delta" in str(e)


def test_resolve_loss_delta_requires_positive_for_berhu() -> None:
    assert resolve_loss_delta({"loss_delta": 0.05}, "accuracy_curve_berhu") == 0.05
    try:
        _ = resolve_loss_delta({}, "accuracy_curve_berhu")
        assert False, "Expected ValueError for missing loss_delta"
    except ValueError as e:
        assert "loss_delta" in str(e)
    try:
        _ = resolve_loss_delta({"loss_delta": 0.0}, "accuracy_curve_berhu")
        assert False, "Expected ValueError for non-positive loss_delta"
    except ValueError as e:
        assert "loss_delta" in str(e)


def test_resolve_loss_delta_ignored_for_other_losses() -> None:
    assert resolve_loss_delta({}, "accuracy_curve_mse") is None
    assert resolve_loss_delta({"loss_delta": 0.5}, "accuracy_nll") is None
