from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.problems.discrete_partitions import DiscreteRulePartition
from src.Bayesian_state.problems.modules.hypo_transitions import DynamicHypothesisModule
from src.Bayesian_state.utils.optimizer_common import derive_run_seed


def _engine(set_size: int = 8, posterior: np.ndarray | None = None, partition=None) -> SimpleNamespace:
    if posterior is None:
        posterior = np.full(set_size, 1.0 / set_size)
    return SimpleNamespace(
        set_size=set_size,
        prior=np.asarray(posterior, dtype=float),
        posterior=np.asarray(posterior, dtype=float),
        hypotheses_mask=None,
        partition=partition,
        modules={},
        observation=None,
    )


def _module(strategies, **kwargs) -> DynamicHypothesisModule:
    return DynamicHypothesisModule(
        _engine(**kwargs.pop("engine_kwargs", {})),
        strategies=strategies,
        init_num=0,
        **kwargs,
    )


def test_strategy_requires_explicit_pool() -> None:
    with pytest.raises(ValueError, match="missing required key.*pool"):
        _module([{"amount": "fixed", "method": "random", "value": 1}])


@pytest.mark.parametrize(
    "strategy, match",
    [
        ({"amount": "fixed", "method": "random", "pool": "bad", "value": 1}, "unsupported pool"),
        ({"amount": "fixed", "method": "bad", "pool": "active", "value": 1}, "unsupported method"),
        ({"amount": "bad", "method": "random", "pool": "active"}, "unsupported amount"),
        ({"amount": "fixed", "method": "random", "pool": "active", "value": 1.5}, "non-integer"),
    ],
)
def test_invalid_strategy_config_raises(strategy, match) -> None:
    with pytest.raises(ValueError, match=match):
        _module([strategy])


def test_pool_resolution_uses_previous_active_and_current_selected() -> None:
    mod = _module([{"amount": "fixed", "method": "random", "pool": "all_unselected", "value": 1}])
    mod.old_active = np.array([1, 2, 3], dtype=int)
    selected = {2, 5}

    assert mod._resolve_pool("active", selected).tolist() == [1, 3]
    assert mod._resolve_pool("inactive", selected).tolist() == [0, 4, 6, 7]
    assert mod._resolve_pool("all_unselected", selected).tolist() == [0, 1, 3, 4, 6, 7]


def test_transition_uses_pool_for_retention_and_exploration() -> None:
    posterior = np.array([0.01, 0.05, 0.10, 0.20, 0.30, 0.25, 0.06, 0.03])
    strategies = [
        {"amount": "fixed", "method": "top_posterior", "pool": "active", "value": 2},
        {"amount": "fixed", "method": "top_posterior", "pool": "inactive", "value": 2},
    ]
    mod = DynamicHypothesisModule(
        _engine(set_size=8, posterior=posterior),
        strategies=strategies,
        init_num=0,
        max_active_hypotheses=4,
    )
    mod.active = np.array([1, 2, 3], dtype=int)

    mod._transition()

    assert mod.active.tolist() == [2, 3, 4, 5]


def test_same_seed_reproducible_and_derived_repeats_differ() -> None:
    strategies = [{"amount": "fixed", "method": "random", "pool": "all_unselected", "value": 4}]
    first = _module(strategies, random_seed=123)
    second = _module(strategies, random_seed=123)

    assert first._sample_from_pool(np.arange(20), 6).tolist() == second._sample_from_pool(np.arange(20), 6).tolist()
    assert derive_run_seed(123, 7, {"gamma": 0.1, "w0": 0.2}, "grid", 0) == derive_run_seed(
        123, 7, {"w0": 0.2, "gamma": 0.1}, "grid", 0
    )
    assert derive_run_seed(123, 7, {"gamma": 0.1, "w0": 0.2}, "grid", 0) != derive_run_seed(
        123, 7, {"gamma": 0.1, "w0": 0.2}, "grid", 1
    )


def test_ksimilar_centers_rejects_discrete_rule_partition() -> None:
    partition = DiscreteRulePartition(n_dims=3, n_cats=2)
    strategies = [
        {
            "amount": "fixed",
            "method": "ksimilar_centers",
            "pool": "all_unselected",
            "value": 1,
        }
    ]

    with pytest.raises(ValueError, match="prototype-backed partition"):
        DynamicHypothesisModule(
            _engine(set_size=partition.length, partition=partition),
            strategies=strategies,
            init_num=0,
        )


def test_nonfinite_posterior_raises() -> None:
    strategies = [{"amount": "fixed", "method": "random", "pool": "all_unselected", "value": 1}]
    mod = DynamicHypothesisModule(
        _engine(set_size=3, posterior=np.array([0.5, np.nan, 0.5])),
        strategies=strategies,
        init_num=0,
    )

    with pytest.raises(ValueError, match="non-finite"):
        mod._transition()
