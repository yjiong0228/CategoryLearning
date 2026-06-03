from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.problems.discrete_partitions import DiscreteRulePartition
from src.Bayesian_state.problems.modules.hypo_transitions import DynamicHypothesisModule
from src.Bayesian_state.utils.optimizer_common import derive_run_seed


class FakePartition:
    def __init__(self, n_cats: int = 2, similarity_matrix: np.ndarray | None = None):
        self.n_cats = n_cats
        self.similarity_matrix = similarity_matrix


class FakePrototypePartition(FakePartition):
    def __init__(self, set_size: int = 8, n_cats: int = 2, n_dims: int = 2):
        super().__init__(n_cats=n_cats, similarity_matrix=np.eye(set_size))
        self.n_dims = n_dims
        base = np.arange(set_size * n_cats * n_dims, dtype=float).reshape(set_size, n_cats, n_dims)
        self.prototypes = base[:, None, :, :] / float(set_size * n_cats * n_dims)


class FarPrototypePartition(FakePartition):
    def __init__(self, set_size: int = 4):
        super().__init__(n_cats=1, similarity_matrix=np.eye(set_size))
        self.n_dims = 1
        self.prototypes = np.zeros((set_size, 1, 1, 1), dtype=float)
        self.prototypes[1:, 0, 0, 0] = 1000.0


class LazySimilarityPartition:
    def __init__(self, matrix: np.ndarray):
        self.n_cats = 2
        self._matrix = matrix
        self.access_count = 0

    @property
    def similarity_matrix(self) -> np.ndarray:
        self.access_count += 1
        return self._matrix


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


def test_strategy_without_pool_uses_method_default() -> None:
    mod = _module([{"amount": "fixed", "method": "random", "value": 1}])

    assert mod.strategies[0]["pool"] == DynamicHypothesisModule.POOL_ALL_UNSELECTED


@pytest.mark.parametrize(
    "strategy, match",
    [
        ({"amount": "fixed", "method": "random", "pool": "bad", "value": 1}, "unsupported pool"),
        ({"amount": "fixed", "method": "bad", "pool": "active", "value": 1}, "unsupported method"),
        ({"amount": "bad", "method": "random", "pool": "active"}, "unsupported amount"),
        ({"amount": "fixed", "method": "random", "pool": "active", "value": 1.5}, "non-integer"),
        ({"amount": "fixed", "method": "top_posterior", "pool": "active", "value": 1, "top_p_scope": "bad"}, "top_p_scope"),
        ({"amount": "fixed", "method": "epsilon_posterior", "pool": "active", "value": 1, "epsilon": 1.5}, "epsilon"),
        ({"amount": "fixed", "method": "temperature_posterior", "pool": "active", "value": 1, "temperature": 0}, "temperature"),
        ({"amount": "fixed", "method": "diverse_posterior", "pool": "active", "value": 1, "diversity_lambda": -0.1}, "diversity_lambda"),
    ],
)
def test_invalid_strategy_config_raises(strategy, match) -> None:
    with pytest.raises(ValueError, match=match):
        _module([strategy])


@pytest.mark.parametrize(
    "strategy, match",
    [
        ({"amount": "recent_accuracy_inverse_7", "method": "random", "pool": "active", "window": 0}, "window"),
        ({"amount": "recent_accuracy_inverse_7", "method": "random", "pool": "active", "gamma": 0}, "gamma"),
        ({"amount": "recent_accuracy_inverse_7", "method": "random", "pool": "active", "padding": "bad"}, "padding"),
        ({"amount": "recent_accuracy_inverse_7", "method": "random", "pool": "active", "padding": "none"}, "padding"),
    ],
)
def test_invalid_history_strategy_config_raises(strategy, match) -> None:
    with pytest.raises(ValueError, match=match):
        _module([strategy], engine_kwargs={"partition": FakePartition(n_cats=2)})


def test_recent_accuracy_chance_padding_requires_n_cats() -> None:
    strategy = {"amount": "recent_accuracy_inverse_7", "method": "random", "pool": "active"}

    with pytest.raises(ValueError, match="padding='chance'"):
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


def test_max_active_hypotheses_budget_follows_strategy_order() -> None:
    posterior = np.array([0.01, 0.02, 0.03, 0.30, 0.25, 0.20, 0.11, 0.09])
    strategies = [
        {"label": "explore_first", "amount": "fixed", "method": "top_posterior", "pool": "inactive", "value": 3},
        {"label": "retain_second", "amount": "fixed", "method": "top_posterior", "pool": "active", "value": 3},
    ]
    mod = DynamicHypothesisModule(
        _engine(set_size=8, posterior=posterior),
        strategies=strategies,
        init_num=0,
        max_active_hypotheses=4,
    )
    mod.active = np.array([0, 1, 2], dtype=int)

    mod._transition()

    assert mod.active.tolist() == [2, 3, 4, 5]
    details = mod.strategy_counts_log[-1]["strategies"]
    assert [item["selected_count"] for item in details] == [3, 1]


def test_strategy_counts_log_keeps_labels_and_method_aggregates() -> None:
    strategies = [
        {"label": "first_random", "amount": "fixed", "method": "random", "pool": "all_unselected", "value": 1},
        {"label": "second_random", "amount": "fixed", "method": "random", "pool": "all_unselected", "value": 2},
    ]
    mod = _module(strategies, module_seed=3)

    mod._transition()

    log = mod.strategy_counts_log[-1]
    assert log["random"] == 3
    assert log["active_total"] == 3
    assert [item["label"] for item in log["strategies"]] == ["first_random", "second_random"]
    assert all("selected" in item for item in log["strategies"])


def test_empty_transition_falls_back_to_best_posterior_hypothesis() -> None:
    posterior = np.array([0.1, 0.2, 0.5, 0.2])
    strategies = [{"amount": "fixed", "method": "random", "pool": "active", "value": 0}]
    mod = _module(strategies, engine_kwargs={"set_size": 4, "posterior": posterior}, module_seed=10)
    mod.active = np.array([0, 2], dtype=int)

    mod._transition()

    assert mod.active.tolist() == [2]
    assert mod.strategy_counts_log[-1]["active_total"] == 1
    assert mod.strategy_counts_log[-1]["strategies"][0]["selected_count"] == 0
    assert mod.strategy_counts_log[-1]["strategies"][-1]["label"] == "fallback_best_posterior"


def test_same_seed_reproducible_and_derived_repeats_differ() -> None:
    strategies = [{"amount": "fixed", "method": "random", "pool": "all_unselected", "value": 4}]
    first = _module(strategies, module_seed=123)
    second = _module(strategies, module_seed=123)

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


def test_random_posterior_zero_mass_pool_falls_back_to_uniform() -> None:
    posterior = np.array([0.0, 0.0, 0.7, 0.3])
    strategy = {"amount": "fixed", "method": "random_posterior", "pool": "active", "value": 1}
    mod = _module([strategy], engine_kwargs={"set_size": 4, "posterior": posterior}, module_seed=4)
    mod.active = np.array([0, 1], dtype=int)

    mod._transition()

    assert len(mod.active) == 1
    assert mod.active[0] in (0, 1)


def test_random_posterior_still_rejects_invalid_candidate_weights() -> None:
    strategy = {"amount": "fixed", "method": "random_posterior", "pool": "active", "value": 1}
    mod = _module([strategy], engine_kwargs={"set_size": 4}, module_seed=4)
    posterior = np.array([0.5, -0.1, 0.4, 0.2])

    with pytest.raises(ValueError, match="invalid values"):
        mod._select_random_posterior(1, np.array([0, 1]), posterior)


def test_top_p_scope_global_vs_pool() -> None:
    posterior = np.array([0.70, 0.10, 0.15, 0.05])
    mod = _module([{"amount": "fixed", "method": "random", "pool": "all_unselected", "value": 1}], engine_kwargs={"set_size": 4, "posterior": posterior})
    candidates = np.array([2, 3])

    global_selected = mod._select_top_posterior(
        0,
        candidates,
        posterior,
        strategy_config={"top_p": 0.5, "top_p_scope": "global"},
    )
    pool_selected = mod._select_top_posterior(
        0,
        candidates,
        posterior,
        strategy_config={"top_p": 0.5, "top_p_scope": "pool"},
    )

    assert global_selected == [2, 3]
    assert pool_selected == [2]


def test_epsilon_posterior_can_drop_high_posterior_hypothesis() -> None:
    posterior = np.array([0.99, 0.01])
    strategy = {"amount": "fixed", "method": "epsilon_posterior", "pool": "all_unselected", "value": 1, "epsilon": 1.0}
    mod = _module([strategy], engine_kwargs={"set_size": 2, "posterior": posterior}, module_seed=0)

    mod._transition()

    assert mod.active.tolist() == [1]


def test_temperature_posterior_is_seed_reproducible() -> None:
    posterior = np.array([0.40, 0.30, 0.20, 0.10])
    strategy = {
        "amount": "fixed",
        "method": "temperature_posterior",
        "pool": "all_unselected",
        "value": 2,
        "temperature": 2.0,
    }
    first = _module([strategy], engine_kwargs={"set_size": 4, "posterior": posterior}, module_seed=44)
    second = _module([strategy], engine_kwargs={"set_size": 4, "posterior": posterior}, module_seed=44)

    first._transition()
    second._transition()

    assert first.active.tolist() == second.active.tolist()


def test_entropy_norm_amounts_move_in_opposite_directions() -> None:
    strategy = {"amount": "fixed", "method": "random", "pool": "all_unselected", "value": 1}
    mod = _module([strategy])
    low_entropy = np.array([0.97, 0.01, 0.01, 0.01])
    high_entropy = np.full(4, 0.25)

    retain_low = mod.adaptive_amount_evaluator("entropy_norm_7", posterior=low_entropy, strategy_config={})
    retain_high = mod.adaptive_amount_evaluator("entropy_norm_7", posterior=high_entropy, strategy_config={})
    explore_low = mod.adaptive_amount_evaluator("opp_entropy_norm_7", posterior=low_entropy, strategy_config={})
    explore_high = mod.adaptive_amount_evaluator("opp_entropy_norm_7", posterior=high_entropy, strategy_config={})

    assert retain_low > retain_high
    assert explore_low < explore_high


def test_recent_accuracy_inverse_uses_chance_padding_by_category_count() -> None:
    strategy = {
        "amount": "recent_accuracy_inverse_7",
        "method": "top_posterior",
        "pool": "active",
        "window": 4,
        "padding": "chance",
    }
    binary = _module([strategy], engine_kwargs={"partition": FakePartition(n_cats=2)})
    four_cat = _module([strategy], engine_kwargs={"partition": FakePartition(n_cats=4)})

    assert binary.adaptive_amount_evaluator("recent_accuracy_inverse_7", posterior=np.full(8, 0.125), strategy_config=strategy) == 4
    assert four_cat.adaptive_amount_evaluator("recent_accuracy_inverse_7", posterior=np.full(8, 0.125), strategy_config=strategy) == 5


def test_recent_accuracy_history_is_recorded_after_transition() -> None:
    posterior = np.array([0.40, 0.20, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01])
    strategy = {
        "label": "history_retention",
        "amount": "recent_accuracy_inverse_7",
        "method": "top_posterior",
        "pool": "active",
        "window": 4,
        "padding": "chance",
    }
    engine = _engine(set_size=8, posterior=posterior, partition=FakePartition(n_cats=2))
    engine.observation = (np.array([0.1, 0.2]), 1, 1.0)
    mod = DynamicHypothesisModule(engine, strategies=[strategy], init_num=6, max_active_hypotheses=7, module_seed=2)

    mod.process()

    assert list(mod.feedback_history) == [1.0]
    assert mod.strategy_counts_log[-1]["strategies"][0]["requested_count"] == 4


def test_exact_feedback_mode_records_half_feedback_as_zero() -> None:
    strategy = {
        "amount": "recent_accuracy_inverse_7",
        "method": "top_posterior",
        "pool": "active",
        "feedback_mode": "exact",
        "padding": 0.5,
    }
    engine = _engine(partition=FakePartition(n_cats=2))
    engine.observation = (np.array([0.1, 0.2]), 1, 0.5)
    mod = DynamicHypothesisModule(engine, strategies=[strategy], init_num=4, max_active_hypotheses=7)

    mod.process()

    assert list(mod.feedback_history) == [0.0]


def test_diverse_posterior_prefers_dissimilar_second_choice() -> None:
    posterior = np.array([0.60, 0.20, 0.19, 0.01])
    sim = np.array(
        [
            [1.00, 0.99, 0.00, 0.00],
            [0.99, 1.00, 0.20, 0.20],
            [0.00, 0.20, 1.00, 0.20],
            [0.00, 0.20, 0.20, 1.00],
        ]
    )
    strategy = {
        "amount": "fixed",
        "method": "diverse_posterior",
        "pool": "all_unselected",
        "value": 2,
        "diversity_lambda": 0.5,
    }
    mod = _module([strategy], engine_kwargs={"set_size": 4, "posterior": posterior, "partition": FakePartition(similarity_matrix=sim)})

    mod._transition()

    assert mod.active.tolist() == [0, 2]


def test_diverse_posterior_requires_similarity_matrix() -> None:
    strategy = {"amount": "fixed", "method": "diverse_posterior", "pool": "all_unselected", "value": 1}

    with pytest.raises(ValueError, match="similarity_matrix"):
        _module([strategy], engine_kwargs={"partition": FakePartition(n_cats=2)})


def test_diverse_posterior_defers_lazy_similarity_matrix_access_until_selection() -> None:
    posterior = np.array([0.60, 0.20, 0.19, 0.01])
    partition = LazySimilarityPartition(np.eye(4))
    strategy = {"amount": "fixed", "method": "diverse_posterior", "pool": "all_unselected", "value": 2}

    mod = _module(
        [strategy],
        engine_kwargs={"set_size": 4, "posterior": posterior, "partition": partition},
    )

    assert partition.access_count == 0

    mod._transition()

    assert partition.access_count == 1
    assert len(mod.active) == 2


def test_diverse_posterior_uses_uniform_relevance_for_zero_mass_inactive_pool() -> None:
    posterior = np.array([0.70, 0.30, 0.0, 0.0])
    strategy = {"amount": "fixed", "method": "diverse_posterior", "pool": "inactive", "value": 2}
    mod = _module(
        [strategy],
        engine_kwargs={"set_size": 4, "posterior": posterior, "partition": FakePartition(similarity_matrix=np.eye(4))},
    )
    mod.active = np.array([0, 1], dtype=int)

    mod._transition()

    assert mod.active.tolist() == [2, 3]


def test_diverse_posterior_still_rejects_nonfinite_candidate_scores() -> None:
    posterior = np.array([0.70, 0.30, np.nan, 0.0])
    strategy = {"amount": "fixed", "method": "diverse_posterior", "pool": "inactive", "value": 2}
    mod = _module(
        [strategy],
        engine_kwargs={"set_size": 4, "posterior": posterior, "partition": FakePartition(similarity_matrix=np.eye(4))},
    )
    mod.active = np.array([0, 1], dtype=int)

    with pytest.raises(ValueError, match="non-finite"):
        mod._transition()


def test_ksimilar_random_zero_scores_falls_back_to_uniform() -> None:
    posterior = np.array([0.7, 0.1, 0.1, 0.1])
    strategy = {
        "amount": "fixed",
        "method": "ksimilar_centers",
        "pool": "inactive",
        "value": 1,
        "proto_hypo_amount": 1,
        "proto_hypo_method": "top",
        "cluster_hypo_method": "random",
    }
    engine = _engine(set_size=4, posterior=posterior, partition=FarPrototypePartition(set_size=4))
    engine.observation = (np.array([0.0]), 0, 1.0)
    mod = DynamicHypothesisModule(engine, strategies=[strategy], init_num=1, max_active_hypotheses=4, module_seed=5)
    mod.active = np.array([0], dtype=int)

    mod._transition()

    assert len(mod.active) == 1
    assert mod.active[0] in (1, 2, 3)


@pytest.mark.parametrize(
    "amount, posterior, partition",
    [
        ("fixed", np.full(8, 0.125), FakePartition(n_cats=2)),
        ("entropy_7", np.array([0.80, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01]), FakePartition(n_cats=2)),
        ("opp_entropy_7", np.full(8, 0.125), FakePartition(n_cats=2)),
        ("random_7", np.array([0.80, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01]), FakePartition(n_cats=2)),
        ("opp_random_7", np.full(8, 0.125), FakePartition(n_cats=2)),
        ("confidence_7", np.array([0.80, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01]), FakePartition(n_cats=2)),
        ("opp_confidence_7", np.full(8, 0.125), FakePartition(n_cats=2)),
        ("max_7", np.array([0.80, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01]), FakePartition(n_cats=2)),
        ("entropy_norm_7", np.array([0.80, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01]), FakePartition(n_cats=2)),
        ("opp_entropy_norm_7", np.full(8, 0.125), FakePartition(n_cats=2)),
        ("recent_accuracy_inverse_7", np.full(8, 0.125), FakePartition(n_cats=2)),
    ],
)
def test_amount_strategy_smoke_runs(amount, posterior, partition) -> None:
    tested = {"amount": amount, "method": "top_posterior", "pool": "all_unselected"}
    if amount == "fixed":
        tested["value"] = 1
    strategies = [
        tested,
        {"amount": "fixed", "method": "random", "pool": "all_unselected", "value": 1},
    ]
    mod = _module(strategies, engine_kwargs={"set_size": 8, "posterior": posterior, "partition": partition}, module_seed=11)

    mod._transition()

    assert len(mod.active) > 0
    assert np.all(np.isfinite(mod.engine.prior))
    assert mod.strategy_counts_log[-1]["active_total"] == len(mod.active)


@pytest.mark.parametrize(
    "method, partition",
    [
        ("top_posterior", FakePartition(n_cats=2)),
        ("random_posterior", FakePartition(n_cats=2)),
        ("random", FakePartition(n_cats=2)),
        ("epsilon_posterior", FakePartition(n_cats=2)),
        ("temperature_posterior", FakePartition(n_cats=2)),
        ("diverse_posterior", FakePartition(n_cats=2, similarity_matrix=np.eye(8))),
        ("ksimilar_centers", FakePrototypePartition(set_size=8)),
    ],
)
def test_method_strategy_smoke_runs(method, partition) -> None:
    posterior = np.array([0.40, 0.20, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01])
    strategy = {"amount": "fixed", "method": method, "pool": "all_unselected", "value": 2}
    if method == "ksimilar_centers":
        strategy.update({"proto_hypo_amount": 1, "proto_hypo_method": "top", "cluster_hypo_method": "top"})
    engine = _engine(set_size=8, posterior=posterior, partition=partition)
    engine.observation = (np.array([0.2, 0.3]), 1, 1.0)
    mod = DynamicHypothesisModule(engine, strategies=[strategy], init_num=3, max_active_hypotheses=7, module_seed=17)

    mod.process()
    mod.process()
    mod.process()

    assert len(mod.active) > 0
    assert len(mod.active) <= 7
    assert np.all(np.isfinite(mod.engine.prior))
    assert len(mod.strategy_counts_log) == 3
