from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.problems.discrete_partitions import DiscreteRulePartition
from src.Bayesian_state.problems.partitions import Partition
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


class FakeLabelMetadataPartition(FakePartition):
    def __init__(self, flags: list[bool]):
        super().__init__(n_cats=2, similarity_matrix=np.eye(len(flags)))
        half = max(1, len(flags) // 2)
        self.hypothesis_metadata = [
            {
                "base_hypo": idx % half,
                "is_label_permuted": bool(flag),
                "label_permutation": (1, 0) if flag else (0, 1),
            }
            for idx, flag in enumerate(flags)
        ]


class FarPrototypePartition(FakePartition):
    def __init__(self, set_size: int = 4):
        super().__init__(n_cats=1, similarity_matrix=np.eye(set_size))
        self.n_dims = 1
        self.prototypes = np.zeros((set_size, 1, 1, 1), dtype=float)
        self.prototypes[1:, 0, 0, 0] = 1000.0


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


def test_strategy_without_pool_raises() -> None:
    with pytest.raises(ValueError, match="pool"):
        _module([{"amount": "fixed", "method": "random", "value": 1}])


@pytest.mark.parametrize(
    "strategy, match",
    [
        ({"amount": "fixed", "method": "random", "pool": "bad", "value": 1}, "unsupported pool"),
        ({"amount": "fixed", "method": "random", "pool": "all_unselected", "value": 1}, "unsupported pool"),
        ({"amount": "fixed", "method": "random", "pool": "unselected", "value": 1}, "unsupported pool"),
        ({"amount": "fixed", "method": "bad", "pool": "active", "value": 1}, "unsupported method"),
        ({"amount": "bad", "method": "random", "pool": "active"}, "unsupported amount"),
        ({"amount": "fixed", "method": "random", "pool": "active", "value": 1.5}, "non-integer"),
        ({"amount": "fixed", "method": "top_posterior", "pool": "active", "value": 1, "top_p_scope": "bad"}, "top_p_scope"),
        ({"amount": "fixed", "method": "epsilon_posterior", "pool": "active", "value": 1, "epsilon": 1.5}, "epsilon"),
        ({"amount": "fixed", "method": "temperature_posterior", "pool": "active", "value": 1, "temperature": 0}, "temperature"),
        ({"amount": "fixed", "method": "diverse_posterior", "pool": "active", "value": 1}, "unsupported method"),
    ],
)
def test_invalid_strategy_config_raises(strategy, match) -> None:
    with pytest.raises(ValueError, match=match):
        _module([strategy])


def test_init_pool_label_permuted_uses_metadata() -> None:
    partition = FakeLabelMetadataPartition([False, False, True, True])
    mod = DynamicHypothesisModule(
        _engine(set_size=4, partition=partition),
        strategies=[{"amount": "fixed", "method": "random", "pool": "active", "value": 1}],
        init_num=2,
        init_pool="label_permuted",
        module_seed=3,
    )
    assert set(mod.active).issubset({2, 3})


def test_init_pool_label_permuted_requires_metadata() -> None:
    with pytest.raises(ValueError, match="hypothesis_metadata"):
        DynamicHypothesisModule(
            _engine(set_size=4, partition=FakePartition()),
            strategies=[{"amount": "fixed", "method": "random", "pool": "active", "value": 1}],
            init_num=1,
            init_pool="label_permuted",
        )


def test_init_hypotheses_forces_initial_active_set() -> None:
    mod = DynamicHypothesisModule(
        _engine(set_size=5, partition=FakePartition()),
        strategies=[{"amount": "fixed", "method": "random", "pool": "active", "value": 1}],
        init_num=2,
        init_hypotheses=[3, 1],
    )
    assert mod.active.tolist() == [1, 3]


def _policy_module(policy_profile, posterior=None, module_seed=11) -> DynamicHypothesisModule:
    if posterior is None:
        posterior = np.full(8, 1.0 / 8.0)
    controller = {
        "method": "feedback_gated_softmax",
        "activation": {"temperature": 0.1},
        "features": {"padding": "chance", "feedback_mode": "exact"},
        "profiles": [policy_profile],
    }
    return DynamicHypothesisModule(
        _engine(set_size=len(posterior), posterior=posterior, partition=FakePartition(n_cats=2)),
        strategy_controller=controller,
        init_num=0,
        max_active_hypotheses=5,
        module_seed=module_seed,
    )


def test_profile_policy_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="policy_method"):
        _policy_module({"id": "bad", "policy_method": "bad"})


def test_profile_policy_conservative_keeps_active_set() -> None:
    posterior = np.array([0.01, 0.03, 0.40, 0.22, 0.18, 0.10, 0.04, 0.02])
    mod = _policy_module({"id": "conservative", "policy_method": "conservative"}, posterior)
    mod.active = np.array([1, 2, 3, 4], dtype=int)

    mod._transition()
    mod._apply_mask()
    mod._posterior_to_prior_transition()

    assert mod.active.tolist() == [1, 2, 3, 4]
    assert mod.strategy_counts_log[-1]["profile_policy"]["policy_method"] == "conservative"
    assert mod.strategy_counts_log[-1]["profile_policy"]["newcomer_count"] == 0


def test_profile_policy_stable_drops_when_full_and_adds_one_newcomer() -> None:
    posterior = np.array([0.01, 0.02, 0.60, 0.20, 0.10, 0.03, 0.02, 0.02])
    profile = {
        "id": "stable",
        "policy_method": "stable",
        "active_limit": 5,
        "explore_count": 1,
        "retain_temperature": 0.1,
        "post_to_prior": {"method": "similarity_novelty"},
    }
    mod = _policy_module(profile, posterior, module_seed=2)
    mod.active = np.array([0, 1, 2, 3, 4], dtype=int)

    mod._transition()

    log = mod.strategy_counts_log[-1]["profile_policy"]
    assert len(mod.active) == 5
    assert log["dropped_count"] == 1
    assert log["newcomer_count"] == 1
    assert any(idx not in {0, 1, 2, 3, 4} for idx in mod.active.tolist())
    assert 2 in mod.active


def test_profile_policy_aggressive_top1_and_newcomer_mass() -> None:
    posterior = np.array([0.02, 0.05, 0.30, 0.10, 0.08, 0.25, 0.12, 0.08])
    profile = {
        "id": "aggressive",
        "policy_method": "aggressive",
        "active_limit": 5,
        "max_newcomers": 4,
    }
    mod = _policy_module(profile, posterior, module_seed=3)
    mod.active = np.array([1, 2, 3, 4, 5], dtype=int)

    mod._transition()
    mod._apply_mask()
    mod._posterior_to_prior_transition()

    assert 2 in mod.active.tolist()
    assert len(mod.active) == 4  # round((1 - 0.30) * 4) newcomers + top1
    p2p = mod.strategy_counts_log[-1]["post_to_prior"]
    assert p2p["policy_method"] == "aggressive"
    assert p2p["top_hypothesis"] == 2
    assert p2p["newcomer_count"] == 3
    assert p2p["newcomer_mass"] == pytest.approx(0.70)


def test_profile_policy_stubborn_explores_less_after_error() -> None:
    posterior = np.array([0.02, 0.05, 0.45, 0.20, 0.12, 0.08, 0.05, 0.03])
    profile = {
        "id": "stubborn",
        "policy_method": "stubborn",
        "active_limit": 5,
        "retain_count": 2,
        "base_explore_prob": 0.0,
        "post_correct_explore_prob": 1.0,
        "newcomer_mass": 0.02,
    }
    correct = _policy_module(profile, posterior, module_seed=4)
    correct.active = np.array([1, 2, 3, 4, 5], dtype=int)
    correct.feedback_history.append(1.0)
    correct._transition()

    wrong = _policy_module(profile, posterior, module_seed=4)
    wrong.active = np.array([1, 2, 3, 4, 5], dtype=int)
    wrong.feedback_history.append(0.0)
    wrong._transition()

    assert correct.strategy_counts_log[-1]["profile_policy"]["newcomer_count"] == 1
    assert wrong.strategy_counts_log[-1]["profile_policy"]["newcomer_count"] == 0
    assert wrong.active.tolist() == [2, 3]


@pytest.mark.parametrize(
    "strategy, match",
    [
        ({"amount": "recent_accuracy_inverse_7", "method": "random", "pool": "active", "window": 0}, "window"),
        ({"amount": "recent_accuracy_inverse_7", "method": "random", "pool": "active", "gamma": 0}, "gamma"),
        ({"amount": "recent_accuracy_inverse_7", "method": "random", "pool": "active", "padding": "bad"}, "padding"),
        ({"amount": "recent_accuracy_inverse_7", "method": "random", "pool": "active", "padding": "none"}, "padding"),
        ({"amount": "post_error_explore_7", "method": "random", "pool": "active", "gamma": 0}, "gamma"),
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
    mod = _module([{"amount": "fixed", "method": "random", "pool": "inactive", "value": 1}])
    mod.old_active = np.array([1, 2, 3], dtype=int)
    selected = {2, 5}

    assert mod._resolve_pool("active", selected).tolist() == [1, 3]
    assert mod._resolve_pool("inactive", selected).tolist() == [0, 4, 6, 7]
    with pytest.raises(ValueError, match="Unsupported pool"):
        mod._resolve_pool("all_unselected", selected)


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
        {"label": "first_random", "amount": "fixed", "method": "random", "pool": "inactive", "value": 1},
        {"label": "second_random", "amount": "fixed", "method": "random", "pool": "inactive", "value": 2},
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
    strategies = [{"amount": "fixed", "method": "random", "pool": "inactive", "value": 4}]
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
            "pool": "inactive",
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
    strategies = [{"amount": "fixed", "method": "random", "pool": "inactive", "value": 1}]
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
    mod = _module([{"amount": "fixed", "method": "random", "pool": "inactive", "value": 1}], engine_kwargs={"set_size": 4, "posterior": posterior})
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
    strategy = {"amount": "fixed", "method": "epsilon_posterior", "pool": "inactive", "value": 1, "epsilon": 1.0}
    mod = _module([strategy], engine_kwargs={"set_size": 2, "posterior": posterior}, module_seed=0)

    mod._transition()

    assert mod.active.tolist() == [1]


def test_temperature_posterior_is_seed_reproducible() -> None:
    posterior = np.array([0.40, 0.30, 0.20, 0.10])
    strategy = {
        "amount": "fixed",
        "method": "temperature_posterior",
        "pool": "inactive",
        "value": 2,
        "temperature": 2.0,
    }
    first = _module([strategy], engine_kwargs={"set_size": 4, "posterior": posterior}, module_seed=44)
    second = _module([strategy], engine_kwargs={"set_size": 4, "posterior": posterior}, module_seed=44)

    first._transition()
    second._transition()

    assert first.active.tolist() == second.active.tolist()


def test_low_posterior_selects_lowest_candidate_scores() -> None:
    posterior = np.array([0.60, 0.30, 0.08, 0.02])
    strategy = {"amount": "fixed", "method": "low_posterior", "pool": "inactive", "value": 2}
    mod = _module([strategy], engine_kwargs={"set_size": 4, "posterior": posterior}, module_seed=44)

    mod._transition()

    assert mod.active.tolist() == [2, 3]


def test_entropy_norm_amounts_move_in_opposite_directions() -> None:
    strategy = {"amount": "fixed", "method": "random", "pool": "inactive", "value": 1}
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


def test_accuracy_static_amounts_match_legacy_step_direction() -> None:
    strategy = {
        "amount": "acc_7",
        "method": "top_posterior",
        "pool": "active",
        "window": 4,
        "padding": "chance",
        "feedback_mode": "exact",
    }
    mod = _module([strategy], engine_kwargs={"partition": FakePartition(n_cats=2)})
    mod.feedback_history.extend([1.0, 1.0, 1.0, 1.0])

    assert mod.adaptive_amount_evaluator("acc_7", posterior=np.full(8, 0.125), strategy_config=strategy) == 7
    assert mod.adaptive_amount_evaluator("accuracy_static_7", posterior=np.full(8, 0.125), strategy_config=strategy) == 7
    assert mod.adaptive_amount_evaluator("opp_acc_7", posterior=np.full(8, 0.125), strategy_config=strategy) == 0
    assert mod.adaptive_amount_evaluator("opp_accuracy_static_7", posterior=np.full(8, 0.125), strategy_config=strategy) == 0

    mod.feedback_history.clear()
    mod.feedback_history.extend([0.0, 0.0, 0.0, 0.0])

    assert mod.adaptive_amount_evaluator("acc_7", posterior=np.full(8, 0.125), strategy_config=strategy) == 0
    assert mod.adaptive_amount_evaluator("opp_acc_7", posterior=np.full(8, 0.125), strategy_config=strategy) == 7


def test_accuracy_delta_amounts_use_left_padding_and_opposite_directions() -> None:
    strategy = {
        "amount": "accuracy_delta_6",
        "method": "top_posterior",
        "pool": "active",
        "window": 4,
        "padding": 0.5,
        "feedback_mode": "exact",
        "scale": 0.5,
    }
    mod = _module([strategy], engine_kwargs={"partition": FakePartition(n_cats=2)})
    assert mod.feedback_history.maxlen == 8
    assert mod.adaptive_amount_evaluator("accuracy_delta_6", posterior=np.full(8, 0.125), strategy_config=strategy) == 0
    assert mod.adaptive_amount_evaluator("opp_accuracy_delta_6", posterior=np.full(8, 0.125), strategy_config=strategy) == 0

    mod.feedback_history.extend([0.0, 0.0, 1.0, 1.0])
    assert mod.adaptive_amount_evaluator("accuracy_delta_6", posterior=np.full(8, 0.125), strategy_config=strategy) == 0

    mod.feedback_history.clear()
    mod.feedback_history.extend([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    assert mod.adaptive_amount_evaluator("accuracy_delta_6", posterior=np.full(8, 0.125), strategy_config=strategy) == 6
    assert mod.adaptive_amount_evaluator("opp_accuracy_delta_6", posterior=np.full(8, 0.125), strategy_config=strategy) == 0

    mod.feedback_history.clear()
    mod.feedback_history.extend([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    assert mod.adaptive_amount_evaluator("accuracy_delta_6", posterior=np.full(8, 0.125), strategy_config=strategy) == 0
    assert mod.adaptive_amount_evaluator("opp_accuracy_delta_6", posterior=np.full(8, 0.125), strategy_config=strategy) == 6


def test_post_error_explore_amount_uses_previous_feedback() -> None:
    strategy = {
        "amount": "post_error_explore_5",
        "method": "random",
        "pool": "inactive",
        "padding": "chance",
        "feedback_mode": "exact",
        "min_count": 1,
        "gamma": 1.0,
    }
    mod = _module([strategy], engine_kwargs={"partition": FakePartition(n_cats=2)})

    assert mod.adaptive_amount_evaluator("post_error_explore_5", posterior=np.full(8, 0.125), strategy_config=strategy) == 3
    mod.feedback_history.append(1.0)
    assert mod.adaptive_amount_evaluator("post_error_explore_5", posterior=np.full(8, 0.125), strategy_config=strategy) == 1
    mod.feedback_history.clear()
    mod.feedback_history.append(0.0)
    assert mod.adaptive_amount_evaluator("post_error_explore_5", posterior=np.full(8, 0.125), strategy_config=strategy) == 5


def test_mixed_history_strategies_must_share_feedback_mode() -> None:
    strategies = [
        {"amount": "recent_accuracy_inverse_7", "method": "top_posterior", "pool": "active"},
        {"amount": "acc_7", "method": "random", "pool": "inactive"},
    ]

    with pytest.raises(ValueError, match="feedback_mode"):
        _module(strategies, engine_kwargs={"partition": FakePartition(n_cats=2)})


def test_history_maxlen_config_is_no_longer_supported() -> None:
    strategy = {
        "amount": "recent_accuracy_inverse_7",
        "method": "top_posterior",
        "pool": "active",
        "window": 4,
        "padding": "chance",
    }

    with pytest.raises(ValueError, match="history_maxlen"):
        _module([strategy], history_maxlen=4, engine_kwargs={"partition": FakePartition(n_cats=2)})


@pytest.mark.parametrize(
    "post_to_prior, match",
    [
        ({"method": "bad"}, "Unsupported post_to_prior"),
        ({"method": "similarity_novelty", "confidence_source": "bad"}, "confidence_source"),
        ({"method": "similarity_novelty", "confidence_source": "recent_accuracy", "window": 0}, "window"),
        ({"method": "conservative_carryover", "newcomer_mass": 1.2}, "newcomer_mass"),
        ({"method": "error_boost_newcomers", "window": 0}, "window"),
        ({"method": "stochastic_reset", "reset_probability": -0.1}, "reset_probability"),
    ],
)
def test_invalid_post_to_prior_config_raises(post_to_prior, match) -> None:
    strategy = {"amount": "fixed", "method": "random", "pool": "inactive", "value": 1}
    with pytest.raises(ValueError, match=match):
        _module([strategy], post_to_prior=post_to_prior, engine_kwargs={"partition": FakePartition(n_cats=2)})


def test_default_similarity_novelty_matches_previous_formula() -> None:
    posterior = np.array([0.6, 0.3, 0.1, 0.0])
    sim = np.eye(4)
    sim[2, 0] = 0.2
    sim[2, 1] = 0.8
    strategy = {"amount": "fixed", "method": "random", "pool": "inactive", "value": 0}
    mod = _module([strategy], engine_kwargs={"set_size": 4, "posterior": posterior, "partition": FakePartition(n_cats=2, similarity_matrix=sim)})
    mod.old_active = np.array([0, 1], dtype=int)
    mod.active = np.array([0, 2], dtype=int)

    mod._posterior_to_prior_transition()

    old_norm = np.array([0.6, 0.3]) / 0.9
    p_sim = np.array([0.2, 0.8]) @ old_norm
    p_nov = 1.0 - 0.8
    confidence = 0.6
    raw_new = max(1.0 - confidence, 0.05) * (confidence * p_sim + (1.0 - confidence) * p_nov)
    expected = np.array([0.6, 0.0, raw_new, 0.0])
    expected /= expected.sum()

    assert np.allclose(mod.engine.prior, expected)
    assert mod.strategy_counts_log[-1]["post_to_prior"]["method"] == "similarity_novelty"


def test_similarity_novelty_keeps_legacy_zero_old_mass_behavior() -> None:
    posterior = np.array([0.0, 0.0, 0.0, 1.0])
    sim = np.eye(4)
    sim[2, 0] = 1.0
    sim[2, 1] = 1.0
    strategy = {"amount": "fixed", "method": "random", "pool": "inactive", "value": 0}
    mod = _module(
        [strategy],
        engine_kwargs={
            "set_size": 4,
            "posterior": posterior,
            "partition": FakePartition(n_cats=2, similarity_matrix=sim),
        },
    )
    mod.old_active = np.array([0, 1], dtype=int)
    mod.active = np.array([0, 2], dtype=int)

    mod._posterior_to_prior_transition()

    assert np.allclose(mod.engine.prior, np.array([0.5, 0.0, 0.5, 0.0]))


def test_post_to_prior_recent_accuracy_confidence_records_history() -> None:
    posterior = np.array([0.45, 0.25, 0.20, 0.10])
    strategy = {"amount": "fixed", "method": "top_posterior", "pool": "active", "value": 1}
    engine = _engine(
        set_size=4,
        posterior=posterior,
        partition=FakePartition(n_cats=2, similarity_matrix=np.eye(4)),
    )
    engine.observation = (np.array([0.1, 0.2]), 1, 1.0)
    mod = DynamicHypothesisModule(
        engine,
        strategies=[strategy],
        init_num=2,
        post_to_prior={
            "method": "similarity_novelty",
            "confidence_source": "recent_accuracy",
            "window": 3,
            "padding": "chance",
            "feedback_mode": "exact",
        },
    )

    assert mod.feedback_history.maxlen >= 3
    mod.process()

    assert list(mod.feedback_history) == [1.0]


def test_conservative_carryover_limits_newcomer_mass() -> None:
    posterior = np.array([0.7, 0.2, 0.1, 0.0])
    strategy = {"amount": "fixed", "method": "random", "pool": "inactive", "value": 0}
    mod = _module(
        [strategy],
        post_to_prior={"method": "conservative_carryover", "newcomer_mass": 0.2},
        engine_kwargs={"set_size": 4, "posterior": posterior, "partition": FakePartition(n_cats=2)},
    )
    mod.old_active = np.array([0, 1], dtype=int)
    mod.active = np.array([0, 2], dtype=int)

    mod._posterior_to_prior_transition()

    assert np.isclose(mod.engine.prior[0], 0.8)
    assert np.isclose(mod.engine.prior[2], 0.2)


def test_error_boost_newcomers_increases_newcomer_mass_after_errors() -> None:
    posterior = np.array([0.7, 0.2, 0.1, 0.0])
    strategy = {"amount": "fixed", "method": "random", "pool": "inactive", "value": 0}
    kwargs = {
        "post_to_prior": {
            "method": "error_boost_newcomers",
            "window": 2,
            "padding": "chance",
            "feedback_mode": "exact",
            "base_newcomer_mass": 0.1,
            "max_newcomer_mass": 0.7,
        },
        "engine_kwargs": {
            "set_size": 4,
            "posterior": posterior,
            "partition": FakePartition(n_cats=2, similarity_matrix=np.eye(4)),
        },
    }
    good = _module([strategy], **kwargs)
    bad = _module([strategy], **kwargs)
    for mod, history in ((good, [1.0, 1.0]), (bad, [0.0, 0.0])):
        mod.feedback_history.extend(history)
        mod.old_active = np.array([0, 1], dtype=int)
        mod.active = np.array([0, 2], dtype=int)
        mod._posterior_to_prior_transition()

    assert bad.engine.prior[2] > good.engine.prior[2]
    assert np.isclose(good.strategy_counts_log[-1]["post_to_prior"]["newcomer_mass"], 0.1)
    assert np.isclose(bad.strategy_counts_log[-1]["post_to_prior"]["newcomer_mass"], 0.7)


def test_stochastic_reset_is_reproducible_and_can_downweight_high_posterior() -> None:
    posterior = np.array([0.99, 0.01, 0.0, 0.0])
    strategy = {"amount": "fixed", "method": "random", "pool": "inactive", "value": 0}
    config = {
        "method": "stochastic_reset",
        "reset_probability": 1.0,
        "newcomer_mass": 0.8,
        "concentration": 1.0,
    }
    first = _module([strategy], post_to_prior=config, engine_kwargs={"set_size": 4, "posterior": posterior}, module_seed=9)
    second = _module([strategy], post_to_prior=config, engine_kwargs={"set_size": 4, "posterior": posterior}, module_seed=9)
    for mod in (first, second):
        mod.old_active = np.array([0, 1], dtype=int)
        mod.active = np.array([0, 2], dtype=int)
        mod._posterior_to_prior_transition()

    assert np.allclose(first.engine.prior, second.engine.prior)
    assert first.engine.prior[0] < 0.99
    assert np.isclose(first.engine.prior[2], 0.8)


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


def test_ksimilar_centers_runs_with_label_reversed_partition() -> None:
    partition = Partition(n_dims=4, n_cats=2, include_label_reversals=True)
    posterior = np.full(partition.length, 1.0 / partition.length)
    posterior[0] = 0.3
    posterior = posterior / posterior.sum()
    strategy = {
        "amount": "fixed",
        "method": "ksimilar_centers",
        "pool": "inactive",
        "value": 1,
        "proto_hypo_amount": 1,
        "proto_hypo_method": "top",
        "cluster_hypo_method": "top",
    }
    engine = _engine(set_size=partition.length, posterior=posterior, partition=partition)
    engine.observation = (np.array([0.25, 0.5, 0.5, 0.5]), 1, 1.0)
    mod = DynamicHypothesisModule(engine, strategies=[strategy], init_num=1, module_seed=17)
    mod.active = np.array([0], dtype=int)

    mod._transition()

    assert len(mod.active) == 1
    assert 0 <= int(mod.active[0]) < partition.length


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
        ("acc_7", np.full(8, 0.125), FakePartition(n_cats=2)),
        ("accuracy_static_7", np.full(8, 0.125), FakePartition(n_cats=2)),
        ("opp_acc_7", np.full(8, 0.125), FakePartition(n_cats=2)),
        ("opp_accuracy_static_7", np.full(8, 0.125), FakePartition(n_cats=2)),
        ("accuracy_delta_7", np.full(8, 0.125), FakePartition(n_cats=2)),
        ("opp_accuracy_delta_7", np.full(8, 0.125), FakePartition(n_cats=2)),
        ("latent_volatility_7", np.full(8, 0.125), FakePartition(n_cats=2)),
        ("opp_latent_volatility_7", np.full(8, 0.125), FakePartition(n_cats=2)),
        ("post_error_explore_7", np.full(8, 0.125), FakePartition(n_cats=2)),
    ],
)
def test_amount_strategy_smoke_runs(amount, posterior, partition) -> None:
    tested = {"amount": amount, "method": "top_posterior", "pool": "inactive"}
    if amount == "fixed":
        tested["value"] = 1
    strategies = [
        tested,
        {"amount": "fixed", "method": "random", "pool": "inactive", "value": 1},
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
        ("low_posterior", FakePartition(n_cats=2)),
        ("ksimilar_centers", FakePrototypePartition(set_size=8)),
    ],
)
def test_method_strategy_smoke_runs(method, partition) -> None:
    posterior = np.array([0.40, 0.20, 0.15, 0.10, 0.08, 0.04, 0.02, 0.01])
    strategy = {"amount": "fixed", "method": method, "pool": "inactive", "value": 2}
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


def test_strategy_candidate_json_uses_explicit_active_or_inactive_pools() -> None:
    path = (
        Path(__file__).parents[1]
        / "src"
        / "Bayesian_state"
        / "problems"
        / "modules"
        / "hypo_transition_strategy_candidates.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert len(payload["cond1"]) >= 12
    assert len(payload["cond23"]) >= 20

    for condition, candidates in payload.items():
        assert candidates, condition
        for candidate in candidates:
            kwargs = candidate["hypo_transitions_kwargs"]
            assert "max_active_hypotheses" not in kwargs
            assert "history_maxlen" not in kwargs
            strategies = kwargs["strategies"]
            assert strategies
            for strategy in strategies:
                assert strategy["pool"] in {"active", "inactive"}
                assert strategy["method"] != "diverse_posterior"
                amount = str(strategy["amount"])
                if amount.startswith(
                    (
                        "recent_accuracy_inverse_",
                        "acc_",
                        "accuracy_static_",
                        "opp_acc_",
                        "opp_accuracy_static_",
                        "accuracy_delta_",
                        "opp_accuracy_delta_",
                    )
                ):
                    assert int(strategy["window"]) > 0
            DynamicHypothesisModule(
                _engine(
                    set_size=16,
                    posterior=np.full(16, 1.0 / 16.0),
                    partition=FakePrototypePartition(set_size=16),
                ),
                module_seed=23,
                **kwargs,
            )


def test_bayesian_m7_candidate_styles_validate() -> None:
    path = (
        Path(__file__).parents[1]
        / "src"
        / "Bayesian_state"
        / "problems"
        / "modules"
        / "hypo_transition_strategy_candidates.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = {
        candidate["id"]: candidate["hypo_transitions_kwargs"]
        for condition in ("cond1", "cond23")
        for candidate in payload[condition]
        if candidate["id"] in {"c1_bayesian_m7_legacy", "c23_bayesian_m7_legacy"}
    }

    assert "c23_bayesian_m7_legacy" in candidates

    posterior = np.full(16, 1.0 / 16.0)
    partition = FakePrototypePartition(set_size=16)
    for kwargs in candidates.values():
        mod = DynamicHypothesisModule(
            _engine(set_size=16, posterior=posterior, partition=partition),
            module_seed=19,
            **kwargs,
        )
        assert len(mod.active) > 0


def test_v10_strategy_candidate_json_validates_post_to_prior_candidates() -> None:
    path = (
        Path(__file__).parents[1]
        / "src"
        / "Bayesian_state"
        / "problems"
        / "modules"
        / "hypo_transition_strategy_v10_candidates.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["cond1_v10"]
    assert payload["cond23_v10"]
    seen_methods = set()
    for candidates in payload.values():
        for candidate in candidates:
            kwargs = candidate["hypo_transitions_kwargs"]
            assert "post_to_prior" in kwargs
            seen_methods.add(kwargs["post_to_prior"]["method"])
            for strategy in kwargs["strategies"]:
                assert strategy["pool"] in {"active", "inactive"}
                assert strategy["method"] != "diverse_posterior"
            DynamicHypothesisModule(
                _engine(
                    set_size=24,
                    posterior=np.full(24, 1.0 / 24.0),
                    partition=FakePrototypePartition(set_size=24),
                ),
                module_seed=29,
                **kwargs,
            )
    assert {"similarity_novelty", "conservative_carryover", "error_boost_newcomers", "stochastic_reset"} <= seen_methods


def test_strategy_controller_uses_history_features_and_logs_profile_probabilities() -> None:
    controller = {
        "method": "feedback_gated_softmax",
        "features": {
            "recent_accuracy_window": 4,
            "accuracy_delta_window": 2,
            "padding": 0.5,
            "feedback_mode": "graded",
        },
        "activation": {"temperature": 1.0},
        "profiles": [
            {
                "id": "exploit",
                "activation": {"recent_accuracy": 6.0},
                "strategies": [
                    {"label": "retain", "amount": "fixed", "value": 1, "method": "top_posterior", "pool": "active"}
                ],
                "post_to_prior": {"method": "similarity_novelty"},
            },
            {
                "id": "refresh",
                "activation": {"recent_error": 6.0},
                "strategies": [
                    {"label": "explore", "amount": "fixed", "value": 1, "method": "random", "pool": "inactive"}
                ],
                "post_to_prior": {"method": "conservative_carryover", "newcomer_mass": 0.1},
            },
        ],
    }
    posterior = np.asarray([0.7, 0.2, 0.08, 0.02], dtype=float)
    engine = _engine(set_size=4, posterior=posterior, partition=FakePartition(n_cats=2))
    mod = DynamicHypothesisModule(
        engine,
        strategy_controller=controller,
        init_num=2,
        module_seed=11,
    )
    mod.feedback_history.extend([0.0, 0.0, 0.0, 0.0])

    mod._transition()
    log = mod.strategy_counts_log[-1]

    assert log["strategy_controller"]["features"]["recent_accuracy"] == pytest.approx(0.0)
    assert log["profile_probabilities"]["refresh"] > log["profile_probabilities"]["exploit"]
    assert log["selected_profile"] in {"exploit", "refresh"}


def test_v11_profile_candidate_json_validates_and_excludes_stochastic_reset() -> None:
    path = (
        Path(__file__).parents[1]
        / "src"
        / "Bayesian_state"
        / "problems"
        / "modules"
        / "hypo_transition_strategies"
        / "hypo_transition_profile_v11_candidates.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["cond1_v11"]
    assert payload["cond23_v11"]
    for candidates in payload.values():
        for candidate in candidates:
            model_kwargs = candidate["model_kwargs"]
            transition_kwargs = model_kwargs["engine.modules.hypo_transitions_mod.kwargs"]
            readout_kwargs = model_kwargs["engine.choice_readout.kwargs"]
            assert readout_kwargs["method"] in {
                "sharpened_expectation",
                "sticky_sample",
                "stubborn_sticky",
                "sample_hypothesis",
                "map_hypothesis",
            }
            controller = transition_kwargs["strategy_controller"]
            for profile in controller["profiles"]:
                assert profile["post_to_prior"]["method"] != "stochastic_reset"
                for strategy in profile["strategies"]:
                    assert strategy["pool"] in {"active", "inactive"}
            DynamicHypothesisModule(
                _engine(
                    set_size=24,
                    posterior=np.full(24, 1.0 / 24.0),
                    partition=FakePrototypePartition(set_size=24),
                ),
                module_seed=31,
                **transition_kwargs,
            )


def test_v13_profile_candidate_json_validates_policy_profiles() -> None:
    path = (
        Path(__file__).parents[1]
        / "src"
        / "Bayesian_state"
        / "problems"
        / "modules"
        / "hypo_transition_strategies"
        / "hypo_transition_profile_v13_candidates.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["cond1_v13"]
    for candidate in payload["cond1_v13"]:
        model_kwargs = candidate["model_kwargs"]
        transition_kwargs = model_kwargs["engine.modules.hypo_transitions_mod.kwargs"]
        readout_kwargs = model_kwargs["engine.choice_readout.kwargs"]

        assert readout_kwargs["method"] in {"expectation", "map_hypothesis"}
        assert transition_kwargs["max_active_hypotheses"] == 5
        controller = transition_kwargs["strategy_controller"]
        methods = {profile["policy_method"] for profile in controller["profiles"]}
        assert methods == {"conservative", "stable", "aggressive", "stubborn"}
        for profile in controller["profiles"]:
            assert "strategies" not in profile
            assert profile.get("post_to_prior", {}).get("method") != "stochastic_reset"

        mod = DynamicHypothesisModule(
            _engine(
                set_size=38,
                posterior=np.full(38, 1.0 / 38.0),
                partition=FakeLabelMetadataPartition([False] * 19 + [True] * 19),
            ),
            module_seed=31,
            **transition_kwargs,
        )
        mod._transition()
        assert 1 <= len(mod.active) <= 5
