from __future__ import annotations

from copy import deepcopy

import pytest
import numpy as np

from scripts.run_cond1_mechanism_screen import (
    _benjamini_hochberg,
    _fit_population_prior,
    _paired_signflip_p,
)
from src.Bayesian_state.utils.newplan_mechanism_variants import (
    apply_candidate,
    capacity_candidates,
    feedback_candidates,
    memory_candidates,
    plasticity_candidates,
    strategy_candidates,
)


def _base() -> dict:
    return {
        "modules": {
            "memory_mod": {"kwargs": {"gamma": 0.55, "feedback_gain": 1.0}},
            "beta_mod": {
                "kwargs": {"correct_additive": 0.50, "decrease_rate": 0.15}
            },
            "hypo_transitions_mod": {"kwargs": {"capacity": 38, "theta": 0.0}},
        }
    }


@pytest.mark.parametrize(
    "factory",
    [feedback_candidates, memory_candidates, plasticity_candidates],
)
def test_continuous_families_have_one_reference(factory) -> None:
    assert sum(candidate.is_reference for candidate in factory()) == 1


def test_capacity_full_set_forces_zero_theta() -> None:
    candidates = capacity_candidates(shared_theta=0.75)
    full = next(candidate for candidate in candidates if candidate.value == 38)
    limited = next(candidate for candidate in candidates if candidate.value == 5)
    assert full.parameter_dict()[
        "engine.modules.hypo_transitions_mod.kwargs.theta"
    ] == 0.0
    assert limited.parameter_dict()[
        "engine.modules.hypo_transitions_mod.kwargs.theta"
    ] == 0.75


def test_plasticity_scale_preserves_rate_ratio() -> None:
    candidate = next(
        candidate for candidate in plasticity_candidates() if candidate.value == 2.0
    )
    config = apply_candidate(_base(), candidate)
    kwargs = config["modules"]["beta_mod"]["kwargs"]
    assert kwargs["correct_additive"] == 1.0
    assert kwargs["decrease_rate"] == 0.30


def test_apply_candidate_does_not_mutate_base() -> None:
    base = _base()
    original = deepcopy(base)
    candidate = feedback_candidates()[0]
    configured = apply_candidate(base, candidate)
    assert base == original
    assert configured["modules"]["memory_mod"]["kwargs"]["feedback_gain"] == 0.4


def test_strategy_candidates_require_limited_capacity() -> None:
    with pytest.raises(ValueError):
        strategy_candidates(capacity=38)


def test_strategy_candidates_use_frozen_theta_as_reference() -> None:
    candidates = strategy_candidates(capacity=3)
    reference = [candidate for candidate in candidates if candidate.is_reference]
    assert len(reference) == 1
    assert reference[0].value == 0.75
    assert all(
        candidate.parameter_dict()[
            "engine.modules.hypo_transitions_mod.kwargs.capacity"
        ]
        == 3
        for candidate in candidates
    )


def test_population_mixture_prior_and_responsibilities_are_normalized() -> None:
    log_evidence = np.asarray(
        [
            [0.0, -8.0],
            [0.0, -6.0],
            [-5.0, 0.0],
        ]
    )
    prior, responsibilities, iterations = _fit_population_prior(
        log_evidence,
        alpha=1.0,
    )
    assert iterations > 0
    assert np.isclose(prior.sum(), 1.0)
    assert np.allclose(responsibilities.sum(axis=1), 1.0)
    assert responsibilities[0, 0] > 0.95
    assert responsibilities[2, 1] > 0.95


def test_exact_signflip_and_bh_are_bounded() -> None:
    assert _paired_signflip_p([1.0, 1.0, 1.0], seed=1, repeats=100) == 0.25
    adjusted = _benjamini_hochberg([0.01, 0.04, 0.03])
    assert np.all((adjusted >= 0.0) & (adjusted <= 1.0))
    assert np.allclose(adjusted, [0.03, 0.04, 0.04])
