from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.Bayesian_state.problems.modules.minimal_hypo_transition import (
    FeedbackSwapHypothesisModule,
)


class _BetaRecorder:
    def __init__(self, beta_init: float = 5.0, size: int = 8):
        self.beta_init = float(beta_init)
        self.beta = np.zeros(size, dtype=float)
        self.calls: list[tuple[np.ndarray, object]] = []

    def initialize_beta_for_hypotheses(self, indices, priors=None) -> None:
        values = np.asarray(indices, dtype=int)
        self.calls.append((values.copy(), priors))
        self.beta[values] = self.beta_init


def _engine(
    posterior: np.ndarray | None = None,
    *,
    set_size: int = 8,
    beta_recorder: _BetaRecorder | None = None,
) -> SimpleNamespace:
    if posterior is None:
        posterior = np.full(set_size, 1.0 / set_size, dtype=float)
    modules = {}
    if beta_recorder is not None:
        modules["beta_mod"] = beta_recorder
    return SimpleNamespace(
        set_size=set_size,
        prior=np.asarray(posterior, dtype=float).copy(),
        posterior=None,
        hypotheses_mask=None,
        observation=None,
        modules=modules,
    )


def _module(
    *,
    theta: float,
    posterior: np.ndarray | None = None,
    module_seed: int = 13,
    init_hypotheses=(0, 1, 2, 3, 4),
    beta_recorder: _BetaRecorder | None = None,
) -> FeedbackSwapHypothesisModule:
    engine = _engine(
        posterior,
        set_size=len(posterior) if posterior is not None else 8,
        beta_recorder=beta_recorder,
    )
    return FeedbackSwapHypothesisModule(
        engine,
        capacity=len(init_hypotheses),
        theta=theta,
        module_seed=module_seed,
        init_hypotheses=list(init_hypotheses),
    )


def _step(
    module: FeedbackSwapHypothesisModule,
    feedback: float,
    posterior: np.ndarray | None = None,
) -> None:
    if posterior is not None:
        module.engine.posterior = np.asarray(posterior, dtype=float).copy()
    elif module.engine.posterior is not None:
        module.engine.posterior = np.asarray(module.engine.prior, dtype=float).copy()
    module.engine.observation = (np.zeros(2, dtype=float), 1, float(feedback))
    module.process()


def test_initial_state_is_uniform_on_exact_capacity() -> None:
    module = _module(theta=0.0)

    assert module.active.tolist() == [0, 1, 2, 3, 4]
    assert np.flatnonzero(module.engine.hypotheses_mask).tolist() == module.active.tolist()
    assert np.allclose(module.engine.prior[module.active], 0.2)
    assert np.isclose(module.engine.prior.sum(), 1.0)
    assert np.all(module.engine.prior[[5, 6, 7]] == 0.0)


@pytest.mark.parametrize("theta", [-0.01, 1.01, float("nan")])
def test_invalid_theta_raises(theta: float) -> None:
    with pytest.raises(ValueError, match="theta"):
        _module(theta=theta)


def test_first_trial_never_reads_current_feedback() -> None:
    error_module = _module(theta=1.0, module_seed=5)
    correct_module = _module(theta=1.0, module_seed=5)

    _step(error_module, feedback=0.0)
    _step(correct_module, feedback=1.0)

    assert error_module.active.tolist() == correct_module.active.tolist()
    assert error_module.transition_log[0]["swap_probability"] == 0.0
    assert correct_module.transition_log[0]["swap_probability"] == 0.0
    assert error_module.transition_log[0]["swap_event"] is False
    assert correct_module.transition_log[0]["swap_event"] is False


def test_theta_zero_never_swaps_after_errors() -> None:
    module = _module(theta=0.0)
    original = module.active.copy()

    for _ in range(20):
        _step(module, feedback=0.0)

    assert np.array_equal(module.active, original)
    assert not any(item["swap_event"] for item in module.transition_log)


def test_correct_previous_feedback_blocks_swap_at_theta_one() -> None:
    module = _module(theta=1.0)
    _step(module, feedback=1.0)
    _step(module, feedback=0.0)

    assert module.transition_log[1]["feedback_used"] == 1.0
    assert module.transition_log[1]["swap_probability"] == 0.0
    assert module.transition_log[1]["swap_event"] is False


def test_error_at_theta_one_drops_minimum_and_sets_exact_prior() -> None:
    recorder = _BetaRecorder()
    module = _module(theta=1.0, beta_recorder=recorder)
    _step(module, feedback=0.0)

    posterior = np.asarray([0.40, 0.20, 0.05, 0.25, 0.10, 0.0, 0.0, 0.0])
    _step(module, feedback=1.0, posterior=posterior)
    event = module.transition_log[1]

    assert event["swap_event"] is True
    assert event["dropped_hypothesis"] == 2
    newcomer = int(event["new_hypothesis"])
    assert newcomer in {5, 6, 7}
    assert len(module.active) == 5
    assert 2 not in module.active
    assert newcomer in module.active
    assert np.isclose(module.engine.prior[newcomer], 0.2)
    survivors = np.asarray([0, 1, 3, 4], dtype=int)
    expected = 0.8 * posterior[survivors] / posterior[survivors].sum()
    assert np.allclose(module.engine.prior[survivors], expected)
    assert np.isclose(module.engine.prior.sum(), 1.0)
    assert recorder.calls[-1][0].tolist() == [newcomer]
    assert recorder.calls[-1][1] is None
    assert recorder.beta[newcomer] == 5.0


def test_tied_minimum_uses_uniform_selector() -> None:
    module = _module(theta=1.0)
    _step(module, feedback=0.0)
    posterior = np.asarray([0.40, 0.05, 0.05, 0.30, 0.20, 0.0, 0.0, 0.0])

    class _FixedRng:
        @staticmethod
        def random(size: int):
            assert size == 3
            # Force swap; choose the second tied minimum; choose last inactive.
            return np.asarray([0.0, 0.75, 0.99], dtype=float)

    module.trial_rng = _FixedRng()
    _step(module, feedback=1.0, posterior=posterior)

    event = module.transition_log[-1]
    assert event["tied_minimum"] == [1, 2]
    assert event["dropped_hypothesis"] == 2
    assert event["new_hypothesis"] == 7


def test_graded_feedback_has_expected_swap_frequency() -> None:
    module = _module(theta=1.0, module_seed=17)
    _step(module, feedback=0.5)
    events = []
    for _ in range(4000):
        _step(module, feedback=0.5)
        events.append(float(module.transition_log[-1]["swap_event"]))

    assert abs(float(np.mean(events)) - 0.5) < 0.03


def test_random_stream_stays_aligned_across_theta_values() -> None:
    fixed = _module(theta=0.0, module_seed=29)
    dynamic = _module(theta=1.0, module_seed=29)

    for _ in range(12):
        _step(fixed, feedback=0.0)
        _step(dynamic, feedback=0.0)

    fixed_uniforms = [item["random_uniform_swap"] for item in fixed.transition_log]
    dynamic_uniforms = [item["random_uniform_swap"] for item in dynamic.transition_log]
    assert fixed_uniforms == dynamic_uniforms
    assert not any(item["swap_event"] for item in fixed.transition_log)
    assert any(item["swap_event"] for item in dynamic.transition_log[1:])


def test_full_set_requires_theta_zero() -> None:
    engine = _engine(set_size=5)
    with pytest.raises(ValueError, match="theta must be 0"):
        FeedbackSwapHypothesisModule(
            engine,
            capacity=5,
            theta=0.1,
            init_hypotheses=[0, 1, 2, 3, 4],
        )
