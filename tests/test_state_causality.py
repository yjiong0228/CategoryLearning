from types import SimpleNamespace

import numpy as np

from src.Bayesian_state.problems.modules.beta import BetaModule
from src.Bayesian_state.problems.modules.memory import DualMemoryModule


class _BinaryPartition:
    n_cats = 2

    @staticmethod
    def get_category_assignment(hypo, stimulus, distance_mode="prototype", beta=1.0):
        return int(hypo)


def test_beta_log_records_pre_feedback_value() -> None:
    engine = SimpleNamespace(
        set_size=2,
        beta=None,
        partition=_BinaryPartition(),
        distance_mode="prototype",
        hypotheses_mask=np.ones(2, dtype=float),
        observation=(np.asarray([0.5]), 1, 1.0),
    )
    module = BetaModule(
        engine,
        beta_init=5.0,
        beta_min=0.1,
        beta_max=25.0,
        correct_additive=0.5,
        decrease_rate=0.15,
    )

    module.process()

    assert len(module.beta_log) == 1
    assert np.allclose(module.beta_log[0], [5.0, 5.0])
    assert module.beta[0] > 5.0
    assert module.beta[1] < 5.0


def test_memory_transition_syncs_prior_and_preserves_channel_offsets() -> None:
    engine = SimpleNamespace(
        set_size=2,
        hypotheses_mask=np.ones(2, dtype=float),
        prior=np.asarray([0.5, 0.5], dtype=float),
        state=None,
    )
    module = DualMemoryModule(engine, gamma=0.8, w0=0.25)
    module.state["static"] = np.asarray([-1.0, -2.0], dtype=float)
    module.state["fade"] = np.asarray([-2.0, -4.0], dtype=float)
    old_offsets = module.state["static"] - module.state["fade"]
    engine.prior = np.asarray([0.8, 0.2], dtype=float)

    module._state_transition(np.ones(2, dtype=float), force_sync=True)

    combined = (
        module.w0 * module.state["static"]
        + (1.0 - module.w0) * module.state["fade"]
    )
    synced = module.translate_from_log(combined.copy(), mask=np.ones(2, dtype=float))
    new_offsets = module.state["static"] - module.state["fade"]

    assert np.allclose(synced, engine.prior)
    assert np.allclose(new_offsets, old_offsets)


def _memory_posterior(feedback_gain: float) -> np.ndarray:
    engine = SimpleNamespace(
        set_size=2,
        hypotheses_mask=np.ones(2, dtype=float),
        prior=np.asarray([0.5, 0.5], dtype=float),
        posterior=None,
        likelihood=np.asarray([0.8, 0.2], dtype=float),
        state=None,
    )
    module = DualMemoryModule(
        engine,
        gamma=1.0,
        w0=1.0,
        feedback_gain=feedback_gain,
    )
    module.process()
    return np.asarray(engine.posterior, dtype=float)


def test_feedback_gain_one_preserves_standard_bayes_update() -> None:
    assert np.allclose(_memory_posterior(1.0), [0.8, 0.2])


def test_feedback_gain_scales_evidence_without_changing_likelihood() -> None:
    weak = _memory_posterior(0.5)
    neutral = _memory_posterior(0.0)
    strong = _memory_posterior(2.0)

    assert np.allclose(neutral, [0.5, 0.5])
    assert 0.5 < weak[0] < 0.8
    assert strong[0] > 0.8
