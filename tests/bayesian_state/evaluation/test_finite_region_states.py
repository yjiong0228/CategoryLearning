import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.finite_region_states import capture_finite_region_states
from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
from src.Bayesian_state.model import StateModel
from src.Bayesian_state.model.modules.hypothesis_transition.finite_region_learning import finite_region_learning
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule as Transition,
)
from tests.bayesian_state.test_finite_region_learning import configuration


@pytest.fixture(autouse=True)
def isolated_geometry_cache(monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)


@pytest.mark.parametrize('rehearse,noise', [(False, 0.), (True, 0.), (True, .03)])
def test_particle_capture_preserves_predictions_and_feedback_causality(rehearse, noise):
    cfg = configuration()
    cfg['modules']['perception_mod']['kwargs']['std'] = [noise]*4
    x = np.random.default_rng(441).uniform(.1, .9, (12, 4))
    y = np.tile([1, 2, 3, 4], 3)
    feedback = np.tile([1., 0., 0.], 4)
    kwargs = dict(engine_config=cfg, subject_id=207, condition=2, stimulus=x, choices=y,
                  particle_count=3, choice_readout_power=1., filter_seed=923,
                  resample_threshold_fraction=1.)
    original = StateModel.begin_trial
    with finite_region_learning(Transition, learning_rate=.5, rehearse_candidates=rehearse):
        reference = run_state_model_particle_filter(feedback=feedback, **kwargs)
        with capture_finite_region_states(choices=y, particle_count=3) as collector:
            result = run_state_model_particle_filter(feedback=feedback, state_observer=collector,
                                                     observe_decision_basis=True, **kwargs)
            saved = collector.arrays()
        assert StateModel.begin_trial is original
        changed = feedback.copy(); changed[-1] = 1.-changed[-1]
        with capture_finite_region_states(choices=y, particle_count=3) as other:
            run_state_model_particle_filter(feedback=changed, state_observer=other,
                                             observe_decision_basis=True, **kwargs)
            altered = other.arrays()
    for field in ('marginal_probabilities', 'final_weights', 'resampled'):
        np.testing.assert_array_equal(getattr(result, field), getattr(reference, field))
    assert result.resampled.any()
    for name in saved:
        np.testing.assert_array_equal(saved[name], altered[name], err_msg=name)
    np.testing.assert_allclose(saved['prediction'], reference.marginal_probabilities, atol=1e-12, rtol=0)
    np.testing.assert_allclose(saved['choice_probability'], reference.marginal_probabilities[np.arange(len(y)), y-1],
                               atol=1e-12, rtol=0)
    valid = saved['rule_ids'] >= 0
    np.testing.assert_allclose(saved['geometry'][valid].sum(-1), 1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(saved['associations'][valid].sum(-1), 1., atol=1e-12, rtol=0)
    np.testing.assert_array_equal(saved['associations'][0][valid[0]], .25)
    assert np.any(saved['associations'][1:][valid[1:]] != .25)
    if noise:
        assert np.any(saved['perceived_stimulus'] != x[:, None, :])
    else:
        np.testing.assert_array_equal(saved['perceived_stimulus'], np.repeat(x[:, None, :], 3, axis=1))
    assert saved['rule_ids'].shape[:2] == (12, 3)


def test_context_restores_method_after_error_and_rejects_nested_installation():
    original = StateModel.begin_trial
    with pytest.raises(RuntimeError, match='example'):
        with capture_finite_region_states(choices=np.array([1]), particle_count=1):
            raise RuntimeError('example')
    assert StateModel.begin_trial is original
    with capture_finite_region_states(choices=np.array([1]), particle_count=1) as collector:
        with pytest.raises(ValueError, match='once'):
            with capture_finite_region_states(choices=np.array([1]), particle_count=1):
                pass
        with pytest.raises(ValueError, match='Incomplete'):
            collector.arrays()
    assert StateModel.begin_trial is original
