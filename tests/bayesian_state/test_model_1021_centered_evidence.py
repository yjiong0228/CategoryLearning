"""Entry normalization preserves incumbent odds and has replayable causal state."""
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.model.modules.hypothesis_transition.centered_evidence import center_support, centered_evidence
from src.Bayesian_state.model.modules.hypothesis_transition.compact_evidence import compact_evidence, evidence_prior
from tests.bayesian_state.test_model_1021_compact_evidence import compact_config


def test_center_preserves_odds_and_puts_neutral_newcomer_at_current_reference():
    base, scores = np.array([1., 3., 2.]), np.array([30., 31., 29.])
    centered, shift = center_support(base, scores)
    np.testing.assert_allclose(np.average(np.exp(centered), weights=base), 1., atol=1e-14)
    np.testing.assert_allclose(evidence_prior(base, np.arange(3), scores),
                               evidence_prior(base, np.arange(3), centered), atol=1e-14)
    np.testing.assert_allclose(np.diff(centered), np.diff(scores), atol=1e-14)
    assert shift > 29
    uniform, _ = center_support(np.ones(2), np.array([20., 20.]))
    np.testing.assert_allclose(evidence_prior(np.ones(3), np.arange(3), np.r_[uniform, 0.]), [1/3]*3)


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_centered_transition_replays_after_snapshot(condition, monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.base_module import ModuleRole
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = compact_config(condition)
    with compact_evidence(Transition), centered_evidence(Transition) as audit:
        a = StateModel(cfg, context=ModelContext(condition=condition))
        for trial in range(8):
            a.begin_trial([.8, .3, .2, .4])
            a.complete_trial(1, .5 if condition == 3 and trial % 2 else float(trial % 2))
        b = StateModel(cfg, context=ModelContext(condition=condition))
        b.engine.load_state_dict(deepcopy(a.engine.state_dict()))
        for trial in range(4):
            pa, pb = [m.begin_trial([.2, .7, .8, .6]) for m in (a, b)]
            np.testing.assert_array_equal(pa.prior, pb.prior)
            np.testing.assert_array_equal(pa.beta, pb.beta)
            for model in (a, b):
                h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
                np.testing.assert_array_equal(h.retention_evidence[h.last_transition_result.selection.newcomers], 0.)
                model.complete_trial(1, float(trial % 2))
        assert audit['centered_feedback_updates'] > 0
