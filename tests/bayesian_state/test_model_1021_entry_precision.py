"""Current precision transfers to newcomers and survives particle replay."""
from contextlib import ExitStack
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.compact_evidence import compact_evidence, evidence_prior
from src.Bayesian_state.model.modules.hypothesis_transition.centered_evidence import centered_evidence
from src.Bayesian_state.model.modules.hypothesis_transition.entry_precision import entry_precision
from tests.bayesian_state.test_model_1021_compact_evidence import compact_config


@pytest.mark.parametrize('condition', [1, 2, 3])
@pytest.mark.parametrize('centered', [False, True])
def test_current_active_mean_initialization_and_replay(condition, centered, monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = compact_config(condition)
    with ExitStack() as stack:
        stack.enter_context(compact_evidence(Transition))
        if centered:
            stack.enter_context(centered_evidence(Transition))
        audit = stack.enter_context(entry_precision(Transition))
        a = StateModel(cfg, context=ModelContext(condition=condition))
        h = a.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        beta = a.engine.get_module(ModuleRole.BETA)
        transferred = 0
        for trial in range(8):
            active = h.active.copy()
            values = beta.beta[active].copy()
            expected = None
            if trial:
                scores = h.retention_evidence.copy()
                scores[active] = h.retention_decay*scores[active]+h._previous_log_evidence()
                weights = evidence_prior(h.base_prior, active, scores)[active]
                expected = np.average(values, weights=weights)
            a.begin_trial([.8, .3, .2, .4])
            selection = h.last_transition_result.selection
            if trial and len(selection.newcomers):
                np.testing.assert_allclose(beta.beta[selection.newcomers], expected, atol=1e-12, rtol=0)
                transferred += len(selection.newcomers)
            if not trial:
                np.testing.assert_allclose(beta.beta[h.active], beta.beta_init, atol=1e-12)
            for rule in selection.survivors:
                np.testing.assert_array_equal(beta.beta[rule], values[np.flatnonzero(active == rule)[0]])
            assert not hasattr(h, 'reentry_precision_seen')
            a.complete_trial(1, .5 if condition == 3 and trial % 2 else float(trial % 2))
        assert transferred > 0 and audit['transferred_entries'] == transferred
        b = StateModel(cfg, context=ModelContext(condition=condition))
        b.engine.load_state_dict(deepcopy(a.engine.state_dict()))
        for trial in range(3):
            pa, pb = [m.begin_trial([.2, .7, .8, .6]) for m in (a, b)]
            np.testing.assert_array_equal(pa.prior, pb.prior)
            np.testing.assert_array_equal(pa.beta, pb.beta)
            for model in (a, b):
                model.complete_trial(1, float(trial % 2))
