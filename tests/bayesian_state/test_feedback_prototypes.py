"""Causal feedback, reproducible particle state and bounded precision updates."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.beta import BetaModule
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule as Transition
from src.Bayesian_state.model.modules.hypothesis_transition.feedback_guided import feedback_intervention, tilt_probability, _ProposalDraw, probability_roundoff, retention_roundoff_guard
from src.Bayesian_state.model.modules.precision_interventions import precision_intervention, precision_step
from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule


def config(condition):
    cfg = yaml.safe_load(Path(f'configs/exp123/model_struct/model_0923_cond{condition}_full_v3.yaml').read_text())
    cfg['oral_readout'] = None
    cfg['modules']['perception_mod']['kwargs'] = dict(features=4, mean=[0.]*4, std=[0.]*4, module_seed=2)
    h = cfg['modules']['hypo_transitions_mod']['kwargs']
    h.update(intervention=dict(kind='E', decay=.95, capacity='evidence_mean', search='accumulate'),
             relation_samples=128, module_seed=3)
    h['search_controller']['baseline_probability'] = .999
    cfg['provenance']['hypothesis_similarity']['sample_count'] = 128
    return cfg


def test_tilt_normalizes_and_preserves_zero_support():
    np.testing.assert_allclose(tilt_probability([.2, .8, 0], [.8, .2, .9]), [.5, .5, 0])
    with pytest.raises(ValueError): tilt_probability([.2, .8], [0, 0])


def test_roundoff_tolerance_does_not_hide_probability_errors():
    np.testing.assert_array_equal(probability_roundoff([0., .5, np.nextafter(1., 2.)]), [0., .5, 1.])
    for invalid in ([1.000000001], [-1e-16], [np.nan]):
        with pytest.raises(ValueError): probability_roundoff(invalid)


def test_pairing_kernel_roundoff_is_logged_and_patch_restored():
    engine = SimpleNamespace(prior=np.array([.5, .5]), hypotheses_mask=np.ones(2),
                             get_module=lambda role: SimpleNamespace(trial_index=3))
    memory = HierarchicalPairingMemoryModule(engine, forgetting_mode='factorized')
    kernel = np.ones((2, 3)); kernel[0, 0] = np.nextafter(1., 2.)
    original = HierarchicalPairingMemoryModule.prepare_feedback
    with pytest.raises(ValueError): memory.prepare_feedback(kernel)
    with retention_roundoff_guard(Transition) as corrections:
        np.testing.assert_allclose(memory.prepare_feedback(kernel), 1.)
        assert len(corrections) == 1 and corrections[0]['phase'] == 'pairing_kernel'
        assert corrections[0]['trial'] == 2
    assert HierarchicalPairingMemoryModule.prepare_feedback is original


def test_adapter_preserves_parent_draw_and_rng_calls():
    a, b = np.random.default_rng(4), np.random.default_rng(4)
    proxy = _ProposalDraw(a, np.array([3, 4]), np.array([.9, .1]))
    assert proxy.choice(2, p=[.6, .4]) == b.choice(2, p=[.6, .4])
    assert proxy.choice([3, 4], p=[.5, .5]) == b.choice([3, 4], p=[.9, .1])
    assert proxy.calls == 1 and a.bit_generator.state == b.bit_generator.state


@pytest.mark.parametrize('old_score,accepted', [(.8, False), (-.1, True)])
def test_candidate_feedback_evidence_controls_capacity_admission(old_score, accepted, monkeypatch, tmp_path):
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = config(1)
    cfg['modules']['hypo_transitions_mod']['kwargs'].update(max_capacity=2, exit_rate=0.)
    with feedback_intervention(Transition, admission=True):
        model = StateModel(cfg, context=ModelContext(condition=1))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        h.active = np.array([0, 1]); h.executed_hypothesis = 0; h.trial_index = 1
        model.engine.posterior = np.zeros(h.total_hypo); model.engine.posterior[:2] = [.8, .2]
        model.engine.prior = model.engine.posterior.copy(); h._apply_mask()
        h.retention_evidence[:2] = [2., old_score]; h.retention_exposure[:2] = 1.
        h.record_outcome((np.array([.8,.3,.2,.4]), 1, 1.))
        h.outcome_pending = False  # Controlled old scores already include the last completed feedback.
        h.current_event_probability = 1.
        monkeypatch.setattr(model.engine.observation_likelihood, 'binary_feedback_likelihood',
                            lambda observation, rules, beta: np.full(len(rules), .9))
        h.process(); event = h.transition_log[-1]; candidate = event['proposed_hypothesis']
        assert bool(candidate in h.active) == accepted
        assert event.get('newcomer_rejected', False) == (not accepted)
        if accepted:
            assert h.retention_evidence[candidate] == pytest.approx(np.log(.9/.5))
            assert h.retention_exposure[candidate] == 1.


@pytest.mark.parametrize('condition', [1, 2, 3])
@pytest.mark.parametrize('proposal,admission', [(True, False), (False, True), (True, True)])
def test_completed_feedback_survives_particle_resume(condition, proposal, admission, monkeypatch, tmp_path):
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    original = Transition.record_outcome
    with feedback_intervention(Transition, proposal=proposal, admission=admission):
        a = StateModel(config(condition), context=ModelContext(condition=condition))
        a.begin_trial([.8, .3, .2, .4])
        memory = a.engine.get_module(ModuleRole.MEMORY)
        before = memory.pairing_marginal() if condition == 3 else None
        a.complete_trial(1, .5 if condition == 3 else 0.)
        h = a.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        if condition == 3: np.testing.assert_array_equal(h.proposal_pairing, before)
        saved = a.engine.state_dict()
        b = StateModel(config(condition), context=ModelContext(condition=condition))
        b.engine.load_state_dict(deepcopy(saved))
        for trial in range(5):
            pa, pb = [m.begin_trial([.2, .7, .8, .6]) for m in (a, b)]
            np.testing.assert_array_equal(pa.prior, pb.prior)
            ha, hb = [m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION) for m in (a, b)]
            np.testing.assert_array_equal(ha.active, hb.active)
            assert ha.trial_rng.bit_generator.state == hb.trial_rng.bit_generator.state
            assert 'candidate_feedback_probability' in ha.transition_log[-1]
            if admission and ha.last_transition_result.selection.newcomers.size:
                ids = ha.last_transition_result.selection.newcomers
                np.testing.assert_allclose(ha.retention_exposure[ids], 1)
                event = ha.transition_log[-1]
                assert np.isnan(event['swap_probability']) and not event['analytic_admission_available']
                np.testing.assert_array_equal(event['retention_evidence'], ha.retention_evidence[ha.active])
            for m in (a, b): m.complete_trial(1, float(trial % 2))
    assert Transition.record_outcome is original


@pytest.mark.parametrize('mode,upper', [('fast', 25.), ('base', 100.), ('log', 100.)])
@pytest.mark.parametrize('condition', [1, 3])
def test_precision_bounds_and_delayed_onset(mode, upper, condition, monkeypatch, tmp_path):
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)
    cfg = config(condition)
    plain = StateModel(cfg, context=ModelContext(condition=condition))
    expected = []
    for t in range(2):
        plain.begin_trial([.8, .3, .2, .4]); plain.complete_trial(1, 1.)
        expected.append(plain.engine.beta.copy())
    with precision_intervention(BetaModule, mode=mode, upper=upper, onset=2):
        model = StateModel(cfg, context=ModelContext(condition=condition))
        for t in range(2):
            model.begin_trial([.8, .3, .2, .4]); model.complete_trial(1, 1.)
            np.testing.assert_array_equal(expected[t], model.engine.beta)
        for t in range(3):
            model.begin_trial([.8, .3, .2, .4]); model.complete_trial(1, 1.)
        assert np.isfinite(model.engine.beta).all()
        assert model.engine.beta.max() <= upper
    assert 'feedback_precision_gradient' not in model.engine.observation_likelihood.__dict__


def test_log_coordinate_chain_rule_and_bounds():
    value = precision_step([5.], [.1], .2, mode='log', lower=.1, upper=100.)
    np.testing.assert_allclose(value, [5 * np.exp(.2 * 5 * .1)])
    value = precision_step([5., 5.], [-1e6, 1e6], 1., mode='log', lower=.1, upper=100.)
    np.testing.assert_allclose(value, [.1, 100.])
