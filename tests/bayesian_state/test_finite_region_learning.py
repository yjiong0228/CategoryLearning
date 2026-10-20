from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition import episodic_admission as admission
from src.Bayesian_state.model.modules.hypothesis_transition.finite_region_learning import (
    finite_region_learning, frozen_association_recall,
)
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule as Transition,
)
from src.Bayesian_state.model.modules.readout.choice import read_choice_probabilities_from_model
from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
from src.Bayesian_state.inference.backends import particle_filter as filter_module
from tests.bayesian_state.test_model_0923_continuous_recalled_belief import config


@pytest.fixture(autouse=True)
def isolated_geometry_cache(monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)


def configuration():
    cfg = config(2)
    cfg['modules']['hypo_transitions_mod']['kwargs']['search_controller']['global_search'] = 1.
    cfg['modules']['hypo_transitions_mod']['kwargs']['max_capacity'] = 2
    cfg['modules']['memory_mod']['kwargs']['gamma'] = .3
    return cfg


def test_real_engine_readout_feedback_precision_and_absence_share_preupdate_state():
    with finite_region_learning(Transition, learning_rate=.5):
        model = StateModel(configuration(), context=ModelContext(condition=2))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        beta = model.engine.get_module(ModuleRole.BETA)
        rng = np.random.default_rng(12)
        seen = set()
        for trial in range(12):
            prepared = model.begin_trial(rng.uniform(.05, .95, 4))
            active = h.active.copy(); seen.update(active)
            before = h.region_associations.state_dict()
            prediction, snapshot = read_choice_probabilities_from_model(
                model, prepared.perceived_stimulus, return_snapshot=True)
            q = np.stack([model.partition_model.get_category_probabilities(
                int(rule), ([prepared.perceived_stimulus], [1], [1.]), float(model.engine.beta[rule]),
                distance_mode='boundary')[:, 0] for rule in active])
            per_rule = h.region_associations.predict(active, q)
            np.testing.assert_allclose(prediction, h.execution_probabilities[active] @ per_rule,
                                       atol=1e-12, rtol=0)
            choice, feedback = trial % 4 + 1, float(trial % 2)
            evaluator = model.engine.observation_likelihood
            evaluation = evaluator.evaluate((prepared.perceived_stimulus, choice, feedback), active,
                                              model.engine.beta[active], precision=True)
            old_beta = model.engine.beta[active].copy()
            model.complete_trial(choice, feedback)
            expected_event = per_rule[:, choice-1] if feedback else np.delete(per_rule, choice-1, axis=1).sum(1)
            np.testing.assert_allclose(model.engine.likelihood[active], expected_event, atol=1e-12, rtol=0)
            expected_beta = np.clip(old_beta + beta.learning_rate*evaluation.precision_gradient,
                                    beta.beta_min, beta.beta_max)
            np.testing.assert_allclose(model.engine.beta[active], expected_beta, atol=1e-12, rtol=0)
            after = h.region_associations.state_dict()
            absent = np.setdiff1d(np.arange(h.total_hypo), active)
            np.testing.assert_array_equal(after['logits'][absent], before['logits'][absent])
            np.testing.assert_array_equal(after['feedback_counts'][active], before['feedback_counts'][active]+1)
            if trial == 0:
                np.testing.assert_allclose(old_beta, model.engine.beta[active], atol=1e-12, rtol=0)
                np.testing.assert_allclose(prediction, .25, atol=1e-12, rtol=0)
        assert len(seen) >= 2
        assert any(row.get('newcomer_rejected') for row in h.transition_log)


def test_recall_uses_saved_associations_and_current_precision_without_learning():
    with finite_region_learning(Transition, learning_rate=.7):
        model = StateModel(configuration(), context=ModelContext(condition=2))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for _ in range(3):
            model.begin_trial([.8, .3, .2, .4]); model.complete_trial(1, 1.)
        evaluator = model.engine.observation_likelihood
        episodes, _ = admission.recalled_episodes(h.admission_history, .8)
        rules = np.arange(h.total_hypo)
        before = h.region_associations.state_dict()
        scores, _ = frozen_association_recall(evaluator, episodes, rules, -999.)
        betas = evaluator.recall_precision(rules)
        for row, episode in enumerate(episodes):
            likelihood = evaluator.binary_feedback_likelihood(
                (episode.stimulus, episode.choice, episode.feedback), rules, betas)
            np.testing.assert_allclose(scores[row], np.log(likelihood)-np.log(.25), atol=1e-12, rtol=0)
        never = before['feedback_counts'] == 0
        np.testing.assert_array_equal(scores[:, never], 0.)
        np.testing.assert_array_equal(h.region_associations.state_dict()['logits'], before['logits'])
        np.testing.assert_array_equal(h.region_associations.state_dict()['feedback_counts'], before['feedback_counts'])


def test_real_particle_snapshot_continuation_and_independent_association_storage():
    cfg = configuration()
    with finite_region_learning(Transition, learning_rate=.5):
        a = StateModel(cfg, context=ModelContext(condition=2))
        for trial in range(5):
            a.begin_trial([.8, .3, .2, .4]); a.complete_trial(1, float(trial % 2))
        b = StateModel(cfg, context=ModelContext(condition=2)); b.engine.load_state_dict(deepcopy(a.engine.state_dict()))
        ha, hb = [m.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION) for m in (a, b)]
        assert ha.region_associations is not hb.region_associations
        assert b.engine.observation_likelihood.associations is hb.region_associations
        for trial in range(5):
            pa, pb = [m.begin_trial([.2, .7, .8, .6]) for m in (a, b)]
            np.testing.assert_array_equal(pa.prior, pb.prior)
            np.testing.assert_array_equal(pa.beta, pb.beta)
            np.testing.assert_array_equal(ha.active, hb.active)
            qa, qb = [read_choice_probabilities_from_model(m, [.2, .7, .8, .6]) for m in (a, b)]
            np.testing.assert_array_equal(qa, qb)
            for m in (a, b): m.complete_trial(3, float(trial % 2))
            np.testing.assert_array_equal(ha.region_associations.associations(range(ha.total_hypo)),
                                          hb.region_associations.associations(range(hb.total_hypo)))


def test_adapter_restores_all_methods_and_rejects_unsupported_oral_route():
    originals = (StateModel.complete_trial, StateModel.predict_oral, Transition.__init__,
                 admission.recalled_log_evidence, filter_module.apply_hypothesis_mapping)
    with finite_region_learning(Transition, learning_rate=.5):
        model = StateModel(configuration(), context=ModelContext(condition=2))
        with pytest.raises(ValueError, match='region_basis'): model.predict_oral(1)
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        model.begin_trial([.3, .2, .7, .4])
        np.testing.assert_allclose(filter_module._map_hypothesis_choice_probability(
            model, np.array([.3, .2, .7, .4]), lapse=0.), .25, atol=1e-12, rtol=0)
        before = h.region_associations.state_dict()
        model.complete_trial(1, 0., update_state=False)
        np.testing.assert_array_equal(h.region_associations.state_dict()['logits'], before['logits'])
        np.testing.assert_array_equal(h.region_associations.state_dict()['feedback_counts'], before['feedback_counts'])
    assert originals == (StateModel.complete_trial, StateModel.predict_oral, Transition.__init__,
                         admission.recalled_log_evidence, filter_module.apply_hypothesis_mapping)


def test_forced_reentry_preserves_associations_and_precision_in_the_real_engine(monkeypatch):
    # Intervene on selection only to cover repeated exit/reentry deterministically.
    # This fixture is a lifecycle check, not a performance simulation.
    def draw(self, before, remaining, gamma, diagnostics):
        candidate = next(rule for rule in (0, 1, 2) if rule not in before)
        diagnostics.update(proposal_source='test_schedule', proposed_hypothesis=candidate)
        return candidate

    monkeypatch.setattr(Transition, '_draw_v3_proposal', draw)
    with finite_region_learning(Transition, learning_rate=.5):
        ordinary_scores = Transition._retention_scores

        def force_admission(self, rules, mean=False):
            values = ordinary_scores(self, rules, mean)
            values[0] = values.min()-1.
            return values

        monkeypatch.setattr(Transition, '_retention_scores', force_admission)
        model = StateModel(configuration(), context=ModelContext(condition=2))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        known, returns = set(), 0
        for trial in range(9):
            before = h.region_associations.state_dict()
            model.begin_trial([.8, .3, .2, .4])
            np.testing.assert_array_equal(h.region_associations.state_dict()['logits'], before['logits'])
            np.testing.assert_array_equal(h.region_associations.state_dict()['feedback_counts'], before['feedback_counts'])
            selection = h.last_transition_result.selection
            for rule in selection.newcomers:
                if rule in known:
                    returns += 1
                    assert model.engine.beta[rule] == h.reentry_precision_values[rule]
            known.update(h.active)
            model.complete_trial(1, float(trial % 2))
        assert returns >= 2


def test_local_label_dependent_proposal_is_rejected():
    with finite_region_learning(Transition, learning_rate=.5):
        with pytest.raises(ValueError, match='global search'):
            StateModel(config(2), context=ModelContext(condition=2))


def test_particle_filter_resampling_reproducibility_and_feedback_causality():
    stimuli = np.random.default_rng(923).uniform(.05, .95, (8, 4))
    kwargs = dict(engine_config=configuration(), subject_id=207, condition=2,
                  stimulus=stimuli, choices=np.tile([1, 2], 4), particle_count=4,
                  choice_readout_power=1., filter_seed=923, resample_threshold_fraction=1.)
    feedback = np.tile([0., 1.], 4)
    with finite_region_learning(Transition, learning_rate=.5):
        first = run_state_model_particle_filter(feedback=feedback, **kwargs)
        again = run_state_model_particle_filter(feedback=feedback, **kwargs)
        changed = feedback.copy(); changed[-1] = 0.
        altered = run_state_model_particle_filter(feedback=changed, **kwargs)
    np.testing.assert_array_equal(first.marginal_probabilities, again.marginal_probabilities)
    np.testing.assert_array_equal(first.final_weights, again.final_weights)
    np.testing.assert_array_equal(first.marginal_probabilities, altered.marginal_probabilities)
    np.testing.assert_array_equal(first.final_weights, altered.final_weights)
    np.testing.assert_allclose(first.marginal_probabilities.sum(1), 1., atol=1e-12, rtol=0)
    np.testing.assert_allclose(first.marginal_probabilities[0], .25, atol=1e-12, rtol=0)
    assert first.resampled.any()
