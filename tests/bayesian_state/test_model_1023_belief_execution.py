"""Belief/execution separation, causal timing, nested mixture and replay."""
from contextlib import contextmanager, ExitStack
from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.belief_execution import belief_execution
from src.Bayesian_state.model.modules.hypothesis_transition.compact_belief import compact_belief
from src.Bayesian_state.model.modules.hypothesis_transition.belief_slot_transfer import belief_slot_transfer
from src.Bayesian_state.model.modules.hypothesis_transition.belief_error_entry import belief_error_entry
from src.Bayesian_state.model.modules.hypothesis_transition.belief_outcome_retention import belief_outcome_retention
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule as Transition)
from tests.bayesian_state.test_model_1021_compact_evidence import compact_config


@contextmanager
def installed(mode):
    with ExitStack() as stack:
        for module in (compact_belief, belief_slot_transfer, belief_outcome_retention, belief_error_entry):
            stack.enter_context(module(Transition))
        audit = stack.enter_context(belief_execution(Transition, mode=mode))
        yield audit


@pytest.fixture(autouse=True)
def isolated_similarity(monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)


def history(condition, mode, kappa=0.):
    config = compact_config(condition)
    config['modules']['hypo_transitions_mod']['kwargs']['execution_inertia'] = kappa
    rows = []
    with installed(mode) as audit:
        model = StateModel(config, context=ModelContext(condition=condition))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for trial in range(16):
            # Include transitions without discovery so execution can switch
            # independently of proposal; other trials exercise forced removal.
            h.baseline_probability = 1e-12 if trial % 4 == 3 else .999
            h.error_gain = 0.
            previous = int(h.executed_hypothesis)
            pending = model.begin_trial([.8, .3, .2, .4])
            row = dict(prior=pending.prior.copy(), beta=pending.beta.copy(), active=h.active.copy(),
                weights=h.execution_probabilities.copy(), previous=previous,
                executed=int(h.executed_hypothesis), search_rng=deepcopy(h.trial_rng.bit_generator.state))
            if mode != 'MIX':
                probability = h.execution_selection_probabilities.copy()
                if mode == 'STICKY':
                    expected = pending.prior * np.exp(kappa * (np.arange(h.total_hypo) == previous))
                else:
                    expected = (pending.prior == pending.prior.max()).astype(float)
                expected /= expected.sum()
                np.testing.assert_allclose(probability, expected, atol=1e-12)
                np.testing.assert_array_equal(h.execution_probabilities, np.eye(h.total_hypo)[h.executed_hypothesis])
                assert h.execution_forced_switch == (previous not in h.active)
                row.update(selection=probability, forced=h.execution_forced_switch)
                if mode == 'STICKY' and kappa == 0:
                    np.testing.assert_allclose(probability, pending.prior, atol=1e-12)
            feedback = (0., .5, 1.)[trial % 3] if condition == 3 else float(trial % 2)
            model.complete_trial(1, feedback)
            row['posterior'] = model.engine.posterior.copy()
            row['beta_after'] = model.engine.beta.copy()
            rows.append(row)
        return rows, dict(audit)


@pytest.mark.parametrize('condition', [1, 2, 3])
@pytest.mark.parametrize('mode,kappa', [('MAP', 0.), ('STICKY', 0.), ('STICKY', 3.)])
def test_execution_preserves_all_rule_learning_search_and_inheritance(condition, mode, kappa):
    baseline, _ = history(condition, 'MIX')
    extended, audit = history(condition, mode, kappa)
    for a, b in zip(baseline, extended):
        for key in ('prior', 'beta', 'active', 'posterior', 'beta_after'):
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        assert a['search_rng'] == b['search_rng']
    assert audit['trials'] == 16
    assert audit['switches'] == audit['forced_switches'] + audit['voluntary_switches']


def test_removed_execution_forces_reselection_without_protection():
    config = compact_config(1)
    config['modules']['hypo_transitions_mod']['kwargs']['execution_inertia'] = 8.
    with installed('STICKY'):
        model = StateModel(config, context=ModelContext(condition=1))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for _ in range(5):
            model.begin_trial([.8, .3, .2, .4]); model.complete_trial(1, 0.)
        assert len(h.active) == h.capacity
        previous = int(h.active[0])
        h.executed_hypothesis = previous
        h.execution_probabilities[:] = 0.; h.execution_probabilities[previous] = 1.
        model.engine.posterior[:] = 0.
        model.engine.posterior[h.active] = 1.
        model.engine.posterior[previous] = .0001
        model.engine.posterior /= model.engine.posterior.sum()
        h.baseline_probability = 1. - 1e-12
        pending = model.begin_trial([.2, .7, .8, .6])
        assert previous not in h.active and h.execution_forced_switch
        assert h.executed_hypothesis != previous and h.current_execution_switch_probability == 1.
        np.testing.assert_allclose(h.execution_selection_probabilities, pending.prior)


@pytest.mark.parametrize('condition', [1, 2, 3])
@pytest.mark.parametrize('mode', ['MAP', 'STICKY'])
def test_snapshot_and_future_reseeding_preserve_execution(condition, mode):
    config = compact_config(condition)
    config['modules']['hypo_transitions_mod']['kwargs']['execution_inertia'] = 2. if mode == 'STICKY' else 0.
    with installed(mode):
        model = StateModel(config, context=ModelContext(condition=condition))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for _ in range(5):
            model.begin_trial([.8, .3, .2, .4]); model.complete_trial(1, 0.)
        clone = StateModel(config, context=ModelContext(condition=condition))
        clone.engine.load_state_dict(deepcopy(model.engine.state_dict()))
        other = clone.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        assert not np.shares_memory(h.execution_selection_probabilities, other.execution_selection_probabilities)
        for index in range(5):
            a, b = [m.begin_trial([.2, .7, .8, .6]) for m in (model, clone)]
            np.testing.assert_array_equal(a.prior, b.prior)
            np.testing.assert_array_equal(h.execution_probabilities, other.execution_probabilities)
            assert h.execution_rng.bit_generator.state == other.execution_rng.bit_generator.state
            for m in (model, clone):
                m.complete_trial(1, .5 if condition == 3 else float(index % 2))
        h.reseed_future(17); other.reseed_future(17)
        assert h.execution_rng.bit_generator.state == other.execution_rng.bit_generator.state
        bad = deepcopy(h.state_dict()); bad['execution_probabilities'] = np.ones(h.total_hypo) / h.total_hypo
        with pytest.raises(ValueError, match='one-hot'):
            h.load_state_dict(bad)
        bad = deepcopy(h.state_dict()); bad['execution_selection_probabilities'][0] = np.nan
        with pytest.raises(ValueError, match='selection distribution'):
            h.load_state_dict(bad)


def test_disabled_exact_validation_and_restoration():
    original = Transition.__init__, Transition._finish_hypothesis_transition, Transition.load_state_dict
    with belief_execution(Transition):
        assert original == (Transition.__init__, Transition._finish_hypothesis_transition, Transition.load_state_dict)
    for mode, value in [('MAP', 1.), ('STICKY', -1.), ('STICKY', np.inf), ('STICKY', True)]:
        cfg = compact_config(1)
        cfg['modules']['hypo_transitions_mod']['kwargs']['execution_inertia'] = value
        with pytest.raises(ValueError, match='kappa'):
            with installed(mode):
                StateModel(cfg, context=ModelContext(condition=1))
    assert original == (Transition.__init__, Transition._finish_hypothesis_transition, Transition.load_state_dict)
    with pytest.raises(ValueError, match='mode'):
        with belief_execution(Transition, mode='unknown'):
            pass


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_actual_readout_uses_execution_and_zero_kappa_marginal_matches_mixture(condition):
    from src.Bayesian_state.model.modules.readout.choice import read_choice_probabilities_from_model
    with installed('STICKY'):
        model = StateModel(compact_config(condition), context=ModelContext(condition=condition))
        h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
        for _ in range(5):
            model.begin_trial([.8, .3, .2, .4]); model.complete_trial(1, 0.)
        pending = model.begin_trial([.2, .7, .8, .6])
        original = h.execution_probabilities.copy()
        selected, snapshot = read_choice_probabilities_from_model(
            model, pending.perceived_stimulus, return_snapshot=True)
        index = snapshot.hypothesis_indices.index(h.executed_hypothesis)
        np.testing.assert_allclose(selected, snapshot.category_probabilities[index], atol=1e-14)
        np.testing.assert_array_equal(snapshot.rule_weights, original[list(snapshot.hypothesis_indices)])
        # Explicitly marginalize every possible executed rule, rather than
        # comparing two implementations of the same weighted readout.
        marginal = np.zeros(model.n_cats)
        for rule in h.active:
            h.execution_probabilities[:] = 0.
            h.execution_probabilities[rule] = 1.
            marginal += h.execution_selection_probabilities[rule] * read_choice_probabilities_from_model(
                model, pending.perceived_stimulus)
        h.execution_probabilities = pending.prior.copy()
        mixture = read_choice_probabilities_from_model(model, pending.perceived_stimulus)
        np.testing.assert_allclose(marginal, mixture, atol=1e-14)
        h.execution_probabilities = original


@pytest.mark.parametrize('condition', [1, 2, 3])
def test_single_active_rule_makes_execution_mechanisms_observationally_identical(condition):
    from src.Bayesian_state.model.modules.readout.choice import read_choice_probabilities_from_model
    traces = []
    for mode in ('MIX', 'MAP', 'STICKY'):
        config = compact_config(condition)
        config['modules']['hypo_transitions_mod']['kwargs']['execution_inertia'] = (
            8. if mode == 'STICKY' else 0.)
        with installed(mode):
            model = StateModel(config, context=ModelContext(condition=condition))
            transition = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
            # Keep the initial one-rule state throughout this fixed-seed
            # sequence. Configured capacity remains in its valid range >= 2.
            transition.baseline_probability = 1e-12
            transition.error_gain = 0.
            predictions = []
            for t in range(12):
                pending = model.begin_trial([.8, .3, .2, .4] if t % 2 else [.2, .7, .8, .6])
                assert len(transition.active) == 1
                predictions.append(read_choice_probabilities_from_model(model, pending.perceived_stimulus))
                model.complete_trial(1, (0., .5, 1.)[t % 3] if condition == 3 else float(t % 2))
            traces.append(np.stack(predictions))
    for observed in traces[1:]:
        np.testing.assert_array_equal(observed, traces[0])
