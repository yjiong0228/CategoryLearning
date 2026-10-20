from copy import deepcopy

import numpy as np
import pytest

from src.Bayesian_state.model import ModelContext, StateModel
from src.Bayesian_state.model.modules.base_module import ModuleRole
from src.Bayesian_state.model.modules.hypothesis_transition.family_balanced_proposal import (
    balanced_global_proposal, family_proposal,
)
from src.Bayesian_state.model.modules.hypothesis_transition.finite_region_learning import finite_region_learning
from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import (
    MechanisticRuleSearchHypothesisTransitionModule as Transition,
)
from src.Bayesian_state.model.modules.readout.choice import read_choice_probabilities_from_model
from tests.bayesian_state.test_finite_region_learning import configuration


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    from src.Bayesian_state.hypothesis_space.similarity import ContinuousSimilarity
    monkeypatch.setattr(ContinuousSimilarity, 'RUNTIME_CACHE_DIR', tmp_path)


def test_equal_family_mass_tracks_only_available_members():
    families = ('a', 'a', 'a', 'b', 'c', 'c')
    for ids in (np.arange(6), np.array([0, 3, 4, 5]), np.array([0, 1, 2, 4, 5])):
        p = family_proposal(families, ids)
        present = {families[i] for i in ids}
        for family in present:
            np.testing.assert_allclose(sum(p[j] for j, i in enumerate(ids) if families[i] == family), 1/len(present))
        np.testing.assert_allclose(p.sum(), 1.)
    for invalid in (np.array([]), np.array([0, 0]), np.array([6]), np.array([1.5])):
        with pytest.raises(ValueError): family_proposal(families, invalid)


def test_real_model_keeps_initial_draw_and_prior_and_runs_rehearsal_after_proposal():
    states = []
    for enabled in (False, True):
        with balanced_global_proposal(Transition, enabled=enabled), finite_region_learning(
                Transition, learning_rate=.5, rehearse_candidates=True):
            model = StateModel(configuration(), context=ModelContext(condition=2))
            h = model.engine.get_module(ModuleRole.HYPOTHESIS_TRANSITION)
            states.append((h.active.copy(), h.base_prior.copy()))
            for _ in range(8):
                model.begin_trial([.8, .3, .2, .4]); model.complete_trial(1, 1.)
            proposals = [event for event in h.transition_log if event.get('search_event')]
            assert proposals
            for event in proposals:
                assert event['candidate_rehearsal'] == (not event['candidate_previously_active'])
                if enabled:
                    assert event['proposal_source'] == 'global_equal_family'
                    inactive = np.setdiff1d(np.arange(h.total_hypo), event['active_before'])
                    expected = family_proposal(h._proposal_families, inactive)
                    index = np.flatnonzero(inactive == event['proposed_hypothesis'])[0]
                    assert event['proposed_rule_probability'] == expected[index]
    for a, b in zip(*states): np.testing.assert_array_equal(a, b)


def test_balanced_proposal_snapshot_roundtrip_reproduces_learning_path():
    with balanced_global_proposal(Transition, enabled=True), finite_region_learning(
            Transition, learning_rate=.5, rehearse_candidates=True):
        model = StateModel(configuration(), context=ModelContext(condition=2))
        for _ in range(5): model.begin_trial([.8, .3, .2, .4]); model.complete_trial(1, 1.)
        state = deepcopy(model.engine.state_dict())
        other = StateModel(configuration(), context=ModelContext(condition=2)); other.engine.load_state_dict(state)
        for trial in range(8):
            predictions = []
            for instance in (model, other):
                prepared = instance.begin_trial([.2, .7, .6, .4])
                predictions.append(read_choice_probabilities_from_model(instance, prepared.perceived_stimulus))
                instance.complete_trial(trial % 4+1, float(trial % 3 == 0))
            np.testing.assert_array_equal(*predictions)
