"""The sampler adapter leaves base validation functions and scoring unchanged."""
from copy import deepcopy

import pytest

from src.Bayesian_state.workflows.analysis.model_0923_h_candidate_validation import (
    HELPER, SAMPLER, configure, core_job, fixed_models, validation,
)


def settings():
    return dict(models={'B': {'kind': 'B'}, 'JF': {'kind': 'J', 'refresh': 'search', 'decay': .95, 'cost': 0., 'candidate_sampler': SAMPLER}},
        fixed_size_models=['JF'])


def test_sampler_protocol_is_explicit_and_preserves_settings():
    original = settings(); copied = deepcopy(original)
    assert fixed_models(original) == {'JF'}
    assert original == copied


@pytest.mark.parametrize('names', [[], ['missing'], ['JF', 'JF'], ['B']])
def test_invalid_sampler_names_are_rejected(names):
    config = settings(); config['fixed_size_models'] = names
    with pytest.raises(ValueError):
        fixed_models(config)


def test_every_trial_refresh_cannot_use_single_proposal_competition():
    config = settings(); config['models']['JF']['refresh'] = 'every_trial'
    with pytest.raises(ValueError):
        fixed_models(config)


def test_sampler_marker_cannot_be_silently_ignored_or_installed_on_baseline():
    config = settings(); del config['models']['JF']['candidate_sampler']
    with pytest.raises(ValueError, match='marker'):
        fixed_models(config)
    job = dict(mechanism=settings()['models']['JF'], candidate_sampler='original')
    with pytest.raises(ValueError, match='differ'):
        core_job(job)
    job['candidate_sampler'] = SAMPLER
    stripped = core_job(job)
    assert 'candidate_sampler' not in stripped['mechanism']
    assert job['mechanism']['candidate_sampler'] == SAMPLER


def test_unadapted_shared_transition_rejects_sampler_marker():
    from types import SimpleNamespace
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import MechanisticRuleSearchHypothesisTransitionModule
    probe = SimpleNamespace(intervention=settings()['models']['JF'], continuous_choice_enabled=True)
    with pytest.raises(ValueError, match='unsupported'):
        MechanisticRuleSearchHypothesisTransitionModule._validate_intervention(probe)


def test_adapter_preserves_scoring_search_and_preparation_implementations():
    v = validation(); original = {name: getattr(v, name) for name in ['score', 'search', 'prepare', 'autonomous', 'evaluate', 'model_space']}
    baseline_hashes = v.source_hashes()
    configure(v, settings())
    assert all(getattr(v, name) is function for name, function in original.items())
    hashes = v.source_hashes()
    assert all(hashes[name] == sha for name, sha in baseline_hashes.items())
    assert HELPER in hashes
    assert 'src/Bayesian_state/workflows/analysis/model_0923_h_candidate_validation.py' in hashes
