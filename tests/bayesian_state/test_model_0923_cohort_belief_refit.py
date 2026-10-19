"""A cohort adapter preserves numerical policy and development boundaries."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.Bayesian_state.workflows.analysis import model_0923_cohort_belief_refit as cohort


def sources():
    reference = yaml.safe_load(Path('configs/shared/model0923_online_evidence_refit_s208.yaml').read_text())
    config = yaml.safe_load(Path('configs/shared/model0923_cohort_belief_refit.yaml').read_text())
    return config, reference


def test_both_mechanisms_inherit_identical_numerical_policy_without_mutating_reference():
    config, reference = sources(); before = deepcopy(reference)
    online = cohort.expanded_settings(config, reference, {107, 207, 306}, 'ONLINE')
    separate = cohort.expanded_settings(config, reference, {107, 207, 306}, 'SEPARATE')
    varied = {'belief_mechanism', 'belief_intervention'}
    assert {k: v for k, v in online.items() if k not in varied} == {k: v for k, v in separate.items() if k not in varied}
    inherited = set(reference) - cohort.REFERENCE_KEYS - {'subjects', 'scope', 'parallel_budget', 'belief_intervention'}
    assert all(online[k] == separate[k] == reference[k] for k in inherited)
    assert reference == before
    online['budgets']['screen']['particles'] = 1
    assert separate['budgets']['screen']['particles'] == reference['budgets']['screen']['particles'] == 128
    online['subjects'].append(208)
    assert separate['subjects'] == config['subjects'] == [107, 207, 306]
    assert not set(online) & cohort.REFERENCE_KEYS


@pytest.mark.parametrize('subjects', [[], [110], [107, 110], [107, 107], ['107'], [True], (107,)])
def test_rejects_unlisted_or_ambiguous_development_participants(subjects):
    config, reference = sources(); config['subjects'] = subjects
    with pytest.raises(ValueError):
        cohort.expanded_settings(config, reference, {107, 207, 306}, 'ONLINE')


@pytest.mark.parametrize('budget', [0, 129, 1.5, True])
def test_rejects_invalid_worker_budget(budget):
    config, reference = sources(); config['parallel_budget'] = budget
    with pytest.raises(ValueError):
        cohort.expanded_settings(config, reference, {107, 207, 306}, 'ONLINE')


@pytest.mark.parametrize('key', ['seed', 'budgets', 'parameter_support', 'data', 'fixed_coordinates'])
def test_config_cannot_silently_override_the_common_policy(key):
    config, reference = sources(); config[key] = 'changed'
    with pytest.raises(ValueError):
        cohort.expanded_settings(config, reference, {107, 207, 306}, 'SEPARATE')


def test_requires_known_mechanism_and_online_reference():
    config, reference = sources()
    with pytest.raises(ValueError):
        cohort.expanded_settings(config, reference, {107, 207, 306}, 'UNKNOWN')
    reference['belief_intervention'] = 'separate_gamma_belief_evidence'
    with pytest.raises(ValueError):
        cohort.expanded_settings(config, reference, {107, 207, 306}, 'ONLINE')
