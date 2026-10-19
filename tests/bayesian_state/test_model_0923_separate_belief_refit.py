"""The new evidence mechanism keeps the completed behavior-search policy."""
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from src.Bayesian_state.workflows.analysis.model_0923_separate_belief_refit import validate_settings


def settings():
    old = yaml.safe_load(Path('configs/shared/model0923_online_evidence_refit_s208.yaml').read_text())
    new = yaml.safe_load(Path('configs/shared/model0923_separate_belief_refit_s208.yaml').read_text())
    return new, old


def test_preserves_search_policy_at_all_existing_variable_coordinates():
    new, old = settings(); validate_settings(new, old)
    assert new['initial_candidates'] == 40 and list(new['arms']) == ['POOL', 'SEARCH']
    assert len(new['fixed_coordinates']) == 1


@pytest.mark.parametrize('change', ['budget', 'seed', 'scope_data', 'warm_start', 'rate', 'decay', 'fixed', 'belief', 'arm'])
def test_rejects_changed_search_policy(change):
    new, old = settings(); new = deepcopy(new)
    if change == 'budget': new['budgets']['confirmation']['particles'] = 128
    elif change == 'seed': new['seed'] += 1
    elif change == 'scope_data': new['data'] = 'other'
    elif change == 'warm_start': new['warm_start_fit'] = 'other'
    elif change == 'rate': new['arms']['SEARCH']['rates'] = [.3]
    elif change == 'decay': new['evidence_decay_values'] = [.3]
    elif change == 'fixed': new['fixed_coordinates']['gamma'] = 1.
    elif change == 'belief': new['belief_intervention'] = 'online_retention_evidence'
    elif change == 'arm': new['arms'].pop('POOL')
    with pytest.raises(ValueError): validate_settings(new, old)
