from copy import deepcopy

import pytest

from src.Bayesian_state.workflows.analysis.model_0923_execution_parameter_cross import cross_job


@pytest.mark.parametrize('policy', ['MIX', 'SELECT'])
@pytest.mark.parametrize('smoke', [False, True])
def test_cross_preserves_parameters_seeds_and_original_job(tmp_path, policy, smoke):
    original = dict(phase='autonomous', candidate=dict(id='point', hyperparams={'a': .3}),
                    seeds=[14, 22, 31, 44], source_sha256={'original': 'hash'},
                    model='SEARCH', mechanism={'kind': 'E'}, folder='source-data')
    before = deepcopy(original)
    job = cross_job(original, policy, tmp_path / 'result', 'cross', smoke)
    assert original == before
    assert job['candidate'] == before['candidate']
    assert job['seeds'] == (before['seeds'][:1] if smoke else before['seeds'])
    assert job['mechanism'] == before['mechanism']
    assert job['folder'] == before['folder']
    assert job['model'] == before['model']
    assert job['execution_policy'] == policy


def test_cross_rejects_choice_filter_job(tmp_path):
    with pytest.raises(ValueError):
        cross_job(dict(phase='evaluate'), 'SELECT', tmp_path / 'result', 'wrong', False)
