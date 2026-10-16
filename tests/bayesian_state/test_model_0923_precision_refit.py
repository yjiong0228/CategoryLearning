"""Matched supports, explicit learning-rate units, and finalist protection."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.workflows.analysis import model_0923_precision_refit as workflow
from src.Bayesian_state.model.modules.precision_interventions import precision_step


@pytest.fixture
def settings():
    return yaml.safe_load(Path('configs/shared/model0923_precision_refit.yaml').read_text())


def fake_nominees(settings):
    space = workflow.model_space(settings, 'ADD1')
    anchor = {k: values[len(values)//2] for k,values in space.items()}
    rows = []
    for subject in settings['subjects']:
        for old in ('FREE', 'CUMULATIVE', 'SHARED'):
            point = deepcopy(anchor)
            if old == 'CUMULATIVE': point[workflow.DECAY] = 1.
            if old == 'SHARED': point.pop(workflow.DECAY)
            rows.append(dict(subject=subject,model=old,candidate=dict(hyperparams=point)))
    return dict(nominees=rows)


def test_equal_banks_and_log_ceiling_isolation(settings):
    banks, required = workflow.starting_banks(settings, fake_nominees(settings), False)
    again, _ = workflow.starting_banks(settings, fake_nominees(settings), False)
    assert banks == again
    assert len(banks) == len(settings['subjects'])*4
    for key, rows in banks.items():
        assert len(rows) == settings['initial_candidates']
        assert len({r['id'] for r in rows}) == len(rows)
        assert required[key]['id'] in {r['id'] for r in rows}
        space = workflow.model_space(settings, key[1])
        assert all(set(r['hyperparams'])==set(space) for r in rows)
        assert all(all(r['hyperparams'][k] in vals for k,vals in space.items()) for r in rows)
    for subject in settings['subjects']:
        assert banks[subject,'LOG25'] == banks[subject,'LOG100']


def test_additive_warm_mapping_preserves_effective_step(settings):
    point = fake_nominees(settings)['nominees'][0]['candidate']['hyperparams']
    for rate in settings['arms']['ADD1']['rates']:
        point[workflow.RATE] = rate
        mapped = workflow.project(point, 'FREE','ADD10',settings)
        assert mapped[workflow.RATE]*10 == pytest.approx(rate)
        np.testing.assert_allclose(
            precision_step([2.,10.],[.3,-.2],rate,mode='base',lower=.1,upper=25),
            precision_step([2.,10.],[.3,-.2],mapped[workflow.RATE],mode='fast',lower=.1,upper=25))


def test_shared_retention_projection_and_log_zero_rate(settings):
    point = fake_nominees(settings)['nominees'][2]['candidate']['hyperparams']
    point[workflow.RATE] = 0.
    mapped = workflow.project(point,'SHARED','LOG25',settings)
    assert mapped[workflow.DECAY] == point[workflow.MEMORY]
    assert mapped[workflow.RATE] == 0.


def test_old_reference_survives_noisy_elimination():
    rows = [dict(id='a',mean_nll=.1),dict(id='b',mean_nll=.2),dict(id='c',mean_nll=.3)]
    required = dict(id='old',mean_nll=.8)
    assert [r['id'] for r in workflow.shortlist(rows,required,2)] == ['old','a']
    with pytest.raises(ValueError): workflow.shortlist(rows,required,5)
