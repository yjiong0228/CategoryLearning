"""Protect path reuse against changing the cognitive task or random replicate."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_online_parameter_cross import compatible_probe, summarize


def fixture():
    point = dict(id='saved', hyperparams={'capacity': 6})
    template = dict(phase='autonomous', folder='original/task', model='SEARCH',
                    mechanism={'kind': 'E'}, runtime='original/runtime', particles=512,
                    arm_settings={'precision': 'fast'}, structural_intervention='retained_rule_precision',
                    retain_precision=True, seeds=[3, 5], smoke=False, online_belief=True)
    probe = dict(deepcopy(template), candidate=deepcopy(point), name='old', target='old/chunk')
    return template, probe, point


def test_reuse_allows_different_output_labels_only():
    template, probe, point = fixture()
    template.update(name='new', target='new/chunk')
    assert compatible_probe(template, probe, point)


@pytest.mark.parametrize('field', ['phase', 'folder', 'model', 'mechanism', 'runtime',
                                  'particles', 'arm_settings', 'structural_intervention',
                                  'retain_precision', 'seeds', 'smoke', 'online_belief', 'candidate'])
def test_refuses_reuse_for_changed_task_mechanism_parameter_or_seed(field):
    template, probe, point = fixture()
    changes = dict(phase='evaluate', folder='other/task', model='POOL', mechanism={'kind': 'R'},
                   runtime='other/runtime', particles=128, arm_settings={'precision': 'old'},
                   structural_intervention='reset', retain_precision=False, seeds=[5, 3],
                   smoke=True, online_belief=False, candidate=dict(id='saved', hyperparams={'capacity': 2}))
    probe[field] = changes[field]
    assert not compatible_probe(template, probe, point)


def test_identical_banks_have_zero_mechanism_and_parameter_contrasts(tmp_path):
    categories = np.tile(np.arange(1, 5), 2)
    choices = np.ones(8, dtype=int)
    conditional = dict(seeds=np.array([1, 2, 3]), choices=choices, categories=categories,
                       observed_probability=np.full((3, 8), .25))
    generated = np.stack([choices, categories, choices, categories])
    automatic = dict(seeds=np.arange(4), choices=generated, categories=categories,
                     probabilities=np.full((4, 8, 4), .25), correct_probability=np.full((4, 8), .25),
                     feedback=(generated == categories).astype(float), present=np.ones((4, 8)),
                     contribution=np.full((4, 8), .5), beta=np.ones((4, 8)), active_count=np.ones((4, 8)),
                     search=np.zeros((4, 8)), target_exited=np.zeros((4, 8)),
                     target_capacity_dropped=np.zeros((4, 8)))
    meta = dict(condition=2, training_trials=4, particles=2, target_rule=0,
                candidate={'hyperparams': {'engine.modules.hypo_transitions_mod.kwargs.max_capacity': 2}})
    cells = ['RETAIN_RETAIN', 'ONLINE_RETAIN', 'RETAIN_ONLINE', 'ONLINE_ONLINE', 'ONLINE_NEAR']
    rows = []
    for cell in cells:
        row = dict(cell=cell)
        for role, arrays in [('evaluate', conditional), ('autonomous', automatic)]:
            target = tmp_path / f'{cell}_{role}'
            np.savez_compressed(target.with_suffix('.npz'), **arrays)
            row[role] = dict(target=str(target), record=deepcopy(meta))
        rows.append(row)
    summarize(tmp_path, {'settings': {'features': {'block_width': 2}}}, rows)
    scores = pd.read_csv(tmp_path / 'scores.csv')
    np.testing.assert_allclose(scores.choice_nll, np.log(4), rtol=0, atol=1e-12)
    contrasts = pd.read_csv(tmp_path / 'contrasts.csv')
    assert len(contrasts) == 15
    assert (contrasts.filter(regex='^delta_') == 0).all().all()
    assert len(pd.read_csv(tmp_path / 'paths.csv')) == 20
