"""Validate the pilot's fixed proposals, semantic summaries and both runtimes."""
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.Bayesian_state.workflows.analysis.model_1031_subject_pilot import (
    DEFAULT, bank, engine_config, active_mask_distribution, rank)
from src.Bayesian_state.model.model_1024 import model_1024_runtime
from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter


def test_candidate_bank_is_frozen_and_coarse():
    p = yaml.safe_load(Path('configs/exp123/specific_models/model_1031_subject118_pilot_v1.yaml').read_text())
    a = bank(p)
    assert a == bank(p) and len(a) == 16
    assert a[0]['parameters'] == DEFAULT
    assert len({tuple(r['parameters'].values()) for r in a}) == 16
    for point in a[1:]:
        assert all(value in p['parameter_values'][key] for key, value in point['parameters'].items())


def test_feature_sets_integrate_particle_weight_and_executed_content():
    state = dict(weights=np.array([.3, .7]), choice_weights=np.array([[1., 0.], [0., 1.]]),
        representation_kind=np.array([[1, 2], [1, 2]]), catalogue_rule=np.array([[-1, 0], [-1, 0]]),
        template_mask=np.array([[[1,1,1,1], [0,0,0,0]], [[1,0,0,0], [0,0,0,0]]], bool))
    result = active_mask_distribution(state, np.array([[0, 1, 0, 0]], bool))
    assert result[15] == pytest.approx(.3) and result[2] == pytest.approx(.7)


def test_ranking_averages_independent_full_record_nll_without_oral():
    records = [dict(job=dict(model='model_1031', candidate=c), nll=n)
               for c, n in [('a', 10.), ('a', 14.), ('b', 11.), ('b', 11.)]]
    assert rank(records)[0]['candidate'] == 'b'
    assert rank(records)[1]['mean_nll'] == 12.


@pytest.mark.parametrize('model', ['model_1031', 'model_1024'])
def test_pilot_profiles_run_shared_filter_and_observer(model):
    cfg = engine_config(model, DEFAULT)
    cfg['modules']['perception_mod']['kwargs'] = dict(mean=[0.] * 4, std=[0.] * 4, module_seed=2)
    seen = []
    with model_1024_runtime() if model == 'model_1024' else nullcontext():
        result = run_state_model_particle_filter(engine_config=cfg, subject_id=118, condition=1,
            stimulus=np.random.default_rng(2).uniform(.1, .9, (4, 4)), choices=[1,2,1,2],
            feedback=[1.,0.,1.,0.], particle_count=2, choice_readout_power=1.,
            state_observer=lambda t, phase, state: seen.append((t, phase)), observe_decision_basis=True)
    np.testing.assert_allclose(result.marginal_probabilities.sum(axis=1), 1.)
    assert len(seen) == 8
