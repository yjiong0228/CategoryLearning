import json

import numpy as np
import pytest

from src.Bayesian_state.evaluation.model_0923_episodic_refit import (
    loss_parts, active_precision_mean, training_chunk_budgets, refit_inactive_channels,
)
from src.Bayesian_state.evaluation.model_0923_h_validation import nll


def test_error_and_correct_losses_add_to_sequence_score():
    p = np.array([[.8,.25,.6],[.6,.5,.5]])
    values = loss_parts(p,[True,False,True])
    assert values['correct_loss']+values['error_loss'] == pytest.approx(nll(p))
    assert values['error_loss'] == pytest.approx(-np.log(.375)/3)


def test_single_correctness_class_has_zero_other_contribution():
    p = np.array([[.1,.8],[.4,.6]])
    assert loss_parts(p,[True,True])['error_loss'] == 0.
    assert loss_parts(p,[False,False])['correct_loss'] == 0.
    with pytest.raises(ValueError): loss_parts(p,[True])


def test_absent_rule_is_not_zero_precision_observation():
    assert active_precision_mean([5.,0.,10.],[1,0,1]) == 7.5
    assert np.isnan(active_precision_mean([0.,0.],[0,0]))
    with pytest.raises(ValueError): active_precision_mean([5.,0.],[1])


def test_precision_sidecar_does_not_double_count_filter_budget(tmp_path):
    stage = tmp_path/'screen'
    chunks = stage/'subject_208'/'POOL'/'chunks'
    chunks.mkdir(parents=True)
    result = dict(subject=208, model='POOL', particles=128, seeds=[4, 8], trials=384,
                  cpu_seconds=12.5, numeric_pools=[dict(num_threads=1)])
    (chunks/'point_C000.json').write_text(json.dumps(result))
    # Sidecars have another schema and refer to the same completed filter job.
    (chunks/'point_C000.experiment.json').write_text('{"feedback_corrections": []}')
    (chunks/'point_C000.precision.json').write_text('{"restores": 17}')
    (chunks/'point_C000.online.json').write_text('{"enabled": true, "reconstructions": 98048}')
    assert training_chunk_budgets(stage) == [dict(subject=208, model='POOL', stage='screen',
        state_updates=98304, filter_runs=2, cpu_seconds=12.5)]
    result['numeric_pools'][0]['num_threads'] = 2
    (chunks/'point_C000.json').write_text(json.dumps(result))
    with pytest.raises(ValueError, match='Nested numeric threads'):
        training_chunk_budgets(stage)


def test_online_reports_fixed_transfer_and_preserves_the_evidence_decay_channel():
    prefix='engine.modules.hypo_transitions_mod.kwargs.'
    frozen=dict(settings=dict(models={'POOL':dict(kind='E',capacity='posterior')},
                              belief_intervention='online_retention_evidence'),
        nominees=[dict(subject=208,model='POOL',candidate=dict(hyperparams={
            prefix+'exit_rate':0.,prefix+'intervention.decay':.8,prefix+'transfer_scale':.01}))])
    rows=refit_inactive_channels(frozen)
    assert rows.parameter.tolist()==[prefix+'transfer_scale']
    assert rows.value.tolist()==[.01]
    assert '固定值仅用于复现' in rows.reason.iloc[0]
    del frozen['settings']['belief_intervention']
    original=refit_inactive_channels(frozen)
    assert original.parameter.tolist()==[prefix+'intervention.decay']
