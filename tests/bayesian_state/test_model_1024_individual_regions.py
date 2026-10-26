"""Scientific definition checks for the bounded individual-parameter pilot."""
import numpy as np
import pytest
import yaml
from pathlib import Path

from src.Bayesian_state.metrics.trajectory_shape import (
    describe_path, paired_difference, sequence_score, shape_check,
)
from src.Bayesian_state.workflows.analysis.model_1024_individual_regions import (
    canonical, KEYS, classify, validate_parameters,
)

POLICY = yaml.safe_load(Path('configs/shared/model1024_individual_regions.yaml').read_text())


def test_complete_sequence_average_differs_from_trial_pooling():
    p = np.array([[.9, .1], [.1, .9]])
    score = sequence_score(p)
    assert score['nll'] == pytest.approx(-np.log(.09)/2)
    assert score['pooled_trial_nll'] == pytest.approx(-np.log(.5))
    assert paired_difference(p, p)['mcse'] == 0
    assert score['effective_systems'] == pytest.approx(2)


@pytest.mark.parametrize('value', [0, np.nan, 1.1])
def test_invalid_probabilities_fail(value):
    with pytest.raises(ValueError):
        sequence_score(np.array([[.5, value]]))


def test_flat_path_is_not_an_abrupt_learning_path():
    record = describe_path(np.tile([0, 1], 256), POLICY['shape'])
    assert not record['supported_transition']
    assert record['label'] == 'no_clear_increase'


def test_shifted_abrupt_paths_preserve_main_shape_features():
    records = []
    for onset in [128, 256, 384]:
        y = np.r_[np.tile([0, 1], onset//2), np.ones(512-onset)]
        records.append(describe_path(y, POLICY['shape']))
    assert all(r['label'] == 'abrupt_maintained' for r in records)
    # Finite path edges and block phase prevent exact invariance; shape remains
    # close while fitted absolute transition centers move by hundreds of trials.
    assert max(r['onset'] for r in records)-min(r['onset'] for r in records)>250
    assert np.max(np.ptp([r['features'] for r in records], axis=0)) < .07


def test_decreasing_and_censored_paths_remain_in_distribution():
    decline = describe_path(np.r_[np.ones(256), np.tile([0, 1],128)], POLICY['shape'])
    late = describe_path(np.r_[np.tile([0, 1],240), np.ones(32)], POLICY['shape'])
    assert decline['label'] == 'decreasing'
    assert not decline['censored']
    assert decline['followup_trials'] == POLICY['shape']['followup']
    assert late['censored']
    assert late['label'] == 'transition_censored'
    assert len(decline['features']) == len(late['features']) == 11


def test_constant_high_and_low_mixture_cannot_be_called_learning():
    high=describe_path(np.ones(512), POLICY['shape'])
    low=describe_path(np.tile([0, 0, 0, 1],128), POLICY['shape'])
    assert high['label']=='stable_high'
    assert low['label']=='no_clear_increase'


def test_shape_tail_uses_all_calibration_paths():
    x=np.zeros((256, 11))
    compatible=shape_check(x,np.zeros(11))
    incompatible=shape_check(x,np.ones(11))
    assert compatible['status']=='compatible'
    assert compatible['tail_count']==128
    assert incompatible['status']=='incompatible'
    assert incompatible['tail_mc_high']>0  # Zero count does not mean zero probability.


def test_only_inactive_parameter_is_canonicalized():
    point={KEYS['error_gain']:0.,KEYS['failure_decay']:.95,KEYS['capacity']:4,KEYS['gamma']:.6}
    a=canonical(point,1,'test')['hyperparams']
    assert a[KEYS['failure_decay']]==0 and a[KEYS['gamma']]==1
    point[KEYS['error_gain']]=.5
    b=canonical(point,3,'test')['hyperparams']
    assert b[KEYS['failure_decay']]==.95 and b[KEYS['gamma']]==.6


def test_empty_gap_between_modes_is_not_accepted_as_typical():
    reference=np.r_[np.zeros((64,2)),np.ones((64,2))]
    samples=np.r_[reference,reference]
    assert shape_check(samples,np.array([.5,.5]))['status']=='incompatible'


@pytest.mark.parametrize('high_probability, expected',[(.6,'retained'),(.59,'numerically_undetermined')])
def test_particle_count_change_can_invalidate_an_otherwise_stable_point(monkeypatch,high_probability,expected):
    from src.Bayesian_state.workflows.analysis import model_1024_individual_regions as runner
    p=np.full((4,32),.6)
    monkeypatch.setattr(runner,'completed_arrays',lambda jobs,key:p)
    row=dict(candidate={'id':'point'},filter_jobs=[],choice_score=sequence_score(p),
             shape_check={'status':'compatible'},particle_check_score=sequence_score(np.full((2,32),high_probability)))
    result=classify([row],POLICY)[0]
    assert result['joint_status']==expected


def test_failure_decay_open_boundary_matches_runtime():
    from src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search import _number
    for value in POLICY['parameter_values']['failure_decay']:
        _number(value,'failure_decay',high=1,open_high=True)
    with pytest.raises(ValueError,match='failure_decay'):
        validate_parameters({KEYS['failure_decay']:1.,KEYS['capacity']:4},1)
