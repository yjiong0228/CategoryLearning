"""Numerical extensions preserve the target set and full likelihood pooling."""
import numpy as np

from src.Bayesian_state.metrics.trajectory_shape import sequence_score
from src.Bayesian_state.workflows.analysis import model_1024_particle_precision as extension


def test_target_rule_does_not_select_favorable_shapes():
    rows={110:[dict(candidate={'id':'a'},particle_check_score={'nll':1},
                   particle_check={'status':'boundary'},shape_check={'status':'incompatible'}),
               dict(candidate={'id':'b'},particle_check_score={'nll':1},
                    particle_check={'status':'stable'},shape_check={'status':'compatible'})]}
    assert extension.select_targets(rows)==[dict(subject=110,candidate='a')]


def test_extension_pools_system_likelihoods_and_preserves_primary(monkeypatch):
    probability=np.array([[.2,.7],[.4,.8],[.9,.9],[.3,.6],[.5,.5],[.6,.7]])
    original=sequence_score(probability[:2]);row=dict(particle_check_jobs=[0,1],particle_check_score=original)
    monkeypatch.setattr(extension,'completed_arrays',lambda jobs,key: probability[jobs])
    assert extension.augment_high_checks(row,[2,3,4,5])
    assert row['primary_particle_check_score']==original
    assert row['particle_check_score']==sequence_score(probability)
    assert row['particle_check_jobs']==list(range(6))
