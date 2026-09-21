"""Validate descriptive shape fitting and sustained-support episode definitions."""
import numpy as np
import pandas as pd
import pytest

from CategoryLearning_codes.figures.fig3.abstract_story_analysis import (
    behavior_shapes, select_examples, support_reversals,
)


def test_constant_behavior_is_not_forced_into_a_learning_group():
    y=np.tile([0,1],128)
    summary,curves=behavior_shapes(y)
    assert summary['preferred']=='constant'
    np.testing.assert_allclose(curves['constant'],.5)


def test_single_upward_step_recovers_boundary_without_smoothing():
    y=np.r_[np.zeros(128),np.ones(128)]
    summary,curves=behavior_shapes(y)
    assert summary['preferred']=='step'
    assert summary['split']==128
    assert summary['delta_bic']>6
    assert np.all(curves['step'][:128]<.001)
    assert np.all(curves['step'][128:]>.999)


def test_gradual_behavior_prefers_trend_over_single_step():
    rng=np.random.default_rng(2826)
    blocks=[]
    for successes in range(10,49,2):
        block=np.r_[np.ones(successes),np.zeros(64-successes)]
        rng.shuffle(block);blocks.append(block)
    summary,_=behavior_shapes(np.concatenate(blocks))
    assert summary['preferred']=='trend'
    assert summary['delta_bic']<-6


def test_block_deletion_preserves_original_time_coordinates():
    y=np.r_[np.zeros(128),np.ones(128)]
    mask=np.ones(256,bool);mask[32:64]=False
    summary,curves=behavior_shapes(y,keep=mask)
    assert summary['split']==128
    assert len(curves['step'])==256
    assert curves['step'][127]<.001 and curves['step'][128]>.999


def test_invalid_behavior_and_insufficient_observations_fail():
    with pytest.raises(ValueError,match='binary'):
        behavior_shapes(np.r_[np.zeros(127),2])
    with pytest.raises(ValueError,match='observed segments'):
        behavior_shapes(np.zeros(256),keep=np.arange(256)<100)


def test_reversals_require_sustained_support_and_rebuilding():
    q=np.r_[np.ones(16)*.8,np.zeros(40),np.ones(15)*.8,np.zeros(8),
            np.ones(16)*.8,np.zeros(8)]
    assert support_reversals(q)==[(1,17),(80,96)]
    # Brief high/low excursions are not sustained state changes.
    assert support_reversals(np.r_[np.ones(15),np.zeros(100)])==[]
    assert support_reversals(np.r_[np.ones(16),np.zeros(7),np.ones(10)])==[]


def test_examples_follow_behavior_and_not_the_belief_column():
    summary=pd.DataFrame({'subject':[1,2,3,4],'criterion':[100,600,800,900],
                          'delta_bic':[0,-20,12,0],
                          'preferred':['constant','trend','step','constant'],
                          'belief':[.1,.9,0,1]})
    assert select_examples(summary)=={'rapid':1,'gradual':2,'abrupt':3}
    changed=summary.assign(belief=summary.belief[::-1].to_numpy())
    assert select_examples(changed)==select_examples(summary)


def test_cohort_expansion_reselects_later_abrupt_case():
    summary=pd.DataFrame({'subject':[122,206,307,104,215,328],
                          'criterion':[139,1378,685,146,732,657],
                          'delta_bic':[1.356,-12.385,13.872,-4.924,22.142,16.729],
                          'preferred':['step','trend','step','trend','step','step']})
    assert select_examples(summary)=={'rapid':122,'gradual':206,'abrupt':215}
