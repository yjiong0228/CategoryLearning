"""Partial content must retain fixed weights and whole-report semantics."""
import numpy as np
import pytest

from src.Bayesian_state.evaluation.oral.constraint_alignment import InconsistentReportError, evaluate_component
from src.Bayesian_state.evaluation.oral.decision_basis_scoring import pool_joint, pooled_mean_and_mcse
from src.Bayesian_state.evaluation.oral.predicate_alignment import fractional_masses, predicate_fractions
from src.Bayesian_state.evaluation.oral.structure_0923 import encode_report
from src.Bayesian_state.workflows.analysis.model_0923_oral_constraint_audit import numerical_constraints


def tokens(text):
    return encode_report(text, ('neck','head','leg','tail')).tokens


def fractions(report, rule):
    return predicate_fractions(tokens(report), *numerical_constraints(tokens(rule)))


@pytest.mark.parametrize('report,rule', [('脖子短','脖子短'),('脖子长','脖子短'),
    ('尾巴长','脖子短'),('头比腿长','头长，腿短'),('头比躯干长','头长')])
def test_single_predicate_reproduces_whole_verdict(report, rule):
    whole = evaluate_component(tokens(report), *numerical_constraints(tokens(rule)))
    actual = fractions(report,rule)
    for d,name in enumerate(('structural','functional')):
        np.testing.assert_array_equal(actual[d], [float(getattr(whole,name)==k) for k in ('support','conflict','undetermined')])


def test_partial_content_does_not_turn_into_a_supported_conjunction():
    report='脖子短，尾巴长'
    a,b=fractions(report,'脖子短'),fractions(report,'尾巴长')
    np.testing.assert_array_equal(a, [[.5,0,.5],[.5,0,.5]])
    mask=np.array([[a,b]])[None]
    score=fractional_masses(np.array([[[.2,.8]]]),mask)
    np.testing.assert_array_equal(score, [[[.5,0,.5],[.5,0,.5]]])
    assert evaluate_component(tokens(report),*numerical_constraints(tokens('脖子短'))).structural=='undetermined'


def test_mixed_support_and_conflict_share_the_report_denominator():
    np.testing.assert_array_equal(fractions('脖子短，尾巴长','脖子短，尾巴短'),[[.5,.5,0],[.5,.5,0]])


def test_symbolic_unknown_keeps_its_share_of_the_denominator():
    np.testing.assert_array_equal(fractions('脖子短，尾巴比躯干长','脖子短'),[[.5,0,.5],[.5,0,.5]])


def test_inconsistent_whole_report_cannot_be_rescued_by_splitting():
    with pytest.raises(InconsistentReportError):
        predicate_fractions(tokens('脖子短')+tokens('脖子长'),*numerical_constraints(tokens('脖子短')))
    with pytest.raises(ValueError,match='unique nonempty'):
        predicate_fractions(tokens('脖子短')*2,*numerical_constraints(tokens('脖子短')))


def test_pooled_joint_mass_preserves_behavior_based_filter_weights():
    yes,no=fractions('脖子短','脖子短'),fractions('脖子短','脖子长')
    mask=np.array([[yes,no]])[None]
    # The first complete filter predicts the observed choice nine times better.
    mass=np.array([[[[.9,0]]],[[[0,.1]]]])
    scored=np.stack([fractional_masses(m,mask) for m in mass])
    np.testing.assert_allclose(pool_joint(scored,np.array([[.9],[.1]])),[[[.9,.1,0],[.9,.1,0]]])


def test_unscored_reports_and_padding_are_distinct_from_unknown_content():
    values=np.full((2,1,2,2,3),np.nan);values[0,0,0]=fractions('尾巴长','脖子短')
    mass=np.array([[[1.,0]],[[0.,1.]]])
    result=fractional_masses(mass,values)
    np.testing.assert_array_equal(result[0],[[0,0,1],[0,0,1]])
    assert np.isnan(result[1]).all()
    mass[0]=[[.9,.1]]
    with pytest.raises(ValueError,match='lacks predicate verdicts'):fractional_masses(mass,values)


def test_reports_are_equally_weighted_after_predicate_averaging():
    value=np.stack([fractions('脖子短','脖子短'),fractions('脖子短，尾巴长','头长')])
    # One supported one-predicate report and one unsupported two-predicate report.
    numerator=np.stack([value,value]);point,error,deleted=pooled_mean_and_mcse(numerator,np.ones((2,2)),np.ones(2,bool))
    np.testing.assert_array_equal(point,[[.5,0,.5],[.5,0,.5]])
    np.testing.assert_array_equal(error,0.);np.testing.assert_array_equal(deleted[0],point)
