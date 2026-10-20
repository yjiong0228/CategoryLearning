import numpy as np
import pytest

from src.Bayesian_state.model.region_catalogue_learner import FixedCatalogueRegionLearner
from src.Bayesian_state.model.region_response_learner import RegionResponseLearner


@pytest.mark.parametrize('gamma',[0., .3, .8, 1.])
def test_fixed_labels_match_independent_sequential_power_update(gamma):
    learner=FixedCatalogueRegionLearner(2,rule_retention=gamma,association_learning_rate=None)
    q=np.array([[.1,.3,.4,.2],[.4,.2,.1,.3]]);prior=np.full(2,.5)
    for choice,feedback in [(3,1.),(1,0.),(4,1.),(2,0.)]:
        np.testing.assert_allclose(learner.predict(q),prior@q,atol=1e-12,rtol=0)
        e=q[:,choice-1] if feedback else np.delete(q,choice-1,axis=1).sum(1)
        prior=prior**gamma*e;prior/=prior.sum()
        learner.update(q,choice,feedback)
        np.testing.assert_allclose(learner.rule_weights,prior,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(learner.associations,np.broadcast_to(np.eye(4),(2,4,4)))


def test_single_geometry_matches_original_independent_row_prototype():
    single=RegionResponseLearner(4,4,learning_rate=.5)
    catalogue=FixedCatalogueRegionLearner(1,rule_retention=.3,association_learning_rate=.5)
    for q,choice,feedback in [(np.array([.1,.2,.3,.4]),2,1.),(np.eye(4)[1],3,0.),(np.eye(4)[2],4,1.)]:
        np.testing.assert_allclose(catalogue.predict(q[None]),single.predict(q),atol=1e-12,rtol=0)
        catalogue.update(q[None],choice,feedback);single.update(q,choice,feedback)
        np.testing.assert_allclose(catalogue.associations[0],single.associations,atol=1e-12,rtol=0)


def test_rule_evidence_uses_associations_before_current_feedback_update():
    learner=FixedCatalogueRegionLearner(2,rule_retention=.8,association_learning_rate=2.)
    q=np.array([[.1,.2,.3,.4],[.6,.2,.1,.1]])
    learner.update(q,1,1.)
    old=learner.rule_weights; a=learner.associations
    per_rule=np.einsum('hg,hgc->hc',q,a)
    expected=old**.8*per_rule[:,2];expected/=expected.sum()
    learner.update(q,3,1.)
    np.testing.assert_allclose(learner.rule_weights,expected,atol=1e-12,rtol=0)


def test_zero_rate_keeps_uniform_associations_across_different_geometries():
    learner=FixedCatalogueRegionLearner(2,rule_retention=0.,association_learning_rate=0.)
    q=np.array([[.1,.2,.3,.4],[.6,.2,.1,.1]])
    for choice,feedback in [(1,1.),(3,0.),(2,1.)]:
        np.testing.assert_allclose(learner.predict(q),.25,atol=1e-12,rtol=0)
        learner.update(q,choice,feedback)
    np.testing.assert_allclose(learner.rule_weights,.5,atol=1e-12,rtol=0)
