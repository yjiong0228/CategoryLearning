"""Choice-loss strata conserve total loss and pair complete filter replicates."""
import numpy as np
import pandas as pd
import pytest

from src.Bayesian_state.workflows.analysis.model_0923_choice_loss_oral import loss_strata, COVERAGE


def data():
    reference=np.array([[.2,.3,.4,.5,.6,.7],[.3,.4,.5,.6,.7,.8],[.25,.35,.45,.55,.65,.75]])
    p=reference*np.array([.8,1.1,.9,1.05,.8,1.1])[None,:]
    return p,reference,np.array([0,1,0,1,0,1]),np.array([*COVERAGE,*COVERAGE]),3


def test_disjoint_outcomes_and_coverage_conserve_total_loss():
    table=pd.DataFrame(loss_strata(*data()))
    for phase,part in table.groupby('phase'):
        total=part[part.outcome.eq('all')&part.coverage.eq('all')].iloc[0]
        outcome=part[part.outcome.ne('all')&part.coverage.eq('all')]
        cells=part[part.outcome.ne('all')&part.coverage.ne('all')]
        assert outcome.trials.sum()==cells.trials.sum()==total.trials
        assert outcome.total_delta_nll.sum()==pytest.approx(total.total_delta_nll)
        assert cells.total_delta_nll.sum()==pytest.approx(total.total_delta_nll)
        assert cells.window_mean_contribution.sum()==pytest.approx(total.mean_delta_nll)


def test_common_filter_variation_cancels_for_constant_probability_ratio():
    _,reference,correct,coverage,cut=data()
    rows=pd.DataFrame(loss_strata(reference*.8,reference,correct,coverage,cut))
    present=rows[rows.trials.gt(0)]
    np.testing.assert_allclose(present.mean_delta_nll,-np.log(.8),atol=1e-14,rtol=0)
    np.testing.assert_allclose(present.mean_delta_nll_mcse,0,atol=1e-14,rtol=0)
    absent=rows[rows.trials.eq(0)]
    assert absent.mean_delta_nll.isna().all()
    assert absent.total_delta_nll.eq(0).all() and absent.window_mean_contribution.eq(0).all()


@pytest.mark.parametrize('fault',['shape','probability','outcome','coverage','cut'])
def test_invalid_inputs_are_rejected(fault):
    p,reference,correct,coverage,cut=data()
    if fault=='shape':p=p[:1]
    elif fault=='probability':p[0,0]=np.nan
    elif fault=='outcome':correct[0]=2
    elif fault=='coverage':coverage[0]='changed'
    else:cut=0
    with pytest.raises(ValueError):loss_strata(p,reference,correct,coverage,cut)
