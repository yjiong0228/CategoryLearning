"""An extension must preserve the original discrepancy and its reference."""
import numpy as np

from src.Bayesian_state.metrics.trajectory_shape import shape_check
from src.Bayesian_state.workflows.analysis.model_1024_shape_extension import fixed_reference_check


def test_original_split_reproduces_primary_definition():
    rng=np.random.default_rng(1903);x=rng.normal(size=(256,11));y=rng.normal(size=11)
    assert fixed_reference_check(x,y,128,.05)==shape_check(x,y,.05)


def test_additional_calibration_preserves_reference_and_statistic():
    rng=np.random.default_rng(1903);x=rng.normal(size=(768,11));y=rng.normal(size=11)
    before=fixed_reference_check(x[:256],y,128,.05);after=fixed_reference_check(x,y,128,.05)
    assert before['observed_discrepancy']==after['observed_discrepancy']
    assert before['neighbors']==after['neighbors']==11
    assert after['calibration_paths']==640 and after['reference_paths']==128
