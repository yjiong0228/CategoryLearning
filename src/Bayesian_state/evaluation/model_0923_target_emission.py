"""Fixed-state effects of the target rule's perception and precision inputs."""
from __future__ import annotations

import numpy as np

from .model_0923_readout_channels import error_channels


def emission_contrasts(present: np.ndarray, contribution: np.ndarray, correct: np.ndarray,
                       original: np.ndarray, clean: np.ndarray, upper: np.ndarray,
                       clean_upper: np.ndarray) -> dict[str, np.ndarray]:
    """Replace only the target emission, retaining all weights and other rules.

The two-factor contributions average each input's improvement over the two
possible orders of changing perception and precision (Shapley allocation).
They can be negative. Their sum plus the remaining target error equals the
original target error. These are instantaneous effects at saved states.
"""
    active=np.asarray(present,bool); c=np.asarray(contribution,float); p=np.asarray(correct,float)
    base=error_channels(present,c,original,p)
    values=[]
    for array in (original,clean,upper,clean_upper):
        q=np.asarray(array,float)
        if q.shape != active.shape or not np.isfinite(q[active]).all() or np.any((q[active]<0)|(q[active]>1)):
            raise ValueError('Target counterfactuals must be probabilities on present trials')
        values.append(np.where(active,q,0.))
    q0,qp,qb,qpb=values
    result=dict(accuracy_original=p,
        accuracy_clean=p+c*(qp-q0),accuracy_upper=p+c*(qb-q0),accuracy_clean_upper=p+c*(qpb-q0),
        perception_share=.5*c*((qp-q0)+(qpb-qb)),
        precision_share=.5*c*((qb-q0)+(qpb-qp)),
        target_error_remaining=c*(1-qpb),target_error_original=base['target_emission_error'])
    np.testing.assert_allclose(result['perception_share']+result['precision_share']+result['target_error_remaining'],
                               result['target_error_original'],rtol=0,atol=2e-12)
    for key in ('accuracy_original','accuracy_clean','accuracy_upper','accuracy_clean_upper'):
        if np.any((result[key]<-1e-12)|(result[key]>1+1e-12)):
            raise ValueError('Target replacement is incompatible with the original mixture')
    return result
