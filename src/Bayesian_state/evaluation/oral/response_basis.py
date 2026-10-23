"""Joint response/report observation bridge for category-power experiments.

A state first generates category c with transformed probability q(c). Given c,
its report basis uses the original relative rule support w(h) p(c|h) / p(c).
Thus P(c,h|state) = q(c) w(h) p(c|h) / p(c). This preserves the conditional
basis definition while allowing the response channel to have its own precision.
It is an explicit measurement assumption; the report is a noisy external test.
"""
from __future__ import annotations

import numpy as np

from .decision_basis import DecisionBasisCollector
from ...model.modules.readout.category_power import category_power


class ResponseBasisCollector(DecisionBasisCollector):
    """Reuse component geometry and accumulation with calibrated joint masses.

    Effective factors passed to the parent are q(c) p(c|h) / p(c). Their rows
    need not sum to one; multiplying by w(h) gives a normalized joint over h,c.
    Raw rule predictions and weights in the incoming snapshot remain intact.
    """
    def __init__(self,*args,exponent: float,**kwargs):
        super().__init__(*args,**kwargs)
        if not np.isfinite(exponent) or exponent<=0:
            raise ValueError('Require a positive finite response exponent')
        self.exponent=float(exponent)

    def __call__(self,trial: int,phase: str,snapshot) -> None:
        if phase=='filtered' or self.exponent==1.:
            return super().__call__(trial,phase,snapshot)
        if phase!='predictive':raise ValueError('Unexpected observation phase')
        weight=np.asarray(snapshot['choice_weights'])
        likelihood=np.asarray(snapshot['rule_choice_probabilities'])
        raw=np.einsum('ph,phc->pc',weight,likelihood)
        q=category_power(raw,self.exponent)
        np.testing.assert_allclose(snapshot['prediction'],q,atol=1e-12,rtol=0)
        ratio=np.divide(q,raw,out=np.zeros_like(q),where=raw>0)
        effective=likelihood*ratio[:,None,:]
        joint=weight[:,:,None]*effective
        np.testing.assert_allclose(joint.sum((1,2)),1.,atol=1e-12,rtol=0)
        bridge=dict(snapshot,rule_choice_probabilities=effective)
        super().__call__(trial,phase,bridge)
