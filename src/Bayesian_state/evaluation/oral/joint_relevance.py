"""Measure joint potential feature effects inside each fixed decision state.

Every probe replaces a declared subset of perceived coordinates x with 1-x.
Only reported coordinates may vary when assessing a report. For each coordinate,
retain its actual-context effect and its largest positive, negative and absolute
change across the other reported coordinates' reflection contexts. A positive
change means keeping that coordinate's actual value raises choice probability.
Take the minimum over reported coordinates inside each particle before averaging.
This retains simultaneous relevance and allows redundant reasons to have an effect
in different contexts. It measures potential effect under these probes, not a
report probability, human causal attribution, or complete semantic agreement.
"""
from __future__ import annotations

import numpy as np


def context_effects(probability: np.ndarray, masks: np.ndarray, reported: np.ndarray) -> dict[str, np.ndarray]:
    """Return particle × reported-feature effects and particle-level minima."""
    q=np.asarray(probability,float);mask=np.asarray(masks);chosen=np.asarray(reported)
    if (q.ndim!=2 or mask.ndim!=2 or not np.issubdtype(mask.dtype,np.bool_)
            or chosen.dtype!=np.bool_ or chosen.shape!=(mask.shape[1],)
            or not chosen.any() or q.shape[1]!=len(mask) or not len(q)
            or not 1<=mask.shape[1]<=10 or len(mask)!=2**mask.shape[1]
            or not np.isfinite(q).all() or np.any((q<0)|(q>1+1e-12))):
        raise ValueError('Require a complete binary probe bank and nonempty reported-feature set')
    keys=mask.astype(int)@(1<<np.arange(mask.shape[1]))
    if len(set(keys.tolist()))!=len(mask):
        raise ValueError('Reflection probes must cover each mask exactly once')
    columns={int(key):i for i,key in enumerate(keys)}
    report_key=int(chosen.astype(int)@(1<<np.arange(len(chosen))))
    actual,positive,negative,absolute=[],[],[],[]
    for feature in np.flatnonzero(chosen):
        bit=1<<int(feature)
        contexts=[key for key in columns if key&~report_key==0 and key&bit==0]
        differences=np.stack([q[:,columns[key]]-q[:,columns[key|bit]] for key in contexts],axis=1)
        actual.append(q[:,columns[0]]-q[:,columns[bit]])
        positive.append(np.maximum(differences.max(1),0.))
        negative.append(np.maximum(-differences.min(1),0.))
        absolute.append(np.abs(differences).max(1))
    result={name:np.stack(values,axis=1) for name,values in
            [('actual_effect',actual),('positive_context_effect',positive),
             ('negative_context_effect',negative),('absolute_context_effect',absolute)]}
    result['minimum_positive_effect']=result['positive_context_effect'].min(1)
    result['minimum_absolute_effect']=result['absolute_context_effect'].min(1)
    result['minimum_actual_positive_effect']=np.maximum(result['actual_effect'],0.).min(1)
    result['mean_positive_effect']=result['positive_context_effect'].mean(1)
    return result


def weighted_profile(values: np.ndarray, weights: np.ndarray, thresholds: np.ndarray,
                     eligible: np.ndarray | None=None) -> dict:
    """Return exact weighted mean and mass above each declared display threshold.

    The weighted mean equals the integral over all thresholds of the survival
    curve. Eligibility retains the original weights and sets excluded values to
    zero; it does not condition again on the oral report. Thresholds are plotting
    coordinates, not an acceptance rule. Individual values retain exact curves.
    """
    x,w,t=(np.asarray(v,float) for v in (values,weights,thresholds))
    if (x.ndim!=1 or not len(x) or w.shape!=x.shape or t.ndim!=1 or not len(t)
            or not all(np.isfinite(v).all() for v in (x,w,t))
            or np.any((x<0)|(x>1+1e-12)) or np.any(w<0)
            or np.any((t<0)|(t>1)) or np.any(np.diff(t)<=0)):
        raise ValueError('Require bounded effects, normalized weights and increasing display thresholds')
    np.testing.assert_allclose(w.sum(),1.,atol=1e-12,rtol=0)
    use=np.ones(len(x),bool) if eligible is None else np.asarray(eligible)
    if use.dtype!=np.bool_ or use.shape!=x.shape:
        raise ValueError('Eligibility must identify these same particles')
    values=np.where(use,x,0.)
    return dict(mean=float(w@values),eligible_mass=float(w@use),
                mass_above=np.array([float(w@(values>threshold)) for threshold in t]))
