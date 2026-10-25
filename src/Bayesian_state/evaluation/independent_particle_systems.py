"""Evidence weights and exact conditional references for particle-system banks.

All systems in a bank have the same cognitive parameters. These functions
operate on numerical estimates; they do not define a second cognitive model.
"""
from __future__ import annotations

import numpy as np

from .particle_smoothing import probabilities


def evidence_weights(log_evidence: np.ndarray) -> np.ndarray:
    """Normalize independent-system evidence estimates along the first axis."""
    value=np.asarray(log_evidence,float)
    if value.ndim not in (1,2) or len(value)<1 or not np.isfinite(value).all():
        raise ValueError('Require finite system or system-by-endpoint log evidence')
    mass=np.exp(value-value.max(axis=0,keepdims=True))
    return probabilities(mass/mass.sum(axis=0,keepdims=True),axis=0)


def bank_diagnostics(weights: np.ndarray) -> dict:
    weights=probabilities(weights,axis=0)
    return dict(ess=1./np.square(weights).sum(axis=0),maximum=weights.max(axis=0))


def pool_curves(curves: np.ndarray, cumulative_log_evidence: np.ndarray,
                endpoints: np.ndarray) -> np.ndarray:
    """Use the observation endpoint of each row, including filter/lag rows.

    curves is system by trial by optional category. endpoints uses one-based
    observation counts. Each cumulative evidence row covers the full sequence.
    """
    curves=np.asarray(curves,float);logz=np.asarray(cumulative_log_evidence,float)
    endpoints=np.asarray(endpoints)
    if (curves.ndim not in (2,3) or logz.ndim!=2 or curves.shape[:2]!=(len(logz),len(endpoints))
            or not np.issubdtype(endpoints.dtype,np.integer) or np.any(endpoints<1)
            or np.any(endpoints>logz.shape[1]) or not np.isfinite(curves).all()):
        raise ValueError('Require aligned curves and valid observation endpoints')
    weights=evidence_weights(logz[:,endpoints-1])
    return np.sum(curves*weights.reshape(weights.shape+(1,)*(curves.ndim-2)),axis=0)


def pimh_indices(log_evidence: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Accept fresh independent PF proposals, retaining evidence on rejection.

    Every element is a fresh particle system, not a resample from a finite bank.
    The first proposal initializes this finite chain; no stationarity is claimed.
    """
    logz=np.asarray(log_evidence,float);u=np.asarray(uniforms,float)
    if logz.ndim!=1 or len(logz)<1 or u.shape!=(len(logz)-1,) or not np.isfinite(logz).all() or np.any((u<=0)|(u>=1)):
        raise ValueError('Require fresh proposal evidence and open-unit uniforms')
    out=np.zeros(len(logz),int);current=0
    for j in range(1,len(logz)):
        if np.log(u[j-1])<min(0.,logz[j]-logz[current]):current=j
        out[j]=current
    return out


def execution_kernel(previous_ids: np.ndarray,current_ids: np.ndarray,belief: np.ndarray,kappa: float) -> np.ndarray:
    """Exact shared persistent-execution law conditional on the other states."""
    previous_ids=np.asarray(previous_ids);current_ids=np.asarray(current_ids);belief=probabilities(belief)
    if (previous_ids.ndim!=1 or current_ids.ndim!=1 or belief.shape!=current_ids.shape
            or len(set(current_ids.tolist()))!=len(current_ids) or not np.isfinite(kappa) or kappa<0):
        raise ValueError('Require distinct active rules, normalized belief and nonnegative inertia')
    mass=np.broadcast_to(belief,(len(previous_ids),len(belief))).copy()
    mass*=np.exp(kappa*(previous_ids[:,None]==current_ids[None,:]))
    return probabilities(mass/mass.sum(axis=1,keepdims=True))


def conditional_execution(ids: list[np.ndarray],beliefs: list[np.ndarray],likelihoods: list[np.ndarray],kappa: float) -> dict:
    """Variable-workspace forward/backward, given the complete other-state path.

    The additional conditioning is essential: this reference is not the full
    behavior-only posterior over workspace, belief, precision and perception.
    """
    n=len(ids)
    if n<1 or len(beliefs)!=n or len(likelihoods)!=n:raise ValueError('Require aligned nonempty state histories')
    kernels=[execution_kernel(ids[t-1],ids[t],beliefs[t],kappa) for t in range(1,n)]
    filtered=[];scales=[];predictive=[]
    for t in range(n):
        prior=probabilities(beliefs[t]) if t==0 else filtered[-1]@kernels[t-1]
        ell=np.asarray(likelihoods[t],float)
        if ell.shape!=prior.shape or np.any((ell<0)|(ell>1)) or not np.isfinite(ell).all():raise ValueError('Invalid execution emission')
        mass=prior*ell;scale=mass.sum()
        if scale<=0:raise ValueError('Impossible conditional choice sequence')
        predictive.append(prior);filtered.append(mass/scale);scales.append(scale)
    smoothed=[None]*n;backward=np.ones_like(filtered[-1]);smoothed[-1]=filtered[-1].copy()
    for t in range(n-2,-1,-1):
        backward=kernels[t]@(np.asarray(likelihoods[t+1])*backward)/scales[t+1]
        mass=filtered[t]*backward;smoothed[t]=probabilities(mass/mass.sum())
    return dict(predictive=predictive,filtered=filtered,smoothed=smoothed,kernels=kernels,
                log_evidence=float(np.log(scales).sum()))
