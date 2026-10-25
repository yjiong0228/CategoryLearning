"""Path-level uncertainty and fixed-bin probability calibration.

Whole independent generating sequences are resampled within parameter context.
Trials from one sequence keep their dependence and their original denominator.
"""
from __future__ import annotations

import numpy as np


def bootstrap_counts(contexts: list[str],repeats: int,seed: int) -> np.ndarray:
    if repeats<1 or not contexts:raise ValueError('Require independent paths and positive repeats')
    rng=np.random.default_rng(seed);contexts=np.asarray(contexts);out=np.zeros((repeats,len(contexts)),int)
    for label in np.unique(contexts):
        group=np.flatnonzero(contexts==label);draw=rng.choice(group,size=(repeats,len(group)),replace=True)
        np.add.at(out,(np.repeat(np.arange(repeats),len(group)),draw.ravel()),1)
    return out


def ratio_interval(numerator,denominator,counts,level: float=.95) -> dict:
    numerator=np.asarray(numerator,float);denominator=np.asarray(denominator,float);counts=np.asarray(counts)
    if (numerator.ndim!=1 or denominator.shape!=numerator.shape or counts.ndim!=2
            or counts.shape[1]!=len(numerator) or np.any(denominator<0) or not 0<level<1):raise ValueError('Invalid path ratio')
    den=counts@denominator;num=counts@numerator;valid=den>0
    draws=num[valid]/den[valid];alpha=(1-level)/2
    ci=np.quantile(draws,[alpha,1-alpha]) if len(draws) else [np.nan,np.nan]
    return dict(value=float(numerator.sum()/denominator.sum()) if denominator.sum() else np.nan,
        lower=float(ci[0]),upper=float(ci[1]),bootstrap_valid_fraction=float(valid.mean()))


def calibration(predictions: list[np.ndarray],truths: list[np.ndarray],counts: np.ndarray,edges: list[float],
                minimum_paths: int,minimum_trials: int,tolerance: float) -> list[dict]:
    """Fixed bins, with only boundary roundoff within 1e-10 clipped to [0,1]."""
    if len(predictions)!=len(truths) or len(predictions)!=counts.shape[1]:raise ValueError('Unaligned independent paths')
    edges=np.asarray(edges,float)
    if edges[0]!=0 or edges[-1]!=1 or np.any(np.diff(edges)<=0):raise ValueError('Invalid probability bins')
    rows=[]
    for b in range(len(edges)-1):
        denominators=[];expected=[];observed=[]
        for p,y in zip(predictions,truths):
            p=np.asarray(p,float);y=np.asarray(y,float)
            if (p.shape!=y.shape or np.any((p < -1e-10)|(p>1+1e-10)) or not np.isfinite(p).all()
                    or not np.isin(y,[0,1]).all()):raise ValueError('Invalid calibration data')
            p=np.clip(p,0,1)
            mask=(p>=edges[b])&((p<edges[b+1]) if b<len(edges)-2 else (p<=edges[b+1]))
            denominators.append(int(mask.sum()));expected.append(p[mask].sum());observed.append(y[mask].sum())
        d=np.asarray(denominators);interval=ratio_interval(np.array(observed)-expected,d,counts)
        frequency=ratio_interval(observed,d,counts)
        covered=int(np.count_nonzero(d));sufficient=covered>=minimum_paths and d.sum()>=minimum_trials
        rows.append(dict(bin=b,bin_lower=float(edges[b]),bin_upper=float(edges[b+1]),paths=covered,trials=int(d.sum()),
            mean_prediction=float(np.sum(expected)/d.sum()) if d.sum() else np.nan,
            observed_frequency=float(np.sum(observed)/d.sum()) if d.sum() else np.nan,
            frequency_lower=frequency['lower'],frequency_upper=frequency['upper'],
            difference=interval['value'],lower=interval['lower'],upper=interval['upper'],
            bootstrap_valid_fraction=interval['bootstrap_valid_fraction'],sufficient=sufficient,
            practical_equivalence=bool(sufficient and interval['bootstrap_valid_fraction']>=.95
                and interval['lower']>=-tolerance and interval['upper']<=tolerance)))
    return rows
