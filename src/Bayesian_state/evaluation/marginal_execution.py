"""Exact local execution conditioning for the zero-inertia belief candidate.

The shared MIX readout supplies the marginalized choice likelihood. Here its
rule contributions are treated as the prior over one latent executed rule.
These helpers do not change cognitive state updates or autonomous simulation.
"""
from __future__ import annotations

import numpy as np

from .particle_smoothing import probabilities, ancestral_weights


def execution_posterior(contributions: np.ndarray, likelihood: np.ndarray,
                        choice: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-particle choice mass and P(execution | state, current choice)."""
    prior=probabilities(contributions); likelihood=np.asarray(likelihood,float)
    if (prior.ndim!=2 or likelihood.ndim!=3 or likelihood.shape[:2]!=prior.shape
            or not isinstance(choice,(int,np.integer)) or not 1<=choice<=likelihood.shape[2]
            or not np.isfinite(likelihood).all() or np.any((likelihood<0)|(likelihood>1))):
        raise ValueError('Require aligned particle/rule/category arrays and a one-based choice')
    # Inactive rows of the existing observer are zero; only prior-supported
    # rule likelihoods need to be normalized categorical distributions.
    if not np.allclose(likelihood.sum(axis=2)[prior>0],1.,atol=1e-12,rtol=0):
        raise ValueError('Supported rule likelihoods must sum to one')
    joint=prior*likelihood[:,:,choice-1];mass=joint.sum(axis=1)
    if np.any(mass<=0):
        raise ValueError('Cannot condition on an impossible choice')
    conditional=joint/mass[:,None];probabilities(conditional)
    return mass,conditional


def marginal_means(values: dict, weights: np.ndarray) -> dict:
    """Aggregate latent-state events and conditional execution probabilities."""
    result={k:np.einsum('tp,tp->t',weights,values[k])
            for k in ('present','high_belief','target_belief','target_execution')}
    result['source']=np.stack([np.sum(weights*values['source_'+str(k)],axis=1) for k in range(3)],axis=1)
    probabilities(result['source'])
    return result


def marginal_smoothing(values: dict, parents: np.ndarray, weights: np.ndarray,
                       tracer, prefix: int, lag: int) -> dict:
    """Marginalize execution along coherent continuous-state ancestry paths."""
    n,particles=weights.shape
    if not 1<=prefix<=n or lag<0 or parents.shape!=(n,particles):
        raise ValueError('Invalid smoothing dimensions or endpoints')
    result={'filtered_'+k:v for k,v in marginal_means(values,weights).items()}
    for mode,end in [('prefix',prefix),('full',n)]:
        paths=tracer(parents[:end]);ancestry=ancestral_weights(paths,weights[end-1])
        result.update({mode+'_'+k:v for k,v in marginal_means(
            {k:v[:end] for k,v in values.items()},ancestry['weights']).items()})
        result[mode+'_indices']=paths.astype(np.int32)
        result[mode+'_unique_ancestors']=ancestry['unique_ancestors']
        result[mode+'_ancestral_ess']=ancestry['ancestral_ess']
    lag_rows=[];counts=[];ess=[]
    for t in range(n):
        end=min(n,t+lag+1);ancestor=np.arange(particles)
        for j in range(end-2,t-1,-1):ancestor=parents[j,ancestor]
        mass=np.bincount(ancestor,weights=weights[end-1],minlength=particles)
        row=marginal_means({k:v[t:t+1] for k,v in values.items()},mass[None,:])
        lag_rows.append(row);counts.append(np.count_nonzero(mass>0));ess.append(1./np.square(mass).sum())
    result.update({'lag_'+k:np.concatenate([r[k] for r in lag_rows]) for k in lag_rows[0]})
    result.update(lag_unique_ancestors=np.array(counts),lag_ancestral_ess=np.array(ess))
    return result


class MarginalExecutionCollector:
    """Passively retain target states and exact local latent execution masses."""
    def __init__(self, choices: np.ndarray, targets: list[int]):
        self.choices=np.asarray(choices);self.targets=np.asarray(targets)
        if (self.choices.ndim!=1 or not np.issubdtype(self.choices.dtype,np.integer)
                or np.any(self.choices<1) or self.targets.ndim!=1 or not len(self.targets)
                or not np.issubdtype(self.targets.dtype,np.integer) or np.any(self.targets<0)
                or len(np.unique(self.targets))!=len(self.targets)):
            raise ValueError('Require integer choices and a unique integer target set')
        self.values=[];self.weights=[];self.predictive=[];self.minimum_choice_mass=1.

    def __call__(self, trial: int, phase: str, snapshot) -> None:
        if phase=='predictive':
            if trial!=len(self.values) or len(self.weights)!=trial:
                raise ValueError('Require alternating ordered snapshots')
            active=np.asarray(snapshot['active']);belief=np.asarray(snapshot['belief'])
            contribution=probabilities(snapshot['choice_weights'])
            if (active.ndim!=2 or belief.shape!=active.shape or contribution.shape!=active.shape
                    or self.targets.max()>=active.shape[1] or not np.isin(active,[0,1]).all()):
                raise ValueError('Invalid active rule or belief dimensions')
            probabilities(belief)
            if np.any(belief[active==0]) or np.any(contribution[active==0]):
                raise ValueError('Inactive rule has positive mass')
            np.testing.assert_allclose(contribution,belief,atol=2e-14,rtol=0)
            mass,q=execution_posterior(contribution,np.asarray(snapshot['rule_choice_probabilities']),
                int(self.choices[trial]))
            np.testing.assert_allclose(mass,np.asarray(snapshot['prediction'])[:,self.choices[trial]-1],atol=2e-14,rtol=0)
            present=active[:,self.targets].any(axis=1);target_belief=belief[:,self.targets].sum(axis=1)
            execution=q[:,self.targets].sum(axis=1)
            self.minimum_choice_mass=min(self.minimum_choice_mass,float(mass.min()))
            self.values.append(dict(present=present.copy(),high_belief=present & (target_belief>=.5),
                target_belief=target_belief.copy(),target_execution=execution.copy(),
                source_0=(~present).astype(float),source_1=present*(1.-execution),source_2=execution.copy()))
            self.predictive.append(float(np.asarray(snapshot['weights'])@contribution[:,self.targets].sum(axis=1)))
        elif phase=='filtered':
            if trial!=len(self.weights) or len(self.values)!=trial+1:
                raise ValueError('Filtered snapshot is out of order')
            self.weights.append(probabilities(snapshot['weights']).copy())
        else:raise ValueError('Unknown snapshot phase')

    def arrays(self) -> dict:
        if not self.values or len(self.values)!=len(self.choices) or len(self.weights)!=len(self.values):
            raise ValueError('Incomplete marginalized execution capture')
        return dict({k:np.stack([r[k] for r in self.values]) for k in self.values[0]},
            filtered_weights=np.stack(self.weights),predictive_target_execution=np.array(self.predictive))
