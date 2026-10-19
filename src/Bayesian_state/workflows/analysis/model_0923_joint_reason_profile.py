"""Overlay frozen oral feature sets on completed fixed-state reflection probes.

Use --sweep FIXED_DECISION_DIRECTORY --oral FROZEN_ORAL_DIRECTORY --arm ARM
--output NEW_DIRECTORY. All source reports are retained in coverage.csv; complete
coded reports with a mentioned feature set receive a relevance profile. Numeric
truth requires every predicate to have an existing numeric interpretation.
Unreported features remain at the original perceived value. Original prior and
choice-conditioned particle weights are retained; oral never reweights states.
The output describes one original filter's potential feature effects under the
declared contrasts. It is not a report likelihood or a population comparison.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...evaluation.model_0923_h_validation import digest,load
from ...evaluation.oral.constraint_alignment import validate_report_constraints
from ...evaluation.oral.joint_relevance import context_effects,weighted_profile
from .model_0923_h_validation import write

FEATURES=('neck','tail','head','leg')
THRESHOLDS=np.array([0.,1e-10,1e-8,1e-6,1e-4,.001,.005,.01,.025,.05,.1,.2,.4,.6,.8,1.])


def report_content(tokens: list[str], perceived: np.ndarray) -> tuple[np.ndarray,np.ndarray | None]:
    """Retain all mentioned coordinates and evaluate only declared numerical predicates."""
    validate_report_constraints(tokens,n_dims=4)
    mentioned=np.zeros(4,bool);truth=np.ones(len(perceived),bool);symbolic=False
    for token in tokens:
        if token.startswith('facet:'):
            values=np.asarray(json.loads(token[6:]),float);a,b=values[:,:4],values[:,4]
            mentioned|=np.any(a!=0,axis=0)
            truth&=np.all(perceived@a.T<=b+1e-12,axis=1)
        else:
            mentioned[int(token.split(':')[1])]=True;symbolic=True
    return mentioned,None if symbolic else truth


def run(sweep: Path,oral: Path,arm: str,output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new joint-reason profile directory')
    inputs={}
    def bind(path):
        path=Path(path).resolve();sha=digest(path)
        if str(path) in inputs and inputs[str(path)]!=sha:raise ValueError('Joint profile input changed')
        inputs[str(path)]=sha;return path
    def read(path):return load(bind(path))
    bind(__file__)
    from ...evaluation.oral import joint_relevance
    bind(joint_relevance.__file__)
    done=read(sweep/'completion.json');manifest=read(oral/'manifest.json')
    if (done['status']!='complete' or manifest['status']!='complete' or done['short_diagnostic']
            or done['parameters_changed'] or done['oral_used'] or len(done['seeds'])!=1 or done['arm']!=arm):
        raise ValueError('Require one completed original full-sequence fixed-state sweep')
    for path,sha in done['inputs_sha256'].items():
        if digest(bind(path))!=sha:raise ValueError('Original state inputs changed')
    for name,sha in done['outputs_sha256'].items():
        if digest(bind(sweep/name))!=sha:raise ValueError('Fixed-state probes changed')
    if digest(bind(oral/'trials.csv'))!=manifest['artifacts']['trials.csv']:
        raise ValueError('Frozen oral rows changed')
    frame=pd.read_csv(oral/'trials.csv',keep_default_na=False)
    frame=frame[frame.arm==arm].sort_values('state_trial').reset_index(drop=True)
    n=done['trials'];cut=done['training_trials'];particles=done['particles']
    if (len(frame)!=n or frame.state_trial.tolist()!=list(range(n)) or frame.iSub.nunique()!=1
            or int(frame.iSub.iloc[0])!=done['subject'] or not 0<cut<n):
        raise ValueError('Oral rows and sweep identity differ')
    with np.load(sweep/'particle_effects.npz') as z:data={k:z[k] for k in z.files}
    np.testing.assert_array_equal(data['choices'],frame.choice)
    np.testing.assert_array_equal(data['seeds'],done['seeds'])
    np.testing.assert_array_equal(data['reflection_masks'],((np.arange(16)[:,None]>>np.arange(4))&1).astype(bool))
    plan=read(sweep/'protocol.json');job=read(plan['source_job']);target=Path(job['target'])
    record=read(target.with_suffix('.json'))
    if digest(bind(target.with_suffix('.npz')))!=record['array_sha256']:
        raise ValueError('Original particle states changed')
    with np.load(target.with_suffix('.npz')) as z:
        perceived=z['particle_basis_perceived_stimulus'];original=z['particle_basis_prediction']
    expected=np.take_along_axis(original,(data['choices']-1)[:,None,None],axis=2)[...,0]
    np.testing.assert_allclose(data['joint_reflection_probability'][:,:,0],expected,atol=1e-12,rtol=0)
    if perceived.shape!=(n,particles,4):raise ValueError('Wrong particle stimulus shape')
    rows=[];feature_rows=[];curves=[];coverage=[];saved=[]
    for trial,r in frame.iterrows():
        base=dict(subject=int(r.iSub),arm=arm,state_trial=int(trial),iSession=int(r.iSession),iTrial=int(r.iTrial),
                  choice=int(r.choice),phase='training' if trial<cut else 'later',text=r.text,status=r.status,scope=r.scope)
        complete=r.status=='coded' and r.scope=='complete'
        mentioned,truth=report_content(json.loads(r.tokens),perceived[trial]) if complete else (np.zeros(4,bool),None)
        usable=bool(complete and mentioned.any())
        coverage.append(dict(base,profile_available=usable,numeric_report_truth_available=truth is not None,
                             reported_features=int(mentioned.sum())))
        if not usable:continue
        effects=context_effects(data['joint_reflection_probability'][trial],data['reflection_masks'],mentioned)
        report_mask=int(mentioned.astype(int)@(1<<np.arange(4)))
        minimum=effects['minimum_positive_effect'];actual=effects['minimum_actual_positive_effect']
        weights={'prior':data['prior_weights'][trial],'choice_conditioned':data['conditional_weights'][trial]}
        saved.append(dict(state_trial=trial,reported_mask=report_mask,minimum_positive_effect=minimum,
            minimum_absolute_effect=effects['minimum_absolute_effect'],mean_positive_effect=effects['mean_positive_effect'],
            minimum_actual_positive_effect=actual,truth=np.full(particles,-1,dtype=np.int8) if truth is None else truth.astype(np.int8),
            prior_weights=weights['prior'],conditional_weights=weights['choice_conditioned']))
        for weighting,w in weights.items():
            raw=weighted_profile(minimum,w,THRESHOLDS)
            gated=weighted_profile(minimum,w,THRESHOLDS,truth) if truth is not None else None
            row=dict(base,weighting=weighting,reported_features=int(mentioned.sum()),reported_mask=report_mask,
                minimum_positive_effect_mean=raw['mean'],mean_positive_effect_mean=float(w@effects['mean_positive_effect']),
                minimum_of_mean_positive_effects=float((w@effects['positive_context_effect']).min()),
                minimum_absolute_effect_mean=float(w@effects['minimum_absolute_effect']),
                minimum_actual_positive_effect_mean=float(w@actual),
                numeric_report_truth_mass=gated['eligible_mass'] if gated else np.nan,
                truth_and_minimum_positive_effect_mean=gated['mean'] if gated else np.nan,
                report_reflection_effect=float(w@(data['joint_reflection_probability'][trial,:,0]
                                                -data['joint_reflection_probability'][trial,:,report_mask])))
            rows.append(row)
            for index,feature in enumerate(np.flatnonzero(mentioned)):
                feature_rows.append(dict(base,weighting=weighting,feature=FEATURES[feature],
                    **{name:float(w@effects[name][:,index]) for name in
                       ('actual_effect','positive_context_effect','negative_context_effect','absolute_context_effect')}))
            for kind,profile in [('potential_relevance',raw),('numeric_truth_and_relevance',gated)]:
                if profile is None:continue
                for threshold,mass in zip(THRESHOLDS,profile['mass_above']):
                    curves.append(dict(subject=int(r.iSub),arm=arm,weighting=weighting,phase=base['phase'],
                        state_trial=trial,kind=kind,threshold=threshold,mass_above=mass))
    if not saved:raise ValueError('No complete reports with declared feature identities')
    output.mkdir(parents=True)
    trials=pd.DataFrame(rows);trials.to_csv(output/'report_trials.csv',index=False)
    pd.DataFrame(feature_rows).to_csv(output/'reported_feature_effects.csv',index=False)
    pd.DataFrame(coverage).to_csv(output/'coverage.csv',index=False)
    keys=['subject','arm','weighting','phase']
    metrics=[k for k in trials if k.endswith('_mean') or k in ('minimum_of_mean_positive_effects','numeric_report_truth_mass','report_reflection_effect')]
    summary=trials.groupby(keys,sort=True)[metrics].mean().reset_index()
    counts=trials.groupby(keys).agg(reports=('state_trial','size'),numeric_reports=('numeric_report_truth_mass','count')).reset_index()
    summary.merge(counts,on=keys,validate='one_to_one').to_csv(output/'summary.csv',index=False)
    curves=pd.DataFrame(curves)
    curves.groupby(keys+['kind','threshold'],sort=True).agg(mass_above=('mass_above','mean'),reports=('state_trial','size')).reset_index().to_csv(output/'threshold_curves.csv',index=False)
    with (output/'particle_profiles.npz').open('xb') as stream:
        np.savez_compressed(stream,**{key:np.array([row[key] for row in saved]) for key in saved[0]},seeds=data['seeds'])
    for path,sha in inputs.items():
        if digest(Path(path))!=sha:raise ValueError('Joint profile source changed during evaluation')
    write(output/'completion.json',dict(status='complete',subject=done['subject'],arm=arm,seeds=done['seeds'],
        particles=particles,trials=n,training_trials=cut,profiles=len(saved),inputs_sha256=inputs,
        artifacts_sha256={p.name:digest(p) for p in sorted(output.iterdir())},parameters_changed=False,oral_used_for_fitting=False,
        scope='One full original filter; within-state potential relevance under report-restricted reflections',
        semantic_scope='Feature relevance and numeric predicate truth retained separately; symbolic full-report truth remains unavailable',
        thresholds=THRESHOLDS.tolist(),threshold_policy='Display coordinates only; exact weighted means and all particle effects retained',
        limitations=['Reflection can miss effects at symmetric or midpoint stimuli.',
            'Different reported features can attain their maximum in different intervention contexts.',
            'These profiles do not define report likelihood, human causal effects, or across-filter uncertainty.']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('sweep','oral','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--arm',required=True)
    args=parser.parse_args();run(args.sweep,args.oral,args.arm,args.output)
