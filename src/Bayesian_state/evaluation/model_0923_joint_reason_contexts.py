"""Compare report-restricted and all-feature backgrounds for potential relevance.

All-feature contexts allow each of the four perceived coordinates to be reflected,
independently of what the participant mentions. Only reported features enter the
within-state minimum. This preserves possible redundancy with an omitted reason.
The existing report-restricted profile remains a distinct diagnostic. Neither
context choice defines a verbal-report likelihood or a human causal effect.
Use --profile COMPLETED_PROFILE --sweep ORIGINAL_SWEEP --output NEW_DIRECTORY.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model_0923_h_validation import digest,load
from .oral.joint_relevance import context_effects,weighted_profile
from ..workflows.analysis.model_0923_h_validation import write


def full_context_effects(probability: np.ndarray,masks: np.ndarray,reported: np.ndarray) -> dict:
    """Use common all-feature contexts, then assess only mentioned coordinates."""
    chosen=np.asarray(reported)
    if chosen.dtype!=np.bool_ or chosen.shape!=(masks.shape[1],) or not chosen.any():
        raise ValueError('Require a nonempty reported-feature set')
    all_effects=context_effects(probability,masks,np.ones(masks.shape[1],bool))
    result={key:all_effects[key][:,chosen] for key in ('actual_effect','positive_context_effect',
                                                     'negative_context_effect','absolute_context_effect')}
    result.update(minimum_positive_effect=result['positive_context_effect'].min(1),
                  minimum_absolute_effect=result['absolute_context_effect'].min(1),
                  minimum_actual_positive_effect=np.maximum(result['actual_effect'],0).min(1),
                  mean_positive_effect=result['positive_context_effect'].mean(1))
    return result


def run(profile: Path,sweep: Path,output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new context-sensitivity directory')
    inputs={}
    def bind(path):
        path=Path(path).resolve();sha=digest(path)
        if str(path) in inputs and inputs[str(path)]!=sha:raise ValueError('Context input changed')
        inputs[str(path)]=sha;return path
    def read(path):return load(bind(path))
    bind(__file__)
    from .oral import joint_relevance
    bind(joint_relevance.__file__)
    done=read(profile/'completion.json');original=read(sweep/'completion.json')
    if (done['status']!='complete' or original['status']!='complete'
            or done['seeds']!=original['seeds'] or done['arm']!=original['arm']
            or done['trials']!=original['trials'] or original['short_diagnostic']):
        raise ValueError('Require matching full original-state diagnostics')
    for path,sha in done['inputs_sha256'].items():
        if digest(bind(path))!=sha:raise ValueError('Original profile source changed')
    for name,sha in done['artifacts_sha256'].items():
        if digest(bind(profile/name))!=sha:raise ValueError('Original profile changed')
    file=bind(sweep/'particle_effects.npz')
    if digest(file)!=original['outputs_sha256']['particle_effects.npz']:
        raise ValueError('Original reflection probes changed')
    with np.load(file) as saved:data={k:saved[k] for k in saved.files}
    with np.load(profile/'particle_profiles.npz') as saved:old={k:saved[k] for k in saved.files}
    rows=pd.read_csv(profile/'report_trials.csv',keep_default_na=False)
    thresholds=np.asarray(done['thresholds']);new_rows=[];curves=[];particle_rows=[]
    for i,t in enumerate(old['state_trial']):
        chosen=((int(old['reported_mask'][i])>>np.arange(4))&1).astype(bool)
        e=full_context_effects(data['joint_reflection_probability'][t],data['reflection_masks'],chosen)
        if np.any(e['minimum_positive_effect']+1e-12<old['minimum_positive_effect'][i]):
            raise ValueError('Enlarging contexts cannot reduce maximum positive effects')
        particle_rows.append(e['minimum_positive_effect'])
        truth=old['truth'][i];available=bool(np.all(truth>=0))
        for weighting,key in [('prior','prior_weights'),('choice_conditioned','conditional_weights')]:
            w=data[key][t];raw=weighted_profile(e['minimum_positive_effect'],w,thresholds)
            gated=weighted_profile(e['minimum_positive_effect'],w,thresholds,truth.astype(bool)) if available else None
            match=rows[(rows.state_trial==t)&(rows.weighting==weighting)]
            if len(match)!=1:raise ValueError('Missing unique report identity')
            record=match.iloc[0].to_dict()
            record.update(context_scope='all_features',minimum_positive_effect_mean=raw['mean'],
                mean_positive_effect_mean=float(w@e['mean_positive_effect']),
                minimum_of_mean_positive_effects=float((w@e['positive_context_effect']).min()),
                minimum_absolute_effect_mean=float(w@e['minimum_absolute_effect']),
                minimum_actual_positive_effect_mean=float(w@e['minimum_actual_positive_effect']),
                truth_and_minimum_positive_effect_mean=gated['mean'] if gated else np.nan)
            new_rows.append(record)
            for kind,item in [('potential_relevance',raw),('numeric_truth_and_relevance',gated)]:
                if item is None:continue
                for threshold,mass in zip(thresholds,item['mass_above']):
                    curves.append(dict(subject=record['subject'],arm=record['arm'],weighting=weighting,phase=record['phase'],
                        state_trial=t,kind=kind,threshold=threshold,mass_above=mass,context_scope='all_features'))
    rows['context_scope']='reported_features';combined=pd.concat([rows,pd.DataFrame(new_rows)],ignore_index=True)
    keys=['subject','arm','weighting','phase','context_scope']
    metrics=[k for k in combined if k.endswith('_mean') or k in ('minimum_of_mean_positive_effects','numeric_report_truth_mass','report_reflection_effect')]
    for key in metrics:combined[key]=pd.to_numeric(combined[key].replace('',np.nan),errors='raise')
    output.mkdir(parents=True)
    combined.to_csv(output/'report_trials.csv',index=False)
    combined.groupby(keys,sort=True)[metrics].mean().reset_index().to_csv(output/'summary.csv',index=False)
    old_curves=pd.read_csv(profile/'threshold_curves.csv');old_curves['context_scope']='reported_features'
    new_curves=pd.DataFrame(curves).groupby(keys+['kind','threshold']).agg(mass_above=('mass_above','mean'),reports=('state_trial','size')).reset_index()
    pd.concat([old_curves,new_curves],ignore_index=True).to_csv(output/'threshold_curves.csv',index=False)
    with (output/'full_context_particle_minima.npz').open('xb') as stream:
        np.savez_compressed(stream,state_trial=old['state_trial'],minimum_positive_effect=np.asarray(particle_rows),seeds=data['seeds'])
    for path,sha in inputs.items():
        if digest(Path(path))!=sha:raise ValueError('Context input changed during evaluation')
    write(output/'completion.json',dict(status='complete',subject=done['subject'],arm=done['arm'],seeds=done['seeds'],
        inputs_sha256=inputs,artifacts_sha256={p.name:digest(p) for p in sorted(output.iterdir())},
        scope='One original filter; common all-feature contexts and report-restricted sensitivity retained',
        interpretation='Primary potential relevance uses common all-feature contexts; report identity determines assessed features only'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','sweep','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.profile,args.sweep,args.output)
