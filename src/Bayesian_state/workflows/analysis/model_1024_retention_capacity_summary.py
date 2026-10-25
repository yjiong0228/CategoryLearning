"""Audit every frozen retention/capacity point and paired factorial contrast.

Use --evaluation COMPLETE --output NEW. Generative equations and path summaries
are shared with the existing execution studies. All profiles, unselected points,
functional events, timings and failed training proposals remain in the tables.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from . import model_1024_retention_capacity as profile
from . import model_1024_generation_selection as shared
from .model_1024_trajectory_diagnosis import checked_new_paths,summarize_paths,probability_record
from .model_1022_belief_summary import jackknife_se
from ...metrics.autonomous_fit import behavior_features,energy_u
from ...metrics.task import category_learning_feedback

ROOT=profile.ROOT
read,write,digest,nll=profile.read,profile.write,profile.digest,profile.nll


def linear_contrast(values: list[np.ndarray], coefficients: list[float]) -> tuple[float,float]:
    """Pair every complete path across the fixed parameter interventions."""
    if not values or len(values)!=len(coefficients) or any(np.asarray(v).shape!=np.asarray(values[0]).shape for v in values):
        raise ValueError('Require aligned path statistics and coefficients')
    combined=sum(c*np.asarray(v,float) for c,v in zip(coefficients,values))
    if combined.ndim!=1 or len(combined)<2 or not np.isfinite(combined).all():
        raise ValueError('Need finite scalar path statistics')
    return float(combined.mean()),float(combined.std(ddof=1)/np.sqrt(len(combined)))


def energy_contrast(features: list[np.ndarray], coefficients: list[float], truth: np.ndarray) -> tuple[float,float]:
    """Delete the same independent path seed from all scored distributions."""
    if len(features)!=len(coefficients) or any(x.shape!=features[0].shape for x in features):
        raise ValueError('Require aligned feature banks and coefficients')
    value=sum(c*energy_u(x,truth)['value'] for c,x in zip(coefficients,features))
    deleted=np.array([sum(c*energy_u(np.delete(x,i,0),truth)['value'] for c,x in zip(coefficients,features))
                      for i in range(len(features[0]))])
    return float(value),jackknife_se(deleted)


def audit_jobs(folder: Path) -> tuple[list[dict],dict]:
    rows=[];hashes={}
    for file in sorted((folder/'jobs').glob('*.json')):
        job=read(file);stem=Path(job['target']);metadata=read(stem.with_suffix('.json'))
        boundary=read(stem.with_suffix('.boundary.json'));mechanism=read(stem.with_suffix('.mechanism.json'))
        expected=metadata['trials']*(metadata['paths'] if 'paths' in metadata else metadata['particles']*len(metadata['seeds']))
        audit=mechanism['audit'];execution=audit['error_entry']['execution']
        if (audit['transitions']!=expected or audit['feedback_updates']!=expected or
                audit['outcome_retention']['updates']!=expected or mechanism['oral_used_for_fitting'] or
                execution['trials']!=(0 if job['execution_mode']=='MIX' else expected) or
                execution['switches']!=execution['forced_switches']+execution['voluntary_switches']):
            raise ValueError('Unexpected belief, learning or execution mechanism audit')
        if any(p.get('num_threads',1)!=1 for p in metadata['numeric_pools']):raise ValueError('Multiple numerical threads')
        train=bool(job.get('training_generation') or job['phase']=='training')
        if boundary['training_only']!=train or boundary['sealed_test_denied']!=train:
            raise ValueError('Training boundary was not enforced')
        if metadata['candidate']['hyperparams']!=job['candidate']['hyperparams']:raise ValueError('Parameter identity changed')
        if digest(stem.with_suffix('.npz'))!=metadata['predictions_sha256']:raise ValueError('Prediction bank changed')
        for suffix in ('.npz','.json','.boundary.json','.mechanism.json'):
            f=stem.with_suffix(suffix);hashes[str(f)]=digest(f)
        rows.append(dict(name=job['name'],training_only=train,phase=job['phase'],mode=job['execution_mode'],
            transitions=expected,cpu_seconds=metadata['cpu_seconds'],seconds=metadata['seconds']))
    return rows,hashes


def timed_events(metrics: dict, truth: dict, events: dict, tolerance: int) -> dict:
    """Attach the prespecified trial-scale timing condition to whole-path events."""
    if tolerance<0:raise ValueError('Timing tolerance must be nonnegative')
    gap=np.abs(metrics['complete_center']-truth['complete_center'])
    return dict(events,jump_maintenance_and_timing=events['jump_and_maintenance'] & (gap<=tolerance),
        jump_maintenance_timing_and_late=events['jump_maintenance_and_late'] & (gap<=tolerance))


def run(evaluation: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new profile summary directory')
    done=read(evaluation/'completion.json');protocol=read(evaluation/'protocol.json')
    training=Path(protocol['training']);frozen=read(training/'nominees.json');policy=protocol['policy']
    if (done['status']!='complete' or done['protocol_sha256']!=digest(evaluation/'protocol.json') or
            frozen['smoke'] or frozen['later_choices_used'] or frozen['profile_sources']!=profile.sources() or
            protocol['frozen_nominees_sha256']!=digest(training/'nominees.json')):
        raise ValueError('Require the complete formal profile evaluation and freeze')
    output.mkdir(parents=True)
    diagnostic=yaml.safe_load((ROOT/'configs/shared/model1024_trajectory_diagnosis.yaml').read_text())
    rows=[];paths=[];events=[];curves=[];observed=[];comparisons=[];factorial=[];input_hashes={};audits=[]
    for folder in (training,evaluation):
        a,h=audit_jobs(folder);audits.extend(a);input_hashes.update(h)
    source_scores=pd.read_csv(training/'screen_scores.csv')
    for nominee in frozen['nominees']:
        sid,mode=nominee['subject'],nominee['mode']
        cases=[c for c in protocol['cases'] if (c['subject'],c['mode'])==(sid,mode)]
        folder=Path(cases[0]['folder']);meta=read(folder/'prepared.json');parts=[]
        for name in ('training.npz','sealed_test.npz'):
            if digest(folder/name)!=meta['array_sha256'][name]:raise ValueError('Data changed since preparation')
            with np.load(folder/name) as z:parts.append({k:z[k] for k in z.files})
        obs={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};cut=meta['training_trials']
        ncat=2 if meta['condition']==1 else 4
        geometry=ROOT/policy['geometry']/f'subject_{sid}'/'audit.json';targets=read(geometry)['training_perfect_rules']
        input_hashes[str(geometry)]=digest(geometry)
        sections=[('full',0,None),('training',0,cut),('late',cut,None)]
        empirical={label:behavior_features(obs['choices'][None,start:stop],obs['categories'][start:stop],
            n_categories=ncat,**policy['features'])[0] for label,start,stop in sections}
        bank={};grid={}
        for case in cases:
            point=case['candidate'];tag=dict(subject=sid,mode=mode,candidate=point['id'],
                delta=point['hyperparams'][profile.DELTA],capacity=point['hyperparams'][profile.CAPACITY],
                is_anchor=case['is_anchor'],is_selected=case['is_selected'],is_nll_reference=case['is_nll_reference'],in_profile=case['in_profile'])
            auto=checked_new_paths(case['autonomous_jobs'],case,obs['categories'])
            if len(auto['choices'])!=policy['evaluation_paths']:raise ValueError('Incomplete full-path budget')
            if (not np.isfinite(auto['precision']).all() or np.any(auto['precision']<0)
                    or np.any(auto['precision']>25.+1e-12) or np.any(auto['precision'][auto['rule']<0])):
                raise ValueError('Invalid generated precision or padded-state encoding')
            expected=np.array([category_learning_feedback(int(c),int(y),condition=meta['condition'])
                for c,y in zip(auto['choices'].ravel(),np.tile(obs['categories'],len(auto['choices'])))]).reshape(auto['choices'].shape)
            np.testing.assert_array_equal(auto['feedback'],expected)
            ordered=np.sort(auto['rule'],axis=-1)
            if np.any((ordered[...,1:]==ordered[...,:-1]) & (ordered[...,1:]>=0)):raise ValueError('Duplicate active rule')
            metrics,truth,event=summarize_paths(auto,obs,cut,ncat,targets,mode,diagnostic)
            event=timed_events(metrics,truth,event,policy['jump_timing_tolerance'])
            metrics['jump_timing_absolute_error']=np.abs(metrics['complete_center']-truth['complete_center'])
            features={label:behavior_features(auto['choices'][:,start:stop],obs['categories'][start:stop],
                n_categories=ncat,**policy['features']) for label,start,stop in sections}
            row=dict(tag,condition=meta['condition'],paths=len(auto['choices']),observed_late_accuracy=truth['late_accuracy'],
                **{k:float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for k,v in metrics.items()})
            for label in features:
                score=energy_u(features[label],empirical[label]);row[label+'_energy']=score['value'];row[label+'_energy_mcse']=score['mc_se']
            for key,value in event.items():
                record=probability_record(value);row[key+'_count']=record['count'];row[key+'_probability']=record['probability']
                events.append(dict(tag,event=key,**record))
            pred=None
            if case['filter_jobs']:
                pred=shared.filter_arrays(case['filter_jobs']);p=pred['probabilities']
                if p.shape!=(policy['evaluate']['seeds'],meta['trials'],ncat):raise ValueError('Wrong conditional probability shape')
                np.testing.assert_allclose(p.sum(-1),1.,atol=1e-12,rtol=0)
                np.testing.assert_allclose(pred['observed_probability'],p[:,np.arange(meta['trials']),obs['choices']-1],atol=1e-12,rtol=0)
                row.update(training_nll=nll(pred['observed_probability'][:,:cut]),test_nll=nll(pred['observed_probability'][:,cut:]))
            rows.append(row)
            bank[point['id']]=dict(row=row,metrics=metrics,features=features,pred=pred,seeds=auto['seeds'])
            if case['in_profile']:grid[tag['delta'],tag['capacity']]=bank[point['id']]
            for i,seed in enumerate(auto['seeds']):
                paths.append(dict(tag,seed=int(seed),**{k:v[i].item() for k,v in metrics.items()},**{k:bool(v[i]) for k,v in event.items()}))
            correct=auto['choices']==obs['categories'][None,:];width=policy['features']['block_width']
            targetmask=np.isin(auto['rule'],targets);contribution=(auto['contribution']*targetmask).sum(-1)
            for first in range(0,meta['trials'],width):
                last=min(first+width,meta['trials']);vals=correct[:,first:last].mean(1)
                curves.append(dict(tag,trial=last,observed=float((obs['choices'][first:last]==obs['categories'][first:last]).mean()),
                    mean=float(vals.mean()),q10=float(np.quantile(vals,.1)),q90=float(np.quantile(vals,.9)),
                    target_present=float(targetmask.any(-1)[:,first:last].mean()),target_contribution=float(contribution[:,first:last].mean()),cut=cut))
        observed.append(dict(subject=sid,mode=mode,**truth))
        anchor,selected=[bank[nominee[k]['id']] for k in ('original','candidate')]
        np.testing.assert_array_equal(anchor['seeds'],selected['seeds'])
        comparison=dict(subject=sid,mode=mode,changed=nominee['original']['id']!=nominee['candidate']['id'],
            anchor=nominee['original']['id'],selected=nominee['candidate']['id'])
        for label in ('full','training','late'):
            value,se=energy_contrast([selected['features'][label],anchor['features'][label]],[1,-1],empirical[label])
            comparison[label+'_energy_delta']=value;comparison[label+'_energy_delta_mcse']=se
        for key in ('late_accuracy','late_target_present','late_target_contribution'):
            value,se=linear_contrast([selected['metrics'][key],anchor['metrics'][key]],[1,-1])
            comparison[key+'_delta']=value;comparison[key+'_delta_mcse']=se
        for key,start,stop in [('training_nll',0,cut),('test_nll',cut,None)]:
            a,b=[x['pred']['observed_probability'][:,start:stop] for x in (anchor,selected)]
            guard=shared.likelihood_guard(a,b,policy['training_nll_tolerance'],policy['mcse_multiplier'])
            comparison[key+'_delta']=guard['nll_difference'];comparison[key+'_delta_mcse']=guard['difference_mcse']
        comparisons.append(comparison)
        d0,k0=[nominee['original']['hyperparams'][k] for k in (profile.DELTA,profile.CAPACITY)]
        for d,k in product([d for d in policy['retention_values'] if d!=d0],[k for k in policy['capacity_values'] if k!=k0]):
            x00,x10,x01,x11=[grid[key] for key in ((d0,k0),(d,k0),(d0,k),(d,k))]
            for item in (x10,x01,x11):np.testing.assert_array_equal(item['seeds'],x00['seeds'])
            for name,banks,weights in [('retention_only',[x10,x00],[1,-1]),('capacity_only',[x01,x00],[1,-1]),
                                      ('joint',[x11,x00],[1,-1]),('interaction',[x11,x10,x01,x00],[1,-1,-1,1])]:
                row=dict(subject=sid,mode=mode,delta_from=d0,delta_to=d,capacity_from=k0,capacity_to=k,contrast=name)
                for label in ('full','late'):
                    value,se=energy_contrast([b['features'][label] for b in banks],weights,empirical[label])
                    row[label+'_energy_delta']=value;row[label+'_energy_delta_mcse']=se
                for key in ('late_accuracy','late_target_present','late_target_contribution','ever_target'):
                    value,se=linear_contrast([b['metrics'][key] for b in banks],weights)
                    row[key+'_delta']=value;row[key+'_delta_mcse']=se
                factorial.append(row)
        print(f'S{sid} {mode}: all profile states and paired contrasts audited',flush=True)
    for name,data in [('summary',rows),('paths',paths),('events',events),('curves',curves),('observed',observed),
                      ('comparisons',comparisons),('factorial_contrasts',factorial),('job_audits',audits)]:
        pd.DataFrame(data).to_csv(output/(name+'.csv'),index=False)
    source_scores.to_csv(output/'training_profile.csv',index=False)
    write(output/'completion.json',dict(status='complete',evaluation=str(evaluation.resolve()),training=str(training.resolve()),
        profile_points=len(rows),paths=len(paths),groups=len(comparisons),changed=sum(x['changed'] for x in comparisons),
        scope=policy['scope'],policy=policy,source_sha256=digest(Path(__file__)),profile_sources=profile.sources(),
        analysis_sources_sha256={name:digest(ROOT/name) for name in (
            'src/Bayesian_state/workflows/analysis/model_1024_trajectory_diagnosis.py',
            'src/Bayesian_state/workflows/analysis/model_1023_execution_summary.py',
            'src/Bayesian_state/metrics/autonomous_fit.py','src/Bayesian_state/metrics/task.py')},
        input_hashes=input_hashes,artifacts_sha256={p.name:digest(p) for p in output.glob('*.csv')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.evaluation,args.output)
