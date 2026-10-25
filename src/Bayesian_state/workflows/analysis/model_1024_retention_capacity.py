"""Profile existing belief retention and capacity with all other parameters fixed.

Use --phase train --config YAML --output NEW [--smoke], then --phase evaluate
--training COMPLETE --config YAML --output NEW. Shared execution/selection
workers provide filtering, simulation and enforced training boundaries. Every
profile point enters frozen generation; only training chooses the final point.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import yaml

from . import model_1024_generation_selection as shared
from .model_1024_trajectory_diagnosis import manifest
from ...metrics.autonomous_fit import behavior_features, energy_u
from ...optimization.search.adaptive_proposals import candidate

ROOT=Path(__file__).resolve().parents[4]
ENTRY=str(Path(__file__).resolve().relative_to(ROOT))
DELTA='engine.modules.hypo_transitions_mod.kwargs.intervention.decay'
CAPACITY='engine.modules.hypo_transitions_mod.kwargs.max_capacity'
read,write,digest,nll=shared.read,shared.write,shared.digest,shared.nll


def sources() -> dict:
    return dict(shared.sources(),**{ENTRY:digest(ROOT/ENTRY)})


def profile(anchor: dict, retentions: list, capacities: list) -> list[dict]:
    """Change exactly the two declared coordinates, retaining the anchor object."""
    if (not retentions or not capacities or len(set(retentions))!=len(retentions)
            or len(set(capacities))!=len(capacities) or any(not np.isfinite(d) or not 0<=d<=1 for d in retentions)
            or any(isinstance(k,bool) or not isinstance(k,int) or k<1 for k in capacities)):
        raise ValueError('Require distinct valid retention and integer-capacity values')
    if anchor['hyperparams'][DELTA] not in retentions or anchor['hyperparams'][CAPACITY] not in capacities:
        raise ValueError('The original anchor must belong to the profile grid')
    rows=[]
    for delta,capacity in product(retentions,capacities):
        parameters=dict(anchor['hyperparams'],**{DELTA:delta,CAPACITY:capacity})
        point=candidate(parameters,'retention_capacity_profile')
        rows.append(deepcopy(anchor) if point['id']==anchor['id'] else point)
    return rows


def choose_profile(scores: list[dict], reference_nll: float, tolerance: float) -> dict | None:
    """Use one fixed likelihood reference rather than rolling tolerance forward."""
    if tolerance<0 or not np.isfinite(reference_nll) or any(
            not np.isfinite([r['training_nll'],r['training_energy']]).all() for r in scores):
        raise ValueError('Require finite training scores and nonnegative tolerance')
    eligible=[r for r in scores if r['training_nll']<=reference_nll+tolerance]
    return min(eligible,key=lambda r:(r['training_energy'],r['training_nll'],r['candidate'])) if eligible else None


def extract_seeds(value) -> set[int]:
    """Collect only explicit simulation-role seed fields from prior protocols."""
    result=set()
    if isinstance(value,dict):
        for key,item in value.items():
            if key in ('seeds','branch_seeds','forbidden') and isinstance(item,list):
                result.update(int(s) for s in item if isinstance(s,int))
            else:result.update(extract_seeds(item))
    elif isinstance(value,list):
        for item in value:result.update(extract_seeds(item))
    return result


def checked_policy(config: Path) -> dict:
    p=yaml.safe_load(config.read_text())
    if (p['subjects']!=[129,229,317] or p['modes']!=['MIX','STICKY'] or
            p['retention_values']!=[0.,.3,.6,.8,1.] or p['capacity_values']!=[2,3,6] or
            p['fallback']!='retain_current_joint_anchor' or not 1<=p['parallel_budget']<=128):
        raise ValueError('Require the registered fixed profile and fallback')
    return p


def clean_template(case: dict) -> dict:
    template=deepcopy(case['autonomous_jobs'][0])
    for key in ('freeze_file','freeze_sha256','selection_sources','training_generation','generation_trials'):
        template.pop(key,None)
    return template


def verify_inputs(bindings: dict) -> None:
    for name,sha in bindings.items():
        if digest(Path(name))!=sha:raise ValueError('Frozen input changed: '+name)


def train(config: Path, output: Path, smoke: bool) -> None:
    if output.exists():raise FileExistsError('Use a new profile training directory')
    p=checked_policy(config); source=ROOT/p['source_evaluation']; study=ROOT/p['source_study']
    previous=read(source/'protocol.json'); original=read(study/'nominees.json')
    if (read(source/'completion.json')['status']!='complete' or original['status']!='frozen_before_test'
            or original['later_choices_used'] or original['oral_used']):
        raise ValueError('Require completed training-only fixed sources')
    current_freeze=Path(previous['training'])/'nominees.json'
    if digest(current_freeze)!=previous['frozen_nominees_sha256']:
        raise ValueError('Current parameter freeze changed')
    old_cases=[c for c in previous['cases'] if c['subject'] in p['subjects'] and c['mode'] in p['modes'] and c['is_selected']]
    if len(old_cases)!=6:raise ValueError('Expected six current joint anchors')
    if smoke:old_cases=old_cases[:1]
    output.mkdir(parents=True)
    inputs={str(config.resolve()):digest(config),str((source/'protocol.json').resolve()):digest(source/'protocol.json'),
        str(current_freeze):digest(current_freeze),str((study/'nominees.json').resolve()):digest(study/'nominees.json')}
    forbidden=extract_seeds(previous)
    for name in p['seed_history']:
        path=ROOT/name;inputs[str(path.resolve())]=digest(path);forbidden.update(extract_seeds(read(path)))
    cases=[];jobs=[];started=time.monotonic()
    for own in old_cases:
        sid,mode=own['subject'],own['mode'];anchor=own['candidate']
        reference=next(r['candidate'] for r in original['nominees'] if (r['subject'],r['arm'])==(sid,mode))
        template=clean_template(own); folder=Path(own['folder']);meta=read(folder/'prepared.json')
        for name in ('prepared.json','engine.json','training.npz'):
            path=folder/name;inputs[str(path)]=digest(path)
        if digest(folder/'training.npz')!=meta['array_sha256']['training.npz']:raise ValueError('Training data changed')
        with np.load(folder/'training.npz') as z:
            count=8 if smoke else len(z['choices'])
            empirical=behavior_features(z['choices'][None,:count],z['categories'][:count],
                n_categories=2 if meta['condition']==1 else 4,**p['features'])[0]
        retentions=sorted({p['retention_values'][0],anchor['hyperparams'][DELTA]}) if smoke else p['retention_values']
        capacities=sorted({p['capacity_values'][0],anchor['hyperparams'][CAPACITY]}) if smoke else p['capacity_values']
        points=profile(anchor,retentions,capacities)
        filters=shared.seed_bank(p,sid,'screen_filter',2 if smoke else p['screen']['seeds'],forbidden)
        paths=shared.seed_bank(p,sid,'screen_generation',4 if smoke else p['screen']['paths'],forbidden|set(filters))
        filter_jobs={};energy_jobs={}
        for point in {c['id']:c for c in points+[reference]}.values():
            fj=shared.point_jobs(template,point,output,'screen_filter',filters,p,phase='training',
                particles=2 if smoke else p['screen']['particles'],smoke=smoke)
            filter_jobs[point['id']]=fj;jobs.extend(fj)
        for point in points:
            aj=shared.point_jobs(template,point,output,'screen_generation',paths,p,training_generation=True,smoke=smoke)
            energy_jobs[point['id']]=aj;jobs.extend(aj)
        cases.append(dict(subject=sid,mode=mode,anchor=anchor,nll_reference=reference,candidates=points,template=template,
            empirical=empirical.tolist(),screen_filters=filter_jobs,screen_paths=energy_jobs,
            forbidden=sorted(forbidden|set(filters)|set(paths))))
    write(output/'protocol.json',dict(policy=p,cases=cases,smoke=smoke,inputs_sha256=inputs,profile_sources=sources(),
        provenance=manifest(config),later_choices_used=False,oral_used=False))
    budget=1 if smoke else p['parallel_budget'];shared.execute(jobs,output,budget)
    scores=[];confirmation=[]
    for case in cases:
        reference_nll=nll(shared.filter_arrays(case['screen_filters'][case['nll_reference']['id']])['observed_probability'])
        own=[]
        for point in case['candidates']:
            pred=shared.filter_arrays(case['screen_filters'][point['id']])
            features=shared.training_features(case['screen_paths'][point['id']],np.array(case['empirical']))
            energy=energy_u(features,np.array(case['empirical']))
            row=dict(subject=case['subject'],mode=case['mode'],candidate=point['id'],delta=point['hyperparams'][DELTA],
                capacity=point['hyperparams'][CAPACITY],training_nll=nll(pred['observed_probability']),
                training_energy=energy['value'],training_energy_mcse=energy['mc_se'],reference_nll=reference_nll)
            row['eligible']=row['training_nll']<=reference_nll+p['training_nll_tolerance'];own.append(row)
        scores.extend(own);winner=choose_profile(own,reference_nll,p['training_nll_tolerance'])
        case['no_eligible']=winner is None
        case['proposed']=next(c for c in case['candidates'] if c['id']==winner['candidate']) if winner else case['anchor']
        case['energy_jobs'],case['filter_jobs']={},{}
        if case['proposed']['id']==case['anchor']['id'] and not smoke:continue
        deny=set(case['forbidden'])
        filters=shared.seed_bank(p,case['subject'],'confirmation_filter',2 if smoke else p['guard']['seeds'],deny)
        paths=shared.seed_bank(p,case['subject'],'confirmation_generation',4 if smoke else p['confirmation_paths'],deny|set(filters))
        for point in {c['id']:c for c in [case['anchor'],case['proposed'],case['nll_reference']]}.values():
            fj=shared.point_jobs(case['template'],point,output,'confirmation_filter',filters,p,phase='training',
                particles=2 if smoke else p['guard']['particles'],smoke=smoke)
            case['filter_jobs'][point['id']]=fj;confirmation.extend(fj)
        for point in {c['id']:c for c in [case['anchor'],case['proposed']]}.values():
            aj=shared.point_jobs(case['template'],point,output,'confirmation_generation',paths,p,
                training_generation=True,smoke=smoke)
            case['energy_jobs'][point['id']]=aj;confirmation.extend(aj)
    pd.DataFrame(scores).to_csv(output/'screen_scores.csv',index=False)
    write(output/'proposals.json',dict(cases=cases,later_choices_used=False,oral_used=False))
    shared.execute(confirmation,output,budget)
    decisions=[];nominees=[]
    for case in cases:
        old,new,reference=case['anchor'],case['proposed'],case['nll_reference'];likelihood={};generation={}
        if case['energy_jobs']:
            a,b=[shared.training_features(case['energy_jobs'][c['id']],np.array(case['empirical'])) for c in (old,new)]
            generation=shared.paired_energy_guard(a,b,np.array(case['empirical']),p['mcse_multiplier'])
            a,b=[shared.filter_arrays(case['filter_jobs'][c['id']]) for c in (reference,new)]
            np.testing.assert_array_equal(a['seeds'],b['seeds'])
            likelihood=shared.likelihood_guard(a['observed_probability'],b['observed_probability'],
                p['training_nll_tolerance'],p['mcse_multiplier'])
        decision=shared.final_decision(old['id'],new['id'],likelihood,generation)
        if case['no_eligible']:decision['reason']='no_profile_point_within_fixed_reference_tolerance'
        decisions.append(dict(subject=case['subject'],mode=case['mode'],nll_reference=reference['id'],**decision))
        nominees.append(dict(subject=case['subject'],mode=case['mode'],original=old,nll_reference=reference,
            candidate=next(c for c in case['candidates'] if c['id']==decision['selected']),candidates=case['candidates'],template=case['template']))
    verify_inputs(inputs)
    write(output/'decisions.json',dict(decisions=decisions,later_choices_used=False,oral_used=False))
    write(output/'nominees.json',dict(status='frozen_before_test',nominees=nominees,smoke=smoke,
        later_choices_used=False,oral_used=False,profile_sources=sources(),decisions_sha256=digest(output/'decisions.json')))
    write(output/'completion.json',dict(status='complete',smoke=smoke,groups=len(cases),replaced=sum(d['replaced'] for d in decisions),
        seconds=time.monotonic()-started,screen_jobs=len(jobs),confirmation_jobs=len(confirmation),
        nominees_sha256=digest(output/'nominees.json'),protocol_sha256=digest(output/'protocol.json')))
    print('Training complete; all two-parameter decisions frozen',flush=True)


def evaluate(config: Path, training: Path, output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new profile evaluation directory')
    p=checked_policy(config);frozen=read(training/'nominees.json');done=read(training/'completion.json');protocol=read(training/'protocol.json')
    if (done['status']!='complete' or done['smoke'] or frozen['status']!='frozen_before_test'
            or frozen['later_choices_used'] or protocol['policy']!=p or frozen['profile_sources']!=sources()
            or done['nominees_sha256']!=digest(training/'nominees.json')):
        raise ValueError('Require the intact formal training freeze')
    verify_inputs(protocol['inputs_sha256']);output.mkdir(parents=True)
    previous=read(training/'proposals.json')['cases'];jobs=[];cases=[]
    for row in frozen['nominees']:
        case=next(c for c in previous if (c['subject'],c['mode'])==(row['subject'],row['mode']))
        deny=set(case['forbidden'])|extract_seeds(case['energy_jobs'])|extract_seeds(case['filter_jobs'])
        filters=shared.seed_bank(p,row['subject'],'frozen_filter',p['evaluate']['seeds'],deny)
        paths=shared.seed_bank(p,row['subject'],'frozen_generation',p['evaluation_paths'],deny|set(filters))
        predicted={c['id'] for c in (row['original'],row['candidate'],row['nll_reference'])}
        for point in {c['id']:c for c in row['candidates']+[row['nll_reference']]}.values():
            aj=shared.point_jobs(row['template'],point,output,'evaluation_generation',paths,p,freeze=training/'nominees.json')
            fj=shared.point_jobs(row['template'],point,output,'evaluation_filter',filters,p,phase='evaluate',
                particles=p['evaluate']['particles'],freeze=training/'nominees.json') if point['id'] in predicted else []
            cases.append(dict(subject=row['subject'],mode=row['mode'],candidate=point,folder=row['template']['folder'],
                autonomous_jobs=aj,filter_jobs=fj,is_anchor=point['id']==row['original']['id'],
                is_selected=point['id']==row['candidate']['id'],is_nll_reference=point['id']==row['nll_reference']['id'],
                in_profile=point['id'] in {c['id'] for c in row['candidates']}))
            jobs.extend(aj+fj)
    write(output/'protocol.json',dict(policy=p,training=str(training.resolve()),cases=cases,
        frozen_nominees_sha256=digest(training/'nominees.json'),profile_sources=sources(),provenance=manifest(config)))
    started=time.monotonic();shared.execute(jobs,output,p['parallel_budget'])
    write(output/'completion.json',dict(status='complete',cases=len(cases),seconds=time.monotonic()-started,
        protocol_sha256=digest(output/'protocol.json'),frozen_nominees_sha256=digest(training/'nominees.json'),
        paths=sum(len(j['seeds']) for c in cases for j in c['autonomous_jobs']),filter_jobs=sum(len(c['filter_jobs']) for c in cases)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('train','evaluate'),required=True)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--training',type=Path);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase=='train':train(args.config,args.output,args.smoke)
        elif args.training and not args.smoke:evaluate(args.config,args.training,args.output)
        else:parser.error('Evaluation requires formal --training')
