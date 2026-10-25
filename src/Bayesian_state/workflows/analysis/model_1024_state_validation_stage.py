"""Bounded stage controller: freeze, generate unselected paths, infer and audit.

Use --phase prepare --config YAML --output NEW, then --phase panel|long
--prepared REGISTERED --output NEW [--smoke]. Long follow-up uses --previous.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import time

import numpy as np
import yaml

from . import model_1024_stage_worker as worker_tools
from . import model_1023_execution_validation as validation
from . import model_1023_execution_recovery as fixture_tools
from . import model_1023_execution_recovery_score as checked

ROOT=worker_tools.ROOT
ENTRY=str(Path(__file__).resolve().relative_to(ROOT))
read,write,digest=worker_tools.read,worker_tools.write,worker_tools.digest


def seed(base: int,label: str,index: int) -> int:
    return int.from_bytes(sha256(f'{base}:{label}:{index}'.encode()).digest()[:4],'little')


def dispatch(jobs: list[dict],output: Path,budget: int) -> None:
    if not jobs:return
    workers=min(budget,len(os.sched_getaffinity(0)),len(jobs));previous=validation.ENTRY
    validation.ENTRY=worker_tools.ENTRY
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            validation.NumericalQueue(pool).execute(jobs,output,workers)
    finally:validation.ENTRY=previous


def prepare(config: Path,output: Path) -> None:
    if output.exists():raise FileExistsError('Use a new stage registration directory')
    p=yaml.safe_load(config.read_text())
    if (p['contexts']!=['C000','C002','C005','C007','C010','C013'] or p['evaluation_paths_per_context']!=8
            or p['filter_seeds_per_path']!=2 or p['particles']!=512 or p['resolution_particles']!=1024
            or p['oral_status']!='deferred_pending_user_manual_review'):
        raise ValueError('Require the frozen stage scope')
    source=ROOT/p['source_cases'];definitions=read(source/'cases.json')['cases'];truths=read(source/'truths.json')['rows']
    banks=read(ROOT/p['source_banks']/'protocol.json')['cases'];inference=read(ROOT/p['source_inference']/'protocol.json')
    contexts=[];forbidden=set()
    for cid in p['contexts']:
        definition=next(c for c in definitions if c['case_id']==cid);truth=next(c for c in truths if c['case_id']==cid)
        bank=next(c for c in banks if (c['subject'],c['mode'],c['candidate']['id'])==(definition['subject'],'STICKY',truth['candidate']['id']))
        reference_seeds=[s for j in bank['autonomous_jobs'] for s in j['seeds']];forbidden.update(reference_seeds)
        if len(reference_seeds)!=128:raise ValueError('Require all 128 cached reference paths')
        draws=[seed(p['generation_seed_base'],cid,i) for i in range(p['evaluation_paths_per_context'])]
        filter_seeds=[[seed(p['filter_seed_base'],cid+'_'+str(i),j) for j in range(p['filter_seeds_per_path'])]
                      for i in range(p['evaluation_paths_per_context'])]
        all_new=draws+[s for ss in filter_seeds for s in ss]
        if len(set(all_new))!=len(all_new) or set(all_new)&forbidden:raise ValueError('Seed banks overlap')
        forbidden.update(all_new)
        contexts.append(dict(context_id=cid,subject=definition['subject'],target_rules=definition['target_rules'],
            template=definition['template'],candidate=truth['candidate'],truth_reference=truth,
            reference_jobs=bank['autonomous_jobs'],generation_seeds=draws,filter_seeds=filter_seeds))
    old_marginal=read(ROOT/p['source_marginal']/'protocol.json');long=[]
    for item in p['long_history']['cases']:
        cid=item['case_id'];case=next(c for c in old_marginal['cases'] if c['case_id']==cid)
        base=next(j for j in old_marginal['jobs'] if j['recovery_case']==cid and j['particles']==item['particles'])
        seeds=[seed(p['long_history']['seed_base'],cid,i) for i in range(64)]
        if len(set(seeds))!=64 or set(seeds)&forbidden:raise ValueError('Long-history seeds overlap')
        forbidden.update(seeds);long.append(dict(**item,case=case,template=base,seeds=seeds))
    output.mkdir(parents=True)
    inputs={str(config.resolve()):digest(config),str(source/'cases.json'):digest(source/'cases.json'),
        str(source/'truths.json'):digest(source/'truths.json'),str(ROOT/p['source_banks']/'protocol.json'):digest(ROOT/p['source_banks']/'protocol.json'),
        str(ROOT/p['source_inference']/'protocol.json'):digest(ROOT/p['source_inference']/'protocol.json')}
    write(output/'protocol.json',dict(policy=p,contexts=contexts,long_history=long,worker_sources=worker_tools.sources(),
        controller_source_sha256=digest(Path(__file__)),inputs_sha256=inputs,
        data_selection='all prespecified new seeds; no selection on behavior or latent states',oral_used=False))
    write(output/'completion.json',dict(status='registered_before_new_generation',protocol_sha256=digest(output/'protocol.json')))
    print('Stage scope, seeds, acceptance and budget registered before new simulation',flush=True)


def registered(prepared: Path) -> dict:
    p=read(prepared/'protocol.json')
    if (read(prepared/'completion.json')['protocol_sha256']!=digest(prepared/'protocol.json')
            or p['worker_sources']!=worker_tools.sources()):raise ValueError('Frozen registration changed')
    for file,sha in p['inputs_sha256'].items():
        if digest(Path(file))!=sha:raise ValueError('Registered input changed')
    return p


def filter_job(template: dict,candidate: dict,targets: list[int],folder: Path,output: Path,
               name: str,cid: str,seed_value: int,particles: int,prefix: int,smoke: bool) -> dict:
    job=deepcopy(template)
    for key in ('original_score','original_score_sha256','original_state','original_state_sha256','genealogy_target',
                'marginal_target','smoothing_sources','marginal_execution_sources','stage_kind','training_generation'):
        job.pop(key,None)
    kappa=candidate['hyperparams'][worker_tools.study.KAPPA];mode='MIX' if kappa==0 else 'STICKY'
    job.update(name=name,stage_kind='filter',kind='filter',model='COMPACT',execution_mode=mode,comparison_arm='ADAPT',
        candidate=candidate,folder=str(folder.resolve()),phase='training' if smoke else 'evaluate',smoke=smoke,
        particles=particles,seeds=[seed_value],target_rules=targets,capture_oral=False,
        source_sha256=worker_tools.study.sources(worker_tools.study.driver.io),capture_sources=worker_tools.marginal.capture.sources(),
        stage_sources=worker_tools.sources(),prefix_trials=prefix,fixed_lag=32,recovery_case=cid,
        target=str((output/'scores'/name).resolve()),state_capture_target=str((output/'states'/name).resolve()))
    job['mechanism']['capacity']='evidence'
    if mode=='MIX':job.update(marginal_execution_sources=worker_tools.marginal.sources(),marginal_target=str((output/'captures'/name).resolve()))
    else:job.update(smoothing_sources=worker_tools.smooth.sources(),genealogy_target=str((output/'captures'/name).resolve()))
    return job


def panel(prepared: Path,output: Path,smoke: bool) -> None:
    if output.exists():raise FileExistsError('Use a new recovery-panel directory')
    r=registered(prepared);p=r['policy'];output.mkdir(parents=True);started=time.monotonic()
    contexts=r['contexts']
    if smoke:contexts=[next(c for c in contexts if c['context_id']==cid) for cid in ('C000','C005','C010')]
    generation=[]
    for c in contexts:
        job=deepcopy(c['template']);name=c['context_id']+'_new_paths'
        job.update(name=name,stage_kind='generation',model='COMPACT',execution_mode='STICKY',comparison_arm='ADAPT',
            candidate=c['candidate'],phase='autonomous',smoke=False,seeds=c['generation_seeds'][:1] if smoke else c['generation_seeds'],
            source_sha256=worker_tools.study.sources(worker_tools.study.driver.io),stage_sources=worker_tools.sources(),
            target=str((output/'generation'/name).resolve()))
        job['mechanism']['capacity']='evidence'
        if smoke:job['generation_trials']=128
        generation.append(job)
    write(output/'registration.json',dict(prepared=str(prepared.resolve()),preparation_sha256=digest(prepared/'protocol.json'),
        generation_jobs=generation,smoke=smoke,selection='all registered seeds; no truth-dependent selection'))
    dispatch(generation,output,1 if smoke else p['parallel_budget'])
    cases=[];jobs=[]
    for c,g in zip(contexts,generation):
        arrays=checked.checked_arrays(Path(g['target']));np.testing.assert_array_equal(arrays['seeds'],g['seeds'])
        original=Path(c['template']['folder'])
        with np.load(original/'training.npz') as a,np.load(original/'sealed_test.npz') as b:
            stimulus=np.concatenate([a['stimulus'],b['stimulus']])
        for i,draw in enumerate(g['seeds']):
            cid=f'U{len(cases):03d}';raw={k:v.copy() if k=='categories' else v[i].copy() for k,v in arrays.items()}
            n=len(raw['choices']);obs=dict(stimulus=stimulus[:n],**{k:raw[k] for k in ('choices','categories','feedback')})
            folder=output/'observations'/cid/f'subject_{c["subject"]}'
            meta=fixture_tools.fixture(original,folder,obs,c['template']['runtime'],.7)
            truth_stem=output/'truth'/cid;truth_stem.parent.mkdir(exist_ok=True)
            with truth_stem.with_suffix('.npz').open('xb') as stream:np.savez_compressed(stream,**raw)
            row=dict(case_id=cid,context_id=c['context_id'],subject=c['subject'],condition=meta['condition'],
                targets=c['target_rules'],candidate=c['candidate'],folder=str(folder.resolve()),trials=n,
                generation_seed=draw,generation_index=i,truth=str(truth_stem.resolve()),truth_sha256=digest(truth_stem.with_suffix('.npz')),
                source_generation=g['target'],source_generation_sha256=digest(Path(g['target']).with_suffix('.npz')))
            cases.append(row)
            for level in ([2] if smoke else [512,1024] if i==p['resolution_path_index'] else [512]):
                for j,fs in enumerate(c['filter_seeds'][i][:1] if smoke else c['filter_seeds'][i]):
                    name=f'{cid}_P{level}_S{j}'
                    jobs.append(filter_job(c['template'],c['candidate'],c['target_rules'],folder,output,name,cid,fs,level,
                        8 if smoke else meta['training_trials'],smoke))
    controls=[]
    if smoke:
        for job in jobs:
            bare=deepcopy(job);bare.update(stage_kind='bare',name=job['name']+'_bare',target=job['target']+'_bare')
            bare.pop('capture_sources');controls.append(bare)
        dispatch(controls,output,1)
        for job,bare in zip(jobs,controls):job.update(original_score=bare['target']+'.npz',original_score_sha256=digest(Path(bare['target']).with_suffix('.npz')))
    write(output/'protocol.json',dict(prepared=str(prepared.resolve()),preparation_sha256=digest(prepared/'protocol.json'),
        policy=p,cases=cases,jobs=jobs,generation_jobs=generation,controls=controls,smoke=smoke,worker_sources=worker_tools.sources(),
        truth_used_for_selection=False,parameters_known=True,oral_used=False))
    dispatch(jobs,output,1 if smoke else p['parallel_budget'])
    write(output/'completion.json',dict(status='complete',smoke=smoke,cases=len(cases),filters=len(jobs),
        seconds=time.monotonic()-started,protocol_sha256=digest(output/'protocol.json'),oral_used=False))
    print('Unselected recovery panel complete',flush=True)


def long_history(prepared: Path,output: Path,previous: Path|None) -> None:
    if output.exists():raise FileExistsError('Use a new long-history directory')
    r=registered(prepared);p=r['policy'];output.mkdir(parents=True);jobs=[];cases=[]
    diagnostics={}
    if previous:
        decision=read(previous/'bank_diagnostics.json')
        if decision['status']!='complete':raise ValueError('Require initial bank diagnostics')
        diagnostics={c['case_id']:c for c in decision['cases']}
    for item in r['long_history']:
        if previous and not diagnostics[item['case_id']]['followup_required']:continue
        case=item['case'];a=32 if previous else 0;b=64 if previous else 32
        cases.append(dict(case,particles=item['particles']))
        for index,fs in enumerate(item['seeds'][a:b],a):
            name=f'{item["case_id"]}_B{index:03d}_P{item["particles"]}'
            template=item['template']
            jobs.append(filter_job(template,template['candidate'],template['target_rules'],Path(template['folder']),output,name,
                item['case_id'],fs,item['particles'],template['prefix_trials'],False))
    write(output/'protocol.json',dict(prepared=str(prepared.resolve()),preparation_sha256=digest(prepared/'protocol.json'),
        policy=p,jobs=jobs,cases=cases,previous=None if previous is None else str(previous.resolve()),
        previous_diagnostics_sha256=None if previous is None else digest(previous/'bank_diagnostics.json'),
        worker_sources=worker_tools.sources(),oral_used=False))
    started=time.monotonic();dispatch(jobs,output,p['parallel_budget'])
    write(output/'completion.json',dict(status='complete',filters=len(jobs),seconds=time.monotonic()-started,
        protocol_sha256=digest(output/'protocol.json'),followup=previous is not None,oral_used=False))
    print('Independent full-history particle systems complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=('prepare','panel','long'),required=True)
    parser.add_argument('--config',type=Path);parser.add_argument('--prepared',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--previous',type=Path);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.phase=='prepare' and args.config:prepare(args.config,args.output)
    elif args.phase=='panel' and args.prepared:panel(args.prepared,args.output,args.smoke)
    elif args.phase=='long' and args.prepared:long_history(args.prepared,args.output,args.previous)
    else:parser.error('Provide the required config or registered input')
