"""Compare saved state distributions across independent banks and resolutions.

This analysis reuses the original 20 belief bins and 25 precision bins. It also
checks the joint distribution of executed rule identity and precision. These
are additional diagnostics; they do not change registered adaptive gates.
Use --phase cache|summarize --source RUN --output NEW_ANALYSIS [--label NAME].
The cache is append-only and reuses immutable complete entries.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .model_1024_individual_states import ROOT, read, write, digest
from .model_1024_individual_state_distributions import weighted_histogram
from ...evaluation.independent_particle_systems import evidence_weights

ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
HELPER = 'src/Bayesian_state/workflows/analysis/model_1024_individual_state_distributions.py'
BELIEF_EDGES = np.linspace(0, 1, 21)
PRECISION_EDGES = np.linspace(0, 25, 26)
FIELDS = ('belief_histogram', 'precision_with_absence', 'executed_rule_precision')


def executed_precision_histogram(ids: np.ndarray, beta: np.ndarray, execution: np.ndarray,
                                 weights: np.ndarray, rules: int) -> np.ndarray:
    """Joint rule/precision probability; inactive slots contribute zero mass."""
    if ids.shape != beta.shape or ids.shape != execution.shape or ids.ndim != 3:
        raise ValueError('Sparse states are not aligned')
    if weights.shape != ids.shape[:2] or np.any(weights < 0) or not np.isfinite(weights).all():
        raise ValueError('Invalid particle weights')
    if not np.isfinite(beta).all() or beta.min() < 0 or beta.max() > 25+1e-10:
        raise ValueError('Precision outside registered histogram support')
    if not np.isfinite(execution).all() or np.any(execution < 0) or np.any(execution[ids<0]!=0):
        raise ValueError('Invalid execution mass')
    result = np.zeros((len(ids), rules*25))
    for t in range(len(ids)):
        valid = ids[t]>=0
        if np.any(ids[t][valid]>=rules):
            raise ValueError('Rule outside catalogue')
        bins = np.clip(np.searchsorted(PRECISION_EDGES,beta[t],side='right')-1,0,24)
        index = ids[t][valid]*25+bins[valid]
        result[t] = np.bincount(index,weights=(weights[t,:,None]*execution[t])[valid],minlength=rules*25)
    np.testing.assert_allclose(result.sum(axis=1),1.,atol=3e-12,rtol=0)
    return result


def cache_one(args: tuple[dict, Path]) -> dict:
    job, output = args; started=time.process_time(); wall=time.time()
    target=output/'jobs'/job['name']; metadata=target.with_suffix('.json')
    sources={ENTRY:digest(ROOT/ENTRY),HELPER:digest(ROOT/HELPER)}
    raw=Path(job['individual_target']).with_suffix('.npz'); expected=read(raw.with_suffix('.json'))['array_sha256']
    if metadata.exists():
        old=read(metadata)
        if old['sources']!=sources or old['raw_sha256']!=expected or digest(target.with_suffix('.npz'))!=old['array_sha256']:
            raise ValueError('Existing distribution cache changed')
        return old
    if digest(raw)!=expected:
        raise ValueError('Captured state archive changed')
    if len(job['target_rules'])!=1:
        raise ValueError('Target precision requires a single registered correct rule')
    target.parent.mkdir(parents=True,exist_ok=True)
    with np.load(raw) as z:
        ids=z['particle_rule']; target_mask=ids==job['target_rules'][0]
        present=target_mask.any(axis=2)
        belief=(target_mask*z['particle_belief']).sum(axis=2)
        beta=z['particle_precision']; target_beta=(target_mask*beta).sum(axis=2)
        execution=z['particle_execution']; rules=len(z['component_counts'])
        result={}
        for mode in ('filtered','full'):
            w=z[mode+'_weights']
            result[mode+'_belief_histogram']=weighted_histogram(belief,w,BELIEF_EDGES)
            hist=weighted_histogram(target_beta,w,PRECISION_EDGES,present)
            result[mode+'_precision_with_absence']=np.column_stack((np.maximum(0.,1-hist.sum(axis=1)),hist))
            np.testing.assert_allclose(hist.sum(axis=1),z[mode+'_target_present'],atol=3e-12,rtol=0)
            result[mode+'_executed_rule_precision']=executed_precision_histogram(ids,beta,execution,w,rules)
            # A joint rule/precision histogram must marginalize to the saved
            # execution distribution at exactly the same conditioning endpoint.
            np.testing.assert_allclose(result[mode+'_executed_rule_precision'].reshape(len(ids),rules,25).sum(axis=2),
                                       z[mode+'_execution'],atol=3e-12,rtol=0)
    with np.load(Path(job['target']).with_suffix('.npz')) as z:
        probability=z['observed_probability'][0]
        if not np.isfinite(probability).all() or np.any(probability<=0):
            raise ValueError('Invalid observed-choice evidence')
        result['log_evidence']=np.log(probability).cumsum()
    for mode in ('filtered','full'):
        for field in FIELDS:
            np.testing.assert_allclose(result[mode+'_'+field].sum(axis=1),1.,atol=3e-12,rtol=0)
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream,**result,belief_edges=BELIEF_EDGES,precision_edges=PRECISION_EDGES)
    meta=dict(name=job['name'],case=job['state_case'],particles=job['particles'],sources=sources,
        raw_sha256=expected,array_sha256=digest(target.with_suffix('.npz')),trials=len(ids),rules=rules,
        cpu_seconds=time.process_time()-started,wall_seconds=time.time()-wall,
        definitions='Target precision includes a distinct absent-rule atom; executed precision includes rule identity.')
    write(metadata,meta)
    return meta


def jobs_for_run(source: Path) -> list[dict]:
    p=read(source/'protocol.json'); jobs=read(source/'baseline_jobs.json')
    for file in (source/'jobs').glob('*.json'):
        ledger=source/'ledger'/file.name
        if ledger.exists() and read(ledger)['status']=='complete':
            jobs.append(read(file))
    return jobs


def cache(source: Path, output: Path, cases: list[str] | None, workers: int) -> None:
    p=read(source/'protocol.json'); scope=cases or p['policy']['pilot_cases']
    jobs=[j for j in jobs_for_run(source) if j['state_case'] in scope]
    output.mkdir(parents=True,exist_ok=True)
    registration=output/'registration.json'
    if not registration.exists():
        write(registration,dict(created_unix=time.time(),source=str(source.resolve()),
            sources={ENTRY:digest(ROOT/ENTRY),HELPER:digest(ROOT/HELPER)},
            belief_edges=BELIEF_EDGES.tolist(),precision_edges=PRECISION_EDGES.tolist(),
            scientific_role='Additional distribution diagnostics; registered adaptive gates unchanged'))
    else:
        if read(registration)['sources']!={ENTRY:digest(ROOT/ENTRY),HELPER:digest(ROOT/HELPER)}:
            raise ValueError('Distribution analysis source changed; use a new output')
    import os
    workers=min(workers,len(os.sched_getaffinity(0)),len(jobs))
    with ProcessPoolExecutor(max_workers=max(1,workers)) as pool:
        for i,meta in enumerate(pool.map(cache_one,[(j,output) for j in jobs])):
            if (i+1)%16==0 or i+1==len(jobs):
                print(f'Cached {i+1}/{len(jobs)} state distributions',flush=True)


def pool(paths: list[Path]) -> dict:
    logs=[]
    for path in paths:
        meta=read(path.with_suffix('.json'))
        if digest(path)!=meta['array_sha256']:
            raise ValueError('Distribution cache changed')
        with np.load(path) as z:
            logs.append(z['log_evidence'])
    logz=np.stack(logs); result={}
    for mode in ('filtered','full'):
        weights=evidence_weights(logz if mode=='filtered' else logz[:,-1])
        if mode=='full':
            weights=np.broadcast_to(weights[:,None],logz.shape)
        for path,outer in zip(paths,weights):
            with np.load(path) as z:
                for field in FIELDS:
                    key=mode+'_'+field
                    if key not in result:
                        result[key]=np.zeros_like(z[key])
                    result[key]+=z[key]*outer[:,None]
        for field in FIELDS:
            np.testing.assert_allclose(result[mode+'_'+field].sum(axis=1),1.,atol=3e-12,rtol=0)
    return result


def compare(a: dict, b: dict) -> tuple[dict, dict]:
    rows={}; trajectories={}
    for mode in ('filtered','full'):
        for field in FIELDS:
            key=mode+'_'+field; tv=.5*np.abs(a[key]-b[key]).sum(axis=1)
            trajectories[key]=tv
            rows[key+'_mean']=float(tv.mean());rows[key+'_q95']=float(np.quantile(tv,.95))
            rows[key+'_maximum']=float(tv.max())
            rows[key+'_early_q95']=float(np.quantile(tv[:max(1,len(tv)//4)],.95))
    return rows,trajectories


def summarize(source: Path, output: Path, label: str, cases: list[str] | None) -> None:
    destination=output/'summaries'/label
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    protocol=read(source/'protocol.json'); scope=cases or protocol['policy']['pilot_cases']
    available=[]
    for folder in sorted((source/'summaries').iterdir(),key=lambda f:f.stat().st_mtime):
        file=folder/'diagnostics.json'
        if file.exists():
            available.extend(read(file))
    unique={}
    for r in available:
        if r['case'] in scope:
            unique[(r['case'],r['particles'],r['systems'],r['confirmation'])]=r
    rows=[];banks={}
    for key,row in sorted(unique.items()):
        cid,n,r,confirmation=key
        paths=[output/'jobs'/(name+'.npz') for name in row['job_names']]
        if not all(path.exists() for path in paths):
            continue
        pooled=pool(paths);own=dict(case=cid,particles=n,systems=r,confirmation=confirmation)
        derived={}
        if r>=2:
            half=r//2; a=pool(paths[:half]);b=pool(paths[half:])
            difference,time_values=compare(a,b)
            own.update({'half_'+k:v for k,v in difference.items()})
            derived.update({'half_'+k:v for k,v in time_values.items()})
        lower=[k for k in banks if k[0]==cid and k[1]<n and not k[3]]
        reference=(cid,n,r,False) if confirmation and (cid,n,r,False) in banks else (
            max(lower,key=lambda k:(k[1],k[2])) if lower else None)
        if reference:
            difference,time_values=compare(pooled,banks[reference])
            own.update(comparison_particles=reference[1],comparison_systems=reference[2])
            own.update({'comparison_'+k:v for k,v in difference.items()})
            derived.update({'comparison_'+k:v for k,v in time_values.items()})
        name=f'{cid}_n{n}_r{r}'+('_confirm' if confirmation else '')
        file=destination/(name+'.npz')
        with file.open('xb') as stream:
            np.savez_compressed(stream,**pooled,**derived,belief_edges=BELIEF_EDGES,precision_edges=PRECISION_EDGES)
        own.update(pooled_file=str(file.resolve()),pooled_sha256=digest(file))
        rows.append(own);banks[key]=pooled
    write(destination/'diagnostics.json',rows)
    pd.DataFrame(rows).to_csv(destination/'diagnostics.csv',index=False)
    write(destination/'completion.json',dict(completed_unix=time.time(),banks=len(rows),cases=scope,
        sources={ENTRY:digest(ROOT/ENTRY),HELPER:digest(ROOT/HELPER)},
        role='Additional state-distribution checks; no change to registered six-quantity gates'))
    print(f'Summarized {len(rows)} distribution banks',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['cache','summarize'],required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--label',default='final')
    parser.add_argument('--cases',nargs='+')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.phase=='cache':
        cache(args.source,args.output,args.cases,args.workers)
    else:
        summarize(args.source,args.output,args.label,args.cases)
