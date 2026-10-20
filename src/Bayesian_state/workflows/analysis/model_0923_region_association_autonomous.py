"""Own-history validation of frozen independent-region diagnostic parameters.

Use --config YAML --output NEW_DIRECTORY [--smoke]. All paths have equal weight;
arms share choice random streams. Observed choices enter only the final scoring.
This condition-2 diagnostic retains all geometries and excludes H capacity/search.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

from .model_0923_region_association_diagnostic import ROOT, verify
from .model_0923_h_validation import write
from ...evaluation.model_0923_h_validation import load, digest, block_paths, distribution_scores, jackknife_se
from ...metrics.autonomous_fit import energy_u
from ...metrics.task import category_learning_feedback
from ...utils.seeding import stable_seed
from ...model.region_catalogue_learner import FixedCatalogueRegionLearner
from ...hypothesis_space.observation_model.continuous_partition import ContinuousPartition
from ...hypothesis_space.spaces.structural_0923 import RELATIONAL_EXTENSION

MODULE='src.Bayesian_state.workflows.analysis.model_0923_region_association_autonomous'


def generate(geometry: np.ndarray, categories: np.ndarray, *, gamma: float,
             rate: float | None, seed: int) -> dict[str,np.ndarray]:
    """Predict, sample a response, obtain task feedback, then update once."""
    q,labels=np.asarray(geometry,float),np.asarray(categories)
    if (q.ndim!=3 or not len(q) or q.shape[2]!=4 or labels.shape!=(len(q),)
            or not np.isfinite(labels).all() or np.any(labels!=labels.astype(int))
            or np.any((labels<1)|(labels>4))):
        raise ValueError('Need aligned four-category condition-2 stimuli')
    learner=FixedCatalogueRegionLearner(q.shape[1],rule_retention=gamma,association_learning_rate=rate)
    rng=np.random.default_rng(seed);p=np.empty((len(q),4));y=np.empty(len(q),dtype=np.int8);f=np.empty(len(q))
    for t in range(len(q)):
        p[t]=learner.predict(q[t]);y[t]=rng.choice(4,p=p[t])+1
        f[t]=category_learning_feedback(y[t],labels[t],condition=2)
        learner.update(q[t],int(y[t]),float(f[t]))
    return dict(probabilities=p,choices=y,feedback=f,final_rule_weights=learner.rule_weights,
                final_associations=learner.associations)


def worker(file: Path) -> None:
    job=load(file);verify(job['bindings']);target=Path(job['target'])
    if job['stage']=='geometry':
        pieces=[];labels=[]
        for path in job['data']:
            with np.load(path) as z:pieces.append(z['stimulus']);labels.append(z['categories'])
        x=np.concatenate(pieces)[:job['length']];labels=np.concatenate(labels)[:job['length']]
        partition=ContinuousPartition(4,4,structural_extension=RELATIONAL_EXTENSION)
        q=np.empty((len(x),152,4))
        for h in range(152):q[:,h]=partition.get_category_probabilities(h,(x,),job['beta'],distance_mode='boundary').T
        arrays=dict(geometry=q,categories=labels)
    elif job['stage']=='autonomous':
        with np.load(job['geometry']) as z:q=z['geometry'];labels=z['categories']
        paths=[generate(q,labels,gamma=job['gamma'],rate=job['rate'],seed=s) for s in job['seeds']]
        arrays={k:np.stack([r[k] for r in paths]) for k in paths[0]}
        arrays.update(categories=labels,seeds=np.asarray(job['seeds'],dtype=np.uint64))
    else:raise ValueError('Unknown autonomous stage')
    verify(job['bindings']);target.parent.mkdir(parents=True,exist_ok=True)
    with target.with_suffix('.npz').open('xb') as handle:np.savez_compressed(handle,**arrays)
    write(target.with_suffix('.json'),dict(status='complete',job_sha256=digest(file),
        arrays_sha256=digest(target.with_suffix('.npz')),stage=job['stage'],subject=job['subject']))


def execute(jobs: list[dict], output: Path, workers: int) -> None:
    (output/'jobs').mkdir(exist_ok=True)
    workers=min(workers,len(jobs),len(os.sched_getaffinity(0)))
    def launch(job):
        file=output/'jobs'/(job['name']+'.json');write(file,job)
        with tempfile.TemporaryDirectory(prefix='region_autonomous_') as cache:
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR=cache,NUMBA_CACHE_DIR=cache,PYTHONDONTWRITEBYTECODE='1')
            with file.with_suffix('.log').open('x') as log:
                subprocess.run([sys.executable,'-m',MODULE,'--worker',str(file)],cwd=ROOT,env=env,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'{jobs[0]["stage"]}: {len(jobs)} jobs, {workers} single-thread workers',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,future in enumerate(as_completed([pool.submit(launch,j) for j in jobs]),1):
            future.result()
            if i%16==0 or i==len(jobs):print(f'{i}/{len(jobs)} complete',flush=True)


def run(config: Path, output: Path, smoke: bool=False) -> None:
    settings=yaml.safe_load(config.read_text());source=ROOT/settings['source']
    if (set(settings)!={'source','paths','chunk','workers','block_width','seed','protocol'}
            or any(type(settings[k]) is not int or settings[k]<1 for k in ('paths','chunk','workers','block_width'))
            or settings['paths']<4 or settings['workers']>64):raise ValueError('Need at least4 paths and at most64 workers')
    if output.exists():raise FileExistsError('Use a new autonomous output directory')
    done=load(source/'completion.json');pro=load(source/'protocol.json');frozen=load(source/'nominees.json')
    if (done['status']!='complete' or done['smoke'] or frozen['status']!='frozen_before_later_choices'
            or digest(source/'nominees.json')!=done['nomination_sha256']
            or digest(source/'protocol.json')!=done['protocol_sha256']):
        raise ValueError('Require a completed full diagnostic and frozen training nominees')
    bindings=dict(pro['bindings']);bindings[str(config.resolve())]=digest(config)
    for name in ('completion.json','nominees.json','protocol.json'):bindings[str(source/name)]=digest(source/name)
    files=[Path(__file__),ROOT/'src/Bayesian_state/metrics/autonomous_fit.py',ROOT/'src/Bayesian_state/metrics/task.py',ROOT/'src/Bayesian_state/utils/seeding.py']
    for p in files:bindings[str(p.resolve())]=digest(p)
    nominees=frozen['nominees']
    if smoke:nominees=[n for n in nominees if n['subject']==nominees[0]['subject']]
    meta={};data={}
    for s in sorted({n['subject'] for n in nominees}):
        folder=ROOT/pro['settings']['data']/f'subject_{s}';meta[s]=load(folder/'prepared.json')
        if meta[s]['condition']!=2:raise ValueError('Condition2 only')
        data[s]=[str(folder/name) for name in ('training.npz','sealed_test.npz')]
        for name,path in zip(('training.npz','sealed_test.npz'),data[s]):bindings[path]=meta[s]['array_sha256'][name]
    verify(bindings);output=output.resolve();output.mkdir(parents=True)
    for p in {Path(k).resolve() for k in bindings if k.endswith('.py')}:
        target=output/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    shutil.copy2(config,output/'config.yaml')
    paths,chunk=(4,4) if smoke else (settings['paths'],settings['chunk'])
    banks={s:[stable_seed(dict(protocol=settings['protocol'],base=settings['seed'],subject=s,path=i)) for i in range(paths)] for s in meta}
    if len({v for bank in banks.values() for v in bank})!=paths*len(meta):raise ValueError('Duplicate path seeds')
    write(output/'protocol.json',dict(settings=settings,smoke=smoke,bindings=bindings,nominees=nominees,
        seed_banks=banks,paths_per_group=paths,chunk=chunk,python=sys.version,versions=pro['versions'],
        parent_nomination_sha256=done['nomination_sha256'],
        scope='Condition2, all152 geometries, fixed beta, own choices and feedback from trial1; equal path weights; no H or oral',
        scoring='RMS-scaled block-accuracy Energy U primary; pair complete paths by common seed; MCSE describes path simulation only',
        resource_policy='At most64 additional workers; with existing33+14+14+2 at most127 numerical/report processes'))
    geometries=[];lookup={}
    for i,(s,beta) in enumerate(sorted({(n['subject'],n['beta']) for n in nominees})):
        target=output/'geometry'/f'S{s}_B{i}';lookup[s,beta]=target.with_suffix('.npz')
        geometries.append(dict(name=f'geometry_S{s}_B{i}',stage='geometry',subject=s,beta=beta,data=data[s],
            length=min(16,meta[s]['trials']) if smoke else meta[s]['trials'],target=str(target),bindings=bindings))
    execute(geometries,output,1 if smoke else settings['workers'])
    jobs=[];groups={}
    for n in nominees:
        s,arm=n['subject'],n['arm'];file=lookup[s,n['beta']];groups[s,arm]=[]
        for first in range(0,paths,chunk):
            name=f'S{s}_{arm}_C{first//chunk:03d}'
            job=dict(n,name=name,stage='autonomous',geometry=str(file),seeds=banks[s][first:first+chunk],
                bindings={**bindings,str(file):digest(file)},target=str(output/'paths'/name))
            jobs.append(job);groups[s,arm].append(job)
    execute(jobs,output,1 if smoke else settings['workers'])
    rows=[];curves=[];contrasts=[];bank={};artifacts={}
    for (s,arm),members in groups.items():
        arrays=[]
        for j in members:
            file=Path(j['target']).with_suffix('.npz');rec=load(file.with_suffix('.json'))
            assert digest(file)==rec['arrays_sha256'];artifacts[str(file)]=rec['arrays_sha256']
            with np.load(file) as z:arrays.append({k:z[k].copy() for k in z.files})
        np.testing.assert_array_equal(np.concatenate([a['seeds'] for a in arrays]),banks[s])
        labels=arrays[0]['categories']
        for a in arrays:np.testing.assert_array_equal(a['categories'],labels)
        generated=np.concatenate([a['choices'] for a in arrays])==labels[None]
        observed=[]
        for file in data[s]:
            with np.load(file) as z:observed.append(z['choices'])
        human=np.concatenate(observed)[:len(labels)]==labels
        n=len(labels);cut=min(meta[s]['training_trials'],n)
        windows=[('whole',0,n),('training',0,cut)]+([('test',cut,n)] if cut<n else [])
        for phase,first,last in windows:
            x=block_paths(generated,first,last,settings['block_width']);y=block_paths(human[None],first,last,settings['block_width'])[0]
            u=energy_u(x/np.sqrt(x.shape[1]),y/np.sqrt(len(y)))
            jack=np.array([energy_u(np.delete(x,i,axis=0)/np.sqrt(x.shape[1]),y/np.sqrt(len(y)))['value'] for i in range(paths)])
            np.testing.assert_allclose(jackknife_se(jack),u['mc_se'],rtol=1e-10,atol=1e-14)
            late=generated[:,max(first,last-128):last].mean(1);actual=human[max(first,last-128):last].mean()
            rows.append(dict(subject=s,arm=arm,phase=phase,paths=paths,**distribution_scores(x,y),energy_u=u['value'],energy_u_mc_se=u['mc_se'],
                late_accuracy=late.mean(),observed_late=actual,late_reaches_observed=(late>=actual).mean()))
            bank[s,arm,phase]=(u['value'],jack)
        x=block_paths(generated,0,n,settings['block_width']);y=block_paths(human[None],0,n,settings['block_width'])[0]
        for b in range(x.shape[1]):curves.append(dict(subject=s,arm=arm,block=b+1,observed=y[b],mean=x[:,b].mean(),q10=np.quantile(x[:,b],.1),q90=np.quantile(x[:,b],.9)))
    for (s,arm,phase),(value,jack) in bank.items():
        if arm=='ROW':
            other,other_jack=bank[s,'FIXED',phase]
            contrasts.append(dict(subject=s,phase=phase,delta_energy_u=value-other,delta_energy_u_mc_se=jackknife_se(jack-other_jack)))
    for name,records in [('scores',rows),('contrasts',contrasts),('curves',curves)]:
        target=output/(name+'.csv');pd.DataFrame(records).to_csv(target,index=False);artifacts[str(target)]=digest(target)
    verify(bindings);verify(artifacts)
    write(output/'completion.json',dict(status='complete',smoke=smoke,groups=len(groups),paths_per_group=paths,
        protocol_sha256=digest(output/'protocol.json'),artifacts=artifacts))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',type=Path);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.config is None or args.output is None:parser.error('--config and --output are required')
    else:run(args.config,args.output,args.smoke)
