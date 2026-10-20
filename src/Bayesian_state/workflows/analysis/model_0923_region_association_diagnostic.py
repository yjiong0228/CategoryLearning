"""Bounded condition-2 comparison of fixed labels and independent associations.

Use --config YAML --output NEW_DIRECTORY [--smoke]. Reuse only the completed
geometry cache, never the historical 24-assignment learner. All 152 structures
stay available. Shared beta/gamma grids and a declared extra association-rate
grid are scored on training choices. Freeze nominees before opening later data.
Later feedback continues learning at fixed parameters. Oral reports are absent.
Actual grid budgets differ because the learned arm has one extra parameter.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

from ...model.region_catalogue_learner import FixedCatalogueRegionLearner
from ...hypothesis_space.observation_model.continuous_partition import ContinuousPartition
from ...hypothesis_space.spaces.structural_0923 import RELATIONAL_EXTENSION
from ...evaluation.model_0923_h_validation import digest, load
from .model_0923_h_validation import write

ROOT=Path(__file__).resolve().parents[4]
MODULE='src.Bayesian_state.workflows.analysis.model_0923_region_association_diagnostic'


def replay(geometry: np.ndarray, choices: np.ndarray, feedback: np.ndarray, *,
           gamma: float, rate: float | None, trace: bool=False) -> dict:
    """Predict before observing this trial's choice and binary feedback."""
    q,y,f=np.asarray(geometry,float),np.asarray(choices),np.asarray(feedback,float)
    if (q.ndim!=3 or q.shape[2]!=4 or not len(q) or y.shape!=(len(q),) or f.shape!=y.shape
            or not np.isfinite(y).all() or np.any(y!=y.astype(int)) or np.any((y<1)|(y>4))
            or not np.isin(f,[0.,1.]).all()):
        raise ValueError('Need aligned four-response condition-2 trial arrays')
    learner=FixedCatalogueRegionLearner(q.shape[1],rule_retention=gamma,association_learning_rate=rate)
    predictions=[];rules=[];associations=[]
    for geometry,choice,outcome in zip(q,y,f):
        predictions.append(learner.predict(geometry))
        if trace:
            rules.append(learner.rule_weights);associations.append(learner.associations)
        learner.update(geometry,int(choice),float(outcome))
    predictions=np.asarray(predictions)
    result=dict(probabilities=predictions,observed_probability=predictions[np.arange(len(q)),y.astype(int)-1])
    if trace:
        result.update(rule_weights=np.asarray(rules),associations=np.asarray(associations),
                      final_rule_weights=learner.rule_weights,final_associations=learner.associations)
    return result


def verify(bindings: dict) -> None:
    for path,expected in bindings.items():
        if digest(ROOT/path)!=expected:
            raise ValueError('Bound source or numerical input changed: '+path)


def worker(file: Path) -> None:
    job=load(file);verify(job['bindings'])
    with np.load(job['geometry']) as z:q=z['geometry'][job['beta_index'],:job['training_length']]
    with np.load(job['training']) as z:y=z['choices'][:len(q)];f=z['feedback'][:len(q)]
    cut=len(q)
    if job['stage']=='evaluation':
        frozen=load(Path(job['nominees']))
        if frozen['status']!='frozen_before_later_choices':
            raise ValueError('Freeze parameters before later data')
        with np.load(job['later']) as z:
            x=z['stimulus'][:job['test_length']];later_y=z['choices'][:len(x)];later_f=z['feedback'][:len(x)]
        partition=ContinuousPartition(4,4,structural_extension=RELATIONAL_EXTENSION)
        later_q=np.empty((len(x),152,4))
        for h in range(152):
            later_q[:,h]=partition.get_category_probabilities(h,(x,),job['beta'],distance_mode='boundary').T
        q=np.concatenate([q,later_q]);y=np.r_[y,later_y];f=np.r_[f,later_f]
    result=replay(q,y,f,gamma=job['gamma'],rate=job['rate'],trace=job['stage']=='evaluation')
    p=result['observed_probability']
    if not np.isfinite(p).all() or np.any(p<=0):
        raise ValueError('Require finite positive observed choice probabilities')
    np.testing.assert_allclose(result['probabilities'].sum(1),1.,atol=1e-12,rtol=0)
    record=dict(status='complete',subject=job['subject'],arm=job['arm'],stage=job['stage'],beta=job['beta'],
                gamma=job['gamma'],rate=job['rate'],trials=len(q),mean_nll=float(-np.log(p).mean()))
    if job['stage']=='evaluation':
        with np.load(job['training_prediction']) as z:
            np.testing.assert_array_equal(result['probabilities'][:cut],z['probabilities'])
        record.update(training_trials=cut,test_trials=len(q)-cut,training_nll=float(-np.log(p[:cut]).mean()),
                      test_nll=float(-np.log(p[cut:]).mean()),exact_training_prefix=True)
    verify(job['bindings']);target=Path(job['target']);target.parent.mkdir(parents=True,exist_ok=True)
    with target.with_suffix('.npz').open('xb') as handle:np.savez_compressed(handle,**result)
    record.update(array_sha256=digest(target.with_suffix('.npz')),job_sha256=digest(file))
    write(target.with_suffix('.json'),record)


def execute(jobs: list[dict], output: Path, workers: int) -> None:
    workers=min(workers,len(jobs),len(os.sched_getaffinity(0)))
    (output/'jobs').mkdir(exist_ok=True)
    def launch(job):
        file=output/'jobs'/(job['name']+'.json');write(file,job)
        with tempfile.TemporaryDirectory(prefix='region_association_') as cache:
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     NUMEXPR_NUM_THREADS='1',MPLCONFIGDIR=cache,NUMBA_CACHE_DIR=cache,PYTHONDONTWRITEBYTECODE='1')
            with file.with_suffix('.log').open('x') as log:
                subprocess.run([sys.executable,'-m',MODULE,'--worker',str(file)],cwd=ROOT,env=env,
                               stdout=log,stderr=subprocess.STDOUT,check=True)
        return job['name']
    print(f'{jobs[0]["stage"]}: {len(jobs)} jobs, {workers} single-thread workers',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,name in enumerate(pool.map(launch,jobs),1):
            if i%16==0 or i==len(jobs):print(f'{i}/{len(jobs)} complete',flush=True)


def run(config: Path, output: Path, smoke: bool=False) -> None:
    if output.exists():raise FileExistsError('Use a new independent-region diagnostic directory')
    settings=yaml.safe_load(config.read_text())
    if (set(settings)!={'subjects','data','geometry_source','beta','rule_retention','association_learning_rate','workers'}
            or not settings['subjects'] or len(set(settings['subjects']))!=len(settings['subjects'])
            or type(settings['workers']) is not int or not 1<=settings['workers']<=64):
        raise ValueError('Declare condition-2 subjects and at most 64 additional workers')
    for key in ('beta','rule_retention','association_learning_rate'):
        v=settings[key]
        if (not v or len(set(v))!=len(v) or not np.isfinite(v).all() or min(v)<0
                or (key=='rule_retention' and max(v)>1)):
            raise ValueError('Invalid declared parameter grid')
    output=output.resolve();bindings={str(config.resolve()):digest(config)}
    files=list((ROOT/'src/Bayesian_state/hypothesis_space').rglob('*.py'))
    files += [Path(__file__),ROOT/'src/Bayesian_state/model/region_catalogue_learner.py',
              ROOT/'src/Bayesian_state/model/modules/memory.py',ROOT/'src/Bayesian_state/model/modules/base_module.py',
              ROOT/'src/Bayesian_state/utils/numeric.py',ROOT/'src/Bayesian_state/evaluation/model_0923_h_validation.py',
              ROOT/'src/Bayesian_state/workflows/analysis/model_0923_h_validation.py']
    bindings.update({str(p.resolve()):digest(p) for p in files})
    subjects=settings['subjects'][:1] if smoke else settings['subjects']
    betas=settings['beta'][:1] if smoke else settings['beta']
    gammas=settings['rule_retention'][:2] if smoke else settings['rule_retention']
    rates=settings['association_learning_rate'][:2] if smoke else settings['association_learning_rate']
    meta={};geometries={};cache=ROOT/settings['geometry_source']
    for subject in subjects:
        folder=ROOT/settings['data']/f'subject_{subject}';prepared=load(folder/'prepared.json');meta[subject]=prepared
        if prepared['subject']!=subject or prepared['condition']!=2 or prepared['test_trials']<1:
            raise ValueError('This diagnostic supports binary-feedback condition 2 only')
        job=load(cache/'jobs'/f'geometry_S{subject}.json');record=load(cache/'geometry'/f'S{subject}.json')
        geometry=cache/'geometry'/f'S{subject}.npz'
        if (record['subject']!=subject or record['phase']!='geometry' or job['phase']!='geometry'
                or record['job_sha256']!=digest(cache/'jobs'/f'geometry_S{subject}.json')
                or record['arrays_sha256']!=digest(geometry) or job['length']!=prepared['training_trials']
                or Path(job['training']).resolve()!=folder/'training.npz'
                or digest(folder/'training.npz')!=prepared['array_sha256']['training.npz']):
            raise ValueError('Frozen geometry cache does not match the training task')
        verify(job['sources']);verify(job['inputs'])
        bindings.update(job['sources']);bindings.update(job['inputs'])
        for p in [folder/'prepared.json',folder/'training.npz',geometry,
                  cache/'jobs'/f'geometry_S{subject}.json',cache/'geometry'/f'S{subject}.json']:
            bindings[str(p.resolve())]=digest(p)
        with np.load(geometry) as z:
            assert z['geometry'].shape==(len(job['betas']),prepared['training_trials'],152,4)
            np.testing.assert_allclose(z['geometry'].sum(-1),1.,atol=1e-12,rtol=0)
        geometries[subject]=(geometry,job['betas'])
    verify(bindings);output.mkdir(parents=True)
    for file in files:
        target=output/'source'/file.resolve().relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,target)
    shutil.copy2(config,output/'config.yaml')
    write(output/'protocol.json',dict(status='running',settings=settings,smoke=smoke,bindings=bindings,
        actual_beta=betas,actual_gamma=gammas,actual_rates=rates,subjects=subjects,
        scope='Condition2,152 always-available geometries,physical stimulus,fixed beta,shared rule memory,no H or oral input',
        association='Independent region-response rows; no permutation-state learner executed',
        selection='Training pre-choice NLL; ties beta then gamma then rate; nominees saved before later file reads',
        resource_policy='At most64 additional numerical workers; existing cohort33,precision-or-oral14,recovery14,reports2 total<=127',
        python=sys.version,versions={name:importlib.metadata.version(name) for name in ['numpy','scipy','pandas','PyYAML']}))
    jobs=[]
    for subject in subjects:
        folder=ROOT/settings['data']/f'subject_{subject}';geometry,cache_betas=geometries[subject]
        for arm,learning_rates in [('FIXED',[None]),('ROW',rates)]:
            for bi,beta in enumerate(betas):
                for gi,gamma in enumerate(gammas):
                    for ri,rate in enumerate(learning_rates):
                        name=f'S{subject}_{arm}_B{bi}_G{gi}_R{ri}'
                        jobs.append(dict(name=name,subject=subject,arm=arm,beta=beta,gamma=gamma,rate=rate,
                            beta_index=cache_betas.index(beta),stage='training',geometry=str(geometry),
                            training=str(folder/'training.npz'),training_length=min(8,meta[subject]['training_trials']) if smoke else meta[subject]['training_trials'],
                            target=str(output/'training'/name),bindings=bindings))
    execute(jobs,output,1 if smoke else settings['workers'])
    records=[load(Path(j['target']).with_suffix('.json')) for j in jobs]
    scores=pd.DataFrame([{k:r[k] for k in ['subject','arm','beta','gamma','rate','trials','mean_nll']} for r in records])
    scores.to_csv(output/'training_scores.csv',index=False);nominees=[];evaluations=[]
    for (subject,arm),group in scores.groupby(['subject','arm'],sort=True):
        ordered=group.sort_values(['mean_nll','beta','gamma','rate'],na_position='first');best=ordered.iloc[0]
        rate=None if arm=='FIXED' else float(best.rate)
        selected=next(j for j in jobs if j['subject']==subject and j['arm']==arm and j['beta']==best.beta
                      and j['gamma']==best.gamma and j['rate']==rate)
        nominees.append(dict(subject=int(subject),arm=arm,beta=float(best.beta),gamma=float(best.gamma),rate=rate,
            mean_nll=float(best.mean_nll),ties=int(ordered.mean_nll.eq(best.mean_nll).sum()),actual_candidates=len(group)))
        later=ROOT/settings['data']/f'subject_{subject}'/'sealed_test.npz'
        evaluations.append(dict(selected,stage='evaluation',name=f'S{subject}_{arm}_evaluation',
            nominees=str(output/'nominees.json'),later=str(later),test_length=min(8,meta[subject]['test_trials']) if smoke else meta[subject]['test_trials'],
            training_prediction=str(Path(selected['target']).with_suffix('.npz')),target=str(output/'evaluation'/f'S{subject}_{arm}'),
            bindings={**bindings,str(later):meta[subject]['array_sha256']['sealed_test.npz']}))
    write(output/'nominees.json',dict(status='frozen_before_later_choices',nominees=nominees,
        protocol_sha256=digest(output/'protocol.json'),training_scores_sha256=digest(output/'training_scores.csv')))
    for job in evaluations:
        job['bindings'].update({str(output/'nominees.json'):digest(output/'nominees.json'),
                               job['training_prediction']:digest(Path(job['training_prediction']))})
    execute(evaluations,output,1 if smoke else settings['workers'])
    summary=pd.DataFrame([{k:r[k] for k in ['subject','arm','beta','gamma','rate','training_trials','test_trials',
        'training_nll','test_nll','mean_nll','exact_training_prefix']} for r in [load(Path(j['target']).with_suffix('.json')) for j in evaluations]])
    summary.to_csv(output/'summary.csv',index=False)
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><h1>独立区域类别关联</h1>'
        '<p>条件2开发对照：152个几何结构始终可用。FIXED保持固定类别；ROW从均匀关联学习。'
        '共享β、γ网格，ROW额外估计关联学习率。参数由训练选择确定；后段继续接收真实反馈。'
        '完整H搜索、容量、知觉噪声及自主生成另行检验。</p>'+summary.to_html(index=False))
    verify(bindings)
    write(output/'completion.json',dict(status='complete',smoke=smoke,training_points=len(jobs),evaluation_groups=len(evaluations),
        protocol_sha256=digest(output/'protocol.json'),nomination_sha256=digest(output/'nominees.json'),
        artifacts={p.name:digest(p) for p in [output/'training_scores.csv',output/'summary.csv',output/'index.html']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',type=Path);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.config is None or args.output is None:parser.error('--config and --output are required')
    else:run(args.config,args.output,args.smoke)
