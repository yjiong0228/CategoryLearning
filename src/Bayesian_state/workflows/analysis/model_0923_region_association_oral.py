"""Evaluate frozen independent-region bases against unchanged oral reasons.

Use --config YAML --output NEW_DIRECTORY [--smoke]. First project saved
behavioral states, using the maintained geometric component kernel. Then read
the existing two oral codings and their frozen constraint labels. Conditional
basis mass is w*q*A(choice)*rho / p(choice); pre-choice mass is w*q*rho.
Complete conjunctions, parsed fragments and unresolved reports stay distinct.
The pre-choice quantity includes stimulus-region uncertainty and is different
from the historical response-selected-region prior used with fixed labels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from .model_0923_region_association_diagnostic import ROOT,verify
from .model_0923_h_validation import write
from .model_0923_decision_basis_alignment import align_identities
from ...evaluation.model_0923_h_validation import load,digest
from ...evaluation.oral.region_basis import region_basis
from ...evaluation.oral.stimulus_controls import component_kernel
from ...evaluation.oral.decision_basis_scoring import label_masses,DOMAINS,LABELS
from ...hypothesis_space.spaces.structural_0923 import build_relational_0923_space
from ...hypothesis_space.geometry.boundary import BoundaryGeometry

MODULE='src.Bayesian_state.workflows.analysis.model_0923_region_association_oral'


def tokens_key(raw: str) -> tuple[str,...]:
    values=json.loads(raw)
    if not isinstance(values,list) or any(not isinstance(v,str) for v in values):raise ValueError('Expected report token list')
    return tuple(sorted(set(values)))


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep report scopes, chronology, feedback strata and both basis timings."""
    rows=[]
    for (reading,subject,arm),g in frame.groupby(['reading','iSub','arm'],sort=True):
        cut=int(g.training_trials.iloc[0]);ordinal=g.state_trial.to_numpy();n=len(g)
        windows={'whole':ordinal>=0,'training':ordinal<cut,'test':ordinal>=cut,'first64':ordinal<64,'last64':ordinal>=max(0,n-64)}
        strata={'all':np.ones(n,bool),'correct':g.feedback.eq(1).to_numpy(),'incorrect':g.feedback.eq(0).to_numpy()}
        for window,window_mask in windows.items():
            for scope in ('complete','parsed_fragment'):
                for stratum,feedback_mask in strata.items():
                    selected=g.loc[window_mask&feedback_mask&g.scope.eq(scope).to_numpy()]
                    if selected.empty:continue
                    for domain in DOMAINS:
                        row=dict(reading=reading,subject=int(subject),arm=arm,window=window,scope=scope,
                            feedback_stratum=stratum,reports=len(selected),domain=domain)
                        for timing in ('conditional','prechoice'):
                            cols=[timing+'_'+domain+'_'+label for label in LABELS]
                            if not np.isfinite(selected[cols]).all().all():raise ValueError('Scorable reports require all verdict masses')
                            np.testing.assert_allclose(selected[cols].sum(axis=1),1.,atol=1e-12,rtol=0)
                            row.update({timing+'_'+label:float(selected[col].mean()) for label,col in zip(LABELS,cols)})
                        rows.append(row)
    return pd.DataFrame(rows)


def worker(file: Path) -> None:
    job=load(file);verify(job['bindings']);space=build_relational_0923_space();target=Path(job['target'])
    counts=np.array([[len(c.components) for c in h.categories] for h in space])
    pieces=[]
    for name in ('training.npz','sealed_test.npz'):
        with np.load(Path(job['folder'])/name) as z:pieces.append({k:z[k] for k in ('stimulus','choices','feedback')})
    data={k:np.concatenate([p[k] for p in pieces])[:job['trials']] for k in pieces[0]}
    if job['stage']=='kernel':
        arrays=dict(components=component_kernel(space,data['stimulus'][:,None,:],np.full(job['trials'],job['region']+1),BoundaryGeometry(space)),component_counts=counts)
    elif job['stage']=='states':
        with np.load(job['geometry']) as z:q=z['geometry'][:job['trials']]
        components=[]
        for p in job['kernels']:
            with np.load(p) as z:components.append(z['components']);np.testing.assert_array_equal(z['component_counts'],counts)
        with np.load(job['reference']) as z:
            arrays=region_basis(geometry=q,rule_weights=z['rule_weights'][:job['trials']],
                associations=z['associations'][:job['trials']],components=np.stack(components,axis=2),choices=data['choices'])
            np.testing.assert_allclose(arrays['prediction'],z['probabilities'][:job['trials']],atol=1e-12,rtol=0)
            np.testing.assert_allclose(arrays['choice_probability'],z['observed_probability'][:job['trials']],atol=1e-12,rtol=0)
        arrays.update(choices=data['choices'],feedback=data['feedback'],component_counts=counts)
    else:raise ValueError('Unknown stage')
    verify(job['bindings']);target.parent.mkdir(parents=True,exist_ok=True)
    with target.with_suffix('.npz').open('xb') as handle:np.savez_compressed(handle,**arrays)
    write(target.with_suffix('.json'),dict(status='complete',stage=job['stage'],subject=job['subject'],trials=job['trials'],
        arrays_sha256=digest(target.with_suffix('.npz')),job_sha256=digest(file),catalogue_signature=space.signature))


def execute(jobs: list[dict], output: Path, workers: int) -> None:
    (output/'jobs').mkdir(exist_ok=True);workers=min(workers,len(jobs),len(os.sched_getaffinity(0)))
    def launch(job):
        file=output/'jobs'/(job['name']+'.json');write(file,job)
        with tempfile.TemporaryDirectory(prefix='region_oral_') as cache:
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
                MPLCONFIGDIR=cache,NUMBA_CACHE_DIR=cache,PYTHONDONTWRITEBYTECODE='1')
            with file.with_suffix('.log').open('x') as log:
                subprocess.run([sys.executable,'-m',MODULE,'--worker',str(file)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'{jobs[0]["stage"]}: {len(jobs)} jobs, {workers} workers',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,future in enumerate(as_completed([pool.submit(launch,j) for j in jobs]),1):
            future.result();print(f'{i}/{len(jobs)} complete',flush=True)


def run(config: Path, output: Path, smoke: bool=False) -> None:
    settings=yaml.safe_load(config.read_text());fit=ROOT/settings['fit'];autonomous=ROOT/settings['autonomous']
    if (set(settings)!={'fit','autonomous','readings','workers'} or type(settings['workers']) is not int
            or not 1<=settings['workers']<=64 or not settings['readings']):raise ValueError('Declare inputs and at most64 additional workers')
    if output.exists():raise FileExistsError('Use a new oral evaluation directory')
    fd,fp=load(fit/'completion.json'),load(fit/'protocol.json');ad,ap=load(autonomous/'completion.json'),load(autonomous/'protocol.json')
    if (fd['status']!='complete' or ad['status']!='complete' or fd['smoke'] or ad['smoke']
            or fd['protocol_sha256']!=digest(fit/'protocol.json') or ad['protocol_sha256']!=digest(autonomous/'protocol.json')
            or fd['nomination_sha256']!=digest(fit/'nominees.json') or ap['parent_nomination_sha256']!=fd['nomination_sha256']):
        raise ValueError('Require matching completed frozen behavior and autonomous protocols')
    bindings=dict(fp['bindings']);bindings[str(config.resolve())]=digest(config)
    for p in [fit/'completion.json',fit/'protocol.json',fit/'nominees.json',autonomous/'completion.json',autonomous/'protocol.json']:
        bindings[str(p)]=digest(p)
    files=[Path(__file__),ROOT/'src/Bayesian_state/evaluation/oral/region_basis.py',ROOT/'src/Bayesian_state/evaluation/oral/stimulus_controls.py',
        ROOT/'src/Bayesian_state/evaluation/oral/decision_basis_scoring.py',ROOT/'src/Bayesian_state/workflows/analysis/model_0923_decision_basis_alignment.py',
        ROOT/'src/Bayesian_state/evaluation/oral/constraint_alignment.py',ROOT/'src/Bayesian_state/evaluation/oral/structure_0923.py',
        ROOT/'src/Bayesian_state/model/modules/readout/report_space.py',ROOT/'src/oral_coding.py']
    for p in files:bindings[str(p.resolve())]=digest(p)
    nominees=load(fit/'nominees.json')['nominees']
    if smoke:nominees=[n for n in nominees if n['subject']==nominees[0]['subject']]
    meta={};folders={};geometries={}
    for s in sorted({n['subject'] for n in nominees}):
        folder=ROOT/fp['settings']['data']/f'subject_{s}';folders[s]=folder;meta[s]=load(folder/'prepared.json')
        if meta[s]['condition']!=2:raise ValueError('Condition2 only')
        for name in ('training.npz','sealed_test.npz'):bindings[str(folder/name)]=meta[s]['array_sha256'][name]
    for file in (autonomous/'jobs').glob('geometry*.json'):
        job=load(file);target=Path(job['target']).with_suffix('.npz');rec=load(target.with_suffix('.json'))
        if rec['job_sha256']!=digest(file):raise ValueError('Geometry provenance changed')
        geometries[job['subject'],job['beta']]=target;bindings[str(target)]=rec['arrays_sha256']
    for n in nominees:
        p=fit/'evaluation'/f'S{n["subject"]}_{n["arm"]}.npz';rec=load(p.with_suffix('.json'))
        if any(rec[k]!=n[k] for k in ('subject','arm','beta','gamma','rate')):raise ValueError('State parameters differ from frozen nominee')
        bindings[str(p)]=rec['array_sha256'];bindings[str(p.with_suffix('.json'))]=digest(p.with_suffix('.json'))
    verify(bindings);output=output.resolve();output.mkdir(parents=True)
    for p in {Path(k).resolve() for k in bindings if k.endswith('.py')}:
        target=output/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    shutil.copy2(config,output/'config.yaml')
    write(output/'protocol.json',dict(settings=settings,smoke=smoke,bindings=bindings,nominees=nominees,
        behavior_nomination_sha256=fd['nomination_sha256'],versions=fp['versions'],python=sys.version,
        timing='Behavior-only bases completed before oral tables; before current feedback; conditional on actual choice and prechoice w*q*rho both retained',
        scope='Complete conjunction and parsed fragment separated; two fixed readings; all reports and feedback strata retained; no oral parameter selection'))
    kernels=[];lookup={}
    for s in meta:
        lookup[s]=[]
        for g in range(4):
            target=output/'kernels'/f'S{s}_G{g}';lookup[s].append(str(target.with_suffix('.npz')))
            kernels.append(dict(name=f'kernel_S{s}_G{g}',stage='kernel',subject=s,region=g,folder=str(folders[s]),
                trials=min(8,meta[s]['trials']) if smoke else meta[s]['trials'],target=str(target),bindings=bindings))
    execute(kernels,output,1 if smoke else settings['workers'])
    jobs=[]
    for n in nominees:
        s=n['subject'];name=f'S{s}_{n["arm"]}'
        jobs.append(dict(n,name=name,stage='states',folder=str(folders[s]),kernels=lookup[s],geometry=str(geometries[s,n['beta']]),
            reference=str(fit/'evaluation'/(name+'.npz')),trials=min(8,meta[s]['trials']) if smoke else meta[s]['trials'],
            bindings={**bindings,**{p:digest(Path(p)) for p in lookup[s]}},target=str(output/'states'/name)))
    execute(jobs,output,1 if smoke else settings['workers'])
    for job in jobs:
        for suffix in ('.npz','.json'):p=Path(job['target']).with_suffix(suffix);bindings[str(p)]=digest(p)
    write(output/'behavior_states_complete.json',dict(status='complete',groups=len(jobs),oral_tables_loaded=False,bindings=bindings))

    # Only frozen semantic labels are reused from the historical diagnostic;
    # its association learner and parameter estimates are never executed.
    frames=[];cache_records=[]
    for reading,paths in settings['readings'].items():
        audit=ROOT/paths['audit'];cache=ROOT/paths['queries'];am=load(audit/'manifest.json');cm=load(cache/'completion.json');cp=load(cache/'protocol.json')
        if am['status']!='complete' or cm['status']!='complete' or cm['smoke'] or digest(cache/'protocol.json')!=cm['protocol_sha256']:
            raise ValueError('Need completed immutable oral audit and query cache')
        semantic_sources={p:h for p,h in cp['source_sha256'].items() if '/hypothesis_space/' in p
            or p.endswith(('/constraint_alignment.py','/structure_0923.py','/report_space.py','/oral_coding.py'))}
        verify(semantic_sources);bindings.update(semantic_sources)
        for p in [audit/'manifest.json',cache/'completion.json',cache/'protocol.json']:bindings[str(p)]=digest(p)
        bindings[str(audit/'trials.csv')]=am['artifacts']['trials.csv'];bindings[str(cache/'queries.csv')]=cm['artifacts']['queries.csv']
        verify(bindings);oral=pd.read_csv(audit/'trials.csv');queries=pd.read_csv(cache/'queries.csv')
        query_lookup={(tokens_key(r.tokens),int(r.region)):str(r.query_id) for r in queries.itertuples()}
        if len(query_lookup)!=len(queries):raise ValueError('Duplicate semantic query key')
        aligned={};labels_cache={}
        for job in jobs:
            s=job['subject'];target=Path(job['target'])
            if s not in aligned:aligned[s]=align_identities(oral,job,job['trials'])
            frame,cut=aligned[s];frame=frame.copy()
            with np.load(target.with_suffix('.npz')) as z:state={k:z[k] for k in z.files}
            np.testing.assert_array_equal(frame.choice,state['choices']);counts=state['component_counts']
            shape=state['prechoice_component_mass'].shape;labels=np.full((*shape,2),-1,dtype=np.int8)
            for t,row in enumerate(frame.itertuples()):
                if pd.isna(row.query_id):continue
                token=tokens_key(row.tokens)
                for g in range(4):
                    key=(token,g)
                    if key not in labels_cache:
                        code=query_lookup[key];p=cache/'queries'/(code+'.npz');rec=load(p.with_suffix('.json'));qjob=cache/'jobs'/('Q_'+code+'.json')
                        if (rec['job_sha256']!=digest(qjob) or tokens_key(json.dumps(rec['tokens']))!=token or rec['region']!=g):
                            raise ValueError('Cached query identity changed')
                        bindings[str(p)]=rec['arrays_sha256'];bindings[str(p.with_suffix('.json'))]=digest(p.with_suffix('.json'));bindings[str(qjob)]=digest(qjob)
                        verify({str(p):rec['arrays_sha256']})
                        with np.load(p) as z:lab=z['labels'].copy()
                        assert lab.shape==(len(counts),int(counts.max()),2)
                        valid=np.arange(int(counts.max()))[None,:]<counts[:,g,None]
                        assert np.isin(lab[valid],[0,1,2]).all() and (lab[~valid]==-1).all()
                        labels_cache[key]=lab;cache_records.append(dict(reading=reading,query_id=code,region=g,tokens=json.dumps(token),array_sha256=rec['arrays_sha256']))
                    labels[t,:,g]=labels_cache[key]
            reshape=(len(frame),counts.shape[0]*4,int(counts.max()))
            before=label_masses(state['prechoice_component_mass'].reshape(reshape),labels.reshape(*reshape,2))
            after=label_masses(state['choice_component_joint_mass'].reshape(reshape),labels.reshape(*reshape,2))/state['choice_probability'][:,None,None]
            frame['reading']=reading;frame['arm']=job['arm'];frame['training_trials']=cut;frame['phase']=np.where(frame.state_trial<cut,'training','test')
            frame['feedback']=state['feedback'];frame['choice_probability']=state['choice_probability']
            for d,domain in enumerate(DOMAINS):
                for k,label in enumerate(LABELS):
                    frame['conditional_'+domain+'_'+label]=after[:,d,k];frame['prechoice_'+domain+'_'+label]=before[:,d,k]
            frames.append(frame)
    trials=pd.concat(frames,ignore_index=True);summary=summarize(trials)
    keys=['reading','subject','window','scope','feedback_stratum','domain']
    a=summary[summary.arm=='ROW'].set_index(keys);b=summary[summary.arm=='FIXED'].set_index(keys)
    assert a.index.equals(b.index);np.testing.assert_array_equal(a.reports,b.reports)
    contrasts=a[['reports']].copy()
    for name in [timing+'_'+label for timing in ('conditional','prechoice') for label in LABELS]:contrasts['delta_'+name]=a[name]-b[name]
    contrasts=contrasts.reset_index()
    group_keys=['reading','iSub','arm','phase','scope','choice','text']
    groups=trials.groupby(group_keys,dropna=False,sort=True).agg(reports=('state_trial','size'),correct=('feedback','sum'),
        conditional_support=('conditional_functional_support','mean'),prechoice_support=('prechoice_functional_support','mean')).reset_index()
    for name,frame in [('trial_scores',trials),('summary',summary),('contrasts',contrasts),('reason_groups',groups),('queries',pd.DataFrame(cache_records))]:
        frame.to_csv(output/(name+'.csv'),index=False)
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><h1>独立区域关联与口述理由</h1>'
        '<p>行为参数和反馈前状态均已冻结。沿用两种固定读法及完整报告／片段范围，比较实际理由与当前局部几何依据的相容性。'
        'conditional给定本题实际选择；prechoice包含刺激区域概率，尚未按本题选择加权。后者与历史固定类别的prior指标定义不同。'
        '这是固定目录开发检验，H搜索和容量尚待接入。</p>'+contrasts[(contrasts.window.isin(['training','test']))&(contrasts.scope=='complete')&(contrasts.feedback_stratum=='all')&(contrasts.domain=='functional')].to_html(index=False))
    verify(bindings)
    write(output/'completion.json',dict(status='complete',smoke=smoke,groups=len(jobs),readings=len(settings['readings']),trials=len(trials),
        reused_semantic_queries=len(cache_records),protocol_sha256=digest(output/'protocol.json'),
        behavior_states_sha256=digest(output/'behavior_states_complete.json'),bindings=bindings,
        artifacts={p.name:digest(p) for p in [output/(n+'.csv') for n in ['trial_scores','summary','contrasts','reason_groups','queries']]+[output/'index.html']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path);parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--worker',type=Path);args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.config is None or args.output is None:parser.error('--config and --output required')
    else:run(args.config,args.output,args.smoke)
