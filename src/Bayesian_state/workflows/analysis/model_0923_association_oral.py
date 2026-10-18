"""Replay frozen association states, then score unchanged oral constraints.

Use --config CONFIG --output NEW_DIRECTORY, optionally --smoke. Behavior-only
state files are completed and checked against the original choice predictions
before oral tables are loaded. Every report conjunction is then checked against
all four geometrical regions using the existing component constraint method.
Overlapping original component labels must match exactly. Association, rule and
component weights are never selected using the oral report.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
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

from . import model_0923_association_diagnostic as diagnostic
from .model_0923_episodic_adaptive_task import SharedWorkerBudget
from .model_0923_decision_basis_alignment import align_identities
from .model_0923_oral_constraint_audit import audit_query
from ...model.association_learner import FixedCatalogueAssociationLearner
from ...evaluation.oral.association_basis import AssociatedDecisionBasis
from ...evaluation.oral.decision_basis_scoring import DOMAINS, LABELS, label_masses
from ...hypothesis_space.observation_model.continuous_partition import ContinuousPartition
from ...hypothesis_space.spaces.structural_0923 import RELATIONAL_EXTENSION

ROOT = diagnostic.ROOT
MODULE = 'src.Bayesian_state.workflows.analysis.model_0923_association_oral'
read, write, digest, verify = diagnostic.read, diagnostic.write, diagnostic.digest, diagnostic.verify


def content_key(raw: str) -> tuple[str, ...]:
    values = json.loads(raw)
    if not isinstance(values,list) or any(not isinstance(x,str) for x in values):
        raise ValueError('Report tokens must be a JSON list of strings.')
    return tuple(sorted(set(values)))


def query_id(tokens: tuple[str,...], region: int) -> str:
    if not tokens or region not in range(4): raise ValueError('Need content and a geometrical region.')
    return hashlib.sha256(json.dumps(dict(tokens=tokens,region=region),sort_keys=True).encode()).hexdigest()[:20]


def component_labels(rows: list[dict], counts: np.ndarray, region: int) -> np.ndarray:
    """Require one label for every real component; padding alone remains -1."""
    counts = np.asarray(counts)
    if (counts.ndim!=2 or counts.shape[1]!=4 or region not in range(4)
            or np.any(counts<1) or np.any(counts!=counts.astype(int))):
        raise ValueError('Need four positive component counts per rule.')
    expected={(h,m) for h in range(len(counts)) for m in range(int(counts[h,region]))}
    keys=[(r['rule'],r['component']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:
        raise ValueError('Missing, duplicated or unexpected catalogue components.')
    labels=np.full((len(counts),int(counts.max()),2),-1,dtype=np.int8)
    for row in rows:
        for d,name in enumerate(DOMAINS):
            if row[name] not in LABELS: raise ValueError('Unknown constraint verdict.')
            labels[row['rule'],row['component'],d]=LABELS.index(row[name])
    return labels


def worker(file: Path) -> None:
    job=read(file);verify(job['sources']);verify(job['inputs']);target=Path(job['target'])
    if job['phase']=='states':
        folder=Path(job['folder']);parts=[]
        for name in ('training.npz','sealed_test.npz'):
            with np.load(folder/name) as z:parts.append({k:z[k].copy() for k in ('stimulus','choices','feedback')})
        data={k:np.concatenate([p[k] for p in parts])[:job['trials']] for k in parts[0]}
        with np.load(job['geometry']) as z:q=z['geometry'][:job['trials']]
        partition=ContinuousPartition(4,4,structural_extension=RELATIONAL_EXTENSION)
        basis=AssociatedDecisionBasis(partition.hypothesis_space,geometry=partition.boundary_geometry)
        arm=job['arm'];learner=FixedCatalogueAssociationLearner(len(partition.hypothesis_space),
            rule_retention=job['gamma'],association_retention=job['gamma'] if arm['association_retention']=='shared' else 1.,
            learn_associations=arm['learn_associations'],hierarchical=job['condition']==3)
        records=[]
        for t in range(len(q)):
            row=basis(probabilities=q[t],rule_weights=learner.rule_weights,associations=learner.association_weights,
                      stimulus=data['stimulus'][t],choice=int(data['choices'][t]))
            np.testing.assert_allclose(row['prediction'],learner.predict(q[t]),atol=1e-14,rtol=0)
            records.append(row)
            learner.update(q[t],int(data['choices'][t]),float(data['feedback'][t]))
            if (t+1)%128==0:print(f'{t+1}/{len(q)} behavior states',flush=True)
        arrays={k:np.stack([r[k] for r in records]) for k in records[0]}
        arrays.update(component_counts=basis.counts,choices=data['choices'])
        with np.load(job['prediction_reference']) as z:reference=z['probabilities'][:len(q)]
        residual=float(np.max(np.abs(arrays['prediction']-reference)))
        np.testing.assert_allclose(arrays['prediction'],reference,atol=job['replay_tolerance'],rtol=0)
        details=dict(subject=job['subject'],arm_name=job['arm_name'],trials=len(q),
            catalogue_version=partition.hypothesis_space.version,catalogue_signature=partition.hypothesis_space.signature,
            max_prediction_error=residual,tolerance=job['replay_tolerance'],verified_behavior_replay=True)
    elif job['phase']=='query':
        summary,rows=audit_query(dict(query_id=job['query_id'],condition=2,choice=job['region']+1,
                                     tokens=job['tokens'],tolerance=job['tolerance']))
        arrays=dict(labels=component_labels(rows,np.asarray(job['counts']),job['region']))
        details=dict(query_id=job['query_id'],tokens=job['tokens'],region=job['region'],summary=summary)
    else:raise ValueError('Unknown association-oral phase.')
    verify(job['sources']);verify(job['inputs'])
    sha=diagnostic.save_arrays(target.with_suffix('.npz'),**arrays)
    write(target.with_suffix('.json'),dict(details,arrays_sha256=sha,job_sha256=digest(file),phase=job['phase']))


def execute(jobs: list[dict], output: Path, settings: dict, smoke: bool) -> None:
    if not jobs:return
    budget=min(1 if smoke else settings['parallel_budget'],len(os.sched_getaffinity(0)))
    slots=SharedWorkerBudget(budget,None if smoke else ROOT/settings['resource_predecessor'])
    def launch(job: dict) -> None:
        file=output/'jobs'/(job['name']+'.json');write(file,job)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        with slots.slot(),tempfile.TemporaryDirectory(prefix='association_oral_') as cache,file.with_suffix('.log').open('x') as log:
            env['MPLCONFIGDIR']=cache
            result=subprocess.run([sys.executable,'-m',MODULE,'--worker',str(file)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Worker failed: {file.with_suffix(".log")}')
    print(f'{jobs[0]["phase"]}: {len(jobs)} jobs; up to {min(budget,len(jobs))} workers',flush=True)
    with ThreadPoolExecutor(max_workers=min(budget,len(jobs))) as pool:
        for n,future in enumerate(as_completed([pool.submit(launch,j) for j in jobs]),1):
            future.result()
            if n%32==0 or n==len(jobs):print(f'{n}/{len(jobs)} complete',flush=True)


def run(config: Path, output: Path, smoke: bool) -> None:
    settings=yaml.safe_load(config.read_text())
    if (type(settings['parallel_budget']) is not int or not 1<=settings['parallel_budget']<=128
            or not np.isfinite(settings['replay_tolerance']) or not 0<settings['replay_tolerance']<=1e-12):
        raise ValueError('Require 1..128 workers and a declared replay tolerance at most 1e-12.')
    if output.exists():raise FileExistsError('Use a new associated oral evaluation directory.')
    output=output.resolve();source=ROOT/settings['source'];done=read(source/'completion.json');pro=read(source/'protocol.json')
    if done['status']!='complete' or done['smoke'] or digest(source/'protocol.json')!=done['protocol_sha256']:
        raise ValueError('Need complete scientific association generation.')
    fit=ROOT/pro['settings']['source'];fit_pro=read(fit/'protocol.json');fit_done=read(fit/'completion.json')
    if (digest(fit/'nominees.json')!=pro['parent_nomination_sha256']
            or fit_done['nomination_sha256']!=pro['parent_nomination_sha256']):
        raise ValueError('Frozen behavioral nominees differ.')
    verify(pro['source_sha256']);verify(pro['input_sha256'])
    sources=dict(pro['source_sha256'])
    for path in [Path(__file__),*(ROOT/n for n in (
        'src/Bayesian_state/evaluation/oral/association_basis.py',
        'src/Bayesian_state/evaluation/oral/decision_basis.py',
        'src/Bayesian_state/evaluation/oral/decision_basis_scoring.py',
        'src/Bayesian_state/evaluation/oral/constraint_alignment.py',
        'src/Bayesian_state/workflows/analysis/model_0923_decision_basis_alignment.py',
        'src/Bayesian_state/workflows/analysis/model_0923_oral_constraint_audit.py',
        'src/Bayesian_state/evaluation/oral/structure_0923.py',
        'src/Bayesian_state/model/modules/readout/report_space.py','src/oral_coding.py'))]:
        sources[str(path.relative_to(ROOT))]=digest(path)
    inputs={str(f):digest(f) for f in (source/'completion.json',source/'protocol.json',fit/'nominees.json',fit/'protocol.json')}
    nominees=pro['nominees']
    if smoke:nominees=[n for n in nominees if n['subject']==nominees[0]['subject']]
    data=ROOT/fit_pro['settings']['data'];geometry={}
    for file in sorted((source/'jobs').glob('geometry*.json')):
        job=read(file);target=Path(job['target']).with_suffix('.npz');record=read(target.with_suffix('.json'))
        if digest(target)!=record['arrays_sha256'] or record['job_sha256']!=digest(file):
            raise ValueError('Geometry cache changed.')
        geometry[job['subject'],job['beta']]=target
    jobs=[]
    for n in nominees:
        s=n['subject'];folder=data/f'subject_{s}';meta=read(folder/'prepared.json');files={str(folder/'prepared.json'):digest(folder/'prepared.json')}
        for name,sha in meta['array_sha256'].items():files[str(folder/name)]=sha
        reference=fit/'evaluation'/f'S{s}_{n["arm_name"]}.npz';r=read(reference.with_suffix('.json'))
        if r['beta']!=n['beta'] or r['gamma']!=n['gamma']:raise ValueError('Behavior reference differs from nominees.')
        files.update({str(reference):r['arrays_sha256'],str(reference.with_suffix('.json')):digest(reference.with_suffix('.json')),
                      str(geometry[s,n['beta']]):digest(geometry[s,n['beta']])})
        verify(files);inputs.update(files)
        jobs.append(dict(n,name=f'S{s}_{n["arm_name"]}',phase='states',folder=str(folder),condition=meta['condition'],
            trials=min(8,meta['trials']) if smoke else meta['trials'],geometry=str(geometry[s,n['beta']]),
            arm=fit_pro['settings']['arms'][n['arm_name']],prediction_reference=str(reference),
            replay_tolerance=settings['replay_tolerance'],inputs=files,sources=sources,target=str(output/'states'/f'S{s}_{n["arm_name"]}')))
    output.mkdir(parents=True)
    for name in sources:
        dest=output/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    shutil.copy2(config,output/'config.yaml')
    write(output/'protocol.json',dict(settings=settings,smoke=smoke,source_sha256=sources,behavior_input_sha256=inputs,
        nominees=nominees,behavior_nomination_sha256=pro['parent_nomination_sha256'],source_versions=pro['versions'],python=sys.version,
        boundary='Behavior state replay and prediction verification complete before loading oral tables; no parameter selection or state update from reports'))
    execute(jobs,output,settings,smoke)
    states={}
    for job in jobs:
        target=Path(job['target']);states[target.name]=read(target.with_suffix('.json'))
        for file in (target.with_suffix('.json'),target.with_suffix('.npz')):inputs[str(file)]=digest(file)
    write(output/'behavior_states_complete.json',dict(status='complete',groups=len(states),states=states,
        protocol_sha256=digest(output/'protocol.json'),oral_tables_loaded=False))

    # The report stage begins only after all behavioral states are frozen.
    audit=ROOT/settings['oral_audit'];audit_manifest=read(audit/'manifest.json')
    if audit_manifest['status']!='complete':raise ValueError('Need a complete oral audit.')
    for name,sha in audit_manifest['artifacts'].items():
        if digest(audit/name)!=sha:raise ValueError('Frozen oral artifact changed.')
        inputs[str(audit/name)]=sha
    inputs[str(audit/'manifest.json')]=digest(audit/'manifest.json')
    oral=pd.read_csv(audit/'trials.csv');old_queries=pd.read_csv(audit/'constraint_queries.csv');old_matches=pd.read_csv(audit/'component_matches.csv')
    trials={};cutoffs={}
    for job in jobs:
        s=job['subject']
        if s not in trials:trials[s],cutoffs[s]=align_identities(oral,job,job['trials'])
    with np.load(Path(jobs[0]['target']).with_suffix('.npz')) as z:counts=z['component_counts']
    needed=sorted({(content_key(row.tokens),g) for frame in trials.values()
                   for row in frame.loc[frame.query_id.notna()].itertuples() for g in range(4)})
    query_jobs=[]
    for tokens,region in needed:
        key=query_id(tokens,region)
        query_jobs.append(dict(name='Q_'+key,phase='query',query_id=key,tokens=list(tokens),region=region,
            tolerance=audit_manifest['config']['tolerance'],counts=counts.tolist(),sources=sources,
            inputs={str(audit/'manifest.json'):inputs[str(audit/'manifest.json')]},target=str(output/'queries'/key)))
    execute(query_jobs,output,settings,smoke)
    lookup={};query_rows=[]
    for j in query_jobs:
        f=Path(j['target']);r=read(f.with_suffix('.json'))
        if digest(f.with_suffix('.npz'))!=r['arrays_sha256']:raise ValueError('Query array changed.')
        with np.load(f.with_suffix('.npz')) as z:lookup[tuple(j['tokens']),j['region']]=z['labels'].copy()
        query_rows.append(dict(query_id=j['query_id'],tokens=json.dumps(j['tokens']),region=j['region']))
        for file in (f.with_suffix('.json'),f.with_suffix('.npz')):inputs[str(file)]=digest(file)
    overlap=set();comparison_rows=[]
    for row in old_queries.loc[old_queries.condition.isin([2,3])].itertuples():
        key=content_key(row.tokens),int(row.choice)-1
        if key not in lookup:continue
        old=component_labels(old_matches.loc[old_matches.query_id.eq(row.query_id)].to_dict('records'),counts,key[1])
        np.testing.assert_array_equal(lookup[key],old);overlap.add(key)
        comparison_rows.append(dict(original_query_id=row.query_id,new_query_id=query_id(*key),identical=True))
    pd.DataFrame(query_rows).to_csv(output/'queries.csv',index=False)
    pd.DataFrame(comparison_rows).to_csv(output/'original_query_checks.csv',index=False)
    write(output/'query_validation.json',dict(total=len(needed),overlapping_content_regions=len(overlap),new_content_regions=len(needed)-len(overlap),
        original_query_checks=len(comparison_rows),all_original_labels_identical=True,
        encoding='existing primary-audit tokens and complete/fragment masks; no re-encoding',
        method='same component constraint evaluator; all four geometrical regions; no oral selection of association or component'))
    results=[];summaries=[]
    for job in jobs:
        s=job['subject'];frame=trials[s].copy();cut=cutoffs[s];target=Path(job['target'])
        with np.load(target.with_suffix('.npz')) as z:a={k:z[k].copy() for k in z.files}
        np.testing.assert_array_equal(a['choices'],frame.choice);np.testing.assert_array_equal(a['component_counts'],counts)
        labels=np.full((*a['prior_component_mass'].shape,2),-1,dtype=np.int8)
        for t,row in enumerate(frame.itertuples()):
            if pd.notna(row.query_id):
                for g in range(4):labels[t,:,g]=lookup[content_key(row.tokens),g]
        shape=(len(frame),counts.shape[0]*4,int(counts.max()))
        prior=label_masses(a['prior_component_mass'].reshape(shape),labels.reshape(*shape,2))
        joint=label_masses(a['choice_component_joint_mass'].reshape(shape),labels.reshape(*shape,2))
        scored=joint/a['choice_probability'][:,None,None]
        frame['arm_name']=job['arm_name'];frame['training_trials']=cut
        frame['phase']=np.where(np.arange(len(frame))<cut,'training','test');frame['choice_probability']=a['choice_probability']
        for d,domain in enumerate(DOMAINS):
            for k,label in enumerate(LABELS):
                frame[domain+'_'+label]=scored[:,d,k];frame['prior_'+domain+'_'+label]=prior[:,d,k]
        results.append(frame);ordinal=np.arange(len(frame))
        windows=dict(whole=ordinal>=0,training=ordinal<cut,test=ordinal>=cut,first64=ordinal<64,last64=ordinal>=max(0,len(frame)-64))
        for window,in_window in windows.items():
            for scope in ('complete','parsed_fragment'):
                mask=frame.scope.eq(scope).to_numpy()&in_window
                if not mask.any():continue
                if not np.isfinite(scored[mask]).all():raise ValueError('Scorable reports lack a full component label set.')
                for d,domain in enumerate(DOMAINS):
                    values=scored[mask,d].mean(axis=0);prior_values=prior[mask,d].mean(axis=0)
                    summaries.append(dict(subject=s,arm_name=job['arm_name'],window=window,scope=scope,domain=domain,reports=int(mask.sum()),
                        **{label:float(values[k]) for k,label in enumerate(LABELS)},
                        **{'prior_'+label:float(prior_values[k]) for k,label in enumerate(LABELS)}))
    result=pd.concat(results,ignore_index=True);summary=pd.DataFrame(summaries)
    result.to_csv(output/'trial_scores.csv',index=False);summary.to_csv(output/'summary.csv',index=False)
    contrasts=[]
    for row in summary.loc[summary.arm_name.str.startswith('LEARN_')].itertuples():
        reference='FIXED_'+row.arm_name.split('_',1)[1]
        baseline=summary.loc[summary.subject.eq(row.subject)&summary.arm_name.eq(reference)&summary.window.eq(row.window)
                             &summary.scope.eq(row.scope)&summary.domain.eq(row.domain)]
        if len(baseline)!=1 or baseline.iloc[0].reports!=row.reports:raise ValueError('Unmatched oral comparison masks.')
        contrasts.append(dict(subject=row.subject,arm_name=row.arm_name,reference=reference,window=row.window,scope=row.scope,domain=row.domain,
                              reports=row.reports,delta_support=row.support-float(baseline.iloc[0].support)))
    pd.DataFrame(contrasts).to_csv(output/'contrasts.csv',index=False)
    primary=summary.loc[summary.scope.eq('complete')&summary.domain.eq('structural')]
    (output/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>类别关联的口述表征检验</title>'
        '<h1>行为拟合后检验类别关联的表征内容</h1><p>先核对行为状态重放，再评价固定口述约束。'
        '所有关联、规则和局部分量按联合质量汇总，随后条件于实际选择。完整报告与解析片段分别保留。'
        '支持率描述表征内容的对应程度；口述未用于参数估计或状态更新。当前为开发样本的固定目录诊断。</p>'
        +primary.to_html(index=False)+'</html>')
    verify(sources);verify(inputs)
    write(output/'completion.json',dict(status='complete',smoke=smoke,groups=len(jobs),trials=len(result),
        unique_queries=len(needed),overlapping_queries=len(overlap),protocol_sha256=digest(output/'protocol.json'),
        behavior_states_sha256=digest(output/'behavior_states_complete.json'),input_sha256=inputs,
        artifacts={f.name:digest(f) for f in [*output.glob('*.csv'),output/'index.html',output/'query_validation.json']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker is not None:worker(args.worker)
    elif args.config is None or args.output is None:parser.error('--config and --output are required')
    else:run(args.config,args.output,args.smoke)
