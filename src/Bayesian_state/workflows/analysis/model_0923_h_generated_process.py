"""Replay frozen autonomous paths and record H events without changing behavior.

The existing autonomous generator and sparse lifecycle observer provide all
cognitive operations. Every replay must reproduce independently saved choices,
feedback and correct-choice probabilities exactly. Physical rule compatibility
is a retrospective diagnostic and never enters the cognitive model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

REPO=Path(__file__).resolve().parents[4]


def validation_module():
    path=REPO/'src/Bayesian_state/workflows/analysis/model_0923_h_validation.py'
    spec=importlib.util.spec_from_file_location('generated_process_validation',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def process_summary(a: dict, compatible: np.ndarray, cut: int) -> list[dict]:
    """Summarize whole-path events with unchanged chronological boundaries."""
    active=a['rule']>=0;good=active&np.isin(a['rule'],compatible)
    n,paths=active.shape[:2];rows=[]
    for phase,first,last in [('training',0,cut),('test',cut,n),('late',max(0,n-64),n)]:
        if first>=last:continue
        for path in range(paths):
            part=(slice(first,last),path);slots=(*part,slice(None))
            searched=(a['proposed'][part]>=0)&(a['source'][part]>0)
            rows.append(dict(phase=phase,path=path,trials=last-first,
                proposals=int(searched.sum()),admissions=int((a['admitted'][part]&searched).sum()),
                rejections=int((a['rejected'][part]&searched).sum()),
                search_probability=float(a['search_probability'][part].mean()),
                search_near_one_fraction=float((a['search_probability'][part]>=.95).mean()),
                active_rules=float(active[slots].sum(axis=-1).mean()),
                compatible_presence=float(good[slots].any(axis=-1).mean()),
                compatible_contribution=float(np.where(good[slots],a['contribution'][slots],0.).sum(axis=-1).mean()),
                capacity_drops=int((a['capacity_dropped'][part]>=0).sum()),
                compatible_capacity_drops=int(np.isin(a['capacity_dropped'][part],compatible).sum()),
                exits=int((a['exits'][slots]>=0).sum()),
                compatible_exits=int(np.isin(a['exits'][slots],compatible).sum()),
                joint_drops=int((a['joint_dropped'][slots]>=0).sum()),
                compatible_joint_drops=int(np.isin(a['joint_dropped'][slots],compatible).sum())))
    return rows


def worker(file: Path) -> None:
    v=validation_module();job=v.read(file)
    workflow=v.intervention_workflow();d=workflow.cohort().load_diagnosis();workflow.install_transition()
    from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
    from src.Bayesian_state.evaluation.model_0923_postfit import rule_catalog
    folder=Path(job['folder']);meta=v.read(folder/'prepared.json')
    engine=v.read(folder/'engine.json');engine['modules']['hypo_transitions_mod']['kwargs']['intervention']=job['mechanism']
    engine=apply_fixed_hyperparams_to_engine_config(engine,job['candidate']['hyperparams'])
    parts=[]
    for name in ['training.npz','sealed_test.npz']:
        with np.load(folder/name) as z:parts.append((z['stimulus'],z['categories']))
    stimulus=np.concatenate([p[0] for p in parts]);categories=np.concatenate([p[1] for p in parts])
    kwargs=engine['modules']['hypo_transitions_mod']['kwargs']
    class Recorder(d.PathRecorder):
        def __init__(self):
            super().__init__(categories,len(job['seeds']),kwargs['max_capacity'],
                engine['partition']['kwargs']['n_cats'],meta['condition']==3)
            shape=self.arrays['choice'].shape
            self.arrays['search_probability']=np.zeros(shape)
            for key in ['admitted','rejected']:self.arrays[key]=np.zeros(shape,bool)

        def before(self,model,choice,feedback):
            token=super().before(model,choice,feedback);t,path=token[0][:2]
            transition=model.engine.get_module(d.ModuleRole.HYPOTHESIS_TRANSITION)
            event=transition.transition_log[-1]
            self.arrays['search_probability'][t,path]=event['search_probability']
            self.arrays['admitted'][t,path]=event['admission_event']
            self.arrays['rejected'][t,path]=event.get('newcomer_rejected',False)
            return token
    recorder=Recorder()
    with d.single_threaded_processes(),d.record_lifecycle(recorder):v.autonomous(job,workflow)
    recorder.validate();a=recorder.arrays;target=Path(job['target'])
    if v.digest(Path(job['reference']))!=job['reference_sha256']:raise ValueError('Autonomous reference changed')
    with np.load(job['reference']) as reference:
        np.testing.assert_array_equal(reference['categories'],categories)
        np.testing.assert_array_equal(reference['seeds'][job['indices']],job['seeds'])
        np.testing.assert_array_equal(reference['choices'][job['indices']],a['choice'].T)
        np.testing.assert_array_equal(reference['feedback'][job['indices']],a['feedback'].T)
        correct=np.take_along_axis(a['prediction'],categories[:,None,None]-1,axis=2)[...,0].T
        np.testing.assert_array_equal(reference['correct_probability'][job['indices']],correct)
    inputs=dict(engine=engine,condition=meta['condition'],subject=meta['subject'],paths=meta['paths'],
                arrays=SimpleNamespace(stimulus=stimulus,categories=categories))
    settings=job['diagnostics']
    catalog=d.geometric_catalog(inputs,rule_catalog(inputs),settings['geometry_beta'])
    compatible=catalog.loc[catalog.physical_argmax_accuracy>=settings['compatibility_threshold'],'rule'].to_numpy()
    trace_file=target.with_name(target.name+'_trace').with_suffix('.npz')
    with trace_file.open('xb') as f:
        np.savez_compressed(f,**a,seeds=job['seeds'],categories=categories)
    if job['indices'][0]==0:
        catalog.to_csv(target.parent/f'S{meta["subject"]}_{job["model"]}_rule_catalog.csv',index=False)
    rows=process_summary(a,compatible,meta['training_trials'])
    for row in rows:row.update(subject=meta['subject'],model=job['model'],seed=job['seeds'][row['path']])
    v.write(target.with_name(target.name+'_process').with_suffix('.json'),dict(rows=rows,
        exact_replay_paths=len(job['seeds']),compatible_rules=compatible.tolist(),trace_sha256=v.digest(trace_file),
        compatibility_threshold=settings['compatibility_threshold'],geometry_beta=settings['geometry_beta'],source_sha256=job['source_sha256']))


def run(fit: Path, autonomous: Path, config: Path, output: Path, smoke: bool) -> None:
    v=validation_module();frozen=v.read(fit/'nominees.json')
    settings=yaml.safe_load(config.read_text())
    if frozen['smoke'] or frozen['status']!='frozen_before_test':raise ValueError('Need frozen scientific nominees')
    if output.exists():raise FileExistsError('Use a new process-diagnostic destination')
    if v.read(autonomous/'evaluation_protocol.json')['nomination_sha256']!=v.digest(fit/'nominees.json'):
        raise ValueError('Autonomous reference uses a different frozen fit')
    for name,sha in frozen['source_sha256'].items():
        if v.digest(REPO/name)!=sha:raise ValueError('Fitted implementation changed')
    output.mkdir(parents=True)
    hashes=v.source_hashes()
    for name in ['src/Bayesian_state/workflows/analysis/model_0923_path_diagnosis.py',str(Path(__file__).resolve().relative_to(REPO))]:
        hashes[name]=v.digest(REPO/name)
    reference_manifest=v.read(autonomous/'evaluation.json')
    if reference_manifest['status']!='complete' or reference_manifest['role']!='autonomous_from_start_prefix_fitted':
        raise ValueError('Need completed autonomous reference paths')
    originals=reference_manifest['rows']
    expected={(r['subject'],r['model']) for r in frozen['nominees']}
    if len(originals)!=len(expected) or {(r['subject'],r['model']) for r in originals}!=expected:
        raise ValueError('Autonomous participant/model coverage differs')
    originals=originals[:1] if smoke else originals
    lookup={(r['subject'],r['model']):r for r in frozen['nominees']};jobs=[]
    banks={(r['subject'],r['model']):[lookup[(r['subject'],r['model'])]['candidate']] for r in originals}
    bases=v.score_jobs(Path(frozen['data']),output,frozen['settings'],banks,'evaluate',False)
    indexed={(r['subject'],r['model']):r for r in originals}
    for base in bases:
        subject=v.read(Path(base['folder'])/'prepared.json')['subject'];row=indexed[(subject,base['model'])]
        if row['candidate']['hyperparams']!=base['candidate']['hyperparams']:raise ValueError('Reference parameters differ')
        reference=autonomous/'evaluate'/f'subject_{subject}'/base['model']/(base['candidate']['id']+'.npz')
        if v.digest(reference)!=row['predictions_sha256']:raise ValueError('Autonomous input changed')
        with np.load(reference) as z:seeds=z['seeds'].tolist()
        if smoke:seeds=seeds[:1]
        width=settings['chunk_size']
        for first in range(0,len(seeds),width):
            name=f'S{subject}_{base["model"]}_C{first:03d}'
            jobs.append(dict(base,phase='autonomous',name=name,seeds=seeds[first:first+width],diagnostics=settings,
                target=str((output/'paths'/name).resolve()),reference=str(reference.resolve()),
                reference_sha256=row['predictions_sha256'],indices=list(range(first,min(first+width,len(seeds)))),source_sha256=hashes))
    v.write(output/'protocol.json',dict(fit=str(fit.resolve()),autonomous=str(autonomous.resolve()),smoke=smoke,
        nomination_sha256=v.digest(fit/'nominees.json'),source_sha256=hashes,settings=settings,
        role='exact_replay_of_frozen_generated_paths_with_observation_only_callbacks',
        compatibility='retrospective physical-stimulus argmax accuracy; ties share credit; threshold and beta from settings'))
    from src.Bayesian_state.utils.parallel import parallel_job_count
    workers=parallel_job_count(1 if smoke else min(settings['parallel_budget'],frozen['settings']['parallel_budget']),len(jobs))
    (output/'jobs').mkdir()
    def launch(job):
        file=output/'jobs'/(job['name']+'.json');v.write(file,job)
        launcher='import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name="__main__")'
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='h_process_') as cache,file.with_suffix('.log').open('x') as log:
            env['NUMBA_CACHE_DIR']=cache
            subprocess.run([sys.executable,'-c',launcher,job['runtime'],str(Path(__file__).resolve()),'--worker',str(file.resolve())],
                env=env,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'Replaying {len(jobs)} chunks with {workers} workers',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index,future in enumerate(as_completed([pool.submit(launch,j) for j in jobs]),1):
            future.result()
            if index%16==0 or index==len(jobs):print(f'{index}/{len(jobs)} complete',flush=True)
    records=[v.read(Path(j['target']).with_name(Path(j['target']).name+'_process').with_suffix('.json')) for j in jobs]
    frame=pd.DataFrame([r for item in records for r in item['rows']]);frame.to_csv(output/'process_paths.csv',index=False)
    metrics=[c for c in frame if c not in ['subject','model','phase','path','seed']]
    frame.groupby(['subject','model','phase'],sort=False)[metrics].mean().reset_index().to_csv(output/'process_means.csv',index=False)
    v.write(output/'completion.json',dict(status='complete',smoke=smoke,exact_replay_paths=sum(r['exact_replay_paths'] for r in records),
        jobs=len(jobs),workers=workers))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['fit','autonomous','config','output','worker']:p.add_argument('--'+name,type=Path)
    p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.worker:worker(args.worker)
    else:
        if any(x is None for x in [args.fit,args.autonomous,args.config,args.output]):p.error('fit, autonomous, config and output are required')
        run(args.fit,args.autonomous,args.config,args.output,args.smoke)


if __name__=='__main__':main()
