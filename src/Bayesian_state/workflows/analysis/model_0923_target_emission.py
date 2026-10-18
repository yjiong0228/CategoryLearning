"""Separate perception and precision in saved correct-rule emissions.

Run --source COMPLETED_READOUT_DIAGNOSIS --output NEW_DIRECTORY. Reconstruct the
same perception streams and require exact agreement with every saved target
emission. Evaluate physical stimuli and the original precision upper bound in
a crossed diagnostic, retaining active sets, weights and all other rules.
Hard nearest-region scores are separate geometric diagnostics. They are not
an information-theoretic bound or autonomous learning curves. --smoke uses one
group, two original paths and sixteen trials, plus one archived active-target
control if the prefix has no target. Oral reports are not read.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[4]
ENTRY=str(Path(__file__).resolve().relative_to(ROOT))
METRICS='src/Bayesian_state/evaluation/model_0923_target_emission.py'
READOUT='src/Bayesian_state/workflows/analysis/model_0923_readout_channels.py'


def helpers():
    spec=importlib.util.spec_from_file_location('target_emission_readout',ROOT/READOUT)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.helpers()


def diagnose(job: dict, workflow, recalled) -> None:
    from src.Bayesian_state.model import ModelContext, StateModel
    from src.Bayesian_state.model.modules.base_module import ModuleRole
    from src.Bayesian_state.model.modules.perception import PerceptionModule
    from src.Bayesian_state.model.modules.readout.choice import normalize_probability_vector
    from threadpoolctl import threadpool_info

    v=recalled.base.validation(); inputs=recalled.base.load('target_emission_inputs',recalled.refit.INPUTS)
    # Load the pure metric with its package name so its relative import resolves.
    recalled.base.load('src.Bayesian_state.evaluation.model_0923_readout_channels',
                       'src/Bayesian_state/evaluation/model_0923_readout_channels.py')
    metrics=recalled.base.load('src.Bayesian_state.evaluation.model_0923_target_emission',METRICS)
    engine,meta,arrays=inputs.inputs(job,v)
    for name,expected in job['diagnostic_input_sha256'].items():
        if v.digest(Path(name)) != expected:raise ValueError('Target-emission input changed')
    with np.load(job['reference']) as saved:
        indices=[int(np.flatnonzero(saved['seeds']==seed).item()) for seed in job['seeds']]
        n=16 if job['smoke'] else len(saved['categories'])
        data={key:saved[key][indices,:n] for key in ('present','contribution','beta','correct_probability')}
        np.testing.assert_array_equal(saved['categories'][:n],arrays['categories'][:n])
    with np.load(job['emission_reference']) as saved:
        indices=[int(np.flatnonzero(saved['seeds']==seed).item()) for seed in job['seeds']]
        original=saved['target_correct'][indices,:n]
    physical,categories=arrays['stimulus'][:n],arrays['categories'][:n]
    model=StateModel(engine,context=ModelContext(subject_id=meta['subject'],condition=meta['condition'],
        processed_data_dir=meta['paths']['processed_dir'],dataset_paths=meta['paths']))
    if model.engine.distance_mode != 'boundary' or model.engine.get_module(ModuleRole.MAPPING) is not None:
        raise ValueError('Require the original fixed-label boundary rule geometry')
    target=v.read(Path(job['audit']))['target']; cap=float(job['arm_settings']['upper'])
    if np.any(data['beta'][data['present'].astype(bool)]>cap+1e-12):raise ValueError('Saved precision exceeds its stated cap')

    def emission(stimuli, beta):
        raw=model.partition_model.get_category_probabilities(hypo=target,
            data=(stimuli,np.ones(len(stimuli),int),np.ones(len(stimuli))),beta=float(beta),distance_mode='boundary')
        return np.stack([normalize_probability_vector(raw[:,i],strict=True) for i in range(len(stimuli))])

    def hard(stimuli):
        distance=model.partition_model.boundary_geometry.category_distances(target,stimuli)
        if not np.isfinite(distance).all() or np.any(distance < -1e-12):raise ValueError('Invalid boundary distances')
        minima=np.isclose(distance,distance.min(axis=1,keepdims=True),rtol=0,atol=1e-12)
        return minima[np.arange(n),categories-1]/minima.sum(axis=1)

    clean_upper=emission(physical,cap)[np.arange(n),categories-1]
    hard_clean=hard(physical)
    q0=np.full_like(original,np.nan);clean=q0.copy();upper=q0.copy();hard_noisy=q0.copy()
    previous=np.random.get_state(); noise=None; active_control=None; start=time.monotonic()
    try:
        for row,seed in enumerate(job['seeds']):
            perception=PerceptionModule(model.engine,**deepcopy(engine['modules']['perception_mod'].get('kwargs',{})))
            noise=dict(mode=perception.noise_mode,mean=perception.mean.tolist(),std=perception.std.tolist(),
                uniform_half_range=None if perception.uniform_half_range is None else perception.uniform_half_range.tolist())
            np.random.seed(int(seed));perceived=[]
            for stimulus in physical:
                model.engine.observation=(stimulus,1,1.)
                perceived.append(perception.process(stimu=stimulus))
            perceived=np.asarray(perceived)
            upper[row]=emission(perceived,cap)[np.arange(n),categories-1]
            hard_noisy[row]=hard(perceived)
            for trial in np.flatnonzero(data['present'][row]):
                beta=data['beta'][row,trial]
                q0[row,trial]=emission(perceived[trial:trial+1],beta)[0,categories[trial]-1]
                clean[row,trial]=emission(physical[trial:trial+1],beta)[0,categories[trial]-1]
        if job['smoke'] and not data['present'].any():
            with np.load(job['emission_reference']) as saved:
                positions=np.argwhere(np.isfinite(saved['target_correct']))
                if not len(positions):raise ValueError('No archived active-target control is available')
                row,trial=map(int,positions[0]);seed=int(saved['seeds'][row])
                expected=float(saved['target_correct'][row,trial])
            with np.load(job['reference']) as saved:
                row=int(np.flatnonzero(saved['seeds']==seed).item())
                beta=float(saved['beta'][row,trial])
                assert saved['present'][row,trial]
            perception=PerceptionModule(model.engine,**deepcopy(engine['modules']['perception_mod'].get('kwargs',{})))
            np.random.seed(seed);perceived=[]
            for stimulus in arrays['stimulus'][:trial+1]:
                model.engine.observation=(stimulus,1,1.)
                perceived.append(perception.process(stimu=stimulus))
            perceived=np.asarray(perceived);category=int(arrays['categories'][trial])-1
            actual=float(emission(perceived[-1:],beta)[0,category])
            np.testing.assert_array_equal(actual,expected)
            batch=float(emission(perceived,cap)[-1,category]);single=float(emission(perceived[-1:],cap)[0,category])
            np.testing.assert_allclose(batch,single,rtol=0,atol=1e-14)
            active_control=dict(seed=seed,trial=trial+1,beta=beta,probability=actual,
                exact_saved_emission=True,upper_batch_single_difference=abs(batch-single),
                role='separate replay check; excluded from sixteen-trial summary')
    finally:
        np.random.set_state(previous)
    np.testing.assert_array_equal(q0,original,err_msg='Reconstructed target emission differs from exact saved diagnosis')
    both=np.broadcast_to(clean_upper,original.shape)
    contrast=metrics.emission_contrasts(data['present'],data['contribution'],data['correct_probability'],q0,clean,upper,both)
    target_file=Path(job['target'])
    with target_file.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream,**contrast,**data,original=q0,clean=clean,upper=upper,clean_upper=both,
            hard_noisy=hard_noisy,hard_clean=np.broadcast_to(hard_clean,original.shape),seeds=job['seeds'],categories=categories)
    rows=[]
    for window,first,last in [('whole',0,n),('late128',max(0,n-128),n)]:
        for row,seed in enumerate(job['seeds']):
            mask=data['present'][row,first:last].astype(bool)
            values={key:float(value[row,first:last].mean()) for key,value in contrast.items()}
            values.update(present_fraction=float(mask.mean()),hard_noisy=float(hard_noisy[row,first:last].mean()),
                hard_clean=float(hard_clean[first:last].mean()),upper_all_trials=float(upper[row,first:last].mean()),
                clean_upper_all_trials=float(clean_upper[first:last].mean()))
            for key,value in [('original',q0),('clean',clean),('upper',upper),('clean_upper',both)]:
                # Save joint numerators; aggregate conditional means with the presence denominator.
                values['present_joint_'+key]=float(np.where(mask,value[row,first:last],0.).mean())
            rows.append(dict(subject=meta['subject'],model=job['model'],candidate=job['candidate']['id'],seed=seed,
                window=window,first_trial=first+1,last_trial=last,**values))
    pd.DataFrame(rows).to_csv(target_file.with_suffix('.csv'),index=False)
    v.write(target_file.with_suffix('.json'),dict(status='complete',subject=meta['subject'],model=job['model'],
        paths=len(job['seeds']),trials=n,smoke=job['smoke'],exact_emissions=True,target_rule=target,precision_upper=cap,
        present_rule_trials=int(data['present'].sum()),smoke_active_control=active_control,
        perception=noise,seconds=time.monotonic()-start,numeric_pools=threadpool_info(),
        array_sha256=v.digest(target_file.with_suffix('.npz')),table_sha256=v.digest(target_file.with_suffix('.csv'))))


def run(source: Path, output: Path, smoke: bool) -> None:
    recalled=helpers();v=recalled.base.validation()
    if output.exists():raise FileExistsError('Use a new target-emission directory')
    done=v.read(source/'completion.json')
    if (done['status']!='complete' or done['smoke'] or done['exact_replay_groups']!=done['groups']
            or done['exact_perception_groups']!=done['groups']):
        raise ValueError('Require a completed exact readout diagnosis')
    files=sorted((source/'jobs').glob('*.json'));files=files[:1] if smoke else files
    jobs=[];inputs={str((source/'completion.json').resolve()):v.digest(source/'completion.json')}
    for file in files:
        job=v.read(file);name=job['name'];record=v.read(source/(name+'.json'));reference=source/(name+'.npz')
        if (not record['exact_replay'] or not record['exact_perception']
                or v.digest(reference)!=record['array_sha256']
                or v.digest(reference)!=done['outputs_sha256'][reference.name]):
            raise ValueError('Saved readout controls or arrays changed')
        if len(job['seeds'])!=record['paths'] or len(set(job['seeds']))!=len(job['seeds']):raise ValueError('Incomplete saved path bank')
        for path_key,hash_key in [('reference','reference_sha256'),('original_job','original_job_sha256'),('audit','audit_sha256')]:
            if v.digest(Path(job[path_key])) != job[hash_key]:raise ValueError('Original autonomous input changed')
        dependencies=[file,reference,source/(name+'.json'),Path(job['reference']),Path(job['original_job']),Path(job['audit'])]
        meta=v.read(Path(job['folder'])/'prepared.json')
        dependencies.extend(Path(meta['paths'][key]) for key in ('perception_summary','perception_summary_72','feature_order_data'))
        hashes={str(p.resolve()):v.digest(p) for p in dependencies}
        inputs.update(hashes)
        job.update(seeds=job['seeds'][:2] if smoke else job['seeds'],smoke=smoke,
            emission_reference=str(reference.resolve()),diagnostic_input_sha256=hashes,
            target=str((output/name).resolve()))
        job['source_sha256'].update({name:v.digest(ROOT/name) for name in (ENTRY,METRICS,READOUT)})
        jobs.append(job)
    output.mkdir(parents=True);(output/'jobs').mkdir()
    for name in jobs[0]['source_sha256']:
        destination=output/'source'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
    v.write(output/'protocol.json',dict(source=str(source.resolve()),smoke=smoke,groups=len(jobs),
        source_sha256=jobs[0]['source_sha256'],input_sha256=inputs,role='fixed_state_target_emission_diagnostic',
        data_use='no fitting; no oral input; task target identity used as oracle diagnosis',
        assumptions='change only target emission; retain all saved weights, active sets, other predictions and future states',
        hard_rule='uniform ties among minimum boundary distances within absolute tolerance1e-12; geometric diagnostic'))
    def launch(job):
        file=output/'jobs'/(job['name']+'.json');v.write(file,job)
        launcher=('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                  "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory(prefix='target_emission_') as cache:
            env.update(NUMBA_CACHE_DIR=cache,MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as log:
                subprocess.run([sys.executable,'-c',launcher,job['runtime'],str(ROOT/ENTRY),'--worker',str(file.resolve())],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(max_workers=min(len(jobs),len(os.sched_getaffinity(0)))) as pool:list(pool.map(launch,jobs))
    table=pd.concat([pd.read_csv(Path(job['target']).with_suffix('.csv')) for job in jobs],ignore_index=True)
    table.to_csv(output/'path_windows.csv',index=False)
    keys=['subject','model','candidate','window','first_trial','last_trial']
    long=table.melt(id_vars=keys+['seed'],var_name='metric',value_name='value')
    summary=long.groupby(keys+['metric']).value.agg(['mean','std','count']).reset_index()
    summary['mc_se']=summary['std']/np.sqrt(summary['count']);summary.to_csv(output/'summary.csv',index=False)
    for name,expected in inputs.items():
        if v.digest(Path(name))!=expected:raise ValueError('Diagnostic input changed')
    v.write(output/'completion.json',dict(status='complete',smoke=smoke,groups=len(jobs),
        exact_emission_groups=len(jobs),paths=sum(len(j['seeds']) for j in jobs),
        artifacts_sha256={p.name:v.digest(p) for p in output.iterdir() if p.suffix in ('.json','.csv','.npz')}))
    print(f'{len(jobs)} target-emission groups complete; all saved emissions reproduced exactly',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:
        recalled=helpers();recalled.refit.autonomous=lambda job,workflow:diagnose(job,workflow,recalled)
        recalled.worker(args.worker)
    elif args.source and args.output:run(args.source,args.output,args.smoke)
    else:parser.error('Supply --source and --output')
