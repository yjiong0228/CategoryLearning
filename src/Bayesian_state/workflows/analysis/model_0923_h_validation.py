"""Bounded, equal-budget H comparisons with training-only parameter search.

prepare seals separate training and test arrays for previously unselected IDs.
search reads training files only, freezing nominees before evaluate opens tests.
Each scoring job starts from the archived cognitive runtime and overrides only
the shared H implementation. One outer process layer uses one numeric thread.
All destinations must be new; existing scientific outputs remain immutable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[4]
H_PATH = 'engine.modules.hypo_transitions_mod.kwargs.'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def intervention_workflow():
    path=REPO/'src/Bayesian_state/workflows/analysis/model_0923_mechanism_interventions.py'
    spec=importlib.util.spec_from_file_location('h_validation_interventions',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def training_boundary(trials: int, fraction: float=.7) -> int:
    if trials < 128:
        raise ValueError('At least 64 training and 64 test trials are required')
    return min(trials-64,max(64,int(trials*fraction)//32*32))


def pooled_nll(observed_probability):
    import numpy as np
    p=np.asarray(observed_probability,dtype=float)
    if p.ndim!=2 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Need finite seed-by-trial observation probabilities')
    return float(-np.log(np.maximum(p.mean(axis=0),np.finfo(float).tiny)).mean())


def prepare_subject(job: dict, workflow) -> None:
    import numpy as np
    from src.Bayesian_state.evaluation.model_0923_fit_source import _base_config
    from src.Bayesian_state.simulation.config import resolve_engine_config
    from src.Bayesian_state.simulation.data import SubjectTrialDataLoader
    from src.Bayesian_state.utils.datasets import resolve_dataset_paths
    folder=Path(job['folder']);folder.mkdir()
    cache=folder/'resources_cache';cache.mkdir()
    subject,condition=job['subject'],job['condition']
    fit=Path(job['source_fit'])
    path,config=_base_config(fit,'initial_cd')
    config.pop('subject_overrides',None)
    config['engine_config_path']=str(fit/'source/configs/exp123/model_struct'/f'model_0923_cond{condition}_full_v3.yaml')
    engine=resolve_engine_config(config,path.parent,subject_id=subject)
    engine['oral_readout']=None
    engine['partition']['kwargs']['similarity_cache_dir']=str(cache)
    engine['inference']['particle_blocks']=1
    reference=REPO/job['cache_sources'][str(condition)]
    previous=read(reference.parent/'prepared.json')
    for name,sha in previous['cognitive_source_sha256'].items():
        if digest(fit/'source'/name)!=sha:
            raise ValueError(f'Archived cognitive source changed: {name}')
    for file in reference.glob('*.npz'):
        shutil.copy2(file,cache/file.name)
    paths=resolve_dataset_paths(config,path.parent)
    loader=SubjectTrialDataLoader(engine,dataset_paths=paths)
    loader.prepare_data(paths['learning_data'])
    frame=loader._get_subject_frame(subject,1.)
    arrays=loader._extract_arrays(frame,None)
    if loader._get_condition_value(frame)!=condition:
        raise ValueError('Declared and actual task condition differ')
    n=len(arrays.choices)
    if n<128:
        write(folder/'excluded.json',dict(subject=subject,condition=condition,trials=n,reason='less_than_128_valid_trials'))
        return
    cut=training_boundary(n,job['training_fraction'])
    hashes={}
    for name,part in [('training',slice(0,cut)),('sealed_test',slice(cut,n))]:
        target=folder/(name+'.npz')
        with target.open('xb') as f:
            np.savez_compressed(f,stimulus=arrays.stimulus[part],choices=arrays.choices[part],
                                feedback=arrays.feedback[part],categories=arrays.categories[part])
        hashes[target.name]=digest(target)
    write(folder/'engine.json',engine)
    write(folder/'prepared.json',dict(subject=subject,condition=condition,trials=n,training_trials=cut,
        test_trials=n-cut,paths={k:str(v) for k,v in paths.items()},array_sha256=hashes,
        engine_sha256=digest(folder/'engine.json'),source_engine_sha256=digest(Path(config['engine_config_path'])),
        frozen_cognitive_source_sha256=previous['cognitive_source_sha256'],
        learning_data_sha256=digest(Path(paths['learning_data'])),split='chronological_approximately_70_percent_at_32_trial_boundary'))


def score(job: dict, workflow) -> None:
    import numpy as np
    from threadpoolctl import threadpool_info
    from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
    from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
    folder=Path(job['folder']); meta=read(folder/'prepared.json')
    for name,sha in job['source_sha256'].items():
        if digest(REPO/name)!=sha:raise ValueError(f'Executing study source changed: {name}')
    if digest(folder/'engine.json')!=meta['engine_sha256']:raise ValueError('Prepared engine changed')
    for name,sha in meta.get('frozen_cognitive_source_sha256',{}).items():
        if digest(Path(job['runtime'])/name)!=sha:raise ValueError('Archived cognitive source changed')
    engine=read(folder/'engine.json')
    engine['modules']['hypo_transitions_mod']['kwargs']['intervention']=job['mechanism']
    engine=apply_fixed_hyperparams_to_engine_config(engine,job['candidate']['hyperparams'])
    files=['training.npz']+(['sealed_test.npz'] if job['phase']=='evaluate' else [])
    parts=[]
    for name in files:
        if digest(folder/name)!=meta['array_sha256'][name]:raise ValueError('Prepared array changed')
        with np.load(folder/name) as z:parts.append({k:z[k] for k in z.files})
    arrays={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if job['smoke']:
        arrays={k:v[:8] for k,v in arrays.items()}
    observed,correct,probabilities=[],[],[]
    start=time.monotonic();cpu=time.process_time()
    for seed in job['seeds']:
        result=run_state_model_particle_filter(engine_config=engine,subject_id=meta['subject'],condition=meta['condition'],
            stimulus=arrays['stimulus'],choices=arrays['choices'],feedback=arrays['feedback'],
            particle_count=job['particles'],particle_blocks=1,filter_seed=seed,choice_readout_power=1.,
            processed_data_dir=meta['paths']['processed_dir'],dataset_paths=meta['paths'])
        p=np.asarray(result.observation_probabilities['prior_t'])
        if not np.isfinite(p).all() or np.any(p<0):raise ValueError('Invalid prior-choice probability')
        np.testing.assert_allclose(p.sum(axis=1),1.,atol=1e-12,rtol=0)
        observed.append(p[np.arange(len(p)),arrays['choices']-1])
        correct.append(p[np.arange(len(p)),arrays['categories']-1])
        if job['phase']=='evaluate':probabilities.append(p)
    observed,correct=np.asarray(observed),np.asarray(correct)
    target=Path(job['target']);target.parent.mkdir(parents=True,exist_ok=True)
    output=dict(observed_probability=observed,correct_probability=correct,seeds=np.array(job['seeds']))
    if probabilities:output.update(probabilities=np.array(probabilities),choices=arrays['choices'],categories=arrays['categories'])
    with target.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**output)
    row=dict(subject=meta['subject'],condition=meta['condition'],model=job['model'],candidate=job['candidate'],
        particles=job['particles'],seeds=job['seeds'],trials=len(observed[0]),mean_nll=pooled_nll(observed),
        seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,numeric_pools=threadpool_info(),
        predictions_sha256=digest(target.with_suffix('.npz')),role=job['phase'])
    if job['phase']=='evaluate':
        cut=meta['training_trials']
        row.update(training_nll=pooled_nll(observed[:,:cut]),test_nll=pooled_nll(observed[:,cut:]),
                   training_trials=cut,test_trials=meta['test_trials'])
    write(target.with_suffix('.json'),row)


def autonomous(job: dict, workflow) -> None:
    """Generate from task start using prefix-fitted parameters and own feedback."""
    import numpy as np
    from threadpoolctl import threadpool_info
    from src.Bayesian_state.simulation.autonomous import run_autonomous_category_learning
    from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
    folder=Path(job['folder']);meta=read(folder/'prepared.json')
    for name,sha in job['source_sha256'].items():
        if digest(REPO/name)!=sha:raise ValueError('Study source changed')
    if digest(folder/'engine.json')!=meta['engine_sha256']:raise ValueError('Prepared engine changed')
    for name,sha in meta.get('frozen_cognitive_source_sha256',{}).items():
        if digest(Path(job['runtime'])/name)!=sha:raise ValueError('Archived cognitive source changed')
    engine=read(folder/'engine.json')
    engine['modules']['hypo_transitions_mod']['kwargs']['intervention']=job['mechanism']
    engine=apply_fixed_hyperparams_to_engine_config(engine,job['candidate']['hyperparams'])
    parts=[]
    for name in ['training.npz','sealed_test.npz']:
        if digest(folder/name)!=meta['array_sha256'][name]:raise ValueError('Prepared array changed')
        # Only physical stimuli and task categories enter autonomous generation.
        with np.load(folder/name) as z:parts.append((z['stimulus'],z['categories']))
    stimulus=np.concatenate([p[0] for p in parts]);categories=np.concatenate([p[1] for p in parts])
    probability,choices,feedback=[],[],[]
    start=time.monotonic();cpu=time.process_time()
    for seed in job['seeds']:
        result=run_autonomous_category_learning(engine_config=engine,subject_id=meta['subject'],condition=meta['condition'],
            stimulus=stimulus,categories=categories,trajectory_seed=seed,
            processed_data_dir=meta['paths']['processed_dir'],dataset_paths=meta['paths'])
        p=np.asarray(result.trajectory.observed_probabilities)
        np.testing.assert_allclose(p.sum(axis=1),1.,atol=1e-12,rtol=0)
        probability.append(p[np.arange(len(p)),categories-1])
        choices.append(result.trajectory.choices);feedback.append(result.trajectory.feedback)
    target=Path(job['target']);target.parent.mkdir(parents=True,exist_ok=True)
    with target.with_suffix('.npz').open('xb') as f:
        np.savez_compressed(f,correct_probability=np.array(probability),choices=np.array(choices),
            feedback=np.array(feedback),seeds=np.array(job['seeds']),categories=categories)
    write(target.with_suffix('.json'),dict(subject=meta['subject'],model=job['model'],role='autonomous_from_start',
        candidate=job['candidate'],paths=len(job['seeds']),training_trials=meta['training_trials'],
        seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,numeric_pools=threadpool_info(),
        predictions_sha256=digest(target.with_suffix('.npz'))))


def worker(path: Path) -> None:
    workflow=intervention_workflow();cohort=workflow.cohort();diagnosis=cohort.load_diagnosis()
    workflow.install_transition();job=read(path)
    with diagnosis.single_threaded_processes():
        {'prepare':prepare_subject,'autonomous':autonomous}.get(job['phase'],score)(job,workflow)


def execute(jobs: list[dict], output: Path, budget: int) -> None:
    """Launch fresh archived-runtime processes; never nest particle pools."""
    from src.Bayesian_state.utils.parallel import parallel_job_count
    workers=parallel_job_count(budget,len(jobs))
    directory=output/'jobs';directory.mkdir(exist_ok=True)
    def launch(job):
        file=directory/(job['name']+'.json');write(file,job)
        launcher=('import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);'
                  "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                 MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='model0923_h_validation_') as cache:
            env['NUMBA_CACHE_DIR']=cache
            with file.with_suffix('.log').open('x') as stream:
                result=subprocess.run([sys.executable,'-c',launcher,job['runtime'],str(Path(__file__).resolve()),
                    '--worker',str(file)],env=env,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Failed worker: {file.with_suffix(".log")}')
    print(f'Running {len(jobs)} jobs with {workers} workers, numeric threads=1',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(launch,job) for job in jobs]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i%16==0 or i==len(jobs):print(f'{i}/{len(jobs)} complete',flush=True)


def source_hashes() -> dict:
    workflow=intervention_workflow()
    return {str(p.relative_to(REPO)):digest(p) for p in [Path(__file__).resolve(),
        REPO/workflow.TRANSITION,REPO/workflow.RETENTION]}


def prepare(config: Path, output: Path, smoke: bool) -> None:
    import yaml
    settings=yaml.safe_load(config.read_text())
    if output.exists():raise FileExistsError('Use a new preparation output')
    output.mkdir(parents=True)
    fit=(REPO/settings['source_fit']).resolve()
    jobs=[]
    subjects=settings['subjects'][:1] if smoke else settings['subjects']
    for subject,condition in subjects:
        jobs.append(dict(phase='prepare',name=f'S{subject}_prepare',subject=subject,condition=condition,
            source_fit=str(fit),runtime=str(fit/'source'),folder=str((output/f'subject_{subject}').resolve()),
            cache_sources=settings['cache_sources'],training_fraction=settings['training_fraction']))
    execute(jobs,output,1 if smoke else settings['parallel_budget'])
    write(output/'preparation_manifest.json',dict(status='complete',settings=settings,smoke=smoke,
        source_sha256=source_hashes(),subjects=[j['subject'] for j in jobs],
        artifacts_sha256={str(p.relative_to(output)):digest(p) for p in output.glob('subject_*/*') if p.is_file()}))


def model_space(settings: dict, model: str) -> tuple[dict,dict]:
    """Use the declared original parameter support, replacing inactive coordinates."""
    import yaml
    original=yaml.safe_load((REPO/settings['parameter_support']).read_text())['hyperparam_space']
    space={k:v['values'] for k,v in original.items()}
    mechanism=settings['models'][model]
    if mechanism['kind']=='J':
        del space[H_PATH+'exit_rate']
        coordinate='stay_bias' if 'stay_bias' in mechanism else 'cost'
        space[H_PATH+'intervention.'+coordinate]=settings['stay_bias_values' if coordinate=='stay_bias' else 'retention_cost_values']
    if mechanism['kind'] in ('E','J') and mechanism['decay']!='memory':
        space[H_PATH+'intervention.decay']=settings['evidence_decay_values']
    anchor={k:values[len(values)//2] for k,values in space.items()}
    return space,anchor


def score_jobs(data: Path, output: Path, settings: dict, banks: dict, stage: str, smoke: bool) -> list[dict]:
    from src.Bayesian_state.utils.seeding import stable_seed
    preparation=read(data/'preparation_manifest.json')
    runtime=str((REPO/preparation['settings']['source_fit']/'source').resolve())
    budget=settings['budgets'][stage]
    hashes=source_hashes(); jobs=[]
    for (subject,model),bank in banks.items():
        seeds=[stable_seed(dict(protocol=settings['protocol'],base=settings['seed'],stage=stage,
                               subject=subject,repeat=i)) for i in range(2 if smoke else budget['seeds'])]
        for candidate in bank:
            target=output/stage/f'subject_{subject}'/model/candidate['id']
            jobs.append(dict(phase='evaluate' if stage=='evaluate' else 'training',
                name=f'{stage}_S{subject}_{model}_{candidate["id"]}',folder=str((data/f'subject_{subject}').resolve()),
                model=model,mechanism=settings['models'][model],candidate=candidate,runtime=runtime,
                seeds=seeds,particles=2 if smoke else budget['particles'],source_sha256=hashes,
                smoke=smoke,target=str(target.resolve())))
    return jobs


def stage_rows(output: Path, stage: str, subject: int, model: str) -> list[dict]:
    rows=[]
    for path in sorted((output/stage/f'subject_{subject}'/model).glob('*.json')):
        record=read(path)
        rows.append(dict(record['candidate'],mean_nll=record['mean_nll'],score_file=str(path.resolve())))
    return sorted(rows,key=lambda r:(r['mean_nll'],r['id']))


def search(config: Path, data: Path, output: Path, smoke: bool) -> None:
    import yaml
    from src.Bayesian_state.optimization.search.adaptive_proposals import initial_points,candidate,propose_round,select_elites
    from src.Bayesian_state.utils.seeding import stable_seed
    settings=yaml.safe_load(config.read_text());preparation=read(data/'preparation_manifest.json')
    if output.exists():raise FileExistsError('Use a new search output')
    output.mkdir(parents=True)
    write(output/'search_protocol.json',dict(settings=settings,data=str(data.resolve()),smoke=smoke,
        source_sha256=source_hashes(),data_manifest_sha256=digest(data/'preparation_manifest.json')))
    subjects=[s for s in preparation['subjects'] if (data/f'subject_{s}'/'prepared.json').exists()]
    if smoke:subjects=subjects[:1]
    banks={};spaces={};seen={}
    for subject in subjects:
        for model in settings['models']:
            key=(subject,model);space,anchor=model_space(settings,model);spaces[key]=space
            points=initial_points(space,anchor,5 if smoke else settings['initial_candidates'],
                stable_seed(dict(base=settings['seed'],subject=subject,role='initial')),cover_key=H_PATH+'max_capacity')
            banks[key]=[candidate(p,'initial') for p in points];seen[key]={r['id'] for r in banks[key]}
    execute(score_jobs(data,output,settings,banks,'screen',smoke),output,1 if smoke else settings['parallel_budget'])
    pooled={key:stage_rows(output,'screen',*key) for key in banks}
    for index in range(1 if smoke else settings['refinement_rounds']):
        stage=f'refine{index+1}'
        settings['budgets'][stage]=settings['budgets']['refine']
        for key in banks:
            elites=select_elites(pooled[key],1 if smoke else settings['elites'],exclude_key=H_PATH+'max_capacity')
            proposals=propose_round(elites,spaces[key],seen[key],1 if smoke else settings['proposals_per_elite'],
                stable_seed(dict(base=settings['seed'],subject=key[0],role=stage)),.5,.25)
            if len(proposals)!=(1 if smoke else settings['elites']*settings['proposals_per_elite']):
                raise ValueError('Unequal refinement proposal budget')
            banks[key]=proposals+elites
            seen[key].update(r['id'] for r in proposals)
        execute(score_jobs(data,output,settings,banks,stage,smoke),output,1 if smoke else settings['parallel_budget'])
        pooled={key:stage_rows(output,stage,*key) for key in banks}
    banks={key:rows[:2 if smoke else settings['finalists']] for key,rows in pooled.items()}
    execute(score_jobs(data,output,settings,banks,'selection',smoke),output,1 if smoke else settings['parallel_budget'])
    nominees=[]
    for key in banks:
        rows=stage_rows(output,'selection',*key)
        nominees.append(dict(subject=key[0],model=key[1],candidate=rows[0],finalists=rows,
            scientific_status='bounded_training_selection',unique_candidates=len(seen[key])))
    write(output/'nominees.json',dict(status='frozen_before_test',settings=settings,smoke=smoke,
        data=str(data.resolve()),source_sha256=source_hashes(),nominees=nominees))
    print('Training selection complete; test arrays remain unopened by scoring jobs',flush=True)


def evaluate(fit: Path, output: Path, generate: bool = False) -> None:
    frozen=read(fit/'nominees.json')
    if frozen['smoke']:raise ValueError('Smoke nominees cannot enter scientific test evaluation')
    if output.exists():raise FileExistsError('Use a new evaluation output')
    for name,sha in frozen['source_sha256'].items():
        if digest(REPO/name)!=sha:raise ValueError('Study code changed after nomination')
    output.mkdir(parents=True)
    banks={(r['subject'],r['model']):[r['candidate']] for r in frozen['nominees']}
    write(output/'evaluation_protocol.json',dict(nomination_sha256=digest(fit/'nominees.json'),
        fit=str(fit.resolve()),source_sha256=source_hashes(),test_opening='after_all_nominees_frozen'))
    jobs=score_jobs(Path(frozen['data']),output,frozen['settings'],banks,'evaluate',False)
    if generate:
        from src.Bayesian_state.utils.seeding import stable_seed
        for job in jobs:
            job['phase']='autonomous'
            job['seeds']=[stable_seed(dict(protocol=frozen['settings']['protocol'],base=frozen['settings']['seed'],
                role='autonomous',subject=read(Path(job['folder'])/'prepared.json')['subject'],path=i))
                for i in range(frozen['settings'].get('autonomous_paths',64))]
    # Independent complete trajectories / filter seeds provide the outer
    # parallel units; no filter creates an inner particle-process pool.
    chunks=[]
    for job in jobs:
        width=frozen['settings'].get('autonomous_chunk',8) if generate else frozen['settings'].get('evaluate_seed_chunk',2)
        for start in range(0,len(job['seeds']),width):
            chunk=deepcopy(job);target=Path(job['target'])
            chunk.update(seeds=job['seeds'][start:start+width],name=job['name']+f'_C{start:03d}',
                target=str(target.parent/'chunks'/(target.name+f'_C{start:03d}')))
            chunks.append(chunk)
    execute(chunks,output,frozen['settings']['parallel_budget'])
    rows=[combine_evaluation_chunks(job,[c for c in chunks if c['model']==job['model'] and c['folder']==job['folder']],generate)
          for job in jobs]
    write(output/'evaluation.json',dict(status='complete',rows=rows,
        role='autonomous_from_start_prefix_fitted' if generate else 'sequential_heldout_prechoice_predictions'))


def combine_evaluation_chunks(job: dict, chunks: list[dict], generate: bool) -> dict:
    """Pool complete seed predictions before taking logs, preserving trial order."""
    import numpy as np
    records=[];arrays=[]
    for chunk in chunks:
        path=Path(chunk['target'])
        row=read(path.with_suffix('.json'))
        if digest(path.with_suffix('.npz'))!=row['predictions_sha256']:raise ValueError('Chunk output changed')
        records.append(row)
        with np.load(path.with_suffix('.npz')) as z:arrays.append({k:z[k] for k in z.files})
    shared={'categories'} if generate else {'categories','choices'}
    combined={}
    for key in arrays[0]:
        if key in shared:
            for other in arrays[1:]:np.testing.assert_array_equal(arrays[0][key],other[key])
            combined[key]=arrays[0][key]
        else:combined[key]=np.concatenate([a[key] for a in arrays],axis=0)
    np.testing.assert_array_equal(combined['seeds'],job['seeds'])
    target=Path(job['target']);row=deepcopy(records[0])
    with target.with_suffix('.npz').open('xb') as f:np.savez_compressed(f,**combined)
    row.update(seconds=sum(r['seconds'] for r in records),cpu_seconds=sum(r['cpu_seconds'] for r in records),
        chunk_files=[c['target'] for c in chunks],chunk_seconds_definition='sum of individual job wall times',
        predictions_sha256=digest(target.with_suffix('.npz')))
    if generate:row['paths']=len(job['seeds'])
    else:
        p=combined['observed_probability'];cut=row['training_trials']
        row.update(seeds=job['seeds'],mean_nll=pooled_nll(p),training_nll=pooled_nll(p[:,:cut]),test_nll=pooled_nll(p[:,cut:]))
    write(target.with_suffix('.json'),row)
    return row


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['prepare','search','evaluate','autonomous']);p.add_argument('--config',type=Path)
    p.add_argument('--data',type=Path);p.add_argument('--fit',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--worker',type=Path);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.worker:worker(args.worker);return
    if args.output is None:p.error('--output is required')
    if args.phase=='prepare':prepare(args.config,args.output,args.smoke)
    elif args.phase=='search':search(args.config,args.data,args.output,args.smoke)
    elif args.phase in ('evaluate','autonomous'):evaluate(args.fit,args.output,args.phase=='autonomous')
    else:p.error('--phase is required')


if __name__=='__main__':main()
