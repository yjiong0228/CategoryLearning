"""Bounded full-record joint parameter sets with autonomous shape distributions.

Run --phase prepare and --phase run with the same --output.
--phase smoke uses a separate new output and one short single-worker case.
Scientific workers reuse the frozen shared runtime and Model 1024 adapters.
Full records use the existing evaluate reader, joining the two archived storage
parts; no chronological split is used for this study's parameter decisions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time

if __package__ in (None, ''):
    import src.Bayesian_state.workflows.analysis as _analysis
    import src.Bayesian_state.metrics as _metrics
    _root = Path(__file__).resolve().parents[4]
    _analysis.__path__ = [str(_root/'src/Bayesian_state/workflows/analysis'), *_analysis.__path__]
    _metrics.__path__ = [str(_root/'src/Bayesian_state/metrics'), *_metrics.__path__]
    __package__ = 'src.Bayesian_state.workflows.analysis'

import numpy as np
import pandas as pd
import yaml

from . import model_1023_execution_study as study
from ...metrics.trajectory_shape import (describe_ensemble, sequence_score, paired_difference,
                                        shape_check, wilson_interval)
from ...metrics.autonomous_fit import energy_u, behavior_features
from ...optimization.search.adaptive_proposals import candidate
from ...utils.seeding import stable_seed

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
METRIC = 'src/Bayesian_state/metrics/trajectory_shape.py'
H = study.driver.H
KEYS = dict(p0=H+'search_controller.baseline_probability', error_gain=H+'search_controller.error_gain',
            failure_decay=H+'search_controller.failure_decay', global_search=H+'search_controller.global_search',
            capacity=H+'max_capacity', retention=H+'intervention.decay', inertia=H+'execution_inertia',
            beta_init='engine.modules.beta_mod.kwargs.beta_init',
            learning_rate='engine.modules.beta_mod.kwargs.learning_rate',
            gamma='engine.modules.memory_mod.kwargs.gamma')
LOW_COST = {110:269.,205:529.,306:611.,129:114.,229:812.,317:887.}
HIGH_COST = {110:439.,205:1051.,306:1213.,129:305.,229:918.,317:773.}
PATH_COST = {110:1.2,205:3.2,306:3.6,129:1.,229:3.7,317:2.2}
OLD_CAPACITY = {110:2,205:6,306:6,129:6,229:3,317:3}


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources() -> dict:
    return dict(study.sources(study.driver.io), **{p:digest(ROOT/p) for p in [ENTRY, METRIC]})


def decision(output: Path, event: str, **fields) -> None:
    with (output/'decisions.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event=event, unix_time=time.time(), **fields), ensure_ascii=False)+'\n')


def seed_bank(p: dict, sid: int, role: str, count: int) -> list[int]:
    return [stable_seed(dict(protocol=p['protocol'], base=p['seed'], subject=sid, role=role, repeat=i))
            for i in range(count)]


def canonical(parameters: dict, condition: int, source: str) -> dict:
    point = deepcopy(parameters)
    if point[KEYS['error_gain']] == 0:
        point[KEYS['failure_decay']] = 0.
    if condition != 3:
        point[KEYS['gamma']] = 1.
    point[KEYS['capacity']] = int(point[KEYS['capacity']])
    return candidate(point, source)


def load_observed(folder: Path) -> tuple[dict, dict]:
    meta = read(folder/'prepared.json')
    parts = []
    for name in ('training.npz','sealed_test.npz'):
        if digest(folder/name) != meta['array_sha256'][name]:
            raise ValueError('Input identity changed')
        with np.load(folder/name) as z:
            parts.append({k:z[k] for k in z.files})
    full = {k:np.concatenate([x[k] for x in parts]) for k in parts[0]}
    if len(full['choices']) != meta['trials']:
        raise ValueError('Full-record reader omitted trials')
    if not np.array_equal(full['choices']==full['categories'], full['feedback']==1):
        raise ValueError('Task category/feedback mismatch')
    return meta, full


def initial_bank(p: dict, case: dict, originals: list[dict], joint: dict | None) -> list[dict]:
    sid, condition = case['subject'], case['condition']
    bank = {}
    def add(point, source):
        x = canonical(point, condition, source)
        bank.setdefault(x['id'],x)
    add(case['template']['candidate']['hyperparams'],'original_choice_anchor')
    if joint:
        add(joint['candidate']['hyperparams'],'previous_joint_anchor')
    for row in sorted(originals, key=lambda r:(r['mean_nll'],r['candidate']['id'])):
        add(row['candidate']['hyperparams'],'previous_confirmation')
        if len(bank) >= 4:
            break
    rng = np.random.default_rng(seed_bank(p,sid,'initial_proposals',1)[0])
    keys = [k for k in KEYS if k != 'gamma' or condition == 3]
    while len(bank) < p['search']['initial']:
        point = deepcopy(case['template']['candidate']['hyperparams'])
        for name in keys:
            values = p['parameter_values'][name]
            point[KEYS[name]] = values[int(rng.integers(len(values)))]
        add(point,'coarse_spread')
    return list(bank.values())


def prepare(config: Path, output: Path, smoke: bool = False) -> dict:
    if (output/'protocol.json').exists():
        raise FileExistsError('Protocol already exists; run or report this registered study')
    p = yaml.safe_load(config.read_text())
    if sum(v for k,v in p['cpu_hours'].items() if k!='total') != p['cpu_hours']['total']:
        raise ValueError('Stage budgets must sum to the total')
    if smoke:
        p['subjects'] = p['subjects'][:1]
        p['search'].update(initial=2,rounds=0,maximum_candidates=2,particles=2,repeats=2,
                           finalists=2,confirmation_particles=4,confirmation_repeats=2,extra_confirmation_repeats=0,
                           particle_check_particles=8,particle_check_repeats=2)
        p['generation'].update(screening_paths=8,confirmation_paths=8,chunk=2)
        p['parallel_budget'] = 1
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT/p['source_study']
    previous = read(source/'protocol.json')
    expected = study.sources(study.driver.io)
    if expected != previous['source_sha256']:
        raise ValueError('Historical cognitive adapters differ; audit before reuse')
    templates = read(source/'autonomous_jobs.json')['jobs']
    joint_file = ROOT/p['joint_study']/'nominees.json'
    joint_rows = read(joint_file)['nominees']
    cases = []
    for definition in p['subjects']:
        sid = definition['id']
        template = deepcopy(next(j for j in templates if j['subject']==sid and j['model']=='STICKY'))
        meta, arrays = load_observed(Path(template['folder']))
        if meta['condition'] != definition['condition']:
            raise ValueError('Subject-condition mismatch')
        records = [read(f) for f in (source/'confirmation'/f'subject_{sid}'/'STICKY').glob('*.json')]
        records = [r for r in records if 'mean_nll' in r and 'candidate' in r]
        joint = next((r for r in joint_rows if r['subject']==sid and r['mode']=='STICKY'),None)
        case = dict(subject=sid,condition=meta['condition'],role=definition['role'],
                    trials=8 if smoke else meta['trials'],template=template,
                    data_hashes=meta['array_sha256'],input_metadata_sha256=digest(Path(template['folder'])/'prepared.json'),
                    original_trials=meta['trials'],first_choice=int(arrays['choices'][0]),last_choice=int(arrays['choices'][-1]))
        case['initial_candidates'] = initial_bank(p,case,records,joint)[:p['search']['initial']]
        cases.append(case)
    snapshot = output/'source'
    for relative in sources():
        dest = snapshot/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,dest)
    shutil.copy2(config,output/'registered_config.yaml')
    protocol = dict(policy=p,cases=cases,smoke=smoke,sources=sources(),created_unix=time.time(),
                    data_use='all valid choices, first through last; development',
                    task_schedule='recorded stimuli and fixed length; response-dependent feedback',
                    oral_used=False,versions={m:importlib.metadata.version(m) for m in ['numpy','scipy','pandas','PyYAML']},
                    python=sys.version,source_nominees_sha256=digest(source/'nominees.json'),
                    joint_nominees_sha256=digest(joint_file))
    write(output/'protocol.json',protocol)
    decision(output,'protocol_frozen',subjects=[c['subject'] for c in cases],smoke=smoke)
    return protocol


def make_jobs(protocol: dict, case: dict, points: list[dict], output: Path, role: str,
              *, particles: int | None = None, repeats: int | None = None, paths: int | None = None) -> list[dict]:
    p = protocol['policy']; sid = case['subject']
    seeds = seed_bank(p,sid,role,paths if paths is not None else repeats)
    size = p['generation']['chunk'] if paths is not None else 1
    jobs = []
    for point in points:
        for offset in range(0,len(seeds),size):
            name = f'{role}_S{sid}_{point["id"]}_{offset:03d}'
            job = deepcopy(case['template'])
            mode = 'STICKY' if paths is not None or point['hyperparams'][KEYS['inertia']] else 'MIX'
            job.update(name=name,phase='autonomous' if paths is not None else ('training' if protocol['smoke'] else 'evaluate'),
                       model='COMPACT',comparison_arm='ADAPT',execution_mode=mode,candidate=point,
                       seeds=seeds[offset:offset+size],smoke=protocol['smoke'],
                       source_sha256=study.sources(study.driver.io),region_sources=protocol['sources'],
                       target=str((output/'numerical'/name).resolve()),region_role=role,
                       expected_trials=(16 if paths is not None else 8) if protocol['smoke'] else case['trials'])
            job['mechanism']['capacity']='evidence'
            if particles is not None:
                job['particles']=particles
            jobs.append(job)
    return jobs


def worker(file: Path) -> None:
    job = read(file)
    if sources() != job['region_sources']:
        raise ValueError('Registered source changed')
    limit = int(job['cpu_limit_seconds'])
    resource.setrlimit(resource.RLIMIT_CPU,(limit,limit+1))
    def exhausted(*_):
        raise TimeoutError('Registered CPU reservation exhausted')
    signal.signal(signal.SIGXCPU,exhausted)
    status = 'failed'
    try:
        study.worker(file)
        with np.load(Path(job['target']).with_suffix('.npz')) as z:
            length = z['choices'].shape[-1] if job['phase']=='autonomous' else z['observed_probability'].shape[-1]
            if length != job['expected_trials']:
                raise ValueError('First-to-last record audit failed')
        status='complete'
    finally:
        usage=resource.getrusage(resource.RUSAGE_SELF)
        write(file.with_suffix('.usage.json'),dict(status=status,stage=job['budget_stage'],
              cpu_seconds=usage.ru_utime+usage.ru_stime,job=job['name'],source_sha256=sources()))


class BudgetQueue:
    """Reserve CPU before launch and enforce each reservation in its worker."""
    def __init__(self, output: Path, protocol: dict):
        self.output,self.protocol=output,protocol
        self.directory=output/'jobs';self.directory.mkdir(exist_ok=True)
        self.analysis_file=output/'coordinator_usage.json'
        self.analysis_prior=read(self.analysis_file)['cpu_seconds'] if self.analysis_file.exists() else 0.
        usage=resource.getrusage(resource.RUSAGE_SELF)
        self.analysis_start=usage.ru_utime+usage.ru_stime
        remaining=protocol['policy']['cpu_hours']['analysis']*3600-self.analysis_prior
        if remaining<=0:
            raise RuntimeError('Coordinator analysis CPU budget exhausted')
        limit=int(np.ceil(self.analysis_start+remaining))
        resource.setrlimit(resource.RLIMIT_CPU,(limit,limit+1))

    def used(self) -> dict:
        totals={k:0. for k in self.protocol['policy']['cpu_hours'] if k!='total'}
        for f in self.directory.glob('*.usage.json'):
            row=read(f);totals[row['stage']]+=row['cpu_seconds']
        usage=resource.getrusage(resource.RUSAGE_SELF)
        totals['analysis']+=self.analysis_prior+usage.ru_utime+usage.ru_stime-self.analysis_start
        write(self.analysis_file,dict(cpu_seconds=totals['analysis']))
        return totals

    def estimate(self,job: dict) -> int:
        if job['smoke']:
            return 180
        sid=job['subject'];k=job['candidate']['hyperparams'][KEYS['capacity']]
        if job['phase']=='autonomous':
            base=PATH_COST[sid]*len(job['seeds'])*max(1.,k/OLD_CAPACITY[sid])
        else:
            base=(LOW_COST[sid] if job['particles']<=128 else HIGH_COST[sid]*job['particles']/512)
            base*=max(.7,k/OLD_CAPACITY[sid])
        return int(max(90,base*2.5+30))

    def launch(self,job: dict) -> dict:
        file=self.directory/(job['name']+'.json')
        write(file,job)
        launcher=('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                  "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='model1024_region_') as temp:
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                     MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
                     NUMBA_CACHE_DIR=temp,MPLCONFIGDIR=temp)
            with file.with_suffix('.log').open('x') as stream:
                result=subprocess.run([sys.executable,'-c',launcher,job['runtime'],str(ROOT/ENTRY),
                                       '--worker',str(file.resolve())],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        usage=file.with_suffix('.usage.json')
        if not usage.exists():
            write(usage,dict(status='terminated_without_receipt',stage=job['budget_stage'],
                            cpu_seconds=job['cpu_limit_seconds'],job=job['name']))
        return dict(name=job['name'],returncode=result.returncode,**read(usage))

    def execute(self,jobs: list[dict],stage: str) -> None:
        p=self.protocol['policy'];pending=[]
        for job in jobs:
            usage=self.directory/(job['name']+'.usage.json')
            if usage.exists():
                continue
            if (self.directory/(job['name']+'.log')).exists():
                raise RuntimeError('Interrupted job requires an explicit resume audit: '+job['name'])
            pending.append(job)
        workers=min(p['parallel_budget'],len(os.sched_getaffinity(0)),max(1,len(pending)))
        in_flight={};reserved=0.;done=0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while pending or in_flight:
                used=self.used()
                remaining=min(p['cpu_hours'][stage]*3600-used[stage],p['cpu_hours']['total']*3600-sum(used.values()))
                submitted=False
                while pending and len(in_flight)<workers:
                    index=next((i for i,j in enumerate(pending) if self.estimate(j)<=remaining-reserved),None)
                    if index is None:
                        break
                    job=pending.pop(index);limit=self.estimate(job)
                    job.update(budget_stage=stage,cpu_limit_seconds=limit)
                    future=pool.submit(self.launch,job);in_flight[future]=limit;reserved+=limit;submitted=True
                write(self.output/'progress.json',dict(stage=stage,pending=len(pending),running=len(in_flight),
                      completed_this_batch=done,cpu_hours={k:v/3600 for k,v in used.items()},unix_time=time.time()))
                if not in_flight:
                    if pending:
                        decision(self.output,'budget_stop',stage=stage,unsubmitted=[j['name'] for j in pending])
                    break
                finished,_=wait(in_flight,timeout=15,return_when=FIRST_COMPLETED)
                for future in finished:
                    reserved-=in_flight.pop(future);row=future.result();done+=1
                    if row['returncode']:
                        decision(self.output,'worker_failure',**row)
                    if done%8==0 or not pending and not in_flight:
                        print(stage,done,'complete; pending',len(pending),'running',len(in_flight),flush=True)


def completed_arrays(jobs: list[dict], key: str) -> np.ndarray | None:
    parts=[]
    for job in jobs:
        path=Path(job['target']);meta=path.with_suffix('.json');data=path.with_suffix('.npz')
        receipt=path.parent.parent/'jobs'/(job['name']+'.usage.json')
        if receipt.exists() and read(receipt)['status']!='complete':
            return None
        if not meta.exists() or not data.exists():
            return None
        row=read(meta)
        if row['candidate']['hyperparams']!=job['candidate']['hyperparams'] or digest(data)!=row['predictions_sha256']:
            raise ValueError('Completed artifact identity changed')
        with np.load(data) as z:
            np.testing.assert_array_equal(z['seeds'],job['seeds'])
            parts.append(z[key])
    return np.concatenate(parts) if parts else None


def evaluate_bank(protocol: dict,case: dict,bank: list[dict],output: Path,queue: BudgetQueue,
                  filter_role: str,generation_role: str,particles: int,repeats: int,paths: int,
                  stage: str, execute: bool = True) -> list[dict]:
    p=protocol['policy'];sid=case['subject']
    jobs=make_jobs(protocol,case,bank,output,filter_role,particles=particles,repeats=repeats)
    autos=make_jobs(protocol,case,bank,output,generation_role,paths=paths)
    if execute:
        queue.execute(jobs,stage)
        queue.execute(autos,'generation')
    _,observed=load_observed(Path(case['template']['folder']))
    n=16 if protocol['smoke'] else case['trials']
    categories=observed['categories'][:n]
    empirical=describe_ensemble(observed['choices'][None,:n],categories,p['shape'])[0]
    rows=[]
    for point in bank:
        filters=[j for j in jobs if j['candidate']['id']==point['id']]
        generations=[j for j in autos if j['candidate']['id']==point['id']]
        probability=completed_arrays(filters,'observed_probability')
        choices=completed_arrays(generations,'choices')
        row=dict(subject=sid,condition=case['condition'],role=case['role'],candidate=point,
                 filter_jobs=filters,generation_jobs=generations,filter_complete=probability is not None,
                 generation_complete=choices is not None)
        if probability is not None:
            row['choice_score']=sequence_score(probability)
        if choices is not None:
            records=describe_ensemble(choices,categories,p['shape'])
            features=np.array([r['features'] for r in records]);actual=np.array(empirical['features'])
            row.update(shape_energy=energy_u(features,actual),shape_check=shape_check(features,actual,p['generation']['compatibility_alpha']),
                       shape_labels={label:sum(r['label']==label for r in records) for label in sorted({r['label'] for r in records})},
                       observed_shape=empirical)
            location=output/'shapes'/f'{generation_role}_S{sid}_{point["id"]}.json'
            write(location,dict(paths=records,observed=empirical,policy=p['shape']))
            row['shape_file']=str(location.resolve())
            classical=behavior_features(choices,categories,n_categories=2 if case['condition']==1 else 4)
            obs_classical=behavior_features(observed['choices'][None,:n],categories,n_categories=2 if case['condition']==1 else 4)[0]
            row['ordered_energy_diagnostic']=energy_u(classical,obs_classical)
        rows.append(row)
    return rows


def proposals(p: dict,case: dict,bank: list[dict],scores: list[dict],round_index: int) -> list[dict]:
    valid=[r for r in scores if r.get('choice_score') and r.get('shape_energy')]
    if not valid:
        return []
    best=min(valid,key=lambda r:r['choice_score']['nll'])
    eligible=[r for r in valid if r['choice_score']['nll']<=best['choice_score']['nll']+.02]
    shape=min(eligible,key=lambda r:r['shape_energy']['value'])
    anchors=[best['candidate'],shape['candidate']]
    rng=np.random.default_rng(seed_bank(p,case['subject'],f'proposal_{round_index}',1)[0])
    seen={x['id'] for x in bank};new=[]
    names=[k for k in KEYS if k!='gamma' or case['condition']==3]
    for attempt in range(300):
        point=deepcopy(anchors[attempt%2]['hyperparams'])
        chosen=rng.choice(names,size=1 if attempt%2==0 else 2,replace=False)
        for name in chosen:
            key=KEYS[name];current=point[key]
            values=sorted(set(p['parameter_values'][name]+[current]))
            index=values.index(current)
            neighbors=[values[i] for i in (index-1,index+1) if 0<=i<len(values)]
            target=neighbors[int(rng.integers(len(neighbors)))]
            if name=='capacity':
                target=current+np.sign(target-current) if round_index>0 else target
            elif round_index>0 and rng.random()<.5:
                target=np.sqrt(target*current) if name in ['learning_rate','beta_init','inertia','error_gain'] and target*current>0 else (target+current)/2
            point[key]=float(target)
        # A registered outward check uses the final round's first proposal.
        if round_index==p['search']['rounds']-1 and attempt==0:
            for name,upper,outside in [('capacity',8,10),('inertia',8,12),('error_gain',8,12)]:
                if best['candidate']['hyperparams'][KEYS[name]]>=upper:
                    point=deepcopy(best['candidate']['hyperparams']);point[KEYS[name]]=outside;break
        c=canonical(point,case['condition'],f'adaptive_round_{round_index}')
        if c['id'] not in seen:
            seen.add(c['id']);new.append(c)
        if len(new)>=p['search']['proposals_per_round']:
            break
    return new[:max(0,p['search']['maximum_candidates']-len(bank))]


def shortlist(rows: list[dict],count: int) -> list[dict]:
    valid=[r for r in rows if r.get('choice_score') and r.get('shape_energy')]
    if not valid:
        return []
    minimum=min(r['choice_score']['nll'] for r in valid)
    eligible=[r for r in valid if r['choice_score']['nll']<=minimum+.02]
    selected={}
    for ordering in [sorted(valid,key=lambda r:r['choice_score']['nll']),
                     sorted(eligible,key=lambda r:r['shape_energy']['value'])]:
        r=ordering[0];selected[r['candidate']['id']]=r['candidate']
    # Keep distant, plausible combinations to describe a region rather than
    # spending every confirmation on numerically adjacent winners.
    candidates=sorted(eligible,key=lambda r:(r['shape_energy']['value'],r['choice_score']['nll']))
    while len(selected)<min(count,len(candidates)):
        remaining=[r for r in candidates if r['candidate']['id'] not in selected]
        def distance(r):
            v=r['candidate']['hyperparams']
            return min(sum(abs(float(v[k])-float(a['hyperparams'][k]))/(1+abs(float(v[k]))+abs(float(a['hyperparams'][k])))
                           for k in KEYS.values()) for a in selected.values())
        r=max(remaining,key=distance);selected[r['candidate']['id']]=r['candidate']
    return list(selected.values())


def classify(rows: list[dict],policy: dict) -> list[dict]:
    valid=[r for r in rows if r.get('choice_score')]
    if not valid:
        return rows
    reference=min(valid,key=lambda r:r['choice_score']['nll'])
    ref=completed_arrays(reference['filter_jobs'],'observed_probability')
    for row in rows:
        row['membership']={}
        if not row.get('choice_score') or not row.get('shape_check'):
            row['joint_status']='incomplete';continue
        probability=completed_arrays(row['filter_jobs'],'observed_probability')
        difference=paired_difference(probability,ref)
        score=row['choice_score'];refscore=reference['choice_score']
        stable=(max(score['maximum_system_weight'],refscore['maximum_system_weight'])<=policy['search']['maximum_system_weight']
                and 2*max(score['mcse'],refscore['mcse'])<=policy['search']['maximum_mc_resolution']
                and 2*difference['mcse']<=policy['search']['maximum_mc_resolution'])
        for compared in [row,reference]:
            high=compared.get('particle_check_score');base=compared['choice_score']
            if high:
                gap=abs(high['nll']-base['nll']);se=float(np.hypot(high['mcse'],base['mcse']))
                tolerance=policy['search']['particle_check_tolerance']
                status='stable' if gap+2*se<=tolerance else ('sensitive' if gap-2*se>tolerance else 'boundary')
                if high['maximum_system_weight']>policy['search']['maximum_system_weight']:
                    status='system_weight_unresolved'
                compared['particle_check']=dict(status=status,absolute_difference=gap,combined_mcse=se,
                                                tolerance=tolerance,particles=policy['search']['particle_check_particles'])
            else:
                compared['particle_check']=dict(status='incomplete')
            stable=stable and compared['particle_check']['status']=='stable'
        row.update(reference_id=reference['candidate']['id'],likelihood_difference=difference,numerically_resolved=stable)
        row['preferred_numerical_precision']=bool(stable and 2*max(score['mcse'],refscore['mcse'],difference['mcse'])
                                                  <=policy['search']['preferred_mc_resolution'])
        for tolerance in policy['search']['score_tolerances']:
            if not stable:
                state='numerically_undetermined'
            elif difference['difference']+2*difference['mcse']<=tolerance:
                state='retained' if row['shape_check']['status']=='compatible' else row['shape_check']['status']
            elif difference['difference']-2*difference['mcse']>tolerance:
                state='likelihood_outside'
            else:
                state='likelihood_boundary'
            row['membership'][str(tolerance)]=state
        row['joint_status']=row['membership']['0.01']
    return rows


def evaluate_many(protocol, batches, output, queue, filter_role, generation_role,
                  particles, repeats, paths, stage):
    """Submit ready cases together, then summarize after all numerical jobs."""
    filters=[];generations=[]
    for case, bank in batches:
        filters.extend(make_jobs(protocol,case,bank,output,filter_role,particles=particles,repeats=repeats))
        generations.extend(make_jobs(protocol,case,bank,output,generation_role,paths=paths))
    queue.execute(filters,stage)
    queue.execute(generations,'generation')
    return {case['subject']:evaluate_bank(protocol,case,bank,output,queue,filter_role,generation_role,
            particles,repeats,paths,stage,False) for case,bank in batches}


def run(output: Path) -> None:
    protocol=read(output/'protocol.json');p=protocol['policy']
    if sources()!=protocol['sources']:
        raise ValueError('Frozen scientific sources changed')
    queue=BudgetQueue(output,protocol)
    # Calibrate fixed anchors before searching. Fresh full-record 128/512 runs
    # are independent of proposal and confirmation seed banks.
    calibration=[]
    for case in protocol['cases']:
        points=case['initial_candidates'][:2]
        calibration+=make_jobs(protocol,case,points,output,'calibration_low',particles=p['search']['particles'],repeats=2)
        calibration+=make_jobs(protocol,case,points[:1],output,'calibration_high',particles=p['search']['confirmation_particles'],repeats=4 if not protocol['smoke'] else 2)
    queue.execute(calibration,'calibration')
    calibration_rows=[]
    for case in protocol['cases']:
        for point in case['initial_candidates'][:2]:
            item=dict(subject=case['subject'],candidate=point['id'])
            for role in ['calibration_low','calibration_high']:
                jobs=[j for j in calibration if j['subject']==case['subject'] and j['candidate']['id']==point['id'] and j['region_role']==role]
                a=completed_arrays(jobs,'observed_probability')
                if a is not None:item[role]=sequence_score(a)
            calibration_rows.append(item)
    write(output/'calibration.json',calibration_rows)
    for item in calibration_rows:
        high=item.get('calibration_high')
        if high and (2*high['mcse']>p['search']['maximum_mc_resolution'] or
                     high['maximum_system_weight']>p['search']['maximum_system_weight']):
            decision(output,'calibration_precision_warning',subject=item['subject'],candidate=item['candidate'],
                     action='exploratory search; independent confirmation must resolve numerical uncertainty',score=high)
    banks={c['subject']:list(c['initial_candidates']) for c in protocol['cases']}
    scores={c['subject']:[] for c in protocol['cases']}
    # Subject order alternates ordinary and difficult cases across conditions.
    ordered=sorted(protocol['cases'],key=lambda c:(c['condition'],c['role']=='challenge'))
    for round_index in range(p['search']['rounds']+1):
        batches=[]
        for case in ordered:
            sid=case['subject']
            new=banks[sid] if round_index==0 else proposals(p,case,banks[sid],scores[sid],round_index-1)
            if round_index:
                banks[sid].extend(new)
            if not new:continue
            decision(output,'proposal_batch',subject=sid,round=round_index,candidates=[x['id'] for x in new])
            batches.append((case,new))
        results=evaluate_many(protocol,batches,output,queue,'screen','screen_generation',
                              p['search']['particles'],p['search']['repeats'],p['generation']['screening_paths'],'search')
        for sid,rows in results.items():
            scores[sid].extend(rows)
            write(output/f'search_S{sid}.json',scores[sid])
        write(output/'candidate_banks.json',banks)
    final=[];batches=[]
    for case in ordered:
        sid=case['subject'];selected=shortlist(scores[sid],p['search']['finalists'])
        decision(output,'confirmation_shortlist',subject=sid,candidates=[x['id'] for x in selected])
        batches.append((case,selected))
    confirmed=evaluate_many(protocol,batches,output,queue,'confirmation','confirmation_generation',
                            p['search']['confirmation_particles'],p['search']['confirmation_repeats'],
                            p['generation']['confirmation_paths'],'confirmation')
    checks=[]
    for case,selected in batches:
        checks.extend(make_jobs(protocol,case,selected,output,'particle_check',
                                particles=p['search']['particle_check_particles'],repeats=p['search']['particle_check_repeats']))
    queue.execute(checks,'confirmation')
    for case,selected in batches:
        for row in confirmed[case['subject']]:
            jobs=[j for j in checks if j['subject']==case['subject'] and j['candidate']['id']==row['candidate']['id']]
            row['particle_check_jobs']=jobs
            probability=completed_arrays(jobs,'observed_probability')
            if probability is not None:row['particle_check_score']=sequence_score(probability)
    extras=[];extra_by_subject={}
    count=p['search']['extra_confirmation_repeats']
    for case,selected in batches:
        rows=classify(confirmed[case['subject']],p)
        if count and any(r.get('choice_score') and (2*r['choice_score']['mcse']>p['search']['maximum_mc_resolution']
                       or r['choice_score']['maximum_system_weight']>p['search']['maximum_system_weight']) for r in rows):
            jobs=make_jobs(protocol,case,selected,output,'extra_confirmation',
                           particles=p['search']['confirmation_particles'],repeats=count)
            extras.extend(jobs);extra_by_subject[case['subject']]=jobs
    queue.execute(extras,'confirmation')
    for case,selected in batches:
        sid=case['subject'];rows=confirmed[sid]
        jobs=extra_by_subject.get(sid,[])
        # Extend every competitor together: unequal seed banks would invalidate
        # the paired numerical comparison. Preserve partial extras separately.
        if jobs and all(r.get('choice_score') for r in rows) and completed_arrays(jobs,'observed_probability') is not None:
            for row in rows:
                more=[j for j in jobs if j['candidate']['id']==row['candidate']['id']]
                row['filter_jobs'].extend(more)
                row['choice_score']=sequence_score(completed_arrays(row['filter_jobs'],'observed_probability'))
            decision(output,'extra_confirmation_integrated',subject=sid,repeats=count)
        rows=classify(rows,p)
        write(output/f'confirmed_S{sid}.json',rows);final.extend(rows)
    write(output/'completion.json',dict(status='bounded_numerical_stage_complete',smoke=protocol['smoke'],
          subjects=len(protocol['cases']),candidates=sum(len(x) for x in banks.values()),
          confirmed_records=len(final),cpu_hours={k:v/3600 for k,v in queue.used().items()},
          protocol_sha256=digest(output/'protocol.json'),scientific_status='per-candidate resolution and compatibility recorded',
          full_parameter_recovery='deferred',oral_used=False))
    decision(output,'numerical_stage_finished',cpu_hours={k:v/3600 for k,v in queue.used().items()})


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['prepare','run','smoke'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/shared/model1024_individual_regions.yaml')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.phase=='prepare':
        prepare(args.config,args.output)
    elif args.phase=='run':
        run(args.output)
    elif args.phase=='smoke':
        prepare(args.config,args.output,True);run(args.output)
    else:
        parser.error('Select a phase or worker')


if __name__=='__main__':
    main()
