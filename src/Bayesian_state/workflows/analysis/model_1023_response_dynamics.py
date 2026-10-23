"""Paired dynamic response-power probes at eighteen frozen TRANSFER points.

Run --config YAML --output NEW [--smoke]. Existing baseline simulations and
prefix continuations are referenced by checked hashes. Each altered mechanism
filters its own observed prefix before continuing with generated feedback.
The smoke uses one selected case per condition, two four-particle filters,
two branches, sixteen task-start trials and one numerical worker.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
PARENT = 'src/Bayesian_state/workflows/analysis/model_1023_prefix_forecast.py'
spec = importlib.util.spec_from_file_location('response_dynamics_parent', ROOT/PARENT)
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
read, write, digest, load, verify = parent.read, parent.write, parent.digest, parent.load, parent.verify
ADAPTER = 'src/Bayesian_state/model/modules/readout/category_power.py'


def worker(file: Path) -> None:
    import numpy as np
    from threadpoolctl import threadpool_info
    job = read(file)
    verify(job['bindings'])
    probe = load('response_dynamics_precision', parent.PROBE)
    power = load('src.Bayesian_state.model.modules.readout.category_power', ADAPTER)
    helper = load('src.Bayesian_state.simulation.filtered_continuation', parent.HELPER)
    diagnostics = load('src.Bayesian_state.simulation.continuation_diagnostics', parent.OBSERVER)
    from src.Bayesian_state.model.modules.readout import choice
    raw_read = choice.read_choice_probabilities_from_model
    from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
    from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
    from src.Bayesian_state.utils.seeding import stable_seed
    folder = Path(job['folder'])
    meta = read(folder/'prepared.json')
    verify({str(Path(job['runtime'])/k): v for k,v in meta['frozen_cognitive_source_sha256'].items()})
    verify({str(folder/k): v for k,v in meta['array_sha256'].items()})
    verify({str(folder/'engine.json'): meta['engine_sha256']})
    start, cpu = time.monotonic(), time.process_time()

    def continuation(installed_job, workflow):
        engine = read(folder/'engine.json')
        engine['modules']['hypo_transitions_mod']['kwargs']['intervention'] = job['mechanism']
        engine = apply_fixed_hyperparams_to_engine_config(engine, job['candidate']['hyperparams'])
        with np.load(folder/'training.npz') as training, np.load(folder/'sealed_test.npz') as later:
            prefix, task = parent.forecast_inputs(training, later, job['smoke'])
        kwargs = dict(engine_config=engine, subject_id=meta['subject'], condition=meta['condition'], **prefix,
            particle_count=job['particles'], particle_blocks=1, filter_seed=job['seeds'][0], choice_readout_power=1.,
            processed_data_dir=meta['paths']['processed_dir'], dataset_paths=meta['paths'])
        models, filtered = helper.capture_prefix(**kwargs)
        prediction = filtered.observation_probabilities['prior_t']
        if job['smoke']:
            plain = run_state_model_particle_filter(**kwargs)
            np.testing.assert_array_equal(plain.observation_probabilities['prior_t'], prediction)
            np.testing.assert_array_equal(plain.final_weights, filtered.final_weights)
        elif job['response_exponent'] == 1.:
            with np.load(job['reference']) as saved:
                np.testing.assert_array_equal(saved['probabilities'][0,:len(prefix['choices'])], prediction)
        endpoints = [diagnostics.endpoint_state(m, job['target_rule']) for m in models]
        seeds = [stable_seed(dict(protocol='model1023_compact_prefix_forecast_v1', base=job['settings']['seed'],
                    subject=meta['subject'], filter_seed=job['seeds'][0], branch=i))
                 for i in range(job['settings']['branches_per_filter'])]
        observer = diagnostics.ContinuationObserver(job['target_rule'])

        def observe(model, decision):
            raw, snapshot = raw_read(model, model._pending_trial.perceived_stimulus, return_snapshot=True)
            np.testing.assert_allclose(power.category_power(raw, job['response_exponent']),
                                       decision.observed_probabilities, atol=1e-12, rtol=0)
            row = observer(model, replace(decision, observed_probabilities=raw, rule_snapshot=snapshot))
            row['cognitive_probabilities'] = raw.copy()
            return row

        generate = dict(models=models, weights=filtered.final_weights, observed_choices=prefix['choices'],
                        observed_feedback=prefix['feedback'], **task, condition=meta['condition'])
        arrays = helper.generate_continuations(**generate, branch_seeds=seeds, observer=observe)
        if job['smoke']:
            plain = helper.generate_continuations(**generate, branch_seeds=seeds)
            reverse = helper.generate_continuations(**generate, branch_seeds=seeds[::-1], observer=observe)
            for k in plain:
                np.testing.assert_array_equal(plain[k], arrays[k], err_msg='Observer changed '+k)
            for k, value in arrays.items():
                other = reverse[k] if k in ('categories','terminal_weights') else reverse[k][::-1]
                np.testing.assert_array_equal(value, other, err_msg='Branch order changed '+k)
        arrays.update({'endpoint_'+k: np.asarray([e[k] for e in endpoints]) for k in endpoints[0]})
        arrays.update(prefix_probabilities=prediction, filter_seed=np.asarray(job['seeds'][0]))
        target = Path(job['target']); target.parent.mkdir(parents=True, exist_ok=True)
        with target.with_suffix('.npz').open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        write(target.with_suffix('.json'), dict(status='complete', subject=meta['subject'], condition=meta['condition'],
            candidate=job['candidate'], case_id=job['case_id'], roles=job['roles'], cohort=job['cohort'],
            filter_seed=job['seeds'][0], particles=job['particles'], paths=len(seeds),
            training_trials=len(prefix['choices']), forecast_trials=len(task['categories']), target_rule=job['target_rule'],
            response_exponent=job['response_exponent'], smoke=job['smoke'],
            independent_prefix_replay=job['smoke'], archived_prefix_exact=job['response_exponent']==1. and not job['smoke'],
            branch_order_check=job['smoke'], observer_passivity_check=job['smoke'],
            seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu, numeric_pools=threadpool_info(),
            predictions_sha256=digest(target.with_suffix('.npz'))))

    original = probe.driver.io.score
    if job['dynamic_role'] == 'prefix':
        probe.driver.io.score = continuation
    try:
        with power.category_power_readout(job['response_exponent']) as audit:
            probe.worker(file)
    finally:
        probe.driver.io.score = original
    write(Path(job['target']).with_suffix('.response.json'), dict(**audit, role=job['dynamic_role'],
        source_sha256={ENTRY: digest(ROOT/ENTRY), ADAPTER: digest(ROOT/ADAPTER)}))
    verify(job['bindings'])


def prepare(config: Path, output: Path, smoke: bool) -> tuple[dict,list[dict]]:
    import yaml
    settings = yaml.safe_load(config.read_text())
    if (settings['subjects'] != [108,205,306,110,209,309] or settings['filters'] != 8
            or settings['particles'] != 1024 or settings['branches_per_filter'] != 16
            or settings['autonomous_paths'] != 128 or settings['exponent'] != 2.):
        raise ValueError('Preserve the registered six-person dynamic comparison')
    source, previous = ROOT/settings['evaluation'], ROOT/settings['previous_prefix']
    plan, done = read(source/'cases.json'), read(source/'completion.json')
    if done['status'] != 'complete' or done['smoke'] or digest(source/'cases.json') != done['cases_sha256']:
        raise ValueError('Require frozen completed evaluations')
    verify(plan['inputs_sha256'])
    old = read(previous/'completion.json')
    if old['status'] != 'complete' or old['smoke'] or digest(previous/'protocol.json') != old['protocol_sha256']:
        raise ValueError('Require completed original prefix study')
    verify(read(previous/'protocol.json')['source_sha256'])
    old_jobs = {}
    for f in sorted((previous/'jobs').glob('*.json')):
        job = read(f)
        old_jobs.setdefault(job['case_id'], []).append(str(f.resolve()))
    cases = [deepcopy(c) for c in plan['cases']]
    if len(cases) != 18 or sorted(set(c['subject'] for c in cases)) != sorted(settings['subjects']):
        raise ValueError('Need all eighteen registered parameter points')
    effective = dict(settings)
    if smoke:
        cases = [next(c for c in cases if c['condition']==condition and 'selected' in c['roles']) for condition in (1,2,3)]
        effective.update(filters=2, particles=4, branches_per_filter=2, autonomous_paths=2, workers=1)
    hashes = {str(ROOT/k): digest(ROOT/k) for k in (ENTRY,ADAPTER,PARENT,parent.HELPER,parent.OBSERVER)}
    inputs = {str(p.resolve()): digest(p) for p in [config,source/'cases.json',source/'completion.json',previous/'protocol.json']}
    jobs, records = [], []
    for c in cases:
        own = Path(c['evaluation'])
        original = [read(f) for f in sorted((own/'jobs').glob('*.json'))
                    if f.name.startswith(('AUTO_','evaluate_'))]
        original = [j for j in original if j['candidate']['id']==c['candidate']]
        conditional = [j for j in original if j['phase']=='evaluate']
        autonomous = [j for j in original if j['phase']=='autonomous']
        if len(conditional)!=8 or sum(len(j['seeds']) for j in autonomous)!=128:
            raise ValueError('Original job bank differs from paired budget')
        if [j['seeds'][0] for j in conditional] != c['reference_seeds']['conditional']:
            raise ValueError('Original filter seed order changed')
        record, = read(own/'autonomous/evaluation.json')['rows']
        c.update(target_rule=record['target_rule'], baseline_task_start=str((own/'autonomous').resolve()),
                 baseline_prefix_jobs=[] if smoke else old_jobs.get(c['case_id'],[]))
        records.append(c)
        for file in c['baseline_prefix_jobs']:
            saved = read(Path(file)); target = Path(saved['target']).with_suffix('.npz')
            sha = read(target.with_suffix('.json'))['predictions_sha256']
            verify({str(target):sha})
            inputs.update({file:digest(Path(file)),str(target):sha})
        for exponent in (1.,2.):
            templates = []
            if exponent==2. or smoke:
                for original_job in autonomous[:1] if smoke else autonomous:
                    j=deepcopy(original_job)
                    if smoke:j['seeds']=j['seeds'][:2]
                    j['dynamic_role']='task_start'; templates.append(j)
            if exponent==2. or not c['baseline_prefix_jobs']:
                for original_job in conditional[:effective['filters']]:
                    j=deepcopy(original_job);j['dynamic_role']='prefix';templates.append(j)
            for j in templates:
                for k in ('decision_basis_target','decision_basis_sources','evaluation_route_sources'):
                    j.pop(k,None)
                ref=Path(j['target']).with_suffix('.npz')
                expected=read(ref.with_suffix('.json'))['predictions_sha256']
                bindings=dict(hashes, **{str(ROOT/k):v for k,v in j['source_sha256'].items()})
                bindings[str(ref)]=expected
                verify(bindings)
                name=f"{c['case_id']}_Q{int(exponent)}_{j['dynamic_role']}_{j['name'].rsplit('_',1)[-1]}"
                j.update(name=name, case_id=c['case_id'], response_exponent=exponent, settings=effective,
                    roles=c['roles'], cohort=c['cohort'], target_rule=c['target_rule'], smoke=smoke,
                    particles=effective['particles'], reference=str(ref), bindings=bindings,
                    target=str((output/'chunks'/name).resolve()))
                jobs.append(j)
    output.mkdir(parents=True);(output/'jobs').mkdir()
    sources=dict(hashes)
    for job in jobs:sources.update({str(ROOT/k):v for k,v in job['source_sha256'].items()})
    for file in sources:
        dest=output/'source'/Path(file).relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,dest)
    shutil.copy2(config,output/'protocol.yaml')
    protocol=dict(settings=effective,smoke=smoke,cases=records,jobs=len(jobs),
        workers=min(effective['workers'],len(os.sched_getaffinity(0)),len(jobs)),
        source_sha256=sources,inputs_sha256=inputs,python=sys.version,
        versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','pandas','PyYAML')},
        conditioning='Each response mechanism filters its own original observed prefix',
        uncertainty_unit='Independent task-start trajectory or prefix-filter cluster',
        data_use='Development comparison; later behavior never enters generated feedback',
        onset='Response transformation active from trial one; no conditional-state transplant')
    write(output/'protocol.json',protocol)
    return protocol,jobs


def run(config: Path, output: Path, smoke: bool) -> None:
    if output.exists():raise FileExistsError('Use a new dynamic comparison directory')
    protocol,jobs=prepare(config,output,smoke)
    start=time.monotonic()
    print(f"{len(jobs)} jobs, {protocol['workers']} single-thread workers",flush=True)

    def launch(job):
        file=output/'jobs'/(job['name']+'.json');write(file,job)
        launcher=('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                  "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                 MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='response_dynamics_') as cache,file.with_suffix('.log').open('x') as log:
            env.update(NUMBA_CACHE_DIR=cache,MPLCONFIGDIR=cache)
            subprocess.run([sys.executable,'-c',launcher,job['runtime'],str(ROOT/ENTRY),'--worker',str(file.resolve())],
                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return {**read(Path(job['target']).with_suffix('.json')), 'name':job['name'],
                'role':job['dynamic_role'], 'response_exponent':job['response_exponent']}

    rows=[]
    # Long prefix filters start first; all jobs still share one numeric queue.
    jobs.sort(key=lambda j: j['dynamic_role']!='prefix')
    with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
        for future in as_completed([pool.submit(launch,j) for j in jobs]):
            rows.append(future.result())
            if len(rows)%16==0 or len(rows)==len(jobs):print(f'{len(rows)}/{len(jobs)} complete',flush=True)
    verify(protocol['inputs_sha256']);verify(protocol['source_sha256'])
    write(output/'completion.json',dict(status='complete',smoke=smoke,rows=rows,jobs=len(jobs),
        seconds=time.monotonic()-start,protocol_sha256=digest(output/'protocol.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('config','output','worker'):parser.add_argument('--'+key,type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.config and args.output:run(args.config,args.output,args.smoke)
    else:parser.error('Supply --config and --output, or --worker')
