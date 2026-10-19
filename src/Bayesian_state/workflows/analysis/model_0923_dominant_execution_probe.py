"""Compare existing mixture and dominant-rule execution at frozen parameters.

Use --config YAML --output NEW_DIRECTORY [--smoke]. The config declares the
completed SEPARATE autonomous component, path/filter counts and worker budget.
Reuse mixture paths; both policies receive equal particle counts and common
filter seeds. The SELECT policy executes the already selected dominant rule.
It retains the current candidate beliefs, precision and search algorithms.
Smoke uses one worker, one original path per cell, 16 autonomous trials and
two-particle eight-trial filters with an explicit 4/4 validation-only split.
Results include actual execution sidecars, scores, paired Monte Carlo errors,
path measures and input hashes. Oral reports do not enter this screen.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
PROBE = 'src/Bayesian_state/workflows/analysis/model_0923_separate_belief_probe.py'
EXECUTION = 'src/Bayesian_state/model/modules/hypothesis_transition/dominant_rule_execution.py'


def helpers():
    spec = importlib.util.spec_from_file_location('dominant_execution_separate', ROOT / PROBE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def worker(file: Path) -> None:
    probe = helpers()
    cross = probe.helpers()
    online = cross.helpers()
    retained = online.helpers()
    recalled = retained.helpers()
    probe.helpers = lambda: cross
    cross.helpers = lambda: online
    online.helpers = lambda: retained
    retained.helpers = lambda: recalled
    original_validation, original_auto = recalled.base.validation, recalled.refit.autonomous
    v = original_validation()
    job = v.read(file)
    if job.get('execution_policy') not in ('MIX', 'SELECT') or job.get('probe_mechanism') != 'SEPARATE':
        raise ValueError('Declare MIX/SELECT with the fixed SEPARATE belief mechanism')

    def with_execution(function, own_job, workflow):
        module = recalled.base.load('src.Bayesian_state.model.modules.hypothesis_transition.dominant_rule_execution', EXECUTION)
        cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
        enabled = own_job['execution_policy'] == 'SELECT'
        with module.dominant_rule_execution(cls, enabled=enabled) as events:
            function(own_job, workflow)
        target = Path(own_job['target'])
        result = v.read(target.with_suffix('.json'))
        copies = len(own_job['seeds']) * (1 if own_job['phase']=='autonomous' else own_job['particles'])
        expected = {t: copies for t in range(result['trials'])} if enabled else {}
        if Counter(e['trial'] for e in events) != expected:
            raise ValueError('Dominant execution must occur once before every choice, including trial zero')
        v.write(target.with_suffix('.execution.json'), dict(enabled=enabled,
            policy=own_job['execution_policy'], rule_execution_policy=module.POLICY if enabled else 'mixture',
            updates=len(events), uses_existing_dominant=True, task_categories_used_for_selection=False))

    def validation():
        module = original_validation()
        score = module.score
        module.score = lambda own_job, workflow: with_execution(score, own_job, workflow)
        return module

    recalled.base.validation = validation
    recalled.refit.autonomous = lambda own_job, workflow: with_execution(original_auto, own_job, workflow)
    probe.worker(file)


def run(config: Path, output: Path, smoke: bool = False) -> None:
    from src.Bayesian_state.evaluation.model_0923_separate_belief_refit import check_execution
    from src.Bayesian_state.evaluation.model_0923_reentry_refit import match_task, window_estimate
    from src.Bayesian_state.evaluation.model_0923_estimation_scope import paired_difference, validate_generation

    if output.exists():
        raise FileExistsError('Use a new fixed-policy probe directory')
    settings = yaml.safe_load(config.read_text())
    if (settings['subject'] != 208 or settings['arms'] != ['POOL', 'SEARCH']
            or not 4 <= settings['paths'] <= 128 or settings['paths'] % 4
            or not 2 <= settings['filter_repeats'] <= 8 or not 4 <= settings['particles'] <= 512
            or not 1 <= settings['parallel_budget'] <= 128 or settings['block_width'] != 32):
        raise ValueError('Declare the bounded 208 comparison and its available worker budget')
    probe = helpers(); cross = probe.helpers()
    v = cross.helpers().helpers().helpers().base.validation()
    source = ROOT / settings['evaluation']
    inputs = {}

    def bind(path):
        path = Path(path).resolve(); value = v.digest(path)
        if str(path) in inputs and inputs[str(path)] != value:
            raise ValueError('Bound input changed')
        inputs[str(path)] = value
        return path

    for path in (config, Path(__file__), ROOT / PROBE, ROOT / EXECUTION):
        bind(path)
    original = v.read(bind(source / 'protocol.json'))
    frozen = v.read(bind(Path(original['fit']) / 'nominees.json'))
    if (frozen['smoke'] or frozen['status'] != 'frozen_before_test'
            or v.digest(Path(original['fit'])/'nominees.json') != original['nomination_sha256']):
        raise ValueError('Require the original frozen scientific nominees')
    executed = {j['target']:j for file in sorted((source/'jobs').glob('*.json')) if (j:=v.read(bind(file)))}
    jobs, banks = [], []
    for arm in settings['arms']:
        nominee = next(n['candidate'] for n in frozen['nominees'] if n['subject']==208 and n['model']==arm)
        for policy in ('MIX','SELECT'):
            bank = dict(cell=f'{policy}_{arm}', arm=arm, policy=policy, candidate=nominee, roles={})
            for phase in ('autonomous','evaluate'):
                templates = [j for j in original['jobs'] if j['model']==arm and j['phase']==phase]
                count = 1 if smoke else settings['paths']//4 if phase=='autonomous' else settings['filter_repeats']
                assert len(templates) >= count
                bank['roles'][phase] = []
                for index, template in enumerate(templates[:count]):
                    original_job = executed[template['target']]
                    assert original_job['candidate']==nominee and not original_job['smoke']
                    for key,value in template.items():
                        assert original_job[key]==value
                    for path,value in original_job['source_sha256'].items():
                        if v.digest(bind(ROOT/path))!=value:
                            raise ValueError('Original execution source changed')
                    if phase=='autonomous':
                        record,_ = cross.check_chunk(original_job,'SEPARATE',v,bind)
                        t = Path(original_job['target'])
                        check_execution(original_job,record,v.read(bind(t.with_suffix('.belief.json'))),
                                        v.read(bind(t.with_suffix('.precision.json'))))
                    reused = policy=='MIX' and phase=='autonomous' and not smoke
                    if reused:
                        job = deepcopy(original_job)
                    else:
                        name=f'{policy}_{arm}_{phase}_C{index:03d}'
                        job=deepcopy(original_job)
                        job.update(name=name, target=str((output/'chunks'/name).resolve()),
                            execution_policy=policy, smoke=smoke,
                            seeds=original_job['seeds'][:1] if smoke else original_job['seeds'],
                            particles=2 if smoke else settings['particles'])
                        if smoke and phase=='evaluate':
                            job['folder']=str((ROOT/settings['smoke_filter_data']).resolve())
                            short=v.read(bind(Path(job['folder'])/'prepared.json'))
                            assert short['training_trials']==4 and short['test_trials']==4 and short['trials']==8
                            for name in ('engine.json','training.npz','sealed_test.npz'):
                                bind(Path(job['folder'])/name)
                        if smoke and phase=='autonomous' and policy=='MIX':
                            job['exact_reference']=str(Path(original_job['target']).with_suffix('.npz'))
                        job['source_sha256'].update({name:v.digest(ROOT/name) for name in (ENTRY,EXECUTION)})
                        jobs.append(job)
                    bank['roles'][phase].append(dict(job=job,reused=reused))
            banks.append(bank)
    output.mkdir(parents=True);(output/'jobs').mkdir();(output/'chunks').mkdir()
    workers=min(1 if smoke else settings['parallel_budget'],len(jobs),len(os.sched_getaffinity(0)))
    protocol=dict(settings=settings,smoke=smoke,jobs=jobs,banks=banks,inputs_sha256=inputs,
        workers=workers,numeric_threads=1,parameters_changed=False,oral_used=False,
        scope='Fixed-parameter execution-policy screen; formal nominees remain unchanged')
    v.write(output/'protocol.json',protocol)

    def launch(job):
        file=output/'jobs'/(job['name']+'.json');v.write(file,job)
        launcher=('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                  "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='dominant_execution_') as cache:
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1',
                NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',NUMBA_CACHE_DIR=cache,MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as stream:
                subprocess.run([sys.executable,'-c',launcher,job['runtime'],str(ROOT/ENTRY),'--worker',str(file.resolve())],
                    cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

    print(f'{len(jobs)} jobs; {workers} workers; numeric threads=1',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index,future in enumerate(as_completed([pool.submit(launch,j) for j in jobs]),1):
            future.result();print(f'{index}/{len(jobs)} complete',flush=True)
    rows, exact = [], 0
    for bank in banks:
        row={k:value for k,value in bank.items() if k!='roles'}
        for phase,items in bank['roles'].items():
            for item in items:
                job=item['job'];t=Path(job['target'])
                record,arrays=cross.check_chunk(job,'SEPARATE',v,bind)
                check_execution(job,record,v.read(bind(t.with_suffix('.belief.json'))),v.read(bind(t.with_suffix('.precision.json'))))
                if not item['reused']:
                    proof=v.read(bind(t.with_suffix('.execution.json')))
                    copies=len(job['seeds'])*(1 if phase=='autonomous' else job['particles'])
                    assert proof['policy']==bank['policy'] and proof['enabled']==(bank['policy']=='SELECT')
                    assert proof['updates']==(copies*record['trials'] if proof['enabled'] else 0)
                    assert proof['uses_existing_dominant'] and not proof['task_categories_used_for_selection']
                    if 'exact_reference' in job:
                        with np.load(bind(job['exact_reference'])) as z:
                            n=record['trials'];indices=[int(np.flatnonzero(z['seeds']==seed).item()) for seed in job['seeds']]
                            for key,value in arrays.items():
                                expected=z[key][:n] if key=='categories' else z[key][indices] if key=='seeds' else z[key][indices,:n]
                                np.testing.assert_array_equal(value,expected);exact+=value.size
                if bank['policy']=='SELECT' and phase=='autonomous':
                    np.testing.assert_array_equal(arrays['contribution'],arrays['contribution'].astype(bool))
            aggregate=deepcopy(items[0]['job']);target=output/'banks'/f'{bank["cell"]}_{phase}';target.parent.mkdir(exist_ok=True)
            aggregate.update(target=str(target.resolve()),seeds=[seed for item in items for seed in item['job']['seeds']])
            record=v.combine_evaluation_chunks(aggregate,[i['job'] for i in items],phase=='autonomous')
            row[phase]=dict(target=str(target.resolve()),record=record)
        rows.append(row)
    v.write(output/'banks.json',dict(status='complete',rows=rows))
    if not smoke:
        estimates,scores,paths,reference={},[],[],None
        for row in rows:
            data=[]
            for phase in ('evaluate','autonomous'):
                with np.load(Path(row[phase]['target']).with_suffix('.npz')) as z:
                    data.append({k:z[k].copy() for k in z.files})
            c,a=data;meta=dict(row['evaluate']['record'],target_rule=row['autonomous']['record']['target_rule'])
            validate_generation(a,row['autonomous']['record'])
            if reference is not None:match_task(reference,(c,a,meta))
            reference=c,a,meta
            for phase,first,last in [('training',0,meta['training_trials']),('test',meta['training_trials'],len(c['choices'])),('whole',0,len(c['choices']))]:
                estimate=window_estimate(c,a,first,last,settings['block_width'])
                estimates[row['cell'],phase]=estimate
                scores.append(dict(cell=row['cell'],phase=phase,**estimate['metrics']))
            for i,seed in enumerate(a['seeds']):
                paths.append(dict(cell=row['cell'],seed=int(seed),late_accuracy=float((a['choices'][i,-128:]==a['categories'][-128:]).mean()),
                    target_late_present=float(a['present'][i,-128:].mean()),target_late_contribution=float(a['contribution'][i,-128:].mean())))
        contrasts=[dict(arm=arm,phase=phase,**paired_difference(estimates[f'SELECT_{arm}',phase],estimates[f'MIX_{arm}',phase]))
                   for arm in settings['arms'] for phase in ('training','test','whole')]
        for name,table in [('scores',scores),('contrasts',contrasts),('paths',paths)]:
            pd.DataFrame(table).to_csv(output/(name+'.csv'),index=False)
    for path,value in inputs.items():
        assert v.digest(Path(path))==value
    v.write(output/'completion.json',dict(status='complete',smoke=smoke,cells=4,new_jobs=len(jobs),
        inputs_sha256=inputs,protocol_sha256=v.digest(output/'protocol.json'),exact_baseline_prefix_elements=exact,
        artifacts={p.name:v.digest(p) for p in output.glob('*.csv')},parameters_changed=False,oral_used=False,
        scope='Fixed-parameter screen; behavior refitting and oral validation remain separate'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','output','worker'):parser.add_argument('--'+name,type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.config and args.output:run(args.config,args.output,args.smoke)
    else:parser.error('Supply --config and --output')
