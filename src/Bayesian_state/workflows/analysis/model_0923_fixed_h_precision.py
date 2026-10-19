"""Select precision coordinates while preserving each completed H parameter point.

Run --config YAML --output NEW_DIRECTORY. Four cells cross MIX/SELECT with
ADD10/LOG25 at their respective frozen SEARCH nominees. A common training-only
grid and an independent confirmation bank select beta_init and learning_rate.
Each confirmation includes its predeclared anchor. Freeze all four nominees
before evaluating later choices and complete autonomous paths. These are
development comparisons; oral reports never enter selection. The actual
precision, belief, retention and execution algorithms use the existing workers.

--smoke executes four eight-trial, two-particle training jobs and four short
autonomous paths on one worker. It does not establish full evaluation results.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from . import model_0923_dominant_execution_probe as execution
from .model_0923_h_validation import combine_evaluation_chunks, write
from .model_0923_online_parameter_cross import check_chunk
from ...evaluation.model_0923_h_validation import digest, load, nll, jackknife_scores, jackknife_se
from ...evaluation.model_0923_separate_belief_refit import check_execution, check_training_arrays
from ...evaluation.model_0923_reentry_refit import window_estimate
from ...evaluation.model_0923_estimation_scope import paired_difference, validate_generation
from ...optimization.search.adaptive_proposals import candidate
from ...utils.seeding import stable_seed

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
INIT = 'engine.modules.beta_mod.kwargs.beta_init'
RATE = 'engine.modules.beta_mod.kwargs.learning_rate'


def precision_points(reference: dict, initials: list, family: dict) -> list[dict]:
    """Change exactly the two precision coordinates; keep H and memory fixed."""
    if (len(set(initials)) != len(initials) or len(set(family['rates'])) != len(family['rates'])
            or any(not 0 < x <= family['upper'] for x in initials)
            or any(not 0 <= x <= 1 for x in family['rates'])
            or family['anchor'][0] not in initials or family['anchor'][1] not in family['rates']):
        raise ValueError('Require distinct supported precision values and an included anchor')
    return [candidate(dict(reference, **{INIT: initial, RATE: rate}), 'fixed_h_precision')
            for initial in initials for rate in family['rates']]


def choose_finalists(rows: list[dict], anchor_id: str, count: int) -> list[str]:
    """Use training NLL only; keep the anchor even when it is outside the top k."""
    if (len({r['id'] for r in rows}) != len(rows) or anchor_id not in {r['id'] for r in rows}
            or count < 1 or any(not np.isfinite(r['training_nll']) for r in rows)):
        raise ValueError('Require unique finite training scores and a declared anchor')
    top = [r['id'] for r in sorted(rows, key=lambda r: (r['training_nll'], r['id']))[:count]]
    return top if anchor_id in top else top + [anchor_id]


def seed_bank(settings: dict, stage: str, count: int) -> list[int]:
    return [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'],
                            subject=settings['subject'], stage=stage, repeat=i)) for i in range(count)]


def run(config: Path, output: Path, smoke: bool = False) -> None:
    if output.exists():
        raise FileExistsError('Use a new precision diagnostic directory')
    settings = yaml.safe_load(config.read_text())
    if (settings['subject'] != 107 or settings['arm'] != 'SEARCH'
            or set(settings['references']) != {'MIX', 'SELECT'}
            or set(settings['families']) != {'ADD10', 'LOG25'}
            or not 1 <= settings['parallel_budget'] <= 14 or settings['block_width'] != 32):
        raise ValueError('Require the declared bounded S107 fixed-H diagnostic')
    for name, mode, scale in [('ADD10', 'fast', 10.), ('LOG25', 'log', 1.)]:
        f = settings['families'][name]
        if (f['precision'], f['rate_scale'], f['upper']) != (mode, scale, 25.):
            raise ValueError('Precision family or ceiling changed')
    for stage in ('screen', 'confirmation', 'evaluation'):
        b = settings[stage]
        if not 2 <= b['repeats'] <= 8 or not 2 <= b['particles'] <= 512:
            raise ValueError('Require bounded independent filters')
    e = settings['evaluation']
    if not 4 <= e['paths'] <= 128 or not 1 <= e['path_chunk'] <= 4:
        raise ValueError('Require bounded autonomous banks')
    inputs, templates, cells, old_seeds = {}, {}, {}, set()

    def bind(path):
        path = Path(path).resolve(); value = digest(path)
        if str(path) in inputs and inputs[str(path)] != value:
            raise ValueError('Bound input changed: ' + str(path))
        inputs[str(path)] = value
        return path

    bind(config)
    for policy, source in settings['references'].items():
        source = ROOT / source
        protocol = load(bind(source / 'protocol.json'))
        done = load(bind(source / 'completion.json'))
        frozen_file = Path(protocol['fit']) / 'nominees.json'
        frozen = load(bind(frozen_file))
        if (done['status'] != 'complete' or frozen['smoke']
                or frozen['status'] != 'frozen_before_test'
                or protocol['nomination_sha256'] != digest(frozen_file)
                or done['nomination_sha256'] != digest(frozen_file)):
            raise ValueError('Require completed frozen source fits and evaluations')
        nominee = next(n['candidate'] for n in frozen['nominees']
                       if n['subject'] == settings['subject'] and n['model'] == settings['arm'])
        all_jobs = [load(bind(f)) for f in sorted((source/'jobs').glob('*.json'))]
        old_seeds.update(seed for j in all_jobs for seed in j['seeds'])
        for phase in ('evaluate', 'autonomous'):
            job = next(j for j in all_jobs if j['model'] == settings['arm'] and j['phase'] == phase)
            if job['candidate'] != nominee or job.get('execution_policy', 'MIX') != policy or job['smoke']:
                raise ValueError('Source job differs from frozen policy or point')
            for name, sha in job['source_sha256'].items():
                if digest(bind(ROOT/name)) != sha:
                    raise ValueError('Source implementation changed: ' + name)
            templates[policy, phase] = job
        for family_name, family in settings['families'].items():
            points = precision_points(nominee['hyperparams'], settings['initial_values'], family)
            anchor = next(p for p in points if [p['hyperparams'][INIT], p['hyperparams'][RATE]] == family['anchor'])
            if family_name == 'ADD10' and anchor['hyperparams'] != nominee['hyperparams']:
                raise ValueError('ADD10 anchor must reproduce the frozen parameter point')
            cells[policy+'_'+family_name] = dict(policy=policy, family=family_name, points=points,
                                               anchor=anchor['id'], reference=nominee)
    # Identical data and stage-specific random banks preserve comparison scope.
    a, b = (templates[p, 'evaluate'] for p in ('MIX', 'SELECT'))
    if a['folder'] != b['folder'] or a['runtime'] != b['runtime']:
        raise ValueError('Require identical task data and archived runtime')
    meta = load(bind(Path(a['folder'])/'prepared.json'))
    if (meta['subject'], meta['training_trials'], meta['test_trials']) != (107, 128, 64):
        raise ValueError('Keep the original 128/64 chronological boundary')
    for name, sha in meta['array_sha256'].items():
        if digest(bind(Path(a['folder'])/name)) != sha:
            raise ValueError('Prepared task data changed')
    banks = {stage: seed_bank(settings, stage, settings[stage]['repeats'])
             for stage in ('screen', 'confirmation', 'evaluation')}
    banks['autonomous'] = seed_bank(settings, 'autonomous', e['paths'])
    seeds = [s for bank in banks.values() for s in bank]
    if len(set(seeds)) != len(seeds) or set(seeds) & old_seeds:
        raise ValueError('New stage banks must be distinct from each other and source evaluations')
    additions = (ENTRY, execution.ENTRY, execution.EXECUTION,
                 'src/Bayesian_state/evaluation/model_0923_separate_belief_refit.py',
                 'src/Bayesian_state/evaluation/model_0923_reentry_refit.py',
                 'src/Bayesian_state/evaluation/model_0923_estimation_scope.py',
                 'src/Bayesian_state/evaluation/model_0923_h_validation.py')
    for name in additions:
        bind(ROOT/name)
    output.mkdir(parents=True); (output/'jobs').mkdir(); (output/'chunks').mkdir(); (output/'banks').mkdir()
    shutil.copyfile(config, output/'config.yaml')
    for name in set(additions) | set(a['source_sha256']) | set(b['source_sha256']):
        target = output/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/name, target)
    workers = min(1 if smoke else settings['parallel_budget'], len(os.sched_getaffinity(0)))
    write(output/'protocol.json', dict(settings=settings, smoke=smoke, cells=cells, seeds=banks,
        inputs_sha256=inputs, workers=workers, numeric_threads=1,
        changed_coordinates=[INIT, RATE], selection_data='training choices only', oral_used=False,
        scope='Fixed-H development diagnostic; original fits remain unchanged',
        versions={name: importlib.metadata.version(name) for name in ('numpy','scipy','pandas','PyYAML')}))

    def make_job(cell, point, stage, seeds):
        c = cells[cell]; phase = 'autonomous' if stage == 'autonomous' else 'evaluate' if stage == 'evaluation' else 'training'
        job = deepcopy(templates[c['policy'], 'autonomous' if phase == 'autonomous' else 'evaluate'])
        name = f'{stage}_{cell}_{point["id"]}_{seeds[0]}'
        family = settings['families'][c['family']]
        job.update(phase=phase, name=name, candidate=point, seeds=seeds, smoke=smoke,
            particles=2 if smoke else settings['evaluation' if stage == 'autonomous' else stage]['particles'],
            execution_policy=c['policy'], target=str((output/'chunks'/name).resolve()),
            arm_settings={k:v for k,v in family.items() if k != 'anchor'},
            fixed_h_precision_cell=cell)
        job['source_sha256'].update({name:digest(ROOT/name) for name in additions})
        return job

    def launch(job):
        file = output/'jobs'/(job['name']+'.json'); write(file, job)
        launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='fixed_h_precision_') as cache:
            env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as stream:
                subprocess.run([sys.executable,'-c',launcher,job['runtime'],str(ROOT/execution.ENTRY),
                                '--worker',str(file.resolve())],cwd=ROOT,env=env,stdout=stream,
                               stderr=subprocess.STDOUT,check=True)

    def execute(jobs):
        print(f'{len(jobs)} jobs; {min(workers,len(jobs))} workers; numeric threads=1', flush=True)
        with ThreadPoolExecutor(max_workers=min(workers,len(jobs))) as pool:
            for i, future in enumerate(as_completed([pool.submit(launch,j) for j in jobs]),1):
                future.result()
                if i % 8 == 0 or i == len(jobs): print(f'{i}/{len(jobs)} complete',flush=True)
        for job in jobs:
            target = Path(job['target']); record = load(bind(target.with_suffix('.json')))
            if digest(bind(target.with_suffix('.npz'))) != record['predictions_sha256']:
                raise ValueError('Executed predictions changed')
            if job['phase'] == 'training':
                with np.load(target.with_suffix('.npz')) as arrays: check_training_arrays(job,record,arrays)
                if record['trials'] != (8 if smoke else 128): raise ValueError('Training crossed its boundary')
            else:
                record, arrays = check_chunk(job,'SEPARATE',SimpleNamespace(read=load,digest=digest),bind)
                if job['phase'] == 'autonomous': validate_generation(arrays,record)
            check_execution(job,record,load(bind(target.with_suffix('.belief.json'))),
                            load(bind(target.with_suffix('.precision.json'))))
            actual = load(bind(target.with_suffix('.experiment.json')))
            # The source-bound recalled-search worker consumes job.arm_settings
            # for precision; its sidecar records proposal and evidence execution.
            if actual['proposal'] is not True or actual['prior'] is not False:
                raise ValueError('Actual proposal mechanism differs from SEARCH')
            proof = load(bind(target.with_suffix('.execution.json')))
            copies = len(job['seeds'])*(1 if job['phase']=='autonomous' else job['particles'])
            enabled = job['execution_policy'] == 'SELECT'
            if (proof['enabled'] != enabled or proof['policy'] != job['execution_policy']
                    or proof['updates'] != (copies*record['trials'] if enabled else 0)):
                raise ValueError('Actual execution policy differs')

    def training_stage(stage, points):
        jobs = [make_job(cell,p,stage,[s]) for cell, ps in points.items() for p in ps
                for s in (banks[stage][:1] if smoke else banks[stage])]
        execute(jobs); rows=[]
        for cell, ps in points.items():
            for p in ps:
                js=[j for j in jobs if j['fixed_h_precision_cell']==cell and j['candidate']['id']==p['id']]
                arrays=[]
                for j in js:
                    with np.load(Path(j['target']).with_suffix('.npz')) as z: arrays.append(z['observed_probability'])
                values=np.concatenate(arrays)
                rows.append(dict(cell=cell,id=p['id'],training_nll=nll(values),
                    mc_se=None if smoke else jackknife_se(jackknife_scores(values)),
                    beta_init=p['hyperparams'][INIT],configured_rate=p['hyperparams'][RATE],
                    score_files=[str(Path(j['target']).with_suffix('.json')) for j in js]))
        write(output/(stage+'.json'),dict(rows=rows)); pd.DataFrame(rows).to_csv(output/(stage+'.csv'),index=False)
        return rows

    points={cell:([next(p for p in c['points'] if p['id']==c['anchor'])] if smoke else c['points'])
            for cell,c in cells.items()}
    screen=training_stage('screen',points)
    if smoke:
        execute([make_job(cell,ps[0],'autonomous',banks['autonomous'][:1]) for cell,ps in points.items()])
        write(output/'completion.json',dict(status='complete',smoke=True,full_evaluation_validated=False,
              protocol_sha256=digest(output/'protocol.json'),inputs_sha256=inputs))
        return
    selected={cell:choose_finalists([r for r in screen if r['cell']==cell],c['anchor'],
                                    settings['confirmation']['finalists']) for cell,c in cells.items()}
    finalists={cell:[p for p in cells[cell]['points'] if p['id'] in ids] for cell,ids in selected.items()}
    confirmed=training_stage('confirmation',finalists)
    nominees={cell:next(p for p in finalists[cell] if p['id']==min(
        [r for r in confirmed if r['cell']==cell],key=lambda r:(r['training_nll'],r['id']))['id']) for cell in cells}
    write(output/'nominees.json',dict(status='frozen_before_test',nominees=nominees,
          confirmation_sha256=digest(output/'confirmation.json'),test_choices_used=False,oral_used=False))
    nomination_sha=digest(output/'nominees.json')
    final_jobs=[]
    for cell,p in nominees.items():
        final_jobs.extend(make_job(cell,p,'evaluation',[s]) for s in banks['evaluation'])
        final_jobs.extend(make_job(cell,p,'autonomous',banks['autonomous'][i:i+e['path_chunk']])
                          for i in range(0,e['paths'],e['path_chunk']))
    execute(final_jobs)
    estimates, scores, contrasts = {}, [], []
    for cell,p in nominees.items():
        combined={}
        for phase in ('evaluate','autonomous'):
            jobs=[j for j in final_jobs if j['fixed_h_precision_cell']==cell and j['phase']==phase]
            job=deepcopy(jobs[0]);target=output/'banks'/f'{cell}_{phase}'
            job.update(target=str(target.resolve()),seeds=[s for j in jobs for s in j['seeds']])
            combine_evaluation_chunks(job,jobs,phase=='autonomous')
            with np.load(bind(target.with_suffix('.npz'))) as z: combined[phase]={k:z[k] for k in z.files}
            bind(target.with_suffix('.json'))
        for window,first,last in [('training',0,128),('later',128,192),('whole',0,192)]:
            c,a=combined['evaluate'],combined['autonomous']
            estimate=window_estimate(c,a,first,last,settings['block_width']);estimates[cell,window]=estimate
            scores.append(dict(cell=cell,window=window,candidate=p['id'],**estimate['metrics'],
                choice_nll_mc_se=jackknife_se(estimate['jackknife']['choice_nll']),
                autonomous_correct=float((a['choices'][:,first:last]==a['categories'][None,first:last]).mean()),
                human_correct=float((c['choices'][first:last]==c['categories'][first:last]).mean())))
    for policy in ('MIX','SELECT'):
        for window in ('training','later','whole'):
            contrasts.append(dict(policy=policy,window=window,comparison='LOG25_minus_ADD10',
                **paired_difference(estimates[policy+'_LOG25',window],estimates[policy+'_ADD10',window])))
    pd.DataFrame(scores).to_csv(output/'scores.csv',index=False)
    pd.DataFrame(contrasts).to_csv(output/'contrasts.csv',index=False)
    if digest(output/'nominees.json') != nomination_sha: raise ValueError('Frozen nominees changed')
    for path,sha in inputs.items():
        if digest(Path(path)) != sha: raise ValueError('Bound input changed: '+path)
    write(output/'completion.json',dict(status='complete',smoke=False,nomination_sha256=nomination_sha,
          protocol_sha256=digest(output/'protocol.json'),inputs_sha256=inputs,
          summary_sha256={name:digest(output/name) for name in ('scores.csv','contrasts.csv')},
          original_fits_changed=False,oral_used=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args.config,args.output,args.smoke)
