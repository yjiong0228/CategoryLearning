"""Register and run the bounded individual-state study.

Run from the repository root with --phase prepare|run --config YAML --output
NEW. Preparation freezes candidates, seeds, sources and simulation cases.
``run`` resumes completed jobs and retains every failed log. ``--extension``
accepts a JSON list of case IDs selected by the registered numerical checks.
The synthetic fixture retains historical storage partitions; every filter
reads the complete record.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

from . import model_1024_individual_state_capture as capture
from . import model_1023_execution_recovery as recovery
from ...utils.seeding import stable_seed

ROOT, read, digest = capture.ROOT, capture.read, capture.digest
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))


def write(path: Path, value) -> None:
    """Create immutable records; only the live progress file is replaceable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.name != 'progress.json':
        if read(path) != value:
            raise FileExistsError(path)
        return
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if path.name == 'progress.json':
        temporary = path.with_suffix('.pending')
        temporary.write_text(text); temporary.replace(path)
    else:
        with path.open('x') as stream:
            stream.write(text)


def make_job(case: dict, p: dict, output: Path, stage: str, particles: int, index: int) -> dict:
    name = f'{stage}_{case["id"]}_{index:03d}'
    job = deepcopy(case['template'])
    for key in ('training_generation', 'selection_sources', 'freeze_file', 'freeze_sha256'):
        job.pop(key, None)
    seed = stable_seed(dict(base=p['seed'], case=case['id'], stage=stage, index=index))
    job.update(name=name, subject=case['subject'], case_id=case['subject'], state_case=case['id'],
        candidate=case['candidate'], phase='evaluate', smoke=False, particles=particles, seeds=[seed],
        model='COMPACT', comparison_arm='ADAPT',
        execution_mode='STICKY' if case['candidate']['hyperparams'][capture.study.KAPPA] else 'MIX',
        capture_oral=False, capture_sources=capture.capture.sources(), individual_state_sources=capture.sources(),
        source_sha256=capture.study.sources(capture.study.driver.io), target_rules=case['target_rules'],
        cpu_limit_seconds=p['budget']['per_job_cpu_seconds'], state_stage=stage)
    job['mechanism']['capacity'] = 'evidence'
    for key, folder in [('target', 'scores'), ('state_capture_target', 'basic'), ('individual_target', 'states')]:
        job[key] = str((output/folder/name).resolve())
    return job


def prepare(config: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new registered output directory')
    p = yaml.safe_load(config.read_text()); source = ROOT/p['source']
    frozen = read(source/'registration.json')['numerical_sources']
    if any(digest(ROOT/k) != v for k, v in frozen.items()):
        raise ValueError('Original individual-parameter numerical sources changed')
    output.mkdir(parents=True)
    cases, truths, inputs = [], [], {str(config): digest(config)}
    for sid in p['subjects']:
        file = source/f'confirmed_S{sid}.json'; inputs[str(file)] = digest(file)
        rows = read(file)
        selected = [r for r in rows if r['joint_status'] == 'retained' or sid == 129]
        geometry = ROOT/p['geometry']/f'subject_{sid}'/'audit.json'
        inputs[str(geometry)] = digest(geometry)
        targets = read(geometry)['training_perfect_rules']
        for r in selected:
            template = deepcopy(r['filter_jobs'][0]); folder = Path(template['folder'])
            meta = read(folder/'prepared.json')
            for name, expected in meta['array_sha256'].items():
                if digest(folder/name) != expected:
                    raise ValueError('Observed data changed')
                inputs[str(folder/name)] = expected
            cases.append(dict(id=f'S{sid}_{r["candidate"]["id"]}', subject=sid, condition=r['condition'],
                candidate=r['candidate'], template=template, target_rules=targets, kind='human',
                parameter_status=r['joint_status'], trials=meta['trials']))
        # Fixed trajectory indices from the first listed eligible candidate.
        # Selection does not inspect simulated states or recovery performance.
        r = selected[0]; generated = r['generation_jobs'][0]
        zfile = Path(generated['target']).with_suffix('.npz')
        if digest(zfile) != read(zfile.with_suffix('.json'))['predictions_sha256']:
            raise ValueError('Simulation bank changed')
        inputs[str(zfile)] = digest(zfile)
        with np.load(zfile) as z:
            bank = {k: z[k] for k in z.files}
        original = Path(r['filter_jobs'][0]['folder'])
        with np.load(original/'training.npz') as a, np.load(original/'sealed_test.npz') as b:
            stimulus = np.concatenate([a['stimulus'], b['stimulus']])
        for index in p['recovery']['path_indices']:
            identity = f'R{sid}_{index}'
            template = deepcopy(r['filter_jobs'][0])
            folder = output/'synthetic'/identity/f'subject_{sid}'
            arrays = dict(stimulus=stimulus, categories=bank['categories'],
                          choices=bank['choices'][index], feedback=bank['feedback'][index])
            meta = recovery.fixture(original, folder, arrays, template['runtime'], .7)
            template['folder'] = str(folder.resolve())
            cases.append(dict(id=identity, subject=sid, condition=r['condition'], candidate=r['candidate'],
                template=template, target_rules=targets, kind='recovery', parameter_status='known_generator',
                trials=meta['trials']))
            truths.append(dict(case=identity, file=str(zfile.resolve()), sha256=digest(zfile),
                index=index, seed=int(bank['seeds'][index]), candidate=r['candidate']))
    jobs = []
    for stage in ('primary', 'resolution', 'recovery'):
        settings = p[stage]
        own = [c for c in cases if (c['kind']=='recovery') == (stage=='recovery')]
        for i in range(settings['systems']):
            jobs.extend(make_job(c, p, output, stage, settings['particles'], i) for c in own)
    # Full-record exact replay checks use old seeds and saved score arrays.
    for sid in (110, 205, 317):
        c = next(c for c in cases if c['kind']=='human' and c['subject']==sid)
        job = make_job(c, p, output, 'replay', c['template']['particles'], 0)
        job['seeds'] = c['template']['seeds']
        job['original_score'] = str(Path(c['template']['target']).with_suffix('.npz'))
        job['original_score_sha256'] = digest(Path(job['original_score']))
        jobs.insert(0, job)
    for j in jobs:
        write(output/'jobs'/(j['name']+'.json'), j)
    write(output/'protocol.json', dict(policy=p, cases=cases, sources=capture.sources(),
        inputs=inputs, created_unix=time.time(), versions=dict(python=sys.version, numpy=np.__version__),
        oral_permission='User explicitly requested provisional external oral validation on 2026-10-28',
        conditioning='full observed choices; all states before current feedback; no oral input',
        parameter_pooling='none; separate fixed-candidate distributions',
        recovery_selection='first eligible candidate in stored order, fixed paths 0 and 1'))
    write(output/'truths.json', truths)
    write(output/'job_order.json', [j['name'] for j in jobs])
    for rel in set(capture.sources()) | {ENTRY, str(config)}:
        target = output/'source'/rel; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/rel, target)
    print(f'Registered {len(cases)} cases and {len(jobs)} initial jobs', flush=True)


def run(output: Path, extension: Path | None = None) -> None:
    protocol = read(output/'protocol.json'); p = protocol['policy']; b = p['budget']
    if protocol['sources'] != capture.sources():
        raise ValueError('Registered capture code changed')
    names = read(output/'job_order.json')
    if extension:
        ids = read(extension)
        own = [c for c in protocol['cases'] if c['id'] in ids and c['kind']=='human']
        if len(own) != len(set(ids)):
            raise ValueError('Unknown extension case')
        names = []
        for i in range(p['extension']['additional_systems']):
            for c in own:
                job = make_job(c, p, output, 'extension', p['extension']['particles'], i)
                file = output/'jobs'/(job['name']+'.json')
                if file.exists() and read(file) != job:
                    raise ValueError('Extension job identity changed')
                write(file, job); names.append(job['name'])
        write(output/'extension_order.json', names)
    clock_file = output/'run_clock.json'
    if not clock_file.exists():
        write(clock_file, dict(started_unix=time.time()))
    started = read(clock_file)['started_unix']
    deadline = started+b['wall_hours']*3600
    done_dir = output/'ledger'; done_dir.mkdir(exist_ok=True)
    existing = {f.stem: read(f) for f in done_dir.glob('*.json')}
    used = sum(r['cpu_seconds'] for r in existing.values())
    pending = [name for name in names if name not in existing]
    workers = min(b['workers'], len(os.sched_getaffinity(0)))
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
        NUMBA_CACHE_DIR='/tmp/model1024_states_numba')
    launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                'sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name="__main__")')
    active, failures, stop = {}, 0, None
    while pending or active:
        now = time.time()
        reserved = sum(j['cpu_limit_seconds'] for _, j, _, _ in active.values())
        if now >= deadline:
            if stop is None:
                stop = 'wall_budget'
            for proc, _, _, _ in active.values():
                if proc.poll() is None:
                    proc.terminate()
        # Keep the initiating cause when budget terminations also exceed the
        # worker-failure limit. Completed scientific outputs are unaffected.
        if failures >= b['failed_jobs_before_stop'] and stop is None:
            stop = 'repeated_worker_failure'
        while pending and len(active) < workers and not stop:
            name = pending[0]; job = read(output/'jobs'/(name+'.json'))
            if used+reserved+job['cpu_limit_seconds'] > b['cpu_hours']*3600:
                if not active:
                    stop = 'cpu_budget'
                break
            pending.pop(0)
            log = output/'logs'/(name+'.log'); log.parent.mkdir(exist_ok=True)
            stream = log.open('x')
            proc = subprocess.Popen([sys.executable, '-c', launcher, job['runtime'], str(ROOT/capture.ENTRY),
                '--worker', str((output/'jobs'/(name+'.json')).resolve())], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT)
            active[name] = (proc, job, stream, time.time()); reserved += job['cpu_limit_seconds']
        for name, (proc, job, stream, launch) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            stream.close(); metadata = Path(job['individual_target']).with_suffix('.json')
            complete = code == 0 and metadata.exists()
            cpu = read(metadata)['cpu_seconds'] if complete else job['cpu_limit_seconds']
            row = dict(name=name, status='complete' if complete else 'failed', returncode=code,
                       cpu_seconds=cpu, cpu_is_upper_bound=not complete, wall_seconds=time.time()-launch)
            write(done_dir/(name+'.json'), row); existing[name] = row; used += cpu
            failures += not complete; del active[name]
            print(f'{name}: {row["status"]}; cumulative CPU {used/3600:.2f} h', flush=True)
        write(output/'progress.json', dict(completed=sum(r['status']=='complete' for r in existing.values()),
            failed=sum(r['status']=='failed' for r in existing.values()), pending=len(pending), active=list(active),
            cpu_hours=used/3600, elapsed_hours=(time.time()-started)/3600, stop=stop, updated_unix=time.time()))
        if stop and not active:
            break
        if active:
            time.sleep(2)
    write(output/('extension_completion.json' if extension else 'initial_completion.json'),
          dict(status='complete' if not pending and not failures else 'limited', stop=stop,
               pending=pending, failures=failures, cpu_hours=used/3600, finished_unix=time.time()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'run'], required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extension', type=Path)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare(args.config, args.output)
    else:
        run(args.output, args.extension)
