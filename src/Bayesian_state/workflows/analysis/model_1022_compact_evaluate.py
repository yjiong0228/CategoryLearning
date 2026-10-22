"""Evaluate all frozen Model 1022 human fits with original-filter oral capture.

Use --study COMPLETE_OR_FROZEN_REFIT --output NEW [--smoke]. All human and
recovery fits must freeze before this stage. If generating-point recovery
scores are still running, their declared upper bound is deducted from the
shared CPU allocation. Each completed truth-score group releases its allowance
immediately; evaluation ultimately expands to the full allocation.
Each conditional filter and complete autonomous path uses the fitted variant.
Behavior equations and decision-basis collection are reused from existing code.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Condition
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
FIT = 'src/Bayesian_state/workflows/analysis/model_1022_compact_refit.py'
CAPTURE = 'src/Bayesian_state/workflows/analysis/model_1021_simplification_capture.py'
spec = importlib.util.spec_from_file_location('model1022_evaluation_capture', ROOT/CAPTURE)
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
driver = capture.driver
read, write, digest = capture.read, capture.write, capture.digest


def sources() -> dict:
    return dict(capture.sources(), **{name: digest(ROOT/name) for name in (ENTRY, FIT)})


def smoke_fixture(data: Path, output: Path, subject: int) -> Path:
    """Split sixteen original training trials 4+12 for software checks only.

    The existing conditional smoke scorer uses the first eight trials, leaving
    four trials on each side of its scoring boundary. The autonomous smoke
    recorder uses sixteen trials. Both retain the existing numerical entry
    points and chronological values. These segments have no scientific use.
    """
    if output.exists():
        raise FileExistsError('Use a new software-fixture directory')
    original=data/f'subject_{subject}';meta=read(original/'prepared.json')
    for name,expected in [('training.npz',meta['array_sha256']['training.npz']),('engine.json',meta['engine_sha256'])]:
        if digest(original/name)!=expected:
            raise ValueError('Source of the short software fixture changed')
    with np.load(original/'training.npz') as z:
        arrays={k:z[k][:16].copy() for k in z.files}
    if any(len(v)!=16 for v in arrays.values()):
        raise ValueError('Need sixteen valid original training trials for the software fixture')
    folder=output/f'subject_{subject}';folder.mkdir(parents=True)
    hashes={}
    for name,part in [('training.npz',slice(0,4)),('sealed_test.npz',slice(4,16))]:
        with (folder/name).open('xb') as stream:
            np.savez_compressed(stream,**{k:v[part] for k,v in arrays.items()})
        hashes[name]=digest(folder/name)
    with (folder/'engine.json').open('xb') as stream:
        stream.write((original/'engine.json').read_bytes())
    meta.update(trials=16,training_trials=4,test_trials=12,array_sha256=hashes,
        split='software_only_first_sixteen_original_training_trials',
        source_prepared_sha256=digest(original/'prepared.json'),source_training=str((original/'training.npz').resolve()),
        source_training_sha256=digest(original/'training.npz'),scientific_inference=False)
    write(folder/'prepared.json',meta)
    prep=read(data/'preparation_manifest.json')
    prep.update(subjects=[subject],data_role='software_only_four_plus_twelve_original_training_trials',smoke=True)
    write(output/'preparation_manifest.json',prep)
    return output


def worker(file: Path) -> None:
    job = read(file)
    if job['evaluation_route_sources'] != sources():
        raise ValueError('Evaluation routing changed')
    model = driver.load('model1022_fitted_variant', FIT)
    if 'decision_basis_target' in job:
        original = capture.driver.worker
        capture.driver.worker = model.worker
        try:
            capture.worker(file)
        finally:
            capture.driver.worker = original
    else:
        model.worker(file)
    write(Path(job['target']).with_suffix('.evaluation_route.json'), dict(
        compact_variant=job['compact_variant'], source_sha256=sources(),
        decision_basis_captured='decision_basis_target' in job))


class ResourceGate:
    """Bound evaluation plus the known final truth-scoring filters by CPU count."""
    def __init__(self, study: Path, maximum: int, remaining: int,
                 pending_stages: dict[Path,int] | None = None):
        self.study, self.maximum, self.remaining = study, maximum, remaining
        self.pending_stages=pending_stages
        if pending_stages is not None and (sum(pending_stages.values())!=remaining
                or any(n<=0 for n in pending_stages.values())):
            raise ValueError('Pending stage budgets must sum to the declared filter bound')
        self.condition = Condition()
        self.active = self.peak = 0
        self.events = []
        self.last_limit = None

    def limit(self) -> int:
        try:
            done = read(self.study/'completion.json')
        except (FileNotFoundError, json.JSONDecodeError):
            done = None
        if done and done['status']=='complete':
            return self.maximum
        remaining=self.remaining
        if self.pending_stages is not None:
            for file,count in self.pending_stages.items():
                try:
                    stage=read(file)
                except (FileNotFoundError,json.JSONDecodeError):
                    continue
                # Stage completion is written after subprocess.run has returned
                # for every filter. Individual score files can precede exit.
                if stage.get('status')=='complete' and stage.get('stage')=='confirmation':
                    remaining-=count
        return self.maximum-remaining

    def enter(self):
        with self.condition:
            while self.active >= self.limit():
                self.condition.wait(timeout=2)
            limit = self.limit()
            if limit != self.last_limit:
                self.events.append(dict(unix=time.time(), evaluation_limit=limit))
                self.last_limit = limit
            self.active += 1
            self.peak = max(self.peak, self.active)

    def leave(self):
        with self.condition:
            self.active -= 1
            self.condition.notify_all()


def launch(job: dict, folder: Path, gate: ResourceGate) -> None:
    directory = folder/'jobs'
    directory.mkdir(exist_ok=True)
    file = directory/(job['name']+'.json')
    gate.enter()
    try:
        write(file, job)
        launcher = ('import runpy,sys;runtime,script=sys.argv[1:3];sys.path.insert(0,runtime);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='compact_evaluation_') as cache:
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
                NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as log:
                result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT/ENTRY),
                    '--worker', str(file)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('Evaluation worker failed: '+str(file.with_suffix('.log')))
    finally:
        gate.leave()


def make_jobs(case: dict, output: Path, smoke: bool) -> tuple[list[dict], dict]:
    from src.Bayesian_state.utils.seeding import stable_seed
    fit = Path(case['output'])
    frozen, complete = read(fit/'nominees.json'), read(fit/'completion.json')
    if (complete['status']!='complete' or complete['test_choices_used'] is not False
            or complete['nomination_sha256']!=digest(fit/'nominees.json')
            or frozen['status']!='frozen_before_test' or frozen['smoke'] is not smoke):
        raise ValueError('Require the matching frozen prefix fit')
    for name, sha in frozen['source_sha256'].items():
        if digest(ROOT/name)!=sha:
            raise ValueError('Fitted scientific source changed: '+name)
    nominee, = frozen['nominees']
    if nominee['compact_variant']!=case['variant']:
        raise ValueError('Cognitive variant differs from fitted model')
    settings = frozen['settings']
    data=Path(frozen['data'])
    if smoke:
        data=smoke_fixture(data,output/'software_fixture',case['subject'])
    bank = {(case['subject'], 'COMPACT'): [nominee['candidate']]}
    groups, chunks = {}, []
    for role in ('conditional', 'autonomous'):
        originals = driver.io.score_jobs(data, output/role,
            driver.base.settings_for_scoring(settings), bank, 'evaluate', smoke)
        groups[role] = []
        for original in originals:
            job = deepcopy(original)
            job.update(arm_settings=settings['arms']['COMPACT'], source_sha256=frozen['source_sha256'],
                compact_variant=case['variant'], case_id=case['case_id'], evaluation_route_sources=sources())
            job['mechanism']['capacity'] = 'evidence'
            if case['variant']=='TRANSFER':
                job['entry_control'] = 'PRECISION'
            if role=='autonomous':
                job.update(phase='autonomous', name='AUTO_'+job['name'])
                job['seeds'] = [stable_seed(dict(protocol=settings['protocol'],base=settings['seed'],
                    role='autonomous',subject=case['subject'],path=i))
                    for i in range(2 if smoke else settings['autonomous_paths'])]
                audit = ROOT/settings['audit']/f'audit/subject_{case["subject"]}/audit.json'
                job.update(audit=str(audit), audit_sha256=digest(audit))
            width = settings['autonomous_chunk'] if role=='autonomous' else 1
            own = []
            for i in range(0,len(job['seeds']),width):
                part = deepcopy(job); target = Path(job['target']); suffix = f'_C{i:03d}'
                part.update(name=job['name']+suffix, seeds=job['seeds'][i:i+width],
                    target=str(target.parent/'chunks'/(target.name+suffix)))
                if role=='conditional':
                    part.update(decision_basis_target=str((output/'state_capture'/part['name']).resolve()),
                                decision_basis_sources=capture.sources())
                own.append(part); chunks.append(part)
            groups[role].append((job,own))
    chunks.sort(key=lambda j: -(j['particles'] if j['phase']=='evaluate' else len(j['seeds'])))
    return chunks, groups


def evaluate_case(case: dict, output: Path, smoke: bool, pool, gate: ResourceGate) -> dict:
    output.mkdir(parents=True)
    chunks, groups = make_jobs(case, output, smoke)
    fit = Path(case['output'])
    write(output/'protocol.json', dict(fit=str(fit), nomination_sha256=digest(fit/'nominees.json'),
        compact_variant=case['variant'], case_id=case['case_id'], cohort=case['cohort'], smoke=smoke,
        data_use='Original training choices selected parameters; later choices opened after freezing',
        route_source_sha256=sources(), jobs=chunks))
    write(output/'decision_basis_capture_protocol.json', dict(source_sha256=capture.sources(),
        route_source_sha256=sources(), additional_filters=0, timing='after_choice_before_current_feedback'))
    for future in as_completed([pool.submit(launch, job, output, gate) for job in chunks]):
        future.result()
    for role, records in groups.items():
        rows = [driver.io.combine_evaluation_chunks(job, parts, role=='autonomous') for job,parts in records]
        write(output/role/'evaluation.json', dict(status='complete',rows=rows))
    write(output/'completion.json',dict(status='complete',smoke=smoke,jobs=len(chunks),
        nomination_sha256=digest(fit/'nominees.json'), compact_variant=case['variant']))
    print(case['case_id']+': frozen evaluation complete',flush=True)
    return dict(case_id=case['case_id'], evaluation=str(output.resolve()), cohort=case['cohort'],
                variant=case['variant'], subject=case['subject'])


def run(study: Path, output: Path, smoke: bool) -> None:
    if output.exists():
        raise FileExistsError('Use a new frozen-evaluation directory')
    frozen = read(study/'human_nominees.json')
    recovered = read(study/'recovery_fit/completion.json')
    if (frozen['status']!='frozen_before_test' or frozen['smoke'] is not smoke
            or recovered['status']!='complete' or recovered['smoke'] is not smoke):
        raise ValueError('Freeze all human and recovery estimates before evaluating')
    all_cases = read(study/'cases.json')['cases']
    cases = [c for c in all_cases if c['kind']=='human']
    if len(cases)!=(2 if smoke else 12):
        raise ValueError('Require the complete registered set of human cases')
    pending = sum(c['settings']['budgets']['confirmation']['seeds'] for c in all_cases if c['kind']=='recovery')
    maximum = 1 if smoke else min(128,len(os.sched_getaffinity(0)))
    if smoke and not (study/'completion.json').exists():
        raise ValueError('Finish the short fitting fixture before its evaluation')
    pending_stages={study/'recovery_scores'/c['case_id']/'confirmation_completion.json':
        c['settings']['budgets']['confirmation']['seeds'] for c in all_cases if c['kind']=='recovery'}
    gate = ResourceGate(study,maximum,pending,pending_stages)
    if gate.limit()<=0:
        raise ValueError('Wait for generating-point scores to release the CPU allocation')
    output.mkdir(parents=True)
    write(output/'protocol.json', dict(study=str(study.resolve()), smoke=smoke, maximum_workers=maximum,
        pending_truth_filter_bound=pending, route_source_sha256=sources(),
        resource_release='Release each completed truth-score group; keep uncompleted groups inside the total cap',
        frozen_human_sha256=digest(study/'human_nominees.json'), cases_sha256=digest(study/'cases.json')))
    print(f'{len(cases)} cases; evaluation initially uses {gate.limit()} of {maximum} worker slots',flush=True)
    with ThreadPoolExecutor(max_workers=maximum) as pool:
        with ThreadPoolExecutor(max_workers=1 if smoke else len(cases)) as controls:
            futures = [controls.submit(evaluate_case,c,output/c['case_id'],smoke,pool,gate) for c in cases]
            records = [f.result() for f in futures]
    write(output/'completion.json',dict(status='complete',smoke=smoke,cases=records,
        peak_evaluation_workers=gate.peak,resource_events=gate.events,additional_filters_for_oral_capture=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('study','output','worker'):
        parser.add_argument('--'+name,type=Path)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.study and args.output:
        run(args.study,args.output,args.smoke)
    else:
        parser.error('Use --study and --output, or internal --worker')
