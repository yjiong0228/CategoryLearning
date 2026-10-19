"""Check one recorded rule's state on completed synthetic recovery sequences.

Use --config YAML --output NEW_DIRECTORY [--smoke]. The configuration selects
completed SEARCH confirmation jobs at the generating and frozen fitted points.
The original recovery worker installs every cognitive intervention. A read-only
observer records target membership, raw choice contribution and precision before
current feedback, both before and after choice conditioning. Behavioral arrays
must exactly reproduce the original jobs. Smoke uses one seed and eight trials
with the original particle count, on one worker.

Only the recorded target rule is evaluated. Parameters were estimated using the
whole synthetic training sequence; these are retrospective development checks.
The output does not establish whole-state recovery or population calibration.
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
import subprocess
import sys
import tempfile

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
PF = 'src/Bayesian_state/inference/backends/particle_filter.py'
RECOVERY = 'src/Bayesian_state/workflows/analysis/model_0923_joint_parameter_recovery.py'
PHASES = ('predictive', 'filtered')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def write(path: Path, value: dict) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify(bindings: dict) -> None:
    for path, expected in bindings.items():
        if digest(Path(path)) != expected:
            raise ValueError(f'State-recovery input changed: {path}')


class TargetCollector:
    """Retain detached particle values, with zero precision when absent.

    The zero is an encoding of joint membership/precision, matching generation.
    Conditional precision summaries must condition on active membership.
    """

    def __init__(self, target: int, trials: int):
        self.target, self.trials = target, trials
        self.rows = {phase: [] for phase in PHASES}

    def __call__(self, trial: int, phase: str, snapshot: dict) -> None:
        if phase not in PHASES or trial != len(self.rows[phase]) or trial >= self.trials:
            raise ValueError('Unexpected observer phase or trial order')
        w = np.asarray(snapshot['weights'])
        active = np.asarray(snapshot['active'])
        u, beta = (np.asarray(snapshot[k]) for k in ('choice_weights', 'beta'))
        if (w.ndim != 1 or active.ndim != 2 or active.shape != u.shape or u.shape != beta.shape
                or active.shape[0] != len(w) or not 0 <= self.target < active.shape[1]
                or not np.isin(active, [0, 1]).all()
                or not all(np.isfinite(x).all() for x in (w, u, beta))
                or np.any(w < 0) or np.any(u < 0) or np.any(beta < 0)):
            raise ValueError('Invalid particle state')
        np.testing.assert_allclose(w.sum(), 1., atol=2e-12, rtol=0)
        np.testing.assert_allclose(u.sum(axis=1), 1., atol=2e-12, rtol=0)
        np.testing.assert_allclose(u[~active.astype(bool)], 0., atol=2e-12, rtol=0)
        a = active[:, self.target].astype(bool)
        self.rows[phase].append(dict(weights=w.copy(), present=a.copy(),
            contribution=u[:, self.target].copy(), active_beta=np.where(a, beta[:, self.target], 0.)))

    def arrays(self) -> dict:
        if any(len(self.rows[p]) != self.trials for p in PHASES):
            raise ValueError('Incomplete observer sequence')
        return {f'{phase}_{key}': np.stack([row[key] for row in self.rows[phase]])
                for phase in PHASES for key in ('weights', 'present', 'contribution', 'active_beta')}


def worker(file: Path) -> None:
    """Add observational capture to the unchanged original recovery worker."""
    job = read(file)
    verify(job['bindings'])
    original = read(Path(job['original_job']))
    actual = deepcopy(original)
    actual.update(target=job['score_target'], smoke=job['smoke'])
    actual_file = file.with_name(file.stem + '_executed.json')
    write(actual_file, actual)
    # The subprocess imports cognition from the original archived runtime.
    # Load the shared observer-enabled filter by its package-qualified alias.
    from src.Bayesian_state.inference.backends import particle_filter as archived
    capture = load('src.Bayesian_state.inference.backends._target_recovery_capture', PF)
    recovery = load('target_recovery_original_worker', RECOVERY)
    collector = TargetCollector(job['target_rule'], job['trials'])
    original_filter = archived.run_state_model_particle_filter
    calls = []

    def observe(**kwargs):
        if calls or kwargs['filter_seed'] != original['seeds'][0]:
            raise ValueError('Require exactly the declared original filter seed')
        calls.append(kwargs['filter_seed'])
        return capture.run_state_model_particle_filter(**kwargs, state_observer=collector)

    archived.run_state_model_particle_filter = observe
    try:
        recovery.worker(actual_file)
    finally:
        archived.run_state_model_particle_filter = original_filter
    if calls != original['seeds']:
        raise ValueError('Original worker did not execute its declared filter')
    score_target = Path(job['score_target'])
    with np.load(job['reference']) as reference, np.load(score_target.with_suffix('.npz')) as actual_arrays:
        for key in ('observed_probability', 'correct_probability'):
            np.testing.assert_array_equal(actual_arrays[key], reference[key][:, :job['trials']])
        np.testing.assert_array_equal(actual_arrays['seeds'], reference['seeds'])
    target = Path(job['target'])
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **collector.arrays(), seeds=original['seeds'])
    verify(job['bindings'])
    write(target.with_suffix('.json'), dict(status='complete', exact_behavior_replay=True,
        target_rule=job['target_rule'], trials=job['trials'], seeds=original['seeds'],
        particles=original['particles'], parameter_id=original['candidate']['id'],
        timing='Both phases precede current feedback; filtered includes current choice',
        scope='One target rule; fitted parameters use the full synthetic training sequence',
        array_sha256=digest(target.with_suffix('.npz')), score_record=read(score_target.with_suffix('.json'))))


def prepare(config: Path, output: Path, smoke: bool) -> dict:
    """Bind the generation trace and every selected original confirmation job."""
    if output.exists():
        raise FileExistsError('Use a new target-state recovery directory')
    settings = yaml.safe_load(config.read_text())
    if (set(settings) != {'prepared', 'parallel_budget', 'groups'}
            or type(settings['parallel_budget']) is not int or not 1 <= settings['parallel_budget'] <= 32):
        raise ValueError('Declare a prepared recovery, groups and up to 32 workers')
    bindings = {}

    def bind(path):
        path = Path(path).resolve()
        bindings[str(path)] = digest(path)
        return path

    for path in (config, ROOT/ENTRY, ROOT/PF, ROOT/RECOVERY):
        bind(path)
    prepared = ROOT/settings['prepared']
    protocol, done, truths = [read(bind(prepared/name)) for name in ('protocol.json', 'completion.json', 'truths.json')]
    if (done['status'] != 'complete' or done['protocol_sha256'] != digest(prepared/'protocol.json')
            or done['truths_sha256'] != digest(prepared/'truths.json')):
        raise ValueError('Original recovery preparation changed')
    cases = {c['replicate']: c for c in protocol['cases']}
    truths = {t['replicate']: t for t in truths['rows']}
    generation = Path(protocol['source'])/'banks/MIX_LOG25_autonomous.npz'
    record = read(bind(generation.with_suffix('.json')))
    if digest(bind(generation)) != record['predictions_sha256']:
        raise ValueError('Synthetic state bank changed')
    source_jobs = sorted((Path(protocol['source'])/'jobs').glob('autonomous_MIX_LOG25_*.json'))
    audit_files = {read(bind(p))['audit'] for p in source_jobs}
    targets = {read(bind(p))['target'] for p in audit_files}
    if len(targets) != 1:
        raise ValueError('Require one recorded target rule')
    target_rule = int(next(iter(targets)))
    groups = settings['groups'][:1] if smoke else settings['groups']
    if len({g['label'] for g in groups}) != len(groups):
        raise ValueError('Duplicate state comparison labels')
    jobs, truth_arrays = [], {}
    seeds_by_replicate = {}
    with np.load(generation) as bank:
        for group in groups:
            if set(group) != {'label', 'replicate', 'source', 'point'} or group['point'] not in ('truth', 'nominee'):
                raise ValueError('Each group must select a completed truth point or nominee')
            label, replicate = group['label'], group['replicate']
            if not label.replace('_', '').isalnum():
                raise ValueError('Use simple distinct group labels')
            case = cases[replicate]
            source = ROOT/group['source']
            completed = read(bind(source/'completion.json'))
            if completed['status'] != 'complete' or completed['smoke']:
                raise ValueError('State checks require complete nonsmoke recovery scores')
            if group['point'] == 'nominee':
                frozen = read(bind(source/'nominees.json'))
                if (frozen['status'] != 'frozen_before_test'
                        or completed['nomination_sha256'] != digest(source/'nominees.json')):
                    raise ValueError('Fitted point is not frozen')
                candidates = [r['candidate'] for r in frozen['nominees']
                              if r['subject'] == case['subject'] and r['model'] == 'SEARCH']
            else:
                candidates = [truths[replicate]['candidate']]
            if len(candidates) != 1:
                raise ValueError('Require one SEARCH point')
            candidate = candidates[0]
            files = sorted((source/'jobs').glob(f'confirmation_S{case["subject"]}_SEARCH_{candidate["id"]}_C*.json'))
            if len(files) != 8:
                raise ValueError('Preserve the eight completed confirmation filters')
            selected = files[:1] if smoke else files
            group_seeds = []
            index, = np.flatnonzero(bank['seeds'] == case['generation_seed'])
            folder = prepared/f'replicate_{replicate}/data'/f'subject_{case["subject"]}'
            meta = read(bind(folder/'prepared.json'))
            for name, expected in case['prepared_sha256'].items():
                if digest(bind(folder/name)) != expected:
                    raise ValueError('Synthetic input changed')
            trials = 8 if smoke else case['training_trials']
            with np.load(folder/'training.npz') as arrays:
                for key in ('choices', 'feedback'):
                    np.testing.assert_array_equal(arrays[key], bank[key][index, :case['training_trials']])
            for key in ('present', 'contribution', 'beta'):
                truth_arrays[f'r{replicate}_{key}'] = bank[key][index, :trials].copy()
            truth_arrays[f'r{replicate}_generation_seed'] = np.array(case['generation_seed'])
            for file in selected:
                own = read(bind(file))
                reference = Path(own['target']).with_suffix('.npz')
                score = read(bind(reference.with_suffix('.json')))
                if (own['phase'] != 'training' or own['smoke'] or own['model'] != 'SEARCH'
                        or own['candidate']['hyperparams'] != candidate['hyperparams']
                        or Path(own['folder']).resolve() != folder.resolve()
                        or len(own['seeds']) != 1 or own['particles'] != 512
                        or own['seeds'][0] == case['generation_seed']
                        or own['retain_precision'] is not True or own['cohort_belief_mechanism'] != 'SEPARATE'
                        or score['predictions_sha256'] != digest(bind(reference))):
                    raise ValueError('Original confirmation identity or recovery data differs')
                for name, expected in own['source_sha256'].items():
                    if digest(bind(ROOT/name)) != expected:
                        raise ValueError('Original recovery implementation changed')
                for name, expected in meta['frozen_cognitive_source_sha256'].items():
                    if digest(bind(Path(own['runtime'])/name)) != expected:
                        raise ValueError('Archived cognition changed')
                for path, expected in own['retention_reference_inputs'].items():
                    if digest(bind(path)) != expected:
                        raise ValueError('Original recovery binding changed')
                bind(Path(own['runtime'])/PF)
                name = label + '_' + file.stem.rsplit('_', 1)[-1]
                jobs.append(dict(name=name, label=label, replicate=replicate, original_job=str(file.resolve()),
                    runtime=own['runtime'], reference=str(reference.resolve()), target_rule=target_rule,
                    trials=trials, smoke=smoke, score_target=str((output/'scores'/name).resolve()),
                    target=str((output/'states'/name).resolve())))
                group_seeds.extend(own['seeds'])
            if len(set(group_seeds)) != len(group_seeds):
                raise ValueError('Repeated filter seed')
            if replicate in seeds_by_replicate and group_seeds != seeds_by_replicate[replicate]:
                raise ValueError('Parameter groups must share the original confirmation seeds')
            seeds_by_replicate[replicate] = group_seeds
    output.mkdir(parents=True)
    for name in ('jobs', 'states', 'scores'):
        (output/name).mkdir()
    with (output/'truth.npz').open('xb') as stream:
        np.savez_compressed(stream, **truth_arrays)
    bind(output/'truth.npz')
    for job in jobs:
        job['bindings'] = bindings
    workers = min(1 if smoke else settings['parallel_budget'], len(os.sched_getaffinity(0)), len(jobs))
    result = dict(settings=settings, smoke=smoke, workers=workers, target_rule=target_rule,
        jobs=jobs, bindings=bindings,
        scope='Retrospective target-rule recovery on two training sequences; no oral or held-out input')
    write(output/'protocol.json', result)
    return result


def run(config: Path, output: Path, smoke: bool) -> None:
    protocol = prepare(config, output, smoke)
    print(f"{len(protocol['jobs'])} fixed confirmation replays, {protocol['workers']} workers", flush=True)

    def launch(job):
        file = output/'jobs'/(job['name']+'.json')
        write(file, job)
        launcher = ('import runpy,sys;root,script=sys.argv[1:3];sys.path.insert(0,root);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='target_state_recovery_') as cache:
            env.update(NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            with file.with_suffix('.log').open('x') as stream:
                result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT/ENTRY),
                    '--worker', str(file.resolve())], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"State replay failed: {file.with_suffix('.log')}")
        record = read(Path(job['target']).with_suffix('.json'))
        if digest(Path(job['target']).with_suffix('.npz')) != record['array_sha256']:
            raise ValueError('Recorded states changed')
        return dict(label=job['label'], replicate=job['replicate'], **record)

    records = []
    with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
        for future in as_completed([pool.submit(launch, job) for job in protocol['jobs']]):
            records.append(future.result())
            print(f"{len(records)}/{len(protocol['jobs'])} exact state replays complete", flush=True)
    verify(protocol['bindings'])
    write(output/'completion.json', dict(status='complete', smoke=smoke, jobs=len(records),
        protocol_sha256=digest(output/'protocol.json'), records=records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'output', 'worker'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.config and args.output:
        run(args.config, args.output, args.smoke)
    else:
        parser.error('Provide --config and a new --output, or --worker')
