"""Factorial fixed-parameter study of recalled proposal and prior formation.

--smoke uses one subject, all four arms, two original seeds and sixteen trials.
The control replays the saved POOL paths exactly. Full runs use the declared
first 64 common path seeds on 207/208/308 and wait for prior production queues.
This is a mechanism probe at frozen POOL parameters, not a refitted ranking.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
HELPER = 'src/Bayesian_state/model/modules/hypothesis_transition/episodic_proposal.py'
REFIT = 'src/Bayesian_state/workflows/analysis/model_0923_episodic_refit.py'


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


def verify(job: dict) -> None:
    for name, expected in job['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Executing source changed: {name}')
    if digest(Path(job['reference'])) != job['reference_sha256']:
        raise ValueError('Original autonomous reference changed')


def filter_check(job: dict, refit, validation) -> dict:
    """Exercise snapshot cloning with explicit smoke-only resampling controls."""
    import numpy as np
    from src.Bayesian_state.inference.backends.particle_filter import run_state_model_particle_filter
    inputs = refit.base.load('recalled_filter_inputs', refit.INPUTS)
    engine, meta, _ = inputs.inputs(job, validation)
    with np.load(Path(job['folder']) / 'training.npz') as saved:
        arrays = {key: saved[key][:16] for key in ('stimulus', 'choices', 'feedback')}
    result = run_state_model_particle_filter(engine_config=engine, subject_id=meta['subject'], condition=meta['condition'],
        **arrays, particle_count=32, particle_blocks=1, filter_seed=job['seeds'][0],
        resample_threshold_fraction=1., choice_readout_power=1.,
        processed_data_dir=meta['paths']['processed_dir'], dataset_paths=meta['paths'])
    probability = np.asarray(result.observation_probabilities['prior_t'])
    np.testing.assert_allclose(probability.sum(axis=1), 1., atol=1e-12, rtol=0)
    if not np.isfinite(probability).all() or not np.any(result.resampled):
        raise ValueError('Particle smoke must exercise resampling and yield finite probabilities')
    target = Path(job['target']).with_suffix('.filter_smoke.npz')
    with target.open('xb') as stream:
        np.savez_compressed(stream, probabilities=probability, resampled=result.resampled)
    return dict(trials=16, particles=32, seed=job['seeds'][0], resample_threshold_fraction=1.,
                resampling_events=int(np.sum(result.resampled)), artifact_sha256=digest(target),
                role='software_validation_only')


def worker(file: Path) -> None:
    import numpy as np
    job = read(file); verify(job)
    refit = load('recalled_study_refit', REFIT)
    v = refit.base.validation(); workflow = v.intervention_workflow(); workflow.install_transition()
    diagnosis = workflow.cohort().load_diagnosis()
    feedback = refit.base.load('src.Bayesian_state.model.modules.hypothesis_transition.feedback_guided', refit.base.FEEDBACK)
    precision = refit.base.load('src.Bayesian_state.model.modules.precision_interventions', refit.base.PRECISION)
    guard = refit.base.load('src.Bayesian_state.model.modules.pairing_snapshot_roundoff', refit.GUARD)
    history = refit.base.load('src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission', refit.HELPER)
    prototype = load('src.Bayesian_state.model.modules.hypothesis_transition.episodic_proposal', HELPER)
    from src.Bayesian_state.model.modules.beta import BetaModule
    from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
    cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
    with ExitStack() as stack:
        stack.enter_context(diagnosis.single_threaded_processes())
        feedback_fixes = stack.enter_context(feedback.retention_roundoff_guard(cls))
        snapshot_fixes = stack.enter_context(guard.pairing_snapshot_roundoff_guard(HierarchicalPairingMemoryModule))
        stack.enter_context(precision.precision_intervention(BetaModule, mode=job['arm_settings']['precision'], upper=job['arm_settings']['upper']))
        recall_fixes = stack.enter_context(history.episodic_admission(cls, mode='POOL'))
        stack.enter_context(prototype.episodic_proposal(cls, **job['prototype']))
        refit.autonomous(job, workflow)
        particle_validation = filter_check(job, refit, v) if job['smoke'] else None
    target = Path(job['target'])
    record = read(target.with_suffix('.json'))
    if digest(target.with_suffix('.npz')) != record['predictions_sha256']:
        raise ValueError('Generated arrays changed')
    exact = None
    if job['model'] == 'POOL':
        with np.load(job['reference']) as previous, np.load(target.with_suffix('.npz')) as current:
            indices = np.array([np.flatnonzero(previous['seeds'] == seed).item() for seed in job['seeds']])
            for key in current.files:
                if key == 'seeds':
                    np.testing.assert_array_equal(current[key], previous[key][indices])
                elif key == 'categories':
                    np.testing.assert_array_equal(current[key], previous[key][:record['trials']])
                else:
                    np.testing.assert_array_equal(current[key], previous[key][indices, :record['trials']])
        exact = True
    verify(job)
    write(target.with_suffix('.validation.json'), dict(exact_control_replay=exact, prototype=job['prototype'],
        feedback_corrections=feedback_fixes, snapshot_corrections=snapshot_fixes, recall_corrections=recall_fixes,
        particle_smoke=particle_validation,
        data_use='frozen_POOL_behavior_parameters; no_oral_input; target_identity_for_recording_only'))


def prepare(config: Path, output: Path, smoke: bool, subjects: list[int] | None) -> tuple[dict, list[dict]]:
    import numpy as np
    import yaml
    settings = yaml.safe_load(config.read_text()); source = ROOT / settings['reference']
    if output.exists():
        raise FileExistsError('Use a new recalled-evidence study directory')
    if read(source / 'completion.json')['status'] != 'complete':
        raise ValueError('Original reference is incomplete')
    selected = subjects if subjects is not None else settings['subjects']
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(settings['subjects']):
        raise ValueError('Choose a unique subset of declared subjects')
    if smoke:
        selected = selected[:1]
    expected = {'POOL': dict(proposal=False, prior=False), 'PRIOR': dict(proposal=False, prior=True),
                'SEARCH': dict(proposal=True, prior=False), 'BOTH': dict(proposal=True, prior=True)}
    if settings['arms'] != expected:
        raise ValueError('Require the declared two-by-two mechanism comparison')
    hashes = {name: digest(ROOT / name) for name in (ENTRY, HELPER)}
    jobs = []
    for subject in selected:
        left = 2 if smoke else settings['paths']
        seeds = []
        for file in sorted((source / 'jobs').glob(f'AUTO_evaluate_S{subject}_POOL_*.json')):
            if left <= 0:
                break
            original = read(file)
            original['reference'] = str(Path(original['target']).with_suffix('.npz'))
            record = read(Path(original['reference']).with_suffix('.json'))
            original['reference_sha256'] = record['predictions_sha256']
            with np.load(original['reference']) as previous:
                np.testing.assert_array_equal(previous['seeds'], original['seeds'])
            original['seeds'] = original['seeds'][:left]
            seeds.extend(original['seeds']); left -= len(original['seeds'])
            for arm, flags in settings['arms'].items():
                name = f'S{subject}_{arm}_{len(seeds):03d}'
                job = dict(original, name=name, model=arm, prototype=flags, smoke=smoke,
                           target=str((output / 'paths' / name).resolve()),
                           source_sha256={**original['source_sha256'], **hashes},
                           original_job=str(file.resolve()), original_job_sha256=digest(file))
                verify(job); jobs.append(job)
        if left or len(set(seeds)) != len(seeds):
            raise ValueError('Insufficient distinct original path seeds')
    output.mkdir(parents=True); (output / 'jobs').mkdir(); (output / 'paths').mkdir()
    all_sources = {name: sha for job in jobs for name, sha in job['source_sha256'].items()}
    for name in all_sources:
        destination = output / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    shutil.copy2(config, output / 'protocol.yaml')
    protocol = dict(settings=settings, smoke=smoke, subjects=selected, jobs=len(jobs),
        workers=min(1 if smoke else settings['parallel_budget'], len(os.sched_getaffinity(0)), len(jobs)),
        source_sha256=all_sources, config_sha256=digest(config), reference=str(source.resolve()),
        original_completion_sha256=digest(source / 'completion.json'), python=sys.version,
        versions={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pandas', 'PyYAML')},
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        scope='fixed-POOL-parameter development mechanism probe; no parameter fitting; all paths retained')
    write(output / 'protocol.json', protocol)
    return protocol, jobs


def run(config: Path, output: Path, smoke: bool, subjects: list[int] | None = None) -> None:
    protocol, jobs = prepare(config, output, smoke, subjects)
    if not smoke:
        for relative in protocol['settings'].get('wait_for', []):
            predecessor = ROOT / relative
            while not (predecessor / 'completion.json').exists():
                print(f'Waiting for declared production queue: {relative}', flush=True)
                time.sleep(30)
            if read(predecessor / 'completion.json')['status'] != 'complete':
                raise ValueError('Predecessor production queue did not complete')
    def launch(job):
        file = output / 'jobs' / (job['name'] + '.json'); write(file, job)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        launcher = ('import runpy,sys;root,script=sys.argv[1:3];sys.path.insert(0,root);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        with tempfile.TemporaryDirectory(prefix='recalled_evidence_') as cache, file.with_suffix('.log').open('x') as log:
            env.update(NUMBA_CACHE_DIR=cache, MPLCONFIGDIR=cache)
            result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(ROOT / ENTRY),
                                     '--worker', str(file.resolve())], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Recalled-evidence worker failed: {file.with_suffix(".log")}')
        return read(Path(job['target']).with_suffix('.json'))
    print(f'{len(jobs)} jobs; {protocol["workers"]} workers; one numeric thread', flush=True)
    start, results = time.monotonic(), []
    with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
        for future in as_completed([pool.submit(launch, job) for job in jobs]):
            results.append(future.result())
            print(f'{len(results)}/{len(jobs)} complete', flush=True)
    if any(p['num_threads'] != 1 for r in results for p in r['numeric_pools']):
        raise ValueError('Numeric worker exceeded one thread')
    write(output / 'completion.json', dict(status='complete', smoke=smoke, jobs=len(jobs),
        paths=sum(r['paths'] for r in results), workers=protocol['workers'], seconds=time.monotonic()-start,
        cpu_seconds=sum(r['cpu_seconds'] for r in results),
        exact_control_paths=sum(r['paths'] for r in results if r['model'] == 'POOL')))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'output', 'worker'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--subjects', type=int, nargs='+')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.config is None or args.output is None:
        parser.error('config and output are required')
    else:
        run(args.config, args.output, args.smoke, args.subjects)


if __name__ == '__main__':
    main()
