"""Check frozen finalists at higher particle budgets and replay autonomous states.

--phase filter, replay or both accepts --fit, --config and a new --output.
Every filter seed is a separate process task. Workers reuse the archived
cognitive runtime and the hash-checked H implementation used by the fit.
The existing training scorer runs unchanged; a read-only wrapper saves its
public numerical/state summaries. Replay records ALL saved autonomous paths
for the configured nominees, requiring exact equality of generated choices.
--smoke uses eight trials, four particles, one seed/path and one worker.
No future choices or new parameter proposals are used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[4]
BASE = 'src/Bayesian_state/workflows/analysis/model_0923_h_validation.py'
DIAGNOSIS = 'src/Bayesian_state/workflows/analysis/model_0923_path_diagnosis.py'
INTERVENTIONS = 'src/Bayesian_state/workflows/analysis/model_0923_mechanism_interventions.py'


def validation():
    spec = importlib.util.spec_from_file_location('precision_validation', REPO/BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_settings(settings: dict) -> None:
    if not settings['subjects'] or len(set(settings['subjects'])) != len(settings['subjects']):
        raise ValueError('Need distinct subjects')
    counts = settings['particle_counts']
    if sorted(set(counts)) != counts or any(not isinstance(p, int) or p < 2 for p in counts):
        raise ValueError('Particle counts must be increasing distinct integers >= 2')
    if settings['filter_repeats'] < 4 or settings['replay_chunk'] < 1 or settings['parallel_budget'] < 1:
        raise ValueError('Invalid repeat, chunk or worker budget')
    if set(settings['replay_models']) - {'NLL', 'GEN'}:
        raise ValueError('Unknown replay nominee')


def filter_seeds(settings: dict, subject: int) -> list[int]:
    from src.Bayesian_state.utils.seeding import stable_seed
    return [stable_seed(dict(protocol=settings['protocol'], base=settings['seed'],
                subject=subject, role='independent_precision_filter', repeat=i))
            for i in range(settings['filter_repeats'])]


def capture_public_arrays(result) -> dict:
    """Copy public outputs only; no particle state or random stream is modified."""
    import numpy as np
    captured = {'probabilities': np.asarray(result.observation_probabilities['prior_t']).copy()}
    for prefix, mapping in [('state', result.state_probabilities), ('latent', result.latent_summaries),
                            ('diagnostic', result.diagnostics)]:
        for name, value in mapping.items():
            if value is not None:
                array = np.asarray(value)
                if array.dtype.kind in 'bifu' and array.ndim in (1, 2, 3):
                    captured[prefix+'__'+name] = array.copy()
    return captured


def filter_job(job: dict, v, workflow) -> None:
    import numpy as np
    import src.Bayesian_state.inference.backends.particle_filter as pf
    original = pf.run_state_model_particle_filter
    captured = []

    def observe(**kwargs):
        result = original(**kwargs)
        captured.append(capture_public_arrays(result))
        return result

    pf.run_state_model_particle_filter = observe
    try:
        v.score(job, workflow)
    finally:
        pf.run_state_model_particle_filter = original
    if len(captured) != 1:
        raise ValueError('Each diagnostic job requires exactly one complete filter seed')
    target = Path(job['diagnostic_target'])
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as handle:
        np.savez_compressed(handle, **captured[0])
    if job['smoke']:
        check = dict(job, target=job['target']+'_uninstrumented')
        v.score(check, workflow)
        with np.load(Path(check['target']).with_suffix('.npz')) as reference:
            with np.load(Path(job['target']).with_suffix('.npz')) as instrumented:
                for key in reference.files:
                    np.testing.assert_array_equal(instrumented[key], reference[key])
    v.write(target.with_suffix('.json'), dict(predictions_sha256=v.digest(target.with_suffix('.npz')),
        keys=sorted(captured[0]), timing='public PF mappings; predictive and filtered names retained',
        instrumented_exact_check=bool(job['smoke']),
        undefined_fields=[key for key, value in captured[0].items() if not np.isfinite(value).all()]))


def replay_job(job: dict, v, workflow) -> None:
    import numpy as np
    from threadpoolctl import threadpool_info
    from src.Bayesian_state.simulation.autonomous import run_autonomous_category_learning
    from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
    folder = Path(job['folder']); meta = v.read(folder/'prepared.json')
    if v.digest(folder/'engine.json') != meta['engine_sha256']:
        raise ValueError('Prepared engine changed')
    if v.digest(folder/'training.npz') != meta['array_sha256']['training.npz']:
        raise ValueError('Training data changed')
    for name, sha in meta['frozen_cognitive_source_sha256'].items():
        if v.digest(Path(job['runtime'])/name) != sha:
            raise ValueError('Archived cognitive source changed')
    source = Path(job['reference'])
    if v.digest(source) != job['reference_sha256']:
        raise ValueError('Saved autonomous reference changed')
    with np.load(folder/'training.npz') as saved:
        stimulus, categories = saved['stimulus'], saved['categories']
    if job['smoke']:
        stimulus, categories = stimulus[:8], categories[:8]
    with np.load(source) as saved:
        np.testing.assert_array_equal(saved['categories'][:len(categories)], categories)
        np.testing.assert_array_equal(saved['seeds'][job['indices']], job['seeds'])
        expected = saved['choices'][job['indices'], :len(categories)]
    engine = v.read(folder/'engine.json')
    engine['modules']['hypo_transitions_mod']['kwargs']['intervention'] = job['mechanism']
    engine = apply_fixed_hyperparams_to_engine_config(engine, job['candidate']['hyperparams'])
    diagnosis = workflow.cohort().load_diagnosis()
    recorder = diagnosis.PathRecorder(categories, len(job['seeds']),
        engine['modules']['hypo_transitions_mod']['kwargs']['max_capacity'],
        category_count=2 if meta['condition'] == 1 else 4, has_pairing=meta['condition'] == 3)
    start, cpu = time.monotonic(), time.process_time()
    with diagnosis.record_lifecycle(recorder):
        for index, seed in enumerate(job['seeds']):
            result = run_autonomous_category_learning(engine_config=deepcopy(engine), subject_id=meta['subject'],
                condition=meta['condition'], stimulus=stimulus, categories=categories, trajectory_seed=seed,
                processed_data_dir=meta['paths']['processed_dir'], dataset_paths=meta['paths'])
            np.testing.assert_array_equal(result.trajectory.choices, expected[index])
            np.testing.assert_array_equal(recorder.arrays['prediction'][:, index], result.trajectory.observed_probabilities)
    recorder.validate()
    np.testing.assert_array_equal(recorder.arrays['choice'].T, expected)
    target = Path(job['target']); target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as handle:
        np.savez_compressed(handle, **recorder.arrays, seeds=job['seeds'], categories=categories)
    v.write(target.with_suffix('.json'), dict(subject=meta['subject'], model=job['model'],
        candidate=job['candidate'], paths=len(job['seeds']), trials=len(categories), exact_choice_replay=True,
        cpu_seconds=time.process_time()-cpu, seconds=time.monotonic()-start,
        numeric_pools=threadpool_info(), predictions_sha256=v.digest(target.with_suffix('.npz'))))


def worker(file: Path) -> None:
    v = validation(); job = v.read(file)
    for name, sha in job['source_sha256'].items():
        if v.digest(REPO/name) != sha:
            raise ValueError(f'Diagnostic implementation changed: {name}')
    workflow = v.intervention_workflow(); workflow.install_transition()
    diagnosis = workflow.cohort().load_diagnosis()
    with diagnosis.single_threaded_processes():
        if job['kind'] == 'filter':
            filter_job(job, v, workflow)
        else:
            replay_job(job, v, workflow)


def prepare_jobs(fit: Path, output: Path, settings: dict, phase: str, smoke: bool) -> tuple[list, dict]:
    import numpy as np
    v = validation(); frozen = v.read(fit/'nominees.json')
    if frozen['smoke'] or frozen['status'] != 'frozen_before_test':
        raise ValueError('Need a frozen scientific candidate set')
    for name, sha in frozen['source_sha256'].items():
        if v.digest(REPO/name) != sha:
            raise ValueError(f'Fitted implementation changed: {name}')
    data = Path(frozen['data'])
    prep = v.read(data/'preparation_manifest.json')
    runtime = str((REPO/prep['settings']['source_fit']/'source').resolve())
    hashes = dict(frozen['source_sha256'])
    for name in (str(Path(__file__).resolve().relative_to(REPO)), DIAGNOSIS, INTERVENTIONS,
                 'src/Bayesian_state/workflows/analysis/model_0923_cohort_path_diagnosis.py'):
        hashes[name] = v.digest(REPO/name)
    inputs = {str(fit/'nominees.json'): v.digest(fit/'nominees.json')}
    subjects = settings['subjects'][:1] if smoke else settings['subjects']
    jobs, points = [], []
    for subject in subjects:
        seeds = filter_seeds(settings, subject)[:1 if smoke else settings['filter_repeats']]
        folder = data/f'subject_{subject}'
        meta = v.read(folder/'prepared.json')
        files = list((fit/'confirmation'/f'subject_{subject}'/'NLL').glob('*.json'))
        if not files:
            raise ValueError(f'No final candidates for {subject}')
        rows = sorted([(v.read(file), file) for file in files], key=lambda pair: (pair[0]['mean_nll'], pair[0]['candidate']['id']))
        if smoke:
            rows = rows[:1]
        for index, (row, file) in enumerate(rows):
            inputs[str(file)] = v.digest(file)
            if set(seeds) & set(row['seeds']):
                raise ValueError('Audit seeds overlap original selection seeds')
            base = dict(folder=str(folder.resolve()), candidate=row['candidate'], mechanism=frozen['settings']['models']['NLL'],
                runtime=runtime, smoke=smoke, source_sha256=hashes, phase='training', subject=subject,
                original_letter=chr(65+index), training_trials=meta['training_trials'])
            points.append(dict(subject=subject, letter=chr(65+index), candidate=row['candidate'], old_nll=row['mean_nll']))
            if phase in ('filter', 'both'):
                for particles in ([4] if smoke else settings['particle_counts']):
                    for repeat, seed in enumerate(seeds):
                        name = f'PF_S{subject}_{row["candidate"]["id"]}_P{particles}_R{repeat:02d}'
                        jobs.append(dict(base, kind='filter', name=name, model='NLL', particles=particles,
                            seeds=[seed], repeat=repeat, target=str((output/'filters'/name).resolve()),
                            diagnostic_target=str((output/'diagnostics'/name).resolve())))
    if phase in ('replay', 'both'):
        selected = [row for row in frozen['nominees'] if row['subject'] in settings['replay_subjects']
                    and row['model'] in settings['replay_models']]
        if smoke:
            selected = selected[:1]
        for nominee in selected:
            subject, model, candidate = nominee['subject'], nominee['model'], nominee['candidate']
            reference = fit/'confirmation'/f'subject_{subject}'/'GEN'/(candidate['id']+'.npz')
            row = v.read(reference.with_suffix('.json'))
            if v.digest(reference) != row['predictions_sha256']:
                raise ValueError('Autonomous reference changed')
            inputs[str(reference)] = row['predictions_sha256']
            with np.load(reference) as saved:
                seeds = saved['seeds'].astype(int).tolist()
            if smoke:
                seeds = seeds[:1]
            folder = data/f'subject_{subject}'; meta = v.read(folder/'prepared.json')
            for first in range(0, len(seeds), settings['replay_chunk']):
                chunk = seeds[first:first+settings['replay_chunk']]
                name = f'REPLAY_S{subject}_{model}_{first:04d}'
                jobs.append(dict(kind='replay', name=name, subject=subject, model=model, candidate=candidate,
                    folder=str(folder.resolve()), mechanism=frozen['settings']['models'][model], runtime=runtime,
                    smoke=smoke, source_sha256=hashes, reference=str(reference.resolve()),
                    reference_sha256=row['predictions_sha256'], indices=list(range(first, first+len(chunk))),
                    seeds=chunk, training_trials=meta['training_trials'], target=str((output/'replay'/name).resolve())))
    protocol = dict(settings=settings, phase=phase, smoke=smoke, source_sha256=hashes, inputs_sha256=inputs,
        fit=str(fit.resolve()), data=str(data.resolve()), candidates=points, jobs=jobs,
        numerical_definition='Original pooled predictive probability NLL; full seed is uncertainty unit.',
        data_boundary='Training arrays only. Autonomous replay uses the exact saved seeds and external schedule.')
    return jobs, protocol


def run(fit: Path, config: Path, output: Path, phase: str, smoke: bool) -> None:
    import yaml
    from src.Bayesian_state.utils.parallel import parallel_job_count
    v = validation(); settings = yaml.safe_load(config.read_text()); check_settings(settings)
    if output.exists():
        raise FileExistsError('Use a new precision output directory')
    jobs, protocol = prepare_jobs(fit, output, settings, phase, smoke)
    output.mkdir(parents=True); (output/'jobs').mkdir()
    v.write(output/'protocol.json', protocol)
    shutil.copy2(__file__, output/'workflow_source.py')
    shutil.copy2(config, output/'config.yaml')
    workers = 1 if smoke else parallel_job_count(settings['parallel_budget'], len(jobs))
    jobs.sort(key=lambda j: -j['training_trials']*j.get('particles', len(j['seeds'])))

    def launch(job):
        file = output/'jobs'/(job['name']+'.json'); v.write(file, job)
        launcher = 'import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name="__main__")'
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='parameter_precision_') as cache, file.with_suffix('.log').open('x') as log:
            env['NUMBA_CACHE_DIR'] = cache
            subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(Path(__file__).resolve()),
                '--worker', str(file.resolve())], env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        return job['name']

    start = time.monotonic()
    print(f'{len(jobs)} jobs, {workers} workers, one numeric thread per worker', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, future in enumerate(as_completed([pool.submit(launch, job) for job in jobs]), 1):
            name = future.result()
            if index % 16 == 0 or index == len(jobs):
                print(f'{index}/{len(jobs)} complete; last={name}; wall={time.monotonic()-start:.1f}s', flush=True)
    records = [v.read(Path(j['target']).with_suffix('.json')) for j in jobs]
    if any(p['num_threads'] != 1 for r in records for p in r['numeric_pools']):
        raise ValueError('Numeric thread allocation exceeded')
    v.write(output/'completion.json', dict(status='complete', jobs=len(jobs), workers=workers, smoke=smoke,
        filter_runs=sum(j['kind'] == 'filter' for j in jobs), replay_paths=sum(r.get('paths', 0) for r in records),
        cpu_seconds=sum(r['cpu_seconds'] for r in records), wall_seconds=time.monotonic()-start,
        protocol_sha256=v.digest(output/'protocol.json')))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('fit', 'config', 'output', 'worker'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--phase', choices=['filter', 'replay', 'both'], default='both')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif all((args.fit, args.config, args.output)):
        run(args.fit, args.config, args.output, args.phase, args.smoke)
    else:
        parser.error('--fit, --config and --output are required')


if __name__ == '__main__':
    main()
