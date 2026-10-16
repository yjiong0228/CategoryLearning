"""Frozen-parameter discovery, rule-use and candidate-admission diagnostics.

Audit all declared participants, then intervene on the targeted participants.
Oracle rules are selected using training-stimulus geometry and validated against
the remaining recorded task schedule. Human choices enter retrospective scoring
only. BASE and REPLAY must reproduce saved autonomous choices and probabilities.
--smoke uses one participant/model/path, 16 trials, onset 8, and one worker.
Every destination is new. Outputs preserve all paths, sparse rule states, RNG
tapes, source hashes, data hashes and the exact frozen parameter points.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import importlib.util
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[4]
HELPER = 'src/Bayesian_state/simulation/h_rule_diagnostic.py'
BASE = 'src/Bayesian_state/workflows/analysis/model_0923_h_validation.py'
RECORDER = 'src/Bayesian_state/workflows/analysis/model_0923_path_diagnosis.py'


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validation():
    return load('rule_diagnosis_validation', BASE)


def inputs(job, v):
    import numpy as np
    from src.Bayesian_state.simulation.parameters import apply_fixed_hyperparams_to_engine_config
    folder = Path(job['folder']); meta = v.read(folder / 'prepared.json')
    for name, sha in job['source_sha256'].items():
        if v.digest(REPO / name) != sha: raise ValueError('Diagnostic source changed')
    for name, sha in meta['frozen_cognitive_source_sha256'].items():
        if v.digest(Path(job['runtime']) / name) != sha: raise ValueError('Archived core changed')
    if v.digest(folder / 'engine.json') != meta['engine_sha256']: raise ValueError('Engine changed')
    engine = v.read(folder / 'engine.json')
    engine['modules']['hypo_transitions_mod']['kwargs']['intervention'] = job['mechanism']
    engine = apply_fixed_hyperparams_to_engine_config(engine, job['candidate']['hyperparams'])
    parts = []
    for name in ('training.npz', 'sealed_test.npz'):
        if v.digest(folder / name) != meta['array_sha256'][name]: raise ValueError('Arrays changed')
        with np.load(folder / name) as saved:
            # Neither oracle selection nor autonomous generation reads human choices.
            parts.append({k: saved[k] for k in ('stimulus', 'categories')})
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return engine, meta, arrays


def audit(job, v):
    import numpy as np
    import pandas as pd
    from src.Bayesian_state.model import StateModel, ModelContext
    engine, meta, arrays = inputs(job, v)
    model = StateModel(engine, context=ModelContext(subject_id=job['subject'], condition=meta['condition'],
        processed_data_dir=meta['paths']['processed_dir'], dataset_paths=meta['paths']))
    x, y = arrays['stimulus'], arrays['categories']; cut = meta['training_trials']
    rows = []
    for h in range(model.engine.set_size):
        p = model.partition_model.get_category_probabilities(hypo=h,
            data=(x, np.ones(len(x), int), np.ones(len(x))), beta=job['geometry_beta'],
            distance_mode=model.engine.distance_mode)
        maxima = np.isclose(p, p.max(axis=0), atol=1e-12, rtol=0)
        score = maxima[y - 1, np.arange(len(y))] / maxima.sum(axis=0)
        rows.append(dict(subject=job['subject'], rule=h, training_accuracy=float(score[:cut].mean()),
            test_accuracy=float(score[cut:].mean()), full_accuracy=float(score.mean()),
            specification=repr(model.partition_model.hypothesis_space[h])))
    frame = pd.DataFrame(rows)
    perfect = frame.loc[frame.training_accuracy == 1, 'rule'].astype(int).tolist()
    target = min(perfect) if perfect else None
    output = Path(job['target']); output.mkdir(parents=True)
    frame.to_csv(output / 'catalog.csv', index=False)
    v.write(output / 'audit.json', dict(subject=job['subject'], condition=meta['condition'],
        training_trials=cut, trials=len(y), target=target, training_perfect_rules=perfect,
        full_perfect_rules=frame.loc[frame.full_accuracy == 1, 'rule'].astype(int).tolist(),
        target_test_accuracy=None if target is None else rows[target]['test_accuracy'],
        selection='smallest_id_among_training_geometry_perfect_rules',
        scope='recorded_physical_stimuli_not_all_possible_task_inputs',
        catalog_sha256=v.digest(output / 'catalog.csv')))


def trace(job, v, w):
    import numpy as np
    from threadpoolctl import threadpool_info
    helper = load('src.Bayesian_state.simulation.h_rule_diagnostic', HELPER)
    d = w.cohort().load_diagnosis()
    cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
    engine, meta, arrays = inputs(job, v)
    n = 16 if job['smoke'] else len(arrays['categories'])
    categories = arrays['categories'][:n]; onset = 8 if job['smoke'] else meta['training_trials']
    kwargs = dict(engine_config=engine, subject_id=job['subject'], condition=meta['condition'],
        stimulus=arrays['stimulus'][:n], categories=categories,
        processed_data_dir=meta['paths']['processed_dir'], dataset_paths=meta['paths'])
    capacity = engine['modules']['hypo_transitions_mod']['kwargs']['max_capacity']
    reference = Path(job['reference'])
    if v.digest(reference) != job['reference_sha256']: raise ValueError('Autonomous reference changed')
    with np.load(reference) as saved:
        indices = [int(np.flatnonzero(saved['seeds'] == seed)[0]) for seed in job['seeds']]
        expected_choice = saved['choices'][indices, :n]
        expected_probability = saved['correct_probability'][indices, :n]
    target = Path(job['target']); target.mkdir(parents=True)
    start, cpu = time.monotonic(), time.process_time()
    banks = {arm: [] for arm in job['arms']}; tapes = []

    class Recorder(d.PathRecorder):
        def __init__(self):
            super().__init__(categories, 1, capacity, 2 if meta['condition'] == 1 else 4, meta['condition'] == 3)
            for key in ('search_probability', 'rejected', 'already_available', 'prevented_exit', 'prevented_capacity'):
                self.arrays[key] = np.zeros((n, 1))
            for key in ('retention_evidence', 'retention_exposure'):
                self.arrays[key] = np.full(self.arrays['rule'].shape, np.nan)

        def before(self, model, choice, feedback):
            token = super().before(model, choice, feedback)
            row, ids, _, _ = token
            transition = model.engine.get_module(d.ModuleRole.HYPOTHESIS_TRANSITION)
            event = transition.transition_log[-1]
            self.arrays['search_probability'][row[:2]] = transition.current_event_probability
            for key, field in [('rejected', 'newcomer_rejected'), ('already_available', 'tape_already_available'),
                               ('prevented_exit', 'oracle_prevented_exit'), ('prevented_capacity', 'oracle_prevented_capacity')]:
                self.arrays[key][row[:2]] = event.get(field, False)
            for key in ('retention_evidence', 'retention_exposure'):
                self.arrays[key][row] = getattr(transition, key)[ids]
            return token

    for index, seed in enumerate(job['seeds']):
        baseline = None; tape = None
        for arm in [*job['arms'], 'REPLAY']:
            recorder = Recorder()
            with helper.intervention(cls, arm=arm, target=job['oracle'], onset=onset, tape=tape) as recorded:
                with d.record_lifecycle(recorder):
                    result = d.autonomous.run_autonomous_category_learning(**kwargs, trajectory_seed=seed)
            recorder.validate()
            a = {k: value[:, 0].copy() for k, value in recorder.arrays.items()}
            p = np.asarray(result.trajectory.observed_probabilities)
            np.testing.assert_array_equal(a['prediction'], p)
            if arm == 'BASE':
                baseline, tape = a, recorded
                tapes.append(dict(seed=seed, transitions=tape))
                np.testing.assert_array_equal(a['choice'], expected_choice[index])
                np.testing.assert_allclose(p[np.arange(n), categories - 1], expected_probability[index], atol=1e-12, rtol=0)
            else:
                # Every arm is identical through the trial preceding onset.
                for key in ('prediction', 'choice', 'rule', 'contribution'):
                    np.testing.assert_array_equal(a[key][:onset], baseline[key][:onset])
            if arm == 'REPLAY':
                for key in ('prediction', 'choice', 'rule', 'contribution', 'proposed', 'exits', 'capacity_dropped'):
                    np.testing.assert_array_equal(a[key], baseline[key])
            else:
                if arm in ('ONCE', 'ONCE_EQUAL', 'KEEP', 'KEEP_EQUAL', 'ONLY'):
                    assert job['oracle'] in a['rule'][onset]
                if arm in ('KEEP', 'KEEP_EQUAL', 'ONLY'):
                    assert np.all(np.any(a['rule'][onset:] == job['oracle'], axis=1))
                banks[arm].append(a)
    files = {}
    for arm, rows in banks.items():
        path = target / (arm + '.npz')
        with path.open('xb') as stream:
            np.savez_compressed(stream, **{k: np.stack([r[k] for r in rows]) for k in rows[0]},
                seeds=job['seeds'], categories=categories, onset=onset, oracle=job['oracle'])
        files[path.name] = v.digest(path)
    v.write(target / 'proposal_rng_tapes.json', dict(paths=tapes))
    files['proposal_rng_tapes.json'] = v.digest(target / 'proposal_rng_tapes.json')
    v.write(target / 'completion.json', dict(subject=job['subject'], model=job['model'], paths=len(job['seeds']),
        arms=job['arms'], baseline_exact=True, replay_exact=True, preintervention_exact=True,
        seconds=time.monotonic()-start, cpu_seconds=time.process_time()-cpu, artifact_sha256=files,
        numeric_pools=threadpool_info()))


def worker(file):
    v = validation(); job = v.read(file); w = v.intervention_workflow(); w.install_transition()
    if job['phase'] == 'audit': audit(job, v)
    else: trace(job, v, w)


def execute(jobs, output, budget):
    from src.Bayesian_state.utils.parallel import parallel_job_count
    v = validation(); folder = output / 'jobs'; folder.mkdir(exist_ok=True)
    workers = parallel_job_count(budget, len(jobs))
    def launch(job):
        path = folder / (job['name'] + '.json'); v.write(path, job)
        launcher = ('import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='h_rule_diagnosis_') as cache:
            env['NUMBA_CACHE_DIR'] = cache
            with path.with_suffix('.log').open('x') as stream:
                result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(Path(__file__).resolve()),
                    '--worker', str(path)], cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'Worker failed: {path.with_suffix(".log")}')
    print(f'{len(jobs)} jobs, {workers} workers, numeric threads=1', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(launch, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 16 == 0 or i == len(jobs): print(f'{i}/{len(jobs)} complete', flush=True)


def run(config, output, smoke):
    import numpy as np
    import yaml
    v = validation(); settings = yaml.safe_load(config.read_text())
    if settings['onset'] != 'training_boundary': raise ValueError('Only the frozen training boundary is supported')
    if settings['arms'] != ['BASE', 'ONCE', 'ONCE_EQUAL', 'KEEP', 'KEEP_EQUAL', 'ONLY', 'REJECT']:
        raise ValueError('The complete ordered diagnostic panel is required')
    for key in ('paths', 'chunk', 'parallel_budget'):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if output.exists(): raise FileExistsError('Use a new diagnostic output')
    fit = REPO / settings['fit']; frozen = v.read(fit / 'nominees.json')
    if frozen['smoke'] or frozen['status'] != 'frozen_before_test': raise ValueError('Need frozen scientific estimates')
    for name, sha in frozen['source_sha256'].items():
        if v.digest(REPO / name) != sha: raise ValueError('Fitted implementation changed')
    data = Path(frozen['data']); preparation = v.read(data / 'preparation_manifest.json')
    runtime = str((REPO / preparation['settings']['source_fit'] / 'source').resolve())
    hashes = dict(v.source_hashes())
    dependencies = (HELPER, RECORDER, str(Path(__file__).resolve().relative_to(REPO)),
        'src/Bayesian_state/workflows/analysis/model_0923_mechanism_interventions.py',
        'src/Bayesian_state/workflows/analysis/model_0923_cohort_path_diagnosis.py')
    for name in dependencies: hashes[name] = v.digest(REPO / name)
    output.mkdir(parents=True)
    for name in hashes:
        archived = output / 'source' / name; archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, archived)
    audit_jobs = []
    audit_subjects = settings['subjects'][:1] if smoke else settings['audit_subjects']
    for subject in audit_subjects:
        nominee = next(r for r in frozen['nominees'] if r['subject'] == subject and r['model'] == 'FREE')
        audit_jobs.append(dict(phase='audit', name=f'audit_{subject}', subject=subject, runtime=runtime,
            folder=str(data / f'subject_{subject}'), mechanism=frozen['settings']['models']['FREE'],
            candidate=nominee['candidate'], geometry_beta=settings['geometry_beta'], source_sha256=hashes,
            target=str((output / 'audit' / f'subject_{subject}').resolve())))
    v.write(output / 'protocol.json', dict(settings=settings, smoke=smoke, source_sha256=hashes,
        nomination_sha256=v.digest(fit / 'nominees.json'), data_manifest_sha256=v.digest(data / 'preparation_manifest.json'),
        intervention_onset='trial immediately after training boundary; no observed prefix conditioning',
        selection='all nine geometry audits; 207,208,308 process failures from previous study',
        oracle_information='task labels on training stimuli; no human choices', audit_jobs=audit_jobs))
    shutil.copy2(config, output / 'config.yaml')
    execute(audit_jobs, output, 1 if smoke else settings['parallel_budget'])
    subjects = settings['subjects'][:1] if smoke else settings['subjects']
    models = settings['models'][:1] if smoke else settings['models']
    jobs = []
    for subject in subjects:
        geometry = v.read(output / 'audit' / f'subject_{subject}' / 'audit.json')
        if geometry['target'] is None or geometry['target_test_accuracy'] != 1:
            raise ValueError('A training-perfect task rule must also cover the remaining recorded stimuli')
        for model in models:
            row = next(r for r in frozen['nominees'] if r['subject'] == subject and r['model'] == model)
            reference = REPO / settings['reference'] / 'evaluate' / f'subject_{subject}' / model / (row['candidate']['id'] + '.npz')
            meta = v.read(reference.with_suffix('.json'))
            if v.digest(reference) != meta['predictions_sha256']: raise ValueError('Reference changed')
            with np.load(reference) as z: seeds = z['seeds'][:1 if smoke else settings['paths']].astype(int).tolist()
            if len(seeds) != (1 if smoke else settings['paths']): raise ValueError('Insufficient reference paths')
            for index in range(0, len(seeds), settings['chunk']):
                name = f'S{subject}_{model}_{index:03d}'
                jobs.append(dict(phase='trace', name=name, subject=subject, model=model, runtime=runtime,
                    folder=str(data / f'subject_{subject}'), mechanism=frozen['settings']['models'][model],
                    candidate=row['candidate'], source_sha256=hashes, oracle=geometry['target'], smoke=smoke,
                    seeds=seeds[index:index+settings['chunk']], arms=settings['arms'], reference=str(reference.resolve()),
                    reference_sha256=meta['predictions_sha256'], target=str((output / 'traces' / name).resolve())))
    v.write(output / 'trace_plan.json', dict(jobs=jobs))
    execute(jobs, output, 1 if smoke else settings['parallel_budget'])
    rows = [v.read(Path(j['target']) / 'completion.json') for j in jobs]
    v.write(output / 'completion.json', dict(status='complete', smoke=smoke, jobs=len(jobs),
        paths=sum(r['paths'] * len(r['arms']) for r in rows), replay_paths=sum(r['paths'] for r in rows),
        cpu_seconds=sum(r['cpu_seconds'] for r in rows), source_sha256=hashes,
        versions={key: importlib.metadata.version(key) for key in ('numpy', 'scipy', 'pandas', 'PyYAML', 'threadpoolctl')},
        python=sys.version))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.config and args.output: run(args.config, args.output, args.smoke)
    else: parser.error('Provide --config and --output')


if __name__ == '__main__': main()
