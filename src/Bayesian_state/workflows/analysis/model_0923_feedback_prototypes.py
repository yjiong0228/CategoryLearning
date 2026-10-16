"""Frozen-point feedback-proposal, evidence-admission and precision experiments.

--stage oracle regenerates fixed-correct-rule paths after the old training
boundary. --stage autonomous runs the complete candidate mechanism from trial
one. --stage score evaluates pre-choice probabilities on training choices only;
score_confirm uses the separately declared numerical confirmation budget.
All participants are development data. Every destination must be new.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import importlib.metadata
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[4]
BASE = 'src/Bayesian_state/workflows/analysis/model_0923_h_rule_diagnosis.py'
FEEDBACK = 'src/Bayesian_state/model/modules/hypothesis_transition/feedback_guided.py'
PRECISION = 'src/Bayesian_state/model/modules/precision_interventions.py'


def workflow():
    spec = importlib.util.spec_from_file_location('prototype_base', REPO / BASE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def worker(file):
    import numpy as np
    from threadpoolctl import threadpool_info
    w = workflow(); v = w.validation(); job = v.read(file)
    intervention = v.intervention_workflow(); intervention.install_transition()
    d = intervention.cohort().load_diagnosis()
    helper = w.load('src.Bayesian_state.simulation.h_rule_diagnostic', w.HELPER)
    feedback = w.load('src.Bayesian_state.model.modules.hypothesis_transition.feedback_guided', FEEDBACK)
    precision = w.load('src.Bayesian_state.model.modules.precision_interventions', PRECISION)
    from src.Bayesian_state.model.modules.beta import BetaModule
    cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
    # Training scores load their arrays through v.score; future stimuli are not
    # opened on that path. Autonomous stages use the complete task schedule.
    scoring = job['stage'].startswith('score')
    if scoring:
        meta = v.read(Path(job['folder']) / 'prepared.json')
    else:
        engine, meta, arrays = w.inputs(job, v)
    arm = job['arm_settings']; started = time.process_time()
    with ExitStack() as stack:
        corrections = stack.enter_context(feedback.retention_roundoff_guard(cls))
        stack.enter_context(feedback.feedback_intervention(cls, proposal=arm['proposal'], admission=arm['admission']))
        precision_onset = (8 if job['smoke'] else meta['training_trials']) if job['stage'] == 'oracle' else 0
        stack.enter_context(precision.precision_intervention(BetaModule, mode=arm['precision'],
                                                          upper=arm['upper'], onset=precision_onset))
        if scoring:
            v.score(job, intervention)
            v.write(Path(job['target']).with_suffix('.roundoff.json'), dict(corrections=corrections))
            return
        n = 16 if job['smoke'] else len(arrays['categories'])
        onset = 8 if job['smoke'] else meta['training_trials']
        if job['stage'] == 'oracle':
            stack.enter_context(helper.intervention(cls, arm='ONLY', target=job['oracle'], onset=onset))
        categories = arrays['categories'][:n]
        bank = []

        class Recorder:
            def __init__(self):
                self.next_path = 0; self.rows = []
            def before(self, model, choice, outcome):
                prepared = model._pending_trial
                h = model.engine.get_module(d.ModuleRole.HYPOTHESIS_TRANSITION)
                ids = h.active; event = h.transition_log[-1]; target = job['oracle']
                self.rows.append(dict(present=int(target in ids),
                    contribution=float(h.execution_probabilities[target]),
                    beta=float(prepared.beta[target]) if target in ids else 0.,
                    proposed=-1 if event['proposed_hypothesis'] is None else event['proposed_hypothesis'],
                    search=int(event['search_event']), rejected=int(event.get('newcomer_rejected', False)),
                    candidate_evidence=event.get('candidate_initial_evidence', 0.),
                    evidence_available=int('candidate_initial_evidence' in event),
                    capacity_dropped=-1 if event['capacity_dropped'] is None else event['capacity_dropped'],
                    target_exited=int(target in event['autonomous_exits'])))
            def after(self, *args): pass

        with np.load(job['reference']) as z:
            if v.digest(Path(job['reference'])) != job['reference_sha256']:
                raise ValueError('Baseline reference changed')
            indices = [int(np.flatnonzero(z['seeds'] == seed)[0]) for seed in job['seeds']]
            expected_choice = z['choices'][indices, :n]
            expected_p = z['correct_probability'][indices, :n]
        for index, seed in enumerate(job['seeds']):
            recorder = Recorder()
            with d.record_lifecycle(recorder):
                result = d.autonomous.run_autonomous_category_learning(engine_config=engine,
                    subject_id=job['subject'], condition=meta['condition'], stimulus=arrays['stimulus'][:n],
                    categories=categories, trajectory_seed=seed, processed_data_dir=meta['paths']['processed_dir'],
                    dataset_paths=meta['paths'])
            p = np.asarray(result.trajectory.observed_probabilities)
            if not np.isfinite(p).all() or np.any(p < 0): raise ValueError('Invalid prediction')
            np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-12, rtol=0)
            row = {key: np.array([r[key] for r in recorder.rows]) for key in recorder.rows[0]}
            row.update(choice=np.asarray(result.trajectory.choices), prediction=p,
                       feedback=np.asarray(result.trajectory.feedback))
            if job['arm'] == 'BASE' or job['stage'] == 'oracle':
                end = onset if job['stage'] == 'oracle' else n
                np.testing.assert_array_equal(row['choice'][:end], expected_choice[index, :end])
                np.testing.assert_allclose(p[np.arange(end), categories[:end]-1], expected_p[index, :end], atol=1e-12, rtol=0)
            bank.append(row)
    target = Path(job['target']); target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as stream:
        np.savez_compressed(stream, **{key: np.stack([r[key] for r in bank]) for key in bank[0]},
                            seeds=job['seeds'], categories=categories, onset=onset, oracle=job['oracle'])
    v.write(target.with_suffix('.json'), dict(subject=job['subject'], model=job['model'], arm=job['arm'],
        paths=len(bank), trials=n, cpu_seconds=time.process_time()-started, numeric_pools=threadpool_info(),
        roundoff_corrections=corrections,
        predictions_sha256=v.digest(target.with_suffix('.npz'))))


def execute(jobs, output, budget):
    from src.Bayesian_state.utils.parallel import parallel_job_count
    v = workflow().validation(); (output / 'jobs').mkdir()
    def launch(job):
        file = output / 'jobs' / (job['name'] + '.json'); v.write(file, job)
        launcher = ('import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='feedback_prototype_') as cache:
            env['NUMBA_CACHE_DIR'] = cache
            with file.with_suffix('.log').open('x') as stream:
                result = subprocess.run([sys.executable, '-c', launcher, job['runtime'], str(Path(__file__).resolve()),
                    '--worker', str(file)], cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'Worker failed: {file.with_suffix(".log")}')
    workers = parallel_job_count(budget, len(jobs))
    print(f'{len(jobs)} jobs; {workers} workers; numeric threads=1', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(launch, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 32 == 0 or i == len(jobs): print(f'{i}/{len(jobs)} complete', flush=True)


def run(config, output, stage, smoke):
    import numpy as np
    import yaml
    w = workflow(); v = w.validation(); settings = yaml.safe_load(config.read_text())
    if output.exists(): raise FileExistsError('Use a new prototype output')
    frozen = v.read(REPO / settings['fit'] / 'nominees.json')
    if frozen['smoke'] or frozen['status'] != 'frozen_before_test': raise ValueError('Need scientific frozen fits')
    for name, sha in frozen['source_sha256'].items():
        if v.digest(REPO / name) != sha: raise ValueError('Baseline fitted source changed')
    data = Path(frozen['data']); prep = v.read(data / 'preparation_manifest.json')
    runtime = str((REPO / prep['settings']['source_fit'] / 'source').resolve())
    hashes = dict(v.source_hashes())
    for name in (BASE, FEEDBACK, PRECISION, w.HELPER, w.RECORDER,
                 str(Path(__file__).resolve().relative_to(REPO)),
                 'src/Bayesian_state/workflows/analysis/model_0923_mechanism_interventions.py',
                 'src/Bayesian_state/workflows/analysis/model_0923_cohort_path_diagnosis.py'):
        hashes[name] = v.digest(REPO / name)
    design = settings[stage]
    subjects = design['subjects'][:1] if smoke else design['subjects']
    models = design['models'][:1] if smoke else design['models']
    jobs = []
    for subject in subjects:
        audit = REPO / settings['audit'] / 'audit' / f'subject_{subject}' / 'audit.json'
        oracle = v.read(audit)['target']
        for model in models:
            row = next(r for r in frozen['nominees'] if r['subject'] == subject and r['model'] == model)
            reference = REPO / settings['reference'] / 'evaluate' / f'subject_{subject}' / model / (row['candidate']['id'] + '.npz')
            with np.load(reference) as z:
                seeds = z['seeds'][:1 if smoke else design.get('paths', 1)].astype(int).tolist()
            for arm in design['arms']:
                arm_settings = settings['arms'][arm]
                if stage == 'oracle' and (arm_settings['proposal'] or arm_settings['admission']):
                    raise ValueError('Oracle stage isolates precision')
                batches = [[s] for s in design['seeds'][:1 if smoke else None]] if stage.startswith('score') else [
                    seeds[i:i+settings['chunk']] for i in range(0, len(seeds), settings['chunk'])]
                for index, batch in enumerate(batches):
                    name = f'S{subject}_{model}_{arm}_{index:03d}'
                    jobs.append(dict(name=name, stage=stage, phase='training', subject=subject, model=model,
                        arm=arm, arm_settings=arm_settings, runtime=runtime, folder=str(data / f'subject_{subject}'),
                        candidate=row['candidate'], mechanism=frozen['settings']['models'][model],
                        oracle=oracle, source_sha256=hashes, smoke=smoke, seeds=batch,
                        particles=8 if smoke else design.get('particles', 0), reference=str(reference),
                        reference_sha256=v.digest(reference), target=str((output / 'chunks' / name).resolve())))
    output.mkdir(parents=True)
    shutil.copy2(config, output / 'config.yaml')
    for name in hashes:
        path = output / 'source' / name; path.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(REPO / name, path)
    v.write(output / 'protocol.json', dict(stage=stage, smoke=smoke, settings=settings, jobs=jobs,
        data_use='development; frozen parameters; training-only choice scoring', source_sha256=hashes,
        nomination_sha256=v.digest(REPO / settings['fit'] / 'nominees.json'),
        data_manifest_sha256=v.digest(data / 'preparation_manifest.json'),
        python=sys.version, versions={k: importlib.metadata.version(k) for k in ('numpy', 'scipy', 'pandas', 'PyYAML')}))
    execute(jobs, output, 1 if smoke else settings['parallel_budget'])
    records = [v.read(Path(j['target']).with_suffix('.json')) for j in jobs]
    v.write(output / 'completion.json', dict(status='complete', smoke=smoke, stage=stage, jobs=len(jobs),
        trajectories=0 if stage.startswith('score') else sum(r['paths'] for r in records),
        cpu_seconds=sum(r['cpu_seconds'] for r in records), protocol_sha256=v.digest(output / 'protocol.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--stage', choices=('oracle', 'autonomous', 'score', 'score_confirm'), default='autonomous')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker: worker(args.worker)
    elif args.config and args.output: run(args.config, args.output, args.stage, args.smoke)
    else: parser.error('Provide --config and --output')


if __name__ == '__main__': main()
