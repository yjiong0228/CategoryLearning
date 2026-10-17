"""Frozen-parameter experiments in retrospective candidate evidence.

BASE replays the previous additive-rate refit. LAST uses one-feedback admission;
NEW recalls history for the candidate; POOL recalls the same history for all
competitors; DRAW samples common episode weights before comparison. No new
parameters are fitted. Score jobs open training arrays only. Autonomous paths
use their own choices and feedback; oracle IDs are recorded for diagnosis only.
Every output directory is new, with source/data hashes and per-seed predictions.
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
ENTRY = 'src/Bayesian_state/workflows/analysis/model_0923_episodic_admission.py'
HELPER = 'src/Bayesian_state/model/modules/hypothesis_transition/episodic_admission.py'


def base_workflow():
    spec = importlib.util.spec_from_file_location('episodic_precision_base',
        REPO/'src/Bayesian_state/workflows/analysis/model_0923_precision_refit.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def autonomous(job: dict, base, workflow) -> None:
    import numpy as np
    from threadpoolctl import threadpool_info
    v = base.validation(); d = workflow.cohort().load_diagnosis()
    w = base.load('episodic_input_helpers', 'src/Bayesian_state/workflows/analysis/model_0923_h_rule_diagnosis.py')
    engine, meta, arrays = w.inputs(job, v)
    n = 16 if job['smoke'] else len(arrays['categories'])
    categories = arrays['categories'][:n]; bank = []
    target_rule = job['oracle']
    if v.digest(Path(job['reference'])) != job['reference_sha256']:
        raise ValueError('Autonomous reference changed')
    with np.load(job['reference']) as z:
        index = [int(np.flatnonzero(z['seeds'] == seed)[0]) for seed in job['seeds']]
        expected_choice = z['choices'][index, :n]
        expected_p = z['correct_probability'][index, :n]

    class Recorder:
        def __init__(self): self.rows = []; self.next_path = 0
        def before(self, model, choice, outcome):
            h = model.engine.get_module(d.ModuleRole.HYPOTHESIS_TRANSITION)
            event = h.transition_log[-1]
            self.rows.append(dict(present=int(target_rule in h.active),
                contribution=float(h.execution_probabilities[target_rule]),
                beta=float(model._pending_trial.beta[target_rule]) if target_rule in h.active else 0.,
                proposed=-1 if event['proposed_hypothesis'] is None else int(event['proposed_hypothesis']),
                search=int(event['search_event']), rejected=int(event.get('newcomer_rejected', False)),
                target_exited=int(target_rule in event['autonomous_exits']),
                target_capacity_dropped=int(event['capacity_dropped'] == target_rule),
                active_count=len(h.active), recalled_episodes=event.get('recalled_episodes', 0),
                recalled_exposure=event.get('recalled_exposure', 0.),
                candidate_evidence=event.get('candidate_initial_evidence', 0.)))
        def after(self, *args): pass

    start = time.process_time()
    for index, seed in enumerate(job['seeds']):
        recorder = Recorder()
        with d.record_lifecycle(recorder):
            result = d.autonomous.run_autonomous_category_learning(engine_config=engine,
                subject_id=job['subject'],condition=meta['condition'], stimulus=arrays['stimulus'][:n],
                categories=categories,trajectory_seed=seed,processed_data_dir=meta['paths']['processed_dir'],
                dataset_paths=meta['paths'])
        p = np.asarray(result.trajectory.observed_probabilities)
        if not np.isfinite(p).all() or np.any(p < 0):raise ValueError('Invalid autonomous probability')
        np.testing.assert_allclose(p.sum(axis=1),1.,atol=1e-12,rtol=0)
        row = {k:np.array([r[k] for r in recorder.rows]) for k in recorder.rows[0]}
        row.update(choices=np.array(result.trajectory.choices),probabilities=p,
                   feedback=np.array(result.trajectory.feedback),correct_probability=p[np.arange(n),categories-1])
        if job['model'] == 'BASE':
            np.testing.assert_array_equal(row['choices'],expected_choice[index])
            np.testing.assert_allclose(row['correct_probability'],expected_p[index],atol=1e-12,rtol=0)
        bank.append(row)
    target = Path(job['target']);target.parent.mkdir(parents=True,exist_ok=True)
    with target.with_suffix('.npz').open('xb') as f:
        np.savez_compressed(f,**{k:np.stack([r[k] for r in bank]) for k in bank[0]},
            seeds=job['seeds'],categories=categories,oracle=target_rule)
    v.write(target.with_suffix('.json'),dict(subject=job['subject'],model=job['model'],condition=meta['condition'],
        candidate=job['candidate'],paths=len(bank),trials=n,training_trials=meta['training_trials'],
        cpu_seconds=time.process_time()-start,numeric_pools=threadpool_info(),
        predictions_sha256=v.digest(target.with_suffix('.npz'))))


def worker(file: Path) -> None:
    base = base_workflow(); v = base.validation(); job = v.read(file)
    workflow = v.intervention_workflow();workflow.install_transition()
    diagnosis = workflow.cohort().load_diagnosis()
    feedback = base.load('src.Bayesian_state.model.modules.hypothesis_transition.feedback_guided',base.FEEDBACK)
    precision = base.load('src.Bayesian_state.model.modules.precision_interventions',base.PRECISION)
    guard = base.load('src.Bayesian_state.model.modules.pairing_snapshot_roundoff',
                     'src/Bayesian_state/model/modules/pairing_snapshot_roundoff.py')
    history = base.load('src.Bayesian_state.model.modules.hypothesis_transition.episodic_admission',HELPER)
    from src.Bayesian_state.model.modules.beta import BetaModule
    from src.Bayesian_state.model.modules.pairing_memory import HierarchicalPairingMemoryModule
    cls = sys.modules['src.Bayesian_state.model.modules.hypothesis_transition.mechanistic_rule_search'].MechanisticRuleSearchHypothesisTransitionModule
    with ExitStack() as stack:
        stack.enter_context(diagnosis.single_threaded_processes())
        corrections = stack.enter_context(feedback.retention_roundoff_guard(cls))
        snapshots = stack.enter_context(guard.pairing_snapshot_roundoff_guard(HierarchicalPairingMemoryModule))
        arm = job['precision']
        stack.enter_context(precision.precision_intervention(BetaModule,mode=arm['precision'],upper=arm['upper']))
        recalled = []
        if job['model'] == 'LAST':stack.enter_context(feedback.feedback_intervention(cls,admission=True))
        elif job['model'] != 'BASE':recalled = stack.enter_context(history.episodic_admission(cls,mode=job['model']))
        if job['phase'] == 'autonomous':autonomous(job,base,workflow)
        else:v.score(job,workflow)
        v.write(Path(job['target']).with_suffix('.corrections.json'),
                dict(feedback=corrections,snapshots=snapshots,recalled=recalled))


def execute(jobs: list[dict], output: Path, budget: int) -> None:
    from src.Bayesian_state.utils.parallel import parallel_job_count
    v = base_workflow().validation(); directory = output/'jobs';directory.mkdir()
    def launch(job):
        file = directory/(job['name']+'.json');v.write(file,job)
        launcher = ('import runpy,sys;source,script=sys.argv[1:3];sys.path.insert(0,source);'
                    "sys.argv=[script]+sys.argv[3:];runpy.run_path(script,run_name='__main__')")
        env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                   MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        with tempfile.TemporaryDirectory(prefix='episodic_admission_') as cache:
            env['NUMBA_CACHE_DIR'] = cache
            with file.with_suffix('.log').open('x') as f:
                result = subprocess.run([sys.executable,'-c',launcher,job['runtime'],str(REPO/ENTRY),
                    '--worker',str(file)],cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(f'Worker failed: {file.with_suffix(".log")}')
    workers = parallel_job_count(budget,len(jobs))
    print(f'{len(jobs)} jobs; {workers} workers; numeric threads=1',flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,future in enumerate(as_completed([pool.submit(launch,j) for j in jobs]),1):
            future.result()
            if i % 16 == 0 or i == len(jobs):print(f'{i}/{len(jobs)} complete',flush=True)


def run(config: Path, output: Path, stage: str, smoke: bool) -> None:
    import numpy as np
    import yaml
    base = base_workflow();v = base.validation();settings = yaml.safe_load(config.read_text())
    if output.exists():raise FileExistsError('Use a new episodic study directory')
    fit = REPO/settings['fit']; frozen = v.read(fit/'nominees.json')
    if frozen['smoke'] or frozen['status'] != 'frozen_before_test':raise ValueError('Need frozen scientific parameters')
    for name,sha in frozen['source_sha256'].items():
        if v.digest(REPO/name) != sha:raise ValueError('Fitted source changed')
    if not set(settings['arms']) <= {'BASE','LAST','NEW','POOL','DRAW'} or 'BASE' not in settings['arms']:
        raise ValueError('Invalid comparison arms')
    data = Path(frozen['data']); prep = v.read(data/'preparation_manifest.json')
    runtime = str((REPO/prep['settings']['source_fit']/'source').resolve())
    hashes = dict(frozen['source_sha256'])
    for name in (ENTRY,HELPER,'src/Bayesian_state/workflows/analysis/model_0923_h_rule_diagnosis.py'):
        hashes[name] = v.digest(REPO/name)
    subjects = settings['subjects'][:1] if smoke else settings['subjects'];jobs=[]
    family = settings['parameter_family']
    for subject in subjects:
        row = next(r for r in frozen['nominees'] if r['subject']==subject and r['model']==family)
        meta = v.read(data/f'subject_{subject}'/'prepared.json')
        reference = REPO/settings['reference']/'evaluate'/f'subject_{subject}'/family/(row['candidate']['id']+'.npz')
        audit = REPO/settings['audit']/'audit'/f'subject_{subject}'/'audit.json'
        # Audit identity is an output-only diagnostic; it never enters the H adapter.
        oracle = v.read(audit)['target']
        if oracle is None:raise ValueError('Missing training-geometry target')
        if stage == 'autonomous':
            with np.load(reference) as z:seeds = z['seeds'][:1 if smoke else settings['paths']].astype(int).tolist()
            if len(seeds) != (1 if smoke else settings['paths']):raise ValueError('Insufficient reference paths')
            chunks = [seeds[i:i+settings['chunk']] for i in range(0,len(seeds),settings['chunk'])]
        else:chunks = [[seed] for seed in settings['score']['seeds'][:1 if smoke else None]]
        for arm in settings['arms']:
            for i,seeds in enumerate(chunks):
                name = f'S{subject}_{arm}_{i:03d}'
                jobs.append(dict(name=name,phase='autonomous' if stage=='autonomous' else 'training',
                    subject=subject,model=arm,candidate=row['candidate'],folder=str(data/f'subject_{subject}'),
                    runtime=runtime,source_sha256=hashes,mechanism=frozen['settings']['mechanism'],
                    precision=frozen['settings']['arms'][family],smoke=smoke,seeds=seeds,oracle=oracle,
                    particles=2 if smoke else settings['score']['particle_overrides'].get(subject,settings['score']['particles']),
                    reference=str(reference),reference_sha256=v.digest(reference),audit_sha256=v.digest(audit),
                    trials=meta['trials'] if stage=='autonomous' else meta['training_trials'],
                    target=str((output/'chunks'/name).resolve())))
    jobs.sort(key=lambda j:-j['trials']*len(j['seeds'])*(1 if stage=='autonomous' else j['particles']))
    output.mkdir(parents=True);shutil.copy2(config,output/'config.yaml')
    for name in hashes:
        path = output/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,path)
    v.write(output/'protocol.json',dict(settings=settings,stage=stage,smoke=smoke,jobs=jobs,
        source_sha256=hashes,nomination_sha256=v.digest(fit/'nominees.json'),
        data_manifest_sha256=v.digest(data/'preparation_manifest.json'),data=str(data.resolve()),
        python=sys.version,versions={k:importlib.metadata.version(k) for k in ('numpy','scipy','pandas','PyYAML')}))
    execute(jobs,output,1 if smoke else settings['parallel_budget'])
    records = [v.read(Path(j['target']).with_suffix('.json')) for j in jobs]
    v.write(output/'completion.json',dict(status='complete',stage=stage,smoke=smoke,jobs=len(jobs),
        cpu_seconds=sum(r['cpu_seconds'] for r in records),protocol_sha256=v.digest(output/'protocol.json')))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',type=Path);parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--stage',choices=('autonomous','score'),default='autonomous')
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.config and args.output:run(args.config,args.output,args.stage,args.smoke)
    else:parser.error('Provide --config and --output')


if __name__=='__main__':main()
