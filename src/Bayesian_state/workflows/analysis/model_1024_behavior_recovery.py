"""Behavior-only resolution study and known-state execution reference.

Run --phase prepare --config YAML --output NEW [--smoke], then --phase run
--prepared PREPARED --output NEW. Workers keep the archived cognitive runtime;
only passive monitors and Bayes conditioning are added. Oral data are deferred.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ENTRY = str(Path(__file__).resolve().relative_to(ROOT))
STUDY = 'src/Bayesian_state/workflows/analysis/model_1023_execution_study.py'
CAPTURE = 'src/Bayesian_state/workflows/analysis/model_1023_execution_capture.py'
COLLECTOR = 'src/Bayesian_state/evaluation/execution_states.py'


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


study = load('behavior_recovery_execution_study', STUDY)
read, write, digest = study.read, study.write, study.digest


def sources() -> dict:
    return dict(study.sources(study.driver.io), **{p: digest(ROOT/p) for p in (ENTRY, CAPTURE, COLLECTOR)})


def execution_reference(rho: np.ndarray, likelihood: np.ndarray, choice: int) -> dict:
    """Condition on choice with the pre-sampling execution prior (not its draw)."""
    rho, likelihood = np.asarray(rho, float), np.asarray(likelihood, float)
    if (rho.ndim != 1 or likelihood.ndim != 2 or likelihood.shape[0] != len(rho)
            or not np.isfinite(rho).all() or not np.isfinite(likelihood).all()
            or np.any(rho < 0) or np.any(likelihood < 0) or not 1 <= choice <= likelihood.shape[1]):
        raise ValueError('Invalid execution prior, likelihood or one-based choice')
    np.testing.assert_allclose(rho.sum(), 1., atol=1e-12)
    np.testing.assert_allclose(likelihood.sum(axis=1), 1., atol=1e-12)
    joint = rho[:, None]*likelihood; marginal = joint.sum(axis=0)
    if marginal[choice-1] <= 0:
        raise ValueError('Observed choice has zero probability')
    posterior = joint[:, choice-1]/marginal[choice-1]
    log_ratio = np.zeros_like(joint)
    valid = joint > 0
    # Zero-prior rules and impossible choices carry no information mass.
    denominator = rho[:, None]*marginal[None, :]
    log_ratio[valid] = np.log(joint[valid]/denominator[valid])
    information = float((joint*log_ratio).sum())
    accuracy = float(joint.max(axis=0).sum())
    if information < -1e-12 or accuracy < rho.max()-1e-12 or accuracy > 1+1e-12:
        raise ValueError('Invalid local information or expected optimal accuracy')
    return dict(posterior=posterior, marginal=marginal,
                information=max(0., information), optimal_accuracy=accuracy,
                prior_optimal_accuracy=float(rho.max()))


def event_prefix(present: np.ndarray, before: int, after: int, block: int) -> tuple[int, int, int]:
    """Register a truth-centred window, with filtering from the task start."""
    present = np.asarray(present, bool)
    indices = np.flatnonzero(present)
    if present.ndim != 1 or not len(indices) or before < 0 or after < 1 or block < 1:
        raise ValueError('Require an observed entry and valid window sizes')
    entry = int(indices[0]); start = max(0, entry-before); stop = min(len(present), entry+after)
    prefix = min(len(present), ((stop+block-1)//block)*block)
    return start, stop, prefix


def monitor_class(base):
    """Add weight/support diagnostics to the unchanged passive state collector."""
    class Monitored(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs); self.monitors = []

        def __call__(self, trial, phase, snapshot):
            if phase == 'predictive':
                self.monitors.append({})
            weights = np.asarray(snapshot['weights'])
            present = np.asarray(snapshot['active'])[:, self.targets].any(axis=1)
            value = weights[present]; total = value.sum()
            row = self.monitors[trial]
            row['monitor_'+phase+'_ess'] = 1./np.square(weights).sum()
            row['monitor_'+phase+'_target_particles'] = int(present.sum())
            row['monitor_'+phase+'_target_ess'] = float(total*total/np.square(value).sum()) if total else 0.
            super().__call__(trial, phase, snapshot)

        def arrays(self):
            arrays = super().arrays()
            if len(self.monitors) != len(self.choices) or any(len(r) != 6 for r in self.monitors):
                raise ValueError('Incomplete weight/support monitor')
            arrays.update({k: np.array([r[k] for r in self.monitors]) for k in self.monitors[0]})
            return arrays
    return Monitored


def oracle_generation(job: dict, workflow) -> None:
    """Replay the shared generator while reading its pre-feedback decision state."""
    from threadpoolctl import threadpool_info
    v = study.base.validation(); d = workflow.cohort().load_diagnosis()
    inputs = study.base.load('behavior_reference_inputs', study.driver.refit.INPUTS)
    engine, meta, arrays = inputs.inputs(job, v)
    from src.Bayesian_state.model.modules.readout.choice import read_choice_probabilities_from_model
    with np.load(job['truth_states']+'.npz') as z:
        truth = {k: z[k].copy() for k in z.files}
    if digest(Path(job['truth_states']).with_suffix('.npz')) != job['truth_sha256']:
        raise ValueError('Generating truth changed')
    n = 8 if job['smoke'] else len(arrays['categories']); capacity = truth['rule'].shape[1]

    class Recorder:
        def __init__(self):
            self.next_path = 0; self.rows = []

        def before(self, model, choice, feedback):
            t = len(self.rows); prepared = model._pending_trial
            h = model.engine.get_module(d.ModuleRole.HYPOTHESIS_TRANSITION)
            active = np.asarray(h.active, int); count = len(active)
            np.testing.assert_array_equal(active, truth['rule'][t, :count])
            for name, actual in [('belief', prepared.prior), ('precision', prepared.beta), ('contribution', h.execution_probabilities)]:
                np.testing.assert_array_equal(actual[active], truth[name][t, :count])
            assert int(h.executed_hypothesis) == int(truth['executed'][t])
            assert model._oral_choice_prediction.output_lapse == 0.
            rng_before = deepcopy(h.execution_rng.bit_generator.state)
            _, snapshot = read_choice_probabilities_from_model(model, prepared.perceived_stimulus, return_snapshot=True)
            assert h.execution_rng.bit_generator.state == rng_before
            lookup = {rule: i for i, rule in enumerate(snapshot.hypothesis_indices)}
            likelihood = snapshot.category_probabilities[[lookup[int(rule)] for rule in active]]
            rho = h.execution_selection_probabilities[active].copy()
            result = execution_reference(rho, likelihood, int(choice))
            selected = int(np.flatnonzero(active == h.executed_hypothesis)[0])
            np.testing.assert_allclose(likelihood[selected], truth['probabilities'][t], atol=1e-14, rtol=0)
            target = np.isin(active, job['target_rules']); present = bool(target.any())
            target_p = float(result['posterior'][target].sum())
            row = dict(information=result['information'], optimal_accuracy=result['optimal_accuracy'],
                prior_optimal_accuracy=result['prior_optimal_accuracy'], choice_probability=result['marginal'][int(choice)-1],
                conditional_decision_sources=np.array([0., 1-target_p, target_p]) if present else np.array([1., 0., 0.]),
                present=present, target_execution_probability=target_p, executed=int(h.executed_hypothesis),
                perceived_stimulus=prepared.perceived_stimulus.copy())
            for name, values in [('execution_prior', rho), ('execution_posterior', result['posterior'])]:
                padded = np.zeros(capacity); padded[:count] = values; row[name] = padded
            padded = np.zeros((capacity, model.n_cats)); padded[:count] = likelihood; row['rule_choice_probabilities'] = padded
            self.rows.append(row)

        def after(self, *args):
            pass

    started, cpu = time.monotonic(), time.process_time(); recorder = Recorder()
    with d.record_lifecycle(recorder):
        generated = d.autonomous.run_autonomous_category_learning(engine_config=engine,
            subject_id=meta['subject'], condition=meta['condition'], stimulus=arrays['stimulus'][:n],
            categories=arrays['categories'][:n], trajectory_seed=job['seeds'][0],
            processed_data_dir=meta['paths']['processed_dir'], dataset_paths=meta['paths'])
    for name, actual in [('choices', generated.trajectory.choices), ('feedback', generated.trajectory.feedback),
                         ('probabilities', generated.trajectory.observed_probabilities)]:
        np.testing.assert_array_equal(actual, truth[name][:n])
    result = {k: np.array([r[k] for r in recorder.rows]) for k in recorder.rows[0]}
    result.update(choices=generated.trajectory.choices, categories=arrays['categories'][:n],
                  rule=truth['rule'][:n], true_executed=truth['executed'][:n], seeds=job['seeds'])
    target = Path(job['target']); target.parent.mkdir(parents=True, exist_ok=True)
    with target.with_suffix('.npz').open('xb') as f:
        np.savez_compressed(f, **result)
    write(target.with_suffix('.json'), dict(status='complete', subject=meta['subject'], case_id=job['recovery_case'],
        condition=meta['condition'], candidate=job['candidate'], seeds=job['seeds'], trials=n, paths=1,
        exact_generation_replay=True, predictions_sha256=digest(target.with_suffix('.npz')),
        seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu, numeric_pools=threadpool_info(),
        reference_conditioning='known pre-execution state and current choice; current execution hidden; no future choices',
        behavior_recovery_sources=job['behavior_recovery_sources']))


def worker(file: Path) -> None:
    job = read(file)
    if job['behavior_recovery_sources'] != sources():
        raise ValueError('Behavior recovery source changed')
    if job['kind'] == 'oracle':
        original = study.autonomous; study.autonomous = oracle_generation
        try:
            study.worker(file)
        finally:
            study.autonomous = original
    elif job['kind'] == 'bare':
        study.worker(file)
    else:
        capture = load('behavior_resolution_capture', CAPTURE)
        original = capture.study.base.load
        def monitored(name, path):
            module = original(name, path)
            if path == COLLECTOR:
                module.ExecutionStateCollector = monitor_class(module.ExecutionStateCollector)
            return module
        capture.study.base.load = monitored
        try:
            capture.worker(file)
        finally:
            capture.study.base.load = original


def prepare(config: Path, output: Path, smoke: bool) -> None:
    import yaml
    from . import model_1024_state_oral_validation as previous
    from . import model_1023_execution_recovery as fixture_tools
    from . import model_1023_execution_recovery_score as checked
    if output.exists():
        raise FileExistsError('Use a new preparation directory')
    p = yaml.safe_load(config.read_text())
    if (p['numerical_cases'] != ['C004','C005','C006','C007','C008','C010','C013']
            or p['particles'] != [512,1024,2048] or p['filter_seeds'] != 3 or p['oral_status'] != 'deferred_pending_user_manual_review'):
        raise ValueError('Require the registered behavioral recovery budget')
    source = ROOT/p['prepared']; inference = ROOT/p['inference']
    done = read(source/'completion.json'); old = read(inference/'protocol.json')
    if done['status'] != 'complete' or read(inference/'completion.json')['status'] != 'complete':
        raise ValueError('Require complete previous simulation recovery')
    assert done['truths_sha256'] == digest(source/'truths.json')
    assert done['cases_sha256'] == digest(source/'cases.json')
    output.mkdir(parents=True); oracle = []; numerical = []
    cases = old['cases'][:5] if smoke else old['cases']
    if smoke:
        cases = [next(c for c in old['cases'] if c['definition']['case_id'] == 'C004')]
    for case in cases:
        definition, truth = case['definition'], case['truth']; cid = definition['case_id']
        original_folder = Path(definition['folder']); raw_file = Path(truth['states'])
        if digest(raw_file.with_suffix('.npz')) != truth['states_sha256']:
            raise ValueError('Generating state hash changed')
        with np.load(raw_file.with_suffix('.npz')) as z:
            raw = {k: z[k].copy() for k in z.files}
        oracle.append(dict(case_id=cid, subject=definition['subject'], scope=definition['scope'],
            template=definition['template'], candidate=truth['candidate'], seed=truth['seed'],
            target_rules=definition['target_rules'], truth=truth))
        if cid not in p['numerical_cases']:
            continue
        present = np.isin(raw['rule'], definition['target_rules']).any(axis=1)
        start, stop, n = event_prefix(present, p['pre_entry_trials'], p['post_entry_trials'], p['prefix_rounding'])
        if smoke:
            n, start, stop = 128, 0, 8
        parts = []
        for name in ('training.npz','sealed_test.npz'):
            with np.load(original_folder/name) as z:
                parts.append({k: z[k] for k in z.files})
        observations = {k: np.concatenate([part[k] for part in parts])[:n] for k in parts[0]}
        folder = output/'prefixes'/cid/f'subject_{definition["subject"]}'
        fixture_tools.fixture(original_folder, folder, observations, definition['template']['runtime'], .7)
        template = deepcopy(definition['template']); template['folder'] = str(folder.resolve())
        seeds = [j['seeds'][0] for j in case['point_jobs'][truth['candidate']['id']]][:p['filter_seeds']]
        refs = []
        for stem in case['states'][truth['candidate']['id']][:p['filter_seeds']]:
            meta = read(Path(stem).with_suffix('.json'))
            refs.append(dict(particles=512, seed=meta['seeds'][0], state=stem,
                state_sha256=digest(Path(stem).with_suffix('.npz')), score=next(j['target'] for j in case['point_jobs'][truth['candidate']['id']] if j['seeds']==meta['seeds'])))
        for j in case['resolution'].get(truth['candidate']['id'], []):
            meta = read(Path(j['state_capture_target']).with_suffix('.json'))
            if meta['seeds'][0] in seeds:
                refs.append(dict(particles=1024, seed=meta['seeds'][0], state=j['state_capture_target'],
                    state_sha256=digest(Path(j['state_capture_target']).with_suffix('.npz')), score=j['target']))
        numerical.append(dict(case_id=cid, subject=definition['subject'], scope=definition['scope'],
            candidate=truth['candidate'], target_rules=definition['target_rules'], template=template,
            prefix_trials=n, window_start=start, window_stop=stop, seeds=seeds, references=refs,
            prefix_control=cid in p['prefix_control_cases'], truth=truth))
    write(output/'protocol.json', dict(policy=p, oracle=oracle, numerical=numerical, smoke=smoke,
        sources=sources(), inputs_sha256={str(config.resolve()):digest(config),
            str((source/'completion.json').resolve()):digest(source/'completion.json'),
            str((inference/'protocol.json').resolve()):digest(inference/'protocol.json')}, oral_used=False))
    write(output/'completion.json', dict(status='complete', smoke=smoke, protocol_sha256=digest(output/'protocol.json')))
    print('Frozen known-state references and targeted numerical prefixes', flush=True)


def run(prepared: Path, output: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    import os
    from . import model_1023_execution_capture as capture
    from . import model_1023_execution_validation as validation
    from . import model_1024_state_oral_validation as previous
    if output.exists():
        raise FileExistsError('Use a new numerical run directory')
    protocol = read(prepared/'protocol.json'); p = protocol['policy']; smoke = protocol['smoke']
    if read(prepared/'completion.json')['protocol_sha256'] != digest(prepared/'protocol.json') or protocol['sources'] != sources():
        raise ValueError('Registered numerical preparation changed')
    output.mkdir(parents=True); jobs = []
    for row in protocol['oracle']:
        job = deepcopy(row['template']); cid = row['case_id']; name = cid+'_known_state'
        job.update(name=name, phase='autonomous', kind='oracle', candidate=row['candidate'], recovery_case=cid,
            seeds=[row['seed']], target_rules=row['target_rules'], truth_states=row['truth']['states'],
            truth_sha256=row['truth']['states_sha256'], target=str((output/'oracle'/name).resolve()),
            smoke=smoke, source_sha256=study.sources(study.driver.io), behavior_recovery_sources=sources())
        jobs.append(job)
    for row in protocol['numerical']:
        c = dict(subject=row['subject'], template=row['template'], target_rules=row['target_rules'])
        levels = [2] if smoke else [1024,2048]
        for level in levels:
            for index, seed in enumerate(row['seeds'][:1] if smoke else row['seeds']):
                if any(r['particles']==level and r['seed']==seed for r in row['references']):
                    continue
                name = f'{row["case_id"]}_P{level}_S{index}'
                job = previous.inference_job(c, row['candidate'], seed, level, output, name, smoke)
                job.update(kind='filter', recovery_case=row['case_id'], behavior_recovery_sources=sources())
                jobs.append(job)
        if row['prefix_control'] and not smoke:
            seed = row['seeds'][0]; name = row['case_id']+'_prefix_control'
            job = previous.inference_job(c, row['candidate'], seed, 512, output, name, False)
            job.update(kind='prefix_control', recovery_case=row['case_id'], behavior_recovery_sources=sources())
            jobs.append(job)
    def dispatch(batch):
        if not batch:
            return
        workers = 1 if smoke else min(p['parallel_budget'], len(os.sched_getaffinity(0)), len(batch))
        original = validation.ENTRY; validation.ENTRY = ENTRY
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                validation.NumericalQueue(pool).execute(batch, output, workers)
        finally:
            validation.ENTRY = original
    if smoke:
        bare = deepcopy(next(j for j in jobs if j['kind']=='filter'))
        bare.update(name='smoke_bare', kind='bare', target=str((output/'scores'/'smoke_bare').resolve()))
        bare.pop('capture_sources'); bare.pop('state_capture_target')
        dispatch([bare])
        actual = next(j for j in jobs if j['kind']=='filter')
        actual.update(original_score=bare['target']+'.npz', original_score_sha256=digest(Path(bare['target']).with_suffix('.npz')))
    write(output/'protocol.json', dict(prepared=str(prepared.resolve()), preparation_sha256=digest(prepared/'protocol.json'),
        jobs=jobs, smoke=smoke, policy=p, sources=sources(), oral_used=False))
    started = time.monotonic(); dispatch(jobs)
    controls = []
    for job in jobs:
        if job['kind'] != 'prefix_control':
            continue
        row = next(r for r in protocol['numerical'] if r['case_id']==job['recovery_case'])
        ref = next(r for r in row['references'] if r['particles']==512 and r['seed']==job['seeds'][0])
        with np.load(ref['state']+'.npz') as old, np.load(job['state_capture_target']+'.npz') as new:
            n = len(new['choices'])
            for key in old.files:
                np.testing.assert_array_equal(new[key], old[key] if key in ('seeds','target_rules') else old[key][:n])
        controls.append(dict(case_id=row['case_id'], exact_prefix_state_replay=True, original_state_sha256=ref['state_sha256']))
    write(output/'completion.json', dict(status='complete', smoke=smoke, jobs=len(jobs), seconds=time.monotonic()-started,
        protocol_sha256=digest(output/'protocol.json'), prefix_controls=controls, oral_used=False,
        oracle_replays=sum(j['kind']=='oracle' for j in jobs), new_filters=sum(j['kind']!='oracle' for j in jobs)))
    print('Known-state references and numerical resolution complete; oral data deferred', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--phase', choices=('prepare','run'))
    parser.add_argument('--prepared', type=Path); parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path); parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.phase=='prepare' and args.config and args.output:
        prepare(args.config,args.output,args.smoke)
    elif args.phase=='run' and args.prepared and args.output:
        run(args.prepared,args.output)
    else:
        parser.error('Provide --worker or the input/output paths for the selected phase')
